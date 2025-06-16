from collections import defaultdict

import numpy as np
import torch
from typing import Any
from numpy.typing import NDArray
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from torch import Tensor

from app.utils.logging_decorator import log_exception, log_flow

device = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess(face: NDArray[np.uint8]) -> NDArray[np.float32]:
    """
    얼굴 이미지를 ArcFace 입력 형식에 맞게 전처리합니다.

    RGB 이미지 배열을 (C, H, W) 순서로 전치하고 float32로 변환한 뒤,
    ArcFace 모델에서 요구하는 정규화 방식으로 스케일링합니다.
    해당 방식은 (pixel - 127.5) / 128.0 형태입니다.

    Args:
        face (np.ndarray): 전처리할 얼굴 이미지 (H, W, C) 형식의 NumPy 배열.

    Returns:
        np.ndarray: 전처리된 얼굴 이미지 (C, H, W) 형식의 float32 NumPy 배열.

    """
    face = np.transpose(face, (2, 0, 1)).astype(np.float32)
    return ((face - 127.5) / 128.0).astype(np.float32)


@log_flow
def cluster_faces(
    images: list[Image.Image], file_names: list[str], arcface_model: Any, yolo_detector: Any
) -> list[list[str]]:
    """
    유사한 얼굴을 클러스터링합니다.

    여러 이미지를 입력으로 받아,
    YOLO 기반 얼굴 탐지 및 ArcFace 임베딩을 수행한 뒤,
    DBSCAN을 통해 동일 인물로 판단되는 얼굴 그룹을 클러스터링합니다.
    이후 클러스터의 평균 및 최대 거리 기준으로 신뢰도 있는 결과만 필터링합니다.

    Args:
        images (list): PIL.Image 형태의 이미지 리스트.
        file_names (list[str]): 각 이미지에 대응하는 파일명 리스트.
        arcface_model: 얼굴 임베딩을 위한 ArcFace 모델.
        yolo_detector: 얼굴 검출 및 정렬을 위한 YOLO 기반 detector 객체.

    Returns:
        list[list[str]]: 동일 인물로 판단된 이미지 파일명 리스트들의 리스트.
            예시: [['1F.jpg', '2F.jpg'], ['3M.jpg', '4M.jpg']]

    """
    detector = yolo_detector
    arcface = arcface_model

    np_images = [np.array(img) for img in images]
    bboxes_list, landmarks_list = detector.predict(np_images)

    crops = []
    metadata = []  # (file_name, bbox) 튜플 저장

    for idx, (img, bboxes, landmarks) in enumerate(
        zip(np_images, bboxes_list, landmarks_list)
    ):
        if len(landmarks) > 0:
            aligned_faces = detector.align(img, landmarks)
            for face_idx, face in enumerate(aligned_faces):
                crops.append(face)
                metadata.append((file_names[idx], bboxes[face_idx]))

    if not crops:
        return []

    input_tensor: Tensor = torch.tensor(
        np.stack([preprocess(f) for f in crops]),
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        embeddings = arcface(input_tensor).cpu().numpy()

    dbscan = DBSCAN(eps=0.6, min_samples=2, metric="cosine")
    labels = dbscan.fit_predict(embeddings)

    # 클러스터별 평균 거리 기반 필터링
    cluster_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        if label != -1:
            cluster_indices[label].append(idx)

    valid_labels = []
    for label, idxs in cluster_indices.items():
        cluster_embeds = embeddings[idxs]
        n = len(cluster_embeds)
        if n < 2:
            continue
        dists = cosine_distances(cluster_embeds)
        avg_dist = np.mean(dists[np.triu_indices(n, k=1)])
        max_dist = np.max(dists[np.triu_indices(n, k=1)])
        if n < 5:
            if max_dist < 0.6:
                valid_labels.append(label)
        else:
            if avg_dist < 0.5:
                valid_labels.append(label)

    # 유효 라벨만 필터링
    filtered_labels = np.array([
        label if label in valid_labels else -1
        for label in labels
    ])

    # 클러스터링 결과 구조화
    cluster_dict = defaultdict(list)
    for label, (name, bbox) in zip(filtered_labels, metadata):
        if label != -1:
            cluster_dict[label].append((name, bbox))

    # 프론트에 맞는 반환 형식 구성
    result = []
    for cluster in sorted(cluster_dict.values(), key=len, reverse=True):
        images = [item[0] for item in cluster]
        rep_image, rep_bbox = cluster[0]
        result.append({
            "images": images,
            "representative_face": {
                "image": rep_image,
                "bbox": rep_bbox
            }
        })

    return result