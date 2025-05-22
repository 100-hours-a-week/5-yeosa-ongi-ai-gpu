import torch

from app.model.yoloface.face_detector import YoloDetector

device = "cuda" if torch.cuda.is_available() else "cpu"
_yolo_detector = None


def load_yolo_detector(target_size: int = 320, min_face: int = 60):
    """
    YOLO 기반 얼굴 검출 모델을 로드하거나 캐시된 인스턴스를 반환합니다.

    지정된 파라미터로 YoloDetector 객체를 초기화하고, 최초 로드 이후에는
    전역 변수에 캐시하여 재사용합니다.

    Args:
        target_size (int): YOLO 모델 입력 이미지의 한 변 크기 (정사각형 기준).
            입력 이미지는 이 크기에 맞게 비율을 유지하여 리사이즈되고, 
            남는 부분은 패딩 처리됩니다. 기본값은 320.
        min_face (int): 감지할 얼굴의 최소 크기(px). 너무 작은 얼굴은 필터링됩니다. 기본값은 60.

    Returns:
        YoloDetector: 초기화되거나 캐시된 YOLO 얼굴 검출기 인스턴스.

    """
    global _yolo_detector
    if _yolo_detector is None:
        _yolo_detector = YoloDetector(
            target_size=target_size, device=device, min_face=min_face
        )
    return _yolo_detector
