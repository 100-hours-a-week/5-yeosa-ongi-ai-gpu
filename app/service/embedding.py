import torch
import time
from concurrent.futures import ThreadPoolExecutor

def format_elapsed(t: float) -> str:
    return f"{t * 1000:.2f} ms" if t < 1 else f"{t:.2f} s"

def embed_images(
    model, preprocess, images, filenames, batch_size=32, device="cuda"
):
    # 이미지 전처리를 배치 단위로 수행
    preprocessed_batches = []
    # print(f"[INFO] 전처리 시작")
    t1 = time.time()
    # ThreadPoolExecutor 생성
    # with ThreadPoolExecutor() as executor:
        # 전처리 함수를 lambda로 정의
        # preprocess_func = lambda img: preprocess(img)
        
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        # 병렬로 전처리 수행
        # preprocessed_batch = list(executor.map(preprocess_func, batch_images))
        # preprocessed_batch = torch.stack(preprocessed_batch)
        preprocessed_batch = preprocess(batch_images)
        preprocessed_batches.append(preprocessed_batch)
    t2 = time.time()
    print(f"[INFO] 전처리 완료: {format_elapsed(t2 - t1)}")
    # 결과를 저장할 딕셔너리
    results = {}
    
    # print(f"[INFO] 임베딩 시작")
    t1 = time.time()
    # 배치 단위로 임베딩 수행
    for i, batch in enumerate(preprocessed_batches):
        batch_filenames = filenames[i * batch_size:(i + 1) * batch_size]
        
        # GPU로 데이터 이동
        image_input = batch.to(device)
        
        # 임베딩 생성
        with torch.no_grad():
            batch_features = model.encode_image(image_input)
        
        # CPU로 결과 이동 및 저장
        batch_features = batch_features.cpu()
        for filename, feature in zip(batch_filenames, batch_features):
            results[filename] = feature.cpu()

    t2 = time.time()
    print(f"[INFO] 임베딩 완료: {format_elapsed(t2 - t1)}")

    return results
