import pickle
import torch
import time
from functools import partial
from datetime import datetime
# import numpy as np

from fastapi import Request, Response

from app.service.embedding import embed_images
from app.utils.image_loader import get_image_loader  # GPU 서버도 image_loader 사용
from app.utils.logging_decorator import log_flow

DEFAULT_BATCH_SIZE = 64

device = "cuda" if torch.cuda.is_available() else "cpu"

def format_elapsed(t: float) -> str:
    return f"{t * 1000:.2f} ms" if t < 1 else f"{t:.2f} s"

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

@log_flow
async def embed_controller(request: Request):
    """
    이미지 파일명을 받아 GPU 서버에서 직접 이미지 로딩, 임베딩 수행.
    결과를 Pickle 바이너리로 반환.
    """
    print("[START] GPU 서버: 임베딩 요청 수신")

    try:
        t0 = time.time()
        receive_time_str = now_str()

        payload = await request.json()
        image_refs = payload["images"]
        client_send_time = payload.get("client_send_time")  # optional

        print(f"[INFO] 요청 수신 시각: {receive_time_str}")
        if client_send_time:
            print(f"[INFO] 클라이언트 전송 시각: {client_send_time}")

        # ✅ 이미지 로딩 시간 측정
        print(f"[INFO] 이미지 로딩 및 디코딩 시작")
        t1 = time.time()
        image_loader = request.app.state.image_loader
        images = await image_loader.load_images(image_refs)
        t2 = time.time()
        print(f"[INFO] 이미지 로딩 및 디코딩 완료: {format_elapsed(t2 - t1)}")
        # images =  [np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8) for _ in image_refs]

        # ✅ 임베딩
        clip_model = request.app.state.clip_model
        clip_preprocess = request.app.state.clip_preprocess
        # loop = request.app.state.loop

        # task_func = partial(
        #     embed_images,
        #     clip_model,
        #     clip_preprocess,
        #     images,
        #     image_refs,
        #     batch_size=DEFAULT_BATCH_SIZE,
        #     device=device
        # )

        # result = await loop.run_in_executor(None, task_func)
        result = await embed_images(clip_model, clip_preprocess, images, image_refs, batch_size=DEFAULT_BATCH_SIZE, device=device)

        # ✅ 직렬화 시간 측정
        print(f"[INFO] 응답 직렬화 시작")
        t5 = time.time()
        response_obj = {
            "message": "success",
            "data": result
        }
        serialized = pickle.dumps(response_obj)
        t6 = time.time()
        print(f"[INFO] 응답 직렬화 완료: {format_elapsed(t6 - t5)}")

        # ✅ 응답 직전 시각
        response_send_time_str = now_str()
        print(f"[INFO] 응답 전송 시각: {response_send_time_str}")
        print(f"[INFO] 총 처리 시간: {format_elapsed(t6 - t0)}")

        return Response(
            content=serialized,
            media_type="application/octet-stream"
        )

    except Exception as e:
        print(f"[EXCEPTION] 임베딩 처리 중 오류 발생: {e}")
        error_response = {
            "message": "fail",
            "data": {}
        }
        return Response(
            content=pickle.dumps(error_response),
            media_type="application/octet-stream",
            status_code=500
        )
