import logging
import pickle
import torch
import time
from functools import partial

from fastapi import Request, Response

from app.service.embedding import embed_images
from app.utils.image_loader import get_image_loader  # GPU 서버도 image_loader 사용
from app.utils.logging_decorator import log_flow

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 16
device = "cuda" if torch.cuda.is_available() else "cpu"

def format_elapsed(t: float) -> str:
    return f"{t * 1000:.2f} ms" if t < 1 else f"{t:.2f} s"

@log_flow
async def embed_controller(request: Request):
    """
    FastAPI 서버로부터 Pickle 직렬화된 이미지 배열과 파일명을 받아
    GPU에서 임베딩을 수행하고 Pickle로 결과를 반환합니다.
    """
    logger.info("임베딩 요청 수신 및 처리 시작")

    try:
        t0 = time.time()

        # ✅ 역직렬화 시간 측정
        body = await request.body()
        t1 = time.time()
        payload = pickle.loads(body)
        t2 = time.time()
        print(f"[INFO] 역직렬화 완료: {format_elapsed(t2 - t1)}")


        images = payload["images"]        # List[np.ndarray]
        filenames = payload["filenames"]  # List[str]

        # 모델 및 전처리기 준비
        clip_model = request.app.state.clip_model
        clip_preprocess = request.app.state.clip_preprocess
        loop = request.app.state.loop

        # 임베딩 함수 준비 (CPU offload 병렬 실행)
        task_func = partial(
            embed_images,
            clip_model,
            clip_preprocess,
            images,
            filenames,
            batch_size=DEFAULT_BATCH_SIZE,
            device=device
        )

        t3 = time.time()
        result = await loop.run_in_executor(None, task_func)
        t4 = time.time()
        print(f"[INFO] 임베딩 완료 - 처리 이미지 수: {len(result)}, 소요 시간: {format_elapsed(t4 - t3)}")

        # ✅ 직렬화 시간 측정
        t5 = time.time()
        response_obj = {
            "message": "success",
            "data": result
        }
        serialized_response = pickle.dumps(response_obj)
        t6 = time.time()
        print(f"[INFO] 응답 직렬화 완료: {format_elapsed(t6 - t5)}")

        # ✅ 응답 전송 전 시간 기록 (GPU → CPU 전송 측정 위해)
        print(f"[INFO] 응답 전송 직전 - 총 처리 시간: {format_elapsed(t6 - t0)}")

        return Response(
            content=pickle.dumps(response_obj),
            media_type="application/octet-stream"
        )

    except Exception as e:
        logger.error("임베딩 처리 중 예외 발생", exc_info=True, extra={"error": str(e)})
        error_response = {
            "message": "fail",
            "data": {}
        }
        return Response(
            content=pickle.dumps(error_response),
            media_type="application/octet-stream",
            status_code=500
        )
