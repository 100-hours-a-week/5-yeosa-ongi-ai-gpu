import logging
import pickle
import torch
from functools import partial

from fastapi import Request, Response

from app.service.embedding import embed_images
from app.utils.image_loader import get_image_loader  # GPU 서버도 image_loader 사용
from app.utils.logging_decorator import log_flow

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 16
device = "cuda" if torch.cuda.is_available() else "cpu"

@log_flow
async def embed_controller(request: Request):
    """
    이미지 파일명을 받아 GPU 서버에서 직접 이미지 로딩, 임베딩 수행.
    결과를 Pickle 바이너리로 반환.
    """
    logger.info("임베딩 요청 수신 및 처리 시작")

    try:
        payload = await request.json()  # ✅ JSON 기반 수신
        image_refs = payload["images"]

        logger.debug("파일명 목록 수신 완료", extra={"count": len(image_refs)})

        # ✅ GPU 서버에서도 image_loader 사용
        image_loader = request.app.state.image_loader
        images = await image_loader.load_images(image_refs)

        logger.info("이미지 로딩 및 디코딩 완료", extra={"count": len(images)})

        clip_model = request.app.state.clip_model
        clip_preprocess = request.app.state.clip_preprocess
        loop = request.app.state.loop

        task_func = partial(
            embed_images,
            clip_model,
            clip_preprocess,
            images,
            image_refs,
            batch_size=DEFAULT_BATCH_SIZE,
            device=device
        )

        result = await loop.run_in_executor(None, task_func)

        logger.info("임베딩 완료", extra={"processed_images": len(result)})

        response_obj = {
            "message": "success",
            "data": result
        }

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
