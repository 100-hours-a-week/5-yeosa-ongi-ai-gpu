import logging
from functools import partial
from typing import Dict, Any

from fastapi import Request
from fastapi.responses import JSONResponse

from app.schemas.album_schema import ImageRequest
from app.service.embedding import embed_images
from app.utils.logging_decorator import log_exception, log_flow

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 16
DEFAULT_DEVICE = "cpu"


@log_flow
async def embed_controller(req: ImageRequest, request: Request):
    """
    이미지 임베딩을 수행하는 컨트롤러입니다.

    Args:
        req: 이미지 파일명 목록을 포함한 요청 객체
        request: FastAPI 요청 객체

    Returns:
        JSONResponse: 성공 메시지와 데이터를 포함한 응답
    """
    logger.info(
        "이미지 임베딩 요청 처리 시작",
        extra={"total_images": len(req.images)},
    )
    
    image_refs = req.images

    image_loader = request.app.state.image_loader
    images = await image_loader.load_images(image_refs)

    logger.debug(
        "이미지 로드 완료",
        extra={"loaded_images": len(images)},
    )

    clip_model = request.app.state.clip_model
    clip_preprocess = request.app.state.clip_preprocess
    loop = request.app.state.loop
    
    task_func = partial(
        embed_images,
        clip_model,
        clip_preprocess,
        images,
        image_refs,
        batch_size=32,
        device="cuda"
    )

    # 임베딩 결과를 직접 반환
    result = await loop.run_in_executor(None, task_func)
    
    logger.info(
        "이미지 임베딩 완료",
        extra={"processed_images": len(image_refs)},
    )
    return {"message": "success", "data": result}
