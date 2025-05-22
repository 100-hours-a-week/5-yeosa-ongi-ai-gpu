import logging
import time
import pickle
import asyncio
from datetime import datetime
from typing import Any
from functools import partial

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from app.service.people import cluster_faces
from app.utils.logging_decorator import log_flow

logger = logging.getLogger(__name__)

PEOPLE_SEMAPHORE_SIZE = 1
people_semaphore = asyncio.Semaphore(PEOPLE_SEMAPHORE_SIZE)

def format_elapsed(t: float) -> str:
    return f"{t * 1000:.2f} ms" if t < 1 else f"{t:.2f} s"

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

@log_flow
async def people_controller(request: Request) -> JSONResponse:
    """
    이미지 파일명을 받아 GPU 서버에 전송하고,
    동일 인물 클러스터링 결과를 반환하는 컨트롤러입니다.
    """
    logger.info("인물 클러스터링 요청 처리 시작")

    try:
        arcface_model = request.app.state.arcface_model
        yolo_detector = request.app.state.yolo_detector
        image_loader = request.app.state.image_loader
        loop = request.app.state.loop

        # ✅ 요청 파싱
        t1 = time.time()
        payload = await request.json()
        filenames = payload["images"]  # List[str]
        logger.info("파일명 수신 완료", extra={"count": len(filenames)})

        # ✅ 이미지 로딩
        images = await image_loader.load_images(filenames)
        logger.info("이미지 디코딩 완료")

        task_func = partial(cluster_faces, images, filenames, arcface_model, yolo_detector)

        # async with people_semaphore:
        clustering_result = await loop.run_in_executor(None, task_func)

        # ✅ 직렬화 및 응답 전송
        response_obj = {
            "message": "success",
            "data": clustering_result
        }

        t3 = time.time()
        serialized = pickle.dumps(response_obj)
        t4 = time.time()
        print(f"[INFO] 응답 직렬화 완료: {format_elapsed(t4 - t3)}")

        logger.info("인물 클러스터링 완료", extra={"clusters": len(clustering_result)})

        return Response(
            content=serialized,
            media_type="application/octet-stream"
        )

    except Exception as e:
        logger.error("인물 클러스터링 처리 중 예외 발생", exc_info=True, extra={"error": str(e)})
        error_response = {
            "message": "fail",
            "data": {}
        }
        return Response(
            content=pickle.dumps(error_response),
            media_type="application/octet-stream",
            status_code=500
        )