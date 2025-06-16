from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.api.controllers.album_people_controller import people_controller
from app.utils.logging_decorator import log_flow

router = APIRouter(tags=["people"])


@router.post("", status_code=201)
@log_flow
async def people(request: Request) -> JSONResponse:
    """동일 인물 얼굴 클러스터링 요청을 people_controller에 전달합니다."""
    return await people_controller(request)