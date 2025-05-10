from fastapi import APIRouter

from app.utils.logging_decorator import log_flow

# HACK: Health check용 임시 라우터

router = APIRouter(tags=["health"])


@router.get("", status_code=200)
@log_flow
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "message": "ONGI AI GPU Server is healthy",
        "version": "1.0.0",
    }
