from fastapi import APIRouter
from app.api.endpoints import (
    album_embedding_router,
    # HACK: Health check용 임시 라우터
    album_health_router
)

api_router = APIRouter()

api_router.include_router(
    album_embedding_router.router,
    prefix="/clip/embedding",
    tags=["embedding"],
)

# HACK: Health check용 임시 라우터
api_router.include_router(
    album_health_router.router,
    prefix="/health/info",
    tags=["health"]
)
