from fastapi import APIRouter, Request

from app.api.controllers.album_embedding_controller import embed_controller
from app.utils.logging_decorator import log_flow

router = APIRouter(tags=["embedding"])


@router.post("", status_code=201)
@log_flow
async def embed(request: Request):
    return await embed_controller(request)