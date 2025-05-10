from functools import partial

from fastapi import Request

from app.schemas.album_schema import ImageRequest
from app.service.embedding import embed_images

async def embed_controller(req: ImageRequest, request: Request):
    filenames = req.images

    image_loader = request.app.state.image_loader
    images = await image_loader.load_images(filenames)

    clip_model = request.app.state.clip_model
    clip_preprocess = request.app.state.clip_preprocess
    loop = request.app.state.loop
    task_func = partial(
        embed_images,
        clip_model,
        clip_preprocess,
        images,
        filenames,
        batch_size=32,
        device="cuda"
    )

    # 임베딩 결과를 직접 반환
    result = await loop.run_in_executor(None, task_func)
    return {"message": "success", "data": result}
