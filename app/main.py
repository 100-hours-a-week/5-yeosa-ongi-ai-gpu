import os
import asyncio
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI

from app.api import api_router
from app.config.settings import IMAGE_MODE
from app.middleware.error_handler import setup_exception_handler
from app.model.clip_loader import load_clip_model
from app.core.task_queue import SerialTaskQueue
from app.utils.image_loader import get_image_loader

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 실행 시, 모델 및 이미지 로더 초기화 로직입니다."""
    # GPU 설정
    if torch.cuda.is_available():
        device = "cuda"
        # # GPU 메모리 최적화
        # torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = False
        # # 기본 CUDA 스트림 설정
        # torch.cuda.set_device(0)
    else:
        device = "cpu"
    
    # CLIP 모델 초기화
    clip_model, clip_preprocess = load_clip_model(device=device)
    
    loop = asyncio.get_running_loop()

    app.state.clip_model = clip_model
    app.state.clip_preprocess = clip_preprocess
    app.state.image_loader = get_image_loader(IMAGE_MODE)
    app.state.embedding_queue = SerialTaskQueue()
    app.state.embedding_queue.start()
    app.state.loop = loop
    yield

app = FastAPI(lifespan=lifespan)

setup_exception_handler(app)

app.include_router(api_router)
