import torch.multiprocessing as mp

try:
    # 현재 설정된 시작 메서드를 확인하고, 없으면 'spawn'으로 설정
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn', force=True) # force=True 추가 권장
        print("DEBUG: multiprocessing start method set to 'spawn'.")
    else:
        # 이미 설정되어 있다면, 어떤 메서드로 설정되었는지 확인
        print(f"DEBUG: multiprocessing start method already set to '{mp.get_start_method()}'.")

except RuntimeError:
    print("DEBUG: RuntimeError when setting multiprocessing start method (possibly already set or unexpected).")
    pass

import os
import asyncio
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI

from app.api import api_router
from app.config.settings import IMAGE_MODE, MODEL_NAME, APP_ENV
from app.middleware.error_handler import setup_exception_handler
from app.model.clip_preprocess import clip_preprocess_np
from app.model.arcface_loader import load_arcface_model
from app.model.yolo_detector_loader import load_yolo_detector
from app.utils.image_loader import (
    get_image_loader,
    GCSImageLoader,
    S3ImageLoader,
)

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
    clip_preprocess = clip_preprocess_np
    loop = asyncio.get_running_loop()

    app.state.clip_preprocess = clip_preprocess
    app.state.image_loader = get_image_loader(IMAGE_MODE)
    app.state.loop = loop
    if IMAGE_MODE == IMAGE_MODE.S3:
        if isinstance(app.state.image_loader, S3ImageLoader):
            await app.state.image_loader.init_client()
    yield

    # 서버 종료 시 리소스 해제
    if IMAGE_MODE == IMAGE_MODE.GCS:
        if isinstance(app.state.image_loader, GCSImageLoader):
            await app.state.image_loader.client.close()

    if IMAGE_MODE == IMAGE_MODE.S3:
        if isinstance(app.state.image_loader, S3ImageLoader):
            await app.state.image_loader.close_client()

app = FastAPI(lifespan=lifespan)

setup_exception_handler(app)

app.include_router(api_router)
