import logging
from typing import Tuple, Optional, Any

import clip
import torch
import torch.nn as nn

from app.model.clip_preprocess import clip_preprocess_np

logger = logging.getLogger(__name__)

# 전역 모델 및 전처리기 인스턴스
_model: Optional[nn.Module] = None
_preprocess: Optional[Any] = None


def load_clip_model(
    model_name: str = "ViT-B/32",
    device: Optional[str] = None,
) -> Tuple[nn.Module, Any]:
    """
    CLIP 모델과 전처리기를 로드하거나 생성합니다.
    
    싱글톤 패턴을 사용하여 모델과 전처리기 인스턴스를 재사용합니다.
    GPU가 사용 가능한 경우 GPU를 사용하고, 그렇지 않은 경우 CPU를 사용합니다.
    
    Args:
        model_name: 사용할 CLIP 모델의 이름 (기본값: "ViT-B/32")
        device: 모델을 로드할 디바이스 (기본값: None, 자동 선택)
        
    Returns:
        Tuple[nn.Module, Any]: (CLIP 모델, 전처리기) 튜플
        
    Raises:
        RuntimeError: CLIP 모델 로드 실패 시
    """
    global _model, _preprocess
    
    # 1. 디바이스 설정
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug(
            "디바이스 자동 선택",
            extra={"device": str(device)},
        )
    
    # 2. 모델 및 전처리기 로드
    if _model is None or _preprocess is None:
        try:
            logger.info(
                "CLIP 모델 로드 시작",
                extra={
                    "model_name": model_name,
                    "device": str(device),
                },
            )
            _model, _ = clip.load(model_name, device=device)
            _preprocess = clip_preprocess_np
            _model.eval()
            logger.info("CLIP 모델 로드 완료")
        except Exception as e:
            logger.error(
                "CLIP 모델 로드 실패",
                extra={"error": str(e)},
            )
            raise RuntimeError(f"CLIP 모델 로드 실패: {str(e)}")
    else:
        logger.debug("기존 CLIP 모델 인스턴스 재사용")
    
    return _model, _preprocess
