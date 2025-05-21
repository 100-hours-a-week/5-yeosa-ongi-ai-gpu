import torch

from app.model.InsightFace_PyTorch.config import im_size
from app.model.InsightFace_PyTorch.models import IRBlock, ResNet

device = "cuda" if torch.cuda.is_available() else "cpu"
_arcface_model = None


def load_arcface_model(weight_path: str = "app/model/insight-face-v3.pt"):
    """
    ArcFace 얼굴 임베딩 모델을 로드하거나 캐시된 모델을 반환합니다.

    지정된 경로의 가중치 파일을 로드하여 ResNet 기반 ArcFace 모델을 초기화하고,
    평가 모드로 설정한 후 캐싱하여 이후 재사용 시 초기화를 생략합니다.

    Args:
        weight_path (str): 모델 가중치 파일 경로. 'app/model/insight-face-v3.pt'.

    Returns:
        torch.nn.Module: 로드된 ArcFace 얼굴 임베딩 모델.

    """
    global _arcface_model
    if _arcface_model is None:
        model = ResNet(IRBlock, [3, 4, 23, 3], use_se=True, im_size=im_size)
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval().to(device)
        _arcface_model = model
    return _arcface_model
