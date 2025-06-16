import os
from enum import Enum

from dotenv import load_dotenv

load_dotenv()


class ImageMode(str, Enum):
    """
    이미지를 어디서 로드할지 선택하는 모드입니다.

    Attributes:
        LOCAL: 로컬 파일 시스템에서 이미지를 로드합니다.
        GCS: Google Cloud Storage에서 이미지를 로드합니다.
        S3: AWS S3에서 이미지를 로드합니다

    """

    LOCAL = "local"
    GCS = "gcs"
    S3 = "s3"


class ModelName(str, Enum):
    """
    CLIP 모델 이름을 선택하는 모드입니다.

    Attributes:
        ViT-B/32: ViT-B/32 모델을 사용합니다.
        ViT-L/14: ViT-L/14 모델을 사용합니다.

    """

    ViT_B_32 = "ViT-B/32"
    ViT_L_14 = "ViT-L/14"


class AppEnv(str, Enum):
    """
    이미지를 어디서 로드할지 선택하는 모드입니다.

    Attributes:
        LOCAL: 로컬 파일 시스템에서 이미지를 로드합니다.
        GCS: Google Cloud Storage에서 이미지를 로드합니다.
        S3: AWS S3에서 이미지를 로드합니다

    """

    PROD = "prod"
    DEV = "dev"


mode_str = os.getenv("IMAGE_MODE", "s3")
app_env = os.getenv("APP_ENV", "prod")
MODEL_NAME = os.getenv("MODEL_NAME", "ViT-B/32")

try:
    IMAGE_MODE = ImageMode(mode_str)
except ValueError:
    raise ValueError(
        f"잘못된 IMAGE_MODE: {mode_str}. 선택 가능한 IMAGE_MODE: {[m.value for m in ImageMode]}"
    )

try:
    MODEL_NAME = ModelName(MODEL_NAME)
except ValueError:
    raise ValueError(
        f"잘못된 MODEL_NAME: {MODEL_NAME}. 선택 가능한 MODEL_NAME: {[m.value for m in ModelName]}"
    )

MODEL_BASE_PATH = f"app/model/{MODEL_NAME.value}"
CATEGORY_FEATURES_FILENAME = "category_features.pt"
QUALITY_FEATURES_FILENAME = "quality_features.pt"
AESTHETIC_REGRESSOR_FILENAME = "aesthetic_regressor.pth"

try:
    APP_ENV = AppEnv(app_env)
except ValueError:
    raise ValueError(
        f"잘못된 APP_ENV: {app_env}. 선택 가능한 APP_ENV: {[m.value for m in AppEnv]}"
    )
