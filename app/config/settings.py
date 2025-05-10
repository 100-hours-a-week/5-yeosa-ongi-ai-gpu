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

    """

    LOCAL = "local"
    GCS = "gcs"


mode_str = os.getenv("IMAGE_MODE", "gcs")

try:
    IMAGE_MODE = ImageMode(mode_str)
except ValueError:
    raise ValueError(
        f"잘못된 IMAGE_MODE: {mode_str}. 선택 가능한 IMAGE_MODE: {[m.value for m in ImageMode]}"
    )
