from pydantic import BaseModel
from typing import Dict, List


# /embedding에서 사용
class ImageRequest(BaseModel):
    """
    이미지 파일명 목록을 전달하기 위한 요청 모델.

    Attributes:
        images (list[str]): 처리할 이미지 파일명 목록입니다.

    """

    images: list[str]