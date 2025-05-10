import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

from dotenv import load_dotenv
from google.cloud import storage
from PIL import Image
from starlette.concurrency import run_in_threadpool

from app.config.settings import ImageMode

load_dotenv()

LOCAL_IMG_PATH_raw = os.getenv("LOCAL_IMG_PATH")
BUCKET_NAME_raw = os.getenv("BUCKET_NAME")
GCP_KEY_PATH_raw = os.getenv("GCP_KEY_PATH")

if LOCAL_IMG_PATH_raw is None:
    raise EnvironmentError("LOCAL_IMG_PATH은 .env에 설정되어야 합니다.")
if BUCKET_NAME_raw is None:
    raise EnvironmentError("BUCKET_NAME은 .env에 설정되어야 합니다.")
if GCP_KEY_PATH_raw is None:
    raise EnvironmentError("GCP_KEY_PATH은 .env에 설정되어야 합니다.")

# 타입이 str로 확정됨 (mypy 추론 가능)
LOCAL_IMG_PATH: str = LOCAL_IMG_PATH_raw
BUCKET_NAME: str = BUCKET_NAME_raw
GCP_KEY_PATH: str = GCP_KEY_PATH_raw

class BaseImageLoader(ABC):
    """
    이미지 로더의 추상 베이스 클래스.

    모든 이미지 로더는 `load_images` 메서드를 구현해야 합니다.
    """

    @abstractmethod
    async def load_images(self, filenames: list[str]) -> list[Image.Image]:
        """
        주어진 이미지 파일 이름 리스트에 대해 이미지를 로드합니다.

        Args:
            filenames (list[str]): 로드할 이미지 파일 이름 목록

        Returns:
            list[Image.Image]: 로드된 PIL 이미지 객체 리스트

        """
        pass


class LocalImageLoader(BaseImageLoader):
    """로컬 파일 시스템에서 이미지를 로드하는 클래스입니다."""

    def __init__(self, image_dir: str = LOCAL_IMG_PATH):
        """
        Args:
            image_dir (str): 이미지가 저장된 로컬 디렉토리 경로

        """
        self.image_dir = image_dir

    async def load_images(self, filenames: list[str]) -> list[Image.Image]:
        """
        비동기적으로 로컬 이미지들을 로드합니다.

        Args:
            filenames (list[str]): 이미지 파일 이름 리스트

        Returns:
            list[Image.Image]: 로드된 이미지 리스트

        """
        return await run_in_threadpool(
            lambda: list(
                Image.open(os.path.join(self.image_dir, filename)).convert(
                    "RGB"
                )
                for filename in filenames
            )
        )


class GCSImageLoader(BaseImageLoader):
    """Google Cloud Storage(GCS)에서 이미지를 로드하는 클래스입니다."""

    def __init__(
        self, bucket_name: str = BUCKET_NAME, key_path: str = GCP_KEY_PATH
    ):
        """
        Args:
            bucket_name (str): GCS 버킷 이름
            key_path (str): 서비스 계정 키 경로 (.json)

        """
        self.client = storage.Client.from_service_account_json(key_path)
        self.bucket = self.client.bucket(bucket_name)
        self.executor = ThreadPoolExecutor(max_workers=10)

    def _download(self, file_name: str) -> Image.Image:
        """
        GCS에서 단일 이미지를 다운로드하고 RGB로 변환합니다.

        Args:
            file_name (str): GCS 내 파일 이름

        Returns:
            Image.Image: 로드된 PIL 이미지

        """
        blob = self.bucket.blob(file_name)
        image_bytes = blob.download_as_bytes()
        return Image.open(BytesIO(image_bytes)).convert("RGB")

    async def load_images(self, filenames: list[str]) -> list[Image.Image]:
        """
        비동기적으로 GCS에서 이미지를 병렬로 다운로드합니다.

        Args:
            filenames (list[str]): GCS 상의 이미지 파일 이름 리스트

        Returns:
            list[Image.Image]: 로드된 이미지 리스트

        """
        return await run_in_threadpool(
            lambda: list(self.executor.map(self._download, filenames))
        )


def get_image_loader(mode: ImageMode) -> BaseImageLoader:
    """
    이미지 로딩 모드를 기반으로 적절한 이미지 로더 인스턴스를 반환합니다.

    Args:
        mode (ImageMode): 이미지 로딩 방식 (로컬 또는 GCS)

    Returns:
        BaseImageLoader: 선택된 이미지 로더 인스턴스

    """
    if mode == ImageMode.GCS:
        return GCSImageLoader()
    return LocalImageLoader()
