import cv2
import numpy as np
import torch

# MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
# STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

# def clip_preprocess_np(img: np.ndarray) -> torch.Tensor:
#     img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
#     tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
#     return (tensor - MEAN) / STD


# MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to('cuda')
# STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to('cuda')

# def clip_preprocess_np(img: np.ndarray, device='cuda') -> torch.Tensor:
#     img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
#     tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
#     tensor = tensor.to(device)
#     return (tensor - MEAN) / STD


# MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
# STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

# def clip_preprocess_np(images: list[np.ndarray]) -> torch.Tensor:
# 	resized = [cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC) for img in images]
# 	np_batch = np.stack(resized)
# 	tensor_batch = torch.from_numpy(np_batch).permute(0, 3, 1, 2).float() / 255.0
# 	return (tensor_batch - MEAN) / STD


MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to('cuda')
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to('cuda')

# def clip_preprocess_np(images: list[np.ndarray], device='cuda') -> torch.Tensor:
# 	resized = [cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC) for img in images]
# 	np_batch = np.stack(resized)
# 	tensor_batch = torch.from_numpy(np_batch).permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.0
# 	return (tensor_batch - MEAN) / STD


def resize_image(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)

async def clip_preprocess_np(images: list[np.ndarray], executor, device='cuda') -> torch.Tensor:
    """
    Resize 이미지만 executor에서 병렬 처리, 이후 GPU에서 정규화 및 텐서화.
    """
    import asyncio
    loop = asyncio.get_running_loop()

    # ✅ 1. resize만 병렬 처리
    resized = await asyncio.gather(*[
        loop.run_in_executor(executor, resize_image, img)
        for img in images
    ])

    # ✅ 2. 텐서 변환 및 GPU 이동 (직렬)
    np_batch = np.stack(resized)
    tensor_batch = torch.from_numpy(np_batch).permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.0

    # ✅ 3. 정규화
    return (tensor_batch - MEAN) / STD