import cv2
import numpy as np
import torch

# mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
# std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

# def clip_preprocess_np(img: np.ndarray) -> torch.Tensor:
#     img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
#     tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
#     return (tensor - mean) / std


# MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to('cuda')
# STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to('cuda')

# def clip_preprocess_np(img: np.ndarray, device='cuda') -> torch.Tensor:
#     img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
#     tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
#     tensor = tensor.to(device)
#     return (tensor - MEAN) / STD


MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to('cuda')
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to('cuda')

def clip_preprocess_np(images: list[np.ndarray], device='cuda') -> torch.Tensor:
	resized = [cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC) for img in images]
	np_batch = np.stack(resized)
	tensor_batch = torch.from_numpy(np_batch).permute(0, 3, 1, 2).float() / 255.0
	return (tensor_batch - MEAN) / STD