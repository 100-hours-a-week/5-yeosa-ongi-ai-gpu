import clip
import torch

_model = None
_preprocess = None


def load_clip_model(model_name="ViT-L/14", device="cuda"):
    global _model, _preprocess
    if _model is None or _preprocess is None:
        _model, _preprocess = clip.load(model_name, device=device)
        _model.eval()
    return _model, _preprocess
