from typing import Dict

import torch
import torch.nn as nn
from transformers import ViTFeatureExtractor, DeiTFeatureExtractor, BeitFeatureExtractor
# from transformers import ViTModel, DeiTModel, BeitModel
from transformers import ViTForImageClassification, DeiTForImageClassification, BeitForImageClassification


class PreTrainedViT(nn.Module):
    def __init__(self, network_params: Dict, activation: bool = True):
        super().__init__()

        pretrained_models = {
            # --- Vision Transformers ---
            # Base
            "ViT": (ViTFeatureExtractor, ViTForImageClassification),
            # Data-efficient Image Transformers
            "DeIT": (DeiTFeatureExtractor, DeiTForImageClassification),
            # BERT pre-training of Image Transformers
            "BeiT": (BeitFeatureExtractor, BeitForImageClassification),
            # A method for self-supervised training of Vision Transformers
            "DINO": None,
            # Masked Autoencoders
            "MAE": None,
            # Swin Transformers
            "Swin": None,
            "SwinV2": None,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.__model(x)
