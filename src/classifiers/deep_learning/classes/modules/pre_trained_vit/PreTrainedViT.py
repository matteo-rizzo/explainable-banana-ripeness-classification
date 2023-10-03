from typing import Dict

import torch
import torch.nn as nn
from torchvision.transforms import transforms
from transformers import ViTForImageClassification, DeiTForImageClassification, BeitForImageClassification


class PreTrainedViT(nn.Module):
    def __init__(self, network_params: Dict):
        super().__init__()

        pretrained_models = {
            # --- Vision Transformers ---
            # Base
            "ViT": ViTForImageClassification,
            # Data-efficient Image Transformers
            "DeIT": DeiTForImageClassification,
            # BERT pre-training of Image Transformers
            "BeiT": BeitForImageClassification,
            # A method for self-supervised training of Vision Transformers
            "DINO": None,
            # Masked Autoencoders
            "MAE": None,
            # Swin Transformers
            "Swin": None,
            "SwinV2": None,
        }

        self.model = pretrained_models[network_params["model_type"]]

        self.model = self.model.from_pretrained(
            network_params["pretrained_model"],
            num_labels=network_params["num_classes"]
        )

        self.__normalization = network_params["normalization"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Normalization ---
        # This variant normalizes here to use faster gpu matrix operations
        mean, std = self.__normalization.values()
        normalization = transforms.Normalize(mean=mean, std=std)
        x = normalization(x)
        # ---------------------
        return self.model(x).logits
