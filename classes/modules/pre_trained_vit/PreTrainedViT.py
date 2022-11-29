from typing import Dict

import torch
import torch.nn as nn
from transformers import ViTFeatureExtractor, DeiTFeatureExtractor, BeitFeatureExtractor
# from transformers import ViTModel, DeiTModel, BeitModel
from transformers import ViTForImageClassification, DeiTForImageClassification, BeitForImageClassification


class PreTrainedViT(nn.Module):
    def __init__(self, network_params: Dict):
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

        self.feature_extractor, self.model = pretrained_models[network_params["model_type"]]

        self.feature_extractor = self.feature_extractor.from_pretrained(network_params["pretrained_model"])
        self.model = self.model.from_pretrained(
            network_params["pretrained_model"],
            num_labels=network_params["num_classes"]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # I know this doesn't go here, but I wanted to test
        # inputs = self.feature_extractor(images=x, return_tensors="pt")
        return self.model(x)
