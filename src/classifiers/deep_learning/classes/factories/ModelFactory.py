from typing import Dict

from src.classifiers.deep_learning.classes.core.Model import Model
from src.classifiers.deep_learning.classes.modules.cnn.ModelCNN import ModelCNN
from src.classifiers.deep_learning.classes.modules.mobilenet_v2.ModelMobileNetV2 import ModelMobileNetV2
from src.classifiers.deep_learning.classes.modules.pre_trained_vit.ModelPreTrainedViT import ModelPreTrainedVit


class ModelFactory:
    models_map = {
        "cnn": ModelCNN,
        "mobilenet_v2": ModelMobileNetV2,
        "pre_trained_vit": ModelPreTrainedVit
    }

    def get(self, model_type: str, model_params: Dict) -> Model:
        if model_type not in self.models_map.keys():
            raise ValueError(f"Model {model_type} is not implemented! "
                             f"\n Implemented models are: {list(self.models_map.keys())}")
        return self.models_map[model_type](model_params)
