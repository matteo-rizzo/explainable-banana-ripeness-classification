from typing import Dict

from classes.core.Model import Model
from classes.modules.cnn_img.ModelImagesCNN import ModelImagesCNN
from classes.modules.pre_trained_cnn.ModelPreTrainedCNN import ModelPreTrainedCNN
from classes.modules.transformer.ModelTransformer import ModelTransformer


class ModelFactory:
    models_map = {
        "cnn_img": ModelImagesCNN,
        "pre_trained_cnn": ModelPreTrainedCNN,
        "transformer": ModelTransformer,
    }

    def get(self, model_type: str, model_params: Dict) -> Model:
        if model_type not in self.models_map.keys():
            raise ValueError("Model {} is not implemented! \n Implemented models are: {}"
                             .format(model_type, list(self.models_map.keys())))
        return self.models_map[model_type](model_params)
