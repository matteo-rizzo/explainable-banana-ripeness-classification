import torch
import torch.nn as nn
from torchvision import transforms


class MobileNetV2(nn.Module):

    def __init__(self):
        super().__init__()
        self.__mobilenet_v2 = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        self.__classifier = nn.Linear(1000, 4)
        self.__normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.__normalize(x)
        x = self.__mobilenet_v2(x)
        x = self.__classifier(x)
        return x
