import torch
from torch import nn


class CriterionFactory:
    criteria_map = {
        "NLLLoss": nn.NLLLoss(),
        "CrossEntropyLoss": nn.CrossEntropyLoss()
    }

    def get(self, criterion_type: str) -> torch.nn.modules.loss:
        if criterion_type not in self.criteria_map.keys():
            raise ValueError(f"Criterion for {criterion_type} is not implemented! "
                             f"\n Supported criteria are: {list(self.criteria_map.keys())}")
        return self.criteria_map[criterion_type]
