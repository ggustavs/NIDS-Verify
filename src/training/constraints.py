import pandas as pd
import property_driven_ml as pdml
import torch


class propertyGoodHTTP(pdml.constraints.Constraint):
    def __init__(self, device: torch.device):
        super().__init__(device)
        ...

class validInput(pdml.constraints.preconditions.Precondition):
    def get_precondition(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x