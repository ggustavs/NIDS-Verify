import property_driven_ml as pdml
import torch


class PropertyGoodHTTP(pdml.constraints.Constraint):
    """
    Implements first property from paper
    """

    def __init__(self, device: torch.device):
        super().__init__(device)
        ...


class ValidInput(pdml.constraints.preconditions.Precondition):
    """
    Helper precondition
    """

    def get_precondition(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x
