"""Model architectures for NIDS (PyTorch)"""
import torch
from loguru import logger
from torch import nn

from src.config import config


class ModelFactory:
    """Factory for creating different model architectures"""

    def __init__(self, input_size: int, seed: int = 42):
        self.input_size = input_size
        self.seed = seed or config.model.initializer_seed
        torch.manual_seed(self.seed)

    def create_model(self, model_type: str) -> nn.Module:
        """Create a model based on the specified type"""
        model_creators = {
            'small': self._create_small_model,
            'mid': self._create_mid_model,
            'mid2': self._create_mid2_model,
            'mid3': self._create_mid3_model,
            'mid4': self._create_mid4_model,
            'big': self._create_big_model,
            'big2': self._create_big2_model,
            'big3': self._create_big3_model,
            'big4': self._create_big4_model,
            'massive': self._create_massive_model,
        }

        if model_type not in model_creators:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(model_creators.keys())}")

        logger.info(f"Creating {model_type} model with input size {self.input_size}")
        model = model_creators[model_type]()
        return model

    def _create_small_model(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def _create_mid_model(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def _create_mid2_model(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def _create_mid3_model(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def _create_mid4_model(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def _create_big_model(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def _create_big2_model(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def _create_big3_model(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def _create_big4_model(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.input_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def _create_massive_model(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.input_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )


def create_model(input_size: int, model_type: str) -> nn.Module:
    """
    Create a model instance

    Args:
        input_size: Size of input features
        model_type: Type of model to create

    Returns:
        PyTorch model instance
    """
    factory = ModelFactory(input_size)
    model = factory.create_model(model_type)

    return model
