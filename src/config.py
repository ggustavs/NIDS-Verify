"""Configuration for NIDS training pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path

_MLRUNS_DIR = Path("./mlruns").resolve()


@dataclass
class MLflowConfig:
    enabled: bool = True
    experiment_name: str = "NIDS_Adversarial_Training"
    tracking_uri: str = _MLRUNS_DIR.as_uri()
    artifact_location: str = str(_MLRUNS_DIR)


@dataclass
class ModelConfig:
    model_type: str = "small"
    onnx_model_dir: str = "models/onnx"
    input_size: int = field(init=False)


@dataclass
class TrainingConfig:
    epochs: int = 64
    steps_per_epoch: int = 4000
    learning_rate: float = 0.001
    pgd_epsilon: float = 0.1
    pgd_steps: int = 3
    pgd_alpha: float = 0.01
    # alfa from Flood et al.: task_loss weight in adversarial blend (1.0 = no adversarial)
    lambda_val: float = 0.7
    vehicle_property: str = "propertyGoodHTTP"


@dataclass
class DataConfig:
    data_dir: str = "data"
    batch_size: int = 16
    pkts_length: int = 10
    num_workers: int = field(default_factory=lambda: min(4, (os.cpu_count() or 2) // 2))


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_dir: str = "./logs"


@dataclass
class Config:
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self):
        self.model.input_size = 2 + self.data.pkts_length * 4


config = Config()
