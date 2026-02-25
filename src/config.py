"""
Configuration management for NIDS training pipeline
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LoggingConfig:
    """Logging configuration (file rotation, level, etc.)."""

    level: str = "INFO"
    log_dir: str = "./logs"
    # loguru handles format and console/file output automatically


@dataclass
class MLflowConfig:
    """MLflow experiment tracking configuration."""

    enabled: bool = True
    experiment_name: str = "NIDS_Adversarial_Training"
    tracking_uri: str | None = "file:./mlruns"
    artifact_location: str | None = "./mlruns"


@dataclass
class LightningConfig:
    """PyTorch Lightning trainer configuration."""

    # Checkpointing
    checkpoint_dir: str = "./models/checkpoints"
    save_top_k: int = 3
    save_last: bool = True

    # Early stopping
    enable_early_stopping: bool = True
    early_stop_patience: int = 5
    early_stop_min_delta: float = 0.0

    # Validation
    val_check_interval: float = 1.0  # Check validation every epoch

    # Learning rate scheduling
    enable_lr_scheduler: bool = False
    lr_scheduler_type: str = "reduce_on_plateau"

    # Progress/Logging
    enable_progress_bar: bool = True
    log_every_n_steps: int = 50
    num_sanity_val_steps: int = 0  # Skip sanity check for faster startup


@dataclass
class ModelConfig:
    """Model configuration"""

    input_size: int = 42
    model_type: str = "small"
    model_types: list[str] = field(
        default_factory=lambda: [
            "small",
            "mid",
            "mid2",
            "mid3",
            "mid4",
            "big",
            "big2",
            "big3",
            "big4",
            "massive",
        ]
    )
    initializer_seed: int = 42
    tf_model_dir: str = field(default="models/tf")
    onnx_model_dir: str = field(default="models/onnx")


@dataclass
class TrainingConfig:
    """Training configuration"""

    epochs: int = 64
    steps_per_epoch: int = 4000
    learning_rate: float = 0.001
    pgd_epsilon: float = 0.1
    pgd_steps: int = 3
    pgd_alpha: float = 0.01
    vehicle_property: str = "propertyGoodHTTP"
    lambda_val: float = 0.7


@dataclass
class DataConfig:
    """Data loading configuration."""

    data_dir: str = "data"
    batch_size: int = 16
    shuffle_buffer_size: int = 10000
    pos_train: str = "CIC2017"
    neg_train: str = "CIC2017"
    pos_test: str = "CIC2018"
    neg_test: str = "DetGenSSH"
    pkts_length: int = 10
    preprocess_dict: dict = field(
        default_factory=lambda: {
            "time_max": 50000000000,
            "iat_max": 5000000000,
            "size_max": 1000,
            "flag_max": 256,
        }
    )
    save: bool = False
    load: bool = True
    resample: str | None = None


@dataclass
class ExperimentConfig:
    """Experiment tracking and artifact configuration."""

    model_save_path: str = "./models"
    attack_name: str = "DoS2"


@dataclass
class Config:
    """Main configuration class combining all sub-configurations."""

    logging: LoggingConfig = field(default_factory=LoggingConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    lightning: LightningConfig = field(default_factory=LightningConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def __post_init__(self):
        """Initialize derived values and create directories."""
        # Calculate derived values
        self.model.input_size = 2 + self.data.pkts_length * 4

        # Ensure critical directories exist
        directories = [
            self.logging.log_dir,
            self.experiment.model_save_path,
            self.lightning.checkpoint_dir,
            self.model.tf_model_dir,
            self.model.onnx_model_dir,
            "mlruns",  # MLflow directory
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


# Global configuration instance
config = Config()
