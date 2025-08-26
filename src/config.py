"""
Configuration management for NIDS training pipeline
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LoggingConfig:
    """Logging configuration"""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_dir: str = "./logs"
    console_output: bool = True
    file_output: bool = True

    # Gradient logging configuration
    gradient_logging_enabled: bool = True
    gradient_console_logging: bool = True
    gradient_csv_logging: bool = True
    gradient_mlflow_logging: bool = True
    gradient_log_frequency: int = 100  # Log to console every N steps
    gradient_mlflow_frequency: int = 25  # Log to MLflow every N steps
    gradient_health_threshold: float = 0.7  # Log immediately if health score < threshold
    gradient_log_dir: str = "logs/gradients"


@dataclass
class MLflowConfig:
    """MLflow configuration"""

    enabled: bool = True
    experiment_name: str = "NIDS_Adversarial_Training"
    tracking_uri: str | None = "file:./mlruns"  # Local file tracking
    artifact_location: str | None = "./mlruns"  # Local artifacts
    run_name_prefix: str = "nids_run"


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


@dataclass
class DataConfig:
    """Data processing configuration"""

    data_dir: str = "data"
    batch_size: int = 32
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
    """Experiment configuration"""

    model_save_path: str = "./models"
    attack_name: str = "DoS2"
    save_adversarial_data: bool = True
    adversarial_data_dir: str = "./adv_data_analysis"


@dataclass
class Config:
    """Main configuration class"""

    logging: LoggingConfig = field(default_factory=LoggingConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def __post_init__(self):
        """Initialize derived values and create directories"""
        # Calculate derived values
        self.model.input_size = 2 + self.data.pkts_length * 4

        # Ensure critical directories exist
        directories = [
            self.logging.log_dir,
            self.logging.gradient_log_dir,
            self.experiment.model_save_path,
            self.experiment.adversarial_data_dir,
            self.model.tf_model_dir,
            self.model.onnx_model_dir,
            "mlruns",  # MLflow directory
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


# Global configuration instance
config = Config()
