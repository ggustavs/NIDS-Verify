"""
Streamlined logging utilities with MLflow integration
"""

import logging
import sys
from pathlib import Path
from typing import Any

import mlflow
import mlflow.tensorflow

from src.config import config


def setup_logging(level: str = None, experiment_name: str = None) -> None:
    """Setup logging configuration"""
    level = level or config.logging.level
    experiment_name = experiment_name or config.mlflow.experiment_name

    # Update config
    config.logging.level = level
    config.mlflow.experiment_name = experiment_name

    # Create log directory
    log_dir = Path(config.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(config.logging.format)

    # Console handler
    if config.logging.console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if config.logging.file_output:
        log_file = log_dir / f"{experiment_name.lower().replace(' ', '_')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    root_logger.info(f"Logging initialized - Level: {level}, Experiment: {experiment_name}")


class NIDSLogger:
    """Streamlined logger with MLflow integration"""

    def __init__(self, name: str = __name__):
        self.name = name
        self.logger = logging.getLogger(name)
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Configure MLflow"""
        if not config.mlflow.enabled:
            return

        try:
            if config.mlflow.tracking_uri:
                mlflow.set_tracking_uri(config.mlflow.tracking_uri)

            experiment_name = config.mlflow.experiment_name
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(
                    experiment_name, artifact_location=config.mlflow.artifact_location
                )
                self.info(f"Created MLflow experiment: {experiment_name}")
            else:
                self.info(f"Using existing MLflow experiment: {experiment_name}")

            mlflow.set_experiment(experiment_name)

        except Exception as e:
            self.error(f"Failed to setup MLflow: {e}")

    # MLflow operations
    def start_run(self, run_name: str | None = None, **kwargs):
        """Start MLflow run"""
        if not config.mlflow.enabled:
            return
        try:
            mlflow.start_run(run_name=run_name, **kwargs)
            self.info(f"Started MLflow run: {run_name or 'unnamed'}")
        except Exception as e:
            self.error(f"Failed to start MLflow run: {e}")

    def end_run(self):
        """End MLflow run"""
        if not config.mlflow.enabled:
            return
        try:
            mlflow.end_run()
            self.info("Ended MLflow run")
        except Exception as e:
            self.error(f"Failed to end MLflow run: {e}")

    def log_params(self, params: dict[str, Any]):
        """Log parameters to MLflow"""
        if not config.mlflow.enabled:
            return
        try:
            mlflow.log_params(params)
            self.debug(f"Logged parameters: {list(params.keys())}")
        except Exception as e:
            self.error(f"Failed to log parameters: {e}")

    def log_metrics(self, metrics: dict[str, float], step: int | None = None):
        """Log metrics to MLflow"""
        if not config.mlflow.enabled:
            return
        try:
            mlflow.log_metrics(metrics, step=step)
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            step_str = f" (step {step})" if step is not None else ""
            self.debug(f"Metrics{step_str}: {metrics_str}")
        except Exception as e:
            self.error(f"Failed to log metrics: {e}")

    def log_gradient_metrics(self, gradient_metrics: dict[str, float], step: int | None = None):
        """Log gradient-specific metrics to MLflow"""
        if not config.mlflow.enabled or not config.logging.gradient_mlflow_logging:
            return
        try:
            # Filter essential gradient metrics
            filtered_metrics = {
                k: v
                for k, v in gradient_metrics.items()
                if any(
                    x in k
                    for x in [
                        "grad_global_norm",
                        "grad_health_score",
                        "grad_vanishing_ratio",
                        "grad_exploding_ratio",
                    ]
                )
            }
            if filtered_metrics:
                mlflow.log_metrics(filtered_metrics, step=step)
                step_str = f" (step {step})" if step is not None else ""
                self.debug(f"Gradient metrics{step_str}: {list(filtered_metrics.keys())}")
        except Exception as e:
            self.error(f"Failed to log gradient metrics: {e}")

    def log_artifact(self, local_path: str, artifact_path: str | None = None):
        """Log artifact to MLflow"""
        if not config.mlflow.enabled:
            return
        try:
            mlflow.log_artifact(local_path, artifact_path)
            self.debug(f"Logged artifact: {local_path}")
        except Exception as e:
            self.error(f"Failed to log artifact: {e}")

    def register_model(
        self, model, model_name: str, description: str = None, tags: dict = None, **kwargs
    ):
        """Register model in MLflow Model Registry"""
        if not config.mlflow.enabled:
            return None
        try:
            # Log the model first
            model_info = mlflow.tensorflow.log_model(
                model,
                model_name,
                signature=kwargs.get("signature"),
                input_example=kwargs.get("input_example"),
                **{k: v for k, v in kwargs.items() if k not in ["signature", "input_example"]},
            )

            # Register the model
            registry_name = model_name.replace("/", "_").replace("\\", "_")
            model_version = mlflow.register_model(
                model_uri=model_info.model_uri, name=registry_name
            )

            # Update with description and tags
            if description or tags:
                from mlflow.tracking import MlflowClient

                client = MlflowClient()

                if description:
                    client.update_model_version(
                        name=registry_name, version=model_version.version, description=description
                    )

                if tags:
                    for key, value in tags.items():
                        client.set_model_version_tag(
                            name=registry_name,
                            version=model_version.version,
                            key=key,
                            value=str(value),
                        )

            self.info(f"Registered model '{registry_name}' version {model_version.version}")
            return model_version

        except Exception as e:
            self.error(f"Failed to register model: {e}")
            return None

    def transition_model_stage(self, model_name: str, version: str, stage: str):
        """Transition model to a specific stage"""
        if not config.mlflow.enabled:
            return
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient()

            client.transition_model_version_stage(name=model_name, version=version, stage=stage)

            self.info(f"Transitioned model '{model_name}' v{version} to {stage}")

        except Exception as e:
            self.error(f"Failed to transition model stage: {e}")

    # Configuration methods
    def configure_gradient_logging(
        self,
        enabled: bool = None,
        console: bool = None,
        csv: bool = None,
        mlflow: bool = None,
        log_frequency: int = None,
        mlflow_frequency: int = None,
        health_threshold: float = None,
        log_dir: str = None,
    ):
        """Configure gradient logging settings dynamically"""
        if enabled is not None:
            config.logging.gradient_logging_enabled = enabled
        if console is not None:
            config.logging.gradient_console_logging = console
        if csv is not None:
            config.logging.gradient_csv_logging = csv
        if mlflow is not None:
            config.logging.gradient_mlflow_logging = mlflow
        if log_frequency is not None:
            config.logging.gradient_log_frequency = log_frequency
        if mlflow_frequency is not None:
            config.logging.gradient_mlflow_frequency = mlflow_frequency
        if health_threshold is not None:
            config.logging.gradient_health_threshold = health_threshold
        if log_dir is not None:
            config.logging.gradient_log_dir = log_dir

        self.info(
            f"Gradient logging configured: enabled={config.logging.gradient_logging_enabled}, "
            f"console={config.logging.gradient_console_logging}, "
            f"csv={config.logging.gradient_csv_logging}, "
            f"mlflow={config.logging.gradient_mlflow_logging}"
        )

    def is_gradient_logging_enabled(self) -> bool:
        """Check if gradient logging is enabled"""
        return config.logging.gradient_logging_enabled

    # Standard logging methods
    def debug(self, message: str):
        self.logger.debug(message)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str, exc_info: bool = False):
        self.logger.error(message, exc_info=exc_info)

    def critical(self, message: str):
        self.logger.critical(message)


def get_logger(name: str = __name__) -> NIDSLogger:
    """Get a logger instance"""
    return NIDSLogger(name)
