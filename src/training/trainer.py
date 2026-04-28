"""PyTorch Lightning Trainer factory."""

import pytorch_lightning as pl
from loguru import logger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from src.config import config

_CHECKPOINT_DIR = "./models/checkpoints"
_SAVE_TOP_K = 3
_EARLY_STOP_PATIENCE = 5


def get_callbacks(model_type: str, training_type: str) -> list[pl.Callback]:
    return [
        ModelCheckpoint(
            dirpath=_CHECKPOINT_DIR,
            filename=f"nids_{model_type}_{training_type}_" + "{epoch:02d}_{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=_SAVE_TOP_K,
            save_last=True,
        ),
        EarlyStopping(monitor="val_loss", patience=_EARLY_STOP_PATIENCE, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
    ]


def create_trainer(
    model_type: str,
    training_type: str = "base",
    epochs: int | None = None,
    steps_per_epoch: int | None = None,
    experiment_name: str | None = None,
    mlflow_run_id: str | None = None,
    progress_bar: bool = True,
) -> pl.Trainer:
    epochs = epochs or config.training.epochs
    steps_per_epoch = steps_per_epoch or config.training.steps_per_epoch
    experiment_name = experiment_name or config.mlflow.experiment_name

    if config.mlflow.enabled:
        trainer_logger = MLFlowLogger(
            experiment_name=experiment_name,
            tracking_uri=config.mlflow.tracking_uri,
            artifact_location=config.mlflow.artifact_location,
            run_id=mlflow_run_id,
            run_name=f"{training_type}_{model_type}_{epochs}ep" if mlflow_run_id is None else None,
        )
    else:
        trainer_logger = False

    trainer = pl.Trainer(
        logger=trainer_logger,
        callbacks=get_callbacks(model_type, training_type),
        max_epochs=epochs,
        limit_train_batches=steps_per_epoch if steps_per_epoch > 0 else None,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=50,
        enable_progress_bar=progress_bar,
        enable_model_summary=progress_bar,
        deterministic=True,
    )

    logger.info(f"Trainer: model={model_type} mode={training_type} epochs={epochs}")
    return trainer
