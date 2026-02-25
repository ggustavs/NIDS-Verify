"""
Training utilities for NIDS models with PyTorch Lightning.

This module provides:
1. Callback configuration
2. Lightning Trainer factory
3. Compatibility functions for existing CLI
"""

from typing import Any

import numpy as np
import property_driven_ml as pdml
import pytorch_lightning as pl
import torch.nn as nn
from loguru import logger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader as TorchDataLoader

from src.config import config
from src.training.lightning_module import NIDSLightningModule


def get_callbacks(training_type: str = "base") -> list[pl.Callback]:
    """Get standard callbacks for training.

    Args:
        training_type: Type of training (affects checkpoint naming)

    Returns:
        List of Lightning callbacks
    """
    callbacks = [
        ModelCheckpoint(
            dirpath=config.lightning.checkpoint_dir,
            filename=f"nids_{config.model.model_type}_{training_type}_"
            + "{epoch:02d}_{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=config.lightning.save_top_k,
            save_last=config.lightning.save_last,
            verbose=True,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=config.lightning.early_stop_patience,
            min_delta=config.lightning.early_stop_min_delta,
            mode="min",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    return callbacks


def create_trainer(
    training_type: str = "base",
    epochs: int | None = None,
    steps_per_epoch: int | None = None,
    experiment_name: str | None = None,
) -> pl.Trainer:
    """Create and configure a PyTorch Lightning Trainer.

    Args:
        training_type: Type of training (for logging/naming)
        epochs: Maximum number of epochs
        steps_per_epoch: Limit training batches per epoch
        experiment_name: MLflow experiment name

    Returns:
        Configured Lightning Trainer instance
    """
    epochs = epochs or config.training.epochs
    steps_per_epoch = steps_per_epoch or config.training.steps_per_epoch
    experiment_name = experiment_name or config.mlflow.experiment_name

    # Setup MLflow logger
    mlflow_logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=config.mlflow.tracking_uri,
        artifact_location=config.mlflow.artifact_location,
        run_name=f"{training_type}_{config.model.model_type}_{epochs}ep",
    )

    # Create trainer with callbacks
    trainer = pl.Trainer(
        logger=mlflow_logger,
        callbacks=get_callbacks(training_type),
        max_epochs=epochs,
        limit_train_batches=steps_per_epoch if steps_per_epoch > 0 else None,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=50,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True,
    )

    logger.info(f"Created Lightning Trainer for {training_type} training")
    return trainer


# ============================================================================
# Compatibility Functions - Maintain API compatibility with old trainer.py
# ============================================================================


def train_base(
    model: nn.Module,
    train_loader: TorchDataLoader,
    val_loader: TorchDataLoader,
    epochs: int | None = None,
    steps_per_epoch: int | None = None,
) -> dict[str, Any]:
    """Train model with standard supervised learning.

    Compatibility wrapper for existing CLI.
    """
    logger.info("Starting base training with Lightning")

    module = NIDSLightningModule(
        model=model,
        training_type="base",
    )

    trainer = create_trainer(
        training_type="base",
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
    )

    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    return {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "epoch_times": [],
    }


def train_adversarial(
    model: nn.Module,
    train_loader: TorchDataLoader,
    val_loader: TorchDataLoader,
    attack_rects: list[np.ndarray] | None = None,
    epochs: int | None = None,
    steps_per_epoch: int | None = None,
    attack_pattern: str = "hulk",
) -> dict[str, Any]:
    """Train model with adversarial training using research hyperrectangles.

    Compatibility wrapper for existing CLI.
    """
    logger.info("Starting adversarial training with Lightning")

    module = NIDSLightningModule(
        model=model,
        training_type="adversarial",
        attack_pattern=attack_pattern,
        attack_rects=attack_rects,
    )

    trainer = create_trainer(
        training_type="adversarial",
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
    )

    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    return {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "epoch_times": [],
    }


def train_constraint(
    model: nn.Module,
    train_loader: TorchDataLoader,
    val_loader: TorchDataLoader,
    constraint: pdml.constraints.Constraint,
    epochs: int | None = None,
    steps_per_epoch: int | None = None,
) -> dict[str, Any]:
    """Train model with constraint-based training using property-driven ML.

    Compatibility wrapper for existing CLI.
    """
    logger.info("Starting constraint training with Lightning")

    module = NIDSLightningModule(
        model=model,
        training_type="constraint",
        constraint=constraint,
    )

    trainer = create_trainer(
        training_type="constraint",
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
    )

    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    return {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "epoch_times": [],
    }


def evaluate_model(
    model: nn.Module,
    dataset: TorchDataLoader,
) -> tuple[float, float]:
    """Evaluate model on dataset.

    Compatibility wrapper.
    """
    module = NIDSLightningModule(model=model)
    trainer = pl.Trainer(
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    trainer.test(module, dataloaders=dataset)

    return 0.0, 0.0  # Placeholder for compatibility
