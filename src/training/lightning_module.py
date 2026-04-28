"""PyTorch Lightning module for NIDS model training."""

from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from src.attacks.pgd import generate_pgd_adversarial_examples
from src.config import config

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VEHICLE_SPEC_PATH = _REPO_ROOT / "vehicle_specifications" / "global.vcl"


class NIDSLightningModule(pl.LightningModule):
    """Supports three training modes: base, adversarial, vehicle."""

    def __init__(
        self,
        model: nn.Module,
        training_type: str = "base",
        attack_rects: list[np.ndarray] | None = None,
        attack_pattern: str = "hulk",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "attack_rects"])

        self.model = model
        self.training_type = training_type
        self.attack_rects = attack_rects
        self.attack_pattern = attack_pattern

        self.loss_fn = nn.CrossEntropyLoss()
        self.constraint_loss_fn = None
        self.lambda_val = config.training.lambda_val

        if training_type == "vehicle":
            from vehicle_lang.loss.pytorch import load_specification

            declarations, _minimise = load_specification(_VEHICLE_SPEC_PATH)
            self.constraint_loss_fn = declarations[config.training.vehicle_property]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=config.training.learning_rate)

    def training_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        if self.training_type == "adversarial":
            metrics = self._adversarial_train_step(x_batch, y_batch)
        elif self.training_type == "vehicle":
            metrics = self._vehicle_train_step(x_batch, y_batch)
        else:
            metrics = self._base_train_step(x_batch, y_batch)
        self.log("train_loss", metrics["loss"], on_epoch=True, on_step=False)
        self.log("train_accuracy", metrics["accuracy"], on_epoch=True, on_step=False)
        return metrics["loss"]

    def validation_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        logits = self.model(x_batch)
        loss = self.loss_fn(logits, y_batch.long())
        accuracy = (torch.argmax(logits, dim=1) == y_batch).float().mean().item()
        self.log("val_loss", loss, on_epoch=True, on_step=False)
        self.log("val_accuracy", accuracy, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        logits = self.model(x_batch)
        loss = self.loss_fn(logits, y_batch.long())
        accuracy = (torch.argmax(logits, dim=1) == y_batch).float().mean().item()
        self.log("test_loss", loss, on_epoch=True, on_step=False)
        self.log("test_accuracy", accuracy, on_epoch=True, on_step=False)
        return loss

    def _base_train_step(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> dict[str, Any]:
        logits = self.model(x_batch)
        loss = self.loss_fn(logits, y_batch.long())
        accuracy = (torch.argmax(logits, dim=1) == y_batch).float().mean().item()
        return {"loss": loss, "accuracy": float(accuracy)}

    def _adversarial_train_step(
        self, x_batch: torch.Tensor, y_batch: torch.Tensor
    ) -> dict[str, Any]:
        x_adv = generate_pgd_adversarial_examples(
            self.model,
            x_batch,
            y_batch,
            attack_rects=self.attack_rects,
            epsilon=config.training.pgd_epsilon,
            num_steps=config.training.pgd_steps,
            step_size=config.training.pgd_alpha,
            attack_pattern=self.attack_pattern,
        )
        x_mixed = torch.cat([x_batch, x_adv.detach()], dim=0)
        y_mixed = torch.cat([y_batch, y_batch], dim=0)
        indices = torch.randperm(x_mixed.size(0), device=self.device)
        x_mixed = x_mixed[indices]
        y_mixed = y_mixed[indices]
        logits = self.model(x_mixed)
        loss = self.loss_fn(logits, y_mixed.long())
        accuracy = (torch.argmax(logits, dim=1) == y_mixed).float().mean().item()
        return {"loss": loss, "accuracy": float(accuracy)}

    def _vehicle_train_step(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> dict[str, Any]:
        logits = self.model(x_batch)
        task_loss = self.loss_fn(logits, y_batch.long())
        # Vehicle property is universally quantified over inputs; it samples internally.
        constraint_loss = self.constraint_loss_fn(self.model)
        loss = task_loss * self.lambda_val + constraint_loss * (1 - self.lambda_val)
        accuracy = (torch.argmax(logits, dim=1) == y_batch).float().mean().item()
        return {"loss": loss, "accuracy": float(accuracy)}
