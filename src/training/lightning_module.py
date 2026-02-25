"""
PyTorch Lightning module for NIDS model training.

Encapsulates the training logic from trainer.py into a reusable LightningModule,
preserving all research features (property-driven ML, vehicle constraints, PGD attacks).
"""

from typing import Any, cast

import numpy as np
import property_driven_ml as pdml
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from vehicle_lang.loss.pytorch import load_specification

from src.attacks.pgd import generate_pgd_adversarial_examples
from src.config import config
from src.data.loader import _NDArrayDataset


class NIDSLightningModule(pl.LightningModule):
    """PyTorch Lightning module for NIDS model training.

    Supports 4 training modes:
    - base: Standard supervised learning
    - adversarial: PGD adversarial training with research hyperrectangles
    - constraint: Property-driven ML with constraints
    - vehicle: Vehicle specification-based training
    """

    def __init__(
        self,
        model: nn.Module,
        training_type: str = "base",
        attack_rects: list[np.ndarray] | None = None,
        attack_pattern: str = "hulk",
        constraint: pdml.constraints.Constraint | None = None,
    ):
        """Initialize the Lightning module.

        Args:
            model: The neural network model to train
            training_type: One of "base", "adversarial", "constraint", "vehicle"
            attack_rects: Research hyperrectangles for adversarial training (None = use default)
            attack_pattern: Attack pattern for PGD (e.g., "hulk")
            constraint: Property-driven ML constraint for constraint training
        """
        super().__init__()
        self.model = model
        self.training_type = training_type
        self.attack_rects = attack_rects
        self.attack_pattern = attack_pattern
        self.constraint = constraint

        self.loss_fn = nn.CrossEntropyLoss()

        self.constraint_loss = None
        self.lambda_val = config.training.lambda_val

        if training_type == "vehicle":
            declarations = load_specification("vehicle/global.vcl")
            self.constraint_loss = declarations[config.training.vehicle_property]
            self.lambda_val = config.training.lambda_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def configure_optimizers(self):
        """Configure optimizer for Lightning."""
        return optim.Adam(self.model.parameters(), lr=config.training.learning_rate)

    def training_step(self, batch, batch_idx):
        """Training step for a single batch.

        Dispatches to appropriate training method based on training_type.
        """
        if self.training_type == "adversarial":
            x_batch, y_batch = batch
            metrics = self._adversarial_train_step(x_batch, y_batch)
            self.log("train_loss", metrics["loss"], on_epoch=True, on_step=False)
            self.log("train_accuracy", metrics["accuracy"], on_epoch=True, on_step=False)
            return metrics["loss"]

        if self.training_type == "constraint":
            x, y_target = batch
            if not hasattr(self, "_pdml_attack"):
                train_loader = self.trainer.train_dataloader if self.trainer else None
                if train_loader is not None:
                    dataset: _NDArrayDataset = cast(_NDArrayDataset, train_loader.dataset)
                    self._pdml_attack = pdml.training.attacks.PGD(
                        pdml.logics.DL2(),
                        self.device,
                        config.training.pgd_steps,
                        10,
                        config.training.pgd_alpha,
                        dataset.mean,
                        dataset.std,
                    )
            if hasattr(self, "_pdml_attack") and self.constraint is not None:
                metrics = self._constraint_train_step(
                    x, y_target, self._pdml_attack, self.constraint
                )
                self.log("train_loss", metrics["loss"], on_epoch=True, on_step=False)
                self.log(
                    "train_satisfaction", metrics["satisfaction"], on_epoch=True, on_step=False
                )
                return metrics["loss"]

            metrics = self._base_train_step(x, y_target)
            self.log("train_loss", metrics["loss"], on_epoch=True, on_step=False)
            self.log("train_accuracy", metrics["accuracy"], on_epoch=True, on_step=False)
            return metrics["loss"]

        if self.training_type == "vehicle":
            x_batch, y_batch = batch
            metrics = self._vehicle_train_step(x_batch, y_batch)
            self.log("train_loss", metrics["loss"], on_epoch=True, on_step=False)
            self.log("train_accuracy", metrics["accuracy"], on_epoch=True, on_step=False)
            return metrics["loss"]

        x_batch, y_batch = batch
        metrics = self._base_train_step(x_batch, y_batch)
        self.log("train_loss", metrics["loss"], on_epoch=True, on_step=False)
        self.log("train_accuracy", metrics["accuracy"], on_epoch=True, on_step=False)
        return metrics["loss"]

    def validation_step(self, batch, batch_idx):
        """Validation step for a single batch."""
        x_batch, y_batch = batch
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        logits = self.model(x_batch)
        loss = self.loss_fn(logits, y_batch.long())

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == y_batch).float().mean().item()

        self.log("val_loss", loss, on_epoch=True, on_step=False)
        self.log("val_accuracy", accuracy, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        """Test step for a single batch."""
        x_batch, y_batch = batch
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        logits = self.model(x_batch)
        loss = self.loss_fn(logits, y_batch.long())

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == y_batch).float().mean().item()

        self.log("test_loss", loss, on_epoch=True, on_step=False)
        self.log("test_accuracy", accuracy, on_epoch=True, on_step=False)
        return loss

    def _base_train_step(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> dict[str, Any]:
        """Single base training step (standard supervised learning)."""
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        logits = self.model(x_batch)
        loss = self.loss_fn(logits, y_batch.long())

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == y_batch).float().mean().item()

        return {"loss": loss, "accuracy": float(accuracy)}

    def _adversarial_train_step(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
    ) -> dict[str, Any]:
        """Single adversarial training step using research-based hyperrectangles (PGD)."""
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)

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

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == y_mixed).float().mean().item()

        return {"loss": loss, "accuracy": float(accuracy)}

    def _constraint_train_step(
        self,
        x: torch.Tensor,
        y_target: torch.Tensor,
        attack: pdml.training.attacks.Attack,
        constraint: pdml.constraints.Constraint,
    ) -> dict[str, Any]:
        """Single constraint training step (property-driven ML)."""
        x = x.to(self.device)
        y_target = y_target.to(self.device)

        adv = attack.attack(self.model, x, y_target, constraint)

        loss_adv, sat_adv = constraint.eval(
            self.model, x, adv, y_target, attack.logic, reduction="mean"
        )

        return {"loss": loss_adv, "satisfaction": float(sat_adv.item())}

    def _vehicle_train_step(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
    ) -> dict[str, Any]:
        """Single vehicle training step (vehicle specification-based)."""
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        logits = self.model(x_batch)
        task_loss = self.loss_fn(logits, y_batch.long())

        if self.constraint_loss is not None:
            constraint_loss = self.constraint_loss.eval(self.model, x_batch, y_batch)
            loss = task_loss * self.lambda_val + constraint_loss * (1 - self.lambda_val)
        else:
            loss = task_loss

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == y_batch).float().mean().item()

        return {"loss": loss, "accuracy": float(accuracy)}
