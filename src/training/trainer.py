"""
Unified training interface for NIDS models (PyTorch)
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader

from src.attacks.pgd import generate_pgd_adversarial_examples
from src.config import config
from src.utils.logging import get_logger
from src.utils.performance import Timer

logger = get_logger(__name__)


class NIDSTrainer:
    """Unified NIDS model trainer with gradient tracking and MLflow integration (Torch)"""

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer | None = None,
        device: torch.device | None = None,
    ):
        self.model = model
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(self.device)
        self.optimizer = optimizer or optim.Adam(
            self.model.parameters(), lr=config.training.learning_rate
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def init_gradient_logging(self, log_dir: str | None = None):
        """Initialize simplified gradient logging"""
        if not config.logging.gradient_logging_enabled:
            return
        logger.debug("Gradient monitoring enabled - will track global gradient norms")

    def train(
        self,
        train_dataset: TorchDataLoader,
        val_dataset: TorchDataLoader,
        training_type: str = "base",
        attack_rects: list[np.ndarray] | None = None,
        epochs: int | None = None,
        steps_per_epoch: int | None = None,
        attack_pattern: str = "hulk",
    ) -> dict[str, Any]:
        """Unified training method for both base and adversarial training."""
        epochs = epochs or config.training.epochs
        steps_per_epoch = steps_per_epoch or config.training.steps_per_epoch

        # Start MLflow run
        run_name = f"{training_type}_{config.model.model_type}_{epochs}ep"
        logger.start_run(run_name=run_name)

        # Log parameters
        params: dict[str, Any] = {
            "training_type": training_type,
            "model_type": config.model.model_type,
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "learning_rate": config.training.learning_rate,
            "batch_size": config.data.batch_size,
        }
        if training_type == "adversarial":
            params.update(
                {
                    "pgd_epsilon": config.training.pgd_epsilon,
                    "pgd_steps": config.training.pgd_steps,
                    "pgd_alpha": config.training.pgd_alpha,
                    "using_research_hyperrectangles": attack_rects is None,
                }
            )
        logger.log_params(params)

        # Initialize gradient logging
        self.init_gradient_logging("logs/gradients")

        history: dict[str, list[float]] = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "epoch_times": [],
        }

        for epoch in range(epochs):
            with Timer() as epoch_timer:
                logger.info(f"Epoch {epoch + 1}/{epochs}")
                epoch_loss: list[float] = []
                epoch_acc: list[float] = []
                step_count = 0
                global_step = epoch * steps_per_epoch

                self.model.train()
                for x_batch, y_batch in train_dataset:
                    if step_count >= steps_per_epoch:
                        break
                    global_step += 1

                    if training_type == "adversarial":
                        metrics = self._adversarial_train_step(
                            x_batch, y_batch, attack_rects, attack_pattern
                        )
                    else:
                        metrics = self._base_train_step(x_batch, y_batch)

                    epoch_loss.append(float(metrics["loss"]))
                    epoch_acc.append(float(metrics["accuracy"]))

                    if "gradients" in metrics and config.logging.gradient_logging_enabled:
                        grad_metrics = self._analyze_gradients(
                            metrics["gradients"], step=global_step, epoch=epoch
                        )
                        if (
                            global_step % config.logging.gradient_mlflow_frequency == 0
                            and grad_metrics
                        ):
                            logger.log_gradient_metrics(
                                {"grad_global_norm": grad_metrics.get("grad_global_norm", 0.0)},
                                step=global_step,
                            )

                    step_count += 1
                    if step_count % 100 == 0:
                        logger.debug(
                            f"Step {step_count}/{steps_per_epoch} - Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}"
                        )

                avg_loss = float(np.mean(epoch_loss)) if epoch_loss else 0.0
                avg_acc = float(np.mean(epoch_acc)) if epoch_acc else 0.0
                val_loss, val_acc = self._evaluate_model(val_dataset)

                history["loss"].append(avg_loss)
                history["accuracy"].append(avg_acc)
                history["val_loss"].append(float(val_loss))
                history["val_accuracy"].append(float(val_acc))

                logger.log_metrics(
                    {
                        "loss": avg_loss,
                        "accuracy": avg_acc,
                        "val_loss": float(val_loss),
                        "val_accuracy": float(val_acc),
                    },
                    step=epoch,
                )

            epoch_time = epoch_timer.elapsed
            history["epoch_times"].append(epoch_time)
            logger.info(
                f"Epoch {epoch + 1}: Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Time: {epoch_time:.4f}s"
            )

        final_metrics = {
            "final_train_loss": history["loss"][-1] if history["loss"] else 0.0,
            "final_train_accuracy": history["accuracy"][-1] if history["accuracy"] else 0.0,
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else 0.0,
            "final_val_accuracy": history["val_accuracy"][-1] if history["val_accuracy"] else 0.0,
            "total_training_time": sum(history["epoch_times"]),
        }
        logger.log_metrics(final_metrics)
        logger.info(
            f"Training completed. Final validation accuracy: {history['val_accuracy'][-1]:.4f}"
            if history["val_accuracy"]
            else "Training completed."
        )

        self._save_and_register_model(training_type, epochs, history)
        logger.end_run()
        return history

    def _base_train_step(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> dict[str, Any]:
        """Single base training step (Torch)"""
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        self.optimizer.zero_grad()
        logits = self.model(x_batch)
        loss = self.loss_fn(logits, y_batch.long())
        loss.backward()

        gradients = [p.grad.detach().clone() for p in self.model.parameters() if p.grad is not None]
        self.optimizer.step()

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == y_batch).float().mean().item()

        return {"loss": float(loss.item()), "accuracy": float(accuracy), "gradients": gradients}

    def _adversarial_train_step(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        attack_rects: list[np.ndarray] | None = None,
        attack_pattern: str = "hulk",
    ) -> dict[str, Any]:
        """Single adversarial training step using research-based hyperrectangles (Torch)"""
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        # Generate adversarial examples using research hyperrectangles
        x_adv = generate_pgd_adversarial_examples(
            self.model,
            x_batch,
            y_batch,
            attack_rects=attack_rects,
            epsilon=config.training.pgd_epsilon,
            num_steps=config.training.pgd_steps,
            step_size=config.training.pgd_alpha,
            attack_pattern=attack_pattern,
        )

        # Mix clean and adversarial examples
        x_mixed = torch.cat([x_batch, x_adv.detach()], dim=0)
        y_mixed = torch.cat([y_batch, y_batch], dim=0)
        indices = torch.randperm(x_mixed.size(0), device=self.device)
        x_mixed = x_mixed[indices]
        y_mixed = y_mixed[indices]

        # Training step
        self.optimizer.zero_grad()
        logits = self.model(x_mixed)
        loss = self.loss_fn(logits, y_mixed.long())
        loss.backward()

        gradients = [p.grad.detach().clone() for p in self.model.parameters() if p.grad is not None]

        self.optimizer.step()

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == y_mixed).float().mean().item()

        return {"loss": float(loss.item()), "accuracy": float(accuracy), "gradients": gradients}

    def _analyze_gradients(self, gradients, step: int = 0, epoch: int = 0) -> dict[str, Any]:
        """Simplified gradient analysis focusing on actionable metrics"""
        if not config.logging.gradient_logging_enabled:
            return {}

        # Calculate global gradient norm (most important metric)
        try:
            norms = []
            for g in gradients:
                if g is None:
                    continue
                # Ensure tensor on CPU for norm computation if needed
                if g.is_cuda:
                    norms.append(g.detach().float().cpu().norm())
                else:
                    norms.append(g.detach().float().norm())
            if norms:
                global_norm = float(torch.stack(norms).norm().item())
            else:
                global_norm = 0.0
        except Exception:
            global_norm = 0.0

        # Log simplified gradient info
        if (
            config.logging.gradient_console_logging
            and step % config.logging.gradient_log_frequency == 0
        ):
            logger.info(f"Gradients: {global_norm:.4f}")

        # Return minimal essential metrics
        return {"grad_global_norm": global_norm}

    def _evaluate_model(self, dataset: TorchDataLoader) -> tuple[float, float]:
        """Evaluate model on dataset (Torch)"""
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        with torch.no_grad():
            for x_batch, y_batch in dataset:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(x_batch)
                loss = self.loss_fn(logits, y_batch.long())

                preds = torch.argmax(logits, dim=1)
                accuracy = (preds == y_batch).float().mean().item()

                total_loss += float(loss.item())
                total_acc += float(accuracy)
                num_batches += 1

        return (
            total_loss / num_batches if num_batches else 0.0,
            total_acc / num_batches if num_batches else 0.0,
        )

    def _save_and_register_model(self, training_type: str, epochs: int, history: dict[str, Any]):
        """
        Save model and register in MLflow following industry best practices.
        This ensures model artifacts are properly tracked and linked to experiments.
        """

        import importlib
        import os
        import time

        import mlflow

        try:
            if not mlflow.active_run():
                logger.warning("No active MLflow run - cannot save model")
                return

            # Create model save directory
            os.makedirs(config.experiment.model_save_path, exist_ok=True)
            timestamp = int(time.time())
            model_name = f"{config.model.model_type}_{training_type}_{timestamp}.pt"
            local_path = os.path.join(config.experiment.model_save_path, model_name)

            # Save Torch model locally
            torch.save(self.model.state_dict(), local_path)
            logger.info(f"Saved PyTorch model to {local_path}")

            # Log artifact
            logger.log_artifact(local_path, artifact_path="model")

            # Try MLflow PyTorch logging/registration
            try:
                mlflow_pytorch = importlib.import_module("mlflow.pytorch")
                mlflow_pytorch.log_model(self.model, artifact_path="torch_model")

                # Register model name
                try:
                    active = mlflow.active_run()
                    if not active:
                        raise RuntimeError("No active MLflow run found during registration")
                    model_uri = f"runs:/{active.info.run_id}/torch_model"
                    registry_name = (
                        f"nids-{config.experiment.attack_name.lower()}-"
                        f"{config.model.model_type}-{training_type}"
                    )
                    mlflow.register_model(model_uri=model_uri, name=registry_name)
                    logger.info(f"Model registered in MLflow registry: {registry_name}")
                except Exception as reg_error:
                    logger.warning(
                        f"Model registry registration failed (non-critical): {reg_error}"
                    )

            except Exception as e:
                logger.warning(f"mlflow.pytorch not available, logged raw artifact only: {e}")

        except Exception as e:
            logger.error(f"Failed to save/register model: {e}")

    def _create_model_signature(self):
        """Create MLflow model signature for better model documentation"""
        try:
            from mlflow.models.signature import infer_signature

            input_dim = self._infer_input_dim()
            sample_input = np.random.rand(1, input_dim).astype(np.float32)
            sample_tensor = torch.from_numpy(sample_input).to(self.device)
            self.model.eval()
            with torch.no_grad():
                sample_output = self.model(sample_tensor).cpu().numpy()

            return infer_signature(sample_input, sample_output)
        except Exception as e:
            logger.debug(f"Could not create model signature: {e}")
            return None

    def _get_model_input_example(self):
        """Get input example for MLflow model documentation"""
        try:
            input_dim = self._infer_input_dim()
            return np.random.rand(1, input_dim).astype(np.float32)
        except Exception as e:
            logger.debug(f"Could not create input example: {e}")
            return None

    def _infer_input_dim(self) -> int:
        """Best-effort inference of the model's expected feature dimension."""
        try:
            for p in self.model.parameters():
                if p.dim() >= 2:
                    return int(p.shape[1])
            return 32
        except Exception:
            return 32


# Compatibility functions for existing code
def train_adversarial(
    model: nn.Module,
    train_dataset: TorchDataLoader,
    val_dataset: TorchDataLoader,
    attack_rects: list[np.ndarray] | None = None,
    epochs: int | None = None,
    steps_per_epoch: int | None = None,
    attack_pattern: str = "hulk",
) -> dict[str, Any]:
    """
    Compatibility wrapper for adversarial training using research-based hyperrectangles.

    Args:
        attack_rects: Optional attack rectangles. If None, uses research hyperrectangles.
    """
    trainer = NIDSTrainer(model)
    return trainer.train(
        train_dataset,
        val_dataset,
        "adversarial",
        attack_rects,
        epochs,
        steps_per_epoch,
        attack_pattern,
    )


def train_base(
    model: nn.Module,
    train_dataset: TorchDataLoader,
    val_dataset: TorchDataLoader,
    epochs: int | None = None,
    steps_per_epoch: int | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for base training"""
    trainer = NIDSTrainer(model)
    return trainer.train(train_dataset, val_dataset, "base", None, epochs, steps_per_epoch)


def evaluate_model(model: nn.Module, dataset: TorchDataLoader) -> tuple[float, float]:
    """Compatibility wrapper for model evaluation"""
    trainer = NIDSTrainer(model)
    return trainer._evaluate_model(dataset)
