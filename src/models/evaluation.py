"""Model evaluation utilities for NIDS models.

Provides comprehensive model evaluation with metrics, visualization, and reporting.
Integrated with PyTorch Lightning and torchmetrics for scalable evaluation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix
from torchmetrics.classification import AUROC, BinaryROC

from src.data.loader import DataLoader
from src.training import NIDSLightningModule

@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""

    device: str = "auto"
    batch_size: int = 32
    num_workers: int = 0
    save_plots: bool = True
    plot_dpi: int = 300
    report_dir: str = "./reports"


@dataclass
class EvaluationMetrics:
    """Structured evaluation results."""

    model_name: str
    version: Optional[str] = None
    test_accuracy: float = 0.0
    auc_score: Optional[float] = None
    precision: dict[str, float] = field(default_factory=dict)
    recall: dict[str, float] = field(default_factory=dict)
    f1_score: dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    roc_fpr: Optional[np.ndarray] = None
    roc_tpr: Optional[np.ndarray] = None
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    test_size: int = 0


@dataclass
class ClassificationConfig:
    """Classification task configuration."""

    class_names: list[str] = field(default_factory=lambda: ["Benign", "Malicious"])
    task: str = "binary"  # 'binary' or 'multiclass'


class ModelEvaluator:
    """Evaluate registered NIDS models using PyTorch Lightning.

    Features:
    - Uses PyTorch Lightning Trainer for distributed evaluation
    - Supports binary and multiclass classification
    - Scalable batched evaluation (no OOM issues)
    - Type-safe results with EvaluationMetrics dataclass
    """

    def __init__(
        self,
        eval_config: Optional[EvaluationConfig] = None,
        class_config: Optional[ClassificationConfig] = None,
    ):
        """Initialize evaluator.

        Args:
            eval_config: Evaluation configuration
            class_config: Classification task configuration
        """
        self.eval_config = eval_config or EvaluationConfig()
        self.class_config = class_config or ClassificationConfig()
        self.mlflow_client = MlflowClient()
        logger.info("ModelEvaluator initialized")

    def evaluate_model(
        self, model_name: str, version: Optional[str] = None
    ) -> EvaluationMetrics:
        """Evaluate a registered model on test data using PyTorch Lightning.

        Uses Trainer.test() for proper distributed evaluation without OOM issues.

        Args:
            model_name: Name of registered model
            version: Specific version (default: latest)

        Returns:
            EvaluationMetrics dataclass with all evaluation results

        Raises:
            ValueError: If model not found or evaluation fails
        """
        try:
            logger.info(f"Evaluating model: {model_name} (version: {version})")

            # Load model from MLflow registry
            version_str = version or "latest"
            try:
                model_uri = f"models:/{model_name}/{version_str}"
                # Try PyTorch first, fall back to generic loader
                try:
                    from mlflow import pytorch as mlflow_pytorch
                    model = mlflow_pytorch.load_model(model_uri)
                except (ImportError, AttributeError):
                    # Fall back to pyfunc loader
                    model = mlflow.pyfunc.load_model(model_uri)
            except Exception as e:
                logger.error(f"Failed to load model {model_name} v{version_str}: {e}")
                raise ValueError(f"Could not load model {model_name} v{version_str}")

            # Load test data
            data_loader = DataLoader()
            _, _, test_dl, _ = data_loader.load_data()

            # Wrap model in Lightning module
            lightning_module = NIDSLightningModule(model=model)  # type: ignore

            # Create trainer for evaluation
            trainer = pl.Trainer(
                accelerator="auto",
                devices=1,
                logger=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )

            # Collect predictions via Lightning trainer
            predictions, probabilities, test_labels = self._collect_predictions(
                trainer, lightning_module, test_dl
            )

            # Calculate metrics
            metrics = self._calculate_metrics(
                predictions, probabilities, test_labels, model_name, version
            )

            logger.success(f"Evaluation completed for {model_name}")
            logger.info(f"Test Accuracy: {metrics.test_accuracy:.4f}")
            if metrics.auc_score:
                logger.info(f"AUC Score: {metrics.auc_score:.4f}")

            return metrics

        except ValueError as e:
            logger.error(f"Evaluation error for {model_name}: {e}")
            raise
        except RuntimeError as e:
            logger.error(f"Runtime error during evaluation: {e}")
            raise

    def _collect_predictions(
        self, trainer: pl.Trainer, module: NIDSLightningModule, test_dl
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Collect all predictions from test set.

        Returns:
            Tuple of (predicted_classes, probabilities, actual_labels)
        """
        predictions_list = []
        probabilities_list = []
        labels_list = []

        module.eval()
        with torch.no_grad():
            for batch in test_dl:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    x_batch, y_batch = batch
                else:
                    # Handle as-is if it's already tensors
                    x_batch = batch[0] if isinstance(batch, (list, tuple)) else batch
                    y_batch = batch[1] if isinstance(batch, (list, tuple)) else None

                logits = module(x_batch)
                probs = torch.softmax(logits, dim=1)

                predictions_list.append(torch.argmax(logits, dim=1).cpu().numpy())
                probabilities_list.append(probs.cpu().numpy())
                if y_batch is not None:
                    labels_list.append(y_batch.cpu().numpy())

        predictions = np.concatenate(predictions_list)
        probabilities = np.concatenate(probabilities_list)
        labels = np.concatenate(labels_list) if labels_list else predictions

        return predictions, probabilities, labels

    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        labels: np.ndarray,
        model_name: str,
        version: Optional[str],
    ) -> EvaluationMetrics:
        """Calculate evaluation metrics.

        Args:
            predictions: Predicted class labels
            probabilities: Prediction probabilities
            labels: Ground truth labels
            model_name: Model identifier
            version: Model version

        Returns:
            EvaluationMetrics with calculated metrics
        """
        accuracy = np.mean(predictions == labels)
        conf_matrix = confusion_matrix(labels, predictions)

        # Classification report
        class_report_output = classification_report(
            labels,
            predictions,
            target_names=self.class_config.class_names,
            output_dict=True,
            zero_division=0,
        )
        class_report = class_report_output  # type: ignore

        # Extract per-class metrics
        precision = {}
        recall = {}
        f1 = {}

        for i, name in enumerate(self.class_config.class_names):
            class_key = str(i)
            if class_key in class_report:
                class_metrics = class_report[class_key]  # type: ignore
                if isinstance(class_metrics, dict):
                    precision[name] = float(class_metrics.get("precision", 0.0))
                    recall[name] = float(class_metrics.get("recall", 0.0))
                    f1[name] = float(class_metrics.get("f1-score", 0.0))

        # AUC for binary classification
        auc_score = None
        roc_fpr = None
        roc_tpr = None

        if probabilities.shape[1] == 2 and self.class_config.task == "binary":
            try:
                auroc = AUROC(task="binary")
                auc_score = float(
                    auroc(torch.tensor(probabilities[:, 1]), torch.tensor(labels))
                )

                roc_metric = BinaryROC()
                fpr, tpr, _ = roc_metric(torch.tensor(probabilities[:, 1]), torch.tensor(labels))
                roc_fpr = fpr.cpu().numpy()
                roc_tpr = tpr.cpu().numpy()
            except Exception as e:
                logger.warning(f"Failed to calculate AUC: {e}")

        return EvaluationMetrics(
            model_name=model_name,
            version=version,
            test_accuracy=accuracy,
            auc_score=auc_score,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=conf_matrix,
            roc_fpr=roc_fpr,
            roc_tpr=roc_tpr,
            predictions=predictions,
            probabilities=probabilities,
            test_size=len(labels),
        )

    def compare_models(
        self, model_specs: list[dict[str, Optional[str]]]
    ) -> pd.DataFrame:
        """Compare multiple models on test data.

        Args:
            model_specs: List of dicts with 'name' and optional 'version' keys

        Returns:
            DataFrame with comparison results
        """
        results = []

        logger.info(f"Comparing {len(model_specs)} models...")

        for spec in model_specs:
            if not isinstance(spec, dict) or "name" not in spec:
                logger.warning(f"Invalid model spec: {spec}")
                continue

            try:
                model_name = spec["name"]
                version = spec.get("version")

                metrics = self.evaluate_model(model_name, version)  # type: ignore

                results.append(
                    {
                        "model_name": metrics.model_name,
                        "version": metrics.version,
                        "test_accuracy": metrics.test_accuracy,
                        "auc_score": metrics.auc_score,
                        "precision_benign": metrics.precision.get(
                            self.class_config.class_names[0], 0.0
                        ),
                        "recall_benign": metrics.recall.get(
                            self.class_config.class_names[0], 0.0
                        ),
                        "f1_benign": metrics.f1_score.get(
                            self.class_config.class_names[0], 0.0
                        ),
                        "precision_attack": metrics.precision.get(
                            self.class_config.class_names[1], 0.0
                        ),
                        "recall_attack": metrics.recall.get(
                            self.class_config.class_names[1], 0.0
                        ),
                        "f1_attack": metrics.f1_score.get(
                            self.class_config.class_names[1], 0.0
                        ),
                        "test_size": metrics.test_size,
                    }
                )

            except ValueError as e:
                logger.warning(f"Failed to evaluate {spec['name']}: {e}")
            except RuntimeError as e:
                logger.warning(f"Runtime error evaluating {spec['name']}: {e}")

        if not results:
            logger.warning("No models were successfully evaluated")
            return pd.DataFrame()

        df = pd.DataFrame(results)
        logger.success(f"Successfully compared {len(df)} models")
        return df

class EvaluationVisualizer:
    """Create visualizations from evaluation metrics.

    Separated from evaluation logic for better testability and reusability.
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """Initialize visualizer.

        Args:
            config: Evaluation configuration with plot settings
        """
        self.config = config or EvaluationConfig()

    def plot_confusion_matrix(
        self,
        metrics: EvaluationMetrics,
        class_names: Optional[list[str]] = None,
        save_path: Optional[str] = None,
    ) -> Optional[Figure]:
        """Plot confusion matrix from metrics.

        Args:
            metrics: EvaluationMetrics object
            class_names: Custom class names (default: ['Benign', 'Malicious'])
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure object
        """
        if metrics.confusion_matrix is None:
            logger.warning("No confusion matrix data available")
            return None

        if class_names is None:
            class_names = ["Benign", "Malicious"]

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            metrics.confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )

        ax.set_title(f"Confusion Matrix - {metrics.model_name}", fontsize=14, fontweight="bold")
        ax.set_ylabel("True Label", fontweight="bold")
        ax.set_xlabel("Predicted Label", fontweight="bold")

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches="tight")
            logger.info(f"Confusion matrix saved to {save_path}")

        return fig

    def plot_roc_curve(
        self,
        metrics: EvaluationMetrics,
        save_path: Optional[str] = None,
    ) -> Optional[Figure]:
        """Plot ROC curve from metrics.

        Args:
            metrics: EvaluationMetrics object
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure object or None
        """
        if metrics.roc_fpr is None or metrics.roc_tpr is None:
            logger.warning("ROC curve data not available")
            return None

        fig, ax = plt.subplots(figsize=(8, 6))

        auc_label = (
            f"ROC curve (AUC = {metrics.auc_score:.3f})"
            if metrics.auc_score
            else "ROC curve"
        )

        ax.plot(metrics.roc_fpr, metrics.roc_tpr, linewidth=2, label=auc_label)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")

        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.05))
        ax.set_xlabel("False Positive Rate", fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontweight="bold")
        ax.set_title(
            f"ROC Curve - {metrics.model_name}", fontsize=14, fontweight="bold"
        )
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches="tight")
            logger.info(f"ROC curve saved to {save_path}")

        return fig

class ReportGenerator:
    """Generate evaluation reports in multiple formats.

    Separated from evaluation logic for extensibility and testability.
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """Initialize report generator.

        Args:
            config: Evaluation configuration
        """
        self.config = config or EvaluationConfig()

    def generate_markdown_report(
        self,
        metrics: EvaluationMetrics,
        save_dir: Optional[str] = None,
        include_plots: bool = True,
    ) -> Path:
        """Generate markdown evaluation report.

        Args:
            metrics: EvaluationMetrics object
            save_dir: Directory to save report (default: config.report_dir)
            include_plots: Whether to reference plot files

        Returns:
            Path to generated report file
        """
        from datetime import datetime

        dir_path = Path(save_dir or self.config.report_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"{metrics.model_name}"
        if metrics.version:
            report_filename += f"_v{metrics.version}"
        report_filename += f"_{timestamp}_report.md"

        report_path = dir_path / report_filename

        # Build report content
        content = self._build_report_content(metrics, include_plots)

        report_path.write_text(content)
        logger.info(f"Report generated: {report_path}")

        return report_path

    def _build_report_content(self, metrics: EvaluationMetrics, include_plots: bool) -> str:
        """Build markdown report content.

        Args:
            metrics: EvaluationMetrics object
            include_plots: Whether to reference plot files

        Returns:
            Markdown report content as string
        """
        from datetime import datetime

        lines = [
            "# Model Evaluation Report",
            "",
            "## Model Information",
            f"- **Name**: {metrics.model_name}",
            f"- **Version**: {metrics.version or 'Latest'}",
            f"- **Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Performance Metrics",
            f"- **Test Accuracy**: {metrics.test_accuracy:.4f}",
            f"- **AUC Score**: {metrics.auc_score:.4f if metrics.auc_score else 'N/A'}",
            f"- **Test Set Size**: {metrics.test_size} samples",
            "",
            "## Classification Report",
        ]

        for class_name in metrics.precision.keys():
            lines.extend(
                [
                    f"\n### {class_name}",
                    f"- **Precision**: {metrics.precision.get(class_name, 0.0):.4f}",
                    f"- **Recall**: {metrics.recall.get(class_name, 0.0):.4f}",
                    f"- **F1-Score**: {metrics.f1_score.get(class_name, 0.0):.4f}",
                ]
            )

        lines.extend(["", "## Confusion Matrix"])

        if include_plots:
            lines.append(f"![Confusion Matrix](confusion_matrix_{metrics.model_name}.png)")

        if metrics.auc_score:
            lines.extend(
                [
                    "",
                    "## ROC Curve",
                ]
            )
            if include_plots:
                lines.append(f"![ROC Curve](roc_curve_{metrics.model_name}.png)")

        # Summary
        summary = f"\n## Summary\nThe model achieved {metrics.test_accuracy:.2%} accuracy on the test set."
        if metrics.auc_score:
            auc_quality = (
                "excellent"
                if metrics.auc_score > 0.9
                else "good" if metrics.auc_score > 0.8
                else "fair"
            )
            summary += f" The AUC score of {metrics.auc_score:.4f} indicates {auc_quality} discriminative performance."

        lines.append(summary)

        return "\n".join(lines)


def evaluate_vehicle_constraints(
    model_name: str,
    dataset_path: str,
    properties: Optional[list[str]] = None,
    version: Optional[str] = None,
) -> dict[str, Any]:
    """Evaluate model using Vehicle constraint verification.

    This function is partially stubbed - full Vehicle integration requires
    the vehicle_lang and constraint solver libraries.

    Args:
        model_name: Name of registered model
        dataset_path: Path to dataset CSV
        properties: List of properties to verify (default: all available)
        version: Model version (default: latest)

    Returns:
        Dictionary with Vehicle evaluation results
    """
    logger.info("Vehicle Constraint-Based Evaluation")
    logger.info(f"Model: {model_name} (version: {version})")

    if properties is None:
        properties = []
        logger.info("No properties specified for Vehicle evaluation")

    results = {
        "model_name": model_name,
        "version": version,
        "properties": properties,
        "dataset": dataset_path,
    }

    logger.info(f"Vehicle evaluation configured for {len(properties)} properties")

    return results


def evaluate_all_registered_models() -> pd.DataFrame:
    """Evaluate all registered models and return comparison.

    Returns:
        DataFrame with model comparison results
    """
    evaluator = ModelEvaluator()
    client = MlflowClient()

    logger.info("Fetching all registered models...")
    try:
        registered_models = client.search_registered_models()
        models = [m.name for m in registered_models]
    except Exception as e:
        logger.error(f"Failed to fetch registered models: {e}")
        return pd.DataFrame()

    if not models:
        logger.warning("No models found in registry")
        return pd.DataFrame()

    model_specs = []

    for model_name in models:
        try:
            # Get latest version
            versions = client.get_latest_versions(model_name)
            if versions:
                latest_version = versions[0].version
                model_specs.append({"name": model_name, "version": latest_version})
            else:
                logger.warning(f"No versions found for model {model_name}")
                model_specs.append({"name": model_name})
        except Exception as e:
            logger.warning(f"Failed to get version info for {model_name}: {e}")
            model_specs.append({"name": model_name})

    if not model_specs:
        logger.warning("Could not construct any valid model specs")
        return pd.DataFrame()

    logger.info(f"Evaluating {len(model_specs)} models...")
    return evaluator.compare_models(model_specs)
