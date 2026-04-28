"""Model evaluation utilities for NIDS models."""

from dataclasses import dataclass, field
from datetime import datetime
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
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)

from src.data.loader import NIDSDataModule

_CLASS_NAMES = ["Benign", "Malicious"]


@dataclass
class EvaluationConfig:
    plot_dpi: int = 300
    report_dir: str = "./reports"


@dataclass
class EvaluationMetrics:
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


class ModelEvaluator:
    def __init__(self, eval_config: Optional[EvaluationConfig] = None):
        self.eval_config = eval_config or EvaluationConfig()

    def evaluate_model(
        self, model: nn.Module, dataset: str = "fixed"
    ) -> EvaluationMetrics:
        """Evaluate a model on test data.

        Args:
            model: PyTorch model to evaluate
            dataset: Which dataset split to use ("original" or "fixed")
        """
        dm = NIDSDataModule(dataset=dataset)
        dm.setup()
        predictions, probabilities, labels = self._collect_predictions(model, dm.test_dataloader())
        return self._calculate_metrics(predictions, probabilities, labels, model.__class__.__name__)

    def _collect_predictions(
        self, model: nn.Module, test_dl
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        device = next(model.parameters()).device
        predictions_list = []
        probabilities_list = []
        labels_list = []

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in test_dl:
                x_batch = x_batch.to(device)
                logits = model(x_batch)
                probs = torch.softmax(logits, dim=1)
                predictions_list.append(torch.argmax(logits, dim=1).cpu().numpy())
                probabilities_list.append(probs.cpu().numpy())
                labels_list.append(y_batch.cpu().numpy())

        return (
            np.concatenate(predictions_list),
            np.concatenate(probabilities_list),
            np.concatenate(labels_list),
        )

    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        labels: np.ndarray,
        model_name: str,
        version: Optional[str] = None,
    ) -> EvaluationMetrics:
        accuracy = float(np.mean(predictions == labels))
        conf_matrix = confusion_matrix(labels, predictions)

        report: dict[str, Any] = classification_report(
            labels,
            predictions,
            target_names=_CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )  # type: ignore[assignment]

        precision = {name: float(report[name]["precision"]) for name in _CLASS_NAMES if name in report}
        recall = {name: float(report[name]["recall"]) for name in _CLASS_NAMES if name in report}
        f1 = {name: float(report[name]["f1-score"]) for name in _CLASS_NAMES if name in report}

        pos_probs = probabilities[:, 1]
        roc_fpr, roc_tpr, _ = roc_curve(labels, pos_probs)
        auc_score = float(auc(roc_fpr, roc_tpr))

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
        self, models: list[tuple[str, nn.Module]], dataset: str = "fixed"
    ) -> pd.DataFrame:
        """Evaluate and compare multiple models on the same test split.

        Args:
            models: List of (name, model) pairs
            dataset: Which dataset split to use
        """
        dm = NIDSDataModule(dataset=dataset)
        dm.setup()
        test_dl = dm.test_dataloader()

        rows = []
        for name, model in models:
            predictions, probabilities, labels = self._collect_predictions(model, test_dl)
            m = self._calculate_metrics(predictions, probabilities, labels, name)
            rows.append({
                "model": name,
                "accuracy": m.test_accuracy,
                "auc": m.auc_score,
                "precision_benign": m.precision.get("Benign", 0.0),
                "recall_benign": m.recall.get("Benign", 0.0),
                "f1_benign": m.f1_score.get("Benign", 0.0),
                "precision_attack": m.precision.get("Malicious", 0.0),
                "recall_attack": m.recall.get("Malicious", 0.0),
                "f1_attack": m.f1_score.get("Malicious", 0.0),
                "n_test": m.test_size,
            })
            logger.info(f"{name}: acc={m.test_accuracy:.4f} auc={m.auc_score:.4f}")

        return pd.DataFrame(rows)


class EvaluationVisualizer:
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()

    def plot_confusion_matrix(
        self,
        metrics: EvaluationMetrics,
        save_path: Optional[str] = None,
    ) -> Optional[Figure]:
        if metrics.confusion_matrix is None:
            return None

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            metrics.confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=_CLASS_NAMES,
            yticklabels=_CLASS_NAMES,
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix — {metrics.model_name}")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches="tight")

        return fig

    def plot_roc_curve(
        self,
        metrics: EvaluationMetrics,
        save_path: Optional[str] = None,
    ) -> Optional[Figure]:
        if metrics.roc_fpr is None or metrics.roc_tpr is None:
            return None

        fig, ax = plt.subplots(figsize=(8, 6))
        label = f"AUC = {metrics.auc_score:.3f}" if metrics.auc_score else "ROC"
        ax.plot(metrics.roc_fpr, metrics.roc_tpr, linewidth=2, label=label)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.05))
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve — {metrics.model_name}")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches="tight")

        return fig


class ReportGenerator:
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self._viz = EvaluationVisualizer(self.config)

    def generate(
        self,
        metrics: EvaluationMetrics,
        save_dir: Optional[str] = None,
    ) -> Path:
        """Write a markdown report and accompanying plots to save_dir."""
        dir_path = Path(save_dir or self.config.report_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        stem = metrics.model_name
        if metrics.version:
            stem += f"_v{metrics.version}"
        stem += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        cm_path = dir_path / f"{stem}_confusion_matrix.png"
        roc_path = dir_path / f"{stem}_roc.png"
        self._viz.plot_confusion_matrix(metrics, save_path=str(cm_path))
        self._viz.plot_roc_curve(metrics, save_path=str(roc_path))

        auc_str = f"{metrics.auc_score:.4f}" if metrics.auc_score is not None else "N/A"
        lines = [
            f"# Evaluation Report: {metrics.model_name}",
            "",
            f"- **Version**: {metrics.version or 'N/A'}",
            f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"- **Test set size**: {metrics.test_size}",
            "",
            "## Metrics",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Accuracy | {metrics.test_accuracy:.4f} |",
            f"| AUC | {auc_str} |",
        ]
        for name in _CLASS_NAMES:
            lines += [
                f"| Precision ({name}) | {metrics.precision.get(name, 0):.4f} |",
                f"| Recall ({name}) | {metrics.recall.get(name, 0):.4f} |",
                f"| F1 ({name}) | {metrics.f1_score.get(name, 0):.4f} |",
            ]

        lines += [
            "",
            "## Confusion Matrix",
            f"![Confusion Matrix]({cm_path.name})",
            "",
            "## ROC Curve",
            f"![ROC Curve]({roc_path.name})",
        ]

        report_path = dir_path / f"{stem}_report.md"
        report_path.write_text("\n".join(lines))
        logger.info(f"Report written to {report_path}")
        return report_path


def evaluate_all_registered_models(dataset: str = "fixed") -> pd.DataFrame:
    """Evaluate all models currently registered in the MLflow model registry."""
    client = MlflowClient()
    evaluator = ModelEvaluator()

    try:
        registered = client.search_registered_models()
    except Exception as e:
        logger.error(f"Failed to fetch registered models: {e}")
        return pd.DataFrame()

    if not registered:
        logger.warning("No models in registry")
        return pd.DataFrame()

    models = []
    for rm in registered:
        versions = client.search_model_versions(f"name='{rm.name}'")
        latest = max(versions, key=lambda v: int(v.version), default=None)
        if latest is None:
            continue
        try:
            model = mlflow.pytorch.load_model(f"models:/{rm.name}/{latest.version}")
            models.append((f"{rm.name}_v{latest.version}", model))
        except Exception as e:
            logger.warning(f"Could not load {rm.name} v{latest.version}: {e}")

    if not models:
        return pd.DataFrame()

    return evaluator.compare_models(models, dataset=dataset)
