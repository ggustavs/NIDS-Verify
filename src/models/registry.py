"""
MLflow Model Registry utilities for NIDS models (in transition to PyTorch)
"""
import mlflow
import mlflow.exceptions
from mlflow.tracking import MlflowClient
from typing import Optional, List, Dict, Any
import pandas as pd
from src.utils.logging import get_logger

logger = get_logger(__name__)


class NIDSModelRegistry:
    """MLflow Model Registry manager for NIDS models"""

    def __init__(self):
        self.client = MlflowClient()

    def list_models(self) -> List[str]:
        """List all registered model names"""
        try:
            models = self.client.search_registered_models()
            return [model.name for model in models]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a registered model"""
        try:
            model = self.client.get_registered_model(model_name)
            versions = self.client.get_latest_versions(model_name)

            return {
                "name": model.name,
                "description": model.description,
                "creation_timestamp": model.creation_timestamp,
                "last_updated_timestamp": model.last_updated_timestamp,
                "tags": model.tags,
                "versions": [
                    {
                        "version": v.version,
                        "description": v.description,
                        "creation_timestamp": v.creation_timestamp,
                        "tags": v.tags,
                        "run_id": v.run_id
                    } for v in versions
                ]
            }
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")
            return None

    def compare_models(self, model_names: List[str]) -> pd.DataFrame:
        """Compare performance metrics across different models"""
        comparison_data = []

        for model_name in model_names:
            try:
                # Get latest version
                versions = self.client.get_latest_versions(model_name)

                if not versions:
                    continue

                version = versions[0]

                # Get run metrics
                run = mlflow.get_run(version.run_id)
                metrics = run.data.metrics
                params = run.data.params

                comparison_data.append({
                    "model_name": model_name,
                    "version": version.version,
                    "final_train_accuracy": metrics.get("final_train_accuracy", None),
                    "final_val_accuracy": metrics.get("final_val_accuracy", None),
                    "final_train_loss": metrics.get("final_train_loss", None),
                    "final_val_loss": metrics.get("final_val_loss", None),
                    "avg_epoch_time": metrics.get("avg_epoch_time", None),
                    "model_type": params.get("model_type", None),
                    "training_type": params.get("training_type", None),
                    "epochs": params.get("epochs", None),
                    "learning_rate": params.get("learning_rate", None)
                })

            except Exception as e:
                logger.warning(f"Failed to get metrics for {model_name}: {e}")
                continue

        return pd.DataFrame(comparison_data)

    def load_model(self, model_name: str, version: str = None):
        """Load model from registry"""
        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
            else:
                # Get latest version
                versions = self.client.get_latest_versions(model_name)
                if not versions:
                    raise ValueError(f"No versions found for model {model_name}")
                model_uri = f"models:/{model_name}/{versions[0].version}"

            # TODO: migrate to mlflow.pytorch; for now, use generic loader if available
            try:
                import mlflow.pytorch  # noqa: F401
                model = mlflow.pytorch.load_model(model_uri)
            except Exception:
                model = mlflow.pyfunc.load_model(model_uri)
            version_str = f"version {version}" if version else f"latest version ({versions[0].version})"
            logger.info(f"Loaded model {model_name} from {version_str}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return None

    def delete_model_version(self, model_name: str, version: str):
        """Delete a specific model version"""
        try:
            self.client.delete_model_version(model_name, version)
            logger.info(f"Deleted {model_name} version {version}")
        except Exception as e:
            logger.error(f"Failed to delete model version: {e}")

    def add_model_description(self, model_name: str, version: str, description: str):
        """Add description to a model version"""
        try:
            self.client.update_model_version(
                name=model_name,
                version=version,
                description=description
            )
            logger.info(f"Updated description for {model_name} v{version}")
        except Exception as e:
            logger.error(f"Failed to update model description: {e}")

    def register_model_from_path(self, model_path: str, model_name: str,
                                params: Dict[str, Any], metrics: Dict[str, float],
                                description: str = None) -> str:
        """
        Register a model from a file path

        Args:
            model_path: Path to the model file
            model_name: Name to register under
            params: Model parameters to log
            metrics: Metrics to log
            description: Optional description

        Returns:
            MLflow run ID
        """
    # NOTE: TensorFlow-specific path being removed; prefer PyTorch in new code paths

        with mlflow.start_run() as run:
            try:
                # Load and validate model
                # Defer exact framework loading; use MLflow pyfunc for registration
                model = None
                logger.info(f"Successfully loaded model from {model_path}")

                # Log parameters and metrics
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)

                # Register the model
                # Prefer artifact logging via pyfunc or pytorch in future
                mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=None,  # Placeholder; switch to mlflow.pytorch in torch trainer
                    registered_model_name=model_name,
                )

                # Add description if provided
                if description:
                    try:
                        versions = self.client.get_latest_versions(model_name)
                        if versions:
                            self.add_model_description(model_name, versions[0].version, description)
                    except Exception as e:
                        logger.warning(f"Failed to add description: {e}")

                logger.info(f"Successfully registered {model_name}")
                return run.info.run_id

            except Exception as e:
                logger.error(f"Failed to register model {model_name}: {e}")
                raise

    def get_best_model(self, metric: str = "final_val_accuracy") -> Optional[Dict[str, Any]]:
        """Get the best performing model based on a specific metric"""
        try:
            models = self.list_models()
            if not models:
                logger.warning("No models found in registry")
                return None

            best_model = None
            best_metric = float('-inf') if 'accuracy' in metric else float('inf')
            errors_encountered = []

            for model_name in models:
                try:
                    versions = self.client.get_latest_versions(model_name)
                    if not versions:
                        logger.debug(f"No versions found for model {model_name}")
                        continue

                    for version in versions:
                        try:
                            run = mlflow.get_run(version.run_id)
                            if not run.data.metrics:
                                logger.debug(f"No metrics found for {model_name} v{version.version}")
                                continue

                            metric_value = run.data.metrics.get(metric)
                            if metric_value is None:
                                logger.debug(f"Metric '{metric}' not found for {model_name} v{version.version}")
                                continue

                            is_better = (
                                metric_value > best_metric if 'accuracy' in metric
                                else metric_value < best_metric
                            )

                            if is_better:
                                best_metric = metric_value
                                best_model = {
                                    "name": model_name,
                                    "version": version.version,
                                    "metric_value": metric_value,
                                    "run_id": version.run_id
                                }

                        except mlflow.exceptions.MlflowException as e:
                            error_msg = f"MLflow error for {model_name} v{version.version}: {e}"
                            logger.warning(error_msg)
                            errors_encountered.append(error_msg)
                        except KeyError as e:
                            error_msg = f"Missing data field for {model_name} v{version.version}: {e}"
                            logger.warning(error_msg)
                            errors_encountered.append(error_msg)
                        except Exception as e:
                            error_msg = f"Unexpected error processing {model_name} v{version.version}: {e}"
                            logger.warning(error_msg)
                            errors_encountered.append(error_msg)

                except mlflow.exceptions.MlflowException as e:
                    error_msg = f"MLflow error accessing model {model_name}: {e}"
                    logger.warning(error_msg)
                    errors_encountered.append(error_msg)
                except Exception as e:
                    error_msg = f"Unexpected error processing model {model_name}: {e}"
                    logger.warning(error_msg)
                    errors_encountered.append(error_msg)

            if errors_encountered:
                logger.info(f"Encountered {len(errors_encountered)} errors while searching for best model")

            if best_model:
                logger.info(f"Best model by {metric}: {best_model['name']} v{best_model['version']} "
                           f"({metric}={best_model['metric_value']:.4f})")
            else:
                logger.warning(f"No model found with metric '{metric}'")

            return best_model

        except Exception as e:
            logger.error(f"Failed to find best model: {e}")
            return None


def create_model_comparison_report(models: List[str] = None) -> str:
    """Create a comparison report of models in the registry"""
    registry = NIDSModelRegistry()

    if models is None:
        models = registry.list_models()

    if not models:
        return "No models found in registry."

    df = registry.compare_models(models)

    if df.empty:
        return "No model data available for comparison."

    # Sort by validation accuracy (descending)
    df_sorted = df.sort_values('final_val_accuracy', ascending=False)

    report = "# NIDS Model Registry Comparison Report\n\n"
    report += f"Total models: {len(df)}\n\n"

    report += "## Performance Summary\n\n"
    report += df_sorted.to_string(index=False)
    report += "\n\n"

    # Best models by category
    best_overall = df_sorted.iloc[0] if not df_sorted.empty else None
    if best_overall is not None:
        report += f"## Best Overall Model\n\n"
        report += f"- **Model**: {best_overall['model_name']}\n"
        report += f"- **Version**: {best_overall['version']}\n"
        report += f"- **Validation Accuracy**: {best_overall['final_val_accuracy']:.4f}\n"
        report += f"- **Training Type**: {best_overall['training_type']}\n"
        report += f"- **Architecture**: {best_overall['model_type']}\n\n"

    # Best by training type
    for training_type in df['training_type'].unique():
        if pd.isna(training_type):
            continue
        subset = df[df['training_type'] == training_type].sort_values('final_val_accuracy', ascending=False)
        if not subset.empty:
            best = subset.iloc[0]
            report += f"## Best {training_type.title()} Model\n\n"
            report += f"- **Model**: {best['model_name']}\n"
            report += f"- **Validation Accuracy**: {best['final_val_accuracy']:.4f}\n"
            report += f"- **Architecture**: {best['model_type']}\n\n"

    return report
