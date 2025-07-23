"""
Model evaluation utilities for registered NIDS models
"""
import mlflow
import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.registry import NIDSModelRegistry
from src.data.loader import load_data
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Evaluate registered NIDS models"""
    
    def __init__(self):
        self.registry = NIDSModelRegistry()
        
    def evaluate_model(self, model_name: str, stage: str = "Production", 
                      version: str = None) -> Dict[str, Any]:
        """
        Evaluate a registered model on test data
        
        Args:
            model_name: Name of registered model
            stage: Model stage to evaluate (Production, Staging, etc.)
            version: Specific version (overrides stage)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            # Load model from registry
            model = self.registry.load_model(model_name, stage, version)
            if model is None:
                raise ValueError(f"Could not load model {model_name}")
            
            # Load test data
            _, _, test_dataset, feature_names = load_data()
            
            # Collect all test data
            test_X, test_y = [], []
            for x_batch, y_batch in test_dataset:
                test_X.append(x_batch.numpy())
                test_y.append(y_batch.numpy())
            
            test_X = np.vstack(test_X)
            test_y = np.hstack(test_y)
            
            # Make predictions
            predictions = model.predict(test_X, verbose=0)
            predicted_probs = tf.nn.softmax(predictions).numpy()
            predicted_classes = np.argmax(predicted_probs, axis=1)
            
            # Calculate metrics
            accuracy = np.mean(predicted_classes == test_y)
            
            # For binary classification
            if predicted_probs.shape[1] == 2:
                auc_score = roc_auc_score(test_y, predicted_probs[:, 1])
                fpr, tpr, _ = roc_curve(test_y, predicted_probs[:, 1])
            else:
                auc_score = None
                fpr, tpr = None, None
            
            # Classification report
            class_report = classification_report(
                test_y, predicted_classes, 
                target_names=['Benign', 'Attack'], 
                output_dict=True
            )
            
            # Confusion matrix
            conf_matrix = confusion_matrix(test_y, predicted_classes)
            
            results = {
                'model_name': model_name,
                'stage': stage,
                'version': version,
                'test_accuracy': accuracy,
                'auc_score': auc_score,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix,
                'roc_curve': {'fpr': fpr, 'tpr': tpr} if fpr is not None else None,
                'predictions': predicted_classes,
                'probabilities': predicted_probs,
                'test_size': len(test_y)
            }
            
            logger.info(f"Evaluation completed for {model_name}")
            logger.info(f"Test Accuracy: {accuracy:.4f}")
            if auc_score:
                logger.info(f"AUC Score: {auc_score:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed for {model_name}: {e}")
            raise
    
    def compare_models(self, model_specs: list) -> pd.DataFrame:
        """
        Compare multiple models on test data
        
        Args:
            model_specs: List of dicts with 'name', 'stage', 'version' keys
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for spec in model_specs:
            try:
                eval_result = self.evaluate_model(
                    spec['name'], 
                    spec.get('stage', 'Production'),
                    spec.get('version')
                )
                
                results.append({
                    'model_name': eval_result['model_name'],
                    'stage': eval_result['stage'],
                    'version': eval_result['version'],
                    'test_accuracy': eval_result['test_accuracy'],
                    'auc_score': eval_result['auc_score'],
                    'precision_benign': eval_result['classification_report']['0']['precision'],
                    'recall_benign': eval_result['classification_report']['0']['recall'],
                    'f1_benign': eval_result['classification_report']['0']['f1-score'],
                    'precision_attack': eval_result['classification_report']['1']['precision'],
                    'recall_attack': eval_result['classification_report']['1']['recall'],
                    'f1_attack': eval_result['classification_report']['1']['f1-score'],
                    'test_size': eval_result['test_size']
                })
                
            except Exception as e:
                logger.warning(f"Failed to evaluate {spec['name']}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def plot_confusion_matrix(self, eval_result: Dict[str, Any], save_path: str = None):
        """Plot confusion matrix for evaluation result"""
        plt.figure(figsize=(8, 6))
        
        conf_matrix = eval_result['confusion_matrix']
        sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt='d',
            cmap='Blues',
            xticklabels=['Benign', 'Attack'],
            yticklabels=['Benign', 'Attack']
        )
        
        plt.title(f"Confusion Matrix - {eval_result['model_name']}")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, eval_result: Dict[str, Any], save_path: str = None):
        """Plot ROC curve for evaluation result"""
        if eval_result['roc_curve'] is None:
            logger.warning("ROC curve data not available")
            return
        
        plt.figure(figsize=(8, 6))
        
        fpr = eval_result['roc_curve']['fpr']
        tpr = eval_result['roc_curve']['tpr']
        auc = eval_result['auc_score']
        
        plt.plot(fpr, tpr, linewidth=2, label=f"ROC curve (AUC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f"ROC Curve - {eval_result['model_name']}")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, model_name: str, stage: str = "Production",
                                 version: str = None, save_dir: str = "./reports") -> str:
        """
        Generate comprehensive evaluation report
        
        Args:
            model_name: Name of registered model
            stage: Model stage to evaluate
            version: Specific version
            save_dir: Directory to save report and plots
            
        Returns:
            Path to generated report
        """
        import os
        from datetime import datetime
        
        # Create reports directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Evaluate model
        eval_result = self.evaluate_model(model_name, stage, version)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create report filename
        report_name = f"{model_name}_{stage}_{timestamp}"
        if version:
            report_name = f"{model_name}_v{version}_{timestamp}"
        
        report_path = os.path.join(save_dir, f"{report_name}_report.md")
        
        # Generate plots
        conf_matrix_path = os.path.join(save_dir, f"{report_name}_confusion_matrix.png")
        roc_curve_path = os.path.join(save_dir, f"{report_name}_roc_curve.png")
        
        self.plot_confusion_matrix(eval_result, conf_matrix_path)
        if eval_result['roc_curve']:
            self.plot_roc_curve(eval_result, roc_curve_path)
        
        # Generate markdown report
        report_content = f"""# Model Evaluation Report

## Model Information
- **Name**: {eval_result['model_name']}
- **Stage**: {eval_result['stage']}
- **Version**: {eval_result.get('version', 'Latest')}
- **Evaluation Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Performance Metrics
- **Test Accuracy**: {eval_result['test_accuracy']:.4f}
- **AUC Score**: {eval_result['auc_score']:.4f if eval_result['auc_score'] else 'N/A'}
- **Test Set Size**: {eval_result['test_size']} samples

## Classification Report

### Benign Traffic
- **Precision**: {eval_result['classification_report']['0']['precision']:.4f}
- **Recall**: {eval_result['classification_report']['0']['recall']:.4f}
- **F1-Score**: {eval_result['classification_report']['0']['f1-score']:.4f}

### Attack Traffic
- **Precision**: {eval_result['classification_report']['1']['precision']:.4f}
- **Recall**: {eval_result['classification_report']['1']['recall']:.4f}
- **F1-Score**: {eval_result['classification_report']['1']['f1-score']:.4f}

## Confusion Matrix
![Confusion Matrix]({os.path.basename(conf_matrix_path)})

"""
        
        if eval_result['roc_curve']:
            report_content += f"""## ROC Curve
![ROC Curve]({os.path.basename(roc_curve_path)})

"""
        
        report_content += f"""## Summary
The model achieved {eval_result['test_accuracy']:.2%} accuracy on the test set. """
        
        if eval_result['auc_score']:
            report_content += f"The AUC score of {eval_result['auc_score']:.4f} indicates {'excellent' if eval_result['auc_score'] > 0.9 else 'good' if eval_result['auc_score'] > 0.8 else 'fair'} discriminative performance."
        
        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Evaluation report generated: {report_path}")
        return report_path


def evaluate_production_models() -> pd.DataFrame:
    """Evaluate all production models and return comparison"""
    evaluator = ModelEvaluator()
    registry = NIDSModelRegistry()
    
    models = registry.list_models()
    model_specs = []
    
    for model_name in models:
        # Get production version
        try:
            versions = registry.client.get_latest_versions(model_name, stages=["Production"])
            if versions:
                model_specs.append({
                    'name': model_name,
                    'stage': 'Production',
                    'version': versions[0].version
                })
        except Exception:
            continue
    
    if not model_specs:
        logger.warning("No production models found")
        return pd.DataFrame()
    
    return evaluator.compare_models(model_specs)
