"""
Streamlined model utilities for NIDS
"""
import os
import tensorflow as tf
from typing import Optional
from src.config import config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def save_model(model: tf.keras.Model, model_name: str, epoch: int = None, 
               save_format: str = 'keras', register_in_mlflow: bool = True) -> str:
    """
    Save and optionally register model
    
    Args:
        model: Trained Keras model
        model_name: Name for the model
        epoch: Optional epoch number for filename
        save_format: 'keras', 'h5', or 'onnx'
        register_in_mlflow: Whether to register in MLflow Model Registry
        
    Returns:
        Path where model was saved
    """
    # Determine save path and extension
    if save_format == 'keras':
        base_dir = os.path.join(config.model.tf_model_dir, config.experiment.attack_name)
        extension = ".keras"  # Use native Keras format
    elif save_format == 'h5':
        base_dir = os.path.join(config.model.tf_model_dir, config.experiment.attack_name)
        extension = ".h5"
    elif save_format == 'onnx':
        base_dir = os.path.join(config.model.onnx_model_dir, config.experiment.attack_name)
        extension = ".onnx"
    else:
        raise ValueError(f"Unsupported save format: {save_format}")
    
    # Create filename
    if epoch is not None:
        filename = f"{model_name}_epoch_{epoch}{extension}"
    else:
        filename = f"{model_name}{extension}"
    
    save_path = os.path.join(base_dir, filename)
    os.makedirs(base_dir, exist_ok=True)
    
    if save_format in ['keras', 'h5']:
        # Save TensorFlow model
        model.save(save_path)
        logger.info(f"Model saved to {save_path}")
        
        # Register in MLflow if requested
        if register_in_mlflow:
            _register_tf_model(model, model_name, save_path)
            
    elif save_format == 'onnx':
        # Convert and save ONNX model
        _save_onnx_model(model, save_path, model_name, register_in_mlflow)
    
    return save_path


def _register_tf_model(model: tf.keras.Model, model_name: str, save_path: str):
    """Register TensorFlow model in MLflow"""
    try:
        import mlflow
        if not mlflow.active_run():
            logger.debug("No active MLflow run - skipping model registration")
            return
            
        logger.debug("Registering model in MLflow registry...")
        
        # Create model signature
        try:
            from mlflow.models.signature import infer_signature
            import numpy as np
            
            sample_input = np.random.random((1, model.input.shape[-1])).astype(np.float32)
            predictions = model(sample_input, training=False)
            signature = infer_signature(sample_input, predictions)
            logger.debug("Model signature created successfully")
        except Exception as e:
            logger.debug(f"Failed to create model signature: {e}")
            signature = None
            sample_input = None
        
        # Get run info for model description
        run = mlflow.active_run()
        training_type = "adversarial" if "adversarial" in model_name else "base"
        
        description = (
            f"NIDS {training_type} model - {config.model.model_type} architecture. "
            f"Trained for network intrusion detection on DoS dataset. "
            f"Run ID: {run.info.run_id}"
        )
        
        tags = {
            "model_type": config.model.model_type,
            "training_type": training_type,
            "framework": "tensorflow",
            "task": "binary_classification",
            "dataset": "DoS"
        }
        
        logger.debug(f"Registering model with name: nids_{model_name}")
        
        # Register model
        model_version = logger.register_model(
            model=model,
            model_name=f"nids_{model_name}",
            description=description,
            tags=tags,
            signature=signature,
            input_example=sample_input if sample_input is not None else None
        )
        
        if model_version:
            logger.info(f"Model registered successfully: nids_{model_name} v{model_version.version}")
            
            # Automatically transition to Staging for review
            logger.transition_model_stage(
                model_name=f"nids_{model_name}",
                version=str(model_version.version),
                stage="Staging"
            )
        else:
            logger.warning("Model registration returned None")
            
    except Exception as e:
        logger.error(f"MLflow model registration failed: {e}", exc_info=True)


def _save_onnx_model(model: tf.keras.Model, output_path: str, model_name: str, register_in_mlflow: bool):
    """Convert and save model to ONNX format"""
    try:
        import tf2onnx
        import onnx
        
        spec = (tf.TensorSpec(model.input.shape, tf.float32, name="input"),)
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
        onnx.save(model_proto, output_path)
        logger.info(f"Model converted and saved to {output_path}")
        
        # Register ONNX model in MLflow if requested
        if register_in_mlflow:
            try:
                import mlflow
                if mlflow.active_run():
                    logger.log_artifact(output_path, f"models/{model_name}")
                    
                    # Register ONNX model
                    run = mlflow.active_run()
                    description = (
                        f"NIDS ONNX model - {config.model.model_type} architecture. "
                        f"Converted from TensorFlow for deployment. "
                        f"Run ID: {run.info.run_id}"
                    )
                    
                    model_uri = f"runs:/{run.info.run_id}/models/{model_name}"
                    mlflow.register_model(
                        model_uri=model_uri,
                        name=f"nids_{model_name}_onnx",
                        description=description
                    )
                    
            except Exception as e:
                logger.debug(f"MLflow ONNX model registration failed: {e}")
                
    except ImportError:
        logger.error("tf2onnx not available for ONNX conversion")
        raise
