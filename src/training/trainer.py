"""
Unified training interface for NIDS models
"""
import time
import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, Any, List, Optional
from src.config import config
from src.utils.logging import get_logger
from src.utils.performance import Timer
from src.attacks.pgd import generate_pgd_adversarial_examples

logger = get_logger(__name__)


class NIDSTrainer:
    """Unified NIDS model trainer with gradient tracking and MLflow integration"""
    
    def __init__(self, model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer = None):
        self.model = model
        self.optimizer = optimizer or tf.keras.optimizers.Adam(learning_rate=config.training.learning_rate)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
    def init_gradient_logging(self, log_dir: str = None):
        """Initialize simplified gradient logging"""
        if not config.logging.gradient_logging_enabled:
            return
        logger.debug("Gradient monitoring enabled - will track global gradient norms")
    
    def train(self, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset, 
              training_type: str = "base", attack_rects: List[np.ndarray] = None,
              epochs: int = None, steps_per_epoch: int = None, 
              attack_pattern: str = "hulk") -> Dict[str, Any]:
        """
        Unified training method for both base and adversarial training
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            training_type: "base" or "adversarial"
            attack_rects: Attack rectangles for adversarial training (None uses research patterns)
            epochs: Number of epochs (uses config if None)
            steps_per_epoch: Steps per epoch (uses config if None)
            attack_pattern: Research attack pattern to use when attack_rects is None
            
        Returns:
            Training history
        """
        epochs = epochs or config.training.epochs
        steps_per_epoch = steps_per_epoch or config.training.steps_per_epoch
        
        # Start MLflow run
        run_name = f"{training_type}_{config.model.model_type}_{epochs}ep"
        logger.start_run(run_name=run_name)
        
        # Log parameters
        params = {
            'training_type': training_type,
            'model_type': config.model.model_type,
            'epochs': epochs,
            'steps_per_epoch': steps_per_epoch,
            'learning_rate': config.training.learning_rate,
            'batch_size': config.data.batch_size
        }
        
        if training_type == "adversarial":
            params.update({
                'pgd_epsilon': config.training.pgd_epsilon,
                'pgd_steps': config.training.pgd_steps,
                'pgd_alpha': config.training.pgd_alpha,
                'using_research_hyperrectangles': attack_rects is None
            })
            
            if attack_rects is None:
                logger.info("Using research-based hyperrectangle definitions from original NIDS paper")
            else:
                logger.info(f"Using provided attack rectangles: {len(attack_rects)} constraints")
            
        logger.log_params(params)
        logger.info(f"Starting {training_type} training for {epochs} epochs, {steps_per_epoch} steps per epoch")
        
        # Initialize gradient logging
        self.init_gradient_logging("logs/gradients")
        
        # Training history
        history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'epoch_times': []
        }
        
        # Training loop
        for epoch in range(epochs):
            with Timer() as epoch_timer:
                logger.info(f"Epoch {epoch + 1}/{epochs}")
                
                # Training
                epoch_loss = []
                epoch_acc = []
                epoch_grad_metrics = []
                
                step_count = 0
                global_step = epoch * steps_per_epoch
                
                for x_batch, y_batch in train_dataset:
                    if step_count >= steps_per_epoch:
                        break
                    
                    global_step += 1
                    
                    # Choose training step based on type
                    if training_type == "adversarial":
                        # Use research-based hyperrectangles (attack_rects can be None)
                        metrics = self._adversarial_train_step(x_batch, y_batch, attack_rects, attack_pattern)
                    else:
                        metrics = self._base_train_step(x_batch, y_batch)
                    
                    epoch_loss.append(metrics['loss'])
                    epoch_acc.append(metrics['accuracy'])
                    
                    # Gradient analysis
                    if 'gradients' in metrics and config.logging.gradient_logging_enabled:
                        grad_metrics = self._analyze_gradients(metrics['gradients'], step=global_step, epoch=epoch)
                        epoch_grad_metrics.append(grad_metrics)
                        
                        # Log essential gradient metrics to MLflow periodically
                        if global_step % config.logging.gradient_mlflow_frequency == 0 and grad_metrics:
                            logger.log_gradient_metrics({
                                'grad_global_norm': grad_metrics.get('grad_global_norm', 0)
                            }, step=global_step)
                    
                    step_count += 1
                    
                    if step_count % 100 == 0:
                        logger.debug(f"Step {step_count}/{steps_per_epoch} - "
                                   f"Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}")
                
                # Calculate epoch metrics
                avg_loss = np.mean(epoch_loss)
                avg_acc = np.mean(epoch_acc)
                
                # Validation
                val_loss, val_acc = self._evaluate_model(val_dataset)
                
                # Store metrics
                history['loss'].append(avg_loss)
                history['accuracy'].append(avg_acc)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
                
                # Log to MLflow
                logger.log_metrics({
                    'loss': avg_loss,
                    'accuracy': avg_acc,
                    'val_loss': val_loss,
                    'val_accuracy': avg_acc
                }, step=epoch)
            
            # Store epoch timing and log (after context manager exits)
            epoch_time = epoch_timer.elapsed
            history['epoch_times'].append(epoch_time)
            
            logger.info(f"Epoch {epoch + 1}: Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Time: {epoch_time:.4f}s")
        
        # Log final metrics
        final_metrics = {
            'final_train_loss': history['loss'][-1],
            'final_train_accuracy': history['accuracy'][-1],
            'final_val_loss': history['val_loss'][-1],
            'final_val_accuracy': history['val_accuracy'][-1],
            'total_training_time': sum(history['epoch_times'])
        }
        logger.log_metrics(final_metrics)
        
        logger.info(f"Training completed. Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
        
        # Save model and register in MLflow (industry best practice)
        self._save_and_register_model(training_type, epochs, history)
        
        logger.end_run()
        
        return history
    
    @tf.function
    def _base_train_step(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> Dict[str, float]:
        """Single base training step"""
        with tf.GradientTape() as tape:
            logits = self.model(x_batch, training=True)
            loss = self.loss_fn(y_batch, logits)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        predictions = tf.nn.softmax(logits)
        accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(predictions, axis=1), tf.cast(y_batch, tf.int64)), 
            tf.float32
        ))
        
        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'gradients': gradients
        }
    
    @tf.function
    def _adversarial_train_step(self, x_batch: tf.Tensor, y_batch: tf.Tensor, 
                               attack_rects: List[np.ndarray] = None,
                               attack_pattern: str = "hulk") -> Dict[str, float]:
        """Single adversarial training step using research-based hyperrectangles"""
        # Generate adversarial examples using research hyperrectangles
        # If attack_rects is None, generate_pgd_adversarial_examples will use research patterns
        x_adv = generate_pgd_adversarial_examples(
            self.model, x_batch, y_batch, 
            attack_rects=attack_rects,  # Can be None to use research hyperrectangles
            epsilon=config.training.pgd_epsilon,
            num_steps=config.training.pgd_steps,
            step_size=config.training.pgd_alpha,
            attack_pattern=attack_pattern  # Use specified research attack pattern
        )
        
        # Mix clean and adversarial examples
        batch_size = tf.shape(x_batch)[0]
        indices = tf.random.shuffle(tf.range(batch_size))
        
        x_mixed = tf.gather(tf.concat([x_batch, x_adv], axis=0), indices)
        y_mixed = tf.gather(tf.concat([y_batch, y_batch], axis=0), indices)
        
        # Training step
        with tf.GradientTape() as tape:
            logits = self.model(x_mixed, training=True)
            loss = self.loss_fn(y_mixed, logits)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        predictions = tf.nn.softmax(logits)
        accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(predictions, axis=1), tf.cast(y_mixed, tf.int64)), 
            tf.float32
        ))
        
        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'gradients': gradients
        }
    
    def _analyze_gradients(self, gradients, step: int = 0, epoch: int = 0) -> Dict[str, float]:
        """Simplified gradient analysis focusing on actionable metrics"""
        if not config.logging.gradient_logging_enabled:
            return {}
        
        # Calculate global gradient norm (most important metric)
        global_norm = float(tf.norm([tf.norm(g) for g in gradients if g is not None]).numpy())
        
        # Simple health assessment based on gradient norm
        if global_norm < 1e-6:
            status = "VANISHING"
            action = "Consider increasing learning rate"
        elif global_norm > 100.0:
            status = "EXPLODING" 
            action = "Consider decreasing learning rate or gradient clipping"
        elif global_norm < 1e-3:
            status = "LOW"
            action = "Training might be slow"
        elif global_norm > 10.0:
            status = "HIGH"
            action = "Monitor for instability"
        else:
            status = "HEALTHY"
            action = "Gradients look good"
        
        # Log simplified gradient info
        if config.logging.gradient_console_logging and step % config.logging.gradient_log_frequency == 0:
            logger.info(f"🎯 Gradients: {global_norm:.4f} ({status}) - {action}")
        
        # Return minimal essential metrics
        return {
            'grad_global_norm': global_norm,
            'grad_status': status
        }
    
    def _evaluate_model(self, dataset: tf.data.Dataset) -> Tuple[float, float]:
        """Evaluate model on dataset"""
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        for x_batch, y_batch in dataset:
            logits = self.model(x_batch, training=False)
            loss = self.loss_fn(y_batch, logits)
            
            predictions = tf.nn.softmax(logits)
            accuracy = tf.reduce_mean(tf.cast(
                tf.equal(tf.argmax(predictions, axis=1), tf.cast(y_batch, tf.int64)), 
                tf.float32
            ))
            
            total_loss += float(loss)
            total_acc += float(accuracy)
            num_batches += 1
        
        return total_loss / num_batches, total_acc / num_batches
    
    def _save_and_register_model(self, training_type: str, epochs: int, history: Dict[str, Any]):
        """
        Save model and register in MLflow following industry best practices.
        This ensures model artifacts are properly tracked and linked to experiments.
        """
        import mlflow
        import mlflow.tensorflow
        import os
        
        try:
            # Check if we have an active MLflow run
            if not mlflow.active_run():
                logger.warning("No active MLflow run - cannot save model")
                return
            
            # Create model name following industry naming conventions
            model_name = f"{config.model.model_type}_{training_type}"
            
            # 1. Save model to MLflow artifacts (industry standard)
            logger.info("Saving model to MLflow artifacts...")
            mlflow.tensorflow.log_model(
                model=self.model,
                artifact_path="model"
            )
            
            # 2. Log model metadata and performance summary
            model_metadata = {
                'total_parameters': self.model.count_params(),
                'trainable_parameters': sum([tf.size(var).numpy() for var in self.model.trainable_variables]),
                'model_architecture': config.model.model_type,
                'training_method': training_type,
                'final_performance': {
                    'train_accuracy': history['accuracy'][-1],
                    'val_accuracy': history['val_accuracy'][-1],
                    'train_loss': history['loss'][-1],
                    'val_loss': history['val_loss'][-1]
                }
            }
            
            # Log as MLflow artifacts and metrics
            mlflow.log_dict(model_metadata, "model_metadata.json")
            mlflow.log_metrics({
                'total_parameters': model_metadata['total_parameters'],
                'trainable_parameters': model_metadata['trainable_parameters']
            })
            
            # 3. Register in model registry
            try:
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
                registry_name = f"nids-{config.experiment.attack_name.lower()}-{model_name}"
                
                mlflow.register_model(
                    model_uri=model_uri,
                    name=registry_name
                )
                logger.info(f"Model registered in MLflow registry: {registry_name}")
            except Exception as reg_error:
                logger.warning(f"Model registry registration failed (non-critical): {reg_error}")
            
            logger.info(f"✅ Model successfully saved and registered:")
            logger.info(f"   - MLflow artifacts: model/")
            logger.info(f"   - Registry: {registry_name}")
            
        except Exception as e:
            logger.error(f"Failed to save/register model: {e}")
    
    def _create_model_signature(self):
        """Create MLflow model signature for better model documentation"""
        try:
            import mlflow
            from mlflow.models.signature import infer_signature
            import numpy as np
            
            # Create sample input matching your model's expected input
            sample_input = np.random.rand(1, config.model.input_size).astype(np.float32)
            sample_output = self.model(sample_input, training=False)
            
            return infer_signature(sample_input, sample_output.numpy())
        except Exception as e:
            logger.debug(f"Could not create model signature: {e}")
            return None
    
    def _get_model_input_example(self):
        """Get input example for MLflow model documentation"""
        try:
            import numpy as np
            return np.random.rand(1, config.model.input_size).astype(np.float32)
        except Exception as e:
            logger.debug(f"Could not create input example: {e}")
            return None


# Compatibility functions for existing code
def train_adversarial(model: tf.keras.Model, train_dataset: tf.data.Dataset,
                     val_dataset: tf.data.Dataset, attack_rects: List[np.ndarray] = None,
                     epochs: int = None, steps_per_epoch: int = None) -> Dict[str, Any]:
    """
    Compatibility wrapper for adversarial training using research-based hyperrectangles.
    
    Args:
        attack_rects: Optional attack rectangles. If None, uses research hyperrectangles.
    """
    trainer = NIDSTrainer(model)
    return trainer.train(train_dataset, val_dataset, "adversarial", attack_rects, epochs, steps_per_epoch)


def train_base(model: tf.keras.Model, train_dataset: tf.data.Dataset,
               val_dataset: tf.data.Dataset, epochs: int = None, 
               steps_per_epoch: int = None) -> Dict[str, Any]:
    """Compatibility wrapper for base training"""
    trainer = NIDSTrainer(model)
    return trainer.train(train_dataset, val_dataset, "base", None, epochs, steps_per_epoch)


def evaluate_model(model: tf.keras.Model, dataset: tf.data.Dataset) -> Tuple[float, float]:
    """Compatibility wrapper for model evaluation"""
    trainer = NIDSTrainer(model)
    return trainer._evaluate_model(dataset)
