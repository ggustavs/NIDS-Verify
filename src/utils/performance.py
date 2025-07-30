"""
Performance optimization and monitoring utilities
"""

import platform
import time
from contextlib import contextmanager
from typing import Any

import psutil
import tensorflow as tf

from src.utils.logging import get_logger

logger = get_logger(__name__)


def configure_tensorflow_performance() -> None:
    """Configure TensorFlow for optimal performance"""
    try:
        # Configure GPU memory growth
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"✓ GPU memory growth configured for {len(gpus)} GPU(s)")
            except RuntimeError as e:
                logger.warning(f"GPU configuration failed: {e}")

        # Set threading configuration for CPU optimization
        tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available cores
        tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all available cores

        # Enable mixed precision if supported (optional, can improve performance)
        try:
            policy = tf.keras.mixed_precision.Policy("mixed_float16")
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.debug("Mixed precision enabled")
        except Exception as e:
            logger.debug(f"Mixed precision not available: {e}")

        logger.info("✓ TensorFlow performance configuration applied")

    except Exception as e:
        logger.error(f"Failed to configure TensorFlow: {e}")


def get_system_info() -> dict[str, Any]:
    """Get comprehensive system information"""
    memory = psutil.virtual_memory()
    gpus = tf.config.list_physical_devices("GPU")

    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "memory_total_gb": round(memory.total / (1024**3), 2),
        "memory_available_gb": round(memory.available / (1024**3), 2),
        "gpu_count": len(gpus),
        "gpu_devices": [gpu.name for gpu in gpus] if gpus else [],
    }


def log_system_info() -> None:
    """Log system information for debugging/monitoring"""
    info = get_system_info()
    logger.info("=== System Information ===")
    logger.info(f"Platform: {info['platform']}")
    logger.info(f"Python: {info['python_version']}")
    logger.info(f"TensorFlow: {info['tensorflow_version']}")
    logger.info(
        f"CPU: {info['cpu_count_physical']} physical, {info['cpu_count_logical']} logical cores"
    )
    logger.info(
        f"Memory: {info['memory_total_gb']} GB total, {info['memory_available_gb']} GB available"
    )
    if info["gpu_count"] > 0:
        logger.info(f"GPU: {info['gpu_count']} device(s) - {', '.join(info['gpu_devices'])}")
    else:
        logger.info("GPU: No devices found")


def get_model_info(model: tf.keras.Model) -> dict[str, Any]:
    """Get model information for monitoring"""
    try:
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])

        return {
            "model_name": getattr(model, "name", "unnamed"),
            "total_parameters": int(total_params),
            "trainable_parameters": int(trainable_params),
            "non_trainable_parameters": int(total_params - trainable_params),
            "layer_count": len(model.layers),
            "estimated_size_mb": round((total_params * 4) / (1024 * 1024), 2),  # Assuming float32
            "input_shape": getattr(model, "input_shape", None),
            "output_shape": getattr(model, "output_shape", None),
        }
    except Exception as e:
        logger.warning(f"Failed to get model info: {e}")
        return {"error": str(e)}


def get_memory_usage() -> dict[str, float]:
    """Get current system memory usage"""
    memory = psutil.virtual_memory()
    return {
        "used_gb": round(memory.used / (1024**3), 2),
        "available_gb": round(memory.available / (1024**3), 2),
        "percent": memory.percent,
    }


def get_process_memory() -> float:
    """Get current process memory usage in MB"""
    try:
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except Exception as e:
        logger.error(f"Failed to get process memory: {e}")
        return 0.0


@contextmanager
def timer(operation_name: str = "Operation"):
    """Context manager for timing operations"""
    start_time = time.time()
    logger.debug(f"Starting {operation_name}")

    try:
        yield
    finally:
        duration = time.time() - start_time
        if duration < 60:
            logger.info(f"{operation_name} completed in {duration:.2f} seconds")
        else:
            minutes = duration // 60
            seconds = duration % 60
            logger.info(f"{operation_name} completed in {int(minutes)}m {seconds:.2f}s")


@contextmanager
def performance_monitor(operation_name: str = "Operation", log_memory: bool = True):
    """Context manager for comprehensive performance monitoring"""
    start_time = time.time()
    start_memory = get_memory_usage() if log_memory else None

    logger.info(f"Starting {operation_name}")
    if start_memory:
        logger.debug(
            f"Initial memory: {start_memory['used_gb']:.1f}GB ({start_memory['percent']:.1f}%)"
        )

    try:
        yield
    finally:
        duration = time.time() - start_time

        # Log timing
        if duration < 60:
            logger.info(f"{operation_name} completed in {duration:.2f} seconds")
        else:
            minutes = duration // 60
            seconds = duration % 60
            logger.info(f"{operation_name} completed in {int(minutes)}m {seconds:.2f}s")

        # Log memory changes if requested
        if log_memory and start_memory:
            end_memory = get_memory_usage()
            memory_diff = end_memory["used_gb"] - start_memory["used_gb"]
            logger.debug(f"Memory change: {memory_diff:+.1f}GB")
            logger.debug(
                f"Final memory: {end_memory['used_gb']:.1f}GB ({end_memory['percent']:.1f}%)"
            )


def log_training_summary(epoch: int, metrics: dict[str, float], epoch_time: float) -> None:
    """Log comprehensive training summary for an epoch"""
    summary_parts = [
        f"Train acc: {metrics.get('train_accuracy', 0):.4f}",
        f"Train loss: {metrics.get('train_loss', 0):.4f}",
    ]

    # Add validation metrics if available
    if "valid_accuracy" in metrics:
        summary_parts.extend(
            [
                f"Valid acc: {metrics['valid_accuracy']:.4f}",
                f"Valid loss: {metrics['valid_loss']:.4f}",
            ]
        )

    # Add test and adversarial metrics
    summary_parts.extend(
        [
            f"Test acc: {metrics.get('test_accuracy', 0):.4f}",
            f"Test loss: {metrics.get('test_loss', 0):.4f}",
            f"Adv acc: {metrics.get('adversarial_accuracy', 0):.4f}",
            f"Time: {epoch_time:.2f}s",
        ]
    )

    logger.info(f"Epoch {epoch} - " + " | ".join(summary_parts))


class Timer:
    """Simple timer class that stores elapsed time"""

    def __init__(self):
        self.elapsed = 0
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time
