"""
Performance optimization and monitoring utilities (PyTorch)
"""

import platform
import time
from contextlib import contextmanager
from typing import Any

import psutil
import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)


def configure_torch_performance() -> None:
    """Configure PyTorch for optimal performance"""
    try:
        # Enable cuDNN benchmark for optimized algorithms when input sizes are static
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # Use higher precision matmul on Ampere+ GPUs for better throughput
        # Use higher matmul precision if supported (PyTorch >= 2.0)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        logger.info("✓ PyTorch performance configuration applied")

    except Exception as e:
        logger.error(f"Failed to configure PyTorch: {e}")


def get_system_info() -> dict[str, Any]:
    """Get comprehensive system information"""
    memory = psutil.virtual_memory()
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_devices = []
    if gpu_count:
        for i in range(gpu_count):
            try:
                gpu_devices.append(torch.cuda.get_device_name(i))
            except Exception:
                gpu_devices.append(f"cuda:{i}")

    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "memory_total_gb": round(memory.total / (1024**3), 2),
        "memory_available_gb": round(memory.available / (1024**3), 2),
        "gpu_count": gpu_count,
        "gpu_devices": gpu_devices,
    }


def log_system_info() -> None:
    """Log system information for debugging/monitoring"""
    info = get_system_info()
    logger.info("=== System Information ===")
    logger.info(f"Platform: {info['platform']}")
    logger.info(f"Python: {info['python_version']}")
    logger.info(f"PyTorch: {info['torch_version']}")
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


def get_model_info(model: Any) -> dict[str, Any]:
    """Get model information for monitoring (PyTorch)"""
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        layer_count = sum(1 for _ in model.modules()) - 1  # exclude the root module

        return {
            "model_name": model.__class__.__name__,
            "total_parameters": int(total_params),
            "trainable_parameters": int(trainable_params),
            "non_trainable_parameters": int(total_params - trainable_params),
            "layer_count": layer_count,
            "estimated_size_mb": round((total_params * 4) / (1024 * 1024), 2),  # Assuming float32
            "input_shape": None,
            "output_shape": None,
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
        # Avoid attribute type annotations inside method for compatibility
        self.elapsed = 0.0
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            self.elapsed = time.time() - self.start_time
        else:
            self.elapsed = 0.0
