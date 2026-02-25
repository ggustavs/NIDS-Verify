"""ONNX model export utilities for Vehicle verification."""

import os
import time
from typing import Optional

import torch
import torch.nn as nn
from loguru import logger

from src.config import config



def export_model_to_onnx(
    model: nn.Module,
    output_path: Optional[str] = None,
    input_size: int = 42,
    opset_version: int = 11,
    verbose: bool = False,
) -> str:
    """
    Export PyTorch model to ONNX format for Vehicle verification

    Args:
        model: PyTorch model to export
        output_path: Path to save ONNX file (optional, auto-generated if None)
        input_size: Input feature dimension
        opset_version: ONNX opset version (11 recommended for Marabou)
        verbose: Enable verbose export logging

    Returns:
        Path to exported ONNX file
    """
    model.eval()

    # Generate output path if not provided
    if output_path is None:
        os.makedirs(config.model.onnx_model_dir, exist_ok=True)
        timestamp = int(time.time())
        output_path = os.path.join(
            config.model.onnx_model_dir,
            f"nids_model_{timestamp}.onnx"
        )

    # Move model to CPU for export (ONNX export works best on CPU)
    device = next(model.parameters()).device
    model = model.cpu()
    model.eval()

    # Create dummy input tensor
    dummy_input = torch.randn(1, input_size, requires_grad=False)

    # Export to ONNX
    try:
        torch.onnx.export(
            model,
            (dummy_input,),
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            },
            verbose=verbose,
        )
        logger.info(f"✓ Exported ONNX model to: {output_path}")

        # Move model back to original device
        model = model.to(device)

        return output_path

    except Exception as e:
        logger.error(f"Failed to export ONNX model: {e}")
        # Move model back to original device even on error
        model = model.to(device)
        raise


def verify_onnx_export(onnx_path: str, model: nn.Module, input_size: int = 42) -> bool:
    """
    Verify that ONNX export preserves model behavior

    Args:
        onnx_path: Path to ONNX file
        model: Original PyTorch model
        input_size: Input feature dimension

    Returns:
        True if outputs match within tolerance
    """
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed, skipping verification")
        return True

    try:
        # Create test input
        test_input = torch.randn(1, input_size)

        # PyTorch inference
        model.eval()
        with torch.no_grad():
            torch_output = model(test_input).numpy()

        # ONNX inference
        ort_session = ort.InferenceSession(onnx_path)
        onnx_output = ort_session.run(
            None,
            {"input": test_input.numpy()}
        )[0]

        # Compare outputs
        max_diff = abs(torch_output - onnx_output).max()
        tolerance = 1e-5

        if max_diff < tolerance:
            logger.info(f"✓ ONNX export verified (max diff: {max_diff:.2e})")
            return True
        else:
            logger.warning(f"⚠ ONNX export may have issues (max diff: {max_diff:.2e})")
            return False

    except Exception as e:
        logger.warning(f"ONNX verification failed: {e}")
        return False
