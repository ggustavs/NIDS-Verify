"""ONNX export for Vehicle/Marabou verification."""

import os
import time

import torch
import torch.nn as nn
from loguru import logger

from src.config import config


def export_model_to_onnx(
    model: nn.Module,
    output_path: str | None = None,
    input_size: int = 42,
    opset_version: int = 11,
) -> str:
    """Export a PyTorch model to ONNX (opset 11 for Marabou compatibility)."""
    if output_path is None:
        os.makedirs(config.model.onnx_model_dir, exist_ok=True)
        output_path = os.path.join(config.model.onnx_model_dir, f"nids_model_{int(time.time())}.onnx")

    device = next(model.parameters()).device
    model.eval()
    model.cpu()
    try:
        torch.onnx.export(
            model,
            (torch.randn(1, input_size),),
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        logger.info(f"Exported ONNX model to {output_path}")
        return output_path
    finally:
        model.to(device)


def verify_onnx_export(onnx_path: str, model: nn.Module, input_size: int = 42) -> bool:
    """Sanity-check that ONNX inference matches PyTorch within tolerance."""
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed, skipping verification")
        return True

    test_input = torch.randn(1, input_size)
    model.eval()
    with torch.no_grad():
        torch_output = model(test_input).numpy()

    session = ort.InferenceSession(onnx_path)
    onnx_output = session.run(None, {"input": test_input.numpy()})[0]

    max_diff = abs(torch_output - onnx_output).max()
    tolerance = 1e-5
    if max_diff < tolerance:
        logger.info(f"ONNX export matches PyTorch (max diff: {max_diff:.2e})")
        return True
    logger.warning(f"ONNX export diverges (max diff: {max_diff:.2e})")
    return False
