"""
ONNX Runtime Everywhere.

Convert PyTorch models to ONNX, run ONNX inference with provider
selection, and DirectML/CUDA/CPU support detection.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("opencut")


@dataclass
class ONNXProviderInfo:
    """Information about an available ONNX execution provider."""
    name: str
    available: bool = False
    priority: int = 0
    device: str = ""
    details: str = ""


def check_onnx_providers(
    on_progress: Optional[Callable] = None,
) -> List[ONNXProviderInfo]:
    """Check which ONNX Runtime execution providers are available.

    Returns:
        List of ONNXProviderInfo with availability status.
    """
    if on_progress:
        on_progress(20, "Checking ONNX Runtime")

    known_providers = [
        ("CUDAExecutionProvider", 1, "NVIDIA CUDA GPU"),
        ("DmlExecutionProvider", 2, "DirectML (AMD/Intel/NVIDIA on Windows)"),
        ("ROCMExecutionProvider", 3, "AMD ROCm GPU"),
        ("CoreMLExecutionProvider", 4, "Apple Core ML"),
        ("OpenVINOExecutionProvider", 5, "Intel OpenVINO"),
        ("CPUExecutionProvider", 99, "CPU fallback"),
    ]

    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except ImportError:
        if on_progress:
            on_progress(100, "ONNX Runtime not installed")
        return [
            ONNXProviderInfo(name=name, available=False, priority=pri, details=desc)
            for name, pri, desc in known_providers
        ]

    results = []
    for name, pri, desc in known_providers:
        results.append(ONNXProviderInfo(
            name=name,
            available=name in available,
            priority=pri,
            details=desc,
        ))

    if on_progress:
        on_progress(100, f"Found {sum(1 for r in results if r.available)} providers")

    return results


def get_optimal_provider(
    on_progress: Optional[Callable] = None,
) -> str:
    """Determine the best available ONNX execution provider.

    Returns:
        Provider name string for onnxruntime session options.
    """
    providers = check_onnx_providers(on_progress=on_progress)
    available = [p for p in providers if p.available]
    if not available:
        return "CPUExecutionProvider"

    available.sort(key=lambda p: p.priority)
    return available[0].name


def convert_to_onnx(
    model: Any,
    output_path: str,
    input_shape: Optional[Tuple] = None,
    opset_version: int = 17,
    dynamic_axes: Optional[Dict] = None,
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Convert a PyTorch model to ONNX format.

    Args:
        model: PyTorch model (nn.Module) to convert.
        output_path: Path for the output .onnx file.
        input_shape: Example input shape (e.g., (1, 3, 224, 224)).
        opset_version: ONNX opset version.
        dynamic_axes: Dict mapping input/output names to dynamic axis indices.

    Returns:
        Dict with output_path, file_size, and conversion details.
    """
    if on_progress:
        on_progress(10, "Preparing model for export")

    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for ONNX conversion")

    if input_shape is None:
        input_shape = (1, 3, 224, 224)

    model.eval()
    dummy_input = torch.randn(*input_shape)

    if on_progress:
        on_progress(30, "Exporting to ONNX")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    input_names = ["input"]
    output_names = ["output"]

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    if on_progress:
        on_progress(70, "Validating ONNX model")

    # Validate
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
    except ImportError:
        logger.warning("onnx package not available for validation")
    except Exception as exc:
        logger.warning("ONNX validation warning: %s", exc)

    file_size = os.path.getsize(output_path)

    if on_progress:
        on_progress(100, "ONNX conversion complete")

    logger.info("Converted model to ONNX: %s (%d bytes)", output_path, file_size)
    return {
        "output_path": output_path,
        "file_size": file_size,
        "opset_version": opset_version,
        "input_shape": list(input_shape),
    }


def run_onnx_inference(
    model_path: str,
    input_data: Any,
    provider: Optional[str] = None,
    input_name: str = "input",
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Run inference on an ONNX model.

    Args:
        model_path: Path to the .onnx model file.
        input_data: Numpy array of input data.
        provider: Execution provider (auto-selected if None).
        input_name: Name of the input tensor.

    Returns:
        Dict with outputs array and inference metadata.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    if on_progress:
        on_progress(10, "Loading ONNX model")

    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime is required for inference")

    import numpy as np

    if provider is None:
        provider = get_optimal_provider()

    providers = [provider]
    if provider != "CPUExecutionProvider":
        providers.append("CPUExecutionProvider")

    if on_progress:
        on_progress(30, f"Creating session with {provider}")

    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception:
        logger.warning("Failed with %s, falling back to CPU", provider)
        session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        provider = "CPUExecutionProvider"

    if on_progress:
        on_progress(60, "Running inference")

    if not isinstance(input_data, np.ndarray):
        input_data = np.array(input_data, dtype=np.float32)

    outputs = session.run(None, {input_name: input_data})

    if on_progress:
        on_progress(100, "Inference complete")

    return {
        "outputs": [o.tolist() if hasattr(o, "tolist") else o for o in outputs],
        "provider": provider,
        "model_path": model_path,
        "output_count": len(outputs),
    }
