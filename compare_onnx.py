#!/usr/bin/env python3
"""Compare two ONNX models by running them on the same input tensor."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DEFAULT_INPUT_NAME = "mel"
DEFAULT_SHAPE = "1,80,3000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two ONNX models with the same tensor and compare outputs."
    )
    parser.add_argument("model_a", type=Path, help="First ONNX model path.")
    parser.add_argument("model_b", type=Path, help="Second ONNX model path.")
    parser.add_argument(
        "--input-name",
        default=DEFAULT_INPUT_NAME,
        help=f"Input tensor name to feed. Default: {DEFAULT_INPUT_NAME}",
    )
    parser.add_argument(
        "--shape",
        default=DEFAULT_SHAPE,
        help=f"Comma-separated random input shape. Default: {DEFAULT_SHAPE}",
    )
    parser.add_argument(
        "--input-npy",
        type=Path,
        default=None,
        help="Optional .npy tensor to use instead of generated random input.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16"),
        default="float32",
        help="Generated input dtype. Ignored when --input-npy is used. Default: float32",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for generated input. Default: 0",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Standard deviation for generated normal input. Default: 1.0",
    )
    parser.add_argument(
        "--provider",
        choices=("cpu", "cuda", "tensorrt"),
        default="cpu",
        help="ONNX Runtime execution provider. Default: cpu",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
        help="Relative tolerance for allclose. Default: 1e-4",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for allclose. Default: 1e-5",
    )
    return parser.parse_args()


def import_onnxruntime():
    try:
        import onnxruntime as ort

        return ort
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The 'onnxruntime' package is not installed in this Python environment.\n"
            "Install it with:\n"
            "  python -m pip install -U onnxruntime"
        ) from exc


def parse_shape(shape: str) -> tuple[int, ...]:
    try:
        dims = tuple(int(part.strip()) for part in shape.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid --shape value: {shape!r}") from exc
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError("--shape must contain positive integer dimensions")
    return dims


def select_providers(ort, provider: str) -> list[str]:
    available = set(ort.get_available_providers())
    requested = {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
    }
    selected = requested[provider]
    if selected not in available:
        raise ValueError(
            f"ONNX Runtime provider is unavailable: {selected}. "
            f"Available providers: {sorted(available)}"
        )
    providers = [selected]
    if selected != "CPUExecutionProvider" and "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    return providers


def make_input(args: argparse.Namespace) -> np.ndarray:
    if args.input_npy is not None:
        if not args.input_npy.is_file():
            raise FileNotFoundError(f"Input tensor file not found: {args.input_npy}")
        return np.load(args.input_npy)

    shape = parse_shape(args.shape)
    dtype = np.dtype(args.dtype)
    rng = np.random.default_rng(args.seed)
    return (rng.standard_normal(shape) * args.scale).astype(dtype)


def run_model(session, input_name: str, input_tensor: np.ndarray) -> list[np.ndarray]:
    model_inputs = {model_input.name for model_input in session.get_inputs()}
    if input_name not in model_inputs:
        raise ValueError(
            f"Input {input_name!r} not found in model. Available inputs: {sorted(model_inputs)}"
        )
    return session.run(None, {input_name: input_tensor})


def compute_metrics(
    a: np.ndarray, b: np.ndarray, rtol: float, atol: float
) -> dict[str, float | bool | tuple[int, ...]]:
    if a.shape != b.shape:
        return {
            "shape_a": a.shape,
            "shape_b": b.shape,
            "same_shape": False,
            "allclose": False,
        }

    a64 = a.astype(np.float64, copy=False)
    b64 = b.astype(np.float64, copy=False)
    diff = a64 - b64
    abs_diff = np.abs(diff)
    abs_a = np.abs(a64)
    abs_b = np.abs(b64)
    denom = np.maximum(np.maximum(abs_a, abs_b), 1e-12)
    rel_diff = abs_diff / denom
    flat_abs = abs_diff.reshape(-1)
    size = int(flat_abs.size)
    nonzero = int(np.count_nonzero(abs_diff))

    return {
        "shape": a.shape,
        "same_shape": True,
        "allclose": bool(np.allclose(a64, b64, rtol=rtol, atol=atol)),
        "max_abs": float(np.max(abs_diff)) if size else 0.0,
        "mean_abs": float(np.mean(abs_diff)) if size else 0.0,
        "median_abs": float(np.median(abs_diff)) if size else 0.0,
        "p95_abs": float(np.percentile(abs_diff, 95)) if size else 0.0,
        "p99_abs": float(np.percentile(abs_diff, 99)) if size else 0.0,
        "rmse": float(np.sqrt(np.mean(diff * diff))) if size else 0.0,
        "max_rel": float(np.max(rel_diff)) if size else 0.0,
        "mean_rel": float(np.mean(rel_diff)) if size else 0.0,
        "cosine_similarity": cosine_similarity(a64, b64),
        "nonzero_diff": nonzero,
        "nonzero_diff_pct": float(nonzero * 100.0 / size) if size else 0.0,
    }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom == 0:
        return 1.0 if np.array_equal(a_flat, b_flat) else 0.0
    return float(np.dot(a_flat, b_flat) / denom)


def print_metrics(output_index: int, metrics: dict[str, object]) -> None:
    print(f"Output {output_index}:")
    if not metrics.get("same_shape", False):
        print(f"  shape_a: {metrics['shape_a']}")
        print(f"  shape_b: {metrics['shape_b']}")
        print("  same_shape: False")
        print("  allclose: False")
        return

    print(f"  shape: {metrics['shape']}")
    for key in [
        "allclose",
        "max_abs",
        "mean_abs",
        "median_abs",
        "p95_abs",
        "p99_abs",
        "rmse",
        "max_rel",
        "mean_rel",
        "cosine_similarity",
        "nonzero_diff",
        "nonzero_diff_pct",
    ]:
        print(f"  {key}: {metrics[key]}")


def main() -> int:
    args = parse_args()
    if not args.model_a.is_file():
        raise FileNotFoundError(f"Model not found: {args.model_a}")
    if not args.model_b.is_file():
        raise FileNotFoundError(f"Model not found: {args.model_b}")

    ort = import_onnxruntime()
    providers = select_providers(ort, args.provider)
    input_tensor = make_input(args)

    session_a = ort.InferenceSession(str(args.model_a), providers=providers)
    session_b = ort.InferenceSession(str(args.model_b), providers=providers)
    outputs_a = run_model(session_a, args.input_name, input_tensor)
    outputs_b = run_model(session_b, args.input_name, input_tensor)

    print(f"Model A: {args.model_a}")
    print(f"Model B: {args.model_b}")
    print(f"Provider: {providers}")
    print(
        f"Input: {args.input_name} shape={input_tensor.shape} dtype={input_tensor.dtype}"
    )
    print(f"Outputs: model_a={len(outputs_a)} model_b={len(outputs_b)}")

    if len(outputs_a) != len(outputs_b):
        raise ValueError(
            f"Output count mismatch: model_a={len(outputs_a)} model_b={len(outputs_b)}"
        )

    all_close = True
    for index, (output_a, output_b) in enumerate(zip(outputs_a, outputs_b)):
        metrics = compute_metrics(output_a, output_b, args.rtol, args.atol)
        print_metrics(index, metrics)
        all_close = all_close and bool(metrics.get("allclose", False))

    print(f"Overall allclose: {all_close}")
    return 0 if all_close else 1


if __name__ == "__main__":
    raise SystemExit(main())
