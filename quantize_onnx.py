#!/usr/bin/env python3
"""Quantize MatMul weights in an ONNX model with per-column absmax scales.

For a MatMul weight W with shape KxN, this pass computes one scale per output
column N:

    scale[n] = max(abs(W[:, n])) / qmax
    Wq[:, n] = round(W[:, n] / scale[n])

The graph is rewritten from:

    Y = MatMul(X, W)

to:

    Yq = MatMul(X, Wq)
    Y  = Mul(Yq, scale)

Wq is stored as a float tensor containing integer-valued quantized weights so
that the graph remains valid with floating-point activations and standard ONNX
MatMul type constraints.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

DEFAULT_INPUT = Path("onnx/tiny/encoder_optimized.onnx")
DEFAULT_OUTPUT = Path("onnx/tiny/encoder_quantized.onnx")


class QuantizationError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize ONNX MatMul weights with per-column absmax scales."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input ONNX model. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output quantized ONNX model. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=8,
        help="Signed quantization bit width. Default: 8",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip onnx.checker validation after quantization.",
    )
    parser.add_argument(
        "--no-shape-inference",
        action="store_true",
        help="Skip ONNX shape inference before saving.",
    )
    return parser.parse_args()


def get_initializer_map(graph) -> dict[str, TensorProto]:
    return {initializer.name: initializer for initializer in graph.initializer}


def replace_initializer(graph, name: str, array: np.ndarray) -> None:
    replacement = numpy_helper.from_array(array, name=name)
    for index, initializer in enumerate(graph.initializer):
        if initializer.name == name:
            graph.initializer[index].CopyFrom(replacement)
            return
    raise QuantizationError(f"Initializer not found: {name}")


def make_unique_name(existing: set[str], base: str) -> str:
    candidate = base
    suffix = 0
    while candidate in existing:
        suffix += 1
        candidate = f"{base}_{suffix}"
    existing.add(candidate)
    return candidate


def collect_names(graph) -> set[str]:
    names: set[str] = set()
    for value in graph.input:
        names.add(value.name)
    for value in graph.output:
        names.add(value.name)
    for value in graph.value_info:
        names.add(value.name)
    for initializer in graph.initializer:
        names.add(initializer.name)
    for node in graph.node:
        if node.name:
            names.add(node.name)
        names.update(name for name in node.input if name)
        names.update(name for name in node.output if name)
    return names


def quantize_weight_absmax(
    weight: np.ndarray, bits: int
) -> tuple[np.ndarray, np.ndarray]:
    if weight.ndim != 2:
        raise QuantizationError(f"Expected 2D MatMul weight, got shape {weight.shape}")
    if bits < 2:
        raise QuantizationError("--bits must be at least 2")

    qmax = float((1 << (bits - 1)) - 1)
    weight_f32 = weight.astype(np.float32, copy=False)
    absmax = np.max(np.abs(weight_f32), axis=0)
    scale = absmax / qmax
    scale = np.where(scale == 0.0, 1.0, scale).astype(np.float32)
    quantized = np.round(weight_f32 / scale.reshape(1, -1))
    quantized = np.clip(quantized, -qmax, qmax).astype(np.float32)
    return quantized, scale


def quantize_matmul_weights(model: onnx.ModelProto, bits: int) -> int:
    graph = model.graph
    initializers = get_initializer_map(graph)
    existing_names = collect_names(graph)
    new_nodes: list[onnx.NodeProto] = []
    quantized_initializers: set[str] = set()
    quantized_count = 0

    for node in graph.node:
        new_nodes.append(node)
        if node.op_type != "MatMul" or len(node.input) < 2 or len(node.output) != 1:
            continue

        weight_name = node.input[1]
        if weight_name not in initializers:
            continue

        weight = numpy_helper.to_array(initializers[weight_name])
        if weight.ndim != 2 or not np.issubdtype(weight.dtype, np.floating):
            continue

        quantized, scale = quantize_weight_absmax(weight, bits)
        if weight_name not in quantized_initializers:
            replace_initializer(graph, weight_name, quantized)
            quantized_initializers.add(weight_name)

        original_output = node.output[0]
        quantized_output = make_unique_name(
            existing_names, f"{original_output}_quantized"
        )
        scale_name = make_unique_name(existing_names, f"{weight_name}_absmax_scale")
        mul_name = make_unique_name(
            existing_names, f"{node.name or original_output}_absmax_dequant"
        )

        node.output[0] = quantized_output
        graph.initializer.append(numpy_helper.from_array(scale, name=scale_name))
        new_nodes.append(
            helper.make_node(
                "Mul",
                inputs=[quantized_output, scale_name],
                outputs=[original_output],
                name=mul_name,
            )
        )
        quantized_count += 1

    del graph.node[:]
    graph.node.extend(new_nodes)
    return quantized_count


def save_model(model: onnx.ModelProto, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(output_path), save_as_external_data=False)


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input ONNX model not found: {args.input}")

    model = onnx.load(str(args.input))
    quantized = quantize_matmul_weights(model, args.bits)
    if quantized == 0:
        raise QuantizationError(
            "No MatMul nodes with 2D floating-point initializer weights were quantized"
        )

    if not args.no_shape_inference:
        model = shape_inference.infer_shapes(model)

    if not args.no_check:
        onnx.checker.check_model(model)

    save_model(model, args.output)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Quantized MatMul weights: {quantized}")
    print(f"Bits: {args.bits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
