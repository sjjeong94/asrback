#!/usr/bin/env python3
"""Optimize exported Whisper encoder ONNX graphs.

The first pass rewrites Attention nodes from the exported 4D layout to ONNX
Attention's 3D layout:

    Q/K/V: Linear -> Reshape -> Transpose -> Attention
    Y:     Attention -> Transpose -> Reshape -> Linear

becomes:

    Q/K/V: Linear -> Attention
    Y:     Attention -> Linear

The pass sets q_num_heads and kv_num_heads on each rewritten Attention node.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

DEFAULT_INPUT = Path("onnx/tiny/encoder.onnx")
DEFAULT_OUTPUT = Path("onnx/tiny/encoder_optimized.onnx")


class OptimizationError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize a Whisper encoder ONNX model."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input encoder ONNX model. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output optimized ONNX model. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip onnx.checker validation after optimization.",
    )
    parser.add_argument(
        "--no-shape-inference",
        action="store_true",
        help="Skip ONNX shape inference before saving.",
    )
    parser.add_argument(
        "--external-data-threshold",
        type=int,
        default=1024,
        help="Tensor size threshold in bytes for external data when saving. Default: 1024",
    )
    return parser.parse_args()


def get_initializer_map(graph) -> dict[str, TensorProto]:
    return {initializer.name: initializer for initializer in graph.initializer}


def get_producer_map(graph) -> dict[str, onnx.NodeProto]:
    producer: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for output in node.output:
            producer[output] = node
    return producer


def get_consumer_map(graph) -> dict[str, list[onnx.NodeProto]]:
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)
    return consumers


def initializer_array(initializers: dict[str, TensorProto], name: str):
    if name not in initializers:
        raise OptimizationError(f"Missing initializer: {name}")
    return numpy_helper.to_array(initializers[name])


def require_single_consumer(
    consumers: dict[str, list[onnx.NodeProto]], tensor_name: str
) -> onnx.NodeProto:
    tensor_consumers = consumers.get(tensor_name, [])
    if len(tensor_consumers) != 1:
        raise OptimizationError(
            f"Expected tensor {tensor_name!r} to have one consumer, "
            f"found {len(tensor_consumers)}"
        )
    return tensor_consumers[0]


def get_ints_attribute(node: onnx.NodeProto, name: str) -> list[int] | None:
    for attr in node.attribute:
        if attr.name == name:
            return list(helper.get_attribute_value(attr))
    return None


def set_int_attribute(node: onnx.NodeProto, name: str, value: int) -> None:
    kept = [attr for attr in node.attribute if attr.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute(name, value))


def remove_nodes(graph, nodes_to_remove: Iterable[onnx.NodeProto]) -> None:
    remove_ids = {id(node) for node in nodes_to_remove}
    kept = [node for node in graph.node if id(node) not in remove_ids]
    del graph.node[:]
    graph.node.extend(kept)


def remove_unused_initializers(graph) -> int:
    used_inputs = {input_name for node in graph.node for input_name in node.input}
    used_inputs.update(input_value.name for input_value in graph.input)
    kept = [init for init in graph.initializer if init.name in used_inputs]
    removed = len(graph.initializer) - len(kept)
    del graph.initializer[:]
    graph.initializer.extend(kept)
    return removed


def match_attention_layout(
    attention: onnx.NodeProto,
    producers: dict[str, onnx.NodeProto],
    consumers: dict[str, list[onnx.NodeProto]],
    initializers: dict[str, TensorProto],
) -> tuple[list[str], str, int, list[onnx.NodeProto]] | None:
    if len(attention.input) < 3 or len(attention.output) < 1:
        return None

    replacement_inputs: list[str] = []
    nodes_to_remove: list[onnx.NodeProto] = []
    num_heads: int | None = None

    for input_name in attention.input[:3]:
        transpose = producers.get(input_name)
        if transpose is None or transpose.op_type != "Transpose":
            return None
        if get_ints_attribute(transpose, "perm") != [0, 2, 1, 3]:
            return None
        if require_single_consumer(consumers, transpose.output[0]) is not attention:
            return None

        reshape = producers.get(transpose.input[0])
        if reshape is None or reshape.op_type != "Reshape":
            return None
        if require_single_consumer(consumers, reshape.output[0]) is not transpose:
            return None
        if len(reshape.input) < 2:
            return None

        shape = initializer_array(initializers, reshape.input[1]).tolist()
        if len(shape) != 4:
            return None
        current_num_heads = int(shape[2])
        if current_num_heads <= 0:
            return None
        if num_heads is None:
            num_heads = current_num_heads
        elif num_heads != current_num_heads:
            raise OptimizationError(
                f"Mismatched attention head counts around {attention.name}: "
                f"{num_heads} vs {current_num_heads}"
            )

        replacement_inputs.append(reshape.input[0])
        nodes_to_remove.extend([reshape, transpose])

    post_transpose = require_single_consumer(consumers, attention.output[0])
    if post_transpose.op_type != "Transpose":
        return None
    if get_ints_attribute(post_transpose, "perm") != [0, 2, 1, 3]:
        return None

    post_reshape = require_single_consumer(consumers, post_transpose.output[0])
    if post_reshape.op_type != "Reshape":
        return None
    if len(post_reshape.input) < 2:
        return None
    output_shape = initializer_array(initializers, post_reshape.input[1]).tolist()
    if len(output_shape) != 3:
        return None

    nodes_to_remove.extend([post_transpose, post_reshape])
    if num_heads is None:
        return None
    return replacement_inputs, post_reshape.output[0], num_heads, nodes_to_remove


def rewrite_attention_to_3d(model: onnx.ModelProto) -> int:
    graph = model.graph
    initializers = get_initializer_map(graph)
    producers = get_producer_map(graph)
    consumers = get_consumer_map(graph)

    nodes_to_remove: list[onnx.NodeProto] = []
    rewritten = 0

    for node in list(graph.node):
        if node.op_type != "Attention":
            continue

        match = match_attention_layout(node, producers, consumers, initializers)
        if match is None:
            continue

        replacement_inputs, replacement_output, num_heads, matched_nodes = match
        for index, input_name in enumerate(replacement_inputs):
            node.input[index] = input_name
        node.output[0] = replacement_output
        set_int_attribute(node, "q_num_heads", num_heads)
        set_int_attribute(node, "kv_num_heads", num_heads)
        nodes_to_remove.extend(matched_nodes)
        rewritten += 1

    if nodes_to_remove:
        remove_nodes(graph, nodes_to_remove)
    remove_unused_initializers(graph)
    return rewritten


def save_model(model: onnx.ModelProto, output_path: Path, threshold: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{output_path.name}.data",
        size_threshold=threshold,
        convert_attribute=False,
    )


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input ONNX model not found: {args.input}")

    model = onnx.load(str(args.input))
    rewritten = rewrite_attention_to_3d(model)
    if rewritten == 0:
        raise OptimizationError(
            "No Attention reshape/transpose patterns were rewritten"
        )

    if not args.no_shape_inference:
        model = shape_inference.infer_shapes(model)

    if not args.no_check:
        onnx.checker.check_model(model)

    save_model(model, args.output, args.external_data_threshold)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Rewritten Attention nodes: {rewritten}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
