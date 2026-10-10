#!/usr/bin/env python3
"""Collect activation min/max statistics for Conv inputs in an ONNX model.

The default dataset path matches scripts/download_librispeech_test_clean.sh. This
script is intended for exported Whisper encoder models: it loads audio samples,
computes Whisper log-mel inputs, runs an augmented ONNX graph, and records min/max
for every tensor used as the activation input of Conv nodes.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, shape_inference

from evaluate import DEFAULT_DATASET, load_samples

DEFAULT_MODEL = Path("onnx/tiny/encoder_quantized.onnx")
DEFAULT_OUTPUT = Path("onnx/tiny/calibration_minmax.json")
DEFAULT_QUANTIZED_MODEL = Path("onnx/tiny/encoder_activation_quantized.onnx")
DEFAULT_INPUT_NAME = "mel"


@dataclass
class TensorStats:
    min_value: float | None = None
    max_value: float | None = None
    samples: int = 0

    def update(self, tensor: np.ndarray) -> None:
        if tensor.size == 0:
            return
        tensor_min = float(np.min(tensor))
        tensor_max = float(np.max(tensor))
        self.min_value = (
            tensor_min if self.min_value is None else min(self.min_value, tensor_min)
        )
        self.max_value = (
            tensor_max if self.max_value is None else max(self.max_value, tensor_max)
        )
        self.samples += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate ONNX Conv activation inputs with dataset min/max statistics."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Input ONNX model. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Path to LibriSpeech test-clean. Default: {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON statistics file. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--quantized-model",
        type=Path,
        default=DEFAULT_QUANTIZED_MODEL,
        help=f"Output ONNX model with quantized Conv inputs. Default: {DEFAULT_QUANTIZED_MODEL}",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=8,
        help="Signed activation quantization bit width. Default: 8",
    )
    parser.add_argument(
        "--input-name",
        default=DEFAULT_INPUT_NAME,
        help=f"Model input tensor name. Default: {DEFAULT_INPUT_NAME}",
    )
    parser.add_argument(
        "--provider",
        default="cuda",
        choices=("cpu", "cuda", "tensorrt"),
        help="ONNX Runtime execution provider. Default: cuda",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Calibrate only the first N utterances for a quick run.",
    )
    return parser.parse_args()


def import_whisper():
    try:
        import whisper

        return whisper
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The 'whisper' package is not installed in this Python environment.\n"
            "Install it with:\n"
            "  python -m pip install -U openai-whisper"
        ) from exc


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


def collect_target_tensors(model: onnx.ModelProto) -> list[str]:
    initializers = {initializer.name for initializer in model.graph.initializer}
    targets: list[str] = []
    seen: set[str] = set()
    for node in model.graph.node:
        if node.op_type != "Conv" or not node.input:
            continue
        tensor_name = node.input[0]
        if not tensor_name or tensor_name in initializers or tensor_name in seen:
            continue
        targets.append(tensor_name)
        seen.add(tensor_name)
    return targets


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


def make_unique_name(existing: set[str], base: str) -> str:
    candidate = base
    suffix = 0
    while candidate in existing:
        suffix += 1
        candidate = f"{base}_{suffix}"
    existing.add(candidate)
    return candidate


def value_info_map(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    values: dict[str, onnx.ValueInfoProto] = {}
    for value in model.graph.input:
        values[value.name] = value
    for value in model.graph.value_info:
        values[value.name] = value
    for value in model.graph.output:
        values[value.name] = value
    return values


def make_augmented_model(
    model: onnx.ModelProto, target_tensors: list[str]
) -> onnx.ModelProto:
    try:
        inferred = shape_inference.infer_shapes(model)
    except Exception:
        inferred = model

    augmented = copy.deepcopy(model)
    values = value_info_map(inferred)
    existing_outputs = {output.name for output in augmented.graph.output}

    for tensor_name in target_tensors:
        if tensor_name in existing_outputs:
            continue
        if tensor_name in values:
            augmented.graph.output.append(copy.deepcopy(values[tensor_name]))
        else:
            augmented.graph.output.append(
                helper.make_tensor_value_info(tensor_name, TensorProto.FLOAT, None)
            )
        existing_outputs.add(tensor_name)

    return augmented


def input_n_mels(session, input_name: str) -> int:
    for model_input in session.get_inputs():
        if model_input.name != input_name:
            continue
        shape = model_input.shape
        if len(shape) >= 2 and isinstance(shape[1], int):
            return int(shape[1])
    return 80


def make_mel_input(whisper, audio_path: Path, n_mels: int) -> np.ndarray:
    audio = whisper.load_audio(str(audio_path))
    audio = whisper.pad_or_trim(audio)
    try:
        mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels)
    except TypeError:
        mel = whisper.log_mel_spectrogram(audio)
    mel_np = mel.detach().cpu().numpy() if hasattr(mel, "detach") else np.asarray(mel)
    return mel_np.astype(np.float32, copy=False)[None, :, :]


def activation_quant_params(
    stats: TensorStats, bits: int
) -> tuple[float, float, float, float]:
    if bits < 2:
        raise ValueError("--bits must be at least 2")
    if stats.min_value is None or stats.max_value is None or stats.samples == 0:
        raise ValueError("Cannot quantize tensor without calibration statistics")

    qmax = float((1 << (bits - 1)) - 1)
    max_abs = max(abs(stats.min_value), abs(stats.max_value))
    scale = max_abs / qmax if max_abs != 0.0 else 1.0
    inv_scale = 1.0 / scale
    return inv_scale, scale, -qmax, qmax


def quantize_model_inputs(
    model: onnx.ModelProto,
    target_tensors: list[str],
    stats: dict[str, TensorStats],
    bits: int,
) -> int:
    graph = model.graph
    existing_names = collect_names(graph)
    targets = set(target_tensors)
    quantized_outputs: dict[str, str] = {}
    dequant_scales: dict[str, str] = {}
    new_nodes: list[onnx.NodeProto] = []
    rewritten = 0

    for node in graph.node:
        if (
            node.op_type == "Conv"
            and node.input
            and node.output
            and node.input[0] in targets
            and node.input[0] in stats
        ):
            original_input = node.input[0]
            quantized_input = quantized_outputs.get(original_input)
            dequant_scale_name = dequant_scales.get(original_input)
            if quantized_input is None or dequant_scale_name is None:
                inv_scale, scale, clip_min, clip_max = activation_quant_params(
                    stats[original_input], bits
                )
                quant_scale_name = make_unique_name(
                    existing_names, f"{original_input}_input_quant_scale"
                )
                dequant_scale_name = make_unique_name(
                    existing_names, f"{original_input}_output_dequant_scale"
                )
                clip_min_name = make_unique_name(
                    existing_names, f"{original_input}_input_quant_min"
                )
                clip_max_name = make_unique_name(
                    existing_names, f"{original_input}_input_quant_max"
                )
                scaled_output = make_unique_name(
                    existing_names, f"{original_input}_input_scaled"
                )
                rounded_output = make_unique_name(
                    existing_names, f"{original_input}_input_rounded"
                )
                quantized_input = make_unique_name(
                    existing_names, f"{original_input}_input_quantized"
                )

                graph.initializer.append(
                    helper.make_tensor(
                        quant_scale_name, TensorProto.FLOAT, dims=(), vals=[inv_scale]
                    )
                )
                graph.initializer.append(
                    helper.make_tensor(
                        dequant_scale_name, TensorProto.FLOAT, dims=(), vals=[scale]
                    )
                )
                graph.initializer.append(
                    helper.make_tensor(
                        clip_min_name, TensorProto.FLOAT, dims=(), vals=[clip_min]
                    )
                )
                graph.initializer.append(
                    helper.make_tensor(
                        clip_max_name, TensorProto.FLOAT, dims=(), vals=[clip_max]
                    )
                )
                new_nodes.append(
                    helper.make_node(
                        "Mul",
                        inputs=[original_input, quant_scale_name],
                        outputs=[scaled_output],
                        name=make_unique_name(
                            existing_names, f"{original_input}_input_quant_mul"
                        ),
                    )
                )
                new_nodes.append(
                    helper.make_node(
                        "Round",
                        inputs=[scaled_output],
                        outputs=[rounded_output],
                        name=make_unique_name(
                            existing_names, f"{original_input}_input_quant_round"
                        ),
                    )
                )
                new_nodes.append(
                    helper.make_node(
                        "Clip",
                        inputs=[rounded_output, clip_min_name, clip_max_name],
                        outputs=[quantized_input],
                        name=make_unique_name(
                            existing_names, f"{original_input}_input_quant_clip"
                        ),
                    )
                )
                quantized_outputs[original_input] = quantized_input
                dequant_scales[original_input] = dequant_scale_name

            original_output = node.output[0]
            quantized_output = make_unique_name(
                existing_names, f"{original_output}_input_quantized_output"
            )
            node.input[0] = quantized_input
            node.output[0] = quantized_output
            new_nodes.append(node)
            new_nodes.append(
                helper.make_node(
                    "Mul",
                    inputs=[quantized_output, dequant_scale_name],
                    outputs=[original_output],
                    name=make_unique_name(
                        existing_names, f"{original_output}_output_dequant_mul"
                    ),
                )
            )
            rewritten += 1
            continue

        new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)
    return rewritten


def write_augmented_model(model: onnx.ModelProto) -> tempfile.NamedTemporaryFile:
    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    tmp.close()
    onnx.save_model(model, tmp.name, save_as_external_data=False)
    return tmp


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than zero")
    if not args.model.is_file():
        raise FileNotFoundError(f"ONNX model not found: {args.model}")
    if args.bits < 2:
        raise ValueError("--bits must be at least 2")

    ort = import_onnxruntime()
    whisper = import_whisper()
    providers = select_providers(ort, args.provider)

    samples = load_samples(args.dataset)
    if args.limit is not None:
        samples = samples[: args.limit]

    model = onnx.load(str(args.model))
    target_tensors = collect_target_tensors(model)
    if not target_tensors:
        raise RuntimeError("No Conv activation input tensors found")

    augmented = make_augmented_model(model, target_tensors)
    tmp_model = write_augmented_model(augmented)

    try:
        session = ort.InferenceSession(tmp_model.name, providers=providers)
        output_names = {output.name for output in session.get_outputs()}
        requested_outputs = [name for name in target_tensors if name in output_names]
        if not requested_outputs:
            raise RuntimeError(
                "None of the target tensors are available as ONNX outputs"
            )

        model_inputs = {model_input.name for model_input in session.get_inputs()}
        if args.input_name not in model_inputs:
            raise ValueError(
                f"Input {args.input_name!r} not found in model. Available inputs: {sorted(model_inputs)}"
            )

        n_mels = input_n_mels(session, args.input_name)
        stats = {name: TensorStats() for name in requested_outputs}

        print(f"Model: {args.model}")
        print(f"Dataset: {args.dataset}")
        print(f"Samples: {len(samples)}")
        print(f"Provider: {providers}")
        print(f"Target tensors: {len(requested_outputs)}")
        sys.stdout.flush()

        for index, sample in enumerate(samples, start=1):
            input_tensor = make_mel_input(whisper, sample.audio_path, n_mels)
            outputs = session.run(requested_outputs, {args.input_name: input_tensor})
            for name, output in zip(requested_outputs, outputs):
                stats[name].update(output)
            if index == 1 or index % 10 == 0 or index == len(samples):
                print(f"Calibrated {index}/{len(samples)} samples")
                sys.stdout.flush()

    finally:
        Path(tmp_model.name).unlink(missing_ok=True)

    rows = [
        {
            "tensor": name,
            "min": stats[name].min_value,
            "max": stats[name].max_value,
            "samples": stats[name].samples,
        }
        for name in requested_outputs
    ]

    quantized_model = copy.deepcopy(model)
    quantized_inputs = quantize_model_inputs(
        quantized_model, requested_outputs, stats, args.bits
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(rows, output_file, indent=2)
        output_file.write("\n")

    args.quantized_model.parent.mkdir(parents=True, exist_ok=True)
    quantized_model = shape_inference.infer_shapes(quantized_model)
    onnx.save_model(
        quantized_model, str(args.quantized_model), save_as_external_data=False
    )

    print(f"Output: {args.output}")
    print(f"Quantized model: {args.quantized_model}")
    print(f"Quantized Conv input uses: {quantized_inputs}")
    for row in rows:
        print(
            f"{row['tensor']}: min={row['min']} max={row['max']} samples={row['samples']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
