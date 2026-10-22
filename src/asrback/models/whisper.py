"""Whisper in plain PyTorch: the encoder-decoder network, which loads Hugging Face
SafeTensors weights, and greedy / sampled decoding over 30-second audio windows.

Concurrent requests are decoded together: callers queue one window at a time and a
single worker thread decodes them in a pool of rows (continuous batching). Between
groups of decoder steps, finished rows return their result and queued windows take
the free rows, so a batch never waits for its longest row.
"""

from __future__ import annotations

import json
import math
import os
import queue
import re
from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import asrfront
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tokenizers import Tokenizer
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from ..transcription import (
    DTYPES,
    AudioSource,
    CapturedGraph,
    Segment,
    Transcription,
    Worker,
    compression_ratio,
    gather_batch,
    load_audio,
    raise_if_device_failed,
    to_device,
)

SAMPLE_RATE = asrfront.whisper.SAMPLE_RATE
WINDOW_SAMPLES = asrfront.whisper.N_SAMPLES
# Trailing windows shorter than this are mostly padding and invite hallucinations.
MIN_TAIL_SAMPLES = SAMPLE_RATE // 10
# With CUDA graphs, decoder steps run this many at a time between EOS checks.
EOS_CHECK_INTERVAL = 4
# Self-attention lengths with their own CUDA graph, besides max_target_positions.
KV_BUCKETS = (64, 128, 256)
# Attention backends for a fixed-shape decoder step, in order of preference. cuDNN
# reads the cross-attention keys about 20% faster than flash attention for one
# query per row; it plans each new shape (tens of ms), and the step has few shapes.
STEP_ATTENTION = [
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]

REQUIRED_FILES = (
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


class MultiHeadAttention(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError(f"d_model ({width}) must be divisible by heads ({heads})")
        self.heads = heads
        self.head_dim = width // heads
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width)
        self.q_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, length, width = x.shape
        return x.view(batch, length, self.heads, width // self.heads).transpose(1, 2)

    def project_key_value(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """``(batch, heads, length, head_dim)`` keys and values for ``x``."""
        return self._split_heads(self.k_proj(x)), self._split_heads(self.v_proj(x))

    def attend(
        self,
        x: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        query = self._split_heads(self.q_proj(x))
        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, is_causal=is_causal
        )
        output = output.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        return self.out_proj(output)

    def forward(self, x: Tensor) -> Tensor:
        """Unmasked self-attention."""
        return self.attend(x, *self.project_key_value(x))


class EncoderLayer(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        width = config["d_model"]
        self.self_attn = MultiHeadAttention(width, config["encoder_attention_heads"])
        self.self_attn_layer_norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, config["encoder_ffn_dim"])
        self.fc2 = nn.Linear(config["encoder_ffn_dim"], width)
        self.final_layer_norm = nn.LayerNorm(width)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.self_attn(self.self_attn_layer_norm(x))
        return x + self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))


class WhisperEncoder(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        width = config["d_model"]
        self.conv1 = nn.Conv1d(config["num_mel_bins"], width, 3, padding=1)
        self.conv2 = nn.Conv1d(width, width, 3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(config["max_source_positions"], width)
        self.layers = nn.ModuleList(
            EncoderLayer(config) for _ in range(config["encoder_layers"])
        )
        self.layer_norm = nn.LayerNorm(width)

    def forward(self, mel: Tensor) -> Tensor:
        x = F.gelu(self.conv1(mel))
        # Contiguous once here: left transposed, every layer norm copies its input
        # and every residual add takes a slower strided kernel.
        x = F.gelu(self.conv2(x)).transpose(1, 2).contiguous()
        if x.shape[1] != self.embed_positions.num_embeddings:
            raise ValueError(
                f"expected {self.embed_positions.num_embeddings * 2} mel frames, "
                f"got {mel.shape[-1]}"
            )
        x = x + self.embed_positions.weight
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)


@dataclass
class DecoderCache:
    """Preallocated keys and values, ``(layers, batch, heads, length, head_dim)``.

    Self-attention entries are written in place at each step's positions, so the
    tensors keep fixed addresses (as CUDA graphs require).
    """

    self_keys: Tensor
    self_values: Tensor
    cross_keys: Tensor
    cross_values: Tensor

    @property
    def batch_size(self) -> int:
        return self.self_keys.shape[1]

    def narrow(self, batch_size: int, length: int | None = None) -> DecoderCache:
        """A view of the first ``batch_size`` rows (and self-attention positions)
        sharing the same storage."""
        return self.rows(0, batch_size, length)

    def rows(self, start: int, stop: int, length: int | None = None) -> DecoderCache:
        """A view of rows ``start:stop`` (and self-attention positions)."""
        return DecoderCache(
            self.self_keys[:, start:stop, :, :length],
            self.self_values[:, start:stop, :, :length],
            self.cross_keys[:, start:stop],
            self.cross_values[:, start:stop],
        )

    def copy_rows(self, source: DecoderCache, rows: Tensor, length: int) -> None:
        """Copy every row of ``source`` (its first ``length`` self-attention
        positions) into rows ``rows`` of this cache."""
        for target, values in (
            (self.self_keys[:, :, :, :length], source.self_keys[:, :, :, :length]),
            (self.self_values[:, :, :, :length], source.self_values[:, :, :, :length]),
            (self.cross_keys, source.cross_keys),
            (self.cross_values, source.cross_values),
        ):
            target.index_copy_(1, rows, values)


class DecoderLayer(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        width = config["d_model"]
        heads = config["decoder_attention_heads"]
        self.self_attn = MultiHeadAttention(width, heads)
        self.self_attn_layer_norm = nn.LayerNorm(width)
        self.encoder_attn = MultiHeadAttention(width, heads)
        self.encoder_attn_layer_norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, config["decoder_ffn_dim"])
        self.fc2 = nn.Linear(config["decoder_ffn_dim"], width)
        self.final_layer_norm = nn.LayerNorm(width)

    def forward(
        self,
        x: Tensor,
        cache: tuple[Tensor, Tensor, Tensor, Tensor],
        positions: Tensor,
        length: int | None,
        mask: Tensor | None,
    ) -> Tensor:
        self_keys, self_values, cross_keys, cross_values = cache

        hidden = self.self_attn_layer_norm(x)
        key, value = self.self_attn.project_key_value(hidden)
        if length is None:
            # Fixed-shape step: one token per row, each at its own position.
            index = positions[:, None, None, None].expand_as(key)
            self_keys.scatter_(2, index, key)
            self_values.scatter_(2, index, value)
        else:
            # Eager: tokens at shared positions; attend to the filled prefix.
            self_keys.index_copy_(2, positions, key)
            self_values.index_copy_(2, positions, value)
            self_keys = self_keys[:, :, :length]
            self_values = self_values[:, :, :length]
        x = x + self.self_attn.attend(
            hidden, self_keys, self_values, mask=mask, is_causal=x.shape[1] > 1
        )

        hidden = self.encoder_attn_layer_norm(x)
        x = x + self.encoder_attn.attend(hidden, cross_keys, cross_values)
        return x + self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))


class WhisperDecoder(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        width = config["d_model"]
        self.embed_tokens = nn.Embedding(config["vocab_size"], width)
        self.embed_positions = nn.Embedding(config["max_target_positions"], width)
        self.layers = nn.ModuleList(
            DecoderLayer(config) for _ in range(config["decoder_layers"])
        )
        self.layer_norm = nn.LayerNorm(width)

    def forward(
        self, tokens: Tensor, positions: Tensor, cache: DecoderCache, length: int | None
    ) -> Tensor:
        mask = None
        if length is None:
            # Fixed-shape step over the whole cache, one position per row: hide
            # positions not yet written.
            x = self.embed_tokens(tokens) + self.embed_positions(positions)[:, None]
            key_positions = torch.arange(cache.self_keys.shape[3], device=x.device)
            mask = (key_positions[None, :] <= positions[:, None])[:, None, None, :]
        else:
            x = self.embed_tokens(tokens) + self.embed_positions(positions)
        step = length is None
        with sdpa_kernel(STEP_ATTENTION, set_priority=True) if step else nullcontext():
            for index, layer in enumerate(self.layers):
                layer_cache = (
                    cache.self_keys[index],
                    cache.self_values[index],
                    cache.cross_keys[index],
                    cache.cross_values[index],
                )
                x = layer(x, layer_cache, positions, length, mask)
        return self.layer_norm(x)


class WhisperModel(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.encoder = WhisperEncoder(config)
        self.decoder = WhisperDecoder(config)


class WhisperForConditionalGeneration(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.model = WhisperModel(config)

    def allocate_cache(self, batch_size: int) -> DecoderCache:
        config = self.config
        weight = self.model.decoder.embed_tokens.weight
        heads = config["decoder_attention_heads"]
        shape = (
            config["decoder_layers"],
            batch_size,
            heads,
            0,
            config["d_model"] // heads,
        )

        def empty(length: int) -> Tensor:
            size = shape[:3] + (length,) + shape[4:]
            return torch.zeros(size, dtype=weight.dtype, device=weight.device)

        target = config["max_target_positions"]
        source = config["max_source_positions"]
        return DecoderCache(empty(target), empty(target), empty(source), empty(source))

    def set_encoder_output(self, cache: DecoderCache, encoder_output: Tensor) -> None:
        """Fill the first ``len(encoder_output)`` rows' cross-attention keys/values."""
        batch_size = encoder_output.shape[0]
        for index, layer in enumerate(self.model.decoder.layers):
            key, value = layer.encoder_attn.project_key_value(encoder_output)
            cache.cross_keys[index, :batch_size] = key
            cache.cross_values[index, :batch_size] = value

    def decode(
        self,
        tokens: Tensor,
        positions: Tensor,
        cache: DecoderCache,
        length: int | None = None,
    ) -> Tensor:
        """Decoder hidden states for ``tokens`` at ``positions``.

        ``length`` is the number of filled cache positions after this call, for
        tokens ``(batch, n)`` at positions ``(n,)`` shared by every row. It is
        ``None`` for a fixed-shape (CUDA graph) step: tokens ``(batch, 1)`` at
        positions ``(batch,)``, one per row, attending over the whole cache.
        """
        return self.model.decoder(tokens, positions, cache, length)

    def logits(self, hidden: Tensor) -> Tensor:
        # As weight @ hidden.T: for F.linear(hidden, weight) with 8-16 rows, cuBLAS
        # picks a kernel up to 3x slower at some widths (384, 768).
        weight = self.model.decoder.embed_tokens.weight
        rows = hidden.flatten(0, -2)
        return (weight @ rows.T).T.unflatten(0, hidden.shape[:-1])


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def required_token_id(tokenizer: Tokenizer, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"tokenizer is missing required token {token!r}")
    return token_id


def language_token_ids(tokenizer_config: dict) -> dict[str, int]:
    languages = {}
    for raw_id, entry in tokenizer_config.get("added_tokens_decoder", {}).items():
        content = entry.get("content", "")
        match = re.fullmatch(r"<\|([a-z]{2,3})\|>", content)
        if match:
            languages[match.group(1)] = int(raw_id)
    if not languages:
        raise ValueError("tokenizer_config.json contains no Whisper language tokens")
    return languages


@dataclass
class _Job:
    """One 30-second window waiting to be decoded."""

    log_mel: np.ndarray
    previous: list[int]
    language: str | None
    task: str
    temperature: float
    max_new_tokens: int | None
    future: Future = field(default_factory=Future)


@dataclass
class _Result:
    language: str
    tokens: list[int]
    avg_logprob: float
    no_speech_prob: float


@dataclass
class _Row:
    """Host-side bookkeeping for an occupied row of the decoding pool."""

    job: _Job
    language: str
    prefix_length: int
    end: int  # last position the row may write
    bound: int  # upper bound of the row's current position


@dataclass
class _DecodeState:
    """Per-row decoding state, updated in place by every step.

    ``position`` holds each row's position of ``tokens`` (its latest token); a
    step writes the row's next token to ``generated`` at ``position + 1``.
    Finished rows, and free rows (marked finished), keep their token and position.
    """

    tokens: Tensor  # (batch, 1)
    position: Tensor  # (batch,)
    end: Tensor  # (batch,), the last position a row may write
    finished: Tensor  # (batch,)
    sum_logprobs: Tensor  # (batch,)
    no_speech_probs: Tensor  # (batch,), set when a row starts
    generated: Tensor  # (batch, max_target_positions)

    @classmethod
    def allocate(
        cls, batch_size: int, max_target: int, device: torch.device
    ) -> _DecodeState:
        return cls(
            torch.zeros((batch_size, 1), dtype=torch.long, device=device),
            torch.zeros(batch_size, dtype=torch.long, device=device),
            torch.zeros(batch_size, dtype=torch.long, device=device),
            torch.ones(batch_size, dtype=torch.bool, device=device),
            torch.zeros(batch_size, device=device),
            torch.zeros(batch_size, device=device),
            torch.zeros((batch_size, max_target), dtype=torch.long, device=device),
        )

    def narrow(self, batch_size: int) -> _DecodeState:
        return _DecodeState(
            self.tokens[:batch_size],
            self.position[:batch_size],
            self.end[:batch_size],
            self.finished[:batch_size],
            self.sum_logprobs[:batch_size],
            self.no_speech_probs[:batch_size],
            self.generated[:batch_size],
        )

    def free(self, rows: Tensor) -> None:
        """Mark ``rows`` free: finished, at position 0."""
        self.finished[rows] = True
        self.position[rows] = 0
        self.tokens[rows] = 0


class Transcriber:
    """Loads a Whisper checkpoint once and transcribes or translates audio.

    ``transcribe`` is thread-safe: windows from concurrent calls are decoded
    together in a pool of ``max_batch_size`` rows; when the pool is empty, the
    worker waits at most ``batch_wait_ms`` for more windows to arrive.
    ``cuda_graphs`` (default: on for CUDA) replays each decoder step from a
    captured graph instead of launching its kernels one by one.
    ``compile_encoder`` compiles the encoder with ``torch.compile`` for every
    batch size up front, which slows startup. With ``condition_on_previous_text``
    off, a window sees only the prompt instead of the text before it, so the
    windows of one recording are decoded together.
    """

    def __init__(
        self,
        model_dir: str | os.PathLike,
        device: str | None = None,
        dtype: str | torch.dtype | None = None,
        max_batch_size: int = 16,
        batch_wait_ms: float = 5.0,
        cuda_graphs: bool | None = None,
        compile_encoder: bool = False,
        condition_on_previous_text: bool = True,
        max_audio_seconds: float | None = None,
    ):
        model_dir = Path(model_dir)
        for filename in REQUIRED_FILES:
            if not (model_dir / filename).is_file():
                raise FileNotFoundError(f"Missing model file: {model_dir / filename}")
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be greater than zero")
        if batch_wait_ms < 0:
            raise ValueError("batch_wait_ms must be non-negative")
        if max_audio_seconds is not None and max_audio_seconds <= 0:
            raise ValueError("max_audio_seconds must be greater than zero")

        self.model_dir = model_dir
        self.config = load_json(model_dir / "config.json")
        tokenizer_config = load_json(model_dir / "tokenizer_config.json")
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if dtype is None:
            dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        elif isinstance(dtype, str):
            dtype = DTYPES[dtype]
        self.dtype = dtype
        if cuda_graphs is None:
            cuda_graphs = self.device.type == "cuda"
        if cuda_graphs and self.device.type != "cuda":
            raise ValueError("cuda_graphs requires a CUDA device")
        self.cuda_graphs = cuda_graphs
        self.max_batch_size = max_batch_size
        self.batch_wait = batch_wait_ms / 1000
        self.max_audio_seconds = max_audio_seconds
        self.condition_on_previous_text = condition_on_previous_text

        model = WhisperForConditionalGeneration(self.config)
        state_dict = load_file(str(model_dir / "model.safetensors"), device="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"weight mismatch: missing={missing}, unexpected={unexpected}"
            )
        self.model = model.eval().to(self.device, self.dtype)

        # English-only (*.en) checkpoints have a smaller vocabulary, start decoding
        # without language/task tokens and cannot translate.
        self.multilingual = self.config["vocab_size"] >= 51865
        tasks = ("transcribe", "translate") if self.multilingual else ("transcribe",)
        if self.multilingual:
            self.languages = language_token_ids(tokenizer_config)
        else:
            self.languages = {"en": required_token_id(self.tokenizer, "<|en|>")}
        self.start_id = required_token_id(self.tokenizer, "<|startoftranscript|>")
        self.prev_id = required_token_id(self.tokenizer, "<|startofprev|>")
        self.no_timestamps_id = required_token_id(self.tokenizer, "<|notimestamps|>")
        self.task_ids = {
            task: required_token_id(self.tokenizer, f"<|{task}|>") for task in tasks
        }
        self.no_speech_id = self.tokenizer.token_to_id(
            "<|nospeech|>"
        ) or self.tokenizer.token_to_id("<|nocaptions|>")

        # Tokens never sampled: configured suppressions and every timestamp token.
        vocab_size = self.config["vocab_size"]
        suppress = torch.zeros(vocab_size, dtype=torch.bool)
        suppress[self.config.get("suppress_tokens", [])] = True
        suppress[self.no_timestamps_id + 1 :] = True
        first_suppress = suppress.clone()
        first_suppress[self.config.get("begin_suppress_tokens", [])] = True
        self._suppress = suppress.to(self.device)
        self._first_suppress = first_suppress.to(self.device)
        self._language_ids = torch.tensor(
            list(self.languages.values()), device=self.device
        )
        self._positions = torch.arange(
            self.config["max_target_positions"], device=self.device
        )

        # Batch sizes with their own CUDA graph; a batch is padded up to one.
        self._buckets = sorted(
            {
                min(2**power, max_batch_size)
                for power in range(max_batch_size.bit_length())
            }
            | {max_batch_size}
        )
        self._kv_buckets = sorted(
            {size for size in KV_BUCKETS if size < self.config["max_target_positions"]}
            | {self.config["max_target_positions"]}
        )
        self._eos = self.config["eos_token_id"]
        self._encoder = self.model.model.encoder
        self.compile_encoder = compile_encoder
        if compile_encoder:
            self._compile_encoder()
        # The pool of rows being decoded, and a staging cache where new windows
        # run their prompt before moving into free rows.
        self._cache = self.model.allocate_cache(max_batch_size)
        self._staging = self.model.allocate_cache(max_batch_size)
        self._state = _DecodeState.allocate(
            max_batch_size, self.config["max_target_positions"], self.device
        )
        self._rows: dict[int, _Row] = {}
        self._graphs: dict[tuple[int, int], CapturedGraph] = {}
        if self.cuda_graphs:
            self._capture_graphs()

        self._queue: queue.SimpleQueue[_Job] = queue.SimpleQueue()
        self._worker = Worker(self._serve, "asrback-decoder")

    @property
    def alive(self) -> bool:
        """Whether the decoding worker still runs."""
        return self._worker.alive

    def transcribe(
        self,
        audio: AudioSource,
        *,
        language: str | None = None,
        task: str = "transcribe",
        prompt: str | None = None,
        temperature: float = 0.0,
        max_new_tokens: int | None = None,
    ) -> Transcription:
        """Decode ``audio`` (a path, encoded file bytes, or a 16 kHz waveform)."""
        if task not in self.task_ids:
            choices = " or ".join(repr(name) for name in self.task_ids)
            raise ValueError(f"task must be {choices} for this model, got {task!r}")
        if language is not None and language not in self.languages:
            choices = ", ".join(sorted(self.languages))
            raise ValueError(
                f"Unsupported language {language!r}. Available codes: {choices}"
            )
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if max_new_tokens is not None and max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")

        waveform = load_audio(audio, self.max_audio_seconds)
        duration = len(waveform) / SAMPLE_RATE

        history = []
        if prompt and prompt.strip():
            history = self.tokenizer.encode(
                " " + prompt.strip(), add_special_tokens=False
            ).ids
        # Like openai-whisper, earlier text fills at most half the context.
        context = self.config["max_target_positions"] // 2 - 1

        def submit(log_mel: np.ndarray, previous: list[int]) -> Future:
            job = _Job(log_mel, previous, language, task, temperature, max_new_tokens)
            self._queue.put(job)
            return job.future

        windows = self._windows(waveform)
        results: list[_Result] = []
        if self.condition_on_previous_text:
            # Windows are decoded in order: each one is conditioned on earlier text.
            for _, _, log_mel in windows:
                result = self._worker.wait(submit(log_mel, history[-context:]))
                language = result.language
                history.extend(result.tokens)
                results.append(result)
        else:
            # Windows share only the prompt and the language, which the first window
            # detects when not given; the rest are queued at once to share batches.
            log_mels = [log_mel for _, _, log_mel in windows]
            if language is None:
                first = submit(log_mels.pop(0), history[-context:])
                results.append(self._worker.wait(first))
                language = results[0].language
            futures = [submit(log_mel, history[-context:]) for log_mel in log_mels]
            results.extend(self._worker.wait(future) for future in futures)

        segments = []
        for (seek, length, _), result in zip(windows, results):
            text = self.tokenizer.decode(result.tokens, skip_special_tokens=True)
            segments.append(
                Segment(
                    id=len(segments),
                    seek=seek // asrfront.whisper.HOP_LENGTH,
                    start=round(seek / SAMPLE_RATE, 3),
                    end=round((seek + length) / SAMPLE_RATE, 3),
                    text=text,
                    tokens=result.tokens,
                    temperature=temperature,
                    avg_logprob=result.avg_logprob,
                    compression_ratio=compression_ratio(text),
                    no_speech_prob=result.no_speech_prob,
                )
            )

        return Transcription(
            task=task,
            language=language,
            duration=round(duration, 3),
            text="".join(segment.text for segment in segments).strip(),
            segments=segments,
        )

    def _windows(self, waveform: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
        """Split into 30-second windows: ``(seek, length, log_mel)`` each."""
        windows = []
        for seek in range(0, max(len(waveform), 1), WINDOW_SAMPLES):
            window = waveform[seek : seek + WINDOW_SAMPLES]
            if windows and len(window) < MIN_TAIL_SAMPLES:
                break
            log_mel = asrfront.whisper.log_mel(
                window, n_mels=self.config["num_mel_bins"]
            )
            windows.append((seek, len(window), log_mel))
        return windows

    @torch.inference_mode()
    def _compile_encoder(self) -> None:
        # Static shapes: batches are padded to a bucket, each compiled here once.
        self._encoder = torch.compile(self.model.model.encoder, dynamic=False)
        frames = 2 * self.config["max_source_positions"]
        for batch_size in self._buckets:
            mel = torch.zeros(
                (batch_size, self.config["num_mel_bins"], frames),
                dtype=self.dtype,
                device=self.device,
            )
            self._encoder(mel)

    def _encode(self, log_mels: list[np.ndarray]) -> Tensor:
        mel = to_device(np.stack(log_mels), self.device).to(self.dtype)
        if not self.compile_encoder:
            return self._encoder(mel)
        batch_size = len(log_mels)
        bucket = next(size for size in self._buckets if size >= batch_size)
        padding = mel.new_zeros((bucket - batch_size,) + mel.shape[1:])
        return self._encoder(torch.cat((mel, padding)))[:batch_size]

    @torch.inference_mode()
    def _capture_graphs(self) -> None:
        if self.device.index is not None:
            torch.cuda.set_device(self.device)
        # One greedy step per (batch size, self-attention length) pair. Warm-up
        # steps advance active rows; keep them inside the shortest length.
        for batch_size in self._buckets:
            state = self._state.narrow(batch_size)
            for length in self._kv_buckets:
                cache = self._cache.narrow(batch_size, length)
                state.finished.zero_()
                state.position.zero_()
                state.end.fill_(self.config["max_target_positions"] - 1)
                self._graphs[batch_size, length] = CapturedGraph(
                    lambda state=state, cache=cache: self._step(state, cache)
                )
        self._state.free(torch.arange(self.max_batch_size, device=self.device))

    def _serve(self) -> None:
        """Worker loop: move queued windows into free rows, run decoder steps,
        and return the results of finished rows."""
        if self.device.type == "cuda" and self.device.index is not None:
            torch.cuda.set_device(self.device)
        pending: deque[_Job] = deque()
        while True:
            if not self._rows and not pending:
                pending.extend(
                    gather_batch(self._queue, self.max_batch_size, self.batch_wait)
                )
            while True:
                try:
                    pending.append(self._queue.get_nowait())
                except queue.Empty:
                    break

            free = self.max_batch_size - len(self._rows)
            if pending and free:
                jobs = [pending.popleft() for _ in range(min(free, len(pending)))]
                self._admit(jobs)
            if not self._rows:
                continue
            try:
                self._advance()
                self._finish()
            except BaseException as exc:
                # A failed step leaves every row's state unknown: fail them all.
                rows = list(self._rows)
                for slot in rows:
                    self._rows.pop(slot).job.future.set_exception(exc)
                raise_if_device_failed(self.device)
                self._state.free(torch.tensor(rows, device=self.device))

    def _sot(self, language: str, task: str) -> list[int]:
        sot = [self.start_id]
        if self.multilingual:
            sot += [self.languages[language], self.task_ids[task]]
        return sot + [self.no_timestamps_id]

    def _layout(self, job: _Job) -> tuple[int, int]:
        """``(prefix_length, max_new_tokens)`` for a job."""
        previous = len(job.previous) + 1 if job.previous else 0
        prefix_length = previous + len(self._sot("en", job.task))
        available = self.config["max_target_positions"] - prefix_length
        return prefix_length, min(job.max_new_tokens or available, available)

    @torch.inference_mode()
    def _admit(self, jobs: list[_Job]) -> None:
        """Encode new windows, run their prompts and pick their first tokens in
        the staging cache, then move them into free rows of the pool.

        Windows with the same prefix length share a prompt pass; after that, rows
        of any prefix length decode together.
        """
        jobs.sort(key=self._layout)
        slots = [slot for slot in range(self.max_batch_size) if slot not in self._rows]
        slots = slots[: len(jobs)]
        try:
            staging = self._staging.narrow(len(jobs))
            encoder_output = self._encode([job.log_mel for job in jobs])
            self.model.set_encoder_output(staging, encoder_output)
            languages = [job.language for job in jobs]
            if None in languages:
                detected = self._detect_languages(staging, len(jobs))
                languages = [
                    given or found for given, found in zip(languages, detected)
                ]

            rows: dict[int, _Row] = {}
            start = 0
            while start < len(jobs):
                layout = self._layout(jobs[start])
                stop = start
                while stop < len(jobs) and self._layout(jobs[stop]) == layout:
                    stop += 1
                rows.update(
                    self._prefill(
                        jobs[start:stop],
                        languages[start:stop],
                        staging.rows(start, stop),
                        slots[start:stop],
                        *layout,
                    )
                )
                start = stop
        except BaseException as exc:
            for job in jobs:
                job.future.set_exception(exc)
            raise_if_device_failed(self.device)
            self._state.free(torch.tensor(slots, device=self.device))
            return
        self._rows.update(rows)

    def _prefill(
        self,
        jobs: list[_Job],
        languages: list[str],
        staging: DecoderCache,
        slots: list[int],
        prefix_length: int,
        limit: int,
    ) -> dict[int, _Row]:
        """Run one prompt pass for ``jobs`` (same prefix length) in ``staging``
        and move them, with their first token, into rows ``slots`` of the pool."""
        prefixes = []
        for job, language in zip(jobs, languages):
            previous = [self.prev_id] + job.previous if job.previous else []
            prefixes.append(previous + self._sot(language, job.task))
        # Copied from pinned memory: a copy from pageable memory (or a transfer to the
        # host) would wait for the decoder steps queued before it.
        tokens = to_device(prefixes, self.device)
        sot_index = prefix_length - len(self._sot("en", jobs[0].task))

        hidden = self.model.decode(
            tokens, self._positions[:prefix_length], staging, length=prefix_length
        )
        logits = self.model.logits(
            torch.stack((hidden[:, sot_index], hidden[:, -1]), dim=1)
        ).float()
        if self.no_speech_id is not None:
            no_speech_probs = logits[:, 0].softmax(dim=-1)[:, self.no_speech_id]
        else:
            no_speech_probs = torch.zeros(len(jobs), device=self.device)
        token, logprob = self._choose(
            logits[:, 1], self._first_suppress, self._temperatures(jobs)
        )

        # The first token sits at ``prefix_length``; a row may write up to ``end``.
        end = prefix_length - 1 + limit
        rows = to_device(slots, self.device)
        self._cache.copy_rows(staging, rows, prefix_length)
        state = self._state
        state.tokens[rows] = token[:, None]
        state.position[rows] = prefix_length
        state.end[rows] = end
        state.finished[rows] = (token == self._eos) | (prefix_length >= end)
        state.sum_logprobs[rows] = logprob
        state.no_speech_probs[rows] = no_speech_probs
        state.generated[rows, prefix_length] = token
        # No transfer to the host here: it would wait for the GPU and leave it idle.
        return {
            slot: _Row(job, language, prefix_length, end, prefix_length)
            for slot, job, language in zip(slots, jobs, languages)
        }

    def _temperatures(self, jobs: list[_Job | None]) -> Tensor | None:
        """Per-row temperatures, or None when every row is greedy."""
        values = [job.temperature if job is not None else 0.0 for job in jobs]
        if not any(value > 0 for value in values):
            return None
        return to_device(values, self.device)

    @torch.inference_mode()
    def _advance(self) -> None:
        """Run up to ``EOS_CHECK_INTERVAL`` decoder steps over the occupied rows."""
        # Rows past the highest occupied one are free (finished) padding.
        bucket = next(size for size in self._buckets if size > max(self._rows))
        rows = self._rows.values()
        count = min(EOS_CHECK_INTERVAL, max(row.end - row.bound for row in rows))
        count = max(count, 1)
        length = max(min(row.bound + count, row.end) for row in rows) + 1
        kv_bucket = next(size for size in self._kv_buckets if size >= length)
        state = self._state.narrow(bucket)
        temperatures = self._temperatures(
            [
                self._rows[slot].job if slot in self._rows else None
                for slot in range(bucket)
            ]
        )
        if self.cuda_graphs and temperatures is None:
            # Graph steps advance the state on their own; the host only checks
            # for finished rows between groups of steps.
            graph = self._graphs[bucket, kv_bucket]
            for _ in range(count):
                graph.replay()
        else:
            cache = self._cache.narrow(bucket, kv_bucket)
            for _ in range(count):
                self._step(state, cache, temperatures)
        for row in rows:
            row.bound = min(row.bound + count, row.end)

    @torch.inference_mode()
    def _finish(self) -> None:
        """Return the results of finished rows and free them."""
        finished = self._state.finished[: max(self._rows) + 1].tolist()
        done = [slot for slot in self._rows if finished[slot]]
        if not done:
            return
        rows = torch.tensor(done, device=self.device)
        positions = self._state.position[rows].tolist()
        sums = self._state.sum_logprobs[rows].tolist()
        no_speech_probs = self._state.no_speech_probs[rows].tolist()
        generated = self._state.generated[rows, : max(positions) + 1].tolist()
        self._state.free(rows)
        for slot, position, sum_logprob, no_speech_prob, tokens in zip(
            done, positions, sums, no_speech_probs, generated
        ):
            row = self._rows.pop(slot)
            tokens = tokens[row.prefix_length : position + 1]
            if tokens and tokens[-1] == self._eos:
                tokens.pop()
            avg_logprob = sum_logprob / (len(tokens) + 1)
            row.job.future.set_result(
                _Result(row.language, tokens, avg_logprob, no_speech_prob)
            )

    def _step(
        self,
        state: _DecodeState,
        cache: DecoderCache,
        temperatures: Tensor | None = None,
    ) -> None:
        """Decode ``state.tokens`` and select the next token for every row."""
        hidden = self.model.decode(state.tokens, state.position, cache)
        logits = self.model.logits(hidden[:, -1]).float()
        token, logprob = self._choose(logits, self._suppress, temperatures)
        # Finished and free rows keep their token and position.
        active = ~state.finished
        position = state.position + active
        token = torch.where(active, token, state.tokens[:, 0])
        state.sum_logprobs.add_(logprob.masked_fill(state.finished, 0.0))
        state.generated.scatter_(1, position[:, None], token[:, None])
        state.finished.logical_or_((token == self._eos) | (position >= state.end))
        state.position.copy_(position)
        state.tokens.copy_(token[:, None])

    def _choose(
        self, logits: Tensor, suppress: Tensor, temperatures: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        """Next token per row (greedy, or sampled where the temperature is
        positive) and its log-probability."""
        logits.masked_fill_(suppress, -math.inf)
        token = logits.argmax(dim=-1)
        if temperatures is not None:
            scaled = logits / temperatures.clamp(min=1e-6)[:, None]
            sampled = torch.multinomial(F.softmax(scaled, dim=-1), 1).squeeze(1)
            token = torch.where(temperatures > 0, sampled, token)
        logprob = F.log_softmax(logits, dim=-1).gather(1, token[:, None]).squeeze(1)
        return token, logprob

    def _detect_languages(self, cache: DecoderCache, batch_size: int) -> list[str]:
        if not self.multilingual:
            return ["en"] * batch_size
        tokens = torch.full((batch_size, 1), self.start_id, device=self.device)
        # Writes cache position 0, which the prefix pass then overwrites.
        hidden = self.model.decode(tokens, self._positions[:1], cache, length=1)
        logits = self.model.logits(hidden[:, -1])[:, self._language_ids]
        names = list(self.languages)
        return [names[index] for index in logits.argmax(dim=-1).tolist()]
