"""Qwen3-ASR (0.6B / 1.7B) in plain PyTorch: an audio encoder whose output fills the
``<|audio_pad|>`` positions of a chat prompt, and a Qwen3 language model that writes
``language <Name><asr_text><transcript>``.

Windows of up to 30 seconds are independent, so every window of a request is queued
at once. A single worker thread decodes them in a pool of rows (continuous batching,
as for Whisper): new windows run their prompts packed back to back, then join the
rows already decoding; finished rows return their result between groups of steps.
"""

from __future__ import annotations

import json
import math
import os
import queue
import re
import threading
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import asrfront
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers
from torch import Tensor, nn
from torch.nn.attention.varlen import varlen_attn

from ..transcription import (
    DTYPES,
    AudioSource,
    CapturedGraph,
    Segment,
    Transcription,
    compression_ratio,
    gather_batch,
    to_device,
)

SAMPLE_RATE = asrfront.qwen3_asr.SAMPLE_RATE
HOP_LENGTH = asrfront.qwen3_asr.HOP_LENGTH
WINDOW_SAMPLES = 30 * SAMPLE_RATE
# Trailing windows shorter than this are mostly padding and invite hallucinations.
MIN_TAIL_SAMPLES = SAMPLE_RATE // 10
# Shorter windows are zero-padded to this length, as qwen-asr does.
MIN_WINDOW_SAMPLES = SAMPLE_RATE // 2
# With CUDA graphs, decoder steps run this many at a time between EOS checks.
EOS_CHECK_INTERVAL = 4
# Positions per row of the KV cache: prompt (about 400 tokens for 30 s of audio)
# plus generated tokens.
CACHE_LENGTH = 1024
# Attention lengths with their own CUDA graph, besides CACHE_LENGTH (without
# varlen_attn, which reads only the keys each row has).
KV_BUCKETS = (128, 256, 512, 768)
MAX_NEW_TOKENS = 512
# Parts each row's keys are split into for a decoder step's flash attention (one
# query per row); the kernel's own choice was about 20% slower at 32 and 64 rows.
DECODE_KV_SPLITS = 4
# 100-frame chunks per convolution call. The checkpoint's ``conv_chunksize`` (500)
# takes about 3 GB of activations at once, which crowds out larger batches; smaller
# calls run as fast.
CONV_CHUNKS = 64
# The request's ``prompt`` (the system message) keeps at most its last this many tokens.
MAX_CONTEXT_TOKENS = 224
# Output repeating a character or a short pattern more often than this is collapsed.
REPETITION_THRESHOLD = 20

REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)
# Qwen2's pre-tokenizer split.
PRETOKENIZE_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
# Request language codes for the languages in config.json's ``support_languages``.
LANGUAGE_CODES = {
    "Arabic": "ar",
    "Cantonese": "yue",
    "Chinese": "zh",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "English": "en",
    "Filipino": "fil",
    "Finnish": "fi",
    "French": "fr",
    "German": "de",
    "Greek": "el",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Indonesian": "id",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Macedonian": "mk",
    "Malay": "ms",
    "Persian": "fa",
    "Polish": "pl",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Russian": "ru",
    "Spanish": "es",
    "Swedish": "sv",
    "Thai": "th",
    "Turkish": "tr",
    "Vietnamese": "vi",
}


@dataclass
class Segments:
    """Boundaries of the sequences laid out back to back in a packed tensor.

    With ``varlen`` off, ``gather`` ``(sequences, longest)`` indexes each
    sequence's tokens, padded with its last one, and ``valid`` marks the real ones.
    """

    starts: Tensor  # cumulative lengths from 0, int32
    longest: int
    varlen: bool  # attend with varlen_attn (flash attention)
    gather: Tensor | None = None
    valid: Tensor | None = None

    @classmethod
    def from_lengths(
        cls, lengths: list[int], device: torch.device, varlen: bool
    ) -> Segments:
        sizes = to_device(lengths, device)
        starts = F.pad(sizes.cumsum(0), (1, 0)).to(torch.int32)
        segments = cls(starts, max(lengths), varlen)
        if not varlen:
            offsets = torch.arange(segments.longest, device=device)
            segments.valid = offsets < sizes[:, None]
            last = (starts[1:] - 1)[:, None]
            segments.gather = torch.minimum(starts[:-1, None] + offsets, last)
        return segments


def attend(q: Tensor, k: Tensor, v: Tensor, segments: Segments, causal: bool) -> Tensor:
    """Attention within each segment of packed ``(tokens, heads, head_dim)`` tensors;
    ``k`` and ``v`` may have fewer heads than ``q`` (grouped-query attention)."""
    if segments.varlen:
        return varlen_attn(
            q,
            k,
            v,
            segments.starts,
            segments.starts,
            segments.longest,
            segments.longest,
            window_size=(-1, 0) if causal else (-1, -1),
            enable_gqa=k.shape[1] != q.shape[1],
        )
    # One padded batch: each sequence's keys stop at its length.
    mask = segments.valid[:, None, None, :]
    if causal:
        mask = (
            mask
            & torch.ones(
                (segments.longest,) * 2, dtype=torch.bool, device=q.device
            ).tril()
        )
    output = F.scaled_dot_product_attention(
        *(t[segments.gather].transpose(1, 2) for t in (q, k, v)),
        attn_mask=mask,
        enable_gqa=True,
    )
    return output.transpose(1, 2)[segments.valid]


def varlen_supported(device: torch.device, dtype: torch.dtype) -> bool:
    """Whether ``varlen_attn`` runs causal grouped-query attention here."""
    if device.type != "cuda" or dtype != torch.bfloat16:
        return False
    q = torch.zeros((2, 2, 64), dtype=dtype, device=device)
    starts = torch.tensor([0, 2], dtype=torch.int32, device=device)
    try:
        varlen_attn(
            q,
            q[:, :1],
            q[:, :1],
            starts,
            starts,
            2,
            2,
            window_size=(-1, 0),
            enable_gqa=True,
        )
    except RuntimeError:
        return False
    return True


class AudioAttention(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)

    def forward(self, x: Tensor, segments: Segments) -> Tensor:
        q, k, v = (
            projection(x).view(len(x), self.heads, -1)
            for projection in (self.q_proj, self.k_proj, self.v_proj)
        )
        return self.out_proj(attend(q, k, v, segments, causal=False).flatten(1))


class AudioEncoderLayer(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        width = config["d_model"]
        self.self_attn_layer_norm = nn.LayerNorm(width)
        self.self_attn = AudioAttention(width, config["encoder_attention_heads"])
        self.final_layer_norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, config["encoder_ffn_dim"])
        self.fc2 = nn.Linear(config["encoder_ffn_dim"], width)

    def forward(self, x: Tensor, segments: Segments) -> Tensor:
        x = x + self.self_attn(self.self_attn_layer_norm(x), segments)
        return x + self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))


def sinusoids(length: int, channels: int) -> Tensor:
    increment = math.log(10000) / (channels // 2 - 1)
    # An explicit device, so the table is real even when the model is built on meta.
    inverse = torch.exp(-increment * torch.arange(channels // 2, device="cpu").float())
    scaled = torch.arange(length, device="cpu")[:, None] * inverse[None, :]
    return torch.cat((torch.sin(scaled), torch.cos(scaled)), dim=1)


class AudioEncoder(nn.Module):
    """Three stride-2 convolutions per 100-frame chunk (13 tokens each), then
    Transformer layers attending within windows of ``n_window_infer`` frames."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        width = config["d_model"]
        channels = config["downsample_hidden_size"]
        self.conv2d1 = nn.Conv2d(1, channels, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(channels, channels, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(channels, channels, 3, 2, padding=1)
        bins = config["num_mel_bins"]
        for _ in range(3):
            bins = (bins + 1) // 2
        self.conv_out = nn.Linear(channels * bins, width, bias=False)
        self.register_buffer(
            "positional_embedding",
            sinusoids(config["max_source_positions"], width),
            persistent=False,
        )
        self.layers = nn.ModuleList(
            AudioEncoderLayer(config) for _ in range(config["encoder_layers"])
        )
        self.ln_post = nn.LayerNorm(width)
        self.proj1 = nn.Linear(width, width)
        self.proj2 = nn.Linear(width, config["output_dim"])
        self.chunk_frames = 2 * config["n_window"]
        self.window_chunks = config["n_window_infer"] // self.chunk_frames

    def forward(
        self,
        chunks: Tensor,
        chunk_tokens: np.ndarray,
        window_tokens: list[int],
        varlen: bool,
    ) -> Tensor:
        """Packed audio embeddings ``(sum(window_tokens), output_dim)``.

        ``chunks`` is ``(chunks, n_mels, chunk_frames)``, every window's chunks in
        order, a window's last chunk zero-padded; ``chunk_tokens`` counts each
        chunk's valid tokens and ``window_tokens`` each window's. Index tensors
        are built on the host, so nothing here waits for the GPU.
        """
        outputs = []
        for part in chunks[:, None].split(CONV_CHUNKS):
            x = F.gelu(self.conv2d1(part))
            x = F.gelu(self.conv2d2(x))
            outputs.append(F.gelu(self.conv2d3(x)))
        x = torch.cat(outputs)
        count, channels, bins, length = x.shape
        x = self.conv_out(x.permute(0, 3, 1, 2).reshape(count, length, channels * bins))
        # Positions restart in every chunk.
        x = x + self.positional_embedding[:length].to(x.dtype)
        valid = np.flatnonzero(np.arange(length) < chunk_tokens[:, None])
        x = x.flatten(0, 1)[to_device(valid, x.device)]

        # Attention windows of ``window_chunks`` chunks within each audio window.
        span = self.window_chunks * length
        lengths = []
        for tokens in window_tokens:
            lengths += [span] * (tokens // span) + (
                [tokens % span] if tokens % span else []
            )
        segments = Segments.from_lengths(lengths, x.device, varlen)
        for layer in self.layers:
            x = layer(x, segments)
        return self.proj2(F.gelu(self.proj1(self.ln_post(x))))


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, self.weight.shape, self.weight, self.eps)


def rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotary embedding of ``(..., heads, head_dim)`` with ``cos``/``sin`` ``(..., head_dim)``."""
    first, second = x.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return x * cos[..., None, :] + rotated * sin[..., None, :]


@dataclass
class KVCache:
    """Preallocated keys and values, ``(layers, batch, length, kv_heads, head_dim)``.

    Entries are written in place, so the tensors keep fixed addresses (as CUDA
    graphs require).
    """

    keys: Tensor
    values: Tensor

    def narrow(self, batch_size: int, length: int) -> KVCache:
        return KVCache(
            self.keys[:, :batch_size, :length], self.values[:, :batch_size, :length]
        )


class TextAttention(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        width = config["hidden_size"]
        self.head_dim = config["head_dim"]
        self.heads = config["num_attention_heads"]
        self.kv_heads = config["num_key_value_heads"]
        self.sizes = [
            size * self.head_dim for size in (self.heads, self.kv_heads, self.kv_heads)
        ]
        # The checkpoint's q_proj, k_proj and v_proj as one matrix.
        self.qkv_proj = nn.Linear(width, sum(self.sizes), bias=False)
        self.o_proj = nn.Linear(self.heads * self.head_dim, width, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config["rms_norm_eps"])
        self.k_norm = RMSNorm(self.head_dim, config["rms_norm_eps"])

    def _project(self, x: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, ...]:
        """``(tokens, heads, head_dim)`` queries, keys and values for ``(tokens, width)``."""
        shape = (len(x), -1, self.head_dim)
        q, k, v = self.qkv_proj(x).split(self.sizes, dim=-1)
        q = rotate(self.q_norm(q.reshape(shape)), cos, sin)
        k = rotate(self.k_norm(k.reshape(shape)), cos, sin)
        return q, k, v.reshape(shape)

    def prefill(
        self,
        x: Tensor,
        keys: Tensor,
        values: Tensor,
        rows: Tensor,
        positions: Tensor,
        segments: Segments,
        cos: Tensor,
        sin: Tensor,
    ) -> Tensor:
        """Causal attention within each packed prompt; every token's key and value
        go to its row and position of the cache."""
        q, k, v = self._project(x, cos, sin)
        keys[rows, positions] = k
        values[rows, positions] = v
        return self.o_proj(attend(q, k, v, segments, causal=True).flatten(1))

    def step(
        self,
        x: Tensor,
        keys: Tensor,
        values: Tensor,
        positions: Tensor,
        mask: Tensor | tuple[Tensor, Tensor, Tensor],
        cos: Tensor,
        sin: Tensor,
    ) -> Tensor:
        """One token per row at its own position, attending over the cache.

        ``mask`` is ``(batch, 1, 1, length)``, or for ``varlen_attn`` over whole
        cache rows: the query starts, each row's start in the flattened cache,
        and each row's key count.
        """
        q, k, v = self._project(x, cos, sin)
        rows = torch.arange(len(x), device=x.device)
        keys[rows, positions] = k
        values[rows, positions] = v
        if isinstance(mask, tuple):
            query_starts, key_starts, lengths = mask
            output = varlen_attn(
                q,
                keys.flatten(0, 1),
                values.flatten(0, 1),
                query_starts,
                key_starts,
                1,
                keys.shape[1],
                enable_gqa=True,
                seqused_k=lengths,
                num_splits=DECODE_KV_SPLITS,
            )
            return self.o_proj(output.flatten(1))
        # The query heads sharing a key/value head attend as one group of queries.
        q = q.view(len(x), self.kv_heads, -1, self.head_dim)
        output = F.scaled_dot_product_attention(
            q, keys.transpose(1, 2), values.transpose(1, 2), attn_mask=mask
        )
        return self.o_proj(output.reshape(len(x), -1))


class TextMLP(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        width, hidden = config["hidden_size"], config["intermediate_size"]
        # The checkpoint's gate_proj and up_proj as one matrix.
        self.gate_up_proj = nn.Linear(width, 2 * hidden, bias=False)
        self.down_proj = nn.Linear(hidden, width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class DecoderLayer(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.self_attn = TextAttention(config)
        self.post_attention_layernorm = RMSNorm(
            config["hidden_size"], config["rms_norm_eps"]
        )
        self.mlp = TextMLP(config)

    def forward(self, x: Tensor, attention: Callable[[TextAttention, Tensor], Tensor]):
        x = x + attention(self.self_attn, self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class TextModel(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList(
            DecoderLayer(config) for _ in range(config["num_hidden_layers"])
        )
        self.norm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])

    def prefill(
        self,
        x: Tensor,
        cache: KVCache,
        rows: Tensor,
        positions: Tensor,
        segments: Segments,
        cos: Tensor,
        sin: Tensor,
    ) -> Tensor:
        """Hidden states of packed prompts ``x`` (tokens, width); token ``i`` is at
        ``positions[i]`` of cache row ``rows[i]``."""
        for index, layer in enumerate(self.layers):
            keys, values = cache.keys[index], cache.values[index]
            x = layer(
                x,
                lambda attention, h: attention.prefill(
                    h, keys, values, rows, positions, segments, cos, sin
                ),
            )
        return self.norm(x)

    def step(
        self,
        tokens: Tensor,
        positions: Tensor,
        cache: KVCache,
        cos: Tensor,
        sin: Tensor,
        varlen: bool,
    ) -> Tensor:
        """Hidden states for one token per row (``tokens``, ``positions``: (batch,))
        over a fixed-shape cache view; ``cos``/``sin`` are the full tables. With
        ``varlen``, the view holds whole cache rows and each row reads only its
        keys up to its position."""
        x = self.embed_tokens(tokens)
        cos, sin = cos[positions], sin[positions]
        length = cache.keys.shape[2]
        if varlen:
            starts = torch.arange(len(x) + 1, dtype=torch.int32, device=x.device)
            mask = (starts, starts * length, (positions + 1).to(torch.int32))
        else:
            key_positions = torch.arange(length, device=x.device)
            mask = (key_positions[None, :] <= positions[:, None])[:, None, None, :]
        for index, layer in enumerate(self.layers):
            keys, values = cache.keys[index], cache.values[index]
            x = layer(
                x,
                lambda attention, h: attention.step(
                    h, keys, values, positions, mask, cos, sin
                ),
            )
        return self.norm(x)


class Thinker(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        text = config["text_config"]
        self.audio_tower = AudioEncoder(config["audio_config"])
        self.model = TextModel(text)
        self.lm_head = nn.Linear(text["hidden_size"], text["vocab_size"], bias=False)


class Qwen3ASRForConditionalGeneration(nn.Module):
    """Module names match the Hugging Face checkpoint (``thinker.*``), except for
    the projections :func:`fuse_projections` joins."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.thinker = Thinker(config["thinker_config"])


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


# Checkpoint projections loaded as one matrix: fewer, larger matrix multiplications.
FUSED_PROJECTIONS = {
    "self_attn.qkv_proj": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    "mlp.gate_up_proj": ("mlp.gate_proj", "mlp.up_proj"),
}


def fuse_projections(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Join the text model's projection weights as :data:`FUSED_PROJECTIONS` lists."""
    for name in list(state_dict):
        for fused, parts in FUSED_PROJECTIONS.items():
            if name.startswith("thinker.model.") and name.endswith(
                f".{parts[0]}.weight"
            ):
                prefix = name[: -len(f"{parts[0]}.weight")]
                state_dict[f"{prefix}{fused}.weight"] = torch.cat(
                    [state_dict.pop(f"{prefix}{part}.weight") for part in parts]
                )
    return state_dict


def load_tokenizer(model_dir: Path) -> Tokenizer:
    """The byte-level BPE tokenizer of ``vocab.json`` and ``merges.txt``, with the
    added tokens of ``tokenizer_config.json`` (Qwen2TokenizerFast's layout)."""
    tokenizer = Tokenizer(
        models.BPE.from_file(
            str(model_dir / "vocab.json"), str(model_dir / "merges.txt")
        )
    )
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(PRETOKENIZE_PATTERN), "isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    added = load_json(model_dir / "tokenizer_config.json")["added_tokens_decoder"]
    for _, entry in sorted(added.items(), key=lambda item: int(item[0])):
        token = entry["content"]
        if entry.get("special"):
            tokenizer.add_special_tokens([token])
        else:
            tokenizer.add_tokens([token])
        if tokenizer.token_to_id(token) is None:
            raise ValueError(f"tokenizer is missing added token {token!r}")
    return tokenizer


def fix_repetitions(text: str, threshold: int = REPETITION_THRESHOLD) -> str:
    """qwen-asr's ``detect_and_fix_repetitions``: a character repeated more than
    ``threshold`` times becomes one; then, scanning on, each pattern of up to 20
    characters repeated at least ``threshold`` times in a row becomes one copy."""
    text = re.sub(r"(.)\1{%d,}" % threshold, r"\1", text, flags=re.DOTALL)
    result = []
    i = 0
    while i <= len(text) - 2 * threshold:
        end = None
        for size in range(1, 21):
            if i + size * threshold > len(text):
                break
            pattern = text[i : i + size]
            if text[i : i + size * threshold] == pattern * threshold:
                end = i + size * threshold
                while text[end : end + size] == pattern:
                    end += size
                break
        if end is None:
            result.append(text[i])
            i += 1
        else:
            result.append(pattern)
            i = end
    result.append(text[i:])
    return "".join(result)


@dataclass
class _Job:
    """One window of up to 30 seconds waiting to be decoded."""

    features: np.ndarray  # (n_mels, frames)
    context: list[int]
    language: str | None
    temperature: float
    max_new_tokens: int | None
    future: Future = field(default_factory=Future)


@dataclass
class _Result:
    language: str | None
    tokens: list[int]
    text: str
    avg_logprob: float
    no_speech_prob: float


@dataclass
class _Row:
    """Host-side bookkeeping for an occupied row of the decoding pool."""

    job: _Job
    prompt_length: int
    end: int  # last position the row may write
    bound: int  # upper bound of the row's current position


@dataclass
class _DecodeState:
    """Per-row decoding state, updated in place by every step.

    ``position`` holds each row's position of ``tokens`` (its latest token); a
    step writes the row's next token to ``generated`` at ``position + 1``.
    Finished rows, and free rows (marked finished), keep their token and position.
    """

    tokens: Tensor  # (batch,)
    position: Tensor  # (batch,)
    end: Tensor  # (batch,), the last position a row may write
    finished: Tensor  # (batch,)
    sum_logprobs: Tensor  # (batch,)
    generated: Tensor  # (batch, CACHE_LENGTH)

    @classmethod
    def allocate(cls, batch_size: int, device: torch.device) -> _DecodeState:
        return cls(
            torch.zeros(batch_size, dtype=torch.long, device=device),
            torch.zeros(batch_size, dtype=torch.long, device=device),
            torch.zeros(batch_size, dtype=torch.long, device=device),
            torch.ones(batch_size, dtype=torch.bool, device=device),
            torch.zeros(batch_size, device=device),
            torch.zeros((batch_size, CACHE_LENGTH), dtype=torch.long, device=device),
        )

    def narrow(self, batch_size: int) -> _DecodeState:
        return _DecodeState(
            self.tokens[:batch_size],
            self.position[:batch_size],
            self.end[:batch_size],
            self.finished[:batch_size],
            self.sum_logprobs[:batch_size],
            self.generated[:batch_size],
        )

    def free(self, rows: Tensor) -> None:
        """Mark ``rows`` free: finished, at position 0."""
        self.finished[rows] = True
        self.position[rows] = 0
        self.tokens[rows] = 0


class Qwen3ASRTranscriber:
    """Loads a Qwen3-ASR checkpoint once and transcribes audio.

    Same interface as :class:`asrback.Transcriber`: ``transcribe`` is thread-safe
    and windows from concurrent calls are decoded together in a pool of
    ``max_batch_size`` rows. Without a language, the model names the language it
    hears before the transcript; with one, the prompt names it. ``prompt`` becomes
    the system message (context such as names or terms). Translation is not
    supported. ``cuda_graphs`` (default: on for CUDA) replays each decoder step
    from a captured graph. ``max_batch_size`` defaults to 64 in bfloat16 and 32 in
    float32, whose KV cache takes twice the memory.
    """

    def __init__(
        self,
        model_dir: str | os.PathLike,
        device: str | None = None,
        dtype: str | torch.dtype | None = None,
        max_batch_size: int | None = None,
        batch_wait_ms: float = 5.0,
        cuda_graphs: bool | None = None,
    ):
        model_dir = Path(model_dir)
        for filename in REQUIRED_FILES:
            if not (model_dir / filename).is_file():
                raise FileNotFoundError(f"Missing model file: {model_dir / filename}")
        if max_batch_size is not None and max_batch_size <= 0:
            raise ValueError("max_batch_size must be greater than zero")
        if batch_wait_ms < 0:
            raise ValueError("batch_wait_ms must be non-negative")

        self.model_dir = model_dir
        self.config = load_json(model_dir / "config.json")
        thinker = self.config["thinker_config"]
        self.tokenizer = load_tokenizer(model_dir)
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
        if max_batch_size is None:
            max_batch_size = 64 if dtype == torch.bfloat16 else 32
        self.max_batch_size = max_batch_size
        self.batch_wait = batch_wait_ms / 1000

        # Built without allocating; the checkpoint's tensors take the parameters' place.
        with torch.device("meta"):
            model = Qwen3ASRForConditionalGeneration(self.config)
        missing, unexpected = model.load_state_dict(
            self._load_weights(), strict=False, assign=True
        )
        if missing or unexpected:
            raise RuntimeError(
                f"weight mismatch: missing={missing}, unexpected={unexpected}"
            )
        self.model = model.eval().to(self.device, self.dtype)
        self._audio = self.model.thinker.audio_tower
        self._text = self.model.thinker.model
        self._lm_head = self.model.thinker.lm_head
        self._varlen = varlen_supported(self.device, self.dtype)

        self.languages = {
            LANGUAGE_CODES[name]: name for name in self.config["support_languages"]
        }
        self._codes = {name: code for code, name in self.languages.items()}
        self.task_ids = {"transcribe": 0}
        encode = self._encode_text
        self._audio_id = thinker["audio_token_id"]
        self._asr_text_id = self.tokenizer.token_to_id("<asr_text>")
        generation = load_json(model_dir / "generation_config.json")
        self._eos_ids = torch.tensor(generation["eos_token_id"], device=self.device)
        # The chat template: the system message holds the context, the user
        # message the audio.
        self._system = encode("<|im_start|>system\n")
        self._user = encode("<|im_end|>\n<|im_start|>user\n<|audio_start|>")
        self._assistant = encode("<|audio_end|><|im_end|>\n<|im_start|>assistant\n")
        self._forced = {
            code: encode(f"language {name}<asr_text>")
            for code, name in self.languages.items()
        }

        text = thinker["text_config"]
        inverse = 1.0 / (
            text["rope_theta"]
            ** (torch.arange(0, text["head_dim"], 2).float() / text["head_dim"])
        )
        angles = torch.arange(CACHE_LENGTH).float()[:, None] * inverse[None, :]
        angles = torch.cat((angles, angles), dim=-1)
        self._cos = angles.cos().to(self.device, self.dtype)
        self._sin = angles.sin().to(self.device, self.dtype)

        # Batch sizes with their own CUDA graph; a batch is padded up to one.
        self._buckets = sorted(
            {
                min(2**power, max_batch_size)
                for power in range(max_batch_size.bit_length())
            }
            | {max_batch_size}
        )
        # varlen_attn reads only each row's keys, so it needs no shorter views.
        self._kv_buckets = sorted(
            {CACHE_LENGTH} if self._varlen else set(KV_BUCKETS) | {CACHE_LENGTH}
        )
        shape = (
            text["num_hidden_layers"],
            max_batch_size,
            CACHE_LENGTH,
            text["num_key_value_heads"],
            text["head_dim"],
        )
        self._cache = KVCache(
            torch.zeros(shape, dtype=self.dtype, device=self.device),
            torch.zeros(shape, dtype=self.dtype, device=self.device),
        )
        self._state = _DecodeState.allocate(max_batch_size, self.device)
        self._rows: dict[int, _Row] = {}
        self._graphs: dict[tuple[int, int], CapturedGraph] = {}
        if self.cuda_graphs:
            self._capture_graphs()

        self._queue: queue.SimpleQueue[_Job] = queue.SimpleQueue()
        threading.Thread(
            target=self._serve, name="asrback-qwen3-asr", daemon=True
        ).start()

    def _load_weights(self) -> dict[str, Tensor]:
        index = self.model_dir / "model.safetensors.index.json"
        if index.is_file():
            files = sorted(set(load_json(index)["weight_map"].values()))
        else:
            files = ["model.safetensors"]
        state_dict = {}
        for filename in files:
            path = self.model_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing model file: {path}")
            state_dict.update(load_file(str(path), device="cpu"))
        return fuse_projections(state_dict)

    def _encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

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
        if task != "transcribe":
            raise ValueError(f"task must be 'transcribe' for this model, got {task!r}")
        if language is not None and language not in self.languages:
            choices = ", ".join(sorted(self.languages))
            raise ValueError(
                f"Unsupported language {language!r}. Available codes: {choices}"
            )
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if max_new_tokens is not None and max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")

        if isinstance(audio, np.ndarray) and audio.dtype != np.uint8:
            waveform = np.asarray(audio, dtype=np.float32)
        else:
            waveform = asrfront.load(audio)
        duration = len(waveform) / SAMPLE_RATE
        context = []
        if prompt and prompt.strip():
            context = self._encode_text(prompt.strip())[-MAX_CONTEXT_TOKENS:]

        # Windows are independent, so all of them are queued at once.
        windows = []
        for seek in range(0, max(len(waveform), 1), WINDOW_SAMPLES):
            window = waveform[seek : seek + WINDOW_SAMPLES]
            if windows and len(window) < MIN_TAIL_SAMPLES:
                break
            length = len(window)
            if length < MIN_WINDOW_SAMPLES:
                window = np.pad(window, (0, MIN_WINDOW_SAMPLES - length))
            features = asrfront.qwen3_asr.log_mel(window)
            job = _Job(features, context, language, temperature, max_new_tokens)
            self._queue.put(job)
            windows.append((seek, length, job))

        segments = []
        detected = None
        for seek, length, job in windows:
            result = self._parse(*job.future.result(), job.language)
            detected = detected or result.language
            segments.append(
                Segment(
                    id=len(segments),
                    seek=seek // HOP_LENGTH,
                    start=round(seek / SAMPLE_RATE, 3),
                    end=round((seek + length) / SAMPLE_RATE, 3),
                    text=result.text,
                    tokens=result.tokens,
                    temperature=temperature,
                    avg_logprob=result.avg_logprob,
                    compression_ratio=compression_ratio(result.text),
                    no_speech_prob=result.no_speech_prob,
                )
            )

        return Transcription(
            task=task,
            language=language or detected or "en",
            duration=round(duration, 3),
            text=" ".join(segment.text for segment in segments if segment.text),
            segments=segments,
        )

    @torch.inference_mode()
    def _capture_graphs(self) -> None:
        if self.device.index is not None:
            torch.cuda.set_device(self.device)
        # One greedy step per (batch size, attention length) pair; they never run
        # at the same time, so they share one memory pool. Warm-up steps advance
        # active rows; keep them inside the shortest length.
        pool = torch.cuda.graph_pool_handle()
        for batch_size in self._buckets:
            state = self._state.narrow(batch_size)
            for length in self._kv_buckets:
                cache = self._cache.narrow(batch_size, length)
                state.finished.zero_()
                state.position.zero_()
                state.end.fill_(CACHE_LENGTH - 1)
                self._graphs[batch_size, length] = CapturedGraph(
                    lambda state=state, cache=cache: self._step(state, cache), pool
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
                self._state.free(torch.tensor(rows, device=self.device))

    def _prompt(self, job: _Job) -> list[int]:
        frames = job.features.shape[1]
        audio = [self._audio_id] * int(asrfront.qwen3_asr.num_audio_tokens(frames))
        prompt = self._system + job.context + self._user + audio + self._assistant
        if job.language is not None:
            prompt += self._forced[job.language]
        return prompt

    def _encode_audio(self, jobs: list[_Job]) -> Tensor:
        """Packed audio embeddings of every job, in order."""
        size = self._audio.chunk_frames
        chunk_frames = []
        window_tokens = []
        for job in jobs:
            frames = job.features.shape[1]
            chunk_frames += [size] * ((frames - 1) // size) + [(frames - 1) % size + 1]
            window_tokens.append(int(asrfront.qwen3_asr.num_audio_tokens(frames)))
        chunks = np.zeros(
            (len(chunk_frames), jobs[0].features.shape[0], size), dtype=np.float32
        )
        index = 0
        for job in jobs:
            for start in range(0, job.features.shape[1], size):
                part = job.features[:, start : start + size]
                chunks[index, :, : part.shape[1]] = part
                index += 1
        chunk_tokens = asrfront.qwen3_asr.num_audio_tokens(np.array(chunk_frames))
        return self._audio(
            to_device(chunks, self.device).to(self.dtype),
            chunk_tokens,
            window_tokens,
            self._varlen,
        )

    @torch.inference_mode()
    def _admit(self, jobs: list[_Job]) -> None:
        """Encode new windows, run their prompts packed back to back straight into
        free rows of the cache, and pick their first tokens."""
        slots = [slot for slot in range(self.max_batch_size) if slot not in self._rows]
        slots = slots[: len(jobs)]
        try:
            # Built on the host: a transfer from GPU (or from pageable memory) here
            # would wait for the work queued before it.
            prompts = [self._prompt(job) for job in jobs]
            lengths = [len(prompt) for prompt in prompts]
            ids = np.concatenate(prompts)
            offsets = np.cumsum(lengths) - lengths
            positions = np.arange(len(ids)) - np.repeat(offsets, lengths)
            # The first token sits at ``prompt_length``; a row may write up to ``end``.
            limits = [
                min(job.max_new_tokens or MAX_NEW_TOKENS, CACHE_LENGTH - length)
                for job, length in zip(jobs, lengths)
            ]
            ends = [length - 1 + limit for length, limit in zip(lengths, limits)]
            audio = np.flatnonzero(ids == self._audio_id)
            ids, rows, positions, audio, slot_rows, start, end = (
                to_device(data, self.device)
                for data in (
                    ids,
                    np.repeat(slots, lengths),
                    positions,
                    audio,
                    slots,
                    lengths,
                    ends,
                )
            )

            x = self._text.embed_tokens(ids)
            x[audio] = self._encode_audio(jobs).to(x.dtype)
            segments = Segments.from_lengths(lengths, self.device, self._varlen)
            hidden = self._text.prefill(
                x,
                self._cache,
                rows,
                positions,
                segments,
                self._cos[positions],
                self._sin[positions],
            )
            last = segments.starts[1:] - 1
            logits = self._lm_head(hidden[last]).float()
            token, logprob = self._choose(logits, self._temperatures(jobs))

            state = self._state
            state.tokens[slot_rows] = token
            state.position[slot_rows] = start
            state.end[slot_rows] = end
            state.finished[slot_rows] = torch.isin(token, self._eos_ids) | (
                start >= end
            )
            state.sum_logprobs[slot_rows] = logprob
            state.generated[slot_rows, start] = token
        except BaseException as exc:
            for job in jobs:
                job.future.set_exception(exc)
            self._state.free(torch.tensor(slots, device=self.device))
            return
        # No transfer to the host here: it would wait for the GPU and leave it idle.
        for slot, job, length, end_position in zip(slots, jobs, lengths, ends):
            self._rows[slot] = _Row(job, length, end_position, length)

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
        generated = self._state.generated[rows, : max(positions) + 1].tolist()
        self._state.free(rows)
        eos = set(self._eos_ids.tolist())
        for slot, position, sum_logprob, tokens in zip(
            done, positions, sums, generated
        ):
            row = self._rows.pop(slot)
            tokens = tokens[row.prompt_length : position + 1]
            if tokens and tokens[-1] in eos:
                tokens.pop()
            # Parsed by the request's thread, so the worker can go on with the GPU.
            row.job.future.set_result((tokens, sum_logprob / (len(tokens) + 1)))

    def _parse(
        self, tokens: list[int], avg_logprob: float, language: str | None
    ) -> _Result:
        """Split ``language <Name><asr_text><text>`` (or just the text, when the
        prompt named the language) into the language code and the text."""
        no_speech_prob = 0.0
        if language is None and self._asr_text_id in tokens:
            split = tokens.index(self._asr_text_id)
            meta = self.tokenizer.decode(tokens[:split], skip_special_tokens=True)
            tokens = tokens[split + 1 :]
            for line in meta.splitlines():
                line = line.strip()
                if line.lower().startswith("language "):
                    name = line[len("language ") :].strip()
                    name = name[:1].upper() + name[1:].lower()
                    if name == "None":
                        # The model's answer for audio without speech.
                        no_speech_prob = 1.0
                    else:
                        language = self._codes.get(name, name.lower() or None)
                    break
        text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        return _Result(
            language, tokens, fix_repetitions(text), avg_logprob, no_speech_prob
        )

    def _step(
        self,
        state: _DecodeState,
        cache: KVCache,
        temperatures: Tensor | None = None,
    ) -> None:
        """Decode ``state.tokens`` and select the next token for every row."""
        hidden = self._text.step(
            state.tokens, state.position, cache, self._cos, self._sin, self._varlen
        )
        logits = self._lm_head(hidden).float()
        token, logprob = self._choose(logits, temperatures)
        # Finished and free rows keep their token and position.
        active = ~state.finished
        position = state.position + active
        token = torch.where(active, token, state.tokens)
        state.sum_logprobs.add_(logprob.masked_fill(state.finished, 0.0))
        state.generated.scatter_(1, position[:, None], token[:, None])
        state.finished.logical_or_(
            torch.isin(token, self._eos_ids) | (position >= state.end)
        )
        state.position.copy_(position)
        state.tokens.copy_(token)

    def _choose(
        self, logits: Tensor, temperatures: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        """Next token per row (greedy, or sampled where the temperature is
        positive) and its log-probability."""
        token = logits.argmax(dim=-1)
        if temperatures is not None:
            scaled = logits / temperatures.clamp(min=1e-6)[:, None]
            sampled = torch.multinomial(F.softmax(scaled, dim=-1), 1).squeeze(1)
            token = torch.where(temperatures > 0, sampled, token)
        logprob = F.log_softmax(logits, dim=-1).gather(1, token[:, None]).squeeze(1)
        return token, logprob


def is_qwen3_asr(model_dir: str | os.PathLike) -> bool:
    """Whether ``model_dir`` holds a Qwen3-ASR checkpoint."""
    config = Path(model_dir) / "config.json"
    if not config.is_file():
        return False
    return load_json(config).get("model_type") == "qwen3_asr"
