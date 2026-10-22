"""Gemma 4 (E2B / E4B) speech recognition in plain PyTorch: a USM-style audio
encoder whose output fills the ``<|audio|>`` positions of a chat prompt, and the
Gemma 4 language model that writes the transcript.

The prompt is the model card's ASR prompt with the audio after the text. Windows
of up to 30 seconds (the model's limit) are independent, so every window of a
request is queued at once. A single worker thread decodes them in a pool of rows
(continuous batching, as for Qwen3-ASR): new windows run their prompts together,
then join the rows already decoding.

The per-layer embedding table (``embed_tokens_per_layer``: 4.7 GB for E2B, 5.6 GB
for E4B in bfloat16) stays in host memory, as the "effective" parameter count
intends: each step looks up the rows of its tokens there and copies them over, so
decoding reads every step's tokens back to the host.
"""

from __future__ import annotations

import json
import math
import os
import queue
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path

import asrfront
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from tokenizers import Tokenizer
from torch import Tensor, nn

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

SAMPLE_RATE = asrfront.gemma4.SAMPLE_RATE
HOP_LENGTH = asrfront.gemma4.HOP_LENGTH
WINDOW_SAMPLES = asrfront.gemma4.MAX_SAMPLES
# Trailing windows shorter than this are mostly padding and invite hallucinations.
MIN_TAIL_SAMPLES = SAMPLE_RATE // 10
# Shorter windows are zero-padded to this length, so that they have audio tokens.
MIN_WINDOW_SAMPLES = SAMPLE_RATE // 10
# Positions per row of the KV cache: prompt (about 820 tokens for 30 s of audio)
# plus generated tokens.
CACHE_LENGTH = 1280
# Attention lengths with their own CUDA graph, besides CACHE_LENGTH.
KV_BUCKETS = (256, 512, 768, 1024)
MAX_NEW_TOKENS = 448
# Windows whose prompts run through the model together; bounds the activations
# (and the per-layer embeddings copied from the host) of a prefill.
PREFILL_ROWS = 8
# The request's ``prompt`` (the system message) keeps at most its last this many tokens.
MAX_CONTEXT_TOKENS = 224

REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
)
PER_LAYER_EMBEDDING = "model.language_model.embed_tokens_per_layer.weight"
# The model card's ASR prompt.
INSTRUCTION = "Transcribe the following speech segment in {0} into {0} text.\n\n"
INSTRUCTION_ANY_LANGUAGE = (
    "Transcribe the following speech segment in its original language.\n\n"
)
FORMATTING = (
    "Follow these specific instructions for formatting the answer:\n"
    "* Only output the transcription, with no newlines.\n"
    "* When transcribing numbers, write the digits, i.e. write 1.7 and not one "
    "point seven, and write 3 instead of three."
)
# Request language codes and the names the prompt gives them.
LANGUAGES = {
    "af": "Afrikaans",
    "am": "Amharic",
    "ar": "Arabic",
    "az": "Azerbaijani",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bs": "Bosnian",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fi": "Finnish",
    "fil": "Filipino",
    "fr": "French",
    "ga": "Irish",
    "gl": "Galician",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "jv": "Javanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "lo": "Lao",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "my": "Burmese",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sq": "Albanian",
    "sr": "Serbian",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "yue": "Cantonese",
    "zh": "Chinese",
    "zu": "Zulu",
}


class RMSNorm(nn.Module):
    """Gemma 4's RMS norm, computed in float32; ``scale=False`` has no weight."""

    def __init__(self, width: int, eps: float, scale: bool = True) -> None:
        super().__init__()
        self.width = width
        self.eps = eps
        if scale:
            self.weight = nn.Parameter(torch.ones(width))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: Tensor) -> Tensor:
        weight = None if self.weight is None else self.weight.float()
        return F.rms_norm(x.float(), (self.width,), weight, self.eps).type_as(x)


class ClippedLinear(nn.Module):
    """A linear layer whose input and output are clamped to the checkpoint's ranges."""

    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__()
        self.linear = nn.Linear(inputs, outputs, bias=False)
        for name in ("input_min", "output_min"):
            self.register_buffer(name, torch.tensor(-math.inf))
        for name in ("input_max", "output_max"):
            self.register_buffer(name, torch.tensor(math.inf))

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x.clamp(self.input_min, self.input_max))
        return x.clamp(self.output_min, self.output_max)


class AudioAttention(nn.Module):
    """Local attention over blocks of ``chunk`` frames: each frame sees itself and
    the ``past - 1`` frames before it, with a relative position bias."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        width = config["hidden_size"]
        self.heads = config["num_attention_heads"]
        self.head_dim = width // self.heads
        self.q_scale = self.head_dim**-0.5 / math.log(2)
        self.k_scale = math.log(1 + math.e) / math.log(2)
        self.chunk = config["attention_chunk_size"]
        self.past = config["attention_context_left"] - 1
        self.future = config["attention_context_right"]
        self.context = self.chunk + self.past + self.future
        self.softcap = config["attention_logit_cap"]
        self.invalid = config["attention_invalid_logits_value"]
        self.q_proj = ClippedLinear(width, width)
        self.k_proj = ClippedLinear(width, width)
        self.v_proj = ClippedLinear(width, width)
        self.post = ClippedLinear(width, width)
        self.relative_k_proj = nn.Linear(width, width, bias=False)
        self.per_dim_scale = nn.Parameter(torch.zeros(self.head_dim))

    def blocked_mask(self, lengths: Tensor, frames: int) -> Tensor:
        """``(batch, 1, blocks, chunk, context)``, True where a query must not
        attend: keys outside its window or past its item's ``lengths``."""
        blocks = -(-frames // self.chunk)
        device = lengths.device
        query = torch.arange(blocks * self.chunk, device=device).view(
            blocks, self.chunk, 1
        )
        key = (torch.arange(blocks, device=device) * self.chunk - self.past).view(
            blocks, 1, 1
        ) + torch.arange(self.context, device=device)
        distance = query - key
        allowed = (distance >= 0) & (distance < self.past) & (key >= 0)
        if self.future:
            allowed |= (distance < 0) & (-distance < self.future)
        allowed = allowed & (key < lengths.view(-1, 1, 1, 1))
        return ~allowed[:, None]

    def _context(self, x: Tensor) -> Tensor:
        """Keys or values around each block: ``(batch, blocks, context, heads,
        head_dim)``."""
        x = F.pad(x, (0, 0, 0, 0, self.past, self.future + self.chunk - 1))
        return torch.movedim(x.unfold(1, self.context, self.chunk), -1, 2)

    def _relative_shift(self, x: Tensor) -> Tensor:
        """Turn scores against relative positions into scores against the block's
        context (Transformer-XL, appendix B)."""
        batch, heads, blocks, chunk, positions = x.shape
        x = F.pad(x, (0, self.context + 1 - positions))
        x = x.view(batch, heads, blocks, chunk * (self.context + 1))
        x = x[..., : chunk * self.context]
        return x.view(batch, heads, blocks, chunk, self.context)

    def forward(self, x: Tensor, positions: Tensor, mask: Tensor) -> Tensor:
        batch, frames, _ = x.shape
        shape = (batch, frames, self.heads, self.head_dim)
        q = self.q_proj(x).float().view(shape)
        q = q * self.q_scale * F.softplus(self.per_dim_scale)
        k = self.k_proj(x).float().view(shape) * self.k_scale
        v = self.v_proj(x).float().view(shape)

        blocks = mask.shape[2]
        q = F.pad(q, (0, 0, 0, 0, 0, blocks * self.chunk - frames))
        q = q.view(batch, blocks, self.chunk, self.heads, self.head_dim)
        q = q.permute(0, 3, 1, 2, 4)
        k = self._context(k).permute(0, 3, 1, 4, 2)
        v = self._context(v).permute(0, 3, 1, 2, 4)
        relative = self.relative_k_proj(positions).float()
        relative = relative.view(-1, self.heads, self.head_dim).permute(1, 2, 0)

        content = q @ k
        position = q.reshape(batch, self.heads, -1, self.head_dim) @ relative
        position = self._relative_shift(
            position.view(batch, self.heads, blocks, self.chunk, -1)
        )
        scores = torch.tanh((content + position) / self.softcap) * self.softcap
        scores = scores.masked_fill(mask, self.invalid)
        output = F.softmax(scores, dim=-1) @ v
        output = output.permute(0, 2, 3, 1, 4).reshape(batch, blocks * self.chunk, -1)
        return self.post(output[:, :frames].to(x.dtype))


# The checkpoint's gradient clipping bound, applied at inference too.
CLIP = 1e10


class AudioFeedForward(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        width = config["hidden_size"]
        self.ffw_layer_1 = ClippedLinear(width, 4 * width)
        self.ffw_layer_2 = ClippedLinear(4 * width, width)
        self.pre_layer_norm = RMSNorm(width, config["rms_norm_eps"])
        self.post_layer_norm = RMSNorm(width, config["rms_norm_eps"])
        self.scale = config["residual_weight"]

    def forward(self, x: Tensor) -> Tensor:
        h = self.pre_layer_norm(x.clamp(-CLIP, CLIP))
        h = self.ffw_layer_2(F.silu(self.ffw_layer_1(h)))
        return x + self.post_layer_norm(h.clamp(-CLIP, CLIP)) * self.scale


class AudioLightConv(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        width = config["hidden_size"]
        self.linear_start = ClippedLinear(width, 2 * width)
        self.linear_end = ClippedLinear(width, width)
        kernel = config["conv_kernel_size"]
        # Causal: padded on the left only.
        self.depthwise_conv1d = nn.Conv1d(
            width, width, kernel, groups=width, bias=False
        )
        self.left_pad = kernel - 1
        self.pre_layer_norm = RMSNorm(width, config["rms_norm_eps"])
        self.conv_norm = RMSNorm(width, config["rms_norm_eps"])

    def forward(self, x: Tensor) -> Tensor:
        h = F.glu(self.linear_start(self.pre_layer_norm(x)), dim=-1)
        h = self.depthwise_conv1d(F.pad(h.transpose(1, 2), (self.left_pad, 0)))
        h = self.conv_norm(h.transpose(1, 2).clamp(-CLIP, CLIP))
        return x + self.linear_end(F.silu(h))


class AudioLayer(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        width = config["hidden_size"]
        self.feed_forward1 = AudioFeedForward(config)
        self.feed_forward2 = AudioFeedForward(config)
        self.self_attn = AudioAttention(config)
        self.lconv1d = AudioLightConv(config)
        self.norm_pre_attn = RMSNorm(width, config["rms_norm_eps"])
        self.norm_post_attn = RMSNorm(width, config["rms_norm_eps"])
        self.norm_out = RMSNorm(width, config["rms_norm_eps"])

    def forward(self, x: Tensor, positions: Tensor, mask: Tensor) -> Tensor:
        x = self.feed_forward1(x)
        h = self.self_attn(self.norm_pre_attn(x.clamp(-CLIP, CLIP)), positions, mask)
        x = x + self.norm_post_attn(h.clamp(-CLIP, CLIP))
        x = self.feed_forward2(self.lconv1d(x))
        return self.norm_out(x.clamp(-CLIP, CLIP))


class SubsampleLayer(nn.Module):
    def __init__(self, inputs: int, outputs: int, eps: float) -> None:
        super().__init__()
        self.conv = nn.Conv2d(inputs, outputs, 3, 2, padding=1, bias=False)
        self.norm = nn.LayerNorm(outputs, eps=eps, bias=False)

    def forward(self, x: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        """``x`` is ``(batch, channels, frames, bins)``; frames past ``valid`` are
        zeroed first."""
        x = self.conv(x * valid[:, None, :, None])
        x = F.relu(self.norm(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        return x, valid[:, ::2]


class Subsample(nn.Module):
    """Two stride-2 convolutions: 4 frames (40 ms) per audio token."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        first, second = config["subsampling_conv_channels"]
        self.layer0 = SubsampleLayer(1, first, config["rms_norm_eps"])
        self.layer1 = SubsampleLayer(first, second, config["rms_norm_eps"])
        self.input_proj_linear = nn.Linear(
            asrfront.gemma4.N_MELS // 4 * second, config["hidden_size"], bias=False
        )

    def forward(self, features: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        x, valid = self.layer0(features[:, None], valid)
        x, valid = self.layer1(x, valid)
        batch, _, frames, _ = x.shape
        x = x.permute(0, 2, 3, 1).reshape(batch, frames, -1)
        return self.input_proj_linear(x), valid


def relative_positions(config: dict) -> Tensor:
    """Sinusoids ``[sin, cos]`` of the relative distances ``past .. 0``."""
    width = config["hidden_size"]
    context = (
        config["attention_chunk_size"]
        + config["attention_context_left"]
        - 1
        + config["attention_context_right"]
    )
    timescales = width // 2
    increment = math.log(10000) / max(timescales - 1, 1)
    # An explicit device, so the table is real even when the model is built on meta.
    inverse = torch.exp(torch.arange(timescales, device="cpu") * -increment)
    distances = torch.arange(context // 2, -1, -1, device="cpu")[:, None]
    scaled = distances * inverse[None, :]
    return torch.cat((torch.sin(scaled), torch.cos(scaled)), dim=-1)


class AudioEncoder(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.subsample_conv_projection = Subsample(config)
        self.register_buffer("positions", relative_positions(config), persistent=False)
        self.layers = nn.ModuleList(
            AudioLayer(config) for _ in range(config["num_hidden_layers"])
        )
        self.output_proj = nn.Linear(config["hidden_size"], config["output_proj_dims"])

    def forward(self, features: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        """Audio tokens ``(batch, tokens, output_proj_dims)`` of ``features``
        ``(batch, frames, n_mels)`` with ``lengths`` valid frames, and the mask of
        the valid tokens."""
        valid = (
            torch.arange(features.shape[1], device=features.device) < lengths[:, None]
        )
        x, valid = self.subsample_conv_projection(features, valid)
        attention = self.layers[0].self_attn
        mask = attention.blocked_mask(valid.sum(-1), x.shape[1])
        positions = self.positions.to(x.dtype)
        for layer in self.layers:
            x = layer(x, positions, mask)
        return self.output_proj(x), valid


class AudioEmbedder(nn.Module):
    """Audio tokens into the language model's embedding space."""

    def __init__(self, audio: dict, text: dict) -> None:
        super().__init__()
        width = audio["output_proj_dims"]
        self.embedding_pre_projection_norm = RMSNorm(
            width, audio["rms_norm_eps"], scale=False
        )
        self.embedding_projection = nn.Linear(width, text["hidden_size"], bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding_projection(self.embedding_pre_projection_norm(x))


def rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotary embedding of ``(..., heads, head_dim)`` with ``cos``/``sin``
    ``(..., head_dim)``."""
    first, second = x.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return x * cos[..., None, :] + rotated * sin[..., None, :]


def rotary_tables(
    text: dict, layer_type: str, head_dim: int, length: int
) -> tuple[Tensor, Tensor]:
    """``cos``/``sin`` ``(length, head_dim)`` of a layer type. Proportional RoPE
    rotates the first ``partial_rotary_factor`` of the frequencies (still counted
    against the whole ``head_dim``) and leaves the rest unrotated."""
    parameters = text["rope_parameters"][layer_type]
    angles = head_dim // 2
    if parameters["rope_type"] == "proportional":
        angles = int(parameters.get("partial_rotary_factor", 1.0) * head_dim // 2)
    elif parameters["rope_type"] != "default":
        raise ValueError(f"unsupported RoPE type {parameters['rope_type']!r}")
    inverse = 1.0 / (
        parameters["rope_theta"] ** (torch.arange(0, 2 * angles, 2).float() / head_dim)
    )
    inverse = F.pad(inverse, (0, head_dim // 2 - angles))
    angles = torch.arange(length).float()[:, None] * inverse[None, :]
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos(), angles.sin()


@dataclass
class LayerKV:
    """Preallocated keys and values of a layer, ``(batch, length, kv_heads, head_dim)``.

    Entries are written in place, so the tensors keep fixed addresses (as CUDA
    graphs require).
    """

    keys: Tensor
    values: Tensor

    def narrow(self, batch_size: int, length: int) -> LayerKV:
        return LayerKV(
            self.keys[:batch_size, :length], self.values[:batch_size, :length]
        )


class TextAttention(nn.Module):
    """Attention with query/key/value norms and no scaling. Layers in the last
    ``num_kv_shared_layers`` have no keys and values of their own: they read those
    of the last earlier layer of their type."""

    def __init__(self, text: dict, layer_type: str, shared: bool) -> None:
        super().__init__()
        width = text["hidden_size"]
        eps = text["rms_norm_eps"]
        full = layer_type == "full_attention"
        self.head_dim = text["global_head_dim"] if full else text["head_dim"]
        self.heads = text["num_attention_heads"]
        self.kv_heads = text["num_key_value_heads"]
        if full and text.get("attention_k_eq_v") and text["num_global_key_value_heads"]:
            self.kv_heads = text["num_global_key_value_heads"]
        self.window = None if full else text["sliding_window"]
        self.layer_type = layer_type
        self.shared = shared
        self.q_norm = RMSNorm(self.head_dim, eps)
        if shared:
            self.q_proj = nn.Linear(width, self.heads * self.head_dim, bias=False)
        else:
            self.sizes = [
                size * self.head_dim
                for size in (self.heads, self.kv_heads, self.kv_heads)
            ]
            # The checkpoint's q_proj, k_proj and v_proj as one matrix.
            self.qkv_proj = nn.Linear(width, sum(self.sizes), bias=False)
            self.k_norm = RMSNorm(self.head_dim, eps)
            self.v_norm = RMSNorm(self.head_dim, eps, scale=False)
        self.o_proj = nn.Linear(self.heads * self.head_dim, width, bias=False)

    def _project(
        self, x: Tensor, cos: Tensor, sin: Tensor
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """``(..., heads, head_dim)`` queries, keys and values (None for a shared
        layer) for ``(..., width)``."""
        shape = (*x.shape[:-1], -1, self.head_dim)
        if self.shared:
            return rotate(self.q_norm(self.q_proj(x).view(shape)), cos, sin), None, None
        q, k, v = self.qkv_proj(x).split(self.sizes, dim=-1)
        q = rotate(self.q_norm(q.reshape(shape)), cos, sin)
        k = rotate(self.k_norm(k.reshape(shape)), cos, sin)
        return q, k, self.v_norm(v.reshape(shape))

    def prefill(
        self,
        x: Tensor,
        cache: LayerKV,
        rows: Tensor,
        kv: dict[str, tuple[Tensor, Tensor]],
        mask: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> Tensor:
        """Attention over padded prompts ``x`` (batch, length, width) written to
        ``rows`` of the cache; ``kv`` carries each layer type's keys and values to
        the shared layers."""
        q, k, v = self._project(x, cos, sin)
        if self.shared:
            k, v = kv[self.layer_type]
        else:
            length = x.shape[1]
            cache.keys[rows, :length] = k
            cache.values[rows, :length] = v
            kv[self.layer_type] = k, v
        output = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=mask,
            scale=1.0,
            enable_gqa=True,
        )
        return self.o_proj(output.transpose(1, 2).flatten(2))

    def step(
        self,
        x: Tensor,
        cache: LayerKV,
        positions: Tensor,
        mask: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> Tensor:
        """One token per row at its own position, attending over the cache with
        ``mask`` ``(batch, 1, 1, length)``."""
        q, k, v = self._project(x, cos, sin)
        if not self.shared:
            rows = torch.arange(len(x), device=x.device)
            cache.keys[rows, positions] = k
            cache.values[rows, positions] = v
        # The query heads sharing a key/value head attend as one group of queries.
        q = q.view(len(x), self.kv_heads, -1, self.head_dim)
        output = F.scaled_dot_product_attention(
            q,
            cache.keys.transpose(1, 2),
            cache.values.transpose(1, 2),
            attn_mask=mask,
            scale=1.0,
        )
        return self.o_proj(output.reshape(len(x), -1))


class TextMLP(nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        # The checkpoint's gate_proj and up_proj as one matrix.
        self.gate_up_proj = nn.Linear(width, 2 * hidden, bias=False)
        self.down_proj = nn.Linear(hidden, width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.gelu(gate, approximate="tanh") * up)


class DecoderLayer(nn.Module):
    def __init__(self, text: dict, index: int) -> None:
        super().__init__()
        width = text["hidden_size"]
        eps = text["rms_norm_eps"]
        first_shared = text["num_hidden_layers"] - text["num_kv_shared_layers"]
        shared = index >= first_shared > 0
        hidden = text["intermediate_size"]
        if shared and text["use_double_wide_mlp"]:
            hidden *= 2
        self.self_attn = TextAttention(text, text["layer_types"][index], shared)
        self.mlp = TextMLP(width, hidden)
        self.input_layernorm = RMSNorm(width, eps)
        self.post_attention_layernorm = RMSNorm(width, eps)
        self.pre_feedforward_layernorm = RMSNorm(width, eps)
        self.post_feedforward_layernorm = RMSNorm(width, eps)
        per_layer = text["hidden_size_per_layer_input"]
        self.per_layer_input_gate = nn.Linear(width, per_layer, bias=False)
        self.per_layer_projection = nn.Linear(per_layer, width, bias=False)
        self.post_per_layer_input_norm = RMSNorm(width, eps)
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(self, x: Tensor, per_layer: Tensor, attention) -> Tensor:
        x = x + self.post_attention_layernorm(
            attention(self.self_attn, self.input_layernorm(x))
        )
        x = x + self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(x))
        )
        h = F.gelu(self.per_layer_input_gate(x), approximate="tanh") * per_layer
        x = x + self.post_per_layer_input_norm(self.per_layer_projection(h))
        return x * self.layer_scalar


class TextModel(nn.Module):
    """The language model without its per-layer embedding table, which the
    transcriber keeps in host memory."""

    def __init__(self, text: dict) -> None:
        super().__init__()
        width = text["hidden_size"]
        self.layer_count = text["num_hidden_layers"]
        self.per_layer_width = text["hidden_size_per_layer_input"]
        self.embed_tokens = nn.Embedding(text["vocab_size"], width)
        self.layers = nn.ModuleList(
            DecoderLayer(text, index) for index in range(self.layer_count)
        )
        self.norm = RMSNorm(width, text["rms_norm_eps"])
        self.per_layer_model_projection = nn.Linear(
            width, self.layer_count * self.per_layer_width, bias=False
        )
        self.per_layer_projection_norm = RMSNorm(
            self.per_layer_width, text["rms_norm_eps"]
        )
        self.embed_scale = width**0.5
        self.per_layer_embed_scale = self.per_layer_width**0.5
        self.softcap = text["final_logit_softcapping"]

    def embed(self, tokens: Tensor) -> Tensor:
        weight = self.embed_tokens.weight
        return self.embed_tokens(tokens) * torch.tensor(
            self.embed_scale, dtype=weight.dtype
        )

    def per_layer_inputs(self, x: Tensor, embeddings: Tensor) -> Tensor:
        """``(..., layers, per_layer_width)`` inputs of every layer: a projection
        of the embeddings ``x`` joined with the tokens' per-layer ``embeddings``
        (looked up, unscaled)."""
        shape = (*x.shape[:-1], self.layer_count, self.per_layer_width)
        projection = self.per_layer_model_projection(x) * x.shape[-1] ** -0.5
        projection = self.per_layer_projection_norm(projection.view(shape))
        embeddings = embeddings.to(x.dtype).view(shape) * self.per_layer_embed_scale
        return (projection + embeddings) * 2**-0.5

    def logits(self, hidden: Tensor) -> Tensor:
        """Float32 logits, soft-capped, of the tied embedding matrix."""
        logits = F.linear(self.norm(hidden), self.embed_tokens.weight).float()
        return torch.tanh(logits / self.softcap) * self.softcap

    def prefill(
        self,
        x: Tensor,
        per_layer: Tensor,
        caches: list[LayerKV],
        rows: Tensor,
        masks: dict[str, Tensor],
        rotary: dict[str, tuple[Tensor, Tensor]],
    ) -> Tensor:
        """Hidden states of padded prompts ``x`` (batch, length, width)."""
        kv: dict[str, tuple[Tensor, Tensor]] = {}
        for index, layer in enumerate(self.layers):
            kind = layer.self_attn.layer_type
            x = layer(
                x,
                per_layer[:, :, index],
                lambda attention, h: attention.prefill(
                    h, caches[index], rows, kv, masks[kind], *rotary[kind]
                ),
            )
        return x

    def step(
        self,
        tokens: Tensor,
        embeddings: Tensor,
        positions: Tensor,
        caches: list[LayerKV],
        rotary: dict[str, tuple[Tensor, Tensor]],
        windows: dict[str, int | None],
    ) -> Tensor:
        """Logits for one token per row (``tokens``, ``positions``: (batch,);
        ``embeddings``: their per-layer embeddings) over fixed-shape cache views;
        ``rotary`` holds the full tables."""
        x = self.embed(tokens)
        per_layer = self.per_layer_inputs(x, embeddings)
        length = caches[0].keys.shape[1]
        key_positions = torch.arange(length, device=x.device)[None, :]
        causal = key_positions <= positions[:, None]
        masks = {}
        for kind, window in windows.items():
            mask = causal
            if window is not None:
                mask = mask & (key_positions > positions[:, None] - window)
            masks[kind] = mask[:, None, None, :]
        tables = {
            kind: (cos[positions], sin[positions])
            for kind, (cos, sin) in rotary.items()
        }
        for index, layer in enumerate(self.layers):
            kind = layer.self_attn.layer_type
            x = layer(
                x,
                per_layer[:, index],
                lambda attention, h: attention.step(
                    h, caches[index], positions, masks[kind], *tables[kind]
                ),
            )
        return self.logits(x)


class Gemma4Model(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.audio_tower = AudioEncoder(config["audio_config"])
        self.embed_audio = AudioEmbedder(config["audio_config"], config["text_config"])
        self.language_model = TextModel(config["text_config"])


class Gemma4ForConditionalGeneration(nn.Module):
    """Module names match the Hugging Face checkpoint (``model.*``), without the
    vision tower, except for the projections :func:`fuse_projections` joins."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.model = Gemma4Model(config)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


# Checkpoint projections loaded as one matrix: fewer, larger matrix multiplications.
FUSED_PROJECTIONS = {
    "self_attn.qkv_proj": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    "mlp.gate_up_proj": ("mlp.gate_proj", "mlp.up_proj"),
}


def fuse_projections(state_dict: dict[str, Tensor], shared: set[int]) -> None:
    """Join the language model's projection weights as :data:`FUSED_PROJECTIONS`
    lists, in place. The ``shared`` layers keep only their query projection; the
    checkpoint's keys and values of theirs are unused."""
    prefix = "model.language_model.layers."
    for name in list(state_dict):
        if not name.startswith(prefix):
            continue
        layer = int(name[len(prefix) :].split(".")[0])
        if layer in shared and any(
            f".self_attn.{part}." in name
            for part in ("k_proj", "v_proj", "k_norm", "v_norm")
        ):
            del state_dict[name]
    for name in list(state_dict):
        for fused, parts in FUSED_PROJECTIONS.items():
            if name.startswith(prefix) and name.endswith(f".{parts[0]}.weight"):
                start = name[: -len(f"{parts[0]}.weight")]
                if not all(f"{start}{part}.weight" in state_dict for part in parts):
                    continue
                state_dict[f"{start}{fused}.weight"] = torch.cat(
                    [state_dict.pop(f"{start}{part}.weight") for part in parts]
                )


@dataclass
class _Job:
    """One window of up to 30 seconds waiting to be decoded."""

    features: np.ndarray  # (frames, n_mels)
    frames: int  # valid frames
    context: list[int]
    language: str | None
    temperature: float
    max_new_tokens: int | None
    future: Future = field(default_factory=Future)


@dataclass
class _Row:
    """An occupied row of the decoding pool. ``tokens[-1]`` is the row's latest
    token, at ``position``; the row may write up to ``end``."""

    job: _Job
    position: int
    end: int
    tokens: list[int]
    sum_logprob: float


class Gemma4Transcriber:
    """Loads a Gemma 4 checkpoint once and transcribes audio.

    Same interface as :class:`asrback.Transcriber`: ``transcribe`` is thread-safe
    and windows from concurrent calls are decoded together in a pool of
    ``max_batch_size`` rows. The prompt names the request's language, or asks for
    the original language without one. ``prompt`` becomes the system message
    (context such as names or terms). Translation is not supported. ``cuda_graphs``
    (default: on for CUDA) replays each decoder step from a captured graph.
    """

    def __init__(
        self,
        model_dir: str | os.PathLike,
        device: str | None = None,
        dtype: str | torch.dtype | None = None,
        max_batch_size: int | None = None,
        batch_wait_ms: float = 5.0,
        cuda_graphs: bool | None = None,
        max_audio_seconds: float | None = None,
    ):
        model_dir = Path(model_dir)
        for filename in REQUIRED_FILES:
            if not (model_dir / filename).is_file():
                raise FileNotFoundError(f"Missing model file: {model_dir / filename}")
        if max_batch_size is not None and max_batch_size <= 0:
            raise ValueError("max_batch_size must be greater than zero")
        if batch_wait_ms < 0:
            raise ValueError("batch_wait_ms must be non-negative")
        if max_audio_seconds is not None and max_audio_seconds <= 0:
            raise ValueError("max_audio_seconds must be greater than zero")

        self.model_dir = model_dir
        self.config = load_json(model_dir / "config.json")
        text = self.config["text_config"]
        if text.get("enable_moe_block") or not text.get("hidden_size_per_layer_input"):
            raise ValueError(
                "Only the Gemma 4 E2B and E4B checkpoints (with audio) are supported"
            )
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
        if max_batch_size is None:
            max_batch_size = 32 if dtype == torch.bfloat16 else 16
        self.max_batch_size = max_batch_size
        self.batch_wait = batch_wait_ms / 1000
        self.max_audio_seconds = max_audio_seconds

        first_shared = text["num_hidden_layers"] - text["num_kv_shared_layers"]
        shared = set(range(first_shared, text["num_hidden_layers"]))
        # Built without allocating; the checkpoint's tensors take the parameters' place.
        with torch.device("meta"):
            model = Gemma4ForConditionalGeneration(self.config)
        state_dict = self._load_weights()
        fuse_projections(state_dict, shared)
        missing, unexpected = model.load_state_dict(
            state_dict, strict=False, assign=True
        )
        if missing or unexpected:
            raise RuntimeError(
                f"weight mismatch: missing={missing}, unexpected={unexpected}"
            )
        del state_dict
        self.model = model.eval().to(self.device, self.dtype)
        self._audio = self.model.model.audio_tower
        self._embed_audio = self.model.model.embed_audio
        self._text = self.model.model.language_model

        self.languages = dict(LANGUAGES)
        self.task_ids = {"transcribe": 0}
        token = self.tokenizer.token_to_id
        self._pad_id = text["pad_token_id"]
        self._audio_id = self.config["audio_token_id"]
        generation = load_json(model_dir / "generation_config.json")
        self._eos_ids = set(generation["eos_token_id"])
        encode = self._encode_text
        self._turn = [token("<|turn>")]
        self._end_turn = [token("<turn|>")] + encode("\n")
        self._bos = [text["bos_token_id"]]
        self._system = self._turn + encode("system\n")
        self._audio_start = [self.config["boa_token_id"]]
        self._audio_end = [self.config["eoa_token_id"]]
        self._model_turn = self._turn + encode("model\n")
        self._instructions = {
            code: self._turn + encode("user\n" + INSTRUCTION.format(name) + FORMATTING)
            for code, name in self.languages.items()
        }
        self._instructions[None] = self._turn + encode(
            "user\n" + INSTRUCTION_ANY_LANGUAGE + FORMATTING
        )

        # Per layer type: cos/sin tables and the attention window.
        self._rotary = {}
        self._windows = {}
        for layer in self._text.layers:
            attention = layer.self_attn
            if attention.layer_type not in self._rotary:
                cos, sin = rotary_tables(
                    text, attention.layer_type, attention.head_dim, CACHE_LENGTH
                )
                self._rotary[attention.layer_type] = (
                    cos.to(self.device, self.dtype),
                    sin.to(self.device, self.dtype),
                )
                self._windows[attention.layer_type] = attention.window

        # Keys and values of the layers with their own; a shared layer reads those
        # of the last earlier layer of its type.
        own = {}
        self._caches: list[LayerKV] = []
        for index, layer in enumerate(self._text.layers):
            attention = layer.self_attn
            if attention.shared:
                self._caches.append(self._caches[own[attention.layer_type]])
                continue
            own[attention.layer_type] = index
            shape = (
                max_batch_size,
                CACHE_LENGTH,
                attention.kv_heads,
                attention.head_dim,
            )
            self._caches.append(
                LayerKV(
                    torch.zeros(shape, dtype=self.dtype, device=self.device),
                    torch.zeros(shape, dtype=self.dtype, device=self.device),
                )
            )

        # Batch sizes with their own CUDA graph; a batch is padded up to one.
        self._buckets = sorted(
            {
                min(2**power, max_batch_size)
                for power in range(max_batch_size.bit_length())
            }
            | {max_batch_size}
        )
        self._kv_buckets = sorted(set(KV_BUCKETS) | {CACHE_LENGTH})
        # Fixed-address step inputs and outputs.
        width = self._text.layer_count * self._text.per_layer_width
        self._step_tokens = torch.zeros(
            max_batch_size, dtype=torch.long, device=self.device
        )
        self._step_positions = torch.zeros_like(self._step_tokens)
        self._step_embeddings = torch.zeros(
            (max_batch_size, width), dtype=self._per_layer.dtype, device=self.device
        )
        self._step_logits = torch.zeros(
            (max_batch_size, text["vocab_size"]), device=self.device
        )
        self._step_host = torch.zeros(
            (max_batch_size, width),
            dtype=self._per_layer.dtype,
            pin_memory=self.device.type == "cuda",
        )
        self._rows: dict[int, _Row] = {}
        self._graphs: dict[tuple[int, int], CapturedGraph] = {}
        if self.cuda_graphs:
            self._capture_graphs()

        self._queue: queue.SimpleQueue[_Job] = queue.SimpleQueue()
        self._worker = Worker(self._serve, "asrback-gemma4")

    @property
    def alive(self) -> bool:
        """Whether the decoding worker still runs."""
        return self._worker.alive

    def _load_weights(self) -> dict[str, Tensor]:
        """The checkpoint's tensors, without the vision tower, read straight to the
        device; the per-layer embedding table stays in host memory."""
        index = self.model_dir / "model.safetensors.index.json"
        if index.is_file():
            files = sorted(set(load_json(index)["weight_map"].values()))
        else:
            files = ["model.safetensors"]
        state_dict = {}
        self._per_layer = None
        for filename in files:
            path = self.model_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing model file: {path}")
            with safe_open(str(path), "pt", device=str(self.device)) as file:
                for name in file.keys():
                    if name == PER_LAYER_EMBEDDING or ".vision_tower." in name:
                        continue
                    if name.startswith("model.embed_vision."):
                        continue
                    state_dict[name] = file.get_tensor(name).to(self.dtype)
            with safe_open(str(path), "pt", device="cpu") as file:
                if PER_LAYER_EMBEDDING in file.keys():
                    self._per_layer = file.get_tensor(PER_LAYER_EMBEDDING)
        if self._per_layer is None:
            raise RuntimeError(f"weight mismatch: missing={[PER_LAYER_EMBEDDING]}")
        return state_dict

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

        waveform = load_audio(audio, self.max_audio_seconds)
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
            job = _Job(
                asrfront.gemma4.log_mel(window),
                asrfront.gemma4.num_frames(len(window)),
                context,
                language,
                temperature,
                max_new_tokens,
            )
            self._queue.put(job)
            windows.append((seek, length, job))

        segments = []
        for seek, length, job in windows:
            tokens, avg_logprob = self._worker.wait(job.future)
            text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
            segments.append(
                Segment(
                    id=len(segments),
                    seek=seek // HOP_LENGTH,
                    start=round(seek / SAMPLE_RATE, 3),
                    end=round((seek + length) / SAMPLE_RATE, 3),
                    text=text,
                    tokens=tokens,
                    temperature=temperature,
                    avg_logprob=avg_logprob,
                    compression_ratio=compression_ratio(text),
                    no_speech_prob=0.0,
                )
            )

        return Transcription(
            task=task,
            language=language or "unknown",
            duration=round(duration, 3),
            text=" ".join(segment.text for segment in segments if segment.text),
            segments=segments,
        )

    @torch.inference_mode()
    def _capture_graphs(self) -> None:
        if self.device.index is not None:
            torch.cuda.set_device(self.device)
        # One step per (batch size, attention length) pair; they never run at the
        # same time, so they share one memory pool. Warm-up steps write position 0
        # of free rows.
        pool = torch.cuda.graph_pool_handle()
        for batch_size in self._buckets:
            for length in self._kv_buckets:
                self._graphs[batch_size, length] = CapturedGraph(
                    lambda b=batch_size, n=length: self._step(b, n), pool
                )

    def _step(self, batch_size: int, length: int) -> None:
        """Logits of the step inputs' first ``batch_size`` rows, into
        ``_step_logits``."""
        caches = [cache.narrow(batch_size, length) for cache in self._caches]
        self._step_logits[:batch_size].copy_(
            self._text.step(
                self._step_tokens[:batch_size],
                self._step_embeddings[:batch_size],
                self._step_positions[:batch_size],
                caches,
                self._rotary,
                self._windows,
            )
        )

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
                for start in range(0, len(jobs), PREFILL_ROWS):
                    self._admit(jobs[start : start + PREFILL_ROWS])
            if not self._rows:
                continue
            try:
                self._advance()
            except BaseException as exc:
                # A failed step leaves every row's state unknown: fail them all.
                for slot in list(self._rows):
                    self._rows.pop(slot).job.future.set_exception(exc)
                raise_if_device_failed(self.device)

    def _prompt(self, job: _Job) -> list[int]:
        audio = [self._audio_id] * asrfront.gemma4.num_audio_tokens(job.frames)
        prompt = list(self._bos)
        if job.context:
            prompt += self._system + job.context + self._end_turn
        return (
            prompt
            + self._instructions[job.language]
            + self._audio_start
            + audio
            + self._audio_end
            + self._end_turn
            + self._model_turn
        )

    def _encode_audio(self, jobs: list[_Job]) -> Tensor:
        """Embeddings of every job's valid audio tokens, in order."""
        frames = max(job.features.shape[0] for job in jobs)
        features = np.zeros((len(jobs), frames, asrfront.gemma4.N_MELS), np.float32)
        for index, job in enumerate(jobs):
            features[index, : job.features.shape[0]] = job.features
        lengths = to_device([job.frames for job in jobs], self.device)
        tokens, valid = self._audio(
            to_device(features, self.device).to(self.dtype), lengths
        )
        return self._embed_audio(tokens[valid])

    def _per_layer_embeddings(
        self, tokens: Tensor, out: Tensor | None = None
    ) -> Tensor:
        """Rows of the host's per-layer embedding table for host ``tokens``."""
        return torch.index_select(self._per_layer, 0, tokens, out=out)

    @torch.inference_mode()
    def _admit(self, jobs: list[_Job]) -> None:
        """Encode new windows, run their prompts as one padded batch straight into
        free rows of the cache, and pick their first tokens."""
        slots = [slot for slot in range(self.max_batch_size) if slot not in self._rows]
        slots = slots[: len(jobs)]
        try:
            prompts = [self._prompt(job) for job in jobs]
            lengths = [len(prompt) for prompt in prompts]
            longest = max(lengths)
            ids = np.full((len(jobs), longest), self._pad_id, dtype=np.int64)
            for index, prompt in enumerate(prompts):
                ids[index, : len(prompt)] = prompt
            audio = ids == self._audio_id
            # Audio positions look up the padding token's embeddings.
            ids[audio] = self._pad_id
            embeddings = self._per_layer_embeddings(torch.from_numpy(ids.reshape(-1)))
            ids, audio, embeddings, rows, sizes = (
                to_device(data, self.device)
                for data in (ids, audio, embeddings, slots, lengths)
            )

            x = self._text.embed(ids)
            x[audio] = self._encode_audio(jobs).to(x.dtype)
            per_layer = self._text.per_layer_inputs(x, embeddings.view(*ids.shape, -1))
            positions = torch.arange(longest, device=self.device)
            causal = (positions[None, :] <= positions[:, None]) & (
                positions[None, None, :] < sizes[:, None, None]
            )
            masks = {}
            for kind, window in self._windows.items():
                mask = causal
                if window is not None:
                    mask = mask & (positions[None, :] > positions[:, None] - window)
                masks[kind] = mask[:, None]
            rotary = {
                kind: (cos[:longest], sin[:longest])
                for kind, (cos, sin) in self._rotary.items()
            }
            hidden = self._text.prefill(x, per_layer, self._caches, rows, masks, rotary)
            last = hidden[torch.arange(len(jobs), device=self.device), sizes - 1]
            logits = self._text.logits(last)
            token, logprob = self._choose(logits, self._temperatures(jobs))
            token, logprob = token.tolist(), logprob.tolist()
        except BaseException as exc:
            for job in jobs:
                job.future.set_exception(exc)
            raise_if_device_failed(self.device)
            return
        for slot, job, length, first, first_logprob in zip(
            slots, jobs, lengths, token, logprob
        ):
            # The first token sits at ``length``; a row may write up to ``end``.
            limit = min(job.max_new_tokens or MAX_NEW_TOKENS, CACHE_LENGTH - length)
            row = _Row(job, length, length - 1 + limit, [first], first_logprob)
            if first in self._eos_ids or row.position >= row.end:
                self._finish(row)
            else:
                self._rows[slot] = row

    def _temperatures(self, jobs: list[_Job | None]) -> Tensor | None:
        """Per-row temperatures, or None when every row is greedy."""
        values = [job.temperature if job is not None else 0.0 for job in jobs]
        if not any(value > 0 for value in values):
            return None
        return to_device(values, self.device)

    @torch.inference_mode()
    def _advance(self) -> None:
        """Run one decoder step over the occupied rows and settle the finished ones."""
        # Rows past the highest occupied one are free padding at position 0.
        bucket = next(size for size in self._buckets if size > max(self._rows))
        tokens = np.zeros(bucket, dtype=np.int64)
        positions = np.zeros(bucket, dtype=np.int64)
        for slot, row in self._rows.items():
            tokens[slot] = row.tokens[-1]
            positions[slot] = row.position
        length = int(positions.max()) + 1
        kv_bucket = next(size for size in self._kv_buckets if size >= length)
        host = self._step_host[:bucket]
        self._per_layer_embeddings(torch.from_numpy(tokens), out=host)
        self._step_embeddings[:bucket].copy_(host, non_blocking=True)
        self._step_tokens[:bucket].copy_(to_device(tokens, self.device))
        self._step_positions[:bucket].copy_(to_device(positions, self.device))
        if self.cuda_graphs:
            self._graphs[bucket, kv_bucket].replay()
        else:
            self._step(bucket, kv_bucket)
        temperatures = self._temperatures(
            [
                self._rows[slot].job if slot in self._rows else None
                for slot in range(bucket)
            ]
        )
        token, logprob = self._choose(self._step_logits[:bucket], temperatures)
        # Waits for the step, after which the host buffer may be refilled.
        token, logprob = token.tolist(), logprob.tolist()
        for slot in list(self._rows):
            row = self._rows[slot]
            row.tokens.append(token[slot])
            row.sum_logprob += logprob[slot]
            row.position += 1
            if token[slot] in self._eos_ids or row.position >= row.end:
                self._finish(self._rows.pop(slot))

    def _finish(self, row: _Row) -> None:
        tokens = row.tokens
        if tokens and tokens[-1] in self._eos_ids:
            tokens = tokens[:-1]
        row.job.future.set_result((tokens, row.sum_logprob / (len(tokens) + 1)))

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


def is_gemma4(model_dir: str | os.PathLike) -> bool:
    """Whether ``model_dir`` holds a Gemma 4 checkpoint."""
    config = Path(model_dir) / "config.json"
    if not config.is_file():
        return False
    return load_json(config).get("model_type") == "gemma4"
