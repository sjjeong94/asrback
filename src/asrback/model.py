"""Whisper encoder-decoder that loads Hugging Face SafeTensors weights."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


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
        x = F.gelu(self.conv2(x)).transpose(1, 2)
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
        return DecoderCache(
            self.self_keys[:, :batch_size, :, :length],
            self.self_values[:, :batch_size, :, :length],
            self.cross_keys[:, :batch_size],
            self.cross_values[:, :batch_size],
        )


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
        self_keys.index_copy_(2, positions, key)
        self_values.index_copy_(2, positions, value)
        if length is not None:
            # Eager: attend to the filled prefix; a multi-token input starts at 0.
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
        x = self.embed_tokens(tokens) + self.embed_positions(positions)
        mask = None
        if length is None:
            # Fixed-shape step over the whole cache: hide positions not yet written.
            key_positions = torch.arange(cache.self_keys.shape[3], device=x.device)
            mask = key_positions[None, :] <= positions[:, None]
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

        ``length`` is the number of filled cache positions after this call, and
        is ``None`` for a fixed-shape (CUDA graph) step that masks the full cache.
        """
        return self.model.decoder(tokens, positions, cache, length)

    def logits(self, hidden: Tensor) -> Tensor:
        return F.linear(hidden, self.model.decoder.embed_tokens.weight)
