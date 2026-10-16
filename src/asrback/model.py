"""Whisper encoder-decoder that loads Hugging Face SafeTensors weights."""

from __future__ import annotations

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
        self.scale = self.head_dim**-0.5
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width)
        self.q_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, length, width = x.shape
        return x.view(batch, length, self.heads, width // self.heads).transpose(1, 2)

    def forward(
        self,
        x: Tensor,
        key_value: Tensor | None = None,
        cache: tuple[Tensor, Tensor] | None = None,
        causal: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        query = self._split_heads(self.q_proj(x))
        if key_value is not None:
            if cache is None:
                key = self._split_heads(self.k_proj(key_value))
                value = self._split_heads(self.v_proj(key_value))
            else:
                key, value = cache
        else:
            key = self._split_heads(self.k_proj(x))
            value = self._split_heads(self.v_proj(x))
            if cache is not None:
                key = torch.cat((cache[0], key), dim=2)
                value = torch.cat((cache[1], value), dim=2)

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if causal:
            query_length = query.shape[2]
            key_length = key.shape[2]
            past_length = key_length - query_length
            query_positions = torch.arange(query_length, device=x.device) + past_length
            key_positions = torch.arange(key_length, device=x.device)
            mask = key_positions[None, :] > query_positions[:, None]
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        attention = F.softmax(scores.float(), dim=-1).to(query.dtype)
        output = torch.matmul(attention, value).transpose(1, 2).contiguous()
        output = output.view(x.shape[0], x.shape[1], -1)
        return self.out_proj(output), (key, value)


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
        residual = x
        x, _ = self.self_attn(self.self_attn_layer_norm(x))
        x = residual + x
        residual = x
        x = self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))
        return residual + x


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


LayerCache = tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]


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
        encoder_output: Tensor,
        cache: LayerCache | None,
    ) -> tuple[Tensor, LayerCache]:
        self_cache = None if cache is None else cache[0]
        cross_cache = None if cache is None else cache[1]

        residual = x
        x, self_cache = self.self_attn(
            self.self_attn_layer_norm(x), cache=self_cache, causal=True
        )
        x = residual + x
        residual = x
        x, cross_cache = self.encoder_attn(
            self.encoder_attn_layer_norm(x),
            key_value=encoder_output,
            cache=cross_cache,
        )
        x = residual + x
        residual = x
        x = self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))
        return residual + x, (self_cache, cross_cache)


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
        self,
        tokens: Tensor,
        encoder_output: Tensor,
        caches: list[LayerCache] | None = None,
    ) -> tuple[Tensor, list[LayerCache]]:
        offset = 0 if caches is None else caches[0][0][0].shape[2]
        end = offset + tokens.shape[1]
        if end > self.embed_positions.num_embeddings:
            raise ValueError("decoder input exceeds max_target_positions")
        x = self.embed_tokens(tokens) + self.embed_positions.weight[offset:end]

        new_caches = []
        for index, layer in enumerate(self.layers):
            cache = None if caches is None else caches[index]
            x, new_cache = layer(x, encoder_output, cache)
            new_caches.append(new_cache)
        return self.layer_norm(x), new_caches


class WhisperModel(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.encoder = WhisperEncoder(config)
        self.decoder = WhisperDecoder(config)


class WhisperForConditionalGeneration(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.model = WhisperModel(config)

    def decode(
        self,
        tokens: Tensor,
        encoder_output: Tensor,
        caches: list[LayerCache] | None = None,
    ) -> tuple[Tensor, list[LayerCache]]:
        hidden, caches = self.model.decoder(tokens, encoder_output, caches)
        logits = F.linear(hidden, self.model.decoder.embed_tokens.weight)
        return logits, caches
