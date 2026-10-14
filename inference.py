#!/usr/bin/env python3
"""Run greedy Whisper transcription without transformers or openai-whisper."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import asrfront
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tokenizers import Tokenizer
from torch import Tensor, nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe audio with Hugging Face Whisper SafeTensors weights."
    )
    parser.add_argument("audio", type=Path, help="Audio file to transcribe.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/huggingface/whisper-tiny"),
        help="Directory containing model.safetensors and tokenizer/config JSON files.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code such as en or ko. Default: detect from the audio.",
    )
    parser.add_argument(
        "--task",
        choices=("transcribe", "translate"),
        default="transcribe",
        help="Decode task. Default: transcribe",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device. Default: cuda when available, otherwise cpu",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Maximum generated text tokens. Default: model maximum",
    )
    return parser.parse_args()


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


def load_audio(path: Path, mel_bins: int) -> Tensor:
    audio = asrfront.load(path)
    log_mel = asrfront.whisper.log_mel(audio, n_mels=mel_bins)
    return torch.from_numpy(log_mel).float().unsqueeze(0)


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


@torch.inference_mode()
def detect_language(
    model: WhisperForConditionalGeneration,
    encoder_output: Tensor,
    start_token: int,
    languages: dict[str, int],
) -> str:
    token = torch.tensor([[start_token]], device=encoder_output.device)
    logits, _ = model.decode(token, encoder_output)
    language_ids = list(languages.values())
    best_index = logits[0, -1, language_ids].argmax().item()
    return list(languages.keys())[best_index]


@torch.inference_mode()
def generate(
    model: WhisperForConditionalGeneration,
    encoder_output: Tensor,
    prompt: list[int],
    config: dict,
    no_timestamps_id: int,
    max_new_tokens: int,
) -> list[int]:
    tokens = torch.tensor([prompt], device=encoder_output.device)
    logits, caches = model.decode(tokens, encoder_output)
    generated = []
    eos = config["eos_token_id"]
    suppress = set(config.get("suppress_tokens", []))
    begin_suppress = set(config.get("begin_suppress_tokens", []))

    for step in range(max_new_tokens):
        next_logits = logits[0, -1].float()
        blocked = suppress | (begin_suppress if step == 0 else set())
        if blocked:
            next_logits[list(blocked)] = -math.inf
        next_logits[no_timestamps_id + 1 :] = -math.inf
        next_token = int(next_logits.argmax().item())
        if next_token == eos:
            break
        generated.append(next_token)
        token = torch.tensor([[next_token]], device=encoder_output.device)
        logits, caches = model.decode(token, encoder_output, caches)
    return generated


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def main() -> int:
    args = parse_args()
    if not args.audio.is_file():
        raise SystemExit(f"Audio file not found: {args.audio}")
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be greater than zero")

    required_files = (
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    for filename in required_files:
        if not (args.model_dir / filename).is_file():
            raise SystemExit(f"Missing model file: {args.model_dir / filename}")

    config = load_json(args.model_dir / "config.json")
    tokenizer_config = load_json(args.model_dir / "tokenizer_config.json")
    tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    device = torch.device(args.device)

    model = WhisperForConditionalGeneration(config)
    state_dict = load_file(str(args.model_dir / "model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"weight mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.eval().to(device)

    mel = load_audio(args.audio, config["num_mel_bins"]).to(device)
    with torch.inference_mode():
        encoder_output = model.model.encoder(mel)

    start_id = required_token_id(tokenizer, "<|startoftranscript|>")
    languages = language_token_ids(tokenizer_config)
    language = args.language or detect_language(
        model, encoder_output, start_id, languages
    )
    if language not in languages:
        choices = ", ".join(sorted(languages))
        raise SystemExit(
            f"Unsupported language {language!r}. Available codes: {choices}"
        )

    task_id = required_token_id(tokenizer, f"<|{args.task}|>")
    no_timestamps_id = required_token_id(tokenizer, "<|notimestamps|>")
    prompt = [start_id, languages[language], task_id, no_timestamps_id]
    available = config["max_target_positions"] - len(prompt)
    max_new_tokens = args.max_new_tokens or available
    max_new_tokens = min(max_new_tokens, available)
    token_ids = generate(
        model,
        encoder_output,
        prompt,
        config,
        no_timestamps_id,
        max_new_tokens,
    )
    print(tokenizer.decode(token_ids, skip_special_tokens=True).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
