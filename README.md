# asrback

OpenAI-compatible speech-to-text server for Whisper checkpoints in Hugging Face
SafeTensors format and for FunASR's SenseVoiceSmall, implemented in plain PyTorch (no
`transformers`, `openai-whisper` or `funasr` at runtime).

## Setup

```bash
uv sync
scripts/download_whisper_hf.sh tiny   # -> models/whisper-tiny (tiny, base, small, *.en, large-v3-turbo)
scripts/download_sensevoice_hf.sh     # -> models/sensevoice-small
```

## Server

```bash
uv run asrback --model-dir models/whisper-tiny --port 8000
uv run asrback --model-dir models/sensevoice-small --port 8000
```

The model type is detected from the directory. SenseVoice transcribes Chinese,
English, Cantonese, Japanese and Korean (`zh`, `en`, `yue`, `ja`, `ko`) with greedy
CTC decoding: it does not translate, and `prompt` and `temperature` have no effect.
Its output includes punctuation and inverse text normalization.

Endpoints:

- `POST /v1/audio/transcriptions`
- `POST /v1/audio/translations`
- `GET /v1/models`
- `GET /health`

Supported response formats are `json`, `text`, `srt`, `vtt` and `verbose_json`.
Audio longer than 30 seconds is decoded in 30-second windows.

Concurrent requests are decoded together in batches. Options that affect speed:

| Option | Default | |
| --- | --- | --- |
| `--dtype` | `bfloat16` on CUDA, else `float32` | Model precision (`float32`, `bfloat16`) |
| `--max-batch-size` | `16` (SenseVoice: `32`) | Most requests decoded together |
| `--batch-wait-ms` | `5` | How long a batch waits to fill |
| `--cuda-graphs` / `--no-cuda-graphs` | on for CUDA | Whisper: replay each decoder step from a captured CUDA graph |
| `--compile-encoder` | off | `torch.compile` the encoder; slower startup, faster for larger models |
| `--condition-on-previous-text` / `--no-condition-on-previous-text` | on | Whisper: condition each 30 s window on the text before it; turned off, the windows of a long file are decoded together |

Any OpenAI client works against the server:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
with open("audio.flac", "rb") as file:
    print(client.audio.transcriptions.create(model="whisper-1", file=file).text)
```

The `model` field is ignored; the server always uses the model it loaded.

## Client

An interactive prompt that transcribes the audio files you enter:

```bash
uv run asrback-client --base-url http://127.0.0.1:8000/v1
audio> path/to/audio.flac
```

## Evaluation

Error rates on LibriSpeech test-clean or a FLEURS test set, against a running server:

```bash
scripts/download_librispeech_test_clean.sh   # -> data/librispeech
uv run asrback-eval --base-url http://127.0.0.1:8000/v1

scripts/download_fleurs_test.sh              # -> data/fleurs/{en_us,ko_kr,ja_jp}
uv run asrback-eval --base-url http://127.0.0.1:8000/v1 --dataset data/fleurs/ko_kr
```

The request language comes from the dataset (`en` for LibriSpeech, `ko` for
`data/fleurs/ko_kr`); override it with `--language`, or pass `--language ''` to let the
server detect it. Japanese, Korean and other languages in which word boundaries are
missing or too coarse are scored by character error rate (spaces ignored), and the
rest by word error rate. `--metric wer|cer` overrides this.

Use `--limit N` for a quick run and `--output predictions.jsonl` to save per-utterance
results; add `--resume` to skip utterances already in that file. Text is normalized
with Whisper's `EnglishTextNormalizer` for English and `BasicTextNormalizer` otherwise
(`pip install asrback[eval]`).

For speed, keep `--concurrency` (default 128) well above the server's
`--max-batch-size` and pass `--sort-by-length` so similar-length files share a batch.
The SenseVoice server also groups queued windows of similar length itself, which
helps more the more requests are in flight. `--base-url` can be repeated to spread
requests over several servers.

```bash
uv run asrback-eval --base-url http://127.0.0.1:8000/v1 --sort-by-length
```

Greedy decoding, server defaults, RTX 5060 Ti; time is the whole test-clean run
(5.4 hours of audio) with `--sort-by-length`:

| Model | WER bfloat16 | WER float32 | Time bfloat16 | Time float32 |
| --- | --- | --- | --- | --- |
| whisper-tiny | 7.52% | 7.54% | 12.9 s | 24.7 s |
| whisper-tiny.en | 5.61% | 5.62% | 11.6 s | 25.0 s |
| whisper-base | 5.05% | 4.99% | 20.7 s | 45.6 s |
| whisper-base.en | 4.20% | 4.23% | 19.1 s | 46.8 s |
| whisper-small | 3.38% | 3.39% | 54.9 s | 144.3 s |
| whisper-small.en | 3.03% | 3.03% | 52.3 s | 134.7 s |
| whisper-large-v3-turbo | 2.07% | 2.10% | 160.1 s | 401.1 s |
| sensevoice-small | 3.44% | 3.45% | 11.9 s | 18.0 s |

`--compile-encoder` brings whisper-small to 49.3 s (small.en 50.2 s) at the cost of
about 20 s more startup; it does not help tiny or base.

FLEURS test, greedy decoding, bfloat16:

| Model | en_us WER | ko_kr CER | ja_jp CER |
| --- | --- | --- | --- |
| whisper-tiny | 15.82% | 17.65% | 43.67% |
| whisper-tiny.en | 14.08% | | |
| whisper-base | 12.84% | 13.89% | 25.12% |
| whisper-base.en | 9.43% | | |
| whisper-small | 8.31% | 5.83% | 12.90% |
| whisper-small.en | 7.77% | | |
| whisper-large-v3-turbo | 6.13% | 3.35% | 6.07% |
| sensevoice-small | 9.06% | 6.61% | 7.56% |

Smaller models sometimes repeat a phrase until the token limit, especially in
Japanese; without temperature fallback (see below) these loops dominate their error
rates.

## Limitations

- Greedy or temperature sampling only; no beam search and no temperature fallback
  on repetitive output.
- No word-level timestamps; segments are fixed 30-second windows.
- One model per server process.
- English-only (`*.en`) and SenseVoice models support transcription only, not
  translation.

## License

The code is released under the [MIT License](LICENSE). Model weights are not part of
this repository and keep their own licenses: Whisper (MIT) and SenseVoiceSmall (see its
[model card](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)).
