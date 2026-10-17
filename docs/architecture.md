# asrback system flow

This document follows the code of the `asrback` package from server startup to request handling,
speech recognition and response. It also covers the evaluation tool (`asrback-eval`) and the
interactive client (`asrback-client`).

- [1. Package layout](#1-package-layout)
- [2. Server startup](#2-server-startup)
- [3. Request handling](#3-request-handling)
- [4. Concurrency model](#4-concurrency-model)
- [5. Whisper inference](#5-whisper-inference)
- [6. SenseVoice inference](#6-sensevoice-inference)
- [7. Responses](#7-responses)
- [8. Evaluation tool (asrback-eval)](#8-evaluation-tool-asrback-eval)
- [9. Interactive client (asrback-client)](#9-interactive-client-asrback-client)
- [10. Error handling](#10-error-handling)
- [11. Defaults](#11-defaults)

## 1. Package layout

```
src/asrback/
├── __init__.py          # Public API: Transcriber, SenseVoiceTranscriber, load_transcriber, ...
├── __main__.py          # python -m asrback → cli.main
├── cli.py               # asrback: parse arguments, load the model, run uvicorn
├── server.py            # FastAPI app: /v1/audio/*, /v1/models, /health
├── transcription.py     # Shared types and helpers: Segment, Transcription, gather_batch, ...
├── models/
│   ├── __init__.py      # load_transcriber: pick Whisper or SenseVoice from the model directory
│   ├── whisper.py       # Whisper network + Transcriber (batched decoding, CUDA graphs)
│   └── sensevoice.py    # SenseVoiceSmall network + SenseVoiceTranscriber (CTC)
├── evaluation.py        # asrback-eval: WER / CER on LibriSpeech or FLEURS
└── client.py            # asrback-client: interactive transcription client
scripts/                 # Model and dataset download scripts
```

What each external dependency is used for:

| Dependency | Used by | Purpose |
| --- | --- | --- |
| `asrfront` | both models | Audio decoding (`asrfront.load`), Whisper log-mel, SenseVoice fbank + LFR + CMVN |
| `torch` | both models | Network computation, CUDA graphs |
| `safetensors`, `tokenizers` | Whisper | Weights (`model.safetensors`) and tokenizer (`tokenizer.json`) |
| `sentencepiece`, `pyyaml` | SenseVoice | BPE tokenizer, `config.yaml` |
| `fastapi`, `uvicorn[standard]`, `python-multipart` | server | HTTP and multipart upload parsing (with uvloop and httptools) |
| `openai` | eval, client | Calls to the OpenAI-compatible API |
| `openai-whisper` (optional, `eval` extra) | eval | Whisper's text normalizers |

## 2. Server startup

```mermaid
flowchart TD
    A["asrback --model-dir DIR ..."] --> B["cli.parse_args"]
    B --> C["torch.set_float32_matmul_precision('high')"]
    C --> D["models.load_transcriber(DIR, ...)"]
    D --> E{"is_sensevoice(DIR)?<br/>model in config.yaml starts with SenseVoice"}
    E -- yes --> F["SenseVoiceTranscriber(...)"]
    E -- no --> G["Transcriber(...) (Whisper)"]
    F --> H["create_app(transcriber, model_name)"]
    G --> H
    H --> I["uvicorn.run(app)"]
    I --> J["lifespan: raise the AnyIO thread limit<br/>to max(40, 8 × max_batch_size)"]
```

1. **Argument parsing** (`cli.parse_args`): `--max-batch-size` defaults to `None`, which means the
   model's own default (16 for Whisper, 32 for SenseVoice).
2. **Model selection** (`models.load_transcriber`): options left as `None` are dropped. If the
   `model` value in `config.yaml` starts with `SenseVoice`, the directory is a SenseVoice checkpoint;
   otherwise it is treated as Whisper. Passing `--cuda-graphs` or `--compile-encoder` with SenseVoice
   raises `ValueError`, which the CLI turns into an exit with the message (`SystemExit`).
3. **Transcriber construction**: the weights are loaded and moved to the device and dtype, the
   model-specific preparation below runs, and **one decoding worker thread is started.** Only that
   thread touches the model.
   - Whisper: token ids (start of transcript, languages, tasks, `<|notimestamps|>`, ...) and the
     suppression masks are built once. With CUDA graphs on, the decoder cache and state buffers are
     allocated at the maximum batch size and one decoder step is captured for every
     (batch bucket × KV length bucket) pair. With `--compile-encoder`, the encoder is compiled for every
     batch bucket up front.
   - SenseVoice: `config.yaml`, the BPE model and `am.mvn` (CMVN statistics) are read, and `model.pt`
     is loaded with `torch.load(weights_only=True)`. float16 is refused because activations overflow it.
4. **App creation and run** (`server.create_app`, `uvicorn.run`): when the app starts (lifespan), the
   AnyIO default thread limit (40) is raised to at least `8 × max_batch_size`. Section
   [4](#4-concurrency-model) explains why.

## 3. Request handling

```mermaid
sequenceDiagram
    participant C as Client
    participant E as uvicorn event loop
    participant H as Request thread (AnyIO thread pool)
    participant Q as Job queue (SimpleQueue)
    participant W as Decoding worker thread
    C->>E: POST /v1/audio/transcriptions (multipart)
    E->>H: run the sync endpoint
    H->>H: validate form, language, response_format, temperature
    H->>H: transcribe(): decode audio, split 30 s windows, compute features
    H->>Q: put one _Job (with a Future) per window
    W->>Q: gather_batch: collect windows from many requests
    W->>W: batched inference (GPU)
    W-->>H: future.set_result(_Result)
    H->>H: build Segment / Transcription
    H->>E: render(): json / text / srt / vtt / verbose_json
    E-->>C: HTTP response
```

Endpoints (`server.py`):

| Endpoint | Handling |
| --- | --- |
| `POST /v1/audio/transcriptions` | 400 if `timestamp_granularities[]` contains `word`, or if `language` is not supported. Then `run(..., task="transcribe")` |
| `POST /v1/audio/translations` | `run(..., task="translate")`. Models that cannot translate (`*.en`, SenseVoice) raise `ValueError` in `transcribe()`, which becomes a 400 |
| `GET /v1/models`, `GET /v1/models/{id}` | Lists the single loaded model (id from `--model-name` or the directory name). The request's `model` field is ignored |
| `GET /health` | `{"status": "ok"}` |

The shared handler `run()`:

1. Checks that `response_format` is one of `json`, `text`, `srt`, `vtt`, `verbose_json` and that
   `temperature` is between 0 and 1.
2. Reads the whole upload into bytes. An empty file is a 400.
3. Calls `transcriber.transcribe(audio_bytes, temperature=..., language=..., task=..., prompt=...)`,
   which blocks the request thread until the result is ready.
4. Formats the result with `render()` (section [7](#7-responses)).

The endpoints are plain `def`, not `async def`, so the event loop stays responsive while a request
waits for inference: FastAPI runs sync endpoints in the AnyIO thread pool.

## 4. Concurrency model

Both transcribers share this structure:

```mermaid
flowchart LR
    subgraph Request threads
        R1["transcribe() #1"]
        R2["transcribe() #2"]
        R3["transcribe() #N"]
    end
    R1 -- "_Job + Future" --> Q[("queue.SimpleQueue")]
    R2 --> Q
    R3 --> Q
    Q --> W["Worker thread _serve()<br/>gather_batch → batched inference"]
    W -- "future.set_result" --> R1
    W --> R2
    W --> R3
```

- **Request threads** validate input, decode audio (`asrfront.load`), split it into 30-second windows
  and compute features (log-mel or fbank). Feature extraction is CPU work, so it runs in parallel across
  request threads and overlaps with the worker's GPU work.
- **The worker thread** collects jobs from the queue into batches, runs the model and puts each job's
  result into its `Future`. It is the only thread that touches the model and its caches, so no lock is
  needed.
- **Batch collection** (`transcription.gather_batch`): wait for one job, then take more until
  `max_batch_size` is reached or `batch_wait_ms` (default 5 ms) has passed. Jobs that queued up while the
  GPU was busy with the previous batch are taken without waiting.
- **AnyIO thread limit**: each request holds a thread while it waits for its result. With the default
  limit of 40, at most 40 requests can wait inside the server, so batches fill less and SenseVoice has
  fewer windows to group by length. The lifespan hook therefore raises the limit to
  `8 × max_batch_size` (at least 40).
- **Errors**: if batched inference raises, the same exception goes into every `Future` of that batch.
  The worker keeps running.

## 5. Whisper inference

Implementation: `models/whisper.py`. The network (`WhisperForConditionalGeneration`) and decoding
(`Transcriber`) live in the same file.

### 5.1 Request thread: `Transcriber.transcribe`

```mermaid
flowchart TD
    A["Audio (path / bytes / 16 kHz waveform)"] --> B["asrfront.load → float32 waveform"]
    B --> C["_windows: split into 30 s windows<br/>drop a trailing window under 0.1 s"]
    C --> D["asrfront.whisper.log_mel per window<br/>(n_mels, 3000)"]
    D --> E["Window i: _Job(log_mel, previous=tail of history,<br/>language, task, temperature)"]
    E --> F["Queue it and wait on future.result()"]
    F --> G["Tokens → text, append tokens to history,<br/>reuse the detected language"]
    G --> H{"Next window?"}
    H -- yes --> E
    H -- no --> I["Transcription(text=joined window texts, segments)"]
```

- **Window order**: windows are queued **one at a time**, because each window is conditioned on the
  text of the previous ones (`history`). As in openai-whisper, at most half the context
  (`max_target_positions // 2 - 1` tokens) is carried over. A `prompt` seeds `history`.
- **Language**: the language detected in the first window is used for the rest.
- **Segments**: one `Segment` per window. `start` and `end` are the window bounds in seconds, and
  `compression_ratio` is the zlib compression ratio of the text.

### 5.2 Worker: forming batches

`_serve()` splits the jobs returned by `gather_batch` by `_layout(job) = (prefix_length, limit)`.
All rows of a batch start decoding at the same position, so they need the same prompt (previous text)
length and the same token budget. Short audio without a prompt (most evaluation data) all lands in one
group.

- `prefix = [<|startofprev|>, ...previous]` (when there is previous text)
  `+ [<|startoftranscript|>, <|lang|>, <|task|>, <|notimestamps|>]`
- English-only (`*.en`) models start with `[<|startoftranscript|>, <|notimestamps|>]`, without language
  or task tokens.
- `limit = min(max_new_tokens or the remaining room, max_target_positions - prefix_length)`

### 5.3 Worker: `_decode_batch`

```mermaid
flowchart TD
    A["Stack log-mels (B, n_mels, 3000)"] --> B["encoder → (B, 1500, d_model)<br/>(padded to a batch bucket with --compile-encoder)"]
    B --> C["set_encoder_output: compute cross-attention K/V<br/>per layer once, into the cache"]
    C --> D{"Rows without a language?"}
    D -- yes --> E["_detect_languages: one startoftranscript token,<br/>argmax over language tokens"]
    D -- no --> F
    E --> F["One decoder pass over the prefix tokens (B, P), eager<br/>writes self-attention K/V to cache positions 0..P-1"]
    F --> G["no_speech_prob = softmax(logits at SOT)[nospeech token]"]
    G --> H["Select the first token (_select, with begin_suppress)"]
    H --> I{"CUDA graphs on &<br/>temperature 0?"}
    I -- yes --> J["graph.replay() 4 steps at a time<br/>pick a KV length bucket, check EOS in between"]
    I -- no --> K["Repeat eager _step<br/>check EOS every step"]
    J --> L["Cut generated tokens at EOS and build results"]
    K --> L
```

**The decoder cache** (`DecoderCache`) is four preallocated tensors shaped
`(layers, batch, heads, length, head_dim)`. Self-attention keys and values are written in place at each
step's position (`index_copy_`); cross-attention keys and values are filled once after the encoder.
Because the tensors never move, the steps can be captured as CUDA graphs.

**The decoding state** (`_DecodeState`) holds, per row, the current token, the shared position, whether
the row has finished, the sum of log-probabilities and a buffer of generated tokens. One step
(`_step` → `_select`) does all of this on the GPU:

1. Decode the current token at the current position to get logits for the next token.
2. Apply the suppression mask (configured `suppress_tokens` and every timestamp token, plus
   `begin_suppress_tokens` on the first step).
3. Pick the token by greedy argmax, or by sampling for rows with temperature > 0.
4. Add the log-probability for rows that have not finished, and mark rows that produced EOS as finished.
5. Advance the position, write the new token to `generated` and feed it back as the next input.

**CUDA graph path** (the default, when greedy):

- A step reads and writes only fixed-address tensors, so one step is captured in advance for every
  (batch bucket × KV length bucket) pair.
  - Batch buckets: powers of two up to `max_batch_size` (1, 2, 4, 8, 16). A batch is padded up to the
    nearest bucket; the extra rows are padding whose results are discarded.
  - KV length buckets: 64, 128, 256 and 448 (`max_target_positions`). Inside a graph the whole cache
    length is masked, so a shorter bucket reads less memory.
- The host synchronizes on `finished.all()` only every `EOS_CHECK_INTERVAL` (4) steps. Rows that finish
  in between keep decoding, but their scores and outputs ignore those steps.
- A batch with any row at temperature > 0 takes the eager path.

**Eager path** (CPU, `--no-cuda-graphs`, sampling): calls the same `_step` directly, attending only to
the filled part of the cache (no mask), and checks for EOS every step.

**Results**: `generated[:, prefix_length:prefix_length+steps]` is moved to the CPU in one transfer and
each row is cut at its first EOS. `avg_logprob = sum of log-probabilities / (tokens + 1)`, counting EOS
as openai-whisper does.

### 5.4 Network

| Part | Details |
| --- | --- |
| Input | log-mel `(B, n_mels, 3000)`; `n_mels` is 80 (128 for the large-v3 family) |
| Encoder | conv1 (k=3) → GELU → conv2 (k=3, stride 2) → GELU → position embedding → N Transformer blocks → LayerNorm; output `(B, 1500, d_model)` |
| Decoder | token + position embedding → N × (self-attention → cross-attention → MLP) → LayerNorm |
| Output | linear layer sharing the token embedding weights (`logits`) |
| Attention | `F.scaled_dot_product_attention` (unmasked in the encoder, `is_causal` for the decoder prefix) |
| Weights | names match Hugging Face `model.safetensors` (`model.encoder...`, `model.decoder...`) |

## 6. SenseVoice inference

Implementation: `models/sensevoice.py`, a plain PyTorch reimplementation of FunASR's SenseVoiceSmall.

### 6.1 Request thread: `SenseVoiceTranscriber.transcribe`

```mermaid
flowchart TD
    A["Audio"] --> B["asrfront.load → float32 waveform"]
    B --> C["Split into 30 s windows"]
    C --> D["asrfront.sensevoice.fbank per window<br/>(80 mel × LFR 7 frames every 6 + CMVN)<br/>→ (T, 560), 60 ms per frame"]
    D --> E["Queue the _Job of every window at once"]
    E --> F["Wait on all futures"]
    F --> G["Build segments, join window texts with spaces"]
```

Unlike Whisper, windows do not depend on each other, so **all windows are queued at once** and the
windows of a long recording can share a batch. `prompt` and `temperature` are unused (greedy CTC), and
translation is not supported.

### 6.2 Worker: grouping windows of similar length

```mermaid
flowchart TD
    A{"Pending list empty?"} -- yes --> B["Collect jobs with gather_batch"]
    A -- no --> C
    B --> C["Move already-queued jobs to the pending list<br/>(up to 4 × max_batch_size)"]
    C --> D["take_similar: the oldest job + the longest jobs no longer than it<br/>→ if short, the shortest jobs longer than it"]
    D --> E["_decode_batch"]
    E --> A
```

A batch is padded to its longest window, so mixed lengths waste computation. `take_similar` anchors on
the job that has waited longest and fills the batch **with shorter jobs first**, since they do not raise
the batch's maximum length. The anchor is always included, so no job waits forever. The more requests
wait inside the server (the more are in flight), the more candidates there are and the better the
grouping.

### 6.3 Worker: `_decode_batch`

```mermaid
flowchart TD
    A["Pad features to the longest (B, T, 560)"] --> B["4 queries: [language (0 for auto), event 1, emotion 2, ITN 14/15]"]
    B --> C["embed(queries) followed by features → (B, 4+T, 560)"]
    C --> D["× √512 + sinusoidal positions (1..4+T)"]
    D --> E["encoders0 (560→512, no residual)<br/>→ 49 encoders → after_norm<br/>→ 20 tp_encoders → tp_norm"]
    E --> F["ctc_lo → logits (B, 4+T, 25055)"]
    F --> G["greedy_ctc: argmax and logsumexp per frame<br/>→ best token, its log-prob, P(nospeech)"]
    G --> H["Mean log-prob over speech frames per row, one transfer to the CPU"]
    H --> I["Per row: merge repeats → drop blanks (0)<br/>→ language from its tag → drop special tokens → BPE decode"]
```

- **Mask**: when every row has the same length, no mask is used. Otherwise a `(B, 4+T)` mask hides
  padding frames from the FSMN input and from attention keys. Batching rows of different lengths gives
  the same results as decoding them one by one.
- **SAN-M attention**: one linear layer produces Q, K and V; the output is SDPA attention plus an FSMN
  memory, a depthwise convolution (k=11) over V.
- **Reading the output**: the CTC output at the first four frames carries a language tag such as
  `<|en|>` and emotion, event and ITN tags. The language tag gives the detected language, and every
  `<|...|>` special token is removed from the text.
- **ITN**: `withitn` (14) by default, so the output includes punctuation and number normalization.
- **dtype**: activations reach about 71,000 in the last layers, beyond the float16 maximum (65,504), so
  the CUDA default is bfloat16 and float16 is refused.

## 7. Responses

`server.render()` turns a `Transcription` into the requested format:

| Format | Content |
| --- | --- |
| `json` | `{"text": ...}` |
| `text` | the text plus a newline |
| `verbose_json` | `task`, `language`, `duration`, `text`, `segments` (id, seek, start, end, text, tokens, temperature, avg_logprob, compression_ratio, no_speech_prob) |
| `srt`, `vtt` | one subtitle block per window segment (`HH:MM:SS,mmm` / `HH:MM:SS.mmm`) |

Segments are 30-second windows; there are no word-level timestamps.

## 8. Evaluation tool (asrback-eval)

Implementation: `evaluation.py`. It sends requests to a running server through the OpenAI client and
measures error rates.

```mermaid
flowchart TD
    A["--dataset path"] --> B{"test.tsv present?"}
    B -- yes --> C["load_fleurs: 7-column TSV → (wav file name, raw transcript)"]
    B -- no --> D["load_librispeech: *.trans.txt → (flac, transcript)"]
    C --> E["Language = directory prefix (ko_kr → ko)"]
    D --> E2["Language = en"]
    E --> F["Metric: CER for ja, ko, zh, ...; WER otherwise (--metric overrides)"]
    E2 --> F
    F --> G["--resume: skip utterances already in --output"]
    G --> H["--sort-by-length: largest files first"]
    H --> I["Check /v1/models on each server (round-robin over several --base-url)"]
    I --> J["ThreadPoolExecutor(--concurrency, default 128)<br/>verbose_json requests"]
    J --> K["As they complete: normalize → split words/characters → add edit distance"]
    K --> L["Write one row per utterance to the --output JSONL"]
    L --> M["Print WER/CER, errors, audio length, elapsed time, RTFx"]
```

- **Normalization**: Whisper's `EnglishTextNormalizer` for English and `BasicTextNormalizer` for other
  languages. Without `openai-whisper`, a basic fallback lowercases the text and turns punctuation and
  symbols into spaces.
- **CER**: spaces are removed from the normalized text and the edit distance is computed over characters.
- **Resume**: `--resume` rewrites the rows of the earlier output file and includes them in the score; a
  file scored with a different metric is refused.
- **Timing**: `RTFx = total audio duration / elapsed time`, excluding utterances skipped by `--resume`.
- **Concurrency**: keep `--concurrency` well above the server's `max_batch_size` so batches fill and
  SenseVoice can group windows of similar length.

## 9. Interactive client (asrback-client)

Implementation: `client.py`. It reads audio file paths at a prompt, sends them to the server and prints
the results.

1. Takes `--task` (transcribe/translate), `--language`, `--prompt` and `--response-format`.
2. Enables readline path completion and accepts quoted paths (from drag and drop).
3. For each path, calls `audio.transcriptions` or `audio.translations` and prints the response and the
   time it took.
4. Prints the status code and message on API errors, or a hint on connection failures, then waits for
   the next path.

## 10. Error handling

| Situation | Handling |
| --- | --- |
| Missing model files, invalid option combinations | `FileNotFoundError` / `ValueError` at startup → the CLI exits with the message |
| Form validation failure (missing required field, ...) | `RequestValidationError` → 400 in the OpenAI `{"error": {...}}` format |
| Unsupported language, format or temperature; empty file; word timestamps requested | `APIError` → 400 |
| Task the model cannot do (translation) | `ValueError` from `transcribe()` → 400 |
| Unknown model id | 404 |
| Exception during batched inference | The same exception reaches every request of that batch (FastAPI answers 500); the worker keeps running |

## 11. Defaults

| Item | Whisper | SenseVoice |
| --- | --- | --- |
| dtype | bfloat16 (CUDA), float32 (CPU) | bfloat16 (CUDA), float32 (CPU); float16 refused |
| `max_batch_size` | 16 | 32 |
| `batch_wait_ms` | 5 | 5 |
| CUDA graphs | on with CUDA (decoder steps) | none |
| `--compile-encoder` | off (optional) | not supported |
| Windows | 30 s windows in order, each conditioned on earlier text | 30 s windows all at once |
| Decoding | greedy or temperature sampling | greedy CTC |
| Server thread limit | max(40, 8 × 16) = 128 | max(40, 8 × 32) = 256 |
| Eval concurrency | 128 | 128 |
