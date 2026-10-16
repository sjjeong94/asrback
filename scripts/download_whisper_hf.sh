#!/usr/bin/env bash
# Download the Whisper files needed by asrback from Hugging Face.
#
# Usage: download_whisper_hf.sh [--tokenizer-only] [tiny|base|small ...]
#   --tokenizer-only  Download only config/tokenizer files (skip model.safetensors).
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-models}"
REVISION="${REVISION:-main}"
FILES=(config.json tokenizer.json tokenizer_config.json model.safetensors)
MODELS=()

for arg in "$@"; do
  case "$arg" in
    --tokenizer-only) FILES=(config.json tokenizer.json tokenizer_config.json) ;;
    tiny|base|small) MODELS+=("$arg") ;;
    *)
      echo "Error: unsupported argument '$arg' (expected --tokenizer-only, tiny, base, or small)." >&2
      exit 1
      ;;
  esac
done
if (( ${#MODELS[@]} == 0 )); then
  MODELS=(tiny base small)
fi

if ! command -v wget >/dev/null 2>&1; then
  echo "Error: install wget to download models." >&2
  exit 1
fi

for model in "${MODELS[@]}"; do
  repo="openai/whisper-$model"
  output_dir="$MODEL_DIR/whisper-$model"
  mkdir -p "$output_dir"

  echo "Downloading $repo@$REVISION to $output_dir..."
  for file in "${FILES[@]}"; do
    wget -c --tries=3 \
      "https://huggingface.co/$repo/resolve/$REVISION/$file" \
      -O "$output_dir/$file"
  done
  echo "Ready: $output_dir"
done

echo "All requested Whisper models are ready in $MODEL_DIR."
