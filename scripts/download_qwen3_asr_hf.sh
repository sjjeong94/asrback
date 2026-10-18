#!/usr/bin/env bash
# Download the Qwen3-ASR files needed for inference from Hugging Face.
#
# Usage: download_qwen3_asr_hf.sh [MODEL ...]
#   MODEL: 0.6B, 1.7B (default: 0.6B 1.7B)
#   Files go to $MODEL_DIR/qwen3-asr-<model> (e.g. models/qwen3-asr-0.6b).
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-models}"
REVISION="${REVISION:-main}"
COMMON_FILES=(
  config.json
  generation_config.json
  preprocessor_config.json
  chat_template.json
  tokenizer_config.json
  vocab.json
  merges.txt
)
MODELS=()

for arg in "$@"; do
  case "$arg" in
    0.6B|1.7B) MODELS+=("$arg") ;;
    *)
      echo "Error: unsupported argument '$arg' (expected 0.6B or 1.7B)." >&2
      exit 1
      ;;
  esac
done
if (( ${#MODELS[@]} == 0 )); then
  MODELS=(0.6B 1.7B)
fi

if ! command -v wget >/dev/null 2>&1; then
  echo "Error: install wget to download models." >&2
  exit 1
fi

for model in "${MODELS[@]}"; do
  repo="Qwen/Qwen3-ASR-$model"
  output_dir="$MODEL_DIR/qwen3-asr-${model,,}"
  case "$model" in
    0.6B) weight_files=(model.safetensors) ;;
    1.7B)
      weight_files=(
        model.safetensors.index.json
        model-00001-of-00002.safetensors
        model-00002-of-00002.safetensors
      )
      ;;
  esac
  mkdir -p "$output_dir"

  echo "Downloading $repo@$REVISION to $output_dir..."
  for file in "${COMMON_FILES[@]}" "${weight_files[@]}"; do
    wget -c --tries=3 \
      "https://huggingface.co/$repo/resolve/$REVISION/$file" \
      -O "$output_dir/$file"
  done
  echo "Ready: $output_dir"
done

echo "All requested Qwen3-ASR models are ready in $MODEL_DIR."
