#!/usr/bin/env bash
# Download the Gemma 4 (instruction-tuned) files from Hugging Face.
#
# Usage: download_gemma4_hf.sh [MODEL ...]
#   MODEL: E2B, E4B (default: E2B E4B)
#   Files go to $MODEL_DIR/gemma-4-<model>-it (e.g. models/gemma-4-e2b-it).
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-models}"
REVISION="${REVISION:-main}"
FILES=(
  config.json
  generation_config.json
  processor_config.json
  chat_template.jinja
  tokenizer.json
  tokenizer_config.json
  model.safetensors
)
MODELS=()

for arg in "$@"; do
  case "$arg" in
    E2B|E4B) MODELS+=("$arg") ;;
    *)
      echo "Error: unsupported argument '$arg' (expected E2B or E4B)." >&2
      exit 1
      ;;
  esac
done
if (( ${#MODELS[@]} == 0 )); then
  MODELS=(E2B E4B)
fi

if ! command -v wget >/dev/null 2>&1; then
  echo "Error: install wget to download models." >&2
  exit 1
fi

for model in "${MODELS[@]}"; do
  repo="google/gemma-4-$model-it"
  output_dir="$MODEL_DIR/gemma-4-${model,,}-it"
  mkdir -p "$output_dir"

  echo "Downloading $repo@$REVISION to $output_dir..."
  for file in "${FILES[@]}"; do
    wget -c --tries=3 \
      "https://huggingface.co/$repo/resolve/$REVISION/$file" \
      -O "$output_dir/$file"
  done
  echo "Ready: $output_dir"
done

echo "All requested Gemma 4 models are ready in $MODEL_DIR."
