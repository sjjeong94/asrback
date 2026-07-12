#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-models/huggingface}"
REVISION="${REVISION:-main}"
if (( $# > 0 )); then
  MODELS=("$@")
else
  MODELS=(tiny base small)
fi

INCLUDE_PATTERNS=(
  "*.json"
  "*.txt"
  "model.safetensors"
)

download_model() {
  local model="$1"
  local repo="openai/whisper-$model"
  local output_dir="$MODEL_DIR/whisper-$model"

  case "$model" in
    tiny|base|small) ;;
    *)
      echo "Error: unsupported model '$model' (expected tiny, base, or small)." >&2
      return 1
      ;;
  esac

  mkdir -p "$output_dir"
  echo "Downloading $repo@$REVISION to $output_dir..."

  if command -v hf >/dev/null 2>&1; then
    hf download "$repo" \
      --revision "$REVISION" \
      --local-dir "$output_dir" \
      --include "${INCLUDE_PATTERNS[@]}"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$repo" \
      --revision "$REVISION" \
      --local-dir "$output_dir" \
      --include "${INCLUDE_PATTERNS[@]}"
  elif python3 -c 'import huggingface_hub' >/dev/null 2>&1; then
    python3 -m huggingface_hub.commands.huggingface_cli download "$repo" \
      --revision "$REVISION" \
      --local-dir "$output_dir" \
      --include "${INCLUDE_PATTERNS[@]}"
  else
    echo "Error: Hugging Face Hub is not installed." >&2
    echo "Install it with: python3 -m pip install -U huggingface_hub" >&2
    return 1
  fi

  echo "Ready: $output_dir"
}

mkdir -p "$MODEL_DIR"
for model in "${MODELS[@]}"; do
  download_model "$model"
done

echo "All requested Whisper models are ready in $MODEL_DIR."
