#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
: "${RUN_DIR:?Set RUN_DIR to the training output directory}"
: "${SERVER_MODULE:?Use a model server launcher}"
export RUN_DIR
export CONFIG_YAML="${CONFIG_YAML:-$RUN_DIR/config.yaml}"
: "${STATS_PATH:?Set STATS_PATH to the statistics JSON used for training}"
export STATS_PATH
if [[ -z "${CHECKPOINT_PATH:-}" ]]; then
  if [[ -f "$RUN_DIR/final_model/pytorch_model.pt" ]]; then
    CHECKPOINT_PATH="$RUN_DIR/final_model/pytorch_model.pt"
  elif [[ -d "$RUN_DIR/checkpoints" ]]; then
    CHECKPOINT_PATH="$(find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -name 'steps_*_pytorch_model.pt' -print | sort -V | tail -n 1)"
  fi
fi
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH or provide a final/step checkpoint under RUN_DIR}"
export CHECKPOINT_PATH
for required in "$CHECKPOINT_PATH" "$CONFIG_YAML" "$STATS_PATH"; do
  [[ -f "$required" ]] || { echo "Required file does not exist: $required" >&2; exit 1; }
done
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GPU_ID="${GPU_ID:-0}"
export HTTP_PORT="${HTTP_PORT:-6070}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec "${PYTHON_BIN:-python}" -m "$SERVER_MODULE" "$@"
