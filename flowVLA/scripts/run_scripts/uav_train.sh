#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
: "${CONFIG_YAML:?Select a model YAML or use a model training launcher}"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
overrides=()
for setting in \
  BASE_VLM:framework.qwenvl.base_vlm \
  DATA_ROOT_DIR:datasets.vla_data.data_root_dir \
  BATCH_SIZE:datasets.vla_data.per_device_batch_size \
  RUN_ROOT_DIR:run_root_dir \
  STATS_PATH:datasets.vla_data.normalization_stats_path \
  FAST_TOKENIZER_NAME:framework.action_model.fast_tokenizer_name \
  FAST_ACTION_TOKENS_FILE:framework.action_model.fast_action_tokens_file \
  TRAIN_FAST_ACTION_TOKEN_EMBEDDINGS:framework.action_model.train_fast_action_token_embeddings; do
  variable="${setting%%:*}"
  [[ -z "${!variable:-}" ]] || overrides+=(--"${setting#*:}" "${!variable}")
done
exec "${PYTHON_BIN:-python}" -m accelerate.commands.launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES:-8}" --main_process_port "${MAIN_PROCESS_PORT:-29535}" \
  starVLA/training/train_starvla_vla.py \
  --config_yaml "$CONFIG_YAML" \
  --run_id "${RUN_ID:-$(basename "$CONFIG_YAML" .yaml)_$(date +%Y%m%d_%H%M%S)}" "${overrides[@]}" "$@"
