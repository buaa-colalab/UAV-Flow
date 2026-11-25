#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG_YAML="${CONFIG_YAML:-starVLA/config/training/qwen3vl_discrete_alllayer.yaml}"
: "${PRETRAINED_CHECKPOINT:?Set PRETRAINED_CHECKPOINT to the discrete baseline checkpoint}"
exec bash "$script_dir/uav_train.sh" --trainer.pretrained_checkpoint "$PRETRAINED_CHECKPOINT" --trainer.reload_modules qwen_vl_interface,action_model "$@"
