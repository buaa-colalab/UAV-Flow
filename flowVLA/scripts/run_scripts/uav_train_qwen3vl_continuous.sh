#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG_YAML="${CONFIG_YAML:-starVLA/config/training/qwen3vl_continuous.yaml}"
exec bash "$script_dir/uav_train.sh" "$@"
