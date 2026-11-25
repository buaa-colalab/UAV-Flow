#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SERVER_MODULE=starVLA.model.framework.ServerQwenOFT_dis
export EXPECTED_ALL_LAYERS=1
exec bash "$script_dir/uav_serve.sh" "$@"
