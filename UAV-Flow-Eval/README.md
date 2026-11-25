# UAV-Flow-Eval

Training environment services and single-model closed-loop evaluation for UAV-Flow.
Choose [training services](#2-start-training-environment-services) or
[closed-loop evaluation](#3-closed-loop-evaluation) after setup.

## 0. Installation

Linux x86_64, Python 3.11, NVIDIA graphics and Vulkan are required for real UE simulation.
Start at the UAV-Flow repository root. Evaluation and simulator services share
this environment; model servers use their own environment.

```bash
cd UAV-Flow-Eval
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install 'pip==24.0' 'setuptools==79.0.1' 'wheel==0.48.0'
python -m pip install .
python -I -B -c 'import env_server, evaluation.closed_loop_runner; print("IMPORT_OK")'
# uav-flow --help
```

## 1. Prepare the UE scene package

Download [Collection_v4_LinuxNoEditor.tar.gz](https://huggingface.co/datasets/wangxiangyu0814/UAV-Flow-Sim-Env/blob/main/Collection_v4_LinuxNoEditor.tar.gz)
from Hugging Face, then extract it from the `UAV-Flow-Eval/` directory:

```bash
mkdir -p Env && tar -xzf /path/to/Collection_v4_LinuxNoEditor.tar.gz -C Env
```

**UE scene package** — set `--ue-package` / `evaluation.ue_package` (or
`training.package`) to the extracted directory containing `Collection/` and
`Engine/`, for example:

```text
Collection_v4_LinuxNoEditor/
├── Collection/
│   └── Binaries/Linux/Collection
└── Engine/
```

UE runs as an ordinary Linux user. Root launchers set `training.user` or
`evaluation.ue_user`; that account needs access to the environment, assets and output.

Use Vulkan adapter indices for `graphics_adapter` and training `gpus`.

## 2. Start training environment services

```bash
cp configs/example.toml configs/local.training.toml
# Edit training.package, output, gpus, ports and user.
uav-flow train-env start --settings configs/local.training.toml --dry-run
uav-flow train-env start --settings configs/local.training.toml
```

Use `uav-flow train-env --help` for status and stop commands.
The RL training algorithm is supplied separately.

## 3. Closed-loop evaluation

```bash
cp configs/example.toml configs/local.evaluation.toml
```

Start a model API using the [FlowVLA server](../flowVLA/README.md#3-inference)
or [OpenVLA-UAV server](../OpenVLA-UAV/README.md#inference--evaluation) in its own
model environment.

Update the existing `[evaluation]` table in `configs/local.evaluation.toml`:

```toml
[evaluation]
server_url = "http://127.0.0.1:6070"  # Match the running model server.
model_name = "flowvla"
ue_package = "../Env/Collection_v4_LinuxNoEditor"
graphics_adapter = 0
output = "../outputs/flowvla_run_01"
# ue_user = "simulator"             # Set an ordinary account when invoked as root.
```

Use the actual extracted scene directory for `ue_package` and a fresh output
directory for each run. TOML paths are relative to the settings file;
CLI arguments override TOML values. When the model runs on another host,
replace `127.0.0.1` with its reachable address and port.

The evaluator accepts FlowVLA action chunks and OpenVLA's single waypoint,
requesting a new observation after the returned actions are executed.

```bash
uav-flow evaluate --settings configs/local.evaluation.toml
```

Results from
the actual evaluation are written under the configured output:

```text
output/
├── trajectories/       # Trajectories, frames and action logs
├── progress_info/      # Process state and logs
└── metrics.json        # Final scores
```

`src/gym_unrealcv/` is maintained directly in this tree, derived from
Gym-UnrealCV / UnrealZoo with UAV-Flow adaptations.
Project-authored code: [Apache-2.0](LICENSE). Upstream notices and previously granted permissions remain applicable;
see the scope and provenance notices in [LICENSE](LICENSE).
