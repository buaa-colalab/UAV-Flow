# UAV-Flow-Eval

Training environment services and single-model closed-loop evaluation for UAV-Flow.
Choose [training services](#2-start-training-environment-services) or
[closed-loop evaluation](#3-closed-loop-evaluation) after setup.

## 0. Installation

Linux x86_64, Python 3.11, NVIDIA graphics and Vulkan are required for real UE simulation.
Run from the repository root. All framework processes use the same environment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install 'pip==24.0' 'setuptools==79.0.1' 'wheel==0.48.0'
python -m pip install '.[dev]'
python -m pip check
python -I -B -c 'import env_server, evaluation.closed_loop_runner; print("IMPORT_OK")'
uav-flow --help
```

## 1. Prepare the UE scene package

Download [Collection_v4_LinuxNoEditor.tar.gz](https://huggingface.co/datasets/wangxiangyu0814/UAV-Flow-Sim-Env/blob/main/Collection_v4_LinuxNoEditor.tar.gz)
from Hugging Face, then extract it from the `UAV-Flow-Eval/` directory:

```bash
mkdir -p external
tar -xzf /path/to/Collection_v4_LinuxNoEditor.tar.gz -C external
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

Set `evaluation.server_url` to a running model API supporting `/reset` and
`/predict`. Configure `ue_package`, `graphics_adapter`, ports and a fresh `output`.

For OpenVLA, start the UAV server as described in
[OpenVLA-UAV](../OpenVLA-UAV/README.md#inference--evaluation) in its own model
environment, then use its URL here. Use `--server-url`, rather than the
QwenOFT-only managed `--checkpoint` launcher. The evaluator accepts OpenVLA's
single waypoint per prediction and feeds back a new observation after each step.

```bash
uav-flow evaluate --settings configs/local.evaluation.toml
```

Results are written under the configured output:

```text
output/
├── trajectories/       # Trajectories, frames and action logs
├── progress_info/      # Process state and logs
└── metrics.json        # Final scores
```

`src/gym_unrealcv/` is maintained directly in this tree, derived from
Gym-UnrealCV / UnrealZoo with UAV-Flow adaptations.
Project-authored code: [Apache-2.0](LICENSE). Project-authored datasets, images,
and documentation: [CC BY-NC 4.0](LICENSE-CONTENT). Code examples follow the code
license. Upstream notices and previously granted permissions remain applicable;
see the scope and provenance notices in [LICENSE](LICENSE). External simulator
assets and model weights have their own terms.
