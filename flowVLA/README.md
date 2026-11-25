# FlowVLA

Train and serve UAVFlow policies with Qwen3-VL, built on [StarVLA](https://github.com/starVLA/starVLA).

## 1. Installation

Run from the repository root. Tested on Linux with Python 3.12 and CUDA 12.6.

```bash
conda create -n flowvla python=3.12 -y
conda activate flowvla
pip install -r requirements.txt
pip install -e .
```

## 2. Training

Download the base model:

```bash
hf download Qwen/Qwen3-VL-2B-Instruct --local-dir /path/to/models/Qwen3-VL-2B-Instruct
```

Prepare the UAVFlow dataset. Each episode directory contains RGB frames and `log.json`.
Set `DATA_ROOT_DIR` to the directory containing these episodes:

```bash
BASE_VLM=/path/to/models/Qwen3-VL-2B-Instruct \
DATA_ROOT_DIR=/path/to/UAVFlow \
RUN_ROOT_DIR=./results RUN_ID=flowvla \
bash scripts/run_scripts/uav_train_qwen3vl_discrete.sh
```

The launcher uses 8 GPUs by default; set `NUM_PROCESSES` to override.
`BASE_VLM`, `DATA_ROOT_DIR`, `BATCH_SIZE`, `RUN_ROOT_DIR`, and `RUN_ID` are shared by all training launchers.
CLI overrides can be appended to the script and take precedence over environment variables and YAML defaults.
Checkpoints are saved under `results/flowvla/checkpoints/`.

Choose a model below; YAML files are in `starVLA/config/training/` and scripts are in `scripts/run_scripts/`.

| Backbone / head | YAML | Training script | Server script |
|---|---|---|---|
| Qwen3-VL / discrete 256-bin | `qwen3vl_discrete.yaml` | `uav_train_qwen3vl_discrete.sh` | `start_qwen3vl_discrete_server.sh` |
| Qwen3-VL / discrete 256-bin, all-layer | `qwen3vl_discrete_alllayer.yaml` | `uav_train_qwen3vl_discrete_alllayer.sh` | `start_qwen3vl_discrete_alllayer_server.sh` |
| Qwen3-VL / continuous, L1 loss | `qwen3vl_continuous.yaml` | `uav_train_qwen3vl_continuous.sh` | `start_qwen3vl_continuous_server.sh` |
| Qwen3-VL / FAST | `qwen3vl_fast.yaml` | `uav_train_qwen3vl_fast.sh` | `start_qwen3vl_fast_server.sh` |

Dataset selection is independent of the model: use the same launcher for mixed, sim-only, or another UAVFlow dataset by changing `DATA_ROOT_DIR`.
The discrete all-layer launcher additionally requires `PRETRAINED_CHECKPOINT` pointing to the discrete baseline; it initializes the backbone and action head from that checkpoint.

The loader computes p1/p99 statistics from the selected dataset. Set `STATS_PATH` to use a frozen statistics file instead.
Statistics contain four-axis `action_min`, `action_max`, `action_mean`, and `action_std` values in local dx/dy/dz and yaw radians.
Each training run saves its normalization statistics alongside the configuration.

## 3. Inference

Pretrained checkpoints:

| Setting | Training steps | Download |
|---|---|---|
| `qwen3vl_discrete` | 80,000 | [Hugging Face](https://huggingface.co/datasets/wangxiangyu0814/UAV-Flow-flowVLA/blob/main/flowvla_pretrain_mixdata_dis_head.pth) |
| `qwen3vl_discrete` + TGRPO | — | [Hugging Face](https://huggingface.co/datasets/wangxiangyu0814/UAV-Flow-flowVLA/blob/main/flowvla_tgrpo_mixdata_dis_head.pth) |

The release also includes the [mixed-data statistics JSON](https://huggingface.co/datasets/wangxiangyu0814/UAV-Flow-flowVLA/blob/main/uavflow_mixed_p1p99.json).

Use the checkpoint, its run configuration, and a statistics JSON matching the training normalization.
For a discrete-head run:

```bash
RUN_DIR=./results/flowvla \
STATS_PATH=/path/to/statistics.json \
CUDA_VISIBLE_DEVICES=0 HTTP_PORT=6070 \
bash scripts/run_scripts/start_qwen3vl_discrete_server.sh
```

The server selects the final checkpoint, or the latest numbered step checkpoint. Set `CHECKPOINT_PATH` to select one explicitly.
Use the matching server in the table for other action heads. Set `BASE_VLM` if the base model directory has moved.
Set `STATS_PATH` to the statistics JSON used for training. It can have any filename and live outside `RUN_DIR`.
For the original mixed-data checkpoints, use `starVLA/config/dataset_statistics/uavflow_mixed_p1p99.json`; for sim-only checkpoints, use `starVLA/config/dataset_statistics/uavflow_sim_p1p99.json`.

Send a prediction request from another terminal in the same environment:

```python
import base64
from pathlib import Path
import requests

url = "http://127.0.0.1:6070"
requests.post(url + "/reset", json={}).raise_for_status()
response = requests.post(url + "/predict", json={
    "image": base64.b64encode(Path("/path/to/frame.png").read_bytes()).decode(),
    "instr": "fly towards the red building",
    "proprio": [0, 0, 0, 0],  # Current [x, y, z, yaw_degrees].
})
response.raise_for_status()
print(response.json()["action"])  # Ten global [x, y, z, yaw_radians] waypoints.
```

Call `/reset` before starting a new episode.

> UAV-Flow-authored code follows [Apache-2.0](../LICENSE); upstream StarVLA
> code retains its [MIT license](LICENSE). 
