

## Installation

Follow the OpenVLA installation guide to set up the environment.

```bash
conda create -n openvla python=3.10 -y
conda activate openvla

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y  
pip install -e .

pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```

Finally, download the pretrained [OpenVLA checkpoint](https://huggingface.co/openvla/openvla-7b) that will be used for finetuning.

## Finetuning



You can launch training using the provided shell script:

```bash
bash vla-scripts/finetune_uav.sh
```

Before running the script, open vla-scripts/finetune_uav.sh and change all paths to match your local setup:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune_uav.py \
  --vla_path /path/to/pretrained_openvla_model \
  --data_root_dir /path/to/dataset \
  --run_root_dir /path/to/run_name \
  --adapter_tmp_dir /path/to/run_name/adapter-tmp \
  --lora_rank 32 \
  --batch_size 2 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug False \
  --wandb_project openvla \
  --wandb_entity your-wandb-username \
  --save_steps 20000
```

## Inference & Evaluation

Start the UAV server in the OpenVLA environment, from this directory:

```bash
CUDA_VISIBLE_DEVICES=0 python vla-scripts/openvla_act.py \
  --model-path /path/to/merged_uav_checkpoint \
  --http-port 6888 --unnorm-key sim
```

Use a **merged UAV-finetuned checkpoint**, with four action dimensions and the
matching normalization statistics stored in its `config.json` (`norm_stats`).
`--unnorm-key` selects that checkpoint's dataset statistics; UAV finetuning uses
`sim`. For an existing OpenVLA installation, install the server dependency with
`pip install 'flask>=2.2,<4'`. Flash Attention 2 remains the default; use
`--attn-implementation sdpa` or `eager` if needed.

In a second terminal, activate the separate
[UAV-Flow-Eval environment](../UAV-Flow-Eval/README.md#0-installation), change to
`UAV-Flow-Eval/`, and check the API without starting UE:

```bash
uav-flow evaluate --server-url http://127.0.0.1:6888 \
  --model-name openvla --output ./outputs/openvla_smoke --smoke
```

For a closed-loop run:

```bash
cp configs/example.toml configs/local.openvla.toml
# Set evaluation.server_url, ue_package, graphics_adapter and a fresh output.
# Set evaluation.ue_user if launching as root.
uav-flow evaluate --settings configs/local.openvla.toml --model-name openvla
```

Use `--server-url` for OpenVLA; the evaluator's managed `--checkpoint` launcher
is for QwenOFT servers. The two Python environments communicate through HTTP.

`openvla_act.py` exposes stateless `POST /reset` and `POST /predict`. Requests
contain a base64 RGB image, `instr` and `proprio=[x,y,z,yaw_degrees]` in the
episode's first-frame coordinate system. It returns one waypoint (`action` has
shape `[1,4]`) in that same coordinate system, with yaw in radians. The evaluator
executes it, captures the next image and requests another prediction. The legacy
state prompt, action decoding and local-to-episode coordinate conversion are
preserved. `deploy.py` provides the upstream robot `/act` API and is not the UAV
evaluation entry point.

Offline interface and closed-loop regression checks (no weights or UE required),
run from the parent `UAV-Flow/` directory in the evaluation environment:

```bash
python -m unittest discover -s OpenVLA-UAV/tests -v
```

UAV-Flow-authored code follows [Apache-2.0](../LICENSE); upstream OpenVLA code
retains its [MIT license](LICENSE). Project-authored documentation follows
[CC BY-NC 4.0](../LICENSE-CONTENT). See [licensing scope](../LICENSING.md).
