

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

If you want to start an OpenVLA server and run evaluation in **UAV-Flow-Eval**, simply run:

```bash
python vla-scripts/openvla_act.py
```

Before running, open openvla_act.py and update the cfg dictionary with your local settings:

```bash
cfg = {
    "gpu_id": 0,                                    
    "model_path": "/path/to/your/finetuned_model",  
    "http_port": 5007,                            
    "unnorm_key": "sim",                          
    "do_sample": False                       
}
```