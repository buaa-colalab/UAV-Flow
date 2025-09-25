"""
finetune_simple.py

LoRA-efficient finetuning of the OpenVLA model using SimpleVLADataset.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
import torch.distributed as dist
from accelerate import PartialState
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader

import wandb
import sys
import logging
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.uav_dataset import SimpleVLADataset
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
import tqdm
import json
import numpy as np
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoImageProcessor, AutoConfig, BitsAndBytesConfig

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def update_model_norm_stats(model, dataset, dataset_key="sim"):
    """
    Update the model's norm_stats with the dataset's action statistics.
    
    Args:
        model: The OpenVLA model
        dataset: SimpleVLADataset instance with computed action statistics
        dataset_key: Key to use in norm_stats (default: "sim")
    """
    # Get action statistics from dataset
    action_mean = dataset.action_mean
    action_std = dataset.action_std
    action_min = dataset.action_min
    action_max = dataset.action_max
    
    # Create norm_stats entry in the format expected by OpenVLA
    norm_stats_entry = {
        "action": {
            "mean": action_mean.tolist(),
            "std": action_std.tolist(),
            "min": action_min.tolist(),
            "max": action_max.tolist(),
            "q01": action_min.tolist(),  # 1st percentile as min
            "q99": action_max.tolist()  # 99th percentile as max
        }
    }
    
    # Update model's norm_stats
    if not hasattr(model, 'norm_stats') or model.norm_stats is None:
        model.norm_stats = {}
    
    model.norm_stats[dataset_key] = norm_stats_entry
    
    # Also update the config if it exists
    if hasattr(model, 'config') and hasattr(model.config, 'norm_stats'):
        if model.config.norm_stats is None:
            model.config.norm_stats = {}
        model.config.norm_stats[dataset_key] = norm_stats_entry
    
    logger.info(f"Updated norm_stats with '{dataset_key}' dataset statistics:")
    logger.info(f"  Mean: {action_mean}")
    logger.info(f"  Std: {action_std}")
    logger.info(f"  Min: {action_min}")
    logger.info(f"  Max: {action_max}")

@dataclass
class FinetuneSimpleConfig:
    vla_path: str = "openvla/openvla-7b"
    data_root_dir: Path = Path("datasets/your_dataset")
    run_root_dir: Path = Path("runs")
    batch_size: int = 16
    max_steps: int = 200_000
    save_steps: int = 5000
    learning_rate: float = 5e-4
    grad_accumulation_steps: int = 1
    image_aug: bool = False
    save_latest_checkpoint_only: bool = True
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    use_quantization: bool = False
    wandb_project: str = "openvla"
    wandb_entity: str = "your-entity"
    run_id_note: Optional[str] = None
    adapter_tmp_dir: Path = Path("adapter-tmp")

@draccus.wrap()
def finetune_simple(cfg: FinetuneSimpleConfig) -> None:
    logger.info("Initializing finetune_simple...")
    assert torch.cuda.is_available(), "At least one GPU is required."
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    run_dir = cfg.run_root_dir / exp_id
    adapter_dir = cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(adapter_dir, exist_ok=True)
    logger.info(f"Artifacts and logs will be saved to: {run_dir}")
    logger.info(f"LoRA adapter checkpoints will be saved to: {adapter_dir}")

    # Register OpenVLA to Hugging Face AutoClasses
    logger.info("Registering OpenVLA with Hugging Face AutoClasses...")
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    quantization_config = None
    if cfg.use_quantization:
        logger.info("Enabling 4-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    logger.info("Loading processor and model...")
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if cfg.use_quantization:
        logger.info("Preparing model for k-bit training...")
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    if cfg.use_lora:
        logger.info("Wrapping model with LoRA adapters...")
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    logger.info("Building DistributedDataParallel (DDP)...")
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)
    optimizer = AdamW([p for p in vla.parameters() if p.requires_grad], lr=cfg.learning_rate)

    logger.info("Initializing SimpleVLADataset...")
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    dataset = SimpleVLADataset(
        data_path=cfg.data_root_dir,
        image_transform=processor.image_processor,
        tokenizer=processor.tokenizer,
        action_tokenizer=action_tokenizer,
        prompt_builder_fn=PurePromptBuilder,
        is_train=True,
        image_aug=cfg.image_aug,
    )
    logger.info(f"Number of samples: {len(dataset)}")
    
    # Update model's norm_stats with dataset statistics
    logger.info("Updating model norm_stats with dataset statistics...")
    update_model_norm_stats(vla.module, dataset, dataset_key="sim")
    logger.info("Creating collator and dataloader...")
    collator = PaddedCollatorForActionPrediction(
        model_max_length=processor.tokenizer.model_max_length,
        pad_token_id=processor.tokenizer.pad_token_id,
        padding_side="right"
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=collator,
        drop_last=True,
        num_workers=4
    )

    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                )
                loss = output.loss

            normalized_loss = loss / cfg.grad_accumulation_steps
            normalized_loss.backward()

            # Compute action accuracy and L1 loss (similar to finetune.py)
            action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float() if mask.sum() > 0 else torch.tensor(0.0)
            # L1 loss
            try:
                continuous_actions_pred = torch.tensor(action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy()))
                continuous_actions_gt = torch.tensor(action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy()))
                
                action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
            except Exception:
                action_l1_loss = torch.tensor(0.0)

            recent_losses = [loss.item()]
            recent_action_accuracies = [action_accuracy.item()]
            recent_l1_losses = [action_l1_loss.item()]

            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
            smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)

            if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                logger.info(f"[METRIC] step={gradient_step_idx} | train_loss={smoothened_loss:.6f} | action_accuracy={smoothened_action_accuracy:.6f} | l1_loss={smoothened_l1_loss:.6f}")

            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                progress.update()

            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                if distributed_state.is_main_process:
                    logger.info(f"Saving model checkpoint for step {gradient_step_idx}...")
                    save_dir = adapter_dir if cfg.use_lora else run_dir
                    
                    # Ensure norm_stats is up to date before saving
                    update_model_norm_stats(vla.module, dataset, dataset_key="sim")
                    
                    processor.save_pretrained(run_dir)
                    vla.module.save_pretrained(save_dir)

                    # Merge LoRA weights and save to run_dir
                    if cfg.use_lora:
                        base_vla = AutoModelForVision2Seq.from_pretrained(
                            cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                        )
                        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
                        merged_vla = merged_vla.merge_and_unload()
                        
                        # Update norm_stats for merged model
                        update_model_norm_stats(merged_vla, dataset, dataset_key="sim")
                        
                        if cfg.save_latest_checkpoint_only:
                            merged_vla.save_pretrained(run_dir)
                            logger.info(f"Checkpoint saved at: {run_dir}")
                        else:
                            checkpoint_dir = Path(str(run_dir) + f"--{gradient_step_idx}_chkpt")
                            os.makedirs(checkpoint_dir, exist_ok=True)
                            processor.save_pretrained(checkpoint_dir)
                            merged_vla.save_pretrained(checkpoint_dir)
                            logger.info(f"Checkpoint saved at: {checkpoint_dir}")
                dist.barrier()

            if gradient_step_idx == cfg.max_steps:
                logger.info(f"Reached max_steps={cfg.max_steps}. Stopping training.")
                break

    if distributed_state.is_main_process:
        logger.info("Training complete.")
    dist.barrier()
    dist.destroy_process_group()
    logger.info("Process group destroyed. Exiting.")

if __name__ == "__main__":
    finetune_simple()
