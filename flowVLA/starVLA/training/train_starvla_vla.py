# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].


"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

import argparse
import json
import os
import hashlib
import math
import warnings
from pathlib import Path
from typing import Tuple
from torch.utils.data import DataLoader
import numpy as np

import torch
import torch.distributed as dist
import wandb
import yaml
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import GradientAccumulationPlugin, set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import get_scheduler

from starVLA.dataloader import build_dataloader
from starVLA.training.trainer_utils.trainer_tools import (
    TrainerUtils,
    build_param_lr_groups,
    normalize_dotlist_args,
)
from starVLA.model.framework import build_framework


os.environ["TOKENIZERS_PARALLELISM"] = "false"


logger = get_logger(__name__)


def validate_training_config(cfg):
    """Reject unsupported settings before CUDA/model/output initialization."""
    trainer = cfg.trainer
    for key in ("max_train_steps", "save_interval", "logging_frequency", "gradient_accumulation_steps"):
        value = trainer[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"trainer.{key} must be a positive integer")
    warmup = trainer.num_warmup_steps
    if isinstance(warmup, bool) or not isinstance(warmup, int) or not 0 <= warmup <= trainer.max_train_steps:
        raise ValueError("num_warmup_steps must be between zero and max_train_steps")
    clipping = trainer.get("gradient_clipping")
    if clipping is not None and (
        isinstance(clipping, bool) or not math.isfinite(float(clipping)) or float(clipping) < 0
    ):
        raise ValueError("gradient_clipping must be finite and nonnegative, or null to disable")
    if trainer.optimizer.get("name", "AdamW") != "AdamW":
        raise ValueError("The VLA trainer supports optimizer.name=AdamW only")
    if (
        trainer.get("is_resume", False)
        or trainer.get("resume_step") is not None
        or trainer.get("resume_epoch") is not None
        or cfg.get("resume_from_checkpoint")
        or trainer.get("resume_from_checkpoint")
    ):
        raise ValueError(
            "Exact resume is unsupported: checkpoints contain model weights only. "
            "Use trainer.pretrained_checkpoint for a weights-only warm start."
        )
    trackers = cfg.get("trackers", ["wandb"])
    if not (isinstance(trackers, list) or OmegaConf.is_list(trackers)) or any(
        t not in ("jsonl", "wandb") for t in trackers
    ):
        raise ValueError("trackers must be a list containing jsonl and/or wandb")
    if cfg.framework.name == "QwenOFT" and cfg.framework.action_model.get("action_head_type") != "discrete":
        raise ValueError("QwenOFT uses categorical cross entropy and requires action_head_type=discrete")
    if (
        cfg.framework.name == "QwenOFT-L1"
        and cfg.framework.action_model.get("action_head_type") not in {None, "continuous"}
    ):
        raise ValueError("Continuous regression requires action_head_type=continuous; legacy configs may omit it")
    if cfg.framework.name == "QwenOFT-L1" and cfg.framework.get("use_all_layer_hidden", False):
        raise ValueError("Continuous all-layer is unsupported; use the discrete all-layer recipe")
    if cfg.datasets.vla_data.dataset_py == "uav_datasets":
        action = cfg.framework.action_model
        if (action.action_dim, action.past_action_window_size, action.future_action_window_size) != (4, 0, 9):
            raise ValueError("uav_datasets emits exactly 10 future actions with 4 dimensions and no past window")
        if cfg.datasets.vla_data.get("image_size") is not None:
            raise ValueError(
                "uav_datasets reads native images; image_size was an inference-only setting. "
                "Use datasets with matching native training and inference resolution."
            )
        for key, minimum in (("per_device_batch_size", 1), ("num_workers", 0)):
            value = cfg.datasets.vla_data.get(key, 4)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"datasets.vla_data.{key} must be an integer >= {minimum}")
    inactive = [
        "framework.dino",
        "framework.layer_qformer",
        "framework.qwenvl.vl_hidden_dim",
        "trainer.epochs",
        "trainer.eval_interval",
        "trainer.loss_scale",
        "trainer.warmup_ratio",
        "trainer.max_grad_norm",
        "trainer.weight_decay",
        "trainer.repeated_diffusion_steps",
        "trainer.enable_mixed_precision_training",
        "datasets.vla_data.load_all_data_for_training",
    ]
    if cfg.framework.name == "QwenOFT":
        inactive += [
            "framework.action_model." + k
            for k in (
                "use_l1_loss",
                "l1_weight",
                "action_model_type",
                "use_ema",
                "repeated_diffusion_steps",
                "use_gaussian",
                "min_std",
                "max_std",
                "use_nll_loss",
                "std_reg_weight",
            )
        ]
    for key in inactive:
        if OmegaConf.select(cfg, key) is not None:
            warnings.warn(f"{key} is unused by this VLA training route", UserWarning, stacklevel=2)


def create_accelerator(cfg):
    """Make training YAML authoritative for accumulation and clipping."""
    plugin = DeepSpeedPlugin()
    steps = int(cfg.trainer.gradient_accumulation_steps)
    clipping = cfg.trainer.get("gradient_clipping")
    plugin.deepspeed_config["gradient_accumulation_steps"] = steps
    plugin.deepspeed_config["gradient_clipping"] = 0.0 if clipping is None else float(clipping)
    plugin.gradient_accumulation_steps = steps
    plugin.gradient_clipping = plugin.deepspeed_config["gradient_clipping"]
    accumulation = GradientAccumulationPlugin(num_steps=steps, sync_with_dataloader=False)
    instance = Accelerator(deepspeed_plugin=plugin, gradient_accumulation_plugin=accumulation)
    instance.dataloader_config.dispatch_batches = False
    return instance


def setup_directories(cfg) -> Path:
    """create output directory and save config"""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

        OmegaConf.save(cfg, output_dir / "config.yaml")
        with open(output_dir / "config.yaml", "r") as f_yaml, open(output_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> Tuple[DataLoader, DataLoader]:
    """prepare training data"""

    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    vlm_train_dataloader = None

    if cfg.datasets.vla_data.dataset_py == "uav_datasets" and accelerator.is_main_process:
        dataset = vla_train_dataloader.dataset
        stats = {
            "action_min": dataset.action_min.tolist(),
            "action_max": dataset.action_max.tolist(),
            "action_mean": dataset.action_mean.tolist(),
            "action_std": dataset.action_std.tolist(),
            "action_order": ["local_dx", "local_dy", "local_dz", "local_dyaw_radians"],
            "percentiles": [1, 99],
            "data_root": cfg.datasets.vla_data.data_root_dir,
        }
        if dataset.action_codec is not None:
            stats["action_codec_version"] = dataset.action_codec.manifest["version"]
            stats["action_codec_sha256"] = dataset.action_codec.sha256
            stats["source_percentiles"] = [1, 99]
            stats["normalization_bounds_policy"] = dataset.action_codec.manifest["bounds_policy"]
            stats["source_p1"] = dataset.action_codec.manifest["source_p1"]
            stats["source_p99"] = dataset.action_codec.manifest["source_p99"]
            stats["num_episodes"] = len(dataset.episodes)
            stats["num_samples_with_repeats"] = len(dataset)
            stats["num_actions"] = sum(len(e["actions"]) for e in dataset.episodes)
            members = [
                {"name": p.name, "source": str(p.resolve())}
                for p in sorted(Path(cfg.datasets.vla_data.data_root_dir).iterdir())
                if p.is_dir()
            ]
            members_json = json.dumps(members, sort_keys=True)
            stats["episode_membership_sha256"] = hashlib.sha256(members_json.encode()).hexdigest()
            (Path(output_dir) / "dataset_membership.json").write_text(members_json)
        with open(Path(output_dir) / "uav_action_statistics.json", "w") as handle:
            json.dump(stats, handle, indent=2)

    accelerator.dataloader_config.dispatch_batches = False

    if dist.is_initialized():
        dist.barrier()

    return vla_train_dataloader, vlm_train_dataloader


def _build_optimizer_and_scheduler(params, cfg, learning_rate):
    optimizer = torch.optim.AdamW(
        params,
        lr=learning_rate,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )
    scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,
    )
    return optimizer, scheduler


def setup_optimizer_and_scheduler_lora(model, cfg):
    """Use one shared rate for every trainable LoRA, query and action-head parameter."""
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters found. Check LoRA config or model wrapping.")
    if dist.is_initialized() and dist.get_rank() == 0:
        logger.info(f"Optimizer will update {sum(p.numel() for p in params)} parameters (trainable).")
    return _build_optimizer_and_scheduler(params, cfg, cfg.trainer.learning_rate.lora)


def setup_optimizer_and_scheduler(model, cfg):
    """Use configured module rates for full-parameter training."""
    groups = build_param_lr_groups(model=model, cfg=cfg)
    if dist.is_initialized() and dist.get_rank() == 0:
        for group in groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")
    return _build_optimizer_and_scheduler(groups, cfg, cfg.trainer.learning_rate.base)


class VLAMTrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, vlm_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.vlm_train_dataloader = vlm_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator, self.model, self.optimizer, self.vla_train_dataloader
        )

        self.total_batch_size = self._calculate_total_batch_size()
        if self.accelerator.is_main_process:
            plugin = self.accelerator.state.deepspeed_plugin
            runtime = {
                "num_processes": self.accelerator.num_processes,
                "gradient_accumulation_steps": self.accelerator.gradient_accumulation_steps,
                "total_batch_size": self.total_batch_size,
                "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                "deepspeed": plugin.deepspeed_config if plugin is not None else None,
                "optimizer_groups": [
                    {
                        "name": group.get("name", "trainable"),
                        "initial_lr": group.get("initial_lr", group["lr"]),
                        "local_optimizer_elements": sum(p.numel() for p in group["params"]),
                    }
                    for group in self.optimizer.param_groups
                ],
            }
            OmegaConf.save(OmegaConf.create(runtime), Path(self.config.output_dir) / "runtime_config.yaml")
        self._init_wandb()
        self._init_checkpointing()

    def _calculate_total_batch_size(self):
        """calculate global batch size"""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """initialize Weights & Biases"""
        if self.accelerator.is_main_process and "wandb" in self.config.get("trackers", ["wandb"]):
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )

    def _init_checkpointing(self):
        """initialize checkpoint directory"""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _save_checkpoint(self):
        """save current training state"""

        state_dict = self.accelerator.get_state_dict(self.model)
        if self.accelerator.is_main_process:

            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            target = checkpoint_path + "_pytorch_model.pt"
            torch.save(state_dict, target + ".tmp")
            os.replace(target + ".tmp", target)

            summary_data = {
                "steps": self.completed_steps,
            }
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")
        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        if self.completed_steps % self.config.trainer.logging_frequency or not self.accelerator.is_main_process:
            return
        metrics = dict(metrics, learning_rate=self.lr_scheduler.get_last_lr()[0])
        trackers = self.config.get("trackers", ["wandb"])
        if "jsonl" in trackers:
            path = Path(self.config.output_dir) / "metrics.jsonl"
            with path.open("a") as handle:
                handle.write(json.dumps(dict(metrics, step=self.completed_steps)) + "\n")
        if "wandb" in trackers:
            wandb.log(metrics, step=self.completed_steps)
        logger.info(f"Step {self.completed_steps}, metrics: {metrics}")

    def _create_data_iterators(self):
        """create data iterators"""
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        """get next batch (automatically handle data loop)"""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        batch_vlm = None
        return batch_vla, batch_vlm

    def train(self):
        """execute training loop"""
        self._log_training_config()

        self._create_data_iterators()

        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps), disable=not self.accelerator.is_local_main_process
        )

        metric_sums = {}
        microbatch_count = 0
        while self.completed_steps < self.config.trainer.max_train_steps:
            batch_vla, batch_vlm = self._get_next_batch()

            step_metrics = self._train_step(batch_vla, batch_vlm)
            microbatch_count += 1
            for key, value in step_metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + value

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.sync_gradients:
                self._log_metrics({key: value / microbatch_count for key, value in metric_sums.items()})
                metric_sums = {}
                microbatch_count = 0
                if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                    self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """
        Evaluate the model on the given dataset using the specified metric function.

        :param eval_dataset: List of evaluation samples, each containing 'image', 'instruction', and 'action'.
        :param metric_fn: Function to compute the distance between predicted and ground truth actions.
        :return: Average metric score across the evaluation dataset.
        """

        if self.accelerator.is_main_process:

            examples, vlm_data = self._get_next_batch()

            score = 0.0

            batch_images = [example["image"] for example in examples]
            instructions = [example["lang"] for example in examples]  # [B, str]
            actions = [example["action"] for example in examples]  # label

            output_dict = self.model.predict_action(
                batch_images=batch_images, instructions=instructions, use_ddim=True, num_ddim_steps=20
            )

            normalized_actions = output_dict["normalized_actions"]  # B, T, D

            actions = np.array(actions)  # convert actions to numpy.ndarray
            num_pots = np.prod(actions.shape)
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            average_score = score / num_pots
            step_metrics["mse_score"] = average_score

        dist.barrier()
        return step_metrics

    def _log_training_config(self):
        """record training config"""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f" Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm):
        """execute single training step"""
        log_dict = {}
        with self.accelerator.accumulate(self.model):

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                total_loss = action_loss
            if not torch.isfinite(total_loss).item():
                raise FloatingPointError("Nonfinite action loss; refusing an optimizer update")
            self.accelerator.backward(total_loss)

            clipping = self.config.trainer.gradient_clipping
            if self.accelerator.sync_gradients and clipping is not None and clipping > 0:
                self.accelerator.clip_grad_norm_(self.model.parameters(), clipping)

            self.optimizer.step()
            if self.accelerator.sync_gradients and not self.accelerator.optimizer_step_was_skipped:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

            log_dict.update(
                {
                    "action_loss": action_loss.item(),
                }
            )
        return log_dict

    def _finalize_training(self):
        """training end processing"""
        state_dict = self.accelerator.get_state_dict(self.model)
        if self.accelerator.is_main_process:
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            target = os.path.join(final_checkpoint, "pytorch_model.pt")
            torch.save(state_dict, target + ".tmp")
            os.replace(target + ".tmp", target)
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        if self.accelerator.is_main_process and "wandb" in self.config.get("trackers", ["wandb"]):
            wandb.finish()

        self.accelerator.wait_for_everyone()


def prepare_model_parameters(model, cfg):
    """Apply weights-only initialization and freezing before optimizer selection."""
    checkpoint = cfg.trainer.get("pretrained_checkpoint")
    if checkpoint:
        TrainerUtils.load_pretrained_backbones(model, checkpoint, cfg.trainer.get("reload_modules"))
    TrainerUtils.freeze_backbones(model, cfg.trainer.get("freeze_modules", ""))
    TrainerUtils.print_trainable_parameters(model)
    return model


def main(cfg) -> None:
    from starVLA.model.action_codec.action_codebook import prepare_action_codec_config

    validate_training_config(cfg)
    prepare_action_codec_config(cfg)
    accelerator = create_accelerator(cfg)
    accelerator.print(accelerator.state)
    logger.info("VLA Training :: Warming Up")
    if cfg.framework.qwenvl.get("model_type") == "qwen3_5":
        set_seed(cfg.seed)  # Seed new action-head and LoRA initialization too.

    vla = build_framework(cfg)
    output_dir = setup_directories(cfg=cfg)

    if getattr(cfg.trainer, "use_lora", False):
        if cfg.framework.name == "QwenFast":
            from starVLA.model.framework.QwenFast import apply_qwen_fast_lora

            vla = apply_qwen_fast_lora(vla, cfg.trainer.lora)
        else:
            from starVLA.model.adapters import apply_backbone_lora

            vla = apply_backbone_lora(vla, cfg.trainer.lora)

        if cfg.trainer.get("run_generation_sanity_check", True):
            from PIL import Image

            image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            sample = {
                "action": np.random.uniform(-1, 1, size=(10, 4)).astype(np.float16),  # action_chunk, action_dim
                "image": [image, image],  # two views
                "lang": "This is a fake instruction for testing. Locate the target object.",
            }

            batch = [sample, sample]
            processor = vla.qwen_vl_interface.processor
            tokenizer = processor.tokenizer
            batch_inputs = vla.qwen_vl_interface.build_qwenvl_inputs(
                images=[batch[0]["image"]], instructions=[batch[0]["lang"]]
            )
            output = vla.qwen_vl_interface.generate(**batch_inputs)
            print(f"Generated Language:\n{tokenizer.decode(output[0], skip_special_tokens=True)}")

        if dist.is_initialized():
            if dist.get_rank() == 0:
                print("LoRA applied with config:", cfg.trainer.lora)
        else:
            print("LoRA applied with config:", cfg.trainer.lora)

    vla = prepare_model_parameters(vla, cfg)
    vla_train_dataloader, vlm_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    if getattr(cfg.trainer, "use_lora", False):
        optimizer, lr_scheduler = setup_optimizer_and_scheduler_lora(model=vla, cfg=cfg)
    else:
        optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLAMTrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        vlm_train_dataloader=vlm_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    accelerator.wait_for_everyone()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="starVLA/config/training/qwen35_discrete.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)  # Normalize CLI args to dotlist format
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    main(cfg)
