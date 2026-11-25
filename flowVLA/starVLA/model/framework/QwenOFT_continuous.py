# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
Qwen-OFT Framework

A lightweight implementation that uses an action special token to parallelly predict continuous actions
conditioned on multi-view images plus a language instruction (shares parameters with the VLM).
Inspired by OpenVLA-OFT
Key Points:
  - Qwen2.5 vision-language backbone
  - Injects an action special token into the VLM
  - Continuous action prediction via L1 regression over the action special token hidden states

Note: How to add special tokens to Qwen2.5:
  See /starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
"""
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.model.tools import FRAMEWORK_REGISTRY


logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.MLP_ActionHeader import get_action_model
from starVLA.training.trainer_utils.trainer_tools import resize_images


@FRAMEWORK_REGISTRY.register("QwenOFT")
class Qwenvl_OFT(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen2.5 VL interface for fused language/vision token embeddings
      - Layer-wise QFormer for multi-layer feature aggregation
      - DINO encoder for dense multi-view spatial tokens
      - DiT diffusion head for future action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        self.config = config
        if config.framework.get("use_all_layer_hidden", False):
            raise ValueError("Continuous all-layer is unsupported; use the discrete all-layer recipe")
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # align dims --> we should put them to config or no?
        config.framework.action_model.action_hidden_dim = self.qwen_vl_interface.model.config.hidden_size
        self.action_model = get_action_model(config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        self.hidden_dim = config.framework.action_model.action_hidden_dim

        self.action_token = "🔍"  # TODO also can add spacail token to Qwen, but too complex
        self.action_token_id = self.qwen_vl_interface.processor.tokenizer("🔍", add_special_tokens=False)["input_ids"][0]

        # L1 损失
        self.l1_loss = nn.L1Loss()

    def forward(
        self,
        examples: List[dict] = None,
        mode: str = "forward",
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        actions=None,
        proprio=None,
        **kwargs,
    ) -> Tuple:
        """
        统一的forward方法，支持训练和log_prob计算两种模式。

        Args:
            examples: List[dict], 用于训练模式，每个dict包含image, lang, action
            mode: str, "forward" (训练) 或 "logp" (计算log_prob)
            input_ids, attention_mask, pixel_values: 用于logp模式的预处理输入
            actions: 用于logp模式的动作tensor [B, chunk_len, action_dim]
            proprio: 预留的proprio输入
            **kwargs: 其他参数

        Returns:
            dict: 根据mode返回不同结果
        """
        if mode == "forward":
            # 训练前向：直接回归未来动作（无扩散）
            batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
            instructions = [example["lang"] for example in examples]  # [B, str]
            actions = [example["action"] for example in examples]  # label [B， len, 7]

            # step 0: add special action token to instruction
            action_tokens = (
                self.action_token * self.chunk_len
            )  # can't add " " between two tokens, otherwise will be tokenized to multiple tokens
            prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
            instructions = [instruction + prompt_suffix for instruction in instructions]

            # Step 1: QWenVL input format
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qwenvl_outputs = self.qwen_vl_interface(
                    **qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                # last_hidden_state: [B, seq_len, H]
                last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

            # Step 4: Action Expert Forward and Loss
            with torch.autocast("cuda", dtype=torch.float32):
                # 提取动作 token embedding 作为动作预测查询
                input_ids = qwen_inputs.get("input_ids", None)
                action_queries = self._gather_action_token_embeddings(
                    last_hidden, input_ids, action_token_id=self.action_token_id
                )  # [B, chunk_len, H]

                # 检查action_model是否支持高斯分布输出（用于RL）
                supports_gaussian = hasattr(self.action_model, "predict_action") and hasattr(
                    self.action_model, "sample_action"
                )

                if supports_gaussian:
                    # 使用高斯策略：输出均值和标准差（通过log_std计算）
                    try:
                        pred_mean, pred_std = self.action_model.predict_action(action_queries, return_distribution=True)
                    except TypeError:
                        pred_mean = self.action_model.predict_action(action_queries)
                        pred_std = None
                else:
                    # 使用确定性预测（L1回归）
                    pred_mean = self.action_model.predict_action(action_queries)
                    pred_std = None
                    pred_log_std = None

                # 标签对齐：取最后 chunk_len 段
                actions = torch.tensor(
                    np.array(actions), device=pred_mean.device, dtype=pred_mean.dtype
                )  # [B, T_full, action_dim]
                actions_target = actions[:, -(self.future_action_window_size + 1) :, :]  # (B, chunk_len, action_dim)

                # 计算损失
                if pred_std is not None:
                    use_nll_loss = getattr(self.config.framework.action_model, "use_nll_loss", False)
                    if use_nll_loss:
                        from torch.distributions import Normal

                        dist = Normal(pred_mean, pred_std)
                        log_probs = dist.log_prob(actions_target)
                        nll_loss = -log_probs.sum(dim=-1).mean()
                        std_reg_weight = getattr(self.config.framework.action_model, "std_reg_weight", 10)
                        std_reg_loss = std_reg_weight * pred_std.mean()
                        action_loss = nll_loss + std_reg_loss
                    else:
                        action_loss = self.l1_loss(pred_mean, actions_target)
                else:
                    action_loss = self.l1_loss(pred_mean, actions_target)

                result = {"action_loss": action_loss}
                if pred_std is not None:
                    result["action_mean"] = pred_mean
                    result["action_std"] = pred_std

                    # 添加诊断信息（用于调试）
                    if self.training:  # 只在训练时记录，避免影响推理性能
                        mean_error = (pred_mean - actions_target).abs().mean().item()
                        std_mean = pred_std.mean().item()
                        std_min = pred_std.min().item()
                        std_max = pred_std.max().item()
                        result["_debug_mean_error"] = mean_error
                        result["_debug_std_mean"] = std_mean
                        result["_debug_std_min"] = std_min
                        result["_debug_std_max"] = std_max
                        if use_nll_loss and pred_std is not None:
                            result["_debug_nll_loss"] = nll_loss.item()
                            result["_debug_std_reg_loss"] = std_reg_loss.item()

        elif mode == "logp":
            # Compute log probabilities for given actions using pre-processed tensors
            qwen_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
            }
            qwen_inputs.update(kwargs)

            qwenvl_outputs = self.qwen_vl_interface(**qwen_inputs, output_hidden_states=True)
            last_hidden = qwenvl_outputs.hidden_states[-1]

            # Gather action tokens
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )

            # Actions to tensor if needed (though usually passed as tensor in PPO)
            if not isinstance(actions, torch.Tensor):
                actions = torch.tensor(actions, device=action_queries.device, dtype=action_queries.dtype)

            # Action Model
            if hasattr(self.action_model, "predict_action") and hasattr(self.action_model, "sample_action"):
                try:
                    # Get distribution parameters
                    # -1 1
                    # 40
                    pred_mean, pred_std = self.action_model.predict_action(action_queries, return_distribution=True)

                    dist = torch.distributions.Normal(pred_mean, pred_std)  # [B, chunk_len, action_dim]
                    # Sum over action dimensions
                    log_probs = dist.log_prob(actions).sum(dim=-1)  # [B, chunk_len]


                    # Compute Entropy (sum over action dimensions)
                    entropy = dist.entropy().sum(dim=-1)  # [B, chunk_len]

                    result = {
                        "action_log_probs": log_probs,
                        "action_entropy": entropy,
                        "action_mean": pred_mean,
                        "action_std": pred_std,
                    }
                except TypeError:
                    # Fallback if return_distribution not supported (shouldn't happen if checked)
                    result = {"action_log_probs": torch.zeros(actions.shape[:2], device=actions.device)}
            else:
                result = {"action_log_probs": torch.zeros(actions.shape[:2], device=actions.device)}
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'forward' or 'logp'")

        return result

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List[List[Image.Image]],  # Batch of PIL Image list as [view1, view2]
        instructions: List[str],
        **kwargs: str,
    ) -> np.ndarray:
        """
        推理：单次前向直接回归未来动作（无扩散采样）。

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Args:
            batch_images: List of samples; each sample is List[PIL.Image] (multi-view).
            instructions: List[str] natural language task instructions.
            cfg_scale: >1 enables classifier-free guidance (scales conditional vs unconditional).
            use_ddim: Whether to use DDIM deterministic sampling.
            num_ddim_steps: Number of DDIM steps if enabled.
            **kwargs: Reserved.

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # step 0: add special action token to instruction
        action_tokens = (
            self.action_token * self.chunk_len
        )  # can't add " " between two tokens, otherwise will be tokenized to multiple tokens
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [instruction + prompt_suffix for instruction in instructions]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # 提取动作 token embedding 作为动作预测查询
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )  # [B, chunk_len, H]

            # 检查是否需要返回分布参数（用于RL）
            return_distribution = kwargs.get("return_distribution", False)
            return_distribution = True

            # 检查action_model是否支持高斯分布输出
            supports_gaussian = hasattr(self.action_model, "predict_action") and hasattr(
                self.action_model, "sample_action"
            )

            if supports_gaussian and return_distribution:
                # 返回均值和标准差（通过log_std计算）
                try:
                    pred_mean, pred_std = self.action_model.predict_action(action_queries, return_distribution=True)
                    # std已经通过exp(log_std)计算得到，log_std在action_model内部进行了范围限制

                    normalized_mean = pred_mean.detach().cpu().numpy()
                    normalized_std = pred_std.detach().cpu().numpy()

                    return {
                        "normalized_actions": normalized_mean,  # 保持向后兼容，返回均值作为主要动作
                        "action_mean": normalized_mean,
                        "action_std": normalized_std,
                    }
                except TypeError:
                    # 如果不支持 return_distribution 参数，回退到只返回均值
                    pred_actions = self.action_model.predict_action(action_queries)
                    normalized_actions = pred_actions.detach().cpu().numpy()
                    return {"normalized_actions": normalized_actions}
            else:
                # 只返回均值（确定性预测）
                # 对于支持高斯分布的模型，如果 return_distribution=False，也返回均值
                if supports_gaussian:
                    pred_actions = self.action_model.predict_action(action_queries, return_distribution=False)
                else:
                    # L1RegressionActionHead 不支持 return_distribution 参数
                    pred_actions = self.action_model.predict_action(action_queries)
                normalized_actions = pred_actions.detach().cpu().numpy()
                return {"normalized_actions": normalized_actions}

    @torch.inference_mode()
    def sample_action(
        self,
        batch_images: List[List[Image.Image]],  # Batch of PIL Image list as [view1, view2]
        instructions: List[str],
        **kwargs: str,
    ) -> np.ndarray:
        """
        从高斯分布中采样动作（用于RL训练）。

        Args:
            batch_images: List of samples; each sample is List[PIL.Image] (multi-view).
            instructions: List[str] natural language task instructions.
            **kwargs: Reserved.

        Returns:
            dict:
                sampled_actions (np.ndarray): Shape [B, T, action_dim], 从分布中采样的动作。
                如果支持高斯分布，还会返回 action_mean 和 action_std。
        """
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # step 0: add special action token to instruction
        action_tokens = self.action_token * self.chunk_len
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [instruction + prompt_suffix for instruction in instructions]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        # Step 4: Sample action from distribution
        with torch.autocast("cuda", dtype=torch.float32):
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )

            # 检查是否支持采样
            if hasattr(self.action_model, "sample_action"):
                # 先获取分布参数（避免在 sample_action 内部重复计算）
                pred_mean, pred_std = self.action_model.predict_action(action_queries, return_distribution=True)
                # std已经在action_model内部进行了范围限制，这里不需要再次clamp

                # 从分布中采样动作
                noise = torch.randn_like(pred_mean)
                sampled_actions = pred_mean + pred_std * noise  # (B, chunk_len, action_dim)

                # 计算对数概率（用于RL训练，如PPO）
                from torch.distributions import Normal

                dist = Normal(pred_mean, pred_std)
                log_probs = dist.log_prob(sampled_actions).sum(dim=-1)  # (B, chunk_len) -> 对每个时间步的action_dim求和

                sampled_actions_np = sampled_actions.detach().cpu().numpy()
                mean_np = pred_mean.detach().cpu().numpy()
                std_np = pred_std.detach().cpu().numpy()
                log_probs_np = log_probs.detach().cpu().numpy()

                return {
                    "sampled_actions": sampled_actions_np,
                    "action_mean": mean_np,
                    "action_std": std_np,
                    "action_log_probs": log_probs_np,  # 用于RL训练
                }
            else:
                # 如果不支持采样，返回确定性预测
                pred_actions = self.action_model.predict_action(action_queries)
                normalized_actions = pred_actions.detach().cpu().numpy()
                return {"sampled_actions": normalized_actions}

    @torch.inference_mode()
    def compute_action_log_prob(
        self,
        batch_images: List[List[Image.Image]],
        instructions: List[str],
        actions: np.ndarray,  # (B, chunk_len, action_dim) 要计算log_prob的动作
        **kwargs: str,
    ) -> np.ndarray:
        """
        计算给定动作的对数概率（用于RL训练，如PPO中计算旧策略的log_prob）。

        Args:
            batch_images: List of samples; each sample is List[PIL.Image] (multi-view).
            instructions: List[str] natural language task instructions.
            actions: np.ndarray, shape (B, chunk_len, action_dim), 要计算log_prob的动作。
            **kwargs: Reserved.

        Returns:
            dict:
                action_log_probs (np.ndarray): Shape [B, chunk_len], 每个动作的对数概率。
                如果支持高斯分布，还会返回 action_mean 和 action_std。
        """
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # step 0: add special action token to instruction
        action_tokens = self.action_token * self.chunk_len
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [instruction + prompt_suffix for instruction in instructions]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        # Step 4: Compute log probability
        with torch.autocast("cuda", dtype=torch.float32):
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )

            # 将actions转换为tensor
            actions_tensor = torch.tensor(actions, device=action_queries.device, dtype=action_queries.dtype)

            # 检查是否支持高斯分布
            if hasattr(self.action_model, "predict_action") and hasattr(self.action_model, "sample_action"):
                try:
                    pred_mean, pred_std = self.action_model.predict_action(action_queries, return_distribution=True)
                    # std已经在action_model内部进行了范围限制，这里不需要再次clamp

                    # 计算对数概率
                    from torch.distributions import Normal

                    dist = Normal(pred_mean, pred_std)
                    log_probs = dist.log_prob(actions_tensor).sum(dim=-1)  # (B, chunk_len)

                    log_probs_np = log_probs.detach().cpu().numpy()
                    mean_np = pred_mean.detach().cpu().numpy()
                    std_np = pred_std.detach().cpu().numpy()

                    return {
                        "action_log_probs": log_probs_np,
                        "action_mean": mean_np,
                        "action_std": std_np,
                    }
                except TypeError:
                    # 如果不支持高斯分布，返回警告
                    import warnings

                    warnings.warn(
                        "Action model does not support distribution, cannot compute log_prob. Returning zeros."
                    )
                    B, chunk_len = actions_tensor.shape[:2]
                    return {
                        "action_log_probs": np.zeros((B, chunk_len)),
                    }
            else:
                # 如果不支持高斯分布，返回警告
                import warnings

                warnings.warn("Action model does not support distribution, cannot compute log_prob. Returning zeros.")
                B, chunk_len = actions_tensor.shape[:2]
                return {
                    "action_log_probs": np.zeros((B, chunk_len)),
                }

    def sample_action_from_inputs(self, input_ids, attention_mask, pixel_values, proprio=None, **kwargs):
        """
        Sample actions directly from pre-processed tensors (used in RL rollout)
        """
        qwen_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
        qwen_inputs.update(kwargs)

        qwenvl_outputs = self.qwen_vl_interface(**qwen_inputs, output_hidden_states=True)
        last_hidden = qwenvl_outputs.hidden_states[-1]

        # Gather action tokens
        action_queries = self._gather_action_token_embeddings(
            last_hidden, input_ids, action_token_id=self.action_token_id
        )

        # Action Model
        if hasattr(self.action_model, "sample_action"):
            # Predict mean and std
            try:
                pred_mean, pred_std = self.action_model.predict_action(action_queries, return_distribution=True)

                # Sample from distribution
                dist = torch.distributions.Normal(pred_mean, pred_std)
                sampled_actions = dist.sample()

            except Exception:
                logger.exception("Failed to sample actions")
                raise

            return {"sampled_actions": sampled_actions, "action_mean": pred_mean, "action_std": pred_std}
        else:
            # Deterministic fallback
            pred_actions = self.action_model.predict_action(action_queries)
            return {"sampled_actions": pred_actions}

    def _gather_action_token_embeddings(
        self,
        last_hidden: torch.Tensor,  # [B, L, H]
        input_ids: torch.Tensor,  # [B, L]
        action_token_id=None,  # 可为 int 或 List[int]
    ) -> torch.Tensor:
        """
        向量化批量提取动作 token embedding:
          - 不再逐样本 for 循环
          - 取每个样本里最靠后的 chunk_len 个动作占位 token
        Args:
            last_hidden: [B, L, H]
            input_ids:   [B, L]
            action_token_id: int 或 List[int]
        Returns:
            action_queries: [B, chunk_len, H]
        """
        if action_token_id is None:
            raise ValueError("action_token_id 不能为空")

        device = input_ids.device
        B, L, H = last_hidden.shape

        # 支持多 id（如多个变体）
        if isinstance(action_token_id, (list, tuple, set)):
            id_list = torch.tensor(list(action_token_id), device=device, dtype=input_ids.dtype)
            # torch.isin 需要 PyTorch >=1.10
            mask = torch.isin(input_ids, id_list)
        else:
            mask = input_ids == action_token_id  # [B, L]

        counts = mask.sum(dim=1)  # [B]
        if (counts < self.chunk_len).any():
            insufficient = (counts < self.chunk_len).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"以下样本动作 token 数量不足 {self.chunk_len}: {insufficient} | counts={counts.tolist()}"
            )

        # 位置索引
        idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # [B, L]
        masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))  # 非动作位置置 -1

        # 取最后 chunk_len 个（索引大的在序列靠后）
        # 注意: 已确保数量足够，不会出现 -1 被错误选中的问题
        topk_pos = masked_pos.topk(k=self.chunk_len, dim=-1).values  # [B, chunk_len] 未排序
        # 时间顺序排序
        selected_pos = topk_pos.sort(dim=-1).values  # [B, chunk_len]

        # Gather
        expanded_index = selected_pos.unsqueeze(-1).expand(-1, -1, H)  # [B, chunk_len, H]
        action_queries = last_hidden.gather(dim=1, index=expanded_index)  # [B, chunk_len, H]
        return action_queries
