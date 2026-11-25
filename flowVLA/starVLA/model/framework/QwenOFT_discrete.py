# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""Discrete UAV action prediction with token or learned-embedding queries.

The backbone and action-head parameter hierarchy is kept stable for existing
Qwen3-VL and Qwen3.5 checkpoints. The continuous L1 framework lives in
QwenOFT_continuous.py.
"""
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    """Predict four categorical action dimensions for each query position."""

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
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # align dims --> we should put them to config or no?
        config.framework.action_model.action_hidden_dim = self.qwen_vl_interface.model.config.hidden_size
        self.action_model = get_action_model(config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        self.hidden_dim = config.framework.action_model.action_hidden_dim

        self.use_learned_action_queries = bool(getattr(self.qwen_vl_interface, "uses_learned_action_queries", False))
        if self.use_learned_action_queries:
            if self.qwen_vl_interface.num_action_queries != self.chunk_len:
                raise ValueError("Learned query count must match the action chunk")
            self.action_token = self.action_token_id = None
        else:
            self.action_token = getattr(config.framework, "action_query_token", "🔍")
            self.action_token_id = self.qwen_vl_interface.processor.tokenizer(
                self.action_token, add_special_tokens=False
            )["input_ids"][0]

        # 离散动作相关
        self.num_action_bins = getattr(config.framework.action_model, "num_action_bins", 256)
        self.ce_loss = nn.CrossEntropyLoss()
        from starVLA.model.action_codec import load_codec_from_config

        self.action_codec = load_codec_from_config(config)

        # ELMo-style scalar mix of all Qwen hidden layers.  The action query
        # token positions are still selected exactly as before; only the
        # hidden feature at each selected position changes from the final
        # layer to a learned softmax-weighted sum of all layers.
        self.use_all_layer_hidden = bool(getattr(config.framework, "use_all_layer_hidden", False))
        if self.use_all_layer_hidden:
            # Qwen3-VL stores the language-layer count under text_config;
            # retain compatibility with models exposing it at the top level.
            model_cfg = self.qwen_vl_interface.model.config
            text_cfg = getattr(model_cfg, "text_config", None)
            num_layers = getattr(model_cfg, "num_hidden_layers", None)
            if num_layers is None and text_cfg is not None:
                num_layers = getattr(text_cfg, "num_hidden_layers", None)
            if num_layers is None:
                raise AttributeError("Cannot determine Qwen hidden-layer count for all-layer feature mixing")
            num_hidden_states = int(num_layers) + 1
            self.layer_weights = nn.Parameter(torch.zeros(num_hidden_states))


    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if self.action_codec is None and prefix + "action_codec._extra_state" in state_dict:
            error_msgs.append("Versioned action checkpoint cannot load into a legacy floor decoder")
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


    def _aggregate_hidden_states(self, qwenvl_outputs):
        """Return final hidden states or an ELMo-style scalar mix."""
        if not self.use_all_layer_hidden:
            return qwenvl_outputs.hidden_states[-1]
        all_hidden = torch.stack(list(qwenvl_outputs.hidden_states), dim=0)
        weights = torch.softmax(self.layer_weights.float(), dim=0).to(all_hidden.dtype)
        return (all_hidden * weights.view(-1, 1, 1, 1)).sum(dim=0)


    def continuous_to_discrete(self, actions: torch.Tensor) -> torch.Tensor:
        """
        将连续动作值(-1, 1)转换为离散类别(0, num_bins-1)

        Args:
            actions: (..., action_dim) 连续动作值，范围[-1, 1]

        Returns:
            discrete_actions: (..., action_dim) 离散类别索引，范围[0, num_bins-1]
        """
        if self.action_codec is not None:
            return self.action_codec.encode_normalized(actions)
        # Legacy checkpoints retain their original label semantics.
        # 先将[-1, 1]映射到[0, 1]，然后乘以(num_bins-1)，再取整
        normalized = (actions + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        discrete = torch.clamp(torch.floor(normalized * (self.num_action_bins - 1)).long(), 0, self.num_action_bins - 1)
        return discrete

    def discrete_to_continuous(self, discrete_actions: torch.Tensor) -> torch.Tensor:
        """
        将离散类别(0, num_bins-1)转换为连续动作值(-1, 1)

        Args:
            discrete_actions: (..., action_dim) 离散类别索引，范围[0, num_bins-1]

        Returns:
            actions: (..., action_dim) 连续动作值，范围[-1, 1]
        """
        if self.action_codec is not None:
            return self.action_codec.decode_normalized(discrete_actions)
        # Legacy checkpoints retain their original label semantics.
        normalized = discrete_actions.float() / (self.num_action_bins - 1)  # [0, num_bins-1] -> [0, 1]
        continuous = normalized * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        return continuous

    def _encode_action_queries(self, qwen_inputs, *, policy=False):
        """Share feature extraction while retaining each path's AMP semantics."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if policy:
                outputs = self.qwen_vl_interface(**qwen_inputs, output_hidden_states=True)
            else:
                outputs = self.qwen_vl_interface(
                    **qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden = self._aggregate_hidden_states(outputs)
        if policy:
            hidden = self._aggregate_hidden_states(outputs)
        return self._gather_action_token_embeddings(
            hidden, qwen_inputs.get("input_ids"), action_token_id=self.action_token_id
        )

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

        temperature = kwargs.get("temperature", 1.6)

        # print('mode:', mode, 'temperature:', temperature)
        if mode == "forward":
            # 训练前向：直接回归未来动作（无扩散）
            batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
            instructions = [example["lang"] for example in examples]  # [B, str]
            actions = [example["action"] for example in examples]  # label [B， len, 7]

            instructions = self._prepare_action_instructions(instructions)

            # Step 1: QWenVL input format
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
            action_queries = self._encode_action_queries(qwen_inputs)

            # Step 4: Action Expert Forward and Loss
            with torch.autocast("cuda", dtype=torch.float32):
                # 离散动作头：输出logits用于分类
                pred_logits = self.action_model.predict_action(action_queries)  # (B, chunk_len, action_dim, num_bins)

                # 标签对齐：取最后 chunk_len 段
                actions = torch.tensor(
                    np.array(actions), device=action_queries.device, dtype=torch.float32
                )  # [B, T_full, action_dim]
                actions_target = actions[:, -(self.future_action_window_size + 1) :, :]  # (B, chunk_len, action_dim)

                # 离散动作：交叉熵损失
                # 将连续标签转换为离散类别
                if self.action_codec is not None:
                    # Labels come directly from physical raw deltas, avoiding
                    # normalization/FP32 roundoff at class boundaries.
                    if any("action_bins" not in example for example in examples):
                        raise ValueError("Versioned-codec training requires dataset action_bins")
                    discrete_targets = torch.as_tensor(
                        np.array([e["action_bins"] for e in examples]), device=action_queries.device, dtype=torch.long
                    )
                    discrete_targets = discrete_targets[:, -(self.future_action_window_size + 1) :, :]
                    if discrete_targets.shape != actions_target.shape:
                        raise ValueError("Action labels and normalized targets have different shapes")
                else:
                    discrete_targets = self.continuous_to_discrete(actions_target)

                # 重塑以适应交叉熵损失：(B*chunk_len*action_dim, num_bins)
                B, chunk_len, action_dim = discrete_targets.shape
                pred_logits_flat = pred_logits.view(B * chunk_len * action_dim, self.num_action_bins)
                discrete_targets_flat = discrete_targets.view(-1)

                ce_loss = self.ce_loss(pred_logits_flat, discrete_targets_flat)
                action_loss = ce_loss

                # 构建返回字典
                result = {"action_loss": action_loss}

                # 离散动作：添加logits和预测的离散类别
                pred_discrete = torch.argmax(pred_logits, dim=-1)  # (B, chunk_len, action_dim)
                pred_continuous = self.discrete_to_continuous(pred_discrete)  # (B, chunk_len, action_dim)

                result["action_logits"] = pred_logits
                result["action_discrete"] = pred_discrete
                result["action_mean"] = pred_continuous  # 兼容性：将预测的连续值作为mean返回

        elif mode == "logp":
            # Compute log probabilities for given actions using pre-processed tensors
            qwen_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
            }
            for key in ("image_grid_thw", "video_grid_thw", "pixel_values_videos", "mm_token_type_ids", "position_ids"):
                if key in kwargs:
                    qwen_inputs[key] = kwargs[key]
            if not np.isfinite(temperature) or temperature <= 0:
                raise ValueError("temperature must be finite and positive")
            action_queries = self._encode_action_queries(qwen_inputs, policy=True)

            # Actions to tensor if needed (though usually passed as tensor in PPO)
            if not isinstance(actions, torch.Tensor):
                actions = torch.tensor(actions, device=action_queries.device, dtype=torch.float32)
            if actions.dim() == 4:
                actions = actions.squeeze(0)
            # Keep the policy head and log-softmax path in float32 for numerical stability.
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                # Action Model - 离散动作版本
                pred_logits = self.action_model.predict_action(action_queries)  # (B, chunk_len, action_dim, num_bins)

                # 温度缩放
                pred_logits = pred_logits / temperature

                # 将连续动作转换为离散类别
                discrete_actions = self.continuous_to_discrete(actions)  # (B, chunk_len, action_dim)

                # 计算交叉熵损失作为负log概率
                # 重塑以适应交叉熵损失：(B*chunk_len*action_dim, num_bins)
                try:
                    B, chunk_len, action_dim = discrete_actions.shape
                except:
                    print(f"discrete_actions: {discrete_actions}")
                pred_logits_flat = pred_logits.view(B * chunk_len * action_dim, self.num_action_bins)
                discrete_actions_flat = discrete_actions.view(-1)

                # 计算log概率：使用交叉熵的负值
                ce_loss_per_element = F.cross_entropy(pred_logits_flat, discrete_actions_flat, reduction="none")
                log_probs = -ce_loss_per_element  # (B*chunk_len*action_dim,)

                # 重塑回原始形状：(B, chunk_len, action_dim)
                log_probs = log_probs.view(B, chunk_len, action_dim)

                # 对每个时间步的action_dim维度求和
                log_probs_sum = log_probs.sum(dim=-1)  # (B, chunk_len)

                # 计算熵
                probs = F.softmax(pred_logits_flat, dim=-1)  # (B*chunk_len*action_dim, num_bins)
                entropy_per_element = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # (B*chunk_len*action_dim,)
                entropy = entropy_per_element.view(B, chunk_len, action_dim).sum(dim=-1)  # (B, chunk_len)

            result = {
                "action_log_probs": log_probs_sum,
                "action_entropy": entropy,
                "action_logits": pred_logits,
            }
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'forward' or 'logp'")

        return result

    def sample_action_from_inputs(self, input_ids, attention_mask, pixel_values, proprio=None, **kwargs):
        """
        Sample actions directly from pre-processed tensors (used in RL rollout)
        """
        # print('mode: rollout', 'temperature:', kwargs.get("temperature", 1.0))
        qwen_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
        for key in ("image_grid_thw", "video_grid_thw", "pixel_values_videos", "mm_token_type_ids", "position_ids"):
            if key in kwargs:
                qwen_inputs[key] = kwargs[key]

        temperature = kwargs.get("temperature", 1.6)
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")

        action_queries = self._encode_action_queries(qwen_inputs, policy=True)

        # Action Model - 离散动作版本
        with torch.autocast("cuda", dtype=torch.float32):
            pred_logits = self.action_model.predict_action(action_queries)

        #
        scaled_logits = pred_logits / temperature

        # 从离散分布中采样
        B, chunk_len, action_dim, num_bins = scaled_logits.shape
        probs = F.softmax(scaled_logits.view(-1, num_bins), dim=-1)  # (B*chunk_len*action_dim, num_bins)
        sampled_bins = torch.multinomial(probs, 1).squeeze(-1)  # (B*chunk_len*action_dim,)
        sampled_bins = sampled_bins.view(B, chunk_len, action_dim)  # (B, chunk_len, action_dim)

        # 将离散bins转换为连续动作值
        sampled_actions = self.discrete_to_continuous(sampled_bins)

        return {
            "sampled_actions": sampled_actions,
            "action_logits": pred_logits,
        }

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

        instructions = self._prepare_action_instructions(instructions)

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        action_queries = self._encode_action_queries(qwen_inputs)

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # 离散动作头：预测logits，然后转换为连续值
            pred_logits = self.action_model.predict_action(action_queries)  # (B, chunk_len, action_dim, num_bins)
            pred_discrete = torch.argmax(pred_logits, dim=-1)  # (B, chunk_len, action_dim)
            pred_continuous = self.discrete_to_continuous(pred_discrete)  # (B, chunk_len, action_dim)

            normalized_actions = pred_continuous.detach().cpu().numpy()
            result = {"normalized_actions": normalized_actions}
            if self.action_codec is not None:
                result["action_bins"] = pred_discrete.detach().cpu().numpy()
                result["local_actions"] = self.action_codec.decode_physical(pred_discrete).cpu().numpy()
            return result

    def _prepare_action_instructions(self, instructions):
        if self.use_learned_action_queries:
            suffix = f" Please predict the next {self.chunk_len} robot actions."
        else:
            action_tokens = self.action_token * self.chunk_len
            suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        return [instruction + suffix for instruction in instructions]

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
        if self.use_learned_action_queries:
            if input_ids is not None and last_hidden.shape[1] != input_ids.shape[1] + self.chunk_len:
                raise ValueError("The backbone must append exactly one full chunk of learned queries")
            return last_hidden[:, -self.chunk_len :, :]

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
