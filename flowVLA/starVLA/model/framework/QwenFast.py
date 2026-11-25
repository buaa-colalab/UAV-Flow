# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
Qwen-Fast Framework

A lightweight implementation for autoregressive discrete action prediction conditioned on multi-view images + instruction.
fast tokenizer is copyright from physical-intelligence/fast

Key Points:
  - Qwen2.5 vision-language backbone
  - Unified action learning via next-token prediction (fast tokenizer)
  - Autoregressive action tokens derived from discretized / symbolized continuous actions

Note: How to add special tokens to Qwen2.5:
  /starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
"""

from typing import List, Optional, Tuple, Any
import torch
import numpy as np
from scipy.fft import idct
from PIL import Image


from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.model.tools import FRAMEWORK_REGISTRY


logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.fast_ActionHeader import get_action_model
from starVLA.training.trainer_utils.trainer_tools import resize_images


def apply_qwen_fast_lora(model, lora_cfg):
    from peft import LoraConfig, get_peft_model

    lora_kwargs = {
        "r": int(lora_cfg.r),
        "lora_alpha": int(lora_cfg.lora_alpha),
        "target_modules": list(lora_cfg.target_modules),
        "lora_dropout": float(lora_cfg.lora_dropout),
        "init_lora_weights": "gaussian",
    }
    action_cfg = model.config.framework.action_model
    train_action_embeddings = bool(getattr(action_cfg, "train_fast_action_token_embeddings", False))

    if not train_action_embeddings:
        vlm = model.qwen_vl_interface.model.model
        model.qwen_vl_interface.model.model = get_peft_model(vlm, LoraConfig(**lora_kwargs))
        return model

    token_ids = sorted(getattr(model.qwen_vl_interface, "fast_action_token_ids", []))
    if not token_ids:
        raise RuntimeError("QwenFast action token IDs are unavailable; cannot train their embeddings")
    if token_ids != list(range(token_ids[0], token_ids[0] + len(token_ids))):
        raise ValueError("QwenFast action token IDs must be contiguous for FAST ID offset decoding")

    expected_tokens = int(getattr(action_cfg, "num_action_bins", len(token_ids)))
    if len(token_ids) != expected_tokens:
        raise ValueError(f"QwenFast registered {len(token_ids)} action tokens, expected {expected_tokens}")

    lora_kwargs["trainable_token_indices"] = token_ids
    model.qwen_vl_interface.model = get_peft_model(
        model.qwen_vl_interface.model,
        LoraConfig(**lora_kwargs),
    )
    logger.info(
        f"QwenFast LoRA will train {len(token_ids)} action-token embedding rows " "with tied input/output weights"
    )
    return model


@FRAMEWORK_REGISTRY.register("QwenFast")
class Qwenvl_Fast(baseframework):
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
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.action_model = get_action_model(config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        self.hidden_dim = config.framework.action_model.action_hidden_dim

        self.action_model.fast_tokenizer.time_horizon = self.future_action_window_size + 1
        self.action_model.fast_tokenizer.action_dim = self.config.framework.action_model.action_dim

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        训练前向：直接next token prediction 预测未来动作（无扩散）。

        Flow:
          1. Build QwenVL inputs (images + instruction tokens)
          2. Extract hidden states from configured layer range
          7. Predict action and compute L1 loss

        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
            **kwargs: Reserved.

        Returns:
            dict:
                action_loss (torch.Tensor): Scalar diffusion noise prediction loss.
        """
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]

        # step 0: map_raw_action_to_vlm_action
        vlm_action_tokens = self.action_model.encoder_action2vlmtoken(actions)  # List[str]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions, solutions=vlm_action_tokens
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )

        vlm_action_loss = qwenvl_outputs.loss
        if vlm_action_loss is None or torch.isnan(vlm_action_loss):
            vlm_action_loss = torch.tensor(0.0, device=self.qwen_vl_interface.model.device)

        return {"action_loss": vlm_action_loss}

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
        instructions = [instruction for instruction in instructions]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)

        action_cfg = self.config.framework.action_model
        num_beams = max(1, int(getattr(action_cfg, "fast_num_beams", 8)))
        generate_kwargs = dict(max_length=4096)
        if num_beams > 1:
            generate_kwargs.update(num_beams=num_beams, num_return_sequences=num_beams)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated_ids = self.qwen_vl_interface.model.generate(
                **qwen_inputs,
                **generate_kwargs,
            )
        # --- Extract and decoder vlm_action to continue actions ---
        # --- extrace token (index based on VLM) ---
        batch_vlm_action_token_ids = self._extract_action_token_ids(generated_ids)
        # --- map index to fast tokenizer index space ---
        batch_fast_action_token_idx = self._decode_action_tokens(batch_vlm_action_token_ids)
        # --- decode fast tokenizer index to action semantic ---
        normalized_actions = self._constrained_fast_decode(batch_fast_action_token_idx, num_beams=num_beams)

        return {"normalized_actions": normalized_actions}

    def _constrained_fast_decode(
        self,
        batch_fast_token_ids: List[Any],
        num_beams: int = 1,
    ) -> np.ndarray:
        """
        约束解码：beam search 返回的 num_beams 个候选里，优先选择 BPE 解码后
        系数数量恰好等于 time_horizon * action_dim 的序列（合法 FAST 编码）；
        全部不合法时退回第一个候选（beam 分数最高）的容错解码。
        batch_fast_token_ids 按 [sample0_beam0..beamK, sample1_beam0..beamK] 排列。
        """
        if num_beams <= 1:
            return self._robust_fast_decode(batch_fast_token_ids)
        tokenizer = self.action_model.fast_tokenizer
        expected = (self.future_action_window_size + 1) * self.config.framework.action_model.action_dim
        batch_size = len(batch_fast_token_ids) // num_beams
        selected = []
        num_constrained = 0
        for b in range(batch_size):
            candidates = batch_fast_token_ids[b * num_beams : (b + 1) * num_beams]
            chosen = None
            for seq in candidates:
                if not seq:
                    continue
                try:
                    if len(tokenizer.bpe_tokenizer.decode(seq)) == expected:
                        chosen = seq
                        break
                except Exception:
                    continue
            if chosen is not None:
                num_constrained += 1
            else:
                chosen = candidates[0] if candidates else None
            selected.append(chosen)
        logger.info(f"Fast constrained decode: {num_constrained}/{batch_size} samples hit a valid sequence")
        return self._robust_fast_decode(selected)

    def _robust_fast_decode(
        self,
        batch_fast_token_ids: List[Any],
    ) -> np.ndarray:
        """
        容错版 FAST 解码。

        原版 tokenizer.decode 要求 BPE 解码出的 DCT 系数恰好是
        time_horizon * action_dim 个，自回归生成只要错 1 个 token 就会
        数量不符并整体清零。这里改为：不足补零、超出截断，再 idct。
        仅在序列为空或硬错误时返回全零。返回 [B, T, action_dim]。
        """
        tokenizer = self.action_model.fast_tokenizer
        horizon = self.future_action_window_size + 1
        action_dim = self.config.framework.action_model.action_dim
        expected = horizon * action_dim
        decoded_actions = []
        num_padded = num_truncated = num_failed = 0
        for seq in batch_fast_token_ids:
            if not seq:
                num_failed += 1
                decoded_actions.append(np.zeros((horizon, action_dim), dtype=np.float32))
                continue
            try:
                decoded_str = tokenizer.bpe_tokenizer.decode(seq)
                dct_coeff = np.array(list(map(ord, decoded_str))) + tokenizer.min_token
                if dct_coeff.size < expected:
                    num_padded += 1
                    dct_coeff = np.pad(dct_coeff, (0, expected - dct_coeff.size))
                elif dct_coeff.size > expected:
                    num_truncated += 1
                    dct_coeff = dct_coeff[:expected]
                dct_coeff = dct_coeff.reshape(horizon, action_dim)
                decoded_actions.append(idct(dct_coeff / tokenizer.scale, axis=0, norm="ortho"))
            except Exception as exc:
                num_failed += 1
                logger.warning(f"Fast robust decode failed, fallback to zeros: {exc}")
                decoded_actions.append(np.zeros((horizon, action_dim), dtype=np.float32))
        if num_padded or num_truncated or num_failed:
            logger.info(f"Fast robust decode stats: padded={num_padded}, truncated={num_truncated}, failed={num_failed}")
        return np.stack(decoded_actions).astype(np.float32)

    def _extract_action_token_ids(
        self,
        generated_ids: torch.LongTensor,
    ) -> List[List[int]]:
        """
        从生成的 token 序列里抽取动作 token（已加偏移），返回一个二维列表：
        ret[b] = [vlm_action_token_id_0, vlm_action_token_id_1, ...]
        规则：保留所有落在 [_ACTION_TOKEN_MIN, _ACTION_TOKEN_MAX] 内的 token，按出现顺序。
        你可按需要改成“只取首次出现后连续段”。
        """
        action_ids = getattr(self.qwen_vl_interface, "fast_action_token_ids", None)
        if action_ids is not None:
            ids = torch.tensor(sorted(action_ids), device=generated_ids.device)
            mask = torch.isin(generated_ids, ids)
        else:
            act_min = self.action_model._ACTION_TOKEN_MIN
            act_max = self.action_model._ACTION_TOKEN_MAX
            mask = (generated_ids >= act_min) & (generated_ids <= act_max)  # [B, L]
        results = []
        for b in range(generated_ids.size(0)):
            idx = mask[b].nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                results.append([])
                continue
            # 全部动作 token
            tokens = generated_ids[b, idx].tolist()
            results.append(tokens)
        return results

    def _decode_action_tokens(self, batch_vlm_tokens: List[List[int]]) -> List[Any]:
        """
        将带偏移的 VLM 动作 token 列表解码回 fast tokenizer 语义。
        fast_tokenizer.decode 预期输入：原始 fast token id 序列（未加偏移）。
        """
        act_min = self.action_model._ACTION_TOKEN_MIN
        batch_fast_token_ids = []
        for seq in batch_vlm_tokens:
            if not seq:
                batch_fast_token_ids.append(None)
                continue
            fast_ids = [t - act_min for t in seq]

            batch_fast_token_ids.append(fast_ids)

        return batch_fast_token_ids
