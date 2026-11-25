# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""Fast Action Tokenizer Adapter
"this file is adapted from https://huggingface.co/physical-intelligence/fast"

Overview:
    This module encapsulates a lightweight "action → language model-readable sequence" converter (Fast_Action_Tokenizer).
    Its core objective is to convert continuous/discrete raw robot actions (raw_actions) into
    pseudo-natural language token strings like <robot_action_12><robot_action_3><robot_action_87> ...
    This facilitates direct integration into multimodal large models (VLM/LLM) dialogue templates,
    leveraging their language modeling capabilities for action prediction.
"""

import torch.nn as nn
from typing import List, Dict, Any, Callable, Optional
import os
import numpy as np
from transformers import AutoProcessor
from transformers.utils.hub import cached_file

from starVLA.model.modules.vlm.QWen2_5 import _ACTION_TOKEN_MAX, _ACTION_TOKEN_MIN


class Fast_Action_Tokenizer(nn.Module):
    """One MLP ResNet block with a residual connection."""

    def __init__(
        self, fast_tokenizer_name="playground/Pretrained_models/fast", action_token_min=None, action_token_max=None
    ):
        super().__init__()

        # FAST stores its BPE tokenizer at the processor root. Transformers 5
        # otherwise interprets bpe_tokenizer as a separate tokenizer subfolder.
        tokenizer_file = cached_file(fast_tokenizer_name, "tokenizer.json")
        self.fast_tokenizer = AutoProcessor.from_pretrained(
            fast_tokenizer_name, trust_remote_code=True, tokenizer_file=tokenizer_file
        )

        # Qwen2.5 checkpoints historically used the constants above.  Qwen3
        # has a different base vocabulary, so QwenFast sets these values after
        # registering its action tokens.  Keeping the fallback preserves the
        # old QwenFast behaviour.
        self._ACTION_TOKEN_MIN = int(action_token_min if action_token_min is not None else _ACTION_TOKEN_MIN)
        self._ACTION_TOKEN_MAX = int(action_token_max if action_token_max is not None else _ACTION_TOKEN_MAX)
        self.fast_action_token_ids = None

    def encoder_action2vlmtoken(self, raw_actions):
        # x: (batch_size, chunck, dim)
        batch_actions = np.stack(raw_actions, axis=0)  # (B, T, D)
        batch_fast_tokens = self.fast_tokenizer(batch_actions)
        # batch_fast_tokens = [self.fast_tokenizer(raw_action)[0] for raw_action in raw_actions]
        batch_vlm_actions = [self.map_fast_token_to_vlm_action(fast_tokens) for fast_tokens in batch_fast_tokens]
        return batch_vlm_actions  # List[str]

    def decoder_action(self, generated_ids):
        # api https://huggingface.co/physical-intelligence/fast
        # return: (batch_size, chunck, dim)
        pred_actions = self.fast_tokenizer.decode([generated_ids - self._ACTION_TOKEN_MIN])
        return pred_actions

    def map_fast_token_to_vlm_action(self, tokens) -> str:
        """Maps fast action tokens to the VLM action format.
        Action token 0 is mapped to the string <robot_action_0>  ... and so on
        """
        return "".join(
            [f"<robot_action_{token}>" for token in tokens]
        )  # you should add <robot_action_{token}> to VLM as special tokens,
        # and set the
        # see ./starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md

    def fit_tokenizer_on_datasets(
        self,
        action_dataset,
        datasets_path="<your_local_path>",
    ):
        # 如果 datasets_path 存在， 直接读取
        if os.path.exists(datasets_path):

            self.fast_tokenizer = AutoProcessor.from_pretrained(datasets_path, trust_remote_code=True)
            return
        else:
            # 如果不存在，Fit the tokenizer on the new dataset
            new_tokenizer = self.fast_tokenizer.tokenizer.fit(action_dataset)
            self.fast_tokenizer = new_tokenizer

            # Save the new tokenizer, optionally push it to the Hugging Face model hub
            self.fast_tokenizer.save_pretrained(datasets_path)


def get_action_model(config=None):
    """
    Factory: build ActionModel from global framework config.

    Args:
        config: Global config (expects config.framework.action_model namespace).
    Returns:
        ActionModel: Initialized diffusion action head.
    """
    action_cfg = config.framework.action_model if config is not None else None
    tokenizer_name = getattr(action_cfg, "fast_tokenizer_name", "playground/Pretrained_models/fast")
    token_min = getattr(action_cfg, "fast_action_token_min", None)
    token_max = getattr(action_cfg, "fast_action_token_max", None)
    action_model = Fast_Action_Tokenizer(
        fast_tokenizer_name=tokenizer_name,
        action_token_min=token_min,
        action_token_max=token_max,
    )

    return action_model


def start_debugpy_once():
    """start debugpy once"""
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10094))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10094 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":

    start_debugpy_once()

    fast_tokenizer_name = "./playground/Pretrained_models/fast"
    fast_tokenizer = Fast_Action_Tokenizer(fast_tokenizer_name=fast_tokenizer_name)
    raw_actions = [np.random.randn(16, 7), np.random.randn(16, 7)]

    # Load the tokenizer from the Hugging Face hub
    tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_name, trust_remote_code=True)

    # basic test
    # Tokenize & decode action chunks (we use dummy data here)
    action_data = np.random.rand(2, 16, 7)  # one batch of action chunks
    tokens = tokenizer(action_data)  # tokens = list[int]
    decoded_actions = tokenizer.decode(tokens)

    # self func test
    vlm_tokens = fast_tokenizer.encoder_action2vlmtoken(raw_actions)
    print(vlm_tokens)
    pred_actions = fast_tokenizer.decoder_action(np.array([12, 3, 45, 87]))
    print(pred_actions)
