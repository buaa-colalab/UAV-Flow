# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""Implementations of various action heads, which serve as alternatives to VLM sequential token prediction."""
"this file is adap from https://github.com/moojink/openvla-oft/blob/main/prismatic/models/action_heads.py"

import torch
import torch.nn as nn


class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""

    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x


class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""

    def __init__(
        self,
        input_dim=2048,
        hidden_dim=4096,
        action_dim=7,
        NUM_ACTIONS_CHUNK=8,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.NUM_ACTIONS_CHUNK = NUM_ACTIONS_CHUNK

        self.model = MLPResNet(num_blocks=2, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=action_dim)

    def predict_action(self, actions_hidden_states):
        """
        actions_hidden_states: (B, chunk_len, hidden_dim)
        返回: (B, chunk_len, action_dim)
        """
        batch_size, chunk_len, hidden_dim = actions_hidden_states.shape
        x = actions_hidden_states.reshape(batch_size * chunk_len, hidden_dim)
        x = self.model(x)  # (B * chunk_len, action_dim)
        actions = x.view(batch_size, chunk_len, self.action_dim)
        return actions

    def forward(self, actions_hidden_states):
        return self.predict_action(actions_hidden_states)


class DiscreteActionHead(nn.Module):
    """
    MLP-based action head that predicts discrete action bins for each action dimension.

    - 输入:  (B, chunk_len, hidden_dim)
    - 输出:  (B, chunk_len, action_dim, num_bins) logits
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 4096,
        action_dim: int = 7,
        NUM_ACTIONS_CHUNK: int = 8,
        num_bins: int = 256,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.NUM_ACTIONS_CHUNK = NUM_ACTIONS_CHUNK
        self.num_bins = num_bins

        # 直接输出 action_dim * num_bins 维度，然后再 reshape
        self.model = MLPResNet(
            num_blocks=2,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim * num_bins,
        )

    def predict_action(self, actions_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions_hidden_states: (B, chunk_len, hidden_dim)

        Returns:
            logits: (B, chunk_len, action_dim, num_bins)
        """
        batch_size, chunk_len, hidden_dim = actions_hidden_states.shape
        x = actions_hidden_states.reshape(batch_size * chunk_len, hidden_dim)
        x = self.model(x)  # (B * chunk_len, action_dim * num_bins)
        x = x.view(batch_size, chunk_len, self.action_dim, self.num_bins)
        return x

    def forward(self, actions_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.predict_action(actions_hidden_states)


def get_action_model(config=None):
    """Build the continuous L1 or discrete action head selected by configuration."""
    action_cfg = config.framework.action_model
    if getattr(action_cfg, "use_gaussian", False):
        raise ValueError("Gaussian action heads are no longer supported")

    head_kwargs = {
        "input_dim": action_cfg.action_hidden_dim,
        "hidden_dim": action_cfg.action_hidden_dim * 2,
        "action_dim": action_cfg.action_dim,
        "NUM_ACTIONS_CHUNK": action_cfg.past_action_window_size + 1 + action_cfg.future_action_window_size,
    }
    if getattr(action_cfg, "action_head_type", None) == "discrete":
        return DiscreteActionHead(**head_kwargs, num_bins=getattr(action_cfg, "num_action_bins", 256))
    return L1RegressionActionHead(**head_kwargs)
