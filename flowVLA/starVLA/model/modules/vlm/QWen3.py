# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import torch
from typing import Optional, List
from PIL import Image
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# from transformers import AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Dict, Optional, List
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchFeature

from qwen_vl_utils import process_vision_info


from accelerate.logging import get_logger

logger = get_logger(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

_ACTION_TOKEN_MIN = (
    151665  # how can we know this range? --> we has other way for this, but is slower see qwenhelix branch
)
_ACTION_TOKEN_MAX = (
    153712  # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
)


import torch.nn as nn
from pathlib import Path


class _QWen3_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen3-VL (Qwen3VLForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen3-VL wrapper.
        Following https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3-VL-4B-Instruct")

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=qwenvl_config.get("attn_implementation", "flash_attention_2"),
            dtype=torch.bfloat16,
            device_map="cuda",
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config
        if config.trainer.get("enable_gradient_checkpointing", False):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()

        # Fast action tokens are optional and isolated to the QwenFast
        # framework.  The existing QwenOFT/L1 paths keep their original
        # tokenizer and vocabulary untouched.
        if config.framework.get("name", None) == "QwenFast":
            self._setup_fast_action_tokens()


        # alin qwen3 with qwen2.5
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

    def _setup_fast_action_tokens(self):
        action_cfg = self.config.framework.action_model
        token_file = action_cfg.get("fast_action_tokens_file", None)
        if token_file is None:
            token_file = (
                Path(__file__).resolve().parents[3]
                / "model"
                / "modules"
                / "vlm"
                / "tools"
                / "add_qwen_special_tokens"
                / "fast_tokens.txt"
            )
        token_file = Path(str(token_file)).expanduser()
        if not token_file.is_absolute():
            token_file = Path(__file__).resolve().parents[3] / token_file
        if not token_file.is_file():
            raise FileNotFoundError(f"Fast action token file does not exist: {token_file}")
        with token_file.open("r", encoding="utf-8") as handle:
            action_tokens = [line.strip() for line in handle if line.strip()]
        if not action_tokens:
            raise ValueError(f"Fast action token file is empty: {token_file}")

        tokenizer = self.processor.tokenizer
        vocabulary = tokenizer.get_vocab()
        missing = [token for token in action_tokens if token not in vocabulary]
        if missing:
            old_size = self.model.get_input_embeddings().weight.shape[0]
            tokenizer.add_special_tokens({"additional_special_tokens": missing})
            self.model.resize_token_embeddings(len(tokenizer))
            # Keep newly introduced action embeddings finite and deterministic.
            with torch.no_grad():
                embedding = self.model.get_input_embeddings().weight
                embedding[old_size:].normal_(mean=0.0, std=0.02)

        token_ids = [int(tokenizer.convert_tokens_to_ids(token)) for token in action_tokens]
        action_cfg.fast_action_token_min = min(token_ids)
        action_cfg.fast_action_token_max = max(token_ids)
        action_cfg.fast_action_token_ids = token_ids
        self.fast_action_token_ids = set(token_ids)
        self.fast_action_token_min = min(token_ids)
        self.fast_action_token_max = max(token_ids)


    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen2.5-VL backbone.
        """

        with torch.autocast("cuda", dtype=torch.bfloat16):
            capture_hidden = kwargs.get(
                "output_hidden_states", getattr(getattr(self.model, "config", None), "output_hidden_states", False)
            )
            if capture_hidden:
                return self._forward_legacy_hidden_states(**kwargs)
            return self.model(**kwargs)

    def _forward_legacy_hidden_states(self, **kwargs):
        """Keep the Transformers 4.57 feature contract used by existing action heads.

        Capture each decoder input (including DeepStack additions), followed by
        the last decoder output before RMSNorm. Transformers 5 returns different
        hidden states even when the decoder's numerical computation is identical.
        """
        backbone = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        layers = backbone.model.language_model.layers
        hidden_states = []
        handles = []
        return_dict = kwargs.pop("return_dict", True)
        try:
            for layer in layers:
                handles.append(layer.register_forward_pre_hook(lambda module, args: hidden_states.append(args[0])))
            handles.append(
                layers[-1].register_forward_hook(lambda module, args, output: hidden_states.append(output))
            )
            outputs = self.model(**kwargs, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        outputs.hidden_states = tuple(hidden_states)
        return outputs if return_dict else outputs.to_tuple()

    def generate(
        self,
        **kwargs,
    ):
        """
        High-level generation interface (auto-regressive decoding), optionally vision-conditioned.

        Args:
            **kwargs: fully follow raw model.generate() signature.
        Returns:
            GenerateOutput | Model-dependent generation return.
        """
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        """
        Build model inputs from raw data (images + instructions + optional solutions).
        Follow Oficial Qwen3-VL Instruct format: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
        """


        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        # import ipdb;ipdb.set_trace()
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            if "CoT_prompt" in self.config.datasets.vla_data:  # If using a grounding prompt to task
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        # Preparation for inference

        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=solutions is None,
            return_dict=True,
            return_tensors="pt",
        )

        # import ipdb;ipdb.set_trace()

        # if solutions, mask out the solution tokens in labels
        if solutions is not None:  #  here only for fast_tokenizer now.
            action_token_min = getattr(self, "fast_action_token_min", _ACTION_TOKEN_MIN)
            action_token_max = getattr(self, "fast_action_token_max", _ACTION_TOKEN_MAX)
            labels = batch_inputs["input_ids"].clone()
            # For each sequence in the batch, find the first occurrence of an action token.
            for i in range(labels.size(0)):
                seq = labels[i]
                # Create a mask for tokens within the action token range.
                mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    # Mask out all tokens before the first action token.
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    # If no action token is found, mask the entire sequence.
                    seq[:] = IGNORE_INDEX
                    RuntimeWarning(
                        f"action token are on in yout tokenizer, plz see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md."
                    )

            labels[labels == self.processor.tokenizer.pad_token_id] = -100  ## mask out pad tokens as well
            batch_inputs["labels"] = labels

        return batch_inputs.to(self.model.device)


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="./starVLA/config/training/internvla_cotrain_custom.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
    qwen_vl = _QWen3_VL_Interface(cfg)
    pass
