"""Qwen3.5 multimodal backbone for the UAV QwenOFT action frameworks."""

import torch
from torch import nn
from transformers import AutoProcessor


class _QWen3_5_VL_Interface(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        try:
            from transformers import Qwen3_5ForConditionalGeneration
        except ImportError as exc:
            raise ImportError("Install the unified requirements.txt environment to use Qwen3.5") from exc

        self.config = config
        cfg = config.framework.qwenvl
        if config.framework.name != "QwenOFT":
            raise ValueError("Qwen3.5 currently supports the QwenOFT discrete action head")
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            cfg.base_vlm,
            dtype=torch.bfloat16,
            attn_implementation=cfg.get("attn_implementation", "sdpa"),
        )
        self.model.config.hidden_size = self.model.config.text_config.hidden_size
        self.model.config.use_cache = False
        self.model.config.text_config.use_cache = False
        # Accelerate selects the rank-local CUDA device before model construction.
        self.model.to(torch.device("cuda", torch.cuda.current_device()))
        self.processor = AutoProcessor.from_pretrained(cfg.base_vlm)
        self.processor.tokenizer.padding_side = "left"
        if config.trainer.get("enable_gradient_checkpointing", True):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.enable_input_require_grads()

        action_cfg = config.framework.action_model
        self.num_action_queries = int(config.framework.get("num_action_queries", 10))
        chunk_len = action_cfg.past_action_window_size + 1 + action_cfg.future_action_window_size
        if self.num_action_queries != 10 or self.num_action_queries != chunk_len:
            raise ValueError("Learned-query UAV training requires 10 queries matching the action chunk")
        if config.framework.get("action_query_type") != "learned_embeddings":
            raise ValueError(
                "Qwen3.5 requires action_query_type=learned_embeddings; token-query checkpoints are incompatible"
            )
        self.uses_learned_action_queries = True
        # Outside model.model: PEFT freezes that backbone, while these queries
        # must remain trainable and be included in the ordinary optimizer/state_dict.
        self.action_query_embeddings = nn.Parameter(
            torch.empty(
                self.num_action_queries,
                self.model.config.hidden_size,
                device=self.model.device,
                dtype=torch.float32,
            )
        )
        nn.init.normal_(self.action_query_embeddings, mean=0.0, std=self.model.config.text_config.initializer_range)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if prefix + "action_query_embeddings" not in state_dict:
            error_msgs.append(
                "Checkpoint is missing learned action_query_embeddings; token-query weights "
                "cannot restore this model, including with strict=False"
            )
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        if solutions is not None:
            raise ValueError("Qwen3.5 adapter expects OFT action queries, not Fast-token language labels")
        if len(images) != len(instructions):
            raise ValueError("Images and instructions must have the same batch size")
        messages = []
        for views, instruction in zip(images, instructions):
            prompt = self.config.datasets.vla_data.get("CoT_prompt", "{instruction}").replace(
                "{instruction}", instruction
            )
            content = [{"type": "image", "image": image.convert("RGB")} for image in views]
            content.append({"type": "text", "text": prompt})
            messages.append([{"role": "user", "content": content}])
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs.to(self.model.device)

    def _prepare_learned_query_inputs(
        self,
        input_ids,
        attention_mask=None,
        mm_token_type_ids=None,
        image_grid_thw=None,
        video_grid_thw=None,
        position_ids=None,
        **kwargs,
    ):
        if input_ids is None or kwargs.get("inputs_embeds") is not None:
            raise ValueError("Provide tokenized prefix input_ids; the adapter builds inputs_embeds and appends queries")
        if kwargs.get("past_key_values") is not None or kwargs.get("cache_position") is not None:
            raise ValueError("Learned action queries require a fresh full-sequence forward without a KV cache")
        core = self.model.model
        base = core.get_base_model() if hasattr(core, "get_base_model") else core
        embeds = base.get_input_embeddings()(input_ids)
        batch_size, prefix_len, _ = embeds.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if attention_mask.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("Expected a 2D prefix attention mask matching input_ids")
        if not attention_mask[:, -1].bool().all():
            raise ValueError("Learned queries require an unpadded final prefix token (use left padding)")

        if position_ids is None:
            if image_grid_thw is not None or video_grid_thw is not None:
                if mm_token_type_ids is None:
                    raise ValueError("Multimodal RoPE requires the processor's mm_token_type_ids")
                position_ids, _ = base.get_rope_index(
                    input_ids,
                    mm_token_type_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attention_mask,
                )
            else:
                positions = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)
                position_ids = positions.unsqueeze(0).expand(3, -1, -1)
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        if position_ids.shape != (3, batch_size, prefix_len):
            raise ValueError("Expected prefix multimodal position_ids shaped [3, batch, length]")
        # Preserve every original vision/text coordinate. Queries continue as
        # text positions, with the same increasing index along all three axes.
        last_position = position_ids.masked_fill(~attention_mask.bool().unsqueeze(0), -1).amax(dim=(0, 2))
        offsets = torch.arange(1, self.num_action_queries + 1, device=embeds.device)
        query_positions = (last_position[:, None] + offsets).unsqueeze(0).expand(3, -1, -1)
        queries = self.action_query_embeddings.to(dtype=embeds.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        prepared = dict(kwargs)
        prepared.update(
            inputs_embeds=torch.cat((embeds, queries), dim=1),
            attention_mask=torch.cat(
                (attention_mask, attention_mask.new_ones(batch_size, self.num_action_queries)), dim=1
            ),
            position_ids=torch.cat((position_ids, query_positions), dim=-1),
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=False,
        )
        if mm_token_type_ids is not None:
            prepared["mm_token_type_ids"] = torch.cat(
                (mm_token_type_ids, mm_token_type_ids.new_zeros(batch_size, self.num_action_queries)), dim=-1
            )
        # Qwen3.5's native inputs_embeds path replaces the prefix image/video
        # embedding placeholders with visual features. Queries need no token IDs.
        return prepared

    def forward(self, **kwargs):
        prepared = self._prepare_learned_query_inputs(**kwargs)
        return self.model.model(**prepared)

    def generate(self, **kwargs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model.generate(**kwargs)
