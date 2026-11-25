"""Shared OFT LoRA construction, preserving the checkpoint parameter hierarchy."""


def apply_backbone_lora(model, lora_config):
    """Attach LoRA to the multimodal core and return the same framework.

    Learned action queries and the action head stay outside PEFT, as in the
    original training implementation. QwenFast's trainable-token variant uses
    its own adapter because it has a different parameter hierarchy.
    """
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        target_modules=lora_config.target_modules,
        lora_dropout=lora_config.lora_dropout,
        init_lora_weights="gaussian",
    )
    core = model.qwen_vl_interface.model.model
    model.qwen_vl_interface.model.model = get_peft_model(core, config)
    return model
