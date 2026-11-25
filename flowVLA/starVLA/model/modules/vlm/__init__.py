def get_vlm_model(config):
    """Explicit model_type wins; infer old configurations from the model path."""
    cfg = config.framework.qwenvl
    model_type = cfg.get("model_type")
    if model_type is None:
        name = cfg.base_vlm
        if "Qwen3.5" in name:
            model_type = "qwen3_5"
        elif "Qwen2.5-VL" in name:
            model_type = "qwen2_5_vl"
        elif "Qwen3-VL" in name:
            model_type = "qwen3_vl"
    if model_type == "qwen3_5":
        from .QWen3_5 import _QWen3_5_VL_Interface

        return _QWen3_5_VL_Interface(config)
    if model_type == "qwen2_5_vl":
        from .QWen2_5 import _QWen_VL_Interface

        return _QWen_VL_Interface(config)
    if model_type == "qwen3_vl":
        from .QWen3 import _QWen3_VL_Interface

        return _QWen3_VL_Interface(config)
    raise NotImplementedError(f"VLM model_type={model_type!r}, base_vlm={cfg.base_vlm!r} not implemented")
