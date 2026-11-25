"""Build a selected framework without importing unrelated model dependencies.

Built-in names have stable mappings; explicitly imported extensions can still
register additional names through FRAMEWORK_REGISTRY.
"""

from importlib import import_module

from starVLA.model.tools import FRAMEWORK_REGISTRY

_FRAMEWORKS = {
    "InternVLA-M1": ("M1", "InternVLA_M1"),
    "QwenOFT": ("QwenOFT_discrete", "Qwenvl_OFT"),
    "QwenOFT-L1": ("QwenOFT_continuous", "Qwenvl_OFT"),
    "QwenFast": ("QwenFast", "Qwenvl_Fast"),
    "QwenFM": ("QwenFM", "Qwenvl_FMHead"),
    "QwenPI": ("QwenPI", "Qwen_PI"),
    "Qwen-Dual": ("QwenDual", "Qwen_Dual"),
    "QwenGR00T": ("QwenGR00T", "Qwen_GR00T"),
}


def build_framework(cfg):
    """Resolve a built-in framework or an explicitly registered extension."""
    if not hasattr(cfg.framework, "name"):
        cfg.framework.name = cfg.framework.framework_py

    framework_id = cfg.framework.name
    if framework_id in _FRAMEWORKS:
        module_name, class_name = _FRAMEWORKS[framework_id]
        module = import_module(f".{module_name}", package=__package__)
        return getattr(module, class_name)(cfg)
    if framework_id not in FRAMEWORK_REGISTRY._registry:
        raise NotImplementedError(f"Framework {framework_id} is not implemented.")
    return FRAMEWORK_REGISTRY[framework_id](cfg)


__all__ = ["build_framework", "FRAMEWORK_REGISTRY"]
