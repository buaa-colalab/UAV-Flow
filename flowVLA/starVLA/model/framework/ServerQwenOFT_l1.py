"""HTTP inference server for the continuous QwenOFT L1 action head.

The request/response contract intentionally matches ``ServerQwenOFT_dis.py``:

* ``POST /reset`` starts a new episode and clears the cached first image.
* ``POST /predict`` accepts ``image`` (base64), ``proprio`` and ``instr``.

Unlike the legacy server, this module loads the exact ``QwenOFT_continuous`` model
used by ``framework.name=QwenOFT-L1`` and treats its output as a continuous
normalized action instead of discrete action-bin logits.
"""

from __future__ import annotations

import base64
import gc
import json
import logging
import os
import re
import sys
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flask import Flask, jsonify, request
from omegaconf import OmegaConf
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("qwen_oft_l1_server")


def _resolve_path(value: str | os.PathLike[str]) -> Path:
    return Path(value).expanduser().resolve()


def _extract_action_range(stats: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    for low_key, high_key in (
        ("action_min", "action_max"),
        ("min", "max"),
        ("q01", "q99"),
    ):
        if low_key in stats and high_key in stats:
            low = np.asarray(stats[low_key], dtype=np.float32)
            high = np.asarray(stats[high_key], dtype=np.float32)
            break
    else:
        nested = stats.get("action")
        if not isinstance(nested, dict):
            raise KeyError("Stats must contain action_min/action_max, min/max, q01/q99, " "or a nested action object.")
        return _extract_action_range(nested)

    if low.shape != (4,) or high.shape != (4,):
        raise ValueError(f"Expected four-dimensional action stats, got {low.shape} and {high.shape}.")
    if not np.isfinite([low, high]).all() or np.any(high <= low):
        raise ValueError("Action bounds must be finite, with each action_max greater than action_min.")
    return low, high


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"steps_(\d+)_pytorch_model\.pt$", path.name)
    return int(match.group(1)) if match else -1


def find_checkpoint(run_dir: Path, explicit_path: str | None = None) -> Path:
    """Resolve an explicit checkpoint or select the final/latest saved model."""
    if explicit_path:
        checkpoint = _resolve_path(explicit_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        return checkpoint

    final_checkpoint = run_dir / "final_model" / "pytorch_model.pt"
    if final_checkpoint.is_file():
        return final_checkpoint.resolve()

    checkpoints = list((run_dir / "checkpoints").glob("steps_*_pytorch_model.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No final_model/pytorch_model.pt or step checkpoint found under {run_dir}.")
    return max(checkpoints, key=_checkpoint_step).resolve()


class QwenOFTL1Server:
    """Single-GPU HTTP server for a deterministic continuous L1 policy."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.gpu_id = int(cfg["gpu_id"])
        if not torch.cuda.is_available():
            raise RuntimeError("QwenOFT-L1 inference requires CUDA on this repository version.")
        if not 0 <= self.gpu_id < torch.cuda.device_count():
            raise ValueError(
                f"GPU_ID={self.gpu_id} is invalid; visible CUDA device count is {torch.cuda.device_count()}."
            )

        # QWen3.py uses device_map="cuda", so set the active CUDA device before
        # constructing the backbone.
        torch.cuda.set_device(self.gpu_id)
        self.device = torch.device("cuda", self.gpu_id)
        self.checkpoint_path = find_checkpoint(cfg["run_dir"], cfg.get("checkpoint_path"))
        self.omega_cfg = self._load_config(cfg)
        self.action_min, self.action_max = self._load_action_stats(cfg.get("stats_path"))
        self.model = self._load_model()

        self.init_img: Image.Image | None = None
        self.inference_lock = threading.Lock()
        self.app = Flask(__name__)
        self.port = int(cfg["http_port"])
        self._setup_routes()

    def _load_config(self, cfg: dict[str, Any]):
        config_path = _resolve_path(cfg["config_yaml"])
        if not config_path.is_file():
            raise FileNotFoundError(f"Training config does not exist: {config_path}")

        omega_cfg = OmegaConf.load(config_path)
        framework_name = OmegaConf.select(omega_cfg, "framework.name")
        framework_py = OmegaConf.select(omega_cfg, "framework.framework_py")
        if framework_name != "QwenOFT-L1" and framework_py != "QwenOFT-L1":
            raise ValueError(
                f"Expected a continuous-head config, got framework.name={framework_name!r}, "
                f"framework.framework_py={framework_py!r}."
            )

        action_dim = int(OmegaConf.select(omega_cfg, "framework.action_model.action_dim", default=-1))
        if action_dim != 4:
            raise ValueError(f"This UAV server expects action_dim=4, got {action_dim}.")

        base_vlm = cfg.get("base_vlm")
        if base_vlm:
            omega_cfg.framework.qwenvl.base_vlm = os.path.expanduser(str(base_vlm))

        log.info("Training config: %s", config_path)
        return omega_cfg

    def _load_action_stats(self, stats_path: str | None) -> tuple[np.ndarray, np.ndarray]:
        if not stats_path:
            raise ValueError("Provide STATS_PATH matching this continuous checkpoint")
        resolved = _resolve_path(stats_path)
        if not resolved.is_file():
            raise FileNotFoundError(f"Action stats file does not exist: {resolved}")
        with resolved.open("r", encoding="utf-8") as handle:
            stats = json.load(handle)
        log.info("Action stats: %s", resolved)
        return _extract_action_range(stats)

    def _load_model(self):
        # This is the implementation selected by build_framework during the
        # QwenOFT-L1 training run. Do not substitute QwenOFT_discrete.py: that file
        # currently contains discrete-bin inference logic.
        from peft import LoraConfig, get_peft_model
        from starVLA.model.framework.QwenOFT_continuous import Qwenvl_OFT

        log.info("Loading QwenOFT-L1 on %s", self.device)
        model = Qwenvl_OFT(self.omega_cfg)

        if bool(OmegaConf.select(self.omega_cfg, "trainer.use_lora", default=False)):
            lora = self.omega_cfg.trainer.lora
            lora_cfg = LoraConfig(
                r=int(lora.r),
                lora_alpha=int(lora.lora_alpha),
                target_modules=list(lora.target_modules),
                lora_dropout=float(lora.lora_dropout),
                init_lora_weights="gaussian",
            )
            vlm = model.qwen_vl_interface.model.model
            model.qwen_vl_interface.model.model = get_peft_model(vlm, lora_cfg)

        # DeepSpeed saved every tensor in this run as bf16. Matching that dtype
        # also keeps the continuous head compatible with the bf16 VLM features.
        model = model.to(device=self.device, dtype=torch.bfloat16)

        log.info("Loading checkpoint: %s", self.checkpoint_path)
        try:
            state_dict = torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        except TypeError:
            # Compatibility with older PyTorch versions.
            state_dict = torch.load(self.checkpoint_path, map_location="cpu")

        # Exact config + exact model implementation should match fully. Strict
        # loading prevents a seemingly healthy server with an uninitialized head.
        model.load_state_dict(state_dict, strict=True)
        del state_dict
        gc.collect()

        model.eval()
        log.info("Model ready: checkpoint=%s, dtype=bf16", self.checkpoint_path)
        return model

    def _setup_routes(self) -> None:
        @self.app.get("/health")
        def health():
            return jsonify(
                {
                    "status": "ok",
                    "model": "QwenOFT-L1",
                    "device": str(self.device),
                    "checkpoint": str(self.checkpoint_path),
                    "action_chunk_length": int(self.model.chunk_len),
                    "action_dim": 4,
                }
            )

        @self.app.post("/reset")
        def reset():
            with self.inference_lock:
                self.init_img = None
            return jsonify({"status": "success", "message": "状态已重置"})

        @self.app.post("/predict")
        def predict():
            try:
                payload = request.get_json(silent=False)
                image, proprio, instruction = self._parse_request(payload)

                start = time.perf_counter()
                with self.inference_lock, torch.inference_mode():
                    if self.init_img is None:
                        self.init_img = image.copy()
                    normalized = self._predict_action([self.init_img, image], instruction)

                raw_min = float(normalized.min())
                raw_max = float(normalized.max())
                clipped = np.clip(normalized, -1.0, 1.0)
                if not np.array_equal(normalized, clipped):
                    log.warning(
                        "Clipped normalized L1 output from range [%.4f, %.4f] to [-1, 1].",
                        raw_min,
                        raw_max,
                    )

                local_actions = self.denormalize(clipped)
                global_actions = self.apply_action_transform(local_actions, proprio)
                elapsed = time.perf_counter() - start
                log.info("Predicted %d actions in %.3fs", global_actions.shape[0], elapsed)

                return jsonify(
                    {
                        "status": "success",
                        "action": global_actions.tolist(),
                        "normalized_action": clipped.tolist(),
                        "message": "Action generated successfully",
                    }
                )
            except (KeyError, TypeError, ValueError) as exc:
                log.warning("Invalid predict request: %s", exc)
                return jsonify({"status": "error", "message": str(exc)}), 400
            except Exception as exc:
                log.exception("Prediction failed")
                return jsonify({"status": "error", "message": str(exc)}), 500

    @staticmethod
    def _parse_request(payload: Any) -> tuple[Image.Image, np.ndarray, str]:
        if not isinstance(payload, dict):
            raise TypeError("Request body must be a JSON object.")

        encoded_image = payload.get("image")
        if not isinstance(encoded_image, str) or not encoded_image:
            raise ValueError("'image' must be a non-empty base64 string.")
        try:
            image_bytes = base64.b64decode(encoded_image, validate=True)
            with Image.open(BytesIO(image_bytes)) as opened:
                image = opened.convert("RGB")
        except Exception as exc:
            raise ValueError(f"Invalid base64 image: {exc}") from exc

        proprio = np.asarray(payload.get("proprio"), dtype=np.float32)
        if proprio.shape != (4,) or not np.isfinite(proprio).all():
            raise ValueError("'proprio' must contain four finite values: [x, y, z, yaw_degrees].")

        target_instruction = payload.get("instr")
        if not isinstance(target_instruction, str) or not target_instruction.strip():
            raise ValueError("'instr' must be a non-empty string.")

        proprio_text = ",".join(str(round(float(value), 1)) for value in proprio)
        instruction = (
            f"Current State: {proprio_text}, What action should the uav take to " f"{target_instruction.strip()}?"
        )
        return image, proprio, instruction

    def _predict_action(self, images: list[Image.Image], instruction: str) -> np.ndarray:
        output = self.model.predict_action(
            batch_images=[images],
            instructions=[instruction],
            state=None,
        )
        normalized = np.asarray(output["normalized_actions"], dtype=np.float32)
        expected_shape = (1, int(self.model.chunk_len), 4)
        if normalized.shape != expected_shape:
            raise RuntimeError(f"Unexpected continuous action shape {normalized.shape}; expected {expected_shape}.")
        if not np.isfinite(normalized).all():
            raise RuntimeError("Model returned NaN or infinite actions.")
        return normalized[0]

    def denormalize(self, normalized_actions: np.ndarray) -> np.ndarray:
        normalized_actions = np.asarray(normalized_actions, dtype=np.float32)
        return (normalized_actions + 1.0) * 0.5 * (self.action_max - self.action_min) + self.action_min

    @staticmethod
    def apply_action_transform(local_actions: np.ndarray, proprio: np.ndarray) -> np.ndarray:
        """Convert the predicted local deltas into global trajectory waypoints."""
        actions = np.asarray(local_actions, dtype=np.float64).copy()
        current_pos = np.asarray(proprio[:3], dtype=np.float64).copy()
        current_yaw = float(np.deg2rad(proprio[3]))

        for action in actions:
            cos_yaw = np.cos(current_yaw)
            sin_yaw = np.sin(current_yaw)
            rotation = np.array(
                [
                    [cos_yaw, -sin_yaw, 0.0],
                    [sin_yaw, cos_yaw, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            action[:3] = current_pos + rotation @ action[:3]
            action[3] += current_yaw
            current_pos = action[:3].copy()
            current_yaw = float(action[3])

        return actions.astype(np.float32)

    def run(self) -> None:
        log.info("Starting QwenOFT-L1 server on 0.0.0.0:%d", self.port)
        # The lock protects model state, but a single request thread also avoids
        # building up queued GPU work if a client sends concurrent commands.
        self.app.run(host="0.0.0.0", port=self.port, debug=False, threaded=False)


def build_runtime_config() -> dict[str, Any]:
    run_dir = _resolve_path(os.environ["RUN_DIR"])
    return {
        "run_dir": run_dir,
        "gpu_id": int(os.environ.get("GPU_ID", "0")),
        "http_port": int(os.environ.get("HTTP_PORT", "6067")),
        "config_yaml": os.environ.get("CONFIG_YAML", str(run_dir / "config.yaml")),
        "base_vlm": os.environ.get("BASE_VLM"),
        "checkpoint_path": os.environ.get("CHECKPOINT_PATH") or None,
        "stats_path": os.environ.get("STATS_PATH") or None,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    server = QwenOFTL1Server(build_runtime_config())
    server.run()


if __name__ == "__main__":
    main()
