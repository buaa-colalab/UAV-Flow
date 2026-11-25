"""HTTP inference server for QwenFast UAV policies."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request
from omegaconf import OmegaConf
from PIL import Image

LOG = logging.getLogger("qwen_fast_server")


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"steps_(\d+)_pytorch_model\.pt$", path.name)
    return int(match.group(1)) if match else -1


def find_checkpoint(run_dir: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    final = run_dir / "final_model/pytorch_model.pt"
    if final.is_file():
        return final.resolve()
    checkpoints = list((run_dir / "checkpoints").glob("steps_*_pytorch_model.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found under {run_dir}")
    return max(checkpoints, key=_checkpoint_step).resolve()


class QwenFastServer:
    def __init__(self, cfg: dict):
        if not cfg.get("stats_path"):
            raise ValueError("Provide STATS_PATH matching this FAST checkpoint")
        from starVLA.model.framework.ServerQwenOFT_l1 import _extract_action_range

        self.action_min, self.action_max = _extract_action_range(json.loads(Path(cfg["stats_path"]).read_text()))
        if not torch.cuda.is_available():
            raise RuntimeError("QwenFast inference requires CUDA")
        self.device = torch.device(f"cuda:{cfg['gpu_id']}")
        torch.cuda.set_device(self.device)
        self.checkpoint = find_checkpoint(cfg["run_dir"], cfg.get("checkpoint_path"))
        self.config = OmegaConf.load(cfg["config_yaml"])
        if cfg.get("base_vlm"):
            self.config.framework.qwenvl.base_vlm = cfg["base_vlm"]

        from starVLA.model.framework.QwenFast import Qwenvl_Fast, apply_qwen_fast_lora

        model = Qwenvl_Fast(self.config)
        if bool(OmegaConf.select(self.config, "trainer.use_lora", default=False)):
            model = apply_qwen_fast_lora(model, self.config.trainer.lora)
        model = model.to(self.device, dtype=torch.bfloat16)
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=True, mmap=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        self.model = model
        self.init_img = None
        self.app = Flask(__name__)
        self._routes()

    def _routes(self):
        @self.app.get("/health")
        def health():
            return jsonify(
                {"status": "ok", "model": "QwenFast", "checkpoint": str(self.checkpoint), "device": str(self.device)}
            )

        @self.app.post("/reset")
        def reset():
            self.init_img = None
            return jsonify({"status": "success"})

        @self.app.post("/predict")
        def predict():
            payload = request.get_json(force=True)
            raw = base64.b64decode(payload["image"], validate=True)
            with Image.open(BytesIO(raw)) as opened:
                image = opened.convert("RGB")
            if self.init_img is None:
                self.init_img = image.copy()
            proprio = np.asarray(payload.get("proprio"), dtype=np.float32)
            if proprio.shape != (4,):
                raise ValueError("proprio must be [x,y,z,yaw_degrees]")
            text = ",".join(str(round(float(v), 1)) for v in proprio)
            instruction = f"Current State: {text}, What action should the uav take to {payload['instr']}?"
            with torch.inference_mode():
                output = self.model.predict_action([[self.init_img, image]], [instruction])
            normalized = np.asarray(output["normalized_actions"], dtype=np.float32)
            if normalized.ndim == 3:
                normalized = normalized[0]
            if normalized.ndim != 2 or normalized.shape[-1] != 4:
                raise RuntimeError(f"Unexpected Fast action shape: {normalized.shape}")
            normalized = np.clip(normalized, -1.0, 1.0)
            local = (normalized + 1.0) * 0.5 * (self.action_max - self.action_min) + self.action_min
            global_actions = self._to_global(local, proprio)
            return jsonify(
                {"status": "success", "action": global_actions.tolist(), "normalized_action": normalized.tolist()}
            )

    @staticmethod
    def _to_global(actions, proprio):
        actions = np.asarray(actions, dtype=np.float64).copy()
        pos = np.asarray(proprio[:3], dtype=np.float64).copy()
        yaw = float(np.deg2rad(proprio[3]))
        for action in actions:
            c, s = np.cos(yaw), np.sin(yaw)
            action[:3] = pos + np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ action[:3]
            action[3] += yaw
            pos, yaw = action[:3].copy(), float(action[3])
        return actions.astype(np.float32)


def main():
    logging.basicConfig(level=logging.INFO)
    run_dir = Path(os.environ["RUN_DIR"]).expanduser().resolve()
    cfg = {
        "run_dir": run_dir,
        "checkpoint_path": os.environ.get("CHECKPOINT_PATH"),
        "config_yaml": Path(os.environ.get("CONFIG_YAML", run_dir / "config.yaml")).expanduser().resolve(),
        "base_vlm": os.environ.get("BASE_VLM"),
        "stats_path": os.environ.get("STATS_PATH"),
        "gpu_id": int(os.environ.get("GPU_ID", "0")),
    }
    server = QwenFastServer(cfg)
    server.app.run(host="0.0.0.0", port=int(os.environ.get("HTTP_PORT", "6069")), threaded=False)


if __name__ == "__main__":
    main()
