"""Qwen3.5 learned-query UAV action HTTP service (same wire format as QwenOFT)."""

import base64
import binascii
import json
import logging
import os
import threading
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request
from omegaconf import OmegaConf
from PIL import Image, UnidentifiedImageError

from starVLA.dataloader.uav_prompts import format_uav_instruction

log = logging.getLogger(__name__)


def load_action_range(path):
    stats = json.loads(Path(path).read_text())
    low, high = (np.asarray(stats[k], dtype=np.float64) for k in ("action_min", "action_max"))
    if low.shape != (4,) or high.shape != (4,) or not np.isfinite([low, high]).all() or (high <= low).any():
        raise ValueError("Expected four finite action_min/action_max values with max > min")
    expected = ["local_dx", "local_dy", "local_dz", "local_dyaw_radians"]
    if stats.get("action_order", expected) != expected:
        raise ValueError("Incompatible normalization action_order")
    return low, high


def integrate_actions(local_actions, proprio):
    """Input state yaw is degrees; output waypoint yaw is radians, as in the old API."""
    result = np.asarray(local_actions, dtype=np.float64).copy()
    position = np.asarray(proprio[:3], dtype=np.float64).copy()
    yaw = np.deg2rad(proprio[3])
    for action in result:
        c, s = np.cos(yaw), np.sin(yaw)
        position += [c * action[0] - s * action[1], s * action[0] + c * action[1], action[2]]
        yaw += action[3]
        action[:3], action[3] = position, yaw
    return result


def decode_image(encoded, image_size):
    if not isinstance(encoded, str):
        raise ValueError("image must be a base64 string")
    try:
        with Image.open(BytesIO(base64.b64decode(encoded, validate=True))) as image:
            image.load()
            image = image.convert("RGB")
    except (ValueError, binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Invalid base64 image") from exc
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return image


class Qwen35OFTServer:
    def __init__(
        self, run_dir, checkpoint_path, *, gpu_id=0, max_gpu_memory_gib=6.25,
        stats_path=None, base_vlm=None, config_yaml=None,
    ):
        self.run_dir = Path(run_dir).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        if not stats_path:
            raise ValueError("Provide stats_path (STATS_PATH) pointing to the statistics JSON used for training")
        stats_path = Path(stats_path).expanduser().resolve()
        self.action_min, self.action_max = load_action_range(stats_path)
        cfg = OmegaConf.load(config_yaml or self.run_dir / "config.yaml")
        if base_vlm:
            cfg.framework.qwenvl.base_vlm = base_vlm
        if cfg.framework.get("action_codec") is not None:
            cfg.datasets.vla_data.normalization_stats_path = str(stats_path)
            from starVLA.model.action_codec.action_codebook import load_codec_from_config

            # Rebuild the trained table from statistics; checkpoint extra state
            # retains the original manifest/hash contract for strict loading.
            load_codec_from_config(cfg)
        if (
            cfg.framework.get("action_query_type") != "learned_embeddings"
            or cfg.framework.qwenvl.get("model_type") != "qwen3_5"
        ):
            raise ValueError("Expected a Qwen3.5 learned-query training run")
        cfg.trainer.enable_gradient_checkpointing = False
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3.5 OFT currently requires CUDA")
        torch.cuda.set_device(gpu_id)
        if max_gpu_memory_gib > 0:
            total = torch.cuda.get_device_properties(gpu_id).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, max_gpu_memory_gib * 1024**3 / total), gpu_id)
        torch.manual_seed(int(cfg.get("seed", 42)))
        from starVLA.model.framework.QwenOFT_discrete import Qwenvl_OFT

        self.model = Qwenvl_OFT(cfg)
        if self.model.action_codec is not None:
            codec = self.model.action_codec
            stats = json.loads(stats_path.read_text())
            if not np.array_equal(self.action_min, codec.low) or not np.array_equal(self.action_max, codec.high):
                raise ValueError("Inference normalization differs from checkpoint codebook")
            if stats.get("action_codec_sha256", codec.sha256) != codec.sha256:
                raise ValueError("Inference statistics/codec checksum mismatch")
        # Keep the exact PEFT nesting and targets used by training, including vision LoRA.
        if cfg.trainer.use_lora:
            from starVLA.model.adapters import apply_backbone_lora

            apply_backbone_lora(self.model, cfg.trainer.lora)
        state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
        query_key = "qwen_vl_interface.action_query_embeddings"
        if query_key not in state or tuple(state[query_key].shape) != (10, 2048):
            raise ValueError("Checkpoint must contain the trained [10,2048] Query Embedding")
        self.model.load_state_dict(state, strict=True)
        del state
        self.model.to(torch.device("cuda", gpu_id)).eval().requires_grad_(False)
        self.info = {
            "run_dir": str(self.run_dir),
            "checkpoint": str(self.checkpoint_path),
            "base_model": cfg.framework.qwenvl.base_vlm,
            "stats_path": str(stats_path),
            "query_count": 10,
            "action_shape": [10, 4],
            "input_yaw_unit": "degrees",
            "output_yaw_unit": "radians",
            "image_size": 256,
            "action_frame": "input proprio reference frame; sequentially integrated local deltas",
            "strict_checkpoint_load": True,
            "action_codec_version": (
                self.model.action_codec.manifest["version"] if self.model.action_codec is not None else "legacy_floor"
            ),
            "action_codec_sha256": self.model.action_codec.sha256 if self.model.action_codec is not None else None,
        }
        self.app = self.create_app()
        log.info("Strictly loaded checkpoint: %s", self.checkpoint_path)

    def create_app(self):
        app = Flask(__name__)
        app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
        sessions = {}
        lock = threading.Lock()

        @app.get("/health")
        def health():
            return jsonify(status="ready", **self.info)

        def session_id(data):
            value = data.get("session_id", "default")
            if not isinstance(value, str) or not 1 <= len(value) <= 128:
                raise ValueError("session_id must be a nonempty string of at most 128 characters")
            return value

        @app.post("/reset")
        def reset():
            data = request.get_json(silent=True)
            if data is None:
                data = {}
            if not isinstance(data, dict):
                return jsonify(status="error", message="Expected a JSON object"), 400
            try:
                sid = session_id(data)
            except ValueError as exc:
                return jsonify(status="error", message=str(exc)), 400
            with lock:
                sessions.pop(sid, None)
            return jsonify(status="success", message="State reset", session_id=sid)

        @app.post("/predict")
        def predict():
            try:
                data = request.get_json(silent=True)
                if not isinstance(data, dict):
                    raise ValueError("Expected a JSON object")
                sid = session_id(data)
                if not isinstance(data.get("instr"), str) or not data["instr"].strip():
                    raise ValueError("instr must be a nonempty task instruction")
                proprio = np.asarray(data.get("proprio", []), dtype=np.float64)
                if proprio.shape != (4,) or not np.isfinite(proprio).all():
                    raise ValueError("proprio must be finite [x,y,z,yaw_degrees] in the episode reference frame")
                current = decode_image(data.get("image"), self.info["image_size"])
                first = decode_image(data["initial_image"], self.info["image_size"]) if "initial_image" in data else None
            except (ValueError, TypeError) as exc:
                return jsonify(status="error", message=str(exc)), 400
            with lock:
                # initial_image allows stateless requests; otherwise preserve the first frame until /reset.
                initial = first if first is not None else sessions.get(sid, current)
                if first is None and sid not in sessions and len(sessions) >= 128:
                    return jsonify(status="error", message="Session limit reached; reset unused sessions"), 429
                instruction = format_uav_instruction(proprio, data["instr"])
                started = time.perf_counter()
                try:
                    with torch.inference_mode():
                        output = self.model.predict_action(
                            batch_images=[[initial, current]], instructions=[instruction], state=None
                        )
                    normalized = np.asarray(output["normalized_actions"][0], dtype=np.float64)
                    if (
                        normalized.shape != (10, 4)
                        or not np.isfinite(normalized).all()
                        or (np.abs(normalized) > 1.000001).any()
                    ):
                        raise RuntimeError("Model returned invalid normalized actions")
                    if self.info.get("action_codec_version", "legacy_floor") != "legacy_floor":
                        # Physical lookup is authoritative: an exact zero must
                        # not pass through a lossy normalized FP32 round trip.
                        local = np.asarray(output["local_actions"][0], dtype=np.float64)
                        if local.shape != (10, 4) or not np.isfinite(local).all():
                            raise RuntimeError("Model returned invalid physical actions")
                    else:
                        local = (normalized + 1) / 2 * (self.action_max - self.action_min) + self.action_min
                    action = integrate_actions(local, proprio)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    log.exception("Inference GPU memory budget exhausted")
                    return jsonify(status="error", message="Inference GPU memory budget exhausted"), 503
                except Exception:
                    log.exception("Inference failed")
                    return jsonify(status="error", message="Inference failed; see server log"), 500
                if first is None:
                    sessions[sid] = initial
                return jsonify(
                    status="success",
                    action=action.tolist(),
                    normalized_action=normalized.tolist(),
                    local_action=local.tolist(),
                    inference_seconds=time.perf_counter() - started,
                    message="Action generated successfully",
                )

        return app


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_dir = Path(os.environ["RUN_DIR"])
    checkpoint = os.environ.get("CHECKPOINT_PATH")
    if not checkpoint:
        raise ValueError("Set CHECKPOINT_PATH or use the server launcher to select a final/latest checkpoint")
    server = Qwen35OFTServer(
        run_dir,
        checkpoint,
        gpu_id=int(os.environ.get("GPU_ID", "0")),
        max_gpu_memory_gib=float(os.environ.get("MAX_GPU_MEMORY_GIB", "6.25")),
        stats_path=os.environ.get("STATS_PATH"),
        base_vlm=os.environ.get("BASE_VLM"),
        config_yaml=os.environ.get("CONFIG_YAML"),
    )
    server.app.run(
        host=os.environ.get("HTTP_HOST", "0.0.0.0"),
        port=int(os.environ.get("HTTP_PORT", "6070")),
        debug=False,
        use_reloader=False,
        threaded=True,
    )


if __name__ == "__main__":
    main()
