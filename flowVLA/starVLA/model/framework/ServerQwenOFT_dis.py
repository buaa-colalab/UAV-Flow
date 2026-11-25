"""
Real-time QwenOFT action server.
Loads the Qwenvl_OFT model, listens for HTTP requests, and returns UAV actions.
"""

import os
import sys
import time
import json
import torch
import logging
import base64
import numpy as np
from io import BytesIO
from pathlib import Path
from PIL import Image
from flask import Flask, request, jsonify
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONFIG_YAML = REPO_ROOT / "starVLA" / "config" / "training" / "qwen3vl_discrete.yaml"
log = logging.getLogger(__name__)


def _resolve_path(path):
    return Path(path).expanduser().resolve()


def _extract_action_range(stats):
    candidates = [
        ("action_min", "action_max"),
        ("min", "max"),
        ("q01", "q99"),
    ]

    for low_key, high_key in candidates:
        if low_key in stats and high_key in stats:
            return np.asarray(stats[low_key], dtype=np.float32), np.asarray(stats[high_key], dtype=np.float32)

    action_stats = stats.get("action")
    if isinstance(action_stats, dict):
        return _extract_action_range(action_stats)

    raise KeyError("stats file must contain action_min/action_max, min/max, or q01/q99.")


def _episode_actions_from_log(log_path):
    with log_path.open("r", encoding="utf-8") as handle:
        episode_data = json.load(handle)

    trajectory = np.asarray(episode_data["raw_logs"], dtype=np.float64)[:, [0, 1, 2, 4]]
    if trajectory.shape[0] == 0:
        return None

    trajectory[:, 3] = np.deg2rad(trajectory[:, 3])
    actions = np.zeros((trajectory.shape[0], 4), dtype=np.float64)

    if trajectory.shape[0] > 1:
        current = trajectory[:-1]
        next_pose = trajectory[1:]
        relative_pos = next_pose[:, :3] - current[:, :3]
        cos_yaw = np.cos(current[:, 3])
        sin_yaw = np.sin(current[:, 3])

        actions[:-1, 0] = cos_yaw * relative_pos[:, 0] + sin_yaw * relative_pos[:, 1]
        actions[:-1, 1] = -sin_yaw * relative_pos[:, 0] + cos_yaw * relative_pos[:, 1]
        actions[:-1, 2] = relative_pos[:, 2]

        relative_yaw = next_pose[:, 3] - current[:, 3]
        actions[:-1, 3] = (relative_yaw + np.pi) % (2 * np.pi) - np.pi

    return actions


def compute_action_stats_from_dataset(data_root, lower_percentile=1.0, upper_percentile=99.0):
    data_root = _resolve_path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Training data root not found: {data_root}")

    log.info(
        "Computing action percentiles from %s (p%s/p%s). This can take a while.",
        data_root,
        lower_percentile,
        upper_percentile,
    )

    all_actions = []
    skipped_episodes = 0
    for index, episode_dir in enumerate(sorted(data_root.iterdir()), start=1):
        if not episode_dir.is_dir():
            continue

        log_path = episode_dir / "log.json"
        if not log_path.exists():
            skipped_episodes += 1
            continue

        try:
            episode_actions = _episode_actions_from_log(log_path)
        except Exception as exc:
            skipped_episodes += 1
            log.warning("Failed to read action stats from %s: %s", log_path, exc)
            continue

        if episode_actions is None:
            skipped_episodes += 1
            continue

        all_actions.append(episode_actions)

        if index % 1000 == 0:
            log.info(
                "Scanned %s episode dirs for action stats; valid=%s, skipped=%s.",
                index,
                len(all_actions),
                skipped_episodes,
            )

    if not all_actions:
        raise RuntimeError(f"No valid episodes found under {data_root}")

    actions = np.concatenate(all_actions, axis=0)
    return {
        "data_root": str(data_root),
        "num_episodes": len(all_actions),
        "num_actions": int(actions.shape[0]),
        "skipped_episodes": skipped_episodes,
        "percentiles": [float(lower_percentile), float(upper_percentile)],
        "action_mean": np.mean(actions, axis=0).astype(float).tolist(),
        "action_std": np.std(actions, axis=0).astype(float).tolist(),
        "action_min": np.percentile(actions, lower_percentile, axis=0).astype(float).tolist(),
        "action_max": np.percentile(actions, upper_percentile, axis=0).astype(float).tolist(),
    }


class QwenOFTServer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.gpu_id = cfg.get("gpu_id", 0)
        self.device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")

        # Load model
        from starVLA.model.framework.QwenOFT_discrete import Qwenvl_OFT

        log.info("🔹 Loading QwenOFT model...")

        self.omega_cfg = OmegaConf.load(cfg["config_yaml"])
        if self.omega_cfg.framework.qwenvl.get("model_type") == "qwen3_5":
            raise ValueError("Use the Qwen3.5 server for learned queries and physical codebook lookup")
        if self.omega_cfg.framework.action_model.get("action_head_type") != "discrete":
            raise ValueError("This server requires a discrete action head; use the continuous server for L1 regression")
        expected_layers = cfg.get("expected_all_layers")
        if (
            expected_layers is not None
            and bool(self.omega_cfg.framework.get("use_all_layer_hidden", False)) != expected_layers
        ):
            raise ValueError(f"Selected server requires use_all_layer_hidden={expected_layers}")
        if cfg.get("base_vlm"):
            self.omega_cfg.framework.qwenvl.base_vlm = cfg["base_vlm"]
        if cfg.get("train_data_root"):
            self.omega_cfg.datasets.vla_data.data_root_dir = cfg["train_data_root"]

        self.model = Qwenvl_OFT(self.omega_cfg)
        if getattr(self.omega_cfg.trainer, "use_lora", False):
            from peft import LoraConfig, get_peft_model

            lora_cfg = LoraConfig(
                r=self.omega_cfg.trainer.lora.r,
                lora_alpha=self.omega_cfg.trainer.lora.lora_alpha,
                target_modules=self.omega_cfg.trainer.lora.target_modules,
                lora_dropout=self.omega_cfg.trainer.lora.lora_dropout,
                init_lora_weights="gaussian",
            )

            vlm = self.model.qwen_vl_interface.model.model
            self.model.qwen_vl_interface.model.model = get_peft_model(vlm, lora_cfg)

        checkpoint = torch.load(cfg["checkpoint_path"], map_location="cpu", weights_only=True, mmap=True)
        self.model.load_state_dict(checkpoint, strict=True)
        self.model = self.model.to(self.device)
        self.model.eval()

        log.info(f"✅ Model loaded from {cfg['checkpoint_path']}")

        # Setup HTTP server
        self.app = Flask(__name__)
        self.port = cfg.get("http_port", 6000)
        self.setup_routes()

        self.init_img = None

        self.action_min, self.action_max = self.load_action_stats(cfg)

    def load_action_stats(self, cfg):
        stats_path = cfg.get("stats_path")
        if stats_path:
            stats_path = _resolve_path(stats_path)
            if stats_path.exists():
                with stats_path.open("r", encoding="utf-8") as handle:
                    stats = json.load(handle)
                action_min, action_max = _extract_action_range(stats)
                log.info("Loaded action normalization stats from %s.", stats_path)
                log.info("action_min=%s", action_min)
                log.info("action_max=%s", action_max)
                return action_min, action_max

        if not cfg.get("recompute_action_stats", False):
            raise ValueError(
                "Provide STATS_PATH matching this discrete checkpoint, or explicitly recompute statistics"
            )

        train_data_root = cfg.get("train_data_root")
        if train_data_root is None:
            train_data_root = OmegaConf.select(self.omega_cfg, "datasets.vla_data.data_root_dir")

        if train_data_root:
            stats = compute_action_stats_from_dataset(
                train_data_root,
                lower_percentile=cfg.get("lower_percentile", 1.0),
                upper_percentile=cfg.get("upper_percentile", 99.0),
            )
            if stats_path:
                try:
                    stats_path.parent.mkdir(parents=True, exist_ok=True)
                    with stats_path.open("w", encoding="utf-8") as handle:
                        json.dump(stats, handle, indent=2)
                    log.info("Cached action normalization stats to %s.", stats_path)
                except OSError as exc:
                    log.warning("Could not cache action stats to %s: %s", stats_path, exc)

            action_min = np.asarray(stats["action_min"], dtype=np.float32)
            action_max = np.asarray(stats["action_max"], dtype=np.float32)
            log.info("Computed action normalization stats from %s.", train_data_root)
            log.info("action_min=%s", action_min)
            log.info("action_max=%s", action_max)
            return action_min, action_max

        raise ValueError("Either stats_path must exist or train_data_root must be configured.")

    def setup_routes(self):

        @self.app.route("/reset", methods=["POST"])
        def reset():
            self.init_img = None
            return jsonify({"status": "success", "message": "状态已重置"})

        @self.app.route("/predict", methods=["POST"])
        def predict():
            # try:
            data = request.json

            # Decode image (support multiple views)
            images = []

            img_bytes = base64.b64decode(data["image"])
            cur_image = Image.open(BytesIO(img_bytes))
            # Decode the image fully without writing it to disk.
            cur_image.load()

            if self.init_img is None:
                self.init_img = cur_image

            images = [self.init_img, cur_image]

            ori_proprio = np.array(data.get("proprio", []), dtype=np.float32)
            proprio = ",".join([str(round(x, 1)) for x in ori_proprio])

            instruction = f"Current State: {proprio}, What action should the uav take to {data['instr']}?"

            start = time.time()
            with torch.inference_mode():
                pred_action_ori = self.predict_action(images, instruction)
            end = time.time()

            # log.info(f"🧠 Inference done in {end - start:.3f}s")
            # log.info(f"Predicted normalized action: {pred_action_ori}")

            # Denormalize to real range
            pred_action = self.denormalize(pred_action_ori)

            # print(pred_action_ori, pred_action, instruction)

            # last_action = pred_action[5]
            # done = False
            # if all(abs(d) < 3.0 for d in last_action[:3]) and abs(np.degrees(last_action[3])) < 1.0:
            #     done = True

            if True:
                pred_action = self.apply_action_transform(pred_action, ori_proprio)

            # pred_action = pred_action[:5]
            # pred_action_ori = pred_action_ori[:5]
            return jsonify(
                {
                    "status": "success",
                    "action": pred_action.tolist(),
                    "normalized_action": pred_action_ori.tolist(),
                    # "done":done,
                    "message": "Action generated successfully",
                }
            )

        # except Exception as e:
        #     import traceback
        #     log.error(traceback.format_exc())
        #     return jsonify({
        #         "status": "error",
        #         "message": str(e) + "\n" + traceback.format_exc()
        #     }), 500

    def denormalize(self, norm_act):
        """Convert normalized action [-1,1] to original range."""
        norm_act = np.array(norm_act)
        return (norm_act + 1) / 2 * (self.action_max - self.action_min) + self.action_min

    def apply_action_transform(self, pred_action, proprio):
        current_yaw = np.deg2rad(proprio[-1])
        current_pos = proprio[0:3]
        cos_yaw, sin_yaw = np.cos(current_yaw), np.sin(current_yaw)
        R = np.array([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]])

        for i in range(pred_action.shape[0]):
            pred_action[i, 0:3] = R @ pred_action[i, 0:3]
            pred_action[i, 0:3] = current_pos + pred_action[i, 0:3]
            pred_action[i, -1] += current_yaw
            current_yaw = pred_action[i, -1]
            current_pos = pred_action[i, 0:3]
            cos_yaw, sin_yaw = np.cos(current_yaw), np.sin(current_yaw)
            R = np.array([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]])
        return pred_action

    def predict_action(self, images, instruction):
        """Run inference on image + instruction"""
        output = self.model.predict_action(batch_images=[images], instructions=[instruction], state=None)
        pred = output["normalized_actions"]
        return pred[0] if isinstance(pred, (list, np.ndarray)) else pred

    def run(self):
        log.info(f"🚀 Starting QwenOFT Server on port {self.port}")
        self.app.run(host="0.0.0.0", port=self.port, debug=False)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    expected_layers = os.environ.get("EXPECTED_ALL_LAYERS")
    if expected_layers not in {None, "0", "1"}:
        raise ValueError("EXPECTED_ALL_LAYERS must be 0 or 1")
    cfg = {
        "expected_all_layers": None if expected_layers is None else expected_layers == "1",
        "gpu_id": int(os.environ.get("GPU_ID", 0)),
        "http_port": int(os.environ.get("HTTP_PORT", 6067)),
        "config_yaml": os.environ.get("CONFIG_YAML", str(DEFAULT_CONFIG_YAML)),
        "base_vlm": os.environ.get("BASE_VLM"),
        "checkpoint_path": os.environ["CHECKPOINT_PATH"],
        "train_data_root": os.environ.get("TRAIN_DATA_ROOT"),
        "stats_path": os.environ.get("STATS_PATH"),
        "recompute_action_stats": os.environ.get("RECOMPUTE_ACTION_STATS", "0") == "1",
        "lower_percentile": float(os.environ.get("LOWER_PERCENTILE", 1.0)),
        "upper_percentile": float(os.environ.get("UPPER_PERCENTILE", 99.0)),
    }

    print(cfg["checkpoint_path"])

    server = QwenOFTServer(cfg)
    server.run()


if __name__ == "__main__":
    main()
