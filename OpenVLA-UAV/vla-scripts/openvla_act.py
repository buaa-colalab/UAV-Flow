"""
Real-time action execution script. Loads the OpenVLA model, serves requests, and returns actions.
"""

import argparse
import base64
import logging
import time
from io import BytesIO

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

log = logging.getLogger(__name__)


class OpenVLAActionAgent:
    def __init__(self, cfg):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        # Device setup
        self.cfg = cfg
        self.gpu_id = cfg.get("gpu_id", 0)
        self.device = torch.device(f"cuda:{self.gpu_id}")

        # Load model and processor
        self.model_path = cfg.get("model_path")
        log.info(f"Loading model: {self.model_path}")

        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        log.info(f"Processor type: {type(self.processor)}")

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            attn_implementation=cfg.get("attn_implementation", "flash_attention_2"),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        log.info(f"VLA model type: {type(self.model)}")

        self.model.eval()

        self.unnorm_key = cfg.get("unnorm_key", "sim")
        self.do_sample = cfg.get("do_sample", False)
        if self.model.get_action_dim(self.unnorm_key) != 4:
            raise ValueError("UAV evaluation requires a fine-tuned four-action checkpoint")

        # Setup HTTP server
        self.app = Flask(__name__)
        self.setup_routes()
        self.port = cfg.get("http_port", 5000)
        self.host = cfg.get("host", "0.0.0.0")

    def setup_routes(self):
        @self.app.route("/reset", methods=["POST"])
        def reset():
            # Each prediction uses only the supplied observation, with no episode state.
            return jsonify({"status": "success"})

        @self.app.route("/predict", methods=["POST"])
        def predict():
            try:
                data = request.json

                # Decode image
                img_bytes = base64.b64decode(data["image"])
                image = Image.open(BytesIO(img_bytes)).convert("RGB")

                # Retrieve instruction and state
                instruction = data["instr"]
                proprio = np.array(data["proprio"], dtype=np.float32)
                if proprio.shape != (4,) or not np.isfinite(proprio).all():
                    raise ValueError("proprio must be finite [x, y, z, yaw_degrees]")
                proprio_str = ",".join([str(round(x, 1)) for x in proprio])

                # Build prompt
                prompt = f"In: Current State: {proprio_str}, What action should the uav take to {instruction}?\nOut:"
                log.info(f"Prompt: {prompt}")

                start_time = time.time()
                pred_action = np.asarray(self.act(image, prompt))
                if pred_action.shape != (4,) or not np.isfinite(pred_action).all():
                    raise ValueError("Model must return finite [dx, dy, dz, dyaw_radians]")
                end_time = time.time()
                log.info(f"Inference time: {end_time - start_time:.3f} seconds")

                # Save raw prediction
                pred_action = pred_action[None, :]
                pred_action_ori = pred_action.copy()
                log.info(f"Raw predicted action: {pred_action_ori}")

                current_yaw = np.deg2rad(proprio[-1])
                current_pos = proprio[0:3]
                cos_yaw = np.cos(current_yaw)
                sin_yaw = np.sin(current_yaw)
                R = np.array([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]])

                pred_action[0, 0:3] = R @ pred_action[0, 0:3]
                pred_action[0, 0:3] = current_pos + pred_action[0, 0:3]
                pred_action[0, -1] = pred_action[0, -1] + current_yaw

                return jsonify(
                    {
                        "status": "success",
                        "action": pred_action.tolist(),
                        "action_ori": pred_action_ori.tolist(),
                        "message": "Action generated successfully",
                    }
                )

            except Exception as e:
                import traceback

                traceback.print_exc()
                return jsonify({"status": "error", "message": str(e) + traceback.format_exc()}), 500

    def run(self):
        """Start the HTTP server."""
        log.info(f"Starting HTTP server on port {self.port}")
        self.app.run(host=self.host, port=self.port)

    def act(self, image, prompt):
        """Run action inference
        Args:
            image: PIL.Image instance
            prompt: str, prompt text
        Returns:
            pred_action: [4] numpy array, local displacement and yaw change in radians
        """
        import torch

        with torch.inference_mode():
            inputs = self.processor(prompt, image)
            inputs = inputs.to(self.device, dtype=torch.bfloat16)
            pred_action = self.model.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=self.do_sample)

        return pred_action


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Merged UAV-finetuned OpenVLA checkpoint")
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA index within visible devices")
    parser.add_argument("--http-port", type=int, default=5007)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--unnorm-key", default="sim", help="Dataset key in the checkpoint's norm_stats")
    parser.add_argument(
        "--attn-implementation",
        choices=("flash_attention_2", "sdpa", "eager"),
        default="flash_attention_2",
    )
    args = parser.parse_args(argv)
    if args.gpu_id < 0 or not 1 <= args.http_port <= 65535:
        parser.error("gpu-id must be nonnegative and http-port must be 1..65535")
    return args


def main(argv=None):
    cfg = vars(parse_args(argv))
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create and run agent
    agent = OpenVLAActionAgent(cfg)
    agent.run()


if __name__ == "__main__":
    main()
