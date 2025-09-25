"""
Real-time action execution script. Loads the OpenVLA model, serves requests, and returns actions.
"""

import logging
import os
import sys
from pathlib import Path
sys.path.append(str(Path(str(os.getcwd())).resolve()))
import numpy as np
import json
from io import BytesIO
import base64
from flask import Flask, request, jsonify
from PIL import Image
import torch
import time

from transformers import AutoModelForVision2Seq, AutoProcessor

log = logging.getLogger(__name__)


class OpenVLAActionAgent:
    def __init__(self, cfg):
        # Device setup
        self.cfg = cfg
        self.gpu_id = cfg.get("gpu_id", 0)
        self.device = torch.device(f"cuda:{self.gpu_id}")
        
        # Load model and processor
        self.model_path = cfg.get("model_path")
        log.info(f"Loading model: {self.model_path}")
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        log.info(f"Processor type: {type(self.processor)}")
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(self.device)
        log.info(f"VLA model type: {type(self.model)}")
        
        self.model.eval()
        
        # Setup HTTP server
        self.app = Flask(__name__)
        self.setup_routes()
        self.port = cfg.get("http_port", 5000)
        
        # Other configurations
        self.unnorm_key = cfg.get("unnorm_key", "sim")
        self.do_sample = cfg.get("do_sample", False)

    def setup_routes(self):
        @self.app.route('/predict', methods=['POST'])
        def predict():
            try:
                data = request.json
                
                # Decode image
                img_bytes = base64.b64decode(data["image"])
                image = Image.open(BytesIO(img_bytes))
                
                # Retrieve instruction and state
                instruction = data["instr"]
                proprio = np.array(data["proprio"], dtype=np.float32)
                proprio_str = ','.join([str(round(x, 1)) for x in proprio])
                
                # Build prompt
                prompt = f"In: Current State: {proprio_str}, What action should the uav take to {instruction}?\nOut:"
                log.info(f"Prompt: {prompt}")
                
                start_time = time.time()
                with torch.inference_mode():
                    pred_action = self.act(image, prompt)
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
                R = np.array([
                    [cos_yaw, -sin_yaw, 0],
                    [sin_yaw, cos_yaw, 0],
                    [0, 0, 1]
                ])
                
                pred_action[0,0:3] = R @ pred_action[0,0:3]
                pred_action[0,0:3] = current_pos + pred_action[0,0:3]
                pred_action[0,-1] = pred_action[0,-1] + current_yaw

                return jsonify({
                    "status": "success",
                    "action": pred_action.tolist(),
                    "action_ori": pred_action_ori.tolist(),
                    "message": "Action generated successfully"
                })
                
            except Exception as e:              
                import traceback
                traceback.print_exc()
                return jsonify({
                    "status": "error",
                    "message": str(e) + traceback.format_exc()
                }), 500

    def run(self):
        """Start the HTTP server."""
        log.info(f"Starting HTTP server on port {self.port}")
        self.app.run(host='0.0.0.0', port=self.port)

    def act(self, image, prompt):
        """Run action inference
        Args:
            image: PIL.Image instance
            prompt: str, prompt text
        Returns:
            pred_action: [1, 4] numpy array, predicted action
        """
        # Prepare inputs
        inputs = self.processor(prompt, image)
        inputs = inputs.to(self.device, dtype=torch.bfloat16)
        
        # Predict action
        pred_action = self.model.predict_action(
            **inputs,
            unnorm_key=self.unnorm_key,
            do_sample=self.do_sample
        )
        
        # Convert to numpy array (already numpy if model returns numpy)
        pred_action = pred_action
        return pred_action

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    
    # Configuration
    cfg = {
        "gpu_id": 0,
        "model_path": "",  # Replace with the actual model path
        "http_port": 5007,
        "unnorm_key": "sim",
        "do_sample": False
    }
    
    # Create and run agent
    agent = OpenVLAActionAgent(cfg)
    agent.run()

if __name__ == "__main__":
    main()