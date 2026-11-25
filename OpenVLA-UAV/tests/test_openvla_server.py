"""Offline API/closed-loop checks; fixed policy outputs replace GPU inference only.

Run in the UAV-Flow-Eval environment from the UAV-Flow repository root:
    python -m unittest discover -s OpenVLA-UAV/tests -v
"""

import base64
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from flask import Flask
from PIL import Image
from werkzeug.serving import make_server

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "UAV-Flow-Eval/src"))
from evaluation import run_single_model as runner  # noqa: E402

spec = importlib.util.spec_from_file_location("openvla_server", ROOT / "OpenVLA-UAV/vla-scripts/openvla_act.py")
server_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server_module)


def make_agent(action):
    agent = server_module.OpenVLAActionAgent.__new__(server_module.OpenVLAActionAgent)
    agent.app = Flask(__name__)
    agent.calls = []

    def act(image, prompt):
        agent.calls.append((image.copy(), prompt))
        return np.array(action, dtype=np.float64)

    agent.act = act
    agent.setup_routes()
    return agent


def payload(proprio=(0, 0, 0, 0)):
    image = Image.new("RGB", (224, 224), (25, 75, 125))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return {
        "image": base64.b64encode(buffer.getvalue()).decode(),
        "instr": "Move forward",
        "proprio": list(proprio),
    }


# The real controller launches the real worker. Only the UE transport/rendering
# and sleeps are replaced; model HTTP, feedback, task resets and logs are real.
BOOTSTRAP = r"""
import importlib.util, sys, types
from pathlib import Path
import numpy as np
source = Path(SOURCE_PATH)
sys.path.insert(0, str(source.parent.parent))
spec = importlib.util.spec_from_file_location('batch_under_test', source)
batch = importlib.util.module_from_spec(spec); spec.loader.exec_module(batch)
class API:
    location = [0,0,300]
    def set_obj_location(self, actor, position): self.location=list(position)
    def set_rotation(self,*args): pass
    def get_obj_location(self,*args): return self.location
    def get_camera_config(self): return {5:{'location':self.location}}
    def get_image(self,*args):
        a=np.arange(256,dtype=np.uint8)
        return np.stack([np.tile(a,(256,1)),np.tile(a[:,None],(1,256)),np.full((256,256),100,dtype=np.uint8)],axis=2)
api=API()
env=types.SimpleNamespace(unwrapped=types.SimpleNamespace(unrealcv=api,player_list=['drone']),close=lambda:None)
batch.ensure_env_ready=lambda *a,**k:env
batch.create_obj_if_needed=lambda *a,**k:None
batch.time.sleep=lambda *a:None
batch.draw_2d_trajectory_from_log=lambda *a,**k:None
batch.draw_3d_trajectory_from_log=lambda *a,**k:None
raise SystemExit(batch.main(sys.argv[3:]))
"""


class OpenVLAProtocolTests(unittest.TestCase):
    def test_act_preserves_processor_inputs_and_model_decode_options(self):
        agent = server_module.OpenVLAActionAgent.__new__(server_module.OpenVLAActionAgent)
        agent.device = "cuda:0"
        agent.unnorm_key = "sim"
        agent.do_sample = False
        output = np.array([1, 2, 3, 0.5])
        inputs = Mock()
        inputs.to.return_value = {"input_ids": "tokens", "pixel_values": "pixels"}
        agent.processor = Mock(return_value=inputs)
        agent.model = types.SimpleNamespace(predict_action=Mock(return_value=output))
        image = Image.new("RGB", (224, 224))
        torch_stub = types.SimpleNamespace(inference_mode=nullcontext, bfloat16="bf16")
        with patch.dict(sys.modules, {"torch": torch_stub}):
            self.assertIs(agent.act(image, "prompt"), output)
        agent.processor.assert_called_once_with("prompt", image)
        inputs.to.assert_called_once_with("cuda:0", dtype="bf16")
        agent.model.predict_action.assert_called_once_with(
            input_ids="tokens", pixel_values="pixels", unnorm_key="sim", do_sample=False
        )

    def test_reset_is_stateless_and_does_not_run_model(self):
        agent = make_agent([10, 0, 0, 0])
        client = agent.app.test_client()
        for _ in range(2):
            response = client.post("/reset", json={})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json["status"], "success")
        self.assertEqual(agent.calls, [])

    def test_prediction_matches_legacy_prompt_and_coordinate_math(self):
        action = np.array([10, -2, 3, 0.25])
        agent = make_agent(action)
        for proprio in [(0, 0, 0, 0), (12.25, -8.5, 30, 90), (-10, 20, -3, -135)]:
            with self.subTest(proprio=proprio):
                response = agent.app.test_client().post("/predict", json=payload(proprio))
                self.assertEqual(response.status_code, 200)
                state = np.array(proprio, dtype=np.float32)
                yaw = np.deg2rad(state[-1])
                c, s = np.cos(yaw), np.sin(yaw)
                rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                expected = action.copy()[None, :]
                expected[0, :3] = rotation @ expected[0, :3]
                expected[0, :3] = state[:3] + expected[0, :3]
                expected[0, -1] += yaw
                np.testing.assert_array_equal(response.json["action"], expected)
                np.testing.assert_array_equal(response.json["action_ori"], action[None, :])
                text = ",".join(str(round(x, 1)) for x in state)
                self.assertEqual(
                    agent.calls[-1][1],
                    f"In: Current State: {text}, What action should the uav take to Move forward?\nOut:",
                )
                self.assertEqual(agent.calls[-1][0].getpixel((0, 0)), (25, 75, 125))

    def test_non_uav_action_and_nonfinite_output_fail(self):
        for action in [[0] * 7, [0, 0, 0, float("nan")], [[0, 0, 0, 0]]]:
            with self.subTest(action=action):
                response = make_agent(action).app.test_client().post("/predict", json=payload())
                self.assertEqual(response.status_code, 500)
                self.assertEqual(response.json["status"], "error")

    def test_cli_help_needs_no_model_or_gpu_libraries(self):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "OpenVLA-UAV/vla-scripts/openvla_act.py"), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in ["--model-path", "--http-port", "--gpu-id", "--unnorm-key", "--attn-implementation"]:
            self.assertIn(option, result.stdout)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Process supervision requires Linux /proc")
    def test_two_episodes_through_http_and_real_closed_loop_worker(self):
        agent = make_agent([10, 0, 1, 0])
        self.assertEqual(agent.app.test_client().post("/reset", json={}).status_code, 200)
        http = make_server("127.0.0.1", 0, agent.app)
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory(prefix="openvla-closed-loop-") as tmp:
                root = Path(tmp)
                tasks = root / "tasks"
                tasks.mkdir()
                for name in ["a.json", "b.json"]:
                    (tasks / name).write_text(
                        json.dumps({"initial_pos": [0, 0, 300, 0, 0, 0], "instruction": "Move forward"})
                    )
                classification = root / "classification.json"
                classification.write_text(json.dumps({"Move": ["a.json", "b.json"]}))
                output = root / "output"
                base = f"http://127.0.0.1:{http.server_port}"
                runner.model_smoke(base)
                agent.calls.clear()
                args = runner.parse_args(
                    [
                        "--server-url",
                        base,
                        "--output",
                        str(output),
                        "--task-dir",
                        str(tasks),
                        "--classification",
                        str(classification),
                        "--ue-package",
                        str(root / "fake-package"),
                        "--graphics-adapter",
                        "1",
                        "--ue-user",
                        runner.default_ue_user() or "root",
                        "--eval-timeout",
                        "60",
                    ]
                )
                command = runner.worker_command
                script = BOOTSTRAP.replace("SOURCE_PATH", repr(str(runner.ROOT / "closed_loop_runner.py")))
                with (
                    patch.object(runner, "bind_check"),
                    patch.object(runner, "prepare_ue", return_value=output / "ue/Collection"),
                    patch.object(
                        runner.runtime,
                        "installed_package",
                        return_value={
                            "version": "test",
                            "modules": {"evaluation.closed_loop_runner": str(runner.ROOT / "closed_loop_runner.py")},
                        },
                    ),
                    patch.object(
                        runner,
                        "worker_command",
                        side_effect=lambda *a: [sys.executable, "-I", "-B", "-u", "-c", script, *command(*a)[4:]],
                    ),
                    patch.object(runner, "score_run", return_value={"overall": {"tasks": 2}}),
                ):
                    self.assertEqual(runner.execute(args), 0)
                status = json.loads((output / "progress_info/status.json").read_text())
                self.assertTrue(status["all_completed"])
                self.assertEqual(len(agent.calls), 200)
                for name in ["a.json", "b.json"]:
                    trajectory = json.loads((output / "trajectories" / name).read_text())
                    self.assertEqual(len(trajectory), 100)
                    self.assertEqual(trajectory[0]["state"][0], [10, 0, 1])
                    self.assertEqual(trajectory[-1]["state"][0], [1000, 0, 100])
                self.assertEqual(agent.calls[0][1], agent.calls[100][1])
                self.assertIn("Current State: 990.0,0.0,99.0,0.0", agent.calls[99][1])
        finally:
            http.shutdown()
            http.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
