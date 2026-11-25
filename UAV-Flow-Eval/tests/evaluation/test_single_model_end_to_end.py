# SPDX-License-Identifier: Apache-2.0
"""Exercise a real batch subprocess and HTTP API; the UE itself is simulated."""

import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from evaluation import run_single_model as runner

BOOTSTRAP = r"""
import importlib.util, sys, types
from pathlib import Path
import numpy as np
source = Path(SOURCE_PATH)
sys.path.insert(0, str(source.parent.parent))
spec = importlib.util.spec_from_file_location('real_batch', source)
batch = importlib.util.module_from_spec(spec);spec.loader.exec_module(batch)
class API:
    location = [0,0,300]
    def set_obj_location(self, actor, position):self.location=list(position)
    def set_rotation(self,*args):pass
    def get_obj_location(self,*args):return self.location
    def get_camera_config(self):return {5:{'location':self.location}}
    def get_image(self,*args):
        a=np.arange(256,dtype=np.uint8)
        return np.stack([np.tile(a,(256,1)),np.tile(a[:,None],(1,256)),np.full((256,256),100,dtype=np.uint8)],axis=2)
api=API()
env=types.SimpleNamespace(unwrapped=types.SimpleNamespace(unrealcv=api,player_list=['drone']),close=lambda:None)
batch.ensure_env_ready=lambda *args,**kwargs:env
batch.create_obj_if_needed=lambda *args,**kwargs:None
batch.time.sleep=lambda *args:None
batch.draw_2d_trajectory_from_log=lambda *args,**kwargs:None
batch.draw_3d_trajectory_from_log=lambda *args,**kwargs:None
raise SystemExit(batch.main(sys.argv[3:]))
"""


class RealProcessFlowTests(unittest.TestCase):
    def run_scenario(self, fail_task_reset):
        with tempfile.TemporaryDirectory(prefix="uav-eval-e2e-") as folder:
            root = Path(folder)
            tasks = root / "tasks"
            tasks.mkdir()
            for name in ["a.json", "b.json"]:
                (tasks / name).write_text(
                    json.dumps({"initial_pos": [0, 0, 300, 0, 0, 0], "instruction": "Move forward"})
                )
            classification = root / "classification.json"
            classification.write_text(json.dumps({"Rotate": ["a.json", "b.json"]}))
            state = {"resets": 0, "predictions": 0, "index": 0}

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *args):
                    pass

                def do_POST(self):
                    self.rfile.read(int(self.headers.get("Content-Length", "0")))
                    if self.path == "/reset":
                        state["resets"] += 1
                        state["index"] = 0
                        code = 500 if fail_task_reset and state["resets"] > 1 else 200
                        result = {"status": "error" if code == 500 else "success"}
                    else:
                        state["predictions"] += 1
                        actions = []
                        for _ in range(8):
                            state["index"] += 1
                            actions.append([state["index"] * 10, 0, 0, 0])
                        code = 200
                        result = {"action": actions}
                    body = json.dumps(result).encode()
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                script = BOOTSTRAP.replace(
                    "SOURCE_PATH", repr(str(runner.ROOT / "closed_loop_runner.py"))
                )
                output = root / "output"
                args = runner.parse_args(
                    [
                        "--server-url",
                        f"http://127.0.0.1:{server.server_port}",
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
                        "root",
                        "--eval-timeout",
                        "60",
                    ]
                )
                command = runner.worker_command
                with (
                    patch.object(runner, "bind_check"),
                    patch.object(runner, "prepare_ue", return_value=output / "ue/Collection"),
                    patch.object(
                        runner.runtime,
                        "installed_package",
                        return_value={
                            "version": "test",
                            "modules": {
                                "evaluation.closed_loop_runner": str(
                                    runner.ROOT / "closed_loop_runner.py"
                                )
                            },
                        },
                    ),
                    patch.object(
                        runner,
                        "worker_command",
                        side_effect=lambda *a: [
                            sys.executable,
                            "-I",
                            "-B",
                            "-u",
                            "-c",
                            script,
                            *command(*a)[4:],
                        ],
                    ),
                    patch.object(
                        runner, "score_run", return_value={"overall": {"tasks": 2}}
                    ) as score,
                ):
                    result = runner.execute(args)
                status = json.loads((output / "progress_info/status.json").read_text())
                if fail_task_reset:
                    self.assertEqual(result, 1)
                    self.assertFalse(status["all_completed"])
                    self.assertEqual(state["predictions"], 0)
                    score.assert_not_called()
                else:
                    self.assertEqual(result, 0)
                    self.assertTrue(status["all_completed"])
                    self.assertEqual(state["resets"], 3)  # readiness + one per task
                    self.assertGreater(state["predictions"], 0)
                    score.assert_called_once()
                    for name in ["a.json", "b.json"]:
                        self.assertEqual(
                            len(json.loads((output / "trajectories" / name).read_text())), 100
                        )
                self.assertEqual(len(list((output / "trajectories").glob("*.json"))), 2)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_two_tasks_complete_once_through_real_http_and_batch_process(self):
        self.run_scenario(False)

    def test_reset_error_propagates_from_real_http_through_batch_to_controller(self):
        self.run_scenario(True)


if __name__ == "__main__":
    unittest.main()
