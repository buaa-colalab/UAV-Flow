# SPDX-License-Identifier: Apache-2.0
import base64
import configparser
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from threading import Barrier
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from env_server import EnvServer
from gym_unrealcv.envs.agent.character import Character_API
from gym_unrealcv.envs.base_env import UnrealCv_base
from tools.rl_envs_linux import prepare_slot


class LinuxStartupTests(unittest.TestCase):
    def test_camera_zero_uses_absolute_camera_commands_and_drone_pose(self):
        api = Character_API.__new__(Character_API)
        api.client = MagicMock()
        api.cam = {0: {}}
        api.get_obj_location = MagicMock(return_value=[100, 200, 300])
        api.get_obj_rotation = MagicMock(return_value=[12, 90, -4])
        server = EnvServer.__new__(EnvServer)
        server.env = MagicMock()
        server.env.unwrapped.player_list = ["BP_Drone01_2"]
        server.env.unwrapped.unrealcv = api
        server.set_cam()
        commands = [call.args[0] for call in api.client.request.call_args_list]
        self.assertEqual(
            commands, ["vset /camera/0/location 100 200 300", "vset /camera/0/rotation -4 90 12"]
        )
        api.get_obj_location.assert_called_once_with("BP_Drone01_2")
        self.assertEqual(api.cam[0]["location"], [100, 200, 300])

    def test_camera_uses_commanded_state_despite_drone_rotation_interpolation(self):
        server = EnvServer.__new__(EnvServer)
        server.env = MagicMock()
        api = server.env.unwrapped.unrealcv
        api.get_obj_rotation.return_value = [0, 168, 0]
        server.rl_camera_id = 5
        server.set_cam({"current_pos": [100, 200, 300], "current_yaw": 429})
        api.set_cam_location.assert_called_once_with(5, [100, 200, 300])
        api.set_cam_rotation.assert_called_once_with(5, [0, 69, 0], rpy=True)
        api.get_obj_rotation.assert_not_called()

    def test_state_pose_does_not_trigger_blueprint_capture_rotation(self):
        server = EnvServer.__new__(EnvServer)
        server.env = MagicMock()
        server.rl_camera_id = 5
        server.env.unwrapped.player_list = ["BP_Drone01_2"]
        server._set_state_pose({"current_pos": [100, 200, 300], "current_yaw": 69})
        api = server.env.unwrapped.unrealcv
        api.set_rotation.assert_not_called()
        api.set_cam_rotation.assert_called_once_with(5, [0, 69, 0], rpy=True)

    def test_capture_discards_previous_frame_before_returning_current_frame(self):
        server = EnvServer.__new__(EnvServer)
        server.env = MagicMock()
        stale = np.zeros((2, 2, 3), dtype=np.uint8)
        current = np.ones((2, 2, 3), dtype=np.uint8)
        server.env.unwrapped.unrealcv.get_image.side_effect = [stale, current]
        with patch("env_server.time.sleep"):
            self.assertIs(server._capture_rl_image(), current)
        self.assertEqual(server.env.unwrapped.unrealcv.get_image.call_count, 2)

    def test_observation_reads_current_image_without_moving_or_resetting(self):
        with (
            patch.object(EnvServer, "isPortFree", return_value=True),
            patch.object(EnvServer, "_initialize_environment"),
        ):
            server = EnvServer({"http_port": 5008, "save_images": False})
        server.env = MagicMock()
        server.env.unwrapped.ue_binary.env = None
        server.env.unwrapped.unrealcv.client.isconnected.return_value = True
        server.env.unwrapped.player_list = ["BP_Drone01_2"]
        api = server.env.unwrapped.unrealcv
        api.get_cam_location.return_value = [100, 200, 300]
        api.get_cam_rotation.return_value = [0, 90, 0]
        api.get_obj_location.return_value = [100, 200, 300]
        api.get_obj_rotation.return_value = [0, 90, 0]
        api.get_image.return_value = np.zeros((256, 256, 3), dtype=np.uint8)
        result = server.app.test_client().get("/observation")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json["camera"]["location"], [100, 200, 300])
        self.assertEqual(result.json["batch_size"], 0)
        self.assertEqual(server.batch_states, [])
        for name in (
            "set_cam",
            "set_cam_location",
            "set_cam_rotation",
            "set_obj_location",
            "set_rotation",
        ):
            getattr(api, name).assert_not_called()
        server.env.reset.assert_not_called()

    def test_concurrent_rewards_for_same_task_do_not_overwrite_other_port(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            definitions = root / "training_tasks"
            definitions.mkdir(parents=True)
            (definitions / "tasks.json").write_text(
                json.dumps(
                    [
                        {
                            "task_id": 512,
                            "prompt": "Rotate 75 degrees to the left",
                            "init_pose": [0, 0, 300, 0, 0, 0],
                        }
                    ]
                )
            )
            (definitions / "task_categories.json").write_text(json.dumps({"Rotate": [512]}))
            with (
                patch("env_server.__file__", str(root / "env_server.py")),
                patch.object(EnvServer, "isPortFree", return_value=True),
                patch.object(EnvServer, "_initialize_environment"),
            ):
                servers = [
                    EnvServer({"http_port": port, "save_images": False, "output": root / "output"})
                    for port in (5008, 5009)
                ]
                barrier = Barrier(2)
                for server in servers:
                    server.task_id_list = [512]
                    original = server.get_check_result_with_details

                    def synchronized_check(*args, original=original, **kwargs):
                        barrier.wait(timeout=5)
                        return original(*args, **kwargs)

                    server.get_check_result_with_details = synchronized_check

                def evaluate(server, yaw):
                    trajectory = [[0, 0, 300, 0, 0, 0], [0, 0, 300, 0, yaw, 0]]
                    return (
                        server.app.test_client()
                        .post("/rewards", json={"trajectories": [trajectory]})
                        .json
                    )

                with ThreadPoolExecutor(max_workers=2) as pool:
                    first = pool.submit(evaluate, servers[0], -75)
                    second = pool.submit(evaluate, servers[1], -15)
                    first_result, second_result = first.result(), second.result()
                self.assertEqual(first_result["rewards"], [1.0])
                self.assertLess(second_result["rewards"][0], 1.0)
                for port, yaw in ((5008, -75), (5009, -15)):
                    trajectory = json.loads((root / f"output/tmp/port_{port}/512.json").read_text())
                    self.assertEqual(trajectory[-1]["state"][1][1], yaw)

    def test_http_reset_and_step_keep_existing_image_and_done_contract(self):
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        with (
            patch.object(EnvServer, "isPortFree", return_value=True),
            patch.object(EnvServer, "_initialize_environment"),
            patch.object(EnvServer, "_apply_initial_state", return_value=image),
            patch.object(EnvServer, "_set_state_pose"),
            patch.object(EnvServer, "set_cam"),
            patch("env_server.time.sleep"),
        ):
            server = EnvServer({"http_port": 5008, "save_images": False})
            server.env = MagicMock()
            server.env.unwrapped.ue_binary.env = None
            server.env.unwrapped.unrealcv.client.isconnected.return_value = True
            server.env.unwrapped.unrealcv.get_image.return_value = image
            client = server.app.test_client()
            self.assertEqual(client.get("/health").json["status"], "healthy")
            self.assertEqual(client.post("/reset", json={}).status_code, 400)
            response = client.post("/reset", json={"task_ids": [0]})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json["task_ids"], [0])
            png = Image.open(BytesIO(base64.b64decode(response.json["images"][0][0])))
            self.assertEqual(png.size, (256, 256))
            response = client.post("/step", json={"actions": [[[0, 0, 0, 0]]]})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json["dones"], [False])
            self.assertEqual(len(response.json["images"]), 1)

    def test_slots_keep_shared_package_and_other_slot_config_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            package = root / "source"
            binary = package / "Collection/Binaries/Linux/Collection"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"fake executable")
            (binary.parent / "unrealcv.ini").write_text(
                "[UnrealCV.Core]\nPort=9000\nWidth=160\nHeight=160\n"
            )
            (package / "Collection/Content").mkdir()
            (package / "Engine/Binaries").mkdir(parents=True)
            first = prepare_slot(package, root / "one", 9400)
            second = prepare_slot(package, root / "two", 9401)
            (first.parent / "unrealcv.ini").write_text("[UnrealCV.Core]\nPort=9402\n")
            config = configparser.ConfigParser()
            config.read(second.parent / "unrealcv.ini")
            self.assertEqual(config.getint("UnrealCV.Core", "Port"), 9401)
            config.read(binary.parent / "unrealcv.ini")
            self.assertEqual(config.getint("UnrealCV.Core", "Port"), 9000)
            self.assertFalse(first.is_symlink())
            self.assertTrue((root / "one/Collection/Content").is_symlink())
            self.assertTrue((root / "one/Engine/Binaries").is_symlink())
            self.assertFalse((root / "one/Collection/Saved").is_symlink())

    def test_busy_http_port_fails_before_launching_unreal(self):
        with (
            patch.object(EnvServer, "isPortFree", return_value=False),
            patch("env_server.gym.make") as make,
        ):
            with self.assertRaisesRegex(RuntimeError, "5008"):
                EnvServer({"http_port": 5008})
            make.assert_not_called()

    def test_failed_reset_cleans_up_unreal_process(self):
        env = MagicMock()
        env.unwrapped = env
        env.reset.side_effect = RuntimeError("reset failed")
        env.ue_binary.env.poll.return_value = None
        with (
            patch.object(EnvServer, "isPortFree", return_value=True),
            patch("env_server.gym.make", return_value=env),
            patch("env_server.configUE.ConfigUEWrapper", side_effect=lambda e, **kwargs: e),
            patch(
                "env_server.augmentation.RandomPopulationWrapper",
                side_effect=lambda e, *a, **kw: e,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "reset failed"):
                EnvServer({"time_dilation": 0, "offscreen": True, "gpu_id": 2})
        env.close.assert_called_once()
        env.ue_binary.env.kill.assert_called_once()

    def test_gpu_and_headless_flags_reach_launcher(self):
        env = MagicMock()
        env.runtime_profile = "training"
        env.docker = False
        env.resolution = (256, 256)
        env.display = None
        env.use_opengl = False
        env.offscreen_rendering = True
        env.nullrhi = False
        env.gpu_id = 2
        env.ue_binary.start.return_value = ("127.0.0.1", 9402)
        with (
            patch.object(UnrealCv_base, "_wait_for_port"),
            patch("gym_unrealcv.envs.base_env.Character_API"),
        ):
            self.assertTrue(UnrealCv_base.launch_ue_env(env))
        kwargs = env.ue_binary.start.call_args.kwargs
        self.assertEqual(kwargs["gpu_id"], 2)
        self.assertTrue(kwargs["offscreen"])
        self.assertFalse(kwargs["nullrhi"])

    def test_unreal_crash_fails_immediately_instead_of_waiting_ten_minutes(self):
        process = MagicMock(returncode=1)
        process.poll.return_value = 1
        with self.assertRaisesRegex(RuntimeError, "exited with code 1"):
            UnrealCv_base._wait_for_port("127.0.0.1", 9400, process=process)


if __name__ == "__main__":
    unittest.main()
