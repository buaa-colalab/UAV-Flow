# SPDX-License-Identifier: Apache-2.0
"""Training HTTP regressions with simulated Unreal and private temporary storage."""

import base64
import copy
import json
import signal
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

import env_server
from env_server import EnvServer


class TrainingServiceRegressions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="uav-training-test-")
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        definitions = root / "training_tasks"
        definitions.mkdir(parents=True)
        tasks = [
            dict(
                task_id=i,
                init_pose=[0, 0, 300, 0, 0, 0],
                prompt="Approach target",
                use_obj=1,
                obj_id=i,
                target_pos=[i * 100, 0, 0, 0, 0, 0],
            )
            for i in (1, 2)
        ]
        (definitions / "tasks.json").write_text(json.dumps(tasks))
        (definitions / "task_categories.json").write_text(json.dumps({"Approach": [1, 2]}))
        self.root = root
        self.world = {"object": None}
        self.observed = []
        env = MagicMock()
        env.unwrapped.player_list = ["drone"]
        env.unwrapped.ue_binary.env = None
        env.unwrapped.unrealcv.client.isconnected.return_value = True

        def initialize(server, cfg):
            server.env = env

        self.enterContext(patch.object(EnvServer, "isPortFree", return_value=True))
        self.enterContext(patch.object(EnvServer, "_initialize_environment", initialize))
        self.enterContext(patch.object(env_server, "__file__", str(root / "env_server.py")))
        self.enterContext(patch("env_server.time.sleep"))
        self.enterContext(patch.dict("os.environ", {"ENV_RUN_ID": "isolated-test"}))
        self.server = EnvServer({"http_port": 5998, "save_images": True, "output": root / "output"})

        def apply_object(obj):
            self.world["object"] = obj["obj_id"] if obj else None

        self.server.create_obj_if_needed = MagicMock(side_effect=apply_object)

        def capture():
            self.observed.append(self.world["object"])
            return np.full((8, 8, 3), self.world["object"] or 0, dtype=np.uint8)

        self.server._capture_rl_image = capture
        self.server.close = MagicMock()
        self.server._restart_engine = MagicMock(side_effect=AssertionError("UE must not restart"))
        self.client = self.server.app.test_client()
        self.assertEqual(self.client.post("/reset", json={"task_ids": [1, 2]}).status_code, 200)

    def step(self, actions=None, **kwargs):
        if actions is None:
            actions = [[[1, 0, 0, 0]], [[2, 0, 0, 0]]]
        return self.client.post("/step", json={"actions": actions}, **kwargs)

    def test_no_accepted_waypoints_returns_second_frame_like_running_service(self):
        from io import BytesIO

        from PIL import Image

        for mode in ("empty", "done", "invalid"):
            for with_id in (False, True):
                with self.subTest(mode=mode, with_id=with_id):
                    self.client.post("/reset", json={"task_ids": [1, 2]})
                    if mode == "done":
                        for state in self.server.batch_states:
                            state.update(done=True, termination_reason="max_steps", step_count=100)
                    actions = {
                        "empty": [[], []],
                        "done": [[[1, 0, 0, 0]], [[2, 0, 0, 0]]],
                        "invalid": [[["bad", 0, 0, 0]], [["bad", 0, 0, 0]]],
                    }[mode]
                    headers = {"X-Rollout-Request-ID": f"second-frame-{mode}"} if with_id else {}
                    frames = [np.full((8, 8, 3), v, dtype=np.uint8) for v in (10, 20, 30, 40)]
                    with patch.object(
                        self.server, "_capture_rl_image", side_effect=frames
                    ) as capture:
                        response = self.step(actions, headers=headers)
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(response.json["accepted_waypoints"], [[], []])
                        for i, value in enumerate((20, 40)):
                            raw = base64.b64decode(response.json["images"][i][0])
                            np.testing.assert_array_equal(
                                np.asarray(Image.open(BytesIO(raw))), frames[2 * i + 1]
                            )
                            self.assertEqual(
                                (self.server.trajectory_dirs[i] / "0001.png").read_bytes(), raw
                            )
                        self.assertEqual(capture.call_count, 4)
                        if with_id:
                            retry = self.step(actions, headers=headers)
                            self.assertEqual(retry.json, response.json)
                            self.assertEqual(capture.call_count, 4)
                    self.assertEqual(self.server.trajectory_image_counts, [2, 2])

    def test_mixed_done_batch_preserves_camera_read_order(self):
        self.server.batch_states[0].update(
            done=True, termination_reason="max_steps", step_count=100
        )
        self.observed.clear()
        response = self.step()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.observed, [1, 1, 2])
        self.assertEqual(response.json["accepted_waypoints"], [[], [[2, 0, 0, 0]]])
        self.assertEqual(response.json["dones"], [True, False])
        self.assertEqual(self.server.trajectory_image_counts, [2, 2])

    def test_all_writes_stay_in_output_and_never_in_reference_package(self):
        with patch.object(
            self.server,
            "get_check_result_with_details",
            return_value=(
                1.0,
                "ok",
                "靠近",
                {"reward": 1.0, "approach_score": 1.0, "collision_penalty": 0.0},
            ),
        ):
            response = self.client.post(
                "/rewards", json={"trajectories": [[[0, 0, 300, 0, 0, 0]]] * 2}
            )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(self.server.save_dir.is_relative_to(self.root / "output"))
        self.assertTrue((self.root / "output/tmp/port_5998/1.json").exists())
        self.assertFalse((self.root / "temp_trajectories").exists())
        self.assertFalse((self.root / "saved_trajectories").exists())
        self.assertFalse(list((self.root / "training_tasks").rglob("*.png")))

    def test_step_restores_each_tasks_target(self):
        self.assertEqual(self.observed, [1, 2])
        self.observed.clear()
        self.assertEqual(self.step().status_code, 200)
        self.assertEqual(self.observed, [1, 2])

    def test_wrong_action_batch_size_is_rejected_before_state_or_engine_changes(self):
        for actions in ([[[1, 0, 0, 0]]], [[[1, 0, 0, 0]]] * 3):
            with self.subTest(size=len(actions)):
                before = copy.deepcopy(self.server.batch_states)
                self.server.env.unwrapped.unrealcv.reset_mock()
                response = self.step(actions)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json["error_code"], "batch_size_mismatch")
                self.assertEqual(self.server.batch_states, before)
                api = self.server.env.unwrapped.unrealcv
                api.set_obj_location.assert_not_called()
                api.set_cam_location.assert_not_called()
                api.set_cam_rotation.assert_not_called()

    def test_normal_step_saves_each_returned_frame_once(self):
        response = self.step()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.server.trajectory_image_counts, [2, 2])
        for index, directory in enumerate(self.server.trajectory_dirs):
            self.assertEqual(len(list(directory.glob("*.png"))), 2)
            self.assertEqual(
                (directory / "0001.png").read_bytes(),
                base64.b64decode(response.json["images"][index][0]),
            )

    def test_empty_actions_keep_image_contract_and_save_once(self):
        response = self.step([[], []])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["accepted_waypoints"], [[], []])
        self.assertEqual(self.server.trajectory_image_counts, [2, 2])

    def test_duplicate_step_does_not_write_duplicate_images(self):
        headers = {"X-Rollout-Request-ID": "isolated-step-1"}
        first = self.step(headers=headers)
        second = self.step(headers=headers)
        self.assertEqual(first.json, second.json)
        self.assertEqual(self.server.trajectory_image_counts, [2, 2])
        self.assertEqual([s["step_count"] for s in self.server.batch_states], [1, 1])

    def test_reward_disk_failure_returns_503_and_service_stays_usable(self):
        self.server.save_images = False
        before = copy.deepcopy(self.server.batch_states)
        with patch("pathlib.Path.mkdir", side_effect=OSError("simulated disk full")):
            response = self.client.post(
                "/rewards", json={"trajectories": [[[0, 0, 300, 0, 0, 0]]] * 2}
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json["error_code"], "storage_unavailable")
        self.assertEqual(self.server.batch_states, before)
        self.server._restart_engine.assert_not_called()
        self.assertEqual(self.step().status_code, 200)

    def test_step_disk_failure_returns_503_without_advancing_batch(self):
        from PIL import Image

        before = copy.deepcopy(self.server.batch_states)
        with patch.object(Image.Image, "save", side_effect=OSError("simulated disk full")):
            response = self.step()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json["error_code"], "storage_unavailable")
        self.assertEqual(self.server.batch_states, before)
        self.server._restart_engine.assert_not_called()

    def test_reward_trajectory_write_failure_is_contained(self):
        self.server.save_images = False
        with patch("env_server.json.dump", side_effect=OSError("simulated write failure")):
            response = self.client.post(
                "/rewards", json={"trajectories": [[[0, 0, 300, 0, 0, 0]]] * 2}
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json["error_code"], "storage_unavailable")
        self.server._restart_engine.assert_not_called()

    def test_reward_success_contract_does_not_require_engine_connection(self):
        self.server.save_images = False
        self.server.env.unwrapped.unrealcv.client.isconnected.return_value = False
        self.server.get_check_result_with_details = MagicMock(
            return_value=(
                0.75,
                "valid score",
                "靠近",
                {"reward": 0.75, "approach_score": 0.8, "collision_penalty": 0.1},
            )
        )
        response = self.client.post("/rewards", json={"trajectories": [[[0, 0, 300, 0, 0, 0]]] * 2})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json,
            dict(
                rewards=[0.75, 0.75], reward_1=[0.8, 0.8], reward_2=[0.1, 0.1], reward_3=[0.0, 0.0]
            ),
        )
        self.server._restart_engine.assert_not_called()

    def test_waypoint_budget_clips_chunk_at_100_and_remains_done(self):
        for state in self.server.batch_states:
            state["step_count"] = 98
        chunk = [[float(i * 10), 0, 0, 0] for i in range(8)]
        response = self.step([chunk, chunk])
        self.assertEqual(response.status_code, 200)
        self.assertEqual([len(a) for a in response.json["accepted_waypoints"]], [2, 2])
        self.assertEqual(response.json["termination_reasons"], ["max_steps", "max_steps"])
        self.assertEqual(response.json["dones"], [True, True])
        again = self.step([chunk, chunk])
        self.assertEqual(again.json["accepted_waypoints"], [[], []])
        self.assertEqual([s["step_count"] for s in self.server.batch_states], [100, 100])

    def test_actions_are_absolute_waypoints_relative_to_reset_pose(self):
        for state in self.server.batch_states:
            state["initial_pos"] = [100, 200, 300, 0, 90, 0]
            state["initial_yaw"] = 90
        response = self.step([[[10, 20, 30, np.pi / 2]]] * 2)
        self.assertEqual(response.status_code, 200)
        for state in self.server.batch_states:
            np.testing.assert_allclose(state["current_pos"], [80, 210, 330])
            self.assertEqual(state["current_yaw"], 180)

    def test_nonfinite_waypoint_stops_without_accepting_invalid_action(self):
        response = self.step([[[float("nan"), 0, 0, 0]], [[0, 0, 0]]])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["accepted_waypoints"], [[], []])
        self.assertEqual(response.json["termination_reasons"], ["invalid_waypoint"] * 2)
        self.assertEqual([s["step_count"] for s in self.server.batch_states], [0, 0])

    def enable_real_recovery(self):
        bound = EnvServer._restart_engine.__get__(self.server, EnvServer)
        self.server._restart_engine = MagicMock(wraps=bound)

    def test_partial_batch_failure_restarts_and_replays_without_double_advancing(self):
        self.enable_real_recovery()
        capture = self.server._capture_rl_image
        calls = 0

        def fail_second_capture():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ConnectionError("simulated broken pipe mid-batch")
            return capture()

        self.server._capture_rl_image = fail_second_capture
        headers = {"X-Rollout-Request-ID": "partial-batch"}
        response = self.step(headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual([s["step_count"] for s in self.server.batch_states], [1, 1])
        self.assertEqual(self.server.trajectory_image_counts, [2, 2])
        self.server._restart_engine.assert_called_once()
        self.assertEqual(self.server._engine_restarts, 1)
        self.assertEqual([len(list(d.glob("*.png"))) for d in self.server.trajectory_dirs], [2, 2])
        self.assertEqual(self.step(headers=headers).json, response.json)
        self.assertEqual(calls, 4)

    @unittest.skipUnless(hasattr(signal, "setitimer"), "Requires Linux/macOS alarm deadlines")
    def test_real_alarm_timeout_restarts_and_replays(self):
        self.enable_real_recovery()
        self.server.cfg["recovery_request_timeout"] = 0.1
        capture = self.server._capture_rl_image
        fired = False

        def timeout_once():
            nonlocal fired
            if not fired:
                fired = True
                # Wait in a real blocking primitive so the actual interval timer fires.
                threading.Event().wait(2)
            return capture()

        self.server._capture_rl_image = timeout_once
        response = self.step()
        self.assertEqual(response.status_code, 200)
        self.server._restart_engine.assert_called_once()
        self.assertEqual([s["step_count"] for s in self.server.batch_states], [1, 1])
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0], 0)

    def test_recovery_exhaustion_keeps_original_state_and_returns_503(self):
        self.enable_real_recovery()
        before = self.server._snapshot_rollout()
        self.server._capture_rl_image = MagicMock(
            side_effect=ConnectionError("simulated lost connection")
        )
        response = self.step()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json["error_code"], "unreal_recovery_exhausted")
        self.assertEqual(response.json["recovery_count"], 2)
        self.assertEqual(self.server._snapshot_rollout(), before)
        self.assertEqual(self.server._restart_engine.call_count, 2)
        self.assertEqual(len(self.server._request_cache), 0)

    def test_zero_total_budget_does_not_execute_or_restart(self):
        self.server.cfg["recovery_total_timeout"] = 0
        self.server._capture_rl_image = MagicMock()
        response = self.step()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json["error_code"], "unreal_recovery_exhausted")
        self.server._capture_rl_image.assert_not_called()
        self.server._restart_engine.assert_not_called()

    def test_disconnected_engine_is_rebuilt_before_executing(self):
        self.enable_real_recovery()
        engine = self.server.env
        engine.unwrapped.unrealcv.client.isconnected.return_value = False

        def initialize(cfg):
            self.server.env = engine
            engine.unwrapped.unrealcv.client.isconnected.return_value = True

        self.server._initialize_environment = MagicMock(side_effect=initialize)
        response = self.step()
        self.assertEqual(response.status_code, 200)
        self.server._initialize_environment.assert_called_once_with(self.server.cfg)
        self.assertEqual([s["step_count"] for s in self.server.batch_states], [1, 1])
        self.assertEqual(self.server._engine_restarts, 1)

    def test_failed_first_rebuild_retries_without_losing_rollout(self):
        self.enable_real_recovery()
        engine = self.server.env
        engine.unwrapped.unrealcv.client.isconnected.return_value = False
        attempts = 0

        def initialize(cfg):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("simulated first bootstrap failure")
            self.server.env = engine
            engine.unwrapped.unrealcv.client.isconnected.return_value = True

        self.server._initialize_environment = MagicMock(side_effect=initialize)
        response = self.step()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(attempts, 2)
        self.assertEqual([s["step_count"] for s in self.server.batch_states], [1, 1])
        self.assertEqual(self.server.trajectory_image_counts, [2, 2])

    def test_health_rejects_dead_process_even_if_socket_looks_connected(self):
        self.server.env.unwrapped.ue_binary.env = MagicMock()
        self.server.env.unwrapped.ue_binary.env.poll.return_value = 1
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json["status"], "unhealthy")
        self.server._restart_engine.assert_not_called()

    def test_cache_detects_request_id_payload_conflict_without_advancing(self):
        headers = {"X-Rollout-Request-ID": "conflicting-step"}
        self.assertEqual(self.step(headers=headers).status_code, 200)
        before = self.server._snapshot_rollout()
        conflict = self.step([[[5, 0, 0, 0]]] * 2, headers=headers)
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(self.server._snapshot_rollout(), before)

    def test_last_step_survives_cache_eviction_by_observations(self):
        headers = {"X-Rollout-Request-ID": "late-step-retry"}
        first = self.step(headers=headers)
        api = self.server.env.unwrapped.unrealcv
        api.get_image.return_value = np.zeros((8, 8, 3), dtype=np.uint8)
        for name in [
            "get_cam_location",
            "get_cam_rotation",
            "get_obj_location",
            "get_obj_rotation",
        ]:
            getattr(api, name).return_value = [0, 0, 0]
        for index in range(20):
            observation = self.client.get(
                "/observation", headers={"X-Rollout-Request-ID": f"observation-{index}"}
            )
            self.assertEqual(observation.status_code, 200)
        self.assertLessEqual(len(self.server._request_cache), 8)
        self.assertEqual(self.step(headers=headers).json, first.json)
        self.assertEqual([s["step_count"] for s in self.server.batch_states], [1, 1])
        self.assertFalse(hasattr(self.server, "_last_step_response"))
        self.assertFalse(hasattr(self.server, "recovery"))

    def test_successful_request_remains_retryable_when_engine_disconnects(self):
        headers = {"X-Rollout-Request-ID": "cached-before-disconnect"}
        response = self.step(headers=headers)
        self.server.env.unwrapped.unrealcv.client.isconnected.return_value = False
        replay = self.step(headers=headers)
        self.assertEqual(replay.status_code, 200)
        self.assertEqual(replay.json, response.json)
        self.server._restart_engine.assert_not_called()

    def test_reset_retry_uses_the_same_single_cache(self):
        headers = {"X-Rollout-Request-ID": "reset-retry"}
        self.server._apply_initial_state = MagicMock(wraps=self.server._apply_initial_state)
        first = self.client.post("/reset", json={"task_ids": [1, 2]}, headers=headers)
        second = self.client.post("/reset", json={"task_ids": [1, 2]}, headers=headers)
        self.assertEqual(first.json, second.json)
        self.assertEqual(self.server._apply_initial_state.call_count, 2)
        self.assertIn(("reset", "reset-retry"), self.server._request_cache)

    def test_health_is_single_and_reports_real_engine_failures(self):
        self.assertEqual(
            sum(rule.endpoint == "health" for rule in self.server.app.url_map.iter_rules()), 1
        )
        healthy = self.client.get("/health")
        self.assertEqual(healthy.status_code, 200)
        self.assertEqual(healthy.json["recovery_count"], 0)
        api = self.server.env.unwrapped.unrealcv
        api.get_obj_location.side_effect = ConnectionError("simulated unresponsive engine")
        unhealthy = self.client.get("/health")
        self.assertEqual(unhealthy.status_code, 503)
        self.assertEqual(unhealthy.json["status"], "unhealthy")
        self.server._restart_engine.assert_not_called()


class TargetPlacementRegressions(unittest.TestCase):
    def test_human_and_car_restore_appearance_position_and_inactive_target(self):
        for use_obj in (1, 2):
            with self.subTest(use_obj=use_obj):
                server = EnvServer.__new__(EnvServer)
                server.env = MagicMock()
                api = server.env.unwrapped.unrealcv
                positions = {}
                api.set_obj_location.side_effect = lambda name, value: positions.update(
                    {name: value}
                )
                api.get_obj_location.side_effect = lambda name: positions[name]
                info = dict(use_obj=use_obj, obj_id=7, obj_pos=[100, 200, 300], obj_rot=[0, 45, 0])
                with patch("env_server.time.sleep"):
                    server.create_obj_if_needed(info)
                active, inactive = (
                    ("BP_Character_21", "BP_Character_22")
                    if use_obj == 1
                    else ("BP_Character_22", "BP_Character_21")
                )
                self.assertEqual(positions[active], [100, 200, 300 if use_obj == 1 else 0])
                self.assertEqual(positions[inactive], [5100, 0, 0])
                api.set_appearance.assert_called_once_with(active, 7 if use_obj == 1 else 2)
                api.set_obj_rotation.assert_called_once_with(active, [0, 45, 0])


if __name__ == "__main__":
    unittest.main()
