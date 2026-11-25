# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO = Path(
    os.environ.get("SURROUND_TEST_REPO", Path(__file__).resolve().parents[2] / "src/evaluation")
)
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location(
    "scoring_under_test", REPO / "check_trajectories_batch.py"
)
scoring = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scoring)


def circle_points(degrees, radius=300):
    return [
        [
            300 + radius * math.cos(math.pi + math.radians(a)),
            radius * math.sin(math.pi + math.radians(a)),
            0,
        ]
        for a in degrees
    ]


def states(points):
    return [{"state": [p, [0, 0, 0]]} for p in points]


class OrbitJudgementTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.task = Path(self.tmp.name) / "task.json"
        self.reference = circle_points(range(361))
        self.task.write_text(
            json.dumps(
                {
                    "initial_pos": [0, 0, 0, 0, 0, 0],
                    "reference_path_raw": self.reference,
                    "instruction": "Circle around target",
                }
            )
        )

    def judge(self, points):
        p = Path(self.tmp.name) / "actual.json"
        p.write_text(json.dumps(states(points)))
        return scoring.check_single_trajectory(p, self.task, "Surround")

    def test_stationary_is_not_a_completed_orbit(self):
        self.assertFalse(self.judge([[0, 0, 0]] * 100)["is_valid"])

    def test_half_circle_is_not_completed(self):
        self.assertFalse(self.judge(circle_points(range(1, 181)))["is_valid"])

    def test_wrong_direction_full_circle_fails(self):
        self.assertFalse(self.judge(circle_points(range(-1, -361, -1)))["is_valid"])

    def test_backtracking_does_not_accumulate_completion(self):
        self.assertFalse(
            self.judge(circle_points(list(range(1, 181)) + list(range(179, -1, -1))))["is_valid"]
        )

    def test_exact_342_degree_boundary_includes_initial_position(self):
        self.assertTrue(self.judge(circle_points(range(1, 343)))["is_valid"])

    def test_341_degrees_is_below_threshold(self):
        self.assertFalse(self.judge(circle_points(range(1, 342)))["is_valid"])

    def test_full_circle_succeeds(self):
        self.assertTrue(self.judge(circle_points(range(1, 361)))["is_valid"])

    def test_radial_error_remains_required(self):
        self.assertFalse(self.judge(circle_points(range(1, 361), 350))["is_valid"])

    def test_reference_determines_direction(self):
        self.task.write_text(
            json.dumps(
                {
                    "initial_pos": [0, 0, 0, 0, 0, 0],
                    "reference_path_raw": circle_points(range(0, -361, -1)),
                }
            )
        )
        self.assertTrue(self.judge(circle_points(range(-1, -361, -1)))["is_valid"])
        self.assertFalse(self.judge(circle_points(range(1, 361)))["is_valid"])


class ControlIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from evaluation import closed_loop_runner

        cls.batch = closed_loop_runner

    def run_loop(self, points, budget, reference=None, chunk=10):
        pending = list(points)
        executed = []
        api = types.SimpleNamespace(
            set_obj_location=lambda actor, pos: executed.append(list(pos)),
            set_rotation=lambda *a: None,
            get_image=lambda *a: np.zeros((16, 16, 3), dtype=np.uint8),
        )
        env = types.SimpleNamespace(
            unwrapped=types.SimpleNamespace(unrealcv=api, player_list=["drone"])
        )

        def predict(**kw):
            group = pending[:chunk]
            del pending[:chunk]
            return {"action": [[*p, 0] for p in group]} if group else {"action": []}

        trajectory = []
        events = []
        with (
            patch.object(self.batch, "send_prediction_request", side_effect=predict),
            patch.object(self.batch.time, "sleep", return_value=None),
            patch.object(self.batch.cv2, "imwrite", return_value=True),
        ):
            ok = self.batch.control_loop(
                [0, 0, 0, 0, 0, 0],
                env,
                "orbit",
                budget,
                trajectory,
                "http://unused",
                0,
                0,
                event_log=events,
                drone_cam_id=5,
                orbit_reference=reference,
            )
        return ok, trajectory, events

    def test_budget_is_400_only_for_surround(self):
        categories = {"Surround": ["orbit.json"], "Rotate": ["rotate.json"]}
        self.assertEqual(self.batch.select_task_max_steps("orbit.json", 100, 400, categories), 400)
        self.assertEqual(self.batch.select_task_max_steps("rotate.json", 100, 400, categories), 100)

    def test_hard_300_action_limit_does_not_overshoot_chunk(self):
        ok, traj, events = self.run_loop(
            [[10 * i, 0, 0] for i in range(1, 400)], 300, circle_points(range(361)), chunk=7
        )
        self.assertTrue(ok)
        self.assertEqual(len(traj), 300)
        self.assertEqual(events[-1]["event"], "max_steps_stop")
        self.assertEqual(events[-1]["step_index"], 300)

    def test_400_budget_completes_orbit_that_needs_more_than_300_actions(self):
        ok, traj, events = self.run_loop(
            circle_points(range(1, 401)), 400, circle_points(range(361)), chunk=7
        )
        self.assertTrue(ok)
        self.assertGreater(len(traj), 300)
        self.assertLess(len(traj), 400)
        self.assertEqual(events[-1]["event"], "orbit_complete_stop")
        self.assertGreaterEqual(events[-1]["directed_net_angle_deg"], 342)

    def test_hard_400_action_limit_does_not_overshoot_chunk(self):
        ok, traj, events = self.run_loop(
            [[10 * i, 0, 0] for i in range(1, 500)], 400, circle_points(range(361)), chunk=7
        )
        self.assertTrue(ok)
        self.assertEqual(len(traj), 400)
        self.assertEqual(events[-1]["event"], "max_steps_stop")

    def test_real_controller_stops_when_shared_geometry_passes(self):
        ok, traj, events = self.run_loop(
            circle_points(range(5, 505, 5)), 300, circle_points(range(361))
        )
        self.assertTrue(ok)
        self.assertLess(len(traj), 300)
        self.assertEqual(events[-1]["event"], "orbit_complete_stop")
        self.assertGreaterEqual(events[-1]["directed_net_angle_deg"], 342)


if __name__ == "__main__":
    unittest.main(verbosity=2)
