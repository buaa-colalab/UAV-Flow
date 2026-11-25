# SPDX-License-Identifier: Apache-2.0
import json
import tempfile
import unittest
from pathlib import Path

from check.check_trajectory_rotate import check_rotate_trajectory


class CheckRotateTrajectoryTests(unittest.TestCase):
    def run_check(self, instruction, yaw_values, roll_values=None, pitch_values=None):
        point_count = len(yaw_values)
        roll_values = roll_values or [0.0] * point_count
        pitch_values = pitch_values or [0.0] * point_count
        trajectory = [
            {
                "state": [
                    [0.0, 0.0, 0.0],
                    [roll_values[index], yaw_values[index], pitch_values[index]],
                ]
            }
            for index in range(point_count)
        ]
        task = {
            "initial_pos": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "instruction": instruction,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            trajectory_path = temp_path / "trajectory.json"
            task_path = temp_path / "task.json"
            trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")
            task_path.write_text(json.dumps(task), encoding="utf-8")
            return check_rotate_trajectory(
                str(trajectory_path),
                test_jsons_file=str(task_path),
            )

    def test_uses_fifth_dimension_yaw_when_first_frame_is_zero(self):
        result = self.run_check(
            "Rotate 30 degrees to the right",
            yaw_values=[0.0, 30.0],
            roll_values=[170.0, -170.0],
            pitch_values=[-160.0, 160.0],
        )

        self.assertEqual(result["angle_type"], "yaw")
        self.assertEqual(result["initial_yaw"], 0.0)
        self.assertEqual(result["final_yaw"], 30.0)
        self.assertAlmostEqual(result["actual_rotation"], 30.0)
        self.assertAlmostEqual(result["expected_rotation"], 30.0)
        self.assertAlmostEqual(result["rotation_diff"], 0.0)
        self.assertAlmostEqual(result["rotation_score"], 0.7)

    def test_left_rotation_is_negative_and_direction_matters(self):
        correct = self.run_check(
            "Rotate 30 degrees to the left",
            yaw_values=[0.0, -30.0],
        )
        wrong_direction = self.run_check(
            "Rotate 30 degrees to the left",
            yaw_values=[0.0, 30.0],
        )

        self.assertAlmostEqual(correct["actual_rotation"], -30.0)
        self.assertAlmostEqual(correct["expected_rotation"], -30.0)
        self.assertAlmostEqual(correct["rotation_diff"], 0.0)
        self.assertAlmostEqual(correct["rotation_score"], 0.7)
        self.assertAlmostEqual(wrong_direction["actual_rotation"], 30.0)
        self.assertAlmostEqual(wrong_direction["rotation_diff"], 60.0)
        self.assertAlmostEqual(wrong_direction["rotation_score"], 0.3)

    def test_crossing_angle_boundary_does_not_create_360_degree_jump(self):
        result = self.run_check(
            "Rotate 200 degrees to the right",
            yaw_values=[0.0, 170.0, -175.0, -160.0],
        )

        self.assertAlmostEqual(result["actual_rotation"], 200.0)
        self.assertAlmostEqual(result["rotation_diff"], 0.0)

    def test_nonzero_first_prediction_is_counted_from_reset_zero(self):
        result = self.run_check(
            "Rotate 33 degrees to the right",
            yaw_values=[3.0, 33.0],
        )

        self.assertAlmostEqual(result["rotation_start_yaw"], 0.0)
        self.assertAlmostEqual(result["actual_rotation"], 33.0)
        self.assertAlmostEqual(result["rotation_diff"], 0.0)

    def test_yaw_history_preserves_left_and_right_at_180_degrees(self):
        left = self.run_check(
            "Rotate 180 degrees to the left",
            yaw_values=[0.0, -90.0, 180.0],
        )
        right = self.run_check(
            "Rotate 180 degrees to the right",
            yaw_values=[0.0, 90.0, 180.0],
        )

        self.assertAlmostEqual(left["actual_rotation"], -180.0)
        self.assertAlmostEqual(left["expected_rotation"], -180.0)
        self.assertAlmostEqual(left["rotation_diff"], 0.0)
        self.assertAlmostEqual(right["actual_rotation"], 180.0)
        self.assertAlmostEqual(right["expected_rotation"], 180.0)
        self.assertAlmostEqual(right["rotation_diff"], 0.0)

    def test_can_accumulate_rotation_beyond_360_degrees(self):
        result = self.run_check(
            "Rotate 450 degrees to the right",
            yaw_values=[0.0, 90.0, 180.0, -90.0, 0.0, 90.0],
        )

        self.assertAlmostEqual(result["actual_rotation"], 450.0)
        self.assertAlmostEqual(result["expected_rotation"], 450.0)
        self.assertAlmostEqual(result["rotation_diff"], 0.0)
        self.assertAlmostEqual(result["rotation_score"], 0.7)

    def test_opposite_165_degree_rotation_is_not_circularly_shortened(self):
        result = self.run_check(
            "Rotate 165 degrees to the left",
            yaw_values=[0.0, 165.0],
        )

        self.assertAlmostEqual(result["actual_rotation"], 165.0)
        self.assertAlmostEqual(result["expected_rotation"], -165.0)
        self.assertAlmostEqual(result["rotation_diff"], 330.0)
        self.assertAlmostEqual(result["rotation_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
