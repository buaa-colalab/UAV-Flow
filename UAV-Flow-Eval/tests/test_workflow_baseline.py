# SPDX-License-Identifier: Apache-2.0
"""Compare shared implementations with outputs captured before consolidation."""

import json
import unittest
from pathlib import Path

from tools.workflow_baseline import capture


class FrozenWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.expected = json.loads((root / "tests/fixtures/workflow_baseline.json").read_text())
        cls.actual = capture(root)

    def test_task_sets_and_classification_are_unchanged(self):
        self.assertEqual(self.actual["data"], self.expected["data"])

    def test_lifecycle_profiles_preserve_launch_and_camera_behavior(self):
        self.assertEqual(self.actual["simulator"], self.expected["simulator"])

    def test_ten_categories_preserve_checker_results_and_ndtw(self):
        self.assertEqual(self.actual["checkers"], self.expected["checkers"])

    def test_orbit_policies_preserve_boundary_direction_and_backtracking(self):
        self.assertEqual(self.actual["orbit_policy"], self.expected["orbit_policy"])

    def test_short_trajectory_trimming_is_unchanged(self):
        self.assertEqual(self.actual["batch"], self.expected["batch"])


if __name__ == "__main__":
    unittest.main()
