# SPDX-License-Identifier: Apache-2.0
"""Exercise shared imports, per-instance policies and detached runtime copies."""

import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import env_server
from evaluation import closed_loop_runner as batch
from gym_unrealcv.envs.base_env import UnrealCv_base
from gym_unrealcv.envs.runtime_policy import runtime_policy

ROOT = Path(__file__).resolve().parents[1]


class SharedWorkflowTests(unittest.TestCase):
    def test_evaluation_import_and_instance_do_not_change_training_profile(self):
        training = UnrealCv_base.__new__(UnrealCv_base)
        evaluation = UnrealCv_base.__new__(UnrealCv_base)
        evaluation.runtime_profile = "evaluation"
        self.assertEqual(training.runtime_profile, "training")
        self.assertEqual(UnrealCv_base.runtime_profile, "training")
        self.assertIs(batch.gym_unrealcv, env_server.gym_unrealcv)
        self.assertEqual(evaluation.runtime_profile, "evaluation")
        with self.assertRaises(ValueError):
            runtime_policy("misspelled-profile")

    def test_evaluation_selects_its_profile_before_first_reset(self):
        env = MagicMock()
        env.unwrapped = env
        env.reset.side_effect = lambda: self.assertEqual(env.runtime_profile, "evaluation")
        args = types.SimpleNamespace(
            env_id="unused",
            time_dilation=0,
            ue_extra_args="",
            seed=0,
            unreal_op_retries=0,
            unreal_retry_sleep=0,
        )
        with (
            patch.object(batch.gym, "make", return_value=env),
            patch.object(batch.configUE, "ConfigUEWrapper", side_effect=lambda value, **kw: value),
            patch.object(
                batch.augmentation,
                "RandomPopulationWrapper",
                side_effect=lambda value, *a, **kw: value,
            ),
        ):
            batch.build_env(args)
        env.reset.assert_called_once()


if __name__ == "__main__":
    unittest.main()
