# SPDX-License-Identifier: Apache-2.0
"""Portable workflow configuration without launching services or using GPUs."""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation import run_single_model as evaluation
from tools import rl_envs_linux as training
from uav_flow import cli

ROOT = Path(__file__).resolve().parents[1]


class PortableConfigurationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="uav-configuration-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.settings = self.root / "workflow.toml"
        self.enterContext(patch.dict(os.environ, {"UAV_FLOW_CONFIG": ""}))
        self.enterContext(contextlib.redirect_stderr(io.StringIO()))

    def write(self, text):
        if "[training]" in text and "output" not in text:
            text = text.replace("[training]", '[training]\noutput="default-training"', 1)
        self.settings.write_text(text)

    def train(self, *arguments):
        defaults = (
            []
            if "output=" in self.settings.read_text() or "output =" in self.settings.read_text()
            else ["--output", str(self.root / "default-training")]
        )
        return training.parse_args(
            ["--settings", str(self.settings), "--dry-run", *defaults, *arguments]
        )

    def evaluate(self, *arguments):
        return evaluation.parse_args(["--settings", str(self.settings), "--dry-run", *arguments])

    def test_training_cli_overrides_toml_and_paths_are_config_relative(self):
        self.write(
            '[training]\npackage="assets/ue"\noutput="runs/train"\ngpus=[1,3]\nbase_port=5100\nuser="simulator"\n'
        )
        args = self.train("--base-port", "5200")
        self.assertEqual(args.package, self.root / "assets/ue")
        self.assertEqual(args.output, self.root / "runs/train")
        self.assertEqual(args.gpus, [1, 3])
        self.assertEqual(args.base_port, 5200)

    def test_cli_relative_path_uses_invocation_directory(self):
        self.write('[training]\ngpus=[0]\noutput="from-toml"\n')
        args = self.train("--output", "from-cli")
        self.assertEqual(args.output, Path.cwd() / "from-cli")

    def test_environment_selects_file_and_cli_can_select_another_file(self):
        self.write("[training]\ngpus=[2]\nbase_port=5100\n")
        another = self.root / "another.toml"
        another.write_text('[training]\noutput="another-training"\ngpus=[3]\nbase_port=5200\n')
        with patch.dict(os.environ, {"UAV_FLOW_CONFIG": str(self.settings)}):
            self.assertEqual(training.parse_args(["--dry-run"]).gpus, [2])
            self.assertEqual(
                training.parse_args(["--settings", str(another), "--dry-run"]).base_port, 5200
            )

    def test_removed_options_are_rejected(self):
        self.write('[training]\ngpus=[0]\npython="other-python"\n')
        with self.assertRaises(SystemExit):
            self.train()
        self.write("[training]\ngpus=[0]\n")
        with self.assertRaises(SystemExit):
            self.train("--python", "other-python")
        for field, options in [
            ("eval_python", ["--eval-python", "other-python"]),
            ("model_python", ["--model-python", "other-python"]),
            ("local", ["--local"]),
        ]:
            self.write(
                '[evaluation]\nserver_url="http://localhost:6888"\noutput="result"\n'
                + field
                + '="other-python"\n'
            )
            with self.assertRaises(SystemExit):
                self.evaluate()
            self.write('[evaluation]\nserver_url="http://localhost:6888"\noutput="result"\n')
            with self.assertRaises(SystemExit):
                self.evaluate(*options)

    def test_model_yaml_and_workflow_settings_are_independent(self):
        self.write(
            '[evaluation]\ncheckpoint="model.pth"\nmodel_gpu=0\nconfig="model.yaml"\noutput="result"\ngraphics_adapter=1\n'
        )
        args = self.evaluate("--config", "override.yaml")
        self.assertEqual(args.checkpoint, self.root / "model.pth")
        self.assertEqual(args.output, self.root / "result")
        self.assertEqual(args.config, Path.cwd() / "override.yaml")

    def test_cli_model_selection_overrides_opposing_config_selection(self):
        self.write(
            '[evaluation]\nserver_url="http://localhost:6888"\noutput="result"\ngraphics_adapter=0\n'
        )
        args = self.evaluate("--checkpoint", "chosen.pth", "--model-gpu", "2")
        self.assertIsNone(args.server_url)
        self.assertEqual(args.checkpoint, Path.cwd() / "chosen.pth")

    def test_ambiguous_config_model_selection_is_rejected(self):
        self.write(
            '[evaluation]\ncheckpoint="model.pth"\nserver_url="http://localhost:6888"\nmodel_gpu=0\noutput="result"\ngraphics_adapter=0\n'
        )
        with self.assertRaises(SystemExit):
            self.evaluate()

    def test_invalid_types_unknown_fields_and_one_shot_flags_are_rejected(self):
        for content in [
            "[training]\ngpus=[]",
            "[training]\ngpus=[true]",
            '[training]\ngpus=[0]\nbase_port="5100"',
            '[training]\ngpus=[0]\nparallel="yes"',
            '[training]\ngpus=[0]\npackge="typo"',
            "[training]\ngpus=[0]\ndry_run=true",
            "[unknown]\nvalue=1",
            "[training",
        ]:
            with self.subTest(content=content):
                self.write(content)
                with self.assertRaises(SystemExit):
                    self.train()

    def test_unset_path_variable_is_rejected(self):
        self.write('[training]\ngpus=[0]\npackage="${UAV_TEST_UNSET_DIRECTORY}/ue"\n')
        with patch.dict(os.environ):
            os.environ.pop("UAV_TEST_UNSET_DIRECTORY", None)
            with self.assertRaises(SystemExit):
                self.train()

    def test_adapters_and_root_user_are_not_guessed_for_launch(self):
        with self.assertRaises(SystemExit):
            training.parse_args(["--dry-run"])
        with patch("uav_flow.settings.os.geteuid", return_value=0):
            with self.assertRaises(SystemExit):
                training.parse_args(
                    [
                        "--output",
                        str(self.root / "training"),
                        "--gpus",
                        "0",
                        "--package",
                        str(self.root),
                    ]
                )

    def test_unknown_status_runtime_does_not_query_live_default_ports(self):
        with patch.object(training, "healthy") as healthy:
            with self.assertRaisesRegex(RuntimeError, "Runtime manifest not found"):
                training.main(["status", "--output", str(self.root / "missing")])
            healthy.assert_not_called()

    def test_training_dry_run_has_no_runtime_network_or_launch_side_effects(self):
        self.write(
            '[training]\ngpus=[0,2]\noutput="train-runtime"\nbase_port=5200\nue_base_port=9500\n'
        )
        output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            patch.object(training, "preflight") as preflight,
            patch.object(training, "prepare") as prepare,
            patch.object(training, "owned_records") as pids,
            patch.object(training, "healthy") as healthy,
        ):
            self.assertEqual(
                cli.main(["train-env", "--settings", str(self.settings), "--dry-run"]), 0
            )
            for function in [preflight, prepare, pids, healthy]:
                function.assert_not_called()
        plan = json.loads(output.getvalue())
        self.assertEqual(
            plan["slots"],
            [
                {"graphics_adapter": 0, "http_port": 5200, "ue_port": 9500},
                {"graphics_adapter": 2, "http_port": 5201, "ue_port": 9501},
            ],
        )
        self.assertFalse((self.root / "train-runtime").exists())

    def test_evaluation_dry_run_preserves_protocol_without_touching_resources(self):
        self.write(
            '[evaluation]\nserver_url="http://localhost:6888"\ngraphics_adapter=0\noutput="result"\n'
        )
        output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            patch.object(evaluation, "start_model") as start,
            patch.object(evaluation, "wait_model") as wait,
            patch.object(evaluation, "prepare_ue") as create,
            patch.object(evaluation, "bind_check") as bind,
        ):
            self.assertEqual(
                cli.main(["evaluate", "--settings", str(self.settings), "--dry-run"]), 0
            )
            for function in [start, wait, create, bind]:
                function.assert_not_called()
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["task_count"], 273)
        self.assertEqual(plan["protocol"], "radial30_net342_surround400_others100")
        self.assertFalse((self.root / "result").exists())
        self.assertFalse((self.root / "eval-runtime").exists())

    def test_missing_full_simulation_config_fails_before_output_or_model_requests(self):
        args = evaluation.parse_args(
            [
                "--server-url",
                "http://localhost:6888",
                "--graphics-adapter",
                "0",
                "--output",
                str(self.root / "result"),
            ]
        )
        with patch.object(evaluation, "wait_model") as wait:
            with self.assertRaisesRegex(ValueError, "requires --ue-package"):
                evaluation.execute(args)
            wait.assert_not_called()
        self.assertFalse((self.root / "result").exists())

    def test_missing_managed_model_config_cannot_launch_a_process(self):
        args = evaluation.parse_args(
            [
                "--checkpoint",
                str(self.root / "model.pth"),
                "--model-gpu",
                "0",
                "--smoke",
                "--output",
                str(self.root / "result"),
            ]
        )
        with patch.object(evaluation.runtime.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(ValueError, "requires --model-repo"):
                evaluation.start_model(args, self.root)
            launch.assert_not_called()

    def test_public_modules_work_from_unrelated_directory(self):
        cases = [
            ["uav_flow", "--help"],
            ["uav_flow", "train-env", "--help"],
            ["uav_flow", "evaluate", "--help"],
            [
                "uav_flow",
                "train-env",
                "start",
                "--gpus",
                "0",
                "--output",
                str(self.root / "runtime"),
                "--dry-run",
            ],
            [
                "uav_flow",
                "evaluate",
                "--server-url",
                "http://localhost:6888",
                "--graphics-adapter",
                "0",
                "--output",
                str(self.root / "output"),
                "--dry-run",
            ],
            ["tools.smoke_rl_http", "--help"],
            ["tools.verify_camera_render", "--help"],
            ["tools.workflow_baseline", "--help"],
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                result = subprocess.run(
                    [sys.executable, "-B", "-m", *arguments],
                    cwd=self.root,
                    env={
                        **os.environ,
                        "PYTHONPATH": str(ROOT / "src"),
                        "PYTHONDONTWRITEBYTECODE": "1",
                    },
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                if "--dry-run" in arguments:
                    plan = json.loads(result.stdout)
                    if "train-env" in arguments:
                        self.assertEqual(plan["action"], "start")
                        self.assertEqual(len(plan["slots"]), 1)
                    else:
                        self.assertEqual(plan["task_count"], 273)
        self.assertFalse((self.root / "runtime").exists())
        self.assertFalse((self.root / "output").exists())


if __name__ == "__main__":
    unittest.main()
