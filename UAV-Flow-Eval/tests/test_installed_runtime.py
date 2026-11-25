# SPDX-License-Identifier: Apache-2.0
"""Ownership and output contracts without real UE or GPU use."""

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import rl_envs_linux as training
from uav_flow import runtime


class RuntimeContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def record(self):
        output = self.root / "a"
        record = dict(
            pid=123,
            starttime=10,
            pgid=123,
            output=str(output),
            group=True,
            pairs=[["-m", "env_server"], ["--output", str(output)]],
            markers=[],
        )
        info = dict(
            pid=123,
            starttime=10,
            pgid=123,
            cwd=str(output / "tmp"),
            argv=["python", "-m", "env_server", "--output", str(output)],
        )
        return record, info

    def test_reused_pid_foreign_cwd_wrong_arguments_and_group_are_never_signaled(self):
        record, original = self.record()
        for change in [
            dict(starttime=11),
            dict(cwd=str(self.root / "other")),
            dict(argv=["python", "-m", "env_server", "--output", "elsewhere"]),
            dict(pgid=999),
        ]:
            info = {**original, **change}
            with (
                self.subTest(change=change),
                patch.object(runtime, "process_info", return_value=info),
                patch.object(runtime.os, "kill") as kill,
                patch.object(runtime.os, "killpg") as killpg,
            ):
                runtime.stop_record(record)
                kill.assert_not_called()
                killpg.assert_not_called()

    def test_unknown_legacy_output_is_not_adopted_or_modified(self):
        output = self.root / "old"
        output.mkdir()
        (output / "current.json").write_text('{"slots": []}')
        before = list(output.iterdir())
        with self.assertRaisesRegex(RuntimeError, "not adopted"):
            with runtime.workflow_lock(output, "training", create=True):
                self.fail("legacy output adopted")
        self.assertEqual(list(output.iterdir()), before)

    def test_manifest_from_another_output_is_rejected(self):
        a, b = self.root / "a", self.root / "b"
        with runtime.workflow_lock(a, "training", create=True) as value:
            copied = copy.deepcopy(value)
        (b / "progress_info").mkdir(parents=True)
        runtime.atomic_json(b / "progress_info/current.json", copied)
        with self.assertRaisesRegex(RuntimeError, "not adopted"):
            runtime.manifest(b, "training")

    def test_foreign_process_record_in_a_manifest_cannot_be_stopped(self):
        output = self.root / "a"
        record, _ = self.record()
        record["output"] = str(self.root / "b")
        value = dict(
            slots=[dict(process=record, binary=str(output / "ue/Collection"))], started_after=0
        )
        with (
            patch.object(runtime, "binary_processes", return_value=[]),
            patch.object(runtime, "stop_record") as stop,
        ):
            training.stop(output, value)
            stop.assert_not_called()

    def test_output_context_contains_cwd_and_temporary_files_and_restores_caller(self):
        output = self.root / "result"
        caller = Path.cwd()
        old_tmp = os.environ.get("TMPDIR")
        with runtime.output_context(output, change_cwd=True):
            self.assertEqual(Path.cwd(), output / "tmp")
            with tempfile.NamedTemporaryFile() as file:
                self.assertTrue(Path(file.name).is_relative_to(output / "tmp"))
            Path("relative.txt").write_text("inside output")
        self.assertEqual(Path.cwd(), caller)
        self.assertEqual(os.environ.get("TMPDIR"), old_tmp)
        self.assertTrue((output / "tmp/relative.txt").exists())

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux /proc ownership")
    def test_stopping_one_real_process_group_leaves_other_output_alive(self):
        processes = []
        try:
            for name in ["a", "b"]:
                output = self.root / name
                script = "import time;time.sleep(120)"
                processes.append(
                    runtime.launch(
                        [sys.executable, "-B", "-c", script],
                        output,
                        output / "tmp",
                        output / "logs/process.log",
                        markers=[script],
                    )
                )
            runtime.stop_process(processes[0])
            self.assertIsNotNone(processes[0].poll())
            self.assertIsNone(processes[1].poll())
        finally:
            for process in processes:
                runtime.stop_process(process)

    def test_package_probe_failure_is_clear_and_does_not_launch_a_service(self):
        result = subprocess.CompletedProcess([], 1, "", "No package")
        with (
            patch.object(runtime.subprocess, "run", return_value=result) as probe,
            self.assertRaisesRegex(RuntimeError, "Install uav-flow-eval"),
        ):
            runtime.installed_package(self.root / "output")
        self.assertEqual(probe.call_args.args[0][0], sys.executable)

    def test_training_command_is_installed_module_and_only_output_is_configurable(self):
        output = self.root / "output"
        args = training.parse_args(["--output", str(output), "--gpus", "0", "--dry-run"])
        command = training.server_command(
            args, dict(port=5200, binary="private/Collection", gpu=0), output / "runs/instance"
        )
        self.assertEqual(command[0], sys.executable)
        self.assertIn("-m", command)
        self.assertEqual(command[command.index("-m") + 1], "env_server")
        self.assertFalse(any(token.endswith(".py") for token in command))
        self.assertFalse(output.exists())
        from evaluation import run_single_model as runner

        evaluation_args = runner.parse_args(
            [
                "--server-url",
                "http://localhost:6888",
                "--graphics-adapter",
                "0",
                "--output",
                str(output),
                "--dry-run",
            ]
        )
        command = runner.worker_command(
            evaluation_args, "http://localhost:6888", Path("private/Collection")
        )
        self.assertEqual(command[0], sys.executable)


class ManagedModelOutputTests(unittest.TestCase):
    def test_external_model_code_uses_output_cwd_logs_and_temporary_files(self):
        import time

        from evaluation import run_single_model as runner

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            repo = root / "model-repository"
            script = repo / "starVLA/model/framework/ServerQwenOFT_dis.py"
            script.parent.mkdir(parents=True)
            script.write_text(
                "import os,sys,json,tempfile,time\nfrom pathlib import Path\nwith tempfile.NamedTemporaryFile(delete=False) as f: temp=f.name\nPath('written.json').write_text(json.dumps({'python':sys.executable,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'checkpoint':os.environ.get('CHECKPOINT_PATH'),'config':os.environ.get('CONFIG_YAML'),'base':os.environ.get('BASE_VLM'),'cwd':os.getcwd(),'temp':temp,'stats':os.environ.get('STATS_PATH'),'cache':os.environ.get('HF_HOME')}))\nprint('fixture model ready',flush=True)\ntime.sleep(120)\n"
            )
            checkpoint = root / "model.pth"
            checkpoint.touch()
            config = root / "model.yaml"
            config.write_text("{}")
            base = root / "base-vlm"
            base.mkdir()
            output = root / "output"
            args = runner.parse_args(
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--model-gpu",
                    "0",
                    "--model-repo",
                    str(repo),
                    "--config",
                    str(config),
                    "--base-vlm",
                    str(base),
                    "--smoke",
                    "--output",
                    str(output),
                ]
            )
            with (
                patch.object(runner, "bind_check"),
                patch.dict(os.environ, {"STATS_PATH": "must-not-be-inherited"}),
            ):
                process = runner.start_model(args, output / "progress_info")
            try:
                deadline = time.monotonic() + 5
                artifact = output / "model/written.json"
                while not artifact.exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                data = json.loads(artifact.read_text())
                self.assertEqual(data["python"], sys.executable)
                self.assertEqual(data["gpu"], "0")
                self.assertEqual(data["checkpoint"], str(checkpoint))
                self.assertEqual(data["config"], str(config))
                self.assertEqual(data["base"], str(base))
                self.assertEqual(Path(data["cwd"]), output / "model")
                self.assertTrue(Path(data["temp"]).is_relative_to(output / "tmp"))
                self.assertTrue(Path(data["cache"]).is_relative_to(output / "cache"))
                self.assertIsNone(data["stats"])
                self.assertEqual(list(repo.rglob("written.json")), [])
                self.assertTrue((output / "progress_info/model.log").exists())
                self.assertFalse(list(output.rglob("*.py")))
            finally:
                runtime.stop_process(process)
