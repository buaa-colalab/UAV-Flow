# SPDX-License-Identifier: Apache-2.0
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from evaluation import check_trajectories_batch as checker
from evaluation import closed_loop_runner as batch
from evaluation import run_single_model as runner
from evaluation import scoring


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tasks = self.root / "tasks"
        self.tasks.mkdir()
        self.out = self.root / "output"
        (self.out / "trajectories").mkdir(parents=True)
        self.classification = self.root / "classification.json"
        self.classification.write_text(json.dumps({"Rotate": ["a.json", "b.json", "unused.json"]}))
        for name in ["a.json", "b.json"]:
            (self.tasks / name).write_text(
                json.dumps(
                    {
                        "reference_path_preprocessed": [
                            [0, 0, 0, 0, yaw, 0] for yaw in range(0, 91, 18)
                        ]
                    }
                )
            )
            (self.out / "trajectories" / name).write_text(
                json.dumps([{"state": [[0, 0, 0], [0, yaw, 0]]} for yaw in range(0, 91, 18)])
            )
        self.enterContext(
            patch.object(
                checker,
                "check_single_trajectory",
                side_effect=lambda trajectory, reference, category, *a: {
                    "filename": trajectory.name,
                    "category": category,
                    "is_valid": True,
                    "reward": 1.0,
                },
            )
        )
        self.enterContext(
            patch.object(
                runner.runtime,
                "installed_package",
                return_value={
                    "version": "test",
                    "modules": {
                        "evaluation.closed_loop_runner": str(runner.ROOT / "closed_loop_runner.py")
                    },
                },
            )
        )
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def test_rescoring_ignores_all_reports_and_keeps_exact_denominator(self):
        for name in [
            "check_results.json",
            "metrics.json",
            "per_task_metrics.json",
            "reuse_note.json",
        ]:
            (self.out / "trajectories" / name).write_text("{}")
        first = scoring.score_run(self.out, self.tasks, self.classification)
        second = scoring.score_run(self.out, self.tasks, self.classification)
        self.assertEqual(first, second)
        self.assertEqual(first["overall"]["tasks"], 2)
        self.assertEqual(first["overall"]["sr_percent"], 100)
        self.assertEqual(first["overall"]["ndtw"], 1)
        self.assertEqual(checker.check_single_trajectory.call_count, 4)

    def test_missing_expected_task_does_not_shrink_denominator(self):
        (self.out / "trajectories/b.json").unlink()
        result = checker.check_all_trajectories(
            self.out / "trajectories", self.tasks, self.classification
        )
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["errors"], 1)
        with self.assertRaises(RuntimeError):
            scoring.score_run(self.out, self.tasks, self.classification)

    def test_explicit_task_manifest_is_used(self):
        result = checker.check_all_trajectories(
            self.out / "trajectories",
            self.tasks,
            self.classification,
            expected_filenames=["a.json"],
        )
        self.assertEqual(result["total"], 1)

    def test_duplicate_and_empty_manifests_are_rejected(self):
        for names in [[], ["a.json", "a.json"]]:
            with self.assertRaises(ValueError):
                checker.check_all_trajectories(
                    self.out / "trajectories",
                    self.tasks,
                    self.classification,
                    expected_filenames=names,
                )


class ResetTests(unittest.TestCase):
    def test_reset_connection_failure_propagates(self):
        with patch.object(batch.requests, "post", side_effect=requests.Timeout("reset timed out")):
            with self.assertRaises(batch.ModelResetError):
                batch.reset_model("http://unused/predict")

    def test_reset_http_failure_propagates(self):
        response = MagicMock()
        response.raise_for_status.side_effect = requests.HTTPError("500")
        with patch.object(batch.requests, "post", return_value=response):
            with self.assertRaises(batch.ModelResetError):
                batch.reset_model("http://unused/predict")

    def test_reset_application_error_propagates(self):
        response = MagicMock()
        response.json.return_value = {"status": "error", "message": "not ready"}
        with patch.object(batch.requests, "post", return_value=response):
            with self.assertRaises(batch.ModelResetError):
                batch.reset_model("http://unused/predict")

    def test_reset_success(self):
        response = MagicMock()
        response.json.return_value = {"status": "success"}
        with patch.object(batch.requests, "post", return_value=response) as post:
            batch.reset_model("http://unused/predict")
            self.assertEqual(post.call_args.args[0], "http://unused/reset")

    def test_failed_reset_prevents_prediction_and_returns_nonzero_batch_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tasks = root / "tasks"
            tasks.mkdir()
            (tasks / "fixture.json").write_text(
                json.dumps({"initial_pos": [0, 0, 300, 0, 0, 0], "instruction": "Rotate left"})
            )
            (root / "out/trajectories/logs").mkdir(parents=True)
            with (
                patch.object(batch, "configure_logging", return_value="isolated.log"),
                patch.object(batch, "ensure_env_ready", return_value=MagicMock()),
                patch.object(batch, "create_obj_if_needed"),
                patch.object(batch, "draw_2d_trajectory_from_log"),
                patch.object(batch, "draw_3d_trajectory_from_log"),
                patch.object(batch, "control_loop") as loop,
                patch.object(batch.requests, "post", side_effect=requests.Timeout("reset timeout")),
                patch.object(batch.time, "sleep"),
            ):
                code = batch.main(
                    ["-f", str(tasks), "-o", str(root / "out"), "--task_retries", "0"]
                )
            self.assertNotEqual(code, 0)
            loop.assert_not_called()
            logs = list((root / "out/trajectories/logs").glob("*_actions.json"))
            self.assertFalse(json.loads(logs[0].read_text())["final"]["execution_ok"])


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tasks = self.root / "tasks"
        self.tasks.mkdir()
        (self.tasks / "a.json").write_text("{}")
        self.classification = self.root / "classification.json"
        self.classification.write_text(json.dumps({"Land": ["a.json"]}))
        self.output = self.root / "output"
        self.base = [
            "--output",
            str(self.output),
            "--task-dir",
            str(self.tasks),
            "--classification",
            str(self.classification),
            "--ue-package",
            str(self.root / "fake-package"),
            "--ue-user",
            "root",
        ]
        self.enterContext(
            patch.object(
                runner.runtime,
                "installed_package",
                return_value={
                    "version": "test",
                    "modules": {
                        "evaluation.closed_loop_runner": str(runner.ROOT / "closed_loop_runner.py")
                    },
                },
            )
        )
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.enterContext(contextlib.redirect_stderr(io.StringIO()))

    def options(self, *extra):
        return runner.parse_args(self.base + list(extra))

    def test_smoke_owned_model_cleanup_on_success(self):
        checkpoint = self.root / "model.pth"
        checkpoint.write_bytes(b"fixture")
        args = self.options("--smoke", "--checkpoint", str(checkpoint), "--model-gpu", "0")
        process = MagicMock()
        with (
            patch.object(runner, "start_model", return_value=process),
            patch.object(runner, "wait_model"),
            patch.object(runner, "model_smoke"),
            patch.object(runner, "stop_process") as stop,
            patch.object(runner, "prepare_ue") as create,
        ):
            self.assertEqual(runner.execute(args), 0)
            stop.assert_called_once_with(process)
            create.assert_not_called()
        self.assertTrue(
            json.loads((self.output / "progress_info/status.json").read_text())["smoke_passed"]
        )

    def test_smoke_cleanup_also_runs_on_predict_failure(self):
        checkpoint = self.root / "model.pth"
        checkpoint.write_bytes(b"fixture")
        args = self.options("--smoke", "--checkpoint", str(checkpoint), "--model-gpu", "0")
        process = MagicMock()
        with (
            patch.object(runner, "start_model", return_value=process),
            patch.object(runner, "wait_model"),
            patch.object(runner, "model_smoke", side_effect=RuntimeError("predict failed")),
            patch.object(runner, "stop_process") as stop,
        ):
            self.assertEqual(runner.execute(args), 1)
            stop.assert_called_once_with(process)

    def test_readiness_failure_still_cleans_owned_model(self):
        checkpoint = self.root / "model.pth"
        checkpoint.write_bytes(b"fixture")
        args = self.options("--smoke", "--checkpoint", str(checkpoint), "--model-gpu", "0")
        process = MagicMock()
        with (
            patch.object(runner, "start_model", return_value=process),
            patch.object(runner, "wait_model", side_effect=TimeoutError()),
            patch.object(runner, "stop_process") as stop,
        ):
            self.assertEqual(runner.execute(args), 1)
            stop.assert_called_once_with(process)

    def test_borrowed_model_smoke_never_starts_or_stops_external_service(self):
        args = self.options("--smoke", "--server-url", "http://localhost:6888")
        with (
            patch.object(runner, "start_model") as start,
            patch.object(runner, "wait_model"),
            patch.object(runner, "model_smoke"),
            patch.object(runner, "stop_process") as stop,
        ):
            self.assertEqual(runner.execute(args), 0)
            start.assert_not_called()
            stop.assert_not_called()

    def fake_workspace(self):
        workspace = self.output
        return workspace, workspace / "ue/Collection"

    def test_nonzero_batch_exit_propagates_and_does_not_score(self):
        args = self.options("--server-url", "http://localhost:6888", "--graphics-adapter", "1")
        workspace, binary = self.fake_workspace()
        with (
            patch.object(runner, "wait_model"),
            patch.object(runner, "bind_check"),
            patch.object(runner, "prepare_ue", return_value=binary),
            patch.object(runner, "run_batch", return_value=17),
            patch.object(runner, "score_run") as score,
            patch.object(runner, "cleanup_ue") as cleanup,
        ):
            self.assertEqual(runner.execute(args), 1)
            score.assert_not_called()
            cleanup.assert_called_once_with(binary, self.output, unittest.mock.ANY)
        self.assertFalse(
            json.loads((self.output / "progress_info/status.json").read_text())["all_completed"]
        )

    def test_single_model_full_run_scores_once_and_completes(self):
        args = self.options("--server-url", "http://localhost:6888", "--graphics-adapter", "1")
        workspace, binary = self.fake_workspace()
        with (
            patch.object(runner, "wait_model"),
            patch.object(runner, "bind_check"),
            patch.object(runner, "prepare_ue", return_value=binary),
            patch.object(runner, "run_batch", return_value=0) as run,
            patch.object(runner, "score_run", return_value={"overall": {"tasks": 1}}) as score,
            patch.object(runner, "cleanup_ue"),
        ):
            self.assertEqual(runner.execute(args), 0)
            run.assert_called_once()
            score.assert_called_once()
            self.assertEqual(score.call_args.args[-1], ["a.json"])
        self.assertTrue(
            json.loads((self.output / "progress_info/status.json").read_text())["all_completed"]
        )

    def test_existing_results_are_not_reused_for_a_new_trial(self):
        self.output.mkdir()
        (self.output / "old.json").write_text("{}")
        args = self.options("--server-url", "http://localhost:6888", "--graphics-adapter", "1")
        with self.assertRaisesRegex(ValueError, "not empty"):
            runner.execute(args)

    def test_score_only_can_be_repeated_without_starting_services(self):
        args = self.options("--score-only")
        with (
            patch.object(runner, "score_run") as score,
            patch.object(runner, "start_model") as start,
            patch.object(runner, "prepare_ue") as create,
        ):
            self.assertEqual(runner.execute(args), 0)
            self.assertEqual(runner.execute(args), 0)
            self.assertEqual(score.call_count, 2)
            start.assert_not_called()
            create.assert_not_called()

    def test_dry_run_writes_nothing(self):
        args = self.options(
            "--dry-run", "--server-url", "http://localhost:6888", "--graphics-adapter", "1"
        )
        self.assertEqual(runner.execute(args), 0)
        self.assertFalse(self.output.exists())

    def test_training_ports_are_rejected(self):
        with self.assertRaises(ValueError):
            runner.api_base("http://127.0.0.1:5008")
        with self.assertRaises(SystemExit):
            self.options(
                "--server-url",
                "http://localhost:6888",
                "--graphics-adapter",
                "1",
                "--ue-port",
                "9400",
            )


if __name__ == "__main__":
    unittest.main()
