#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One model, one fresh pass over the test set. No SSH or historical job queues."""

import argparse
import base64
import hashlib
import io
import json
import math
import os
import signal
import socket
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from evaluation.runtime import cleanup_ue, prepare_ue
from uav_flow import runtime
from uav_flow.runtime import atomic_json, stop_process
from uav_flow.settings import default_ue_user, parse_settings


def score_run(*args, **kwargs):
    from evaluation.scoring import score_run as score

    return score(*args, **kwargs)


ROOT = Path(__file__).resolve().parent
HTTP = build_opener(ProxyHandler({}))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    model = parser.add_mutually_exclusive_group()
    model.add_argument(
        "--checkpoint", type=Path, help="Launch a local QwenOFT-compatible model server"
    )
    model.add_argument(
        "--server-url", help="Use an existing model API; this controller never stops it"
    )
    parser.add_argument("--model-name", default="single_model")
    parser.add_argument(
        "--output", type=Path, help="Fresh output directory (CLI or evaluation.output in settings)"
    )
    parser.add_argument("--task-dir", type=Path, default=ROOT / "test_jsons")
    parser.add_argument("--classification", type=Path, default=ROOT / "classified_instr.json")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Only test model reset/predict, then clean up owned model; no UE",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Re-score an existing output using expected task filenames",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without creating files or starting services",
    )
    parser.add_argument(
        "--graphics-adapter", type=int, help="Explicit Vulkan adapter; verify against GPU UUIDs"
    )
    parser.add_argument("--ue-port", type=int, default=9488)
    parser.add_argument(
        "--ue-package", type=Path, help="External UE package; required for a full simulation"
    )
    parser.add_argument(
        "--ue-user",
        default=default_ue_user(),
        help="UE account; root callers must choose explicitly",
    )
    parser.add_argument("--model-gpu", type=int)
    parser.add_argument("--model-port", type=int, default=6888)
    parser.add_argument("--model-repo", type=Path)
    parser.add_argument(
        "--config", type=Path, help="Model YAML configuration; independent of workflow --settings"
    )
    parser.add_argument("--base-vlm", type=Path)
    parser.add_argument(
        "--stats-path",
        type=Path,
        help="Explicit normalization statistics for the checkpoint, if required",
    )
    parser.add_argument("--start-timeout", type=float, default=1200)
    parser.add_argument("--eval-timeout", type=float, default=21600)
    args = parse_settings(parser, "evaluation", argv)
    if args.output is None:
        parser.error("Choose --output or evaluation.output in --settings")
    if args.checkpoint and args.server_url:
        parser.error("Choose exactly one of --checkpoint and --server-url")
    if not args.score_only and not (args.checkpoint or args.server_url):
        parser.error("Choose --checkpoint or --server-url")
    if args.score_only and args.smoke:
        parser.error("--score-only and --smoke are mutually exclusive")
    if (
        not args.score_only
        and not args.smoke
        and (args.graphics_adapter is None or args.graphics_adapter < 0)
    ):
        parser.error("A full evaluation requires --graphics-adapter")
    if args.checkpoint and (args.model_gpu is None or args.model_gpu < 0):
        parser.error("--checkpoint requires an explicit --model-gpu")
    if args.ue_port in range(9400, 9404) or args.model_port in range(5008, 5012):
        parser.error("These default training ports are reserved; choose private evaluation ports")
    if not 1 <= args.ue_port <= 9999 or not 1 <= args.model_port <= 65535:
        parser.error("UE port must be 1..9999; HTTP port must be 1..65535")
    if (
        not math.isfinite(args.start_timeout)
        or not math.isfinite(args.eval_timeout)
        or args.start_timeout <= 0
        or args.eval_timeout <= 0
    ):
        parser.error("Timeouts must be positive")
    return args


def api_base(url):
    base = url.rstrip("/").removesuffix("/predict")
    parts = urlsplit(base)
    if (
        parts.scheme not in ("http", "https")
        or not parts.hostname
        or parts.username
        or parts.password
    ):
        raise ValueError("Expected an HTTP model API URL without embedded credentials")
    if parts.hostname in ("localhost", "127.0.0.1") and parts.port in range(5008, 5012):
        raise ValueError("Training environment ports are not model inference endpoints")
    return base


def request_json(base, endpoint, payload):
    request = Request(
        base + endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with HTTP.open(request, timeout=15) as response:
        data = json.load(response)
    if isinstance(data, dict) and data.get("status") == "error":
        raise RuntimeError(data.get("message", "Model API rejected request"))
    return data


def task_manifest(args):
    classification = json.loads(args.classification.read_text())
    records = []
    for path in sorted(args.task_dir.glob("*.json")):
        categories = [name for name, names in classification.items() if path.name in names]
        if len(categories) != 1:
            raise ValueError(f"Task must have exactly one category: {path.name}")
        records.append(
            dict(
                filename=path.name,
                category=categories[0],
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    if not records:
        raise ValueError("The test set is empty")
    return records


def bind_check(port):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))


def start_model(args, progress_info):
    for name in ("model_repo", "config", "base_vlm"):
        if getattr(args, name) is None:
            raise ValueError(
                f"Managed model requires --{name.replace('_', '-')} or its evaluation setting"
            )
    for path in [
        args.checkpoint,
        args.config,
        args.base_vlm,
        args.model_repo / "starVLA/model/framework/ServerQwenOFT_dis.py",
    ]:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.stats_path is not None and not args.stats_path.is_file():
        raise FileNotFoundError(args.stats_path)
    bind_check(args.model_port)
    env = os.environ.copy()
    env.update(runtime.environment(args.output))
    env.update(
        CUDA_VISIBLE_DEVICES=str(args.model_gpu),
        GPU_ID="0",
        HTTP_PORT=str(args.model_port),
        CHECKPOINT_PATH=str(args.checkpoint.resolve()),
        CONFIG_YAML=str(args.config.resolve()),
        BASE_VLM=str(args.base_vlm.resolve()),
        PYTHONPATH=str(args.model_repo.resolve()),
    )
    # Do not inherit another experiment's normalization selection accidentally.
    env.pop("STATS_PATH", None)
    if args.stats_path:
        env["STATS_PATH"] = str(args.stats_path.resolve())
    script = args.model_repo / "starVLA/model/framework/ServerQwenOFT_dis.py"
    return runtime.launch(
        [sys.executable, "-B", "-u", str(script)],
        args.output,
        args.output / "model",
        progress_info / "model.log",
        markers=[str(script)],
        env=env,
    )


def wait_model(base, process, timeout):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Model server exited with {process.returncode}")
        try:
            request_json(base, "/reset", {})
            return
        except Exception as error:
            last_error = error
            time.sleep(1)
    raise TimeoutError(f"Model readiness/reset failed: {last_error}")


def model_smoke(base):
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (224, 224), (100, 100, 100)).save(buffer, format="PNG")
    result = request_json(
        base,
        "/predict",
        {
            "image": base64.b64encode(buffer.getvalue()).decode(),
            "proprio": [0, 0, 0, 0],
            "instr": "Rotate slightly to the left",
        },
    )
    if (
        not isinstance(result, dict)
        or not isinstance(result.get("action"), list)
        or not result["action"]
    ):
        raise RuntimeError("Model smoke returned no valid action chunk")


def worker_command(args, base, binary):
    return [
        sys.executable,
        "-I",
        "-B",
        "-u",
        "-m",
        "evaluation.closed_loop_runner",
        "--json_folder",
        str(args.task_dir),
        "--classification",
        str(args.classification),
        "--output",
        str(args.output),
        "--server-url",
        base,
        "--max_steps",
        "100",
        "--surround-max-steps",
        "400",
        "--save-frames",
        "--ue_extra_args",
        f"-RenderOffScreen -graphicsadapter={args.graphics_adapter}",
    ]


def run_batch(args, base, binary, progress_info, value):
    env = runtime.environment(args.output, args.ue_user)
    env["UNREALZOO_ENV_BIN"] = str(binary)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    process = runtime.launch(
        worker_command(args, base, binary),
        args.output,
        args.output / "tmp",
        progress_info / "batch.log",
        user=args.ue_user,
        pairs=[("-m", "evaluation.closed_loop_runner"), ("--output", str(args.output))],
        env=env,
    )
    value["worker"] = process.identity
    atomic_json(args.output / "progress_info/current.json", value)
    try:
        return process.wait(timeout=args.eval_timeout)
    finally:
        stop_process(process)


def execute(args):
    records = task_manifest(args)
    expected = [r["filename"] for r in records]
    base = api_base(args.server_url or f"http://127.0.0.1:{args.model_port}")
    plan = dict(
        model=args.model_name,
        checkpoint=str(args.checkpoint) if args.checkpoint else None,
        server_url=base,
        output=str(args.output),
        task_count=len(records),
        tasks=records,
        protocol="radial30_net342_surround400_others100",
        fresh_trial=True,
        smoke=args.smoke,
        resources=dict(
            ue_package=str(args.ue_package) if args.ue_package else None,
            ue_user=args.ue_user,
            ue_port=args.ue_port,
            graphics_adapter=args.graphics_adapter,
            python=sys.executable,
            model_gpu=args.model_gpu,
            model_port=args.model_port,
            model_repo=str(args.model_repo) if args.model_repo else None,
            model_config=str(args.config) if args.config else None,
            base_vlm=str(args.base_vlm) if args.base_vlm else None,
        ),
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if args.score_only:
        saved = args.output / "progress_info/run_manifest.json"
        if saved.exists() and json.loads(saved.read_text())["tasks"] != records:
            raise ValueError("Reference tasks changed since the original run")
        with runtime.output_context(args.output):
            score_run(args.output, args.task_dir, args.classification, expected)
        return 0
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(
            "Output directory is not empty; use a new directory for a fresh single run"
        )
    if not args.smoke:
        if args.ue_package is None:
            raise ValueError(
                "Full simulation requires --ue-package or evaluation.ue_package in --settings"
            )
        if args.ue_user is None:
            raise ValueError("Root must choose --ue-user or evaluation.ue_user in --settings")
    with runtime.workflow_lock(args.output, "evaluation", create=True) as value:
        runtime.make_output(args.output, args.ue_user if not args.smoke else None)
        with runtime.output_context(args.output):
            return execute_run(args, records, base, plan, value)


def execute_run(args, records, base, plan, value):
    expected = [record["filename"] for record in records]
    progress_info = args.output / "progress_info"
    atomic_json(progress_info / "run_manifest.json", plan)
    process = None
    binary = None
    status = dict(state="running", all_completed=False, task_count=len(records), smoke=args.smoke)
    value.update(state="running", started_after=runtime.start_ticks())
    atomic_json(progress_info / "current.json", value)
    atomic_json(progress_info / "status.json", status)
    try:
        if not args.smoke:
            package = runtime.installed_package(args.output, args.ue_user)
            value["package"] = package
            selected = Path(package["modules"]["evaluation.closed_loop_runner"]).parent
            if args.task_dir == ROOT / "test_jsons":
                args.task_dir = selected / "test_jsons"
            if args.classification == ROOT / "classified_instr.json":
                args.classification = selected / "classified_instr.json"
            if task_manifest(args) != records:
                raise RuntimeError("Controller and worker task data differ")
        if args.checkpoint:
            process = start_model(args, progress_info)
            if isinstance(process, runtime.OwnedProcess):
                value["model"] = process.identity
                atomic_json(progress_info / "current.json", value)
        wait_model(base, process, args.start_timeout)
        if args.smoke:
            model_smoke(base)
        else:
            bind_check(args.ue_port)
            binary = prepare_ue(args.ue_package, args.output / "ue", args.ue_port)
            runtime.make_output(args.output, args.ue_user)
            value["binary"] = str(binary)
            atomic_json(progress_info / "current.json", value)
            result = run_batch(args, base, binary, progress_info, value)
            if result != 0:
                raise RuntimeError(
                    f"Closed-loop batch failed with exit code {result}; see progress_info/batch.log"
                )
            if task_manifest(args) != records:
                raise RuntimeError("Reference tasks changed during evaluation")
            metrics = score_run(args.output, args.task_dir, args.classification, expected)
            status["overall"] = metrics["overall"]
        status.update(state="completed", all_completed=not args.smoke, smoke_passed=args.smoke)
    except BaseException as error:
        status.update(state="failed", error=f"{type(error).__name__}: {error}")
    finally:
        try:
            if process is not None:
                stop_process(process)
        except Exception as error:
            status.update(state="failed", all_completed=False, cleanup_error=str(error))
        try:
            if binary is not None:
                value["ue_processes"] = runtime.binary_processes(
                    binary, args.output, value["started_after"]
                )
                cleanup_ue(binary, args.output, value["started_after"])
        except Exception as error:
            status.update(state="failed", all_completed=False, cleanup_error=str(error))
        value["state"] = status["state"]
        atomic_json(progress_info / "current.json", value)
        atomic_json(progress_info / "status.json", status)
    return 0 if status["state"] == "completed" else 1


def main(argv=None):
    try:
        return execute(parse_args(argv))
    except Exception as error:
        print(f"Evaluation failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


def cli(argv=None):
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Interrupted by signal {signum}")

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        return main(argv)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(cli())
