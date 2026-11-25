#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Manage isolated training UE instances and installed environment-server modules."""

import argparse
import configparser
import json
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

from uav_flow import runtime
from uav_flow.settings import adapter_indices, default_ue_user, parse_settings

HTTP = build_opener(ProxyHandler({}))


def prepare_slot(package, destination, port):
    """Share assets, but give every process its own executable, ini and Saved."""
    source_binary = package / "Collection/Binaries/Linux/Collection"
    binary = destination / "Collection/Binaries/Linux/Collection"
    binary.parent.mkdir(parents=True, exist_ok=True)
    (destination / "Engine").mkdir(parents=True, exist_ok=True)
    # A symlink to the executable would resolve /proc/self/exe back to the
    # shared source package and defeat unrealcv.ini isolation.
    if not binary.exists() or binary.stat().st_size != source_binary.stat().st_size:
        subprocess.run(["cp", "--reflink=auto", str(source_binary), str(binary)], check=True)
    binary.chmod(0o755)
    for parent in (
        package,
        package / "Collection",
        package / "Collection/Binaries",
        package / "Collection/Binaries/Linux",
        package / "Engine",
    ):
        local_parent = destination / parent.relative_to(package)
        local_parent.mkdir(parents=True, exist_ok=True)
        for source in parent.iterdir():
            target = local_parent / source.name
            if source.name in ("Saved", "unrealcv.ini"):
                continue
            if not target.exists() and not target.is_symlink():
                target.symlink_to(source, target_is_directory=source.is_dir())
    for name in ("Collection/Saved", "Engine/Saved"):
        (destination / name).mkdir(parents=True, exist_ok=True)
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(package / "Collection/Binaries/Linux/unrealcv.ini")
    if not config.has_section("UnrealCV.Core"):
        config.add_section("UnrealCV.Core")
    config["UnrealCV.Core"].update(Port=str(port), Width="256", Height="256")
    with (binary.parent / "unrealcv.ini").open("w") as output:
        config.write(output, space_around_delimiters=False)
    return binary


def prepare(args):
    slots = []
    for index, gpu in enumerate(args.gpus):
        port = args.base_port + index
        binary = prepare_slot(
            args.package, args.output / "ue" / f"slot_{port}", args.ue_base_port + index
        )
        slots.append(
            dict(port=port, ue_port=args.ue_base_port + index, gpu=gpu, binary=str(binary))
        )
    runtime.make_output(args.output, args.user)
    return slots


def preflight(args):
    if not (args.package / "Collection/Binaries/Linux/Collection").is_file():
        raise RuntimeError(f"Unreal executable not found in {args.package}")
    package = runtime.installed_package(args.output, args.user)
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError("NVIDIA GPU unavailable: " + result.stderr)
    available = {int(line.split(",")[0]) for line in result.stdout.splitlines() if line.strip()}
    if not set(args.gpus).issubset(available):
        raise RuntimeError(f"GPU indices {args.gpus} not available: {sorted(available)}")
    for port in [
        *(args.base_port + i for i in range(len(args.gpus))),
        *(args.ue_base_port + i for i in range(len(args.gpus))),
    ]:
        with socket.socket() as sock:
            sock.bind(("0.0.0.0", port))
    return package


def healthy(port):
    try:
        with HTTP.open(f"http://127.0.0.1:{port}/health", timeout=3) as response:
            payload = json.load(response)
            return payload.get("status") == "healthy" and "recovery_count" in payload
    except (OSError, ValueError):
        return False


def owned_records(output, value):
    records = []
    for slot in value["slots"]:
        record = slot.get("process")
        if record and record.get("output") == str(output.resolve()) and runtime.matches(record):
            records.append(record)
        records.extend(
            runtime.binary_processes(Path(slot["binary"]), output, value["started_after"])
        )
    return records


def stop(output, value):
    # Only schema-validated records from this output are considered, never module-name scans.
    for record in owned_records(output, value):
        runtime.stop_record(record)
    value["state"] = "stopped"
    runtime.atomic_json(output / "progress_info/current.json", value)


def server_command(args, slot, instance):
    return [
        sys.executable,
        "-I",
        "-B",
        "-u",
        "-m",
        "env_server",
        "--output",
        str(instance),
        "-p",
        str(slot["port"]),
        "--env_bin",
        slot["binary"],
        "--offscreen",
        "--gpu_id",
        str(slot["gpu"]),
        "--host",
        args.host,
    ]


def start(args, value):
    package = preflight(args)
    slots = prepare(args)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    value.update(
        state="starting",
        slots=slots,
        run_id=run_id,
        package=package,
        started_after=runtime.start_ticks(),
    )
    path = args.output / "progress_info/current.json"
    runtime.atomic_json(path, value)
    started = []

    def wait_ready(slot, process, deadline):
        while not healthy(slot["port"]):
            if process.poll() is not None:
                raise RuntimeError(f"HTTP {slot['port']} exited; see {slot['log']}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"HTTP {slot['port']} startup exceeded {args.timeout}s")
            time.sleep(3)
        slot["ue_processes"] = runtime.binary_processes(
            Path(slot["binary"]), args.output, value["started_after"]
        )
        runtime.atomic_json(path, value)

    try:
        for slot in slots:
            instance = args.output / "runs" / run_id / f"port_{slot['port']}"
            log = instance / "logs/http.log"
            process = runtime.launch(
                server_command(args, slot, instance),
                args.output,
                instance / "tmp",
                log,
                user=args.user,
                pairs=[
                    ("-m", "env_server"),
                    ("-p", str(slot["port"])),
                    ("--output", str(instance)),
                ],
            )
            slot.update(process=process.identity, output=str(instance), log=str(log))
            runtime.atomic_json(path, value)
            deadline = time.monotonic() + args.timeout
            if args.parallel:
                started.append((slot, process, deadline))
            else:
                wait_ready(slot, process, deadline)
        for slot, process, deadline in started:
            wait_ready(slot, process, deadline)
        value["state"] = "running"
        runtime.atomic_json(path, value)
    except BaseException:
        stop(args.output, value)
        value["state"] = "failed"
        runtime.atomic_json(path, value)
        raise
    print(f"All {len(slots)} environments ready. Output: {args.output}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("start", "prepare", "status", "stop", "check"),
        nargs="?",
        default="start",
    )
    parser.add_argument("--package", type=Path, help="External UE package")
    parser.add_argument(
        "--output", type=Path, help="Single output root; CLI or training.output in settings"
    )
    parser.add_argument("--user", default=default_ue_user())
    parser.add_argument("--gpus", type=adapter_indices, help="Explicit Vulkan adapter indices")
    parser.add_argument("--base-port", type=int, default=5008)
    parser.add_argument("--ue-base-port", type=int, default=9400)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without writes, processes or network access",
    )
    args = parse_settings(parser, "training", argv)
    if args.output is None:
        parser.error("Choose --output or training.output in --settings")
    if args.action in ("start", "prepare", "check") and args.gpus is None:
        parser.error("Choose Vulkan adapters with --gpus or training.gpus")
    args.gpus = args.gpus or []
    for first in [args.base_port, args.ue_base_port]:
        if first < 1 or first + len(args.gpus) - 1 > 9999:
            parser.error("Ports must be between 1 and 9999")
    if set(range(args.base_port, args.base_port + len(args.gpus))) & set(
        range(args.ue_base_port, args.ue_base_port + len(args.gpus))
    ):
        parser.error("HTTP and UE port ranges must not overlap")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not args.dry_run and args.action in ("start", "prepare", "check"):
        if args.package is None or args.user is None:
            parser.error("Choose the external --package and UE --user")
    args.output = args.output.resolve()
    if args.package is not None:
        args.package = args.package.resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    action=args.action,
                    output=str(args.output),
                    package=str(args.package) if args.package else None,
                    python=sys.executable,
                    user=args.user,
                    host=args.host,
                    parallel=args.parallel,
                    slots=[
                        dict(
                            graphics_adapter=gpu,
                            http_port=args.base_port + i,
                            ue_port=args.ue_base_port + i,
                        )
                        for i, gpu in enumerate(args.gpus)
                    ],
                ),
                indent=2,
            )
        )
        return 0
    with runtime.workflow_lock(
        args.output, "training", create=args.action in ("start", "prepare", "check")
    ) as value:
        if args.action in ("start", "prepare", "check") and owned_records(args.output, value):
            raise RuntimeError("This output already owns running environments; stop them first")
        if args.action == "start":
            start(args, value)
        elif args.action == "prepare":
            preflight(args)
            value.update(slots=prepare(args), state="prepared")
            runtime.atomic_json(args.output / "progress_info/current.json", value)
        elif args.action == "check":
            preflight(args)
        elif args.action == "stop":
            stop(args.output, value)
        else:
            for slot in value["slots"]:
                active = bool(slot.get("process") and runtime.matches(slot["process"]))
                print(
                    f"HTTP {slot['port']}: {'healthy' if active and healthy(slot['port']) else 'unavailable'}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
