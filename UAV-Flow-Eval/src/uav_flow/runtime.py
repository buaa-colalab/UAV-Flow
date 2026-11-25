# SPDX-License-Identifier: Apache-2.0
"""Installed-process execution, output containment and manifest-bound ownership."""

import contextlib
import fcntl
import json
import os
import pwd
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def environment(output: Path, user: str | None = None) -> dict[str, str]:
    root = output.resolve()
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        MPLBACKEND="Agg",
        TMPDIR=str(root / "tmp"),
        TMP=str(root / "tmp"),
        TEMP=str(root / "tmp"),
        XDG_CACHE_HOME=str(root / "cache"),
        XDG_CONFIG_HOME=str(root / "cache/config"),
        MPLCONFIGDIR=str(root / "cache/matplotlib"),
        HF_HOME=str(root / "cache/huggingface"),
        TORCH_HOME=str(root / "cache/torch"),
        TRITON_CACHE_DIR=str(root / "cache/triton"),
        CUDA_CACHE_PATH=str(root / "cache/cuda"),
    )
    if user is not None:
        account = pwd.getpwnam(user)
        env.update(HOME=account.pw_dir, USER=account.pw_name, LOGNAME=account.pw_name)
    return env


def user_options(user: str | None) -> dict[str, Any]:
    if user is None:
        return {}
    account = pwd.getpwnam(user)
    if os.geteuid() != 0:
        if account.pw_uid != os.geteuid():
            raise PermissionError("A non-root launcher must use its own UE account")
        return {}
    return dict(
        user=account.pw_uid,
        group=account.pw_gid,
        extra_groups=os.getgrouplist(account.pw_name, account.pw_gid),
    )


def make_output(output: Path, user: str | None = None) -> None:
    for name in ["progress_info", "tmp", "cache", "logs"]:
        (output / name).mkdir(parents=True, exist_ok=True)
    if os.geteuid() == 0 and user is not None:
        account = pwd.getpwnam(user)
        for path in [output, *output.rglob("*")]:
            if not path.is_symlink():
                os.chown(path, account.pw_uid, account.pw_gid)


@contextlib.contextmanager
def output_context(output: Path, change_cwd: bool = False) -> Iterator[None]:
    output = output.resolve()
    make_output(output)
    previous = dict(os.environ)
    old_cwd, old_temp = Path.cwd(), tempfile.tempdir
    os.environ.update(environment(output))
    tempfile.tempdir = None
    try:
        if change_cwd:
            os.chdir(output / "tmp")
        yield
    finally:
        os.chdir(old_cwd)
        os.environ.clear()
        os.environ.update(previous)
        tempfile.tempdir = old_temp


def installed_package(output: Path, user: str | None = None) -> dict[str, Any]:
    """Probe the current interpreter/account with source-path injection disabled."""
    make_output(output, user)
    script = """
import importlib, importlib.metadata, json
from pathlib import Path
dist=importlib.metadata.distribution('uav-flow-eval')
modules={}
for name, relative in [('env_server','env_server.py'),('evaluation.closed_loop_runner','evaluation/closed_loop_runner.py'),('training_tasks','training_tasks/__init__.py')]:
 module=importlib.import_module(name)
 actual=Path(module.__file__).resolve();expected=Path(dist.locate_file(relative)).resolve()
 assert actual==expected,(name,str(actual),str(expected))
 modules[name]=str(actual)
print(json.dumps({'version':dist.version,'modules':modules}))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        cwd=output / "tmp",
        env=environment(output, user),
        capture_output=True,
        text=True,
        timeout=90,
        **user_options(user),
    )
    if result.returncode:
        raise RuntimeError(
            f"Install uav-flow-eval in {sys.executable} for the selected account:\n{result.stderr}"
        )
    return dict(json.loads(result.stdout.splitlines()[-1]))


def start_ticks() -> int:
    return int(time.clock_gettime(time.CLOCK_BOOTTIME) * os.sysconf("SC_CLK_TCK"))


def process_info(pid: int) -> dict[str, Any] | None:
    root = Path("/proc") / str(pid)
    try:
        fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return dict(
            pid=pid,
            starttime=int(fields[19]),
            pgid=os.getpgid(pid),
            argv=(root / "cmdline").read_bytes().decode().rstrip("\0").split("\0"),
            cwd=str((root / "cwd").resolve(strict=True)),
        )
    except (OSError, ValueError, UnicodeError):
        return None


def matches(record: dict[str, Any]) -> bool:
    current = process_info(int(record["pid"]))
    if current is None or current["starttime"] != record["starttime"]:
        return False
    if not Path(current["cwd"]).is_relative_to(Path(record["output"])):
        return False
    if record.get("group") and (
        current["pgid"] != record["pgid"] or record["pgid"] != record["pid"]
    ):
        return False
    argv = current["argv"]
    if record.get("binary") and (not argv or argv[0] != record["binary"]):
        return False
    if any(marker not in argv for marker in record.get("markers", [])):
        return False
    for flag, value in record.get("pairs", []):
        if not any(a == flag and b == value for a, b in zip(argv, argv[1:])):
            return False
    return True


@dataclass
class OwnedProcess:
    process: subprocess.Popen[Any]
    identity: dict[str, Any]

    def poll(self) -> int | None:
        return self.process.poll()

    def wait(self, timeout: float | None = None) -> int:
        return self.process.wait(timeout=timeout)


def launch(
    command: Sequence[str],
    output: Path,
    cwd: Path,
    log: Path,
    user: str | None = None,
    pairs: Sequence[tuple[str, str]] = (),
    markers: Sequence[str] = (),
    env: dict[str, str] | None = None,
) -> OwnedProcess:
    cwd.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    make_output(output, user)
    with log.open("w") as stream:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=env or environment(output, user),
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=stream,
            start_new_session=True,
            **user_options(user),
        )
    info = process_info(process.pid)
    if info is None:
        process.wait(timeout=5)
        raise RuntimeError(f"Process exited before registration; see {log}")
    identity = dict(
        pid=process.pid,
        starttime=info["starttime"],
        pgid=process.pid,
        output=str(output.resolve()),
        group=True,
        pairs=list(pairs),
        markers=list(markers),
    )
    return OwnedProcess(process, identity)


def stop_record(record: dict[str, Any]) -> None:
    for sig, timeout in [(signal.SIGTERM, 10), (signal.SIGKILL, 3)]:
        if not matches(record):
            return
        try:
            if record.get("group"):
                os.killpg(int(record["pgid"]), sig)
            else:
                os.kill(int(record["pid"]), sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + timeout
        while matches(record) and time.monotonic() < deadline:
            time.sleep(0.05)
    if matches(record):
        raise RuntimeError(f"Owned process did not exit: {record['pid']}")


def stop_process(process: OwnedProcess | None) -> None:
    if process is None:
        return
    stop_record(process.identity)
    process.process.wait(timeout=5)


def binary_processes(binary: Path, output: Path, since: int) -> list[dict[str, Any]]:
    original = binary
    binary, output = binary.resolve(), output.resolve()
    if not binary.is_relative_to(output) or original.is_symlink():
        raise ValueError("UE binary must be private to the selected output")
    records = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
            if not argv or argv[0] != str(binary).encode():
                continue
        except OSError:
            continue
        info = process_info(int(entry.name))
        if info and info["starttime"] >= since and info["argv"] and info["argv"][0] == str(binary):
            records.append(
                dict(
                    pid=info["pid"],
                    starttime=info["starttime"],
                    output=str(output),
                    binary=str(binary),
                    pgid=info["pgid"],
                    group=info["pgid"] == info["pid"],
                )
            )
    return records


def manifest(output: Path, workflow: str) -> dict[str, Any]:
    path = output / "progress_info/current.json"
    if not path.is_file():
        raise RuntimeError(f"Runtime manifest not found: {path}; old deployments are not adopted")
    value = json.loads(path.read_text())
    if (
        value.get("schema") != SCHEMA_VERSION
        or value.get("workflow") != workflow
        or value.get("output") != str(output.resolve())
    ):
        raise RuntimeError(
            "Manifest does not belong to this output/workflow; old deployments are not adopted"
        )
    return dict(value)


@contextlib.contextmanager
def workflow_lock(output: Path, workflow: str, create: bool = False) -> Iterator[dict[str, Any]]:
    output = output.resolve()
    path = output / "progress_info/current.json"
    if create:
        if output.exists() and any(output.iterdir()) and not path.is_file():
            raise RuntimeError("Output is not a managed run; old deployments are not adopted")
        (output / "progress_info").mkdir(parents=True, exist_ok=True)
    elif not path.is_file():
        raise RuntimeError(f"Runtime manifest not found: {path}; old deployments are not adopted")
    with (output / "progress_info/.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        if not path.exists():
            atomic_json(
                path,
                dict(
                    schema=SCHEMA_VERSION,
                    workflow=workflow,
                    output=str(output),
                    state="new",
                    slots=[],
                    started_after=start_ticks(),
                ),
            )
        yield manifest(output, workflow)
