# SPDX-License-Identifier: Apache-2.0
import configparser
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from evaluation import runtime


class RuntimeIsolationTests(unittest.TestCase):
    def test_private_ue_keeps_shared_binary_ini_and_saved_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            package = root / "package"
            binary = package / "Collection/Binaries/Linux/Collection"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"fake UE binary")
            ini = binary.parent / "unrealcv.ini"
            ini.write_text("[UnrealCV.Core]\nPort=9000\n")
            (package / "Collection/Saved").mkdir()
            (package / "Collection/Content").mkdir()
            (package / "Engine").mkdir()
            private = runtime.prepare_ue(package, root / "private", 9488)
            self.assertEqual(binary.read_bytes(), b"fake UE binary")
            self.assertEqual(ini.read_text(), "[UnrealCV.Core]\nPort=9000\n")
            config = configparser.ConfigParser()
            config.read(private.parent / "unrealcv.ini")
            self.assertEqual(config.getint("UnrealCV.Core", "Port"), 9488)
            self.assertFalse(private.is_symlink())
            self.assertFalse((root / "private/Collection/Saved").is_symlink())
            self.assertTrue((root / "private/Collection/Content").is_symlink())

    def test_process_matching_requires_exact_private_binary_and_start_time(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "output"
            binary = output / "ue/Collection"
            binary.parent.mkdir(parents=True)
            binary.touch()
            proc = root / "proc"
            proc.mkdir()
            commands = {
                123: [str(binary)],
                124: ["/another/Collection"],
                125: ["/bin/sh", "-c", str(binary)],
                126: [str(binary) + "_old"],
                127: [str(binary)],
            }
            for pid, args in commands.items():
                entry = proc / str(pid)
                entry.mkdir()
                (entry / "cmdline").write_bytes(b"\0".join(a.encode() for a in args) + b"\0")

            def info(pid):
                return dict(
                    pid=pid,
                    starttime=10 if pid == 127 else 30,
                    pgid=pid,
                    argv=commands[pid],
                    cwd=str(output),
                )

            with (
                patch.object(
                    runtime.runtime,
                    "Path",
                    side_effect=lambda value: proc if value == "/proc" else Path(value),
                ),
                patch.object(runtime.runtime, "process_info", side_effect=info),
            ):
                records = runtime.runtime.binary_processes(binary, output, 20)
            self.assertEqual([r["pid"] for r in records], [123])

    def test_cleanup_only_signals_recorded_private_processes(self):
        record = {"pid": 123, "starttime": 50, "output": "/private"}
        with (
            patch.object(runtime.runtime, "binary_processes", return_value=[record]),
            patch.object(runtime.runtime, "stop_record") as stop,
        ):
            runtime.cleanup_ue("/private/Collection", "/private", 0)
            stop.assert_called_once_with(record)


if __name__ == "__main__":
    unittest.main()
