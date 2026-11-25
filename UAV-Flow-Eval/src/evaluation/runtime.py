# SPDX-License-Identifier: Apache-2.0
"""Private UE files and cleanup; Python code is loaded from the installed package."""

import configparser
import subprocess
from pathlib import Path

from uav_flow import runtime


def prepare_ue(package, destination, port):
    package, destination = Path(package), Path(destination)
    source = package / "Collection/Binaries/Linux/Collection"
    if not source.is_file():
        raise FileNotFoundError(f"UE executable missing: {source}")
    binary = destination / "Collection/Binaries/Linux/Collection"
    binary.parent.mkdir(parents=True)
    (destination / "Engine").mkdir()
    subprocess.run(["cp", "--reflink=auto", str(source), str(binary)], check=True)
    binary.chmod(0o755)
    for parent in [
        package,
        package / "Collection",
        package / "Collection/Binaries",
        package / "Collection/Binaries/Linux",
        package / "Engine",
    ]:
        local = destination / parent.relative_to(package)
        local.mkdir(parents=True, exist_ok=True)
        for item in parent.iterdir():
            target = local / item.name
            if (
                item.name not in ("Saved", "unrealcv.ini")
                and not target.exists()
                and not target.is_symlink()
            ):
                target.symlink_to(item, target_is_directory=item.is_dir())
    for name in ["Collection/Saved", "Engine/Saved"]:
        (destination / name).mkdir(parents=True, exist_ok=True)
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(source.parent / "unrealcv.ini")
    if not config.has_section("UnrealCV.Core"):
        config.add_section("UnrealCV.Core")
    config["UnrealCV.Core"].update(Port=str(port), Width="256", Height="256")
    with (binary.parent / "unrealcv.ini").open("w") as handle:
        config.write(handle, space_around_delimiters=False)
    return binary


def cleanup_ue(binary, output, since):
    if binary is not None:
        for record in runtime.binary_processes(Path(binary), Path(output), since):
            runtime.stop_record(record)
