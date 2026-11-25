#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify camera tracking on real HTTP environments and export before/after frames."""

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.smoke_rl_http import REPO, frame, measured_camera, request


def verify(port, task_id, tasks, output):
    folder = output / f"port_{port}"
    folder.mkdir()
    record = dict(port=port, task_id=task_id, passed=False)
    try:
        task = next(case for case in tasks if case["task_id"] == task_id)
        pose = task.get("init_pose") or task["initial_pos"]
        reset, _ = request(port, "reset", {"task_ids": [task_id]})
        first = frame(reset, folder / "reset.png")
        initial = measured_camera(port, pose[:3], pose[4])
        turned, _ = request(port, "step", {"actions": [[[0, 0, 0, math.pi / 4]]]})
        rotated = frame(turned, folder / "turn_45_degrees.png")
        rotation = measured_camera(port, pose[:3], initial["rotation"][1] + 45)
        translation, _ = request(port, "step", {"actions": [[[100, 0, 0, math.pi / 4]]]})
        frame(translation, folder / "forward_100cm.png")
        theta = math.radians(pose[4])
        position = [pose[0] + 100 * math.cos(theta), pose[1] + 100 * math.sin(theta), pose[2]]
        moved = measured_camera(port, position, initial["rotation"][1] + 45)
        difference = float(np.abs(rotated.astype(float) - first.astype(float)).mean())
        assert difference > 5, f"45-degree rotation did not change the scene: {difference}"
        record.update(
            passed=True,
            initial_camera=initial,
            rotated_camera=rotation,
            translated_camera=moved,
            rotation_image_difference=difference,
        )
        # Return this instance to the task's initial pose after verification.
        request(port, "reset", {"task_ids": [task_id]})
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ports",
        required=True,
        help="Explicit HTTP ports of disposable environments; this test resets/steps them",
    )
    parser.add_argument("--tasks", default="35,390,799,671")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--baseline-dir", type=Path, help="Optional previous images named port_<port>_0.png"
    )
    parser.add_argument(
        "--training-output",
        type=Path,
        help="Optional managed training output, read only for context",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True)
    tasks = json.loads((REPO / "training_tasks/tasks.json").read_text())
    ports = [int(value) for value in args.ports.split(",")]
    tasks_to_test = [int(value) for value in args.tasks.split(",")]
    if len(ports) != len(tasks_to_test):
        parser.error("--ports and --tasks must have the same length")
    pairs = list(zip(ports, tasks_to_test))
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(verify, port, task, tasks, output) for port, task in pairs]
        records = [future.result() for future in futures]
    comparison = Image.new("RGB", (len(pairs) * 300, 3 * 300), (24, 24, 24))
    draw = ImageDraw.Draw(comparison)
    font = ImageFont.load_default()
    for column, (port, task) in enumerate(pairs):
        images = [
            (
                "Previous image",
                args.baseline_dir / f"port_{port}_0.png" if args.baseline_dir else None,
            ),
            ("Reset", output / f"port_{port}/reset.png"),
            ("Yaw +45 degrees", output / f"port_{port}/turn_45_degrees.png"),
        ]
        for row, (label, path) in enumerate(images):
            draw.text(
                (column * 300 + 8, row * 300 + 4),
                f"Port {port} | Task {task}",
                fill="white",
                font=font,
            )
            draw.text((column * 300 + 8, row * 300 + 22), label, fill=(200, 200, 200), font=font)
            if path is not None and path.exists():
                with Image.open(path) as image:
                    comparison.paste(
                        image.convert("RGB").resize((256, 256)), (column * 300 + 22, row * 300 + 42)
                    )
    comparison.save(output / "before_after.png")
    report = dict(passed=all(record["passed"] for record in records), ports=records)
    manifest = args.training_output / "progress_info/current.json" if args.training_output else None
    if manifest is not None and manifest.exists():
        report["main_runtime_snapshot"] = json.loads(manifest.read_text())
        report["main_runtime_snapshot_note"] = (
            "Informational; tested ports are listed above and may be an isolated runtime."
        )
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"Artifacts: {output}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
