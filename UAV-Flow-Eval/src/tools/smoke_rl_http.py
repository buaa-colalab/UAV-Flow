#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise real RL HTTP services; save rendered frames and a JSON report."""

import argparse
import base64
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
from check.check_trajectory_approach import check_approach_trajectory
from check.check_trajectory_land import check_land_trajectory
from check.check_trajectory_pass import check_pass_trajectory
from check.check_trajectory_rotate import check_rotate_trajectory


def request(port, route, payload=None):
    opener = build_opener(ProxyHandler({}))
    data = None if payload is None else json.dumps(payload).encode()
    req = Request(
        f"http://127.0.0.1:{port}/{route}", data=data, headers={"Content-Type": "application/json"}
    )
    before = time.monotonic()
    with opener.open(req, timeout=90) as response:
        result = json.load(response)
    if result.get("status") == "error":
        raise RuntimeError(result)
    return result, time.monotonic() - before


def frame(result, destination, count=1):
    assert len(result["images"]) == count, result.keys()
    raw = base64.b64decode(result["images"][0][0], validate=True)
    image = Image.open(BytesIO(raw)).convert("RGB")
    assert image.size == (256, 256), image.size
    pixels = np.asarray(image)
    assert float(pixels.std()) > 3, "Rendered image is blank or nearly uniform"
    destination.write_bytes(raw)
    return pixels


def measured_camera(port, location, yaw=None):
    observation, _ = request(port, "observation")
    camera = observation["camera"]
    agent = observation["agent"]
    error = float(np.linalg.norm(np.asarray(camera["location"]) - location))
    assert error < 1.0, f"Camera position error {error:.3f} cm: {camera}"
    assert np.linalg.norm(np.asarray(camera["location"]) - agent["location"]) < 1.0
    # The Blueprint actor interpolates yaw; camera follows commanded RL state directly.
    if yaw is not None:
        error = (camera["rotation"][1] - yaw + 180) % 360 - 180
        assert abs(error) < 1.0, f"Camera yaw differs from commanded yaw: {camera}, expected {yaw}"
    return camera


def test_port(port, index, tasks, output):
    folder = output / f"port_{port}"
    folder.mkdir()
    record = dict(port=port, passed=False, steps=0)
    try:
        health, _ = request(port, "health")
        assert health["status"] == "healthy"
        task = next(t for t in tasks if t["task_id"] == 512)
        assert task["prompt"] == "Rotate 75 degrees to the left"
        reset, record["reset_seconds"] = request(port, "reset", {"task_ids": [512]})
        assert reset["task_ids"] == [512] and reset["batch_size"] == 1
        initial = frame(reset, folder / "reset.png")
        pose = task.get("init_pose") or task["initial_pos"]
        camera = measured_camera(port, pose[:3], pose[4])
        record["initial_camera"] = camera
        initial_camera_yaw = camera["rotation"][1]
        # Different commanded rotations expose shared temporary-file races
        # when the same task is evaluated concurrently on multiple ports.
        target_degrees = (-75, -15, -45, -105)[index % 4]
        points = [list(pose)]
        times = []
        previous = initial
        changed = 0
        for step in range(1, 7):
            degrees = target_degrees * step / 6
            response, seconds = request(
                port, "step", {"actions": [[[0, 0, 0, math.radians(degrees)]]]}
            )
            assert len(response["dones"]) == 1 and isinstance(response["dones"][0], bool)
            pixels = frame(response, folder / f"step_{step:02d}.png")
            camera = measured_camera(port, pose[:3], initial_camera_yaw + degrees)
            if float(np.abs(pixels.astype(float) - previous.astype(float)).mean()) > 3.0:
                changed += 1
            previous = pixels
            points.append([*pose[:3], 0, pose[4] + degrees, 0])
            times.append(seconds)
            record["steps"] += 1
        assert changed >= 3, f"Only {changed} rendered frames changed after rotating"
        record.update(changed_frames=changed, step_mean_seconds=sum(times) / len(times))
        trajectory = folder / "commanded_trajectory.json"
        trajectory.write_text(json.dumps([{"state": [p[:3], p[3:]]} for p in points]))
        expected = check_rotate_trajectory(
            str(trajectory),
            combined_json_path=str(REPO / "training_tasks/tasks.json"),
            task_id=512,
        )
        record["expected_reward"] = float(expected["reward"])
        for iteration in range(3):
            rewards, _ = request(port, "rewards", {"trajectories": [points]})
            for key in ("rewards", "reward_1", "reward_2", "reward_3"):
                assert len(rewards[key]) == 1 and math.isfinite(float(rewards[key][0]))
            actual = rewards["rewards"][0]
            assert abs(actual - round(record["expected_reward"], 2)) < 0.011, (
                actual,
                record["expected_reward"],
            )
        record["reward"] = actual
        moved, _ = request(port, "step", {"actions": [[[500, 0, 0, math.radians(target_degrees)]]]})
        frame(moved, folder / "translation_500cm.png")
        theta = math.radians(pose[4])
        expected_location = [
            pose[0] + 500 * math.cos(theta),
            pose[1] + 500 * math.sin(theta),
            pose[2],
        ]
        record["translated_camera"] = measured_camera(
            port, expected_location, initial_camera_yaw + target_degrees
        )
        record["camera_pose_tracking_passed"] = True
        batch, _ = request(port, "reset", {"task_ids": [512, 513]})
        assert batch["batch_size"] == 2 and batch["task_ids"] == [512, 513]
        frame(batch, folder / "batch_reset.png", count=2)
        batch_step, _ = request(port, "step", {"actions": [[[0, 0, 0, 0]], [[0, 0, 0, 0]]]})
        assert len(batch_step["dones"]) == 2
        frame(batch_step, folder / "batch_step.png", count=2)
        record["batch_size_two_passed"] = True
        # Exercise all four categories present in the 1024-task RL dataset.
        # Select distinct locations; tasks 0/256/768 start within 60 cm of each other.
        category_ids = [35, 390, 671, 799]
        mixed, _ = request(port, "reset", {"task_ids": category_ids})
        assert mixed["task_ids"] == category_ids
        frame(mixed, folder / "four_categories_reset.png", count=4)
        category_images = [
            np.asarray(Image.open(BytesIO(base64.b64decode(images[0]))).convert("RGB")).astype(
                float
            )
            for images in mixed["images"]
        ]
        scene_differences = [
            float(np.abs(image - category_images[0]).mean()) for image in category_images[1:]
        ]
        assert min(scene_differences) > 5.0, (
            f"Different task locations show the same scene: {scene_differences}"
        )
        record["different_task_image_differences"] = scene_differences
        mixed_step, _ = request(port, "step", {"actions": [[[0, 0, 0, 0]]] * 4})
        assert len(mixed_step["dones"]) == 4
        frame(mixed_step, folder / "four_categories_step.png", count=4)
        trajectories = []
        expected_rewards = []
        for task_id, checker in zip(
            category_ids,
            (
                check_pass_trajectory,
                check_approach_trajectory,
                check_rotate_trajectory,
                check_land_trajectory,
            ),
        ):
            case = next(t for t in tasks if t["task_id"] == task_id)
            initial_pose = case.get("init_pose") or case["initial_pos"]
            trajectories.append([initial_pose, initial_pose])
            file = folder / f"category_{task_id}_trajectory.json"
            file.write_text(json.dumps([{"state": [initial_pose[:3], initial_pose[3:]]}] * 2))
            result = checker(
                str(file),
                combined_json_path=str(REPO / "training_tasks/tasks.json"),
                task_id=task_id,
            )
            expected_rewards.append(round(float(result["reward"]), 2))
        mixed_rewards, _ = request(port, "rewards", {"trajectories": trajectories})
        for key in ("rewards", "reward_1", "reward_2", "reward_3"):
            assert len(mixed_rewards[key]) == 4
        assert all(
            abs(a - b) < 0.011 for a, b in zip(mixed_rewards["rewards"], expected_rewards)
        ), (mixed_rewards, expected_rewards)
        record["four_categories_passed"] = True
        record["four_category_rewards"] = mixed_rewards["rewards"]
        record["passed"] = True
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ports",
        required=True,
        help="Explicit HTTP ports of disposable test environments; this test resets/steps them",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tasks = json.loads((REPO / "training_tasks/tasks.json").read_text())
    ports = [int(p) for p in args.ports.split(",")]
    with ThreadPoolExecutor(max_workers=len(ports)) as pool:
        futures = [pool.submit(test_port, port, i, tasks, output) for i, port in enumerate(ports)]
        records = [future.result() for future in futures]
    report = dict(
        passed=all(item["passed"] for item in records),
        note="Camera and agent poses checked through the read-only observation endpoint; rewards checked against supplied commanded world poses.",
        ports=records,
    )
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"Artifacts: {output}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
