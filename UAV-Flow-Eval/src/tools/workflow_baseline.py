# SPDX-License-Identifier: Apache-2.0
"""Capture deterministic workflow behavior without starting UE or model services.

The committed fixture was captured before shared-code consolidation. Updating it
requires an explicit protocol decision, not just accepting a refactor's output.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import sys
import tempfile
from functools import partial
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

CHECKERS = {
    "Turn": ("turn_to_object", "check_turn_to_object_trajectory"),
    "Move": ("translate_move_to_object", "check_move_to_object_distance_trajectory"),
    "Shift": ("translate", "check_translate_trajectory"),
    "Rotate": ("rotate", "check_rotate_trajectory"),
    "Surround": ("orbit", "check_orbit_trajectory"),
    "Ascend/Descend": ("translate_up_and_down", "check_ascent_descent_trajectory"),
    "Approach": ("approach", "check_approach_trajectory"),
    "Retreat": ("back", "check_backward_trajectory"),
    "Pass": ("pass", "check_pass_trajectory"),
    "Land": ("land", "check_land_trajectory"),
}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize(value, temporary=""):
    if isinstance(value, np.ndarray):
        return normalize(value.tolist(), temporary)
    if isinstance(value, np.generic):
        return normalize(value.item(), temporary)
    if isinstance(value, float):
        return round(value, 10) if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(k): normalize(v, temporary) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalize(v, temporary) for v in value]
    if isinstance(value, str) and temporary:
        return value.replace(temporary, "<fixture>")
    return value


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def reference_manifest(root, relative, recursive=False):
    folder = root / relative
    paths = folder.rglob("*.json") if recursive else folder.glob("*.json")
    hashes = {
        str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
    }
    return {"files": len(hashes), "sha256": fingerprint(hashes)}


def simulator_behavior(root, profile):
    legacy = root / "evaluation/gym_unrealcv/envs/base_env.py"
    source = (
        legacy
        if profile == "evaluation" and legacy.exists()
        else root / "gym_unrealcv/envs/base_env.py"
    )
    module = load("baseline_base_" + profile, source)
    cls = module.UnrealCv_base
    env = cls.__new__(cls)
    env.runtime_profile = profile
    for name, value in {
        "docker": False,
        "resolution": (256, 256),
        "display": None,
        "use_opengl": False,
        "offscreen_rendering": True,
        "nullrhi": False,
        "gpu_id": 2,
        "comm_mode": "tcp",
        "env_name": "DowntownWest",
    }.items():
        setattr(env, name, value)
    env.ue_binary = MagicMock()
    env.ue_binary.start.return_value = ("127.0.0.1", 9488)
    env._wait_for_port = MagicMock()
    with patch.object(module, "Character_API"):
        launched = env.launch_ue_env()
    start = env.ue_binary.start.call_args.kwargs
    launch = {
        "success": launched,
        "kwargs": normalize(start),
        "wait_checks_process": "process" in env._wait_for_port.call_args.kwargs,
    }
    settling = {}
    for scenario in ("settles", "deadline", "query_timeout"):
        env.player_list = ["drone", "npc"]
        env.cam_list = [0, 1]
        env.action_space = [0, 1]
        env.observation_space = [0, 1]
        env.agents = {"drone": {}, "npc": {}}
        env.remove_cam = MagicMock(return_value=[0])
        env.unrealcv = MagicMock()
        if scenario == "settles":
            env.unrealcv.get_camera_num.side_effect = [4, 3]
            env.unrealcv.client.request.side_effect = ["0 1 2 3", "0 1 2"]
        elif scenario == "deadline":
            env.unrealcv.get_camera_num.return_value = 4
            env.unrealcv.client.request.return_value = "0 1 2 3"
        else:
            env.unrealcv.get_camera_num.return_value = 3
            env.unrealcv.client.request.return_value = None
        error = None
        with (
            patch.object(module.time, "sleep"),
            patch.object(
                module.time,
                "monotonic",
                side_effect=[0, 1, 2] if scenario == "settles" else [0, 6, 7],
            ),
            patch.object(
                module.time,
                "time",
                side_effect=[0, 0, 1, 2] if scenario == "settles" else [0, 0, 31, 32],
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            try:
                env.remove_agent("npc")
            except (TimeoutError, TypeError) as exc:
                error = type(exc).__name__
        settling[scenario] = {
            "error": error,
            "bounded_queries": [
                normalize(call.kwargs) for call in env.unrealcv.client.request.call_args_list
            ],
            "legacy_queries": env.unrealcv.get_camera_num.call_count,
            "remaining_agents": env.player_list,
        }
    return {"launch": launch, "camera_settling": settling}


def batch_behavior(root, classification):
    batch = load("baseline_batch", root / "evaluation/closed_loop_runner.py")
    result = {}
    with tempfile.TemporaryDirectory(prefix="uav-batch-fixtures-") as folder:
        for category, steps in [("Rotate", 9), ("Rotate", 15), ("Rotate", 100), ("Surround", 15)]:
            case = Path(folder) / (category + str(steps))
            tasks = case / "tasks"
            tasks.mkdir(parents=True)
            name = next(
                name
                for name in classification[category]
                if (root / "evaluation/test_jsons" / name).exists()
            )
            (tasks / name).write_bytes((root / "evaluation/test_jsons" / name).read_bytes())
            (case / "out/trajectories/logs").mkdir(parents=True)

            def control(*args, **kwargs):
                kwargs["trajectory_log"].extend(
                    {"state": [[i, 0, 0], [0, 0, 0]]} for i in range(steps)
                )
                return True

            with contextlib.ExitStack() as stack:
                for target, value in [
                    ("configure_logging", "isolated.log"),
                    ("ensure_env_ready", MagicMock()),
                    ("create_obj_if_needed", None),
                    ("reset_model", None),
                    ("draw_2d_trajectory_from_log", None),
                    ("draw_3d_trajectory_from_log", None),
                ]:
                    stack.enter_context(patch.object(batch, target, return_value=value))
                stack.enter_context(patch.object(batch, "control_loop", side_effect=control))
                stack.enter_context(patch.object(batch.time, "sleep"))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                code = batch.main(
                    ["-f", str(tasks), "-o", str(case / "out"), "--task_retries", "0"]
                )
            final = json.loads(
                next((case / "out/trajectories/logs").glob("*_actions.json")).read_text()
            )["final"]
            result[category + "/" + str(steps)] = {
                "exit_code": code,
                **{
                    k: final[k]
                    for k in (
                        "execution_ok",
                        "termination_reason",
                        "effective_max_steps",
                        "raw_trajectory_steps",
                        "saved_trajectory_steps",
                        "trimmed_tail_steps",
                    )
                },
            }
    return result


def capture(root):
    root = Path(root).resolve()
    if (root / "src").is_dir():
        root = root / "src"
    sys.path[:0] = [str(root), str(root / "evaluation")]
    classification = json.loads((root / "evaluation/classified_instr.json").read_text())
    task_names = {p.name for p in (root / "evaluation/test_jsons").glob("*.json")}
    counts = {
        category: len(task_names.intersection(names)) for category, names in classification.items()
    }
    training = json.loads((root / "training_tasks/tasks.json").read_text())
    training_categories = json.loads((root / "training_tasks/task_categories.json").read_text())
    result = {
        "schema_version": 1,
        "data": {
            "training_tasks": len(training),
            "training_unique_ids": len({r["task_id"] for r in training}),
            "training_category_counts": {k: len(v) for k, v in training_categories.items()},
            "evaluation_tasks": len(task_names),
            "evaluation_category_counts": counts,
            "training_references": reference_manifest(root, "training_tasks", True),
            "evaluation_references": reference_manifest(root, "evaluation/test_jsons"),
            "classification_sha256": hashlib.sha256(
                (root / "evaluation/classified_instr.json").read_bytes()
            ).hexdigest(),
        },
        "simulator": {
            profile: simulator_behavior(root, profile) for profile in ("training", "evaluation")
        },
        "batch": batch_behavior(root, classification),
        "checkers": {},
        "orbit_policy": {},
    }
    scoring = load("baseline_scoring", root / "evaluation/scoring.py")
    with tempfile.TemporaryDirectory(prefix="uav-behavior-fixtures-") as temporary:
        temporary = Path(temporary)
        for profile in ("training", "evaluation"):
            functions = {}
            for category, (suffix, function) in CHECKERS.items():
                path = (
                    root
                    / ("evaluation/check" if profile == "evaluation" else "check")
                    / ("check_trajectory_" + suffix + ".py")
                )
                if not path.exists():
                    path = root / "check" / path.name
                module = load("baseline_" + profile + "_" + suffix, path)
                checker = getattr(module, function)
                if profile == "evaluation" and category == "Surround":
                    checker = partial(checker, min_directed_angle_deg=module.MIN_DIRECTED_ANGLE_DEG)
                functions[category] = checker
            cases = {}
            for category in sorted(CHECKERS):
                name = sorted(task_names.intersection(classification[category]))[0]
                reference = root / "evaluation/test_jsons" / name
                task = json.loads(reference.read_text())
                poses = task["reference_path_preprocessed"]
                for variant, points in {
                    "reference": poses,
                    "stationary": [[0] * 6] * 100,
                    "reverse": list(reversed(poses)),
                }.items():
                    actual = temporary / "actual.json"
                    actual.write_text(json.dumps([{"state": [p[:3], p[3:6]]} for p in points]))
                    with contextlib.redirect_stdout(io.StringIO()):
                        outcome = normalize(
                            normalize(
                                functions[category](str(actual), str(reference)), str(temporary)
                            ),
                            str(root),
                        )
                    case = {
                        "reference": name,
                        "result_sha256": fingerprint(outcome),
                        "is_valid": outcome.get("is_valid"),
                        "reward": outcome.get("reward"),
                        "error": outcome.get("error"),
                    }
                    if profile == "evaluation":
                        case["ndtw"] = normalize(scoring.task_ndtw(actual, reference, category))
                    cases[category + "/" + variant] = case
            result["checkers"][profile] = cases

            def circle(angles, radius=300):
                return [
                    [
                        300 + radius * math.cos(math.pi + math.radians(a)),
                        radius * math.sin(math.pi + math.radians(a)),
                        0,
                    ]
                    for a in angles
                ]

            reference = temporary / "orbit_reference.json"
            reference.write_text(
                json.dumps({"initial_pos": [0] * 6, "reference_path_raw": circle(range(361))})
            )
            for name, points in {
                "stationary": [[0, 0, 0]] * 100,
                "half": circle(range(1, 181)),
                "341deg": circle(range(1, 342)),
                "342deg": circle(range(1, 343)),
                "full": circle(range(1, 361)),
                "wrong_direction": circle(range(-1, -361, -1)),
                "backtrack": circle([*range(1, 181), *range(179, -1, -1)]),
                "radial_error": circle(range(1, 361), 350),
            }.items():
                actual.write_text(json.dumps([{"state": [p, [0, 0, 0]]} for p in points]))
                outcome = normalize(
                    functions["Surround"](str(actual), str(reference)), str(temporary)
                )
                result["orbit_policy"][profile + "/" + name] = {
                    "is_valid": outcome["is_valid"],
                    "result_sha256": fingerprint(outcome),
                }
    return normalize(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--write", type=Path, required=True)
    args = parser.parse_args()
    args.write.write_text(
        json.dumps(capture(args.root), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
