# SPDX-License-Identifier: Apache-2.0
"""Score exactly the reference task manifest; reports never enter the denominator."""

import csv
import json
import math
from pathlib import Path

from evaluation import metric
from evaluation.check_trajectories_batch import check_all_trajectories, convert_to_serializable


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(convert_to_serializable(data), indent=2, ensure_ascii=False, allow_nan=False)
    )
    temporary.replace(path)


def task_ndtw(trajectory, reference, category):
    stride = 2 if category in ("Turn", "Move") else 5
    zero_pos = category in ("Turn", "Rotate")
    if category == "Surround":
        actual_full = metric.get_sampled_state6d_from_model_rule(str(trajectory), 1)
        truth_full = metric.get_sampled_state6d_from_gt_rule(str(reference), 1, max_points=None)
        actual, truth = actual_full[::5], truth_full[::5]
        for full, sampled in [(actual_full, actual), (truth_full, truth)]:
            if full and (len(full) - 1) % 5:
                sampled.append(full[-1])
    else:
        actual = metric.get_sampled_state6d_from_model_rule(
            str(trajectory), stride, zero_pos=zero_pos
        )
        truth = metric.get_sampled_state6d_from_gt_rule(
            str(reference), stride, max_points=20, zero_pos=zero_pos
        )
    value = metric.ndtw(metric.dtw_distance(truth, actual), metric.path_length(truth), eta=1)
    if value is not None and not math.isfinite(float(value)):
        raise ValueError(f"Nonfinite nDTW: {trajectory}")
    return float(value) if value is not None else None


def score_run(output, tasks, classification, expected_filenames=None):
    output, tasks = Path(output), Path(tasks)
    expected = sorted(
        expected_filenames
        if expected_filenames is not None
        else (p.name for p in tasks.glob("*.json"))
    )
    checks = check_all_trajectories(
        output / "trajectories", tasks, Path(classification), expected_filenames=expected
    )
    atomic_json(output / "check_results.json", checks)
    if checks["total"] != len(expected) or checks["errors"]:
        raise RuntimeError(
            f"Incomplete or invalid evaluation: {checks['errors']} scoring errors for {len(expected)} expected tasks"
        )
    rows = []
    for category, details in checks["details"].items():
        for score in details:
            filename = score["filename"]
            rows.append(
                dict(
                    filename=filename,
                    category=category,
                    success=bool(score["is_valid"]),
                    cs=score.get("reward")
                    if category in ("Pass", "Land", "Approach", "Rotate")
                    else None,
                    ndtw=task_ndtw(output / "trajectories" / filename, tasks / filename, category),
                )
            )

    def summarize(group):
        values = [r["ndtw"] for r in group if r["ndtw"] is not None]
        continuous = [float(r["cs"]) for r in group if r["cs"] is not None]
        return dict(
            tasks=len(group),
            successes=sum(r["success"] for r in group),
            sr_percent=100 * sum(r["success"] for r in group) / len(group),
            ndtw=sum(values) / len(values) if values else None,
            ndtw_count=len(values),
            cs=sum(continuous) / len(continuous) if continuous else None,
        )

    metrics = dict(
        protocol="radial30_net342_surround400_others100",
        ndtw_version="surround_fullref_stride5_terminal_v1",
        overall=summarize(rows),
        by_category={
            category: summarize([r for r in rows if r["category"] == category])
            for category in sorted({r["category"] for r in rows})
        },
    )
    atomic_json(output / "metrics.json", metrics)
    atomic_json(output / "per_task_metrics.json", rows)
    with (output / "per_task_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return metrics
