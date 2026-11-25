# SPDX-License-Identifier: Apache-2.0
# Modified for UAV-Flow-Eval; upstream attribution and notices: LICENSE.
"""Frozen simulator lifecycle differences between training and evaluation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimePolicy:
    camera_settle_seconds: int
    camera_query_seconds: int | None
    forward_gpu_id: bool
    detect_startup_exit: bool


TRAINING = RuntimePolicy(5, 2, True, True)
EVALUATION = RuntimePolicy(30, None, False, False)


def runtime_policy(profile: str) -> RuntimePolicy:
    if profile == "training":
        return TRAINING
    if profile == "evaluation":
        return EVALUATION
    raise ValueError(f"Unknown simulator runtime profile: {profile!r}")
