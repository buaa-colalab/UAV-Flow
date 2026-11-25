# SPDX-License-Identifier: Apache-2.0
# Modified for UAV-Flow-Eval; upstream attribution and notices: LICENSE.
"""Frozen nDTW primitives migrated without numeric changes."""

import json

import numpy as np
from scipy.spatial.distance import cdist


def get_gt_states_from_rule_log(gt_path):
    """Load preprocessed ground-truth states from a rule-based log JSON file.

    Args:
        gt_path: Path to the ground-truth JSON file.
    Returns:
        A list of 6D states under the key 'reference_path_preprocessed'.
    """
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["reference_path_preprocessed"]


def get_sampled_state6d_from_model_rule(model_path, step=5, zero_pos=False):
    """Sample 6D states from a model trajectory log.

    The output vector per sample is [x, y, z, cos(roll), cos(yaw), cos(pitch)],
    where position can be zeroed when zero_pos is True.

    Args:
        model_path: Path to the model trajectory JSON.
        step: Sampling stride over frames.
        zero_pos: If True, set positions to zeros; otherwise use positions/100.
    Returns:
        A list of np.ndarray vectors (length 6).
    """
    with open(model_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = []
    for idx, item in enumerate(data):
        if idx % step == 0:
            if zero_pos:
                pos = np.zeros(3)
            else:
                pos = np.array(item["state"][0]) / 100
            rot = np.array(item["state"][1])
            rot_cos = np.cos(np.deg2rad(rot))
            vec = np.concatenate([pos, rot_cos])
            result.append(vec)
    return result


def get_sampled_state6d_from_gt_rule(gt_path, step=5, max_points=20, zero_pos=False):
    """Sample 6D states from ground-truth states.

    Args:
        gt_path: Path to the ground-truth JSON file.
        step: Sampling stride over frames.
        max_points: Maximum sampled points, or None to keep the complete reference.
        zero_pos: If True, set positions to zeros; otherwise use positions/100.
    Returns:
        A list of np.ndarray vectors (length 6), truncated to max_points.
    """
    states = get_gt_states_from_rule_log(gt_path)
    result = []
    for idx, item in enumerate(states):
        if idx % step == 0:
            if zero_pos:
                pos = np.zeros(3)
            else:
                pos = np.array(item[:3]) / 100
            rot = np.array(item[3:6])
            rot_cos = np.cos(np.deg2rad(rot))
            vec = np.concatenate([pos, rot_cos])
            result.append(vec)
    return result[:max_points]


def dtw_distance(vecs1, vecs2):
    """Compute DTW (Dynamic Time Warping) distance between two sequences.

    Args:
        vecs1: Sequence of vectors (list of np.ndarray) for path 1.
        vecs2: Sequence of vectors (list of np.ndarray) for path 2.
    Returns:
        A float DTW distance, or None if either sequence is empty.
    """
    if len(vecs1) == 0 or len(vecs2) == 0:
        return None
    dist_matrix = cdist(vecs1, vecs2, metric="euclidean")
    n, m = dist_matrix.shape
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = dist_matrix[i - 1, j - 1]
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return dtw[n, m]


def path_length(points):
    """Compute the polyline length for a sequence of points.

    Args:
        points: List of np.ndarray points.
    Returns:
        The total Euclidean length as float.
    """
    if len(points) < 2:
        return 0
    length = 0
    for i in range(1, len(points)):
        length += np.linalg.norm(points[i] - points[i - 1])
    return length


def ndtw(dtw_dist, gt_len, eta=1):
    """Compute normalized DTW (nDTW) score.

    nDTW = exp(- DTW / (eta * L_gt)).

    Args:
        dtw_dist: DTW distance value.
        gt_len: Ground-truth path length.
        eta: Normalization hyper-parameter (default 1).
    Returns:
        nDTW score in [0,1], or None if inputs are invalid.
    """
    if dtw_dist is None or gt_len == 0:
        return None
    return np.exp(-dtw_dist / (eta * gt_len))
