import json
import numpy as np
import os
from scipy.spatial.distance import cdist
import sys


def _print_table(headers, rows, align=None):
    """Print a clean ASCII table.

    Args:
        headers: List of header strings.
        rows: List of rows (iterables of cell strings).
        align: Optional list of 'l' or 'r' for left/right alignment per column.
    """
    headers = [str(h) for h in headers]
    str_rows = [["" if c is None else str(c) for c in r] for r in rows]
    widths = [len(h) for h in headers]
    for r in str_rows:
        for i, c in enumerate(r):
            if i >= len(widths):
                widths.append(len(c))
            else:
                widths[i] = max(widths[i], len(c))
    if align is None:
        align = ['l'] * len(widths)
    def fmt_cell(i, s):
        if align[i] == 'r':
            return s.rjust(widths[i])
        return s.ljust(widths[i])
    sep = "+" + "+".join(["-" * (w + 2) for w in widths]) + "+"
    # header
    print(sep)
    print("| " + " | ".join(fmt_cell(i, headers[i]) for i in range(len(widths))) + " |")
    print(sep)
    # rows
    for r in str_rows:
        print("| " + " | ".join(fmt_cell(i, r[i] if i < len(r) else "") for i in range(len(widths))) + " |")
    print(sep)


def get_gt_states_from_rule_log(gt_path):
    """Load preprocessed ground-truth states from a rule-based log JSON file.

    Args:
        gt_path: Path to the ground-truth JSON file.
    Returns:
        A list of 6D states under the key 'reference_path_preprocessed'.
    """
    with open(gt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['reference_path_preprocessed']


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
    with open(model_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    result = []
    for idx, item in enumerate(data):
        if idx % step == 0:
            if zero_pos:
                pos = np.zeros(3)
            else:
                pos = np.array(item['state'][0]) / 100
            rot = np.array(item['state'][1])
            rot_cos = np.cos(np.deg2rad(rot))
            vec = np.concatenate([pos, rot_cos])
            result.append(vec)
    return result


def get_sampled_state6d_from_gt_rule(gt_path, step=5, max_points=20, zero_pos=False):
    """Sample 6D states from ground-truth states.

    Args:
        gt_path: Path to the ground-truth JSON file.
        step: Sampling stride over frames.
        max_points: Maximum number of sampled points to return.
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
    dist_matrix = cdist(vecs1, vecs2, metric='euclidean')
    n, m = dist_matrix.shape
    dtw = np.full((n+1, m+1), np.inf)
    dtw[0, 0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = dist_matrix[i-1, j-1]
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
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
        length += np.linalg.norm(points[i] - points[i-1])
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


def evaluate_by_classification(classified_json_path, model_dir, gt_rule_dir, default_step=5):
    """Evaluate trajectories grouped by classification.

    For each class, sample model and ground-truth states, compute nDTW,
    and report per-class and overall statistics.

    Args:
        classified_json_path: Path to a JSON mapping class_name -> [file names].
        model_dir: Directory containing model JSON files.
        gt_rule_dir: Directory containing ground-truth JSON files.
        default_step: Default sampling stride if not class-specific.
    """
    with open(classified_json_path, 'r', encoding='utf-8') as f:
        class_dict = json.load(f)
    all_ndtw_results = []
    table_rows = []

    for class_name, file_list in class_dict.items():
        ndtw_results = []
        # Position suppression for rotation-only classes
        zero_pos = class_name in ["Turn", "Rotate"]
        # Class-specific sampling stride
        if class_name in ["Turn", "Move"]:
            step = 2
        else:
            step = default_step
        num_valid = 0
        for file_name in file_list:
            folder = file_name.replace('.json', '')
            gt_path = os.path.join(gt_rule_dir, file_name)
            model_path = os.path.join(model_dir, file_name)
            if not os.path.exists(gt_path) or not os.path.exists(model_path):
                continue
            model_vecs = get_sampled_state6d_from_model_rule(model_path, step, zero_pos=zero_pos)
            gt_vecs = get_sampled_state6d_from_gt_rule(gt_path, step, max_points=20, zero_pos=zero_pos)
            dtw_dist = dtw_distance(gt_vecs, model_vecs)
            gt_len = path_length(gt_vecs)
            ndtw_score = ndtw(dtw_dist, gt_len, eta=1)
            if ndtw_score is not None:
                ndtw_results.append(ndtw_score)
                all_ndtw_results.append(ndtw_score)
            num_valid += 1
        mean_ndtw = (np.mean(ndtw_results) if len(ndtw_results) > 0 else None)
        table_rows.append([
            class_name,
            str(len(file_list)),
            str(num_valid),
            ("{:.4f}".format(mean_ndtw) if mean_ndtw is not None else "-")
        ])

    # Pretty print per-class table (nDTW only)
    print("\nUAV-Flow Evaluation by Class (nDTW)")
    _print_table(
        headers=["Class", "#Tasks", "#Evaluated", "Mean nDTW"],
        rows=table_rows,
        align=['l', 'r', 'r', 'r']
    )

    # Overall summary (nDTW only)
    print("\nOverall Summary (nDTW)")
    overall_rows = [[
        str(len(all_ndtw_results)),
        ("{:.4f}".format(np.mean(all_ndtw_results)) if len(all_ndtw_results) else "-")
    ]]
    _print_table(
        headers=["#nDTW Samples", "Overall Mean nDTW"],
        rows=overall_rows,
        align=['r', 'r']
    )


if __name__ == '__main__':
    model_list = ['openvla']

    # Redirect print output to a file for reproducible logging
    log_file = f'./metric.txt'
    sys.stdout = open(log_file, 'w', encoding='utf-8')
    
    for model in model_list:
        print("\n\n=========================")
        print(f"Model: {model}")
        print("=========================")
        model_dir = r'.\results\UnrealTrack-DowntownWest-ContinuousColor-v0\{}'.format(model)
        # Ground-truth directory
        gt_dir = r'.\test_jsons'
        classified_json_path = r'.\classified_instr.json'
        default_step = 5
        
        evaluate_by_classification(classified_json_path, model_dir, gt_dir, default_step=default_step)
        

    