# SPDX-License-Identifier: Apache-2.0
import json
import math

import numpy as np

# ============================================================================
# 可调阈值配置（用于判定轨迹是否符合"环绕"指令）
# ============================================================================

# 环绕判定阈值
LOSS_THRESHOLD = 30.0  # 平均距离阈值（单位：厘米）
MIN_DIRECTED_ANGLE_DEG = 342.0
# 说明：判定轨迹是否符合环绕指令的主要指标
#       计算方式：从真值轨迹拟合出完整的圆，计算实际轨迹中每个点到这个圆的距离，取平均值
#       如果平均距离小于LOSS_THRESHOLD，则认为环绕成功
#       平均距离越小，说明实际轨迹越接近真值轨迹形成的圆形路径

# ============================================================================


def world_to_local(target, init_pos):
    """
    将世界坐标系中的点转换到无人机坐标系（判定坐标系）
    与closed_loop_runner.py中的world_to_local函数一致

    参数:
        target: 世界坐标系中的点 [x, y, z] 或 [x, y, z, ...]
        init_pos: 初始位置 [x, y, z, roll, pitch, yaw]

    返回:
        无人机坐标系中的点 [x, y, z]
    """
    x0, y0, z0 = init_pos[0:3]
    yaw0 = init_pos[4]
    dx = target[0] - x0
    dy = target[1] - y0
    dz = target[2] - z0
    theta = -np.radians(yaw0)
    x_rel = dx * np.cos(theta) - dy * np.sin(theta)
    y_rel = dx * np.sin(theta) + dy * np.cos(theta)
    z_rel = dz
    return np.array([x_rel, y_rel, z_rel])


def fit_circle_from_points(points):
    """
    从点集拟合圆（使用最小二乘法）

    参数:
        points: 点集，形状 (n, 2) 或 (n, 3)，在判定坐标系中

    返回:
        tuple: (center_x, center_y, radius) 圆心和半径
    """
    points = np.array(points)
    if points.shape[1] > 2:
        points_2d = points[:, :2]
    else:
        points_2d = points

    if len(points_2d) < 3:
        # 点数太少，无法拟合圆，返回默认值
        return (0.0, 0.0, 0.0)

    # 最小二乘法拟合圆
    # 圆的一般方程: (x - cx)^2 + (y - cy)^2 = r^2
    # 展开: x^2 + y^2 - 2*cx*x - 2*cy*y + (cx^2 + cy^2 - r^2) = 0
    # 设: A = -2*cx, B = -2*cy, C = cx^2 + cy^2 - r^2
    # 则: x^2 + y^2 + A*x + B*y + C = 0

    x = points_2d[:, 0]
    y = points_2d[:, 1]

    # 构建线性方程组
    A = np.column_stack([x, y, np.ones(len(x))])
    b = -(x**2 + y**2)

    # 求解线性方程组
    try:
        solution = np.linalg.lstsq(A, b, rcond=None)[0]

        # 从解中提取圆心和半径
        center_x = -solution[0] / 2.0
        center_y = -solution[1] / 2.0
        radius = np.sqrt(center_x**2 + center_y**2 - solution[2])

        # 确保半径为正值
        if radius < 0 or not np.isfinite(radius):
            # 如果拟合失败，使用所有点的中心作为圆心，平均距离作为半径
            center_x = np.mean(x)
            center_y = np.mean(y)
            distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            radius = np.mean(distances)

        return (center_x, center_y, radius)
    except BaseException:
        # 拟合失败，使用所有点的中心作为圆心，平均距离作为半径
        center_x = np.mean(x)
        center_y = np.mean(y)
        distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        radius = np.mean(distances)
        return (center_x, center_y, radius)


def point_to_circle_distance(point, circle_center, circle_radius):
    """
    计算点到圆的距离

    参数:
        point: 点坐标 [x, y] 或 [x, y, z]
        circle_center: 圆心 [x, y]
        circle_radius: 半径

    返回:
        点到圆的距离（点在圆外为正，圆内为负，但取绝对值）
    """
    point = np.array(point)
    if len(point) > 2:
        point_2d = point[:2]
    else:
        point_2d = point

    circle_center = np.array(circle_center)

    # 计算点到圆心的距离
    dist_to_center = np.linalg.norm(point_2d - circle_center)

    # 点到圆的距离 = |点到圆心的距离 - 半径|
    distance = abs(dist_to_center - circle_radius)

    return distance


def compute_trajectory_loss(actual_trajectory, reference_trajectory, *, include_progress=False):
    """
    计算实际轨迹和真值轨迹之间的loss
    从真值轨迹拟合出完整的圆，计算实际轨迹中每个点到这个圆的距离，累加作为loss

    参数:
        actual_trajectory: 实际轨迹点，形状 (n, 2) 或 (n, 3)，在判定坐标系中
        reference_trajectory: 真值轨迹点，形状 (m, 2) 或 (m, 3)，在判定坐标系中

    返回:
        dict: 包含loss和圆信息的字典
    """
    actual_trajectory = np.array(actual_trajectory)
    reference_trajectory = np.array(reference_trajectory)

    # 确保都是2D坐标（只考虑x, y）
    if actual_trajectory.shape[1] > 2:
        actual_trajectory_2d = actual_trajectory[:, :2]
    else:
        actual_trajectory_2d = actual_trajectory

    if reference_trajectory.shape[1] > 2:
        reference_trajectory_2d = reference_trajectory[:, :2]
    else:
        reference_trajectory_2d = reference_trajectory

    # 从真值轨迹拟合圆
    circle_center_x, circle_center_y, circle_radius = fit_circle_from_points(
        reference_trajectory_2d
    )
    circle_center = np.array([circle_center_x, circle_center_y])

    # 计算实际轨迹中每个点到拟合出的圆的距离
    distances_to_circle = []
    for actual_point in actual_trajectory_2d:
        dist = point_to_circle_distance(actual_point, circle_center, circle_radius)
        distances_to_circle.append(dist)

    distances_to_circle = np.array(distances_to_circle)

    # 累加所有距离作为loss
    total_loss = np.sum(distances_to_circle)

    # 计算平均距离
    mean_distance = np.mean(distances_to_circle)

    # 计算最大距离
    max_distance = np.max(distances_to_circle)

    # 计算实际轨迹的总长度
    actual_length = 0.0
    for i in range(len(actual_trajectory_2d) - 1):
        actual_length += np.linalg.norm(actual_trajectory_2d[i + 1] - actual_trajectory_2d[i])

    # 计算真值轨迹的总长度（圆周长）
    reference_length = 2 * math.pi * circle_radius if circle_radius > 0 else 0.0

    # 计算拟合出的圆的拟合误差（真值轨迹点到拟合圆的距离）
    reference_distances_to_circle = []
    for ref_point in reference_trajectory_2d:
        dist = point_to_circle_distance(ref_point, circle_center, circle_radius)
        reference_distances_to_circle.append(dist)

    reference_distances_to_circle = np.array(reference_distances_to_circle)
    circle_fit_error = np.mean(reference_distances_to_circle)

    result = {
        "total_loss": total_loss,  # 累加的loss
        "mean_distance": mean_distance,  # 平均距离
        "max_distance": max_distance,  # 最大距离
        "circle_center": [circle_center_x, circle_center_y],  # 拟合出的圆心
        "circle_radius": circle_radius,  # 拟合出的半径
        "circle_fit_error": circle_fit_error,  # 拟合误差
        "actual_length": actual_length,
        "reference_length": reference_length,
        "num_actual_points": len(actual_trajectory_2d),
        "num_reference_points": len(reference_trajectory_2d),
        "distances_to_circle": distances_to_circle.tolist(),  # 每个点到圆的距离
    }
    if include_progress:
        # Evaluation logs omit the initial pose (the local origin). Preserve the
        # reference direction and let reversals cancel rather than add progress.
        actual_with_start = np.vstack(([0.0, 0.0], actual_trajectory_2d))
        actual_angles = np.unwrap(
            np.arctan2(
                actual_with_start[:, 1] - circle_center_y,
                actual_with_start[:, 0] - circle_center_x,
            )
        )
        reference_angles = np.unwrap(
            np.arctan2(
                reference_trajectory_2d[:, 1] - circle_center_y,
                reference_trajectory_2d[:, 0] - circle_center_x,
            )
        )
        reference_signed_angle_deg = float(np.degrees(reference_angles[-1] - reference_angles[0]))
        actual_signed_angle_deg = float(np.degrees(actual_angles[-1] - actual_angles[0]))
        result.update(
            actual_signed_angle_deg=actual_signed_angle_deg,
            reference_signed_angle_deg=reference_signed_angle_deg,
            directed_net_angle_deg=float(np.sign(reference_signed_angle_deg))
            * actual_signed_angle_deg,
        )
    return result


def orbit_is_complete(
    metrics, loss_threshold=LOSS_THRESHOLD, min_directed_angle_deg=MIN_DIRECTED_ANGLE_DEG
):
    """Evaluation's online/offline success rule, shared by both callers."""
    return bool(
        metrics["mean_distance"] < loss_threshold
        and metrics["directed_net_angle_deg"] + 1e-6 >= min_directed_angle_deg
    )


def check_orbit_trajectory(
    qwenoft_file, test_jsons_file, loss_threshold=None, *, min_directed_angle_deg=None
):
    """
    判定轨迹是否符合"环绕"指令

    参数:
        qwenoft_file: qwenoft文件夹中的json文件路径(无人机坐标系轨迹数据)
        test_jsons_file: test_jsons文件夹中的json文件路径
        loss_threshold: loss阈值（平均距离），小于此值认为成功（默认使用LOSS_THRESHOLD）

    返回:
        dict: 包含判定结果的字典
    """
    # 如果未提供loss_threshold，使用默认值
    if loss_threshold is None:
        loss_threshold = LOSS_THRESHOLD
    # 读取qwenoft文件(轨迹数据)
    with open(qwenoft_file, "r", encoding="utf-8") as f:
        trajectory_data = json.load(f)

    # 读取test_jsons文件(初始位置和目标位置)
    with open(test_jsons_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 提取数据
    initial_pos = np.array(test_data["initial_pos"])  # [x, y, z, roll, pitch, yaw]
    instruction = test_data.get("instruction", "")
    instruction_unified = test_data.get("instruction_unified", "")
    reference_path_raw = test_data.get("reference_path_raw", [])

    if len(reference_path_raw) == 0:
        return {
            "instruction": instruction,
            "instruction_unified": instruction_unified,
            "is_valid": False,
            "error": "No reference path found",
        }

    # 提取实际轨迹位置(无人机坐标系中,以第一帧为原点)
    actual_trajectory_points = []
    for item in trajectory_data:
        state = item["state"]
        pos = np.array(state[0])  # [x, y, z] 在无人机坐标系中
        actual_trajectory_points.append(pos)

    actual_trajectory_points = np.array(actual_trajectory_points)

    if len(actual_trajectory_points) == 0:
        return {
            "instruction": instruction,
            "instruction_unified": instruction_unified,
            "is_valid": False,
            "error": "No trajectory points",
        }

    # 处理真值轨迹（reference_path_raw是世界坐标系，需要转换到无人机坐标系）
    reference_trajectory_points = []
    for ref_point in reference_path_raw:
        pos_world = np.array(ref_point[:3])  # [x, y, z] 在世界坐标系中
        pos_local = world_to_local(pos_world, initial_pos)
        reference_trajectory_points.append(pos_local)

    reference_trajectory_points = np.array(reference_trajectory_points)

    # 计算loss
    loss_results = compute_trajectory_loss(
        actual_trajectory_points,
        reference_trajectory_points,
        include_progress=min_directed_angle_deg is not None,
    )

    # 使用平均距离作为主要loss指标
    main_loss = loss_results["mean_distance"]

    # 判定是否成功：平均距离小于阈值
    is_valid = main_loss < loss_threshold
    if min_directed_angle_deg is not None:
        is_valid = orbit_is_complete(loss_results, loss_threshold, min_directed_angle_deg)

    # 构建结果字典
    result = {
        "instruction": instruction,
        "instruction_unified": instruction_unified,
        "is_valid": is_valid,
        "main_loss": main_loss,  # 平均距离作为loss
        "total_loss": loss_results["total_loss"],  # 累加的loss（保留供参考）
        "mean_distance": loss_results["mean_distance"],  # 平均距离
        "max_distance": loss_results["max_distance"],  # 最大距离
        "loss_threshold": loss_threshold,
        "loss_details": loss_results,
        "circle_center": loss_results["circle_center"],  # 拟合出的圆心
        "circle_radius": loss_results["circle_radius"],  # 拟合出的半径
        "actual_trajectory_points": actual_trajectory_points[:, :2].tolist(),  # 只保存2D坐标
        "reference_trajectory_points": reference_trajectory_points[:, :2].tolist(),  # 只保存2D坐标
    }

    if min_directed_angle_deg is not None:
        result.update(
            directed_net_angle_deg=loss_results["directed_net_angle_deg"],
            min_directed_angle_deg=min_directed_angle_deg,
            radial_error_ok=bool(main_loss < loss_threshold),
            angle_coverage_ok=bool(
                loss_results["directed_net_angle_deg"] + 1e-6 >= min_directed_angle_deg
            ),
        )
    return result
