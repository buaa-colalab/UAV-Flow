# SPDX-License-Identifier: Apache-2.0
import json
import re

import numpy as np

# ============================================================================
# 可调阈值配置（用于判定轨迹是否符合"上升和下降"指令）
# ============================================================================

# 平均距离判定阈值
LOSS_THRESHOLD = 20.0  # 平均距离阈值（单位：与坐标单位相同，通常为厘米）
# 说明：判定轨迹是否符合"上升和下降"指令的主要指标之一（条件1）
#       计算方式：从真值轨迹拟合出完整的3D直线，计算实际轨迹中每个点到这条直线的距离，取平均值
#       如果平均距离小于LOSS_THRESHOLD，则认为轨迹与真值轨迹的偏差在可接受范围内
#       平均距离越小，说明实际轨迹越接近真值轨迹形成的直线路径

# 长度差值判定阈值
LENGTH_DIFF_THRESHOLD = 100.0  # 长度差值阈值（单位：与坐标单位相同，通常为厘米）
# 说明：判定轨迹长度是否接近真值轨迹长度的指标（条件2）
#       计算方式：计算修正后的真值轨迹长度（乘以100）与无人机轨迹长度的差值的绝对值
#       如果长度差值的绝对值小于等于LENGTH_DIFF_THRESHOLD，则认为轨迹长度满足要求
#       长度差值越小，说明实际轨迹长度越接近真值轨迹长度

# 终点距离判定阈值
ENDPOINT_DISTANCE_THRESHOLD = 100.0  # 终点距离阈值（单位：与坐标单位相同，通常为厘米）
# 说明：判定轨迹终点是否接近真值轨迹终点的指标（条件3）
#       计算方式：计算修正后的真值轨迹最后一个点（乘以100）与无人机轨迹最后一个点之间的欧氏距离（3D）
#       如果终点距离小于等于ENDPOINT_DISTANCE_THRESHOLD，则认为终点位置满足要求
#       终点距离越小，说明实际轨迹终点越接近真值轨迹终点

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


def extract_length_from_instruction(instruction):
    """
    从instruction中提取数字（长度或高度）
    只提取带".0"格式的数字，如"1.0"、"3.0"、"9.0"、"8.0"等

    参数:
        instruction: 指令字符串，例如 "Move 9.0 meters to the left at a 35-degree angle"
                     或 "Climb to an altitude of 8.0 meters with an 85-degree ascent angle"

    返回:
        float: 提取到的长度或高度，如果提取失败返回None
    """
    if not instruction:
        return None

    # 使用正则表达式提取带".0"格式的数字
    # 匹配模式：一个或多个数字，后跟".0"
    pattern = r"\d+\.0"
    matches = re.findall(pattern, instruction)

    if matches:
        # 取第一个匹配的数字作为长度或高度
        try:
            length = float(matches[0])
            return length
        except ValueError:
            return None

    return None


def fit_line_from_points_3d(points, target_length=None):
    """
    从点集拟合3D直线（使用最小二乘法）

    参数:
        points: 点集，形状 (n, 3)，在判定坐标系中
        target_length: 目标长度，如果提供，会将直线延长到该长度

    返回:
        tuple: (line_start, line_end, direction_vector) 直线的起点、终点和方向向量（3D）
    """
    points = np.array(points)
    if points.shape[1] < 3:
        # 如果点数不足3维，补零
        points_3d = np.zeros((len(points), 3))
        points_3d[:, : points.shape[1]] = points
    else:
        points_3d = points[:, :3]

    if len(points_3d) < 2:
        # 点数太少，无法拟合直线
        line_start = np.array([0.0, 0.0, 0.0])
        direction_vector = np.array([1.0, 0.0, 0.0])
        if target_length is not None:
            line_end = line_start + direction_vector * target_length
        else:
            line_end = line_start + direction_vector
        return (line_start, line_end, direction_vector)

    # 使用最小二乘法拟合3D直线
    # 方法：使用PCA或者直接计算主方向
    # 计算所有点的质心
    centroid = np.mean(points_3d, axis=0)

    # 计算协方差矩阵
    centered_points = points_3d - centroid
    cov_matrix = np.cov(centered_points.T)

    # 计算主方向（最大特征值对应的特征向量）
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    max_idx = np.argmax(eigenvalues)
    direction_vector = eigenvectors[:, max_idx]

    # 归一化方向向量
    direction_vector = direction_vector / np.linalg.norm(direction_vector)

    # 确保方向向量指向正确的方向（从第一个点到最后一个点的方向）
    first_to_last = points_3d[-1] - points_3d[0]
    if np.dot(direction_vector, first_to_last) < 0:
        direction_vector = -direction_vector

    # 确定起点：第一个点
    line_start = points_3d[0]

    # 如果提供了目标长度，延长直线到该长度
    if target_length is not None:
        line_end = line_start + direction_vector * target_length
    else:
        # 否则使用最后一个点作为终点
        line_end = points_3d[-1]

    return (line_start, line_end, direction_vector)


def point_to_line_segment_distance_3d(point, line_start, line_end, line_direction):
    """
    计算3D点到直线的垂直距离（直线是无限延伸的，不考虑线段端点限制）

    参数:
        point: 点坐标 [x, y, z]
        line_start: 直线上的一点（起点） [x, y, z]
        line_end: 直线上的一点（终点，用于确定方向） [x, y, z]
        line_direction: 直线的方向向量 [x, y, z]（已归一化）

    返回:
        点到直线的垂直距离（3D）
    """
    point = np.array(point)
    if len(point) < 3:
        # 如果点数不足3维，补零
        point_3d = np.zeros(3)
        point_3d[: len(point)] = point
    else:
        point_3d = point[:3]

    line_start = np.array(line_start)
    if len(line_start) < 3:
        line_start_3d = np.zeros(3)
        line_start_3d[: len(line_start)] = line_start
    else:
        line_start_3d = line_start[:3]

    line_end = np.array(line_end)
    line_direction = np.array(line_direction)
    if len(line_direction) < 3:
        line_direction_3d = np.zeros(3)
        line_direction_3d[: len(line_direction)] = line_direction
    else:
        line_direction_3d = line_direction[:3]

    # 确保方向向量已归一化
    line_direction_3d = line_direction_3d / np.linalg.norm(line_direction_3d)

    # 从line_start到point的向量
    vec_to_point = point_3d - line_start_3d

    # 计算投影长度
    projection_length = np.dot(vec_to_point, line_direction_3d)

    # 投影点
    projection_point = line_start_3d + projection_length * line_direction_3d

    # 点到直线的距离（3D）
    distance = np.linalg.norm(point_3d - projection_point)

    return distance


def compute_trajectory_loss_3d(actual_trajectory, reference_trajectory, target_length=None):
    """
    计算实际轨迹和真值轨迹之间的loss（3D）
    从真值轨迹拟合出完整的3D直线（如果提供了target_length会延长），计算实际轨迹中每个点到这条延长后的直线的距离

    参数:
        actual_trajectory: 实际轨迹点，形状 (n, 3)，在判定坐标系中
        reference_trajectory: 真值轨迹点，形状 (m, 3)，在判定坐标系中
        target_length: 目标长度，如果提供，会将拟合后的直线延长到该长度

    返回:
        dict: 包含loss和直线信息的字典
    """
    actual_trajectory = np.array(actual_trajectory)
    reference_trajectory = np.array(reference_trajectory)

    # 确保都是3D坐标
    if actual_trajectory.shape[1] < 3:
        actual_trajectory_3d = np.zeros((len(actual_trajectory), 3))
        actual_trajectory_3d[:, : actual_trajectory.shape[1]] = actual_trajectory
    else:
        actual_trajectory_3d = actual_trajectory[:, :3]

    if reference_trajectory.shape[1] < 3:
        reference_trajectory_3d = np.zeros((len(reference_trajectory), 3))
        reference_trajectory_3d[:, : reference_trajectory.shape[1]] = reference_trajectory
    else:
        reference_trajectory_3d = reference_trajectory[:, :3]

    # 从真值轨迹使用最小二乘法拟合3D直线（如果提供了目标长度，会延长到该长度）
    line_start, line_end, line_direction = fit_line_from_points_3d(
        reference_trajectory_3d, target_length
    )

    # 计算实际轨迹中每个点到拟合后并延长后的真值直线的距离
    distances_to_line = []
    for actual_point in actual_trajectory_3d:
        dist = point_to_line_segment_distance_3d(actual_point, line_start, line_end, line_direction)
        distances_to_line.append(dist)

    distances_to_line = np.array(distances_to_line)

    # 累加所有距离作为loss
    total_loss = np.sum(distances_to_line)

    # 计算平均距离
    mean_distance = np.mean(distances_to_line)

    # 计算最大距离
    max_distance = np.max(distances_to_line)

    # 计算实际轨迹的总长度（3D）
    actual_length = 0.0
    for i in range(len(actual_trajectory_3d) - 1):
        actual_length += np.linalg.norm(actual_trajectory_3d[i + 1] - actual_trajectory_3d[i])

    # 计算真值轨迹的总长度（3D）
    reference_length = 0.0
    for i in range(len(reference_trajectory_3d) - 1):
        reference_length += np.linalg.norm(
            reference_trajectory_3d[i + 1] - reference_trajectory_3d[i]
        )

    # 计算拟合出的直线的拟合误差（真值轨迹点到拟合直线的距离）
    reference_distances_to_line = []
    for ref_point in reference_trajectory_3d:
        dist = point_to_line_segment_distance_3d(ref_point, line_start, line_end, line_direction)
        reference_distances_to_line.append(dist)

    reference_distances_to_line = np.array(reference_distances_to_line)
    line_fit_error = np.mean(reference_distances_to_line)

    # 计算延长后的直线长度（3D）
    extended_line_length = np.linalg.norm(line_end - line_start)

    return {
        "total_loss": total_loss,  # 累加的loss
        "mean_distance": mean_distance,  # 平均距离
        "max_distance": max_distance,  # 最大距离
        "line_start": line_start.tolist(),  # 拟合出的直线的起点（3D）
        "line_end": line_end.tolist(),  # 拟合出的直线的终点（3D）
        "line_direction": line_direction.tolist(),  # 拟合出的直线的方向向量（3D）
        "line_fit_error": line_fit_error,  # 拟合误差
        "extended_line_length": extended_line_length,  # 延长后的直线长度
        "target_length": target_length,  # 目标长度
        "actual_length": actual_length,
        "reference_length": reference_length,
        "num_actual_points": len(actual_trajectory_3d),
        "num_reference_points": len(reference_trajectory_3d),
        "distances_to_line": distances_to_line.tolist(),  # 每个点到直线的距离
    }


def check_ascent_descent_trajectory(qwenoft_file, test_jsons_file, loss_threshold=None):
    """
    判定轨迹是否符合"上升"或"下降"指令（3D轨迹）

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

    # 提取实际轨迹位置(无人机坐标系中,以第0帧为原点，3D)
    # 注意：trajectory_data中的第一个点是从原点移动一帧之后的位置
    # 所以起点应该是[0,0,0]，需要添加到轨迹开头
    actual_trajectory_points = []
    # 首先添加起点[0,0,0]
    actual_trajectory_points.append(np.array([0.0, 0.0, 0.0]))
    # 然后添加所有轨迹点
    for item in trajectory_data:
        state = item["state"]
        pos = np.array(state[0])  # [x, y, z] 在无人机坐标系中
        actual_trajectory_points.append(pos)

    actual_trajectory_points = np.array(actual_trajectory_points)

    # 检查是否有有效的轨迹点（除了起点[0,0,0]之外至少还要有一个点）
    if len(actual_trajectory_points) <= 1:
        return {
            "instruction": instruction,
            "instruction_unified": instruction_unified,
            "is_valid": False,
            "error": "No trajectory points (only origin point)",
        }

    # 处理真值轨迹（reference_path_raw是世界坐标系，需要转换到无人机坐标系）
    reference_trajectory_points = []
    for ref_point in reference_path_raw:
        pos_world = np.array(ref_point[:3])  # [x, y, z] 在世界坐标系中
        pos_local = world_to_local(pos_world, initial_pos)
        reference_trajectory_points.append(pos_local)

    reference_trajectory_points = np.array(reference_trajectory_points)

    # 从instruction中提取高度（或长度）
    target_length = extract_length_from_instruction(instruction)

    # 计算loss（3D，如果提取到了高度，会延长直线到该长度）
    loss_results = compute_trajectory_loss_3d(
        actual_trajectory_points, reference_trajectory_points, target_length
    )

    # 使用平均距离作为主要loss指标
    main_loss = loss_results["mean_distance"]

    # 计算修正后的真值轨迹长度和无人机轨迹长度的差值
    # 修正后的真值轨迹长度需要乘以100来匹配仿真系统的单位长度
    extended_line_length_raw = loss_results["extended_line_length"]  # 原始延长后的直线长度
    extended_line_length = extended_line_length_raw * 100.0  # 修正后的真值轨迹长度（乘以100）
    actual_length = loss_results["actual_length"]  # 无人机轨迹长度（沿着路径累加的距离，3D）
    length_diff = extended_line_length - actual_length  # 长度差值

    # 计算实际轨迹起点到终点的直线距离（3D）
    actual_first_point_3d = actual_trajectory_points[0, :3]  # 无人机轨迹起点（3D），应该是[0,0,0]
    actual_last_point_3d = actual_trajectory_points[-1, :3]  # 无人机轨迹最后一个点（3D）
    actual_straight_distance = np.linalg.norm(
        actual_last_point_3d - actual_first_point_3d
    )  # 起点到终点的直线距离（3D）

    # 计算修正后的真值轨迹最后一个点（长度乘以100后，终点也要跟着变化，3D）
    line_start_3d = np.array(loss_results["line_start"])  # 起点（3D）
    direction_vector_3d = np.array(loss_results["line_direction"])  # 方向向量（3D）
    line_end_raw_3d = np.array(loss_results["line_end"])  # 原始终点（3D）

    # 如果提供了target_length，终点沿着方向向量延伸target_length * 100的距离
    if target_length is not None:
        line_end_corrected_3d = line_start_3d + direction_vector_3d * target_length * 100.0
    else:
        # 如果没有target_length，根据原始长度延长100倍
        original_length = np.linalg.norm(line_end_raw_3d - line_start_3d)
        line_end_corrected_3d = line_start_3d + direction_vector_3d * original_length * 100.0

    # 计算修正后的真值轨迹最后一个点和无人机轨迹最后一个点的距离（3D）
    endpoint_distance = np.linalg.norm(
        line_end_corrected_3d - actual_last_point_3d
    )  # 最后一个点的距离（3D）

    # 计算修正后的真值轨迹起点到终点的直线距离（应该等于extended_line_length，因为是直线）
    corrected_straight_distance = np.linalg.norm(
        line_end_corrected_3d - line_start_3d
    )  # 修正后的真值轨迹起点到终点的直线距离（3D）

    # 判定是否成功：需要同时满足三个条件
    # 1. main_loss < loss_threshold
    # 2. length_diff误差不超过阈值（绝对值）
    # 3. endpoint_distance不超过阈值
    condition1 = main_loss < loss_threshold
    condition2 = abs(length_diff) <= LENGTH_DIFF_THRESHOLD
    condition3 = endpoint_distance <= ENDPOINT_DISTANCE_THRESHOLD
    is_valid = condition1 and condition2 and condition3

    # 构建结果字典
    # 真值轨迹相关的字段名称带GT前缀，无人机轨迹相关的字段不带GT前缀
    result = {
        "instruction": instruction,
        "instruction_unified": instruction_unified,
        "is_valid": is_valid,
        # 无人机轨迹相关的指标（不带GT前缀）
        "main_loss": main_loss,  # 平均距离作为loss（基于无人机轨迹计算）
        "total_loss": loss_results["total_loss"],  # 累加的loss（基于无人机轨迹计算）
        "mean_distance": loss_results["mean_distance"],  # 平均距离（基于无人机轨迹计算）
        "max_distance": loss_results["max_distance"],  # 最大距离（基于无人机轨迹计算）
        "actual_length": actual_length,  # 无人机轨迹长度（沿着路径累加的距离，3D）
        "actual_last_point": actual_last_point_3d.tolist(),  # 无人机轨迹最后一个点（3D）
        "actual_straight_distance": actual_straight_distance,  # 无人机轨迹起点到终点的直线距离（3D）
        # 真值轨迹相关的字段（带GT前缀）
        "GT_start": loss_results["line_start"],  # 拟合出的直线的起点（来自真值轨迹，3D）
        "GT_end": loss_results["line_end"],  # 原始拟合出的直线的终点（来自真值轨迹，3D）
        "GT_end_point": line_end_corrected_3d.tolist(),  # 修正后的真值轨迹最后一个点（乘以100，3D）
        "GT_length": loss_results["target_length"],  # 从instruction中提取的目标高度或长度（真值）
        "GT_length_corrected": extended_line_length,  # 修正后的真值轨迹长度（乘以100）
        "GT_straight_distance": corrected_straight_distance,  # 修正后的真值轨迹起点到终点的直线距离（3D）
        # 差值或距离（不是单一来源，保持原名称）
        "length_diff": length_diff,  # 修正后的真值轨迹长度和无人机轨迹长度的差值
        "endpoint_distance": endpoint_distance,  # 修正后的真值轨迹最后一个点和无人机轨迹最后一个点的距离（3D）
        # 判定条件
        "condition1_main_loss": condition1,  # 条件1：main_loss < loss_threshold
        "condition2_length_diff": condition2,  # 条件2：abs(length_diff) <= LENGTH_DIFF_THRESHOLD
        "condition3_endpoint_distance": condition3,  # 条件3：endpoint_distance <= ENDPOINT_DISTANCE_THRESHOLD
    }

    return result
