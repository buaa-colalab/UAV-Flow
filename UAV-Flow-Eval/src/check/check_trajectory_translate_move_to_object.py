# SPDX-License-Identifier: Apache-2.0
import json
import math
import re
from pathlib import Path

import numpy as np

# ============================================================================
# 可调阈值配置（用于判定轨迹是否符合"移动到距物体xx米"指令）
# ============================================================================

# 平均距离判定阈值
LOSS_THRESHOLD = 30.0  # 平均距离阈值（单位：与坐标单位相同，通常为厘米）
# 说明：判定轨迹是否符合"移动到距物体xx米"指令的主要指标
#       计算方式：从真值轨迹拟合出完整的直线，计算实际轨迹中每个点到这条直线的距离，取平均值
#       如果平均距离小于LOSS_THRESHOLD，则认为轨迹与真值轨迹的偏差在可接受范围内
#       平均距离越小，说明实际轨迹越接近真值轨迹形成的直线路径

# 终点距离判定阈值
ENDPOINT_THRESHOLD = 80.0  # 终点距离阈值（单位：与坐标单位相同，通常为厘米）
# 说明：判定轨迹终点是否接近真值轨迹终点的指标
#       计算方式：计算实际轨迹最后一个点与真值轨迹最后一个点之间的欧氏距离（2D）
#       如果终点距离小于ENDPOINT_THRESHOLD，则认为终点位置满足要求
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
    从instruction中提取数字（长度）
    只提取带".0"格式的数字，如"1.0"、"3.0"、"9.0"等

    参数:
        instruction: 指令字符串，例如 "Move 9.0 meters to the left at a 35-degree angle"

    返回:
        float: 提取到的长度，如果提取失败返回None
    """
    if not instruction:
        return None

    # 使用正则表达式提取带".0"格式的数字
    # 匹配模式：一个或多个数字，后跟".0"
    pattern = r"\d+\.0"
    matches = re.findall(pattern, instruction)

    if matches:
        # 取第一个匹配的数字作为长度
        try:
            length = float(matches[0])
            return length
        except ValueError:
            return None

    return None


def fit_line_from_points(points, target_length=None):
    """
    从点集拟合直线（使用最小二乘法）

    参数:
        points: 点集，形状 (n, 2) 或 (n, 3)，在判定坐标系中
        target_length: 目标长度，如果提供，会将直线延长到该长度

    返回:
        tuple: (line_start, line_end, direction_vector) 直线的起点、终点和方向向量
    """
    points = np.array(points)
    if points.shape[1] > 2:
        points_2d = points[:, :2]
    else:
        points_2d = points

    if len(points_2d) < 2:
        # 点数太少，无法拟合直线
        line_start = np.array([0.0, 0.0])
        direction_vector = np.array([1.0, 0.0])
        if target_length is not None:
            line_end = line_start + direction_vector * target_length
        else:
            line_end = line_start + direction_vector
        return (line_start, line_end, direction_vector)

    # 确定起点：第一个点
    line_start = points_2d[0]

    # 计算从第一个点到最后一个点的方向向量（用于确定方向）
    first_to_last = points_2d[-1] - points_2d[0]
    first_to_last_normalized = first_to_last / (np.linalg.norm(first_to_last) + 1e-10)

    # 使用最小二乘法拟合直线
    x = points_2d[:, 0]
    y = points_2d[:, 1]

    # 判断直线是否接近垂直（x的方差小）
    x_var = np.var(x)
    y_var = np.var(y)

    if x_var < y_var:
        # 如果x方差小，说明直线接近垂直，拟合 x = cy + d
        # 使用最小二乘法: x = a*y + b
        A = np.vstack([y, np.ones(len(y))]).T
        coeffs = np.linalg.lstsq(A, x, rcond=None)[0]
        a, _b = coeffs[0], coeffs[1]

        # 直线的方向向量（归一化）
        # 对于 x = a*y + b，方向向量是 (a, 1) 归一化
        direction_vector = np.array([a, 1.0])
        direction_vector = direction_vector / np.linalg.norm(direction_vector)

        # 确保方向向量指向从第一个点到最后一个点的方向
        if np.dot(direction_vector, first_to_last_normalized) < 0:
            direction_vector = -direction_vector

        # 如果提供了目标长度，延长直线到该长度
        if target_length is not None:
            line_end = line_start + direction_vector * target_length
        else:
            # 否则使用最后一个点作为终点
            line_end = points_2d[-1]
    else:
        # 直线不接近垂直，拟合 y = ax + b
        # 使用最小二乘法: y = a*x + b
        A = np.vstack([x, np.ones(len(x))]).T
        coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
        a, _b = coeffs[0], coeffs[1]

        # 直线的方向向量（归一化）
        # 对于 y = a*x + b，方向向量是 (1, a) 归一化
        direction_vector = np.array([1.0, a])
        direction_vector = direction_vector / np.linalg.norm(direction_vector)

        # 确保方向向量指向从第一个点到最后一个点的方向
        if np.dot(direction_vector, first_to_last_normalized) < 0:
            direction_vector = -direction_vector

        # 如果提供了目标长度，延长直线到该长度
        if target_length is not None:
            line_end = line_start + direction_vector * target_length
        else:
            # 否则使用最后一个点作为终点
            line_end = points_2d[-1]

    return (line_start, line_end, direction_vector)


def point_to_line_segment_distance(point, line_start, line_end, line_direction):
    """
    计算点到直线的垂直距离（直线是无限延伸的，不考虑线段端点限制）

    参数:
        point: 点坐标 [x, y] 或 [x, y, z]
        line_start: 直线上的一点（起点） [x, y]
        line_end: 直线上的一点（终点，用于确定方向） [x, y]
        line_direction: 直线的方向向量 [x, y]（已归一化）

    返回:
        点到直线的垂直距离
    """
    point = np.array(point)
    if len(point) > 2:
        point_2d = point[:2]
    else:
        point_2d = point

    line_start = np.array(line_start)
    line_end = np.array(line_end)
    line_direction = np.array(line_direction)

    # 确保方向向量已归一化
    line_direction = line_direction / np.linalg.norm(line_direction)

    # 从line_start到point的向量
    vec_to_point = point_2d - line_start

    # 计算投影长度
    projection_length = np.dot(vec_to_point, line_direction)

    # 投影点
    projection_point = line_start + projection_length * line_direction

    # 点到直线的距离
    distance = np.linalg.norm(point_2d - projection_point)

    return distance


def compute_trajectory_loss(actual_trajectory, reference_trajectory, target_length=None):
    """
    计算实际轨迹和真值轨迹之间的loss
    从真值轨迹拟合出完整的直线（如果提供了target_length会延长），计算实际轨迹中每个点到这条延长后的直线的距离

    参数:
        actual_trajectory: 实际轨迹点，形状 (n, 2) 或 (n, 3)，在判定坐标系中
        reference_trajectory: 真值轨迹点，形状 (m, 2) 或 (m, 3)，在判定坐标系中
        target_length: 目标长度，如果提供，会将拟合后的直线延长到该长度

    返回:
        dict: 包含loss和直线信息的字典
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

    # 从真值轨迹使用最小二乘法拟合直线（如果提供了目标长度，会延长到该长度）
    line_start, line_end, line_direction = fit_line_from_points(
        reference_trajectory_2d, target_length
    )

    # 计算实际轨迹中每个点到拟合后并延长后的真值直线的距离
    distances_to_line = []
    for actual_point in actual_trajectory_2d:
        dist = point_to_line_segment_distance(actual_point, line_start, line_end, line_direction)
        distances_to_line.append(dist)

    distances_to_line = np.array(distances_to_line)

    # 累加所有距离作为loss
    total_loss = np.sum(distances_to_line)

    # 计算平均距离
    mean_distance = np.mean(distances_to_line)

    # 计算最大距离
    max_distance = np.max(distances_to_line)

    # 计算实际轨迹的总长度
    actual_length = 0.0
    for i in range(len(actual_trajectory_2d) - 1):
        actual_length += np.linalg.norm(actual_trajectory_2d[i + 1] - actual_trajectory_2d[i])

    # 计算真值轨迹的总长度（直线长度）
    reference_length = 0.0
    for i in range(len(reference_trajectory_2d) - 1):
        reference_length += np.linalg.norm(
            reference_trajectory_2d[i + 1] - reference_trajectory_2d[i]
        )

    # 计算拟合出的直线的拟合误差（真值轨迹点到拟合直线的距离）
    reference_distances_to_line = []
    for ref_point in reference_trajectory_2d:
        dist = point_to_line_segment_distance(ref_point, line_start, line_end, line_direction)
        reference_distances_to_line.append(dist)

    reference_distances_to_line = np.array(reference_distances_to_line)
    line_fit_error = np.mean(reference_distances_to_line)

    # 计算延长后的直线长度
    extended_line_length = np.linalg.norm(line_end - line_start)

    return {
        "total_loss": total_loss,  # 累加的loss
        "mean_distance": mean_distance,  # 平均距离
        "max_distance": max_distance,  # 最大距离
        "line_start": line_start.tolist(),  # 拟合出的直线的起点
        "line_end": line_end.tolist(),  # 拟合出的直线的终点
        "line_direction": line_direction.tolist(),  # 拟合出的直线的方向向量
        "line_fit_error": line_fit_error,  # 拟合误差
        "extended_line_length": extended_line_length,  # 延长后的直线长度
        "target_length": target_length,  # 目标长度
        "actual_length": actual_length,
        "reference_length": reference_length,
        "num_actual_points": len(actual_trajectory_2d),
        "num_reference_points": len(reference_trajectory_2d),
        "distances_to_line": distances_to_line.tolist(),  # 每个点到直线的距离
    }


def check_move_to_object_distance_trajectory(
    qwenoft_file,
    test_jsons_file=None,
    combined_json_path=None,
    json_name=None,
    task_id=None,
    loss_threshold=None,
):
    """
    判定轨迹是否符合"移动到物体xx米"指令

    参数:
        qwenoft_file: qwenoft文件夹中的json文件路径(无人机坐标系轨迹数据)
        test_jsons_file: test_jsons文件夹中的json文件路径（可选，如果提供combined_json_path则不需要）
        combined_json_path: combined.json文件路径（可选，如果提供则从此文件读取数据）
        json_name: json文件名（用于在combined.json中查找对应的任务定义，可选）
        task_id: 任务ID（用于在combined.json中查找对应的任务定义，可选）
        loss_threshold: 平均距离阈值（可选，默认使用文件顶部定义的LOSS_THRESHOLD）

    返回:
        dict: 包含判定结果的字典
    """
    # 如果未提供loss_threshold，使用默认值
    if loss_threshold is None:
        loss_threshold = LOSS_THRESHOLD

    # 读取qwenoft文件(轨迹数据)
    with open(qwenoft_file, "r", encoding="utf-8") as f:
        trajectory_data = json.load(f)

    # 从combined.json或test_jsons文件读取数据
    instruction = ""
    instruction_unified = ""
    _use_obj = None
    initial_pos = None  # [x, y, z, roll, pitch, yaw]
    reference_path_raw = []  # 参考轨迹数据（世界坐标系）

    if combined_json_path is not None:
        # 从combined.json读取数据
        combined_json_file = Path(combined_json_path)
        if combined_json_file.exists():
            with open(combined_json_file, "r", encoding="utf-8") as f:
                combined_data = json.load(f)
            # 查找对应的task
            for task in combined_data:
                if isinstance(task, dict) and task.get("task_id") == task_id:
                    # 提取数据（兼容init_pose和initial_pos两种字段名）
                    initial_pos_raw = task.get("initial_pos") or task.get("init_pose")
                    if initial_pos_raw is not None and len(initial_pos_raw) >= 5:
                        initial_pos = np.array(initial_pos_raw)

                    instruction = task.get("instruction", "") or task.get("prompt", "")
                    instruction_unified = task.get("instruction_unified", "")
                    _use_obj = task.get("use_obj", None)

                    # 优先使用reference_path_raw字段
                    reference_path_raw = task.get("reference_path_raw", [])

                    break
    elif test_jsons_file is not None:
        # 从test_jsons文件读取数据
        with open(test_jsons_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        # 提取数据（兼容init_pose和initial_pos两种字段名）
        initial_pos_raw = test_data.get("initial_pos") or test_data.get("init_pose")
        if initial_pos_raw is not None and len(initial_pos_raw) >= 5:
            initial_pos = np.array(initial_pos_raw)

        instruction = test_data.get("instruction", "") or test_data.get("prompt", "")
        instruction_unified = test_data.get("instruction_unified", "")
        _use_obj = test_data.get("use_obj", None)
        reference_path_raw = test_data.get("reference_path_raw", [])

    # 如果仍然没有initial_pos，使用默认值
    if initial_pos is None:
        return {
            "instruction": instruction,
            "instruction_unified": instruction_unified,
            "is_valid": False,
            "error": "无法加载测试数据：缺少initial_pos或init_pose字段",
            "distance_valid": False,
            "endpoint_valid": False,
        }

    if len(reference_path_raw) == 0:
        return {
            "instruction": instruction,
            "instruction_unified": instruction_unified,
            "is_valid": False,
            "error": "No reference path found",
            "distance_valid": False,
            "endpoint_valid": False,
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
            "distance_valid": False,
            "endpoint_valid": False,
        }

    # 处理真值轨迹（reference_path_raw是世界坐标系，需要转换到无人机坐标系）
    reference_trajectory_points = []
    for ref_point in reference_path_raw:
        pos_world = np.array(ref_point[:3])  # [x, y, z] 在世界坐标系中
        pos_local = world_to_local(pos_world, initial_pos)
        reference_trajectory_points.append(pos_local)

    reference_trajectory_points = np.array(reference_trajectory_points)

    # 计算loss（不延长直线）
    loss_results = compute_trajectory_loss(
        actual_trajectory_points, reference_trajectory_points, None
    )

    # 使用平均距离作为主要loss指标
    main_loss = loss_results["mean_distance"]

    # 提取位置信息
    first_pos = actual_trajectory_points[0]
    last_pos = actual_trajectory_points[-1]

    # 计算位置距离
    _position_distance = np.linalg.norm(last_pos - first_pos)

    # 计算真值轨迹最后一个点和实际轨迹最后一个点的距离
    reference_last_point = reference_trajectory_points[-1]
    endpoint_distance = np.linalg.norm(last_pos[:2] - reference_last_point[:2])

    # 计算实际轨迹起点到终点的直线距离（2D）
    actual_first_point_2d = actual_trajectory_points[0, :2]  # 无人机轨迹第一个点（2D）
    actual_last_point_2d = actual_trajectory_points[-1, :2]  # 无人机轨迹最后一个点（2D）
    actual_straight_distance = np.linalg.norm(
        actual_last_point_2d - actual_first_point_2d
    )  # 起点到终点的直线距离

    # 计算实际轨迹长度
    actual_length = loss_results["actual_length"]

    # 判定条件1：平均距离小于阈值
    distance_valid = main_loss < loss_threshold

    # 判定条件2：终点距离小于阈值
    endpoint_threshold = ENDPOINT_THRESHOLD
    endpoint_valid = endpoint_distance < endpoint_threshold

    # 两个条件都满足才算成功
    is_valid = distance_valid and endpoint_valid

    # ========== 新增Reward计算 ==========
    # Reward计算：停止位置距离标准位置的距离
    # 100以内得1分，每远100扣0.1，最少得0分
    # 使用任务内参考轨迹的终点距离。
    dist_to_standard = endpoint_distance

    # 计算reward
    # 离散分段扣分：每100为一个段，每段扣0.1分
    if dist_to_standard <= 100:
        reward = 1.0
    else:
        # 离散分段扣分：每100为一个段，每段扣0.1分
        deduction_steps = (dist_to_standard - 100) / 100.0
        deduction_steps = math.ceil(deduction_steps)  # 向上取整
        reward = max(0.0, 1.0 - deduction_steps * 0.1)

    # 确保reward在[0, 1]范围内
    reward = max(0.0, min(1.0, reward))

    # ========== Reward计算结束 ==========

    # 构建结果字典
    # 真值轨迹相关的字段名称带GT前缀，无人机轨迹相关的字段不带GT前缀
    # 兼容prompt和instruction两种字段名
    prompt = (
        task.get("prompt", "") if combined_json_path is not None else test_data.get("prompt", "")
    )
    result = {
        "instruction": instruction,
        "instruction_unified": instruction_unified,
        "prompt": prompt,
        "is_valid": is_valid,
        # 无人机轨迹相关的指标（不带GT前缀）
        "main_loss": main_loss,  # 平均距离作为loss（基于无人机轨迹计算）
        "total_loss": loss_results["total_loss"],  # 累加的loss（基于无人机轨迹计算）
        "mean_distance": loss_results["mean_distance"],  # 平均距离（基于无人机轨迹计算）
        "max_distance": loss_results["max_distance"],  # 最大距离（基于无人机轨迹计算）
        "actual_length": actual_length,  # 无人机轨迹长度（沿着路径累加的距离）
        "actual_last_point": actual_last_point_2d.tolist(),  # 无人机轨迹最后一个点（2D）
        "actual_straight_distance": actual_straight_distance,  # 无人机轨迹起点到终点的直线距离
        # 真值轨迹相关的字段（带GT前缀）
        "GT_start": loss_results["line_start"],  # 拟合出的直线的起点（来自真值轨迹）
        "GT_end": loss_results["line_end"],  # 原始拟合出的直线的终点（来自真值轨迹）
        "GT_end_point": reference_last_point[:2].tolist(),  # 真值轨迹最后一个点（2D）
        # 差值或距离（不是单一来源，保持原名称）
        "endpoint_distance": endpoint_distance,  # 真值轨迹最后一个点和实际轨迹最后一个点的距离
        # 判定条件
        "distance_valid": distance_valid,  # 平均距离判定结果
        "endpoint_valid": endpoint_valid,  # 终点距离判定结果
        "endpoint_threshold": endpoint_threshold,  # 终点距离阈值
        # 新增Reward相关字段
        "dist_to_standard": dist_to_standard,  # 停止位置距离标准位置的距离
        "reward": reward,  # 总reward（0-1之间）
    }

    return result
