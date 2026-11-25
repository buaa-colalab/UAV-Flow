# SPDX-License-Identifier: Apache-2.0
import json

import numpy as np

# ============================================================================
# 可调阈值配置（用于判定轨迹是否符合"转向物体"指令）
# ============================================================================

# 旋转角度判定阈值
ROTATION_THRESHOLD = 10.0  # 旋转角度差值阈值（单位：度）
# 说明：判定旋转是否成功的主要指标（条件1）
#       计算方式：计算实际旋转角度与真值旋转角度的差值（绝对值）
#       如果角度差值小于ROTATION_THRESHOLD度，则认为旋转角度满足要求
#       角度差值越小，说明实际旋转越接近真值旋转

# 位置距离判定阈值
DISTANCE_THRESHOLD = 100.0  # 位置距离阈值（单位：与坐标单位相同，通常为厘米）
# 说明：判定转向过程中位置是否保持稳定的指标（条件2）
#       计算方式：计算第一个轨迹点和最后一个轨迹点之间的欧氏距离
#       如果位置距离小于DISTANCE_THRESHOLD，则认为位置保持稳定（转向过程中没有明显移动）
#       位置距离越小，说明转向过程中位置变化越小

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
    yaw0 = init_pos[5]  # yaw是第6个元素（索引5）
    dx = target[0] - x0
    dy = target[1] - y0
    dz = target[2] - z0
    theta = -np.radians(yaw0)
    x_rel = dx * np.cos(theta) - dy * np.sin(theta)
    y_rel = dx * np.sin(theta) + dy * np.cos(theta)
    z_rel = dz
    return np.array([x_rel, y_rel, z_rel])


def normalize_angle(angle):
    """
    将角度归一化到 [-180, 180] 范围

    参数:
        angle: 角度（度）

    返回:
        float: 归一化后的角度
    """
    angle = angle % 360
    if angle > 180:
        angle -= 360
    elif angle < -180:
        angle += 360
    return angle


def compute_rotation_angle_diff(start_angle, end_angle):
    """
    计算从起始角度到结束角度的旋转角度差

    参数:
        start_angle: 起始角度（度）
        end_angle: 结束角度（度）

    返回:
        float: 旋转角度差（度），范围 [-180, 180]
    """
    # 归一化角度
    start_angle = normalize_angle(start_angle)
    end_angle = normalize_angle(end_angle)

    # 计算差值
    diff = end_angle - start_angle

    # 归一化差值到 [-180, 180]
    diff = normalize_angle(diff)

    return diff


def check_turn_to_object_trajectory(
    qwenoft_file, test_jsons_file, rotation_threshold=None, distance_threshold=None
):
    """
    判定轨迹是否符合"转向物体"指令

    参数:
        qwenoft_file: qwenoft文件夹中的json文件路径(无人机坐标系轨迹数据)
        test_jsons_file: test_jsons文件夹中的json文件路径
        rotation_threshold: 旋转角度差值阈值（度），小于此值认为成功（如果为None，则使用ROTATION_THRESHOLD常量）
        distance_threshold: 位置距离阈值，小于此值认为成功（如果为None，则使用DISTANCE_THRESHOLD常量）

    返回:
        dict: 包含判定结果的字典
    """
    # 如果未提供阈值，使用默认值
    if rotation_threshold is None:
        rotation_threshold = ROTATION_THRESHOLD
    if distance_threshold is None:
        distance_threshold = DISTANCE_THRESHOLD
    # 读取qwenoft文件(轨迹数据)
    with open(qwenoft_file, "r", encoding="utf-8") as f:
        trajectory_data = json.load(f)

    # 读取test_jsons文件(初始位置和目标位置)
    with open(test_jsons_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 提取数据
    _initial_pos = np.array(test_data["initial_pos"])  # [x, y, z, roll, pitch, yaw]
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

    if len(trajectory_data) == 0:
        return {
            "instruction": instruction,
            "instruction_unified": instruction_unified,
            "is_valid": False,
            "error": "No trajectory data",
        }

    # 提取第一个和最后一个state
    first_state = trajectory_data[0]["state"]
    last_state = trajectory_data[-1]["state"]

    # 提取位置和姿态
    # state[0] 是位置 [x, y, z]
    # state[1] 是姿态 [roll, pitch, yaw]
    first_pos = np.array(first_state[0])  # [x, y, z]
    _first_attitude = np.array(first_state[1])  # [roll, pitch, yaw]

    last_pos = np.array(last_state[0])  # [x, y, z]
    last_attitude = np.array(last_state[1])  # [roll, pitch, yaw]

    # 从reference_path_raw中提取真值路径（世界坐标系）
    # reference_path_raw中的每个点是 [x, y, z, roll, pitch, yaw]
    reference_first_point = np.array(reference_path_raw[0])  # 第一个点
    reference_last_point = np.array(reference_path_raw[-1])  # 最后一个点

    # 提取真值路径的初始和最终姿态（世界坐标系）
    reference_initial_attitude = reference_first_point[3:6]  # [roll, pitch, yaw]
    reference_final_attitude = reference_last_point[3:6]  # [roll, pitch, yaw]

    # 计算真值路径的旋转角度变化（在世界坐标系中）
    reference_initial_roll = reference_initial_attitude[0]
    reference_initial_pitch = reference_initial_attitude[1]
    reference_initial_yaw = reference_initial_attitude[2]

    reference_final_roll = reference_final_attitude[0]
    reference_final_pitch = reference_final_attitude[1]
    reference_final_yaw = reference_final_attitude[2]

    # 计算各个角度的变化量（世界坐标系中）
    reference_roll_diff = abs(
        compute_rotation_angle_diff(reference_initial_roll, reference_final_roll)
    )
    reference_pitch_diff = abs(
        compute_rotation_angle_diff(reference_initial_pitch, reference_final_pitch)
    )
    reference_yaw_diff = abs(
        compute_rotation_angle_diff(reference_initial_yaw, reference_final_yaw)
    )

    # 选择变化最大的角度作为真值旋转角度
    if reference_yaw_diff >= reference_pitch_diff and reference_yaw_diff >= reference_roll_diff:
        reference_rotation = reference_yaw_diff
        _reference_angle_type = "yaw"
        reference_initial_angle = reference_initial_yaw
        reference_final_angle = reference_final_yaw
    elif reference_pitch_diff >= reference_roll_diff:
        reference_rotation = reference_pitch_diff
        _reference_angle_type = "pitch"
        reference_initial_angle = reference_initial_pitch
        reference_final_angle = reference_final_pitch
    else:
        reference_rotation = reference_roll_diff
        _reference_angle_type = "roll"
        reference_initial_angle = reference_initial_roll
        reference_final_angle = reference_final_roll

    # 从qwenoft的最后一个state提取实际旋转角度（无人机坐标系中）
    # 在无人机坐标系中，初始角度为0
    actual_roll = last_attitude[0]
    actual_pitch = last_attitude[1]
    actual_yaw = last_attitude[2]

    # 计算各个角度的绝对值
    actual_roll_abs = abs(actual_roll)
    actual_pitch_abs = abs(actual_pitch)
    actual_yaw_abs = abs(actual_yaw)

    # 选择绝对值最大的角度作为实际旋转角度
    if actual_yaw_abs >= actual_pitch_abs and actual_yaw_abs >= actual_roll_abs:
        actual_rotation = actual_yaw_abs
        _actual_angle_type = "yaw"
        actual_final_angle = actual_yaw
    elif actual_pitch_abs >= actual_roll_abs:
        actual_rotation = actual_pitch_abs
        _actual_angle_type = "pitch"
        actual_final_angle = actual_pitch
    else:
        actual_rotation = actual_roll_abs
        _actual_angle_type = "roll"
        actual_final_angle = actual_roll

    # 计算旋转角度差值（取绝对值）
    rotation_diff = abs(actual_rotation - reference_rotation)

    # 计算最后一个位置点和初始位置点的距离
    position_distance = np.linalg.norm(last_pos - first_pos)

    # 判定条件1：旋转角度差值小于阈值
    rotation_valid = rotation_diff < rotation_threshold

    # 判定条件2：位置距离小于阈值
    distance_valid = position_distance < distance_threshold

    # 两个条件都满足才算成功
    is_valid = rotation_valid and distance_valid

    # 构建结果字典
    result = {
        "instruction": instruction,
        "instruction_unified": instruction_unified,
        "is_valid": is_valid,
        "rotation_valid": rotation_valid,
        "distance_valid": distance_valid,
        # 'reference_angle_type': reference_angle_type,  # 真值使用的角度类型
        # 'actual_angle_type': actual_angle_type,  # 实际使用的角度类型
        "reference_initial_angle": reference_initial_angle,  # 真值初始角度（世界坐标系）
        "reference_final_angle": reference_final_angle,  # 真值最终角度（世界坐标系）
        "reference_rotation": reference_rotation,  # 真值旋转角度（世界坐标系中）
        "actual_rotation": actual_rotation,  # 实际旋转角度（无人机坐标系中）
        "actual_final_angle": actual_final_angle,  # 实际最终角度（无人机坐标系中）
        "rotation_diff": rotation_diff,  # 旋转角度差值
        "rotation_threshold": rotation_threshold,
        "position_distance": position_distance,
        "distance_threshold": distance_threshold,
        "first_pos": first_pos.tolist(),
        "last_pos": last_pos.tolist(),
        "num_points": len(trajectory_data),
        "num_reference_points": len(reference_path_raw),
    }

    return result
