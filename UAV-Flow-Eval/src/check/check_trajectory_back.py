# SPDX-License-Identifier: Apache-2.0
import json

import numpy as np

# ============================================================================
# 可调阈值配置（用于判定轨迹是否符合"后退"指令）
# ============================================================================

# 轨迹偏移判定阈值
MAX_DISTANCE_FROM_X_AXIS = 100.0  # 轨迹点距离x轴的最大偏移量阈值（单位：米）
# 说明：在后退过程中，所有轨迹点距离x轴（无人机前进方向）的偏移量（在y-z平面内的距离）
#       不能超过MAX_DISTANCE_FROM_X_AXIS，超过则判定失败
#       计算公式：sqrt(y^2 + z^2) <= MAX_DISTANCE_FROM_X_AXIS

# 停止判定阈值（用于信息记录，不直接影响is_valid）
STOP_THRESHOLD = 100  # 停止判定的状态数量阈值
# 说明：如果轨迹的状态数量少于STOP_THRESHOLD，则认为无人机已停止
#       此阈值仅用于信息记录，不直接用于is_valid的计算

# ============================================================================


def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    将欧拉角(roll, pitch, yaw)转换为旋转矩阵
    角度单位: 度
    """
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)

    # 绕X轴旋转(roll)
    R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])

    # 绕Y轴旋转(pitch)
    R_y = np.array(
        [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]]
    )

    # 绕Z轴旋转(yaw)
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])

    # 组合旋转: R = R_z * R_y * R_x
    R = R_z @ R_y @ R_x
    return R


def check_backward_trajectory(qwenoft_file, test_jsons_file):
    """
    判定轨迹是否符合"后退"指令

    参数:
        qwenoft_file: qwenoft文件夹中的json文件路径
        test_jsons_file: test_jsons文件夹中的json文件路径

    返回:
        dict: 包含判定结果的字典
    """
    # 读取qwenoft文件(轨迹数据)
    with open(qwenoft_file, "r", encoding="utf-8") as f:
        trajectory_data = json.load(f)

    # 读取test_jsons文件(初始位置和目标位置)
    with open(test_jsons_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 提取数据
    _initial_pos = np.array(test_data["initial_pos"])  # [x, y, z, roll, pitch, yaw]
    _target_pos = np.array(test_data["target_pos"])  # [x, y, z, roll, pitch, yaw]
    instruction = test_data.get("instruction", "")
    instruction_unified = test_data.get("instruction_unified", "")

    # 提取轨迹位置(以第一帧为原点的坐标系)
    trajectory_points = []
    for item in trajectory_data:
        state = item["state"]
        # state[0]是位置 [x, y, z]
        pos = np.array(state[0])
        trajectory_points.append(pos)

    trajectory_points = np.array(trajectory_points)

    # 检查停止条件: state数量少于阈值
    num_states = len(trajectory_data)
    is_stopped = num_states < STOP_THRESHOLD

    # 轨迹点已经在无人机坐标系中(以第一帧为原点)
    # 在无人机坐标系中：
    # - x轴是无人机朝向(正方向),对应initial_pos的朝向
    # - y轴和z轴是垂直于x轴的平面内的方向
    # 检查轨迹点距离x轴的偏移量(在y-z平面内的距离,即 sqrt(y^2 + z^2))

    all_points_in_range = True
    max_x_distance = 0

    # 检查所有轨迹点距离x轴的偏移量
    for point in trajectory_points:
        # point已经在无人机坐标系中 [x, y, z]
        # x轴是[1, 0, 0]方向,计算点距离x轴的距离
        # 距离x轴的距离 = sqrt(y^2 + z^2)
        distance_from_x_axis = np.sqrt(point[1] ** 2 + point[2] ** 2)

        if distance_from_x_axis > max_x_distance:
            max_x_distance = distance_from_x_axis

        if distance_from_x_axis > MAX_DISTANCE_FROM_X_AXIS:
            all_points_in_range = False

    # 检查整体是否在后退: 最后一个点相较于第一个点,x坐标应该减小
    if len(trajectory_points) > 0:
        first_point = trajectory_points[0]
        last_point = trajectory_points[-1]
        # 如果最后一个点的x坐标小于第一个点的x坐标,说明在后退
        is_backward = last_point[0] < first_point[0]
    else:
        is_backward = False

    # 判定结果: 所有点在x轴左右100范围内,并且无人机在后退,并且停止
    is_valid = all_points_in_range and is_backward

    result = {
        "instruction": instruction,
        "instruction_unified": instruction_unified,
        "num_states": num_states,
        "is_stopped": is_stopped,
        "is_backward": is_backward,
        "all_points_in_range": all_points_in_range,
        "max_distance_from_x_axis": max_x_distance,
        "is_valid": is_valid,
    }

    return result
