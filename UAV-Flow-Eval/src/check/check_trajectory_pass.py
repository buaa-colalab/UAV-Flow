# SPDX-License-Identifier: Apache-2.0
import json
import math
from pathlib import Path

import numpy as np

# ============================================================================
# 可调阈值配置（用于判定轨迹是否符合"穿过"指令）
# ============================================================================

# 轨迹经过区域判定阈值
REGION_LENGTH = 1000.0  # 判定区域长度（沿着主要方向，单位：厘米）
# 说明：用于判定轨迹是否经过目标物左侧/右侧区域的长度阈值
#       区域以目标物边界为中心，沿着绕过方向（left为Y轴负方向，right为Y轴正方向）延伸REGION_LENGTH

REGION_WIDTH = 200.0  # 判定区域宽度（垂直方向，单位：厘米）
# 说明：用于判定轨迹是否经过目标物左侧/右侧区域的宽度阈值
#       区域在垂直方向（X方向）的宽度为REGION_WIDTH

# 最后位置y坐标判定阈值
Y_FINAL_RANGE = 200.0  # y轴数值范围阈值（单位：厘米）
# 说明：判定最后位置y坐标是否在±Y_FINAL_RANGE范围内
#       如果最后位置y在[-Y_FINAL_RANGE, Y_FINAL_RANGE]内，则认为满足条件之一

Y_TARGET_RANGE = 300.0  # 目标点相关y坐标范围阈值（单位：厘米）
# 说明：判定最后位置y坐标是否在目标点相关范围内
#       left类型：最后位置y需在[target_y, Y_TARGET_RANGE]范围内
#       right类型：最后位置y需在[-Y_TARGET_RANGE, target_y]范围内
#       这是判定最后位置的另一个条件（满足任一条件即可）

# 目标物图形尺寸（用于碰撞检测和区域计算）
TARGET_WIDTH = 400.0  # 长方形目标物的宽度（x方向，forward方向，单位：厘米）
# 说明：当use_obj==2（汽车）时，目标物为长方形，宽度为TARGET_WIDTH

TARGET_HEIGHT = 200.0  # 长方形目标物的高度（y方向，right方向，单位：厘米）
# 说明：当use_obj==2（汽车）时，目标物为长方形，高度为TARGET_HEIGHT

TARGET_RADIUS = 25.0  # 圆形目标物的半径（单位：厘米）
# 说明：当use_obj!=2（非汽车）时，目标物为圆形，半径为TARGET_RADIUS

# 惩罚系数（用于reward计算）
PENALTY_COEFFICIENT = 8.0  # 当trajectory_passed_region为false时，min_y_diff的惩罚系数
# 说明：如果轨迹未经过目标侧方区域，会计算轨迹y轴方向和目标物位置的y轴值之差的最小值，
#       然后从reward中减去 min_y_diff * PENALTY_COEFFICIENT

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


def classify_instruction(instruction):
    """
    根据instruction分类绕过类型
    返回: 'left', 'right'
    """
    instruction_lower = instruction.lower()
    if "left" in instruction_lower:
        return "left"
    elif "right" in instruction_lower:
        return "right"
    else:
        # 如果没有明确指定，默认返回'right'
        return "right"


def segment_intersects_rectangle(p1, p2, center, width, height):
    """
    检查线段是否与长方形相交

    参数:
        p1: 线段起点 [x, y]
        p2: 线段终点 [x, y]
        center: 长方形中心 [x, y]
        width: 宽度（x方向）
        height: 高度（y方向）

    返回:
        True如果相交，False否则
    """
    half_width = width / 2.0
    half_height = height / 2.0

    # 长方形边界
    x_min = center[0] - half_width
    x_max = center[0] + half_width
    y_min = center[1] - half_height
    y_max = center[1] + half_height

    # 检查线段端点是否在长方形内
    if x_min <= p1[0] <= x_max and y_min <= p1[1] <= y_max:
        return True
    if x_min <= p2[0] <= x_max and y_min <= p2[1] <= y_max:
        return True

    # 使用分离轴定理检查线段与长方形是否相交
    # 检查线段是否与长方形的四条边相交
    def line_segment_intersect(seg1_p1, seg1_p2, seg2_p1, seg2_p2):
        """检查两条线段是否相交"""

        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return ccw(seg1_p1, seg2_p1, seg2_p2) != ccw(seg1_p2, seg2_p1, seg2_p2) and ccw(
            seg1_p1, seg1_p2, seg2_p1
        ) != ccw(seg1_p1, seg1_p2, seg2_p2)

    # 长方形的四条边
    corners = [
        np.array([x_min, y_min]),  # 左下
        np.array([x_max, y_min]),  # 右下
        np.array([x_max, y_max]),  # 右上
        np.array([x_min, y_max]),  # 左上
    ]

    # 检查线段是否与四条边相交
    for i in range(4):
        edge_start = corners[i]
        edge_end = corners[(i + 1) % 4]
        if line_segment_intersect(p1, p2, edge_start, edge_end):
            return True

    return False


def segment_intersects_circle(p1, p2, center, radius):
    """
    检查线段是否与圆形相交

    参数:
        p1: 线段起点 [x, y]
        p2: 线段终点 [x, y]
        center: 圆心 [x, y]
        radius: 半径

    返回:
        True如果相交，False否则
    """
    # 检查端点是否在圆内
    if np.linalg.norm(p1 - center) <= radius:
        return True
    if np.linalg.norm(p2 - center) <= radius:
        return True

    # 计算线段到圆心的最短距离
    # 向量化计算
    seg_vec = p2 - p1
    seg_len_sq = np.dot(seg_vec, seg_vec)

    if seg_len_sq < 1e-10:  # 线段长度为0
        dist = np.linalg.norm(p1 - center)
        return dist <= radius

    # 计算投影点
    t = np.dot(center - p1, seg_vec) / seg_len_sq
    t = np.clip(t, 0.0, 1.0)  # 限制在[0, 1]

    closest_point = p1 + t * seg_vec
    dist_to_center = np.linalg.norm(closest_point - center)

    return dist_to_center <= radius


def check_pass_trajectory(
    qwenoft_file, test_jsons_file=None, combined_json_path=None, json_name=None, task_id=None
):
    """
    判定轨迹是否符合"从左侧/右侧绕过xx"指令

    参数:
        qwenoft_file: qwenoft文件夹中的json文件路径(无人机坐标系轨迹数据)
        test_jsons_file: test_jsons文件夹中的json文件路径（可选，如果提供combined_json_path则不需要）
        combined_json_path: combined.json文件路径（可选，如果提供则从此文件读取数据）
        json_name: json文件名（用于在combined.json中查找对应的任务定义，可选）
        task_id: 任务ID（用于在combined.json中查找对应的任务定义，可选）

    返回:
        dict: 包含判定结果的字典
    """
    # 读取qwenoft文件(轨迹数据)
    with open(qwenoft_file, "r", encoding="utf-8") as f:
        trajectory_data = json.load(f)

    # 从combined.json或test_jsons文件读取数据
    if combined_json_path is not None:
        # 从combined.json读取数据
        combined_json_path = Path(combined_json_path)
        if not combined_json_path.exists():
            raise FileNotFoundError(f"combined.json文件不存在: {combined_json_path}")

        with open(combined_json_path, "r", encoding="utf-8") as f:
            combined_data = json.load(f)

        task_def = None

        # 优先按 task_id 查找
        if task_id is not None:
            for task in combined_data:
                if isinstance(task, dict) and task.get("task_id") == task_id:
                    task_def = task
                    break

        # 如果没有按 task_id 找到，按 json_name 查找
        if task_def is None and json_name is not None:
            for task in combined_data:
                if isinstance(task, dict) and task.get("json_name") == json_name:
                    task_def = task
                    break

        if task_def is None:
            error_msg = "在combined.json中找不到"
            if task_id is not None:
                error_msg += f"task_id为'{task_id}'"
            elif json_name is not None:
                error_msg += f"json_name为'{json_name}'"
            else:
                error_msg += "对应的任务定义（未提供task_id或json_name）"
            raise ValueError(error_msg)

        # 提取数据（兼容init_pose和initial_pos两种字段名）
        initial_pos_raw = task_def.get("initial_pos") or task_def.get("init_pose")
        if initial_pos_raw is None or len(initial_pos_raw) < 5:
            raise ValueError("任务定义中缺少有效的initial_pos或init_pose字段")
        initial_pos = np.array(initial_pos_raw)  # [x, y, z, roll, pitch, yaw]

        target_pos_raw = task_def.get("target_pos")
        if target_pos_raw is None or len(target_pos_raw) < 3:
            raise ValueError("任务定义中缺少有效的target_pos字段")
        target_pos = np.array(target_pos_raw)  # [x, y, z, roll, pitch, yaw]

        # 兼容prompt和instruction两种字段名
        instruction = task_def.get("instruction") or task_def.get("prompt") or ""
        instruction_unified = task_def.get("instruction_unified", "")
        use_obj = task_def.get("use_obj", None)  # 获取use_obj字段
    else:
        # 从test_jsons文件读取数据（保持向后兼容）
        if test_jsons_file is None:
            raise ValueError("必须提供test_jsons_file或combined_json_path+json_name")

        with open(test_jsons_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        # 提取数据
        initial_pos = np.array(test_data["initial_pos"])  # [x, y, z, roll, pitch, yaw]
        target_pos = np.array(test_data["target_pos"])  # [x, y, z, roll, pitch, yaw]
        instruction = test_data.get("instruction", "")
        instruction_unified = test_data.get("instruction_unified", "")
        use_obj = test_data.get("use_obj", None)  # 获取use_obj字段

    # 分类绕过类型
    pass_type = classify_instruction(instruction)

    # 确定目标物图形类型
    # use_obj == 2 表示汽车，使用长方形
    # 其他情况使用圆形
    is_car = use_obj == 2
    if is_car:
        target_shape = "rectangle"
        target_width = TARGET_WIDTH  # x方向（forward）
        target_height = TARGET_HEIGHT  # y方向（right）
    else:
        target_shape = "circle"
        target_radius = TARGET_RADIUS

    # 提取轨迹位置(无人机坐标系中,以第一帧为原点)
    trajectory_points = []
    for item in trajectory_data:
        state = item["state"]
        pos = np.array(state[0])  # [x, y, z] 在无人机坐标系中
        trajectory_points.append(pos)

    trajectory_points = np.array(trajectory_points)

    if len(trajectory_points) == 0:
        return {
            "instruction": instruction,
            "instruction_unified": instruction_unified,
            "pass_type": pass_type,
            "is_valid": False,
            "error": "No trajectory points",
        }

    # 获取最后一个轨迹点(无人机最终位置)
    last_point_drone = trajectory_points[-1]

    # 计算轨迹累计距离（基于前三维坐标）
    if len(trajectory_points) >= 2:
        diffs = np.diff(trajectory_points[:, :3], axis=0)
        path_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
    else:
        path_length = 0.0
    path_length_penalty = max(0.0, path_length - 3000.0)

    # 将最后一个点从无人机坐标系转换到世界坐标系
    # 参考closed_loop_runner.py中的world_to_local函数
    x0, y0, z0 = initial_pos[0:3]
    yaw0 = initial_pos[4]
    theta = np.radians(yaw0)  # 逆变换使用正yaw

    x_rel = last_point_drone[0]
    y_rel = last_point_drone[1]
    z_rel = last_point_drone[2]

    # 从无人机坐标系转换到世界坐标系（world_to_local的逆变换）
    x_world = x0 + x_rel * np.cos(theta) - y_rel * np.sin(theta)
    y_world = y0 + x_rel * np.sin(theta) + y_rel * np.cos(theta)
    z_world = z0 + z_rel

    last_point_world = np.array([x_world, y_world, z_world])

    # 建立二维判定坐标系（与closed_loop_runner.py中的2D图逻辑一致）
    # 使用无人机坐标系本身：以第一帧为原点，x轴是forward（前向），y轴是right（右侧）

    # 无人机最后一个位置在判定坐标系中（就是无人机坐标系）
    last_point_judge_2d = np.array(
        [last_point_drone[0], last_point_drone[1]]
    )  # [x_forward, y_right]

    # 将目标物从世界坐标系转换到无人机坐标系（使用world_to_local）
    dx = target_pos[0] - x0
    dy = target_pos[1] - y0
    _dz = target_pos[2] - z0
    theta = -np.radians(yaw0)  # 注意这里是负yaw
    target_x_judge = dx * np.cos(theta) - dy * np.sin(theta)
    target_y_judge = dx * np.sin(theta) + dy * np.cos(theta)
    target_judge_2d = np.array([target_x_judge, target_y_judge])  # [x_forward, y_right]

    # 初始位置在判定坐标系中是原点
    initial_judge_2d = np.array([0.0, 0.0])  # [x_forward, y_right]

    # 检查轨迹是否与目标物图形相交（碰撞检测）
    trajectory_2d_points = trajectory_points[:, :2]  # 只取x, y坐标
    has_collision = False

    if target_shape == "rectangle":
        # 检查轨迹是否与长方形相交
        for i in range(len(trajectory_2d_points) - 1):
            p1 = trajectory_2d_points[i]
            p2 = trajectory_2d_points[i + 1]
            if segment_intersects_rectangle(p1, p2, target_judge_2d, target_width, target_height):
                has_collision = True
                break
    else:  # circle
        # 检查轨迹是否与圆形相交
        for i in range(len(trajectory_2d_points) - 1):
            p1 = trajectory_2d_points[i]
            p2 = trajectory_2d_points[i + 1]
            if segment_intersects_circle(p1, p2, target_judge_2d, target_radius):
                has_collision = True
                break

    # 判定区域：长方形
    region_length = REGION_LENGTH  # 长（沿着主要方向）
    region_width = REGION_WIDTH  # 宽（垂直方向）
    _half_length = region_length / 2.0
    half_width = region_width / 2.0

    # 根据目标物图形类型计算边界
    if target_shape == "rectangle":
        # 对于长方形，需要考虑长方形的边界
        half_target_width = target_width / 2.0
        half_target_height = target_height / 2.0
        # 长方形的边界
        target_x_min = target_judge_2d[0] - half_target_width
        target_x_max = target_judge_2d[0] + half_target_width
        target_y_min = target_judge_2d[1] - half_target_height
        target_y_max = target_judge_2d[1] + half_target_height
    else:  # circle
        # 对于圆形，需要考虑圆的边界
        target_x_min = target_judge_2d[0] - target_radius
        target_x_max = target_judge_2d[0] + target_radius
        target_y_min = target_judge_2d[1] - target_radius
        target_y_max = target_judge_2d[1] + target_radius

    # 检查轨迹中是否有任何一点经过目标物的左侧/右侧区域
    trajectory_passed_region = False

    if pass_type == "left":
        # 从左侧绕过: 目标左侧区域（Y轴负方向）
        # 区域: 以目标物左侧边界为中心，Y方向在[target_Y_min-region_length, target_Y_min], X在[target-half_width, target+half_width]
        region_x_min = target_judge_2d[0] - half_width
        region_x_max = target_judge_2d[0] + half_width
        region_y_min = target_y_min - region_length
        region_y_max = target_y_min

        # 检查轨迹中是否有任何一点在区域内
        for point in trajectory_points:
            point_judge_2d = np.array([point[0], point[1]])  # [x_forward, y_right]
            if (
                region_x_min <= point_judge_2d[0] <= region_x_max
                and region_y_min <= point_judge_2d[1] <= region_y_max
            ):
                trajectory_passed_region = True
                break

        # 四个角点（按顺序：左下、右下、右上、左上）
        corner_points = np.array(
            [
                [region_x_min, region_y_min],  # 左下角
                [region_x_max, region_y_min],  # 右下角
                [region_x_max, region_y_max],  # 右上角
                [region_x_min, region_y_max],  # 左上角
            ]
        )

    else:  # right
        # 从右侧绕过: 目标右侧区域（Y轴正方向）
        # 区域: 以目标物右侧边界为中心，Y方向在[target_Y_max, target_Y_max+region_length], X在[target-half_width, target+half_width]
        region_x_min = target_judge_2d[0] - half_width
        region_x_max = target_judge_2d[0] + half_width
        region_y_min = target_y_max
        region_y_max = target_y_max + region_length

        # 检查轨迹中是否有任何一点在区域内
        for point in trajectory_points:
            point_judge_2d = np.array([point[0], point[1]])  # [x_forward, y_right]
            if (
                region_x_min <= point_judge_2d[0] <= region_x_max
                and region_y_min <= point_judge_2d[1] <= region_y_max
            ):
                trajectory_passed_region = True
                break

        # 四个角点（按顺序：左下、右下、右上、左上）
        corner_points = np.array(
            [
                [region_x_min, region_y_min],  # 左下角
                [region_x_max, region_y_min],  # 右下角
                [region_x_max, region_y_max],  # 右上角
                [region_x_min, region_y_max],  # 左上角
            ]
        )

    # 检查无人机最后停留位置：y轴数值在阈值范围内
    y_final_in_range = -Y_FINAL_RANGE <= last_point_judge_2d[1] <= Y_FINAL_RANGE

    # 检查新条件：根据绕过类型判断
    # left类型：最后位置y在[target_y, Y_TARGET_RANGE]之间
    # right类型：最后位置y在[-Y_TARGET_RANGE, target_y]之间
    target_y = target_judge_2d[1]
    y_final_value = last_point_judge_2d[1]

    if pass_type == "left":
        # left类型：最后位置y在[target_y, Y_TARGET_RANGE]之间
        y_final_in_target_range = target_y <= y_final_value <= Y_TARGET_RANGE
    else:  # right
        # right类型：最后位置y在[-Y_TARGET_RANGE, target_y]之间
        y_final_in_target_range = -Y_TARGET_RANGE <= y_final_value <= target_y

    # 最后位置y满足条件：在±Y_FINAL_RANGE范围内 或者 在目标点相关范围内
    y_final_valid = y_final_in_range or y_final_in_target_range

    # ================= 新的终点 Y 轴合法性判定 =================
    target_y = target_judge_2d[1]
    y_final_value = last_point_judge_2d[1]
    start_y = initial_judge_2d[1]  # 初始位置在判定坐标系中是 0.0

    # 1. 确定起点和target点的y值边界
    base_y_min = min(start_y, target_y)
    base_y_max = max(start_y, target_y)

    # 2. 向y正轴和负轴各扩展300个单位
    Y_EXTENSION = 300.0
    valid_y_min = base_y_min - Y_EXTENSION
    valid_y_max = base_y_max + Y_EXTENSION

    # 3. 判定最后位置是否在这个动态范围内
    y_final_valid = valid_y_min <= y_final_value <= valid_y_max

    # 为了不破坏原代码最后 return result 字典里的键值对，将原有的两个中间变量直接设为等同结果
    y_final_in_range = y_final_valid
    y_final_in_target_range = y_final_valid
    # ===========================================================

    # 综合判定: 轨迹经过区域 且 最后位置y满足条件 且 不能与目标物碰撞
    # 如果轨迹与目标物图形相交（碰撞），视为失败
    is_valid = trajectory_passed_region and y_final_valid and (not has_collision)
    # is_valid = trajectory_passed_region and (not has_collision)
    # 计算 reward：命中条件 +1000，再惩罚距离Y轴的距离
    # 距离y轴（无人机坐标系中 y 轴即 right 方向，距离为 |y|）
    distance_to_y_axis = abs(last_point_judge_2d[1])
    reward = 0.0
    if is_valid:
        reward += 10000.0
    reward -= distance_to_y_axis
    # 长路径惩罚：超过3000的部分按1:1扣减
    reward -= path_length_penalty
    # 如果轨迹未经过区域或发生碰撞，额外惩罚：计算轨迹和目标物的y轴方向的交点，距离目标物的y轴的差值
    min_y_diff_penalty = 0.0
    min_y_diff = None
    intersection_y_diff = None  # 轨迹与目标物在y轴方向交点的y坐标与目标物中心y坐标的差值

    if not trajectory_passed_region or has_collision:
        # 计算目标物在x方向上的范围
        if target_shape == "rectangle":
            half_target_width = target_width / 2.0
            target_x_min = target_judge_2d[0] - half_target_width
            target_x_max = target_judge_2d[0] + half_target_width
        else:  # circle
            target_x_min = target_judge_2d[0] - target_radius
            target_x_max = target_judge_2d[0] + target_radius

        target_center_y = target_judge_2d[1]  # 目标物中心的y坐标

        # 遍历所有轨迹线段，找到在x方向上与目标物重叠的线段
        min_y_diff = float("inf")
        for i in range(len(trajectory_2d_points) - 1):
            p1 = trajectory_2d_points[i]
            p2 = trajectory_2d_points[i + 1]

            # 计算线段在x方向上的范围
            seg_x_min = min(p1[0], p2[0])
            seg_x_max = max(p1[0], p2[0])

            # 检查线段是否在x方向上与目标物重叠
            if seg_x_max < target_x_min or seg_x_min > target_x_max:
                continue  # 线段在x方向上与目标物不重叠

            # 计算重叠部分
            overlap_x_min = max(seg_x_min, target_x_min)
            overlap_x_max = min(seg_x_max, target_x_max)

            # 如果线段不是垂直的，计算在重叠x范围内，线段对应的y坐标
            if abs(p2[0] - p1[0]) > 1e-10:  # 线段不是垂直的
                # 线段参数方程: y = p1[1] + (p2[1] - p1[1]) * (x - p1[0]) / (p2[0] - p1[0])
                # 计算重叠区域中点对应的y坐标
                overlap_x_center = (overlap_x_min + overlap_x_max) / 2.0
                t = (overlap_x_center - p1[0]) / (p2[0] - p1[0])
                intersection_y = p1[1] + (p2[1] - p1[1]) * t
            else:  # 线段是垂直的，使用p1的y坐标
                intersection_y = p1[1]

            # 计算交点的y坐标与目标物中心y坐标的差值
            y_diff = abs(intersection_y - target_center_y)
            if y_diff < min_y_diff:
                min_y_diff = y_diff
                intersection_y_diff = y_diff

        # 如果没有找到交点，使用原来的方法作为备选
        if min_y_diff == float("inf"):
            # 计算所有轨迹点的y坐标与目标物中心y坐标的差值的最小绝对值
            trajectory_y_coords = trajectory_points[:, 1]
            y_diffs = np.abs(trajectory_y_coords - target_center_y)
            min_y_diff = np.min(y_diffs)
            intersection_y_diff = min_y_diff

        min_y_diff_penalty = min_y_diff * PENALTY_COEFFICIENT
        reward -= min_y_diff_penalty
    ########################################################################################################
    # 新的reward计算：轨迹的y轴值越大，reward就越大（要考虑正负，不是绝对值）
    # 累加整条轨迹所有点的y坐标值
    # reward = 0.0
    # for point in trajectory_points:
    #     reward += point[1]  # 累加每个轨迹点的y坐标值
    ########################################################################################################
    ########################################################################################################
    reward = 0
    # 新reward逻辑：
    # 以起点(0,0)和目标点为对角顶点生成轴对齐矩形，
    # 以过目标点且平行y轴的直线镜像该矩形，再将矩形的x方向长度加倍。
    px, py = last_point_judge_2d
    start_x, start_y = 0.0, 0.0
    target_x, target_y = target_judge_2d

    base_x_min = min(start_x, target_x)
    base_x_max = max(start_x, target_x)
    base_y_min = min(start_y, target_y)
    base_y_max = max(start_y, target_y)

    mirror_axis_x = target_x
    mirror_x_min = 2 * mirror_axis_x - base_x_max
    mirror_x_max = 2 * mirror_axis_x - base_x_min
    mirror_y_min = base_y_min
    mirror_y_max = base_y_max

    center_x = (mirror_x_min + mirror_x_max) / 2.0
    half_len_x = (mirror_x_max - mirror_x_min) / 2.0
    half_len_x *= 2.0  # 长度加倍
    rect_x_min = center_x - half_len_x
    rect_x_max = center_x + half_len_x
    rect_y_min = mirror_y_min
    rect_y_max = mirror_y_max

    in_reward_rect = (rect_x_min <= px <= rect_x_max) and (rect_y_min <= py <= rect_y_max)
    dx_out = max(rect_x_min - px, 0.0, px - rect_x_max)
    dy_out = max(rect_y_min - py, 0.0, py - rect_y_max)
    dist_to_rect = math.hypot(dx_out, dy_out)

    # 计算distance_penalty：距离每500取上整扣20分
    distance_penalty = math.ceil(dist_to_rect / 500.0) * 20.0

    # 第一部分（50%）：根据距离计算位置reward，最高0.5分
    # 在矩形内：0.5分
    # 每远离200个单位长度，扣0.1分（因为最高是0.5，所以扣分也是原来的一半）
    # 最低为0分
    if in_reward_rect:
        position_reward = 0.5
    else:
        # 每200个单位为一个扣分段（向上取整）
        penalty_steps = math.ceil(dist_to_rect / 200.0)
        position_reward = max(0.0, 0.5 - penalty_steps * 0.1)

    # 第二部分（50%）：根据轨迹是否经过目标物对应一侧计算reward
    # 经过区域得分：0.5分基础分
    # 如果轨迹经过对应一侧：得0.5分
    # 如果有碰撞：扣0.3分
    # 否则：0分
    if trajectory_passed_region:
        pass_reward = 0.5
    else:
        pass_reward = 0.0

    # 碰撞扣分：扣0.3分
    collision_penalty = -0.3 if has_collision else 0.0

    # 总reward = 位置reward + 经过区域reward + 碰撞扣分
    reward = position_reward + pass_reward + collision_penalty
    # 确保reward在[0, 1]范围内
    reward = max(0.0, min(1.0, reward))
    #############################################################################
    # 兼容prompt和instruction两种字段名
    prompt = (
        task_def.get("prompt", "")
        if combined_json_path is not None
        else test_data.get("prompt", "")
    )
    result = {
        "instruction": instruction,
        "instruction_unified": instruction_unified,
        "prompt": prompt,
        "pass_type": pass_type,
        "use_obj": use_obj,
        "target_shape": target_shape,  # 'rectangle' 或 'circle'
        "path_length": path_length,  # 轨迹累计距离
        "path_length_penalty": path_length_penalty,  # 路径惩罚
        "reward_rectangle": {
            "x_min": rect_x_min,
            "x_max": rect_x_max,
            "y_min": rect_y_min,
            "y_max": rect_y_max,
        },
        "distance_to_reward_rectangle": dist_to_rect,
        "distance_penalty": distance_penalty,
        "last_point_world": last_point_world.tolist(),
        "last_point_judge_2d": last_point_judge_2d.tolist(),
        "target_judge_2d": target_judge_2d.tolist(),
        "initial_judge_2d": initial_judge_2d.tolist(),
        "region_corners": corner_points.tolist(),  # 判定区域四个角点坐标
        "has_collision": has_collision,  # 是否与目标物碰撞
        "trajectory_passed_region": trajectory_passed_region,  # 轨迹是否经过区域
        "y_final_in_range": y_final_in_range,  # 最后位置y是否在±Y_FINAL_RANGE范围内
        "y_final_in_target_range": y_final_in_target_range,  # 最后位置y是否在目标点相关范围内
        "y_final_valid": y_final_valid,  # 最后位置y是否满足条件（两个条件之一）
        "y_final": last_point_judge_2d[1],  # 最后位置的y坐标
        "target_y": target_y,  # 目标点的y坐标
        "distance_to_y_axis": distance_to_y_axis,  # 终点到y轴的距离
        "intersection_y_diff": intersection_y_diff,  # 轨迹与目标物在y轴方向交点的y坐标与目标物中心y坐标的差值（当trajectory_passed_region为false或has_collision为true时）
        "min_y_diff": min_y_diff,  # 轨迹与目标物在y轴方向交点的y坐标与目标物中心y坐标的差值（当trajectory_passed_region为false或has_collision为true时）
        "min_y_diff_penalty": min_y_diff_penalty,  # 基于min_y_diff的惩罚值（min_y_diff * PENALTY_COEFFICIENT）
        "position_reward": position_reward,  # 位置reward（50%，基于距离理想位置的距离）
        "pass_reward": pass_reward,  # 经过区域reward（50%，基于轨迹是否经过目标物对应一侧）
        "collision_penalty": collision_penalty,  # 碰撞扣分（有碰撞扣0.3）
        "reward": reward,  # 总打分（position_reward + pass_reward + collision_penalty，范围[0, 1]）
        "is_valid": is_valid,  # 综合判定结果
    }

    # 添加目标物图形信息
    if target_shape == "rectangle":
        result["target_shape_info"] = {"width": target_width, "height": target_height}
    else:  # circle
        result["target_shape_info"] = {"radius": target_radius}

    return result
