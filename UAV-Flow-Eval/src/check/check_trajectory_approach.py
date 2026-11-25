# SPDX-License-Identifier: Apache-2.0
import json
import math
from pathlib import Path

import numpy as np

# ============================================================================
# 可调阈值配置（用于判定轨迹是否符合"靠近"指令）
# ============================================================================

# 判定区域相关阈值
SQUARE_SIZE = 800.0  # 判定区域的正方形边长（单位：米）
# 说明：无人机最终位置需要在以目标物边界为中心的指定区域内，该区域为正方形，边长为SQUARE_SIZE

Z_TOLERANCE = 50.0  # Z轴（高度）变化容忍范围（单位：米）
# 说明：无人机在靠近过程中，高度变化（相对于初始位置）不能超过Z_TOLERANCE，超过则判定失败

# 夹角判定阈值
ANGLE_THRESHOLD = 45.0  # 夹角阈值（单位：度）
# 说明：无人机最后位置与目标物中心点连线，与参考方向（前/左/右）的夹角必须小于ANGLE_THRESHOLD度

# 目标物图形尺寸（用于碰撞检测和距离计算）
TARGET_WIDTH = 400.0  # 长方形目标物的宽度（x方向，forward方向，单位：米）
# 说明：当use_obj==2（汽车）时，目标物为长方形，宽度为TARGET_WIDTH

TARGET_HEIGHT = 200.0  # 长方形目标物的高度（y方向，right方向，单位：米）
# 说明：当use_obj==2（汽车）时，目标物为长方形，高度为TARGET_HEIGHT

TARGET_RADIUS = 25.0  # 圆形目标物的半径（单位：米）
# 说明：当use_obj!=2（非汽车）时，目标物为圆形，半径为TARGET_RADIUS

# 理想位置距离目标物边缘的距离（单位：米）
IDEAL_DISTANCE_FROM_TARGET = 150.0  # 理想位置距离目标物边缘的距离
# 说明：根据instruction中的left/right，理想位置在目标物的左边或右边，距离目标物边缘IDEAL_DISTANCE_FROM_TARGET

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
    根据instruction分类靠近类型
    返回: 'left', 'right', 'forward'
    """
    instruction_lower = instruction.lower()
    if "left" in instruction_lower:
        return "left"
    elif "right" in instruction_lower:
        return "right"
    else:
        return "forward"


def point_to_rectangle_distance(point, center, width, height):
    """
    计算点到长方形的最小距离
    长方形以center为中心，宽度为width，高度为height（与坐标轴对齐）

    参数:
        point: 点坐标 [x, y]
        center: 长方形中心 [x, y]
        width: 宽度（x方向）
        height: 高度（y方向）

    返回:
        距离（如果点在长方形内返回0）
    """
    dx = abs(point[0] - center[0])
    dy = abs(point[1] - center[1])

    half_width = width / 2.0
    half_height = height / 2.0

    # 如果点在长方形内
    if dx <= half_width and dy <= half_height:
        return 0.0

    # 计算到长方形边界的距离
    if dx <= half_width:
        # 点在长方形的上下区域
        return dy - half_height
    elif dy <= half_height:
        # 点在长方形的左右区域
        return dx - half_width
    else:
        # 点在长方形的角区域
        return math.sqrt((dx - half_width) ** 2 + (dy - half_height) ** 2)


def point_to_circle_distance(point, center, radius):
    """
    计算点到圆形的最小距离

    参数:
        point: 点坐标 [x, y]
        center: 圆心 [x, y]
        radius: 半径

    返回:
        距离（如果点在圆内返回0）
    """
    dist = np.linalg.norm(point - center)
    if dist <= radius:
        return 0.0
    return dist - radius


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


def point_to_region_distance(point, x_min, x_max, y_min, y_max):
    """
    计算点到轴对齐矩形区域的最短距离
    区域由x,y的最小/最大值描述，若点在区域内则距离为0
    """
    clamped_x = min(max(point[0], x_min), x_max)
    clamped_y = min(max(point[1], y_min), y_max)
    dx = point[0] - clamped_x
    dy = point[1] - clamped_y
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0
    return math.sqrt(dx * dx + dy * dy)


def point_to_triangle_distance(point, triangle_vertices):
    """
    计算点到三角形的最短距离（如果在三角形内返回0）

    参数:
        point: 点坐标 [x, y]
        triangle_vertices: 三角形三个顶点 [[x1, y1], [x2, y2], [x3, y3]]

    返回:
        距离（如果在三角形内返回0）
    """

    # 使用重心法判断点是否在三角形内
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    def in_triangle(pt, v1, v2, v3):
        d1 = sign(pt, v1, v2)
        d2 = sign(pt, v2, v3)
        d3 = sign(pt, v3, v1)
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)

    # 检查点是否在三角形内
    v1 = np.array(triangle_vertices[0])
    v2 = np.array(triangle_vertices[1])
    v3 = np.array(triangle_vertices[2])
    p = np.array(point)

    if in_triangle(p, v1, v2, v3):
        return 0.0

    # 计算点到三角形三边的最短距离
    def point_to_line_distance(pt, line_start, line_end):
        line_vec = line_end - line_start
        pt_vec = pt - line_start
        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-10:
            return np.linalg.norm(pt - line_start)
        t = np.clip(np.dot(pt_vec, line_vec) / (line_len * line_len), 0.0, 1.0)
        closest = line_start + t * line_vec
        return np.linalg.norm(pt - closest)

    d1 = point_to_line_distance(p, v1, v2)
    d2 = point_to_line_distance(p, v2, v3)
    d3 = point_to_line_distance(p, v3, v1)

    return min(d1, d2, d3)


def check_approach_trajectory(
    qwenoft_file, test_jsons_file=None, combined_json_path=None, json_name=None, task_id=None
):
    """
    判定轨迹是否符合"靠近"指令

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
            raise ValueError("必须提供test_jsons_file或combined_json_path")

        with open(test_jsons_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        # 提取数据
        initial_pos = np.array(test_data["initial_pos"])  # [x, y, z, roll, pitch, yaw]
        target_pos = np.array(test_data["target_pos"])  # [x, y, z, roll, pitch, yaw]
        instruction = test_data.get("instruction", "")
        instruction_unified = test_data.get("instruction_unified", "")
        use_obj = test_data.get("use_obj", None)  # 获取use_obj字段

    # 分类靠近类型
    approach_type = classify_instruction(instruction)

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
            "approach_type": approach_type,
            "is_valid": False,
            "error": "No trajectory points",
        }

    # 获取最后一个轨迹点(无人机最终位置)
    last_point_drone = trajectory_points[-1]

    # 将最后一个点从无人机坐标系转换到世界坐标系
    # 参考closed_loop_runner.py中的world_to_local函数：
    # world_to_local: x_rel = dx * cos(-yaw) - dy * sin(-yaw)
    #                  y_rel = dx * sin(-yaw) + dy * cos(-yaw)
    # 其中 theta = -np.radians(yaw0)
    # 逆变换（local_to_world）：直接使用逆矩阵
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
    # 参考closed_loop_runner.py中的world_to_local函数和draw_2d_trajectory_from_log函数

    # 无人机最后一个位置在判定坐标系中（就是无人机坐标系）
    # trajectory_points已经是无人机坐标系，所以直接使用
    last_point_judge_2d = np.array(
        [last_point_drone[0], last_point_drone[1]]
    )  # [x_forward, y_right]

    # 将目标物从世界坐标系转换到无人机坐标系（使用world_to_local）
    # world_to_local: x_rel = dx * cos(-yaw) - dy * sin(-yaw)
    #                  y_rel = dx * sin(-yaw) + dy * cos(-yaw)
    x0, y0, z0 = initial_pos[0:3]
    yaw0 = initial_pos[4]
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

    # 判定是否在指定区域内(正方形)
    square_size = SQUARE_SIZE
    half_size = square_size / 2.0

    # Z轴检查: 相较于初始位置变化在容忍范围内
    # 初始位置z坐标
    _initial_z = initial_pos[2]
    z_change = abs(last_point_drone[2] - 0.0)  # 在无人机坐标系中，初始位置的z=0
    z_in_range = z_change <= Z_TOLERANCE

    # 计算无人机最后位置到目标物图形的距离（而不是点到点的距离）
    if target_shape == "rectangle":
        # 计算到长方形边界的距离
        dist_to_target = point_to_rectangle_distance(
            last_point_judge_2d, target_judge_2d, target_width, target_height
        )
    else:  # circle
        # 计算到圆形边界的距离
        dist_to_target = point_to_circle_distance(
            last_point_judge_2d, target_judge_2d, target_radius
        )

    # 计算无人机最后位置到目标物中心点的连线向量（只考虑x、y平面）
    vec_from_drone_to_target = target_judge_2d - last_point_judge_2d

    # 根据靠近类型判定区域（只考虑x、y平面）
    # 判定坐标系：x轴是forward（前向），y轴是right（右侧）
    # 初始位置在原点[0, 0]，目标物在[target_x, target_y]
    # 注意：判定区域需要考虑目标物的图形大小
    if target_shape == "rectangle":
        # 对于长方形，需要考虑长方形的边界
        half_width = target_width / 2.0
        half_height = target_height / 2.0
        # 长方形的边界
        target_x_min = target_judge_2d[0] - half_width
        _target_x_max = target_judge_2d[0] + half_width
        target_y_min = target_judge_2d[1] - half_height
        target_y_max = target_judge_2d[1] + half_height
    else:  # circle
        # 对于圆形，需要考虑圆的边界
        target_x_min = target_judge_2d[0] - target_radius
        _target_x_max = target_judge_2d[0] + target_radius
        target_y_min = target_judge_2d[1] - target_radius
        target_y_max = target_judge_2d[1] + target_radius

    if approach_type == "forward":
        # Forward: 区域在目标物的 X- 方向（下方）
        # vec_from_drone_to_target 指向 +X
        # vec_ref = [1, 0]
        # 区域: X在[target_x_min-square_size, target_x_min], Y在[target_y-half_size, target_y+half_size]
        region_x_min = target_x_min - square_size
        region_x_max = target_x_min
        region_y_min = target_judge_2d[1] - half_size
        region_y_max = target_judge_2d[1] + half_size
        xy_in_region = (
            region_x_min <= last_point_judge_2d[0] <= region_x_max
            and region_y_min <= last_point_judge_2d[1] <= region_y_max
        )
        corner_points = np.array(
            [
                [region_x_min, region_y_min],
                [region_x_max, region_y_min],
                [region_x_max, region_y_max],
                [region_x_min, region_y_max],
            ]
        )
        # 计算夹角: 与+X方向的夹角
        vec_ref = np.array([1.0, 0.0])
        angle = math.degrees(
            math.acos(
                np.clip(
                    np.dot(vec_from_drone_to_target, vec_ref)
                    / (np.linalg.norm(vec_from_drone_to_target) + 1e-10),
                    -1.0,
                    1.0,
                )
            )
        )

    elif approach_type == "left":
        # Left: 区域在目标物的 Y- 方向（左侧）
        # vec_from_drone_to_target 指向 +Y
        # vec_ref = [0, 1]
        # 区域: X在[target_x-half_size, target_x+half_size], Y在[target_y_min-square_size, target_y_min]
        region_x_min = target_judge_2d[0] - half_size
        region_x_max = target_judge_2d[0] + half_size
        region_y_min = target_y_min - square_size
        region_y_max = target_y_min
        xy_in_region = (
            region_x_min <= last_point_judge_2d[0] <= region_x_max
            and region_y_min <= last_point_judge_2d[1] <= region_y_max
        )
        corner_points = np.array(
            [
                [region_x_min, region_y_min],
                [region_x_max, region_y_min],
                [region_x_max, region_y_max],
                [region_x_min, region_y_max],
            ]
        )
        # 计算夹角: 与+Y方向的夹角
        vec_ref = np.array([0.0, 1.0])
        angle = math.degrees(
            math.acos(
                np.clip(
                    np.dot(vec_from_drone_to_target, vec_ref)
                    / (np.linalg.norm(vec_from_drone_to_target) + 1e-10),
                    -1.0,
                    1.0,
                )
            )
        )

    else:  # right
        # Right: 区域在目标物的 Y+ 方向（右侧）
        # vec_from_drone_to_target 指向 -Y
        # vec_ref = [0, -1]
        # 区域: X在[target_x-half_size, target_x+half_size], Y在[target_y_max, target_y_max+square_size]
        region_x_min = target_judge_2d[0] - half_size
        region_x_max = target_judge_2d[0] + half_size
        region_y_min = target_y_max
        region_y_max = target_y_max + square_size
        xy_in_region = (
            region_x_min <= last_point_judge_2d[0] <= region_x_max
            and region_y_min <= last_point_judge_2d[1] <= region_y_max
        )
        corner_points = np.array(
            [
                [region_x_min, region_y_min],
                [region_x_max, region_y_min],
                [region_x_max, region_y_max],
                [region_x_min, region_y_max],
            ]
        )
        # 计算夹角: 与-Y方向的夹角
        vec_ref = np.array([0.0, -1.0])
        angle = math.degrees(
            math.acos(
                np.clip(
                    np.dot(vec_from_drone_to_target, vec_ref)
                    / (np.linalg.norm(vec_from_drone_to_target) + 1e-10),
                    -1.0,
                    1.0,
                )
            )
        )

    # ========== 使用三角形区域作为判定区域（与 reward 一致）==========
    # 定义理想区域（直角三角形）：根据instruction中的left/right/forward确定
    # 注意坐标系：X轴是forward（前向，+X向前），Y轴是right（右侧，+Y向右）
    # 三角形以目标物中心为顶点，角度与ANGLE_THRESHOLD一致（45度），高为1000
    triangle_height = 1000.0  # 三角形高度
    # 直角三角形：底边的一半 = 高 * tan(angle)
    # 当 angle = 45度时，tan(45) = 1，所以底边的一半 = triangle_height
    triangle_base_half = triangle_height * math.tan(math.radians(ANGLE_THRESHOLD))

    if approach_type == "forward":
        # Forward: 三角形在 X- 方向（目标物下方）
        # 直角三角形的直角顶点在目标物
        # 底边平行于Y轴（local坐标的right方向）
        # v1: 目标物中心位置（直角顶点）
        v1 = np.array([target_judge_2d[0], target_judge_2d[1]])
        # v2: 底边左上角（x更小，y更大）
        v2 = np.array(
            [target_judge_2d[0] - triangle_height, target_judge_2d[1] - triangle_base_half]
        )
        # v3: 底边左下角（x更小，y更小）
        v3 = np.array(
            [target_judge_2d[0] - triangle_height, target_judge_2d[1] + triangle_base_half]
        )
        ideal_triangle = [v1.tolist(), v2.tolist(), v3.tolist()]
        ideal_triangle_np = [v1, v2, v3]

    elif approach_type == "left":
        # Left: 三角形在 Y- 方向（目标物左侧）
        # 直角三角形的直角顶点在目标物
        # 底边平行于X轴（local坐标的forward方向）
        # v1: 目标物中心位置（直角顶点）
        v1 = np.array([target_judge_2d[0], target_judge_2d[1]])
        # v2: 底边左上角（x更大，y更小）
        v2 = np.array(
            [target_judge_2d[0] - triangle_base_half, target_judge_2d[1] - triangle_height]
        )
        # v3: 底边左下角（x更小，y更小）
        v3 = np.array(
            [target_judge_2d[0] + triangle_base_half, target_judge_2d[1] - triangle_height]
        )
        ideal_triangle = [v1.tolist(), v2.tolist(), v3.tolist()]
        ideal_triangle_np = [v1, v2, v3]

    else:  # right
        # Right: 三角形在 Y+ 方向（目标物右侧）
        # 直角三角形的直角顶点在目标物
        # 底边平行于X轴（local坐标的forward方向）
        # v1: 目标物中心位置（直角顶点）
        v1 = np.array([target_judge_2d[0], target_judge_2d[1]])
        # v2: 底边右上角（x更大，y更大）
        v2 = np.array(
            [target_judge_2d[0] - triangle_base_half, target_judge_2d[1] + triangle_height]
        )
        # v3: 底边右下角（x更小，y更大）
        v3 = np.array(
            [target_judge_2d[0] + triangle_base_half, target_judge_2d[1] + triangle_height]
        )
        ideal_triangle = [v1.tolist(), v2.tolist(), v3.tolist()]
        ideal_triangle_np = [v1, v2, v3]

    # 计算无人机最终位置到理想三角形的距离
    dist_to_ideal = point_to_triangle_distance(last_point_judge_2d, ideal_triangle_np)

    # 使用三角形区域判断 xy_in_region（距离为0表示在三角形内）
    xy_in_region = dist_to_ideal == 0.0

    # 综合判定: XY平面在区域内（三角形），且Z轴变化在容忍范围内
    in_region = xy_in_region and z_in_range

    # 夹角判定: 夹角必须小于阈值
    angle_valid = angle < ANGLE_THRESHOLD

    # 最终判定: 需要在区域内且夹角满足要求，且不能与目标物碰撞
    # 如果轨迹与目标物图形相交（碰撞），视为失败
    is_valid = in_region and angle_valid and (not has_collision)

    # ========== Reward计算 ==========
    # Reward分为两部分：
    # 1. 靠近理想区域得分：1分基础分，每远离200个单位扣0.1分，最少0分
    # 2. 碰撞扣分：如果有碰撞扣分，否则不扣
    # 总分在0到1之间

    # 靠近理想区域得分：1分基础分
    # 离散分段扣分：每远离200个单位扣0.1分（每200为一个段）
    approach_score_base = 1.0
    if dist_to_ideal == 0:
        # 在理想区域内，得满分
        approach_score = approach_score_base
    else:
        # 离散分段扣分：每50为一个段，每段扣0.1分
        deduction_steps = dist_to_ideal / 50.0
        deduction_steps = math.ceil(deduction_steps)  # 向上取整
        approach_score = max(0.0, approach_score_base - deduction_steps * 0.1)

    # 碰撞扣分：扣0.3分
    collision_penalty = -0.3 if has_collision else 0.0

    # 总分 = 靠近得分 + 碰撞扣分
    reward = approach_score + collision_penalty
    # 确保分数在0到1之间
    reward = max(0.0, min(1.0, reward))

    # ========== Reward计算结束 ==========

    result = {
        "instruction": instruction,
        "instruction_unified": instruction_unified,
        "approach_type": approach_type,
        "use_obj": use_obj,
        "target_shape": target_shape,  # 'rectangle' 或 'circle'
        "last_point_world": last_point_world.tolist(),
        "last_point_judge_2d": last_point_judge_2d.tolist(),
        "target_judge_2d": target_judge_2d.tolist(),
        "initial_judge_2d": initial_judge_2d.tolist(),  # 无人机起点在判定坐标系中的坐标
        "region_corners": corner_points.tolist(),  # 判定区域四个角点坐标
        "dist_to_target": dist_to_target,  # 到目标物图形的距离
        "has_collision": has_collision,  # 是否与目标物碰撞
        "xy_in_region": xy_in_region,
        "z_in_range": z_in_range,
        "z_change": z_change,
        "in_region": in_region,
        "angle": angle,  # 夹角（度）
        "angle_valid": angle_valid,  # 夹角是否小于45度
        "is_valid": is_valid,
        # Reward相关字段
        "ideal_triangle": ideal_triangle,  # 理想区域三角形顶点
        "dist_to_ideal": dist_to_ideal,  # 到理想区域的距离
        "approach_score": approach_score,  # 靠近得分（1分基础）
        "collision_penalty": collision_penalty,  # 碰撞扣分
        "reward": reward,  # 总reward（0-1之间）
    }

    # 添加目标物图形信息
    if target_shape == "rectangle":
        result["target_shape_info"] = {"width": target_width, "height": target_height}
    else:  # circle
        result["target_shape_info"] = {"radius": target_radius}

    return result
