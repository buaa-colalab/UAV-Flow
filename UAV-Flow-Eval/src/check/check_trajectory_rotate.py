# SPDX-License-Identifier: Apache-2.0
import json
import math
import re
from pathlib import Path

import numpy as np

# ============================================================================
# 可调阈值配置（用于判定轨迹是否符合"旋转"指令）
# ============================================================================

# 旋转角度判定阈值
ROTATION_THRESHOLD = 10.0  # 旋转角度差值阈值（单位：度）
# 说明：判定旋转是否成功的主要指标
#       计算方式：从指令中提取带方向的期望旋转度数，与轨迹第5维yaw的有符号旋转量进行比较
#       如果角度差值（绝对值）小于ROTATION_THRESHOLD度，则认为旋转角度满足要求
#       角度差值越小，说明实际旋转越接近期望旋转

# 位置距离判定阈值
DISTANCE_THRESHOLD = 30.0  # 位置距离阈值（单位：与坐标单位相同，通常为厘米）
# 说明：判定旋转过程中位置是否保持稳定的指标
#       计算方式：计算第一个轨迹点和最后一个轨迹点之间的欧氏距离
#       如果位置距离小于DISTANCE_THRESHOLD，则认为位置保持稳定（旋转过程中没有明显移动）
#       位置距离越小，说明旋转过程中位置变化越小

# ============================================================================


def extract_rotation_degree_from_instruction(instruction):
    """
    从指令文本中提取旋转度数

    参数:
        instruction: 指令文本，例如 "Rotate 105 degrees to the right"

    返回:
        float: 提取到的度数，如果提取失败返回 None
    """
    # 使用正则表达式提取数字
    # 匹配模式：数字（可能包含小数点）后面跟着 "degree" 或 "度"
    pattern = r"(\d+\.?\d*)\s*(?:degrees?|度)"
    match = re.search(pattern, instruction, re.IGNORECASE)

    if match:
        return float(match.group(1))

    # 如果上面的模式没匹配到，尝试匹配任何数字
    # 但优先匹配大于10的数字（因为旋转角度通常不会小于10度）
    pattern_all = r"(\d+\.?\d*)"
    matches = re.findall(pattern_all, instruction)

    if matches:
        # 尝试找到最可能的旋转角度（通常旋转角度在10-360度之间）
        for num_str in matches:
            num = float(num_str)
            if 10 <= num <= 360:
                return num
        # 如果没找到合适的，返回第一个数字
        return float(matches[0])

    return None


def extract_rotation_direction_from_instruction(instruction):
    """从旋转指令中提取方向；右转为正，左转为负。"""
    instruction_lower = instruction.lower()
    left_markers = (
        "left",
        "counterclockwise",
        "counter-clockwise",
        "anticlockwise",
        "anti-clockwise",
        "左",
        "逆时针",
    )
    right_markers = ("right", "clockwise", "顺时针", "右")

    # counterclockwise/anticlockwise 都包含 clockwise，必须先判断左转标记。
    if any(marker in instruction_lower for marker in left_markers):
        return -1.0
    if any(marker in instruction_lower for marker in right_markers):
        return 1.0
    # 无方向文本时保持向后兼容，按正方向处理。
    return 1.0


def normalize_angle(angle):
    """
    将角度归一化到 [-180, 180] 范围

    参数:
        angle: 角度（度）

    返回:
        float: 归一化后的角度
    """
    original_angle = float(angle)
    normalized = (original_angle + 180.0) % 360.0 - 180.0
    # 180度的方向仅凭最终朝向无法区分；这里保留输入差值本身的符号。
    if math.isclose(normalized, -180.0, abs_tol=1e-9) and original_angle > 0:
        return 180.0
    return normalized


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


def compute_cumulative_yaw_rotation(yaw_angles):
    """累计相邻帧的有符号yaw变化，避免跨越±180度时产生跳变。"""
    if len(yaw_angles) < 2:
        return 0.0

    return float(
        sum(
            compute_rotation_angle_diff(yaw_angles[index - 1], yaw_angles[index])
            for index in range(1, len(yaw_angles))
        )
    )


def check_rotate_trajectory(
    qwenoft_file,
    test_jsons_file=None,
    combined_json_path=None,
    json_name=None,
    task_id=None,
    rotation_threshold=None,
    distance_threshold=None,
):
    """
    判定轨迹是否符合"旋转"指令

    参数:
        qwenoft_file: qwenoft文件夹中的json文件路径(无人机坐标系轨迹数据)
        test_jsons_file: test_jsons文件夹中的json文件路径（可选，如果提供combined_json_path则不需要）
        combined_json_path: combined.json文件路径（可选，如果提供则从此文件读取数据）
        json_name: json文件名（用于在combined.json中查找对应的任务定义，可选）
        task_id: 任务ID（用于在combined.json中查找对应的任务定义，可选）
        rotation_threshold: 旋转角度差值阈值（可选，默认使用文件顶部定义的ROTATION_THRESHOLD）
        distance_threshold: 位置距离阈值（可选，默认使用文件顶部定义的DISTANCE_THRESHOLD）

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

    # 从combined.json或test_jsons文件读取数据
    instruction = ""
    instruction_unified = ""

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
        _initial_pos = np.array(initial_pos_raw)  # [x, y, z, roll, yaw, pitch]

        # 兼容prompt和instruction两种字段名
        instruction = task_def.get("instruction") or task_def.get("prompt") or ""
        instruction_unified = task_def.get("instruction_unified", "")

    elif test_jsons_file is not None:
        # 从test_jsons文件读取数据（保持向后兼容）
        with open(test_jsons_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        # 提取数据
        _initial_pos = np.array(test_data["initial_pos"])  # [x, y, z, roll, yaw, pitch]
        instruction = test_data.get("instruction", "")
        instruction_unified = test_data.get("instruction_unified", "")
    else:
        raise ValueError("必须提供test_jsons_file或combined_json_path")

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
    # state[1] 是姿态 [roll, yaw, pitch]，对应扁平轨迹的第4/5/6维
    first_pos = np.array(first_state[0])  # [x, y, z]
    first_attitude = np.array(first_state[1])  # [roll, yaw, pitch]

    last_pos = np.array(last_state[0])  # [x, y, z]
    last_attitude = np.array(last_state[1])  # [roll, yaw, pitch]

    # yaw固定为姿态中的第2个元素，即扁平轨迹的第5维。
    initial_roll = first_attitude[0]
    initial_yaw = first_attitude[1]
    initial_pitch = first_attitude[2]
    final_roll = last_attitude[0]
    final_yaw = last_attitude[1]
    final_pitch = last_attitude[2]
    # /rewards 接收的是相对初始机体坐标系的轨迹；reset 时的相对 yaw 固定为 0°。
    # 轨迹本身通常从第一条预测 action 开始，并不包含 reset 这一帧。
    rotation_start_yaw = 0.0
    initial_angle = rotation_start_yaw
    final_angle = final_yaw
    angle_type = "yaw"

    # 对 reset(0°) 到每个预测 state 的相邻yaw差进行有符号累计。
    # 对于不跨±180度的轨迹，这等价于 0° 到末帧的旋转；跨边界时
    # 仍按最短有符号增量展开，不会产生 ±360° 的表示跳变。
    yaw_angles = [
        rotation_start_yaw,
        *[float(state_entry["state"][1][1]) for state_entry in trajectory_data],
    ]
    actual_rotation = compute_cumulative_yaw_rotation(yaw_angles)

    # 从指令中提取期望旋转度数和方向：右转为正，左转为负。
    expected_rotation_magnitude = extract_rotation_degree_from_instruction(instruction)

    if expected_rotation_magnitude is None:
        return {
            "instruction": instruction,
            "instruction_unified": instruction_unified,
            "is_valid": False,
            "error": "Cannot extract rotation degree from instruction",
        }

    expected_rotation = expected_rotation_magnitude * extract_rotation_direction_from_instruction(
        instruction
    )

    # 计算旋转角度差值（取绝对值）
    rotation_diff = abs(actual_rotation - expected_rotation)

    # 计算最后一个位置点和初始位置点的距离
    position_distance = np.linalg.norm(last_pos - first_pos)

    # 判定条件1：旋转角度差值小于阈值
    rotation_valid = rotation_diff < rotation_threshold

    # 判定条件2：位置距离小于阈值
    distance_valid = position_distance < distance_threshold

    # 两个条件都满足才算成功
    is_valid = rotation_valid and distance_valid

    # ========== 新增Reward计算 ==========
    # Reward由两部分组成：
    # 1. 停止位置得分：0.3分，100以内得0.3分，每远离100扣0.1，最少0分
    # 2. 旋转角度得分：10度以内0.7分；10-40度每10度扣0.1；40-100度每20度扣0.1；最少0分
    # 从指令中提取期望旋转度数进行比较

    # 计算停止位置得分
    if position_distance <= 100:
        position_score = 0.3
    else:
        # 每远离100扣0.1分
        deduction_steps = math.ceil((position_distance - 100) / 100.0)
        position_score = max(0.0, 0.3 - deduction_steps * 0.1)

    # 计算旋转角度得分
    # 使用指令中的标准角度
    angle_diff_for_reward = rotation_diff  # 默认使用与期望角度的差值

    if angle_diff_for_reward <= 10:
        rotation_score = 0.7
    elif angle_diff_for_reward <= 20:
        rotation_score = 0.6
    elif angle_diff_for_reward <= 30:
        rotation_score = 0.5
    elif angle_diff_for_reward <= 40:
        rotation_score = 0.4
    elif angle_diff_for_reward <= 60:
        rotation_score = 0.3
    elif angle_diff_for_reward <= 80:
        rotation_score = 0.2
    elif angle_diff_for_reward <= 100:
        rotation_score = 0.1
    else:
        rotation_score = 0.0

    # 总reward
    reward = position_score + rotation_score
    # 确保reward在[0, 1]范围内
    reward = max(0.0, min(1.0, reward))

    # ========== Reward计算结束 ==========

    # 兼容prompt和instruction两种字段名
    prompt = task_def.get("prompt", "") if combined_json_path is not None else ""
    result = {
        "instruction": instruction,
        "instruction_unified": instruction_unified,
        "prompt": prompt,
        "is_valid": is_valid,
        "rotation_valid": rotation_valid,
        "distance_valid": distance_valid,
        "angle_type": angle_type,
        "initial_angle": initial_angle,
        "rotation_start_yaw": rotation_start_yaw,
        "final_angle": final_angle,
        "initial_roll": initial_roll,
        "initial_pitch": initial_pitch,
        "initial_yaw": initial_yaw,
        "final_roll": final_roll,
        "final_pitch": final_pitch,
        "final_yaw": final_yaw,
        "actual_rotation": actual_rotation,
        "expected_rotation": expected_rotation,
        "rotation_diff": rotation_diff,
        "rotation_threshold": rotation_threshold,
        "position_distance": position_distance,
        "distance_threshold": distance_threshold,
        "first_pos": first_pos.tolist(),
        "last_pos": last_pos.tolist(),
        "num_points": len(trajectory_data),
        # 新增Reward相关字段
        "position_score": position_score,  # 停止位置得分（0.3分基础）
        "rotation_score": rotation_score,  # 旋转角度得分（0.7分基础）
        "reward": reward,  # 总reward（0-1之间）
    }

    return result
