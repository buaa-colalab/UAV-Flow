# SPDX-License-Identifier: Apache-2.0
"""
批量检查轨迹文件，判断任务是否完成
根据 classified_instr.json 中的分类，调用对应的检查函数
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# 导入检查函数
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from check.check_trajectory_approach import check_approach_trajectory
from check.check_trajectory_back import check_backward_trajectory
from check.check_trajectory_land import check_land_trajectory
from check.check_trajectory_orbit import (
    MIN_DIRECTED_ANGLE_DEG,
    check_orbit_trajectory,
)
from check.check_trajectory_pass import check_pass_trajectory
from check.check_trajectory_rotate import check_rotate_trajectory
from check.check_trajectory_translate import check_translate_trajectory
from check.check_trajectory_translate_move_to_object import check_move_to_object_distance_trajectory
from check.check_trajectory_translate_up_and_down import check_ascent_descent_trajectory
from check.check_trajectory_turn_to_object import check_turn_to_object_trajectory


def convert_to_serializable(obj: Any) -> Any:
    """
    将对象转换为可 JSON 序列化的类型
    处理 numpy 类型、Path 对象等
    """
    # 处理 numpy 整数类型
    if isinstance(
        obj,
        (
            np.integer,
            np.intc,
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    ):
        return int(obj)
    # 处理 numpy 浮点数类型（兼容 NumPy 2.0）
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    # 处理 numpy 布尔类型
    elif isinstance(obj, np.bool_):
        return bool(obj)
    # 处理 numpy 数组
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # 处理 Path 对象
    elif isinstance(obj, Path):
        return str(obj)
    # 递归处理字典
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    # 递归处理列表和元组
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


# 类别到检查函数的映射
CATEGORY_TO_FUNCTION = {
    "Turn": check_turn_to_object_trajectory,
    "Move": check_move_to_object_distance_trajectory,
    "Shift": check_translate_trajectory,
    "Rotate": check_rotate_trajectory,
    "Surround": check_orbit_trajectory,
    "Ascend/Descend": check_ascent_descent_trajectory,
    "Approach": check_approach_trajectory,
    "Retreat": check_backward_trajectory,
    "Pass": check_pass_trajectory,
    "Land": check_land_trajectory,
}


def find_category(filename: str, classified_instr: Dict) -> Optional[str]:
    """
    根据文件名在 classified_instr.json 中查找分类

    参数:
        filename: json文件名（如 "2025-05-12_21-50-33.json"）
        classified_instr: classified_instr.json 的内容

    返回:
        类别名称，如果找不到返回 None
    """
    for category, files in classified_instr.items():
        if filename in files:
            return category
    return None


def check_single_trajectory(
    trajectory_file: Path,
    test_jsons_file: Path,
    category: str,
    classified_instr_path: Optional[Path] = None,
) -> Dict:
    """
    检查单个轨迹文件

    参数:
        trajectory_file: 轨迹文件路径（qwenoft文件夹中的json文件）
        test_jsons_file: 测试数据文件路径（test_jsons文件夹中的json文件）
        category: 类别名称
        classified_instr_path: classified_instr.json 的路径（可选，当前未使用）

    返回:
        包含检查结果的字典
    """
    if category not in CATEGORY_TO_FUNCTION:
        return {
            "filename": trajectory_file.name,
            "category": category,
            "is_valid": False,
            "error": f"Unknown category: {category}",
        }

    check_function = CATEGORY_TO_FUNCTION[category]

    try:
        # 根据类别调用不同的检查函数
        if category == "Pass":
            # Pass 类别可以使用 test_jsons_file
            result = check_function(str(trajectory_file), test_jsons_file=str(test_jsons_file))
        elif category == "Surround":
            result = check_function(
                str(trajectory_file),
                str(test_jsons_file),
                min_directed_angle_deg=MIN_DIRECTED_ANGLE_DEG,
            )
        elif category == "Approach":
            # Approach 类别可以使用 test_jsons_file 或 combined_json_path
            # 优先使用 test_jsons_file
            result = check_function(str(trajectory_file), test_jsons_file=str(test_jsons_file))
        else:
            # 其他类别直接使用 test_jsons_file
            result = check_function(str(trajectory_file), str(test_jsons_file))

        # 确保结果包含文件名和类别
        result["filename"] = trajectory_file.name
        result["category"] = category

        return result

    except Exception as e:
        return {
            "filename": trajectory_file.name,
            "category": category,
            "is_valid": False,
            "error": f"Exception during check: {str(e)}",
        }


def check_all_trajectories(
    trajectory_dir: Path,
    test_jsons_dir: Path,
    classified_instr_path: Path,
    output_file: Optional[Path] = None,
    expected_filenames=None,
) -> Dict:
    """
    批量检查所有轨迹文件

    参数:
        trajectory_dir: 轨迹文件目录（qwenoft_grounding_gt_bbox_full）
        test_jsons_dir: 测试数据目录（test_jsons）
        classified_instr_path: classified_instr.json 文件路径
        output_file: 输出结果文件路径（可选）

    返回:
        包含所有检查结果的字典
    """
    # 读取分类信息
    with open(classified_instr_path, "r", encoding="utf-8") as f:
        classified_instr = json.load(f)

    # 获取所有轨迹文件
    # The reference/manifest defines the denominator, never arbitrary output JSON.
    expected = sorted(
        expected_filenames
        if expected_filenames is not None
        else (p.name for p in test_jsons_dir.glob("*.json"))
    )
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("Expected task filenames must be nonempty and unique")
    if any(Path(name).name != name for name in expected):
        raise ValueError("Task names must be plain filenames")
    trajectory_files = [trajectory_dir / name for name in expected]

    results = {
        "total": len(trajectory_files),
        "valid": 0,
        "invalid": 0,
        "errors": 0,
        "success_rate": 0.0,  # 总成功率
        "by_category": {},
        "details": {},  # 按类别组织的详细信息
    }

    print(f"开始检查 {len(trajectory_files)} 个轨迹文件...")

    for i, trajectory_file in enumerate(trajectory_files, 1):
        filename = trajectory_file.name
        print(f"[{i}/{len(trajectory_files)}] 检查: {filename}")

        if not trajectory_file.is_file():
            results["errors"] += 1
            results["details"].setdefault("Missing", []).append(
                {"filename": filename, "is_valid": False, "error": "Expected trajectory is missing"}
            )
            continue
        # 查找类别
        category = find_category(filename, classified_instr)
        if category is None:
            result = {
                "filename": filename,
                "category": None,
                "is_valid": False,
                "error": "Category not found in classified_instr.json",
            }
            results["errors"] += 1
            # 将未分类的结果放入 'Unknown' 类别
            if "Unknown" not in results["details"]:
                results["details"]["Unknown"] = []
            results["details"]["Unknown"].append(result)
            continue

        # 查找对应的测试数据文件
        test_jsons_file = test_jsons_dir / filename
        if not test_jsons_file.exists():
            result = {
                "filename": filename,
                "category": category,
                "is_valid": False,
                "error": f"Test data file not found: {test_jsons_file}",
            }
            results["errors"] += 1
            # 按类别组织结果
            if category not in results["details"]:
                results["details"][category] = []
            results["details"][category].append(result)
            continue

        # 检查轨迹
        result = check_single_trajectory(
            trajectory_file, test_jsons_file, category, classified_instr_path
        )

        # 统计结果
        if result.get("is_valid", False):
            results["valid"] += 1
        elif "error" in result:
            results["errors"] += 1
        else:
            results["invalid"] += 1

        # 按类别统计
        if category not in results["by_category"]:
            results["by_category"][category] = {
                "total": 0,
                "valid": 0,
                "invalid": 0,
                "errors": 0,
                "success_rate": 0.0,  # 类别成功率
            }

        results["by_category"][category]["total"] += 1
        if result.get("is_valid", False):
            results["by_category"][category]["valid"] += 1
        elif "error" in result:
            results["by_category"][category]["errors"] += 1
        else:
            results["by_category"][category]["invalid"] += 1

        # 更新类别成功率（四舍五入保留一位小数）
        if results["by_category"][category]["total"] > 0:
            results["by_category"][category]["success_rate"] = round(
                results["by_category"][category]["valid"]
                / results["by_category"][category]["total"]
                * 100.0,
                1,
            )

        # 按类别组织结果
        if category not in results["details"]:
            results["details"][category] = []
        results["details"][category].append(result)

        # 打印进度
        status = "✓" if result.get("is_valid", False) else "✗"
        print(f"  {status} {category}: {result.get('is_valid', False)}")
        if "error" in result:
            print(f"    错误: {result['error']}")

    # 计算总成功率（四舍五入保留一位小数）
    if results["total"] > 0:
        results["success_rate"] = round(results["valid"] / results["total"] * 100.0, 1)

    # 打印汇总
    print("\n" + "=" * 60)
    print("检查结果汇总")
    print("=" * 60)
    print(f"总计: {results['total']}")
    print(f"通过: {results['valid']} ({results['success_rate']:.1f}%)")
    print(f"失败: {results['invalid']} ({results['invalid'] / results['total'] * 100:.1f}%)")
    print(f"错误: {results['errors']} ({results['errors'] / results['total'] * 100:.1f}%)")
    print("\n按类别统计:")
    for category, stats in results["by_category"].items():
        total = stats["total"]
        valid = stats["valid"]
        invalid = stats["invalid"]
        errors = stats["errors"]
        print(f"  {category}:")
        print(f"    总计: {total}")
        print(f"    通过: {valid} ({stats['success_rate']:.1f}%)")
        print(f"    失败: {invalid} ({invalid / total * 100:.1f}%)")
        print(f"    错误: {errors} ({errors / total * 100:.1f}%)")

    # 保存结果（先转换为可序列化的类型）
    if output_file:
        # 转换所有 numpy 类型为 Python 原生类型
        serializable_results = convert_to_serializable(results)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_file}")

    compact_categories = ("Pass", "Land", "Rotate", "Approach")
    compact_summary = []
    for category in compact_categories:
        stats = results["by_category"].get(category, {})
        compact_summary.append(
            f"{category.lower()} {stats.get('valid', 0)}/{stats.get('total', 0)}"
        )

    print(f"\n成功率{results['valid']}/{results['total']} ({results['success_rate']:.1f}%)")
    print("，".join(compact_summary))

    return results
