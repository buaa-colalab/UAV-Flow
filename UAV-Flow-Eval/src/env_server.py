# SPDX-License-Identifier: Apache-2.0
import argparse
import base64
import copy
import hashlib
import json
import logging
import os
import signal
import sys
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from uav_flow.runtime import environment, make_output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="UAV环境服务器")
    parser.add_argument(
        "-e", "--env_id", default="UnrealTrack-DowntownWest-ContinuousColor-v0", help="环境ID"
    )
    parser.add_argument("-t", "--time_dilation", default=10, type=int, help="时间膨胀参数")
    parser.add_argument("-s", "--seed", default=0, type=int, help="随机种子")
    parser.add_argument("-p", "--port", default=5008, type=int, help="HTTP服务器端口")
    parser.add_argument("--env_bin", help="Linux/Windows Unreal executable override")
    parser.add_argument(
        "--offscreen", action="store_true", help="Enable GPU rendering without a display"
    )
    parser.add_argument("--gpu_id", type=int, help="Vulkan graphics adapter index")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind address")
    parser.add_argument(
        "--output", type=Path, required=True, help="Single instance output directory"
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="日志级别",
    )

    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    if args.env_bin:
        args.env_bin = str(Path(args.env_bin).resolve())
    return args


if __name__ == "__main__":
    _cli_args = parse_args()
    _cli_args.output = _cli_args.output.resolve()
    make_output(_cli_args.output)
    os.environ.update(environment(_cli_args.output))
    os.chdir(_cli_args.output / "tmp")

import cv2
import gym
import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

import gym_unrealcv  # noqa: F401 - registration and public simulator module identity
from check.check_trajectory_approach import check_approach_trajectory
from check.check_trajectory_land import check_land_trajectory
from check.check_trajectory_pass import check_pass_trajectory
from check.check_trajectory_rotate import check_rotate_trajectory
from check.check_trajectory_translate_move_to_object import check_move_to_object_distance_trajectory
from gym_unrealcv.envs.wrappers import augmentation, configUE, time_dilation

logger = logging.getLogger(__name__)

# ====== Constants ======
SLEEP_SHORT_S: float = 1.0
SLEEP_AFTER_RESET_S: float = 2.0
ACTION_SMALL_DELTA_POS: float = 3.0
ACTION_SMALL_DELTA_YAW: float = 1.0
ACTION_SMALL_STEPS: int = 10


class UnrealDeadline(BaseException):
    """Escape the route's generic exception handlers so the engine can be restarted."""


class StorageUnavailable(BaseException):
    """Bypass generic scorer handlers without misclassifying disk I/O as engine failure."""


@contextmanager
def storage_operation():
    try:
        yield
    except (OSError, UnrealDeadline) as error:
        raise StorageUnavailable(f"{type(error).__name__}: {error}") from error


@contextmanager
def engine_deadline(seconds):
    if (
        not hasattr(signal, "setitimer")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    def expired(signum, frame):
        raise UnrealDeadline(f"Unreal operation timed out after {seconds:.1f} seconds")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class EnvServer:
    _ROLLOUT_STATE_FIELDS = (
        "batch_states",
        "batch_size",
        "task_id_list",
        "current_instruction_list",
        "current_instruction",
        "trajectory_dirs",
        "trajectory_image_counts",
    )

    def __init__(self, cfg: Dict[str, Any]):
        "初始化环境服务器"
        self.cfg = dict(cfg)
        cfg = self.cfg
        self.output = Path(cfg.get("output", Path.cwd() / "outputs/training")).resolve()
        (self.output / "logs").mkdir(parents=True, exist_ok=True)
        cfg["ue_log"] = str(self.output / "logs/ue.log")
        self.env_id = cfg.get("env_id", "UnrealTrack-DowntownWest-ContinuousColor-v0")
        self.time_dilation = cfg.get("time_dilation", 10)
        self.seed = cfg.get("seed", 0)

        self.port = int(cfg.get("http_port", 5008))
        self.env = None
        if not self.isPortFree("0.0.0.0", self.port):
            raise RuntimeError(f"HTTP port {self.port} is already in use")
        try:
            self._initialize_environment(cfg)
        except BaseException:
            self.close()
            raise

        # 批次状态
        self.batch_states: List[Dict[str, Any]] = []
        self.batch_size: int = 0
        self.max_step_count = 100
        self._request_lock = threading.RLock()
        self._request_cache = OrderedDict()
        # This is only a key into the single cache, never a second response copy.
        self._latest_step_key = None
        self._engine_ready = True
        self._engine_restarts = 0
        self._recovery_max_restarts = cfg.get("recovery_max_restarts", 2)

        # 从tasks.json加载测试任务
        default_combined_path = Path(__file__).parent / "training_tasks" / "tasks.json"
        self.combined_json_path = Path(
            cfg.get("combined_json_path", default_combined_path)
        ).resolve()
        self.task_definitions: List[Dict[str, Any]] = self._load_task_definitions(
            self.combined_json_path
        )
        self.task_id_to_case: Dict[int, Dict[str, Any]] = {
            tc.get("task_id"): tc
            for tc in self.task_definitions
            if isinstance(tc, dict) and isinstance(tc.get("task_id"), int)
        }
        if not self.task_definitions:
            logger.warning(f"tasks.json为空或加载失败: {self.combined_json_path}")
        else:
            logger.info(
                f"已加载 {len(self.task_definitions)} 个任务定义自 {self.combined_json_path}"
            )
        self.current_instruction: Optional[str] = None
        self.current_instruction_list: List[Optional[str]] = []

        self.save_images: bool = cfg.get("save_images", True)
        self.save_dir = self.output / "trajectories" if self.save_images else None
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_dirs: List[Optional[Path]] = []  # 每个batch对应的轨迹文件夹
        self.trajectory_image_counts: List[int] = []  # 每个轨迹的图片计数
        self.task_id_list: List[int] = []  # 当前batch对应的task_id列表，用于按task_id分类

        # 设置Flask应用
        self.app = Flask(__name__)
        self.setup_routes()

    def _initialize_environment(self, cfg):
        logger.info(f"正在初始化环境: {self.env_id}")
        self.env = gym.make(self.env_id)
        if int(self.time_dilation) > 0:
            self.env = time_dilation.TimeDilationWrapper(self.env, int(self.time_dilation))
        self.env.unwrapped.agents_category = ["drone"]
        self.env = configUE.ConfigUEWrapper(
            self.env,
            resolution=(256, 256),
            offscreen=cfg.get("offscreen", False),
            gpu_id=cfg.get("gpu_id"),
        )
        ue_log = cfg.get("ue_log")
        if ue_log:
            launcher = self.env.unwrapped.ue_binary
            original_options = launcher.set_ue_options

            def options_with_log(*args, **kwargs):
                options = original_options(*args, **kwargs)
                options.append(f"-abslog={Path(ue_log).resolve()}")
                return options

            launcher.set_ue_options = options_with_log
        self.env = augmentation.RandomPopulationWrapper(self.env, 2, 2, random_target=False)
        self.env.seed(int(self.seed))
        self.env.reset()
        # Camera 0 captures the game viewport and can retain its previous view.
        # Use an independent scene-capture camera for deterministic RL observations.
        api = self.env.unwrapped.unrealcv
        self.rl_camera_id = api.get_camera_num()
        reply = api.client.request("vset /cameras/spawn")
        if reply is None or str(reply).lower().startswith("error"):
            raise RuntimeError(f"Cannot create independent RL camera: {reply}")
        api.register_camera(self.rl_camera_id)
        self.env.unwrapped.unrealcv.set_phy(self.env.unwrapped.player_list[0], 0)
        # Camera sits at the drone origin; exclude its mesh from first-person views.
        self.env.unwrapped.unrealcv.set_hide_obj(self.env.unwrapped.player_list[0])
        logger.info(f"环境配置: {self.env.unwrapped.unrealcv.get_camera_config()}")

        # 初始化物体
        time.sleep(SLEEP_SHORT_S)
        self.env.unwrapped.unrealcv.new_obj("bp_character_C", "BP_Character_21", [0, 0, 0])
        self.env.unwrapped.unrealcv.set_appearance("BP_Character_21", 0)
        self.env.unwrapped.unrealcv.set_obj_rotation("BP_Character_21", [0, 0, 0])
        time.sleep(SLEEP_SHORT_S)
        self.env.unwrapped.unrealcv.new_obj("BP_BaseCar_C", "BP_Character_22", [1000, 0, 0])
        self.env.unwrapped.unrealcv.set_appearance("BP_Character_22", 2)
        self.env.unwrapped.unrealcv.set_obj_rotation("BP_Character_22", [0, 0, 0])
        self.env.unwrapped.unrealcv.set_phy("BP_Character_22", 0)
        time.sleep(SLEEP_SHORT_S)

        logger.info("环境初始化完成")

    def isPortFree(self, ip, port):
        "Check if the port is free."
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if "linux" in sys.platform or "darwin" in sys.platform:
                sock.bind((ip, port))
            elif "win" in sys.platform:
                sock.bind((ip, port))
                sock.connect((ip, port))
        except Exception:
            sock.close()
            return False
        sock.close()
        return True

    def set_cam(self, state: Optional[Dict[str, Any]] = None) -> None:
        """Set the RL capture camera from state, independently of Blueprint drone rotation."""
        api = self.env.unwrapped.unrealcv
        if state is None:
            cam_loc = api.get_obj_location(self.env.unwrapped.player_list[0])
            cam_rot = api.get_obj_rotation(self.env.unwrapped.player_list[0])
        else:
            cam_loc = list(state["current_pos"])
            cam_rot = [0, self.normalize_angle(float(state["current_yaw"])), 0]
        # Character_API.set_cam targets a Blueprint actor, not an absolute capture camera.
        api.set_cam_location(getattr(self, "rl_camera_id", 0), cam_loc)
        api.set_cam_rotation(getattr(self, "rl_camera_id", 0), cam_rot, rpy=True)

    def _capture_rl_image(self) -> np.ndarray:
        """Flush the previous capture before returning the image for the current pose."""
        api = self.env.unwrapped.unrealcv
        api.get_image(getattr(self, "rl_camera_id", 0), "lit")
        time.sleep(0.1)
        return api.get_image(getattr(self, "rl_camera_id", 0), "lit")

    def _encode_observation(self, image: np.ndarray, batch_index: Optional[int] = None) -> str:
        """Encode once; persist exactly the bytes returned to the training client."""
        buffer = BytesIO()
        with storage_operation():
            Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(buffer, format="PNG")
            data = buffer.getvalue()
            if (
                self.save_images
                and batch_index is not None
                and self.trajectory_dirs[batch_index] is not None
            ):
                path = (
                    self.trajectory_dirs[batch_index]
                    / f"{self.trajectory_image_counts[batch_index]:04d}.png"
                )
                path.write_bytes(data)
                self.trajectory_image_counts[batch_index] += 1
        return base64.b64encode(data).decode("utf-8")

    def create_obj_if_needed(self, obj_info: Optional[Dict[str, Any]]) -> None:
        """Restore the active task's target and park the other target outside its view."""
        if not obj_info or obj_info.get("use_obj") not in (1, 2):
            return
        api = self.env.unwrapped.unrealcv
        human = obj_info["use_obj"] == 1
        active, inactive = (
            ("BP_Character_21", "BP_Character_22")
            if human
            else ("BP_Character_22", "BP_Character_21")
        )
        position = list(obj_info["obj_pos"])
        if not human:
            position[2] = 0
        api.set_appearance(active, obj_info.get("obj_id") if human else 2)
        api.set_obj_location(active, position)
        api.set_obj_rotation(active, obj_info["obj_rot"])
        if not human:
            api.set_phy(active, 0)

        def verify_position():
            for _ in range(10):
                actual = api.get_obj_location(active)
                if all(abs(actual[i] - position[i]) < 50.0 for i in range(3 if human else 2)):
                    return
                api.set_obj_location(active, position)
                time.sleep(0.01)
            logger.warning("Target %s did not settle at %s", active, position)

        verify_position()
        api.set_obj_location(inactive, [position[0] + 5000, 0, 0])
        api.set_obj_location(active, position)
        verify_position()
        time.sleep(SLEEP_SHORT_S)

    def _load_task_definitions(self, path: Path) -> List[Dict[str, Any]]:
        """从tasks.json加载任务列表"""
        if not path.exists():
            logger.error(f"tasks.json不存在: {path}")
            return []
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                logger.error(f"tasks.json格式错误，期望列表: {path}")
                return []
            return data
        except Exception as e:
            logger.error(f"读取tasks.json失败: {path}, 错误: {e}")
            return []

    def get_test_case_by_task_id(self, task_id: int) -> Optional[Dict[str, Any]]:
        """根据task_id获取任务定义"""
        task = self.task_id_to_case.get(task_id)
        return copy.deepcopy(task) if task is not None else None

    def _prepare_obj_info(self, test_case: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """根据测试用例准备物体信息"""
        use_obj = test_case.get("use_obj")
        if use_obj not in [1, 2]:
            return None
        target_pos = test_case.get("target_pos")
        if not (isinstance(target_pos, Sequence) and len(target_pos) >= 6):
            logger.warning(
                f"测试用例{test_case.get('_json_path')}缺少有效的target_pos，跳过物体创建"
            )
            return None
        obj_info = {
            "use_obj": use_obj,
            "obj_id": test_case.get("obj_id"),
            "target_pos": list(target_pos[:6]),
            "obj_pos": list(target_pos[:3]),
            "obj_rot": list(target_pos[3:6]),
        }
        return obj_info

    def _set_state_pose(
        self, state: Dict[str, Any], sleep_after: bool = False, apply_obj: bool = False
    ) -> None:
        """根据状态设置环境中的无人机姿态"""
        pos = state.get("current_pos")
        yaw = state.get("current_yaw")
        if pos is None:
            pos = state["initial_pos"][0:3]
        if yaw is None:
            yaw = state["initial_yaw"]
        self.env.unwrapped.unrealcv.set_obj_location(self.env.unwrapped.player_list[0], pos)
        # The hidden drone mesh need not rotate. Blueprint rotation also changes
        # capture components asynchronously; RL yaw belongs to the capture camera.
        if apply_obj:
            self.create_obj_if_needed(state.get("obj_info"))
        self.set_cam(state)
        if sleep_after:
            time.sleep(SLEEP_AFTER_RESET_S)

    def _apply_initial_state(self, state: Dict[str, Any]) -> np.ndarray:
        """将环境重置到state的初始姿态并返回当前图像"""
        state["current_pos"] = list(state["initial_pos"][0:3])
        state["current_yaw"] = float(state["initial_yaw"])
        state["last_pose"] = None
        state["small_count"] = 0
        state["step_count"] = 0
        self._set_state_pose(state, sleep_after=True, apply_obj=True)
        return self._capture_rl_image()

    def transform_to_global(self, x: float, y: float, initial_yaw: float) -> tuple:
        """将相对坐标转换为全局坐标"""
        theta = np.radians(initial_yaw)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        global_x = x * cos_theta - y * sin_theta
        global_y = x * sin_theta + y * cos_theta
        return global_x, global_y

    def normalize_angle(self, angle: float) -> float:
        """归一化角度到[-180, 180]"""
        angle = angle % 360
        if angle > 180:
            angle -= 360
        return angle

    def format_check_result(
        self, result: Dict[str, Any], chinese_category: str, trajectory_path: str
    ) -> str:
        """Keep scorer evidence in one generic report; scoring never depends on formatting."""

        def serializable(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            return str(value)

        details = json.dumps(result, ensure_ascii=False, indent=2, default=serializable)
        return f"检查轨迹文件: {trajectory_path}\n任务类别: {chinese_category}\n{details}"

    def _build_reward_list(
        self, reward: float, result: Dict[str, Any], category: Optional[str]
    ) -> List[float]:
        """Preserve the training contract: total and three category-specific components."""
        components = {
            "降落": ("approach_score", "descent_score", "collision_penalty"),
            "旋转": ("rotation_score", "position_score", None),
            "穿过": ("pass_reward", "position_reward", "collision_penalty"),
            "靠近": ("approach_score", "collision_penalty", None),
        }
        values = [reward] + [result.get(key, 0.0) for key in components.get(category, (None,) * 3)]
        if category == "移动到距物体xx米":
            values[1] = reward
        return [float(round(float(value), 2)) for value in values]

    def get_check_result_with_details(
        self, trajectory_json_path: str, project_root: str, task_id: int
    ) -> tuple:
        "获取检查结果和详细信息，返回(reward, formatted_log, chinese_category, result)"
        try:
            trajectory_path = Path(trajectory_json_path)
            project_root_path = Path(project_root)

            # 读取分类文件（按task_id分类，使用 task_categories.json）
            classified_instr_file = project_root_path / "training_tasks" / "task_categories.json"

            if not classified_instr_file.exists():
                return 0, f"错误: 分类文件不存在: {classified_instr_file}\n", None, {}

            with open(classified_instr_file, "r", encoding="utf-8") as f:
                classified_instr = json.load(f)

            # 按 task_id 查找类别
            category = None
            for cat, task_ids in classified_instr.items():
                if task_id in task_ids:
                    category = cat
                    break

            if category is None:
                return 0, f"错误: 在分类文件中找不到 task_id={task_id} 对应的类别\n", None, {}

            scorers = {
                "Pass": ("穿过", check_pass_trajectory),
                "Move": ("移动到距物体xx米", check_move_to_object_distance_trajectory),
                "Rotate": ("旋转", check_rotate_trajectory),
                "Land": ("降落", check_land_trajectory),
                "Approach": ("靠近", check_approach_trajectory),
            }
            if category not in scorers:
                return 0, f"错误: 不支持的训练任务类别: {category}\n", None, {}
            chinese_category, check_function = scorers[category]

            # 查找tasks.json文件
            combined_json_file = project_root_path / "training_tasks" / "tasks.json"

            if not combined_json_file.exists():
                return 0, f"错误: tasks.json文件不存在: {combined_json_file}\n", None, {}

            # 调用检查函数，传入tasks.json路径和task_id
            result = check_function(
                str(trajectory_path), combined_json_path=str(combined_json_file), task_id=task_id
            )

            # 格式化结果
            formatted_log = self.format_check_result(result, chinese_category, str(trajectory_path))

            # 返回reward（优先使用检查结果提供的数值，否则退回0/1）
            if "reward" in result:
                reward = float(result["reward"])
            else:
                reward = 1 if result.get("is_valid", False) else 0

            return reward, formatted_log, chinese_category, result

        except Exception as e:
            import traceback

            error_msg = f"错误: 处理轨迹文件时发生异常: {str(e)}\n{traceback.format_exc()}\n"
            return 0, error_msg, None, {}

    def setup_routes(self):
        """Register each route once, with one server-owned dispatch path."""
        for endpoint, methods in (
            ("reset", ["POST"]),
            ("step", ["POST"]),
            ("observation", ["GET"]),
            ("health", ["GET"]),
            ("rewards", ["POST"]),
        ):
            self.app.add_url_rule(
                "/" + endpoint, endpoint, partial(self._dispatch_request, endpoint), methods=methods
            )

    def _storage_error(self, error):
        logger.error("[UAV storage] %s failed; preserving engine: %s", request.path, error)
        return jsonify(status="error", error_code="storage_unavailable", message=str(error)), 503

    def _cache_response(self, key, digest, response):
        if request.endpoint == "reset":
            self._latest_step_key = None
        elif request.endpoint == "step":
            self._latest_step_key = key
        if key is None:
            return
        self._request_cache[key] = (
            digest,
            (response.get_data(), response.status_code, list(response.headers)),
        )
        while len(self._request_cache) > 8:
            oldest = next(iter(self._request_cache))
            if oldest == self._latest_step_key:
                # Keep the last step retryable even after many observation requests.
                self._request_cache.move_to_end(oldest)
            else:
                self._request_cache.popitem(last=False)

    def _dispatch_request(self, endpoint):
        with self._request_lock:
            if endpoint == "health":
                return self._health()
            view = getattr(self, "_" + endpoint)
            if endpoint == "rewards":
                # Scoring uses supplied trajectories and does not require a live UE.
                try:
                    return view()
                except StorageUnavailable as error:
                    return self._storage_error(error)
            request_id = request.headers.get("X-Rollout-Request-ID")
            key = (endpoint, request_id) if request_id else None
            digest = hashlib.sha256(request.get_data()).hexdigest()
            if key in self._request_cache:
                previous_digest, cached = self._request_cache[key]
                if digest != previous_digest:
                    return jsonify(
                        status="error", message="Request ID reused with different payload"
                    ), 409
                return self.app.make_response(cached)
            response = self._run_engine_request(view)
            if response.status_code < 400:
                self._cache_response(key, digest, response)
            return response

    def _run_engine_request(self, view):
        before = self._snapshot_rollout()
        deadline = time.monotonic() + self.cfg.get("recovery_total_timeout", 450)
        response = None
        for attempt in range(self._recovery_max_restarts + 1):
            try:
                if not self._engine_connected():
                    raise ConnectionError("Unreal process or connection unavailable")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise UnrealDeadline("Unreal recovery total timeout exhausted")
                with engine_deadline(min(remaining, self.cfg.get("recovery_request_timeout", 60))):
                    response = self.app.make_response(view())
                if not self._recoverable_engine_error(response):
                    if response.status_code >= 400:
                        self._restore_rollout(before)
                    return response
            except StorageUnavailable as error:
                self._restore_rollout(before)
                return self.app.make_response(self._storage_error(error))
            except (Exception, UnrealDeadline) as error:
                logger.exception("Environment request failed")
                response = self.app.make_response(
                    (jsonify(status="error", message=str(error)), 503)
                )
                if self._engine_connected() and not isinstance(
                    error, (ConnectionError, TimeoutError, OSError, UnrealDeadline)
                ):
                    self._restore_rollout(before)
                    return response
            # Roll back the complete request before reinitializing and replaying it.
            self._restore_rollout(before)
            remaining = deadline - time.monotonic()
            if attempt == self._recovery_max_restarts or remaining <= 0:
                detail = (response.get_json(silent=True) or {}).get("message", "Unreal unavailable")
                return self.app.make_response(
                    (
                        jsonify(
                            status="error",
                            error_code="unreal_recovery_exhausted",
                            message=detail,
                            recovery_count=self._engine_restarts,
                        ),
                        503,
                    )
                )
            try:
                with engine_deadline(min(remaining, self.cfg.get("recovery_restart_timeout", 210))):
                    self._restart_engine(before)
            except (Exception, UnrealDeadline) as error:
                logger.exception("Unreal restart failed")
                response = self.app.make_response(
                    (jsonify(status="error", message=str(error)), 503)
                )

    def _snapshot_rollout(self):
        return {
            key: copy.deepcopy(getattr(self, key))
            for key in self._ROLLOUT_STATE_FIELDS
            if hasattr(self, key)
        }

    def _restore_rollout(self, state):
        for key in self._ROLLOUT_STATE_FIELDS:
            if key in state:
                setattr(self, key, copy.deepcopy(state[key]))
            elif hasattr(self, key):
                delattr(self, key)

    def _engine_connected(self):
        if not self._engine_ready:
            return False
        try:
            raw = self.env.unwrapped
            process = getattr(raw.ue_binary, "env", None)
            return (process is None or process.poll() is None) and raw.unrealcv.client.isconnected()
        except (AttributeError, OSError):
            return False

    def _restart_engine(self, state):
        logger.warning(
            "Restarting Unreal; restoring %s task states", len(state.get("batch_states", []))
        )
        self._engine_ready = False
        try:
            self.close()
        except Exception:
            logger.exception("Cleanup of failed Unreal instance raised an error")
        self.env = None
        self._initialize_environment(self.cfg)
        self._restore_rollout(state)
        # The server multiplexes a batch on one physical drone. Each task must
        # restore its objects as well as pose when it becomes active.
        if self.batch_states:
            self._set_state_pose(self.batch_states[-1], apply_obj=True)
        self._engine_ready = True
        self._engine_restarts += 1

    def _recoverable_engine_error(self, response):
        if response.status_code < 500:
            return False
        if not self._engine_connected():
            return True
        message = str((response.get_json(silent=True) or {}).get("message", "")).lower()
        return any(
            word in message
            for word in (
                "socket",
                "broken pipe",
                "timed out",
                "connection",
                "decode_img",
                "get_image",
                "cannot receive",
                "can not receive",
            )
        )

    def _health(self):
        try:
            if not self._engine_connected():
                raise ConnectionError("Unreal process or connection unavailable")
            # A live Flask server/socket alone does not establish engine health.
            with engine_deadline(5):
                self.env.unwrapped.unrealcv.get_obj_location(self.env.unwrapped.player_list[0])
            return jsonify(
                status="healthy", env_id=self.env_id, recovery_count=self._engine_restarts
            )
        except (Exception, UnrealDeadline) as exc:
            return jsonify(
                status="unhealthy",
                env_id=self.env_id,
                message=str(exc),
                recovery_count=self._engine_restarts,
            ), 503

    def _reset(self):
        "重置环境接口"
        try:
            data = request.json or {}
            task_ids = data.get("task_ids", None)

            if task_ids is None:
                return jsonify(
                    {"status": "error", "message": "task_ids is required and must be a list."}
                ), 400

            if not isinstance(task_ids, list) or len(task_ids) == 0:
                return jsonify(
                    {"status": "error", "message": "task_ids must be a non-empty list."}
                ), 400

            # 验证所有task_id都是整数
            try:
                task_ids_int = [int(tid) for tid in task_ids]
            except (ValueError, TypeError):
                return jsonify(
                    {"status": "error", "message": "All task_ids must be integers."}
                ), 400

            batch_size = len(task_ids_int)

            # 为每个task_id加载测试用例
            test_cases = []
            for task_id_int in task_ids_int:
                test_case_raw = self.get_test_case_by_task_id(task_id_int)
                if test_case_raw is None:
                    return jsonify(
                        {
                            "status": "error",
                            "message": f"task_id {task_id_int} not found in tasks.json.",
                        }
                    ), 400
                test_cases.append(copy.deepcopy(test_case_raw))

            # 初始化批次状态
            self.batch_size = batch_size
            self.batch_states = []
            self.current_instruction_list = []
            self.task_id_list = task_ids_int  # 保存task_id列表

            # 初始化图片保存相关
            if self.save_images and self.save_dir is not None:
                # 为每个batch创建轨迹文件夹
                self.trajectory_dirs = []
                self.trajectory_image_counts = []
                base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                for idx in range(batch_size):
                    # 使用 task_id 命名文件夹
                    task_id = task_ids_int[idx]

                    # 使用base_timestamp + 毫秒 + batch索引确保唯一性
                    traj_timestamp = (
                        f"{base_timestamp}_{datetime.now().strftime('%f')[:-3]}_{idx:03d}"
                    )
                    traj_dir = self.save_dir / str(task_id) / traj_timestamp
                    with storage_operation():
                        traj_dir.mkdir(parents=True, exist_ok=True)
                    self.trajectory_dirs.append(traj_dir)
                    self.trajectory_image_counts.append(0)
                    logger.debug(f"创建轨迹文件夹 {idx}: {traj_dir}")
                    # 添加小延迟确保时间戳不同（如果仍然相同，索引会保证唯一性）
                    time.sleep(0.001)
            else:
                # 如果不保存图片，初始化空列表
                self.trajectory_dirs = [None] * batch_size
                self.trajectory_image_counts = [0] * batch_size

            images_batch: List[List[str]] = []
            instructions: List[Optional[str]] = []

            # 为每个task_id准备状态并应用到环境
            for idx, (task_id_int, test_case) in enumerate(zip(task_ids_int, test_cases)):
                initial_pos = test_case.get("initial_pos") or test_case.get("init_pose")
                if not (isinstance(initial_pos, (list, tuple)) and len(initial_pos) >= 5):
                    return jsonify(
                        {
                            "status": "error",
                            "message": f"Test case task_id={task_id_int} missing valid initial_pos/init_pose.",
                        }
                    ), 500

                # prompt即instruction
                instruction = test_case.get("instruction") or test_case.get("prompt")
                if idx == 0:
                    print(f"instruction: {instruction}")
                    self.current_instruction = instruction

                obj_info = self._prepare_obj_info(test_case)

                # 准备状态
                state = {
                    "initial_pos": list(initial_pos),
                    "initial_yaw": float(initial_pos[4]),
                    "obj_info": obj_info,
                    "task_id": task_id_int,
                    "instruction": instruction,
                    "done": False,
                    "current_pos": list(initial_pos[:3]),
                    "current_yaw": float(initial_pos[4]),
                    "last_pose": None,
                    "small_count": 0,
                    "step_count": 0,
                }

                # 应用初始状态到环境（每个task都需要重新设置环境）
                image = self._apply_initial_state(state)
                img_base64 = self._encode_observation(image, idx)

                self.batch_states.append(state)
                self.current_instruction_list.append(instruction)
                images_batch.append([img_base64])
                instructions.append(instruction)

            logger.info(f"环境重置成功，batch_size={batch_size}, task_ids={task_ids_int}")
            response_payload: Dict[str, Any] = {
                "status": "success",
                "images": images_batch,
                "message": "Environment reset successfully",
                "batch_size": batch_size,
                "task_ids": task_ids_int,
                "instructions": instructions,
            }
            # print(images_batch)
            return jsonify(response_payload)

        except Exception as e:
            import traceback

            logger.error(f"Reset error: {e}\n{traceback.format_exc()}")
            return jsonify(
                {"status": "error", "message": str(e) + "\n" + traceback.format_exc()}
            ), 500

    def _step(self):
        "执行动作接口"
        try:
            data = request.json or {}
            actions_batch = data.get("actions", None)

            if (
                actions_batch is None
                or not isinstance(actions_batch, list)
                or len(actions_batch) == 0
            ):
                return jsonify(
                    {
                        "status": "error",
                        "message": "actions is required and must be a non-empty list",
                    }
                ), 400

            if not self.batch_states:
                return jsonify(
                    {
                        "status": "error",
                        "message": "Environment not reset. Please call /reset first.",
                        "error_code": "episode_missing",
                    }
                ), 409

            if len(actions_batch) != len(self.batch_states):
                return jsonify(
                    status="error",
                    error_code="batch_size_mismatch",
                    message="Actions batch size must match the active episode batch",
                    expected_batch_size=len(self.batch_states),
                    received_batch_size=len(actions_batch),
                ), 400

            images_batch: List[List[str]] = []
            done_batch: List[bool] = []
            accepted_batch = []
            termination_reasons = []
            # Publish updated state only after building the complete response.
            next_states = [state.copy() for state in self.batch_states]

            for batch_idx, state in enumerate(next_states):
                actions = actions_batch[batch_idx]
                if actions is None:
                    actions = []

                self._set_state_pose(state, apply_obj=True)

                initial_x, initial_y, initial_z = state["initial_pos"][0:3]
                initial_yaw = state["initial_yaw"]

                accepted = []
                done_flag = state.get("done", False)
                reason = state.get("termination_reason") if done_flag else None
                if done_flag and reason is None:
                    reason = "already_done"
                if not done_flag and state["step_count"] >= self.max_step_count:
                    done_flag, reason = True, "max_steps"

                for i, action_pose in enumerate(actions):
                    if done_flag:
                        break
                    try:
                        if not isinstance(action_pose, (list, tuple)) or len(action_pose) < 4:
                            raise ValueError("Expected four waypoint coordinates")
                        waypoint = [float(value) for value in action_pose[:4]]
                        if not np.isfinite(waypoint).all():
                            raise ValueError("Non-finite waypoint coordinates")
                    except (TypeError, ValueError, OverflowError) as error:
                        logger.warning(
                            f"Invalid waypoint for batch {batch_idx}, index {i}: {error}"
                        )
                        done_flag, reason = True, "invalid_waypoint"
                        break

                    relative_x, relative_y, relative_z, yaw_rad = waypoint
                    relative_yaw = (float(np.degrees(yaw_rad)) + 180) % 360 - 180
                    global_x, global_y = self.transform_to_global(
                        relative_x, relative_y, initial_yaw
                    )
                    state["current_pos"] = [
                        global_x + initial_x,
                        global_y + initial_y,
                        relative_z + initial_z,
                    ]
                    state["current_yaw"] = self.normalize_angle(relative_yaw + initial_yaw)
                    accepted.append(waypoint)
                    state["step_count"] += 1

                    pose_now = [relative_x, relative_y, relative_z, relative_yaw]
                    last_pose = state.get("last_pose")
                    if last_pose is not None:
                        diffs = [abs(a - b) for a, b in zip(pose_now, last_pose)]
                        if (
                            all(d < ACTION_SMALL_DELTA_POS for d in diffs[:3])
                            and diffs[3] < ACTION_SMALL_DELTA_YAW
                        ):
                            state["small_count"] = state.get("small_count", 0) + 1
                        else:
                            state["small_count"] = 0
                    state["last_pose"] = pose_now

                    if state["step_count"] >= self.max_step_count:
                        done_flag, reason = True, "max_steps"
                        break
                    if state.get("small_count", 0) >= ACTION_SMALL_STEPS:
                        done_flag, reason = True, "small_motion"
                        break

                self.env.unwrapped.unrealcv.set_obj_location(
                    self.env.unwrapped.player_list[0], state["current_pos"]
                )
                self.set_cam(state)
                image = self._capture_rl_image()
                time.sleep(0.1)

                if not accepted:
                    # Preserve the running service's second camera read when no
                    # waypoint is applied, including completed and empty actions.
                    self.set_cam(state)
                    image = self._capture_rl_image()

                # 确保有图像用于返回（如果actions为空，使用上面获取的图像；否则使用最后一个动作后的图像）
                if image is None:
                    self.set_cam(state)
                    image = self._capture_rl_image()

                img_base64 = self._encode_observation(image, batch_idx)

                state["current_pos"] = state.get("current_pos", state["initial_pos"][:3])
                state["current_yaw"] = state.get("current_yaw", state["initial_yaw"])

                state["done"] = done_flag
                state["termination_reason"] = reason
                images_batch.append([img_base64])
                done_batch.append(done_flag)
                accepted_batch.append(accepted)
                termination_reasons.append(reason)

            payload = {
                "status": "success",
                "images": images_batch,
                "dones": done_batch,
                "accepted_waypoints": accepted_batch,
                "termination_reasons": termination_reasons,
                "message": "Step executed successfully",
                "batch_size": len(images_batch),
            }
            response = jsonify(payload)
            self.batch_states = next_states
            return response

        except Exception as e:
            import traceback

            logger.error(f"Step error: {e}\n{traceback.format_exc()}")
            return jsonify(
                {"status": "error", "message": str(e) + "\n" + traceback.format_exc()}
            ), 500

    def _observation(self):
        """Read the current rendered frame and measured poses without changing the rollout."""
        try:
            api = self.env.unwrapped.unrealcv
            agent = self.env.unwrapped.player_list[0]
            camera_location = api.get_cam_location(getattr(self, "rl_camera_id", 0))
            camera_rotation = api.get_cam_rotation(getattr(self, "rl_camera_id", 0))
            agent_location = api.get_obj_location(agent)
            agent_rotation = api.get_obj_rotation(agent)
            image = api.get_image(getattr(self, "rl_camera_id", 0), "lit")
            return jsonify(
                {
                    "status": "success",
                    "images": [[self._encode_observation(image)]],
                    "camera": {
                        "id": getattr(self, "rl_camera_id", 0),
                        "location": camera_location,
                        "rotation": camera_rotation,
                    },
                    "agent": {
                        "name": agent,
                        "location": agent_location,
                        "rotation": agent_rotation,
                    },
                    "task_ids": self.task_id_list,
                    "batch_size": self.batch_size,
                }
            )
        except Exception as exc:
            logger.exception("Observation error")
            return jsonify({"status": "error", "message": str(exc)}), 500

    def _rewards(self):
        "计算批次轨迹的rewards接口"
        try:
            data = request.json or {}
            trajectories = data.get("trajectories", None)

            if trajectories is None or not isinstance(trajectories, list):
                return jsonify(
                    {"status": "error", "message": "trajectories is required and must be a list"}
                ), 400

            if len(trajectories) == 0:
                return jsonify(
                    {"status": "error", "message": "trajectories list cannot be empty"}
                ), 400

            # 检查task_id_list是否已设置
            if not self.task_id_list:
                return jsonify(
                    status="error",
                    error_code="episode_missing",
                    message="Environment not reset. Please call /reset first.",
                ), 409
            if len(self.task_id_list) != len(trajectories):
                return jsonify(
                    {
                        "status": "error",
                        "message": f"task_id_list length ({len(self.task_id_list) if self.task_id_list else 0}) does not match trajectories length ({len(trajectories)})",
                    }
                ), 400

            # 自动推断项目根目录
            current_file = Path(__file__)
            project_root = current_file.parent

            # 创建临时目录用于保存轨迹文件
            temp_dir = self.output / "tmp" / f"port_{self.port}"
            with storage_operation():
                temp_dir.mkdir(parents=True, exist_ok=True)

            rewards_list: List[float] = []
            reward_1_list: List[float] = []
            reward_2_list: List[float] = []
            reward_3_list: List[float] = []

            # 用于收集每个task_id的log数据 (格式: {task_id: [(formatted_log, reward_list), ...]})
            task_id_to_logs: Dict[int, List[tuple]] = {}

            # 处理每个轨迹
            for idx, trajectory in enumerate(trajectories):
                if not isinstance(trajectory, list) or len(trajectory) == 0:
                    logger.warning(f"轨迹 {idx} 为空或格式不正确，reward设为0")
                    rewards_list.append(0.0)
                    reward_1_list.append(0.0)
                    reward_2_list.append(0.0)
                    reward_3_list.append(0.0)
                    continue

                # 获取对应的task_id
                task_id = self.task_id_list[idx] if idx < len(self.task_id_list) else None
                if task_id is None:
                    logger.warning(f"轨迹 {idx} 没有对应的task_id，reward设为0")
                    rewards_list.append(0.0)
                    reward_1_list.append(0.0)
                    reward_2_list.append(0.0)
                    reward_3_list.append(0.0)
                    continue

                # 轨迹文件名使用 task_id.json
                json_filename = f"{task_id}.json"

                try:
                    # 将轨迹转换为JSON格式
                    # 轨迹格式: [[x, y, z, roll, yaw, pitch], ...]
                    # 需要转换为: [{"state": [[x, y, z], [roll, yaw, pitch]]}, ...]
                    trajectory_json = []
                    for point in trajectory:
                        if not isinstance(point, (list, tuple)) or (
                            len(point) != 6 and len(point) != 4
                        ):
                            continue
                        if len(point) == 6:
                            x, y, z = float(point[0]), float(point[1]), float(point[2])
                            roll, yaw, pitch = float(point[3]), float(point[4]), float(point[5])
                            trajectory_json.append({"state": [[x, y, z], [roll, yaw, pitch]]})
                        elif len(point) == 4:
                            x, y, z, yaw = (
                                float(point[0]),
                                float(point[1]),
                                float(point[2]),
                                float(point[3]),
                            )
                            yaw = self.normalize_angle(np.degrees(yaw))
                            trajectory_json.append({"state": [[x, y, z], [0, yaw, 0]]})

                    if len(trajectory_json) == 0:
                        logger.warning(f"轨迹 {idx} 没有有效点，reward设为0")
                        rewards_list.append(0.0)
                        reward_1_list.append(0.0)
                        reward_2_list.append(0.0)
                        reward_3_list.append(0.0)
                        continue

                    # 保存轨迹到临时文件（用于计算reward）
                    temp_trajectory_path = temp_dir / json_filename
                    with storage_operation():
                        with open(temp_trajectory_path, "w", encoding="utf-8") as f:
                            json.dump(trajectory_json, f, indent=2)

                    # 保存轨迹到轨迹目录（与图片保存在同一文件夹）
                    if (
                        self.save_images
                        and idx < len(self.trajectory_dirs)
                        and self.trajectory_dirs[idx] is not None
                    ):
                        try:
                            trajectory_save_path = self.trajectory_dirs[idx] / json_filename
                            with storage_operation():
                                with open(trajectory_save_path, "w", encoding="utf-8") as f:
                                    json.dump(trajectory_json, f, indent=2)
                            logger.debug(f"轨迹 {idx} 已保存到: {trajectory_save_path}")
                        except Exception as e:
                            logger.warning(f"保存轨迹 {idx} 到轨迹目录失败: {e}")
                    logger.debug(f"计算轨迹 {idx} 的reward: task_id={task_id}")

                    # 调用get_check_result_with_details获取详细结果
                    # 只传入task_id，不再使用json_name
                    with storage_operation():
                        reward, formatted_log, chinese_category, result = (
                            self.get_check_result_with_details(
                                str(temp_trajectory_path), str(project_root), task_id=task_id
                            )
                        )

                    if chinese_category is None or not isinstance(result, dict) or not result:
                        raise RuntimeError(formatted_log.strip() or "Scorer returned no result")
                    if result.get("error"):
                        raise RuntimeError(str(result["error"]))
                    # 根据类别构建分数列表 [reward, score1, score2, score3]
                    reward_list = self._build_reward_list(reward, result, chinese_category)
                    if not np.isfinite(reward_list).all():
                        raise ValueError("Scorer returned a non-finite reward")

                    # 收集log数据
                    if task_id not in task_id_to_logs:
                        task_id_to_logs[task_id] = []
                    task_id_to_logs[task_id].append((formatted_log, reward_list))

                    rewards_list.append(reward_list[0])
                    reward_1_list.append(reward_list[1])
                    reward_2_list.append(reward_list[2])
                    reward_3_list.append(reward_list[3])

                    logger.debug(
                        f"轨迹 {idx} reward计算完成: task_id={task_id}, reward_list={reward_list}"
                    )

                except Exception as error:
                    logger.exception("Scoring failed for task_id=%s trajectory=%s", task_id, idx)
                    return jsonify(
                        status="error",
                        error_code="scoring_failed",
                        task_id=task_id,
                        trajectory_index=idx,
                        message=str(error),
                    ), 500

            logger.info(f"批次reward计算完成，共 {len(rewards_list)} 个轨迹")

            # 保存log文件到每个task_id对应的文件夹
            if self.save_images and self.save_dir is not None:
                # 获取当前时间戳，用于区分不同批次的记录
                batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 精确到毫秒

                for task_id, logs in task_id_to_logs.items():
                    if not logs:
                        continue

                    # 获取task_id对应的文件夹
                    task_id_dir = self.save_dir / str(task_id)
                    if task_id_dir.exists():
                        # 保存log文件（追加模式，使用时间戳区分不同批次）
                        log_file_path = task_id_dir / "check_result.log"
                        try:
                            # 检查文件是否存在，决定是否添加分隔符
                            file_exists = log_file_path.exists()

                            # 使用追加模式，并在每次追加前添加时间戳分隔符
                            with storage_operation():
                                with open(log_file_path, "a", encoding="utf-8") as f:
                                    # 如果文件已存在，添加批次分隔符和时间戳
                                    if file_exists:
                                        f.write(f"\n{'=' * 80}\n")
                                        f.write(f"批次时间戳: {batch_timestamp}\n")
                                        f.write(f"{'=' * 80}\n\n")
                                    else:
                                        # 第一次写入时，也添加时间戳（但不在开头加换行）
                                        f.write(f"{'=' * 80}\n")
                                        f.write(f"批次时间戳: {batch_timestamp}\n")
                                        f.write(f"{'=' * 80}\n\n")

                                    for log_content, reward_list in logs:
                                        f.write(log_content)
                                        # 添加reward列表记录
                                        f.write(f"\n[Reward列表] {reward_list}\n")
                                        f.write("\n")

                            action = "追加" if file_exists else "创建"
                            logger.info(
                                f"已{action}log记录到文件: {log_file_path} (批次: {batch_timestamp})"
                            )
                        except Exception as e:
                            logger.error(f"保存log文件失败: {log_file_path}, 错误: {e}")

            return jsonify(
                {
                    "rewards": rewards_list,
                    "reward_1": reward_1_list,
                    "reward_2": reward_2_list,
                    "reward_3": reward_3_list,
                }
            )

        except Exception as e:
            import traceback

            logger.error(f"计算rewards时发生错误: {e}\n{traceback.format_exc()}")
            return jsonify({"status": "error", "message": f"计算rewards时发生错误: {str(e)}"}), 500

    def run(self):
        """启动HTTP服务器"""
        logger.info(f"启动环境服务器，监听端口: {self.port}")
        self.app.run(
            host=self.cfg.get("host", "0.0.0.0"), port=self.port, threaded=False, use_reloader=False
        )

    def close(self):
        """关闭环境"""
        if self.env is not None:
            try:
                self.env.close()
            finally:
                # A reset can fail before env.launched becomes True.
                process = getattr(self.env.unwrapped.ue_binary, "env", None)
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait()
            logger.info("环境已关闭")


def main(args=None):
    """主函数"""
    args = parse_args() if args is None else args

    # 配置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )

    # 配置
    cfg = {
        "env_id": args.env_id,
        "time_dilation": args.time_dilation,
        "seed": args.seed,
        "http_port": args.port,
        "offscreen": args.offscreen,
        "gpu_id": args.gpu_id,
        "host": args.host,
        "output": args.output,
    }

    # 创建并运行服务器
    if args.env_bin:
        os.environ["UNREALZOO_ENV_BIN"] = str(Path(args.env_bin).resolve())

    def stop_server(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_server)
    server = None
    try:
        server = EnvServer(cfg)
        server.run()
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭...")
    finally:
        if server is not None:
            server.close()


if __name__ == "__main__":
    main(_cli_args)
