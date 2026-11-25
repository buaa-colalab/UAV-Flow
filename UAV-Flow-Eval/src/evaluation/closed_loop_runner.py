# SPDX-License-Identifier: Apache-2.0
# Modified for UAV-Flow-Eval; upstream attribution and notices: LICENSE.
import argparse
import base64
import os
import signal
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from uav_flow.runtime import environment, make_output, output_context

DEFAULT_TICK_X: Optional[float] = 100
DEFAULT_TICK_Y: Optional[float] = 100
DEFAULT_TICK_Z: Optional[float] = 100
DEFAULT_MIN_SPAN_X: Optional[float] = 400
DEFAULT_MIN_SPAN_Y: Optional[float] = 400
DEFAULT_MIN_SPAN_Z: Optional[float] = 400


def parse_args(argv=None):
    # ====== Default parameters ======
    DEFAULT_ENV_ID = "UnrealTrack-DowntownWest-ContinuousColor-v0"
    DEFAULT_TIME_DILATION = 10
    DEFAULT_SEED = 0
    DEFAULT_JSON_FOLDER = str(Path(__file__).resolve().parent / "test_jsons")
    DEFAULT_SERVER_PORT = 6067
    DEFAULT_MAX_STEPS = 100
    DEFAULT_INSTRUCTION_TYPE = "instruction_unified"  # "instruction" or "instruction_unified"
    # Fixed tick spacing (grid size)
    # Minimum display span

    import argparse

    parser = argparse.ArgumentParser(
        description="Batch run UAV control with instruction-conditioned policy"
    )
    parser.add_argument("-e", "--env_id", default=DEFAULT_ENV_ID, help="Environment ID to run")
    parser.add_argument(
        "-t",
        "--time_dilation",
        default=DEFAULT_TIME_DILATION,
        type=int,
        help="Time dilation parameter to keep FPS in simulator",
    )
    parser.add_argument("-s", "--seed", default=DEFAULT_SEED, type=int, help="Random seed")
    parser.add_argument(
        "-f",
        "--json_folder",
        default=DEFAULT_JSON_FOLDER,
        help="Folder path containing batch task json files",
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help="Single output root")
    parser.add_argument(
        "--classification",
        type=Path,
        default=Path(__file__).resolve().parent / "classified_instr.json",
    )
    parser.add_argument(
        "-p", "--server_port", default=DEFAULT_SERVER_PORT, type=int, help="Inference server port"
    )
    parser.add_argument(
        "--server-url", default="", help="Inference API base URL, or full /predict URL"
    )
    parser.add_argument(
        "-m", "--max_steps", default=DEFAULT_MAX_STEPS, type=int, help="Maximum inference steps"
    )
    parser.add_argument(
        "--surround-max-steps",
        "--surround_max_steps",
        dest="surround_max_steps",
        default=400,
        type=int,
        help="Maximum action points for Surround tasks only",
    )
    parser.add_argument(
        "-i",
        "--instruction_type",
        default=DEFAULT_INSTRUCTION_TYPE,
        choices=["instruction", "instruction_unified"],
        help="Choose which field to use: instruction or instruction_unified",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    # Fixed tick spacing (grid size)
    parser.add_argument("--tick_x", type=float, default=None, help="Tick step for X (forward) axis")
    parser.add_argument("--tick_y", type=float, default=None, help="Tick step for Y (right) axis")
    parser.add_argument("--tick_z", type=float, default=None, help="Tick step for Z (up) axis")
    # Minimum display span
    parser.add_argument(
        "--min_span_x", type=float, default=None, help="Minimum span for X (forward) axis"
    )
    parser.add_argument(
        "--min_span_y", type=float, default=None, help="Minimum span for Y (right) axis"
    )
    parser.add_argument(
        "--min_span_z", type=float, default=None, help="Minimum span for Z (up) axis"
    )
    parser.add_argument(
        "--unreal_op_retries",
        type=int,
        default=2,
        help="Extra retry count for each UnrealCV operation",
    )
    parser.add_argument(
        "--unreal_retry_sleep",
        type=float,
        default=0.5,
        help="Sleep seconds between UnrealCV operation retries",
    )
    parser.add_argument(
        "--task_retries",
        type=int,
        default=1,
        help="Extra retry count for each task in the same env",
    )
    parser.add_argument(
        "--env_boot_retries", type=int, default=2, help="Extra retry count when creating the env"
    )
    parser.add_argument(
        "--env_retry_sleep",
        type=float,
        default=2.0,
        help="Sleep seconds between task retry attempts",
    )
    parser.add_argument(
        "--restart_env_every",
        type=int,
        default=0,
        help="Deprecated and ignored; env will not be recreated during the batch",
    )
    parser.add_argument(
        "--ue_extra_args",
        default="",
        help="Extra command-line args appended when launching Collection.exe",
    )
    parser.add_argument(
        "--save-frames", action="store_true", help="Save model-input frames under the output root"
    )
    parser.add_argument(
        "--drone_cam_id",
        type=int,
        default=-1,
        help="Camera id used for model input; -1 = auto-resolve the camera closest to the drone",
    )
    args = parser.parse_args(argv)
    if args.max_steps <= 0 or args.surround_max_steps <= 0:
        parser.error("Action point budgets must be positive")
    args.output = args.output.resolve()
    args.json_folder = str(Path(args.json_folder).resolve())
    args.classification = args.classification.resolve()
    return args


if __name__ == "__main__":
    _cli_args = parse_args()
    _cli_args.output = _cli_args.output.resolve()
    make_output(_cli_args.output)
    os.environ.update(environment(_cli_args.output))
    os.chdir(_cli_args.output / "tmp")

import json
import time
from io import BytesIO

import cv2
import gym
import numpy as np
import requests
from PIL import Image

import gym_unrealcv  # noqa: F401 - registration and public simulator module identity

glob = __import__("glob")
import logging

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from check.check_trajectory_orbit import (
    LOSS_THRESHOLD,
    MIN_DIRECTED_ANGLE_DEG,
    compute_trajectory_loss,
    orbit_is_complete,
)
from gym_unrealcv.envs.wrappers import augmentation, configUE, time_dilation

logger = logging.getLogger(__name__)

# ====== Constants ======
IMG_INPUT_SIZE: Tuple[int, int] = (224, 224)
SLEEP_SHORT_S: float = 1.0
SLEEP_AFTER_RESET_S: float = 3.0
ACTION_SMALL_DELTA_POS: float = 3.0
ACTION_SMALL_DELTA_YAW: float = 1.0
ACTION_SMALL_STEPS: int = 10
TRAJ_IMG_SIZE_2D: Tuple[int, int] = (10, 10)
TRAJ_IMG_SIZE_3D: Tuple[int, int] = (12, 10)
PLOT_YAW_ARROW_MIN_LEN_2D: int = 10
PLOT_YAW_ARROW_MIN_LEN_3D: int = 12


def local_timestamp() -> str:
    """Return a local timestamp that is stable for log records."""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def configure_logging(logs_dir: str, log_level: str, run_id: str) -> str:
    """Log to both terminal and a timestamped file under the run logs folder."""
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"batch_run_{run_id}.log")
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        ],
        force=True,
    )
    return log_path


def append_task_event(event_log: Optional[List[Dict[str, Any]]], event: str, **fields: Any) -> None:
    """Append a structured event to a per-task log when enabled."""
    if event_log is None:
        return
    item = {
        "timestamp": local_timestamp(),
        "event": event,
    }
    item.update(fields)
    event_log.append(item)


def retry_unreal_op(
    op_name: str, func: Callable[[], Any], retries: int, retry_sleep_s: float
) -> Any:
    """Retry a UnrealCV operation a few times before surfacing the error."""
    last_error: Optional[Exception] = None
    total_attempts = max(1, retries + 1)
    for attempt in range(1, total_attempts + 1):
        try:
            return func()
        except Exception as e:
            last_error = e
            if attempt >= total_attempts:
                break
            logger.warning(
                f"{op_name} failed on attempt {attempt}/{total_attempts}: {e}. "
                f"Retrying in {retry_sleep_s:.2f}s"
            )
            time.sleep(retry_sleep_s)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{op_name} failed without raising a concrete exception")


def close_env_quietly(env: Optional[Any]) -> None:
    """Close env and suppress teardown errors so retries can continue."""
    if env is None:
        return
    try:
        env.close()
    except Exception as e:
        logger.warning(f"Failed to close env cleanly: {e}")


def kill_stale_ue_processes() -> None:
    """Only clean the explicit private UE binary belonging to this evaluation."""
    binary = os.environ.get("UNREALZOO_ENV_BIN")
    if not binary:
        logger.warning("No private UE binary configured; refusing broad process cleanup")
        return
    expected = str(Path(binary).resolve()).encode()
    if not Path("/proc").is_dir():
        return
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc / "cmdline").read_bytes().split(b"\0")
            if command and command[0] == expected:
                os.kill(int(proc.name), signal.SIGTERM)
        except (OSError, PermissionError):
            pass


def probe_env(env: Any, retries: int, retry_sleep_s: float) -> bool:
    """Check whether the UnrealCV connection is still alive."""
    try:
        player = env.unwrapped.player_list[0]
        retry_unreal_op(
            "probe_env.get_obj_location",
            lambda: env.unwrapped.unrealcv.get_obj_location(player),
            retries,
            retry_sleep_s,
        )
        return True
    except Exception as e:
        logger.warning(f"Env health probe failed: {e}")
        return False


def append_ue_extra_args(env: Any, extra_args: str) -> None:
    """Append optional UE command-line args without editing unrealcv launcher."""
    extra_args = extra_args.strip()
    if not extra_args:
        return
    ue_binary = env.unwrapped.ue_binary
    original_set_ue_options = ue_binary.set_ue_options
    parsed_args = extra_args.split()

    def set_ue_options_with_extra(
        cmd_exe=None, opengl=False, offscreen=False, nullrhi=False, gpu_id=None
    ):
        if cmd_exe is None:
            cmd_exe = []
        cmd_exe = original_set_ue_options(cmd_exe, opengl, offscreen, nullrhi, gpu_id)
        cmd_exe.extend(parsed_args)
        return cmd_exe

    ue_binary.set_ue_options = set_ue_options_with_extra
    logger.info(f"Appended UE launch args: {parsed_args}")


def build_env(args: argparse.Namespace) -> Any:
    """Create and initialize a fresh gym/UnrealCV env instance."""
    env = gym.make(args.env_id)
    # Instance-scoped selection: importing evaluation never changes training.
    env.unwrapped.runtime_profile = "evaluation"
    if int(args.time_dilation) > 0:
        env = time_dilation.TimeDilationWrapper(env, int(args.time_dilation))
    env.unwrapped.agents_category = ["drone"]
    env = configUE.ConfigUEWrapper(env, resolution=(256, 256))
    env = augmentation.RandomPopulationWrapper(env, 2, 2, random_target=False)
    append_ue_extra_args(env, args.ue_extra_args)
    env.seed(int(args.seed))
    retry_unreal_op(
        "env.reset", lambda: env.reset(), args.unreal_op_retries, args.unreal_retry_sleep
    )
    retry_unreal_op(
        "set_viewport",
        lambda: env.unwrapped.unrealcv.set_viewport(env.unwrapped.player_list[0]),
        args.unreal_op_retries,
        args.unreal_retry_sleep,
    )
    retry_unreal_op(
        "set_phy",
        lambda: env.unwrapped.unrealcv.set_phy(env.unwrapped.player_list[0], 0),
        args.unreal_op_retries,
        args.unreal_retry_sleep,
    )
    camera_config = retry_unreal_op(
        "get_camera_config",
        lambda: env.unwrapped.unrealcv.get_camera_config(),
        args.unreal_op_retries,
        args.unreal_retry_sleep,
    )
    logger.info(camera_config)
    print(camera_config)
    return env


def initialize_runtime_objects(env: Any, retries: int, retry_sleep_s: float) -> None:
    """Spawn the helper character/car objects used by downstream tasks."""
    time.sleep(SLEEP_SHORT_S)
    retry_unreal_op(
        "new_obj.BP_Character_21",
        lambda: env.unwrapped.unrealcv.new_obj("bp_character_C", "BP_Character_21", [0, 0, 0]),
        retries,
        retry_sleep_s,
    )
    retry_unreal_op(
        "set_appearance.BP_Character_21",
        lambda: env.unwrapped.unrealcv.set_appearance("BP_Character_21", 0),
        retries,
        retry_sleep_s,
    )
    retry_unreal_op(
        "set_obj_rotation.BP_Character_21",
        lambda: env.unwrapped.unrealcv.set_obj_rotation("BP_Character_21", [0, 0, 0]),
        retries,
        retry_sleep_s,
    )
    time.sleep(SLEEP_SHORT_S)
    retry_unreal_op(
        "new_obj.BP_Character_22",
        lambda: env.unwrapped.unrealcv.new_obj("BP_BaseCar_C", "BP_Character_22", [1000, 0, 0]),
        retries,
        retry_sleep_s,
    )
    retry_unreal_op(
        "set_appearance.BP_Character_22",
        lambda: env.unwrapped.unrealcv.set_appearance("BP_Character_22", 2),
        retries,
        retry_sleep_s,
    )
    retry_unreal_op(
        "set_obj_rotation.BP_Character_22",
        lambda: env.unwrapped.unrealcv.set_obj_rotation("BP_Character_22", [0, 0, 0]),
        retries,
        retry_sleep_s,
    )
    retry_unreal_op(
        "set_phy.BP_Character_22",
        lambda: env.unwrapped.unrealcv.set_phy("BP_Character_22", 0),
        retries,
        retry_sleep_s,
    )
    time.sleep(SLEEP_SHORT_S)


def ensure_env_ready(
    env: Optional[Any], args: argparse.Namespace, force_recreate: bool = False
) -> Any:
    """Return a healthy env, recreating it if requested or if health checks fail."""
    if force_recreate:
        close_env_quietly(env)
        env = None
    if env is not None and probe_env(env, args.unreal_op_retries, args.unreal_retry_sleep):
        return env
    close_env_quietly(env)
    last_error: Optional[Exception] = None
    total_attempts = max(1, args.env_boot_retries + 1)
    for attempt in range(1, total_attempts + 1):
        try:
            logger.info(f"Creating env attempt {attempt}/{total_attempts}")
            env = build_env(args)
            initialize_runtime_objects(env, args.unreal_op_retries, args.unreal_retry_sleep)
            return env
        except Exception as e:
            last_error = e
            logger.warning(f"Env creation failed on attempt {attempt}/{total_attempts}: {e}")
            close_env_quietly(env)
            env = None
            kill_stale_ue_processes()
            if attempt < total_attempts:
                time.sleep(args.env_retry_sleep)
    if last_error is not None:
        raise RuntimeError(f"Unable to create env after {total_attempts} attempts: {last_error}")
    raise RuntimeError("Unable to create env for an unknown reason")


def send_prediction_request(
    image: Image, proprio: np.ndarray, instr: str, server_url: str
) -> Optional[Dict[str, Any]]:
    """Send a request to the inference service and return JSON response.

    Args:
        image: PIL image object, resized to 224x224 and sent as PNG.
        proprio: Vehicle state vector (np.ndarray), converted to list.
        instr: Text instruction.
        server_url: Inference service /predict endpoint URL.
    Returns:
        dict or None: Parsed JSON if successful, otherwise None on error.
    """
    proprio_list = proprio.tolist()
    img_io = BytesIO()
    if image.size != IMG_INPUT_SIZE:
        image = image.resize(IMG_INPUT_SIZE)
    image.save(img_io, format="PNG")
    img_data = img_io.getvalue()
    img_base64 = base64.b64encode(img_data).decode("utf-8")
    payload: Dict[str, Any] = {"image": img_base64, "proprio": proprio_list, "instr": instr}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(
            server_url,
            data=json.dumps(payload),
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return None


def draw_2d_trajectory_from_log(
    log_path: str,
    save_path: str,
    instruction: str,
    target: Optional[List[float]],
    tick_x: Optional[float] = None,
    tick_y: Optional[float] = None,
    min_span_x: Optional[float] = None,
    min_span_y: Optional[float] = None,
) -> None:
    """Plot 2D trajectory from log and save the image.

    - Fixed tick step: use tick_x/tick_y to specify grid spacing
    - Minimum span: use min_span_x/min_span_y to expand view when data span is smaller
    """
    with open(log_path, "r") as f:
        log = json.load(f)
    if not isinstance(log, list) or len(log) == 0:
        plt.figure(figsize=TRAJ_IMG_SIZE_2D)
        plt.title(instruction + "\n" + log_path + " (no data)")
        plt.xlabel("Y (right)")
        plt.ylabel("X (forward)")
        ax = plt.gca()
        if tick_y is not None and tick_y > 0:
            ax.xaxis.set_major_locator(MultipleLocator(tick_y))
        if tick_x is not None and tick_x > 0:
            ax.yaxis.set_major_locator(MultipleLocator(tick_x))
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        return
    trajectory = np.array([item["state"][0] + item["state"][1] for item in log])
    if trajectory.ndim != 2 or trajectory.shape[1] < 5:
        plt.figure(figsize=TRAJ_IMG_SIZE_2D)
        plt.title(instruction + "\n" + log_path + " (invalid data)")
        plt.xlabel("Y (right)")
        plt.ylabel("X (forward)")
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        return
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    yaw = np.deg2rad(trajectory[:, 4])
    arrow_length = max(PLOT_YAW_ARROW_MIN_LEN_2D, np.sqrt(x.var() + y.var()) // 4)
    dx = np.cos(yaw) * arrow_length
    dy = np.sin(yaw) * arrow_length

    plt.figure(figsize=TRAJ_IMG_SIZE_2D)
    if target is not None:
        plt.scatter(target[1], target[0], color="red", label="target")
    plt.plot(y, x, color="blue", label="trajectory")
    plt.quiver(
        y,
        x,
        dy,
        dx,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="green",
        width=0.003,
        label="yaw",
    )

    ax = plt.gca()
    # Set equal aspect ratio first, then expand to minimum span based on this
    ax.set_aspect("equal", adjustable="box")

    # Expand to minimum span without shrinking: X-axis shows Y, Y-axis shows X
    cur_xmin, cur_xmax = ax.get_xlim()
    cur_ymin, cur_ymax = ax.get_ylim()
    span_y_axis = cur_xmax - cur_xmin
    span_x_axis = cur_ymax - cur_ymin
    if min_span_y is not None and min_span_y > 0 and span_y_axis < min_span_y:
        cx = (cur_xmin + cur_xmax) / 2.0
        half = min_span_y / 2.0
        ax.set_xlim(cx - half, cx + half)
    if min_span_x is not None and min_span_x > 0 and span_x_axis < min_span_x:
        cy = (cur_ymin + cur_ymax) / 2.0
        half = min_span_x / 2.0
        ax.set_ylim(cy - half, cy + half)

    # Fixed tick spacing (grid size)
    if tick_y is not None and tick_y > 0:
        ax.xaxis.set_major_locator(MultipleLocator(tick_y))
    if tick_x is not None and tick_x > 0:
        ax.yaxis.set_major_locator(MultipleLocator(tick_x))

    plt.legend()
    plt.title(instruction + "\n" + log_path)
    plt.xlabel("Y (right)")
    plt.ylabel("X (forward)")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def draw_3d_trajectory_from_log(
    log_path: str,
    save_path: str,
    instruction: str,
    target: Optional[List[float]],
    tick_x: Optional[float] = None,
    tick_y: Optional[float] = None,
    tick_z: Optional[float] = None,
    min_span_x: Optional[float] = None,
    min_span_y: Optional[float] = None,
    min_span_z: Optional[float] = None,
) -> None:
    """Plot 3D trajectory from log and save the image.

    - Fixed tick step: use tick_x/tick_y/tick_z to specify grid spacing
    - Minimum span: use min_span_x/min_span_y/min_span_z to expand view when data span is smaller
    """
    with open(log_path, "r") as f:
        log = json.load(f)
    if not isinstance(log, list) or len(log) == 0:
        fig = plt.figure(figsize=TRAJ_IMG_SIZE_3D)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(f"3D Trajectory with Yaw Direction\nInstruction: {instruction} (no data)")
        ax.set_xlabel("Y (right)")
        ax.set_ylabel("X (forward)")
        ax.set_zlabel("Z (up)")
        if tick_y is not None and tick_y > 0:
            ax.xaxis.set_major_locator(MultipleLocator(tick_y))
        if tick_x is not None and tick_x > 0:
            ax.yaxis.set_major_locator(MultipleLocator(tick_x))
        if tick_z is not None and tick_z > 0:
            ax.zaxis.set_major_locator(MultipleLocator(tick_z))
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return
    trajectory = np.array([item["state"][0] + item["state"][1] for item in log])
    if trajectory.ndim != 2 or trajectory.shape[1] < 5:
        fig = plt.figure(figsize=TRAJ_IMG_SIZE_3D)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(f"3D Trajectory with Yaw Direction\nInstruction: {instruction} (invalid data)")
        ax.set_xlabel("Y (right)")
        ax.set_ylabel("X (forward)")
        ax.set_zlabel("Z (up)")
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]
    yaw = np.deg2rad(trajectory[:, 4])
    arrow_length = max(PLOT_YAW_ARROW_MIN_LEN_3D, np.sqrt(x.var() + y.var()) // 4)
    dx = np.cos(yaw) * arrow_length
    dy = np.sin(yaw) * arrow_length
    dz = np.zeros_like(dx)

    fig = plt.figure(figsize=TRAJ_IMG_SIZE_3D)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(y, x, z, color="blue", label="Trajectory")
    ax.quiver(
        y,
        x,
        z,
        dy,
        dx,
        dz,
        color="green",
        length=1.0,
        normalize=False,
        linewidth=0.5,
        label="Yaw direction",
    )
    ax.scatter(y[0], x[0], z[0], color="blue", s=50)
    ax.text(y[0], x[0], z[0], "Start", color="blue", fontsize=12)
    if target is not None:
        ax.scatter(target[1], target[0], target[2], color="red", s=50)
        ax.text(target[1], target[0], target[2], "Target", color="red", fontsize=12)

    # Expand to minimum span without shrinking. Axis mapping: x-axis shows Y, y-axis shows X
    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()
    cur_zlim = ax.get_zlim()
    span_y = cur_xlim[1] - cur_xlim[0]
    span_x = cur_ylim[1] - cur_ylim[0]
    span_z = cur_zlim[1] - cur_zlim[0]
    if min_span_y is not None and min_span_y > 0 and span_y < min_span_y:
        c = (cur_xlim[0] + cur_xlim[1]) / 2.0
        half = min_span_y / 2.0
        ax.set_xlim(c - half, c + half)
    if min_span_x is not None and min_span_x > 0 and span_x < min_span_x:
        c = (cur_ylim[0] + cur_ylim[1]) / 2.0
        half = min_span_x / 2.0
        ax.set_ylim(c - half, c + half)
    if min_span_z is not None and min_span_z > 0 and span_z < min_span_z:
        c = (cur_zlim[0] + cur_zlim[1]) / 2.0
        half = min_span_z / 2.0
        ax.set_zlim(c - half, c + half)

    # Fixed tick spacing
    if tick_y is not None and tick_y > 0:
        ax.xaxis.set_major_locator(MultipleLocator(tick_y))
    if tick_x is not None and tick_x > 0:
        ax.yaxis.set_major_locator(MultipleLocator(tick_x))
    if tick_z is not None and tick_z > 0:
        ax.zaxis.set_major_locator(MultipleLocator(tick_z))

    # Equal aspect box: use current axis limits to avoid flattening when Z is zero
    _xl = ax.get_xlim()
    _yl = ax.get_ylim()
    _zl = ax.get_zlim()
    ax.set_box_aspect(
        ((max(_xl[1] - _xl[0], 1e-6)), (max(_yl[1] - _yl[0], 1e-6)), (max(_zl[1] - _zl[0], 1e-6)))
    )
    ax.set_title(f"3D Trajectory with Yaw Direction\nInstruction: {instruction}")
    ax.set_xlabel("Y (right)")
    ax.set_ylabel("X (forward)")
    ax.set_zlabel("Z (up)")
    ax.legend()
    ax.grid(True)
    ax.view_init(elev=30, azim=-100)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def control_loop(
    initial_pos: List[float],
    env: Any,
    instruction: str,
    max_steps: Optional[int],
    trajectory_log: List[Dict[str, Any]],
    server_url: str,
    unreal_op_retries: int,
    unreal_retry_sleep: float,
    action_log: Optional[List[Dict[str, Any]]] = None,
    event_log: Optional[List[Dict[str, Any]]] = None,
    task_attempt: Optional[int] = None,
    save_frames_dir: Optional[str] = None,
    drone_cam_id: int = -1,
    orbit_reference: Optional[Any] = None,
    debug_image_path: Optional[str] = None,
) -> bool:
    """Main control loop: capture image/state, call inference, act in env, log trajectory."""
    initial_x, initial_y, initial_z = initial_pos[0:3]
    initial_yaw = initial_pos[4]
    retry_unreal_op(
        "control_loop.set_initial_location",
        lambda: env.unwrapped.unrealcv.set_obj_location(
            env.unwrapped.player_list[0], initial_pos[0:3]
        ),
        unreal_op_retries,
        unreal_retry_sleep,
    )
    retry_unreal_op(
        "control_loop.set_initial_rotation",
        lambda: env.unwrapped.unrealcv.set_rotation(
            env.unwrapped.player_list[0], initial_pos[4] - 180
        ),
        unreal_op_retries,
        unreal_retry_sleep,
    )
    if drone_cam_id < 0:
        drone_cam_id = resolve_drone_cam_id(env, unreal_op_retries, unreal_retry_sleep)
    time.sleep(SLEEP_AFTER_RESET_S)
    image = retry_unreal_op(
        "control_loop.get_initial_image",
        lambda: env.unwrapped.unrealcv.get_image(drone_cam_id, "lit"),
        unreal_op_retries,
        unreal_retry_sleep,
    )

    current_pose: List[float] = [0, 0, 0, 0]
    logger.info("Start control loop!")
    last_pose: Optional[List[float]] = None
    small_count = 0
    step_count = 0
    inference_count = 0

    def transform_to_global(x: float, y: float, initial_yaw: float) -> Tuple[float, float]:
        theta = np.radians(initial_yaw)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        global_x = x * cos_theta - y * sin_theta
        global_y = x * sin_theta + y * cos_theta
        return global_x, global_y

    def normalize_angle(angle: float) -> float:
        angle = angle % 360
        if angle > 180:
            angle -= 360
        return angle

    try:
        while True:
            logger.debug(f"current_pose: {current_pose}")
            inference_count += 1
            request_started_at = local_timestamp()
            if save_frames_dir:
                frame_path = os.path.join(save_frames_dir, f"frame_{inference_count:04d}.png")
                try:
                    cv2.imwrite(frame_path, image)
                except Exception as e:
                    logger.debug(f"Failed to save frame {frame_path}: {e}")
            t1 = time.time()
            response = send_prediction_request(
                image=Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
                proprio=np.array(current_pose),
                instr=instruction,
                server_url=server_url,
            )
            t2 = time.time()
            inference_time_s = round(float(t2 - t1), 6)
            # logger.info(f"Prediction time: {t2 - t1} seconds")
            if response is None:
                append_task_event(
                    event_log,
                    "prediction_request_failed",
                    attempt=task_attempt,
                    inference_index=inference_count,
                    request_started_at=request_started_at,
                    inference_time_s=inference_time_s,
                    current_pose=current_pose,
                )
                logger.warning("No valid response, ending control")
                return False

            try:
                action_poses = response.get("action")
                # print(action_poses)
                if not isinstance(action_poses, list) or len(action_poses) == 0:
                    append_task_event(
                        event_log,
                        "invalid_action_response",
                        attempt=task_attempt,
                        inference_index=inference_count,
                        request_started_at=request_started_at,
                        inference_time_s=inference_time_s,
                        response=response,
                    )
                    logger.warning("Response 'action' is empty or invalid, stopping.")
                    return False
                for i, action_pose in enumerate(action_poses):
                    if max_steps is not None and step_count >= max_steps:
                        break
                    if not (isinstance(action_pose, (list, tuple)) and len(action_pose) >= 4):
                        append_task_event(
                            event_log,
                            "invalid_action_element",
                            attempt=task_attempt,
                            inference_index=inference_count,
                            action_index_in_response=i,
                            raw_action=action_pose,
                        )
                        logger.warning(f"Invalid action element at {i}: {action_pose}")
                        continue
                    relative_x, relative_y = float(action_pose[0]), float(action_pose[1])
                    relative_z = float(action_pose[2])
                    relative_yaw = float(np.degrees(action_pose[3]))
                    relative_yaw = (relative_yaw + 180) % 360 - 180
                    global_x, global_y = transform_to_global(relative_x, relative_y, initial_yaw)
                    absolute_yaw = normalize_angle(relative_yaw + initial_yaw)
                    absolute_pos = [
                        global_x + initial_x,
                        global_y + initial_y,
                        relative_z + initial_z,
                        absolute_yaw,
                    ]
                    if action_log is not None:
                        action_log.append(
                            {
                                "timestamp": local_timestamp(),
                                "attempt": task_attempt,
                                "inference_index": inference_count,
                                "action_index_in_response": i,
                                "step_index": step_count + 1,
                                "request_started_at": request_started_at,
                                "inference_time_s": inference_time_s,
                                "proprio_before_request": current_pose,
                                "raw_action": list(action_pose),
                                "relative_pose": {
                                    "x": relative_x,
                                    "y": relative_y,
                                    "z": relative_z,
                                    "yaw_deg": relative_yaw,
                                },
                                "absolute_pose": {
                                    "x": absolute_pos[0],
                                    "y": absolute_pos[1],
                                    "z": absolute_pos[2],
                                    "yaw_deg": absolute_pos[3],
                                },
                                "response_done": response.get("done"),
                            }
                        )
                    retry_unreal_op(
                        "control_loop.set_obj_location",
                        lambda pos=absolute_pos[:3]: env.unwrapped.unrealcv.set_obj_location(
                            env.unwrapped.player_list[0], pos
                        ),
                        unreal_op_retries,
                        unreal_retry_sleep,
                    )
                    retry_unreal_op(
                        "control_loop.set_rotation",
                        lambda yaw=absolute_pos[3] - 180: env.unwrapped.unrealcv.set_rotation(
                            env.unwrapped.player_list[0], yaw
                        ),
                        unreal_op_retries,
                        unreal_retry_sleep,
                    )
                    if i == len(action_poses) - 1:
                        current_pose = [relative_x, relative_y, relative_z, relative_yaw]
                        image = retry_unreal_op(
                            "control_loop.get_image",
                            lambda: env.unwrapped.unrealcv.get_image(drone_cam_id, "lit"),
                            unreal_op_retries,
                            unreal_retry_sleep,
                        )
                        try:
                            if debug_image_path is not None:
                                cv2.imwrite(debug_image_path, image)
                        except Exception as e:
                            logger.debug(f"Failed to write debug image: {e}")
                    step_count += 1
                    trajectory_log.append(
                        {"state": [[relative_x, relative_y, relative_z], [0, relative_yaw, 0]]}
                    )
                    pose_now = [relative_x, relative_y, relative_z, relative_yaw]
                    if last_pose is not None:
                        diffs = [abs(a - b) for a, b in zip(pose_now, last_pose)]
                        if (
                            all(d < ACTION_SMALL_DELTA_POS for d in diffs[:3])
                            and diffs[3] < ACTION_SMALL_DELTA_YAW
                        ):
                            small_count += 1
                        else:
                            small_count = 0
                        if small_count >= ACTION_SMALL_STEPS:
                            append_task_event(
                                event_log,
                                "small_motion_stop",
                                attempt=task_attempt,
                                step_index=step_count,
                                consecutive_small_steps=ACTION_SMALL_STEPS,
                            )
                            logger.info(
                                f"Detected x,y,z,yaw continuous {ACTION_SMALL_STEPS} steps change is very small, automatically end task."
                            )
                            return True
                    last_pose = pose_now
                    time.sleep(0.1)
                if orbit_reference is not None and trajectory_log:
                    orbit_metrics = compute_trajectory_loss(
                        [item["state"][0] for item in trajectory_log],
                        orbit_reference,
                        include_progress=True,
                    )
                    if orbit_is_complete(
                        orbit_metrics,
                        loss_threshold=LOSS_THRESHOLD,
                        min_directed_angle_deg=MIN_DIRECTED_ANGLE_DEG,
                    ):
                        append_task_event(
                            event_log,
                            "orbit_complete_stop",
                            attempt=task_attempt,
                            step_index=step_count,
                            directed_net_angle_deg=orbit_metrics["directed_net_angle_deg"],
                            mean_radial_error_cm=orbit_metrics["mean_distance"],
                        )
                        logger.info(
                            "Surround complete: angle=%.2f deg, radial error=%.2f cm",
                            orbit_metrics["directed_net_angle_deg"],
                            orbit_metrics["mean_distance"],
                        )
                        return True
                if max_steps is not None and step_count >= max_steps:
                    append_task_event(
                        event_log,
                        "max_steps_stop",
                        attempt=task_attempt,
                        step_index=step_count,
                        max_steps=max_steps,
                    )
                    logger.info(
                        f"Already inferred {max_steps} steps, automatically switch to next task."
                    )
                    return True
            except Exception as e:
                append_task_event(
                    event_log,
                    "action_execution_error",
                    attempt=task_attempt,
                    inference_index=inference_count,
                    error=str(e),
                )
                logger.error(f"Error executing action: {e}")
                return False
            try:
                if response.get("done") is True:
                    append_task_event(
                        event_log,
                        "server_done_stop",
                        attempt=task_attempt,
                        inference_index=inference_count,
                        step_index=step_count,
                    )
                    logger.info("Server returned done=True. Ending control loop.")
                    return True
            except Exception:
                pass
    except Exception as e:
        append_task_event(
            event_log,
            "control_loop_error",
            attempt=task_attempt,
            error=str(e),
        )
        logger.error(f"Control loop error: {e}")
        return False


def set_cam(env: Any, retries: int, retry_sleep_s: float) -> None:
    """Compute and set camera pose based on current object pose."""
    x, y, z = retry_unreal_op(
        "set_cam.get_obj_location",
        lambda: env.unwrapped.unrealcv.get_obj_location(env.unwrapped.player_list[0]),
        retries,
        retry_sleep_s,
    )
    roll, yaw, pitch = retry_unreal_op(
        "set_cam.get_obj_rotation",
        lambda: env.unwrapped.unrealcv.get_obj_rotation(env.unwrapped.player_list[0]),
        retries,
        retry_sleep_s,
    )

    cam_loc = [x, y, z]
    cam_rot = [roll, pitch, yaw]

    retry_unreal_op(
        "set_cam.set_cam",
        lambda: env.unwrapped.unrealcv.set_cam(0, cam_loc, cam_rot),
        retries,
        retry_sleep_s,
    )


def resolve_drone_cam_id(
    env: Any, retries: int, retry_sleep_s: float, max_dist_cm: float = 200.0
) -> int:
    """Return the id of the camera bound to the drone (the camera closest to the drone).

    Camera indices are not stable across env boots, and `vbp 0 set_cam` is a no-op,
    so resolve the bound camera geometrically after every task reset.
    """
    drone_loc = retry_unreal_op(
        "resolve_drone_cam.get_obj_location",
        lambda: env.unwrapped.unrealcv.get_obj_location(env.unwrapped.player_list[0]),
        retries,
        retry_sleep_s,
    )
    cam_cfg = retry_unreal_op(
        "resolve_drone_cam.get_camera_config",
        lambda: env.unwrapped.unrealcv.get_camera_config(),
        retries,
        retry_sleep_s,
    )
    best_id: Optional[int] = None
    best_dist: Optional[float] = None
    drone_xyz = np.asarray(drone_loc, dtype=np.float64)
    for cid, cfg in cam_cfg.items():
        dist = float(np.linalg.norm(np.asarray(cfg["location"], dtype=np.float64) - drone_xyz))
        if best_dist is None or dist < best_dist:
            best_id, best_dist = cid, dist
    logger.info(
        f"Drone camera resolved: cam_id={best_id}, distance={best_dist:.1f}cm (drone at {drone_loc})"
    )
    if best_id is None or best_dist is None or best_dist > max_dist_cm:
        raise RuntimeError(f"Failed to resolve drone-bound camera (best distance: {best_dist})")
    return int(best_id)


def create_obj_if_needed(
    env: Any, obj_info: Optional[Dict[str, Any]], retries: int, retry_sleep_s: float
) -> None:
    """Create or place objects in the scene if needed."""
    if obj_info is None:
        return
    use_obj = obj_info.get("use_obj", None)
    obj_id = obj_info.get("obj_id", None)
    obj_pos = obj_info.get("obj_pos", None)
    obj_rot = obj_info.get("obj_rot", None)
    if use_obj == 1:
        retry_unreal_op(
            "create_obj.set_appearance.BP_Character_21",
            lambda: env.unwrapped.unrealcv.set_appearance("BP_Character_21", obj_id),
            retries,
            retry_sleep_s,
        )
        retry_unreal_op(
            "create_obj.set_obj_location.BP_Character_21",
            lambda: env.unwrapped.unrealcv.set_obj_location("BP_Character_21", obj_pos),
            retries,
            retry_sleep_s,
        )
        retry_unreal_op(
            "create_obj.set_obj_rotation.BP_Character_21",
            lambda: env.unwrapped.unrealcv.set_obj_rotation("BP_Character_21", obj_rot),
            retries,
            retry_sleep_s,
        )
        retry_unreal_op(
            "create_obj.hide.BP_Character_22",
            lambda: env.unwrapped.unrealcv.set_obj_location("BP_Character_22", [0, 0, -1000]),
            retries,
            retry_sleep_s,
        )
        retry_unreal_op(
            "create_obj.set_obj_location.BP_Character_21.final",
            lambda: env.unwrapped.unrealcv.set_obj_location("BP_Character_21", obj_pos),
            retries,
            retry_sleep_s,
        )
    elif use_obj == 2:
        retry_unreal_op(
            "create_obj.set_appearance.BP_Character_22",
            lambda: env.unwrapped.unrealcv.set_appearance("BP_Character_22", 2),
            retries,
            retry_sleep_s,
        )
        retry_unreal_op(
            "create_obj.set_obj_location.BP_Character_22",
            lambda: env.unwrapped.unrealcv.set_obj_location(
                "BP_Character_22", [obj_pos[0], obj_pos[1], 0]
            ),
            retries,
            retry_sleep_s,
        )
        retry_unreal_op(
            "create_obj.set_obj_rotation.BP_Character_22",
            lambda: env.unwrapped.unrealcv.set_obj_rotation("BP_Character_22", obj_rot),
            retries,
            retry_sleep_s,
        )
        retry_unreal_op(
            "create_obj.set_phy.BP_Character_22",
            lambda: env.unwrapped.unrealcv.set_phy("BP_Character_22", 0),
            retries,
            retry_sleep_s,
        )
        retry_unreal_op(
            "create_obj.hide.BP_Character_21",
            lambda: env.unwrapped.unrealcv.set_obj_location("BP_Character_21", [0, 0, -1000]),
            retries,
            retry_sleep_s,
        )
        retry_unreal_op(
            "create_obj.set_obj_location.BP_Character_22.final",
            lambda: env.unwrapped.unrealcv.set_obj_location(
                "BP_Character_22", [obj_pos[0], obj_pos[1], 0]
            ),
            retries,
            retry_sleep_s,
        )

    if use_obj in [1, 2]:
        logger.debug(
            retry_unreal_op(
                "create_obj.get_camera_config",
                lambda: env.unwrapped.unrealcv.get_camera_config(),
                retries,
                retry_sleep_s,
            )
        )
        time.sleep(SLEEP_SHORT_S)


class ModelResetError(RuntimeError):
    pass


def reset_model(server_url: str) -> None:
    """A failed reset aborts the task attempt before any prediction is made."""
    try:
        response = requests.post(server_url.removesuffix("/predict") + "/reset", timeout=10)
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict) and payload.get("status") == "error":
            raise ValueError(payload.get("message", "Model rejected reset"))
    except Exception as error:
        raise ModelResetError(f"Model reset failed: {error}") from error


def world_to_local(target: List[float], init_pos: List[float]) -> List[float]:
    """Convert world-frame target coordinates to the first-frame local frame."""
    x0, y0, z0 = init_pos[0:3]
    yaw0 = init_pos[4]
    dx = target[0] - x0
    dy = target[1] - y0
    dz = target[2] - z0
    theta = -np.radians(yaw0)
    x_rel = dx * np.cos(theta) - dy * np.sin(theta)
    y_rel = dx * np.sin(theta) + dy * np.cos(theta)

    z_rel = dz
    return [x_rel, y_rel, z_rel]


def select_instruction(manual_data: Dict[str, Any], instruction_type: str) -> str:
    """Choose the configured instruction field with compatibility fallback."""
    if instruction_type == "instruction_unified":
        return manual_data.get("instruction_unified", manual_data.get("instruction", ""))
    return manual_data.get("instruction", manual_data.get("instruction_unified", ""))


def select_task_max_steps(filename, default_max_steps, surround_max_steps, categories):
    """Use the benchmark task category rather than matching instruction text."""
    return surround_max_steps if filename in categories.get("Surround", []) else default_max_steps


def _run(args):
    with open(
        args.classification,
        encoding="utf-8",
    ) as handle:
        task_categories = json.load(handle)

    images_dir = str(args.output / "trajectories")
    os.makedirs(images_dir, exist_ok=True)
    save_frames_root = os.path.join(images_dir, "frames") if args.save_frames else ""
    logs_dir = os.path.join(images_dir, "logs")
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_log_path = configure_logging(logs_dir, args.log_level, run_id)
    logger.info(f"Run log saved: {run_log_path}")

    tick_x = args.tick_x if args.tick_x is not None else DEFAULT_TICK_X
    tick_y = args.tick_y if args.tick_y is not None else DEFAULT_TICK_Y
    tick_z = args.tick_z if args.tick_z is not None else DEFAULT_TICK_Z

    min_span_x = args.min_span_x if args.min_span_x is not None else DEFAULT_MIN_SPAN_X
    min_span_y = args.min_span_y if args.min_span_y is not None else DEFAULT_MIN_SPAN_Y
    min_span_z = args.min_span_z if args.min_span_z is not None else DEFAULT_MIN_SPAN_Z

    server_url = (
        args.server_url.rstrip("/") if args.server_url else f"http://127.0.0.1:{args.server_port}"
    )
    if not server_url.endswith("/predict"):
        server_url += "/predict"

    env: Optional[Any] = ensure_env_ready(None, args)
    if args.restart_env_every > 0:
        logger.warning("--restart_env_every is ignored; env will stay alive for the whole batch.")

    json_folder = args.json_folder
    json_files = glob.glob(os.path.join(json_folder, "*.json"))
    json_files.sort(reverse=True)
    logger.info(f"Detected {len(json_files)} json task files")

    execution_failures = 0
    for idx, json_file in enumerate(json_files):
        # if "2025-03-30_11-49-27" not in json_file:
        #     continue

        base_name = os.path.splitext(os.path.basename(json_file))[0]
        logger.info(f"\n===== Start task {idx + 1}/{len(json_files)} file: {json_file} =====")
        with open(json_file, "r", encoding="utf-8") as f:
            manual_data = json.load(f)
        if not isinstance(manual_data, dict):
            execution_failures += 1
            logger.warning(f"Unsupported json format, skipping: {json_file}")
            continue
        # Choose instruction field based on argument (compatible with legacy/new format)
        instruction = select_instruction(manual_data, args.instruction_type)
        is_surround = os.path.basename(json_file) in task_categories.get("Surround", [])
        effective_max_steps = select_task_max_steps(
            os.path.basename(json_file), args.max_steps, args.surround_max_steps, task_categories
        )
        initial_pos = manual_data.get("initial_pos", None)
        target_pos = manual_data.get("target_pos", None)
        # Only create objects when both obj_id and use_obj are provided
        obj_info = None
        if "obj_id" in manual_data and "use_obj" in manual_data:
            if (
                "target_pos" in manual_data
                and isinstance(manual_data["target_pos"], list)
                and len(manual_data["target_pos"]) == 6
            ):
                obj_pos = manual_data["target_pos"][:3]
                obj_rot = manual_data["target_pos"][3:]
            else:
                obj_pos = manual_data.get("obj_pos", None)
                obj_rot = manual_data.get("obj_rot", [0, 0, 0])
            if obj_pos is not None:
                obj_info = {
                    "use_obj": manual_data["use_obj"],
                    "obj_id": manual_data["obj_id"],
                    "obj_pos": obj_pos,
                    "obj_rot": obj_rot,
                }
        logger.info(f"instruction: {instruction}")
        if not initial_pos or len(initial_pos) < 5:
            execution_failures += 1
            logger.error("Invalid or missing 'initial_pos' in task json; skipping.")
            continue
        orbit_reference = None
        if is_surround:
            reference_path = manual_data.get("reference_path_raw", [])
            if not reference_path:
                raise ValueError(f"Surround task has no reference trajectory: {json_file}")
            orbit_reference = [world_to_local(point[:3], initial_pos) for point in reference_path]
        logger.info(
            "Task category=%s, maximum action points=%s",
            "Surround" if is_surround else "other",
            effective_max_steps,
        )
        # Transform target_pos to the coordinate system of the first frame.
        target_local = None
        if (
            isinstance(target_pos, list)
            and len(target_pos) >= 3
            and initial_pos is not None
            and len(initial_pos) > 0
        ):
            target_local = world_to_local(target_pos, initial_pos)

        task_action_log_path = os.path.join(logs_dir, f"{base_name}_{run_id}_actions.json")
        task_action_log: Dict[str, Any] = {
            "run_id": run_id,
            "task_index": idx + 1,
            "task_count": len(json_files),
            "task_name": base_name,
            "task_file": json_file,
            "instruction_type": args.instruction_type,
            "instruction": instruction,
            "task_category": "Surround" if is_surround else "other",
            "effective_max_steps": effective_max_steps,
            "initial_pos": initial_pos,
            "target_pos": target_pos,
            "target_local": target_local,
            "obj_info": obj_info,
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "server_url": server_url,
            "started_at": local_timestamp(),
            "attempts": [],
        }
        trajectory_log: List[Dict[str, Any]] = []
        task_success = False
        total_task_attempts = max(1, args.task_retries + 1)
        for task_attempt in range(1, total_task_attempts + 1):
            if env is None:
                raise RuntimeError("Env became unavailable and automatic restart is disabled")

            attempt_log: List[Dict[str, Any]] = []
            attempt_action_log: Dict[str, Any] = {
                "attempt": task_attempt,
                "started_at": local_timestamp(),
                "actions": [],
                "events": [],
            }
            task_action_log["attempts"].append(attempt_action_log)
            append_task_event(
                attempt_action_log["events"],
                "task_attempt_start",
                attempt=task_attempt,
                total_attempts=total_task_attempts,
            )
            try:
                reset_model(server_url)
                create_obj_if_needed(env, obj_info, args.unreal_op_retries, args.unreal_retry_sleep)
                time.sleep(SLEEP_SHORT_S)
                attempt_frames_dir = None
                if save_frames_root:
                    attempt_frames_dir = os.path.join(
                        save_frames_root, base_name, f"attempt_{task_attempt}"
                    )
                    os.makedirs(attempt_frames_dir, exist_ok=True)
                task_success = control_loop(
                    initial_pos,
                    env=env,
                    instruction=instruction,
                    max_steps=effective_max_steps,
                    trajectory_log=attempt_log,
                    server_url=server_url,
                    unreal_op_retries=args.unreal_op_retries,
                    unreal_retry_sleep=args.unreal_retry_sleep,
                    action_log=attempt_action_log["actions"],
                    event_log=attempt_action_log["events"],
                    task_attempt=task_attempt,
                    save_frames_dir=attempt_frames_dir,
                    drone_cam_id=args.drone_cam_id,
                    orbit_reference=orbit_reference,
                    debug_image_path=str(args.output / "debug.jpg"),
                )
            except Exception as e:
                append_task_event(
                    attempt_action_log["events"],
                    "task_attempt_crashed",
                    attempt=task_attempt,
                    error=str(e),
                )
                logger.error(f"Task attempt {task_attempt}/{total_task_attempts} crashed: {e}")
                task_success = False

            trajectory_log = attempt_log
            attempt_action_log["finished_at"] = local_timestamp()
            attempt_action_log["success"] = task_success
            attempt_action_log["trajectory_steps"] = len(attempt_log)
            if task_success:
                break

            if task_attempt < total_task_attempts:
                append_task_event(
                    attempt_action_log["events"],
                    "task_attempt_retry",
                    attempt=task_attempt,
                    retry_sleep_s=args.env_retry_sleep,
                )
                logger.warning(
                    f"Task {base_name} failed on attempt {task_attempt}/{total_task_attempts}. "
                    f"Retrying in the same env."
                )
                time.sleep(args.env_retry_sleep)

        if not task_success:
            execution_failures += 1
            logger.error(
                f"Task {base_name} exhausted retries and will be saved with the latest partial trajectory."
            )
        traj_json_path = os.path.join(images_dir, base_name + ".json")
        raw_trajectory_steps = len(trajectory_log)
        trimmed_tail_steps = 0
        # If the task stopped before max_steps, drop the final repeated small-action tail.
        if not is_surround and len(trajectory_log) != effective_max_steps:
            trajectory_log = trajectory_log[:-ACTION_SMALL_STEPS]
            trimmed_tail_steps = raw_trajectory_steps - len(trajectory_log)
        with open(traj_json_path, "w", encoding="utf-8") as f:
            json.dump(trajectory_log, f, indent=2)
        plot_2d_path = os.path.join(images_dir, base_name + "_2d.png")
        plot_3d_path = os.path.join(images_dir, base_name + "_3d.png")
        plotting_error = None
        try:
            draw_2d_trajectory_from_log(
                traj_json_path,
                plot_2d_path,
                instruction,
                target_local,
                tick_x=tick_x,
                tick_y=tick_y,
                min_span_x=min_span_x,
                min_span_y=min_span_y,
            )
            draw_3d_trajectory_from_log(
                traj_json_path,
                plot_3d_path,
                instruction,
                target_local,
                tick_x=tick_x,
                tick_y=tick_y,
                tick_z=tick_z,
                min_span_x=min_span_x,
                min_span_y=min_span_y,
                min_span_z=min_span_z,
            )
            logger.info(f"Trajectory and images saved: {base_name}")
        except Exception as e:
            plotting_error = str(e)
            logger.error(f"Plotting failed: {e}")
        task_action_log["finished_at"] = local_timestamp()
        termination_reason = next(
            (
                event["event"]
                for event in reversed(task_action_log["attempts"][-1]["events"])
                if event["event"].endswith("_stop") or event["event"].endswith("_error")
            ),
            "execution_finished" if task_success else "execution_failed",
        )
        orbit_metrics = None
        semantic_success = task_success
        if is_surround:
            semantic_success = False
            if trajectory_log:
                metrics = compute_trajectory_loss(
                    [item["state"][0] for item in trajectory_log],
                    orbit_reference,
                    include_progress=True,
                )
                semantic_success = orbit_is_complete(
                    metrics,
                    loss_threshold=LOSS_THRESHOLD,
                    min_directed_angle_deg=MIN_DIRECTED_ANGLE_DEG,
                )
                orbit_metrics = {
                    "mean_radial_error_cm": float(metrics["mean_distance"]),
                    "directed_net_angle_deg": float(metrics["directed_net_angle_deg"]),
                }
        task_action_log["final"] = {
            "success": semantic_success,
            "execution_ok": task_success,
            "termination_reason": termination_reason,
            "truncated": termination_reason == "max_steps_stop",
            "effective_max_steps": effective_max_steps,
            "orbit_metrics": orbit_metrics,
            "raw_trajectory_steps": raw_trajectory_steps,
            "saved_trajectory_steps": len(trajectory_log),
            "trimmed_tail_steps": trimmed_tail_steps,
            "trajectory_path": traj_json_path,
            "plot_2d_path": plot_2d_path,
            "plot_3d_path": plot_3d_path,
            "plotting_error": plotting_error,
            "action_log_path": task_action_log_path,
        }
        try:
            with open(task_action_log_path, "w", encoding="utf-8") as f:
                json.dump(task_action_log, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Action log saved: {task_action_log_path}")
        except Exception as e:
            logger.error(f"Failed to save action log: {e}")
        logger.info(f"===== Task {idx + 1} finished =====\n")
    close_env_quietly(env)
    return 1 if execution_failures else 0


def main(argv=None):
    args = parse_args(argv) if not isinstance(argv, argparse.Namespace) else argv
    args.output = args.output.resolve()
    with output_context(args.output, change_cwd=True):
        return _run(args)


if __name__ == "__main__":
    raise SystemExit(main(_cli_args))
