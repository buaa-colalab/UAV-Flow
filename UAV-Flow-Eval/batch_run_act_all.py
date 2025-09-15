import base64
import requests
import sys
sys.path.append('./')
from io import BytesIO
import json
import numpy as np
from PIL import Image
import gym
import gym_unrealcv
import time
import argparse
import os
import cv2
from relative import calculate_new_pose
glob = __import__('glob')
from gym_unrealcv.envs.wrappers import time_dilation, configUE, augmentation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
from typing import Any, Dict, List, Optional, Tuple
from matplotlib.ticker import MultipleLocator
logger = logging.getLogger(__name__)

# ====== Constants ======
IMG_INPUT_SIZE: Tuple[int, int] = (224, 224)
SLEEP_SHORT_S: float = 1.0
SLEEP_AFTER_RESET_S: float = 2.0
ACTION_SMALL_DELTA_POS: float = 3.0
ACTION_SMALL_DELTA_YAW: float = 1.0
ACTION_SMALL_STEPS: int = 10
DEBUG_IMAGE_PATH: str = './debug.jpg'
TRAJ_IMG_SIZE_2D: Tuple[int, int] = (10, 10)
TRAJ_IMG_SIZE_3D: Tuple[int, int] = (12, 10)
PLOT_YAW_ARROW_MIN_LEN_2D: int = 10
PLOT_YAW_ARROW_MIN_LEN_3D: int = 12



def send_prediction_request(image: Image, proprio: np.ndarray, instr: str, server_url: str) -> Optional[Dict[str, Any]]:
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
    image.save(img_io, format='PNG')
    img_data = img_io.getvalue()
    img_base64 = base64.b64encode(img_data).decode('utf-8')
    payload: Dict[str, Any] = {
        'image': img_base64,
        'proprio': proprio_list,
        'instr': instr
    }
    headers = {
        'Content-Type': 'application/json'
    }
    try:
        response = requests.post(
            server_url,
            data=json.dumps(payload),
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return None


def draw_2d_trajectory_from_log(log_path: str, save_path: str, instruction: str, target: Optional[List[float]], tick_x: Optional[float] = None, tick_y: Optional[float] = None, min_span_x: Optional[float] = None, min_span_y: Optional[float] = None) -> None:
    """Plot 2D trajectory from log and save the image.

    - Fixed tick step: use tick_x/tick_y to specify grid spacing
    - Minimum span: use min_span_x/min_span_y to expand view when data span is smaller
    """
    with open(log_path, 'r') as f:
        log = json.load(f)
    if not isinstance(log, list) or len(log) == 0:
        plt.figure(figsize=TRAJ_IMG_SIZE_2D)
        plt.title(instruction + '\n' + log_path + ' (no data)')
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
    trajectory = np.array([item['state'][0] + item['state'][1] for item in log])
    if trajectory.ndim != 2 or trajectory.shape[1] < 5:
        plt.figure(figsize=TRAJ_IMG_SIZE_2D)
        plt.title(instruction + '\n' + log_path + ' (invalid data)')
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
        plt.scatter(target[1], target[0], color='red', label='target')
    plt.plot(y, x, color='blue', label='trajectory')
    plt.quiver(y, x, dy, dx, angles='xy', scale_units='xy', scale=1, color='green', width=0.003, label='yaw')

    ax = plt.gca()
    # Set equal aspect ratio first, then expand to minimum span based on this
    ax.set_aspect('equal', adjustable='box')
    
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
    plt.title(instruction + '\n' + log_path)
    plt.xlabel("Y (right)")
    plt.ylabel("X (forward)")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def draw_3d_trajectory_from_log(log_path: str, save_path: str, instruction: str, target: Optional[List[float]], tick_x: Optional[float] = None, tick_y: Optional[float] = None, tick_z: Optional[float] = None, min_span_x: Optional[float] = None, min_span_y: Optional[float] = None, min_span_z: Optional[float] = None) -> None:
    """Plot 3D trajectory from log and save the image.

    - Fixed tick step: use tick_x/tick_y/tick_z to specify grid spacing
    - Minimum span: use min_span_x/min_span_y/min_span_z to expand view when data span is smaller
    """
    with open(log_path, 'r') as f:
        log = json.load(f)
    if not isinstance(log, list) or len(log) == 0:
        fig = plt.figure(figsize=TRAJ_IMG_SIZE_3D)
        ax = fig.add_subplot(111, projection='3d')
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
    trajectory = np.array([item['state'][0] + item['state'][1] for item in log])
    if trajectory.ndim != 2 or trajectory.shape[1] < 5:
        fig = plt.figure(figsize=TRAJ_IMG_SIZE_3D)
        ax = fig.add_subplot(111, projection='3d')
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
    arrow_length =  max(PLOT_YAW_ARROW_MIN_LEN_3D, np.sqrt(x.var() + y.var()) // 4)
    dx = np.cos(yaw) * arrow_length
    dy = np.sin(yaw) * arrow_length
    dz = np.zeros_like(dx)

    fig = plt.figure(figsize=TRAJ_IMG_SIZE_3D)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(y, x, z, color='blue', label='Trajectory')
    ax.quiver(y, x, z, dy, dx, dz, color='green', length=1.0, normalize=False, linewidth=0.5, label='Yaw direction')
    ax.scatter(y[0], x[0], z[0], color='blue', s=50)
    ax.text(y[0], x[0], z[0], 'Start', color='blue', fontsize=12)
    if target is not None:
        ax.scatter(target[1], target[0], target[2], color='red', s=50)
        ax.text(target[1], target[0], target[2], 'Target', color='red', fontsize=12)

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
    _xl = ax.get_xlim(); _yl = ax.get_ylim(); _zl = ax.get_zlim()
    ax.set_box_aspect(((max(_xl[1]-_xl[0], 1e-6)), (max(_yl[1]-_yl[0], 1e-6)), (max(_zl[1]-_zl[0], 1e-6))))
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


def control_loop(initial_pos: List[float], env: Any, instruction: str, max_steps: Optional[int], trajectory_log: List[Dict[str, Any]], server_url: str) -> None:
    """Main control loop: capture image/state, call inference, act in env, log trajectory."""
    initial_x, initial_y, initial_z = initial_pos[0:3]
    initial_yaw = initial_pos[4]
    env.unwrapped.unrealcv.set_obj_location(env.unwrapped.player_list[0], initial_pos[0:3])
    env.unwrapped.unrealcv.set_rotation(env.unwrapped.player_list[0], initial_pos[4] - 180)
    set_cam(env)
    time.sleep(SLEEP_AFTER_RESET_S)
    image = env.unwrapped.unrealcv.get_image(0, 'lit')
    
    current_pose: List[float] = [0, 0, 0, 0]
    logger.info('Start control loop!')
    last_pose: Optional[List[float]] = None
    small_count = 0
    step_count = 0

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
            set_cam(env)
            logger.debug(f"current_pose: {current_pose}")
            t1 = time.time()
            response = send_prediction_request(
                image=Image.fromarray(image),
                proprio=np.array(current_pose),
                instr=instruction,
                server_url=server_url
            )
            t2 = time.time()
            # logger.info(f"Prediction time: {t2 - t1} seconds")
            if response is None:
                logger.warning("No valid response, ending control")
                break
            
            try:
                action_poses = response.get('action')
                if not isinstance(action_poses, list) or len(action_poses) == 0:
                    logger.warning("Response 'action' is empty or invalid, stopping.")
                    break
                for i, action_pose in enumerate(action_poses):
                    if not (isinstance(action_pose, (list, tuple)) and len(action_pose) >= 4):
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
                        absolute_yaw
                    ]
                    env.unwrapped.unrealcv.set_obj_location(env.unwrapped.player_list[0], absolute_pos[:3])
                    env.unwrapped.unrealcv.set_rotation(env.unwrapped.player_list[0], absolute_pos[3] - 180)
                    set_cam(env)
                    if i == len(action_poses) - 1:
                        current_pose = [relative_x, relative_y, relative_z, relative_yaw]
                        image = env.unwrapped.unrealcv.get_image(0, 'lit')
                        try:
                            cv2.imwrite(DEBUG_IMAGE_PATH, image)
                        except Exception as e:
                            logger.debug(f"Failed to write debug image: {e}")
                    step_count += 1
                    trajectory_log.append({'state': [[relative_x, relative_y, relative_z], [0, relative_yaw, 0]]})
                    pose_now = [relative_x, relative_y, relative_z, relative_yaw]
                    if last_pose is not None:
                        diffs = [abs(a - b) for a, b in zip(pose_now, last_pose)]
                        if all(d < ACTION_SMALL_DELTA_POS for d in diffs[:3]) and diffs[3] < ACTION_SMALL_DELTA_YAW:
                            small_count += 1
                        else:
                            small_count = 0
                        if small_count >= ACTION_SMALL_STEPS:
                            logger.info(f"Detected x,y,z,yaw continuous {ACTION_SMALL_STEPS} steps change is very small, automatically end task.")
                            return
                    last_pose = pose_now
                    time.sleep(0.1)
                if max_steps is not None and step_count >= max_steps:
                    logger.info(f"Already inferred {max_steps} steps, automatically switch to next task.")
                    break
            except Exception as e:
                logger.error(f"Error executing action: {e}")
                break
            try:
                if response.get('done') is True:
                    logger.info("Server returned done=True. Ending control loop.")
                    return
            except Exception:
                pass
    except Exception as e:
        logger.error(f"Control loop error: {e}")


def set_cam(env: Any) -> None:
    """Compute and set camera pose based on current object pose."""
    x, y, z = env.unwrapped.unrealcv.get_obj_location(env.unwrapped.player_list[0])
    roll, yaw, pitch = env.unwrapped.unrealcv.get_obj_rotation(env.unwrapped.player_list[0])

    cam_loc = [x, y, z]
    cam_rot = [roll, pitch, yaw]
    
    env.unwrapped.unrealcv.set_cam(0, cam_loc, cam_rot)


def create_obj_if_needed(env: Any, obj_info: Optional[Dict[str, Any]]) -> None:
    """Create or place objects in the scene if needed."""
    if obj_info is None:
        return
    use_obj = obj_info.get('use_obj', None)
    obj_id = obj_info.get('obj_id', None)
    obj_pos = obj_info.get('obj_pos', None)
    obj_rot = obj_info.get('obj_rot', None)
    if use_obj == 1:
        env.unwrapped.unrealcv.set_appearance("BP_Character_21", obj_id)
        env.unwrapped.unrealcv.set_obj_location("BP_Character_21", obj_pos)
        env.unwrapped.unrealcv.set_obj_rotation("BP_Character_21", obj_rot)
        env.unwrapped.unrealcv.set_obj_location("BP_Character_22", [0, 0, -1000])
        env.unwrapped.unrealcv.set_obj_location("BP_Character_21", obj_pos)
    elif use_obj == 2:
        env.unwrapped.unrealcv.set_appearance("BP_Character_22", 2)
        env.unwrapped.unrealcv.set_obj_location("BP_Character_22", [obj_pos[0], obj_pos[1], 0])
        env.unwrapped.unrealcv.set_obj_rotation("BP_Character_22", obj_rot)
        env.unwrapped.unrealcv.set_phy("BP_Character_22", 0)
        env.unwrapped.unrealcv.set_obj_location("BP_Character_21", [0, 0, -1000])
        env.unwrapped.unrealcv.set_obj_location("BP_Character_22", [obj_pos[0], obj_pos[1], 0])

    if use_obj in [1, 2]:
        logger.debug(env.unwrapped.unrealcv.get_camera_config())
        time.sleep(SLEEP_SHORT_S)


def reset_model(server_url: str) -> None:
    """Call server /reset to reset the model."""
    try:
        resp = requests.post(server_url.replace('/predict', '/reset'), timeout=10)
        logger.info(f"Model reset response: {resp.status_code}")
    except Exception as e:
        logger.error(f"Model reset failed: {e}")


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

if __name__ == '__main__':
    # ====== Default parameters ======
    DEFAULT_ENV_ID = 'UnrealTrack-DowntownWest-ContinuousColor-v0'
    DEFAULT_TIME_DILATION = 10
    DEFAULT_SEED = 0
    DEFAULT_JSON_FOLDER = r'./test_jsons'
    DEFAULT_IMAGES_DIR = f'./results/{DEFAULT_ENV_ID}/openvla'
    DEFAULT_SERVER_PORT = 5007
    DEFAULT_MAX_STEPS = 100
    DEFAULT_INSTRUCTION_TYPE = "instruction"
    # Fixed tick spacing (grid size)
    DEFAULT_TICK_X: Optional[float] = 100
    DEFAULT_TICK_Y: Optional[float] = 100
    DEFAULT_TICK_Z: Optional[float] = 100
    # Minimum display span
    DEFAULT_MIN_SPAN_X: Optional[float] = 400
    DEFAULT_MIN_SPAN_Y: Optional[float] = 400
    DEFAULT_MIN_SPAN_Z: Optional[float] = 400

    import argparse
    parser = argparse.ArgumentParser(description='Batch run UAV control with instruction-conditioned policy')
    parser.add_argument("-e", "--env_id", default=DEFAULT_ENV_ID, help='Environment ID to run')
    parser.add_argument("-t", '--time_dilation', default=DEFAULT_TIME_DILATION, type=int, help='Time dilation parameter to keep FPS in simulator')
    parser.add_argument("-s", '--seed', default=DEFAULT_SEED, type=int, help='Random seed')
    parser.add_argument("-f", '--json_folder', default=DEFAULT_JSON_FOLDER, help='Folder path containing batch task json files')
    parser.add_argument("-o", '--images_dir', default=DEFAULT_IMAGES_DIR, help='Directory to save images and trajectory logs')
    parser.add_argument("-p", '--server_port', default=DEFAULT_SERVER_PORT, type=int, help='Inference server port')
    parser.add_argument("-m", '--max_steps', default=DEFAULT_MAX_STEPS, type=int, help='Maximum inference steps')
    parser.add_argument("-i", "--instruction_type", default=DEFAULT_INSTRUCTION_TYPE, choices=["instruction", "instruction_unified"], help='Choose which field to use: instruction or instruction_unified')
    parser.add_argument('--log_level', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], help='Logging level')
    # Fixed tick spacing (grid size)
    parser.add_argument('--tick_x', type=float, default=None, help='Tick step for X (forward) axis')
    parser.add_argument('--tick_y', type=float, default=None, help='Tick step for Y (right) axis')
    parser.add_argument('--tick_z', type=float, default=None, help='Tick step for Z (up) axis')
    # Minimum display span
    parser.add_argument('--min_span_x', type=float, default=None, help='Minimum span for X (forward) axis')
    parser.add_argument('--min_span_y', type=float, default=None, help='Minimum span for Y (right) axis')
    parser.add_argument('--min_span_z', type=float, default=None, help='Minimum span for Z (up) axis')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s')

    tick_x = args.tick_x if args.tick_x is not None else DEFAULT_TICK_X
    tick_y = args.tick_y if args.tick_y is not None else DEFAULT_TICK_Y
    tick_z = args.tick_z if args.tick_z is not None else DEFAULT_TICK_Z

    min_span_x = args.min_span_x if args.min_span_x is not None else DEFAULT_MIN_SPAN_X
    min_span_y = args.min_span_y if args.min_span_y is not None else DEFAULT_MIN_SPAN_Y
    min_span_z = args.min_span_z if args.min_span_z is not None else DEFAULT_MIN_SPAN_Z

    server_url = f"http://127.0.0.1:{args.server_port}/predict"

    env = gym.make(args.env_id)
    if int(args.time_dilation) > 0:
        env = time_dilation.TimeDilationWrapper(env, int(args.time_dilation))
    env.unwrapped.agents_category = ['drone']
    env = configUE.ConfigUEWrapper(env, resolution=(256, 256))
    env = augmentation.RandomPopulationWrapper(env, 2, 2, random_target=False)
    env.seed(int(args.seed))
    env.reset()
    env.unwrapped.unrealcv.set_viewport(env.unwrapped.player_list[0])
    env.unwrapped.unrealcv.set_phy(env.unwrapped.player_list[0], 0)
    logger.info(env.unwrapped.unrealcv.get_camera_config())
    print(env.unwrapped.unrealcv.get_camera_config())

    json_folder = args.json_folder
    json_files = glob.glob(os.path.join(json_folder, '*.json'))
    logger.info(f"Detected {len(json_files)} json task files")
    images_dir = args.images_dir
    os.makedirs(images_dir, exist_ok=True)

    # init object
    time.sleep(SLEEP_SHORT_S)
    env.unwrapped.unrealcv.new_obj("bp_character_C", "BP_Character_21", [0,0,0])
    env.unwrapped.unrealcv.set_appearance("BP_Character_21", 0)
    env.unwrapped.unrealcv.set_obj_rotation("BP_Character_21",  [0,0,0])
    time.sleep(SLEEP_SHORT_S)
    env.unwrapped.unrealcv.new_obj("BP_BaseCar_C", "BP_Character_22", [1000,0,0])
    env.unwrapped.unrealcv.set_appearance("BP_Character_22", 2)
    env.unwrapped.unrealcv.set_obj_rotation("BP_Character_22", [0,0,0])
    env.unwrapped.unrealcv.set_phy("BP_Character_22", 0)
    time.sleep(SLEEP_SHORT_S)


    for idx, json_file in enumerate(json_files):

        base_name = os.path.splitext(os.path.basename(json_file))[0]
        img2d_path = os.path.join(images_dir, base_name + '_2d.png')
        img3d_path = os.path.join(images_dir, base_name + '_3d.png')
        if os.path.exists(img2d_path) and os.path.exists(img3d_path):
            logger.info(f"{base_name} images already exist, skipping.")
            continue
        logger.info(f"\n===== Start task {idx+1}/{len(json_files)} file: {json_file} =====")
        reset_model(server_url)
        with open(json_file, 'r') as f:
            manual_data = json.load(f)
        if not isinstance(manual_data, dict):
            logger.warning(f"Unsupported json format, skipping: {json_file}")
            continue
        # Choose instruction field based on argument (compatible with legacy/new format)
        if args.instruction_type == 'instruction_unified':
            instruction = manual_data.get('instruction_unified', manual_data.get('instruction', ''))
        else:
            instruction = manual_data.get('instruction', manual_data.get('instruction_unified', ''))
        initial_pos = manual_data.get('initial_pos', None)
        target_pos = manual_data.get('target_pos', None)
        # Only create objects when both obj_id and use_obj are provided
        obj_info = None
        if 'obj_id' in manual_data and 'use_obj' in manual_data:
            if 'target_pos' in manual_data and isinstance(manual_data['target_pos'], list) and len(manual_data['target_pos']) == 6:
                obj_pos = manual_data['target_pos'][:3]
                obj_rot = manual_data['target_pos'][3:]
            else:
                obj_pos = manual_data.get('obj_pos', None)
                obj_rot = manual_data.get('obj_rot', [0, 0, 0])
            if obj_pos is not None:
                obj_info = {
                    'use_obj': manual_data['use_obj'],
                    'obj_id': manual_data['obj_id'],
                    'obj_pos': obj_pos,
                    'obj_rot': obj_rot
                }
        create_obj_if_needed(env, obj_info)
        set_cam(env)
        time.sleep(SLEEP_SHORT_S)
        logger.info(f"instruction: {instruction}")
        trajectory_log: List[Dict[str, Any]] = []
        if not initial_pos or len(initial_pos) < 5:
            logger.error("Invalid or missing 'initial_pos' in task json; skipping.")
            continue
        control_loop(
            initial_pos,
            env=env,
            instruction=instruction,
            max_steps=args.max_steps,
            trajectory_log=trajectory_log,
            server_url=server_url
        )
        traj_json_path = os.path.join(images_dir, base_name + '.json')
        with open(traj_json_path, 'w') as f:
            json.dump(trajectory_log, f, indent=2)
        # Transform target_pos to the coordinate system of the first frame
        target_local = None
        if target_pos is not None and initial_pos is not None and len(initial_pos) > 0:
            target_local = world_to_local(target_pos, initial_pos)
        try:
            draw_2d_trajectory_from_log(traj_json_path, os.path.join(images_dir, base_name + '_2d.png'), instruction, target_local, tick_x=tick_x, tick_y=tick_y, min_span_x=min_span_x, min_span_y=min_span_y)
            draw_3d_trajectory_from_log(traj_json_path, os.path.join(images_dir, base_name + '_3d.png'), instruction, target_local, tick_x=tick_x, tick_y=tick_y, tick_z=tick_z, min_span_x=min_span_x, min_span_y=min_span_y, min_span_z=min_span_z)
            logger.info(f"Trajectory and images saved: {base_name}")
        except Exception as e:
            logger.error(f"Plotting failed: {e}")
        logger.info(f"===== Task {idx+1} finished =====\n")
    env.close() 