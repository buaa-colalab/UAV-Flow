"""
uav_datasets.py

A simple dataset class that directly matches the format required by OpenVLA training
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from starVLA.dataloader.uav_prompts import format_uav_instruction


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

logger = logging.getLogger(__name__)


class SimpleVLADataset(Dataset):
    """
    A simple dataset class that directly matches the format required by OpenVLA training
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        is_train: bool = True,
        image_aug: bool = False,
        action_codec=None,
        stats_path=None,
    ):
        """
        Initialize the dataset

        Args:
            data_path: Path to the dataset directory
            is_train: Whether the dataset is for training mode
            image_aug: Whether to enable image augmentation
        """
        self.data_path = Path(data_path)
        self.is_train = is_train
        self.image_aug = image_aug
        self.action_codec = action_codec
        self.stats_path = stats_path
        if action_codec is not None and not stats_path:
            raise ValueError("Versioned codec requires frozen normalization statistics")

        # Load all episodes
        self.episodes = self._load_data()

        # Compute action statistics
        self._compute_action_stats()

        # Collect all frames into self.samples
        self.samples = []
        self.last_frame_repeat_count = 5
        for episode in self.episodes:
            frame_count = len(episode["image_paths"])
            for frame_idx in range(frame_count):
                repeat = frame_idx == 0 or frame_count - 5 <= frame_idx <= frame_count - 1
                copies = 1 + (self.last_frame_repeat_count if repeat else 0)
                for _ in range(copies):
                    # Keep each repeated sample's action array independent.
                    actions = np.zeros((10, 4))
                    for i in range(10):
                        future_idx = frame_idx + i
                        if future_idx < len(episode["actions"]):
                            actions[i] = episode["actions"][future_idx]
                    self.samples.append(
                        {
                            "image_path": [episode["image_paths"][0], episode["image_paths"][frame_idx]],
                            "action": actions,
                            "proprio": episode["proprio"][frame_idx],
                            "instruction": episode["instruction"],
                        }
                    )

        # Keep sample indices stable across ranks. DataLoader's sampler owns
        # shuffling, worker dispatch, and (via Accelerate) distributed sharding.

    def _load_data(self) -> List[Dict]:
        """
        Load episodes from the dataset directory
        """
        episodes = []

        # import ipdb; ipdb.set_trace()

        # Iterate through episode directories
        for episode_dir in sorted(self.data_path.iterdir()):
            if not episode_dir.is_dir():
                continue

            # Read log.json
            # print(episode_dir)
            log_path = episode_dir / "log.json"
            if not log_path.exists():
                continue

            with open(log_path, "r") as f:
                episode_data = json.load(f)

            # Process a single episode
            processed_episode = self._process_episode(episode_data, episode_dir)
            if processed_episode is not None:
                episodes.append(processed_episode)

        return episodes

    def _process_episode(self, episode_data: Dict, episode_dir: Path) -> Optional[Dict]:
        """
        Process a single episode
        """
        try:
            # Get image files
            image_files = sorted(list(episode_dir.glob("*.[jp][pn][g]")))
            if not image_files:
                if self.action_codec is not None:
                    raise ValueError("Episode has no images")
                return None

            # Process trajectory data, keep only x, y, z, yaw
            trajectory_raw = np.array(episode_data["raw_logs"], dtype=np.float64)  # [T, 6]
            trajectory = np.array(episode_data["preprocessed_logs"], dtype=np.float64)
            if self.action_codec is not None:
                if (
                    trajectory_raw.ndim != 2
                    or trajectory.ndim != 2
                    or trajectory_raw.shape[1] < 5
                    or trajectory.shape[1] < 5
                ):
                    raise ValueError("Pose logs must be [T,>=5]")
                if (
                    not len(image_files) == len(trajectory_raw) == len(trajectory)
                    or not np.isfinite(trajectory_raw).all()
                    or not np.isfinite(trajectory).all()
                ):
                    raise ValueError("Image/log lengths differ or poses contain nonfinite values")

            # Keep only x, y, z, yaw (indices: 0,1,2,4)
            trajectory_raw = trajectory_raw[:, [0, 1, 2, 4]]  # [T, 4]
            trajectory = trajectory[:, [0, 1, 2, 4]]  # [T, 4]

            # proprio: current position in the coordinate frame of the first frame; yaw in degrees
            proprio = trajectory

            trajectory_raw[:, 3] = np.deg2rad(trajectory_raw[:, 3])  # convert yaw only

            actions = np.zeros_like(trajectory)  # [T, 4]
            for i in range(len(trajectory) - 1):
                current_pose = trajectory_raw[i]  # [4]
                next_pose = trajectory_raw[i + 1]  # [4]
                actions[i] = self._transform_to_local_frame(current_pose, next_pose)

            # Set the action of the last frame to zero
            actions[-1] = np.zeros(4)

            if actions.shape[0] == 0:
                return None

            return {
                "image_paths": image_files,
                "actions": actions,
                "proprio": proprio,
                "instruction": episode_data.get("instruction", "default instruction"),
                # 'grounding': episode_data["grounding"]
            }

        except Exception as e:
            if self.action_codec is not None:
                raise ValueError(f"Invalid versioned-codec episode {episode_dir}: {e}") from e
            logger.warning(f"Failed to process episode: {str(e)}")
            return None

    def _transform_to_local_frame(self, current_pose: np.ndarray, next_pose: np.ndarray) -> np.ndarray:
        """
        Transform the next pose into the local frame of the current pose
        """
        # Extract position and yaw
        current_pos = current_pose[:3]
        current_yaw = current_pose[3]

        next_pos = next_pose[:3]
        next_yaw = next_pose[3]

        # Build 2D rotation matrix
        cos_yaw = np.cos(current_yaw)
        sin_yaw = np.sin(current_yaw)
        R = np.array([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]])

        # Compute relative position
        relative_pos = next_pos - current_pos
        local_pos = np.linalg.inv(R) @ relative_pos

        # Compute relative yaw
        relative_yaw = next_yaw - current_yaw
        relative_yaw = (relative_yaw + np.pi) % (2 * np.pi) - np.pi

        return np.array([local_pos[0], local_pos[1], local_pos[2], relative_yaw])

    def _compute_action_stats(self):
        if self.stats_path:
            from starVLA.model.action_codec.action_codebook import load_normalization_stats

            stats = load_normalization_stats(self.stats_path, self.action_codec)
            for name in ("action_min", "action_max", "action_mean", "action_std"):
                setattr(self, name, np.asarray(stats[name], dtype=np.float64))
            if not self.episodes:
                raise ValueError("No valid episodes")
            return
        """
        Compute action statistics
        """
        all_actions = np.concatenate([episode["actions"] for episode in self.episodes])
        self.action_mean = np.mean(all_actions, axis=0)
        self.action_std = np.std(all_actions, axis=0)
        self.action_min = np.percentile(all_actions, 1, axis=0)
        self.action_max = np.percentile(all_actions, 99, axis=0)

        logger.info("Action statistics:")
        logger.info(f"mean: {self.action_mean}")
        logger.info(f"std: {self.action_std}")
        logger.info(f"1st percentile (min approx): {self.action_min}")
        logger.info(f"99th percentile (max approx): {self.action_max}")
        print("Action statistics:")
        print(f"mean: {self.action_mean}")
        print(f"std: {self.action_std}")
        print(f"1st percentile (min approx): {self.action_min}")
        print(f"99th percentile (max approx): {self.action_max}")

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """
        Normalize action values to [-1, 1]
        """
        if getattr(self, "action_codec", None) is not None:
            return self.action_codec.normalize_numpy(action)
        normalized_actions = 2 * (action - self.action_min) / (self.action_max - self.action_min) - 1
        normalized_actions = np.clip(normalized_actions, -1, 1)
        return normalized_actions

    def __getitem__(self, idx):
        sample = self.samples[idx]
        first_img_path, img_path = sample["image_path"]
        image = Image.open(img_path).convert("RGB")
        image_first = Image.open(first_img_path).convert("RGB")

        instruction = format_uav_instruction(sample["proprio"], sample["instruction"])
        action = self._normalize_action(sample["action"])
        result = {
            "image": [image_first, image],
            "lang": instruction,
            "action": action,
        }
        if getattr(self, "action_codec", None) is not None:
            result["action_bins"] = self.action_codec.encode_numpy(sample["action"])
        return result

    def __len__(self):
        return len(self.samples)


def collate_fn(batch):
    return batch


def make_uav_datasets(cfg, is_debug=False):
    data_path = cfg.datasets.vla_data.data_root_dir
    batch_size = cfg.datasets.vla_data.per_device_batch_size

    from starVLA.model.action_codec import load_codec_from_config

    codec = load_codec_from_config(cfg) if "framework" in cfg else None
    train_dataset = SimpleVLADataset(
        data_path, action_codec=codec, stats_path=cfg.datasets.vla_data.get("normalization_stats_path")
    )
    low, high = train_dataset.action_min, train_dataset.action_max
    if (
        low.shape != (4,)
        or high.shape != (4,)
        or not np.isfinite(low).all()
        or not np.isfinite(high).all()
        or np.any(high <= low)
    ):
        raise ValueError("Action normalization requires finite, increasing bounds in every dimension")

    if is_debug:
        num_workers = 0
    else:
        num_workers = int(cfg.datasets.vla_data.get("num_workers", 4))

    from torch.utils.data import DataLoader

    # A dedicated common seed prevents rank-local model/dropout RNG from
    # changing the index permutation before Accelerate divides its batches.
    generator = torch.Generator().manual_seed(int(getattr(cfg, "seed", 42)))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=True,
        generator=generator,
    )

    return train_dataloader
