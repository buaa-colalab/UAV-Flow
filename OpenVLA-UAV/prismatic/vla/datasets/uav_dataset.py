"""
simple_dataset.py

A simple dataset class that directly matches the format required by OpenVLA training
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.vla.action_tokenizer import ActionTokenizer

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

logger = logging.getLogger(__name__)

class SimpleVLADataset(IterableDataset):
    """
    A simple dataset class that directly matches the format required by OpenVLA training
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
        action_tokenizer: ActionTokenizer,
        prompt_builder_fn: type[PromptBuilder],
        is_train: bool = True,
        image_aug: bool = False,
    ):
        """
        Initialize the dataset
        
        Args:
            data_path: Path to the dataset directory
            image_transform: Image transform pipeline
            tokenizer: Text tokenizer
            action_tokenizer: Action tokenizer
            prompt_builder_fn: Prompt builder class/factory
            is_train: Whether the dataset is for training mode
            image_aug: Whether to enable image augmentation
        """
        self.data_path = Path(data_path)
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.action_tokenizer = action_tokenizer
        self.prompt_builder_fn = prompt_builder_fn
        self.is_train = is_train
        self.image_aug = image_aug
        
        # Load all episodes
        self.episodes = self._load_data()
        
        # Compute action statistics
        self._compute_action_stats()
        
        # Collect all frames into self.samples
        self.samples = []
        self.last_frame_repeat_count = 5
        for episode in self.episodes:
            # print(len(episode['image_paths']), episode['image_paths'][0])
            for frame_idx in range(len(episode['image_paths'])):
                self.samples.append({
                    'image_path': episode['image_paths'][frame_idx],
                    'action': episode['actions'][frame_idx],
                    'proprio': episode['proprio'][frame_idx],
                    'instruction': episode['instruction'],
                })
                if frame_idx == len(episode['image_paths']) - 1 or frame_idx == 0:
                    for i in range(self.last_frame_repeat_count):
                        self.samples.append({
                            'image_path': episode['image_paths'][frame_idx],
                            'action': episode['actions'][frame_idx],
                            'proprio': episode['proprio'][frame_idx],
                            'instruction': episode['instruction'],
                        })
        
        # Shuffle samples at the end of __init__
        np.random.shuffle(self.samples)
        
    def _load_data(self) -> List[Dict]:
        """
        Load episodes from the dataset directory
        """
        episodes = []
        
        # import ipdb; ipdb.set_trace()
        
        # Iterate through episode directories
        for episode_dir in self.data_path.iterdir():
            if not episode_dir.is_dir():
                continue
                
            # Read log.json
            # print(episode_dir)
            log_path = episode_dir / "log.json"
            if not log_path.exists():
                continue
                
            with open(log_path, 'r') as f:
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
            image_files = sorted(list(episode_dir.glob('*.[jp][pn][g]')))
            if not image_files:
                return None
            
            # Process trajectory data, keep only x, y, z, yaw
            trajectory_raw = np.array(episode_data['raw_logs'])  # [T, 6]
            trajectory = np.array(episode_data['preprocessed_logs'])          # [T, 6]
            
            # Keep only x, y, z, yaw (indices: 0,1,2,4)
            trajectory_raw = trajectory_raw[:, [0,1,2,4]]  # [T, 4]
            trajectory = trajectory[:, [0,1,2,4]]          # [T, 4]
            
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
                'image_paths': image_files,
                'actions': actions,
                'proprio': proprio,
                'instruction': episode_data.get('instruction', 'default instruction'),
            }
            
        except Exception as e:
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
        R = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        
        # Compute relative position
        relative_pos = next_pos - current_pos
        local_pos = np.linalg.inv(R) @ relative_pos
        
        # Compute relative yaw
        relative_yaw = next_yaw - current_yaw
        relative_yaw = (relative_yaw + np.pi) % (2 * np.pi) - np.pi
        
        return np.array([local_pos[0], local_pos[1], local_pos[2], relative_yaw])
    
    def _compute_action_stats(self):
        """
        Compute action statistics
        """
        all_actions = np.concatenate([episode['actions'] for episode in self.episodes])
        self.action_mean = np.mean(all_actions, axis=0)
        self.action_std = np.std(all_actions, axis=0)
        self.action_min = np.percentile(all_actions, 1, axis=0)
        self.action_max = np.percentile(all_actions, 99, axis=0)
        
        logger.info("Action statistics:")
        logger.info(f"mean: {self.action_mean}")
        logger.info(f"std: {self.action_std}")
        logger.info(f"1st percentile (min approx): {self.action_min}")
        logger.info(f"99th percentile (max approx): {self.action_max}")
    
    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """
        Normalize action values to [-1, 1]
        """
        normalized_actions = 2 * (action - self.action_min) / (self.action_max - self.action_min) - 1
        normalized_actions = np.clip(normalized_actions, -1, 1)
        return normalized_actions
    
    def __iter__(self):
        indices = np.random.permutation(len(self.samples))
        for idx in indices:
            sample = self.samples[idx]
            image = Image.open(sample['image_path']).convert('RGB')
            out = self.image_transform(image)
            pixel_values = out['pixel_values'][0]
            pixel_values = torch.from_numpy(pixel_values)
            action = self._normalize_action(sample['action'])
            # print(f"[INFO] action: {action}, action_ori: {sample['action']}")
            proprio = sample['proprio']
            proprio = ','.join([str(round(x, 1)) for x in proprio])
            prompt_builder = self.prompt_builder_fn("openvla")
            conversation = [
                {"from": "human", "value": f"Current State: {proprio}, What action should the uav take to {sample['instruction']}?"},
                {"from": "gpt", "value": self.action_tokenizer(action)},
            ]
            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])
            # print(f"[INFO] prompt_builder.get_prompt(): {prompt_builder.get_prompt()}")
            input_ids = self.tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
            labels = list(input_ids)
            input_ids = torch.tensor(input_ids)
            labels = torch.tensor(labels)
            attention_mask = torch.ones_like(input_ids)
            labels[: -(len(action) + 1)] = IGNORE_INDEX
            yield {
                'pixel_values': pixel_values,
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': attention_mask,
                "dataset_name": "uav",
            }

    def __len__(self):
        return len(self.samples) 