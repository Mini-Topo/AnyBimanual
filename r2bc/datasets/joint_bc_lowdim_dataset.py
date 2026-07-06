# r2bc/datasets/joint_bc_lowdim_dataset.py

import glob
import pickle
from typing import Optional, Dict, Any, List

import numpy as np
import torch
from torch.utils.data import Dataset


class JointBCLowDimDataset(Dataset):
    """
    expert_keyframe pkl を読む low_dim_state only Joint BC 用 Dataset。

    1 sample:
      input  = right_low_dim_state + left_low_dim_state + keyframe one-hot
      target = full_action shape (18,)
    """

    def __init__(
        self,
        data_root: str,
        task_name: str = "bimanual_lift_long_block",
        max_episodes: Optional[int] = None,
        num_keyframes: int = 2,
    ):
        self.data_root = data_root
        self.task_name = task_name
        self.num_keyframes = int(num_keyframes)

        pattern = f"{data_root}/{task_name}_expert_keyframe_ep*.pkl"
        self.episode_paths = sorted(glob.glob(pattern))

        if max_episodes is not None:
            self.episode_paths = self.episode_paths[: int(max_episodes)]

        if len(self.episode_paths) == 0:
            raise FileNotFoundError(f"No expert keyframe pkl found: pattern={pattern}")

        self.samples: List[Dict[str, Any]] = []

        for path in self.episode_paths:
            with open(path, "rb") as f:
                episode = pickle.load(f)

            steps = episode["steps"]

            for step in steps:
                obs = step["obs"]

                right_low = np.asarray(obs["right_low_dim_state"], dtype=np.float32)
                left_low = np.asarray(obs["left_low_dim_state"], dtype=np.float32)

                step_id = int(step["step_id"])
                key_onehot = np.zeros(self.num_keyframes, dtype=np.float32)
                if 0 <= step_id < self.num_keyframes:
                    key_onehot[step_id] = 1.0

                x = np.concatenate([right_low, left_low, key_onehot]).astype(np.float32)
                y = np.asarray(step["full_action"], dtype=np.float32)

                if y.shape != (18,):
                    raise ValueError(f"full_action must be shape (18,), got {y.shape}, path={path}")

                self.samples.append(
                    {
                        "x": x,
                        "y": y,
                        "episode_path": path,
                        "step_id": step_id,
                    }
                )

        self.x_dim = int(self.samples[0]["x"].shape[0])
        self.y_dim = 18

        xs = np.stack([s["x"] for s in self.samples], axis=0)
        ys = np.stack([s["y"] for s in self.samples], axis=0)

        self.x_mean = xs.mean(axis=0).astype(np.float32)
        self.x_std = (xs.std(axis=0) + 1e-6).astype(np.float32)

        self.y_mean = ys.mean(axis=0).astype(np.float32)
        self.y_std = (ys.std(axis=0) + 1e-6).astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        x = (s["x"] - self.x_mean) / self.x_std
        y = (s["y"] - self.y_mean) / self.y_std

        return {
            "x": torch.from_numpy(x).float(),
            "y": torch.from_numpy(y).float(),
            "raw_y": torch.from_numpy(s["y"]).float(),
            "step_id": torch.tensor(s["step_id"], dtype=torch.long),
        }

    def summary(self):
        return {
            "data_root": self.data_root,
            "task_name": self.task_name,
            "num_episodes": len(self.episode_paths),
            "num_samples": len(self.samples),
            "x_dim": self.x_dim,
            "y_dim": self.y_dim,
            "num_keyframes": self.num_keyframes,
        }