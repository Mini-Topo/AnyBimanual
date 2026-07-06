# Standard Library
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
from typing import Union

# Third Party
import numpy as np


class R2BCEpisodeBuffer:
    """
    Simple episode-level buffer for R2BC data collection.

    保存単位:
        1 episode = 1 pickle file

    各stepに保存するもの:
        obs
        next_obs
        human_arm
        human_action
        success
        terminate
        description
        step_id
    """

    def __init__(
        self,
        task_name: str,
        variation: int,
        episode_id: int,
        human_arm: str,
        description: Optional[str] = None,
    ):
        self.task_name = task_name
        self.variation = variation
        self.episode_id = episode_id
        self.human_arm = human_arm
        self.description = description

        self.steps: List[Dict[str, Any]] = []

    def add(
        self,
        obs: Any,
        human_action: np.ndarray,
        next_obs: Any,
        success: bool,
        terminate: bool,
        step_id: int,
        target_action_9d: Optional[np.ndarray] = None,
        target_arm: Optional[str] = None,
        policy_arm: Optional[str] = None,
        policy_action: Optional[np.ndarray] = None,
        full_action: Optional[Any] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        human_action = np.asarray(human_action, dtype=np.float32)

        if target_action_9d is not None:
            target_action_9d = np.asarray(target_action_9d, dtype=np.float32)

        if policy_action is not None:
            policy_action = np.asarray(policy_action, dtype=np.float32)

        if full_action is not None:
            full_action = np.asarray(full_action, dtype=np.float32)

        self.steps.append(
            {
                "step_id": int(step_id),
                "obs": obs,
                "human_action": human_action,
                "target_arm": target_arm or self.human_arm,
                "target_action_9d": target_action_9d,
                "next_obs": next_obs,
                "human_arm": self.human_arm,
                "policy_arm": policy_arm,
                "policy_action": policy_action,
                "full_action": full_action,
                "success": bool(success),
                "terminate": bool(terminate),
                "description": self.description,
                "info": info or {},
            }
        )
        
    def summary(self) -> Dict[str, Any]:
        final_success = False
        final_terminate = False

        if len(self.steps) > 0:
            final_success = bool(self.steps[-1]["success"])
            final_terminate = bool(self.steps[-1]["terminate"])

        return {
            "task_name": self.task_name,
            "variation": self.variation,
            "episode_id": self.episode_id,
            "human_arm": self.human_arm,
            "description": self.description,
            "num_steps": len(self.steps),
            "final_success": final_success,
            "final_terminate": final_terminate,
        }

    def save(self, path: Union[str, Path]) -> Path:
        """
        Save episode buffer as pickle.

        path can be:
            /path/to/episode_000000.pkl
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "meta": self.summary(),
            "steps": self.steps,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        return path

    def __len__(self) -> int:
        return len(self.steps)


def make_episode_path(
    save_root: Union[str, Path],
    task_name: str,
    episode_id: int,
    human_arm: str,
) -> Path:
    """
    Example:
        data/r2bc/bimanual_lift_long_block/right_human/episode_000000.pkl
    """
    save_root = Path(save_root)

    return (
        save_root
        / task_name
        / f"{human_arm}_human"
        / f"episode_{episode_id:06d}.pkl"
    )