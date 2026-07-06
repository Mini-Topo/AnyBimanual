import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class R2BCDiskDataset(Dataset):
    """
    Disk-backed dataset for R2BC episode pickle files.

    Expected pickle format:
        {
            "meta": {...},
            "steps": [
                {
                    "obs": dict,
                    "human_action": np.ndarray shape (4,),
                    "target_arm": "right" or "left",
                    "target_action_9d": np.ndarray shape (9,),
                    "full_action": np.ndarray shape (18,),
                    ...
                },
                ...
            ],
        }

    This dataset does not eagerly load all observations into memory.
    It loads episode files once to build an index, and loads the needed
    episode on __getitem__.
    """

    def __init__(
        self,
        data_root: str,
        task_name: Optional[str] = None,
        max_episodes: Optional[int] = None,
        include_success: bool = True,
        episode_glob: str = "episode_*.pkl",
    ):
        self.data_root = Path(data_root)
        self.task_name = task_name
        self.max_episodes = max_episodes
        self.include_success = include_success
        self.episode_glob = episode_glob

        if task_name is None:
            search_root = self.data_root
        else:
            search_root = self.data_root / task_name

        if not search_root.exists():
            raise FileNotFoundError(f"Data directory not found: {search_root}")

        episode_paths = sorted(search_root.rglob(episode_glob))

        if max_episodes is not None:
            episode_paths = episode_paths[-max_episodes:]

        if len(episode_paths) == 0:
            raise FileNotFoundError(
                f"No episode pickle files found under {search_root} with glob {episode_glob}"
            )

        self.episode_paths: List[Path] = episode_paths

        # index: list of (episode_index, step_index)
        self.index: List[tuple[int, int]] = []
        self.episode_metas: List[Dict[str, Any]] = []
        self.episode_num_steps: List[int] = []

        for epi_idx, path in enumerate(self.episode_paths):
            episode = self._load_episode(path)
            steps = episode.get("steps", [])
            meta = episode.get("meta", {})

            self.episode_metas.append(meta)
            self.episode_num_steps.append(len(steps))

            for step_idx, step in enumerate(steps):
                if not self._is_valid_step(step):
                    continue

                if not include_success:
                    if bool(step.get("success", False)):
                        continue

                self.index.append((epi_idx, step_idx))

        if len(self.index) == 0:
            raise RuntimeError(
                "Episode files were found, but no valid transitions were indexed."
            )

        self._episode_cache_idx: Optional[int] = None
        self._episode_cache: Optional[Dict[str, Any]] = None

    def _load_episode(self, path: Path) -> Dict[str, Any]:
        with path.open("rb") as f:
            return pickle.load(f)

    def _get_episode(self, epi_idx: int) -> Dict[str, Any]:
        if self._episode_cache_idx == epi_idx and self._episode_cache is not None:
            return self._episode_cache

        episode = self._load_episode(self.episode_paths[epi_idx])
        self._episode_cache_idx = epi_idx
        self._episode_cache = episode
        return episode

    def _is_valid_step(self, step: Dict[str, Any]) -> bool:
        if "obs" not in step:
            return False
        if "target_arm" not in step:
            return False
        if "target_action_9d" not in step:
            return False
        if step["target_action_9d"] is None:
            return False
        return True

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        epi_idx, step_idx = self.index[idx]
        episode = self._get_episode(epi_idx)

        step = episode["steps"][step_idx]
        meta = episode.get("meta", {})

        target_arm = step["target_arm"]
        target_arm_id = 0 if target_arm == "right" else 1

        sample = {
            "obs": step["obs"],
            "next_obs": step.get("next_obs", None),

            "target_arm": target_arm,
            "target_arm_id": np.array(target_arm_id, dtype=np.int64),
            "target_action_9d": np.asarray(step["target_action_9d"], dtype=np.float32),

            "human_action": self._as_float_array(step.get("human_action", None)),
            "full_action": self._as_float_array(step.get("full_action", None)),
            "policy_action": self._as_float_array(step.get("policy_action", None)),

            "success": np.array(bool(step.get("success", False)), dtype=np.bool_),
            "terminate": np.array(bool(step.get("terminate", False)), dtype=np.bool_),

            "episode_path": str(self.episode_paths[epi_idx]),
            "episode_index": np.array(epi_idx, dtype=np.int64),
            "step_index": np.array(step_idx, dtype=np.int64),
            "step_id": np.array(step.get("step_id", step_idx), dtype=np.int64),

            "meta": meta,
            "info": step.get("info", {}),
        }

        return sample

    def _as_float_array(self, value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None
        return np.asarray(value, dtype=np.float32)

    def summary(self) -> Dict[str, Any]:
        return {
            "data_root": str(self.data_root),
            "task_name": self.task_name,
            "num_episode_files": len(self.episode_paths),
            "num_transitions": len(self.index),
            "episode_num_steps_min": min(self.episode_num_steps),
            "episode_num_steps_max": max(self.episode_num_steps),
            "episode_num_steps_sum": sum(self.episode_num_steps),
        }


def r2bc_disk_collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Minimal custom collate.

    Keep obs / next_obs / meta / info as Python objects because RLBench obs dicts
    contain large nested arrays and camera data. Stack only small numeric labels.
    """

    out: Dict[str, Any] = {}

    out["obs"] = [b["obs"] for b in batch]
    out["next_obs"] = [b["next_obs"] for b in batch]
    out["target_arm"] = [b["target_arm"] for b in batch]

    out["target_arm_id"] = torch.as_tensor(
        np.stack([b["target_arm_id"] for b in batch]),
        dtype=torch.long,
    )

    out["target_action_9d"] = torch.as_tensor(
        np.stack([b["target_action_9d"] for b in batch]),
        dtype=torch.float32,
    )

    out["success"] = torch.as_tensor(
        np.stack([b["success"] for b in batch]),
        dtype=torch.bool,
    )

    out["terminate"] = torch.as_tensor(
        np.stack([b["terminate"] for b in batch]),
        dtype=torch.bool,
    )

    out["episode_index"] = torch.as_tensor(
        np.stack([b["episode_index"] for b in batch]),
        dtype=torch.long,
    )

    out["step_index"] = torch.as_tensor(
        np.stack([b["step_index"] for b in batch]),
        dtype=torch.long,
    )

    out["step_id"] = torch.as_tensor(
        np.stack([b["step_id"] for b in batch]),
        dtype=torch.long,
    )

    out["human_action"] = _stack_optional_float(batch, "human_action")
    out["full_action"] = _stack_optional_float(batch, "full_action")
    out["policy_action"] = _stack_optional_float(batch, "policy_action")

    out["episode_path"] = [b["episode_path"] for b in batch]
    out["meta"] = [b["meta"] for b in batch]
    out["info"] = [b["info"] for b in batch]

    return out


def _stack_optional_float(
    batch: Sequence[Dict[str, Any]],
    key: str,
) -> Optional[torch.Tensor]:
    values = [b[key] for b in batch]
    if any(v is None for v in values):
        return None

    return torch.as_tensor(
        np.stack(values),
        dtype=torch.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/r2bc_peract")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    dataset = R2BCDiskDataset(
        data_root=args.data_root,
        task_name=args.task,
        max_episodes=args.max_episodes,
    )

    print("[r2bc_disk_buffer] Dataset summary:")
    for k, v in dataset.summary().items():
        print(f"  {k}: {v}")

    sample = dataset[0]

    print("\n[r2bc_disk_buffer] Sample keys:")
    print(list(sample.keys()))

    print("\n[r2bc_disk_buffer] Sample shapes:")
    print("  target_arm:", sample["target_arm"])
    print("  target_arm_id:", sample["target_arm_id"], sample["target_arm_id"].shape)
    print("  target_action_9d:", sample["target_action_9d"].shape)

    if sample["human_action"] is not None:
        print("  human_action:", sample["human_action"].shape)

    if sample["full_action"] is not None:
        print("  full_action:", sample["full_action"].shape)

    if sample["policy_action"] is not None:
        print("  policy_action:", sample["policy_action"].shape)

    obs = sample["obs"]
    if isinstance(obs, dict):
        print("\n[r2bc_disk_buffer] Obs keys:")
        print(list(obs.keys()))

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=r2bc_disk_collate_fn,
    )

    batch = next(iter(loader))

    print("\n[r2bc_disk_buffer] Batch check:")
    print("  batch obs len:", len(batch["obs"]))
    print("  target_arm:", batch["target_arm"])
    print("  target_arm_id:", batch["target_arm_id"].shape)
    print("  target_action_9d:", batch["target_action_9d"].shape)

    if batch["full_action"] is not None:
        print("  full_action:", batch["full_action"].shape)


if __name__ == "__main__":
    main()