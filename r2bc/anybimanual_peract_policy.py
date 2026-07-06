# Standard Library
import os
import sys
from os.path import join, dirname, abspath

# Project paths
PROJECT_ROOT = dirname(dirname(abspath(__file__)))

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, join(PROJECT_ROOT, "third_party", "RLBench"))
sys.path.insert(0, join(PROJECT_ROOT, "third_party", "PyRep"))
sys.path.insert(0, join(PROJECT_ROOT, "third_party", "YARR"))

# Third Party
import numpy as np
import torch


def _get_type(value):
    """
    YARR rollout_generator.py の _get_type 相当。
    基本的には float32 に寄せる。
    uint8 画像だけ uint8 のままにする。
    """
    arr = np.asarray(value)

    if arr.dtype == np.uint8:
        return np.uint8

    return np.float32


class AnyBimanualPerActPolicy:
    """
    AnyBimanual + PerAct を R2BC collection から使うための薄い wrapper。

    重要:
    - 入力 obs は CustomRLBenchEnv.reset()/extract_obs() が返す obs_dict
    - 出力 action は 18D
      right 9D + left 9D
    """

    def __init__(
        self,
        cfg,
        ckpt_dir,
        device="cuda:0",
        timesteps=1,
        deterministic=True,
    ):
        self.cfg = cfg
        self.ckpt_dir = ckpt_dir
        self.device = torch.device(device)
        self.timesteps = int(timesteps)
        self.deterministic = bool(deterministic)

        if self.timesteps <= 0:
            raise ValueError(f"timesteps must be positive, got {self.timesteps}")

        self.agent = self._build_agent()

    def _build_agent(self):
        from agents.agent_factory import create_agent

        print("[AnyBimanualPerActPolicy] creating agent...")
        agent = create_agent(self.cfg)

        print("[AnyBimanualPerActPolicy] building agent...")
        agent.build(training=False, device=self.device)

        print(f"[AnyBimanualPerActPolicy] loading weights from: {self.ckpt_dir}")
        agent.load_weights(self.ckpt_dir)

        print("[AnyBimanualPerActPolicy] load success!")
        return agent

    def reset(self):
        if hasattr(self.agent, "reset"):
            self.agent.reset()

    def make_prepped_data(self, obs_dict):
        """
        CustomRLBenchEnv obs_dict -> YARR/PerAct 用 batch tensor

        出力:
            prepped_data[key].shape == [1, timesteps, ...]
        """
        obs_history = {}

        for k, v in obs_dict.items():
            dtype = _get_type(v)
            arr = np.asarray(v, dtype=dtype)

            # timesteps 分だけ同じ obs を履歴として積む
            obs_history[k] = [arr] * self.timesteps

        prepped_data = {}

        for k, v in obs_history.items():
            arr = np.asarray(v)

            if k == "lang_goal_tokens":
                tensor = torch.as_tensor(
                    arr[None],
                    dtype=torch.long,
                    device=self.device,
                )
            else:
                tensor = torch.as_tensor(
                    arr[None],
                    dtype=torch.float32,
                    device=self.device,
                )

            prepped_data[k] = tensor

        return prepped_data

    @torch.no_grad()
    def act_full(self, obs_dict, step_id):
        """
        Returns:
            np.ndarray shape=(18,)
            right 9D + left 9D
        """
        prepped_data = self.make_prepped_data(obs_dict)

        act_result = self.agent.act(
            step_id,
            prepped_data,
            deterministic=self.deterministic,
        )

        full_action = np.asarray(act_result.action, dtype=np.float32)

        if full_action.shape != (18,):
            raise ValueError(
                f"AnyBimanual/PerAct action must be shape (18,), got {full_action.shape}"
            )

        return full_action