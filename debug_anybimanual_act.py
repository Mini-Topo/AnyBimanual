# Standard Library
import os
import sys
from os.path import join, dirname, abspath

# pygame / SDL noise reduction
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")

# Project paths
PROJECT_ROOT = dirname(abspath(__file__))
sys.path.insert(0, join(PROJECT_ROOT, "third_party", "RLBench"))
sys.path.insert(0, join(PROJECT_ROOT, "third_party", "PyRep"))

# Third Party
import numpy as np
import torch
import hydra
from omegaconf import DictConfig

from rlbench.backend.utils import task_file_to_task_class
from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import BimanualDiscrete
from rlbench.action_modes.arm_action_modes import BimanualEndEffectorPoseViaPlanning

# Local
from agents.agent_factory import create_agent
from helpers.custom_rlbench_env import CustomRLBenchEnv
from helpers.observation_utils import create_obs_config


def _get_type(x):
    arr = np.asarray(x)
    if arr.dtype == np.float64:
        return np.float32
    return arr.dtype


def make_peract_obs_config(cfg):
    camera_names = list(cfg.rlbench.cameras)
    camera_resolution = list(cfg.rlbench.camera_resolution)

    print("[debug] camera_names:", camera_names)
    print("[debug] camera_resolution:", camera_resolution)

    return create_obs_config(
        camera_names=camera_names,
        camera_resolution=camera_resolution,
        method_name=cfg.method.name,
        robot_name="bimanual",
    )


def make_action_mode():
    return BimanualMoveArmThenGripper(
        BimanualEndEffectorPoseViaPlanning(),
        BimanualDiscrete(),
    )


def make_prepped_data(obs, timesteps, device):
    """
    third_party/YARR/yarr/utils/rollout_generator.py と同じ形にする。

    obs[key] -> [timesteps, ...]
             -> [batch=1, timesteps, ...]
             -> torch.Tensor
    """
    obs_history = {
        k: [np.array(v, dtype=_get_type(v))] * timesteps
        for k, v in obs.items()
    }

    prepped_data = {
        k: torch.tensor(np.array(v)[None], device=device)
        for k, v in obs_history.items()
    }

    return prepped_data


def print_obs_shapes(title, obs):
    print(f"\n[debug] {title}")
    for k, v in obs.items():
        arr = np.asarray(v)
        print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")


def print_tensor_shapes(title, data):
    print(f"\n[debug] {title}")
    for k, v in data.items():
        print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device}")


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # -------------------------
    # agent config
    # -------------------------
    cfg.method.name = "PERACT_BC"
    cfg.method.agent_type = "independent"
    cfg.method.robot_name = "bimanual"

    cfg.framework.anybimanual = True
    cfg.framework.checkpoint_name_prefix = "checkpoint"

    cfg.ddp.num_devices = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("[debug] method:", cfg.method.name)
    print("[debug] agent_type:", cfg.method.agent_type)
    print("[debug] anybimanual:", cfg.framework.anybimanual)
    print("[debug] checkpoint_name_prefix:", cfg.framework.checkpoint_name_prefix)
    print("[debug] device:", device)

    # -------------------------
    # load agent
    # -------------------------
    print("[debug] creating agent...")
    agent = create_agent(cfg)

    print("[debug] building agent...")
    agent.build(training=False, device=device)

    ckpt_dir = (
        "/home/tappei-m/Project/AnyBimanual_checkpoints/"
        "PERACT_BC_leader_as_independent"
    )

    print("[debug] loading weights from:", ckpt_dir)
    agent.load_weights(ckpt_dir)
    print("[debug] load success!")

    # -------------------------
    # make env
    # -------------------------
    task_name = "bimanual_lift_long_block"
    task_class = task_file_to_task_class(task_name, True)

    obs_config = make_peract_obs_config(cfg)
    action_mode = make_action_mode()

    episode_length = int(getattr(cfg.rlbench, "episode_length", 25))
    timesteps = int(getattr(cfg.framework, "timesteps", 1))

    print("[debug] task:", task_name)
    print("[debug] episode_length:", episode_length)
    print("[debug] timesteps:", timesteps)

    env = CustomRLBenchEnv(
        task_class=task_class,
        observation_config=obs_config,
        action_mode=action_mode,
        episode_length=episode_length,
        dataset_root="",
        channels_last=False,
        reward_scale=100.0,
        headless=False,
        time_in_state=True,
        include_lang_goal_in_obs=bool(
            getattr(cfg.framework, "include_lang_goal_in_obs", True)
        ),
        record_every_n=999999,
    )

    try:
        print("[debug] launching env...")
        env.launch()

        print("[debug] resetting env...")
        obs = env.reset()
        print_obs_shapes("raw obs from CustomRLBenchEnv.reset()", obs)

        prepped_data = make_prepped_data(
            obs=obs,
            timesteps=timesteps,
            device=device,
        )
        print_tensor_shapes("prepped_data for agent.act()", prepped_data)

        # -------------------------
        # act
        # -------------------------
        print("\n[debug] calling agent.act...")
        agent.reset()

        with torch.no_grad():
            act_result = agent.act(
                step=0,
                observation=prepped_data,
                deterministic=True,
            )

        action = np.asarray(act_result.action, dtype=np.float32)

        print("\n[debug] act_result type:", type(act_result))
        print("[debug] action type:", type(act_result.action))
        print("[debug] action shape:", action.shape)
        print("[debug] action:", action)

        if action.shape == (18,):
            print("[debug] OK: action is 18D = right 9D + left 9D")
            print("[debug] right_action:", action[:9])
            print("[debug] left_action :", action[9:18])
        else:
            print("[debug] WARNING: expected action shape (18,), got", action.shape)

        # -------------------------
        # optional one-step env.step
        # -------------------------
        print("\n[debug] trying env.step(act_result)...")
        transition = env.step(act_result)
        print("[debug] env.step returned")
        print("[debug] reward:", transition.reward)
        print("[debug] terminal:", transition.terminal)
        print("[debug] transition info:", transition.info)

    finally:
        print("[debug] shutting down env...")
        try:
            env.shutdown()
        except Exception as e:
            print("[debug] env.shutdown failed:", repr(e))


if __name__ == "__main__":
    main()