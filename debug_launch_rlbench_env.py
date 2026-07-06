import time
import os

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf

import peract_config
from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper, MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import (
    BimanualEndEffectorPoseViaPlanning,
    EndEffectorPoseViaPlanning,
)
from rlbench.action_modes.gripper_action_modes import BimanualDiscrete, Discrete

from rlbench.backend import task as rlbench_task
from rlbench.backend.utils import task_file_to_task_class

from helpers import observation_utils
from helpers.custom_rlbench_env import CustomRLBenchEnv


@hydra.main(config_name="eval", config_path="conf")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    gripper_mode = eval(cfg.rlbench.gripper_mode)()
    arm_action_mode = eval(cfg.rlbench.arm_action_mode)()
    action_mode = eval(cfg.rlbench.action_mode)(arm_action_mode, gripper_mode)

    is_bimanual = cfg.method.robot_name == "bimanual"

    task_path = (
        rlbench_task.BIMANUAL_TASKS_PATH
        if is_bimanual
        else rlbench_task.TASKS_PATH
    )

    task_files = [
        t.replace(".py", "")
        for t in os.listdir(task_path)
        if t != "__init__.py" and t.endswith(".py")
    ]

    task_name = cfg.rlbench.tasks[0]
    if task_name not in task_files:
        raise ValueError(f"Task {task_name} not recognised.")

    task_class = task_file_to_task_class(task_name, is_bimanual)

    cameras = (
        cfg.rlbench.cameras
        if isinstance(cfg.rlbench.cameras, ListConfig)
        else [cfg.rlbench.cameras]
    )

    obs_config = observation_utils.create_obs_config(
        cameras,
        cfg.rlbench.camera_resolution,
        cfg.method.name,
        cfg.method.robot_name,
    )

    env = CustomRLBenchEnv(
        task_class=task_class,
        observation_config=obs_config,
        action_mode=action_mode,
        episode_length=cfg.rlbench.episode_length,
        dataset_root=cfg.rlbench.demo_path,
        headless=cfg.rlbench.headless,
        include_lang_goal_in_obs=False,
        time_in_state=cfg.rlbench.time_in_state,
    )

    print("Launching RLBench...")
    env.launch()

    print("Resetting task...")
    obs = env.reset()

    print("Task launched.")
    print("Observation keys:")
    for k in obs.keys():
        print("  ", k)

    input("Press Enter to shutdown the environment...")

    env.shutdown()


if __name__ == "__main__":
    peract_config.on_init()
    main()