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

    print("[debug] env attrs related to task/robot:")
    for name in dir(env):
        if "task" in name.lower() or "robot" in name.lower() or "scene" in name.lower():
            print("  ", name)

    input("[debug] Press Enter to inspect internal env and send fixed 18D action...")

    # まず内部に何があるか見る
    print("[debug] env.__dict__.keys():")
    for k in env.__dict__.keys():
        print("  ", k)

    # CustomRLBenchEnv / YARR RLBenchEnv は内部に task を持っている可能性が高い
    raw_task_env = None
    for candidate in ["_task", "_task_env", "task_env"]:
        if hasattr(env, candidate):
            raw_task_env = getattr(env, candidate)
            print(f"[debug] found raw_task_env as env.{candidate}")
            break

    if raw_task_env is None:
        raise RuntimeError(
            "Could not find raw task env inside CustomRLBenchEnv. "
            "Check printed env.__dict__.keys()."
        )

    print("[debug] raw_task_env type:", type(raw_task_env))
    print("[debug] raw_task_env attrs related to robot/scene:")
    for name in dir(raw_task_env):
        if "robot" in name.lower() or "scene" in name.lower():
            print("  ", name)

    robot = getattr(raw_task_env, "_robot", None)
    scene = getattr(raw_task_env, "_scene", None)

    print("[debug] robot:", robot)
    print("[debug] scene:", scene)

    if robot is None:
        raise RuntimeError("raw_task_env._robot is None or does not exist.")

    print("[debug] robot attrs:")
    for name in dir(robot):
        if "arm" in name.lower() or "gripper" in name.lower() or "right" in name.lower() or "left" in name.lower():
            print("  ", name)

    input("[debug] Press Enter to send fixed 18D action...")

    import numpy as np

    def get_pose7_from_arm(arm):
        tip = arm.get_tip()
        pos = tip.get_position()
        quat = tip.get_quaternion()
        return np.array(
            [pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]],
            dtype=np.float32,
        )

    def make_9d_action(pose7, gripper=1.0, ignore_collisions=0.0):
        return np.array(
            [
                pose7[0], pose7[1], pose7[2],
                pose7[3], pose7[4], pose7[5], pose7[6],
                gripper,
                ignore_collisions,
            ],
            dtype=np.float32,
        )

    right_pose = get_pose7_from_arm(robot.right_arm)
    left_pose = get_pose7_from_arm(robot.left_arm)

    print("[debug] current right pose7:", right_pose)
    print("[debug] current left  pose7:", left_pose)

    right_target_pose = right_pose.copy()
    left_target_pose = left_pose.copy()

    # right arm だけ少し上に動かす
    right_target_pose[2] += 0.01

    right_action = make_9d_action(
        right_target_pose,
        gripper=1.0,
        ignore_collisions=0.0,
    )

    left_action = make_9d_action(
        left_target_pose,
        gripper=1.0,
        ignore_collisions=0.0,
    )

    full_action = np.concatenate([right_action, left_action]).astype(np.float32)

    print("[debug] right_action:", right_action)
    print("[debug] left_action :", left_action)
    print("[debug] full_action shape:", full_action.shape)
    print("[debug] full_action:", full_action)

    input("[debug] Press Enter to call env.step(full_action)...")

    obs, reward, terminate = raw_task_env.step(full_action)

    print("[debug] raw_task_env.step success!")
    print("[debug] obs type:", type(obs))
    print("[debug] reward:", reward)
    print("[debug] terminate:", terminate)

    input("[debug] Press Enter to shutdown...")

    env.shutdown()

if __name__ == "__main__":
    peract_config.on_init()
    main()