# Standard Library
import os
import sys
import time
import argparse
from os.path import join, dirname, abspath
import copy
import time
from collections import defaultdict

# pygame の余計なログを消す
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")

# Project paths
PROJECT_ROOT = dirname(dirname(abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, join(PROJECT_ROOT, "third_party", "RLBench"))
sys.path.insert(0, join(PROJECT_ROOT, "third_party", "PyRep"))
sys.path.insert(0, join(PROJECT_ROOT, "third_party", "YARR"))

# Third Party
import numpy as np

try:
    from hydra import compose, initialize_config_dir
except ImportError:
    from hydra.experimental import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlbench.backend.utils import task_file_to_task_class
from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import BimanualDiscrete
from rlbench.action_modes.arm_action_modes import BimanualEndEffectorPoseViaPlanning

import torch
from torch.utils.data import DataLoader

# AnyBimanual / PerAct
from helpers.custom_rlbench_env import CustomRLBenchEnv
from helpers.observation_utils import create_obs_config

# Local
from r2bc.teleop_toggle import JoystickTeleop
from r2bc.anybimanual_peract_policy import AnyBimanualPerActPolicy
from r2bc.replay_buffer import R2BCEpisodeBuffer, make_episode_path

from r2bc.datasets.r2bc_disk_buffer import (
    R2BCDiskDataset,
    r2bc_disk_collate_fn,
)
from agents.agent_factory import create_agent

from r2bc.peract_update import (
    update_from_disk_buffer,
    sync_anybimanual_modules,
)

DT = 0.05
MAX_STEPS = 500


def debug_train_from_disk_buffer(args):
    dataset = R2BCDiskDataset(
        data_root=args.save_root,
        task_name=args.task,
        max_episodes=args.train_max_episodes,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=r2bc_disk_collate_fn,
    )

    batch = next(iter(loader))

    print("[debug train] disk buffer loaded")
    print(f"[debug train] num episodes: {len(dataset.episode_paths)}")
    print(f"[debug train] num transitions: {len(dataset)}")
    print(f"[debug train] target_arm: {batch['target_arm']}")
    print(f"[debug train] target_arm_id: {batch['target_arm_id'].shape}")
    print(f"[debug train] target_action_9d: {batch['target_action_9d'].shape}")

    if batch["full_action"] is not None:
        print(f"[debug train] full_action: {batch['full_action'].shape}")

def get_arm_by_name(robot, arm_name):
    if arm_name == "right":
        return robot.right_arm
    if arm_name == "left":
        return robot.left_arm
    raise ValueError(f"Unknown arm_name: {arm_name}")


def get_pose7_from_arm(arm):
    tip = arm.get_tip()
    pos = tip.get_position()
    quat = tip.get_quaternion()
    return np.array(
        [
            pos[0], pos[1], pos[2],
            quat[0], quat[1], quat[2], quat[3],
        ],
        dtype=np.float32,
    )


def make_9d_action_from_pose7(
    pose7,
    gripper_open=True,
    ignore_collisions=0.0,
):
    return np.array(
        [
            pose7[0], pose7[1], pose7[2],
            pose7[3], pose7[4], pose7[5], pose7[6],
            float(gripper_open),
            float(ignore_collisions),
        ],
        dtype=np.float32,
    )


def apply_delta_to_pose7(pose7, action_4d):
    """
    action_4d = [dx, dy, dz, gripper_toggle_or_state]
    rotation は固定。
    """
    target_pose = pose7.copy()
    target_pose[0] += float(action_4d[0])
    target_pose[1] += float(action_4d[1])
    target_pose[2] += float(action_4d[2])
    return target_pose

def make_delta_9d_action(robot, arm_name, action_4d, gripper_open=True):
    arm = get_arm_by_name(robot, arm_name)
    current_pose7 = get_pose7_from_arm(arm)
    target_pose7 = apply_delta_to_pose7(current_pose7, action_4d)
    return make_9d_action_from_pose7(
        target_pose7,
        gripper_open=gripper_open,
        ignore_collisions=0.0,
    )

def make_idle_9d_action(robot, arm_name, gripper_open=True, ignore_collisions=0.0):
    arm = get_arm_by_name(robot, arm_name)
    pose7 = get_pose7_from_arm(arm)
    return make_9d_action_from_pose7(
        pose7,
        gripper_open=gripper_open,
        ignore_collisions=ignore_collisions,
    )


def make_safe_policy_9d_action(
    robot,
    arm_name,
    policy_action_9d,
    gripper_open=True,
    max_policy_delta=0.05,
    min_policy_delta=1e-4,
    idle_epsilon=1e-4,
    step_id=0,
):
    """
    R2BC collect 用の安全な policy-arm action。

    方針:
    - raw PerAct policy_action_9d は実行には使わない
    - policy arm は基本 idle にする
    - ただし current pose そのものを target にすると
      planner が zero-length path で壊れる可能性があるため、
      tiny offset を入れた non-zero idle action を返す

    注意:
    - policy_action_9d は shape check のためだけに受け取る
    - raw policy action 自体は episode_buffer の policy_action に保存されるので、
      評価には使える
    """
    policy_action_9d = np.asarray(policy_action_9d, dtype=np.float32)

    if policy_action_9d.shape != (9,):
        raise ValueError(
            f"policy_action_9d must be shape (9,), got {policy_action_9d.shape}"
        )

    arm = get_arm_by_name(robot, arm_name)
    current_pose7 = get_pose7_from_arm(arm)

    safe_pose7 = current_pose7.copy()

    # drift を抑えるため、step ごとに tiny offset の向きを反転する。
    # z方向だけに入れる。まずは一番単純な zero-length path 回避。
    sign = 1.0 if (step_id % 2 == 0) else -1.0
    safe_pose7[2] += sign * float(idle_epsilon)

    safe_action = make_9d_action_from_pose7(
        safe_pose7,
        gripper_open=gripper_open,
        ignore_collisions=0.0,
    )

    return safe_action

def sync_grasp_attachment(robot, task, arm_name, was_closed, is_closed):
    """
    RLBench/PyRep 側の grasp attachment を同期する。
    """
    if (not was_closed) and is_closed:
        for obj in task.get_graspable_objects():
            robot.grasp(obj, arm_name)

    if was_closed and (not is_closed):
        robot.release_gripper(arm_name)


def load_cfg(config_dir, config_name, overrides):
    """
    debug_anybimanual_act.py と同じ cfg を作るための関数。
    """
    config_dir = abspath(config_dir)

    print(f"[collect_peract] config_dir: {config_dir}")
    print(f"[collect_peract] config_name: {config_name}")
    print("[collect_peract] overrides:")
    for o in overrides:
        print("  ", o)

    try:
        hydra_ctx = initialize_config_dir(
            config_dir=config_dir,
            version_base=None,
        )
    except TypeError:
        hydra_ctx = initialize_config_dir(
            config_dir=config_dir,
        )

    with hydra_ctx:
        cfg = compose(
            config_name=config_name,
            overrides=overrides,
        )

    return cfg


def make_action_mode():
    return BimanualMoveArmThenGripper(
        BimanualEndEffectorPoseViaPlanning(),
        BimanualDiscrete(),
    )


def make_custom_env(cfg, task_name, variation, headless):
    """
    PerAct / AnyBimanual 用の CustomRLBenchEnv を作る。

    重要:
    - obs_config は create_obs_config() を使う
    - time_in_state=True を env 引数で渡す
    """
    task_class = task_file_to_task_class(task_name, True)

    obs_config = create_obs_config(
        camera_names=list(cfg.rlbench.cameras),
        camera_resolution=list(cfg.rlbench.camera_resolution),
        method_name=cfg.method.name,
        robot_name="bimanual",
    )

    action_mode = make_action_mode()

    env = CustomRLBenchEnv(
        task_class=task_class,
        observation_config=obs_config,
        action_mode=action_mode,
        dataset_root="",
        episode_length=cfg.rlbench.episode_length,
        headless=headless,
        include_lang_goal_in_obs=True,
        time_in_state=True,
        record_every_n=-1,
    )

    env.launch()
    env._task.set_variation(variation)

    return env


def get_description_from_env(env):
    """
    CustomRLBenchEnv では description の持ち方が通常 task_env と少し違う可能性があるので、
    落ちないようにゆるく取る。
    """
    try:
        descriptions = env._task.get_descriptions()
        if len(descriptions) > 0:
            return descriptions[0]
    except Exception:
        pass
    return None

def print_time_summary(time_stats):
    print("[timing] summary")
    for k, values in time_stats.items():
        if not values:
            continue
        arr = np.array(values)
        print(
            f"  {k:16s} "
            f"mean={arr.mean():.4f}s "
            f"p50={np.percentile(arr, 50):.4f}s "
            f"p90={np.percentile(arr, 90):.4f}s "
            f"max={arr.max():.4f}s "
            f"n={len(arr)}"
        )

def get_arm_tip_xyz(robot, arm_name: str) -> np.ndarray:
    arm = get_arm_by_name(robot, arm_name)
    pose7 = get_pose7_from_arm(arm)
    return pose7[:3].copy()

def compute_action_delta_xyz(robot, arm_name: str, action_9d: np.ndarray) -> np.ndarray:
    """
    action_9d の target xyz が、現在の EE pose からどれだけ離れているかを返す。
    delta_xyz = target_xyz - current_xyz
    """
    action_9d = np.asarray(action_9d, dtype=np.float32)

    if action_9d.shape != (9,):
        raise ValueError(f"action_9d must be shape (9,), got {action_9d.shape}")

    current_xyz = get_arm_tip_xyz(robot, arm_name)
    target_xyz = action_9d[:3]

    return target_xyz - current_xyz


def compute_action_delta_z(robot, arm_name: str, action_9d: np.ndarray) -> float:
    return float(compute_action_delta_xyz(robot, arm_name, action_9d)[2])


def main():
    parser = argparse.ArgumentParser()

    # task / collection
    parser.add_argument("--task", type=str, default="bimanual_lift_long_block")
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--human-arm", type=str, default="right", choices=["right", "left"])
    parser.add_argument("--alternate-human-arm", action="store_true")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--save-root", type=str, default="data/r2bc_peract")
    parser.add_argument("--policy-every-steps", type=int, default=1, help="Run PerAct policy every K steps and reuse the previous policy action between calls." )

    # training
    parser.add_argument("--enable-train", action="store_true")
    parser.add_argument("--train-every-episodes", type=int, default=2)
    parser.add_argument("--train-max-episodes", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--train-num-updates", type=int, default=10)
    parser.add_argument("--train-target-arm", type=str, default="both", choices=["both", "right", "left"])

    # AnyBimanual / Hydra config
    parser.add_argument("--config-dir", type=str, default=join(PROJECT_ROOT, "conf"))
    parser.add_argument("--config-name", type=str, default="config")
    parser.add_argument("--ckpt-dir", type=str, default="/home/tappei-m/Project/AnyBimanual_checkpoints/PERACT_BC_leader_as_independent")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--timesteps", type=int, default=1)
    parser.add_argument("--max-policy-delta", type=float, default=0.01, help="Maximum xyz step size for capped policy action in meters.")
    parser.add_argument("--idle-epsilon", type=float, default=1e-4, help="Tiny offset added to policy-arm idle target to avoid zero-length planning.")

    # debug
    parser.add_argument("--print-obs-keys", action="store_true")
    parser.add_argument("--no-sleep", action="store_true")

    args = parser.parse_args()

    env = None
    teleop = None

    try:
        overrides = [
            "method=PERACT_BC",
            "method.agent_type=independent",
            "framework.anybimanual=True",
            "framework.checkpoint_name_prefix=checkpoint",
            "ddp.num_devices=1",
        ]

        cfg = load_cfg(
            config_dir=args.config_dir,
            config_name=args.config_name,
            overrides=overrides,
        )

        print("[collect_peract] cfg loaded")
        print("[collect_peract] method:", cfg.method.name)
        print("[collect_peract] agent_type:", cfg.method.agent_type)
        print("[collect_peract] anybimanual:", cfg.framework.anybimanual)
        print("[collect_peract] cameras:", list(cfg.rlbench.cameras))
        print("[collect_peract] camera_resolution:", list(cfg.rlbench.camera_resolution))

        print("[collect_peract] launching CustomRLBenchEnv...")
        env = make_custom_env(
            cfg=cfg,
            task_name=args.task,
            variation=args.variation,
            headless=args.headless,
        )

        robot = env._task._robot
        task = env._task._task

        print("[collect_peract] creating PerAct policy...")
        peract_policy = AnyBimanualPerActPolicy(
            cfg=cfg,
            ckpt_dir=args.ckpt_dir,
            device=args.device,
            timesteps=args.timesteps,
            deterministic=True,
        )

        print("[debug] peract_policy type:", type(peract_policy))
        print("[debug] peract_policy dict keys:", peract_policy.__dict__.keys())


        train_agent = None
        clip_agent = None
        train_cfg = None

        if args.enable_train:
            print("[collect_peract] creating train/clip agents for R2BC update...")

            train_cfg = copy.deepcopy(cfg)
            train_cfg.method.name = "PERACT_BC"
            train_cfg.method.agent_type = "independent"
            train_cfg.method.robot_name = "bimanual"
            train_cfg.method.transform_augmentation.apply_se3 = False
            train_cfg.framework.anybimanual = True
            train_cfg.framework.checkpoint_name_prefix = "checkpoint"
            train_cfg.ddp.num_devices = 1
            train_cfg.replay.batch_size = 1

            clip_cfg = copy.deepcopy(cfg)
            clip_cfg.method.name = "PERACT_BC"
            clip_cfg.method.agent_type = "independent"
            clip_cfg.method.robot_name = "bimanual"
            clip_cfg.method.transform_augmentation.apply_se3 = False
            clip_cfg.framework.anybimanual = True
            clip_cfg.framework.checkpoint_name_prefix = "checkpoint"
            clip_cfg.ddp.num_devices = 1
            clip_cfg.replay.batch_size = 1

            device = torch.device(args.device)

            clip_agent = create_agent(clip_cfg)
            clip_agent.build(training=False, device=device)

            train_agent = create_agent(train_cfg)
            train_agent.build(training=True, device=device)
            train_agent.load_weights(args.ckpt_dir)

            print("[collect_peract] train/clip agents ready.")

        teleop = JoystickTeleop()

        print("[collect_peract] Teleop started.")
        print("  stick  : move human arm")
        print("  button0: toggle human gripper")
        print("  Ctrl+C : quit")

        for ep_i in range(args.num_episodes):
            episode_id = args.episode_id + ep_i

            if args.alternate_human_arm:
                human_arm = "right" if episode_id % 2 == 0 else "left"
            else:
                human_arm = args.human_arm

            policy_arm = "left" if human_arm == "right" else "right"

            print("=" * 80)
            print(f"[collect_peract] Episode {episode_id}")
            print(f"[collect_peract] human_arm={human_arm}, policy_arm={policy_arm}")

            env._task.set_variation(args.variation)

            obs_dict = env.reset()
            peract_policy.reset()

            description = get_description_from_env(env)
            print("[collect_peract] description:", description)

            episode_buffer = R2BCEpisodeBuffer(
                task_name=args.task,
                variation=args.variation,
                episode_id=episode_id,
                human_arm=human_arm,
                description=description,
            )

            if args.print_obs_keys:
                print("[collect_peract] obs keys:")
                for k, v in obs_dict.items():
                    try:
                        print(f"  {k}: shape={np.asarray(v).shape}, dtype={np.asarray(v).dtype}")
                    except Exception:
                        print(f"  {k}: type={type(v)}")

            right_closed = False
            left_closed = False
            prev_gripper_button_pressed = False

            episode_t0 = time.perf_counter()
            time_stats = defaultdict(list)


            cached_peract_full_action = None
            cached_policy_step_id = None
            for step_id in range(args.max_steps):
                step_t0 = time.perf_counter()

                prev_obs_dict = obs_dict

                # -------------------------
                # 1. PerAct full 18D action
                # -------------------------
                t0 = time.perf_counter()

                run_policy = (cached_peract_full_action is None or step_id % args.policy_every_steps == 0)

                if run_policy:
                    try:
                        peract_full_action = peract_policy.act_full(obs_dict, step_id)
                        peract_full_action = np.asarray(peract_full_action, dtype=np.float32)

                        if peract_full_action.shape != (18,):
                            raise ValueError(
                                f"peract_full_action must be shape (18,), got {peract_full_action.shape}"
                            )

                        cached_peract_full_action = peract_full_action.copy()
                        cached_policy_step_id = step_id
                        step_policy_ok = True

                    except Exception as e:
                        print(f"[collect_peract] peract_policy.act_full failed at step={step_id}: {e}")
                        step_policy_ok = False
                        break
                else:
                    peract_full_action = cached_peract_full_action.copy()
                    step_policy_ok = True

                right_policy_action = peract_full_action[:9]
                left_policy_action = peract_full_action[9:18]

                time_stats["policy_act"].append(time.perf_counter() - t0)

                # -------------------------
                # 2. human joystick 4D action
                # -------------------------
                t0 = time.perf_counter()
                human_action_4d = teleop.read_action().as_array()
                human_action_4d = np.asarray(human_action_4d, dtype=np.float32)

                if human_action_4d.shape != (4,):
                    raise ValueError(
                        f"human_action_4d must be shape (4,), got {human_action_4d.shape}"
                    )

                time_stats["joystick_read"].append(time.perf_counter() - t0)

                # -------------------------
                # 3. human gripper toggle
                # -------------------------
                t0 = time.perf_counter()
                gripper_button_pressed = bool(human_action_4d[3] > 0.5)
                gripper_toggled = False

                if gripper_button_pressed and not prev_gripper_button_pressed:
                    if human_arm == "right":
                        was_closed = right_closed
                        right_closed = not right_closed
                        sync_grasp_attachment(
                            robot=robot,
                            task=task,
                            arm_name="right",
                            was_closed=was_closed,
                            is_closed=right_closed,
                        )
                        print(f"[human gripper] right_closed -> {right_closed}")
                    else:
                        was_closed = left_closed
                        left_closed = not left_closed
                        sync_grasp_attachment(
                            robot=robot,
                            task=task,
                            arm_name="left",
                            was_closed=was_closed,
                            is_closed=left_closed,
                        )
                        print(f"[human gripper] left_closed -> {left_closed}")

                    gripper_toggled = True

                prev_gripper_button_pressed = gripper_button_pressed

                right_gripper_open = not right_closed
                left_gripper_open = not left_closed

                time_stats["gripper_toggle"].append(time.perf_counter() - t0)

                # -------------------------
                # 4. human arm だけ delta 9D に変換
                # -------------------------
                t0 = time.perf_counter()
                if human_arm == "right":
                    human_right_action = make_delta_9d_action(
                        robot=robot,
                        arm_name="right",
                        action_4d=human_action_4d,
                        gripper_open=right_gripper_open,
                    )

                    safe_left_policy_action = make_safe_policy_9d_action(
                        robot=robot,
                        arm_name="left",
                        policy_action_9d=left_policy_action,
                        gripper_open=left_gripper_open,
                        max_policy_delta=args.max_policy_delta,
                        idle_epsilon=args.idle_epsilon,
                        step_id=step_id,
                    )

                    right_action = human_right_action
                    left_action = safe_left_policy_action

                else:
                    safe_right_policy_action = make_safe_policy_9d_action(
                        robot=robot,
                        arm_name="right",
                        policy_action_9d=right_policy_action,
                        gripper_open=right_gripper_open,
                        max_policy_delta=args.max_policy_delta,
                        idle_epsilon=args.idle_epsilon,
                        step_id=step_id,
                    )

                    human_left_action = make_delta_9d_action(
                        robot=robot,
                        arm_name="left",
                        action_4d=human_action_4d,
                        gripper_open=left_gripper_open,
                    )

                    right_action = safe_right_policy_action
                    left_action = human_left_action

                full_action = np.concatenate([right_action, left_action]).astype(np.float32)

                if full_action.shape != (18,):
                    raise ValueError(f"full_action must be shape (18,), got {full_action.shape}")

                time_stats["delta_9d_action"].append(time.perf_counter() - t0)

                # -------------------------
                # 4.5. delta_z debug
                # -------------------------
                right_policy_delta_xyz = compute_action_delta_xyz(
                    robot=robot,
                    arm_name="right",
                    action_9d=right_policy_action,
                )
                left_policy_delta_xyz = compute_action_delta_xyz(
                    robot=robot,
                    arm_name="left",
                    action_9d=left_policy_action,
                )

                right_exec_delta_xyz = compute_action_delta_xyz(
                    robot=robot,
                    arm_name="right",
                    action_9d=right_action,
                )
                left_exec_delta_xyz = compute_action_delta_xyz(
                    robot=robot,
                    arm_name="left",
                    action_9d=left_action,
                )

                right_policy_delta_z = float(right_policy_delta_xyz[2])
                left_policy_delta_z = float(left_policy_delta_xyz[2])
                right_exec_delta_z = float(right_exec_delta_xyz[2])
                left_exec_delta_z = float(left_exec_delta_xyz[2])

                # -------------------------
                # 5. 実行
                # -------------------------
                t0 = time.perf_counter()
                try:
                    raw_obs_tp1, reward, env_terminal = env._task.step(full_action)
                    obs_dict_tp1 = env.extract_obs(raw_obs_tp1)
                    step_ok = True
                except Exception as e:
                    print(f"[collect_peract] env._task.step failed at step={step_id}: {e}")
                    reward = 0.0
                    env_terminal = True
                    obs_dict_tp1 = obs_dict
                    step_ok = False
                time_stats["exec"].append(time.perf_counter() - t0)

                # -------------------------
                # 6. success / terminate
                # -------------------------

                if step_ok:
                    success, task_terminate = env._task._task.success()
                else:
                    success = False
                    task_terminate = True

                terminate = bool(env_terminal or task_terminate)

                # -------------------------
                # 7. buffer add
                # -------------------------
                if human_arm == "right":
                    target_action_9d = right_action
                else:
                    target_action_9d = left_action

                episode_buffer.add(
                    obs=prev_obs_dict,
                    human_action=human_action_4d,
                    target_action_9d=target_action_9d,
                    target_arm=human_arm,
                    next_obs=obs_dict_tp1,
                    success=success,
                    terminate=terminate,
                    step_id=step_id,
                    policy_arm=policy_arm,
                    policy_action=peract_full_action,
                    full_action=full_action,
                    info={
                        "step_ok": step_ok,
                        "reward": reward,

                        "human_arm": human_arm,
                        "policy_arm": policy_arm,

                        "policy_every_steps": args.policy_every_steps,
                        "policy_run_this_step": run_policy,
                        "cached_policy_step_id": cached_policy_step_id,
                        "max_policy_delta": args.max_policy_delta,
                        "idle_epsilon": args.idle_epsilon,

                        "right_closed": right_closed,
                        "left_closed": left_closed,
                        "right_gripper_open": right_gripper_open,
                        "left_gripper_open": left_gripper_open,

                        "gripper_button_pressed": gripper_button_pressed,
                        "gripper_toggled": gripper_toggled,

                        "peract_full_action_18d": peract_full_action,
                        "right_policy_action_9d": right_policy_action,
                        "left_policy_action_9d": left_policy_action,

                        "right_executed_action_9d": right_action,
                        "left_executed_action_9d": left_action,

                        "right_policy_delta_xyz": right_policy_delta_xyz,
                        "left_policy_delta_xyz": left_policy_delta_xyz,
                        "right_exec_delta_xyz": right_exec_delta_xyz,
                        "left_exec_delta_xyz": left_exec_delta_xyz,

                        "right_policy_delta_z": right_policy_delta_z,
                        "left_policy_delta_z": left_policy_delta_z,
                        "right_exec_delta_z": right_exec_delta_z,
                        "left_exec_delta_z": left_exec_delta_z,

                        "target_arm": human_arm,
                        "target_action_9d": target_action_9d,

                        "control_mode": "r2bc_anybimanual_peract_safe",
                    },
                )

                if step_id % 10 == 0 or success or terminate:
                    print(
                        f"[debug] step={step_id} "
                        f"human_arm={human_arm} "
                        f"policy_arm={policy_arm} "
                        f"policy_run={run_policy} "
                        f"cached_policy_step={cached_policy_step_id} "
                        f"human_action_4d={human_action_4d} "
                        f"right_xyz={right_action[:3]} "
                        f"left_xyz={left_action[:3]} "
                        f"right_policy_dz={right_policy_delta_z:+.4f} "
                        f"left_policy_dz={left_policy_delta_z:+.4f} "
                        f"right_exec_dz={right_exec_delta_z:+.4f} "
                        f"left_exec_dz={left_exec_delta_z:+.4f} "
                        f"reward={reward} "
                        f"success={success} "
                        f"terminate={terminate}"
                    )

                obs_dict = obs_dict_tp1

                if terminate:
                    print("[collect_peract] Terminated.")
                    break

                if not args.no_sleep:
                    t0 = time.perf_counter()
                    time.sleep(DT)
                    time_stats["sleep"].append(time.perf_counter() - t0)

                time_stats["step_total"].append(time.perf_counter() - step_t0)

            save_path = make_episode_path(
                save_root=args.save_root,
                task_name=args.task,
                episode_id=episode_id,
                human_arm=human_arm,
            )

            saved_path = episode_buffer.save(save_path)

            

            print(f"[collect_peract] Episode saved to {saved_path}")
            print(f"[collect_peract] Summary: {episode_buffer.summary()}")
            print_time_summary(time_stats)

            if args.enable_train and ((episode_id + 1) % args.train_every_episodes == 0):
                print(
                    f"[collect_peract] Training trigger at episode {episode_id + 1} "
                    f"(every {args.train_every_episodes} episodes)"
                )

                train_target_arm = None
                if args.train_target_arm != "both":
                    train_target_arm = args.train_target_arm

                train_stats = update_from_disk_buffer(
                    train_agent=train_agent,
                    clip_agent=clip_agent,
                    cfg=train_cfg,
                    device=torch.device(args.device),
                    data_root=args.save_root,
                    task_name=args.task,
                    num_updates=args.train_num_updates,
                    batch_size=args.train_batch_size,
                    max_episodes=args.train_max_episodes,
                    target_arm=train_target_arm,
                    shuffle=True,
                    debug=True,
                    raise_on_error=True,
                )

                print("[collect_peract] train_stats:", train_stats)

                print("[collect_peract] syncing AnyBimanual modules to act policy...")
                sync_anybimanual_modules(
                    src_agent=train_agent,
                    dst_agent=peract_policy.agent,
                )
                print("[collect_peract] sync done.")
        print("[collect_peract] done.")

    except KeyboardInterrupt:
        print("\n[collect_peract] Stopping by Ctrl+C...")

    finally:
        if teleop is not None:
            teleop.close()

        if env is not None:
            env.shutdown()

        print("[collect_peract] Shutdown complete.")


if __name__ == "__main__":
    main()