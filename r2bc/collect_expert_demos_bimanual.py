# r2bc/collect_expert_demos_bimanual.py

import os
import sys
import argparse
import pickle
from os.path import join, dirname, abspath

PROJECT_ROOT = dirname(dirname(abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, join(PROJECT_ROOT, "third_party", "RLBench"))
sys.path.insert(0, join(PROJECT_ROOT, "third_party", "PyRep"))
sys.path.insert(0, join(PROJECT_ROOT, "third_party", "YARR"))

import numpy as np

try:
    from hydra import compose, initialize_config_dir
except ImportError:
    from hydra.experimental import compose, initialize_config_dir

from rlbench.backend.utils import task_file_to_task_class
from helpers.observation_utils import create_obs_config
from helpers.custom_rlbench_env import CustomRLBenchEnv

from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import BimanualDiscrete
from rlbench.action_modes.arm_action_modes import BimanualEndEffectorPoseViaPlanning

from pyrep.objects.dummy import Dummy


def load_cfg(config_dir, config_name, overrides):
    config_dir = abspath(config_dir)

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
        BimanualDiscrete(
            attach_grasped_objects=False,
        ),
    )


def make_env(cfg, task_name, variation, headless):
    task_class = task_file_to_task_class(task_name, True)

    obs_config = create_obs_config(
        camera_names=list(cfg.rlbench.cameras),
        camera_resolution=list(cfg.rlbench.camera_resolution),
        method_name=cfg.method.name,
        robot_name="bimanual",
    )

    env = CustomRLBenchEnv(
        task_class=task_class,
        observation_config=obs_config,
        action_mode=make_action_mode(),
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


def safe_get_action_from_obs(obs):
    misc = getattr(obs, "misc", None)
    if misc is None:
        return None

    right = misc.get("right_executed_demo_joint_position_action", None)
    left = misc.get("left_executed_demo_joint_position_action", None)

    if right is None or left is None:
        return None

    right = np.asarray(right, dtype=np.float32)
    left = np.asarray(left, dtype=np.float32)

    return np.concatenate([right, left]).astype(np.float32)

def make_9d_action_from_pose7(
    pose7,
    gripper_open=True,
    ignore_collisions=0.0,
):
    pose7 = np.asarray(pose7, dtype=np.float32)

    if pose7.shape != (7,):
        raise ValueError(f"pose7 must be shape (7,), got {pose7.shape}")

    return np.array(
        [
            pose7[0], pose7[1], pose7[2],
            pose7[3], pose7[4], pose7[5], pose7[6],
            float(gripper_open),
            float(ignore_collisions),
        ],
        dtype=np.float32,
    )


def get_pose7_from_dummy(dummy_name: str) -> np.ndarray:
    dummy = Dummy(dummy_name)
    pos = dummy.get_position()
    quat = dummy.get_quaternion()

    return np.array(
        [
            pos[0], pos[1], pos[2],
            quat[0], quat[1], quat[2], quat[3],
        ],
        dtype=np.float32,
    )


def make_9d_action_from_waypoint(
    waypoint_name: str,
    gripper_open: bool,
    ignore_collisions: float = 0.0,
) -> np.ndarray:
    pose7 = get_pose7_from_dummy(waypoint_name)

    return make_9d_action_from_pose7(
        pose7,
        gripper_open=gripper_open,
        ignore_collisions=ignore_collisions,
    )

def make_waypoint_pair_action_18d(keyframe_id: int) -> np.ndarray:
    waypoint_pairs = [
        ("waypoint0", "waypoint2"),
        ("waypoint1", "waypoint3"),
    ]

    right_wp, left_wp = waypoint_pairs[keyframe_id]

    right_pose7 = get_pose7_from_dummy(right_wp)
    left_pose7 = get_pose7_from_dummy(left_wp)

    print(
        f"[waypoint action debug] keyframe={keyframe_id} "
        f"right_wp={right_wp} pose7={right_pose7} "
        f"left_wp={left_wp} pose7={left_pose7}"
    )

    right_action_9d = make_9d_action_from_pose7(
        right_pose7,
        gripper_open=False,
        ignore_collisions=0.0,
    )

    left_action_9d = make_9d_action_from_pose7(
        left_pose7,
        gripper_open=False,
        ignore_collisions=0.0,
    )

    return np.concatenate([right_action_9d, left_action_9d]).astype(np.float32)

def cache_waypoint_pair_actions_18d():
    """
    init_episode 直後、demo 実行前に waypoint action を固定しておく。
    get_demos / scene.get_demo 後に waypoint pose が変わる可能性があるため、
    保存時には必ずこの cache を使う。
    """

    actions = []

    for keyframe_id in range(2):
        action_18d = make_waypoint_pair_action_18d(keyframe_id)
        actions.append(action_18d.copy())

        print(
            f"[waypoint cache] keyframe={keyframe_id} "
            f"right_xyz={action_18d[:3]} right_grip={action_18d[7]} "
            f"left_xyz={action_18d[9:12]} left_grip={action_18d[16]}"
        )

    return actions

def has_executed_demo_action(obs) -> bool:
    """
    obs.misc に demo 実行中 action が入っているかだけを見る。
    14D/18D など shape は問わない。
    keyframe segment 検出用。
    """
    misc = getattr(obs, "misc", None)
    if misc is None:
        return False

    right = misc.get("right_executed_demo_joint_position_action", None)
    left = misc.get("left_executed_demo_joint_position_action", None)

    return right is not None and left is not None


    """
    1つの keyframe segment から、学習用 transition を作る。
    action は obs_after の EE pose から 9D+9D=18D として作る。
    """

    right_pose7 = get_pose7_from_obs(obs_after, "right")
    left_pose7 = get_pose7_from_obs(obs_after, "left")

    right_open = get_gripper_open_from_obs(obs_after, "right")
    left_open = get_gripper_open_from_obs(obs_after, "left")

    right_action_9d = make_9d_action_from_pose7(
        right_pose7,
        gripper_open=right_open,
        ignore_collisions=0.0,
    )

    left_action_9d = make_9d_action_from_pose7(
        left_pose7,
        gripper_open=left_open,
        ignore_collisions=0.0,
    )

    full_action_18d = np.concatenate(
        [right_action_9d, left_action_9d]
    ).astype(np.float32)

    return {
        "obs": env.extract_obs(obs_before),
        "next_obs": env.extract_obs(obs_after),

        "target_arm": "both",
        "target_action_18d": full_action_18d.copy(),
        "full_action": full_action_18d.copy(),

        "right_action_9d": right_action_9d.copy(),
        "left_action_9d": left_action_9d.copy(),

        "target_action_9d": None,
        "human_action": None,

        "success": False,
        "terminate": False,
        "step_id": int(step_id),

        "info": {
            "source": "rlbench_expert_keyframe_ee18",
            "demo_start_t": int(start_t),
            "demo_end_t": int(end_t),
            "task": task_name,
            "variation": int(variation),
            "episode_id": int(episode_id),
            "description": description,
            "right_pose7": right_pose7.copy(),
            "left_pose7": left_pose7.copy(),
            "right_gripper_open": bool(right_open),
            "left_gripper_open": bool(left_open),
        },
    }

def extract_keyframe_steps_from_demo(
    env,
    demo,
    task_name: str,
    variation: int,
    episode_id: int,
    description: str,
    cached_actions_18d,
):
    """
    RLBench live demo から、学習用 keyframe transition を抽出する。

    方針:
    - obs.misc に action がある連続区間を 1つの keyframe segment とみなす
    - obs は「前 keyframe 後の観測」
    - next_obs は「今回 keyframe segment の終端観測」
    - action は obs.misc の 14D joint action ではなく、
      CoppeliaSim waypoint pose から作った 18D EE action を使う

    bimanual_lift_long_block では:
      keyframe0: right=waypoint0, left=waypoint2
      keyframe1: right=waypoint1, left=waypoint3
    """

    steps = []

    if len(demo) == 0:
        return steps

    prev_keyframe_obs = demo[0]

    in_segment = False
    prev_has_action = False

    segment_start_t = None
    segment_start_obs = None
    last_obs_in_segment = None

    for t, obs in enumerate(demo):
        has_action = has_executed_demo_action(obs)

        # action が出始めた瞬間 = keyframe segment 開始
        if has_action and not prev_has_action:
            in_segment = True
            segment_start_t = t
            segment_start_obs = prev_keyframe_obs
            last_obs_in_segment = obs

        # action が出ている間は segment 終端候補を更新
        if in_segment and has_action:
            last_obs_in_segment = obs

        # action が消えた瞬間 = 1 keyframe segment 終了
        if (not has_action) and prev_has_action and in_segment:
            segment_end_t = t - 1
            keyframe_id = len(steps)

            # 今回は waypoint pair が 2つだけなので、余分な segment は無視
            if keyframe_id >= 2:
                print(
                    f"[keyframe extract] ignore extra segment "
                    f"keyframe_id={keyframe_id} "
                    f"start_t={segment_start_t} end_t={segment_end_t}"
                )
            else:
                full_action_18d = np.asarray(cached_actions_18d[keyframe_id], dtype=np.float32).copy()
                right_action_9d = full_action_18d[:9].copy()
                left_action_9d = full_action_18d[9:18].copy()

                step = {
                    "obs": env.extract_obs(segment_start_obs),
                    "next_obs": env.extract_obs(last_obs_in_segment),

                    "target_arm": "both",
                    "target_action_18d": full_action_18d.copy(),
                    "full_action": full_action_18d.copy(),

                    "right_action_9d": right_action_9d,
                    "left_action_9d": left_action_9d,

                    "target_action_9d": None,
                    "human_action": None,

                    "success": False,
                    "terminate": False,
                    "step_id": keyframe_id,

                    "info": {
                        "source": "rlbench_expert_keyframe_waypoint_ee18",
                        "demo_start_t": int(segment_start_t),
                        "demo_end_t": int(segment_end_t),
                        "task": task_name,
                        "variation": int(variation),
                        "episode_id": int(episode_id),
                        "description": description,
                        "right_waypoint": "waypoint0" if keyframe_id == 0 else "waypoint1",
                        "left_waypoint": "waypoint2" if keyframe_id == 0 else "waypoint3",
                    },
                }

                steps.append(step)

                print(
                    f"[keyframe extract] add keyframe={keyframe_id} "
                    f"segment=({segment_start_t}->{segment_end_t}) "
                    f"right_xyz={right_action_9d[:3]} "
                    f"right_grip={right_action_9d[7]} "
                    f"left_xyz={left_action_9d[:3]} "
                    f"left_grip={left_action_9d[7]}"
                )

            # 次 keyframe の obs は、今回 keyframe 実行後の観測
            prev_keyframe_obs = last_obs_in_segment

            in_segment = False
            segment_start_t = None
            segment_start_obs = None
            last_obs_in_segment = None

        prev_has_action = has_action

    # demo の最後まで action が続いていた場合
    if in_segment and last_obs_in_segment is not None:
        segment_end_t = len(demo) - 1
        keyframe_id = len(steps)

        if keyframe_id < 2:
            full_action_18d = np.asarray(cached_actions_18d[keyframe_id], dtype=np.float32).copy()
            right_action_9d = full_action_18d[:9].copy()
            left_action_9d = full_action_18d[9:18].copy()

            step = {
                "obs": env.extract_obs(segment_start_obs),
                "next_obs": env.extract_obs(last_obs_in_segment),

                "target_arm": "both",
                "target_action_18d": full_action_18d.copy(),
                "full_action": full_action_18d.copy(),

                "right_action_9d": right_action_9d,
                "left_action_9d": left_action_9d,

                "target_action_9d": None,
                "human_action": None,

                "success": False,
                "terminate": False,
                "step_id": keyframe_id,

                "info": {
                    "source": "rlbench_expert_keyframe_waypoint_ee18",
                    "demo_start_t": int(segment_start_t),
                    "demo_end_t": int(segment_end_t),
                    "task": task_name,
                    "variation": int(variation),
                    "episode_id": int(episode_id),
                    "description": description,
                    "right_waypoint": "waypoint0" if keyframe_id == 0 else "waypoint1",
                    "left_waypoint": "waypoint2" if keyframe_id == 0 else "waypoint3",
                },
            }

            steps.append(step)

            print(
                f"[keyframe extract] add keyframe={keyframe_id} "
                f"segment=({segment_start_t}->{segment_end_t}) "
                f"right_xyz={right_action_9d[:3]} "
                f"right_grip={right_action_9d[7]} "
                f"left_xyz={left_action_9d[:3]} "
                f"left_grip={left_action_9d[7]}"
            )

    if len(steps) > 0:
        steps[-1]["success"] = True
        steps[-1]["terminate"] = True

    return steps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="bimanual_lift_long_block")
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--save-root", type=str, default="data/expert_demos")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--config-dir", type=str, default=join(PROJECT_ROOT, "conf"))
    parser.add_argument("--config-name", type=str, default="config")
    parser.add_argument("--max-demo-attempts", type=int, default=10)
    args = parser.parse_args()

    overrides = [
        "method=PERACT_BC",
        "method.agent_type=independent",
        "framework.anybimanual=True",
        "framework.checkpoint_name_prefix=checkpoint",
        "ddp.num_devices=1",
    ]

    cfg = load_cfg(args.config_dir, args.config_name, overrides)

    env = None
    try:
        env = make_env(
            cfg=cfg,
            task_name=args.task,
            variation=args.variation,
            headless=args.headless,
        )

        os.makedirs(args.save_root, exist_ok=True)

        for ep in range(args.num_episodes):
            print("=" * 80)
            print(f"[expert collect] episode={ep}")

            env._task.set_variation(args.variation)

            description = "lift the long block with both arms"
            env._lang_goal = description

            try:
                # reset で init_episode が走る。
                # ここでブロック初期位置ランダム化も走る。
                env.reset()

                # 重要:
                # demo 実行前の waypoint pose を cache する。
                cached_actions_18d = cache_waypoint_pair_actions_18d()

                # reset 済みの current episode から expert demo を生成する。
                # randomly_place=False にして、ここで再ランダム化させない。
                demo = env._task._scene.get_demo(
                    record=True,
                    randomly_place=False,
                )

            except Exception as e:
                import traceback
                print(f"[expert collect] demo failed: episode={ep}, error={e}")
                traceback.print_exc()
                continue

            print(f"[expert collect] demo length={len(demo)}")

            keyframe_steps = extract_keyframe_steps_from_demo(
                env=env,
                demo=demo,
                task_name=args.task,
                variation=args.variation,
                episode_id=ep,
                description=description,
                cached_actions_18d=cached_actions_18d,
            )

            print(
                f"[expert collect] episode={ep} "
                f"num_demo_obs={len(demo)} "
                f"num_keyframes={len(keyframe_steps)}"
            )

            for k, step in enumerate(keyframe_steps):
                action = step["full_action"]
                info = step.get("info", {})

                print(
                    f"  keyframe={k} "
                    f"demo_segment=({info.get('demo_start_t')}->{info.get('demo_end_t')}) "
                    f"right_wp={info.get('right_waypoint')} "
                    f"left_wp={info.get('left_waypoint')} "
                    f"right_xyz={action[:3]} "
                    f"right_grip={action[7]} "
                    f"left_xyz={action[9:12]} "
                    f"left_grip={action[16]}"
                )

            if len(keyframe_steps) == 0:
                print(
                    "[expert collect] WARNING: no keyframe actions extracted. "
                    "Check obs.misc keys."
                )

                for t, obs in enumerate(demo[:10]):
                    misc = getattr(obs, "misc", {}) or {}
                    print(f"  [debug misc] t={t} keys={list(misc.keys())}")

                continue

            episode_data = {
                "meta": {
                    "task": args.task,
                    "variation": int(args.variation),
                    "episode_id": int(ep),
                    "mode": "expert_keyframe_joint_bc",
                    "description": description,
                    "num_demo_obs": int(len(demo)),
                    "num_keyframes": int(len(keyframe_steps)),
                    "action_dim": 18,
                    "action_type": "right_9d_plus_left_9d",
                },
                "steps": keyframe_steps,
            }

            save_path = join(
                args.save_root,
                f"{args.task}_expert_keyframe_ep{ep:04d}.pkl",
            )

            with open(save_path, "wb") as f:
                pickle.dump(episode_data, f)

            print(f"[expert collect] saved keyframe demo: {save_path}")
    finally:
        if env is not None:
            env.shutdown()


if __name__ == "__main__":
    main()