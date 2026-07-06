# Standard Library
import os
import sys
import time
import argparse
from os.path import join, dirname, abspath

# pygame の余計なログを消す
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")

# Project paths
PROJECT_ROOT = dirname(dirname(abspath(__file__)))
sys.path.insert(0, join(PROJECT_ROOT, "third_party", "RLBench"))
sys.path.insert(0, join(PROJECT_ROOT, "third_party", "PyRep"))

# Third Party
from rlbench import ObservationConfig
from rlbench.environment import Environment
from rlbench.backend.utils import task_file_to_task_class

from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import BimanualDiscrete
from rlbench.action_modes.arm_action_modes import BimanualEndEffectorPoseViaPlanning

import numpy as np

# Local
# from r2bc.teleop import JoystickTeleop
from r2bc.teleop_toggle import JoystickTeleop
from r2bc.replay_buffer import R2BCEpisodeBuffer, make_episode_path
from r2bc.policies import make_policy


DT = 0.05
MAX_STEPS = 500

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
    まずは rotation は固定する。
    """
    target_pose = pose7.copy()
    target_pose[0] += float(action_4d[0])
    target_pose[1] += float(action_4d[1])
    target_pose[2] += float(action_4d[2])
    return target_pose


def make_idle_9d_action(robot, arm_name, gripper_open=True):
    arm = get_arm_by_name(robot, arm_name)
    pose7 = get_pose7_from_arm(arm)
    return make_9d_action_from_pose7(
        pose7,
        gripper_open=gripper_open,
        ignore_collisions=0.0,
    )


def make_delta_9d_action(robot, arm_name, action_4d, gripper_open=True):
    arm = get_arm_by_name(robot, arm_name)
    current_pose7 = get_pose7_from_arm(arm)
    target_pose7 = apply_delta_to_pose7(current_pose7, action_4d)
    return make_9d_action_from_pose7(
        target_pose7,
        gripper_open=gripper_open,
        ignore_collisions=0.0,
    )

def make_obs_config():
    """
    最初は画像なしで軽くする。
    joint position, gripper pose などの低次元情報だけ取る。
    """
    obs_config = ObservationConfig()
    obs_config.set_all(False)

    obs_config.joint_positions = True
    obs_config.joint_velocities = True
    obs_config.gripper_open = True
    obs_config.gripper_pose = True
    obs_config.gripper_joint_positions = True
    obs_config.task_low_dim_state = True

    return obs_config


def make_env(headless=False):
    """
    RLBench Environmentを作る。
    今回は task_env.step(action) を使わず直接 robot を動かすが、
    Environment生成には action_mode が必要。
    """
    obs_config = make_obs_config()

    env = Environment(
        action_mode=BimanualMoveArmThenGripper(
            BimanualEndEffectorPoseViaPlanning(),
            BimanualDiscrete(),
        ),
        obs_config=obs_config,
        robot_setup="dual_panda",
        headless=headless,
    )

    env.launch()
    return env

def sync_grasp_attachment(robot, task, arm_name, was_closed, is_closed):
    """
    RLBench/PyRep側の grasp attachment を同期する。

    was_closed=False, is_closed=True:
        closeした瞬間なので graspable objects をattach

    was_closed=True, is_closed=False:
        openした瞬間なので release
    """
    if (not was_closed) and is_closed:
        for obj in task.get_graspable_objects():
            robot.grasp(obj, arm_name)

    if was_closed and (not is_closed):
        robot.release_gripper(arm_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="bimanual_lift_long_block")
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--human-arm", type=str, default="right", choices=["right", "left"])
    parser.add_argument("--alternate-human-arm", action="store_true")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--verbose-ik", action="store_true")
    parser.add_argument("--save-root", type=str, default="data/r2bc")
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--policy", type=str, default="idle", choices=["idle", "constant_up", "constant_down", "scripted_lift"])
    args = parser.parse_args()

    env = None
    teleop = None

    try:
        print("[collect_rlbench] Launching RLBench...")
        env = make_env(headless=args.headless)

        task_class = task_file_to_task_class(args.task, True)
        task_env = env.get_task(task_class)

        print("[collect_rlbench] Task:", task_env.get_name())
        if args.alternate_human_arm:
            print("[collect_rlbench] Human arm: alternating")
        else:
            print("[collect_rlbench] Human arm:", args.human_arm)
        print("[collect_rlbench] Policy:", args.policy)

        teleop = JoystickTeleop()

        robot = task_env._robot
        scene = task_env._scene

        print("[collect_rlbench] Teleop started.")
        print("  stick  : move human arm in z")
        print("  button0: toggle human gripper")
        print("  Ctrl+C : quit")

        # input("Press Enter to start collecting episodes...")

        for ep_i in range(args.num_episodes):
            episode_id = args.episode_id + ep_i

            if args.alternate_human_arm:
                human_arm = "right" if episode_id % 2 == 0 else "left"
            else:
                human_arm = args.human_arm

            policy_arm = "left" if human_arm == "right" else "right"

            print("=" * 80)
            print(f"[collect_rlbench] Episode {episode_id}")
            print(f"[collect_rlbench] human_arm={human_arm}, policy_arm={policy_arm}")
            
            task_env.set_variation(args.variation)
            descriptions, obs = task_env.reset()

            description = descriptions[0] if len(descriptions) > 0 else None

            policy = make_policy(args.policy)
            policy.reset()

            episode_buffer = R2BCEpisodeBuffer(
                task_name=task_env.get_name(),
                variation=args.variation,
                episode_id=episode_id,
                human_arm=human_arm,
                description=description,
            )

            right_closed = False
            left_closed = False

            prev_gripper_button_pressed = False
            prev_policy_gripper_button_pressed = False

            for step_id in range(args.max_steps):
                prev_obs = obs

                # action_4d = [dx, dy, dz, gripper_button]
                human_action = teleop.read_action().as_array()

                # -------------------------
                # human gripper toggle
                # -------------------------
                gripper_button_pressed = bool(human_action[3] > 0.5)
                gripper_toggled = False

                if gripper_button_pressed and not prev_gripper_button_pressed:
                    if human_arm == "right":
                        was_closed = right_closed
                        right_closed = not right_closed
                        sync_grasp_attachment(
                            robot=robot,
                            task=task_env._task,
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
                            task=task_env._task,
                            arm_name="left",
                            was_closed=was_closed,
                            is_closed=left_closed,
                        )
                        print(f"[human gripper] left_closed -> {left_closed}")

                    gripper_toggled = True

                prev_gripper_button_pressed = gripper_button_pressed

                # -------------------------
                # policy action
                # -------------------------
                policy_action_4d = policy.act(obs, policy_arm)
                policy_action_4d = np.asarray(policy_action_4d, dtype=np.float32)

                if policy_action_4d.shape != (4,):
                    raise ValueError(
                        f"policy_action_4d must be shape (4,), got {policy_action_4d.shape}"
                    )

                # -------------------------
                # policy gripper toggle
                # -------------------------
                policy_gripper_button_pressed = bool(policy_action_4d[3] > 0.5)
                policy_gripper_toggled = False

                if policy_gripper_button_pressed and not prev_policy_gripper_button_pressed:
                    if policy_arm == "right":
                        was_closed = right_closed
                        right_closed = not right_closed
                        sync_grasp_attachment(
                            robot=robot,
                            task=task_env._task,
                            arm_name="right",
                            was_closed=was_closed,
                            is_closed=right_closed,
                        )
                        print(f"[policy gripper] right_closed -> {right_closed}")
                    else:
                        was_closed = left_closed
                        left_closed = not left_closed
                        sync_grasp_attachment(
                            robot=robot,
                            task=task_env._task,
                            arm_name="left",
                            was_closed=was_closed,
                            is_closed=left_closed,
                        )
                        print(f"[policy gripper] left_closed -> {left_closed}")

                    policy_gripper_toggled = True

                prev_policy_gripper_button_pressed = policy_gripper_button_pressed

                # True = open, False = close
                right_gripper_open = not right_closed
                left_gripper_open = not left_closed


                if human_arm == "right":
                    right_action = make_delta_9d_action(
                        robot=robot,
                        arm_name="right",
                        action_4d=human_action,
                        gripper_open=right_gripper_open,
                    )

                    left_action = make_delta_9d_action(
                        robot=robot,
                        arm_name="left",
                        action_4d=policy_action_4d,
                        gripper_open=left_gripper_open,
                    )

                else:
                    right_action = make_delta_9d_action(
                        robot=robot,
                        arm_name="right",
                        action_4d=policy_action_4d,
                        gripper_open=right_gripper_open,
                    )

                    left_action = make_delta_9d_action(
                        robot=robot,
                        arm_name="left",
                        action_4d=human_action,
                        gripper_open=left_gripper_open,
                    )

                full_action = np.concatenate([right_action, left_action]).astype(np.float32)
                if step_id % 10 == 0:
                    print("[debug] right_action xyz:", right_action[:3])
                    print("[debug] left_action  xyz:", left_action[:3])

                try:
                    next_obs, reward, env_terminate = task_env.step(full_action)
                    step_ok = True
                except Exception as e:
                    print(f"[collect_rlbench] task_env.step failed at step={step_id}: {e}")
                    next_obs = obs
                    reward = 0.0
                    env_terminate = True
                    step_ok = False

                if step_ok:
                    success, task_terminate = task_env._task.success()
                else:
                    success = False
                    task_terminate = True

                terminate = bool(env_terminate or task_terminate)

                episode_buffer.add(
                    obs=prev_obs,
                    human_action=human_action,
                    next_obs=next_obs,
                    success=success,
                    terminate=terminate,
                    step_id=step_id,
                    policy_arm=policy_arm,
                    policy_action=policy_action_4d,
                    full_action=full_action,
                    info={
                        "step_ok": step_ok,
                        "reward": reward,
                        "right_closed": right_closed,
                        "left_closed": left_closed,
                        "right_gripper_open": right_gripper_open,
                        "left_gripper_open": left_gripper_open,

                        "gripper_button_pressed": gripper_button_pressed,
                        "gripper_toggled": gripper_toggled,
                        "policy_gripper_button_pressed": policy_gripper_button_pressed,
                        "policy_gripper_toggled": policy_gripper_toggled,

                        "policy_action_4d": policy_action_4d,
                        "right_action_9d": right_action,
                        "left_action_9d": left_action,
                        "policy_name": args.policy,
                        "control_mode": "bimanual_ee_pose_via_planning",
                    }
                )

                obs = next_obs

                if step_id % 10 == 0 or success or terminate:
                    print(
                        f"[debug] human_arm={human_arm} "
                        f"policy_arm={policy_arm} "
                        f"human_action={human_action} "
                        f"policy_action_4d={policy_action_4d} "
                        f"right_xyz={right_action[:3]} "
                        f"left_xyz={left_action[:3]}"
                    )

                if terminate:
                    print("[collect_rlbench] Terminated.")
                    break

                time.sleep(DT)
            save_path = make_episode_path(
                save_root=args.save_root,
                task_name=task_env.get_name(),
                episode_id=episode_id,
                human_arm=human_arm,
            )
            saved_path = episode_buffer.save(save_path)
            print(f"[collect_rlbench] Episode saved to {saved_path}")
            print(f"[collect_rlbench] Summary: {episode_buffer.summary()}")

    except KeyboardInterrupt:
        print("\n[collect_rlbench] Stopping by Ctrl+C...")

    finally:
        if teleop is not None:
            teleop.close()

        if env is not None:
            env.shutdown()

        print("[collect_rlbench] Shutdown complete.")


if __name__ == "__main__":
    main()