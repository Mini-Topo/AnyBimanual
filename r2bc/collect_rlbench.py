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
from rlbench.action_modes.arm_action_modes import BimanualJointPosition
from rlbench.action_modes.gripper_action_modes import BimanualDiscrete

# Local
# from r2bc.teleop import JoystickTeleop
from r2bc.teleop_toggle import JoystickTeleop
from r2bc.ee_controller import apply_single_arm_action, set_gripper_target
from r2bc.replay_buffer import R2BCEpisodeBuffer, make_episode_path
from r2bc.policies import make_policy


DT = 0.05
MAX_STEPS = 500


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
            BimanualJointPosition(),
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
    parser.add_argument("--policy", type=str, default="idle", choices=["idle", "constant_up", "scripted_lift"])
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

        input("Press Enter to start collecting episodes...")

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

            for step_id in range(args.max_steps):
                # action = [dx, dy, dz, gripper_toggle]
                human_action = teleop.read_action().as_array()
                policy_action = policy.act(obs, policy_arm)

                if human_arm == "right":
                    prev_right_closed = right_closed
                    ik_ok_human, right_closed = apply_single_arm_action(
                        arm=robot.right_arm,
                        gripper=robot.right_gripper,
                        action=human_action,
                        gripper_closed=right_closed,
                        verbose=args.verbose_ik,
                    )

                    sync_grasp_attachment(
                        robot=robot,
                        task=task_env._task,
                        arm_name="right",
                        was_closed=prev_right_closed,
                        is_closed=right_closed,
                    )

                    prev_left_closed = left_closed
                    ik_ok_policy, left_closed = apply_single_arm_action(
                        arm=robot.left_arm,
                        gripper=robot.left_gripper,
                        action=policy_action,
                        gripper_closed=left_closed,
                        verbose=args.verbose_ik,
                    )

                    sync_grasp_attachment(
                        robot=robot,
                        task=task_env._task,
                        arm_name="left",
                        was_closed=prev_left_closed,
                        is_closed=left_closed,
                    )

                else:
                    prev_right_closed = right_closed
                    ik_ok_policy, right_closed = apply_single_arm_action(
                        arm=robot.right_arm,
                        gripper=robot.right_gripper,
                        action=policy_action,
                        gripper_closed=right_closed,
                        verbose=args.verbose_ik,
                    )

                    sync_grasp_attachment(
                        robot=robot,
                        task=task_env._task,
                        arm_name="right",
                        was_closed=prev_right_closed,
                        is_closed=right_closed,
                    )

                    prev_left_closed = left_closed
                    ik_ok_human, left_closed = apply_single_arm_action(
                        arm=robot.left_arm,
                        gripper=robot.left_gripper,
                        action=human_action,
                        gripper_closed=left_closed,
                        verbose=args.verbose_ik,
                    )

                    sync_grasp_attachment(
                        robot=robot,
                        task=task_env._task,
                        arm_name="left",
                        was_closed=prev_left_closed,
                        is_closed=left_closed,
                    )

                prev_obs = obs

                scene.step()

                next_obs = scene.get_observation()
                success, terminate = task_env._task.success()

                episode_buffer.add(
                    obs=prev_obs,
                    human_action=human_action,
                    next_obs=next_obs,
                    success=success,
                    terminate=terminate,
                    step_id=step_id,
                    policy_arm=policy_arm,
                    policy_action=policy_action,
                    full_action=None,
                    info={
                        "ik_ok_human": ik_ok_human,
                        "ik_ok_policy": ik_ok_policy,
                        "right_closed": right_closed,
                        "left_closed": left_closed,
                        "policy_name": args.policy,
                    },
                )

                obs = next_obs

                if step_id % 10 == 0 or success or terminate:
                    print(
                        f"episode={episode_id:06d} "
                        f"step={step_id:04d} "
                        f"success={success} "
                        f"terminate={terminate} "
                        f"ik_ok_human={ik_ok_human} "
                        f"right_closed={right_closed} "
                        f"left_closed={left_closed}"
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