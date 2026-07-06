# r2bc/eval_joint_bc_lowdim.py

import os
import sys
import argparse
from os.path import join, dirname, abspath

import numpy as np
import torch

PROJECT_ROOT = dirname(dirname(abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, join(PROJECT_ROOT, "third_party", "RLBench"))
sys.path.insert(0, join(PROJECT_ROOT, "third_party", "PyRep"))
sys.path.insert(0, join(PROJECT_ROOT, "third_party", "YARR"))

try:
    from hydra import compose, initialize_config_dir
except ImportError:
    from hydra.experimental import compose, initialize_config_dir

from rlbench.backend.utils import task_file_to_task_class
from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import BimanualDiscrete
from rlbench.action_modes.arm_action_modes import BimanualEndEffectorPoseViaPlanning

from helpers.custom_rlbench_env import CustomRLBenchEnv
from helpers.observation_utils import create_obs_config

from r2bc.train_joint_bc_lowdim import JointBCLowDimMLP


PROJECT_ROOT = dirname(dirname(abspath(__file__)))


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


def make_action_mode(attach_grasped_objects: bool):
    return BimanualMoveArmThenGripper(
        BimanualEndEffectorPoseViaPlanning(),
        BimanualDiscrete(
            attach_grasped_objects=attach_grasped_objects,
        ),
    )


def make_env(cfg, task_name, variation, headless, attach_grasped_objects):
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
        action_mode=make_action_mode(
            attach_grasped_objects=attach_grasped_objects,
        ),
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


def make_input_from_obs(obs_dict, keyframe_id: int, num_keyframes: int):
    right_low = np.asarray(obs_dict["right_low_dim_state"], dtype=np.float32)
    left_low = np.asarray(obs_dict["left_low_dim_state"], dtype=np.float32)

    key_onehot = np.zeros(num_keyframes, dtype=np.float32)
    key_onehot[int(keyframe_id)] = 1.0

    x = np.concatenate([right_low, left_low, key_onehot]).astype(np.float32)
    return x


def normalize_quat(q):
    q = np.asarray(q, dtype=np.float32)
    norm = float(np.linalg.norm(q))

    if norm < 1e-6:
        # fallback: identity-ish
        out = np.zeros(4, dtype=np.float32)
        out[3] = 1.0
        return out

    return (q / norm).astype(np.float32)


def sanitize_action_18d(
    action,
    force_gripper_close: bool = True,
    force_ignore_collisions: float = 0.0,
):
    """
    MLP の連続出力を RLBench の planning action として使えるように整える。
    action format:
      right: [x,y,z,qx,qy,qz,qw,gripper_open,ignore_collisions]
      left : [x,y,z,qx,qy,qz,qw,gripper_open,ignore_collisions]
    """

    a = np.asarray(action, dtype=np.float32).copy()

    if a.shape != (18,):
        raise ValueError(f"action must be shape (18,), got {a.shape}")

    # quaternion normalization
    a[3:7] = normalize_quat(a[3:7])
    a[12:16] = normalize_quat(a[12:16])

    if force_gripper_close:
        # このタスクでは掴んで持ち上げるため、両方 close に固定する。
        # gripper_open: 1=open, 0=close
        a[7] = 0.0
        a[16] = 0.0
    else:
        a[7] = 1.0 if a[7] > 0.5 else 0.0
        a[16] = 1.0 if a[16] > 0.5 else 0.0

    a[8] = float(force_ignore_collisions)
    a[17] = float(force_ignore_collisions)

    return a


def predict_action(
    model,
    obs_dict,
    keyframe_id,
    ckpt,
    device,
):
    num_keyframes = int(ckpt["num_keyframes"])

    x = make_input_from_obs(
        obs_dict=obs_dict,
        keyframe_id=keyframe_id,
        num_keyframes=num_keyframes,
    )

    x_norm = (x - ckpt["x_mean"]) / ckpt["x_std"]

    with torch.no_grad():
        pred_norm = model(
            torch.from_numpy(x_norm).float().unsqueeze(0).to(device)
        )[0].cpu().numpy()

    action = pred_norm * ckpt["y_std"] + ckpt["y_mean"]
    return action.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--task", type=str, default="bimanual_lift_long_block")
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--config-dir", type=str, default=join(PROJECT_ROOT, "conf"))
    parser.add_argument("--config-name", type=str, default="config")

    parser.add_argument(
        "--attach-grasped-objects",
        action="store_true",
        help="Use RLBench attachment on gripper close. If disabled, relies on physics only.",
    )
    parser.add_argument(
        "--no-force-gripper-close",
        action="store_true",
        help="Use model gripper output instead of forcing close.",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=0.0,
        help="Sleep after each keyframe for visual inspection.",
    )

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)

    model = JointBCLowDimMLP(
        input_dim=ckpt["input_dim"],
        output_dim=ckpt["output_dim"],
        hidden_dim=ckpt["hidden_dim"],
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

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
            attach_grasped_objects=args.attach_grasped_objects,
        )

        successes = []

        for ep in range(args.num_episodes):
            print("=" * 80)
            print(f"[eval joint bc lowdim] episode={ep}")

            env._task.set_variation(args.variation)

            obs_dict = env.reset()

            description = "lift the long block with both arms"
            env._lang_goal = description

            ep_success = False
            ep_terminate = False

            for keyframe_id in range(int(ckpt["num_keyframes"])):
                raw_action = predict_action(
                    model=model,
                    obs_dict=obs_dict,
                    keyframe_id=keyframe_id,
                    ckpt=ckpt,
                    device=device,
                )

                action = sanitize_action_18d(
                    raw_action,
                    force_gripper_close=not args.no_force_gripper_close,
                    force_ignore_collisions=0.0,
                )

                print(
                    f"[eval] ep={ep} keyframe={keyframe_id}\n"
                    f"  raw_right_xyz={raw_action[:3]} raw_right_grip={raw_action[7]:.3f}\n"
                    f"  raw_left_xyz ={raw_action[9:12]} raw_left_grip ={raw_action[16]:.3f}\n"
                    f"  exec_right_xyz={action[:3]} exec_right_grip={action[7]} collide={action[8]}\n"
                    f"  exec_left_xyz ={action[9:12]} exec_left_grip ={action[16]} collide={action[17]}"
                )

                try:
                    raw_obs_tp1, reward, terminate = env._task.step(action)
                    obs_dict = env.extract_obs(raw_obs_tp1)
                except Exception as e:
                    import traceback
                    print(f"[eval] step failed: ep={ep} keyframe={keyframe_id} error={e}")
                    traceback.print_exc()
                    ep_success = False
                    ep_terminate = True
                    break

                try:
                    success, task_terminate = env._task._task.success()
                except Exception:
                    success, task_terminate = False, False

                ep_success = bool(success)
                ep_terminate = bool(terminate or task_terminate)

                print(
                    f"[eval] after keyframe={keyframe_id} "
                    f"reward={reward} success={ep_success} terminate={ep_terminate}"
                )

                if args.hold_seconds > 0:
                    import time
                    time.sleep(args.hold_seconds)

                if ep_terminate:
                    break

            successes.append(float(ep_success))

            print(
                f"[eval] episode={ep} final_success={ep_success} "
                f"running_success_rate={np.mean(successes):.3f}"
            )

        print("=" * 80)
        print("[eval joint bc lowdim] done")
        print(f"num_episodes={args.num_episodes}")
        print(f"successes={successes}")
        print(f"success_rate={np.mean(successes):.3f}")

    finally:
        if env is not None:
            env.shutdown()


if __name__ == "__main__":
    main()