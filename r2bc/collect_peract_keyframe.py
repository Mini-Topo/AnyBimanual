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

from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape
from pyrep.backend import sim

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

from typing import Optional
import ctypes


def print_current_physics_engine():
    try:
        engine_id = sim.simGetInt32Parameter(sim.sim_intparam_dynamic_engine)

        engine_names_guess = {
            0: "Bullet",
            1: "ODE",
            2: "Vortex",
            3: "Newton",
            4: "MuJoCo or newer Bullet depending on version",
        }

        print(
            "[physics engine] "
            f"engine_id={engine_id} "
            f"name_guess={engine_names_guess.get(engine_id, 'UNKNOWN')}"
        )
    except Exception as e:
        print(f"[physics engine] failed to get dynamic engine: {e}")

def make_arm_progress_log_path(path: Optional[str], arm: str) -> Optional[str]:
    if path is None:
        return None

    root, ext = os.path.splitext(path)
    if ext == "":
        ext = ".jsonl"

    return f"{root}_{arm}{ext}"


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

def print_grasped_objects(robot, prefix="[grasp state]"):
    try:
        right_objs = [obj.get_name() for obj in robot.right_gripper.get_grasped_objects()]
    except Exception as e:
        right_objs = [f"<error: {e}>"]

    try:
        left_objs = [obj.get_name() for obj in robot.left_gripper.get_grasped_objects()]
    except Exception as e:
        left_objs = [f"<error: {e}>"]

    print(f"{prefix} right_grasped={right_objs} left_grasped={left_objs}")

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


def make_action_mode(grasp_mode="rlbench_attachment"):
    attach_grasped_objects = grasp_mode != "physics_only"

    print(
        "[make_action_mode] "
        f"grasp_mode={grasp_mode} "
        f"attach_grasped_objects={attach_grasped_objects}"
    )

    return BimanualMoveArmThenGripper(
        BimanualEndEffectorPoseViaPlanning(),
        BimanualDiscrete(
            attach_grasped_objects=attach_grasped_objects,
        ),
    )


def make_custom_env(cfg, task_name, variation, headless, grasp_mode):
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

    action_mode = make_action_mode(grasp_mode=grasp_mode)

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
    print_current_physics_engine()

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

def create_debug_dummy(name, position, size=0.05):
    """
    CoppeliaSim 上にデバッグ用 Dummy を出す。
    Shape と違い、カメラ画像や point cloud に写り込みにくい。
    position は world 座標 [x, y, z]。
    """
    dummy = Dummy.create(size=size)

    try:
        dummy.set_name(name)
    except Exception as e:
        print(f"[debug dummy] set_name failed: name={name}, error={e}")

    dummy.set_position(np.asarray(position, dtype=np.float32).tolist())
    return dummy


def safe_remove_debug_object(obj, name="debug_object"):
    """
    Dummy/Shape などの PyRep object を安全に消す。
    """
    if obj is None:
        return

    try:
        obj.remove()
    except Exception as e:
        print(f"[debug marker] remove failed: name={name}, error={e}")

def step_simulation_for_marker_update(env):
    """
    CoppeliaSim上のdebug marker表示を更新するために、
    RLBench Sceneを1 step進める。
    """
    try:
        env._task._scene.step()
        return True
    except Exception as e:
        print(f"[marker update] env._task._scene.step() failed: {e}")
        return False

def hold_scene_for_visual_check(
    env,
    seconds: float,
    message: str = "",
    update_fn=None,
):
    """
    CoppeliaSim の画面を一定時間維持する。
    update_fn を渡した場合、各 scene.step の前後で pseudo grasp などを更新する。
    """
    if message:
        print(message)

    if seconds <= 0:
        return

    t_end = time.time() + float(seconds)
    while time.time() < t_end:
        if update_fn is not None:
            update_fn()

        step_simulation_for_marker_update(env)

        if update_fn is not None:
            update_fn()

        time.sleep(0.05)

def get_arm_tip_xyz(robot, arm_name: str) -> np.ndarray:
    arm = get_arm_by_name(robot, arm_name)
    pose7 = get_pose7_from_arm(arm)
    return pose7[:3].copy()

def _safe_names(objs):
    names = []
    for obj in objs:
        try:
            names.append(obj.get_name())
        except Exception:
            names.append(str(obj))
    return names


def print_grasped_objects(robot, prefix="[grasp state]"):
    try:
        right_objs = _safe_names(robot.right_gripper.get_grasped_objects())
    except Exception as e:
        right_objs = [f"<error: {e}>"]

    try:
        left_objs = _safe_names(robot.left_gripper.get_grasped_objects())
    except Exception as e:
        left_objs = [f"<error: {e}>"]

    print(f"{prefix} right_grasped={right_objs} left_grasped={left_objs}")


def release_standard_grasps(robot):
    """
    RLBench/PyRep の標準 single-arm attachment を外す。
    gripper を開くのではなく、object attachment だけ release する想定。
    """
    try:
        robot.right_gripper.release()
    except Exception as e:
        print(f"[pseudo dual grasp] right_gripper.release failed: {e}")

    try:
        robot.left_gripper.release()
    except Exception as e:
        print(f"[pseudo dual grasp] left_gripper.release failed: {e}")


def update_pseudo_dual_grasp(
    robot,
    state: dict,
    right_action: np.ndarray,
    left_action: np.ndarray,
    item_name: str = "item0",
    activate_distance: float = 0.75,
    verbose: bool = False,
):
    """
    Task-specific pseudo dual grasp.

    標準 RLBench grasp は item0 を片腕にしか attach できないため、
    左右両方が close のときだけ item0 を左右 tip midpoint に追従させる。

    state:
      {
        "active": bool,
        "offset": np.ndarray shape (3,) or None,
      }
    """
    right_action = np.asarray(right_action, dtype=np.float32)
    left_action = np.asarray(left_action, dtype=np.float32)

    right_closed = bool(right_action[7] <= 0.5)
    left_closed = bool(left_action[7] <= 0.5)

    try:
        item = Shape(item_name)
    except Exception as e:
        print(f"[pseudo dual grasp] failed to get Shape({item_name}): {e}")
        return

    # どちらかが開いたら pseudo grasp 解除
    if not (right_closed and left_closed):
        if state.get("active", False):
            print("[pseudo dual grasp] deactivate: one or both grippers opened")
            try:
                item.set_dynamic(True)
                print("[pseudo dual grasp] item dynamic -> True")
            except Exception as e:
                print(f"[pseudo dual grasp] item.set_dynamic(True) failed: {e}")

        state["active"] = False
        state["offset"] = None
        return

    right_tip = get_arm_tip_xyz(robot, "right")
    left_tip = get_arm_tip_xyz(robot, "left")
    midpoint = 0.5 * (right_tip + left_tip)

    item_pos = np.asarray(item.get_position(), dtype=np.float32)

    # 初回 activation
    if not state.get("active", False):
        right_dist = float(np.linalg.norm(right_tip - item_pos))
        left_dist = float(np.linalg.norm(left_tip - item_pos))

        print(
            "[pseudo dual grasp] candidate "
            f"item_pos={item_pos} "
            f"right_tip={right_tip} "
            f"left_tip={left_tip} "
            f"right_dist={right_dist:.4f} "
            f"left_dist={left_dist:.4f}"
        )

        if right_dist > activate_distance or left_dist > activate_distance:
            print(
                "[pseudo dual grasp] not activated: tips too far from item "
                f"(threshold={activate_distance})"
            )
            return

        # 標準 single-arm attachment を解除してから pseudo に切り替える
        print_grasped_objects(robot, prefix="[pseudo dual grasp] before release")
        release_standard_grasps(robot)
        print_grasped_objects(robot, prefix="[pseudo dual grasp] after release")

        try:
            item.set_dynamic(False)
            print("[pseudo dual grasp] item dynamic -> False")
        except Exception as e:
            print(f"[pseudo dual grasp] item.set_dynamic(False) failed: {e}")

        state["active"] = True
        state["offset"] = item_pos - midpoint

        print(
            "[pseudo dual grasp] ACTIVATED "
            f"offset={state['offset']} "
            f"midpoint={midpoint}"
        )

    offset = state.get("offset", None)
    if offset is None:
        offset = item_pos - midpoint
        state["offset"] = offset

    new_pos = midpoint + offset

    try:
        item.set_position(new_pos.tolist())
    except Exception as e:
        print(f"[pseudo dual grasp] item.set_position failed: {e}")
        return

    print(
        "[pseudo dual grasp] update "
        f"right_closed={right_closed} "
        f"left_closed={left_closed} "
        f"midpoint={midpoint} "
        f"new_item_pos={new_pos}"
    )

def task_step_with_pseudo_update(env, full_action, update_fn=None):
    """
    env._task.step(full_action) の内部で scene.pyrep.step() が呼ばれるたびに
    update_fn() を差し込む。

    これにより、arm motion の途中でも pseudo dual grasp が更新される。
    """
    if update_fn is None:
        return env._task.step(full_action)

    scene = env._task._scene
    pyrep = scene.pyrep

    original_step = pyrep.step

    def wrapped_step(*args, **kwargs):
        out = original_step(*args, **kwargs)
        update_fn()
        return out

    pyrep.step = wrapped_step

    try:
        return env._task.step(full_action)
    finally:
        pyrep.step = original_step

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

def quat_angle_deg(q1, q2):
    """
    q1, q2: [qx, qy, qz, qw]
    quaternion は q と -q が同じ姿勢なので abs(dot) で角度差を見る。
    """
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)

    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    q2 = q2 / (np.linalg.norm(q2) + 1e-12)

    dot = abs(float(np.dot(q1, q2)))
    dot = np.clip(dot, -1.0, 1.0)

    return float(np.degrees(2.0 * np.arccos(dot)))

def create_policy_target_marker(
    arm_name: str,
    action_9d: np.ndarray,
    episode_id: int,
    keyframe_id: int,
    size: float = 0.04,
):
    """
    policy が出力した target 位置を Dummy で可視化する。
    左右の区別は Dummy 名で見る。
    """
    action_9d = np.asarray(action_9d, dtype=np.float32)

    if action_9d.shape != (9,):
        raise ValueError(f"action_9d must be shape (9,), got {action_9d.shape}")

    return create_debug_dummy(
        name=f"ep{episode_id}_{arm_name}_policy_target_keyframe_{keyframe_id}",
        position=action_9d[:3],
        size=size,
    )

def edit_human_keyframe_target(
    env,
    robot,
    teleop,
    arm_name: str,
    initial_gripper_closed: bool,
    episode_id: int,
    keyframe_id: int,
    dt: float = 0.05,
    marker_radius: float = 0.03,
):
    """
    ジョイコンで human arm の keyframe target を編集する。

    操作:
    - stick  : target marker を移動
    - button0: gripper open/close target を切り替え
    - button1: target を確定して motion planner に渡す

    注意:
    - この関数内では robot.grasp() / release_gripper() は呼ばない。
    - button0 は「実行後の gripper state」を指定するだけ。
    - 実際の grasp/release は env._task.step(full_action) 側に任せる。
    """

    arm = get_arm_by_name(robot, arm_name)

    # 現在の EE pose を keyframe target の初期値にする
    start_pose7 = get_pose7_from_arm(arm)
    target_pose7 = start_pose7.copy()

    gripper_closed = bool(initial_gripper_closed)

    marker = create_debug_dummy(
        name=f"ep{episode_id}_{arm_name}_human_target_keyframe_{keyframe_id}",
        position=target_pose7[:3],
        size=marker_radius * 2.0,
    )

    gripper_toggled_count = 0
    edit_steps = 0

    print(
        f"\n[keyframe edit] keyframe={keyframe_id}, arm={arm_name}\n"
        f"  stick  : move target marker\n"
        f"  button0: toggle gripper target\n"
        f"  button1: confirm target and execute planner\n"
        f"  initial xyz={target_pose7[:3]}\n"
        f"  initial gripper_closed={gripper_closed}\n"
    )

    while True:
        teleop_action = teleop.read_action()
        action_4d = np.asarray(teleop_action.as_array(), dtype=np.float32)

        if action_4d.shape != (4,):
            raise ValueError(
                f"teleop action must be shape (4,), got {action_4d.shape}"
            )

        dx, dy, dz = action_4d[:3]

        # marker target を少しずつ動かす
        target_pose7[0] += float(dx)
        target_pose7[1] += float(dy)
        target_pose7[2] += float(dz)

        marker.set_position(target_pose7[:3].tolist())
        step_simulation_for_marker_update(env)

        # button0: gripper target toggle
        # teleop_toggle.py 側で edge trigger 済みなので、そのまま使える
        if bool(teleop_action.gripper_toggle):
            gripper_closed = not gripper_closed
            gripper_toggled_count += 1
            print(
                f"[keyframe edit] {arm_name}_gripper_closed target -> "
                f"{gripper_closed}"
            )

        # button1: confirm
        # teleop_toggle.py 側で edge trigger 済み
        if bool(teleop_action.confirm):
            print("[keyframe edit] target confirmed.")
            break

        edit_steps += 1

        if edit_steps % 20 == 0:
            print(
                f"[keyframe edit] target xyz={target_pose7[:3]} "
                f"gripper_closed={gripper_closed}"
            )

        time.sleep(dt)

    gripper_open = not gripper_closed

    action_9d = make_9d_action_from_pose7(
        target_pose7,
        gripper_open=gripper_open,
        ignore_collisions=0.0,
    )

    accumulated_delta = target_pose7[:3] - start_pose7[:3]

    # 保存用:
    # ここでは「最終的に決めた keyframe target の delta」と
    # 「最終 gripper_open」を入れる
    human_action_4d = np.array(
        [
            accumulated_delta[0],
            accumulated_delta[1],
            accumulated_delta[2],
            float(gripper_open),
        ],
        dtype=np.float32,
    )

    info = {
        "start_pose7": start_pose7.copy(),
        "target_pose7": target_pose7.copy(),
        "accumulated_delta_xyz": accumulated_delta.copy(),
        "gripper_toggled_count": gripper_toggled_count,
        "final_gripper_closed": gripper_closed,
        "final_gripper_open": gripper_open,
        "edit_steps": edit_steps,
    }

    print(
        f"[keyframe edit] final target\n"
        f"  xyz={target_pose7[:3]}\n"
        f"  delta={accumulated_delta}\n"
        f"  gripper_open={gripper_open}\n"
    )

    return action_9d, human_action_4d, gripper_closed, info, marker

def main():
    parser = argparse.ArgumentParser()

    # task / collection
    parser.add_argument("--task", type=str, default="bimanual_lift_long_block")
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--human-arm", type=str, default="right", choices=["right", "left"])
    parser.add_argument("--alternate-human-arm", action="store_true")
    parser.add_argument("--max-keyframes", type=int, default=3, help="Maximum number of executed keyframe actions per episode.")

    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--save-root", type=str, default="data/r2bc_peract")

    # training
    parser.add_argument("--enable-train", action="store_true")
    parser.add_argument("--train-every-episodes", type=int, default=2)
    parser.add_argument("--train-max-episodes", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--train-num-updates", type=int, default=10)
    parser.add_argument("--train-target-arm", type=str, default="both", choices=["both", "right", "left"])
    parser.add_argument(
        "--train-only-disk",
        action="store_true",
        help="Do not launch RLBench or collect new episodes. Only run update_from_disk_buffer on existing saved episodes.",
    )
    parser.add_argument("--grad-accum-steps", type=int, default=1)

    # AnyBimanual / Hydra config
    parser.add_argument("--config-dir", type=str, default=join(PROJECT_ROOT, "conf"))
    parser.add_argument("--config-name", type=str, default="config")
    parser.add_argument("--ckpt-dir", type=str, default="/home/tappei-m/Project/AnyBimanual_checkpoints/PERACT_BC_leader_as_independent")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--timesteps", type=int, default=1)
    parser.add_argument("--idle-epsilon", type=float, default=1e-4, help="Tiny offset added to policy-arm idle target to avoid zero-length planning.")

    # debug
    parser.add_argument("--print-obs-keys", action="store_true")
    parser.add_argument("--no-sleep", action="store_true")
    parser.add_argument("--progress-log-path", type=str, default=None)
    parser.add_argument("--progress-log-every", type=int, default=10)
    parser.add_argument("--save-ckpt-dir", type=str, default=None, help="Directory to save updated PerAct/AnyBimanual weights after training.")
    parser.add_argument(
        "--episode-end-hold-seconds",
        type=float,
        default=0.0,
        help="Seconds to keep the scene at the end of each episode, even if not successful. Set 0 to disable.",
    )
    parser.add_argument(
        "--keyframe-hold-seconds",
        type=float,
        default=0.0,
        help="Seconds to hold the scene after each keyframe execution.",
    )
    parser.add_argument(
        "--grasp-mode",
        type=str,
        default="physics_only",
        choices=["pseudo_dual", "rlbench_attachment", "physics_only"],
    )

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
        print("[collect_peract] scene_bounds:", list(cfg.rlbench.scene_bounds))

        if args.train_only_disk:
            print("[collect_peract] train-only-disk mode")
            print("[collect_peract] data_root:", args.save_root)
            print("[collect_peract] task:", args.task)
            print("[collect_peract] train_num_updates:", args.train_num_updates)
            print("[collect_peract] train_batch_size:", args.train_batch_size)
            print("[collect_peract] train_target_arm:", args.train_target_arm)
            print("[collect_peract] grad_accum_steps:", args.grad_accum_steps)
            print("[collect_peract] effective_batch_size:", args.train_batch_size * args.grad_accum_steps)

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

            print("[collect_peract] creating clip_agent...")
            clip_agent = create_agent(clip_cfg)
            clip_agent.build(training=False, device=device)

            print("[collect_peract] creating train_agent...")
            train_agent = create_agent(train_cfg)
            train_agent.build(training=True, device=device)
            train_agent.load_weights(args.ckpt_dir)

            train_target_arm = None
            if args.train_target_arm != "both":
                train_target_arm = args.train_target_arm

            train_stats = update_from_disk_buffer(
                train_agent=train_agent,
                clip_agent=clip_agent,
                cfg=train_cfg,
                device=device,
                data_root=args.save_root,
                task_name=args.task,
                num_updates=args.train_num_updates,
                batch_size=args.train_batch_size,
                max_episodes=args.train_max_episodes,
                target_arm=train_target_arm,
                shuffle=True,
                debug=False,
                raise_on_error=True,
                grad_accum_steps=args.grad_accum_steps,
                progress_log_path=args.progress_log_path,
                progress_log_every=args.progress_log_every,
                save_ckpt_dir=args.save_ckpt_dir,
            )

            print("[collect_peract] train_stats summary:")

            for k in [
                "num_success",
                "num_skipped",
                "num_failed",
                "target_arm",
                "filtered_target_arm",
                "filtered_num_transitions",
                "num_updates_requested",
                "batch_size",
                "grad_accum_steps",
                "effective_batch_size",
                "num_epochs",
                "loss_first",
                "loss_last",
                "loss_min",
                "loss_max",
                "trans_losses_first",
                "trans_losses_last",
                "rot_losses_first",
                "rot_losses_last",
                "grip_losses_first",
                "grip_losses_last",
                "collision_losses_first",
                "collision_losses_last",
            ]:
                print(f"  {k}: {train_stats.get(k)}")

            print("  dataset_summary:", train_stats.get("dataset_summary"))
            print("  last_gt_voxel:", train_stats.get("gt_voxels", [None])[-1])
            print("  last_pred_voxel:", train_stats.get("pred_voxels", [None])[-1])

            print("[collect_peract] train-only-disk done.")
            return

        print("[collect_peract] launching CustomRLBenchEnv...")
        env = make_custom_env(
            cfg=cfg,
            task_name=args.task,
            variation=args.variation,
            headless=args.headless,
            grasp_mode=args.grasp_mode,
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

            print("[debug] obs_dict language-related keys:")
            for k, v in obs_dict.items():
                if "lang" in k.lower() or "desc" in k.lower() or "goal" in k.lower():
                    try:
                        print(f"  {k}: type={type(v)}, shape={np.asarray(v).shape}, value={v}")
                    except Exception:
                        print(f"  {k}: type={type(v)}, value={v}")

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
            debug_markers = []

            pseudo_dual_grasp_state = {
                "active": False,
                "offset": None,
            }
            last_pseudo_update_fn = None

            for keyframe_id in range(args.max_keyframes):
                step_id = keyframe_id
                step_t0 = time.perf_counter()

                prev_obs_dict = obs_dict

                # -------------------------
                # 1. PerAct full 18D action
                # -------------------------
                t0 = time.perf_counter()

                try:
                    peract_full_action = peract_policy.act_full(obs_dict, keyframe_id)
                    peract_full_action = np.asarray(peract_full_action, dtype=np.float32)

                    if peract_full_action.shape != (18,):
                        raise ValueError(
                            f"peract_full_action must be shape (18,), got {peract_full_action.shape}"
                        )

                except Exception as e:
                    import traceback
                    print(f"[collect_keyframe] peract_policy.act_full failed at keyframe={keyframe_id}: {e}")
                    traceback.print_exc()
                    break

                right_policy_action = peract_full_action[:9].copy()
                left_policy_action = peract_full_action[9:18].copy()

                # ------------------------------------------------------------
                # DEBUG: policy arm の rotation を current tip pose に固定する
                # ------------------------------------------------------------
                right_current_pose7 = get_pose7_from_arm(get_arm_by_name(robot, "right"))
                left_current_pose7 = get_pose7_from_arm(get_arm_by_name(robot, "left"))

                right_policy_rot_err = quat_angle_deg(
                    right_current_pose7[3:7],
                    right_policy_action[3:7],
                )
                left_policy_rot_err = quat_angle_deg(
                    left_current_pose7[3:7],
                    left_policy_action[3:7],
                )

                print(
                    f"\n[policy quat debug BEFORE override] keyframe={keyframe_id}\n"
                    f"  right_current quat={right_current_pose7[3:7]}\n"
                    f"  right_policy  quat={right_policy_action[3:7]} "
                    f"rot_err_deg={right_policy_rot_err:.3f}\n"
                    f"  left_current  quat={left_current_pose7[3:7]}\n"
                    f"  left_policy   quat={left_policy_action[3:7]} "
                    f"rot_err_deg={left_policy_rot_err:.3f}\n"
                )

                # if policy_arm == "left":
                #     left_policy_action[3:7] = left_current_pose7[3:7]
                # elif policy_arm == "right":
                #     right_policy_action[3:7] = right_current_pose7[3:7]

                # # 2. gripper は反転して挙動確認
                # if policy_arm == "left":
                #     left_policy_action[7] = 1.0 - left_policy_action[7]
                # elif policy_arm == "right":
                #     right_policy_action[7] = 1.0 - right_policy_action[7]

                right_policy_rot_err_after = quat_angle_deg(
                    right_current_pose7[3:7],
                    right_policy_action[3:7],
                )
                left_policy_rot_err_after = quat_angle_deg(
                    left_current_pose7[3:7],
                    left_policy_action[3:7],
                )

                print(
                    f"[policy quat debug AFTER override] keyframe={keyframe_id}\n"
                    f"  right_policy quat={right_policy_action[3:7]} "
                    f"rot_err_deg={right_policy_rot_err_after:.3f}\n"
                    f"  left_policy  quat={left_policy_action[3:7]} "
                    f"rot_err_deg={left_policy_rot_err_after:.3f}\n"
                )

                #######################################################################

                right_current_pose7 = get_pose7_from_arm(get_arm_by_name(robot, "right"))
                left_current_pose7 = get_pose7_from_arm(get_arm_by_name(robot, "left"))

                right_policy_rot_err = quat_angle_deg(
                    right_current_pose7[3:7],
                    right_policy_action[3:7],
                )
                left_policy_rot_err = quat_angle_deg(
                    left_current_pose7[3:7],
                    left_policy_action[3:7],
                )

                print(
                    f"\n[policy quat debug] keyframe={keyframe_id}\n"
                    f"  right_current quat={right_current_pose7[3:7]}\n"
                    f"  right_policy  quat={right_policy_action[3:7]} "
                    f"rot_err_deg={right_policy_rot_err:.3f}\n"
                    f"  left_current  quat={left_current_pose7[3:7]}\n"
                    f"  left_policy   quat={left_policy_action[3:7]} "
                    f"rot_err_deg={left_policy_rot_err:.3f}\n"
                )

                policy_action_9d = (
                    right_policy_action if policy_arm == "right"
                    else left_policy_action
                )

                policy_marker = create_policy_target_marker(
                    arm_name=policy_arm,
                    action_9d=policy_action_9d,
                    episode_id=episode_id,
                    keyframe_id=keyframe_id,
                    size=0.04,
                )
                debug_markers.append(policy_marker)

                step_simulation_for_marker_update(env)

                time_stats["policy_act"].append(time.perf_counter() - t0)

                # -------------------------
                # 2. human keyframe target editing
                # -------------------------
                t0 = time.perf_counter()

                if human_arm == "right":
                    human_action_9d, human_action_4d, right_closed, human_edit_info, human_marker = edit_human_keyframe_target(
                        env=env,
                        robot=robot,
                        teleop=teleop,
                        arm_name="right",
                        initial_gripper_closed=right_closed,
                        episode_id=episode_id,
                        keyframe_id=keyframe_id,
                        dt=DT,
                    )
                    debug_markers.append(human_marker)

                    right_action = human_action_9d
                    left_action = left_policy_action.copy()

                else:
                    human_action_9d, human_action_4d, left_closed, human_edit_info, human_marker = edit_human_keyframe_target(
                        env=env,
                        robot=robot,
                        teleop=teleop,
                        arm_name="left",
                        initial_gripper_closed=left_closed,
                        episode_id=episode_id,
                        keyframe_id=keyframe_id,
                        dt=DT,
                    )
                    debug_markers.append(human_marker)

                    right_action = right_policy_action.copy()
                    left_action = human_action_9d

                right_gripper_open = not right_closed
                left_gripper_open = not left_closed

                time_stats["human_keyframe_edit"].append(time.perf_counter() - t0)

                full_action = np.concatenate([right_action, left_action]).astype(np.float32)

                if full_action.shape != (18,):
                    raise ValueError(f"full_action must be shape (18,), got {full_action.shape}")

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
                    # -------------------------
                    # debug marker: policy target
                    # -------------------------
                    # create_debug_sphere(
                    #     name=f"left_policy_target_step_{step_id}",
                    #     position=left_policy_action[:3],
                    #     radius=0.03,
                    #     color=[0.0, 0.0, 1.0],  # blue
                    # )
                    right_current_before = get_pose7_from_arm(get_arm_by_name(robot, "right"))
                    left_current_before = get_pose7_from_arm(get_arm_by_name(robot, "left"))

                    right_exec_rot_err = quat_angle_deg(
                        right_current_before[3:7],
                        right_action[3:7],
                    )
                    left_exec_rot_err = quat_angle_deg(
                        left_current_before[3:7],
                        left_action[3:7],
                    )

                    print(
                        f"\n[before planner] step={step_id} "
                        f"human_arm={human_arm} policy_arm={policy_arm}\n"
                        f"  right_action xyz={right_action[:3]} quat={right_action[3:7]} "
                        f"grip={right_action[7]} collide={right_action[8]} "
                        f"rot_err_deg={right_exec_rot_err:.3f}\n"
                        f"  left_action  xyz={left_action[:3]} quat={left_action[3:7]} "
                        f"grip={left_action[7]} collide={left_action[8]} "
                        f"rot_err_deg={left_exec_rot_err:.3f}\n"
                        f"  right_policy_delta_xyz={right_policy_delta_xyz}\n"
                        f"  left_policy_delta_xyz={left_policy_delta_xyz}\n"
                    )

                    print_grasped_objects(
                        robot,
                        prefix=f"[grasp state BEFORE step={step_id}]",
                    )

                    def _update_pseudo_dual_grasp_current():
                        update_pseudo_dual_grasp(
                            robot=robot,
                            state=pseudo_dual_grasp_state,
                            right_action=right_action,
                            left_action=left_action,
                            item_name="item0",
                            activate_distance=0.75,
                            verbose=False,
                        )
                    last_pseudo_update_fn = None

                    if args.grasp_mode == "pseudo_dual":
                        raw_obs_tp1, reward, env_terminal = task_step_with_pseudo_update(
                            env=env,
                            full_action=full_action,
                            update_fn=_update_pseudo_dual_grasp_current,
                        )
                    elif args.grasp_mode == "physics_only":
                        raw_obs_tp1, reward, env_terminal = env._task.step(full_action)
                    else:
                        raw_obs_tp1, reward, env_terminal = env._task.step(full_action)

                    print_grasped_objects(
                        robot,
                        prefix=f"[grasp state AFTER rlbench step={step_id}]",
                    )

                    if args.grasp_mode == "pseudo_dual":
                        _update_pseudo_dual_grasp_current()

                        print_grasped_objects(
                            robot,
                            prefix=f"[grasp state AFTER pseudo step={step_id}]",
                        )
                    else:
                        print(
                            f"[grasp state AFTER pseudo step={step_id}] "
                            f"skip pseudo update because grasp_mode={args.grasp_mode}"
                        )

                    if args.keyframe_hold_seconds > 0:
                        hold_scene_for_visual_check(
                            env,
                            args.keyframe_hold_seconds,
                            message=f"[visual check] after keyframe {step_id}: hold {args.keyframe_hold_seconds:.1f}s",
                            update_fn=(
                                _update_pseudo_dual_grasp_current
                                if args.grasp_mode == "pseudo_dual"
                                else None
                            ),
                        )

                    # pseudo dual grasp で item0 を動かした後の obs を取る
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

                        "policy_run_this_keyframe": True,
                        "policy_keyframe_id": keyframe_id,

                        "idle_epsilon": args.idle_epsilon,

                        "right_closed": right_closed,
                        "left_closed": left_closed,
                        "right_gripper_open": right_gripper_open,
                        "left_gripper_open": left_gripper_open,

                        "human_edit_info": human_edit_info,
                        "gripper_toggled_count": human_edit_info["gripper_toggled_count"],

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

                        "control_mode": "human_keyframe_marker_confirm",
                    },
                )

                if step_id % 10 == 0 or success or terminate:
                    print(
                        f"[debug] step={step_id} "
                        f"human_arm={human_arm} "
                        f"policy_arm={policy_arm} "
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

            episode_summary = episode_buffer.summary()

            print(f"[collect_peract] Episode saved to {saved_path}")
            print(f"[collect_peract] Summary: {episode_summary}")
            print_time_summary(time_stats)

            if args.episode_end_hold_seconds > 0:
                print(
                    "\n[visual check] EPISODE END. "
                    f"success={episode_summary.get('final_success')} "
                    f"terminate={episode_summary.get('final_terminate')}. "
                    "Holding final scene for visual inspection..."
                )
                hold_scene_for_visual_check(
                    env,
                    args.episode_end_hold_seconds,
                    message=f"[visual check] episode-end hold {args.episode_end_hold_seconds:.1f}s",
                    update_fn=last_pseudo_update_fn
                    if pseudo_dual_grasp_state.get("active", False)
                    else None,
                )

            for i, marker in enumerate(debug_markers):
                safe_remove_debug_object(
                    marker,
                    name=f"ep{episode_id}_debug_marker_{i}",
                )

            debug_markers.clear()

            step_simulation_for_marker_update(env)

            if args.enable_train and ((episode_id + 1) % args.train_every_episodes == 0):
                print(
                    f"[collect_peract] Training trigger at episode {episode_id + 1} "
                    f"(every {args.train_every_episodes} episodes)"
                )

                if args.train_target_arm == "both":
                    train_stats_by_arm = {}

                    for arm in ["left", "right"]:
                        print("=" * 80)
                        print(f"[collect_peract] Training target_arm={arm}")

                        arm_progress_log_path = make_arm_progress_log_path(
                            args.progress_log_path,
                            arm,
                        )

                        # left の途中では checkpoint 保存しない。
                        # right 終了後に save_ckpt_dir を渡すことで、
                        # left/right 更新済みの train_agent 全体を保存する。
                        arm_save_ckpt_dir = args.save_ckpt_dir if arm == "right" else None

                        train_stats_arm = update_from_disk_buffer(
                            train_agent=train_agent,
                            clip_agent=clip_agent,
                            cfg=train_cfg,
                            device=torch.device(args.device),
                            data_root=args.save_root,
                            task_name=args.task,
                            num_updates=args.train_num_updates,
                            batch_size=args.train_batch_size,
                            max_episodes=args.train_max_episodes,
                            target_arm=arm,
                            shuffle=True,
                            debug=False,
                            raise_on_error=True,
                            grad_accum_steps=args.grad_accum_steps,
                            progress_log_path=arm_progress_log_path,
                            progress_log_every=args.progress_log_every,
                            save_ckpt_dir=arm_save_ckpt_dir,
                        )

                        train_stats_by_arm[arm] = train_stats_arm

                        print(f"[collect_peract] train_stats summary arm={arm}:")
                        for k in [
                            "num_success",
                            "num_skipped",
                            "num_failed",
                            "target_arm",
                            "filtered_target_arm",
                            "filtered_num_transitions",
                            "num_updates_requested",
                            "batch_size",
                            "grad_accum_steps",
                            "effective_batch_size",
                            "num_epochs",
                            "loss_first",
                            "loss_last",
                            "loss_min",
                            "loss_max",
                            "trans_losses_first",
                            "trans_losses_last",
                            "rot_losses_first",
                            "rot_losses_last",
                            "grip_losses_first",
                            "grip_losses_last",
                            "collision_losses_first",
                            "collision_losses_last",
                            "dataset_summary",
                        ]:
                            if k in train_stats_arm:
                                print(f"  {k}: {train_stats_arm[k]}")

                        if train_stats_arm.get("gt_voxels"):
                            print(f"  last_gt_voxel: {train_stats_arm['gt_voxels'][-1]}")
                        if train_stats_arm.get("pred_voxels"):
                            print(f"  last_pred_voxel: {train_stats_arm['pred_voxels'][-1]}")

                    train_stats = {
                        "mode": "both_sequential_left_right",
                        "left": train_stats_by_arm.get("left"),
                        "right": train_stats_by_arm.get("right"),
                    }

                else:
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
                        target_arm=args.train_target_arm,
                        shuffle=True,
                        debug=False,
                        raise_on_error=True,
                        grad_accum_steps=args.grad_accum_steps,
                        progress_log_path=args.progress_log_path,
                        progress_log_every=args.progress_log_every,
                        save_ckpt_dir=args.save_ckpt_dir,
                    )

                    print("[collect_peract] train_stats summary:")
                    for k in [
                        "num_success",
                        "num_skipped",
                        "num_failed",
                        "target_arm",
                        "num_updates_requested",
                        "batch_size",
                        "grad_accum_steps",
                        "effective_batch_size",
                        "num_epochs",
                        "loss_first",
                        "loss_last",
                        "loss_min",
                        "loss_max",
                        "trans_losses_first",
                        "trans_losses_last",
                        "rot_losses_first",
                        "rot_losses_last",
                        "grip_losses_first",
                        "grip_losses_last",
                        "collision_losses_first",
                        "collision_losses_last",
                        "dataset_summary",
                    ]:
                        if k in train_stats:
                            print(f"  {k}: {train_stats[k]}")

                    if train_stats.get("gt_voxels"):
                        print(f"  last_gt_voxel: {train_stats['gt_voxels'][-1]}")
                    if train_stats.get("pred_voxels"):
                        print(f"  last_pred_voxel: {train_stats['pred_voxels'][-1]}")

                if args.save_ckpt_dir is not None:
                    print("[collect_peract] reloading updated weights into act policy...")
                    print("[collect_peract] reload ckpt:", args.save_ckpt_dir)

                    old_random_init = os.environ.pop("R2BC_RANDOM_INIT_ANYBIMANUAL", None)
                    try:
                        peract_policy.agent.load_weights(args.save_ckpt_dir)
                    finally:
                        if old_random_init is not None:
                            os.environ["R2BC_RANDOM_INIT_ANYBIMANUAL"] = old_random_init

                    print("[collect_peract] reload done.")
                else:
                    print("[collect_peract] save_ckpt_dir is None; skip reload into act policy.")
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