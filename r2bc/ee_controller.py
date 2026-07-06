# Standard Library
from typing import Optional, Sequence, Tuple

# Third Party
import numpy as np


def apply_ee_delta(
    arm,
    delta_pos: Sequence[float],
    max_delta: float = 0.03,
    verbose: bool = False,
) -> bool:
    """
    Move an end-effector by a small Cartesian delta using IK.

    Parameters
    ----------
    arm:
        PyRep arm object. e.g. PandaRight(), PandaLeft(),
        or task_env._robot.right_arm / left_arm.
    delta_pos:
        [dx, dy, dz] in meters.
    max_delta:
        Safety clamp for each delta component.
    verbose:
        If True, print IK errors.

    Returns
    -------
    bool:
        True if IK succeeded and joint targets were set.
        False otherwise.
    """
    delta_pos = np.asarray(delta_pos, dtype=np.float64)

    if delta_pos.shape != (3,):
        raise ValueError(f"delta_pos must have shape (3,), got {delta_pos.shape}")

    # 入力が大きすぎるとIKが壊れやすいので一応クリップ
    delta_pos = np.clip(delta_pos, -max_delta, max_delta)

    if np.allclose(delta_pos, 0.0):
        return True

    tip = arm.get_tip()
    pose = tip.get_pose()  # [x, y, z, qx, qy, qz, qw]

    target_pos = np.asarray(pose[:3], dtype=np.float64) + delta_pos
    target_quat = pose[3:]

    try:
        q = arm.solve_ik_via_jacobian(
            position=target_pos.tolist(),
            quaternion=target_quat,
        )
        arm.set_joint_target_positions(q)
        return True

    except Exception as e:
        if verbose:
            print(f"[apply_ee_delta] IK failed: {type(e).__name__}: {e}")
        return False


def set_gripper_target(
    gripper,
    target_open_amount: float,
    velocity: float = 0.04,
) -> bool:
    """
    Non-blocking gripper control.

    Parameters
    ----------
    gripper:
        PyRep gripper object.
    target_open_amount:
        1.0 = open, 0.0 = close.
    velocity:
        Gripper actuation velocity.

    Returns
    -------
    bool:
        True if gripper reached target, False otherwise.
    """
    target_open_amount = float(np.clip(target_open_amount, 0.0, 1.0))
    return bool(gripper.actuate(target_open_amount, velocity))


def action_to_gripper_target(is_closed: bool) -> float:
    """
    Convert closed/open state to gripper target.

    closed=True  -> 0.0
    closed=False -> 1.0
    """
    return 0.0 if is_closed else 1.0


def apply_single_arm_action(
    arm,
    gripper,
    action: Sequence[float],
    gripper_closed: bool,
    max_delta: float = 0.03,
    gripper_velocity: float = 0.04,
    verbose: bool = False,
) -> Tuple[bool, bool]:
    """
    Apply one single-arm R2BC action.

    action:
        [dx, dy, dz, gripper_toggle]

    gripper_closed:
        Current logical gripper state.

    Returns
    -------
    ik_ok:
        Whether IK succeeded.
    gripper_closed:
        Updated logical gripper state.
    """
    action = np.asarray(action, dtype=np.float64)

    if action.shape != (4,):
        raise ValueError(f"action must have shape (4,), got {action.shape}")

    delta_pos = action[:3]
    gripper_toggle = bool(action[3] > 0.5)

    if gripper_toggle:
        gripper_closed = not gripper_closed

    ik_ok = apply_ee_delta(
        arm=arm,
        delta_pos=delta_pos,
        max_delta=max_delta,
        verbose=verbose,
    )

    target = action_to_gripper_target(gripper_closed)
    set_gripper_target(
        gripper=gripper,
        target_open_amount=target,
        velocity=gripper_velocity,
    )

    return ik_ok, gripper_closed