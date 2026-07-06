# Standard Library
from typing import Any, Dict, Optional

# Third Party
import numpy as np


class BasePolicy:
    """
    Policy interface for one arm.

    action format:
        np.array([dx, dy, dz, gripper_toggle], dtype=np.float32)

    dx, dy, dz:
        EE delta position [m]

    gripper_toggle:
        1.0 only when toggling gripper open/close
        0.0 otherwise
    """

    def reset(self) -> None:
        pass

    def act(self, obs: Any, arm_name: str) -> np.ndarray:
        raise NotImplementedError


class IdlePolicy(BasePolicy):
    """
    Do nothing.

    This keeps the arm at the current target position.
    Gripper state is not toggled.
    """

    def act(self, obs: Any, arm_name: str) -> np.ndarray:
        return np.zeros(4, dtype=np.float32)


class ConstantDeltaPolicy(BasePolicy):
    """
    Always output the same EE delta.

    Useful for quick tests.
    Example:
        slowly move upward:
            ConstantDeltaPolicy(dx=0, dy=0, dz=0.002)
    """

    def __init__(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        gripper_toggle: float = 0.0,
    ):
        self.action = np.array(
            [dx, dy, dz, gripper_toggle],
            dtype=np.float32,
        )

    def act(self, obs: Any, arm_name: str) -> np.ndarray:
        return self.action.copy()


class ScriptedLiftPolicy(BasePolicy):
    """
    Very simple scripted policy for debugging.

    It does not yet use object position.
    It only executes a fixed phase sequence:

        1. wait
        2. close gripper once
        3. lift upward for N steps
        4. idle

    This is intentionally simple.
    Later, this can be replaced by a task-aware scripted policy.
    """

    def __init__(
        self,
        wait_steps: int = 30,
        lift_steps: int = 80,
        lift_dz: float = 0.003,
    ):
        self.wait_steps = wait_steps
        self.lift_steps = lift_steps
        self.lift_dz = lift_dz
        self.t = 0
        self.has_toggled_gripper = False

    def reset(self) -> None:
        self.t = 0
        self.has_toggled_gripper = False

    def act(self, obs: Any, arm_name: str) -> np.ndarray:
        action = np.zeros(4, dtype=np.float32)

        # Phase 1: wait
        if self.t < self.wait_steps:
            pass

        # Phase 2: close gripper once
        elif not self.has_toggled_gripper:
            action[3] = 1.0
            self.has_toggled_gripper = True

        # Phase 3: lift upward
        elif self.t < self.wait_steps + self.lift_steps:
            action[2] = self.lift_dz

        # Phase 4: idle
        else:
            pass

        self.t += 1
        return action


class LearnedPolicy(BasePolicy):
    """
    Placeholder for future learned policy.

    Later:
        obs -> model -> [dx, dy, dz, gripper_toggle]
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.model_path = model_path
        self.device = device

        if model_path is not None:
            raise NotImplementedError(
                "LearnedPolicy is not implemented yet. "
                "Use IdlePolicy or ScriptedLiftPolicy for now."
            )

    def act(self, obs: Any, arm_name: str) -> np.ndarray:
        raise NotImplementedError("LearnedPolicy.act() is not implemented yet.")


def make_policy(policy_name: str, kwargs: Optional[Dict[str, Any]] = None) -> BasePolicy:
    """
    Factory function.

    policy_name:
        idle
        constant_up
        constant_down
        scripted_lift
    """
    kwargs = kwargs or {}

    if policy_name == "idle":
        return IdlePolicy()

    if policy_name == "constant_up":
        return ConstantDeltaPolicy(dz=kwargs.get("dz", 0.0005))

    if policy_name == "constant_down":
        return ConstantDeltaPolicy(dz=kwargs.get("dz", -0.01))

    if policy_name == "scripted_lift":
        return ScriptedLiftPolicy(
            wait_steps=kwargs.get("wait_steps", 30),
            lift_steps=kwargs.get("lift_steps", 80),
            lift_dz=kwargs.get("lift_dz", -0.003),
        )

    raise ValueError(f"Unknown policy_name: {policy_name}")