# r2bc/controllers/async_bimanual_planner.py

import numpy as np

from rlbench.action_modes.arm_action_modes import BimanualEndEffectorPoseViaPlanning
from rlbench.backend.exceptions import InvalidActionError
from pyrep.errors import ConfigurationPathError


def get_arm_by_name(robot, arm_name):
    if arm_name == "right":
        return robot.right_arm
    if arm_name == "left":
        return robot.left_arm
    raise ValueError(f"Unknown arm_name: {arm_name}")


def get_gripper_by_name(robot, arm_name):
    if arm_name == "right":
        return robot.right_gripper
    if arm_name == "left":
        return robot.left_gripper
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


class AsyncBimanualPlanner:
    def __init__(self, scene):
        self.scene = scene
        self.robot = scene.robot
        self.ee_planner = BimanualEndEffectorPoseViaPlanning()

        self.paths = {
            "right": None,
            "left": None,
        }
        self.done = {
            "right": True,
            "left": True,
        }

    def make_path_from_action9d(self, arm_name, action_9d):
        action_9d = np.asarray(action_9d, dtype=np.float32)

        if action_9d.shape != (9,):
            raise ValueError(
                f"action_9d must be shape (9,), got {action_9d.shape}"
            )

        pose7 = action_9d[:7]
        ignore_collisions = bool(action_9d[8])

        return self.make_path_from_pose7(
            arm_name=arm_name,
            pose7=pose7,
            ignore_collisions=ignore_collisions,
        )

    def make_path_from_pose7(self, arm_name, pose7, ignore_collisions=True):
        arm = get_arm_by_name(self.robot, arm_name)
        gripper = get_gripper_by_name(self.robot, arm_name)

        pose7 = np.asarray(pose7, dtype=np.float32)

        try:
            path = self.ee_planner.get_path(
                self.scene,
                pose7,
                bool(ignore_collisions),
                arm,
                gripper,
            )
        except (ConfigurationPathError, InvalidActionError) as e:
            # print(f"[AsyncBimanualPlanner] {arm_name} path failed: {repr(e)}")
            self.paths[arm_name] = None
            self.done[arm_name] = True
            return False

        self.paths[arm_name] = path
        self.done[arm_name] = False
        # print(f"[AsyncBimanualPlanner] {arm_name} path created")
        return True

    def step_arm(self, arm_name):
        path = self.paths[arm_name]

        if path is None or self.done[arm_name]:
            return True

        self.done[arm_name] = path.step()
        return self.done[arm_name]

    def step_scene(self):
        self.scene.step()

    def step(self):
        right_done = self.step_arm("right")
        left_done = self.step_arm("left")
        self.step_scene()

        success, terminate = self.scene.task.success()
        return {
            "right_done": right_done,
            "left_done": left_done,
            "success": success,
            "terminate": terminate,
        }

    def step_selective(
        self,
        policy_arm: str,
        human_arm: str,
        policy_step: bool = True,
        human_step: bool = True,
    ):
        """
        policy/human のどちらの path.step() を呼ぶかを選び、
        scene は毎tick進める。

        human_arm:
          基本的に毎tick進める。

        policy_arm:
          policy_step=True のtickだけ進める。
          これにより policy arm を間引いて遅くできる。
        """
        if policy_arm not in ["right", "left"]:
            raise ValueError(f"Unknown policy_arm: {policy_arm}")
        if human_arm not in ["right", "left"]:
            raise ValueError(f"Unknown human_arm: {human_arm}")

        if policy_step:
            self.step_arm(policy_arm)

        if human_step:
            self.step_arm(human_arm)

        self.step_scene()

        success, terminate = self.scene.task.success()
        return {
            "right_done": bool(self.done["right"]),
            "left_done": bool(self.done["left"]),
            "success": bool(success),
            "terminate": bool(terminate),
        }

    def is_done(self, arm_name):
        return bool(self.done[arm_name])

    def clear_path(self, arm_name):
        self.paths[arm_name] = None
        self.done[arm_name] = True