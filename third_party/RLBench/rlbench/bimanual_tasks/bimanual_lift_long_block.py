from collections import defaultdict
from typing import List, Tuple

import numpy as np
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.conditions import NothingGrasped
from rlbench.backend.task import BimanualTask
from rlbench.backend.spawn_boundary import SpawnBoundary
from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object
from rlbench.backend.conditions import Condition


class LiftedCondition(Condition):

    def __init__(self, item: Shape, min_height: float):
        self.item = item
        self.min_height = min_height

    def condition_met(self):
        pos = self.item.get_position()
        return pos[2] >= self.min_height, False

class BimanualLiftLongBlock(BimanualTask):

    def init_task(self) -> None:

        self.task_root = Dummy('bimanual_lift_long_block')
        self.task_root_initial_pose = np.array(self.task_root.get_pose())

        self.item = Shape('item0')

        self.register_graspable_objects([self.item])

        self.waypoint_mapping = defaultdict(lambda: 'left')
        self.waypoint_mapping.update({'waypoint0': 'right', 'waypoint1': 'right'})

        self.randomize_xy_range = 0.03


    def init_episode(self, index: int) -> List[str]:

        self._variation_index = index

        # Randomize the whole task root so item0 and waypoints move together.
        dx = np.random.uniform(-self.randomize_xy_range, self.randomize_xy_range)
        dy = np.random.uniform(-self.randomize_xy_range, self.randomize_xy_range)

        root_pose = self.task_root_initial_pose.copy()
        root_pose[0] += dx
        root_pose[1] += dy
        self.task_root.set_pose(root_pose.tolist())

        print(
            f"[BimanualLiftLongBlock] random root offset: "
            f"dx={dx:.4f}, dy={dy:.4f}, "
            f"root_pos={self.task_root.get_position()}, "
            f"item_pos={self.item.get_position()}"
        )

        print(
            "[BimanualLiftLongBlock] waypoint positions:",
            "w0=", Dummy('waypoint0').get_position(),
            "w1=", Dummy('waypoint1').get_position(),
            "w2=", Dummy('waypoint2').get_position(),
            "w3=", Dummy('waypoint3').get_position(),
        )

        right_success_sensor = ProximitySensor('Panda_rightArm_gripper_attachProxSensor')
        left_success_sensor = ProximitySensor('Panda_leftArm_gripper_attachProxSensor')

        self.register_success_conditions(
            [DetectedCondition(self.item, right_success_sensor),
            DetectedCondition(self.item, left_success_sensor),
            LiftedCondition(self.item, 0.95)]
        )

        return [
            'lift the long block with both arms',
            'use both arms to lift the long block',
            'pick up the long block using both grippers',
            'raise the long block from the table',
        ]

    def variation_count(self) -> int:
        return 1

    # def boundary_root(self) -> Object:
    #     return Shape('item0')

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        # return [0, 0, - np.pi / 8], [0, 0, np.pi / 8]
        return [0, 0, 0], [0, 0, 0]

    def is_static_workspace(self) -> bool:
        return True
