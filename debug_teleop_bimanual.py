import os
import sys
import time
from os.path import join, dirname, abspath

# pygameの余計なログを消す
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")

import pygame
import numpy as np

# local RLBench / PyRep を使う
ROOT = dirname(abspath(__file__))
sys.path.insert(0, join(ROOT, "third_party", "RLBench"))
sys.path.insert(0, join(ROOT, "third_party", "PyRep"))

from pyrep import PyRep
from pyrep.errors import ConfigurationPathError, IKError

from rlbench.backend.const import BIMANUAL_TTT_FILE

from pyrep.robots.arms.dual_panda import PandaRight, PandaLeft
from pyrep.robots.end_effectors.dual_panda_gripper import (
    PandaGripperRight,
    PandaGripperLeft,
)


DT = 0.05
POS_STEP = 0.01      # 1 stepあたりのEE移動量[m]
Z_STEP = 0.01
ROT_STEP = 0.02     # 今回は未使用でもOK
DEADZONE = 0.15


def dz(x, deadzone=DEADZONE):
    if abs(x) < deadzone:
        return 0.0
    return float(x)


def move_ee_delta(pr, arm, dx, dy, dz_):
    """現在のEE poseから少しだけ動かす。IKが解けなければ何もしない。"""
    if dx == 0.0 and dy == 0.0 and dz_ == 0.0:
        return

    tip = arm.get_tip()
    pose = tip.get_pose()  # [x, y, z, qx, qy, qz, qw]
    target_pos = np.array(pose[:3], dtype=np.float64)
    target_quat = pose[3:]

    target_pos += np.array([dx, dy, dz_], dtype=np.float64)

    try:
        q = arm.solve_ik_via_jacobian(
            position=target_pos.tolist(),
            quaternion=target_quat,
        )
        arm.set_joint_target_positions(q)
    except Exception as e:
        print(f"[IK failed] {type(e).__name__}: {e}")


def actuate_gripper_until(pr, gripper, target_open_amount, velocity=0.04, max_steps=50):
    """
    target_open_amount:
      1.0 = open
      0.0 = close
    """
    for _ in range(max_steps):
        done = gripper.actuate(target_open_amount, velocity)
        pr.step()
        if done:
            break


def main():
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("No joystick found.")
        print("まずはJoy-Con/ゲームパッドがpygameから見えるか確認してください。")
        return

    joy = pygame.joystick.Joystick(0)
    joy.init()

    print("Joystick:", joy.get_name())
    print("axes:", joy.get_numaxes(), "buttons:", joy.get_numbuttons())

    pr = PyRep()

    ttt_file = join(
        ROOT,
        "third_party",
        "RLBench",
        "rlbench",
        BIMANUAL_TTT_FILE,
    )

    print("Launching:", ttt_file)
    pr.launch(ttt_file, headless=False, responsive_ui=True)
    pr.start()

    right_arm = PandaRight()
    left_arm = PandaLeft()
    right_gripper = PandaGripperRight()
    left_gripper = PandaGripperLeft()

    # 最初はグリッパを開いておく
    actuate_gripper_until(pr, right_gripper, 1.0)
    actuate_gripper_until(pr, left_gripper, 1.0)

    right_closed = False
    left_closed = False
    prev_buttons = {}

    print("Teleop started.")
    print("Left stick : right arm x/y")
    print("Right stick: left arm x/y")
    print("Buttons 0/1: toggle right/left gripper")
    print("Ctrl+C to quit.")

    try:
        while True:
            pygame.event.pump()

            # 環境によってaxis番号は違う可能性あり
            # まずは一般的なゲームパッド想定
            ax0 = dz(joy.get_axis(0)) if joy.get_numaxes() > 0 else 0.0
            ax1 = dz(joy.get_axis(1)) if joy.get_numaxes() > 1 else 0.0
            ax2 = dz(joy.get_axis(2)) if joy.get_numaxes() > 2 else 0.0
            ax3 = dz(joy.get_axis(3)) if joy.get_numaxes() > 3 else 0.0

            # 右腕: 左スティック
            r_dx = POS_STEP * ax0
            r_dy = -POS_STEP * ax1
            r_dz = 0.0

            # 左腕: 右スティック
            l_dx = POS_STEP * ax2
            l_dy = -POS_STEP * ax3
            l_dz = 0.0

            move_ee_delta(pr, right_arm, r_dx, r_dy, r_dz)
            move_ee_delta(pr, left_arm, l_dx, l_dy, l_dz)

            # button 0: right gripper toggle
            b0 = joy.get_button(0) if joy.get_numbuttons() > 0 else 0
            if b0 and not prev_buttons.get(0, False):
                right_closed = not right_closed
                target = 0.0 if right_closed else 1.0
                print("right gripper:", "close" if right_closed else "open")
                actuate_gripper_until(pr, right_gripper, target)

            # button 1: left gripper toggle
            b1 = joy.get_button(1) if joy.get_numbuttons() > 1 else 0
            if b1 and not prev_buttons.get(1, False):
                left_closed = not left_closed
                target = 0.0 if left_closed else 1.0
                print("left gripper:", "close" if left_closed else "open")
                actuate_gripper_until(pr, left_gripper, target)

            prev_buttons[0] = bool(b0)
            prev_buttons[1] = bool(b1)

            pr.step()
            time.sleep(DT)

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        pr.stop()
        pr.shutdown()
        pygame.quit()


if __name__ == "__main__":
    main()