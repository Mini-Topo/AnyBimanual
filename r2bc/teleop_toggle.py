# Standard Library
import os
from dataclasses import dataclass

# Third Party
import numpy as np
import pygame


os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")


@dataclass
class TeleopAction:
    dx: float
    dy: float
    dz: float
    gripper_toggle: bool
    confirm: bool

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.dx, self.dy, self.dz, float(self.gripper_toggle)],
            dtype=np.float32,
        )


class JoystickTeleop:
    """
    Joystick input wrapper.

    This class does NOT depend on RLBench or PyRep.
    It only reads joystick input and returns:
        [dx, dy, dz, gripper_toggle]

    gripper_toggle:
        True only on the frame where the button is pressed.
    """

    def __init__(
        self,
        joystick_id: int = 0,
        z_step: float = 0.01,
        deadzone: float = 0.15,
        axis_z: int = 1,
        gripper_button: int = 0,
        confirm_button: int = 1,
    ):
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() <= joystick_id:
            raise RuntimeError(
                f"No joystick found at id={joystick_id}. "
                f"pygame detected {pygame.joystick.get_count()} joystick(s)."
            )

        self.joy = pygame.joystick.Joystick(joystick_id)
        self.joy.init()

        self.z_step = z_step
        self.deadzone = deadzone

        self.axis_z = axis_z
        self.gripper_button = gripper_button
        self.confirm_button = confirm_button

        self._prev_gripper_button_pressed = False
        self._prev_confirm_button_pressed = False

        print("[JoystickTeleop] joystick:", self.joy.get_name())
        print("[JoystickTeleop] axes:", self.joy.get_numaxes())
        print("[JoystickTeleop] buttons:", self.joy.get_numbuttons())
        print("[JoystickTeleop] axis_z:", self.axis_z)
        print("[JoystickTeleop] gripper_button:", self.gripper_button)
        print("[JoystickTeleop] confirm_button:", self.confirm_button)
    def close(self):
        pygame.joystick.quit()
        pygame.quit()

    def _axis(self, axis_id: int) -> float:
        if axis_id < 0 or axis_id >= self.joy.get_numaxes():
            return 0.0

        value = float(self.joy.get_axis(axis_id))
        if abs(value) < self.deadzone:
            return 0.0
        return value

    def _button(self, button_id: int) -> bool:
        if button_id < 0 or button_id >= self.joy.get_numbuttons():
            return False
        return bool(self.joy.get_button(button_id))

    def read_action(self) -> TeleopAction:
        pygame.event.pump()

        az = self._axis(self.axis_z)

        dx = 0.0
        dy = 0.0
        dz = -self.z_step * az

        gripper_pressed = self._button(self.gripper_button)
        gripper_toggle = gripper_pressed and not self._prev_gripper_button_pressed
        self._prev_gripper_button_pressed = gripper_pressed

        confirm_pressed = self._button(self.confirm_button)
        confirm = confirm_pressed and not self._prev_confirm_button_pressed
        self._prev_confirm_button_pressed = confirm_pressed

        return TeleopAction(
            dx=dx,
            dy=dy,
            dz=dz,
            gripper_toggle=gripper_toggle,
            confirm=confirm,
        )


def main():
    teleop = JoystickTeleop()

    print("Teleop test started.")
    print("Move stick 1 up/down to see dz.")
    print("Press button 0 to see gripper_toggle=True.")
    print("Press button 1 to see confirm=True.")
    print("Ctrl+C to quit.")

    try:
        while True:
            action = teleop.read_action()
            arr = action.as_array()

            if abs(action.dz) > 0.0 or action.gripper_toggle or action.confirm:
                print(arr, "confirm=", action.confirm)

            pygame.time.wait(50)

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        teleop.close()


if __name__ == "__main__":
    main()