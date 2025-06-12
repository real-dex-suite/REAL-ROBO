import rospy
from std_msgs.msg import Float64MultiArray

from holodex.utils.files import *
from holodex.utils.vec_ops import coord_in_bound, best_fit_transform, normalize_vector
from holodex.constants import *
from copy import deepcopy as copy
from scipy.spatial.transform import Rotation as R

try:
    from .robot import RobotController
except:
    from robot import RobotController

from scipy.interpolate import CubicSpline
from scipy.special import comb
from scipy.spatial.transform import Slerp
from termcolor import cprint

from pynput import keyboard
import time

# translation_step = 3.0
translation_step = 0.003
rotation_step = 0.05


class KBArmTeleop(object):
    def __init__(self, simulator=None, gripper=None, arm_type="franka", gripper_init_state="open"):
        self.arm_type = arm_type
        rospy.init_node("keyboard_arm_teleop")

        rospy.Subscriber(
            JAKA_EE_POSE_TOPIC,
            Float64MultiArray,
            self._callback_arm_ee_pose,
            queue_size=1,
        )

        # create a publisher to send the desired arm pose
        self.arm_pose_publisher = rospy.Publisher(
            KEYBOARD_EE_TOPIC, Float64MultiArray, queue_size=1
        )
        self.hand_dof_pos_publisher = rospy.Publisher(
            KEYBOARD_HAND_TOPIC, Float64MultiArray, queue_size=1
        )

        # Initializing the robot controller
        self.robot = RobotController(teleop=True, simulator=simulator, gripper=gripper, arm_type=arm_type, gripper_init_state=gripper_init_state)

        # Initializing the arm pose
        self.arm_ee_pose = self.robot.arm.get_tcp_position(euler=False)
        self.desired_arm_pose = list(self.arm_ee_pose)
        self.tmp_desired_arm_pose = copy(self.desired_arm_pose)

        # Initializing the hand pose
        self.hand_dof_pos = (
            self.robot.get_hand_position() if HAND_TYPE is not None else None
        )
        self.desired_hand_dof_pos = LEAP_HOME_POSITION
        self.tmp_desired_hand_dof_pos = copy(self.desired_hand_dof_pos)

        self.prev_hand_joint_angles = (
            self.robot.get_hand_position() if HAND_TYPE is not None else None
        )
        # print(f"Initial hand joint angles: {self.prev_hand_joint_angles}")

        self.predefined_hand_dof_step = PREDEFINED_HAND_DOF_STEP

    def _callback_arm_ee_pose(self, msg):
        self.arm_ee_pose = msg.data
        # self.desired_arm_pose = copy(self.arm_ee_pose)
        # self.desired_arm_pose = list(self.arm_ee_pose)

    def motion(self):

        desired_cmd = []

        desired_cmd = np.concatenate(
            [desired_cmd, self.tmp_desired_arm_pose, self.tmp_desired_hand_dof_pos]
        )
        return desired_cmd

    def _on_press(self, key):

        print(f"Key pressed: {key}")
        try:
            if self.send_keyboard_cmd:
                return

            print(f"Key pressed: {key.char}")
            if key.char in [
                "w",
                "s",
                "a",
                "d",
                "q",
                "e",
                "r",
                "f",
                "t",
                "g",
                "y",
                "h",
                "p",
            ]:
                self.send_keyboard_cmd = True

            if key.char == "w":
                self.tmp_desired_arm_pose[1] = (
                    self.desired_arm_pose[1] - translation_step
                )
            elif key.char == "s":
                self.tmp_desired_arm_pose[1] = (
                    self.desired_arm_pose[1] + translation_step
                )
            elif key.char == "a":
                self.tmp_desired_arm_pose[0] = (
                    self.desired_arm_pose[0] + translation_step
                )
            elif key.char == "d":
                self.tmp_desired_arm_pose[0] = (
                    self.desired_arm_pose[0] - translation_step
                )
            elif key.char == "q":
                self.tmp_desired_arm_pose[2] = (
                    self.desired_arm_pose[2] + translation_step
                )
            elif key.char == "e":
                self.tmp_desired_arm_pose[2] = (
                    self.desired_arm_pose[2] - translation_step
                )
            elif key.char == "r":
                self.tmp_desired_arm_pose[3] = self.desired_arm_pose[3] + rotation_step
            elif key.char == "f":
                self.tmp_desired_arm_pose[3] = self.desired_arm_pose[3] - rotation_step
            elif key.char == "t":
                self.tmp_desired_arm_pose[4] = self.desired_arm_pose[4] + rotation_step
            elif key.char == "g":
                self.tmp_desired_arm_pose[4] = self.desired_arm_pose[4] - rotation_step
            elif key.char == "y":
                self.tmp_desired_arm_pose[5] = self.desired_arm_pose[5] + rotation_step
            elif key.char == "h":
                self.tmp_desired_arm_pose[5] = self.desired_arm_pose[5] - rotation_step
            # move hand by predefined step
            elif key.char == "p":
                if self.predefined_hand_dof_step is not None:
                    self.tmp_desired_hand_dof_pos = (
                        self.desired_hand_dof_pos + self.predefined_hand_dof_step
                    )
                else:
                    raise ValueError("Predefined hand DOF step is not set.")
            else:
                print(f"Unknown key pressed: {key.char}")
        except AttributeError:
            print(f"Special key {key} pressed")

    def move(self):
        print(
            "\n******************************************************************************"
        )
        cprint("[   ok   ]     Controller initiated. ", "green", attrs=["bold"])
        print(
            "******************************************************************************\n"
        )
        print("Start controlling the robot hand using the Keyboard.\n")

        keyboard_listener = keyboard.Listener(on_press=self._on_press)

        keyboard_listener.start()
        while True:
            # reset the desired
            self.send_keyboard_cmd = False
            self.tmp_desired_arm_pose = copy(self.desired_arm_pose)
            self.tmp_desired_hand_dof_pos = copy(self.desired_hand_dof_pos)

            print(self.tmp_desired_arm_pose)

            # from ipdb import set_trace; set_trace()

            # get the desired command
            desired_cmd = self.motion()

            # from ipdb import set_trace; set_trace()

            self.robot.move(desired_cmd)

            # update the desired
            self.desired_arm_pose = copy(self.tmp_desired_arm_pose)
            self.desired_hand_dof_pos = copy(self.tmp_desired_hand_dof_pos)

            # publish the desired command
            if self.send_keyboard_cmd:
                self.arm_pose_publisher.publish(
                    Float64MultiArray(data=self.desired_arm_pose)
                )
                self.hand_dof_pos_publisher.publish(
                    Float64MultiArray(data=self.desired_hand_dof_pos)
                )


if __name__ == "__main__":
    teleop = KBArmTeleop()
    teleop.move()
