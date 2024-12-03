import threading
import pyspacemouse
import numpy as np
from typing import Tuple
import time
import multiprocessing
import rospy
from scipy.spatial.transform import Rotation as R
from holodex.constants import *
from copy import deepcopy as copy
from termcolor import cprint
from std_msgs.msg import Float64MultiArray
from holodex.utils import converter

try:
    from .robot import RobotController
except:
    from robot import RobotController

import os
import sys

## Potential Bug: The SpaceMouseExpert class is not working as expected. The action and buttons are not being printed.
## sudo chmod 666 /dev/hidraw* to fix the issue
# using os to run the cmd: sudo chmod 666 /dev/hidraw*

os.system("cd /usr/lib64")
os.system("sudo chmod 666 /dev/hidraw*")

move_linear_velocity = 0.002
move_angular_velocity = 0.002


class SpaceMouseTeleop(object):
    def __init__(self):
        pyspacemouse.open()
        rospy.init_node("space_mouse_teleop")

        # create a subscriber to get the current arm pose
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
        self.robot = RobotController(teleop=True)

        # Initializing the arm pose
        self.arm_ee_pose = self.robot.arm.get_tcp_position(euler=True, degree=False)
        self.desired_arm_pose = list(self.arm_ee_pose)
        self.tmp_desired_arm_pose = copy(self.desired_arm_pose)

        # Initializing the hand pose
        self.hand_dof_pos = (
            self.robot.get_hand_position() if HAND_TYPE is not None else None
        )
        self.desired_hand_dof_pos = LEAP_HOME_POSITION
        self.tmp_desired_hand_dof_pos = copy(self.desired_hand_dof_pos)

        # Initializing the hand joint angles
        self.prev_hand_joint_angles = (
            self.robot.get_hand_position() if HAND_TYPE is not None else None
        )
        self.predefined_hand_dof_step = PREDEFINED_HAND_DOF_STEP

        # Initializing the space mouse arguments
        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        self.latest_data["action"] = [0.0] * 6  # Using lists for compatibility
        self.latest_data["buttons"] = [0, 0, 0, 0]

        # Start a process to continuously read the SpaceMouse state
        self.process = multiprocessing.Process(target=self._read_spacemouse)
        self.process.daemon = True
        self.process.start()

    def _read_spacemouse(self):
        while True:
            state = pyspacemouse.read()
            action = [0.0] * 6
            buttons = [0, 0, 0, 0]

            self.latest_data["action"] = np.array(
                [-state.y, state.x, state.z, -state.roll, -state.pitch, -state.yaw]
            )  # spacemouse axis matched with robot base frame
            self.latest_data["buttons"] = state.buttons

    def get_action(self) -> Tuple[np.ndarray, list]:
        """Returns the latest action and button state of the SpaceMouse."""
        action = self.latest_data["action"]
        buttons = self.latest_data["buttons"]
        return np.array(action), buttons

    def close(self):
        # pyspacemouse.close()
        self.process.terminate()

    def _callback_arm_ee_pose(self, msg):
        self.arm_ee_pose = msg.data
        # self.desired_arm_pose = copy(self.arm_ee_pose)
        # self.desired_arm_pose = list(self.arm_ee_pose)

    def motion(self):
        desired_cmd = []
        action, buttons = self.get_action()
        self.tmp_desired_arm_pose[:3] = (
            self.desired_arm_pose[:3] + action[:3] * move_linear_velocity
        )
        self.tmp_desired_arm_pose[3:6] = (
            self.desired_arm_pose[3:] + action[3:] * move_angular_velocity
        )

        # euler to quat
        tmp_desired_arm_euler = self.tmp_desired_arm_pose[3:6]
        tmp_desired_arm_quat = self.robot.arm.eulerZYX2quat(tmp_desired_arm_euler)

        desired_cmd = np.concatenate(
            [
                desired_cmd,
                self.tmp_desired_arm_pose[:3],
                tmp_desired_arm_quat,
                self.tmp_desired_hand_dof_pos,
            ]
        )
        return desired_cmd

    def move(self):
        print(
            "\n******************************************************************************"
        )
        cprint("[   ok   ]     Controller initiated. ", "green", attrs=["bold"])
        print(
            "******************************************************************************\n"
        )
        print("Start controlling the robot hand using the SpaceMouse.\n")

        calibrate = True

        while True:
            # reset the desired
            self.tmp_desired_arm_pose = copy(self.desired_arm_pose)
            self.tmp_desired_hand_dof_pos = copy(self.desired_hand_dof_pos)
            desired_cmd = self.motion()
            print(f"transformed tcp quat: {desired_cmd[3:7]}")

            self.robot.move(desired_cmd)

            # update the desired
            self.desired_arm_pose = copy(self.tmp_desired_arm_pose)
            self.desired_hand_dof_pos = copy(self.tmp_desired_hand_dof_pos)

            if calibrate:
                tcp_pose = self.robot.get_arm_tcp_position()
                self.arm_pose_publisher.publish(Float64MultiArray(data=tcp_pose))


if __name__ == "__main__":
    teleop = SpaceMouseTeleop()
    teleop.move()
