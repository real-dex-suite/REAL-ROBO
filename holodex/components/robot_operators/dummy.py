import rospy
from std_msgs.msg import Float64MultiArray, Bool, Float64
from geometry_msgs.msg import Pose

from holodex.utils.files import *
from holodex.utils.vec_ops import coord_in_bound, best_fit_transform, normalize_vector
from holodex.constants import *
from copy import deepcopy as copy
from scipy.spatial.transform import Rotation as R
from termcolor import cprint
from typing import Tuple
import spdlog
import numpy as np
from transforms3d.quaternions import quat2mat, mat2quat

try:
    from .robot import RobotController
except ImportError:
    from robot import RobotController

class DummyDexArmTeleOp:
    def __init__(self, simulator=None, gripper=None, arm_type="franka", gripper_init_state="open"):
        self.arm_type = arm_type
        self.gripper_control = float(gripper_init_state == "close")
        self.logger = spdlog.ConsoleLogger("RobotController")

        # Initialize robot controller
        self.robot = RobotController(teleop=True, simulator=simulator, gripper=gripper, arm_type=arm_type, gripper_init_state=gripper_init_state)
        self.init_arm_ee_pose = self._get_tcp_position()

    def _get_tcp_position(self):
        """Get the TCP position based on the arm type"""
        if self.arm_type == "flexiv":
            return self.robot.arm.get_tcp_position(euler=False, degree=False)
        else:
            return self.robot.arm.get_tcp_position()
        
    def move(self):
        """Main control loop for robot movement"""
        print("\n" + "*" * 78)
        cprint("[   ok   ]     Controller initiated. ", "green", attrs=["bold"])
        print("*" * 78 + "\n")
        print("Start controlling the robot hand using the Dummy Teleop.\n")

        while True:
            # Generate desired joint angles based on current joystick pose
            desired_cmd = self.init_arm_ee_pose.copy()
            if self.robot.arm.with_gripper:
                self.robot.move(np.concatenate([desired_cmd, np.expand_dims(self.gripper_control, axis=0)]))
            else:
                self.robot.move(desired_cmd)