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
        self.trans_scale = 1
        self.gripper_control = float(gripper_init_state == "close")
        self.logger = spdlog.ConsoleLogger("RobotController")

        # Initialize state variables
        self.stop_move = False
        self.end_robot = False

        # Set up ROS subscribers
        self._setup_subscribers()

        # Initialize robot controller
        self.robot = RobotController(teleop=True, simulator=simulator, gripper=gripper, arm_type=arm_type, gripper_init_state=gripper_init_state)
        self.init_arm_ee_pose = self._get_tcp_position()
        self.init_arm_ee_to_world = np.eye(4)
        self.init_arm_ee_to_world[:3, 3] = self.init_arm_ee_pose[:3]
        self.init_arm_ee_to_world[:3, :3] = quat2mat(self.init_arm_ee_pose[3:7])
        self.arm_ee_pose = None

    def _setup_subscribers(self):
        """Set up all ROS subscribers"""
        topics_callbacks = [
            ("/data_collector/reset_done", Bool, self._callback_reset_done),
            ("/data_collector/reset_robot", Bool, self._callback_reset_robot),
            ("/data_collector/stop_move", Bool, self._callback_stop_move),
            ("/data_collector/end_robot", Bool, self._callback_end_robot),
        ]
        for topic, msg_type, callback in topics_callbacks:
            rospy.Subscriber(topic, msg_type, callback, queue_size=1)

    def _get_tcp_position(self):
        """Get the TCP position based on the arm type"""
        if self.arm_type == "flexiv":
            return self.robot.arm.get_tcp_position(euler=False, degree=False)
        else:
            return self.robot.arm.get_tcp_position()

    def _callback_end_robot(self, msg):
        """Callback function to set end_robot flag"""
        self.end_robot = msg.data

    def _callback_stop_move(self, msg):
        """Callback function to set stop_move flag"""
        self.stop_move = msg.data

    def _callback_reset_robot(self, msg):
        """Callback function to reset robot position"""
        if self.robot.arm.with_gripper:
            self.robot.move(np.concatenate([self.init_arm_ee_pose, np.expand_dims(self.robot.arm.gripper_init_state == "close", axis=0)]))
        else:
            self.robot.move(np.concatenate([self.init_arm_ee_pose]))
        
    def _callback_reset_done(self, msg):
        """Callback function to handle reset done event"""
        if self.robot.arm.with_gripper:
            self.robot.move(np.concatenate([self.init_arm_ee_pose, np.expand_dims(self.robot.arm.gripper_init_state == "close", axis=0)]))
        else:
            self.robot.move(np.concatenate([self.init_arm_ee_pose]))


    def move(self):
        """Main control loop for robot movement"""
        print("\n" + "*" * 78)
        cprint("[   ok   ]     Controller initiated. ", "green", attrs=["bold"])
        print("*" * 78 + "\n")
        print("Start controlling the robot hand using the PICO VR.\n")

        while True:
            if self.stop_move:
                continue
            if self.end_robot:
                break
            # Generate desired joint angles based on current joystick pose
            desired_cmd = self.init_arm_ee_pose.copy()
            if self.robot.arm.with_gripper:
                self.robot.move(np.concatenate([desired_cmd, np.expand_dims(self.gripper_control, axis=0)]))
            else:
                self.robot.move(desired_cmd)