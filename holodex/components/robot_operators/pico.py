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

def swap_y_z_axis(T):
    """
    Swap Y and Z axes in a 4x4 transformation matrix.
    
    Args:
        T (np.ndarray): 4x4 transformation matrix
    
    Returns:
        np.ndarray: New transformation matrix with Y and Z swapped
    """
    # Make a copy to avoid modifying the original
    T_new = T.copy()
    
    # Swap rotation rows (Y and Z)
    T_new[1, :], T_new[2, :] = T[2, :], T[1, :]
    
    # Swap rotation columns (Y and Z)
    T_new[:, 1], T_new[:, 2] = T_new[:, 2], T_new[:, 1].copy()
    
    return T_new

def rfu_to_flu(T_rfu):
    """
    Convert a transformation matrix from RFU (Right, Front, Up) to FLU (Front, Left, Up).
    
    Args:
        T_rfu (np.ndarray): 4x4 transformation matrix in RFU coordinates
    
    Returns:
        np.ndarray: 4x4 transformation matrix in FLU coordinates
    """
    # Transformation matrix C (RFU -> FLU)
    C = np.array([
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Compute T_flu = C @ T_rfu @ C^{-1}
    # Since C is orthonormal, C^{-1} = C.T
    C_inv = C.T
    
    T_flu = C @ T_rfu @ C_inv
    
    return T_flu
    
class PICODexArmTeleOp:
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
        self.joystick_pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) # xyz, wxyz

    def _setup_subscribers(self):
        """Set up all ROS subscribers"""
        topics_callbacks = [
            (JAKA_EE_POSE_TOPIC, Float64MultiArray, self._callback_arm_ee_pose),
            ("/data_collector/reset_done", Bool, self._callback_reset_done),
            ("/data_collector/reset_robot", Bool, self._callback_reset_robot),
            ("/data_collector/stop_move", Bool, self._callback_stop_move),
            ("/data_collector/end_robot", Bool, self._callback_end_robot),
            ("vr/gripper", Float64, self._callback_gripper),
            ("vr/ee_pose", Pose, self._callback_ee_pose),
        ]
        for topic, msg_type, callback in topics_callbacks:
            rospy.Subscriber(topic, msg_type, callback, queue_size=1)

    def _get_tcp_position(self):
        """Get the TCP position based on the arm type"""
        if self.arm_type == "flexiv":
            return self.robot.arm.get_tcp_position(euler=False, degree=False)
        else:
            return self.robot.arm.get_tcp_position()

    def _callback_ee_pose(self, pose):
        """Callback function to update joystick pose from VR data
        
        Args:
            pose: Pose message containing VR end-effector pose

        Note:
            This is based on the assumption that the VR end-effector pose is in the left hand coordinate system.
            Please modify the callback function if the VR end-effector pose is in a different coordinate system.
        """
        pos = np.array([
            pose.position.x,
            pose.position.y,
            pose.position.z
        ])
        
        # 转换四元数为旋转矩阵
        quat = [
            pose.orientation.w,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z
        ]
            
        rot = quat2mat(quat)
        transmat = np.zeros((4,4))
        transmat[:3, :3] = rot
        transmat[:3, 3] = pos
        transmat = swap_y_z_axis(transmat)
        transmat = rfu_to_flu(transmat)
        rot = transmat[:3, :3]
        pos = transmat[:3, 3]
        rot = mat2quat(rot)
        self.joystick_pose = np.concatenate([pos, rot], axis=0)

    def _callback_gripper(self, data):
        """Callback function to update gripper from VR data"""
        self.gripper_control = np.array(data.data)

    def _callback_arm_ee_pose(self, data):
        """Callback function to update arm end-effector pose"""
        self.arm_ee_pose = np.array(data.data)

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

    def _retarget_base(self):
        """Retarget the base position of the robot arm"""
        current_arm_pose = self.init_arm_ee_pose.copy()
        current_arm_pose[:3]  = self.joystick_pose[:3] * self.trans_scale + self.init_arm_ee_to_world[:3, 3]
        current_arm_pose[3:7] = mat2quat(quat2mat(self.joystick_pose[3:7]) @ self.init_arm_ee_to_world[:3, :3])
        return current_arm_pose
    
    def move(self, finger_configs):
        """Main control loop for robot movement"""
        print("\n" + "*" * 78)
        cprint("[   ok   ]     Controller initiated. ", "green", attrs=["bold"])
        print("*" * 78 + "\n")
        print("Start controlling the robot hand using the PICO VR.\n")

        while True:
            if self.joystick_pose is not None:
                if self.stop_move:
                    continue
                if self.end_robot:
                    break
                # Generate desired joint angles based on current joystick pose
                desired_cmd = self._retarget_base()
                if self.robot.arm.with_gripper:
                    self.robot.move(np.concatenate([desired_cmd, np.expand_dims(self.gripper_control, axis=0)]))
                else:
                    self.robot.move(desired_cmd)