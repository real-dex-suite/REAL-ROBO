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

def remove_euler_component_scipy(quat, remove_roll=False, remove_pitch=False, remove_yaw=False):
    """
    Remove specified Euler angle components from a quaternion using SciPy
    
    Args:
        quat: Input quaternion in [w, x, y, z] format (scalar first)
        remove_roll: Flag to remove roll component (x-axis rotation)
        remove_pitch: Flag to remove pitch component (y-axis rotation)
        remove_yaw: Flag to remove yaw component (z-axis rotation)
        
    Returns:
        New quaternion with specified components removed [w, x, y, z]
    """
    # Note: SciPy uses xyzw format (scalar last), so we need to convert input
    quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
    
    # Create Rotation object from quaternion
    rotation = R.from_quat(quat_xyzw)
    
    # Extract Euler angles (using 'xyz' convention: roll, pitch, yaw)
    euler_angles = rotation.as_euler('xyz', degrees=False)
    
    # Zero out the components we want to remove
    if remove_roll:
        euler_angles[0] = 0.0  # Roll (x-axis)
    if remove_pitch:
        euler_angles[1] = 0.0  # Pitch (y-axis)
    if remove_yaw:
        euler_angles[2] = 0.0  # Yaw (z-axis)
    
    # Create new rotation from modified Euler angles
    new_rotation = R.from_euler('xyz', euler_angles)
    
    # Convert back to quaternion and adjust to wxyz format
    new_quat_xyzw = new_rotation.as_quat()
    new_quat = np.array([new_quat_xyzw[3], new_quat_xyzw[0], new_quat_xyzw[1], new_quat_xyzw[2]])
    
    return new_quat

class PICODexArmTeleOp:
    def __init__(self, simulator=None, gripper=None, arm_type="franka", gripper_init_state="open", lock_rotation=["pitch", "roll"], lock_z=False):
        self.arm_type = arm_type
        self.trans_scale = 1
        self.gripper_control = float(gripper_init_state == "close")
        self.logger = spdlog.ConsoleLogger("RobotController")

        # Initialize state variables
        self.stop_move = False
        self.end_robot = False

        self._setup_params()
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
        self.lock_rotation = lock_rotation
        self.lock_z = lock_z

    def _setup_params(self):
        rospy.set_param("/data_collector/stop_move", False)
        rospy.set_param("/data_collector/end_robot", False)
        rospy.set_param("/data_collector/reset_robot", False)
        
    def _setup_subscribers(self):
        """Set up all ROS subscribers"""
        topics_callbacks = [
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
        if pose is not None:
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

    def _retarget_base(self):
        """Retarget the base position of the robot arm"""
        current_arm_pose = self.init_arm_ee_pose.copy()
        if self.lock_z:
            current_arm_pose[:2]  = self.joystick_pose[:2] * self.trans_scale + self.init_arm_ee_to_world[:2, 3]
            current_arm_pose[2:3] = self.init_arm_ee_to_world[2:3, 3].copy()
        else:
            current_arm_pose[:3]  = self.joystick_pose[:3] * self.trans_scale + self.init_arm_ee_to_world[:3, 3]
        # NOTE: quat is wxyz.
        filtered_quat = remove_euler_component_scipy(self.joystick_pose[3:7], 
                                                    remove_roll="roll" in self.lock_rotation,
                                                    remove_pitch="pitch" in self.lock_rotation,
                                                    remove_yaw="yaw" in self.lock_rotation,)
        current_arm_pose[3:7] = mat2quat(quat2mat(filtered_quat) @ self.init_arm_ee_to_world[:3, :3])
        return current_arm_pose
    
    def move(self):
        """Main control loop for robot movement"""
        print("\n" + "*" * 78)
        cprint("[   ok   ]     Controller initiated. ", "green", attrs=["bold"])
        print("*" * 78 + "\n")
        print("Start controlling the robot hand using the PICO VR.\n")

        while True:
            if rospy.get_param("/data_collector/reset_robot"):
                self.robot.home_robot()
                rospy.set_param("/data_collector/reset_robot", False)
            if rospy.get_param("/data_collector/end_robot"):
                os._exit(0)
            if rospy.get_param("/data_collector/stop_move"):
                continue
            # Generate desired joint angles based on current joystick pose
            desired_cmd = self._retarget_base()
            if self.robot.arm.with_gripper:
                self.robot.move(np.concatenate([desired_cmd, np.expand_dims(self.gripper_control, axis=0)]))
            else:
                self.robot.move(desired_cmd)