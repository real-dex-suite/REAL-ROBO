#!/usr/bin/env python
import time
import numpy as np
try:
    from holodex.robot.arm.franka.kinematics_solver import FrankaSolver
except:
    from kinematics_solver import FrankaSolver
from scipy.spatial.transform import Rotation as R
import os
os.environ['MUJOCO_GL'] = 'glx'
os.environ['PYOPENGL_PLATFORM'] = 'glx'
import torch

from std_msgs.msg import Float64MultiArray, Bool
import rospy

class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha 
        self.prev_value = None
    
    def __call__(self, new_value):
        if self.prev_value is None:
            self.prev_value = new_value
        else:
            self.prev_value = self.alpha * new_value + (1 - self.alpha) * self.prev_value
        return self.prev_value
    
class FrankaGenesisEnvWrapper:
    def __init__(self, control_mode="joint", teleop=False):
        rospy.init_node('genesis_tele', anonymous=True)
        rospy.sleep(1.0)

        self.ik_solver = FrankaSolver()
        self.current_joint_state = None
        self.current_ee_state = None
        self.joint_control_pub = rospy.Publisher(
            "/genesis/joint_control",
            Float64MultiArray,
            queue_size=1,
        )
        self.gripper_control_pub = rospy.Publisher(
            "/genesis/gripper_control",
            Bool,
            queue_size=1,
        )
        self.reset_pub = rospy.Publisher(
            "/genesis/reset_robot",
            Bool,
            queue_size=1,
        )
        self.joint_state_sub = rospy.Subscriber(
            "/genesis/joint_states",
            Float64MultiArray,
            self._callback_current_joint_state,
            queue_size=1,
        )
        self.ee_state_sub = rospy.Subscriber(
            "/genesis/ee_states",
            Float64MultiArray,
            self._callback_current_ee_state,
            queue_size=1,
        )

        self._initialize_state()
        
    def _initialize_state(self):
        if self.current_joint_state is None:
            msg = rospy.wait_for_message('/genesis/joint_states', Float64MultiArray, timeout=5.0)
            self.current_joint_state = msg.data
        if self.current_ee_state is None:
            msg = rospy.wait_for_message('/genesis/ee_states', Float64MultiArray, timeout=5.0)
            self.current_ee_state = msg.data
        
    def _callback_current_joint_state(self, msg):
        self.current_joint_state = msg.data

    def _callback_current_ee_state(self, msg):
        self.current_ee_state = msg.data

    def eulerZYX2quat(self, euler, degree=False):
        if degree:
            euler = np.radians(euler)

        tmp_quat = R.from_euler("xyz", euler).as_quat().tolist()
        quat = [tmp_quat[3], tmp_quat[0], tmp_quat[1], tmp_quat[2]]
        return quat

    def get_arm_position(self):
        # Get the current joint positions of the arm
        return np.array(self.current_joint_state)

    def get_tcp_position(self):
        """
        Get the TCP position of the robot
        return:
            Translation: [x, y, z]
            Quaternion: [w, x, y, z]
        """
        # Retrieve the current end-effector pose and return it as a concatenated array
        return np.array(self.current_ee_state)

    def ee2joint(self, ee_pose):
        # Convert end-effector pose to joint positions using inverse kinematics
        np.set_printoptions(precision=4, suppress=True)
        ik_res = self.ik_solver.solve_ik(ee_pose[:3], ee_pose[3:])
        return ik_res

    def open_gripper(self):
        # Open the robot's gripper
        gripper_msg = Bool(data=True)
        self.gripper_control_pub.publish(gripper_msg)
        
    def close_gripper(self):
        # Open the robot's gripper
        gripper_msg = Bool(data=False)
        self.gripper_control_pub.publish(gripper_msg)

    def reset(self):
        """
        This function is used to reset the robot to the home position from the frankapy.
        """
        print("publishing...")
        rate = rospy.Rate(10)
        for _ in range(5):
            reset_msg = Bool(data=True)
            self.reset_pub.publish(reset_msg)
            rate.sleep()

    def solve_ik(self, ee_pose: list) -> list:
        """
        Solve inverse kinematics.

        Args:
            ee_pose (list): The end-effector pose in the form [x, y, z, qx, qy, qz, qw].

        Returns:
            list: The joint positions that achieve the desired end-effector pose.

        Raises:
            ValueError: If no IK solution found
        """
        
        ik_res = self.ik_solver.solve_ik_by_motion_gen(
            self.get_arm_position(), ee_pose[:3], ee_pose[3:]
        )
        if ik_res is None:
            return None
        ik_res = np.array(ik_res[-1])
        return ik_res

    def move_joint(self, target_joint): #! double check the type of target_joint
        target_joint = self.solve_ik(target_joint)
        if target_joint is not None:
            joint_pos_msg = Float64MultiArray(data=target_joint)
            self.joint_control_pub.publish(joint_pos_msg)

    def run(self):
        pass

    def shutdown(self):
        # Placeholder for shutdown procedures
        pass

if __name__ == '__main__':
    controller = FrankaGenesisEnvWrapper()
    controller.run()