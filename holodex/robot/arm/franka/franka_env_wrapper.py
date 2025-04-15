#!/usr/bin/env python
import time
import rospy
import numpy as np
import roslib

roslib.load_manifest("franka_interface_msgs")
from frankapy import FrankaArm, SensorDataMessageType, FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointPositionSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
from holodex.robot.arm.franka.kinematics_solver import FrankaSolver
from scipy.spatial.transform import Rotation as R
from frankapy.proto import (
    PosePositionSensorMessage,
    CartesianImpedanceSensorMessage,
)


class FrankaEnvWrapper:
    """
    Wrapper class for controlling Franka robot arm.

    This class provides an interface to the Franka Emika Panda robot using frankapy.
    It handles joint and Cartesian control, gripper operations, and inverse kinematics.

    Dependencies:
        - frankapy: https://github.com/iamlab-cmu/frankapy
        - curobo: High-performance IK solver (https://curobo.org/)

    The wrapper simplifies robot control by providing high-level methods for common
    operations while handling the underlying ROS communication and state management.
    """

    def __init__(self):
        """Initialize robot arm controller."""
        self.arm = FrankaArm("franka_arm_reader")
        rospy.loginfo("Initializing FrankaWrapper...")

        self._initialize_state()
        self._initialize_joint_control_config()

        self.cmd_pub = rospy.Publisher(
            FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000
        )

        self._fa_cmd_id = 0
        self._init_time = rospy.Time.now().to_time()
        self.ik_solver = FrankaSolver("ik_solver")

    def _initialize_state(self):
        """Initialize robot state variables."""
        self.current_joint_state = self.arm.get_joints()
        self.joint_state = self.current_joint_state[:]
        self.current_ee_pose = self.arm.get_pose()
        self.ee_pose = self.current_ee_pose

    def _initialize_joint_control_config(self):
        """Configure joint control parameters."""
        self.arm.goto_joints(
            self.current_joint_state,
            duration=10000,
            k_gains=[100.0, 100.0, 150.0, 400.0, 400.0, 600.0, 80.0],
            d_gains=[30.0, 30.0, 40.0, 150.0, 100.0, 30.0, 15.0],
            dynamic=True,
            buffer_time=10,
        )

    def _initialize_cartesian_control_config(self):
        """Configure Cartesian control parameters."""
        self.arm.goto_pose(
            FC.HOME_POSE,
            duration=10000,
            dynamic=True,
            buffer_time=10000,
            cartesian_impedances=[1200.0, 1200.0, 1200.0, 50.0, 50.0, 50.0],
        )

    def get_arm_position(self) -> list:
        """
        Get current joint positions.

        Returns:
            list: The current joint state of the robot arm.
        """
        return self.arm.get_joints()

    def get_tcp_position(self) -> np.ndarray:
        """
        Get TCP position and orientation.

        Returns:
            np.ndarray: [x, y, z, qw, qx, qy, qz]
        """
        self.current_ee_pose = self.arm.get_pose()
        trans = self.current_ee_pose.translation
        rot_quat = self.current_ee_pose.quaternion
        return np.concatenate([trans, rot_quat])

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
        np.set_printoptions(precision=4, suppress=True)
        print("ee_pose", ee_pose)
        print("cur_ee_pose", self.get_tcp_position())
        ik_res = self.ik_solver.solve_ik(ee_pose[:3], ee_pose[3:])
        if ik_res is None:
            raise ValueError("IK solution not found")
        return ik_res

    def get_pose_as_matrix(self) -> np.ndarray:
        """
        Get current end-effector pose as 4x4 transformation matrix.

        Returns:
            np.ndarray: A 4x4 NumPy array representing the transformation matrix.
        """
        trans = self.current_ee_pose.translation
        rotation_mat = self.current_ee_pose.rotation
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_mat
        transformation_matrix[:3, 3] = trans
        return transformation_matrix

    def open_gripper(self):
        """
        Open gripper to maximum width.

        Returns:
            None
        """
        self.arm.open_gripper(block=True, skill_desc="OpenGripper")

    def close_gripper(self):
        """Close gripper and attempt to grasp object."""
        self.arm.close_gripper(grasp=True, block=True, skill_desc="CloseGripper")

    def get_gripper_width(self) -> float:
        """
        Get current gripper width.

        Returns:
            float: The current width of the gripper.
        """
        return self.arm.get_gripper_width()

    def get_gripper_is_grasped(self) -> bool:
        """
        Check if gripper is currently grasping an object.

        Returns:
            bool: True if the gripper is grasping something, False otherwise.
        """
        return self.arm.get_gripper_is_grasped()

    def move_gripper(self, target_width: float):
        """
        Move gripper to target width.

        Args:
            target_width (float): Target gripper width in meters

        """
        raise NotImplementedError(
            "Gripper control not implemented yet. Contact Jinzhou"
        )

    def move_cartesian(self, target_pose: list):
        """
        Move end-effector to target Cartesian pose.

        Args:
            target_pose (list): The target pose for the robot in the form [x, y, z, qx, qy, qz, qw].

        """
        assert len(target_pose) == 7, "target_pose must be a list of length 7"
        timestamp = rospy.Time.now().to_time() - self._init_time
        self._fa_cmd_id += 1

        traj_gen_proto_msg = PosePositionSensorMessage(
            id=self._fa_cmd_id,
            timestamp=timestamp,
            position=target_pose[:3],
            quaternion=target_pose[3:],
        )

        fb_ctrlr_proto = CartesianImpedanceSensorMessage(
            id=self._fa_cmd_id,
            timestamp=timestamp,
            translational_stiffnesses=[600.0, 600.0, 600.0],
            rotational_stiffnesses=FC.DEFAULT_ROTATIONAL_STIFFNESSES,
        )

        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION
            ),
            feedback_controller_sensor_msg=sensor_proto2ros_msg(
                fb_ctrlr_proto, SensorDataMessageType.CARTESIAN_IMPEDANCE
            ),
        )

        self.cmd_pub.publish(ros_msg)

    def move_joint(self, target_joint: list):
        """
        Move joints to target positions.

        Args:
            target_joint (list): The target joint position for the robot.

        Returns:
            None
        """
        timestamp = rospy.Time.now().to_time() - self._init_time

        self._fa_cmd_id += 1
        traj_gen_proto_msg = JointPositionSensorMessage(
            id=self._fa_cmd_id, timestamp=timestamp, joints=target_joint
        )
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION
            )
        )
        self.cmd_pub.publish(ros_msg)

    def eulerZYX2quat(self, euler: list, degree: bool = False) -> list:
        """
        Convert Euler ZYX angles to quaternion.

        Args:
            euler (list): [roll, pitch, yaw]
            degree (bool): If True, input is in degrees

        Returns:
            list: [qw, qx, qy, qz]
        """
        if degree:
            euler = np.radians(euler)
        tmp_quat = R.from_euler("xyz", euler).as_quat().tolist()
        return [tmp_quat[3], tmp_quat[0], tmp_quat[1], tmp_quat[2]]

    def eulerXYZ2quat(self, euler: list, degree: bool = False) -> list:
        """
        Convert Euler XYZ angles to quaternion.

        Args:
            euler (list): [roll, pitch, yaw]
            degree (bool): If True, input is in degrees

        Returns:
            list: [qw, qx, qy, qz]
        """
        if degree:
            euler = np.radians(euler)
        tmp_quat = R.from_euler("xyz", euler).as_quat().tolist()
        return [tmp_quat[3], tmp_quat[0], tmp_quat[1], tmp_quat[2]]

    def run(self):
        """
        Run test motion sequence.

        Returns:
            None
        """
        rate = rospy.Rate(10)
        for i in range(11):
            if rospy.is_shutdown():
                break
            x = self.get_arm_position()
            x = [j + 0.05 for j in x]  # Increment all joints by 0.05
            self.move_joint(x)
            print(self.current_joint_state)
            rate.sleep()
        self.open_gripper()

    def shutdown(self):
        """
        Clean shutdown of robot controller.

        Returns:
            None
        """
        pass


if __name__ == "__main__":
    try:
        controller = FrankaEnvWrapper()
        controller.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        controller.shutdown()
