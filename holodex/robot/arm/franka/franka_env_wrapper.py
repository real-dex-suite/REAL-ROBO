#!/usr/bin/env python
import time
import rospy
import numpy as np
import roslib


roslib.load_manifest("franka_interface_msgs")
from std_msgs.msg import Float64MultiArray
from frankapy import FrankaArm, SensorDataMessageType, FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointPositionSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
from holodex.robot.arm.franka.kinematics_solver import FrankaSolver

# from kinematics_solver import FrankaSolver
from scipy.spatial.transform import Rotation as R
from frankapy.utils import min_jerk, min_jerk_weight
from frankapy.proto import (
    PosePositionSensorMessage,
    ShouldTerminateSensorMessage,
    CartesianImpedanceSensorMessage,
)


class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.prev_value = None

    def __call__(self, new_value):
        if self.prev_value is None:
            self.prev_value = new_value
        else:
            self.prev_value = (
                self.alpha * new_value + (1 - self.alpha) * self.prev_value
            )
        return self.prev_value


class FrankaEnvWrapper:
    def __init__(self):
        self.arm = FrankaArm("franka_arm_reader")
        rospy.loginfo("Initializing FrankaWrapper...")

        self._initialize_state()
        self._initialize_joint_control_config()

        # publishers msg to control the robot
        self.cmd_pub = rospy.Publisher(
            FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000
        )

        self._fa_cmd_id = 0
        self.ik_solver = FrankaSolver()  # implement by curobo
        # if we use vr, this is no longer needed
        self.low_pass_filter = LowPassFilter(0.2)  # higher meaning more smoothing

        # TODO: add logic to process the gripper from franka or third party like robotiq

    def _initialize_state(self):
        # Initialize the current joint state and end-effector pose
        self.current_joint_state = self.arm.get_joints()
        self.joint_state = self.current_joint_state[:]
        self.current_ee_pose = self.arm.get_pose()
        self.ee_pose = self.current_ee_pose

    def _initialize_joint_control_config(self):
        # This configuration is calibrated between the real robot and simulation parameters
        self.arm.goto_joints(
            self.current_joint_state,
            duration=10000,
            k_gains=[100.0, 100.0, 150.0, 400.0, 400.0, 600.0, 80.0],
            d_gains=[30.0, 30.0, 40.0, 150.0, 100.0, 30.0, 15.0],
            dynamic=True,
            buffer_time=10,
        )

    def _initialize_cartesian_control_config(self):
        self.arm.goto_pose(
            FC.HOME_POSE,  #! check this
            duration=10000,
            dynamic=True,
            buffer_time=10000,
            cartesian_impedances=[1200.0, 1200.0, 1200.0, 50.0, 50.0, 50.0],
        )


    def get_arm_position(self)-> list:
        """
        Get the current joint state of the robot arm.
        return:
            list: The current joint state of the robot arm.
        """
        return self.arm.get_joints()


    def get_tcp_position(self):
        """
        Get the TCP position of the robot
        return:
            Translation: [x, y, z]
            Quaternion: [w, x, y, z]
        """
        # Retrieve the current end-effector pose and return it as a concatenated array
        self.current_ee_pose = self.arm.get_pose()
        trans = self.current_ee_pose.translation
        rot_quat = self.current_ee_pose.quaternion
        ee_pose = np.concatenate([trans, rot_quat])
        return ee_pose
    

    def solve_ik(self, ee_pose):
        """
        Solve the inverse kinematics for the given end-effector pose.
        Parameters:
            ee_pose (list): The end-effector pose in the form [x, y, z, qx, qy, qz, qw].
        Returns:
            list: The joint positions that achieve the desired end-effector pose.
        """
        np.set_printoptions(precision=4, suppress=True)
        print("ee_pose", ee_pose)
        print("cur_ee_pose", self.get_tcp_position())
        ik_res = self.ik_solver.solve_ik(ee_pose[:3], ee_pose[3:])
        if ik_res is None:
            raise ValueError("IK solution not found")
        return ik_res


    def get_transformation_matrix(self):
        # Compute the transformation matrix from the current end-effector pose
        trans = self.current_ee_pose.translation
        rotation_mat = self.current_ee_pose.rotation
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_mat
        transformation_matrix[:3, 3] = trans

        self.arm.inverse_kinematics(transformation_matrix)
        return transformation_matrix
    

    def get_pose_as_matrix(self) -> np.ndarray:
        """
        Gets the current end-effector pose (from self.current_ee_pose)
        as a 4x4 homogeneous transformation matrix.

        Warning: Relies on self.current_ee_pose being updated externally.

        Returns:
            np.ndarray: A 4x4 NumPy array representing the transformation matrix.
        """
        # Using the potentially cached pose:
        trans = self.current_ee_pose.translation
        rotation_mat = self.current_ee_pose.rotation

        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_mat
        transformation_matrix[:3, 3] = trans

        # Removed the incorrect self.arm.inverse_kinematics call

        return transformation_matrix


    def open_gripper(self):
        """
        Opens gripper to maximum width

        Parameters:
            block (bool) - Function blocks by default. If False, the function becomes asynchronous
                           and can be preempted.
            skill_desc (str) - Skill description to use for logging on control-pc.

        """
        self.arm.open_gripper(block=True, skill_desc="OpenGripper")


    def close_gripper(self):
        """
        Closes the gripper as much as possible

        Parameters:
            grasp (bool) - Flag that signals whether to grasp.
            block (bool) - Function blocks by default. If False, the function becomes asynchronous
                           and can be preempted.
            skill_desc (str) - Skill description to use for logging on control-pc.
        """
        # Close the robot's gripper
        self.arm.close_gripper(grasp=True, block=True, skill_desc="CloseGripper")


    def get_gripper_width(self):
        """
        Returns the current width of the gripper.

        Returns:
            float: The current width of the gripper.

        """
        return self.arm.get_gripper_width()


    def get_gripper_is_grasped(self):
        """
        Returns a flag that represents if the gripper is grasping something.

        Returns:
            True if the gripper is grasping something, False otherwise.
        """
        return self.arm.get_gripper_is_grasped()


    def move_gripper(self, target_width):
        """
        Move the gripper to a target width.

        Parameters:
            target_width (float): The target width for the gripper.
        """

        raise NotImplementedError(
            "Gripper control not implemented yet. Contact Jinzhou"
        )


    def move_cartesian(self, target_pose):
        '''
        Move the robot to a target pose in Cartesian space.
        Parameters:
            target_pose (list): The target pose for the robot in the form [x, y, z, qx, qy, qz, qw].
        This function uses the Franka arm's pose position sensor message to control the robot's end-effector.
        It publishes the pose position command to the appropriate ROS topic.
        '''
        assert len(target_pose) == 7, "target_pose must be a list of length 7"
        init_time = rospy.Time.now().to_time()
        timestamp = rospy.Time.now().to_time() - init_time
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
            translational_stiffnesses=[600.0, 600.0, 600.0],  # ! Double Check
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


    def move_joint(self, target_joint):
        '''
        Move the robot to a target joint position.

        Parameters:
            target_joint (list): The target joint position for the robot.

        This function uses the Franka arm's joint position sensor message to control the robot's joints.
        It publishes the joint position command to the appropriate ROS topic.
        
        '''
        init_time = rospy.Time.now().to_time()
        timestamp = rospy.Time.now().to_time() - init_time

        self._fa_cmd_id += 1
        traj_gen_proto_msg = JointPositionSensorMessage(
            id=0, timestamp=timestamp, joints=target_joint
        )
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION
            )
        )
        self.cmd_pub.publish(ros_msg)


    def eulerZYX2quat(self, euler, degree=False):
        if degree:
            euler = np.radians(euler)

        tmp_quat = R.from_euler("xyz", euler).as_quat().tolist()
        quat = [tmp_quat[3], tmp_quat[0], tmp_quat[1], tmp_quat[2]]
        return quat

    def eulerXYZ2quat(self, euler, degree=False):
        pass


    def run(self):
        # Main loop to move the robot and open the gripper
        rate = rospy.Rate(10)  # test in 10 Hz
        for i in range(11):
            if rospy.is_shutdown():
                break
            x = self.get_arm_position()
            x[6] += 0.05
            x[5] += 0.05
            x[4] += 0.05
            x[3] += 0.05
            x[2] += 0.05
            x[1] += 0.05
            x[0] += 0.05
            self.move_joint(x)
            print(self.current_joint_state)
            rate.sleep()
        self.open_gripper()

    def shutdown(self):
        # Placeholder for shutdown procedures
        pass


if __name__ == "__main__":
    try:
        controller = FrankaEnvWrapper()
        controller.run()

    except rospy.ROSInterruptException:
        pass
    finally:
        controller.shutdown()
