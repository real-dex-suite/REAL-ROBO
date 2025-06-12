#!/usr/bin/env python
import rospy
import numpy as np
import roslib

try:
    roslib.load_manifest("franka_interface_msgs")
    from frankapy import FrankaArm, SensorDataMessageType, FrankaConstants as FC
    from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
    from frankapy.proto import JointPositionSensorMessage
    from franka_interface_msgs.msg import SensorDataGroup
    from frankapy.proto import (
        PosePositionSensorMessage,
        CartesianImpedanceSensorMessage,
    )
except:
    rospy.logwarn("frankapy not loaded! Please check whether we are doing teleop in simulation.")
try:
    from holodex.robot.arm.franka.kinematics_solver import FrankaSolver
except:
    from kinematics_solver import FrankaSolver
from scipy.spatial.transform import Rotation as R
from termcolor import cprint
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

    def __init__(self, control_mode: str = "joint", teleop: bool = False, gripper="ctek", gripper_init_state="open"):
        """
        Initialize robot arm controller.

        Args:
            control_mode (str): Control mode for the robot arm.
                Options: "joint" (default) or "cartesian"
        """
        assert gripper in ["ctek", "panda"] or gripper is None, f"Gripper {gripper} is not supported for FrankaEnvWrapper."
        self.arm = FrankaArm(rosnode_name="franka_arm_reader", with_gripper=gripper == "panda")
        rospy.loginfo("Initializing FrankaWrapper...")

        self._initialize_state()
        self.teleop = teleop
        self.dof = 7
        self.gripper = gripper
        self.with_gripper = gripper is not None
        if self.with_gripper:
            self.dof += 1
        
        if self.gripper == "ctek":
            from holodex.robot.gripper.ctek import CTekGripper
            self.gripper_wrapper = CTekGripper()
        elif self.gripper == "panda":
            self.gripper_wrapper = self.arm
        else:
            raise NotImplementedError(f"Gripper {self.gripper} is not supported for FrankaEnvWrapper.")
        # Set up the robot control configuration based on the specified mode
        if control_mode == "joint":
            # TODO: make these configurable
            self.joint_k_gains = [400.0, 350.0, 400.0, 400.0, 400.0,150.0,80.0]
            self.joint_d_gains = [100.0,  100.0,  80.0,  80.0,  80.0, 50.0,  15.0]
            # self.joint_k_gains  = [100.0, 100.0, 100.0, 100.0, 250.0, 150.0, 50.0]
            # self.joint_d_gains = [50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0]

            self._initialize_joint_control_config()
        elif control_mode == "cartesian":
            self._initialize_cartesian_control_config()
        else:
            raise ValueError(
                f"Unsupported control mode: '{control_mode}'. "
                "Supported modes are 'joint' or 'cartesian'."
            )

        self.cmd_pub = rospy.Publisher(
            FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000
        )

        self._fa_cmd_id = 0
        self._init_time = rospy.Time.now().to_time()
        self.ik_solver = FrankaSolver(ik_type="motion_gen", ik_sim=not (gripper=="panda"))
        self.gripper_init_state = gripper_init_state
        if gripper_init_state == "open":
            self.open_gripper()
        elif gripper_init_state == "close":
            self.close_gripper()
        else:
            raise NotImplementedError(f"Unknown gripper_init_state {gripper_init_state}")

    def _initialize_state(self):
        """Initialize robot state variables."""
        self.current_joint_state = self.arm.get_joints()

    def _initialize_joint_control_config(self):
        """Configure joint control parameters."""
        self.arm.goto_joints(
            self.current_joint_state,
            duration=10000,
            k_gains=self.joint_k_gains,
            d_gains=self.joint_d_gains,
            dynamic=True,
            buffer_time=10,
            ignore_virtual_walls=True,
        )

    def _initialize_cartesian_control_config(self):
        """Configure Cartesian control parameters."""
        self.arm.goto_pose(
            FC.HOME_POSE,
            duration=10000,
            dynamic=True,
            buffer_time=10000,
            cartesian_impedances=[1200.0, 1200.0, 1200.0, 50.0, 50.0, 50.0],
            ignore_virtual_walls=True,
        )

    def eulerZYX2quat(self, euler, degree=False):
        if degree:
            euler = np.radians(euler)

        tmp_quat = R.from_euler("xyz", euler).as_quat().tolist()
        quat = [tmp_quat[3], tmp_quat[0], tmp_quat[1], tmp_quat[2]]
        return quat

    def home_robot(self):
        pass
        
    def get_arm_position(self) -> list:
        """
        Get current joint positions.

        Returns:
            list: The current joint state of the robot arm.
        """
        return self.arm.get_joints()

    def get_tcp_position(self, fk_pose=True) -> np.ndarray:
        """
        Get TCP position and orientation.

        Returns:
            np.ndarray: [x, y, z, qw, qx, qy, qz]
        """
        if fk_pose:
            trans, rot_quat = self.ik_solver.compute_fk(self.get_arm_position())
        else:
            current_ee_pose = self.arm.get_pose()
            trans = current_ee_pose.translation
            rot_quat = current_ee_pose.quaternion
        return np.concatenate([trans, rot_quat])

    def solve_ik(self, ee_pose: list) -> list:
        """
        Solve inverse kinematics.

        Args:
            ee_pose (list): The end-effector pose in the form [x, y, z, qw, qx, qy, qz].

        Returns:
            list: The joint positions that achieve the desired end-effector pose.

        Raises:
            ValueError: If no IK solution found
        """
        ik_res = self.ik_solver.solve_ik_by_motion_gen(
            self.get_arm_position(), ee_pose[:3], ee_pose[3:]
        )
        ik_res = np.array(ik_res[-1])
        if ik_res is None:
            raise ValueError("IK solution not found")
        return ik_res

    def open_gripper(self, block=True):
        """
        Open gripper to maximum width.

        Returns:
            None
        """
        if self.gripper == "panda":
            self.gripper_wrapper.open_gripper(block=block, skill_desc="OpenGripper")
        elif self.gripper == "ctek":
            self.gripper_wrapper.open_gripper(block=block)
        else:
            pass

    def close_gripper(self, block=True):
        """Close gripper and attempt to grasp object."""
        if self.gripper == "panda":
            self.gripper_wrapper.close_gripper(grasp=True, block=block, skill_desc="CloseGripper")
        elif self.gripper == "ctek":
            self.gripper_wrapper.close_gripper(block=block)
        else:
            pass

    def move_joint(self, target_joint: list):
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
        
    def move_joint_ik(self, target_ee: list):
        """
        Move joints to target positions.

        Args:
            target_joint (list): The target joint position for the robot.

        Returns:
            None
        """
        target_joint = self.solve_ik(target_ee)
        self.move_joint(target_joint)
        
    def move_gripper(self, gripper_cmd: bool = True):
        """
        Control gripper for teleoperation with binary open/close command.
        Includes debouncing to avoid too frequent control commands.
        
        Args:
            gripper_cmd (float or int): Binary command for gripper
                - Values <= 0.05: Close the gripper
                - Values > 0.05: Open the gripper
                
        """
        if self.with_gripper:
            if self.gripper == "panda":
                if not gripper_cmd:
                    self.open_gripper()
                else:
                    self.close_gripper()
            elif self.gripper == "ctek":
                if not gripper_cmd:
                    self.open_gripper()
                else:
                    self.close_gripper()
            else:
                raise NotImplementedError(f"Gripper {self.gripper} is not implemented.")
        else:
            raise RuntimeError("No gripper equipped in Franka. move_gripper should not work.")
             
    def move(self, target_cmd):
        self.move_joint_ik(target_cmd[:7])
        if self.with_gripper and len(target_cmd) > 7:
            self.move_gripper(target_cmd[7])
        
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
        if self.with_gripper:
            self.open_gripper()

    def shutdown(self):
        """
        Clean shutdown of robot controller.

        Returns:
            None
        """
        # TODO: shutdown the robot
        pass

    def reset(self):
        """
        This function is used to reset the robot to the home position from the frankapy.
        """
        self.arm.reset_joints()

if __name__ == "__main__":
    try:
        controller = FrankaEnvWrapper()
        # controller.run()
        # TODO: run the robot, and test at local side
        i = 1
        while i < 100:
            target_pose = controller.get_tcp_position()
            solved_joint = controller.solve_ik(target_pose)
            solved_pose = controller.ik_solver.compute_fk(solved_joint)
            print(f"----------------i={i}--------------------------------")
            print("original_joint:", controller.get_arm_position())
            print("solved_joint:", solved_joint)
            print("target_pose:", target_pose)
            print("solved pose:", solved_pose)
            print("current pose:", controller.ik_solver.compute_fk(controller.get_arm_position()))
            print("------------------------------------------------------")
            i += 1

    except rospy.ROSInterruptException:
        pass
    finally:
        controller.shutdown()
