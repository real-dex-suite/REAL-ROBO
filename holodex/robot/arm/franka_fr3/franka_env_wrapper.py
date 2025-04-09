#!/usr/bin/env python

import rospy
import numpy as np
import roslib

roslib.load_manifest("franka_interface_msgs")
from std_msgs.msg import Float64MultiArray
from frankapy import FrankaArm, SensorDataMessageType, FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointPositionSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
from holodex.robot.arm.franka_fr3.kinematics_solver import FrankaSolver


class FrankaEnvWrapper:
    def __init__(self):
        self.arm = FrankaArm("franka_arm_reader")
        rospy.loginfo("Initializing FrankaWrapper...")

        self._initialize_state()
        self._initialize_arm()
        self.pub = rospy.Publisher(
            FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000
        )
        self.ik_solver = FrankaSolver()

    def _initialize_state(self):
        self.current_joint_state = self.arm.get_joints()
        self.joint_state = self.current_joint_state[:]
        self.current_ee_pose = self.arm.get_pose()
        self.ee_pose = self.current_ee_pose

    def _initialize_arm(self):
        # This configuration is calibrated between the real robot and simulation parameters
        self.arm.goto_joints(
            self.current_joint_state,
            duration=5,
            k_gains=[100.0, 100.0, 150.0, 400.0, 400.0, 600.0, 80.0],
            d_gains=[30.0, 30.0, 40.0, 150.0, 100.0, 30.0, 15.0],
            dynamic=True,
            buffer_time=10,
        )

    def get_arm_position(self):
        self.current_joint_state = self.arm.get_joints()
        return self.current_joint_state

    def ee2joint(self, ee_pose):
        return self.ik_solver.solve_ik(ee_pose[:3], ee_pose[3:])

    def get_transformation_matrix(self):
        trans = self.current_ee_pose.translation
        rotation_mat = self.current_ee_pose.rotation
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_mat
        transformation_matrix[:3, 3] = trans

        self.arm.inverse_kinematics(transformation_matrix)
        return transformation_matrix

    def open_gripper(self):
        self.arm.open_gripper()

    def close_gripper(self):
        self.arm.close_gripper()

    def move_joint(self, target_joint):
        init_time = rospy.Time.now().to_time()
        traj_gen_proto_msg = JointPositionSensorMessage(
            id=0, timestamp=rospy.Time.now().to_time() - init_time, joints=target_joint
        )
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION
            )
        )
        rospy.loginfo("Publishing: ID {}".format(traj_gen_proto_msg.id))
        self.pub.publish(ros_msg)

    def get_tcp_position(self):
        self.current_ee_pose = self.arm.get_pose()
        trans = self.current_ee_pose.translation
        rot_quat = self.current_ee_pose.quaternion
        ee_pose = np.concatenate([trans, rot_quat])
        return ee_pose

    def run(self):
        rate = rospy.Rate(10)  # test in 10 Hz
        for i in range(11):
            if rospy.is_shutdown():
                break
            x = self.get_arm_position()
            x[6] += 0.1
            self.move_joint(x)
            print(self.current_joint_state)
            rate.sleep()
        self.open_gripper()

    def shutdown(self):
        pass


if __name__ == "__main__":
    try:
        controller = FrankaWrapper()
        controller.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        controller.shutdown()
