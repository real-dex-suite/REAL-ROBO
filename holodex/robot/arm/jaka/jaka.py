import numpy as np
import rospy
from holodex.constants import (
    JAKA_IP,
    JAKA_POSITIONS,
    JAKA_DOF,
    JAKA_JOINT_STATE_TOPIC,
    JAKA_COMMANDED_JOINT_STATE_TOPIC,
    JAKA_EE_POSE_TOPIC,
)
from holodex.utils.network import JointStatePublisher, FloatArrayPublisher
import random
from termcolor import cprint


class JakaArm(object):
    def __init__(
        self,
        servo_mode=True,
        control_mode="ik",
        teleop=False,
        safety_moving_trans=100,
        random_jaka_home=False,
        gripper=None,
    ):
        # rospy.init_node('jaka_arm_controller')

        # Creating ROS Publishers
        self.joint_state_publisher = JointStatePublisher(
            publisher_name=JAKA_JOINT_STATE_TOPIC
        )
        self.command_joint_state_publisher = JointStatePublisher(
            publisher_name=JAKA_COMMANDED_JOINT_STATE_TOPIC
        )
        self.ee_pose_publisher = FloatArrayPublisher(publisher_name=JAKA_EE_POSE_TOPIC)
        
        from jkrc import jkrc
        
        self.robot = jkrc.RC(JAKA_IP)
        ret = self.robot.login()
        print("Robot logging: ", ret)

        ret = self.robot.power_on()
        print("Robot power on: ", ret)
        # if has collision, recover from collision
        success, collision = self.robot.is_in_collision()
        if collision:
            self.robot.collision_recover()
            self.robot.enable_robot()
        ret = self.robot.enable_robot()
        print("Robot enable: ", ret)
        self.jaka_joint_state = None
        # TODO change to ros?
        # rospy.Subscriber(KINOVA_JOINT_STATE_TOPIC, JointState, self._callback_joint_state, queue_size = 1)
        self.teleop = teleop

        self.move_mode = 0
        self.is_block = True
        self.speed = 10
        self.acc = 5
        self.tol = 0.01
        self.dof = JAKA_DOF

        self.with_gripper = gripper is not None
        if self.with_gripper:
            self.dof += 1
            
        self.safety_moving_trans = safety_moving_trans
        self.joint_vel_limit = 0.5  # TODO configureable
        self.joint_pos_limit = np.array([6.28, 2.09, 2.27, 6.28, 2.09, 6.28])

        self.servo_mode = servo_mode
        self.control_mode = control_mode
        if self.teleop:
            assert self.control_mode == "ik"

        self.robot.servo_move_enable(self.servo_mode)

        self.random_jaka_home = random_jaka_home

        # self.home_robot()

    def _callback_joint_state(self):
        self.jaka_joint_state = self.robot.get_joint_position()[1]

    def get_arm_position(self):
        # if self.jaka_joint_state is None:
        #     return None
        self._callback_joint_state()
        return np.array(self.jaka_joint_state, dtype=np.float32)

    def get_tcp_position(self):
        output = self.robot.get_tcp_position()
        # while len(output) == 1:
        #     output = self.robot.get_tcp_position()
        if len(output) == 1:
            print(output)
        return np.array(output[1])
    
    def get_gripper_position(self):
        raise NotImplementedError("JAKA with gripper is not implemented now.")
    
    def open_gripper(self):
        raise NotImplementedError("JAKA with gripper is not implemented now.")
    
    def close_gripper(self):
        raise NotImplementedError("JAKA with gripper is not implemented now.")
    
    def move_gripper(self, gripper_cmd):
        """
        Control gripper for teleoperation with binary open/close command.
        Includes debouncing to avoid too frequent control commands.
        
        Args:
            gripper_cmd (float or int): Binary command for gripper
                - Values <= 0.05: Close the gripper
                - Values > 0.05: Open the gripper
                
        """
        raise NotImplementedError("JAKA with gripper is not implemented now.")
    
    def set_tcp_position(self, input_pose):
        self.robot.linear_move_extend(
            input_pose, self.move_mode, self.is_block, self.speed, self.acc, self.tol
        )

    def home_robot(self):
        # self.move_joint(JAKA_POSITIONS['home'])
        tcp_home = JAKA_POSITIONS["tcp_home"]
        if self.random_jaka_home:
            tcp_home[0] += random.uniform(-40, 40)
            tcp_home[1] += random.uniform(-40, 0)
            tcp_home[2] += random.uniform(-10, 10)

        home_joint = self.compute_joint(tcp_home)
        self.move_joint(home_joint)

    def reset(self):
        self.move_joint(JAKA_POSITIONS["home"])

    def move_joint(self, input_angles):
        self.robot.joint_move(input_angles, self.move_mode, self.is_block, self.speed)

    def safety_check(self, target_arm_pose):
        current_arm_pose = self.get_tcp_position()

        if np.any(
            np.abs(target_arm_pose[:3] - current_arm_pose[:3])
            > self.safety_moving_trans
        ):
            cprint(
                "Target position is too far from current, arm will not moving", "red"
            )
            self.robot.motion_abort()
            return current_arm_pose
        else:
            return target_arm_pose

    def limit_joint_vel(self, target_joint):
        current_joint = self.get_arm_position()
        assert (-self.joint_vel_limit < target_joint - current_joint).any() and (
            target_joint - current_joint < self.joint_vel_limit
        ).any(), "Joint velocity limit exceeded"
        target_joint = current_joint + np.clip(
            target_joint - current_joint, -self.joint_vel_limit, self.joint_vel_limit
        )
        return target_joint

    def compute_ik(self, current_joint, cart_pose):
        joint_tuple = self.robot.kine_inverse(current_joint, cart_pose)
        if len(joint_tuple) > 1:
            return joint_tuple[1]
        else:
            print(joint_tuple)
            print("Inverse kinematics failed, arm will not moving")
            return current_joint

    def compute_joint(self, cart_pose):
        current_joint = self.get_arm_position()
        joint_tuple = self.robot.kine_inverse(current_joint, cart_pose)
        if len(joint_tuple) > 1:
            return joint_tuple[1]
        else:
            print(joint_tuple)
            print("Inverse kinematics failed, arm will not moving")
            return current_joint

    def limit_joint_pos(self, target_joint):
        target_joint = np.clip(
            target_joint, -self.joint_pos_limit, self.joint_pos_limit
        )
        return target_joint

    def publish_state(self, input_cmd=None):
        current_joint = self.get_arm_position()
        self.joint_state_publisher.publish(current_joint)

        current_ee_pose = self.get_tcp_position()
        self.ee_pose_publisher.publish(current_ee_pose)

        if input_cmd is not None:
            self.command_joint_state_publisher.publish(input_cmd)

    def move(self, input_cmd):
        if self.teleop:
            input_cmd = self.safety_check(input_cmd)
        # TODO add pose command
        if self.servo_mode:
            self.robot.servo_move_enable(True)
            if self.control_mode == "ik":
                input_cmd = self.compute_joint(input_cmd)
                input_cmd = self.limit_joint_vel(input_cmd)
                input_cmd = self.limit_joint_pos(input_cmd)

            if self.teleop:
                input_cmd = self.limit_joint_vel(input_cmd)
                input_cmd = self.limit_joint_pos(input_cmd)

                self.publish_state(input_cmd)  # TODO maybe change to use ros
            else:
                input_cmd = self.limit_joint_vel(input_cmd)
                input_cmd = self.limit_joint_pos(input_cmd)

            self.robot.servo_j(input_cmd, self.move_mode)
        else:
            self.robot.joint_move(input_cmd, self.move_mode, self.is_block, self.speed)


if __name__ == "__main__":
    # init ros node
    rospy.init_node("jaka_arm_controller")
    jaka = JakaArm()
    current_tcp_position = jaka.get_tcp_position()
    import time

    while True:
        print("target:", current_tcp_position)
        # jaka.move(current_tcp_position)
        jaka.publish_state()
