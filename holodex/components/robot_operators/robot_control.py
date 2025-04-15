import rospy
import numpy as np
from holodex.constants import *
import warnings
import time
from termcolor import cprint

warnings.filterwarnings(
    "ignore",
    message="Link .* is of type 'fixed' but set as active in the active_links_mask.*",
)

if ARM_TYPE is not None:
    # Load module according to arm type
    arm_module = __import__("holodex.robot.arm")
    Arm_module_name = f"{ARM_TYPE}Arm"
    Arm = getattr(arm_module.robot, Arm_module_name)

class RobotController(object):
    def __init__(
        self,
        teleop,
        servo_mode=True,
        arm_control_mode="ik",
        hand_control_mode="joint",
        home=True,
        random_arm_home=False,
    ) -> None:
        if ARM_TYPE == "Flexiv":
            self.arm = Arm()
            cprint("Call Flexiv Arm", "red")
        elif ARM_TYPE == "Franka":
            from holodex.robot.arm.franka.franka_env_wrapper import FrankaEnvWrapper
            self.arm = FrankaEnvWrapper(control_mode="joint", teleop=teleop)
            cprint("Call Franka Arm", "red")
        else:
            self.arm = (
                Arm(
                    servo_mode=servo_mode,
                    teleop=teleop,
                    control_mode=arm_control_mode,
                    safety_moving_trans=JAKA_SAFE_MOVING_TRANS,
                    random_jaka_home=random_arm_home,
                )
                if ARM_TYPE is not None
                else None
            )

        self.arm_control_mode = arm_control_mode
        cprint(f"self.arm_control_mode: {self.arm_control_mode}", "red")
        self.teleop = teleop

        self.home = home
        if self.home:
            self.home_robot()

    def home_robot(self):
        # Implementation placeholder
        pass

    def reset_robot(self):
        if ARM_TYPE is not None:
            self.arm.reset()

    def get_arm_position(self):
        return self.arm.get_arm_position()

    def get_arm_velocity(self):
        return self.arm.get_arm_velocity()

    def get_arm_tcp_position(self):
        return self.arm.get_tcp_position()

    def get_arm_torque(self):
        return self.arm.get_arm_torque()

    def get_hand_position(self):
        return self.hand.get_hand_position()

    def get_hand_velocity(self):
        return self.hand.get_hand_velocity()

    def get_hand_torque(self):
        return self.hand.get_hand_torque()

    def move_hand(self, input_angles):
        if "LEAP" in HAND_TYPE.upper() and self.teleop:
            input_angles = np.array(input_angles)[
                [1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15]
            ]
        self.hand.move(input_angles)

    def move_arm(self, input_angles: np.ndarray):
        self.arm.move_joint(input_angles)
        rospy.sleep(SLEEP_TIME)

    def move_arm_and_hand(self, input_angles):
        assert (
            self.arm is not None and self.hand is not None
        ), "Arm and hand are not initialized"
        self.move_arm(input_angles[: self.arm.dof])
        self.move_hand(input_angles[self.arm.dof :])

    def move(self, input_angles):
        print(f"input_angles: {input_angles}")
        if self.arm is not None:
            self.move_arm(input_angles[: self.arm.dof])
            if self.hand is not None:
                self.move_hand(input_angles[self.arm.dof :])
        elif self.hand is not None:
            self.move_hand(input_angles)
            rospy.sleep(SLEEP_TIME)

    def move_seperate(self, action: dict):
        if self.arm is not None:
            self.move_arm(action["arm"])
        if self.hand is not None:
            self.move_hand(action["hand"])
        rospy.sleep(1/5)
        
    def move_gripper(self, gripper_cmd):
        """
        Control gripper for teleoperation with binary open/close command.
        Includes debouncing to avoid too frequent control commands.
        
        Args:
            gripper_cmd (float or int): Binary command for gripper
                - Values <= 0.05: Close the gripper
                - Values > 0.05: Open the gripper
        """
        # Initialize state tracking if not already set
        if not hasattr(self, '_gripper_state'):
            self._gripper_state = None
            
        # Debounce logic - only send commands when state actually changes
        if float(gripper_cmd) > 0.05:
            # Open gripper command
            if self._gripper_state != 'open':
                self.arm.open_gripper()
                self._gripper_state = 'open'
        else:
            # Close gripper command
            if self._gripper_state != 'closed':
                self.arm.close_gripper()
                self._gripper_state = 'closed'

    def get_gripper_state(self):
        return self.arm.get_gripper_is_grasped()


import sensor_msgs

class RobotControlNode:
    def __init__(self) -> None:
        self.robot = RobotController(teleop=False)
        self.joint_position_commands = [
            -1.5707487,
            0.24192421,
            -1.4037328,
            0.02739489,
            -1.8208425,
            -2.1729174,
        ]
        rospy.Timer(rospy.Duration(0.1), self._main_loop)
        self.publisher = rospy.Publisher(
            "/holodex/joint_states", sensor_msgs.msg.JointState, queue_size=10
        )
        rospy.Subscriber(
            "/holodex/joint_commands",
            sensor_msgs.msg.JointState,
            self._joint_commands_callback,
        )
        rospy.spin()

    def _main_loop(self, event):
        # Publish joint states
        arm_position = self.robot.get_arm_position()
        js = sensor_msgs.msg.JointState()
        js.header.stamp = rospy.Time.now()
        js.header.frame_id = "link_0"
        js.position = arm_position
        js.name = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        self.publisher.publish(js)

        # Send joint commands
        self.robot.arm.move_joint(self.joint_position_commands)

    def _joint_commands_callback(self, msg):
        self.joint_position_commands = msg.position


if __name__ == "__main__":
    rospy.init_node("robot_control")
    robot = RobotController(teleop=False, random_arm_home=False, home=True)
