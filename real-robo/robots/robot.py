import time
import argparse
from pathlib import Path
import numpy as np
import time
import sys

sys.path.append("dependencies/deoxys_control_research3/deoxys")


from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse
from real_robo_logger import get_real_robo_logger


class RobotController(object):
    def __init__(
        self,
        interface="teleop",
        arm_type="franka",
        arm_config=None,
        simulator=None,
        gripper_type=None,
        gripper_init_state="open",
    ) -> None:
        self.interface = interface
        self.arm_type = arm_type
        self.arm_config = arm_config
        self.simulator = simulator
        self.gripper_type = gripper_type
        self.gripper_state = gripper_init_state
        self.logger = get_real_robo_logger(__name__)

        # initilize the franka (deoxys)

        


      
    def reset(self):
        """Reset the robot to a safe state."""
        pass

    def move_arm(self, position):
        """Move the robot arm to a target position."""
        pass

    def move_manipulator(self, position):
        """Move the end effector/manipulator."""
        pass

    ##### Arm methods #####
    def get_arm_position(self):
        """Get current arm position."""
        self.logger.debug("Fetching arm position.")
        return [0.0, 0.0, 0.0]

    def get_arm_tcp_position(self):
        """Get current arm tcp position."""
        self.logger.debug("Fetching arm TCP position.")
        return [0.0, 0.0, 0.0]

    def get_arm_joint_positions(self):
        """Get current joint positions."""
        self.logger.debug("Fetching arm joint positions.")
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def get_arm_torques(self):
        """Get current joint torques."""
        self.logger.debug("Fetching arm joint torques.")
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    ##### Gripper methods #####
    def open_gripper(self):
        """Open the gripper."""
        self.logger.info("Opening gripper.")
        self.gripper_state = "open"
        # ...actual open code...

    def close_gripper(self):
        """Close the gripper."""
        self.logger.info("Closing gripper.")
        self.gripper_state = "closed"
        # ...actual close code...

    def get_gripper_state(self):
        """Get current gripper state."""
        self.logger.debug("Fetching gripper state.")
        return self.gripper_state

    def get_status(self):
        """Get overall robot status (errors, health, etc.)."""
        self.logger.info("Fetching robot status.")
        return {"status": "ok", "errors": []}

    def get_full_state(self):
        """Get full robot state (arm, gripper, sensors, etc.)."""
        self.logger.info("Fetching full robot state.")
        return {
            "arm_position": self.get_arm_position(),
            "arm_tcp_position": self.get_arm_tcp_position(),
            "arm_joint_positions": self.get_arm_joint_positions(),
            "arm_torques": self.get_arm_torques(),
            "gripper_state": self.get_gripper_state(),
            "status": self.get_status(),
        }

    def emergency_stop(self):
        """Trigger an emergency stop."""
        self.logger.critical("EMERGENCY STOP TRIGGERED!")
        # ...actual emergency stop code...

if __name__ == "__main__":
    robot = RobotController()
    robot.reset()
    robot.move_arm([0.5, 0.0, 0.5])
    robot.move_manipulator([0.5, 0.0, 0.5])
    robot.open_gripper()
    robot.close_gripper()
    robot.emergency_stop()
    action = {"arm": [0.5, 0.0, 0.5], "hand": [0.0]*16}
    robot.move_arm(action["arm"])
    robot.move_manipulator(action["hand"])
    robot.get_gripper_state()
