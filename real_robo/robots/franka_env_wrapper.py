#!/usr/bin/env python3
"""
FrankaEnvWrapper: A High-Level Interface for Franka Robot Control

Author: Jinzhou Li

This module provides a convenient wrapper around the Deoxys FrankaInterface,
enabling simplified and robust interaction with Franka robots for research and development.
It offers user-friendly methods for querying robot state, controlling motion, and accessing
robot data, while handling initialization and connection management.

Adapted from: deoxys_control_research3 (https://github.com/UT-Austin-RPL/deoxys_control)

Additional helper functions and safety checks are included to streamline common robotics workflows.
"""

from __future__ import annotations
import time
import logging
import time
import sys
import numpy as np
from typing import Optional

# Add deoxys to path
from real_robo.robots.deoxys_imports import setup_deoxys_path
setup_deoxys_path()

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.config_utils import get_default_controller_config
from real_robo.robots.real_robo_logger import get_real_robo_logger
logger = get_real_robo_logger("demo", console_level=logging.DEBUG)


class FrankaEnvWrapper:
    def __init__(
        self,
        controller_type: str = "JOINT_POSITION",
        interface_cfg: str = "charmander.yml",
        controller_cfg: Optional[dict] = 'joint-position-controller.yml',
    
    ) -> None:
        
        # initilize deoxys
        self.robot_interface =FrankaInterface(config_root + f"/{interface_cfg}", use_visualizer=False)
        self.controller_cfg = YamlConfig(config_root + f"/{controller_cfg}").as_easydict()
        self.controller_type = controller_type

        # self.controller_cfg = get_default_controller_config(controller_type=controller_type)
        # Wait for the robot to be ready
        self._wait_for_robot_state()


    def _wait_for_robot_state(self, *, timeout: float = 5.0, poll_interval: float = 0.05) -> None:
        """Poll until the interface reports the first state or raise a timeout."""
        deadline = time.time() + timeout
        while not getattr(self.robot_interface, "received_states", False):
            if time.time() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for FrankaInterface state updates; verify robot connection."
                )
            time.sleep(poll_interval)
        elapsed = timeout - (deadline - time.time())
        logger.info("Robot state received, FrankaEnvWrapper is ready to use (waited %.2f seconds)", elapsed)


    def _get_last_state(self) -> dict:
        """Get the current raw state of the robot.

        Returns:
            dict: A dictionary containing the robot's state information.
        """
        robot_state = {}
        state = self.robot_interface._state_buffer[-1]
        for field, value in state.ListFields():
            robot_state[field.name] = value

        return robot_state
    

    def get_ee_pose(self) -> np.ndarray:
        """Get the current end-effector pose.

        Returns:
            np.ndarray: A 7-dimensional array containing the end-effector position and orientation (quaternion).
        """
        return self.robot_interface.last_eef_quat_and_pos
    

    def get_ee_T(self) -> np.ndarray:
        """Get the current end-effector pose as a 4x4 transformation matrix.

        Returns:
            np.ndarray: A 4x4 transformation matrix representing the end-effector pose.
        """
        return self.robot_interface.last_pose
    
    
    def get_joint_positions(self) -> np.ndarray:
        """Get the current joint positions.

        Returns:
            np.ndarray: A 7-dimensional array containing the joint positions.
        """
        return self.robot_interface.last_q
    

    def get_joint_velocities(self) -> np.ndarray:
        """Get the current joint velocities.

        Returns:
            np.ndarray: A 7-dimensional array containing the joint velocities.
        """
        return self.robot_interface.last_dq
    

    def get_joint_external_torques(self) -> np.ndarray:
        """Get the current joint torques.

        Returns:
            np.ndarray: A 7-dimensional array containing the joint torques.
        """
        # return self.robot_interface.last_tau_ext
        return self._get_last_state()['tau_ext_hat_filtered']


    def get_ee_wrench(self) -> np.ndarray:
        """Get the current end-effector wrench eastimated from joint torques.

        Returns:
            np.ndarray: A 6-dimensional array containing the end-effector wrench (force and torque).
        """
        return self._get_last_state()['O_F_ext_hat_K']


    def get_base_wrench(self) -> np.ndarray:
        """Get the current base wrench eastimated from joint torques.

        Returns:
            np.ndarray: A 6-dimensional array containing the base wrench (force and torque).
        """
        return self._get_last_state()['K_F_ext_hat_K']


    def get_tau_J(self) -> np.ndarray:
        """Get the current joint torques.

        Returns:
            np.ndarray: A 7-dimensional array containing the joint torques.
        """
        return self._get_last_state()['tau_J']
    
    
    ########################################################################
    ########################## control the robot ###########################
    ########################################################################

    #TODO 1: add safety check, if two cmd are too far away, don't do it. Which also important for VR teleop!
    def move_ee_pose(self, pose: np.ndarray) -> None:
        """Move the robot to the target end-effector pose.
        Args:
            pose (np.ndarray): A 7-dimensional array containing the target end-effector position and orientation (quaternion).
        """
        self.robot_interface.control(
            controller_type="CARTESIAN_VELOCITY",
            action=pose,
            controller_cfg=self.controller_cfg,
        )


    def move_joint_positions(self, joint_positions_cmd: np.ndarray) -> None:
        """Move the robot to the target joint positions.
        Args:
            joint_positions (np.ndarray): A 7-dimensional array containing the target joint positions.
        """
        
        #Do we need blocking here until the robot reaches the target position?
        self.robot_interface.control(
            controller_type="JOINT_POSITION",
            action=joint_positions_cmd,
            controller_cfg=self.controller_cfg,
        )

    #TODO 2: torque control


if __name__ == "__main__":
    robot = FrankaEnvWrapper()
    while True:
        # print(robot.get_joint_positions())
        print(robot.get_ee_pose())
        time.sleep(0.1)
