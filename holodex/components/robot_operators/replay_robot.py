import os
from typing import Tuple, List, Any

import rospy
import cv2
import numpy as np
from holodex.constants import SLEEP_TIME
from holodex.components.robot_operators.robot import RobotController
from holodex.robot.hand.leap.leap_kdl import LeapKDL
import pyrealsense2 as rs


class ReplayController(RobotController):
    def __init__(self, configs, hand_state=None, arm_state=None) -> None:
        super().__init__(teleop=False, servo_mode=True, arm_control_mode="joint")
        self.configs = configs
        self.arm_joint_positions = arm_state
        self.hand_joint_positions = hand_state

        # camera setup for image capture -> robot_camera.yaml
        # TODO: refactor this part to use the RealSenseRobotStream class
        self.save_dir = (
            configs.relpay_image_path
        )  # "/home/agibot/Projects/Real-Robo/replay_data"
        self.cam_serial_num = "211422061450"
        self.num_cams = 1

        os.makedirs(self.save_dir, exist_ok=True)

        self.pipeline = None
        self.start_camera_stream()

    def replay_hand_motion(self, n: int = 20) -> None:
        for i in range(len(self.hand_joint_positions) - 1):
            start_pos = self.hand_joint_positions[i]
            end_pos = self.hand_joint_positions[i + 1]
            for step in range(1, n + 1):
                interpolated_pos = []
                for j in range(len(start_pos)):
                    interpolated_joint_position = (
                        start_pos[j] + (end_pos[j] - start_pos[j]) * step / n
                    )
                    interpolated_pos.append(interpolated_joint_position)
                self.move_hand(interpolated_pos)
        self.move_hand(self.hand_joint_positions[-1])

    def replay_arm_motion(self, n: int = 20) -> None:
        for i in range(len(self.arm_joint_positions) - 1):
            # arm interpolation
            start_arm_pos = self.arm_joint_positions[i]
            end_arm_pos = self.arm_joint_positions[i + 1]

            for step in range(1, n + 1):
                interpolated_arm_pos = []
                for j in range(len(start_arm_pos)):
                    interpolated_joint_position = (
                        start_arm_pos[j]
                        + (end_arm_pos[j] - start_arm_pos[j]) * step / n
                    )
                    interpolated_arm_pos.append(interpolated_joint_position)

                self.move_arm(interpolated_arm_pos)
        print("Replay complete!")

    def replay_arm_and_hand_motion(self, n_interpolations: int = 30) -> None:
        assert len(self.hand_joint_positions) == len(
            self.arm_joint_positions
        ), "Hand and arm data length mismatch"

        for i in range(len(self.hand_joint_positions) - 1):
            # hand interpolation
            start_hand_pos = np.array(self.hand_joint_positions[i])
            end_hand_pos = np.array(self.hand_joint_positions[i + 1])

            # arm interpolation
            start_arm_pos = np.array(self.arm_joint_positions[i])
            end_arm_pos = np.array(self.arm_joint_positions[i + 1])

            # make the robot move
            for step in range(1, n_interpolations + 1):
                interpolated_hand_pos = (
                    start_hand_pos
                    + (end_hand_pos - start_hand_pos) * step / n_interpolations
                )
                interpolated_arm_pos = (
                    start_arm_pos
                    + (end_arm_pos - start_arm_pos) * step / n_interpolations
                )

                self.move_arm(interpolated_arm_pos)
                self.move_hand(interpolated_hand_pos)
                rospy.sleep(SLEEP_TIME)

            new_arm_pos = self.get_arm_position()
            new_hand_pos = self.get_hand_position()

            # compute difference between current and target position
            arm_diff = np.abs(new_arm_pos - self.arm_joint_positions[i + 1])
            hand_diff = np.abs(new_hand_pos - self.hand_joint_positions[i + 1])
            print(f"Arm diff: {arm_diff}, Hand diff: {hand_diff}")

            self.capture_save_images(i + 1)
        self.stop_camera_stream()
        print("Replay complete!")

    def start_camera_stream(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.cam_serial_num)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.pipeline.start(config)

    def capture_save_images(self, i: int) -> None:
        try:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            color_image_path = os.path.join(self.save_dir, f"replay_image_{i}.png")
            cv2.imwrite(color_image_path, color_image)
            print(f"Color image saved at {color_image_path}")
        except Exception as e:
            print(f"Failed to capture and save image: {e}")

    def stop_camera_stream(self):
        if self.pipeline:
            self.pipeline.stop()
            print("Camera stream stopped.")


class DataProcessor(object):
    def __init__(self, configs) -> None:
        self.arm_state_path = configs.data_path
        self.hand_state_path = configs.data_path
        self.kdl_solver = LeapKDL()

    # TODO 1: read data from the extracted data or filtered data
    # TODO 2: optimize the saving format.
    def load_data(self) -> Tuple[List[List[Any]], List[List[Any]]]:
        """Loads arm and hand state data."""
        arm_data, hand_data = [], []
        demo_list = os.listdir(self.arm_state_path)
        demo_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

        for file_name in demo_list:
            file_path = os.path.join(self.arm_state_path, file_name)
            state = np.load(file_path, allow_pickle=True)

            arm_joint_position = list(state["arm_joint_positions"])
            hand_joint_position = list(state["hand_joint_positions"])

            arm_data.append(arm_joint_position)
            hand_data.append(hand_joint_position)

        return arm_data, hand_data
