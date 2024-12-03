import os
from typing import Tuple, List, Any

import rospy
import cv2
import numpy as np
from holodex.constants import SLEEP_TIME
from holodex.components.robot_operators.robot import RobotController
from holodex.robot.hand.leap.leap_kdl import LeapKDL
import pyrealsense2 as rs
import torch


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
        # self.cam_serial_num = "211422061450" # 415
        self.cam_serial_num = "f1230963"  # 515
        self.num_cams = 1

        os.makedirs(self.save_dir, exist_ok=True)

        self.pipeline = None
        self.start_camera_stream()
        print("camera stream started")

    def replay_hand_motion(self, n: int = 1) -> None:
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

    def replay_arm_motion_tcp(self):
        for i in range(len(self.arm_joint_positions)):
            self.move_arm(self.arm_joint_positions[i])

        print("Replay complete!")

    def replay_arm_motion(self, n: int = 1) -> None:
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

    def replay_arm_and_hand_motion(self, n_interpolations: int = 1) -> None:
        print(len(self.hand_joint_positions), len(
            self.arm_joint_positions
        ))
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
            # print(end_arm_pos)
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
            # print("euler:", self.get_arm_tcp_position()[3:])
            # print(
            #     "quaternion",
            #     self.arm.robot.rot_matrix_to_quaternion(
            #         self.arm.robot.rpy_to_rot_matrix(self.get_arm_tcp_position()[3:])[1]
            #     )[1],
            # )
            # compute difference between current and target position
            arm_diff = np.abs(new_arm_pos - self.arm_joint_positions[i + 1])
            hand_diff = np.abs(new_hand_pos - self.hand_joint_positions[i + 1])
            # print(f"Arm diff: {arm_diff}, Hand diff: {hand_diff}")

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
        self.data_path = configs.data_path
        self.extract_path = configs.extract_path
        self.kdl_solver = LeapKDL()

    def load_data(self, data_type: str) -> Tuple[List[List[Any]], List[List[Any]]]:
        """Loads arm and hand data based on data_type."""
        if data_type == "extracted":
            return self._load_extracted_data()
        elif data_type == "raw":
            return self._load_raw_data()
        else:
            raise ValueError(
                f"Invalid data type: {data_type}. Choose 'extracted' or 'raw'."
            )

    def _load_extracted_data(self) -> Tuple[List[List[Any]], List[List[Any]]]:
        """Loads extracted arm and hand joint position data."""
        if not os.path.exists(self.extract_path):
            raise FileNotFoundError(f"File {self.extract_path} does not exist.")

        data = torch.load(self.extract_path)
        from termcolor import cprint
        # arm_positions = [pos.tolist() for pos in data["arm_cmd_abs_joint"]]
        arm_positions = [pos.tolist() for pos in data["arm_cmd_ee_pose"]]
        hand_positions = [pos.tolist() for pos in data["hand_abs_joint"]]

        return arm_positions, hand_positions

    def _load_raw_data(self) -> Tuple[List[List[Any]], List[List[Any]]]:
        """Loads raw arm and hand state data."""
        arm_data, hand_data = [], []
        demo_list = sorted(
            os.listdir(self.data_path),
            key=lambda f: int("".join(filter(str.isdigit, f))),
        )

        for file_name in demo_list:
            file_path = os.path.join(self.data_path, file_name)
            state = np.load(file_path, allow_pickle=True)

            arm_data.append(list(state["arm_cmd_abs_joint"]))
            hand_data.append(list(state["hand_joint_positions"]))

        return arm_data, hand_data
