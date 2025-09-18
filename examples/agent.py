"""Agent utilities for gathering batched observations from RealSense and Franka."""

# TODO: add gopro support

import time
from collections import deque
from typing import Deque, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np

from real_robo.device.camera.realsense import RealSenseRGBDCamera
from real_robo.robots.franka_env_wrapper import RobotController


DEFAULT_WARMUP_FRAMES = 30
DEFAULT_CROP_SIZE = (480, 480)
DEFAULT_CROP_OFFSET = (0, -200)
DEFAULT_OUTPUT_SIZE = (224, 224)
DEFAULT_BLOCKING_SLEEP = 0.1


class Agent:
    """Evaluation agent that buffers recent camera frames and end-effector poses."""

    def __init__(self, camera_serial: str, obs_num: int, **kwargs) -> None:
        self.camera_serial = camera_serial
        self.obs_num = obs_num

        self.warmup_frames: int = kwargs.pop("warmup_frames", DEFAULT_WARMUP_FRAMES)
        self.crop_size: Tuple[int, int] = kwargs.pop("crop_size", DEFAULT_CROP_SIZE)
        self.crop_offset: Tuple[int, int] = kwargs.pop("crop_offset", DEFAULT_CROP_OFFSET)
        self.output_size: Tuple[int, int] = kwargs.pop("output_size", DEFAULT_OUTPUT_SIZE)
        self.blocking_sleep: float = kwargs.pop("blocking_sleep", DEFAULT_BLOCKING_SLEEP)

        camera: Optional[RealSenseRGBDCamera] = kwargs.pop("camera", None)
        robot: Optional[RobotController] = kwargs.pop("robot", None)
        teleop: bool = kwargs.pop("teleop", True)

        # Allow callers to stash additional config without failing fast.
        self.extra_config = kwargs

        self.rgb_buffer: Deque[np.ndarray] = deque(maxlen=obs_num)
        self.pose_buffer: Deque[np.ndarray] = deque(maxlen=obs_num)

        self.camera = camera or RealSenseRGBDCamera(serial=self.camera_serial)
        self.robot = robot or RobotController(teleop=teleop)

        self._warmup_camera()
        self._fill_initial_buffer()

    def _warmup_camera(self) -> None:
        """Capture a few frames so the auto-exposure converges before logging data."""
        for _ in range(self.warmup_frames):
            self.camera.get_rgbd_image()

    def _fill_initial_buffer(self) -> None:
        """Grab the first ``obs_num`` observations so the buffers start full."""
        for _ in range(self.obs_num):
            self._add_single_observation()

    def _add_single_observation(self) -> None:
        """Capture and buffer one frame/pose pair."""
        colors, _ = self.camera.get_rgbd_image()
        processed_img = self._process_image(colors)
        pose = np.asarray(self.robot.get_arm_tcp_position(), dtype=float)

        self.rgb_buffer.append(processed_img)
        self.pose_buffer.append(pose[:3])

    def _process_image(self, colors: np.ndarray) -> np.ndarray:
        """
        Center-crop to the configured ``crop_size`` and resize to ``output_size``.

        The RealSense frames are larger than the network input; we focus on the
        workspace by cropping and performing a final color conversion for PyTorch.
        """

        crop_w, crop_h = self.crop_size
        out_w, out_h = self.output_size
        offset_x, offset_y = self.crop_offset

        height, width = colors.shape[:2]
        start_x = (width - crop_w) // 2 + offset_x
        start_y = (height - crop_h) // 2 + offset_y

        start_x = max(0, start_x)
        start_y = max(0, start_y)

        end_x = min(width, start_x + crop_w)
        end_y = min(height, start_y + crop_h)

        # Keep the crop anchored even when the desired window exceeds the frame.
        start_x = max(0, end_x - crop_w)
        start_y = max(0, end_y - crop_h)

        cropped = colors[start_y:end_y, start_x:end_x]
        resized = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    def get_observation(self) -> Dict[str, np.ndarray]:
        """Return stacked camera frames and poses after appending a fresh sample."""
        self._add_single_observation()

        return {
            "camera_0": np.stack(self.rgb_buffer),
            "robot_eef_pose": np.stack(self.pose_buffer),
        }

    def set_tcp_pose(self, pose: Iterable[float], blocking: bool = False) -> None:
        """Command the robot TCP pose, optionally blocking for a short dwell."""
        self.robot.move_arm(list(pose))
        if blocking:
            # Brief dwell ensures we provide time for the controller to react.
            time.sleep(self.blocking_sleep)


if __name__ == "__main__":
    agent = Agent(camera_serial="xxxxxxxxxxxxxxxx", obs_num=2)
