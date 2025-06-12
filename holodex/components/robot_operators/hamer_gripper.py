import rospy
from std_msgs.msg import Float64MultiArray, Bool

from holodex.utils.files import *
from holodex.utils.vec_ops import coord_in_bound, best_fit_transform, normalize_vector
from holodex.constants import *
from copy import deepcopy as copy
from scipy.spatial.transform import Rotation as R

try:
    from .robot import RobotController
except:
    from robot import RobotController
from pykalman import KalmanFilter
from scipy.interpolate import CubicSpline
from scipy.special import comb
from scipy.spatial.transform import Slerp
from termcolor import cprint
from typing import Tuple

import multiprocessing
import time
import os
import sys
import spdlog
import numpy as np

# Load constants according to hand type
hand_type = HAND_TYPE.lower() if HAND_TYPE is not None else None
JOINTS_PER_FINGER = (
    eval(f"{hand_type.upper()}_JOINTS_PER_FINGER") if HAND_TYPE is not None else None
)
JOINT_OFFSETS = (
    eval(f"{hand_type.upper()}_JOINT_OFFSETS") if HAND_TYPE is not None else None
)

move_linear_velocity = 0.002
move_angular_velocity = 0.002


def get_mano_coord_frame(keypoint_3d_array, oculus=False):
    """
    Compute the 3D coordinate frame (orientation only) from detected 3d key points
    :param points: keypoint3 detected from MediaPipe detector. Order: [wrist, index, middle, pinky]
    :return: the coordinate frame of wrist in MANO convention
    """
    if oculus:
        assert keypoint_3d_array.shape == (24, 3)
        points = keypoint_3d_array[[0, 6, 9], :]  # TODO check if this is correct
    else:
        assert keypoint_3d_array.shape == (21, 3)
        points = keypoint_3d_array[[0, 5, 9], :]

    # Compute vector from palm to the first joint of middle finger
    x_vector = points[0] - points[2]

    # Normal fitting with SVD
    points = points - np.mean(points, axis=0, keepdims=True)
    u, s, v = np.linalg.svd(points)

    normal = v[2, :]

    # Gram–Schmidt Orthonormalize
    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / np.linalg.norm(x)
    z = np.cross(x, normal)

    # We assume that the vector from pinky to index is similar the z axis in MANO convention
    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1
        z *= -1
    frame = np.stack([x, normal, z], axis=1)
    return frame


class HamerGripperDexArmTeleOp(object):
    '''
    TODO
    '''
    def __init__(self, simulator=None, gripper=None, arm_type="franka", gripper_init_state="open"):
        self.arm_type = arm_type
        raise NotImplementedError("HamerGripperDexArmTeleOp is not finished.")
        if RETARGET_TYPE == "dexpilot":
            print("Loading the retargeting configuration")
            from holodex.components.retargeting.retargeting_config import RetargetingConfig

            config_path = f"holodex/components/retargeting/configs/teleop/{HAND_TYPE.lower()}_hand_right_{RETARGET_TYPE}.yml"
            RetargetingConfig.set_default_urdf_dir("holodex/robot/hand")
            retarget_config = RetargetingConfig.load_from_file(config_path)
            self.retargeting = retarget_config.build()
        elif RETARGET_TYPE == "distance_gripper":
            self.retargeting = None

        self.trans_scale = 1
        self.logger = spdlog.ConsoleLogger("RobotController")
        self.finger_distance = 0.1
        
        # Initialize state variables
        self.hand_coords = None
        self.arm_coords = None
        self.arm_ee_pose = None
        self.stop_move = False
        self.end_robot = False
        self.translation_state = None
        
        # Set up ROS subscribers
        self._setup_subscribers()

        # Initialize robot controller
        self.robot = RobotController(teleop=True, simulator=simulator, gripper=gripper, arm_type=arm_type, gripper_init_state=gripper_init_state)
        self.init_tcp = np.array(self._get_tcp_position())
        self.arm_ee_pose = self._get_tcp_position()

        # Initialize moving average queues
        self.moving_average_queues = {
            "thumb": [],
            "index": [],
            "middle": [],
            "ring": [],
        }
        if self.arm_type is not None:
            self._calibrate_arm_bounds()
            self.leap2flange = np.eye(4)

            if self.arm_type == "jaka":
                self.leap2flange[:3, :3] = R.from_euler(
                    "xyz", [0, 0, 214.5], degrees=True
                ).as_matrix()
            elif self.arm_type == "franka":
                self.leap2flange[:3, :3] = R.from_euler(
                    "xyz", [0, 0, -90], degrees=True
                ).as_matrix()
            else:
                self.leap2flange[:3, :3] = R.from_euler(
                    "xyz", [0, 0, 90], degrees=True
                ).as_matrix()

    def _setup_subscribers(self):
        """Set up all ROS subscribers"""
        rospy.Subscriber(
            HAMER_HAND_TRANSFORM_COORDS_TOPIC,
            Float64MultiArray,
            self._callback_hand_coords,
            queue_size=1,
        )
        rospy.Subscriber(
            HAMER_ARM_TRANSFORM_COORDS_TOPIC,
            Float64MultiArray,
            self._callback_arm_coords,
            queue_size=1,
        )
        rospy.Subscriber(
            "/data_collector/reset_done",
            Bool,
            self._callback_reset_done,
            queue_size=1,
        )
        rospy.Subscriber(
            "/data_collector/reset_robot",
            Bool,
            self._callback_reset_robot,
            queue_size=1,
        )
        rospy.Subscriber(
            "/data_collector/stop_move",
            Bool,
            self._callback_stop_move,
            queue_size=1,
        )
        rospy.Subscriber(
            "/data_collector/end_robot",
            Bool,
            self._callback_end_robot,
            queue_size=1,
        )
        rospy.Subscriber(
            HAMER_FINGER_DISTANCE_TOPIC,
            Float64MultiArray,
            self._callback_finger_distance,
            queue_size=1,
        )

    def _get_tcp_position(self):
        """Get the TCP position based on the arm type"""
        if self.arm_type == "flexiv":
            return self.robot.arm.get_tcp_position(euler=True, degree=False)
        elif self.arm_type == "franka":
            tcp_pose = self.robot.arm.get_tcp_position()  # w, x, y, z
            # Convert to euler
            tcp_quat_wxyz = tcp_pose[3:7]
            tcp_quat_xyzw = [tcp_quat_wxyz[1], tcp_quat_wxyz[2], tcp_quat_wxyz[3], tcp_quat_wxyz[0]]
            tcp_rot = R.from_quat(tcp_quat_xyzw).as_euler("xyz", degrees=False)
            return np.concatenate([tcp_pose[:3], tcp_rot])
        else:
            return self.robot.arm.get_tcp_position()

    def _callback_finger_distance(self, data):
        self.finger_distance = np.array(list(data.data))
            
    def _callback_hand_coords(self, coords):
        self.hand_coords = np.array(list(coords.data)).reshape(HAMER_NUM_KEYPOINTS, 3)

    def _callback_arm_coords(self, coords):
        self.arm_coords = np.array(list(coords.data)).reshape(
            HAMER_ARM_NUM_KEYPOINTS, 3
        )

    def _callback_end_robot(self, msg):
        self.end_robot = msg.data

    def _callback_stop_move(self, msg):
        self.stop_move = msg.data

    def _callback_reset_robot(self, msg):
        if msg.data:
            self.robot.home_robot()

    def _callback_reset_done(self, msg):
        self.robot.home_robot()
        if msg.data and self.arm_type is not None:
            self._calibrate_arm_bounds()

    # Filter functions for smoothing
    def _low_pass_filter(self, new_value, state, alpha=0.4):
        if state is None:
            state = new_value
        else:
            state = alpha * new_value + (1 - alpha) * state
        return state

    def _kalman_filter(self, observation):
        if self.state_mean is None:
            self.state_mean = observation
            self.state_covariance = np.eye(3) * 1e-1

        self.state_mean, self.state_covariance = self.kf.filter_update(
            self.state_mean, self.state_covariance, observation
        )
        return self.state_mean

    def _stable_translation(self, new_translation, alpha=0.4):
        self.translation_state = self._low_pass_filter(
            new_translation, self.translation_state, alpha
        )
        return self.translation_state

    def _arm_filter(self, translation, filter_type):
        if filter_type == "low_pass":
            return self._stable_translation(translation)
        elif filter_type == "kalman":
            return self._kalman_filter(translation)
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

    def vr_to_robot(self, armpoints):
        """Convert VR arm points to robot coordinates"""
        
        scaled_points = armpoints * np.array([2, 2, 1 / 10])
        timestamps = np.arange(len(scaled_points))

        # Create cubic splines for smooth interpolation
        cs_x = CubicSpline(timestamps, scaled_points[:, 0])
        cs_y = CubicSpline(timestamps, scaled_points[:, 1])
        cs_z = CubicSpline(timestamps, scaled_points[:, 2])

        dense_timestamps = np.linspace(
            0, len(scaled_points) - 1, num=len(scaled_points) * 2
        )
        interpolated_points = np.column_stack(
            (cs_x(dense_timestamps), cs_y(dense_timestamps), cs_z(dense_timestamps))
        )

        translation_vector = np.average(interpolated_points, axis=0)
        rotation_vectors = armpoints - np.average(armpoints, axis=0)

        rotation_vectors = self._arm_filter(rotation_vectors, "low_pass")
        index_knuckle_coord = rotation_vectors[1]
        pinky_knuckle_coord = rotation_vectors[2]

        palm_normal = normalize_vector(
            np.cross(index_knuckle_coord, pinky_knuckle_coord)
        )
        palm_direction = normalize_vector(index_knuckle_coord + pinky_knuckle_coord)
        cross_product = normalize_vector(np.cross(palm_direction, palm_normal))

        return translation_vector, cross_product, palm_direction, palm_normal

    def _compute_transformation(self, init_hand_transformation, hand2vr_transformation):
        """Compute the transformation matrix for the robot arm"""
        new_hand2init_hand = init_hand_transformation @ hand2vr_transformation
        init_flange2base = self.init_arm_transformation_matrix
        init_leap2base = init_flange2base @ self.leap2flange
        new_ee_transformation_matrix = (
            init_leap2base @ new_hand2init_hand @ np.linalg.inv(self.leap2flange)
        )
        return new_ee_transformation_matrix

    def _get_transformation(self, points_in_hand_space, points_in_vr_space):
        """Get transformation matrices between hand space and VR space"""
        vr2init_hand_transformation, _, _ = best_fit_transform(
            self.init_points_in_vr_space, points_in_hand_space
        )
        hand2vr_transformation, _, _ = best_fit_transform(
            points_in_hand_space, points_in_vr_space
        )
        return vr2init_hand_transformation, hand2vr_transformation

    def _retarget_base(self):
        """Retarget the base position of the robot arm"""
        # Get the scaled hand center and direction vectors
        hand_center, hand_x, hand_y, hand_z = self.vr_to_robot(self.arm_coords)

        # Define points in hand space
        points_in_hand_space = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )

        points_in_vr_space = np.array(
            [
                hand_center,
                hand_center + hand_x,
                hand_center + hand_y,
                hand_center + hand_z,
            ]
        )

        vr2init_hand_transformation, hand2vr_transformation = self._get_transformation(
            points_in_hand_space, points_in_vr_space
        )

        # Compute the composed transformation
        composed_transformation = self._compute_transformation(
            vr2init_hand_transformation, hand2vr_transformation
        )

        composed_translation = composed_transformation[:3, 3]
        composed_rotation = composed_transformation[:3, :3]
        composed_rotation_quat = R.from_matrix(composed_rotation).as_quat()
        current_arm_pose = self.arm_ee_pose

        # Apply exponential smoothing to translation
        exponential_smoothing = 0.8
        if not hasattr(self, "previous_filtered_translation"):
            self.previous_filtered_translation = current_arm_pose[:3]

        self.previous_filtered_translation = np.array(self.previous_filtered_translation)
        composed_translation = np.array(composed_translation)
        current_filtered_translation = (
            (1.0 - exponential_smoothing) * self.previous_filtered_translation
        ) + (exponential_smoothing * composed_translation)

        # Use Slerp to smooth the rotation
        alpha = 0.2 # alpha composed_rotation + (1- alpha) of previous_rotation.
        if not hasattr(self, "previous_filtered_rotation"):
            self.previous_filtered_rotation = R.from_euler(
                "xyz", current_arm_pose[3:6]
            ).as_quat()
        previous_rotation = R.from_quat(self.previous_filtered_rotation)
        composed_rotation = R.from_quat(composed_rotation_quat)

        key_times = [0, 1]  # Time keyframes for the start and end
        key_rotations = R.from_quat(
            [previous_rotation.as_quat(), composed_rotation.as_quat()]
        )

        # Create Slerp interpolator with keyframes
        slerp = Slerp(key_times, key_rotations)
        smoothed_rotation = slerp(alpha).as_quat()
        self.previous_filtered_rotation = smoothed_rotation
        smoothed_rotation_euler = R.from_quat(smoothed_rotation).as_euler("xyz")

        # Update the arm pose
        current_arm_pose[:3] = current_filtered_translation * self.trans_scale
        current_arm_pose[3:6] = smoothed_rotation_euler
        
        # current_arm_pose[3:6] = self._limited_rot_ws(
        #     smoothed_rotation_euler, max_rot=np.pi / 2, min_rot=-np.pi / 2
        # )

        return current_arm_pose

    def _limited_rot_ws(self, current_tcp, max_rot, min_rot):
        """Limit rotation within workspace"""
        self.init_tcp_rot = self.init_tcp[3:6]
        current_tcp_rot = np.array(current_tcp)
        return current_tcp_rot

    def motion(self):
        """Generate motion commands for the robot"""
        desired_cmd = []

        if self.arm_type is not None:
            desired_arm_pose = self._retarget_base()
            tmp_desired_arm_euler = desired_arm_pose[3:6]
            tmp_desired_arm_quat = self.robot.arm.eulerZYX2quat(
                tmp_desired_arm_euler
            )
            desired_cmd = np.concatenate(
                [desired_cmd, desired_arm_pose[:3], tmp_desired_arm_quat]
            )

        return desired_cmd

    def _calibrate_arm_bounds(self):
        """Calibrate the arm bounds based on initial positions"""
        inital_frame_number = 1
        frame_number = 0

        initial_hand_centers = []
        initial_hand_xs = []
        initial_hand_ys = []
        initial_hand_zs = []

        initial_arm_poss = []
        initial_arm_rots = []

        while frame_number < inital_frame_number:
            hand_center, hand_x, hand_y, hand_z = self.vr_to_robot(self.arm_coords)
            initial_hand_centers.append(hand_center)
            initial_hand_xs.append(hand_x)
            initial_hand_ys.append(hand_y)
            initial_hand_zs.append(hand_z)

            initial_arm_poss.append(
                np.array(self._get_tcp_position()[:3]) / self.trans_scale
            )
            initial_arm_rots.append(np.array(self._get_tcp_position()[3:6]))

            frame_number += 1

        # Calculate mean values
        init_hand_center = np.mean(initial_hand_centers, axis=0)
        init_hand_x = np.mean(initial_hand_xs, axis=0)
        init_hand_y = np.mean(initial_hand_ys, axis=0)
        init_hand_z = np.mean(initial_hand_zs, axis=0)

        self.init_points_in_vr_space = np.array(
            [
                init_hand_center,
                init_hand_center + init_hand_x,
                init_hand_center + init_hand_y,
                init_hand_center + init_hand_z,
            ]
        )

        self.init_arm_pos = np.mean(initial_arm_poss, axis=0)
        self.init_arm_rot = np.mean(initial_arm_rots, axis=0)
        self.init_arm_transformation_matrix = np.eye(4)
        self.init_arm_transformation_matrix[:3, :3] = R.from_euler(
            "xyz", self.init_arm_rot
        ).as_matrix()
        self.init_arm_transformation_matrix[:3, 3] = self.init_arm_pos.reshape(3)


    def move(self):
        """Main control loop for robot movement"""
        print("\n" + "*" * 78)
        cprint("[   ok   ]     Controller initiated. ", "green", attrs=["bold"])
        print("*" * 78 + "\n")
        print("Start controlling the robot hand using the Hamer Framework.\n")

        while True:
            if (self.arm_coords is not None) and (self.hand_coords is not None):
                if self.stop_move:
                    continue
                if self.end_robot:
                    break
                # Generate desired joint angles based on current joystick pose
                desired_cmd = self.motion()
                self.robot.move(np.concatenate([desired_cmd, self.finger_distance < 0.05]))
                
if __name__ == "__main__":
    hamer = HamerGripperDexArmTeleOp()
    hamer.move()