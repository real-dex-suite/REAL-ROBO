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
import pyspacemouse

# load constants according to hand type
hand_type = HAND_TYPE.lower() if HAND_TYPE is not None else None
JOINTS_PER_FINGER = (
    eval(f"{hand_type.upper()}_JOINTS_PER_FINGER") if HAND_TYPE is not None else None
)
JOINT_OFFSETS = (
    eval(f"{hand_type.upper()}_JOINT_OFFSETS") if HAND_TYPE is not None else None
)

SPACE_MOUSE_CONTROL = False

import os
import sys
import spdlog

## Potential Bug: The SpaceMouseExpert class is not working as expected. The action and buttons are not being printed.
## sudo chmod 666 /dev/hidraw* to fix the issue
# using os to run the cmd: sudo chmod 666 /dev/hidraw*

if SPACE_MOUSE_CONTROL:
    os.system("cd /usr/lib64")
    os.system("sudo chmod 666 /dev/hidraw*")

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

    # Gram–Schmidt Or thonormalize
    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / np.linalg.norm(x)
    z = np.cross(x, normal)

    # We assume that the vector from pinky to index is similar the z axis in MANO convention
    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1
        z *= -1
    frame = np.stack([x, normal, z], axis=1)
    return frame


class HamerDexArmTeleOp(object):
    def __init__(self):
        # Initializing the ROS Node
        rospy.init_node("hamer_dexarm_teleop")

        if RETARGET_TYPE == "dexpilot":
            print("Loading the retargeting configuration")
            from holodex.components.retargeting.retargeting_config import (
                RetargetingConfig,
            )  # -> bug

            print("Loading the retargeting configuration")
            config_path = f"holodex/components/retargeting/configs/teleop/{HAND_TYPE.lower()}_hand_right_{RETARGET_TYPE}.yml"
            RetargetingConfig.set_default_urdf_dir("holodex/robot/hand")
            retarget_config = RetargetingConfig.load_from_file(config_path)
            self.retargeting = retarget_config.build()

        self.trans_scale = 1 if ARM_TYPE == "Flexiv" else 1000

        self.logger = spdlog.ConsoleLogger("RobotController")

        # Storing the transformed hand coordinates
        self.hand_coords = None
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
            JAKA_EE_POSE_TOPIC,
            Float64MultiArray,
            self._callback_arm_ee_pose,
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

        self.stop_move = False
        rospy.Subscriber(
            "/data_collector/stop_move",
            Bool,
            self._callback_stop_move,
            queue_size=1,
        )

        self.end_robot = False
        rospy.Subscriber(
            "/data_collector/end_robot",
            Bool,
            self._callback_end_robot,
            queue_size=1,
        )


        # Initializing the robot controller
        self.robot = RobotController(teleop=True)

        # self.arm_ee_pose = self.robot.arm.get_tcp_position()
        self.init_tcp = np.array(self._get_tcp_position())
        self.arm_ee_pose = self._get_tcp_position()

        # Initializing the solvers
        self.fingertip_solver = self.robot.hand_KDLControl
        self.finger_joint_solver = self.robot.hand_JointControl

        self.translation_state = None

        # Initialzing the moving average queues
        self.moving_average_queues = {
            "thumb": [],
            "index": [],
            "middle": [],
            "ring": [],
        }

        self.prev_hand_joint_angles = self.robot.get_hand_position()

        if ARM_TYPE is not None:
            self._calibrate_arm_bounds()
            # if ARM_TYPE == "Jaka":
            # TODO configureable
            self.leap2flange = np.eye(4)

            if ARM_TYPE == "Jaka":
                self.leap2flange[:3, :3] = R.from_euler(
                    "xyz", [0, 0, 214.5], degrees=True
                ).as_matrix()
            else:
                self.leap2flange[:3, :3] = R.from_euler(
                    "xyz", [0, 0, 90], degrees=True
                ).as_matrix()

        # Initializing SpaceMouse
        if SPACE_MOUSE_CONTROL:  # TODO: intergrate to processes.py

            try:
                pyspacemouse.open()
                self.logger.info("Space Mouse is started!")
            except Exception as e:
                self.logger.error(f"Failed to start space mouse.")

            # cprint("HamerDexArmTeleOp Initializing Successfully!2", "green")

            self.desired_arm_pose = list(self.arm_ee_pose)
            self.tmp_desired_arm_pose = copy(self.desired_arm_pose)

            self.lockhand = False
            self.lockrotation = False

            # Initializing the space mouse arguments
            self.manager = multiprocessing.Manager()
            self.latest_data = self.manager.dict()
            self.latest_data["action"] = [0.0] * 6  # Using lists for compatibility
            self.latest_data["buttons"] = [0, 0, 0, 0]

            # Start a process to continuously read the SpaceMouse state
            self.process = multiprocessing.Process(target=self._read_spacemouse)
            self.process.daemon = True
            self.process.start()

    def _get_tcp_position(self):
        if ARM_TYPE == "Flexiv":
            return self.robot.arm.get_tcp_position(euler=True, degree=False)
        else:
            return self.robot.arm.get_tcp_position()

    def get_action(self) -> Tuple[np.ndarray, list]:
        """Returns the latest action and button state of the SpaceMouse."""
        action = self.latest_data["action"]
        buttons = self.latest_data["buttons"]
        return np.array(action), buttons

    def _read_spacemouse(self):
        while True:
            state = pyspacemouse.read()
            action = [0.0] * 6
            buttons = [0, 0, 0, 0]

            self.latest_data["action"] = np.array(
                [-state.y, state.x, state.z, -state.roll, -state.pitch, -state.yaw]
            )  # spacemouse axis matched with robot base frame
            self.latest_data["buttons"] = state.buttons

    def _callback_hand_coords(self, coords):
        self.hand_coords = np.array(list(coords.data)).reshape(HAMER_NUM_KEYPOINTS, 3)

    def _callback_arm_coords(self, coords):
        self.arm_coords = np.array(list(coords.data)).reshape(
            HAMER_ARM_NUM_KEYPOINTS, 3
        )

    def _callback_arm_ee_pose(self, data):
        self.arm_ee_pose = np.array(data.data)

    def _retarget_hand(self, finger_configs):
        if RETARGET_TYPE == "dexpilot":
            indices = self.retargeting.optimizer.target_link_human_indices
            origin_indices = indices[0, :]
            task_indices = indices[1, :]
            ref_value = (
                self.hand_coords[task_indices, :] - self.hand_coords[origin_indices, :]
            )
            desired_joint_angles = self.retargeting.retarget(ref_value)

        return desired_joint_angles

    def _callback_end_robot(self, msg):
        self.end_robot = msg.data

    def _callback_stop_move(self, msg):
        self.stop_move = msg.data

    def _callback_reset_robot(self, msg):
        if msg.data: self.robot.home_robot()

    def _callback_reset_done(self, msg):
        self.robot.home_robot()
        if msg.data:
            if ARM_TYPE is not None:
                self._calibrate_arm_bounds()

    # Low-pass filter function to smooth translation values
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
        # translation_vector = np.average(armpoints * np.array([1.5, 1.5, 1/15]), axis=0)
        # print(armpoints)
        scaled_points = armpoints * np.array([1, 1, 1 / 30])
        timestamps = np.arange(len(scaled_points))

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

        # print("rotation_vectors: ", rotation_vectors)

        rotation_vectors = self._arm_filter(rotation_vectors, "low_pass")
        index_knuckle_coord = rotation_vectors[1]
        pinky_knuckle_coord = rotation_vectors[2]

        palm_normal = normalize_vector(
            np.cross(index_knuckle_coord, pinky_knuckle_coord)
        )
        palm_direction = normalize_vector(index_knuckle_coord + pinky_knuckle_coord)
        cross_product = normalize_vector(np.cross(palm_direction, palm_normal))

        # print("----------------------")
        # print(translation_vector, cross_product, palm_direction, palm_normal)

        return translation_vector, cross_product, palm_direction, palm_normal

    def _compute_transformation(self, init_hand_transformation, hand2vr_transformation):
        new_hand2init_hand = init_hand_transformation @ hand2vr_transformation
        init_flange2base = self.init_arm_transformation_matrix
        init_leap2base = init_flange2base @ self.leap2flange
        new_ee_transformation_matrix = (
            init_leap2base @ new_hand2init_hand @ np.linalg.inv(self.leap2flange)
        )

        return new_ee_transformation_matrix

    def _get_transformation(self, points_in_hand_space, points_in_vr_space):
        vr2init_hand_transformation, _, _ = best_fit_transform(
            self.init_points_in_vr_space, points_in_hand_space
        )
        hand2vr_transformation, _, _ = best_fit_transform(
            points_in_hand_space, points_in_vr_space
        )

        return vr2init_hand_transformation, hand2vr_transformation

    def _retarget_base(self):
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

        cprint(
            f"vr2init_hand_transformation: {vr2init_hand_transformation.shape}", "green"
        )

        # Extract translation and rotation parts
        composed_transformation = self._compute_transformation(
            vr2init_hand_transformation, hand2vr_transformation
        )  # 4 * 4

        # flexiv
        # this transformation is different from jaka, so we need to transform 180 degrees by z axis

        if ARM_TYPE == "Flexiv":
            # F_X_J = np.eye(4)
            # # F_X_J[:3, :3] = R.from_euler('xyz', [0, 0, -90], degrees=True).as_matrix()

            # tcpJ_X_tcpF = np.eye(4)
            # # tcpJ_X_tcpF[:3, :3] = R.from_euler('xyz', [0, 0, 180], degrees=True).as_matrix()
            # composed_transformation =  F_X_J @ composed_transformation @ tcpJ_X_tcpF
            pass

        composed_translation = composed_transformation[:3, 3]
        composed_rotation = composed_transformation[:3, :3]

        composed_rotation_quat = R.from_matrix(composed_rotation).as_quat()
        current_arm_pose = self.arm_ee_pose

        # exponential_smoothing = 0.2  # Smoothing factor (adjust as needed)
        exponential_smoothing = 1.0  # Smoothing factor (adjust as needed)

        if not hasattr(self, "previous_filtered_translation"):
            self.previous_filtered_translation = current_arm_pose[:3]

        # current_filtered_translation = ((1.0 - exponential_smoothing) * self.previous_filtered_translation) + (exponential_smoothing * composed_translation)
        # self.previous_filtered_translation = current_filtered_translation

        self.previous_filtered_translation = np.array(
            self.previous_filtered_translation
        )
        composed_translation = np.array(composed_translation)
        current_filtered_translation = (
            (1.0 - exponential_smoothing) * self.previous_filtered_translation
        ) + (exponential_smoothing * composed_translation)

        # use Slrp to smooth the rotation
        alpha = 0.03  # 0.2
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

        # if space mouse -> dont move translation
        if not SPACE_MOUSE_CONTROL:
            current_arm_pose[:3] = current_filtered_translation * self.trans_scale
        else:
            current_arm_pose[:3] = self._get_tcp_position()[:3]

        # current_arm_pose[3:6] = smoothed_rotation_euler
        current_arm_pose[3:6] = self._limited_rot_ws(
            smoothed_rotation_euler, max_rot=np.pi / 2, min_rot=-np.pi / 2
        )

        return current_arm_pose

    def get_space_mouse_motion(self, desired_joint_angles):
        desired_cmd = []
        action, buttons = self.get_action()

        if buttons[0] == 1:
            self.lockhand = not self.lockhand
            self.prev_desired_joint_angles = desired_joint_angles[6:]

        self.tmp_desired_arm_pose[:3] = (
            self.desired_arm_pose[:3] + action[:3] * move_linear_velocity
        )

        # euler to quat
        tmp_desired_arm_euler = desired_joint_angles[3:6]
        tmp_desired_arm_quat = self.robot.arm.eulerZYX2quat(tmp_desired_arm_euler)

        if self.lockhand:
            tmp_desired_joint_angles = self.prev_desired_joint_angles
        else:
            tmp_desired_joint_angles = desired_joint_angles[6:]

        if buttons[1] == 1:
            self.lockrotation = not self.lockrotation
            self.prev_desired_arm_quat = tmp_desired_arm_quat

        if self.lockrotation:
            tmp_desired_arm_quat = self.prev_desired_arm_quat

        desired_cmd = np.concatenate(
            [
                desired_cmd,
                self.tmp_desired_arm_pose[:3],
                tmp_desired_arm_quat,
                tmp_desired_joint_angles,
            ]
        )
        return desired_cmd

    def _filter(self, desired_hand_joint_angles):
        desired_hand_joint_angles = (
            desired_hand_joint_angles * SMOOTH_FACTOR
            + self.prev_hand_joint_angles * (1 - SMOOTH_FACTOR)
        )
        self.prev_hand_joint_angles = desired_hand_joint_angles
        return desired_hand_joint_angles

    def _limited_rot_ws(self, current_tcp, max_rot, min_rot):
        # TODO: finish this
        self.init_tcp_rot = self.init_tcp[3:6]
        current_tcp_rot = np.array(current_tcp)

        cprint(f"current tcp rot: {current_tcp_rot}", "red")
        # cprint(f"self.init_tcp_rot: {self.init_tcp_rot}", "green")
        # cprint(f"clip limits: {self.init_tcp_rot - max_rot, self.init_tcp_rot + max_rot}", "yellow")

        # current_tcp_rot = np.clip(current_tcp_rot, self.init_tcp_rot - max_rot, self.init_tcp_rot + max_rot)

        # if current

        return current_tcp_rot

    def _limited_trans_ws(self, current_tcp, max_dist, min_dist):
        pass

    def motion(self, finger_configs):
        desired_cmd = []

        if ARM_TYPE is not None:
            desired_arm_pose = self._retarget_base()

            if not SPACE_MOUSE_CONTROL:
                tmp_desired_arm_euler = desired_arm_pose[3:6]
                tmp_desired_arm_quat = self.robot.arm.eulerZYX2quat(
                    tmp_desired_arm_euler
                )
                desired_cmd = np.concatenate(
                    [desired_cmd, desired_arm_pose[:3], tmp_desired_arm_quat]
                )
            else:
                desired_cmd = np.concatenate([desired_cmd, desired_arm_pose])

        if HAND_TYPE is not None:
            # hand control based on the retargeted hand coordinates
            desired_hand_joint_angles = self._retarget_hand(finger_configs)
            desired_hand_joint_angles = self._filter(desired_hand_joint_angles)
            desired_cmd = np.concatenate([desired_cmd, desired_hand_joint_angles])

        return desired_cmd

    def _calibrate_arm_bounds(self):
        inital_frame_number = 1  # set to 50 will cause collision
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

    def move(self, finger_configs):
        print(
            "\n******************************************************************************"
        )
        cprint("[   ok   ]     Controller initiated. ", "green", attrs=["bold"])
        print(
            "******************************************************************************\n"
        )
        print("Start controlling the robot hand using the Hamer Framework.\n")

        while True:
            if (
                (self.arm_coords is not None)
                and (self.hand_coords is not None)
                and (self.robot.get_hand_position() is not None)
            ):

                # Original target_arm    [___x___] [___y___] [___z___] [___r___] [___p___] [___y___]
                # target arm             [_______] [_______] [_______] [___r___] [___p___] [___y___]
                #                                                          |         |         |
                #                                                          v         v         v
                # space mouse control    [___x___] [___y___] [___z___] [_______] [_______] [_______]
                #                            |         |         |         |         |         |
                #                            v         v         v         v         v         v
                # Desired_joint_angles   [_______] [_______] [_______] [_______] [_______] [_______] + [hand]
                if self.stop_move:
                    continue
                if self.end_robot:
                    break

                desired_joint_angles = self.motion(finger_configs)
                if SPACE_MOUSE_CONTROL:
                    self.tmp_desired_arm_pose = copy(self.desired_arm_pose)
                    desired_joint_angles = self.get_space_mouse_motion(
                        desired_joint_angles
                    )
                    self.desired_arm_pose = copy(self.tmp_desired_arm_pose)

                self.robot.move(desired_joint_angles)
