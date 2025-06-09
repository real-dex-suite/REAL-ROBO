import rospy
from std_msgs.msg import Float64MultiArray

from holodex.utils.files import *
from holodex.utils.vec_ops import coord_in_bound, best_fit_transform, normalize_vector
from holodex.constants import *
from copy import deepcopy as copy
from scipy.spatial.transform import Rotation as R

from scipy.spatial.transform import Slerp
from .robot import RobotController
from pykalman import KalmanFilter
from scipy.interpolate import CubicSpline
from scipy.special import comb

from termcolor import cprint

# load constants according to hand type
hand_type = HAND_TYPE.lower()
JOINTS_PER_FINGER = eval(f"{hand_type.upper()}_JOINTS_PER_FINGER")
JOINT_OFFSETS = eval(f"{hand_type.upper()}_JOINT_OFFSETS")


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

        # Storing the transformed hand coordinates
        self.hand_coords = None

        rospy.Subscriber(
            HAMER_HAND_TRANSFORM_COORDS_TOPIC,
            Float64MultiArray,
            self._callback_hand_coords,
            queue_size=1,
        )
        rospy.Subscriber(
            "/hamer/aruco_pose",
            Float64MultiArray,
            self._callback_aruco_pose,
            queue_size=1,
        )

        # Initializing the robot controller
        self.robot = RobotController(teleop=True)
        # Initializing the solvers
        self.fingertip_solver = self.robot.hand_KDLControl
        self.finger_joint_solver = self.robot.hand_JointControl
        # smoothing factor
        self.window_size = 50
        self.translation_history = []
        process_noise_cov = np.eye(3) * 1e-2
        measurement_noise_cov = np.eye(3) * 1e-1
        self.kf = KalmanFilter(
            transition_matrices=np.eye(3),
            observation_matrices=np.eye(3),
            transition_covariance=process_noise_cov,
            observation_covariance=measurement_noise_cov,
        )
        self.state_mean = None
        self.state_covariance = None
        self.translation_state = None
        # Initialzing the moving average queues
        self.moving_average_queues = {
            "thumb": [],
            "index": [],
            "middle": [],
            "ring": [],
        }

        self.prev_hand_joint_angles = self.robot.get_hand_position()

        if RETARGET_TYPE == "dexpilot":
            from holodex.components.retargeting.retargeting_config import (
                RetargetingConfig,
            )

            config_path = f"holodex/components/retargeting/configs/teleop/{HAND_TYPE.lower()}_hand_right_{RETARGET_TYPE}.yml"
            RetargetingConfig.set_default_urdf_dir("holodex/robot/hand")
            self.retargeting = RetargetingConfig.load_from_file(config_path).build()

        if ARM_TYPE is not None:
            self._calibrate_vr_arm_bounds()
            if ARM_TYPE == "Jaka":
                # TODO configureable
                self.leap2flange = np.eye(4)
                self.leap2flange[:3, :3] = R.from_euler(
                    "xyz", [0, 0, 214.5], degrees=True
                ).as_matrix()

    def _callback_hand_coords(self, coords):
        self.hand_coords = np.array(list(coords.data)).reshape(HAMER_NUM_KEYPOINTS, 3)

    def _callback_aruco_pose(self, msg):
        self.arm_coords = np.array(list(msg.data)).reshape(4, 3)

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

    def _low_pass_filter(self, new_value, state, alpha=0.4):
        if state is None:
            state = new_value
        else:
            state = alpha * new_value + (1 - alpha) * state
        return state

    def _stable_translation(self, new_translation, alpha=0.4):
        self.translation_state = self._low_pass_filter(
            new_translation, self.translation_state, alpha
        )
        return self.translation_state

    def _kalman_filter(self, observation):
        if self.state_mean is None:
            self.state_mean = observation
            self.state_covariance = np.eye(3) * 1e-1

        self.state_mean, self.state_covariance = self.kf.filter_update(
            self.state_mean, self.state_covariance, observation
        )

        return self.state_mean

    def _arm_filter(self, translation, filter_type):
        if filter_type == "low_pass":
            return self._stable_translation(translation)
        elif filter_type == "kalman":
            return self._kalman_filter(translation)
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

    def _bezier_curve(self, points, num_samples=100):
        num_points = len(points)
        t = np.linspace(0, 1, num_samples)

        curve = np.zeros((num_samples, 3))
        for i in range(num_samples):
            for j in range(num_points):
                # B(t) = Sum((n choose k) * (1-t)^(n-k) * t^k * P_k)
                curve[i] += (
                    comb(num_points - 1, j)
                    * (1 - t[i]) ** (num_points - 1 - j)
                    * t[i] ** j
                    * points[j]
                )

        return curve

    # exponential_smoothing_for_orientation = 0.2 # 0.25 #0.1
    # exponential_smoothing_for_position = 0.2 #0.5 #0.1

    def _retarget_base(self):
        if self.arm_coords is None:
            raise ValueError("arm_coords not set")

        # Get the aruco pose from arm_coords
        print("arm coords: ", self.arm_coords)
        print(self.arm_coords[0])
        hand_center = self.arm_coords[0]
        hand_x = self.arm_coords[1]
        hand_y = self.arm_coords[2]
        hand_z = self.arm_coords[3]

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

        # Extract translation and rotation parts
        composed_translation = self._compute_transformation(
            vr2init_hand_transformation, hand2vr_transformation
        )[:3, 3]
        composed_rotation = self._compute_transformation(
            vr2init_hand_transformation, hand2vr_transformation
        )[:3, :3]
        composed_rotation = R.from_matrix(composed_rotation).as_euler("xyz")

        # Get the current arm TCP position
        current_arm_pose = self.robot.arm.get_tcp_position()

        exponential_smoothing = 0.25  # Smoothing factor (adjust as needed)
        if not hasattr(self, "previous_filtered_translation"):
            self.previous_filtered_translation = current_arm_pose[:3]

        current_filtered_translation = (
            (1.0 - exponential_smoothing) * self.previous_filtered_translation
        ) + (exponential_smoothing * composed_translation)
        self.previous_filtered_translation = current_filtered_translation

        exponential_smoothing_for_position = 0.3
        if not hasattr(self, "previous_filtered_rotation"):
            self.previous_filtered_rotation = current_arm_pose[3:6]
        current_filtered_rotation = (
            (1.0 - exponential_smoothing_for_position) * self.previous_filtered_rotation
        ) + (exponential_smoothing_for_position * composed_rotation)
        self.previous_filtered_rotation = current_filtered_rotation

        current_arm_pose[:3] = composed_translation * 1000

        current_arm_pose[3:6] = composed_rotation

        return current_arm_pose

    def motion(self, finger_configs):
        desired_cmd = []

        if ARM_TYPE is not None:
            desired_arm_pose = self._retarget_base()
            desired_cmd = np.concatenate([desired_cmd, desired_arm_pose])

        # if HAND_TYPE is not None:
        #     desired_hand_joint_angles = self._retarget_hand(finger_configs)
        #     desired_hand_joint_angles = self._filter(desired_hand_joint_angles)
        #     desired_cmd = np.concatenate([desired_cmd, desired_hand_joint_angles])

        return desired_cmd

    def _calibrate_vr_arm_bounds(self):
        inital_frame_number = 5  # set to 50 will cause collision
        frame_number = 0

        initial_hand_centers = []
        initial_hand_xs = []
        initial_hand_ys = []
        initial_hand_zs = []

        initial_arm_poss = []
        initial_arm_rots = []

        while frame_number < inital_frame_number:
            print("calibration initial pose, id: ", frame_number)
            # hand_center, hand_x, hand_y, hand_z = self.vr_to_robot(self.arm_coords)
            hand_center = self.arm_coords[0]
            hand_x = self.arm_coords[1]
            hand_y = self.arm_coords[2]
            hand_z = self.arm_coords[3]

            initial_hand_centers.append(hand_center)
            initial_hand_xs.append(hand_x)
            initial_hand_ys.append(hand_y)
            initial_hand_zs.append(hand_z)

            initial_arm_poss.append(
                np.array(self.robot.arm.get_tcp_position()[:3]) / 1000
            )
            initial_arm_rots.append(np.array(self.robot.arm.get_tcp_position()[3:6]))

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
            # if self.hand_coords is not None and self.robot.get_hand_position() is not None:
            # Obtaining the desired angles
            if self.robot.get_hand_position() is not None:
                print("1")
                desired_joint_angles = self.motion(finger_configs)
                print("current joint angles: ", self.robot.get_arm_tcp_position())
                print("desired joint angles: ", desired_joint_angles)

                # Move the hand based on the desired angles
                self.robot.move(desired_joint_angles)
