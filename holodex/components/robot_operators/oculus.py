import rospy
from std_msgs.msg import Float64MultiArray
from .calibrators import OculusThumbBoundCalibrator

from holodex.utils.files import *
from holodex.utils.vec_ops import coord_in_bound, best_fit_transform, normalize_vector
from holodex.constants import *
from copy import deepcopy as copy
from scipy.spatial.transform import Rotation as R

from .robot import RobotController

# load constants according to hand type
hand_type = HAND_TYPE.lower() if HAND_TYPE is not None else None
JOINTS_PER_FINGER = eval(f"{hand_type.upper()}_JOINTS_PER_FINGER") if HAND_TYPE is not None else None
JOINT_OFFSETS = eval(f"{hand_type.upper()}_JOINT_OFFSETS") if HAND_TYPE is not None else None


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


class VRDexArmTeleOp(object):
    def __init__(self):
        # Initializing the ROS Node
        rospy.init_node("vr_dexarm_teleop")

        # Storing the transformed hand coordinates
        self.hand_coords = None
        rospy.Subscriber(
            VR_RIGHT_TRANSFORM_COORDS_TOPIC,
            Float64MultiArray,
            self._callback_hand_coords,
            queue_size=1,
        )
        rospy.Subscriber(
            VR_RIGHT_ARM_TRANSFORM_COORDS_TOPIC,
            Float64MultiArray,
            self._callback_arm_coords,
            queue_size=1,
        )

        # Initializing the robot controller
        self.robot = RobotController(teleop=True)
        # Initializing the solvers
        self.fingertip_solver = self.robot.hand_KDLControl
        self.finger_joint_solver = self.robot.hand_JointControl

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
        elif RETARGET_TYPE == "joint":
            # Calibrating to get the thumb bounds
            self._calibrate_bounds()

            # Getting the bounds for the robot hand
            robohand_bounds_path = get_path_in_package(
                f"components/robot_operators/configs/{hand_type}_vr.yaml"
            )
            with open(robohand_bounds_path, "r") as file:
                self.robohand_bounds = yaml.safe_load(file)
        elif RETARGET_TYPE == "dexjoint":
            from holodex.components.retargeting.retargeting_config import (
                RetargetingConfig,
            )

            config_path = f"holodex/components/retargeting/configs/teleop/{HAND_TYPE.lower()}_hand_right_dexpilot.yml"
            RetargetingConfig.set_default_urdf_dir("holodex/robot/hand")
            self.retargeting = RetargetingConfig.load_from_file(config_path).build()

            # Calibrating to get the thumb bounds
            self._calibrate_bounds()

            # Getting the bounds for the robot hand
            robohand_bounds_path = get_path_in_package(
                f"components/robot_operators/configs/{hand_type}_vr.yaml"
            )
            with open(robohand_bounds_path, "r") as file:
                self.robohand_bounds = yaml.safe_load(file)

        if ARM_TYPE is not None:
            self._calibrate_vr_arm_bounds()
            if ARM_TYPE == "Jaka":
                # TODO configureable
                self.leap2flange = np.eye(4)
                self.leap2flange[:3, :3] = R.from_euler(
                    "xyz", [0, 0, 214.5], degrees=True
                ).as_matrix()

    def _calibrate_bounds(self):
        print("***************************************************************")
        print("     Starting calibration process ")
        print("***************************************************************")
        calibrator = OculusThumbBoundCalibrator()
        self.thumb_index_bounds, self.thumb_middle_bounds, self.thumb_ring_bounds = (
            calibrator.get_bounds()
        )

    def _callback_hand_coords(self, coords):
        self.hand_coords = np.array(list(coords.data)).reshape(24, 3)

    def _callback_arm_coords(self, coords):
        self.arm_coords = np.array(list(coords.data)).reshape(
            OCULUS_ARM_NUM_KEYPOINTS, 3
        )

    def _get_finger_coords(self, finger_type):
        return np.vstack(
            [self.hand_coords[0], self.hand_coords[OCULUS_JOINTS[finger_type]]]
        )

    def _get_2d_thumb_angles(self, curr_angles):
        if (
            coord_in_bound(
                self.thumb_index_bounds[:4], self._get_finger_coords("thumb")[-1][:2]
            )
            > -1
        ):
            return self.fingertip_solver.thumb_motion_2D(
                hand_coordinates=self._get_finger_coords("thumb")[-1],
                xy_hand_bounds=self.thumb_index_bounds[:4],
                yz_robot_bounds=[
                    self.robohand_bounds["thumb"]["top_right"],
                    self.robohand_bounds["thumb"]["bottom_right"],
                    self.robohand_bounds["thumb"]["index_bottom"],
                    self.robohand_bounds["thumb"]["index_top"],
                ],
                robot_x_val=self.robohand_bounds["thumb"]["x_coord"],
                moving_avg_arr=self.moving_average_queues["thumb"],
                curr_angles=curr_angles,
            )
        elif (
            coord_in_bound(
                self.thumb_middle_bounds[:4], self._get_finger_coords("thumb")[-1][:2]
            )
            > -1
        ):
            return self.fingertip_solver.thumb_motion_2D(
                hand_coordinates=self._get_finger_coords("thumb")[-1],
                xy_hand_bounds=self.thumb_middle_bounds[:4],
                yz_robot_bounds=[
                    self.robohand_bounds["thumb"]["index_top"],
                    self.robohand_bounds["thumb"]["index_bottom"],
                    self.robohand_bounds["thumb"]["middle_bottom"],
                    self.robohand_bounds["thumb"]["middle_top"],
                ],
                robot_x_val=self.robohand_bounds["thumb"]["x_coord"],
                moving_avg_arr=self.moving_average_queues["thumb"],
                curr_angles=curr_angles,
            )
        elif (
            coord_in_bound(
                self.thumb_ring_bounds[:4], self._get_finger_coords("thumb")[-1][:2]
            )
            > -1
        ):
            return self.fingertip_solver.thumb_motion_2D(
                hand_coordinates=self._get_finger_coords("thumb")[-1],
                xy_hand_bounds=self.thumb_ring_bounds[:4],
                yz_robot_bounds=[
                    self.robohand_bounds["thumb"]["middle_top"],
                    self.robohand_bounds["thumb"]["middle_bottom"],
                    self.robohand_bounds["thumb"]["ring_bottom"],
                    self.robohand_bounds["thumb"]["ring_top"],
                ],
                robot_x_val=self.robohand_bounds["thumb"]["x_coord"],
                moving_avg_arr=self.moving_average_queues["thumb"],
                curr_angles=curr_angles,
            )
        else:
            return curr_angles

    def _get_3d_thumb_angles(self, curr_angles):
        if (
            coord_in_bound(
                self.thumb_index_bounds[:4], self._get_finger_coords("thumb")[-1][:2]
            )
            > -1
        ):
            return self.fingertip_solver.thumb_motion_3D(
                hand_coordinates=self._get_finger_coords("thumb")[-1],
                xy_hand_bounds=self.thumb_index_bounds[:4],
                yz_robot_bounds=[
                    self.robohand_bounds["thumb"]["top_right"],
                    self.robohand_bounds["thumb"]["bottom_right"],
                    self.robohand_bounds["thumb"]["index_bottom"],
                    self.robohand_bounds["thumb"]["index_top"],
                ],
                z_hand_bound=self.thumb_index_bounds[4],
                x_robot_bound=[
                    self.robohand_bounds["thumb"]["index_x_bottom"],
                    self.robohand_bounds["thumb"]["index_x_top"],
                ],
                moving_avg_arr=self.moving_average_queues["thumb"],
                curr_angles=curr_angles,
            )
        elif (
            coord_in_bound(
                self.thumb_middle_bounds[:4], self._get_finger_coords("thumb")[-1][:2]
            )
            > -1
        ):
            return self.fingertip_solver.thumb_motion_3D(
                hand_coordinates=self._get_finger_coords("thumb")[-1],
                xy_hand_bounds=self.thumb_middle_bounds[:4],
                yz_robot_bounds=[
                    self.robohand_bounds["thumb"]["index_top"],
                    self.robohand_bounds["thumb"]["index_bottom"],
                    self.robohand_bounds["thumb"]["middle_bottom"],
                    self.robohand_bounds["thumb"]["middle_top"],
                ],
                z_hand_bound=self.thumb_middle_bounds[4],
                x_robot_bound=[
                    self.robohand_bounds["thumb"]["middle_x_bottom"],
                    self.robohand_bounds["thumb"]["middle_x_top"],
                ],
                moving_avg_arr=self.moving_average_queues["thumb"],
                curr_angles=curr_angles,
            )
        elif (
            coord_in_bound(
                self.thumb_ring_bounds[:4], self._get_finger_coords("thumb")[-1][:2]
            )
            > -1
        ):
            return self.fingertip_solver.thumb_motion_3D(
                hand_coordinates=self._get_finger_coords("thumb")[-1],
                xy_hand_bounds=self.thumb_ring_bounds[:4],
                yz_robot_bounds=[
                    self.robohand_bounds["thumb"]["middle_top"],
                    self.robohand_bounds["thumb"]["middle_bottom"],
                    self.robohand_bounds["thumb"]["ring_bottom"],
                    self.robohand_bounds["thumb"]["ring_top"],
                ],
                z_hand_bound=self.thumb_ring_bounds[4],
                x_robot_bound=[
                    self.robohand_bounds["thumb"]["ring_x_bottom"],
                    self.robohand_bounds["thumb"]["ring_x_top"],
                ],
                moving_avg_arr=self.moving_average_queues["thumb"],
                curr_angles=curr_angles,
            )
        else:
            return curr_angles

    def _retarget_hand(self, finger_configs):
        if RETARGET_TYPE == "dexpilot":
            indices = self.retargeting.optimizer.target_link_human_indices
            origin_indices = indices[0, :]
            task_indices = indices[1, :]
            ref_value = (
                self.hand_coords[task_indices, :] - self.hand_coords[origin_indices, :]
            )
            desired_joint_angles = self.retargeting.retarget(ref_value)
        elif RETARGET_TYPE == "joint":
            desired_joint_angles = copy(self.robot.get_hand_position())

            # Movement for the index finger
            if not finger_configs["freeze_index"]:
                desired_joint_angles = self.finger_joint_solver.calculate_finger_angles(
                    finger_type="index",
                    finger_joint_coords=self._get_finger_coords("index"),
                    curr_angles=desired_joint_angles,
                    moving_avg_arr=self.moving_average_queues["index"],
                )
            else:
                for idx in range(JOINTS_PER_FINGER):
                    if idx > 0:
                        desired_joint_angles[idx + JOINT_OFFSETS["index"]] = 0.05
                    else:
                        desired_joint_angles[idx + JOINT_OFFSETS["index"]] = 0

            # Movement for the middle finger
            if not finger_configs["freeze_middle"]:
                desired_joint_angles = self.finger_joint_solver.calculate_finger_angles(
                    finger_type="middle",
                    finger_joint_coords=self._get_finger_coords("middle"),
                    curr_angles=desired_joint_angles,
                    moving_avg_arr=self.moving_average_queues["middle"],
                )
            else:
                for idx in range(JOINTS_PER_FINGER):
                    if idx > 0:
                        desired_joint_angles[idx + JOINT_OFFSETS["middle"]] = 0.05
                    else:
                        desired_joint_angles[idx + JOINT_OFFSETS["middle"]] = 0

            # Movement for the ring finger
            # Calculating the translatory joint angles
            desired_joint_angles = self.finger_joint_solver.calculate_finger_angles(
                finger_type="ring",
                finger_joint_coords=self._get_finger_coords("ring"),
                curr_angles=desired_joint_angles,
                moving_avg_arr=self.moving_average_queues["ring"],
            )

            # Movement for the thumb finger - we disable 3D motion just for the thumb
            if finger_configs["three_dim"]:
                desired_joint_angles = self._get_3d_thumb_angles(desired_joint_angles)
            else:
                desired_joint_angles = self._get_2d_thumb_angles(desired_joint_angles)

            desired_joint_angles = np.array(desired_joint_angles)
            if "LEAP" in HAND_TYPE.upper():
                desired_joint_angles = desired_joint_angles[
                    [1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15]
                ]  # the original hand order is 0,1, but in robot the order change to 1,0 for dexpilot, so for oculus we need to change back

        elif RETARGET_TYPE == "dexjoint":
            indices = self.retargeting.optimizer.target_link_human_indices
            origin_indices = indices[0, :]
            task_indices = indices[1, :]

            translated_coords = self.hand_coords.copy()
            original_coord_frame = get_mano_coord_frame(translated_coords, oculus=True)
            transformed_coords = (
                translated_coords @ original_coord_frame @ OPERATOR2MANO_RIGHT
            )
            # transform coords around z axis for 180 degree
            transformed_coords[:, 0] *= -1

            ref_value = (
                transformed_coords[task_indices, :]
                - transformed_coords[origin_indices, :]
            )
            desired_joint_angles = self.retargeting.retarget(ref_value)
            desired_thumb_angles = desired_joint_angles[-5:].copy()

            desired_joint_angles = copy(self.robot.get_hand_position())

            # Movement for the index finger
            if not finger_configs["freeze_index"]:
                desired_joint_angles = self.finger_joint_solver.calculate_finger_angles(
                    finger_type="index",
                    finger_joint_coords=self._get_finger_coords("index"),
                    curr_angles=desired_joint_angles,
                    moving_avg_arr=self.moving_average_queues["index"],
                )
            else:
                for idx in range(JOINTS_PER_FINGER):
                    if idx > 0:
                        desired_joint_angles[idx + JOINT_OFFSETS["index"]] = 0.05
                    else:
                        desired_joint_angles[idx + JOINT_OFFSETS["index"]] = 0

            # Movement for the middle finger
            if not finger_configs["freeze_middle"]:
                desired_joint_angles = self.finger_joint_solver.calculate_finger_angles(
                    finger_type="middle",
                    finger_joint_coords=self._get_finger_coords("middle"),
                    curr_angles=desired_joint_angles,
                    moving_avg_arr=self.moving_average_queues["middle"],
                )
            else:
                for idx in range(JOINTS_PER_FINGER):
                    if idx > 0:
                        desired_joint_angles[idx + JOINT_OFFSETS["middle"]] = 0.05
                    else:
                        desired_joint_angles[idx + JOINT_OFFSETS["middle"]] = 0

            # Movement for the ring finger
            # Calculating the translatory joint angles
            desired_joint_angles = self.finger_joint_solver.calculate_finger_angles(
                finger_type="ring",
                finger_joint_coords=self._get_finger_coords("ring"),
                curr_angles=desired_joint_angles,
                moving_avg_arr=self.moving_average_queues["ring"],
            )

            # Movement for the thumb finger - we disable 3D motion just for the thumb
            if finger_configs["three_dim"]:
                desired_joint_angles = self._get_3d_thumb_angles(desired_joint_angles)
            else:
                desired_joint_angles = self._get_2d_thumb_angles(desired_joint_angles)

            desired_joint_angles = np.array(desired_joint_angles)

            if "LEAP" in HAND_TYPE.upper():
                desired_joint_angles = desired_joint_angles[
                    [1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15]
                ]  # the original hand order is 0,1, but in robot the order change to 1,0 for dexpilot, so for oculus we need to change back

            desired_joint_angles[-5:] = desired_thumb_angles

        return desired_joint_angles

    def vr_to_robot(self, armpoints):
        # wrist_position = np.dot(VR_TO_ROBOT,np.dot(LEFT_TO_RIGHT, armpoints[0]))
        # index_knuckle_coord = np.dot(VR_TO_ROBOT,np.dot(LEFT_TO_RIGHT, armpoints[1]))
        # pinky_knuckle_coord = np.dot(VR_TO_ROBOT,np.dot(LEFT_TO_RIGHT, armpoints[2]))
        # important! do not count use coordinate in world space!
        armpoints[0] = np.average(armpoints, axis=0)  # use mean as palm center position
        wrist_position = np.dot(LEFT_TO_RIGHT, armpoints[0])
        index_knuckle_coord = np.dot(LEFT_TO_RIGHT, armpoints[1] - armpoints[0])
        pinky_knuckle_coord = np.dot(LEFT_TO_RIGHT, armpoints[2] - armpoints[0])

        palm_normal = normalize_vector(
            np.cross(index_knuckle_coord, pinky_knuckle_coord)
        )  # Current Z
        palm_direction = normalize_vector(
            index_knuckle_coord + pinky_knuckle_coord
        )  # Current Y
        cross_product = normalize_vector(
            np.cross(palm_direction, palm_normal)
        )  # Current X
        # print(f'index_knuckle_coord: {index_knuckle_coord}, pinky_knuckle_coord: {pinky_knuckle_coord}')
        # print(f'palm_normal: {palm_normal}, palm_direction: {palm_direction}')
        return wrist_position, cross_product, palm_direction, palm_normal

    def _retarget_base(self):
        hand_center, hand_x, hand_y, hand_z = self.vr_to_robot(self.arm_coords)
        points_in_hand_space = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        points_in_vr_sapce = np.array(
            [
                hand_center,
                hand_center + hand_x,
                hand_center + hand_y,
                hand_center + hand_z,
            ]
        )

        # print('init_points_in_vr_space', init_points_in_vr_space)
        # print('points_in_vr_sapce', points_in_vr_sapce)

        vr2init_hand_transformation, init_vr2hand_rotation, init_vr2hand_translation = (
            best_fit_transform(self.init_points_in_vr_space, points_in_hand_space)
        )
        hand2vr_transformation, vr2hand_rotation, vr2hand_translation = (
            best_fit_transform(points_in_hand_space, points_in_vr_sapce)
        )
        new_hand2init_hand = vr2init_hand_transformation @ hand2vr_transformation
        init_flange2base = self.init_arm_transformation_matrix
        init_leap2base = init_flange2base @ self.leap2flange
        new_ee_transformation_matrix = (
            init_leap2base @ new_hand2init_hand @ np.linalg.inv(self.leap2flange)
        )

        # print('new_hand2init_hand:', '\n', new_hand2init_hand)
        # print('init_leap2base', '\n', init_leap2base)
        # print('new_ee_transformation_matrix', '\n', new_ee_transformation_matrix)
        # print('leap2flange', leap2flange)
        # print(hand_palm_direction, hand_palm_normal)
        # hand_wrist_rel_pos = hand_wrist_position-self.init_hand_wrist_position
        # points = np.array([hand_wrist_rel_pos, hand_wrist_rel_pos+hand_palm_normal, hand_wrist_rel_pos+hand_palm_direction])
        # transfomation, rotation, translation = best_fit_transform(self.init_points, points)

        # points_tran = np.ones((self.init_points.shape[0],4))
        # points_tran[:,:3] = self.init_points
        # print(((transfomation@points_tran.T).T)[:,:3]-points, R.from_matrix(rotation).as_euler('xyz'), translation)
        # convert rotation vector to rotation matrix
        # new_hand2init_hand = np.linalg.inv(transfomation)

        # new_ee_transformation_matrix = (self.init_arm_transformation_matrix @ (self.fake_to_ee_transformation_matrix@transfomation)) @ np.linalg.inv(self.fake_to_ee_transformation_matrix)

        composed_translation = new_ee_transformation_matrix[:3, 3]
        composed_rotation = new_ee_transformation_matrix[:3, :3]

        # convert rotation matrix to rotation vector
        composed_rotation = R.from_matrix(composed_rotation).as_euler("xyz")
        new_arm_pose = self.robot.arm.get_tcp_position()
        new_arm_pose[:3] = composed_translation * 1000
        new_arm_pose[3:6] = composed_rotation

        # print('current_arm_pose', self.robot.arm.get_tcp_position())
        # print('new_arm_pose', new_arm_pose)

        return new_arm_pose

    def _filter(self, desired_hand_joint_angles):
        desired_hand_joint_angles = (
            desired_hand_joint_angles * SMOOTH_FACTOR
            + self.prev_hand_joint_angles * (1 - SMOOTH_FACTOR)
        )
        self.prev_hand_joint_angles = desired_hand_joint_angles
        return desired_hand_joint_angles

    def motion(self, finger_configs):
        desired_cmd = []

        if ARM_TYPE is not None:
            desired_arm_pose = self._retarget_base()
            desired_cmd = np.concatenate([desired_cmd, desired_arm_pose])

        if HAND_TYPE is not None:
            desired_hand_joint_angles = self._retarget_hand(finger_configs)
            desired_hand_joint_angles = self._filter(desired_hand_joint_angles)
            desired_cmd = np.concatenate([desired_cmd, desired_hand_joint_angles])

        return desired_cmd

    def _calibrate_vr_arm_bounds(self):
        inital_frame_number = 1  # set to 50 will cause collision
        initial_hand_centers = []
        initial_hand_xs = []
        initial_hand_ys = []
        initial_hand_zs = []

        initial_arm_poss = []
        inital_arm_rots = []
        frame_number = 0

        while frame_number < inital_frame_number:
            print("calibration initial pose, id: ", frame_number)
            hand_center, hand_x, hand_y, hand_z = self.vr_to_robot(self.arm_coords)
            initial_hand_centers.append(hand_center)
            initial_hand_xs.append(hand_x)
            initial_hand_ys.append(hand_y)
            initial_hand_zs.append(hand_z)

            initial_arm_poss.append(
                np.array(self.robot.arm.get_tcp_position()[:3]) / 1000
            )
            inital_arm_rots.append(np.array(self.robot.arm.get_tcp_position()[3:6]))

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
        self.init_arm_rot = np.mean(inital_arm_rots, axis=0)
        self.init_arm_transformation_matrix = np.eye(4)
        self.init_arm_transformation_matrix[:3, :3] = R.from_euler(
            "xyz", self.init_arm_rot
        ).as_matrix()
        self.init_arm_transformation_matrix[:3, 3] = self.init_arm_pos.reshape(3)

    def move(self, finger_configs):
        print(
            "\n******************************************************************************"
        )
        print("     Controller initiated. ")
        print(
            "******************************************************************************\n"
        )
        print("Start controlling the robot hand using the Oculus Headset.\n")

        while True:
            if (
                self.hand_coords is not None
                and self.robot.get_hand_position() is not None
            ):
                # Obtaining the desired angles
                desired_cmd = self.motion(finger_configs)
                # Move the hand based on the desired angles
                self.robot.move(desired_cmd)
