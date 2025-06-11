import rospy
from std_msgs.msg import Float64MultiArray

# from .calibrators import LeapThumbBoundCalibrator

from holodex.utils.files import *
from holodex.utils.vec_ops import coord_in_bound, best_fit_transform
from holodex.constants import *
from copy import deepcopy as copy

from .robot import RobotController
from scipy.spatial.transform import Rotation as R


class LPDexArmTeleOp(object):
    def __init__(self, simulator=None):
        # Initializing the ROS Node
        rospy.init_node("lp_dexarm_teleop")

        # Storing the transformed hand coordinates
        self.hand_coords = None
        self.arm_coords = None
        rospy.Subscriber(
            LP_HAND_TRANSFORM_COORDS_TOPIC,
            Float64MultiArray,
            self._callback_hand_coords,
            queue_size=1,
        )
        rospy.Subscriber(
            LP_ARM_TRANSFORM_COORDS_TOPIC,
            Float64MultiArray,
            self._callback_arm_coords,
            queue_size=1,
        )

        # Initializing the robot controller
        self.robot = RobotController(teleop=True, simulator=simulator)
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

        if RETARGET_TYPE == "dexpilot":
            from holodex.components.retargeting.retargeting_config import (
                RetargetingConfig,
            )

            config_path = f"holodex/components/retargeting/configs/teleop/{HAND_TYPE.lower()}_hand_right_{RETARGET_TYPE}.yml"
            RetargetingConfig.set_default_urdf_dir("holodex/robot/hand")
            self.retargeting = RetargetingConfig.load_from_file(config_path).build()

        elif RETARGET_TYPE == "joint+spatial":
            # Calibrating to get the thumb bounds
            self._calibrate_bounds()

            # Getting the bounds for the robot hand
            robohand_bounds_path = get_path_in_package(
                "components/robot_operators/configs/{hand_type}_lp.yaml"
            )
            with open(robohand_bounds_path, "r") as file:
                self.robohand_bounds = yaml.safe_load(file)

        if ARM_TYPE is not None:
            self._calibrate_lp_arm_bounds()

    def _calibrate_bounds(self):
        print("***************************************************************")
        print("     Starting calibration process ")
        print("***************************************************************")
        calibrator = LeapThumbBoundCalibrator()
        self.thumb_index_bounds, self.thumb_middle_bounds, self.thumb_ring_bounds = (
            calibrator.get_bounds()
        )

    def _callback_hand_coords(self, coords):
        self.hand_coords = np.array(list(coords.data)).reshape(LP_NUM_KEYPOINTS, 3)

    def _callback_arm_coords(self, coords):
        self.arm_coords = np.array(list(coords.data)).reshape(LP_ARM_NUM_KEYPOINTS, 3)

    def _get_finger_coords(self, finger_type):
        return np.vstack(
            [self.hand_coords[0], self.hand_coords[LP_JOINTS[finger_type]]]
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
        else:
            # use allegro representation, -3.14
            desired_joint_angles = copy(self.robot.get_hand_position()) - 3.14

            # Movement for the index finger
            if not finger_configs["freeze_index"]:
                desired_joint_angles = self.finger_joint_solver.calculate_finger_angles(
                    finger_type="index",
                    finger_joint_coords=self._get_finger_coords("index"),
                    curr_angles=desired_joint_angles,
                    moving_avg_arr=self.moving_average_queues["index"],
                )
            else:
                for idx in range(self.robot.joints_per_finger):
                    if idx > 0:
                        desired_joint_angles[
                            idx + self.robot.joint_offsets["index"]
                        ] = 0.05
                    else:
                        desired_joint_angles[
                            idx + self.robot.joint_offsets["index"]
                        ] = 0

            # Movement for the middle finger
            if not finger_configs["freeze_middle"]:
                desired_joint_angles = self.finger_joint_solver.calculate_finger_angles(
                    finger_type="middle",
                    finger_joint_coords=self._get_finger_coords("middle"),
                    curr_angles=desired_joint_angles,
                    moving_avg_arr=self.moving_average_queues["middle"],
                )
            else:
                for idx in range(self.robot.joints_per_finger):
                    if idx > 0:
                        desired_joint_angles[
                            idx + self.robot.joint_offsets["middle"]
                        ] = 0.05
                    else:
                        desired_joint_angles[
                            idx + self.robot.joint_offsets["middle"]
                        ] = 0

            # Movement for the ring finger
            # Calculating the translatory joint angles
            desired_joint_angles = self.finger_joint_solver.calculate_finger_angles(
                finger_type="ring",
                finger_joint_coords=self._get_finger_coords("ring"),
                curr_angles=desired_joint_angles,
                moving_avg_arr=self.moving_average_queues["ring"],
            )

            if RETARGET_TYPE == "joint+spatial":
                # Movement for the thumb finger - we disable 3D motion just for the thumb
                if finger_configs["three_dim"]:
                    desired_joint_angles = self._get_3d_thumb_angles(
                        desired_joint_angles
                    )
                else:
                    desired_joint_angles = self._get_2d_thumb_angles(
                        desired_joint_angles
                    )
            elif RETARGET_TYPE == "joint":
                desired_joint_angles = self.finger_joint_solver.calculate_finger_angles(
                    finger_type="thumb",
                    finger_joint_coords=self._get_finger_coords("thumb"),
                    curr_angles=desired_joint_angles,
                    moving_avg_arr=self.moving_average_queues["thumb"],
                )

        return desired_joint_angles

    def _retarget_base(self):
        hand_direction, hand_palm_normal, hand_wrist_position, hand_palm_position = (
            self.leap_motion_to_robot(self.arm_coords)
        )
        hand_wrist_rel_pos = hand_wrist_position - self.init_hand_wrist_position
        points = np.array(
            [
                hand_wrist_rel_pos,
                hand_wrist_rel_pos + hand_palm_normal,
                hand_wrist_rel_pos + hand_direction,
            ]
        )
        transfomation, rotation, translation = best_fit_transform(
            self.init_points, points
        )

        # use open3d visualize points and init_points
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        # pcd_init = o3d.geometry.PointCloud()
        # pcd_init.points = o3d.utility.Vector3dVector(init_points)
        # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=[0, 0, 0])

        # o3d.visualization.draw_geometries([pcd,pcd_init,axis])

        # compute new pose
        # new_arm_transformation_matrix = np.dot(init_arm_transformation_matrix, transfomation)
        # new_arm_pose = robot.robot.get_tcp_position()[1]
        # # new_arm_pose = np.zeros(6)
        # new_arm_pose[:3] = new_arm_transformation_matrix[:3,3]*1000
        # new_arm_pose[3:6] = R.from_matrix(new_arm_transformation_matrix[:3,:3]).as_euler('xyz')

        # convert rotation vector to rotation matrix
        new_ee_transformation_matrix = (
            self.init_arm_transformation_matrix
            @ (self.fake_to_ee_transformation_matrix @ transfomation)
        ) @ np.linalg.inv(self.fake_to_ee_transformation_matrix)

        composed_translation = new_ee_transformation_matrix[:3, 3]
        composed_rotation = new_ee_transformation_matrix[:3, :3]

        # convert rotation matrix to rotation vector
        composed_rotation = R.from_matrix(composed_rotation).as_euler("xyz")
        new_arm_pose = self.robot.arm.get_tcp_position()
        new_arm_pose[:3] = composed_translation * 1000
        new_arm_pose[3:6] = composed_rotation

        return new_arm_pose

    def motion(self, finger_configs):
        desired_cmd = []

        if ARM_TYPE is not None:
            desired_arm_pose = self._retarget_base()
            desired_cmd = np.concatenate([desired_cmd, desired_arm_pose])

        if HAND_TYPE is not None:
            desired_hand_joint_angles = self._retarget_hand(finger_configs)
            desired_cmd = np.concatenate([desired_cmd, desired_hand_joint_angles])

        return desired_cmd

    def leap_motion_to_robot(self, armpoints):
        direction = np.dot(LP_TO_ROBOT, armpoints[0]) * LP_WORKSPACE_SCALE
        palm_normal = np.dot(LP_TO_ROBOT, armpoints[1]) * LP_WORKSPACE_SCALE
        wrist_position = np.dot(LP_TO_ROBOT, armpoints[2]) * LP_WORKSPACE_SCALE
        palm_position = np.dot(LP_TO_ROBOT, armpoints[3]) * LP_WORKSPACE_SCALE
        return direction, palm_normal, wrist_position, palm_position

    def _calibrate_lp_arm_bounds(self):
        inital_frame_number = 50
        initial_hand_directions = []
        initial_hand_palm_normals = []
        initial_hand_wrist_positions = []
        initial_hand_palm_positions = []

        initial_arm_poss = []
        inital_arm_rots = []
        frame_number = 0

        while frame_number < inital_frame_number:
            print("calibration initial pose, id: ", frame_number)
            (
                hand_direction,
                hand_palm_normal,
                hand_wrist_position,
                hand_palm_position,
            ) = self.leap_motion_to_robot(self.arm_coords)
            initial_hand_directions.append(hand_direction)
            initial_hand_palm_normals.append(hand_palm_normal)
            initial_hand_wrist_positions.append(hand_wrist_position)
            initial_hand_palm_positions.append(hand_palm_position)

            initial_arm_poss.append(
                np.array(self.robot.arm.get_tcp_position()[:3]) / 1000
            )
            inital_arm_rots.append(np.array(self.robot.arm.get_tcp_position()[3:6]))

            frame_number += 1

        self.init_hand_direction = np.mean(initial_hand_directions, axis=0)
        self.init_hand_palm_normal = np.mean(initial_hand_palm_normals, axis=0)
        self.init_hand_wrist_position = np.mean(initial_hand_wrist_positions, axis=0)
        # init_hand_palm_position = np.mean(initial_hand_palm_positions,axis=0)
        self.init_points = np.array(
            [
                self.init_hand_wrist_position * 0,
                self.init_hand_palm_normal,
                self.init_hand_direction,
            ]
        )  # set to zero TODO

        self.init_arm_pos = np.mean(initial_arm_poss, axis=0)
        self.init_arm_rot = np.mean(inital_arm_rots, axis=0)
        self.init_arm_transformation_matrix = np.eye(4)
        self.init_arm_transformation_matrix[:3, :3] = R.from_euler(
            "xyz", self.init_arm_rot
        ).as_matrix()
        self.init_arm_transformation_matrix[:3, 3] = self.init_arm_pos.reshape(3)

        # algin ee coordinate to fake coordinate consistent with human hand coordinate
        self.fake_to_ee_transformation_matrix = np.eye(4)
        self.fake_to_ee_transformation_matrix[:3, :3] = np.linalg.inv(
            self.init_arm_transformation_matrix[:3, :3]
        )

    def move(self, finger_configs):
        print(
            "\n******************************************************************************"
        )
        print("     Controller initiated. ")
        print(
            "******************************************************************************\n"
        )
        print("Start controlling the robot hand using the Leapmotion.\n")

        while True:
            if (
                self.hand_coords is not None
                and self.robot.get_hand_position() is not None
            ):
                # Obtaining the desired angles
                desired_cmd = self.motion(finger_configs)
                # Move the hand based on the desired angles
                self.robot.move(desired_cmd)
