import rospy
from std_msgs.msg import Float64MultiArray, Bool, Float64
from geometry_msgs.msg import Pose

from holodex.utils.files import *
from holodex.utils.vec_ops import coord_in_bound, best_fit_transform, normalize_vector
from holodex.constants import *
from copy import deepcopy as copy
from scipy.spatial.transform import Rotation as R
from termcolor import cprint
from typing import Tuple
import spdlog
import numpy as np

try:
    from .robot import RobotController
except ImportError:
    from robot import RobotController


class HamerDexArmTeleOp:
    def __init__(self):
        self.trans_scale = 1.5
        self.finger_distance = 0.0
        self.logger = spdlog.ConsoleLogger("RobotController")

        # Initialize state variables
        self.arm_ee_pose = None
        self.stop_move = False
        self.end_robot = False
        self.translation_state = None

        # Set up ROS subscribers
        self._setup_subscribers()

        # Initialize robot controller
        self.robot = RobotController(teleop=True)
        self.init_tcp = np.array(self._get_tcp_position())
        self.arm_ee_pose = self._get_tcp_position()

        # Calibrate arm bounds and set correct flange rotation if ARM_TYPE is defined
        if ARM_TYPE:
            self._calibrate_arm_bounds()
            self.correct_flange = np.eye(4)
            self._set_correct_flange_rotation()

    def _setup_subscribers(self):
        """Set up all ROS subscribers"""
        topics_callbacks = [
            (JAKA_EE_POSE_TOPIC, Float64MultiArray, self._callback_arm_ee_pose),
            ("/data_collector/reset_done", Bool, self._callback_reset_done),
            ("/data_collector/reset_robot", Bool, self._callback_reset_robot),
            ("/data_collector/stop_move", Bool, self._callback_stop_move),
            ("/data_collector/end_robot", Bool, self._callback_end_robot),
            ("vr/gripper", Float64, self._callback_finger_distance),
            ("vr/ee_pose", Pose, self._callback_ee_pose),
        ]
        for topic, msg_type, callback in topics_callbacks:
            rospy.Subscriber(topic, msg_type, callback, queue_size=1)

    def _get_tcp_position(self):
        """Get the TCP position based on the arm type"""
        if ARM_TYPE == "Flexiv":
            return self.robot.arm.get_tcp_position(euler=True, degree=False)
        elif "Franka" in ARM_TYPE:
            tcp_pose = self.robot.arm.get_tcp_position()  # w, x, y, z
            tcp_quat_wxyz = tcp_pose[3:7]
            tcp_quat_xyzw = [
                tcp_quat_wxyz[1],
                tcp_quat_wxyz[2],
                tcp_quat_wxyz[3],
                tcp_quat_wxyz[0],
            ]
            tcp_rot = R.from_quat(tcp_quat_xyzw).as_euler("xyz", degrees=False)
            return np.concatenate([tcp_pose[:3], tcp_rot])
        else:
            return self.robot.arm.get_tcp_position()

    def _callback_ee_pose(self, pose):
        """Callback function to update joystick pose from VR data
        
        Args:
            pose: Pose message containing VR end-effector pose

        Note:
            This is based on the assumption that the VR end-effector pose is in the left hand coordinate system.
            Please modify the callback function if the VR end-effector pose is in a different coordinate system.
        """
        self.joystick_pose = np.array(
            [
                pose.position.x,
                -pose.position.y,
                pose.position.z,
                0, 
                0, 
                0, 
                1,
                # -pose.orientation.x,
                # pose.orientation.y,
                # -pose.orientation.z,
                # pose.orientation.w,
            ]
        )

    def _callback_finger_distance(self, data):
        """Callback function to update finger distance from VR data"""
        self.finger_distance = np.array(data.data)

    def _callback_arm_ee_pose(self, data):
        """Callback function to update arm end-effector pose"""
        self.arm_ee_pose = np.array(data.data)

    def _callback_end_robot(self, msg):
        """Callback function to set end_robot flag"""
        self.end_robot = msg.data

    def _callback_stop_move(self, msg):
        """Callback function to set stop_move flag"""
        self.stop_move = msg.data

    def _callback_reset_robot(self, msg):
        """Callback function to reset robot position"""
        if msg.data:
            self.robot.home_robot()

    def _callback_reset_done(self, msg):
        """Callback function to handle reset done event"""
        self.robot.home_robot()
        if msg.data and ARM_TYPE:
            self._calibrate_arm_bounds()

    def vr_to_robot(self, pose):
        """Convert joystick end-effector pose to robot coordinates"""
        translation_vector = pose[:3]
        rotation_quat = pose[3:7]
        rotation_matrix = R.from_quat(rotation_quat).as_matrix()
        forward_vector = normalize_vector(rotation_matrix[:, 0])
        up_vector = normalize_vector(rotation_matrix[:, 1])
        side_vector = normalize_vector(rotation_matrix[:, 2])

        return translation_vector, side_vector, forward_vector, up_vector

    def _compute_transformation(self, init_hand_transformation, hand2vr_transformation):
        """Compute the transformation matrix for the robot arm"""
        new_hand2init_hand = init_hand_transformation @ hand2vr_transformation
        init_flange2base = self.init_arm_transformation_matrix @ self.correct_flange
        return init_flange2base @ new_hand2init_hand @ np.linalg.inv(self.correct_flange)

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
        base_center, x_vector, y_vector, z_vector = self.vr_to_robot(self.joystick_pose)

        reference_points_in_base = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        reference_points_in_vr = np.array(
            [
                base_center,
                base_center + x_vector,
                base_center + y_vector,
                base_center + z_vector,
            ]
        )

        vr_to_base_transformation, base_to_vr_transformation = self._get_transformation(
            reference_points_in_base, reference_points_in_vr
        )

        composed_transformation = self._compute_transformation(
            vr_to_base_transformation, base_to_vr_transformation
        )

        composed_translation = composed_transformation[:3, 3]
        composed_rotation = composed_transformation[:3, :3]
        composed_rotation_quat = R.from_matrix(composed_rotation).as_quat()
        current_arm_pose = self.arm_ee_pose

        current_arm_pose[:3] = composed_translation * self.trans_scale
        current_arm_pose[3:6] = R.from_quat(composed_rotation_quat).as_euler("xyz")

        return current_arm_pose

    def motion(self, finger_configs):
        """Generate motion commands for the robot"""
        if ARM_TYPE:
            desired_arm_pose = self._retarget_base()
            tmp_desired_arm_euler = desired_arm_pose[3:6]
            tmp_desired_arm_quat = self.robot.arm.eulerZYX2quat(tmp_desired_arm_euler)
            return np.concatenate([desired_arm_pose[:3], tmp_desired_arm_quat])
        return []

    def _calibrate_arm_bounds(self):
        """Calibrate the arm bounds based on initial positions"""
        initial_centers, initial_xs, initial_ys, initial_zs = (
            [],
            [],
            [],
            [],
        )
        initial_arm_poss, initial_arm_rots = [], []

        for _ in range(1):
            center, x, y, z = self.vr_to_robot(self.joystick_pose)
            initial_centers.append(center)
            initial_xs.append(x)
            initial_ys.append(y)
            initial_zs.append(z)

            initial_arm_poss.append(
                np.array(self._get_tcp_position()[:3]) / self.trans_scale
            )
            initial_arm_rots.append(np.array(self._get_tcp_position()[3:6]))

        avg_center = np.mean(initial_centers, axis=0)
        avg_x = np.mean(initial_xs, axis=0)
        avg_y = np.mean(initial_ys, axis=0)
        avg_z = np.mean(initial_zs, axis=0)

        self.init_points_in_vr_space = np.array(
            [
                avg_center,
                avg_center + avg_x,
                avg_center + avg_y,
                avg_center + avg_z,
            ]
        )

        self.init_arm_pos = np.mean(initial_arm_poss, axis=0)
        self.init_arm_rot = np.mean(initial_arm_rots, axis=0)
        self.init_arm_transformation_matrix = np.eye(4)
        self.init_arm_transformation_matrix[:3, :3] = R.from_euler(
            "xyz", self.init_arm_rot
        ).as_matrix()
        self.init_arm_transformation_matrix[:3, 3] = self.init_arm_pos

    def _set_correct_flange_rotation(self):
        """Set the correct flange rotation based on the arm type"""
        if "Franka" in ARM_TYPE:
            self.correct_flange[:3, :3] = R.from_euler(
                "xyz", [0, 0, 0], degrees=True
            ).as_matrix()
        else:
            self.correct_flange[:3, :3] = R.from_euler(
                "xyz", [0, 0, 90], degrees=True
            ).as_matrix()

    def move(self, finger_configs):
        """Main control loop for robot movement"""
        print("\n" + "*" * 78)
        cprint("[   ok   ]     Controller initiated. ", "green", attrs=["bold"])
        print("*" * 78 + "\n")
        print("Start controlling the robot hand using the Hamer Framework.\n")

        while True:
            if self.joystick_pose is not None:
                if self.stop_move:
                    continue
                if self.end_robot:
                    break

                # Generate desired joint angles based on current joystick pose
                desired_joint_angles = self.motion(finger_configs)
                x = np.array(self.robot.get_arm_tcp_position())
                np.set_printoptions(precision=5, suppress=True)
                print(f"current_joint{self.robot.get_arm_position()}")
                self.robot.move_arm(desired_joint_angles)

                # Move the gripper based on the current finger distance
                print(f"gripper: {self.finger_distance}")
                # self.robot.move_gripper(1 - self.finger_distance)
                self.robot.move_gripper(self.finger_distance)