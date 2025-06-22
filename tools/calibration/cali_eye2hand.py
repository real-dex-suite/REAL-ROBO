import os
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image, CameraInfo
from scipy.spatial.transform import Rotation as R
import time
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
import random
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped
import argparse
from frankapy import FrankaArm


class Targeting:
    def __init__(self, test_views=20):
        """Initialize Targeting class for camera-robot calibration"""
        # Initialize robot arm
        self.fa = FrankaArm(with_gripper=False)

        # Set up ROS subscribers
        self.rgb_sub = rospy.Subscriber(
            "/camera/color/image_raw", 
            Image, 
            self._bgr_callback
        )
        self.camera_info_sub = rospy.Subscriber(
            "/camera/color/camera_info",
            CameraInfo,
            self._camera_info_callback,
        )

        # Set up ROS publishers 
        self.aruco_rgb_pub = rospy.Publisher("/aruco_rgb", Image, queue_size=10)
        self.target_pose_pub = rospy.Publisher("/target_pose", Pose, queue_size=10)

        # Initialize camera and ArUco parameters
        self.camera_info_loaded = False
        self._cv_bridge = CvBridge()
        self.marker_size = 0.078  # ArUco marker size in meters
        self.trans_mats = [None]  # Transformation matrices
        self.aruco_id = 582       # ArUco marker ID
        self.test_views = test_views

    def _g2r_callback(self):
        """Callback function to get robot base to gripper transformation.
        
        Gets the current pose of the robot's end effector and converts it to a 4x4
        transformation matrix representing gripper to robot base transform.
        """
        tcp_pose = self.fa.get_pose()

        pose_array = [
            tcp_pose.translation[0],
            tcp_pose.translation[1],
            tcp_pose.translation[2],
            tcp_pose.quaternion[0],
            tcp_pose.quaternion[1],
            tcp_pose.quaternion[2],
            tcp_pose.quaternion[3],
        ]
        self.cur_tcp_pose = np.array(pose_array)
        print(self.cur_tcp_pose)

        ret = self.cur_tcp_pose
        rot_matrix = R.from_quat([ret[4], ret[5], ret[6], ret[3]]).as_matrix()
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rot_matrix
        transformation_matrix[:3, 3] = np.array(ret[0:3])
        print("transformation_matrix:", transformation_matrix)
        self.g2r = transformation_matrix

    def _camera_intrinsics_callback(self, msg):
        """Callback function to load camera intrinsics from CameraInfo message.
        
        Extracts camera intrinsic parameters from the CameraInfo message and stores them
        in a dictionary for later use.
        """
        if not self.camera_info_loaded:
            print("msg:", msg)
            self.intrinsic_matrix = {
                "fx": msg.data[0],
                "fy": msg.data[4],
                "cx": msg.data[2],
                "cy": msg.data[5],
            }
            rospy.loginfo("Camera intrinsics loaded.")
            self.camera_info_loaded = True

    def _camera_info_callback(self, msg):
        if not self.camera_info_loaded:
            self.intrinsic_matrix = {
                "fx": msg.K[0],
                "fy": msg.K[4],
                "cx": msg.K[2],
                "cy": msg.K[5],
            }
            self.distortion_coefficients = np.array(msg.D)
            self.camera_info_loaded = True

    def _bgr_callback(self, msg):
        self.origin_image = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.bgr_image = self.origin_image.copy()

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        aruco_params = cv2.aruco.DetectorParameters()

        corners, ids, rejected = cv2.aruco.detectMarkers(
            self.bgr_image, aruco_dict, parameters=aruco_params
        )

        # Camera matrix
        mtx = np.array(
            [
                [self.intrinsic_matrix["fx"], 0, self.intrinsic_matrix["cx"]],
                [0, self.intrinsic_matrix["fy"], self.intrinsic_matrix["cy"]],
                [0, 0, 1],
            ]
        ) 

        dist = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ) 
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_size, mtx, dist
        ) 

        if ids is not None:
            self.trans_mats = []
            filter_corners = []
            filter_ids = []
            for i, marker_id in enumerate(ids):
                if marker_id == self.acruco_id:
                    rvec, tvec = rvecs[i], tvecs[i]

                    R, _ = cv2.Rodrigues(rvec[0])

                    trans_mat = np.eye(4)
                    trans_mat[:3, :3] = R
                    trans_mat[:3, 3] = tvec
                    cv2.drawFrameAxes(
                        self.bgr_image,
                        mtx,
                        dist,
                        rvec,
                        tvec,
                        0.05,
                    )

                    self.trans_mats.append(trans_mat)
                    filter_corners.append(corners[i])
                    filter_ids.append(ids[i])

            image_markers = cv2.aruco.drawDetectedMarkers(
                self.bgr_image.copy(), filter_corners, np.array(filter_ids)
            )
            self.aruco_rgb_pub.publish(
                self._cv_bridge.cv2_to_imgmsg(image_markers, encoding="bgr8")
            )
        else:
            rospy.loginfo("No AruCo markers detected.")
            self.aruco_rgb_pub.publish(
                self._cv_bridge.cv2_to_imgmsg(self.bgr_image, encoding="bgr8")
            )

    def vis_targeting(self):
        test = 0
        if self.test_views > 0:
            test = 1  # Test calibration accuracy
        if test:
            # Load camera to base transformation matrix
            T_camera_to_base = np.load(f"calibration_results/{self.test_views}_views_c2r.npy")
            T_base_to_camera = np.linalg.inv(T_camera_to_base)

            # Define coordinate axes
            axis_length = 0.1
            axes_points_base = np.array([
                [0, 0, 0],           # Origin
                [axis_length, 0, 0], # X-axis
                [0, axis_length, 0], # Y-axis
                [0, 0, axis_length]  # Z-axis
            ])

            # Convert to homogeneous coordinates    
            ones = np.ones((axes_points_base.shape[0], 1))
            axes_points_base_homogeneous = np.hstack([axes_points_base, ones])

            # Transform points to camera frame
            axes_points_camera = (T_base_to_camera @ axes_points_base_homogeneous.T).T
            points_3D = axes_points_camera[:, :3]

            # Camera parameters
            mtx = np.array([
                [self.intrinsic_matrix["fx"], 0, self.intrinsic_matrix["cx"]],
                [0, self.intrinsic_matrix["fy"], self.intrinsic_matrix["cy"]],
                [0, 0, 1]
            ])
            dist = np.zeros(5)
            rvec = np.zeros((3, 1))
            tvec = np.zeros((3, 1))

            # Project 3D points to image
            projected_points, _ = cv2.projectPoints(points_3D, rvec, tvec, mtx, dist)
            projected_points = projected_points.reshape(-1, 2)

            # Extract point coordinates
            origin = tuple(projected_points[0].astype(int))
            x_axis = tuple(projected_points[1].astype(int))
            y_axis = tuple(projected_points[2].astype(int))
            z_axis = tuple(projected_points[3].astype(int))

            # Draw coordinate frame
            image = self.origin_image.copy()
            cv2.line(image, origin, x_axis, (0, 0, 255), 2)    # X-axis (red)
            cv2.line(image, origin, y_axis, (0, 255, 0), 2)    # Y-axis (green) 
            cv2.line(image, origin, z_axis, (255, 0, 0), 2)    # Z-axis (blue)
            cv2.circle(image, origin, radius=5, color=(0, 0, 0), thickness=-1)

            # Display result
            cv2.imshow("Base Position and Orientation", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if self.trans_mats == []:
            return None

        return self.trans_mats[0]

    def calibrate(self):
        o2cs = []  # Object to camera transforms
        g2rs = []  # Gripper to robot base transforms

        while True:
            flag = input("press q to end calibration, else continue: ")
            if flag == "q":
                break

            o2c = self.vis_targeting()
            if o2c is not None:
                o2cs.append(o2c)
                self._g2r_callback()
                g2rs.append(np.linalg.inv(self.g2r))
                rospy.loginfo(f"Calibration data collected. {len(o2cs)} views.")
            else:
                rospy.loginfo("No AruCo markers detected.")

            if len(o2cs) >= 3:
                # Extract rotation and translation components
                R_gripper2base = [g[:3, :3] for g in g2rs]
                t_gripper2base = [g[:3, 3] for g in g2rs]
                R_obj2cam = [o[:3, :3] for o in o2cs]
                t_obj2cam = [o[:3, 3] for o in o2cs]

                # Perform hand-eye calibration
                R_cam2base, t_cam2base = cv2.calibrateHandEye(
                    R_gripper2base,
                    t_gripper2base,
                    R_obj2cam,
                    t_obj2cam,
                    method=cv2.CALIB_HAND_EYE_TSAI
                )

                # Build transformation matrix
                c2r = np.eye(4)
                c2r[:3, :3] = R_cam2base
                c2r[:3, 3] = t_cam2base[:, 0]

                rospy.loginfo(f"Current Calibration {len(o2cs)} views. c2r: {c2r}")
                os.makedirs("calibration_results", exist_ok=True)
                np.save(f"calibration_results/{len(o2cs)}views_c2r.npy", c2r)

                g2c = np.linalg.inv(c2r) @ self.g2r
            else:
                g2c = np.eye(4)
                g2c[2, 3] = 0.3

        np.save('o2cs.npy',np.stack(o2cs[:20]))
        np.save('g2rs.npy',np.stack(g2rs[:20]))

@click.command()
@click.option("-t", "--test_views", default=20, help="Number of views to use for calibration")
def main(test_views):
    targeting = Targeting(test_views=test_views)
    time.sleep(2)
    targeting.calibrate()

if __name__ == "__main__":
    main()