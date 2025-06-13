import os
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image,CameraInfo
from scipy.spatial.transform import Rotation as R
from tf.transformations import quaternion_matrix
import time
from cv_bridge import CvBridge
from flask import Flask, request, jsonify
from geometry_msgs.msg import Pose
import random
from holodex.components import RobotController
import spdlog
from holodex.constants import *

from std_msgs.msg import Float64MultiArray

rospy.init_node('targeting',anonymous=True)

class Targeting:
    def __init__(self,):
        self.logger = spdlog.ConsoleLogger("Targeting")
        
        self.robot_tcp_position_sub = rospy.Subscriber(KEYBOARD_EE_TOPIC, Float64MultiArray,self._read_tcp_position_sub)
        self.rgb_sub = rospy.Subscriber("/camera/color/image_raw", Image, self._bgr_callback)
        self.camera_info_sub = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self._camera_info_callback)
        self.aruco_rgb_pub = rospy.Publisher('/aruco_rgb', Image, queue_size=10)
        self.camera_info_loaded = False
        self._cv_bridge = CvBridge()
        self.target_pose_pub = rospy.Publisher('/target_pose',Pose,queue_size=10)

        self.marker_size = 0.078
        self.trans_mats = [None]
        self.acruco_id = 582


    def _read_tcp_position_sub(self, msg):
        self.cur_tcp_pose = np.array(msg.data)

    def _g2r_callback(self):
        ret = self.cur_tcp_pose
        rot_matrix = R.from_quat([ret[4], ret[5], ret[6], ret[3]]).as_matrix()
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rot_matrix
        transformation_matrix[:3, 3] = np.array(ret[0:3])
        print('transformation_matrix:', transformation_matrix)
        self.g2r = transformation_matrix

    def _camera_intrinsics_callback(self, msg):
        if not self.camera_info_loaded:
            print('msg:', msg)
            self.intrinsic_matrix = {'fx':msg.data[0],
                                    'fy':msg.data[4],
                                    'cx':msg.data[2],
                                    'cy':msg.data[5]}
            rospy.loginfo("Camera intrinsics loaded.")
            self.camera_info_loaded = True

    def _camera_info_callback(self, msg):
        if not self.camera_info_loaded:
            self.intrinsic_matrix = {'fx':msg.K[0],
                                    'fy':msg.K[4],
                                    'cx':msg.K[2],
                                    'cy':msg.K[5]}
            self.distortion_coefficients = np.array(msg.D)
            self.camera_info_loaded = True 

    def _bgr_callback(self, msg):
        self.origin_image = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.bgr_image = self.origin_image.copy()

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        aruco_params = cv2.aruco.DetectorParameters()

        corners, ids, rejected = cv2.aruco.detectMarkers(self.bgr_image, aruco_dict, parameters=aruco_params)
        mtx = np.array([[self.intrinsic_matrix['fx'], 0, self.intrinsic_matrix['cx']], [0, self.intrinsic_matrix['fy'], self.intrinsic_matrix['cy']], [0, 0, 1]]) # 替换为实际的相机内参 

        dist = np.array([0.,0.,0.,0.,0.]) # 替换为实际的畸变系数 # 对每个标记进行姿态估计 
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_size, mtx,dist) # 遍历每个标记，绘制坐标轴，并输出三维坐标 

        if ids is not None:
            self.trans_mats = []
            filter_corners = []
            filter_ids = []
            for i, marker_id in enumerate(ids): 
                if marker_id == self.acruco_id:
                    rvec, tvec = rvecs[i], tvecs[i]

                    R,_ = cv2.Rodrigues(rvec[0])

                    trans_mat = np.eye(4)
                    trans_mat[:3,:3] = R
                    trans_mat[:3,3] = tvec
                    cv2.drawFrameAxes(self.bgr_image, mtx, dist, rvec, tvec, 0.05,) 

                    self.trans_mats.append(trans_mat)
                    filter_corners.append(corners[i])
                    filter_ids.append(ids[i])

            image_markers = cv2.aruco.drawDetectedMarkers(self.bgr_image.copy(),filter_corners,np.array(filter_ids))
            self.aruco_rgb_pub.publish(self._cv_bridge.cv2_to_imgmsg(image_markers, encoding='bgr8'))
        else:
            self.logger.error("No AruCo markers detected.")
            self.aruco_rgb_pub.publish(self._cv_bridge.cv2_to_imgmsg(self.bgr_image, encoding='bgr8'))

    def vis_targeting(self):
        test = 1 # 檢測標定精度
        if test:
            # 加载相机到基座的变换矩阵
            T_camera_to_base = np.load('32views_c2r.npy')  # 确保路径正确

            # 计算基座到相机的变换矩阵（取逆）
            T_base_to_camera = np.linalg.inv(T_camera_to_base)

            # 定义基座坐标系的轴方向，在基座坐标系中
            axis_length = 0.1  # 坐标轴长度，单位：米
            axes_points_base = np.array([
                [0, 0, 0],  # 原点
                [axis_length, 0, 0],  # X轴
                [0, axis_length, 0],  # Y轴
                [0, 0, axis_length]   # Z轴
            ])

            # 转换为齐次坐标
            ones = np.ones((axes_points_base.shape[0], 1))
            axes_points_base_homogeneous = np.hstack([axes_points_base, ones])  # 形状：(4, 4)

            # 将轴方向从基座坐标系变换到相机坐标系
            axes_points_camera = (T_base_to_camera @ axes_points_base_homogeneous.T).T  # 形状：(4, 4)

            # 提取3D点坐标
            points_3D = axes_points_camera[:, :3]
            
            # 定义新的3D点
            # new_point_3D = np.array([-0.103, -0.398, 0.049])  # 基座坐标系中的点，替换为需要的坐标
            # new_point_3D_homogeneous = np.append(new_point_3D, 1)  # 转换为齐次坐标

            # 将新的3D点从基座坐标系变换到相机坐标系
            # new_point_camera = T_base_to_camera @ new_point_3D_homogeneous
            # new_point_camera_3D = new_point_camera[:3]  # 提取3D坐标

            # 相机内参
            fx = self.intrinsic_matrix['fx']
            fy = self.intrinsic_matrix['fy']
            cx = self.intrinsic_matrix['cx']
            cy = self.intrinsic_matrix['cy']

            # 内参矩阵
            mtx = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0,  0,  1]
            ])

            # 畸变系数（假设为零）
            dist = np.zeros(5)

            # 将3D点投影到2D图像平面
            rvec = np.zeros((3, 1))  # 旋转向量（为零）
            tvec = np.zeros((3, 1))  # 平移向量（为零）

            projected_points, _ = cv2.projectPoints(points_3D, rvec, tvec, mtx, dist)
            projected_points = projected_points.reshape(-1, 2)
            
            # 投影新3D点
            # projected_new_point, _ = cv2.projectPoints(new_point_camera_3D.reshape(1, 3), rvec, tvec, mtx, dist)
            # projected_new_point = tuple(projected_new_point.reshape(-1, 2)[0].astype(int))

            # 提取投影点坐标
            origin = tuple(projected_points[0].astype(int))
            x_axis = tuple(projected_points[1].astype(int))
            y_axis = tuple(projected_points[2].astype(int))
            z_axis = tuple(projected_points[3].astype(int))

            # 读取图像
            image = self.origin_image.copy()

            # 绘制坐标轴
            cv2.line(image, origin, x_axis, (0, 0, 255), 2)  # X轴，红色
            cv2.line(image, origin, y_axis, (0, 255, 0), 2)  # Y轴，绿色
            cv2.line(image, origin, z_axis, (255, 0, 0), 2)  # Z轴，蓝色

            # 绘制原点
            cv2.circle(image, origin, radius=5, color=(0, 0, 0), thickness=-1)
            
            # 绘制新投影点
            # cv2.circle(image, projected_new_point, radius=5, color=(255, 255, 0), thickness=-1)  # 黄色标记


            # 显示图像
            cv2.imshow("Base Position and Orientation", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if self.trans_mats == []:
            return None

        return self.trans_mats[0]

    def calibrate(self,):
        o2cs = []
        g2rs = []
        while True:
            flag = input('press y to end calibration,else continue:')
            if flag == 'y':
                break
            else:
                o2c = self.vis_targeting()
                # print('o2c: ', o2c)
                if o2c is not None:
                    o2cs.append(o2c)
                    self._g2r_callback()
                    g2r = self.g2r
                    g2rs.append(np.linalg.inv(g2r))
                    self.logger.info(f"Calibration data collected. {len(o2cs)} views.")
                else:
                    self.logger.error("No AruCo markers detected.")
        
            if len(o2cs) >=3:
                # 不同view数量都可视化一下看看
                R_gripper2base = [g[:3, :3] for g in g2rs]
                t_gripper2base = [g[:3, 3] for g in g2rs]
                R_obj2cam = [o[:3, :3] for o in o2cs]
                t_obj2cam = [o[:3, 3] for o in o2cs]
                
                R_cam2base, t_cam2base = cv2.calibrateHandEye(
                    R_gripper2base, t_gripper2base,
                    R_obj2cam, t_obj2cam,
                    method=cv2.CALIB_HAND_EYE_TSAI)   

                c2r = np.eye(4)
                c2r[:3,:3] = R_cam2base
                c2r[:3,3] = t_cam2base[:,0]

                self.logger.info(f"Current Calibration {len(o2cs)} views. c2r: {c2r}")

                np.save(f'{len(o2cs)}views_c2r.npy',c2r)

                g2c = np.linalg.inv(c2r) @ self.g2r
            else:
                g2c = np.eye(4)
                g2c[2,3] = 0.3

            target_msg = Pose()
            target_msg.position.x = g2c[0,3]
            target_msg.position.y = g2c[1,3]
            target_msg.position.z = g2c[2,3]
            quaternion = R.from_matrix(g2c[:3,:3]).as_quat()
            target_msg.orientation.x = quaternion[0]
            target_msg.orientation.y = quaternion[1]
            target_msg.orientation.z = quaternion[2]
            target_msg.orientation.w = quaternion[3]
            self.target_pose_pub.publish(target_msg)
        
        # np.save('o2cs.npy',np.stack(o2cs[:20]))
        # np.save('g2rs.npy',np.stack(g2rs[:20]))

    def calibrate_from_npy(self,views=5):
        for epoch in range(5):
            input('start!')
            sample_ids = random.sample(range(20),views)

            o2cs = np.load('o2cs.npy')
            g2rs = np.load('g2rs.npy')

            # 不同view数量都可视化一下看看
            R_gripper2base = [g2rs[i,:3, :3] for i in sample_ids]
            t_gripper2base = [g2rs[i,:3, 3] for i in sample_ids]
            R_obj2cam = [o2cs[i,:3, :3] for i in sample_ids]
            t_obj2cam = [o2cs[i,:3, 3] for i in sample_ids]

            R_cam2base, t_cam2base = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base,
            R_obj2cam, t_obj2cam,
            method=cv2.CALIB_HAND_EYE_PARK)   

            c2r = np.eye(4)
            c2r[:3,:3] = R_cam2base
            c2r[:3,3] = t_cam2base[:,0]

            if not os.path.exists(f'{views}'):
                os.mkdir(f'{views}')
            np.save(f'{views}/{epoch}_c2r.npy',c2r)

if __name__ == '__main__':
    targeting = Targeting()
    time.sleep(2)
    # targeting.calibrate_from_npy(20)
    targeting.calibrate()