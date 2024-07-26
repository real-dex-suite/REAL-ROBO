import numpy as np
import rospy
# import pyrealsense2 as rs
import pyrealsense2.pyrealsense2 as rs

from holodex.utils.network import ImagePublisher, FloatArrayPublisher
from holodex.utils.images import *
from holodex.constants import *
from holodex.camera.realsense_helper import get_profiles


class RealSenseRobotStream(object):
    def __init__(self, cam_serial_num, robot_cam_num, rotation_angle = 0, mode='rgbd'):
        self.mode = mode
        self.cam_serial_num = cam_serial_num

        # Initializing ROS Node
        rospy.init_node('robot_cam_{}_stream'.format(robot_cam_num))

        # Disabling scientific notations
        np.set_printoptions(suppress=True)

        # Creating ROS Publishers
        self.color_image_publisher = ImagePublisher(publisher_name = '/robot_camera_{}/color_image'.format(robot_cam_num), color_image = True)
        if self.mode == 'rgbd':
            self.depth_image_publisher = ImagePublisher(publisher_name = '/robot_camera_{}/depth_image'.format(robot_cam_num), color_image = False)
        self.intrinsics_publisher = FloatArrayPublisher(publisher_name = '/robot_camera_{}/intrinsics'.format(robot_cam_num))

        # Setting rotation settings
        self.rotation_angle = rotation_angle

        # Setting ROS frequency
        self.rate = rospy.Rate(CAM_FPS)

        # Starting the realsense camera stream
        self._start_realsense(PROCESSING_PRESET)    
        
        print(f"Started the Realsense pipeline for camera: {self.cam_serial_num}!")

    def _start_realsense(self, processing_preset):
        
        config = rs.config()
        pipeline = rs.pipeline()
        config.enable_device(self.cam_serial_num)


        # Enabling camera streams
        if self.cam_serial_num == "211422061450": # D435
            config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, CAM_FPS)
            if self.mode == 'rgbd':
                config.enable_stream(rs.stream.depth, WIDTH, HEIGHT,rs.format.z16, CAM_FPS)

        if self.cam_serial_num == "f1231617": # L515
            color_profiles, depth_profiles = get_profiles(self.cam_serial_num)

            w, h, fps, fmt = color_profiles[26]
            config.enable_stream(rs.stream.color, w, h, fmt, fps)

            if self.mode == 'rgbd':
                w, h, fps, fmt = depth_profiles[0]
                config.enable_stream(rs.stream.depth, w, h, fmt, fps)


        # Starting the pipeline
        cfg = pipeline.start(config)
        device = cfg.get_device()

        if self.cam_serial_num == "211422061450": # D435
            device.hardware_reset()


        # if self.mode == 'rgbd':
        #     # Setting the depth mode to high accuracy mode
            # depth_sensor = device.first_depth_sensor()
            # depth_sensor.set_option(rs.option.visual_preset, processing_preset) # High accuracy post-processing mode
        self.realsense = pipeline

        # Obtaining the color intrinsics matrix for aligning the color and depth images
        profile = pipeline.get_active_profile()
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        intrinsics = color_profile.get_intrinsics()
        self.intrinsics_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy], 
            [0, 0, 1]
        ])
        # print(f"Camera {self.cam_serial_num} intrinsics matrix: {self.intrinsics_matrix}")

        # Align function - aligns other frames with the color frame
        self.align = rs.align(rs.stream.color)

    def get_rgb_depth_images(self):
        frames = None

        while frames is None:
            # Obtaining and aligning the frames
            frames = self.realsense.wait_for_frames()
            aligned_frames = self.align.process(frames)

            if self.mode == 'rgbd':
                aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Getting the images from the frames
            if self.mode == 'rgbd':
                if self.cam_serial_num == "f1231617": 
                    depth_image = (
                        np.asanyarray(aligned_depth_frame.get_data()) // 4
                    )  # L515 camera need to divide by 4 to get metric in meter   
                elif self.cam_serial_num == "211422061450":
                    depth_image = np.asanyarray(aligned_depth_frame.get_data()) 
            color_image = np.asanyarray(color_frame.get_data())

            if self.mode == 'rgbd':
                return color_image, depth_image
            else:
                return color_image
            
    def stream(self):
        print("Starting stream!\n")
        while True:
            if self.mode == 'rgbd':
                color_image, depth_image = self.get_rgb_depth_images()
                color_image, depth_image = rotate_image(color_image, self.rotation_angle), rotate_image(depth_image, self.rotation_angle)

                # Publishing the original color and depth images
                self.color_image_publisher.publish(color_image)
                self.depth_image_publisher.publish(depth_image) 
            elif self.mode == 'rgb':
                color_image = self.get_rgb_depth_images()
                color_image = rotate_image(color_image, self.rotation_angle)

                # Publishing the original color image
                self.color_image_publisher.publish(color_image)

            # Publishing the intrinsics of the camera
            self.intrinsics_publisher.publish(self.intrinsics_matrix.reshape(9).tolist())

            self.rate.sleep()