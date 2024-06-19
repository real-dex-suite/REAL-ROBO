import rospy
from std_msgs.msg import Float64MultiArray

from holodex.utils.files import *
from holodex.utils.vec_ops import coord_in_bound, best_fit_transform, normalize_vector
from holodex.constants import *
from copy import deepcopy as copy
from scipy.spatial.transform import Rotation as R

from .robot import RobotController

# load constants according to hand type
hand_type = HAND_TYPE.lower()
JOINTS_PER_FINGER = eval(f'{hand_type.upper()}_JOINTS_PER_FINGER')
JOINT_OFFSETS = eval(f'{hand_type.upper()}_JOINT_OFFSETS')

def get_mano_coord_frame(keypoint_3d_array, oculus=False):
        """
        Compute the 3D coordinate frame (orientation only) from detected 3d key points
        :param points: keypoint3 detected from MediaPipe detector. Order: [wrist, index, middle, pinky]
        :return: the coordinate frame of wrist in MANO convention
        """
        if oculus:
            assert keypoint_3d_array.shape == (24, 3)
            points = keypoint_3d_array[[0, 6, 9], :] # TODO check if this is correct
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

        rospy.Subscriber(HAMER_HAND_TRANSFORM_COORDS_TOPIC, Float64MultiArray, self._callback_hand_coords, queue_size = 1)
        rospy.Subscriber(HAMER_ARM_TRANSFORM_COORDS_TOPIC, Float64MultiArray, self._callback_arm_coords, queue_size = 1)

        # Initializing the robot controller
        self.robot = RobotController(teleop = True)
        # Initializing the solvers
        self.fingertip_solver = self.robot.hand_KDLControl
        self.finger_joint_solver = self.robot.hand_JointControl
        
        # Initialzing the moving average queues
        self.moving_average_queues = {
            'thumb': [],
            'index': [],
            'middle': [],
            'ring': []
        }

        self.prev_hand_joint_angles = self.robot.get_hand_position()

        if RETARGET_TYPE == 'dexpilot':
            from holodex.components.retargeting.retargeting_config import RetargetingConfig

            config_path = f"holodex/components/retargeting/configs/teleop/{HAND_TYPE.lower()}_hand_right_{RETARGET_TYPE}.yml"
            RetargetingConfig.set_default_urdf_dir("holodex/robot/hand")
            self.retargeting = RetargetingConfig.load_from_file(config_path).build()
        
        if ARM_TYPE is not None:
            self._calibrate_vr_arm_bounds()
            if ARM_TYPE == "Jaka":
                # TODO configureable
                self.leap2flange = np.eye(4)
                self.leap2flange[:3, :3] = R.from_euler('xyz', [0, 0, 214.5], degrees=True).as_matrix()

    def _callback_hand_coords(self, coords):
        self.hand_coords = np.array(list(coords.data)).reshape(HAMER_NUM_KEYPOINTS, 3)
    
    def _callback_arm_coords(self, coords):
        self.arm_coords = np.array(list(coords.data)).reshape(HAMER_ARM_NUM_KEYPOINTS, 3)

    def _retarget_hand(self, finger_configs):
        if RETARGET_TYPE == 'dexpilot':
            indices = self.retargeting.optimizer.target_link_human_indices
            origin_indices = indices[0, :]
            task_indices = indices[1, :]
            ref_value = self.hand_coords[task_indices, :] - self.hand_coords[origin_indices, :]
            desired_joint_angles = self.retargeting.retarget(ref_value)

        return desired_joint_angles
    
    def vr_to_robot(self, armpoints):
        # wrist_position = np.dot(VR_TO_ROBOT,np.dot(LEFT_TO_RIGHT, armpoints[0]))
        # index_knuckle_coord = np.dot(VR_TO_ROBOT,np.dot(LEFT_TO_RIGHT, armpoints[1]))
        # pinky_knuckle_coord = np.dot(VR_TO_ROBOT,np.dot(LEFT_TO_RIGHT, armpoints[2]))
        # important! do not count use coordinate in world space!
        # armpoints[:,0]*=1.5
        # armpoints[:,1]*=1.5
        # armpoints[:,2]/=15
        armpoints[0] = np.average(armpoints, axis=0) # use mean as palm center position
        wrist_position = armpoints[0]
        index_knuckle_coord = armpoints[1] - armpoints[0]
        pinky_knuckle_coord = armpoints[2] - armpoints[0]
        
        palm_normal = normalize_vector(np.cross(index_knuckle_coord, pinky_knuckle_coord))   # Current Z
        palm_direction = normalize_vector(index_knuckle_coord + pinky_knuckle_coord)  # Current Y 
        cross_product = normalize_vector(np.cross(palm_direction, palm_normal))                # Current X
        # print(f'index_knuckle_coord: {index_knuckle_coord}, pinky_knuckle_coord: {pinky_knuckle_coord}')
        # print(f'palm_normal: {palm_normal}, palm_direction: {palm_direction}')
        return wrist_position, cross_product, palm_direction, palm_normal
    

    def _retarget_base(self):
        hand_center, hand_x, hand_y, hand_z = self.vr_to_robot(self.arm_coords)
        points_in_hand_space = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        points_in_vr_sapce = np.array([
            hand_center,
            hand_center + hand_x,
            hand_center + hand_y,
            hand_center + hand_z
        ])
        
        # print('init_points_in_vr_space', init_points_in_vr_space)
        # print('points_in_vr_sapce', points_in_vr_sapce)

        vr2init_hand_transformation, init_vr2hand_rotation, init_vr2hand_translation = best_fit_transform(self.init_points_in_vr_space, points_in_hand_space)
        
        hand2vr_transformation, vr2hand_rotation, vr2hand_translation = best_fit_transform(points_in_hand_space, points_in_vr_sapce)

        new_hand2init_hand = vr2init_hand_transformation @ hand2vr_transformation        
        init_flange2base = self.init_arm_transformation_matrix
        init_leap2base = init_flange2base @ self.leap2flange
        new_ee_transformation_matrix = init_leap2base @ new_hand2init_hand @ np.linalg.inv(self.leap2flange)

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

        composed_translation = new_ee_transformation_matrix[:3,3]
        composed_rotation = new_ee_transformation_matrix[:3,:3]

        # convert rotation matrix to rotation vector
        composed_rotation = R.from_matrix(composed_rotation).as_euler('xyz')
        new_arm_pose = self.robot.arm.get_tcp_position()
        new_arm_pose[:3] = composed_translation*1000
        new_arm_pose[3:6] = composed_rotation
        
        # print('current_arm_pose', self.robot.arm.get_tcp_position())
        # print('new_arm_pose', new_arm_pose)

        return new_arm_pose
    
    def _filter(self, desired_hand_joint_angles):
        desired_hand_joint_angles = desired_hand_joint_angles * SMOOTH_FACTOR + self.prev_hand_joint_angles * (1 - SMOOTH_FACTOR)
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
        inital_frame_number = 1 # set to 50 will cause collision
        initial_hand_centers = []
        initial_hand_xs = []
        initial_hand_ys = []
        initial_hand_zs = []

        initial_arm_poss = []
        inital_arm_rots = []
        frame_number =0

        while frame_number < inital_frame_number:
            print('calibration initial pose, id: ', frame_number)
            hand_center, hand_x, hand_y, hand_z = self.vr_to_robot(self.arm_coords)
            initial_hand_centers.append(hand_center)
            initial_hand_xs.append(hand_x)
            initial_hand_ys.append(hand_y)
            initial_hand_zs.append(hand_z)
            
            initial_arm_poss.append(np.array(self.robot.arm.get_tcp_position()[:3])/1000)
            inital_arm_rots.append(np.array(self.robot.arm.get_tcp_position()[3:6]))

            frame_number += 1
        
        init_hand_center = np.mean(initial_hand_centers,axis=0)
        init_hand_x = np.mean(initial_hand_xs,axis=0)
        init_hand_y = np.mean(initial_hand_ys,axis=0)
        init_hand_z = np.mean(initial_hand_zs,axis=0)

        self.init_points_in_vr_space = np.array([
            init_hand_center,
            init_hand_center + init_hand_x,
            init_hand_center + init_hand_y,
            init_hand_center + init_hand_z
        ])

        self.init_arm_pos = np.mean(initial_arm_poss,axis=0)
        self.init_arm_rot = np.mean(inital_arm_rots,axis=0)
        self.init_arm_transformation_matrix = np.eye(4)
        self.init_arm_transformation_matrix[:3,:3] = R.from_euler('xyz', self.init_arm_rot).as_matrix()
        self.init_arm_transformation_matrix[:3,3] = self.init_arm_pos.reshape(3)


    def move(self, finger_configs):
        print("\n******************************************************************************")
        print("     Controller initiated. ")
        print("******************************************************************************\n")
        print("Start controlling the robot hand using the Hamer Framework.\n")

        while True:
            if self.hand_coords is not None and self.robot.get_hand_position() is not None:
                # Obtaining the desired angles
                desired_joint_angles = self.motion(finger_configs)
                # Move the hand based on the desired angles
                self.robot.move(desired_joint_angles)