import numpy as np
import rospy

from std_msgs.msg import Float64MultiArray
from holodex.utils.network import FloatArrayPublisher, frequency_timer
from holodex.utils.vec_ops import *
from holodex.constants import *


class TransformHandCoords(object):
    def __init__(self, detector_type, moving_average_limit = 1):
        # Initializing the ROS node
        rospy.init_node('hand_transformation_coords_{}'.format(detector_type))

        self.detector_type = detector_type

        # Initializing subscriber to get the raw keypoints
        self.hand_coords = None
        self.arm_coords = None

        if detector_type == 'MP':
            self.num_keypoints = MP_NUM_KEYPOINTS
            self.knuckle_points = (MP_JOINTS['knuckles'][0], MP_JOINTS['knuckles'][-1])
            rospy.Subscriber(MP_KEYPOINT_TOPIC, Float64MultiArray, self._callback_hand_coords, queue_size = 1)
            self.keypoint_publisher = FloatArrayPublisher(MP_HAND_TRANSFORM_COORDS_TOPIC)
        
        elif detector_type == 'LP':
            self.num_keypoints = LP_NUM_KEYPOINTS
            rospy.Subscriber(LP_HAND_KEYPOINT_TOPIC, Float64MultiArray, self._callback_hand_coords, queue_size = 1)
            self.keypoint_publisher = FloatArrayPublisher(LP_HAND_TRANSFORM_COORDS_TOPIC)
            if ARM_TYPE is not None:
                self.num_arm_keypoints = LP_ARM_NUM_KEYPOINTS
                rospy.Subscriber(LP_ARM_KEYPOINT_TOPIC, Float64MultiArray, self._callback_arm_coords, queue_size = 1)
                self.arm_keypoint_publisher = FloatArrayPublisher(LP_ARM_TRANSFORM_COORDS_TOPIC)

        elif detector_type == 'VR_RIGHT':
            self.num_keypoints = OCULUS_NUM_KEYPOINTS
            self.knuckle_points = (OCULUS_JOINTS['knuckles'][0], OCULUS_JOINTS['knuckles'][-1])
            rospy.Subscriber(VR_RIGHT_HAND_KEYPOINTS_TOPIC, Float64MultiArray, self._callback_hand_coords, queue_size = 1)
            self.keypoint_publisher = FloatArrayPublisher(VR_RIGHT_TRANSFORM_COORDS_TOPIC)

        elif detector_type == 'VR_LEFT':
            self.num_keypoints = OCULUS_NUM_KEYPOINTS
            self.knuckle_points = (OCULUS_JOINTS['knuckles'][0], OCULUS_JOINTS['knuckles'][-1])
            rospy.Subscriber(VR_LEFT_HAND_KEYPOINTS_TOPIC, Float64MultiArray, self._callback_hand_coords, queue_size = 1)
            self.keypoint_publisher = FloatArrayPublisher(VR_LEFT_TRANSFORM_DIR_TOPIC)
        
        else:
            raise NotImplementedError("There are no other detectors available. \
            The only options are Mediapipe or Leapmotion or Oculus!")

        # Setting the frequency to 30 Hz
        if detector_type == 'MP':
            self.frequency_timer = frequency_timer(MP_FREQ)  
        elif detector_type == 'LP':
            self.frequency_timer = frequency_timer(LP_FREQ)
        elif detector_type == 'VR_RIGHT' or 'VR_LEFT': 
            self.frequency_timer = frequency_timer(VR_FREQ)

        # Moving average queue
        self.moving_average_limit = moving_average_limit
        self.moving_average_queue = []

    def _callback_hand_coords(self, coords):
        self.hand_coords = np.array(list(coords.data)).reshape(self.num_keypoints, 3)
    
    def _callback_arm_coords(self, coords):
        self.arm_coords = np.array(list(coords.data)).reshape(self.num_arm_keypoints, 3)

    def _translate_coords(self, hand_coords):
        return hand_coords - hand_coords[0]

    def _get_coord_frame(self, index_knuckle_coord, pinky_knuckle_coord):
        palm_normal = normalize_vector(np.cross(index_knuckle_coord, pinky_knuckle_coord))   # Current Z
        palm_direction = normalize_vector(index_knuckle_coord + pinky_knuckle_coord)         # Current Y
        cross_product = normalize_vector(np.cross(palm_direction, palm_normal))                # Current X
        return [cross_product, palm_direction, palm_normal]
    
    def _get_mano_coord_frame(self, keypoint_3d_array, oculus=False):
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
    
    def transform_right_keypoints(self, hand_coords):
        translated_coords = self._translate_coords(hand_coords)
        original_coord_frame = self._get_coord_frame(translated_coords[self.knuckle_points[0]], translated_coords[self.knuckle_points[1]])

        # Finding the rotation matrix and rotating the coordinates
        rotation_matrix = np.linalg.solve(original_coord_frame, np.eye(3)).T
        transformed_coords = (rotation_matrix @ translated_coords.T).T
        return transformed_coords
    
    def transform_left_keypoints(self, hand_coords):
        translated_coords = self._translate_coords(hand_coords)
        original_coord_frame = self._get_coord_frame(translated_coords[self.knuckle_points[0]], translated_coords[self.knuckle_points[1]])

        translated_coord_frame = np.hstack([
            self.hand_coords[0], 
            original_coord_frame[0] + self.hand_coords[0], 
            original_coord_frame[1] + self.hand_coords[0],
            original_coord_frame[2] + self.hand_coords[0]
        ])
        return translated_coord_frame

    def transform_lp_right_keypoints(self, hand_coords, oculus=False):
        translated_coords = self._translate_coords(hand_coords)
        original_coord_frame = self._get_mano_coord_frame(translated_coords, oculus=oculus)

        transformed_coords = translated_coords @ original_coord_frame @ OPERATOR2MANO_RIGHT
        return transformed_coords

    def transform_lp_arm_keypoints(self, arm_coords):
        return arm_coords

    def stream(self):
        while True:
            if self.hand_coords is None:
                continue
            if ARM_TYPE is not None and self.arm_coords is None:
                continue
            # Shift the points to required axes
            if self.detector_type == "VR_LEFT":
                transformed_coords = self.transform_left_keypoints(self.hand_coords)
            elif self.detector_type == "LP":
                transformed_coords = self.transform_lp_right_keypoints(self.hand_coords)
                if ARM_TYPE is not None:
                    transformed_arm_coords = self.transform_lp_arm_keypoints(self.arm_coords)
            elif self.detector_type == "VR_RIGHT" or self.detector_type == "MP":
                transformed_coords = self.transform_right_keypoints(self.hand_coords)
                if RETARGET_TYPE == "dexpilot":
                    transformed_coords = self.transform_lp_right_keypoints(transformed_coords, oculus=True)
                    # transform coords around z axis for 180 degree
                    transformed_coords[:, 0] *= -1
            
            # TODO why moving average?
            if self.detector_type == "LP":
                averaged_coords = transformed_coords
            else:
                # Passing the transformed coords into a moving average
                averaged_coords = moving_average(transformed_coords, self.moving_average_queue, self.moving_average_limit)
            
            self.keypoint_publisher.publish(averaged_coords.flatten().tolist())

            if ARM_TYPE is not None:
                if self.detector_type == "LP":
                    averaged_arm_coords = transformed_arm_coords

                self.arm_keypoint_publisher.publish(averaged_arm_coords.flatten().tolist())

            self.frequency_timer.sleep()