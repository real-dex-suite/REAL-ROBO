import mediapipe as mp
import mediapipe.framework as framework
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import hands_connections
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.hands import HandLandmark

OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ]
)


class SingleHandDetector:
    def __init__(self, hand_type="Right", min_detection_confidence=0.8, min_tracking_confidence=0.8, selfie=False):
        self.hand_detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.selfie = selfie
        self.operator2mano = OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT
        inverse_hand_dict = {"Right": "Left", "Left": "Right"}
        self.detected_hand_type = hand_type if selfie else inverse_hand_dict[hand_type]

    @staticmethod
    def draw_skeleton_on_image(image, keypoint_2d: landmark_pb2.NormalizedLandmarkList, style="white"):
        if style == "default":
            mp.solutions.drawing_utils.draw_landmarks(
                image,
                keypoint_2d,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style(),
            )
        elif style == "white":
            landmark_style = {}
            for landmark in HandLandmark:
                landmark_style[landmark] = DrawingSpec(color=(255, 48, 48), circle_radius=4, thickness=-1)

            connections = hands_connections.HAND_CONNECTIONS
            connection_style = {}
            for pair in connections:
                connection_style[pair] = DrawingSpec(thickness=2)

            mp.solutions.drawing_utils.draw_landmarks(
                image, keypoint_2d, mp.solutions.hands.HAND_CONNECTIONS, landmark_style, connection_style
            )

        return image

    def detect(self, rgb, depth=None, realsense_controller=None):
        results = self.hand_detector.process(rgb)
        if not results.multi_hand_landmarks:
            return 0, None, None, None, None

        desired_hand_num = -1
        for i in range(len(results.multi_hand_landmarks)):
            label = results.multi_handedness[i].ListFields()[0][1][0].label
            if label == self.detected_hand_type:
                desired_hand_num = i
                break
        if desired_hand_num < 0:
            return 0, None, None, None, None

        keypoint_3d = results.multi_hand_world_landmarks[desired_hand_num]
        keypoint_2d = results.multi_hand_landmarks[desired_hand_num]
        num_box = len(results.multi_hand_landmarks)

        # Parse 3d keypoints from MediaPipe hand detector
        if realsense_controller is not None:
            keypoint_3d_array = self.get_keypoints(keypoint_2d, depth, realsense_controller)
        else:
            keypoint_3d_array = self.parse_keypoint_3d(keypoint_3d)
        keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :]
        mediapipe_wrist_rot = self.estimate_frame_from_hand_points(keypoint_3d_array)
        joint_pos = keypoint_3d_array @ mediapipe_wrist_rot @ self.operator2mano

        return num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot, results

    def transform_keypoints_to_jointpos(self, keypoint_3d_array):
        keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :]
        mediapipe_wrist_rot = self.estimate_frame_from_hand_points(keypoint_3d_array)
        joint_pos = keypoint_3d_array @ mediapipe_wrist_rot @ self.operator2mano
        return joint_pos
    
    def get_keypoints(self, hand_landmarks, depth_frame, realsense_controller):
        keypoint = np.empty([21, 3])
        for idx, point in enumerate(hand_landmarks.landmark):
            # Getting the pixel value
            x_pixel, y_pixel = int(realsense_controller.resolution[0] * point.x), int(realsense_controller.resolution[1] * point.y)

            # Obtaining the depth value for the wrist joint
            new_depth = realsense_controller._get_depth_value(x_pixel, y_pixel, depth_frame)

            # # Using the temporal moving average filter
            if new_depth > 0:
                keypoint[idx][2] = realsense_controller._temporal_depth_average(keypoint, idx, new_depth, alpha = realsense_controller.alpha)

            # Finding the x and y coordinates
            keypoint[idx][0] = keypoint[idx][2] * (x_pixel - realsense_controller.intrinsics_matrix[0][2]) / realsense_controller.intrinsics_matrix[0][0]
            keypoint[idx][1] = keypoint[idx][2] * (y_pixel - realsense_controller.intrinsics_matrix[1][2]) / realsense_controller.intrinsics_matrix[1][1]
        return keypoint
    
    @staticmethod
    def parse_keypoint_3d(keypoint_3d: framework.formats.landmark_pb2.LandmarkList) -> np.ndarray:
        keypoint = np.empty([21, 3])
        for i in range(21):
            keypoint[i][0] = keypoint_3d.landmark[i].x
            keypoint[i][1] = keypoint_3d.landmark[i].y
            keypoint[i][2] = keypoint_3d.landmark[i].z
        return keypoint

    @staticmethod
    def parse_keypoint_2d(keypoint_2d: landmark_pb2.NormalizedLandmarkList, img_size) -> np.ndarray:
        keypoint = np.empty([21, 2])
        for i in range(21):
            keypoint[i][0] = keypoint_2d.landmark[i].x
            keypoint[i][1] = keypoint_2d.landmark[i].y
        keypoint = keypoint * np.array([img_size[1], img_size[0]])[None, :]
        return keypoint

    @staticmethod
    def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        """
        Compute the 3D coordinate frame (orientation only) from detected 3d key points
        :param points: keypoint3 detected from MediaPipe detector. Order: [wrist, index, middle, pinky]
        :return: the coordinate frame of wrist in MANO convention
        """
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

    def plot_world_landmarks(
            self,
            plt,
            ax_list,
            multi_hands_landmarks,
            multi_handedness,
            visibility_th=0.5,
        ):
            ax_list[0].cla()
            ax_list[0].set_xlim3d(-0.1, 0.1)
            ax_list[0].set_ylim3d(-0.1, 0.1)
            ax_list[0].set_zlim3d(-0.1, 0.1)
            ax_list[1].cla()
            ax_list[1].set_xlim3d(-0.1, 0.1)
            ax_list[1].set_ylim3d(-0.1, 0.1)
            ax_list[1].set_zlim3d(-0.1, 0.1)

            for landmarks, handedness in zip(multi_hands_landmarks, multi_handedness):
                handedness_index = 0
                if handedness.classification[0].label == 'Left':
                    handedness_index = 0
                elif handedness.classification[0].label == 'Right':
                    handedness_index = 1

                landmark_point = []

                for index, landmark in enumerate(landmarks.landmark):
                    landmark_point.append(
                        [landmark.visibility, (landmark.x, landmark.y, landmark.z)])

                palm_list = [0, 1, 5, 9, 13, 17, 0]
                thumb_list = [1, 2, 3, 4]
                index_finger_list = [5, 6, 7, 8]
                middle_finger_list = [9, 10, 11, 12]
                ring_finger_list = [13, 14, 15, 16]
                pinky_list = [17, 18, 19, 20]

                # 掌
                palm_x, palm_y, palm_z = [], [], []
                for index in palm_list:
                    point = landmark_point[index][1]
                    palm_x.append(point[0])
                    palm_y.append(point[2])
                    palm_z.append(point[1] * (-1))

                # 親指
                thumb_x, thumb_y, thumb_z = [], [], []
                for index in thumb_list:
                    point = landmark_point[index][1]
                    thumb_x.append(point[0])
                    thumb_y.append(point[2])
                    thumb_z.append(point[1] * (-1))

                # 人差し指
                index_finger_x, index_finger_y, index_finger_z = [], [], []
                for index in index_finger_list:
                    point = landmark_point[index][1]
                    index_finger_x.append(point[0])
                    index_finger_y.append(point[2])
                    index_finger_z.append(point[1] * (-1))

                # 中指
                middle_finger_x, middle_finger_y, middle_finger_z = [], [], []
                for index in middle_finger_list:
                    point = landmark_point[index][1]
                    middle_finger_x.append(point[0])
                    middle_finger_y.append(point[2])
                    middle_finger_z.append(point[1] * (-1))

                # 薬指
                ring_finger_x, ring_finger_y, ring_finger_z = [], [], []
                for index in ring_finger_list:
                    point = landmark_point[index][1]
                    ring_finger_x.append(point[0])
                    ring_finger_y.append(point[2])
                    ring_finger_z.append(point[1] * (-1))

                # 小指
                pinky_x, pinky_y, pinky_z = [], [], []
                for index in pinky_list:
                    point = landmark_point[index][1]
                    pinky_x.append(point[0])
                    pinky_y.append(point[2])
                    pinky_z.append(point[1] * (-1))

                ax_list[handedness_index].plot(palm_x, palm_y, palm_z)
                ax_list[handedness_index].plot(thumb_x, thumb_y, thumb_z)
                ax_list[handedness_index].plot(index_finger_x, index_finger_y,
                                            index_finger_z)
                ax_list[handedness_index].plot(middle_finger_x, middle_finger_y,
                                            middle_finger_z)
                ax_list[handedness_index].plot(ring_finger_x, ring_finger_y,
                                            ring_finger_z)
                ax_list[handedness_index].plot(pinky_x, pinky_y, pinky_z)

            plt.pause(.001)

            return