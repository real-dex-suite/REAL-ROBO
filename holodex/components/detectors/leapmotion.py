# import cv2
# import numpy as np
# import rospy
# from holodex.utils.network import *
# from holodex.constants import *

# import sys
# sys.path.append("/home/agibot/Packages/leap-sdk-python3")
# import Leap

# class LPHandDetector(object):
#     def __init__(self):
#         # Initializing the ROS node
#         rospy.init_node("leapmotion_keypoint_extractor")

#         # Disabling scientific notations
#         np.set_printoptions(suppress=True)
        
#         # Initializing Leapmotion pipeline
#         self._start_leapmotion()

#         # Creating an empty keypoint array
#         self.keypoints = np.empty((MP_NUM_KEYPOINTS, 3))

#         # Initializing ROS publishers
#         self.hand_keypoint_publisher = FloatArrayPublisher(publisher_name = LP_HAND_KEYPOINT_TOPIC)
#         self.arm_keypoint_publisher = FloatArrayPublisher(publisher_name = LP_ARM_KEYPOINT_TOPIC)

#     def _start_leapmotion(self):
#         self.controller = Leap.Controller()
#         print("Started the Leapmotion pipeline!")

#     def leap_vector_to_numpy(self, vector) -> np.ndarray:
#         """Converts a Leap Motion `Vector` to a numpy array."""
#         return np.array([vector.x, vector.y, vector.z])

#     def leap_motion_to_keypoints(self, hand) -> np.ndarray:
#         """Converts a Leap Motion `Hand` to a numpy array of keypoints."""
#         # print(hand.palm_position)
#         keypoints = np.zeros((21, 3))
#         armpoints = np.zeros((4, 3))
#         keypoints[0, :] = self.leap_vector_to_numpy(hand.wrist_position)

#         for finger in hand.fingers:
#             finger_index = finger.type
#             for bone_index in range(0, 4):
#                 bone = finger.bone(bone_index)
#                 index = 1 + finger_index * 4 + bone_index
#                 keypoints[index, :] = self.leap_vector_to_numpy(bone.next_joint)
#         armpoints[0, :] = self.leap_vector_to_numpy(hand.direction)
#         armpoints[1, :] = self.leap_vector_to_numpy(hand.palm_normal)
#         armpoints[2, :] = self.leap_vector_to_numpy(hand.wrist_position)
#         armpoints[3, :] = self.leap_vector_to_numpy(hand.palm_position)
#         return keypoints, armpoints

#     def stream(self):
#         # Starting the video loop
#         while True:
#             # Obtaining the leapmotion coords
#             frame = self.controller.frame()

#             if len(frame.hands) == 1:
#                 handpoints, armpoints = self.leap_motion_to_keypoints(frame.hands[0])
#                 # mm to m
#                 # TODO arm should be same? 
#                 armpoints = armpoints / 1000
#                 handpoints = handpoints / 1000
                
#                 # Publishing the detected data
#                 self.arm_keypoint_publisher.publish(armpoints.flatten().tolist())
#                 self.hand_keypoint_publisher.publish(handpoints.flatten().tolist())
#             else:
#                 continue