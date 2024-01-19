from jaka.jaka import JakaArm
import os
import time
import sys

sys.path.append("/home/agibot/Packages/leap-sdk-python3")
import time

import Leap
import numpy as np
from scipy.spatial.transform import Rotation as R

import open3d as o3d

def leap_vector_to_numpy(vector) -> np.ndarray:
    """Converts a Leap Motion `Vector` to a numpy array."""
    return np.array([vector.x, vector.y, vector.z])


def leap_hand_to_keypoints(hand) -> np.ndarray:
    """Converts a Leap Motion `Hand` to a numpy array of keypoints."""
    # print(hand.palm_position)
    keypoints = np.zeros((21, 3))
    armpoints = np.zeros((4, 3))
    keypoints[0, :] = leap_vector_to_numpy(hand.wrist_position)

    for finger in hand.fingers:
        finger_index = finger.type
        for bone_index in range(0, 4):
            bone = finger.bone(bone_index)
            index = 1 + finger_index * 4 + bone_index
            keypoints[index, :] = leap_vector_to_numpy(bone.next_joint)
    armpoints[0, :] = leap_vector_to_numpy(hand.direction)
    armpoints[1, :] = leap_vector_to_numpy(hand.palm_normal)
    armpoints[2, :] = leap_vector_to_numpy(hand.wrist_position)
    armpoints[3, :] = leap_vector_to_numpy(hand.palm_position)
    return keypoints, armpoints

def leap_motion_to_robot(armpoints):
    direction = np.dot(SENSOR_TO_ROBOT,armpoints[0]/1000)
    palm_normal = np.dot(SENSOR_TO_ROBOT,armpoints[1]/1000)
    wrist_position = np.dot(SENSOR_TO_ROBOT,armpoints[2]/1000)
    palm_position = np.dot(SENSOR_TO_ROBOT,armpoints[3]/1000)
    return direction, palm_normal, wrist_position, palm_position

def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    """

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t

class SampleListener(Leap.Listener):
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    bone_names = ["Metacarpal", "Proximal", "Intermediate", "Distal"]
    state_names = ["STATE_INVALID", "STATE_START", "STATE_UPDATE", "STATE_END"]

    def __init__(self, rtde_c, rate: float = 10.0) -> None:
        super().__init__()
        self.rtde_c = rtde_c
        self.rate = rate
        self.latest = None
        self.last_action = None
        self.tran_max = 0.05
        self.rot_max = 0.349
        self.abort = False
        self.finish = False

    def on_init(self, controller):
        print("Initialized")

    def on_connect(self, controller):
        print("Connected")

    def on_disconnect(self, controller):
        # Note: not dispatched when running in a debugger.
        print("Disconnected")

    def on_exit(self, controller):
        print("Exited")

    def on_frame(self, controller):
        # print('manual')
        # Get the most recent frame and report some basic information
        frame = controller.frame()
        # collect_data(frame)

        # print("Frame id %d" % frame.id)
        for hand in frame.hands:
            keypoints, armpoints = leap_hand_to_keypoints(hand)
            print()

if __name__ == "__main__":
    robot = JakaArm()


        