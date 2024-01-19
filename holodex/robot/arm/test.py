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
    controller = Leap.Controller()
    robot = JakaArm()
    R.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()

    SENSOR_TO_ROBOT = np.array([-1, 0, 0,
                                 0, 0, 1,
                                 0, 1, 0]).reshape(3, 3)

    inital_frame_number = 50
    initial_hand_directions = []
    initial_hand_palm_normals = []
    initial_hand_wrist_positions = []
    initial_hand_palm_positions = []

    initial_arm_poss = []
    inital_arm_rots = []
    frame_number =0

    robot.robot.servo_move_enable(True)

    while frame_number < inital_frame_number:
        frame = controller.frame()
        # print("Frame id %d" % frame.id)
        hand = frame.hands[0]
        keypoints, armpoints = leap_hand_to_keypoints(hand)
        print('calibration initial pose, id: ', frame_number)
        hand_direction, hand_palm_normal, hand_wrist_position, hand_palm_position = leap_motion_to_robot(armpoints)
        initial_hand_directions.append(hand_direction)
        initial_hand_palm_normals.append(hand_palm_normal)
        initial_hand_wrist_positions.append(hand_wrist_position)
        initial_hand_palm_positions.append(hand_palm_position)
        
        initial_arm_poss.append(np.array(robot.robot.get_tcp_position()[1][:3])/1000)
        inital_arm_rots.append(np.array(robot.robot.get_tcp_position()[1][3:6]))

        init_hand_direction = np.mean(initial_hand_directions,axis=0)
        init_hand_palm_normal = np.mean(initial_hand_palm_normals,axis=0)
        init_hand_wrist_position = np.mean(initial_hand_wrist_positions,axis=0)
        # init_hand_palm_position = np.mean(initial_hand_palm_positions,axis=0)
        init_points = np.array([init_hand_wrist_position*0,init_hand_palm_normal,init_hand_direction])

        init_arm_pos = np.mean(initial_arm_poss,axis=0)
        init_arm_rot = np.mean(inital_arm_rots,axis=0)
        init_arm_transformation_matrix = np.eye(4)
        init_arm_transformation_matrix[:3,:3] = R.from_euler('xyz', init_arm_rot).as_matrix()
        init_arm_transformation_matrix[:3,3] = init_arm_pos.reshape(3)

        frame_number += 1

    while True:
        frame = controller.frame()
        # print("Frame id %d" % frame.id)
        hand = frame.hands[0]
        keypoints, armpoints = leap_hand_to_keypoints(hand)

        hand_direction, hand_palm_normal, hand_wrist_position, hand_palm_position = leap_motion_to_robot(armpoints)
        hand_wrist_rel_pos = hand_wrist_position-init_hand_wrist_position
        points = np.array([hand_wrist_rel_pos,hand_wrist_rel_pos+hand_palm_normal,hand_wrist_rel_pos+hand_direction])
        transfomation, rotation, translation = best_fit_transform(init_points, points)    

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
        init_rotation = init_arm_transformation_matrix[:3,:3]

        # create composed rotation and translation
        composed_rotation = np.dot(rotation, init_rotation)
        composed_translation = np.dot(rotation, translation) + init_arm_pos

        # convert rotation matrix to rotation vector
        composed_rotation = R.from_matrix(composed_rotation).as_euler('xyz')
        new_arm_pose = robot.robot.get_tcp_position()[1]
        new_arm_pose[:3] = composed_translation*1000
        new_arm_pose[3:6] = composed_rotation

        # robot.robot.linear_move_extend(new_arm_pose,0,True,100,50,0.1)
        robot.robot.servo_p(new_arm_pose,0)
    
    robot.robot.servo_move_enable(False)


        