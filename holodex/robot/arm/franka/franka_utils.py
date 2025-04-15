import numpy as np
from scipy.spatial.transform import Rotation as R

# transformation utils

def eulerZYX2quat(euler, degree=False):
    if degree:
        euler = np.radians(euler)

    tmp_quat = R.from_euler("xyz", euler).as_quat().tolist()
    quat = [tmp_quat[3], tmp_quat[0], tmp_quat[1], tmp_quat[2]]
    return quat



def eulerZYX2quat(euler: list, degree: bool = False) -> list:
    """
    Convert Euler ZYX angles to quaternion.

    Args:
        euler (list): [roll, pitch, yaw]
        degree (bool): If True, input is in degrees

    Returns:
        list: [qw, qx, qy, qz]
    """
    if degree:
        euler = np.radians(euler)
    tmp_quat = R.from_euler("xyz", euler).as_quat().tolist()
    return [tmp_quat[3], tmp_quat[0], tmp_quat[1], tmp_quat[2]]


def eulerXYZ2quat(euler: list, degree: bool = False) -> list:
    """
    Convert Euler XYZ angles to quaternion.

    Args:
        euler (list): [roll, pitch, yaw]
        degree (bool): If True, input is in degrees

    Returns:
        list: [qw, qx, qy, qz]
    """
    if degree:
        euler = np.radians(euler)
    tmp_quat = R.from_euler("xyz", euler).as_quat().tolist()
    return [tmp_quat[3], tmp_quat[0], tmp_quat[1], tmp_quat[2]]


def get_pose_as_matrix(trans: list, rotation_mat: list) -> np.ndarray:
    """
    Get current end-effector pose as 4x4 transformation matrix.

    Args:
        trans (list): [x, y, z]
        rotation_mat (list): [rx, ry, rz]

    Returns:
        np.ndarray: A 4x4 NumPy array representing the transformation matrix.
    """
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_mat
    transformation_matrix[:3, 3] = trans
    return transformation_matrix