import numpy as np
from scipy.spatial.transform import Rotation as R

# transformation utils

def eulerZYX2quat(self, euler, degree=False):
    if degree:
        euler = np.radians(euler)

    tmp_quat = R.from_euler("xyz", euler).as_quat().tolist()
    quat = [tmp_quat[3], tmp_quat[0], tmp_quat[1], tmp_quat[2]]
    return quat
