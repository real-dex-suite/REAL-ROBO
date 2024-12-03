import numpy as np
import os
from scipy.spatial.transform import Rotation as R

from holodex.camera.realsense_camera import RealSenseRobotStream

if __name__ == "__main__":
    cam_sn = "f1230963"
    extrinsics_npy = np.load("/home/agibot/Projects/Real-Robo/32views_c2r.npy") # c2r

    rs = RealSenseRobotStream(cam_sn, 1)
    rs.stream()

    get_pointcloud_from_realsense(cam_sn)