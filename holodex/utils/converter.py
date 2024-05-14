import torch
from scipy.spatial.transform import Rotation as R
from holodex.constants import *

def normalize_arm_ee_pose(arm_ee_pose: torch.Tensor) -> torch.Tensor:
    arm_ee_pose[:3] /= ARM_POS_SCALE
    arm_ee_pose[3:] /= ARM_ORI_SCALE

    return arm_ee_pose

def denormalize_arm_ee_pose(arm_ee_pose: torch.Tensor, ARM_POS_SCALE: float, ARM_ORI_SCALE: float) -> torch.Tensor:
   # Ensure arm_ee_pose is at least 2-dimensional
   if arm_ee_pose.dim() == 1:
       arm_ee_pose = arm_ee_pose.unsqueeze(0)  # Reshape from [N] to [1, N]
  
   # Apply scaling
   arm_ee_pose[:, :3] *= ARM_POS_SCALE
   arm_ee_pose[:, 3:] *= ARM_ORI_SCALE
  
   # If originally it was 1-dimensional, return it back to its 1D shape
   if arm_ee_pose.shape[0] == 1:
       arm_ee_pose = arm_ee_pose.squeeze(0)
  
   return arm_ee_pose

def scale_transform(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    offset = (lower + upper) * 0.5
    return 2 * (x - offset) / (upper - lower)

def unscale_transform(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    offset = (lower + upper) * 0.5
    return x * (upper - lower) * 0.5 + offset


def from_euler_angle_to_quaternion(angles: torch.Tensor) -> torch.Tensor:
    quat = R.from_euler('xyz', angles, degrees=False).as_quat()
    
    return quat

def from_quaternion_to_euler_angle(quat: np.array) -> np.array:
    euler_angle = R.from_quat(quat).as_euler('xyz', degrees=False)
    
    return euler_angle

