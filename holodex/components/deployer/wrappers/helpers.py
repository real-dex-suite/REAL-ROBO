import os
import torch
from torch.utils.data import DataLoader
from holodex.datasets.image import ImageActionDataset
from tqdm import tqdm
from copy import deepcopy as copy
from holodex.utils import converter
from holodex.constants import *

def load_encoder_representations(
    data_path, 
    encoder, 
    selected_view, 
    transform, 
    demo_list = None
):
    transform_dict = {
        'color_image': [transform],
        'depth_image': []
    }

    dataset = ImageActionDataset(
        data_path = data_path, 
        selected_views = [selected_view], 
        image_type = 'color',
        demos_list = demo_list,
        absolute = None, 
        transforms = transform_dict
    )
    encoder = encoder.cuda()
    dataloader = DataLoader(dataset = dataset, batch_size = 64, shuffle = False)

    input_representation_array = []
    print('Obtaining all the representations:')
    for input_images, _ in tqdm(dataloader):
        input_images = input_images[0].cuda().float()
        input_representation = encoder(input_images).detach()
        input_representation_array.append(input_representation)

    input_representations = torch.cat(input_representation_array, dim = 0)
    torch.save(input_representations, os.path.join(data_path, 'input_representations.pt'))

    encoder = encoder.cpu()
    return input_representations

# def load_tensors(path, demo_list):
#     if demo_list is None:
#         demo_list = os.listdir(path)
#     else:
#         demo_list = ['{}.pth'.format(demo_name) for demo_name in demo_list]

#     demo_list.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))
#     tensor_paths = [os.path.join(path, tensor_name) for tensor_name in demo_list]

#     tensor_array = []
#     for tensor_path in tensor_paths:
#         tensor_array.append(torch.load(tensor_path))

#     return torch.cat(tensor_array, dim = 0)

def load_tensors(path, demos_list):
    input_type = "joint"
    tensor_paths = get_tensor_paths(path, demos_list)
    tensor_array = []

    for tensor_path in tensor_paths:
        data = torch.load(tensor_path)
        processed_data = process_data(data, input_type)
        tensor_array.append(processed_data)

    return torch.cat(tensor_array, dim=0)


def get_tensor_paths(path, demos_list):
    """
    Get the paths of tensor files based on the specified path and demos list.
    """
    if demos_list is None:
        tensor_names = os.listdir(path)
        tensor_names.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        tensor_paths = [os.path.join(path, tensor_name) for tensor_name in tensor_names]
    else:
        demos_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        tensor_paths = [os.path.join(path, f"{demo}.pth") for demo in demos_list]
    return tensor_paths


def process_data(data, input_type):
    arm_tensor, hand_tensor = data["arm_ee_pose"], data["hand_abs_joint"]
    arm_joint = data["arm_abs_joint"]

    processed_hand_abs_joint = converter.scale_transform(hand_tensor, HAND_JOINT_LOWER_LIMIT, HAND_JOINT_UPPER_LIMIT)

    if input_type == "ee_pose":
        processed_arm_data = process_arm_ee_pose(arm_tensor)
    elif input_type == "joint":
        processed_arm_data = arm_joint
        processed_arm_data = converter.scale_transform(processed_arm_data, ARM_JOINT_LOWER_LIMIT, ARM_JOINT_UPPER_LIMIT)
    else:
        raise ValueError(f"Invalid input type: {input_type}. Choose either 'ee_pose' or 'joint'.")

    return torch.cat([processed_arm_data, processed_hand_abs_joint], dim=1)


def process_arm_ee_pose(arm_tensor):
    """
    Process the arm end-effector pose data based on the orientation type.
    """
    ORI_TYPE = "quat_pose"  # define the orientation type: euler_pose or quat_pose

    if ORI_TYPE == "euler_pose":
        processed_arm_ee_pose = converter.normalize_arm_ee_pose(arm_tensor)
        return processed_arm_ee_pose
    elif ORI_TYPE == "quat_pose":
        arm_position = arm_tensor[:, :3] / ARM_POS_SCALE
        arm_orientation_eulers = arm_tensor[:, 3:]
        new_arm_pose = torch.zeros(arm_tensor.shape[0], 7)
        new_arm_pose[:, :3] = arm_position
        prev_quat = None
        for idx, euler in enumerate(arm_orientation_eulers):
            quat = torch.tensor(R.from_euler('xyz', euler.cpu().numpy(), degrees=False).as_quat(), device='cuda:0')
            if idx != 0:
                quat = converter.transform_quat(prev_quat, quat)
            prev_quat = quat
            new_arm_pose[idx, 3:] = quat
        return new_arm_pose


def load_image_paths(data_path, selected_view, demo_list = None):
    images_path = os.path.join(data_path, 'images')

    if demo_list is None:
        demo_list = os.listdir(images_path)

    demo_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    input_image_paths = []
    output_image_paths = []
    traj_idx = []
    cumm_len = [0]
    
    for idx, demo in enumerate(demo_list):
        image_names = os.listdir(os.path.join(images_path, demo, 'camera_{}_color_image'.format(selected_view)))
        image_names.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))
        
        cumm_value = cumm_len[-1] + len(image_names) - 1
        cumm_len.append(cumm_value)

        for _ in range(len(image_names) - 1):
            traj_idx.append(idx)

        demo_input_image_paths = [os.path.join(images_path, demo, 'camera_{}_color_image'.format(selected_view), image_names[image_num]) for image_num in range(len(image_names) - 1)]
        demo_output_image_paths = [os.path.join(images_path, demo, 'camera_{}_color_image'.format(selected_view), image_names[image_num + 1]) for image_num in range(len(image_names) - 1)]
        input_image_paths.append(copy(demo_input_image_paths))
        output_image_paths.append(copy(demo_output_image_paths))

    return input_image_paths, output_image_paths, cumm_len, traj_idx

def get_traj_state_idxs(idx, traj_idx, cumm_len):
    traj_idx = traj_idx[idx]
    path_idx = idx - cumm_len[traj_idx]
    return traj_idx, path_idx