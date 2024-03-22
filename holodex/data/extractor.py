import os
import numpy as np
import cv2
from PIL import Image
import torch
import yaml

from holodex.utils.files import make_dir, get_pickle_data
from holodex.constants import *
from utils.tactile_data_vis.tactile_visualizer_2d import Tactile2DVisualizer

# load module according to hand type
hand_module = __import__("holodex.robot.hand")
HandKDL_module_name = f'{HAND_TYPE}KDL'
# get relevant classes
HandKDL = getattr(hand_module.robot, HandKDL_module_name)

class TactileExtractor(object):
    def __init__(self, data_path, extract_tactile_types):
        self.data_path = data_path
        self.extract_tactile_types = extract_tactile_types
        if "image" in extract_tactile_types:
            with open("configs/teleop.yaml", "r") as file:
                tactile_type = yaml.safe_load(file)['defaults'][2]['tactile']
            self.tactile_visualizer = Tactile2DVisualizer(tactile_type)

    def extract_demo(self, demo_path, demo_data_target_path=None, demo_image_target_path=None):
        states = os.listdir(demo_path)
        states.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))

        demo_tactile_data = {}
        for tactile_type in self.extract_tactile_types:
            if "image" not in tactile_type:
                demo_tactile_data[tactile_type] = []

        for state in states:
            tactile_data = get_pickle_data(os.path.join(demo_path, state))["tactile_data"]

            # example 
            for tactile_type in self.extract_tactile_types:
                if tactile_type == "image":
                    self.tactile_visualizer.plot_once(tactile_data, save_img_path=os.path.join(demo_image_target_path, f'{state}.PNG'))
                elif tactile_type == "raw_data":
                    raw_data = []
                    for sensor_name in tactile_data:
                        raw_data.extend(tactile_data[sensor_name].reshape(-1).tolist())
                    demo_tactile_data[tactile_type].append(raw_data)

        if demo_data_target_path is not None:
            for tactile_type in demo_tactile_data:
                demo_tactile_data[tactile_type] = torch.tensor(np.array(demo_tactile_data[tactile_type])).squeeze()

            torch.save(demo_tactile_data, demo_data_target_path)

    def extract(self, target_path):
        demo_list = os.listdir(self.data_path)
        demo_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        for demo in demo_list:
            demo_path = os.path.join(self.data_path, demo)
            demo_image_target_path = None
            demo_data_target_path = None
             
            for extrace_tactile_type in self.extract_tactile_types:
                if "image" in extrace_tactile_type:
                    demo_image_target_path = os.path.join(target_path, demo)
                    make_dir(demo_image_target_path)
                else:
                    demo_data_target_path = os.path.join(target_path, f'{demo}.pth')

            print(f"Extracting tactiles from {demo_path}")
            self.extract_demo(demo_path, demo_data_target_path, demo_image_target_path)

class ColorImageExtractor(object):
    def __init__(self, data_path, num_cams, image_size, crop_sizes = None):
        # TODO BUG: if set num_cam to 1 in demo_extract, size not match
        # uncomment if camera > 1
        # assert num_cams == len(crop_sizes)
        
        self.data_path = data_path
        self.num_cams = num_cams
        self.image_size = image_size
        self.crop_sizes = crop_sizes

    def extract_demo(self, demo_path, target_path):
        states = os.listdir(demo_path)
        states.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))

        color_cam_image_paths = []
        for cam_num in range(self.num_cams):
            color_cam_image_path = os.path.join(target_path, 'camera_{}_color_image').format(cam_num + 1)
            color_cam_image_paths.append(color_cam_image_path)
            make_dir(color_cam_image_path)

        for state in states:
            state_data = get_pickle_data(os.path.join(demo_path, state))
            
            color_images = [state_data['camera_{}_color_image'.format(cam_num + 1)] for cam_num in range(self.num_cams)]
            
            if self.crop_sizes is not None:
                for cam_num in range(self.num_cams):
                    color_image = Image.fromarray(color_images[cam_num])
                    color_image = color_image.crop(self.crop_sizes[cam_num])
                    color_image = color_image.resize((self.image_size, self.image_size))
                    color_image = np.array(color_image)
                    cv2.imwrite(os.path.join(color_cam_image_paths[cam_num], f'{state}.PNG'), color_image)

    def extract(self, target_path):
        demo_list = os.listdir(self.data_path)
        demo_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        for demo in demo_list:
            demo_path = os.path.join(self.data_path, demo)
            demo_target_path = os.path.join(target_path, demo)
            make_dir(demo_target_path)

            print(f"\nExtracting images from {demo_path}")
            self.extract_demo(demo_path, demo_target_path)


class DepthImageExtractor(object):
    def __init__(self, data_path, num_cams, image_size, crop_sizes = None):
        # TODO BUG uncomment if camera > 1
        # assert num_cams == len(crop_sizes)
        
        self.data_path = data_path
        self.num_cams = num_cams
        self.image_size = image_size
        self.crop_sizes = crop_sizes

    def extract_demo(self, demo_path, target_path):
        states = os.listdir(demo_path)
        states.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))

        depth_cam_image_paths = []
        for cam_num in range(self.num_cams):
            depth_cam_image_path = os.path.join(target_path, 'camera_{}_depth_image').format(cam_num + 1)
            depth_cam_image_paths.append(depth_cam_image_path)
            make_dir(depth_cam_image_path)

        for state in states:
            state_data = get_pickle_data(os.path.join(demo_path, state))
            
            depth_images = [state_data['camera_{}_depth_image'.format(cam_num + 1)] for cam_num in range(self.num_cams)]
            
            if self.crop_sizes is not None:
                for cam_num in range(self.num_cams):
                    depth_image = Image.fromarray(depth_images[cam_num])
                    depth_image = depth_image.crop(self.crop_sizes[cam_num])
                    depth_image = depth_image.resize((self.image_size, self.image_size))
                    depth_image = np.array(depth_image)
                    np.save(os.path.join(depth_cam_image_paths[cam_num], f'{state}.npy'), depth_image)

    def extract(self, target_path):
        demo_list = os.listdir(self.data_path)
        demo_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        for demo in demo_list:
            demo_path = os.path.join(self.data_path, demo)
            demo_target_path = os.path.join(target_path, demo)
            make_dir(demo_target_path)

            print(f"\nExtracting images from {demo_path}")
            self.extract_demo(demo_path, demo_target_path)


class StateExtractor(object):
    def __init__(self, data_path, extract_state_types):
        self.data_path = data_path
        self.kdl_solver = HandKDL()
        self.extract_state_types = extract_state_types

    def _get_coords(self, joint_angles):
        if "LEAP" in HAND_TYPE.upper():
            # Highlight!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
            # the urdf we use to calculate the finger coordinates use first palm-mcp, then mcp-pip which is different
            # with current hand joint angles, so we need to change the order of joint angles
            joint_angles = np.array(joint_angles)[[1,0,2,3,5,4,6,7,9,8,10,11,12,13,14,15]]
            
        index_coords, _ = self.kdl_solver.finger_forward_kinematics('index', list(joint_angles)[0:4])
        middle_coords, _ = self.kdl_solver.finger_forward_kinematics('middle', list(joint_angles)[4:8])
        ring_coords, _ = self.kdl_solver.finger_forward_kinematics('ring', list(joint_angles)[8:12])
        thumb_coords, _ = self.kdl_solver.finger_forward_kinematics('thumb', list(joint_angles)[12:16])

        finger_coords = list(index_coords) + list(middle_coords) + list(ring_coords) + list(thumb_coords)
        return np.array(finger_coords)

    def extract_demo(self, demo_path, target_path):
        states = os.listdir(demo_path)
        states.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))

        demo_state_data = {}
        for state_type in self.extract_state_types:
            demo_state_data[state_type] = []

        for idx in range(len(states) - 1):
            state_data = get_pickle_data(os.path.join(demo_path, states[idx]))
            for state_type in demo_state_data.keys():   
                if state_type == "arm_abs_joint":
                    arm_state_abs_joint = state_data['arm_joint_positions']
                    demo_state_data[state_type].append(arm_state_abs_joint)
                if state_type == "hand_abs_joint":
                    hand_state_abs_joint = state_data['hand_joint_positions']
                    demo_state_data[state_type].append(hand_state_abs_joint)
                if state_type == "hand_coords":
                    state_joint_angles = state_data['hand_joint_positions']
                    state_joint_coords = self._get_coords(state_joint_angles)
        
        for state_type in self.extract_state_types:
             demo_state_data[state_type] = torch.tensor(np.array(demo_state_data[state_type])).squeeze()

        torch.save(demo_state_data, target_path)

    def extract(self, target_path):
        demo_list = os.listdir(self.data_path)
        demo_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        for demo in demo_list:
            demo_path = os.path.join(self.data_path, demo)
            demo_target_path = os.path.join(target_path, f'{demo}.pth')

            print(f"Extracting states from {demo_path}")
            self.extract_demo(demo_path, demo_target_path)


class ActionExtractor(object):
    def __init__(self, data_path, extract_action_types):
        self.data_path = data_path
        self.kdl_solver = HandKDL()
        self.extract_action_types = extract_action_types

    def _get_coords(self, joint_angles):
        if "LEAP" in HAND_TYPE.upper():
            # Highlight!!!!!!!!!!!!!!!!!!!!!!!!!!!!  the urdf we use to calculate the finger coordinates use first palm-mcp, then mcp-pip which is different
            # with current hand joint angles, so we need to change the order of joint angles
            joint_angles = np.array(joint_angles)[[1,0,2,3,5,4,6,7,9,8,10,11,12,13,14,15]]

        index_coords, _ = self.kdl_solver.finger_forward_kinematics('index', list(joint_angles)[0:4])
        middle_coords, _ = self.kdl_solver.finger_forward_kinematics('middle', list(joint_angles)[4:8])
        ring_coords, _ = self.kdl_solver.finger_forward_kinematics('ring', list(joint_angles)[8:12])
        thumb_coords, _ = self.kdl_solver.finger_forward_kinematics('thumb', list(joint_angles)[12:16])

        finger_coords = list(index_coords) + list(middle_coords) + list(ring_coords) + list(thumb_coords)
        return np.array(finger_coords)

    def extract_demo(self, demo_path, target_path):
        states = os.listdir(demo_path)
        states.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))

        demo_action_data = {}
        for action_type in self.extract_action_types:
            demo_action_data[action_type] = []

        for idx in range(1, len(states)):
            state_data = get_pickle_data(os.path.join(demo_path, states[idx]))

            # example 
            for state_type in demo_action_data.keys():
                if state_type == "arm_cmd_abs_joint":
                    arm_cmd_abs_joint = state_data['arm_commanded_joint_position']
                    demo_action_data[state_type].append(arm_cmd_abs_joint) 
                elif state_type == "arm_abs_joint":
                    arm_abs_joint = state_data['arm_joint_positions']
                    demo_action_data[state_type].append(arm_abs_joint) 
                elif state_type == "hand_cmd_abs_joint":
                    hand_cmd_abs_joint = state_data['hand_commanded_joint_position']
                    demo_action_data[state_type].append(hand_cmd_abs_joint)
                elif state_type == "hand_abs_joint":
                    hand_abs_joint = state_data['hand_joint_positions']
                    demo_action_data[state_type].append(hand_abs_joint) 

        for state_type in self.extract_action_types:
             demo_action_data[state_type] = torch.tensor(np.array(demo_action_data[state_type])).squeeze()

        torch.save(demo_action_data, target_path)

    def extract(self, target_path):
        demo_list = os.listdir(self.data_path)
        demo_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        for demo in demo_list:
            demo_path = os.path.join(self.data_path, demo)
            demo_target_path = os.path.join(target_path, f'{demo}.pth')

            print(f"Extracting actions from {demo_path}")
            self.extract_demo(demo_path, demo_target_path)