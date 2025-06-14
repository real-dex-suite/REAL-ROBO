import os
import numpy as np
import cv2
from PIL import Image
import torch
import yaml
import struct

from holodex.utils.files import make_dir, get_pickle_data
from holodex.constants import *
from utils.tactile_data_vis.tactile_visualizer_2d import Tactile2DVisualizer
from termcolor import cprint

# Highlight!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# from jkrc import jkrc
# robot = jkrc.RC("192.168.130.95")
# robot.login()

# load module according to hand type
hand_module = __import__("holodex.robot.hand")  if HAND_TYPE is not None else None
HandKDL_module_name = f'{HAND_TYPE}KDL'  if HAND_TYPE is not None else None
# get relevant classes
HandKDL = getattr(hand_module.robot, HandKDL_module_name)  if HAND_TYPE is not None else None

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
                        # # only for old data
                        # raw_sensor_data = tactile_data[sensor_name]
                        # unsign_z_data = self.conver_sign_to_unsigned(raw_sensor_data[:,2])
                        # raw_sensor_data[:,2] = unsign_z_data
                        # reshaped_data = raw_sensor_data.reshape(-1).tolist()
                        # raw_data.extend(reshaped_data)
                        raw_data.extend(tactile_data[sensor_name].reshape(-1).tolist())
                    demo_tactile_data[tactile_type].append(raw_data)
                elif tactile_type == "dict_raw_data":
                    demo_tactile_data[tactile_type].append(tactile_data)

        if demo_data_target_path is not None:
            for tactile_type in demo_tactile_data:
                if tactile_type == 'dict_raw_data':
                    demo_tactile_data[tactile_type] = demo_tactile_data[tactile_type]
                else:
                    demo_tactile_data[tactile_type] = torch.tensor(np.array(demo_tactile_data[tactile_type])).squeeze()

            torch.save(demo_tactile_data, demo_data_target_path)
        
    def conver_sign_to_unsigned(self, data):
        # conver data to int
        data = [int(d) for d in data]
        original_bytes = struct.pack(f'<{len(data)}b', *data)
        unsigned_integers = list(struct.unpack(f'<{len(original_bytes)}B', original_bytes))
        return unsigned_integers

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
    def __init__(self, data_path, num_cams, image_size, crop_sizes = None, crop_image = True):
        # TODO BUG: if set num_cam to 1 in demo_extract, size not match
        # uncomment if camera > 1
        # assert num_cams == len(crop_sizes)
        
        self.data_path = data_path
        self.num_cams = num_cams
        self.image_size = image_size
        self.crop_sizes = crop_sizes
        self.crop_image = crop_image

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
                    if self.crop_image:
                        color_image = color_image.crop(self.crop_sizes[cam_num])
                        if len(self.image_size) > 1:
                            color_image = color_image.resize((self.image_size[0], self.image_size[1]))
                        else:
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

            print(f"Extracting images from {demo_path}")
            self.extract_demo(demo_path, demo_target_path)


class DepthImageExtractor(object):
    def __init__(self, data_path, num_cams, image_size, crop_sizes = None, extract_depth_types = None, crop_image = True):
        # TODO BUG uncomment if camera > 1
        # assert num_cams == len(crop_sizes)
        
        self.data_path = data_path
        self.num_cams = num_cams
        self.image_size = image_size
        self.crop_sizes = crop_sizes
        self.crop_image = crop_image
        self.extract_depth_types = extract_depth_types

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
                    if self.crop_image:
                        depth_image = depth_image.crop(self.crop_sizes[cam_num])
                        if len(self.image_size) > 1:
                            depth_image = depth_image.resize((self.image_size[0], self.image_size[1]))
                        else:
                            depth_image = depth_image.resize((self.image_size, self.image_size))
                    depth_image = np.array(depth_image)

                    for depth_type in self.extract_depth_types:
                        if depth_type == "image":
                            depth_image = np.clip(depth_image, 0, 1024)
                            cv2.imwrite(os.path.join(depth_cam_image_paths[cam_num], f'{state}.PNG'), depth_image*255/np.max(depth_image))
                        elif depth_type == "raw_data":
                            np.save(os.path.join(depth_cam_image_paths[cam_num], f'{state}.npy'), depth_image)

    def extract(self, target_path):
        demo_list = os.listdir(self.data_path)
        demo_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        for demo in demo_list:
            demo_path = os.path.join(self.data_path, demo)
            demo_target_path = os.path.join(target_path, demo)
            make_dir(demo_target_path)

            print(f"Extracting images from {demo_path}")
            self.extract_demo(demo_path, demo_target_path)


class StateExtractor(object):
    def __init__(self, data_path, extract_state_types):
        self.data_path = data_path
        self.kdl_solver = HandKDL() if HAND_TYPE is not None else None
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

        for idx in range(len(states)):
            state_data = get_pickle_data(os.path.join(demo_path, states[idx]))
            for state_type in demo_state_data.keys():   
                if state_type == "arm_abs_joint":
                    arm_state_abs_joint = state_data['arm_joint_positions']
                    demo_state_data[state_type].append(arm_state_abs_joint)
                if state_type == "arm_ee_pose":
                    arm_ee_pose = state_data['arm_ee_pose']
                    demo_state_data[state_type].append(arm_ee_pose)
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
        self.kdl_solver = HandKDL() if HAND_TYPE is not None else None
        self.extract_action_types = extract_action_types

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

        demo_action_data = {}
        for action_type in self.extract_action_types:
            demo_action_data[action_type] = []

        # depends on what kind of actions, cmd action or delta should start from 0, abs joint should start from 1 
        start_idx = 0
        
        for idx in range(start_idx, len(states)): 
            state_data = get_pickle_data(os.path.join(demo_path, states[idx]))

            # example 
            for state_type in demo_action_data.keys():
                if state_type == "arm_cmd_abs_joint":
                    arm_cmd_abs_joint = state_data['arm_commanded_joint_position']
                    demo_action_data[state_type].append(arm_cmd_abs_joint)
                if state_type == "arm_cmd_ee_pose":
                    # TODO: ARM_TYPE --> arm_type in configs
                    cprint("ARM_TYPE is deprecated, and will move to self.arm_type. Set 'Flexiv' by default.", "yellow")
                    # if ARM_TYPE == "jaka":
                    #     arm_cmd_ee_pose = robot.kine_forward(state_data['arm_commanded_joint_position'])[1]
                    #     demo_action_data[state_type].append(arm_cmd_ee_pose)
                    if ARM_TYPE == "Flexiv":
                        arm_cmd_ee_pose = state_data['arm_commanded_ee_pose']
                        demo_action_data[state_type].append(arm_cmd_ee_pose)


                if state_type == "arm_ee_pose" and idx > 0:
                    arm_ee_pose = state_data['arm_ee_pose']
                    demo_action_data[state_type].append(arm_ee_pose)
                if state_type == "arm_abs_joint" and idx > 0:
                    arm_abs_joint = state_data['arm_joint_positions']
                    demo_action_data[state_type].append(arm_abs_joint) 
                if state_type == "hand_cmd_abs_joint":
                    hand_cmd_abs_joint = state_data['hand_commanded_joint_position']
                    demo_action_data[state_type].append(hand_cmd_abs_joint)
                if state_type == "hand_abs_joint" and idx > 0:
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