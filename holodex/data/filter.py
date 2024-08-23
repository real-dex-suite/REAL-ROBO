import os
import numpy as np
from holodex.utils.files import make_dir, get_pickle_data, store_pickle_data
from holodex.utils.vec_ops import get_distance

from holodex.constants import *

# load module according to hand type
hand_module = __import__("holodex.robot.hand")
HandKDL_module_name = f'{HAND_TYPE}KDL'
# get relevant classes
HandKDL = getattr(hand_module.robot, HandKDL_module_name)

class FilterData(object):
    def __init__(self, data_path, hand_delta = None, arm_delta = None, tactile_delta = None, play_data=False, last_frame_save_number=0) -> None:
        self.hand_delta = hand_delta # Threshold for filtering
        self.arm_delta = arm_delta # Threshold for filtering
        self.tactile_delta = tactile_delta # Threshold for filtering
        self.data_path = data_path           
        self.play_data = play_data
        self.last_frame_save_number = last_frame_save_number

        self.kdl_solver = HandKDL() # # Solver for forward kinematics
        
    def _get_coords(self, joint_angles):
        # Calculate finger coordinates based on joint angles

        if "LEAP" in HAND_TYPE.upper():
            # Highlight!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
            # the urdf we use to calculate the finger coordinates 
            # use first palm-mcp, then mcp-pip which is different
            # with current hand joint angles, so we need to change the order of joint angles
            joint_angles = np.array(joint_angles)[[1,0,2,3,5,4,6,7,9,8,10,11,12,13,14,15]]

        index_coords, _ = self.kdl_solver.finger_forward_kinematics('index', list(joint_angles)[0:4])
        middle_coords, _ = self.kdl_solver.finger_forward_kinematics('middle', list(joint_angles)[4:8])
        ring_coords, _ = self.kdl_solver.finger_forward_kinematics('ring', list(joint_angles)[8:12])
        thumb_coords, _ = self.kdl_solver.finger_forward_kinematics('thumb', list(joint_angles)[12:16])

        return index_coords, middle_coords, ring_coords, thumb_coords

    def _get_coords_from_state(self, state_data):
        # Read joint angles from the state file and calculate finger coordinates
        joint_angles = state_data['hand_joint_positions']
        return self._get_coords(joint_angles)
    
    def _get_arm_ee_pose_from_state(self, state_data):
        # Read end-effector pose from the state file
        ee_position = np.array(state_data['arm_ee_pose'][:3])
        ee_orientation = np.array(state_data['arm_ee_pose'][3:])
        return ee_position, ee_orientation

    def _get_robot_poses_from_state(self, demo_path, state_path):
        # Read the state file and get the finger coordinates and end-effector pose
        state_path = os.path.join(demo_path, state_path)
        state_data = get_pickle_data(state_path)

        if "hand_joint_positions" in state_data.keys():
            index_coords, middle_coords, ring_coords, thumb_coords = self._get_coords_from_state(state_data)
        else:
            index_coords, middle_coords, ring_coords, thumb_coords = None, None, None, None

        if "arm_ee_pose" in state_data.keys():
            arm_ee_position, arm_ee_orientation = self._get_arm_ee_pose_from_state(state_data)
        else:
            arm_ee_position, arm_ee_orientation = None, None

        return index_coords, middle_coords, ring_coords, thumb_coords, arm_ee_position, arm_ee_orientation

    def _get_tactile_from_state(self, demo_path, state_path):
        # Read the state file and get the tactile data
        state_path = os.path.join(demo_path, state_path)
        state_data = get_pickle_data(state_path)

        if "tactile_data" in state_data.keys():
            tactile_data = state_data['tactile_data']
            raw_data = []
            for sensor_name in tactile_data:
                raw_data.extend(tactile_data[sensor_name].reshape(-1).tolist())
        else:
            raw_data = None

        return np.array(raw_data)

    def filter_demo(self, demo_path, target_path):
        # Filter a single demonstration and store the filtered data in the target path
        states = os.listdir(demo_path)
        states.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))

        filtered_state_idxs = []

        # Get the initial state of the robot
        prev_index_coords, prev_middle_coords, prev_ring_coords, prev_thumb_coords, prev_arm_ee_pos, prev_arm_ee_ori = self._get_robot_poses_from_state(demo_path, states[0])
        prev_tactile = self._get_tactile_from_state(demo_path, states[0])

        for idx in range(1, len(states)):
            index_coords, middle_coords, ring_coords, thumb_coords, arm_ee_pos, arm_ee_ori = self._get_robot_poses_from_state(demo_path, states[idx])
            tactile = self._get_tactile_from_state(demo_path, states[idx])
            

            save_current_frame = False
            
            # hand delta
            if index_coords is not None and middle_coords is not None and ring_coords is not None and thumb_coords is not None:    
                delta_index = get_distance(prev_index_coords, index_coords)*100
                delta_middle = get_distance(prev_middle_coords, middle_coords)*100
                delta_ring = get_distance(prev_ring_coords, ring_coords)*100
                delta_thumb = get_distance(prev_thumb_coords, thumb_coords)*100
                hand_delta_total = delta_index + delta_middle + delta_ring + delta_thumb
                
                hand_delta_satisfied = True if hand_delta_total >= self.hand_delta else False
                save_current_frame = hand_delta_satisfied
            
            # arm delta        
            if arm_ee_pos is not None and arm_ee_ori is not None:    
                arm_delta = get_distance(prev_arm_ee_pos, arm_ee_pos)/10
                arm_delta_total = arm_delta
                
                arm_delta_satisfied = True if arm_delta_total >= self.arm_delta else False
                save_current_frame = arm_delta_satisfied
            
            # tactile delta
            if tactile is not None:
                tactile_delta = get_distance(tactile, prev_tactile)
                tactile_delta_satisfied = True if tactile_delta >= self.tactile_delta else False
                save_current_frame = tactile_delta_satisfied
                # tactile_delta_satisfied = True
            
            if index_coords is not None and arm_ee_pos is not None:
                save_current_frame = hand_delta_satisfied and arm_delta_satisfied
            
            if tactile is not None and self.tactile_delta > 0:
                save_current_frame = save_current_frame or tactile_delta_satisfied
            
            if self.play_data:
                save_current_frame = (hand_delta_total+arm_delta_total) >= self.hand_delta + self.arm_delta
                if tactile is not None and self.tactile_delta > 0:
                    save_current_frame = save_current_frame or tactile_delta_satisfied

            # for last five frames, must save
            if idx >= len(states) - self.last_frame_save_number:
                save_current_frame = True

            # if np.max(np.abs(tactile)) > 50:
            #     save_current_frame = False

            if save_current_frame:
                filtered_state_idxs.append(idx)
                prev_index_coords, prev_middle_coords, prev_ring_coords, prev_thumb_coords = index_coords, middle_coords, ring_coords, thumb_coords
                prev_arm_ee_pos, prev_arm_ee_ori = arm_ee_pos, arm_ee_ori
                prev_tactile = tactile

        for counter, idx in enumerate(filtered_state_idxs):
            state_data = get_pickle_data(os.path.join(demo_path, states[idx]))
            state_pickle_path = os.path.join(target_path, f'{counter + 1}')
            store_pickle_data(state_pickle_path, state_data)

    def filter(self, target_path, fresh_data = False):
        # Filter demonstrations and store the filtered data in the target path
        demo_list = os.listdir(self.data_path)
        demo_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        for demo in demo_list:
            demo_path = os.path.join(self.data_path, demo)
            demo_target_path = os.path.join(target_path, demo)

            if os.path.exists(demo_target_path) and fresh_data is False:
                print('{} already filtered!'.format(demo))
                continue

            make_dir(demo_target_path)

            print(f"Filtering demonstration from {demo_path}")
            self.filter_demo(demo_path, demo_target_path)