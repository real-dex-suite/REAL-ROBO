'''
Just for removing useless data. Don't use it in the main code.
'''

import os
import pickle
import numpy as np
from tqdm import tqdm
import shutil

def find_zero_diff_indices_both(arm_ee_poses):
    """
    找到相邻两帧之间差为0的索引，包含两帧
    
    参数:
        arm_ee_poses: Nx3的numpy数组，表示机械臂末端执行器的位姿
    
    返回:
        一个排序后的列表，包含所有相邻两帧差为0的索引对
    """
    if len(arm_ee_poses) < 2:
        return []
    
    # 计算相邻帧之间的差值
    diffs = np.diff(arm_ee_poses, axis=0)
    
    # 检查每个维度的差值是否为0
    zero_diff_mask = np.all(diffs == 0, axis=1)
    
    # 获取差为0的索引对
    zero_diff_indices = []
    for i in np.where(zero_diff_mask)[0]:
        zero_diff_indices.extend([i, i+1])  # 添加两帧的索引
    
    # 去除可能的重复索引（当连续三帧相同时）
    zero_diff_indices = sorted(list(set(zero_diff_indices)))
    
    return zero_diff_indices

if __name__ == "__main__":
    task_names = ["banana_plate", "carrot_plate", "cup_cup", "cup_plate", "open_door", "close_door"]
    ori_data_dir = "/home/wts/workspace/REAL-ROBO/expert_dataset/recorded_data_filtered"
    target_data_dir =  "/home/wts/workspace/REAL-ROBO/expert_dataset/recorded_data_filtered_1"
    for task_name in task_names:
        task_path = os.path.join(ori_data_dir, task_name)
        if task_name not in ["open_door", "close_door"]:
            demo_paths = []
            timestamps = os.listdir(task_path)
            for timestamp in timestamps:
                timestamp_path = os.path.join(task_path, timestamp)
                demo_names = os.listdir(timestamp_path)
                for demo_name in demo_names:
                    demo_path = os.path.join(timestamp_path, demo_name)
                    demo_paths.append(demo_path)
        else:
            demo_paths = [os.path.join(task_path, demo_name) for demo_name in os.listdir(task_path)]
        for demo_path in tqdm(demo_paths, desc=task_name):
            data_files = os.listdir(demo_path)
            data_files = np.array(sorted([int(x) for x in data_files]))
            arm_ee_poses = []
            for data_file in data_files:
                data_path = os.path.join(demo_path, str(data_file))
                arm_ee_poses.append(pickle.load(open(data_path, "rb"))["arm_ee_pose"])
            arm_ee_poses = np.array(arm_ee_poses)
            zero_diff_indexes = find_zero_diff_indices_both(arm_ee_poses)
            print(arm_ee_poses[:, :3])
            # for data_file_index, data_file in enumerate(data_files):
            #     if data_file_index not in zero_diff_indexes:
            #         data_path = os.path.join(demo_path, str(data_file))
            #         target_dir = demo_path.replace(ori_data_dir, target_data_dir)
            #         os.makedirs(target_dir, exist_ok=True)
            #         target_path = os.path.join(target_dir, str(data_file))
            #         shutil.copy(data_path, target_path)