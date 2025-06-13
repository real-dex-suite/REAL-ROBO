import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
path = "/home/zhiyuan/REAL-ROBO/expert_dataset/recorded_data/20250613_003950/demonstration_1"
data_files = sorted([int(x) for x in os.listdir(path)])


# deploy with python3 view_data.py
for data_file in tqdm(data_files):
    # load data
    data = np.load(os.path.join(path, str(data_file)), allow_pickle=True)

    vis_dir = "expert_dataset/visualization/camera_1"
    os.makedirs(vis_dir, exist_ok=True)
    img = data["camera_1_color_image"][:, :, ::-1] # BGR --> RGB
    plt.figure(figsize=(8, 6))  # Optional: Adjust figure size
    plt.imshow(img)
    plt.axis('off')  # Optional: Hide axes
    plt.title(f"Camera 1 Color Image, gripper={data['gripper_joint_positions']}")  # Optional: Add a title
    plt.savefig(os.path.join(vis_dir, f"{data_file:03d}.jpg"))
    plt.close()

    vis_dir = "expert_dataset/visualization/camera_2"
    os.makedirs(vis_dir, exist_ok=True)
    img2 = data["camera_2_color_image"][:, :, ::-1] # BGR --> RGB
    plt.figure(figsize=(8, 6))  # Optional: Adjust figure size
    plt.imshow(img2)
    plt.axis('off')  # Optional: Hide axes
    plt.title(f"Camera 2 Color Image, gripper={data['gripper_joint_positions']}")  # Optional: Add a title
    plt.savefig(os.path.join(vis_dir, f"{data_file:03d}.jpg"))
    plt.close()