import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
timestamp = "close_door"
path = f"expert_dataset/recorded_data/{timestamp}"
for demo_name in os.listdir(path):
    demo_path = os.path.join(path, demo_name)
    data_files = sorted([int(x) for x in os.listdir(demo_path)])

    vis_dir = f"expert_dataset/visualization/camera_1/{timestamp}/{demo_name}"
    os.makedirs(vis_dir, exist_ok=True)
    output_path = os.path.join(vis_dir, 'vis.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4编码器
    video_writer = cv2.VideoWriter(output_path, fourcc, 5, (1280, 720))

    # deploy with python3 view_data.py
    for data_file in tqdm(data_files):
        # load data
        data = np.load(os.path.join(demo_path, str(data_file)), allow_pickle=True)
        print(data["arm_joint_positions"])
        video_writer.write(data["camera_1_color_image"])
    video_writer.release()
    # plt.figure(figsize=(8, 6))  # Optional: Adjust figure size
    # plt.imshow(img)
    # plt.axis('off')  # Optional: Hide axes
    # plt.title(f"Camera 1 Color Image, gripper={data['gripper_joint_positions']}")  # Optional: Add a title
    # plt.savefig(os.path.join(vis_dir, f"{data_file:03d}.jpg"))
    # plt.close()

    # vis_dir = "expert_dataset/visualization/camera_2"
    # os.makedirs(vis_dir, exist_ok=True)
    # img2 = data["camera_2_color_image"][:, :, ::-1] # BGR --> RGB
    # plt.figure(figsize=(8, 6))  # Optional: Adjust figure size
    # plt.imshow(img2)
    # plt.axis('off')  # Optional: Hide axes
    # plt.title(f"Camera 2 Color Image, gripper={data['gripper_joint_positions']}")  # Optional: Add a title
    # plt.savefig(os.path.join(vis_dir, f"{data_file:03d}.jpg"))
    # plt.close()