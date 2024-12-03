#!/bin/bash

# path -> Modify this to the path of the dataset
base_directory="/home/agibot/Projects/Real-Robo/expert_dataset/openbox_v2"

# Step 1: extract data
echo "Step 1: Filtering data augmentations"
python extract_data.py \
    storage_path="$base_directory/recorded_data" \
    filter_path="$base_directory/filtered_data" \
    target_path="$base_directory/extracted_data/" \
    play_data=True \
    num_cams=2 \
    hand_min_action_distance=0.0 \
    arm_min_action_distance=0 \
    tactile_min_force_distance=0.0 \
    last_frame_save_number=1 \
    tactiles=True \
    color_images=True \
    depth_images=False \
    crop_image=True \
    states=True \
    actions=True \
    depth_data_types=["image"] \
    tactile_data_types=["image"] \
    tactile_image=False \
    tactile_img_representation="none" \
    state_data_types=["arm_ee_pose","arm_abs_joint","hand_abs_joint"] \
    action_data_types=["arm_ee_pose","arm_abs_joint","hand_abs_joint"] \
    image_parameters="openbox"

# Step 2: generate concatenated images
echo "Step 2: Generating concatenated images to check data quality"
python utils/concat_rgb_tactile.py \
    --base_directory "$base_directory"