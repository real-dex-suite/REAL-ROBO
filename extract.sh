#####################
### Step 1: filter data augs
#####################

# # # ############  Modify this to the path of the dataset  ############
# base_directory="/home/agibot/Projects/Real-Robo/expert_dataset/pouring-v1"

# # echo "Step 1: Filtering data"
# python extract_data.py \
#     storage_path="$base_directory/recorded_data" \
#     filter_path="$base_directory/filtered_data" \
#     target_path="$base_directory/extracted_data/" \
#     play_data=True \
#     num_cams=1 \
#     hand_min_action_distance=0.0 \
#     arm_min_action_distance=0 \
#     tactile_min_force_distance=0.0 \
#     last_frame_save_number=1 \
#     tactiles=True \
#     color_images=True \
#     depth_images=False \
#     crop_image=True \
#     states=True\
#     actions=True \
#     depth_data_types=["image"] \
#     tactile_data_types=["raw_data","dict_raw_data"] \
#     tactile_image=True\
#     tactile_img_representation="none" \
#     state_data_types=["arm_ee_pose","arm_abs_joint","hand_abs_joint"] \
#     action_data_types=["arm_ee_pose","arm_abs_joint","hand_abs_joint"] \
#     image_parameters="openbox" \

# # Step2: Generate concat Image to check data quality
# # This will automatically run after step 1
# # echo "Step 2: Generating concatenated images to check data quality"
# python utils/concat_rgb_tactile.py \
#     --base_directory "$base_directory"

#####################
### extract data augs
#####################
python extract_data.py \
    storage_path='expert_dataset/pick_basketball_v1/recorded_data' \
    filter_path='expert_dataset/pick_basketball_v1/filtered_data' \
    target_path='expert_dataset/pick_basketball_v1/extracted_data/' \
    play_data=True \
    num_cams=1 \
    hand_min_action_distance=0.0 \
    arm_min_action_distance=0.0 \
    tactile_min_force_distance=0.0 \
    last_frame_save_number=1 \
    tactiles=False \
    color_images=True \
    depth_images=False \
    crop_image=True \
    states=True\
    actions=True \
    depth_data_types=["image"] \
    tactile_data_types=["raw_data","dict_raw_data"] \
    tactile_image=False \
    tactile_img_representation="none" \
    state_data_types=["arm_ee_pose","hand_abs_joint"] \
    action_data_types=["arm_ee_pose","arm_cmd_ee_pose","hand_abs_joint"] \
    image_parameters="openbox" \
    
# state_data_types=["arm_ee_pose","arm_abs_joint","hand_abs_joint"] \
# action_data_types=["arm_ee_pose","arm_cmd_ee_pose","arm_abs_joint","arm_cmd_abs_joint","hand_abs_joint","hand_cmd_abs_joint"] \

# python extract_data.py \
#     storage_path='expert_dataset/tactile_play_data_v1_numbered/recorded_data' \
#     filter_path='expert_dataset/tactile_play_data_v1_numbered/filtered_data' \
#     target_path='expert_dataset/tactile_play_data_v1_numbered/extracted_data/' \
#     play_data=True \
#     num_cams=2 \
#     hand_min_action_distance=1.0 \
#     arm_min_action_distance=0 \
#     tactile_min_force_distance=5.0 \
#     last_frame_save_number=1 \
#     tactiles=True \
#     color_images=True \
#     depth_images=True \
#     crop_image=True \
#     states=True\
#     actions=True \
#     depth_data_types=["image"] \
#     tactile_data_types=["raw_data","dict_raw_data"] \
#     tactile_image=True\
#     tactile_img_representation="none" \
#     state_data_types=["arm_ee_pose","arm_abs_joint","hand_abs_joint"] \
#     action_data_types=["arm_ee_pose","arm_abs_joint","hand_abs_joint"] \
#     image_parameters="openbox" \


# python extract_data.py \
#     storage_path='expert_dataset/reori_v1_refined/recorded_data' \
#     filter_path='expert_dataset/reori_v1_refined/filtered_data' \
#     target_path='expert_dataset/reori_v1_refined/extracted_data/' \
#     play_data=True \
#     num_cams=2 \
#     hand_min_action_distance=0 \
#     arm_min_action_distance=0 \
#     last_frame_save_number=1 \
#     tactiles=True \
#     color_images=True \
#     depth_images=True \
#     crop_image=True \
#     states=True\
#     actions=True \
#     depth_data_types=["image"] \
#     tactile_data_types=["image"] \
#     tactile_image=True\
#     tactile_img_representation="none" \
#     state_data_types=["arm_ee_pose","arm_abs_joint","hand_abs_joint"] \
#     action_data_types=["arm_ee_pose","arm_abs_joint","hand_abs_joint"] \
#     image_parameters="pouring" \