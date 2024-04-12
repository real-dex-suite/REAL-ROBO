python extract_data.py \
    storage_path='expert_dataset/pouring_latest/recorded_data/' \
    filter_path='expert_dataset/pouring_latest/filtered_data' \
    target_path='expert_dataset/pouring_latest/extracted_data/' \
    num_cams=1 \
    hand_min_action_distance=0 \
    arm_min_action_distance=0 \
    tactiles=True \
    color_images=True \
    depth_images=True \
    states=True \
    actions=True \
    depth_data_types=["raw_data"] \
    tactile_data_types=["raw_data"] \
    state_data_types=["arm_ee_pose","arm_abs_joint","hand_abs_joint"] \
    action_data_types=["arm_ee_pose","arm_abs_joint","hand_abs_joint"] \
    image_parameters="pouring" \
