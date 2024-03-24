python extract_data.py \
    storage_path='expert_dataset/pouring/recorded_data/' \
    filter_path='expert_dataset/pouring/filtered_data/' \
    target_path='expert_dataset/pouring/extracted_data/' \
    num_cams=1 \
    hand_min_action_distance=0 \
    arm_min_action_distance=0 \
    tactiles=True \
    color_images=True \
    depth_images=False \
    states=True \
    actions=True \
    tactile_data_types=["image","raw_data"] \
    state_data_types=["arm_abs_joint","hand_abs_joint"] \
    action_data_types=["arm_cmd_abs_joint","arm_abs_joint","hand_cmd_abs_joint","hand_abs_joint"] \
