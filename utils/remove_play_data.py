import os
import shutil
import json

dataset_name = "tactile_play_data_v1_refined"
target_dataset_name = "tactile_play_data_v1_numbered"
demo_path = os.path.join("/home/agibot/Projects/Real-Robo/expert_dataset", dataset_name, "recorded_data")
trget_path = os.path.join("/home/agibot/Projects/Real-Robo/expert_dataset", target_dataset_name, "recorded_data")
idx = 1
demo_dict = {}
for demo in os.listdir(demo_path):
    demo_dir = os.path.join(demo_path, demo)
    demo_dict[idx] = demo
    # copy demo folder to target path
    target_demo_dir = os.path.join(trget_path, f"demonstration_{idx}")
    os.makedirs(target_demo_dir, exist_ok=True)
    for file in os.listdir(demo_dir):
        file_path = os.path.join(demo_dir, file)
        target_file_path = os.path.join(target_demo_dir, file)
        shutil.copy(file_path, target_file_path)

    idx += 1

with open(os.path.join(trget_path, "demonstration_dict.json"), 'w') as f:
    json.dump(demo_dict, f, indent=4)

    