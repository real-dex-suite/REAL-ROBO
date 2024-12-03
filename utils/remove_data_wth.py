import os
import shutil
task_name = "assembly_fix_v2"
demonstration_number = 3
target_demonstration_number = demonstration_number + 30
demonstration_path = f"/home/agibot/Projects/Real-Robo/expert_dataset/{task_name}/recorded_data/demonstration_{demonstration_number}"
target_demonstration_path = f"/home/agibot/Projects/Real-Robo/expert_dataset/{task_name}_refined/recorded_data/demonstration_{target_demonstration_number}"

start_idx = 5
end_idx = 214

os.makedirs(target_demonstration_path, exist_ok=True)

for file in os.listdir(demonstration_path):
    file_path = os.path.join(demonstration_path, file)
    target_file_path = os.path.join(target_demonstration_path, file)
    if start_idx <= int(file) <= end_idx:
        shutil.copy(file_path, target_file_path)
        print(f"Copying {file_path} to {target_file_path}")