import os
import shutil
task_name = "openbottle_v1"
demonstration_number = 30
demonstration_path = f"/home/agibot/Projects/Real-Robo/expert_dataset/{task_name}/recorded_data/demonstration_{demonstration_number}"
target_path = demonstration_path.replace(task_name, f'{task_name}_refined')

start_idx = 5
end_idx = 88

os.makedirs(target_path, exist_ok=True)

for file in os.listdir(demonstration_path):
    file_path = os.path.join(demonstration_path, file)
    target_file_path = os.path.join(target_path, file)
    if start_idx <= int(file) <= end_idx:
        shutil.copy(file_path, target_file_path)
        print(f"Copying {file_path} to {target_file_path}")
        