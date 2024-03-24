import os

demonstration_path = "/home/agibot/Projects/Real-Robo/expert_dataset/pouring/recorded_data/demonstration_2"

start_idx = 7
end_idx = 71

for file in os.listdir(demonstration_path):
    file_path = os.path.join(demonstration_path, file)
    
    if int(file) < start_idx or int(file) > end_idx:
        # print(file)
        os.remove(file_path)