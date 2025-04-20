import numpy as np


data_path = "expert_dataset/test_ros/recorded_data/demonstration_10/5"

data = np.load(data_path, allow_pickle=True)

print(data)