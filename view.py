import numpy as np

data_path = '/home/agibot/Projects/Real-Robo/expert_dataset/pouring-v1/recorded_data/demonstration_8/2'

# Load the data
data = np.load(data_path, allow_pickle=True)
print(data)