import numpy as np
import matplotlib.pyplot as plt
path = "/home/wts/workspace/REAL-ROBO/expert_dataset/twin_stacking_all/recorded_data/demonstration_20/1"

# deploy with python3 view_data.py

# load data
data = np.load(path, allow_pickle=True)
print(data.keys())
img = data["camera_1_color_image"]
# plt
print(data["gripper_joint_positions"])


plt.figure(figsize=(8, 6))  # Optional: Adjust figure size
plt.imshow(img)
plt.axis('off')  # Optional: Hide axes
plt.title("Camera 1 Color Image")  # Optional: Add a title
plt.show()

## 0.039 - 0.024