import numpy as np

# generate random int from 1 to 10
data_idxs = []

for i in range(100):
    random_target_pos_idx = np.random.randint(1, 11)
    random_obj_pos_idx = np.random.randint(1, 11)
    random_obj_ori_idx = np.random.randint(1, 6)
    each_idx = [random_target_pos_idx, random_obj_pos_idx, random_obj_ori_idx]
    print(each_idx)
    data_idxs.append(each_idx)

