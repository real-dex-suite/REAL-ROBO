import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

class TactileVisualizer:
    def __init__(self, figsize=(16, 8)):
        self.figsize = figsize
        self.fig, self.axs = plt.subplots(2, 4, figsize=self.figsize)
  

        for ax in self.axs.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor('white')
        
        self.imgs = [None] * self.axs.size
        plt.ion()
        plt.show(block=False)

    def plot_tactile_pad_tip(self, pad_data, ax):
        img_shape = (240, 240, 3)
        blank_image = np.ones(img_shape, np.uint8) * 255
        idx = np.where(self.axs == ax)
        img_idx = idx[0][0] * self.axs.shape[1] + idx[1][0]
        if self.imgs[img_idx] is None:
            self.imgs[img_idx] = ax.imshow(blank_image)

        tactile_coordinates = []
        for j in range(200, 40 - 1, -80):
            for i in range(200, 40 - 1, -40):
                tactile_coordinates.append((j, i))
        
        tactile_coordinates[4] = (160, 40)   # 1,1
        tactile_coordinates[9] = (120, 20)  
        tactile_coordinates[14] = (80, 40)  
                
        frame_axis = blank_image.copy()
        pad_data = np.array(pad_data).reshape(5, 3, 3)
        
        min_radius = 10
        max_radius = 30
        
        for i in range(5):
            for j in range(3):
                data = pad_data[i, j]
                print(data)
                center_coordinates = tactile_coordinates[i * 3 + j]

                x_value = data[0]
                y_value = data[1]
                z_value = data[2]
                
                if z_value >= 0:
                    z_value = np.sqrt(x_value**2 + y_value**2 + z_value**2)
                    radius = min_radius + int((max_radius - min_radius) * z_value / 30)
                    
                    color =  (255, 80, 80)
                    offset_x = max(-10, min(int(data[0] * 10), 10))
                    offset_y = max(-10, min(int(data[1] * 10), 10))
                    
                    center_coordinates = (center_coordinates[0] + offset_x, center_coordinates[1] + offset_y)
                    
                    frame_axis = cv2.circle(
                        frame_axis, center_coordinates, radius, color=color, thickness=-1
                    )

        self.imgs[img_idx].set_data(frame_axis)

        return self.imgs[img_idx]
    
    def plot_tactile_pad(self, pad_data, ax):
        img_shape = (240, 240, 3)
        blank_image = np.ones(img_shape, np.uint8) * 255
        idx = np.where(self.axs == ax)
        img_idx = idx[0][0] * self.axs.shape[1] + idx[1][0]
        if self.imgs[img_idx] is None:
            self.imgs[img_idx] = ax.imshow(blank_image)

        tactile_coordinates = []
        for j in range(200, 40 - 1, -80):
            for i in range(200, 40 - 1, -40):
                tactile_coordinates.append((j, i))
                
        frame_axis = blank_image.copy()
        pad_data = np.array(pad_data).reshape(5, 3, 3)
        
        min_radius = 10
        max_radius = 30
        
        for i in range(5):
            for j in range(3):
                data = pad_data[i, j]
                print(data)
                center_coordinates = tactile_coordinates[i * 3 + j]
            
                x_value = data[0]
                y_value = data[1]
                z_value = data[2]
                
                if z_value >= 0:
                    z_value = np.sqrt(x_value**2 + y_value**2 + z_value**2)
                    radius = min_radius + int((max_radius - min_radius) * z_value / 30)
                    
                    # light red rgb = (255, 204, 204)
                    color = (255, 80, 80)
                    offset_x = max(-10, min(int(data[0] * 10), 10))
                    offset_y = max(-10, min(int(data[1] * 10), 10))
                    
                    center_coordinates = (center_coordinates[0] + offset_x, center_coordinates[1] + offset_y)
                    
                    frame_axis = cv2.circle(
                        frame_axis, center_coordinates, radius, color=color, thickness=-1
                    )

        self.imgs[img_idx].set_data(frame_axis)

        return self.imgs[img_idx]

    def visualize_tactile_data(self, tactile_data, save_path='tactile_visualization.png'):
        for i, pad_data in enumerate(tactile_data):
            row = i // 4
            col = i % 4
            if i < 4:
                self.plot_tactile_pad_tip(pad_data, self.axs[row, col])
            else:
                self.plot_tactile_pad(pad_data, self.axs[row, col])

        plt.tight_layout()
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
        
        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved as {save_path}")

if __name__ == "__main__":
    visualizer = TactileVisualizer()

    # Generate random tactile data for 8 pads, each with 15 points (x, y, z)
    #tactile_data = [np.random.rand(15, 3) for _ in range(8)]

    # all  = 0
    # tactile_data = [np.zeros((15, 3)) for _ in range(8)]
    tacilte_data_path = 'expert_dataset/tactile_play_data_v2_numbered/recorded_data/demonstration_33/42'
    # # load pickle file
    tactile_data = np.load(tacilte_data_path, allow_pickle=True)
    tactile_data = tactile_data['tactile_data']

    # print(tactile_data.keys())  
    import numpy as np
    #read npy file

    # data_path = "/home/agibot/Projects/real_diff_dex/rollouts/reorient/2024.09.13_reorient_random_20v1_refined_cmd_3dc_50scale_handbase_835_ours_DinoV2_p:True_f:False_MAEGAT_p:True_f:False_2000.ckpt_ori_1/7/129_tactile.npy"
    # tactile_data = np.load( data_path, allow_pickle=True)
    
    rollout = False
    if rollout:
        # change the 360 - > 8, 15, 3
        tactile_data = tactile_data.reshape(8, 15, 3)
        #  # dict_keys(['thumb_tip', 'thumb_pulp', 'index_tip', 'index_pulp', 'middle_tip', 'middle_pulp', 'ring_tip', 'ring_pulp'])
        tactiles_tip = [tactile_data[0], tactile_data[2], tactile_data[4], tactile_data[6]]
        tactiles_pulp = [tactile_data[1], tactile_data[3], tactile_data[5], tactile_data[7]]

        tactile_data = []
        for i in range(4):
            tactile_data.append(tactiles_tip[i])

        for j in range(4):
            tactile_data.append(tactiles_pulp[j])
    
        visualizer.visualize_tactile_data(tactile_data, 'tactile_sensor_data.png')
    

    else:
        # dict_keys(['thumb_tip', 'thumb_pulp', 'index_tip', 'index_pulp', 'middle_tip', 'middle_pulp', 'ring_tip', 'ring_pulp'])
        tactiles_tip = [tactile_data['thumb_tip'], tactile_data['index_tip'], tactile_data['middle_tip'], tactile_data['ring_tip']]
        tactiles_pulp = [tactile_data['thumb_pulp'], tactile_data['index_pulp'], tactile_data['middle_pulp'], tactile_data['ring_pulp']]

        tactile_data = []
        for i in range(4):
            tactile_data.append(tactiles_tip[i])
        
        for j in range(4):
            tactile_data.append(tactiles_pulp[j])
        # Visualize the data and save the image
        visualizer.visualize_tactile_data(tactile_data, 'tactile_sensor_data.png')
    