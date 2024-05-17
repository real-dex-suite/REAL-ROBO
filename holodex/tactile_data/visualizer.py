import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from holodex.tactile_data.tactile_image import TactileImage

class TactileVisualizer:
    def __init__(self, figsize=(10, 10), layout="tactile_pattern"):
        # initialize the tactile image
        self.tactile_image = TactileImage()
        
        # initialize the figure
        self.figsize = figsize
        self.layout = layout

        if layout == "3x3":
            self.fig, self.axs = plt.subplots(3, 3, figsize=self.figsize)
        elif layout == "4x4":
            self.fig, self.axs = plt.subplots(4, 4, figsize=self.figsize)
        elif layout == "tactile_pattern":
            self.fig, self.axs = plt.subplots(1, 1, figsize=self.figsize, squeeze=False)
        elif layout == "concat":
            self.fig, self.axs = plt.subplots(3, 6, figsize=self.figsize)  
        
        self.fig.subplots_adjust(wspace=0.1, hspace=0.1)
        
        for ax in self.axs.flatten():
            ax.get_yaxis().set_ticks([])
            ax.get_xaxis().set_ticks([])
            if layout == "3x3" and ax == self.axs[2, 2]:
                ax.axis("off")
        self.imgs = [None] * self.axs.size
        plt.ion()
        plt.show(block=False)
        
    def plot_one_tactile_sensor(self, sensor_values, ax):
        img_shape = (240, 240, 3)
        blank_image = np.ones(img_shape, np.uint8) * 255
        idx = np.where(self.axs == ax)
        img_idx = idx[0][0] * self.axs.shape[1] + idx[1][0]
        if self.imgs[img_idx] is None:
            self.imgs[img_idx] = ax.imshow(blank_image)

        tactile_coordinates = []
        # Generate tactile sensor coordinates
        # tactile_coordinates = []
        # for j in range(40, 200 + 1, 80):
        #     for i in range(200, 40 - 1, -40):  # Y
        #         tactile_coordinates.append((j, i))
        
        tactile_coordinates = []
        for j in range(200, 40 - 1, -80):
            for i in range(200, 40 - 1, -40):
                tactile_coordinates.append((j, i))
                
        frame_axis = blank_image.copy()
        for i, row in enumerate(sensor_values):
            for j, data in enumerate(row):
                center_coordinates = tactile_coordinates[i * len(row) + j]
                
                # Set the minimum and maximum radius
                min_radius = 2
                max_radius = 30
                
                # Calculate the radius based on data[2] and apply the limit
                radius = max(min_radius, min(int(data[2] * 4), max_radius))
                # radius = int(data[2] * 2)  # Adjust the scaling factor as needed
                
                # Calculate the offset based on x and y values
                offset_x = int(data[0] * 2)  # Adjust the scaling factor as needed
                offset_y = int(data[1] * 2)  # Adjust the scaling factor as needed
                
                # Apply the offset to the center coordinates
                center_coordinates = (center_coordinates[0] + offset_x, center_coordinates[1] + offset_y)
                
                frame_axis = cv2.circle(
                    frame_axis, center_coordinates, radius, color=(0, 255, 0), thickness=-1
                )

        self.imgs[img_idx].set_data(frame_axis)
        return self.imgs[img_idx]

    def plot_all_tactile_sensors(self, sensor_values):
        if self.layout == "3x3":
            # currently using shape of 3x3 for 8 tactile sensors
            sensor_order = [2, 4, 6, 3, 5, 7, 0, 1]
        elif (
            self.layout == "4x4"
        ):  # this is more like real tactile sensor layout for easy understanding
            sensor_order = [0, 7, 5, 3, 0, 8, 6, 4, 1, 0, 0, 0, 2, 0, 0, 0]

        for i, ax in enumerate(self.axs.flatten()):
            if self.layout == "3x3" and i < 8:
                self.plot_one_tactile_sensor(sensor_values[sensor_order[i]], ax)
            elif self.layout == "4x4":
                if sensor_order[i] != 0 and sensor_order[i] <= len(sensor_values):
                    self.plot_one_tactile_sensor(sensor_values[sensor_order[i] - 1], ax)
                else:
                    ax.axis("off")
                      
    def get_whole_tactile_repre_pattern(self, tactile_data):
        tactile_data = torch.tensor(tactile_data, dtype=torch.float64)
        # (8,5,3,3) -> (360,)
        tactile_data = tactile_data.view(-1)
        tactile_image = self.tactile_image.get_whole_hand_tactile_image(tactile_data, shuffle_type="none")
        
        # (3, 224, 224) -> (224, 224, 3)
        tactile_image = torch.permute(tactile_image, (1, 2, 0))
        tactile_image = tactile_image.numpy()
        
        # update the tactile image
        if self.layout == "tactile_pattern":
            for i, ax in enumerate(self.axs.flatten()):
                if self.imgs[i] is None:
                    self.imgs[i] = ax.imshow(tactile_image, aspect='auto')  # `aspect='auto'` padding the image to fit the axis
                else:
                    self.imgs[i].set_data(tactile_image)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
        
    # TODO
    def get_single_tactile_repre_pattern(self, tactile_data):
        tactile_data = torch.tensor(tactile_data, dtype=torch.float64)
        # (8,5,3,3) -> (360,)
        tactile_data = tactile_data.view(-1)
        tactile_images = self.tactile_image.get_single_tactile_image(tactile_data)
        
        # now we have 8 tactile images
        tactile_images = tactile_images.numpy()
        
        # write a for loop to draw tactile images
        # we only have 8, the last one is draw empty image
        
        for i, ax in enumerate(self.axs.flatten()):
            if i < tactile_images.shape[0]:
                if self.imgs[i] is None:
                    self.imgs[i] = ax.imshow(tactile_images[i], aspect='auto')
                else:
                    self.imgs[i].set_data(tactile_images[i])
            else:
                ax.axis("off")
        # update the tactile image
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
        
        print("test")
        

    # def concat_image(self, tactile_data):
    #     # concatenate the image show tactile sensor and tactile repre pattern
    #     # tactile sensor

    #     pass 
  
    # TODO
    def concat(self, sensor_values, tactile_data):
        pass

    def img_to_video(self):
        pass
    
    def visualize_tactile_data(self, tactile_data, figsize=(5, 5)):
        self.layout = "3x3"
        self.figsize = figsize
        self.fig.set_size_inches(figsize)
        self.plot_all_tactile_sensors(sensor_values=tactile_data)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
        
    def visualize_tactile_repre_pattern(self, tactile_data, figsize=(5, 5)):
        self.layout = "tactile_pattern"
        tactile_data = self.get_tactile_repre_pattern(tactile_data)
        self.figsize = figsize
        self.fig.set_size_inches(self.figsize)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
        
    # TODO: generate a rgb + sensor + tactile repre pattern -> video
    def draw_concat_image(self, tactile_data):
        pass
