import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from holodex.utils.network import frequency_timer
from holodex.constants import *


class Plot2DTACTILE(object):
    def __init__(self, sensor_info):
        self.totoal_force_limit = np.sqrt(FORCE_LIMIT**2 * 3)
        self.sensor_info = sensor_info
        self.sensor_type = self.sensor_info['sensor_type']
        self.sensor_name = []
        # Figure settings
        self.fig = plt.figure()
        self.sub_plot = []
        fig_id = 0
        for id in self.sensor_info['id']:
            for sensor_name in self.sensor_info['id'][id]:
                self.sub_plot.append(self.fig.add_subplot(self.sensor_info['finger_num'], self.sensor_info['sensor_per_finger'], fig_id+1))
                self.sensor_name.append(sensor_name)
                self.sub_plot[fig_id].set_title(sensor_name)
                self.sub_plot[fig_id].set_xlim(-0.6, 0.6)
                self.sub_plot[fig_id].set_ylim(-0.6, 0.6)
                fig_id += 1          
        plt.subplots_adjust(wspace=0.5, hspace=1.0)
        self.point_size = 20

    def draw_points(self, X, Y, Z, color='r'):
        for (fig_id,fig) in enumerate(self.sub_plot):
            fig.cla()
            if self.sensor_type[fig_id] == 'IP':
                total_force = np.sqrt(X[fig_id]**2 + Y[fig_id]**2 + Z[fig_id]**2) / self.totoal_force_limit
                fig.scatter(PAXINI_IP_VIS_COORDS_2D[:,0], PAXINI_IP_VIS_COORDS_2D[:,1], c=color, marker='o', s=self.point_size*total_force)
            elif self.sensor_type[fig_id] == 'DP':
                total_force = np.sqrt(X[fig_id]**2 + Y[fig_id]**2 + Z[fig_id]**2) / self.totoal_force_limit
                fig.scatter(PAXINI_DP_VIS_COORDS_2D[:,0], PAXINI_DP_VIS_COORDS_2D[:,1], c=color, marker='o', s=self.point_size*total_force)
            fig.set_title(self.sensor_name[fig_id])

    def draw(self, X, Y, Z):
        # Setting the plot limits
        # self._set_limits()

        # Plotting the lines to visualize the hand
        self.draw_points(X, Y, Z)
        # plt.draw()

        # Resetting and Pausing the 3D plot
        self.fig.canvas.flush_events()
        plt.pause(0.01)
        # plt.cla()
    
    def draw_single(self, X, Y, Z):
        # Setting the plot limits
        # self._set_limits()

        # Plotting the lines to visualize the hand
        self.draw_points(X, Y, Z)
        plt.show()

if __name__ == "__main__":
    plotter = Plot2DTACTILE(None)
    force_mag = 1
    import time
    while True:
        st = time.time()
        force = np.ones([8,15,3])*force_mag
        plotter.draw(force[:,:,0], force[:,:,1], force[:,:,2])
        force_mag += 0.5
        print(time.time() - st)