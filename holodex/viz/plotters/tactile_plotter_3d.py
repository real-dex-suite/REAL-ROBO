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
        # Figure settings
        self.fig = plt.figure()
        self.sub_plot = []
        fig_id = 0
        for id in self.sensor_info['id']:
            for sensor_name in self.sensor_info['id'][id]:
                self.sub_plot.append(self.fig.add_subplot(self.sensor_info['finger_num'], self.sensor_info['sensor_per_finger'], fig_id+1, projection='3d'))
                self.sub_plot[fig_id].set_title(sensor_name)
                fig_id += 1          
        plt.subplots_adjust(wspace=0.5, hspace=1.0)

    def fetch_paxini_info(self):    
        self.sensor_info = {}

        self.start_tags = []
        for group_id in PAXINI_GROUP_INFO: # read each group first
            for finger_part_id in PAXINI_FINGER_PART_INFO:    
                self.start_tags.append(PAXINI_FINGER_PART_INFO[finger_part_id]+PAXINI_GROUP_INFO[group_id])

        # get finger name which has tactile
        self.sensor_info = {}
        self.sensor_info['id'] = {}
        self.sensor_info['finger_num'] = len(PAXINI_LEAPHAND)
        self.sensor_info['sensor_per_finger'] = len(PAXINI_FINGER_PART_INFO)
        self.sensor_info['sensor_type'] = []

        self.serial_port_number = []
        for finger in PAXINI_LEAPHAND.keys():
            cur_serial_port_number = PAXINI_LEAPHAND[finger]['serial_port_number']
            self.serial_port_number.append(cur_serial_port_number)
        self.serial_port_number = list(set(self.serial_port_number))
        self.serial_port_number.sort()  
        
        # for each port, port has to be consistent with defined, and same with collection and deploy
        for serial_port_number in self.serial_port_number:
            cur_id = int(serial_port_number[serial_port_number.find("USB")+3])+1
            if cur_id not in self.sensor_info['id']:
                self.sensor_info['id'][cur_id] = []

            for start_tag in self.start_tags:
                for finger in PAXINI_LEAPHAND:
                    if PAXINI_LEAPHAND[finger]['serial_port_number'] == serial_port_number:
                        current_group_id = PAXINI_LEAPHAND[finger]['group_id']
                        for finger_part_id in PAXINI_FINGER_PART_INFO:
                            if  PAXINI_FINGER_PART_INFO[finger_part_id]+PAXINI_GROUP_INFO[current_group_id] == start_tag:
                                self.sensor_info['id'][cur_id].append(finger+'_'+finger_part_id)
                                if 'tip' in finger_part_id:
                                    self.sensor_info['sensor_type'].append('IP')
                                elif 'pulp' in finger_part_id:
                                    self.sensor_info['sensor_type'].append('DP')
    
        self.tactile_topic = []
        self.raw_data = {}
        for cur_serial_port_number in self.serial_port_number:
            idx = int(cur_serial_port_number[cur_serial_port_number.find("USB")+3])
            self.tactile_topic.append('/tactile_{}/raw_data'.format(idx+1))
            self.raw_data[idx+1] = None

        self.sensor_per_board = len(PAXINI_FINGER_PART_INFO) * len(PAXINI_GROUP_INFO)

    def _set_limits(self):
        plt.axis([-0.12, 0.12, -0.02, 0.2])
    
    def draw_force(self, X, Y, Z, color='black'):
        # TODO draw according to force direction
        for (fig_id,fig) in enumerate(self.sub_plot):
            if self.sensor_type[fig_id] == 'IP':
                fig.scatter(PAXINI_IP_VIS_COORDS[:,0], PAXINI_IP_VIS_COORDS[:,1], PAXINI_IP_VIS_COORDS[:,2], c=color, marker='o', s=10)
                force_x_coords = PAXINI_IP_VIS_COORDS[:,0] + X[fig_id]
                force_y_coords = PAXINI_IP_VIS_COORDS[:,1] + Y[fig_id]
                force_z_coords = PAXINI_IP_VIS_COORDS[:,2] + Z[fig_id]
                fig.quiver(PAXINI_IP_VIS_COORDS[:,0], PAXINI_IP_VIS_COORDS[:,1], PAXINI_IP_VIS_COORDS[:,2], force_x_coords, PAXINI_IP_VIS_COORDS[:,1], PAXINI_IP_VIS_COORDS[:,2], color='r')
                fig.quiver(PAXINI_IP_VIS_COORDS[:,0], PAXINI_IP_VIS_COORDS[:,1], PAXINI_IP_VIS_COORDS[:,2], PAXINI_IP_VIS_COORDS[:,0], force_y_coords, PAXINI_IP_VIS_COORDS[:,2], color='g')
                fig.quiver(PAXINI_IP_VIS_COORDS[:,0], PAXINI_IP_VIS_COORDS[:,1], PAXINI_IP_VIS_COORDS[:,2], PAXINI_IP_VIS_COORDS[:,0], PAXINI_IP_VIS_COORDS[:,1], force_z_coords, color='b')
                
            elif self.sensor_type[fig_id] == 'DP':
                fig.scatter(PAXINI_DP_VIS_COORDS[:,0], PAXINI_DP_VIS_COORDS[:,1], PAXINI_DP_VIS_COORDS[:,2], c=color, marker='o', s=10)
                force_x_coords = PAXINI_DP_VIS_COORDS[:,0] + X[fig_id]
                force_y_coords = PAXINI_DP_VIS_COORDS[:,1] + Y[fig_id]
                force_z_coords = PAXINI_DP_VIS_COORDS[:,2] + Z[fig_id]
                fig.quiver(PAXINI_DP_VIS_COORDS[:,0], PAXINI_DP_VIS_COORDS[:,1], PAXINI_DP_VIS_COORDS[:,2], force_x_coords, PAXINI_DP_VIS_COORDS[:,1], PAXINI_DP_VIS_COORDS[:,2], color='r')
                fig.quiver(PAXINI_DP_VIS_COORDS[:,0], PAXINI_DP_VIS_COORDS[:,1], PAXINI_DP_VIS_COORDS[:,2], PAXINI_DP_VIS_COORDS[:,0], force_y_coords, PAXINI_DP_VIS_COORDS[:,2], color='g')
                fig.quiver(PAXINI_DP_VIS_COORDS[:,0], PAXINI_DP_VIS_COORDS[:,1], PAXINI_DP_VIS_COORDS[:,2], PAXINI_DP_VIS_COORDS[:,0], PAXINI_DP_VIS_COORDS[:,1], force_z_coords, color='b')

    def draw_points(self, X, Y, Z, color='r'):
        for (fig_id,fig) in enumerate(self.sub_plot):
            if self.sensor_type[fig_id] == 'IP':
                total_force = np.sqrt(X[fig_id]**2 + Y[fig_id]**2 + Z[fig_id]**2) / self.totoal_force_limit
                fig.scatter(PAXINI_IP_VIS_COORDS[:,0], PAXINI_IP_VIS_COORDS[:,1], PAXINI_IP_VIS_COORDS[:,2], c=color, marker='o', s=10*total_force)
            elif self.sensor_type[fig_id] == 'DP':
                total_force = np.sqrt(X[fig_id]**2 + Y[fig_id]**2 + Z[fig_id]**2) / self.totoal_force_limit
                fig.scatter(PAXINI_DP_VIS_COORDS[:,0], PAXINI_DP_VIS_COORDS[:,1], PAXINI_DP_VIS_COORDS[:,2], c=color, marker='o', s=10*total_force)

    def draw(self, X, Y, Z):
        # Setting the plot limits
        # self._set_limits()

        # Plotting the lines to visualize the hand
        self.draw_points(X, Y, Z)
        plt.draw()

        # Resetting and Pausing the 3D plot
        self.fig.canvas.flush_events()
        plt.pause(0.01)
        plt.cla()

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