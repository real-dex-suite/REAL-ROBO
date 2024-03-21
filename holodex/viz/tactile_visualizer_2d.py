import numpy as np
import rospy

from std_msgs.msg import Float64MultiArray

from holodex.viz.plotters.tactile_plotter_2d import *
from holodex.tactile.utils import fetch_paxini_info
from holodex.constants import *

from mpl_toolkits.mplot3d import Axes3D

class Tactile2DVisualizer(object):
    def __init__(self, tactile_type, *args):
        # Initializing a ROS node
        try:
            rospy.init_node("{}_tactile_2d_visualizer".format(tactile_type))
        except:
            pass

        # Selecting the tactile
        if tactile_type == 'paxini':
            self.sensor_info, self.tactile_topic, self.raw_data, self.sensor_per_board = fetch_paxini_info()
            for rostopic in self.tactile_topic:
                rospy.Subscriber(
                    rostopic, 
                    Float64MultiArray, 
                    self._callback_raw_data, 
                    queue_size = 1
                )
            self.plotter2D = Plot2DTACTILE(self.sensor_info)
        else:
            raise NotImplementedError("There are no other tactile available. \
            The only options are paxini!")

    # def fetch_paxini_info(self):    
    #     self.sensor_info = {}

    #     self.start_tags = []
    #     for group_id in PAXINI_GROUP_INFO: # read each group first
    #         for finger_part_id in PAXINI_FINGER_PART_INFO:    
    #             self.start_tags.append(PAXINI_FINGER_PART_INFO[finger_part_id]+PAXINI_GROUP_INFO[group_id])

    #     # get finger name which has tactile
    #     self.sensor_info = {}
    #     self.sensor_info['id'] = {}
    #     self.sensor_info['finger_num'] = len(PAXINI_LEAPHAND)
    #     self.sensor_info['sensor_per_finger'] = len(PAXINI_FINGER_PART_INFO)
    #     self.sensor_info['sensor_type'] = []

    #     self.serial_port_number = []
    #     for finger in PAXINI_LEAPHAND.keys():
    #         cur_serial_port_number = PAXINI_LEAPHAND[finger]['serial_port_number']
    #         self.serial_port_number.append(cur_serial_port_number)
    #     self.serial_port_number = list(set(self.serial_port_number))
    #     self.serial_port_number.sort()  
        
    #     # for each port, port has to be consistent with defined, and same with collection and deploy
    #     for serial_port_number in self.serial_port_number:
    #         cur_id = int(serial_port_number[serial_port_number.find("ACM")+3])+1
    #         if cur_id not in self.sensor_info['id']:
    #             self.sensor_info['id'][cur_id] = []

    #         for start_tag in self.start_tags:
    #             for finger in PAXINI_LEAPHAND:
    #                 if PAXINI_LEAPHAND[finger]['serial_port_number'] == serial_port_number:
    #                     current_group_id = PAXINI_LEAPHAND[finger]['group_id']
    #                     for finger_part_id in PAXINI_FINGER_PART_INFO:
    #                         if  PAXINI_FINGER_PART_INFO[finger_part_id]+PAXINI_GROUP_INFO[current_group_id] == start_tag:
    #                             self.sensor_info['id'][cur_id].append(finger+'_'+finger_part_id)
    #                             if 'tip' in finger_part_id:
    #                                 self.sensor_info['sensor_type'].append('IP')
    #                             elif 'pulp' in finger_part_id:
    #                                 self.sensor_info['sensor_type'].append('DP')
    
    #     self.tactile_topic = []
    #     self.raw_data = {}
    #     for cur_serial_port_number in self.serial_port_number:
    #         idx = int(cur_serial_port_number[cur_serial_port_number.find("ACM")+3])
    #         self.tactile_topic.append('/tactile_{}/raw_data'.format(idx+1))
    #         self.raw_data[idx+1] = None

    #     self.sensor_per_board = len(PAXINI_FINGER_PART_INFO) * len(PAXINI_GROUP_INFO)

    def _callback_raw_data(self, raw_data):
        id = raw_data.layout.data_offset
        self.raw_data[id] = np.array(raw_data.data).reshape(self.sensor_per_board, POINT_PER_SENSOR, FORCE_DIM_PER_POINT)

    def stream(self):
        while True:
            raw_data = np.zeros((self.sensor_info['finger_num']* self.sensor_info['sensor_per_finger'], POINT_PER_SENSOR, FORCE_DIM_PER_POINT))
            for id in self.raw_data.keys():
                if self.raw_data[id] is not None:
                    raw_data[(id-1)*self.sensor_per_board:id*self.sensor_per_board,:,:] = self.raw_data[id]
            
            self.plotter2D.draw(raw_data[:,:,0], raw_data[:,:,1], raw_data[:,:,2])
            # if self.raw_data is None:
            #     raw_data = np.zeros((self.num_keypoints, 2))
            #     self.plotter2D.draw(keypoints[:, 0], keypoints[:, 1])
            # else:
            #     self.plotter2D.draw(self.keypoints[:, 0], self.keypoints[:, 1])

if __name__ == "__main__":
    tactile_visualizer = Tactile2DVisualizer('paxini')
    tactile_visualizer.stream()