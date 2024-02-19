import numpy as np
import rospy

from std_msgs.msg import Float64MultiArray

from holodex.viz.plotters.tactile_plotter_2d import *
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
            self.fetch_paxini_info()
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

    def fetch_paxini_info(self):    
        self.sensor_info = {}

        self.start_tag = []
        for group in PAXINI_GROUP_IDS: # read each group first
            for finger in PAXINI_FINGER_IDS:    
                self.start_tag.append(finger+group)

        # get finger name which has tactile
        self.sensor_info = {}
        self.sensor_info['id'] = {}
        self.sensor_info['finger_num'] = 0
        self.sensor_info['sensor_per_finger'] = len(PAXINI_FINGER_IDS)
        self.sensor_info['sensor_type'] = []

        self.serial_port_number = []
        for finger in PAXINI_LEAPHAND.keys():
            self.sensor_info['finger_num'] += 1

            cur_serial_port_number = PAXINI_LEAPHAND[finger]['serial_port_number']
            self.serial_port_number.append(cur_serial_port_number)

            cur_id = int(cur_serial_port_number[cur_serial_port_number.find("ACM")+3])+1
            if cur_id not in self.sensor_info['id']:
                self.sensor_info['id'][cur_id] = []

            for part_name in PAXINI_FINGER_PART_NAMES.keys():
                self.sensor_info['id'][cur_id].append(finger+'_'+part_name)
                if 'tip' in part_name:
                    self.sensor_info['sensor_type'].append('IP')
                elif 'pulp' in part_name:
                    self.sensor_info['sensor_type'].append('DP')
    
        self.serial_port_number = list(set(self.serial_port_number))

        self.tactile_topic = []
        self.raw_data = {}
        for cur_serial_port_number in self.serial_port_number:
            idx = int(cur_serial_port_number[cur_serial_port_number.find("ACM")+3])
            self.tactile_topic.append('/tactile_{}/raw_data'.format(idx+1))
            self.raw_data[idx+1] = None

        self.sensor_per_board = len(PAXINI_FINGER_IDS) * len(PAXINI_GROUP_IDS)

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