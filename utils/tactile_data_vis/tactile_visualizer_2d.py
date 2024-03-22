import numpy as np

from utils.tactile_data_vis.tactile_plotter_2d import *
from utils.tactile_data_vis.tactile_constants import *

class Tactile2DVisualizer(object):
    def __init__(self, tactile_type='paxini'):
        # Selecting the tactile
        if tactile_type == 'paxini':
            self.fetch_paxini_info()
            self.plotter2D = Plot2DTACTILE(self.sensor_info)
        else:
            raise NotImplementedError("There are no other tactile available. \
            The only options are paxini!")

    def fetch_paxini_info(self):    
        sensor_info = {}

        start_tags = []
        for group_id in PAXINI_GROUP_INFO: # read each group first
            for finger_part_id in PAXINI_FINGER_PART_INFO:    
                start_tags.append(PAXINI_FINGER_PART_INFO[finger_part_id]+PAXINI_GROUP_INFO[group_id])

        # get finger name which has tactile
        sensor_info = {}
        sensor_info['id'] = {}
        sensor_info['finger_num'] = len(PAXINI_LEAPHAND)
        sensor_info['sensor_per_finger'] = len(PAXINI_FINGER_PART_INFO)
        sensor_info['sensor_type'] = []

        # for each port, port has to be consistent with defined, and same with collection and deploy
        for serial_port_number in SERIAL_PORT_NUMBERS:
            cur_id = int(serial_port_number[serial_port_number.find("ACM")+3])+1
            if cur_id not in sensor_info['id']:
                sensor_info['id'][cur_id] = []

            for start_tag in start_tags:
                for finger in PAXINI_LEAPHAND:
                    if PAXINI_LEAPHAND[finger]['serial_port_number'] == serial_port_number:
                        current_group_id = PAXINI_LEAPHAND[finger]['group_id']
                        for finger_part_id in PAXINI_FINGER_PART_INFO:
                            if  PAXINI_FINGER_PART_INFO[finger_part_id]+PAXINI_GROUP_INFO[current_group_id] == start_tag:
                                sensor_info['id'][cur_id].append(finger+'_'+finger_part_id)
                                if 'tip' in finger_part_id:
                                    sensor_info['sensor_type'].append('IP')
                                elif 'pulp' in finger_part_id:
                                    sensor_info['sensor_type'].append('DP')

        self.sensor_info = sensor_info  

    def stream(self, raw_data):
        sensor_order = []
        for id in self.sensor_info['id']:
            sensor_names = self.sensor_info['id'][id]
            for sensor_name in sensor_names:
                assert sensor_name in raw_data.keys()
                sensor_order.append(sensor_name)
                
        processed_data = np.zeros((self.sensor_info['finger_num']* self.sensor_info['sensor_per_finger'], POINT_PER_SENSOR, FORCE_DIM_PER_POINT))

        for (sensor_index, sensor_name) in enumerate(sensor_order):
            processed_data[sensor_index,:,:] = raw_data[sensor_name]

        self.plotter2D.draw(processed_data[:,:,0], processed_data[:,:,1], processed_data[:,:,2])

    def plot_once(self, raw_data, save_img_path=None):
        sensor_order = []
        for id in self.sensor_info['id']:
            sensor_names = self.sensor_info['id'][id]
            for sensor_name in sensor_names:
                assert sensor_name in raw_data.keys()
                sensor_order.append(sensor_name)
                
        processed_data = np.zeros((self.sensor_info['finger_num']* self.sensor_info['sensor_per_finger'], POINT_PER_SENSOR, FORCE_DIM_PER_POINT))

        for (sensor_index, sensor_name) in enumerate(sensor_order):
            processed_data[sensor_index,:,:] = raw_data[sensor_name]

        self.plotter2D.draw_single(processed_data[:,:,0], processed_data[:,:,1], processed_data[:,:,2], save_img_path=save_img_path)

if __name__ == "__main__":
    tactile_visualizer = Tactile2DVisualizer('paxini')
    # based on SERIAL_PORT_NUMBERS
    # stream_1 = PaxiniTactileStream('/dev/ttyACM0', 1)
    # stream_2 = PaxiniTactileStream('/dev/ttyACM1', 2)
    # while True:
    #     raw_data_1 = stream_1.get_data()
    #     raw_data_2 = stream_2.get_data()
    #     raw_data = {**raw_data_1, **raw_data_2}

    #     tactile_visualizer.stream(raw_data)

    # plot a single tactile data
    import pickle
    from holodex.utils.files import get_pickle_data
    data = get_pickle_data("/home/agibot/Projects/Real-Robo/expert_dataset/recorded_data/demonstration_1/117")
    tactile_visualizer.plot_once(data['tactile_data'])