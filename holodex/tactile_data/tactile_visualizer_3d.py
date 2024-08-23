import rospy
from holodex.tactile.tactile_plotter_2d import *
from holodex.tactile.tactile_constants import *
from holodex.tactile_data.visualizer import TactileVisualizer
import torch

class Tactile3DVisualizer(object):
    def __init__(self, tactile_type="paxini", figsize=(12, 12), layout="3x3"):
        self.layout = layout
        self.figsize = figsize

        # Selecting the tactile
        if tactile_type == "paxini":
            self.fetch_paxini_info()
            self.tactile_visualizer = TactileVisualizer(figsize=self.figsize, layout=self.layout)
        else:
            raise NotImplementedError("There are no other tactile available. The only options are paxini!")

    def fetch_paxini_info(self):
        sensor_info = {}
        start_tags = []

        for group_id in PAXINI_GROUP_INFO:
            for finger_part_id in PAXINI_FINGER_PART_INFO:
                start_tags.append(PAXINI_FINGER_PART_INFO[finger_part_id] + PAXINI_GROUP_INFO[group_id])

        sensor_info = {
            "id": {},
            "finger_num": len(PAXINI_LEAPHAND),
            "sensor_per_finger": len(PAXINI_FINGER_PART_INFO),
            "sensor_type": []
        }

        for serial_port_number in SERIAL_PORT_NUMBERS:
            cur_id = int(serial_port_number[serial_port_number.find("USB") + 3]) + 1
            if cur_id not in sensor_info["id"]:
                sensor_info["id"][cur_id] = []

            for start_tag in start_tags:
                for finger in PAXINI_LEAPHAND:
                    if PAXINI_LEAPHAND[finger]["serial_port_number"] == serial_port_number:
                        current_group_id = PAXINI_LEAPHAND[finger]["group_id"]
                        for finger_part_id in PAXINI_FINGER_PART_INFO:
                            if PAXINI_FINGER_PART_INFO[finger_part_id] + PAXINI_GROUP_INFO[current_group_id] == start_tag:
                                sensor_info["id"][cur_id].append(finger + "_" + finger_part_id)
                                if "tip" in finger_part_id:
                                    sensor_info["sensor_type"].append("IP")
                                elif "pulp" in finger_part_id:
                                    sensor_info["sensor_type"].append("DP")

        self.sensor_info = sensor_info

    def plot_tactile(self, raw_data):
        sensor_order = []
        for id in self.sensor_info["id"]:
            sensor_names = self.sensor_info["id"][id]
            for sensor_name in sensor_names:
                assert sensor_name in raw_data.keys()
                sensor_order.append(sensor_name)

        processed_data = np.zeros(
            (
                self.sensor_info["finger_num"] * self.sensor_info["sensor_per_finger"],
                POINT_PER_SENSOR,
                FORCE_DIM_PER_POINT,
            )
        )

        for sensor_index, sensor_name in enumerate(sensor_order):
            processed_data[sensor_index, :, :] = raw_data[sensor_name]

        processed_data = processed_data.reshape((8, 5, 3, 3))

        if self.layout == "tactile_pattern":
            self.tactile_visualizer.get_whole_tactile_repre_pattern(processed_data)
        elif self.layout == "3x3":
            self.tactile_visualizer.visualize_tactile_data(processed_data)

    def plot_tactile_repre(self, raw_data):
        sensor_order, data = [], []
        for id in self.sensor_info["id"]:
            sensor_names = self.sensor_info["id"][id]
            for sensor_name in sensor_names:
                assert sensor_name in raw_data.keys()
                sensor_order.append(sensor_name)

        for sensor_name in sensor_order:
            data.extend(raw_data[sensor_name].reshape(-1).tolist())

        data = torch.tensor(data, dtype=torch.float64)