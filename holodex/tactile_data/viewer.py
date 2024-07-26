import rospy
import numpy as np
from holodex.utils.network import TactileSubscriber
from holodex.tactile.utils import fetch_paxini_info
from holodex.constants import *
from holodex.tactile_data.tactile_visualizer_3d import Tactile3DVisualizer
from holodex.tactile_data.tactile_visualizer_2d import Tactile2DVisualizer

class TactileDataViewer(object):
    def __init__(self, tactile_viewer_type):
        rospy.init_node('tactile_view_listener', anonymous=True)
        self.num_tactiles = 2
        self.tactile_info, _, _, self.sensor_per_board = fetch_paxini_info()
        self.tactile_subscribers = [TactileSubscriber(tactile_num=tactile_num + 1) for tactile_num in range(self.num_tactiles)]
        self.viewer_type = tactile_viewer_type

        if tactile_viewer_type == "3D":
            self.tactile_visualizer = Tactile3DVisualizer("paxini", layout="3x3")
        elif tactile_viewer_type == "2D":
            self.tactile_visualizer = Tactile2DVisualizer()
        else:
            rospy.logwarn("Invalid tactile viewer type")

        rospy.loginfo("Tactile subscribers initialized")

    def retrieve_tactile_data(self):
        tactile_data = {}
        for tactile_num in range(self.num_tactiles):
            tactile_subscriber = self.tactile_subscribers[tactile_num]
            data = tactile_subscriber.get_data()
            if data is None:
                rospy.logwarn(f'Tactile {tactile_num + 1} data not available!')
                continue
           
            raw_datas = np.array(data).reshape(self.sensor_per_board, POINT_PER_SENSOR, FORCE_DIM_PER_POINT)
            for tactile_id, raw_data in enumerate(raw_datas):
                tactile_data[self.tactile_info['id'][tactile_num + 1][tactile_id]] = raw_data

        return tactile_data
    
    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            tactile_data = self.retrieve_tactile_data()
            # print(tactile_data.keys)
            if self.viewer_type == "3D":
                 self.tactile_visualizer.plot_tactile(tactile_data)
            elif self.viewer_type == "2D":
                 self.tactile_visualizer.stream(tactile_data)
            else:
                rospy.logwarn("Invalid tactile viewer type")
            rate.sleep()

if __name__ == '__main__':
    listener = TactileDataViewer("3D")
    viewer_type = "2D"
    if viewer_type == "3D":
        viewer = Tactile3DVisualizer("paxini", layout="3x3")
    elif viewer_type == "2D":
        viewer = Tactile2DVisualizer()


    listener.run()
