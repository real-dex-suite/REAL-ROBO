import os
import sys
import rospy 
import time
from sensor_msgs.msg import JointState
from holodex.utils.files import *
from holodex.constants import *
from holodex.utils.network import ImageSubscriber, frequency_timer, Float64MultiArray, TactileSubscriber
from holodex.tactile.utils import fetch_paxini_info

# load module according to hand type
hand_module = __import__("holodex.robot.hand")
Hand_module_name = f'{HAND_TYPE}Hand'
Hand = getattr(hand_module.robot, Hand_module_name)

class DataCollector(object):
    def __init__(
        self,
        num_tactiles,
        num_cams,
        storage_path
    ):
        rospy.init_node('data_extractor', disable_signals = True)

        self.storage_path = storage_path

        # ROS Subscribers based on the number of tactile board used
        self.num_tactiles = num_tactiles
        if self.num_tactiles > 0:
            self.tactile_info, _, _, self.sensor_per_board = fetch_paxini_info()
        self.tactile_subscribers = []
        for tactile_num in range(self.num_tactiles):
            self.tactile_subscribers.append(
                TactileSubscriber(
                    tactile_num = tactile_num + 1
                )
            )

        # ROS Subscribers based on the number of cameras used
        self.num_cams = num_cams

        self.color_image_subscribers, self.depth_image_subscribers = [], []
        for cam_num in range(self.num_cams):
            self.color_image_subscribers.append(
                ImageSubscriber(
                    subscriber_name = '/robot_camera_{}/color_image'.format(cam_num + 1),
                    node_name = 'robot_camera_{}_color_data_collector'.format(cam_num + 1)
                )
            )
            self.depth_image_subscribers.append(
                ImageSubscriber(
                    subscriber_name = '/robot_camera_{}/depth_image'.format(cam_num + 1),
                    node_name = 'robot_camera_{}_depth_data_collector'.format(cam_num + 1),
                    color = False
                )
            )

        # Hand controller initialization
        self.hand = Hand()

        # ROS Subscriber to get the arm information
        self.arm_joint_state = None
        self.arm_ee_pose = None
        rospy.Subscriber(JAKA_JOINT_STATE_TOPIC, JointState, self._callback_arm_joint_state, queue_size = 1)
        rospy.Subscriber(JAKA_COMMANDED_JOINT_STATE_TOPIC, JointState, self._callback_arm_commanded_joint_state, queue_size = 1)
        rospy.Subscriber(JAKA_EE_POSE_TOPIC, Float64MultiArray, self._callback_arm_ee_pose, queue_size = 1)

        self.frequency_timer = frequency_timer(RECORD_FPS)

    def _callback_arm_joint_state(self, data):
        self.arm_joint_state = data
    
    def _callback_arm_commanded_joint_state(self, data):
        self.arm_commanded_joint_state = data
    
    def _callback_arm_ee_pose(self, data):
        self.arm_ee_pose = data

    def extract(self, offset = 0):
        counter = offset + 1
        try:
            while True:
                skip_loop = False

                # Checking for broken data streams
                for tactile_subscriber in self.tactile_subscribers:
                    if tactile_subscriber.get_data() is None:
                        print('Tactile data not available!')
                        skip_loop = True

                if self.hand.get_hand_position() is None:
                    print('Hand data not available!')
                    skip_loop = True
                
                if self.arm_joint_state is None or self.arm_ee_pose is None:
                    print('Arm data not available!')
                    skip_loop = True

                for color_image_subscriber in self.color_image_subscribers:
                    if color_image_subscriber.get_image() is None:
                        print('Color image not available!')
                        skip_loop = True
                
                # Comment out the depth image subscriber for now
                # for depth_image_subscriber in self.depth_image_subscribers:
                #     if depth_image_subscriber.get_image() is None:
                #         skip_loop = True

                if skip_loop:
                    continue
                
                print('Valid Data', time.time())
                state = dict()
                
                # tactile data
                tactile_data = {}
                for tactile_num in range(self.num_tactiles):
                    raw_datas = np.array(self.tactile_subscribers[tactile_num].get_data()).reshape(self.sensor_per_board, POINT_PER_SENSOR, FORCE_DIM_PER_POINT)
                    for (tactile_id, raw_data) in enumerate(raw_datas):
                        tactile_data[self.tactile_info['id'][tactile_num + 1][tactile_id]] = raw_data
                state['tactile_data'] = tactile_data

                # Hand data
                state['hand_joint_positions'] = self.hand.get_hand_position() # follow orignal joint order, first mcp-pip, then palm-mcp
                state['hand_joint_velocity'] = self.hand.get_hand_velocity()
                state['hand_joint_effort'] = self.hand.get_hand_torque()
                state['hand_commanded_joint_position'] = self.hand.get_commanded_joint_position()

                # Arm data
                state['arm_joint_positions'] = self.arm_joint_state.position
                state['arm_ee_pose'] = self.arm_ee_pose.data
                state['arm_commanded_joint_position'] = self.arm_commanded_joint_state.position

                for cam_num in range(self.num_cams):
                    state['camera_{}_color_image'.format(cam_num + 1)] = self.color_image_subscribers[cam_num].get_image()
                    # state['camera_{}_depth_image'.format(cam_num + 1)] = self.depth_image_subscribers[cam_num].get_image()

                # Temporal information
                state['time'] = time.time()

                # Saving the pickle file
                state_pickle_path = os.path.join(self.storage_path, f'{counter}')
                store_pickle_data(state_pickle_path, state)

                counter += 1

                self.frequency_timer.sleep()
        
        except KeyboardInterrupt:
            print('Finished recording! Data can be found in {}'.format(self.storage_path))
            sys.exit(0)