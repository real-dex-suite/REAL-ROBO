import os
import sys
import rospy 
import time
from sensor_msgs.msg import JointState
from holodex.utils.files import *
from holodex.constants import *
from holodex.utils.network import ImageSubscriber, frequency_timer, Float64MultiArray, TactileSubscriber
from holodex.tactile.utils import fetch_paxini_info
from termcolor import cprint

# load module according to hand type
hand_module = __import__("holodex.robot.hand")
Hand_module_name = f'{HAND_TYPE}Hand' if HAND_TYPE is not None else None
Hand = getattr(hand_module.robot, Hand_module_name) if HAND_TYPE is not None else None

class DataCollector(object):
    def __init__(
        self,
        num_tactiles,
        num_cams,
        keyboard_control,
        storage_path,
        data_collection_type,
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
        self.hand = Hand() if HAND_TYPE is not None else None

        # ROS Subscriber to get the arm information
        # self.arm_joint_state = None
        # self.arm_commanded_joint_state = None
        # self.arm_ee_pose = None
        # self.arm_commanded_ee_pose = None

        # rospy.Subscriber(JAKA_JOINT_STATE_TOPIC, JointState, self._callback_arm_joint_state, queue_size = 1)
        # rospy.Subscriber(JAKA_COMMANDED_JOINT_STATE_TOPIC, JointState, self._callback_arm_commanded_joint_state, queue_size = 1)
        # rospy.Subscriber(JAKA_EE_POSE_TOPIC, Float64MultiArray, self._callback_arm_ee_pose, queue_size = 1)
        # rospy.Subscriber(JAKA_COMMANDED_EE_POSE_TOPIC, Float64MultiArray, self._callback_arm_commanded_ee_pose, queue_size = 1)    
        self.keyboard_control = keyboard_control

        if 'arm_ee_pose' in data_collection_type:
            self.arm_ee_pose = None
            # Keyboard control subscriber
            if self.keyboard_control:
                self.hand_commanded_joint_position = None
                rospy.Subscriber(KEYBOARD_EE_TOPIC, Float64MultiArray, self._callback_keyboard_control_ee, queue_size = 1)
                rospy.Subscriber(KEYBOARD_HAND_TOPIC, Float64MultiArray, self._callback_keyboard_control_hand, queue_size = 1)
            else:
                rospy.Subscriber(JAKA_EE_POSE_TOPIC, Float64MultiArray, self._callback_arm_ee_pose, queue_size = 1)

        if 'arm_commanded_ee_pose' in data_collection_type:
            self.arm_commanded_ee_pose = None
            rospy.Subscriber(JAKA_COMMANDED_EE_POSE_TOPIC, Float64MultiArray, self._callback_arm_commanded_ee_pose, queue_size = 1)

        if 'arm_joint_positions' in data_collection_type:
            self.arm_joint_state = None
            rospy.Subscriber(JAKA_JOINT_STATE_TOPIC, JointState, self._callback_arm_joint_state, queue_size = 1)

        if 'arm_commanded_joint_state' in data_collection_type:
            self.arm_commanded_joint_state = None
            rospy.Subscriber(JAKA_COMMANDED_JOINT_STATE_TOPIC, JointState, self._callback_arm_commanded_joint_state, queue_size = 1)
        
        # Frequency timer
        self.frequency_timer = frequency_timer(RECORD_FPS)

    def _callback_keyboard_control_ee(self, data):
        self.arm_ee_pose = data

    def _callback_keyboard_control_hand(self, data):
        self.hand_commanded_joint_position = data

    def _callback_arm_commanded_ee_pose(self, data):
        self.arm_commanded_ee_pose = data

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
                        cprint('Tactile data not available!', 'red')
                        skip_loop = True

                if self.hand is not None and self.hand.get_hand_position() is None:
                    cprint('Hand data not available!', 'red')
                    skip_loop = True
                
                
                # if self.arm_joint_state is None or self.arm_ee_pose is None or self.arm_commanded_joint_state is None:
                #     cprint('Arm data not available!', 'red')
                #     skip_loop = True

                # TODO: fix this
                if self.arm_joint_state is None or self.arm_ee_pose is None:
                    cprint('Arm data not available!', 'red')
                    skip_loop = True

                for color_image_subscriber in self.color_image_subscribers:
                    if color_image_subscriber.get_image() is None:
                        cprint('Color image not available!', 'red')
                        skip_loop = True
                
                # Comment out the depth image subscriber for now
                for depth_image_subscriber in self.depth_image_subscribers:
                    if depth_image_subscriber.get_image() is None:
                        cprint('Depth image not available!', 'red')
                        skip_loop = True

                if skip_loop:
                    continue
                
                #print('Valid Data', time.time())
                cprint(f'Valid Data at {time.time()}', 'green', 'on_black', attrs=['bold'])
                state = dict()

                # Hand data
                if self.hand is not None:
                    state['hand_joint_positions'] = self.hand.get_hand_position() # follow orignal joint order, first mcp-pip, then palm-mcp
                    state['hand_joint_velocity'] = self.hand.get_hand_velocity()
                    state['hand_joint_effort'] = self.hand.get_hand_torque()
                
                if self.keyboard_control:
                    if self.hand_commanded_joint_position is not None:
                        state['hand_commanded_joint_position'] = self.hand_commanded_joint_position.data
                else:
                    state['hand_commanded_joint_position'] = self.hand.get_commanded_joint_position()

                # Arm data
                if getattr(self, 'arm_joint_state', None) is not None:
                    state['arm_joint_positions'] = self.arm_joint_state.position
                
                if getattr(self, 'arm_commanded_joint_state', None) is not None:
                    state['arm_commanded_joint_position'] = self.arm_commanded_joint_state.position

                if getattr(self, 'arm_ee_pose', None) is not None:
                    state['arm_ee_pose'] = self.arm_ee_pose.data

                if getattr(self, 'arm_commanded_ee_pose', None) is not None:
                    state['arm_commanded_ee_pose'] = self.arm_commanded_ee_pose.data

                #TODO: Arm Effort, Velocity

                # Image data
                for cam_num in range(self.num_cams):
                    state['camera_{}_color_image'.format(cam_num + 1)] = self.color_image_subscribers[cam_num].get_image()
                    state['camera_{}_depth_image'.format(cam_num + 1)] = self.depth_image_subscribers[cam_num].get_image()
                
                # tactile data
                tactile_data = {}
                for tactile_num in range(self.num_tactiles):
                    raw_datas = np.array(self.tactile_subscribers[tactile_num].get_data()).reshape(self.sensor_per_board, POINT_PER_SENSOR, FORCE_DIM_PER_POINT)
                    for (tactile_id, raw_data) in enumerate(raw_datas):
                        tactile_data[self.tactile_info['id'][tactile_num + 1][tactile_id]] = raw_data
                state['tactile_data'] = tactile_data

                # Temporal information
                state['time'] = time.time()

                # Saving the pickle file
                state_pickle_path = os.path.join(self.storage_path, f'{counter}')
                store_pickle_data(state_pickle_path, state)

                counter += 1
                # reset
                if self.keyboard_control:
                    self.arm_ee_pose = None
                    self.hand_commanded_joint_position = None

                self.frequency_timer.sleep()
        
        except KeyboardInterrupt:
            cprint('Finished recording! Data can be found in {}'.format(self.storage_path), 'green')
            sys.exit(0)