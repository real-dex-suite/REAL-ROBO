import os
import sys
import rospy 
import time
import shutil
import termios
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

from holodex.utils.files import *
from holodex.constants import *
from holodex.utils.network import ImageSubscriber, frequency_timer, Float64MultiArray, TactileSubscriber
from holodex.tactile.utils import fetch_paxini_info
from termcolor import cprint
from pynput import keyboard

def clear_input_buffer():
    termios.tcflush(sys.stdin, termios.TCIFLUSH)

if HAND_TYPE is not None:
    # load module according to hand type
    hand_module = __import__("holodex.robot.hand")
    Hand_module_name = f'{HAND_TYPE}Hand'
    Hand = getattr(hand_module.robot, Hand_module_name)

class AutoDataCollector(object):
    def __init__(
        self,
        num_tactiles,
        num_cams,
        storage_root,
        arm_type="franka",
        gripper="ctek",
    ):
        rospy.init_node('data_extractor', disable_signals = True)

        # read the first demo number
        demo_num = input("Enter the first demo number: ")

        self.storage_root = storage_root
        self.demo_start_idx = int(demo_num)
        self.demo_num = int(demo_num)
        self.num_tactiles = num_tactiles
        self.storage_path = os.path.join(self.storage_root, f'demonstration_{self.demo_num}')
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

        self.with_gripper = gripper is not None
        self.gripper = gripper
        # arm collector initialization
        if self.arm_type == "flexiv":
            self.data_collection_topic_type = "jaka"
        elif self.arm_type == "franka":
            self.data_collection_topic_type = "franka"
        elif self.arm_type == "jaka":
            self.data_collection_topic_type = "jaka"
        else:
            raise NotImplementedError(f"Unknown arm type {arm_type}")
   
        # Hand controller initialization
        self.hand = Hand() if HAND_TYPE is not None else None

        # keyboard cmd
        self.keyboard_listener = keyboard.Listener(on_press=self._on_press)
        self.keyboard_listener.start()
      
        # Frequency timer
        self.frequency_timer = frequency_timer(RECORD_FPS)

        # ros publish for reset
        self.stop = False
        self.reset_publisher = rospy.Publisher("/data_collector/reset_robot", Bool, queue_size=1)
        self.stop_publisher = rospy.Publisher("/data_collector/stop_move", Bool, queue_size=1)   
        self.reset_done_publisher = rospy.Publisher("/data_collector/reset_done", Bool, queue_size=1)
        self.end_publisher = rospy.Publisher("/data_collector/end_robot", Bool, queue_size=1)

    def _setup_state_collection(self):
        self.arm_ee_pose = None
        rospy.Subscriber(f"/{self.data_collection_topic_type}/ee_pose", Float64MultiArray, self._callback_arm_ee_pose, queue_size = 1)

        self.arm_commanded_ee_pose = None
        rospy.Subscriber(f"/{self.data_collection_topic_type}/commanded_ee_pose", Float64MultiArray, self._callback_arm_commanded_ee_pose, queue_size = 1)

        self.arm_joint_state = None
        rospy.Subscriber(f"/{self.data_collection_topic_type}/joint_states", JointState, self._callback_arm_joint_state, queue_size = 1)

        self.arm_commanded_joint_state = None
        rospy.Subscriber(f"/{self.data_collection_topic_type}/commanded_joint_states", JointState, self._callback_arm_commanded_joint_state, queue_size = 1)
    
    def _callback_arm_commanded_ee_pose(self, data):
        self.arm_commanded_ee_pose = data

    def _callback_arm_joint_state(self, data):
        self.arm_joint_state = data
    
    def _callback_arm_commanded_joint_state(self, data):
        self.arm_commanded_joint_state = data
    
    def _callback_arm_ee_pose(self, data):
        self.arm_ee_pose = data

    def _collect_state_data(self):
        """
        Collect all state data into a dictionary

        Returns:
            dict: Dictionary containing all collected state data
        """
        state = {}

        # Add arm data directly to state for compatibility with other collectors
        if self.arm_joint_state is not None:
            state['arm_joint_positions'] = self.arm_joint_state.position
        if self.arm_commanded_joint_state is not None:
            state['arm_commanded_joint_position'] = self.arm_commanded_joint_state.position
        if self.arm_ee_pose is not None:
            state['arm_ee_pose'] = self.arm_ee_pose.data
        if self.arm_commanded_ee_pose is not None:
            state['arm_commanded_ee_pose'] = self.arm_commanded_ee_pose.data

        return state
    
    def _on_press(self, key):
        try:
            if not self.stop: 
                if key.char == 's':
                    print(f"Key pressed: {key.char}")
                    self.stop = True
        except AttributeError:
            # print(f"Special key {key} pressed")
            pass
   
    def extract(self):
        counter = 1
        try:
            while True:
                if not self.keyboard_listener.is_alive():
                    self.keyboard_listener = keyboard.Listener(on_press=self._on_press)
                    self.keyboard_listener.start()
                skip_loop = False

                # Checking for broken data streams
                for tactile_subscriber in self.tactile_subscribers:
                    if tactile_subscriber.get_data() is None:
                        cprint('Tactile data not available!', 'red')
                        skip_loop = True

                if self.hand is not None and self.hand.get_hand_position() is None:
                    cprint('Hand data not available!', 'red')
                    skip_loop = True

                # TODO: fix this
                if self.arm_collector.arm_joint_state is None or self.arm_collector.arm_ee_pose is None:
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

                # Arm data
                arm_state = self._collect_state_data()
                state.update(arm_state)

                # Hand data
                if self.hand is not None:
                    state['hand_joint_positions'] = self.hand.get_hand_position() # follow orignal joint order, first mcp-pip, then palm-mcp
                    state['hand_joint_velocity'] = self.hand.get_hand_velocity()
                    state['hand_joint_effort'] = self.hand.get_hand_torque()
                    state['hand_commanded_joint_position'] = self.hand.get_commanded_joint_position()

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
                state['time'] = rospy.Time.now().to_time()

                # Saving the pickle file save path
                # # TODO: add delete functionality
                if not os.path.exists(self.storage_path):
                    os.makedirs(self.storage_path)
                state_pickle_path = os.path.join(self.storage_path, f'{counter}')
                store_pickle_data(state_pickle_path, state)

                counter += 1
                self.frequency_timer.sleep()

                #################################################### 
                if self.stop:
                    cprint(f'Successfully record {self.demo_num} traj! Data can be found in {self.storage_path}', 'green')
                    self.stop = False

                    bool_true_msg = Bool()
                    bool_true_msg.data = True

                    bool_false_msg = Bool()
                    bool_false_msg.data = False

                    # stop robot move
                    self.stop_publisher.publish(bool_true_msg)

                    # stuck here waiting for the next command, c for continue, d for delete, r for reset, s for stop
                    while True:
                        cprint('Waiting for the next command: ', 'yellow')
                        cprint('c -> continue, d -> delete, r -> reset, q -> quit, x -> stop robot', 'yellow')
                        clear_input_buffer()
                        input_cmd = input("Enter the next command: ")

                        if input_cmd == 'c':
                            # update demo_num and continue recording
                            counter = 1
                            self.demo_num += 1
                            self.storage_path = os.path.join(self.storage_root, f'demonstration_{self.demo_num}')
                            self.reset_done_publisher.publish(bool_true_msg)
                            cprint(f"Start recording at {self.storage_path}", 'green')

                            # reset
                            self.reset_publisher.publish(bool_true_msg)
                            rospy.sleep(1)

                            # start robot move
                            self.stop_publisher.publish(bool_false_msg)

                            break

                        elif input_cmd == 'd':
                            # remove the last data
                            shutil.rmtree(self.storage_path)
                            cprint(f"Removing the last data at {self.storage_path}", 'red')
                            self.demo_num -= 1
                            continue

                        elif input_cmd == 'r':
                            # reset the robot
                            cprint(f"Resetting the robot...", 'blue')
                            self.reset_publisher.publish(bool_true_msg)
                            rospy.sleep(1)
                            continue

                        elif input_cmd == 'q':
                            # quit the program
                            cprint(f'Finished recording! Sucessfully record {self.demo_start_idx} ~ {self.demo_num} traj! Data can be found in {self.storage_root}', 'green')
                            self.end_publisher.publish(bool_true_msg)
                            sys.exit(0)

                        elif input_cmd == 'x':
                            # stop the robot move
                            cprint(f"Stop the robot move...", 'red')
                            self.stop_publisher.publish(bool_true_msg)
                            continue

                        else:

                            cprint(f'Invalid command {input_cmd}!', 'red')
                
        except KeyboardInterrupt:
            cprint('Finished recording! Data can be found in {}'.format(self.storage_path), 'green')
            sys.exit(0)