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
from tqdm import tqdm

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

        self.storage_root = storage_root
        self.demo_num = 1
        self.num_tactiles = num_tactiles
        self.arm_type = arm_type

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
      
        # Frequency timer
        self.frequency_timer = frequency_timer(RECORD_FPS)

        # ros publish for reset
        self.start = False
        self.stop = False
        self.c_pressed = False
        self.r_pressed = False
        self.q_pressed = False
        self.e_pressed = False
        self.i_pressed = False
        self.p_pressed = False
        self._setup_state_collection()

    def _setup_state_collection(self):
        self.arm_ee_pose = None
        rospy.Subscriber(f"/{self.data_collection_topic_type}/ee_pose", Float64MultiArray, self._callback_arm_ee_pose, queue_size = 1)

        self.arm_commanded_ee_pose = None
        rospy.Subscriber(f"/{self.data_collection_topic_type}/commanded_ee_pose", Float64MultiArray, self._callback_arm_commanded_ee_pose, queue_size = 1)

        self.arm_joint_state = None
        rospy.Subscriber(f"/{self.data_collection_topic_type}/joint_states", JointState, self._callback_arm_joint_state, queue_size = 1)

        self.arm_commanded_joint_state = None
        rospy.Subscriber(f"/{self.data_collection_topic_type}/commanded_joint_states", JointState, self._callback_arm_commanded_joint_state, queue_size = 1)

        if self.with_gripper:
            self.gripper_control = None
            rospy.Subscriber(f"/{self.data_collection_topic_type}/gripper_control", Bool, self._callback_gripper_control, queue_size = 1)

    def _callback_arm_commanded_ee_pose(self, data):
        self.arm_commanded_ee_pose = data

    def _callback_arm_joint_state(self, data):
        self.arm_joint_state = data
    
    def _callback_arm_commanded_joint_state(self, data):
        self.arm_commanded_joint_state = data
    
    def _callback_arm_ee_pose(self, data):
        self.arm_ee_pose = data

    def _callback_gripper_control(self, data):
        self.gripper_control = data

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
        if self.gripper_control is not None and self.with_gripper:
            state['gripper_joint_positions'] = self.gripper_control.data
        return state

    def extract(self, reset_timeout=10):
        states = []
        state_cnt = 0
        pbar = None

        def _on_press(key):
            nonlocal self
            try:
                if not self.start: 
                    if key.char == 's':
                        self.start = True
                if not self.stop:
                    if key.char == 't':
                        self.stop = True
                if key.char == 'c':
                    self.c_pressed = True
                if key.char == 'r':
                    self.r_pressed = True
                if key.char == 'q':
                    self.q_pressed = True    
                if key.char == 'e':
                    self.e_pressed = True     
                if key.char == "i":
                    self.i_pressed = True   
                if key.char == "p":
                    self.p_pressed = True            
            except AttributeError:
                pass
                # keyboard cmd

        def reset_robot():
            nonlocal states
            nonlocal state_cnt
            # reset the robot
            cprint(f"Resetting the robot...", 'blue')
            states = []
            state_cnt = 0

            # reset
            rospy.set_param("/data_collector/reset_robot", True)
            wait_reset_start = time.time()
            while True:
                if not rospy.get_param("/data_collector/reset_robot"):
                    rospy.set_param("/data_collector/stop_move", False)
                    break
                if time.time() - wait_reset_start > reset_timeout:
                    cprint(f"Reset failed after {reset_timeout} s. Please manually reset. Turn to the next demo by default.")
                    break

        def quit_program():
            # quit the program
            cprint(f"------------------------------------------", "green", attrs=['bold'])
            cprint(f'Finished recording! Data can be found in {self.storage_root}', 'green', attrs=['bold'])
            cprint(f"------------------------------------------", "green", attrs=['bold'])

            rospy.set_param("/data_collector/end_robot", True)
            sys.exit(0)

        def quit_program_silently():
            # quit the program
            cprint(f"------------------------------------------", "green", attrs=['bold'])
            cprint(f'Finished recording! Data can be found in {self.storage_root}', 'green', attrs=['bold'])
            cprint(f'Teleop process will not be killed.', 'green', attrs=['bold'])
            cprint(f"------------------------------------------", "green", attrs=['bold'])
            # quit the program
            rospy.set_param("/data_collector/end_robot", False)
            rospy.set_param("/data_collector/reset_robot", False)
            rospy.set_param("/data_collector/stop_move", False)
            sys.exit(0)

        def save_states():
            nonlocal states
            nonlocal self
            # Saving the pickle file save path
            os.makedirs(self.storage_path, exist_ok=True)
            for state_idx, state in enumerate(tqdm(states, desc=f'Saving demo {self.demo_num}...')):
                state_pickle_path = os.path.join(self.storage_path, f'{state_idx + 1}')
                store_pickle_data(state_pickle_path, state)
            self.demo_num += 1
            self.storage_path = os.path.join(self.storage_root, f'demonstration_{self.demo_num}')

        def wait_for_start():
            nonlocal pbar
            nonlocal self
            clear_input_buffer()
            cprint(f"---------------------------------------------", "yellow", attrs=['bold'])
            cprint(f"| s -> start recording, t -> stop recording |", "yellow", attrs=['bold'])
            cprint(f"---------------------------------------------", "yellow", attrs=['bold'])
            while True:
                if self.start:
                    cprint(f"Start recording demo {self.demo_num}.", "green", attrs=['bold'])
                    self.start = False
                    break
            pbar = tqdm(total=None)
        reset_robot()
        self.keyboard_listener = keyboard.Listener(on_press=_on_press)
        self.keyboard_listener.start()
        try:
            wait_for_start()
            while True:
                if not self.keyboard_listener.is_alive():
                    self.keyboard_listener = keyboard.Listener(on_press=_on_press)
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
                if self.num_cams > 0:
                    for cam_num in range(self.num_cams):
                        state['camera_{}_color_image'.format(cam_num + 1)] = self.color_image_subscribers[cam_num].get_image()
                        state['camera_{}_depth_image'.format(cam_num + 1)] = self.depth_image_subscribers[cam_num].get_image()
                    
                # tactile data
                if self.num_tactiles > 0:
                    tactile_data = {}
                    for tactile_num in range(self.num_tactiles):
                        raw_datas = np.array(self.tactile_subscribers[tactile_num].get_data()).reshape(self.sensor_per_board, POINT_PER_SENSOR, FORCE_DIM_PER_POINT)
                        for (tactile_id, raw_data) in enumerate(raw_datas):
                            tactile_data[self.tactile_info['id'][tactile_num + 1][tactile_id]] = raw_data
                    state['tactile_data'] = tactile_data

                # Temporal information
                state['time'] = rospy.Time.now().to_time()
                states.append(state)
                state_cnt += 1
                pbar.update(1)
                self.frequency_timer.sleep()

                #################################################### 
                if self.stop:
                    self.stop = False
                    pbar.close()

                    bool_true_msg = Bool()
                    bool_true_msg.data = True

                    bool_false_msg = Bool()
                    bool_false_msg.data = False

                    # stop robot move
                    rospy.set_param("/data_collector/stop_move", True)

                    clear_input_buffer()
                    cprint(f'>>> Collected demo {self.demo_num} with {state_cnt} timesteps.', 'green', attrs=['bold'])
                    cprint(f"--------------------------------------------", "yellow", attrs=['bold'])
                    cprint(f'| r -> restart, c -> continue              |', "yellow", attrs=['bold'])
                    cprint(f'| q -> save, dont kill teleop and quit     |', 'yellow', attrs=['bold'])
                    cprint(f'| e -> discard, dont kill teleop and quit  |', 'yellow', attrs=['bold'])
                    cprint(f'| i -> save, kill teleop and quit          |', 'yellow', attrs=['bold'])
                    cprint(f'| p -> discard, kill teleop and quit       |', 'yellow', attrs=['bold'])
                    cprint(f"--------------------------------------------", "yellow", attrs=['bold'])
                    while True:
                        if self.c_pressed:
                            self.c_pressed = False
                            save_states()
                            reset_robot()
                            wait_for_start()
                            break
                        
                        elif self.r_pressed:
                            self.r_pressed = False
                            reset_robot()
                            wait_for_start()
                            break

                        elif self.q_pressed:
                            # save, dont kill teleop and quit
                            self.q_pressed = False
                            save_states()
                            reset_robot()
                            quit_program_silently()  

                        elif self.e_pressed:
                            # discard, dont kill teleop and quit
                            self.e_pressed = False
                            reset_robot()
                            quit_program_silently()  

                        elif self.i_pressed:
                            # save, kill teleop and quit
                            self.i_pressed = False
                            save_states()
                            reset_robot()
                            quit_program()

                        elif self.p_pressed:
                            self.p_pressed = False
                            reset_robot()
                            quit_program()

        except KeyboardInterrupt:
            cprint('Finished recording! Data can be found in {}'.format(self.storage_path), 'green')
            sys.exit(0)