import os
import sys
import rospy
import time
import shutil
import termios
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from holodex.utils.files import store_pickle_data, make_dir
from holodex.constants import (
    RECORD_FPS,
    KEYBOARD_EE_TOPIC,
    KEYBOARD_HAND_TOPIC,
)
from holodex.utils.network import (
    ImageSubscriber,
    frequency_timer,
    Float64MultiArray,
)
from holodex.robot.arm.franka.franka_env_wrapper import FrankaEnvWrapper
from franka_interface_msgs.msg import RobotState
from termcolor import cprint
from pynput import keyboard


def clear_input_buffer():
    termios.tcflush(sys.stdin, termios.TCIFLUSH)


class FrankaDataCollector:
    def __init__(
        self,
        num_cams=0,
        keyboard_control=False,
        storage_path=None,
        data_collection_type=None,
    ):
        """
        Initialize the data collector for Franka robot.

        Args:
            num_cams (int): Number of cameras
            keyboard_control (bool): Whether to use keyboard control
            storage_path (str): Path to store collected data
            data_collection_type (list): Types of data to collect
        """
        self.storage_path = storage_path
        self.keyboard_control = keyboard_control
        self.num_cams = num_cams
        self.storage_root = storage_path

        # For auto collection
        self.stop = False
        self.demo_num = 1
        if storage_path:
            self.storage_path = os.path.join(
                self.storage_root, f"demonstration_{self.demo_num}"
            )
            make_dir(self.storage_path)

        # Initialize Franka environment wrapper
        self.franka_env_wrapper = FrankaEnvWrapper()

        # Set up cameras
        # self._setup_cameras()

        # Set up ROS topics
        self._setup_ros_topics()

        # Set up data collection based on type
        if data_collection_type:
            self._setup_data_collection(data_collection_type)
        else:
            self._setup_franka_state_collection()

        # Frequency timer for data collection
        self.frequency_timer = frequency_timer(RECORD_FPS)

        # Initialize state variables
        self.robot_state = None

        # For keyboard control if enabled
        if self.keyboard_control:
            self._setup_keyboard_control()

        # Set up keyboard listener for auto collection commands
        self.keyboard_listener = keyboard.Listener(on_press=self._on_press)
        self.keyboard_listener.start()

        # Set up publishers for robot control
        self._setup_robot_control_publishers()

    def _setup_robot_control_publishers(self):
        """Set up publishers for robot control"""
        self.reset_publisher = rospy.Publisher(
            "/data_collector/reset_robot", Bool, queue_size=1
        )
        self.stop_publisher = rospy.Publisher(
            "/data_collector/stop_move", Bool, queue_size=1
        )
        self.hamer_recalib_publisher = rospy.Publisher(
            "/data_collector/reset_done", Bool, queue_size=1
        )
        self.end_publisher = rospy.Publisher(
            "/data_collector/end_robot", Bool, queue_size=1
        )

    def _on_press(self, key):
        """Callback for keyboard press events"""
        try:
            if not self.stop and hasattr(key, "char") and key.char == "s":
                print(f"Key pressed: {key.char}")
                self.stop = True
        except AttributeError:
            pass

    def _setup_cameras(self):
        """Set up camera subscribers"""
        self.color_image_subscribers, self.depth_image_subscribers = [], []
        for cam_num in range(self.num_cams):
            self.color_image_subscribers.append(
                ImageSubscriber(
                    subscriber_name=f"/robot_camera_{cam_num + 1}/color_image",
                    node_name=f"robot_camera_{cam_num + 1}_color_data_collector",
                )
            )
            self.depth_image_subscribers.append(
                ImageSubscriber(
                    subscriber_name=f"/robot_camera_{cam_num + 1}/depth_image",
                    node_name=f"robot_camera_{cam_num + 1}_depth_data_collector",
                    color=False,
                )
            )

    def _setup_ros_topics(self):
        """Set up ROS topic names"""
        self.franka_state_topic = "/robot_state_publisher_node_1/robot_state"

    def _setup_keyboard_control(self):
        """Set up keyboard control subscribers"""
        self.arm_ee_pose = None
        self.hand_commanded_joint_position = None
        rospy.Subscriber(
            KEYBOARD_EE_TOPIC,
            Float64MultiArray,
            self._callback_keyboard_control_ee,
            queue_size=1,
        )
        rospy.Subscriber(
            KEYBOARD_HAND_TOPIC,
            Float64MultiArray,
            self._callback_keyboard_control_hand,
            queue_size=1,
        )

    def _callback_keyboard_control_ee(self, data):
        """Callback for keyboard EE control"""
        self.arm_ee_pose = data

    def _callback_keyboard_control_hand(self, data):
        """Callback for keyboard hand control"""
        self.hand_commanded_joint_position = data

    def _setup_data_collection(self, data_collection_type):
        """
        Set up data collection based on specified types

        Args:
            data_collection_type (list): List of data types to collect
        """
        if "robot_state" in data_collection_type:
            self._setup_franka_state_collection()

    def _setup_franka_state_collection(self):
        """Set up franka state collection"""
        rospy.Subscriber(
            self.franka_state_topic,
            RobotState,
            self._callback_robot_state,
            queue_size=1,
        )

    def _callback_robot_state(self, data):
        """Callback for robot state"""
        self.robot_state = data

    def _check_data_availability(self):
        """
        Check if all required data streams are available

        Returns:
            bool: True if all data is available, False otherwise
        """
        # Check gripper data
        if self.franka_env_wrapper.get_gripper_width() is None:
            cprint("Gripper width data not available!", "red")
            return False

        if self.franka_env_wrapper.get_gripper_is_grasped() is None:
            cprint("Gripper grasp status not available!", "red")
            return False

        # Check robot state data if needed
        if hasattr(self, "robot_state") and self.robot_state is None:
            cprint("Robot state data not available!", "red")
            return False

        # Check camera data
        for cam_num in range(self.num_cams):
            if self.color_image_subscribers[cam_num].get_image() is None:
                cprint(f"Camera {cam_num+1} color image not available!", "red")
                return False
            if self.depth_image_subscribers[cam_num].get_image() is None:
                cprint(f"Camera {cam_num+1} depth image not available!", "red")
                return False

        return True

    def _collect_state_data(self):
        """
        Collect all state data into a dictionary

        Returns:
            dict: Dictionary containing all collected state data
        """
        state = {}

        # Gripper data
        state["gripper_joint_positions"] = self.franka_env_wrapper.get_gripper_width()
        state["gripper_status"] = self.franka_env_wrapper.get_gripper_is_grasped()

        # Full robot state if available
        if getattr(self, "robot_state", None) is not None:
            # Add arm data directly to state for compatibility with other collectors
            state["arm_joint_positions"] = self.robot_state.q
            state["arm_ee_pose"] = self.robot_state.O_T_EE
            state["arm_commanded_joint_position"] = self.robot_state.q_d
            state["arm_commanded_ee_pose"] = self.robot_state.O_T_EE_d

        # Keyboard control data if enabled
        if self.keyboard_control:
            if getattr(self, "arm_ee_pose", None) is not None:
                state["arm_ee_pose"] = self.arm_ee_pose.data
            if getattr(self, "hand_commanded_joint_position", None) is not None:
                state["hand_commanded_joint_position"] = (
                    self.hand_commanded_joint_position.data
                )

        # Image data
        for cam_num in range(self.num_cams):
            state[f"camera_{cam_num + 1}_color_image"] = self.color_image_subscribers[
                cam_num
            ].get_image()
            state[f"camera_{cam_num + 1}_depth_image"] = self.depth_image_subscribers[
                cam_num
            ].get_image()

        # Temporal information
        state["time"] = time.time()

        return state

    def _handle_user_command(self, counter):
        """
        Handle user commands during auto collection

        Args:
            counter (int): Current counter value

        Returns:
            int: Updated counter value
            bool: Whether to continue collection
        """
        bool_true_msg = Bool()
        bool_true_msg.data = True

        bool_false_msg = Bool()
        bool_false_msg.data = False

        # Stop robot movement
        self.stop_publisher.publish(bool_true_msg)

        # Wait for next command
        while True:
            cprint("Waiting for the next command: ", "yellow")
            cprint(
                "c -> continue, d -> delete, r -> reset, q -> quit, x -> stop robot",
                "yellow",
            )
            clear_input_buffer()
            input_cmd = input("Enter the next command: ")

            if input_cmd == "c":
                # Update demo number and continue recording
                counter = 1
                self.demo_num += 1
                self.storage_path = os.path.join(
                    self.storage_root, f"demonstration_{self.demo_num}"
                )
                make_dir(self.storage_path)
                self.hamer_recalib_publisher.publish(bool_true_msg)
                cprint(f"Start recording at {self.storage_path}", "green")

                # Reset robot
                self.reset_publisher.publish(bool_true_msg)
                rospy.sleep(1)

                # Start robot movement
                self.stop_publisher.publish(bool_false_msg)
                return counter, True

            elif input_cmd == "d":
                # Remove the last data
                shutil.rmtree(self.storage_path)
                cprint(f"Removing the last data at {self.storage_path}", "red")
                self.demo_num -= 1
                continue

            elif input_cmd == "r":
                # Reset the robot
                cprint(f"Resetting the robot...", "blue")
                self.reset_publisher.publish(bool_true_msg)
                rospy.sleep(1)
                continue

            elif input_cmd == "q":
                # Quit the program
                cprint(
                    f"Finished recording! Successfully recorded {self.demo_num} demonstrations! Data can be found in {self.storage_root}",
                    "green",
                )
                self.end_publisher.publish(bool_true_msg)
                sys.exit(0)

            elif input_cmd == "x":
                # Stop the robot movement
                cprint(f"Stop the robot movement...", "red")
                self.stop_publisher.publish(bool_true_msg)
                continue

            else:
                cprint(f"Invalid command {input_cmd}!", "red")

    def extract(self, offset=0):
        """
        Extract and save data from all sources

        Args:
            offset (int): Counter offset for file naming
        """
        counter = offset + 1
        try:
            while True:
                # Check if all data is available
                if not self._check_data_availability():
                    time.sleep(0.1)  # Short sleep to avoid CPU spinning
                    continue

                # Log valid data
                cprint(
                    f"Valid Data at {time.time()}", "green", "on_black", attrs=["bold"]
                )

                # Collect all state data
                state = self._collect_state_data()

                # Save data to file
                state_pickle_path = os.path.join(self.storage_path, f"{counter}")
                store_pickle_data(state_pickle_path, state)

                counter += 1

                # Reset keyboard control data if needed
                if self.keyboard_control:
                    self.arm_ee_pose = None
                    self.hand_commanded_joint_position = None

                # Sleep to maintain collection frequency
                self.frequency_timer.sleep()

                # Handle auto collection stop
                if self.stop:
                    cprint(
                        f"Successfully recorded demonstration {self.demo_num}! Data can be found in {self.storage_path}",
                        "green",
                    )
                    self.stop = False

                    # Handle user commands
                    counter, continue_collection = self._handle_user_command(counter)
                    if not continue_collection:
                        break

        except KeyboardInterrupt:
            cprint(
                f"Finished recording! Data can be found in {self.storage_path}",
                "green",
            )
            sys.exit(0)


if __name__ == "__main__":
    import os
    import time
    from holodex.utils.files import make_dir

    # Define storage path
    storage_path = "expert_dataset/test_data/franka_collector_test"

    # Create directory if it doesn't exist
    make_dir(storage_path)

    # Create a data collector instance for testing
    collector = FrankaDataCollector(
        num_cams=1,
        keyboard_control=False,
        data_collection_type=["robot_state"],
        storage_path=storage_path,
    )

    # Print initialization message
    print(
        f"FrankaDataCollector test initialized. Data will be saved to: {storage_path}"
    )

    try:
        # Start data collection
        print("Starting data collection test...")

        # Collect data for 10 seconds
        start_time = time.time()
        duration = 10  # seconds

        collector.extract(offset=0)

        while time.time() - start_time < duration:
            print(
                f"Collecting data... {int(time.time() - start_time)}/{duration} seconds"
            )
            time.sleep(1)

        # Check if data was collected
        import glob

        collected_files = glob.glob(f"{storage_path}/*")
        print(f"Test results: Collected {len(collected_files)} data points")

        if len(collected_files) > 0:
            print("Test PASSED: Data collection successful")
        else:
            print("Test FAILED: No data collected")

    except KeyboardInterrupt:
        print("Test terminated by user")
    except Exception as e:
        print(f"Test FAILED with error: {e}")
    finally:
        print("Test completed")
