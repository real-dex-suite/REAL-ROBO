import flexivrdk
import rospy
import numpy as np
import time
import spdlog
from holodex.constants import *
from holodex.utils.network import JointStatePublisher, FloatArrayPublisher
from sensor_msgs.msg import JointState
import time
from scipy.spatial.transform import Rotation as R
import threading
from termcolor import cprint
from queue import Queue
from collections import deque


class FlexivArm(object):
    def __init__(self):
        self.robot_sn = "Rizon4-062521"
        self.logger = spdlog.ConsoleLogger("RobotController")
        self.mode = flexivrdk.Mode
        print(self.mode)

        self.initialize_connection()
        self.vel = 1.0
        self.dof = 7

        # Control parameters
        self.control_freq = 30
        self.control_period = 1 / self.control_freq

        # Publish msg
        self.joint_state_publisher = JointStatePublisher(
            publisher_name=JAKA_JOINT_STATE_TOPIC
        )
        self.command_joint_state_publisher = JointStatePublisher(
            publisher_name=JAKA_COMMANDED_JOINT_STATE_TOPIC
        )
        self.ee_pose_publisher = FloatArrayPublisher(publisher_name=JAKA_EE_POSE_TOPIC)

        # Threading setup
        self.tcp_position = None
        self.move_queue = Queue()
        self.stop_event = threading.Event()

        # Start threads
        self.tcp_reader_thread = threading.Thread(target=self.read_tcp_position)
        self.movement_thread = threading.Thread(target=self.movement_handler)
        self.tcp_reader_thread.daemon = True
        self.movement_thread.daemon = True
        self.tcp_reader_thread.start()
        self.movement_thread.start()

    def initialize_connection(self):
        # Initialize connection to the Flexiv robot
        try:
            self.flexiv = flexivrdk.Robot(self.robot_sn)
            if self.flexiv.fault():
                self.logger.warn("Fault occured the connected robot")

                if not self.flexiv.ClearFault():
                    return 1
                self.logger.info("cleared")

            self.flexiv.Enable()

            self.flexiv.SwitchMode(self.mode.NRT_CARTESIAN_MOTION_FORCE)

            while not self.flexiv.operational():
                time.sleep(1)

        except Exception as e:
            self.logger.error(f"Failed to connect to Flexiv robot: {e}")

    def _parse_pt_states(self, pt_states, parse_target):
        """
        Parse the value of a specified primitive state from the pt_states string list.

        Parameters
        ----------
        pt_states : str list
            Primitive states string list returned from Robot::primitive_states().
        parse_target : str
            Name of the primitive state to parse for.

        Returns
        ----------
        str
            Value of the specified primitive state in string format. Empty string is
            returned if parse_target does not exist.
        """
        for state in pt_states:
            # Split the state sentence into words
            words = state.split()

            if words[0] == parse_target:
                return words[-1]

        return ""

    def home_robot(self):
        self.flexiv.SwitchMode(self.mode.NRT_PRIMITIVE_EXECUTION)

        flexiv_home_js = FLEXIV_POSITIONS["home_js"]
        self.flexiv.ExecutePrimitive(
            f"MoveJ(target={' '.join(map(str, flexiv_home_js))}, jntVelScale=10)"
        )

        while (
            self._parse_pt_states(self.flexiv.primitive_states(), "reachedTarget")
            != "1"
        ):
            time.sleep(1)

        self.flexiv.SwitchMode(self.mode.NRT_CARTESIAN_MOTION_FORCE)

    def get_tcp_position(self, euler=False, degree=True):
        # p_x, p_y, p_z, wx, r_x, r_y, r_z
        try:
            tcp_position = self.flexiv.states().tcp_pose

            # print(f"now tcp quat: {tcp_position}")

            if euler:
                rot = self.quat2eulerZYX(tcp_position[3:], degree)
                trans = tcp_position[:3]
                return trans + rot

            return tcp_position
        except Exception as e:
            self.logger.error(f"Failed to get TCP position: {e}")
            return None

    def read_tcp_position(self):
        while not self.stop_event.is_set():
            try:
                self.tcp_position = self.flexiv.states().tcp_pose
                self.publish_state()

            except Exception as e:
                self.logger.error(f"TCP position reading error: {e}")
                time.sleep(0.1)

    def movement_handler(self):
        while not self.stop_event.is_set():
            try:

                if not self.move_queue.empty():
                    target_pose = self.move_queue.get()
                    if self.tcp_position is not None:
                        # send command
                        try:
                            self.flexiv.SendCartesianMotionForce(
                                list(target_pose), [0] * 6, 0.5, 1.0, 0.4
                            )
                            time.sleep(1 / 30)
                        except Exception as e:
                            self.logger.error(f"SendCartesianMotionForce failed: {e}")
                        

            except Exception as e:
                self.logger.error(f"Movement execution error: {e}")
                time.sleep(0.1)

    def move_to_position(self, target_pose):
        """Queue a movement command"""
        self.move_queue.put(list(target_pose))
        self.publish_state(target_pose)

    def move(self, target_arm_pose):
        # input 7
        try:
            # self.flexiv.ExecutePrimitive(
            #     f"MovePTP(target={target_arm_pose[0]} {target_arm_pose[1]} {target_arm_pose[2]} {target_arm_pose[3]} {target_arm_pose[4]} {target_arm_pose[5]} WORLD WORLD_ORIGIN)" #vel={self.vel})"
            # )
            # self.flexiv.SendCartesianMotionForce(target_arm_pose, [0]*6, 0.12, 1.0) # space mouse
            self.publish_state(target_arm_pose)  # TODO maybe change to use ros
            self.flexiv.SendCartesianMotionForce(
                target_arm_pose, [0] * 6, 0.5, 1.0, 0.4
            )
            tcp_position = self.get_tcp_position()
            # cprint(f"tcp_position: {tcp_position}", "yellow")
            time.sleep(1 / 30)

        except Exception as e:
            self.logger.error(f"Failed to move the robot: {e}")

    def move_joint(self, target_joint):
        pass


    def quat2eulerZYX(self, quat, degree=False):
        eulerZYX = (
            R.from_quat([quat[1], quat[2], quat[3], quat[0]])
            .as_euler("xyz", degrees=degree)
            .tolist()
        )
        return eulerZYX

    def eulerZYX2quat(self, euler, degree=False):
        if degree:
            euler = np.radians(euler)

        tmp_quat = R.from_euler("xyz", euler).as_quat().tolist()
        quat = [tmp_quat[3], tmp_quat[0], tmp_quat[1], tmp_quat[2]]
        return quat

    def publish_state(self, input_cmd=None):
        # TODO 3: read joint and read
        # current_joint = self.get_arm_position()
        # self.joint_state_publisher.publish(current_joint)

        try:
            current_ee_pose = self.flexiv.states().tcp_pose
        except Exception as e:
            self.logger.error(f"Failed to get TCP position: {e}")
            return None

        self.ee_pose_publisher.publish(current_ee_pose)

        # TODO 1: publish joint states
        # TODO 2: publish ee pose
        if input_cmd is not None:
            self.command_joint_state_publisher.publish(input_cmd)

    def cleanup(self):
        """Cleanup function to properly stop all threads"""
        self.stop_event.set()
        self.tcp_reader_thread.join()
        self.movement_thread.join()


# Example usage
if __name__ == "__main__":
    rospy.init_node("flexiv_arm_controller")
    controller = FlexivArm()
    controller.home_robot()

    # tcp_pose = controller.get_tcp_position()

    # for i in range(10):
    #     tcp_pose_new = list(tcp_pose)
    #     tcp_pose_new = tcp_pose_new.copy()
    #     tcp_pose_new[2] += 0.1 * i / 10
    #     print('============')
    #     cprint(tcp_pose_new[:3], 'red')
    #     controller.move_to_position(tcp_pose_new)
    #     time.sleep(1/30)
    #     # new_tcp_pose = controller.get_tcp_position()
    #     new_tcp_pose = controller.tcp_position
    #     cprint(new_tcp_pose[:3], 'green')

    try:
        time.sleep(1.0)

        initial_pose = controller.tcp_position
        if initial_pose is None:
            raise RuntimeError("Failed to get initial pose")

        print(f"Initial pose: {initial_pose[:3]}")

        test_positions = []
        for i in range(10):  #!!!不要调太多！！！
            new_pose = list(initial_pose)
            new_pose[2] += 0.01 * (i + 1)  # !!!不要调太多！！！
            test_positions.append(new_pose)

        # for target_pose in test_positions:
        #     cprint(f"\nMoving to: {target_pose[:3]}", "red")
        #     controller.move_to_position(target_pose)
        #     wait_start = time.monotonic()
        #     current_pos = (
        #         controller.tcp_position[:3] if controller.tcp_position else None
        #     )
        #     time.sleep(1 / 30)
        #     cprint(f"Final position: {controller.tcp_position[:3]}", "green")

        start_time = time.monotonic()
        iterations = 0
        positions = []

        for target_pose in test_positions:
            controller.logger.warn(f"\nMoving: {target_pose[2]:.5f}")
            iterations += 1
            controller.move_to_position(target_pose)
            time.sleep(1/10) # 5和10 基本align # TODO: add time check and align

            positions.append(controller.tcp_position[:3])
            controller.logger.info(f"Final: {controller.tcp_position[2]:.5f}")
   
        end_time = time.monotonic()
        total_time = end_time - start_time
        actual_frequency = iterations / total_time
        print(f"Total time: {total_time:.4f} seconds")
        print(f"Number of iterations: {iterations}")
        print(f"Actual control frequency: {actual_frequency:.2f} Hz")

        # get precise final position
        time.sleep(1)
        controller.logger.info(f"Real final position: {controller.tcp_position[2]:.5f}")

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        controller.cleanup()
        print("Controller shutdown completed")
