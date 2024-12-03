import flexivrdk
import rospy
import numpy as np
import time
import spdlog
from holodex.constants import *
from holodex.utils.network import JointStatePublisher, FloatArrayPublisher
from sensor_msgs.msg import JointState
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

        self.vel = 1.0
        self.dof = 7

        # Threading setup
        self.tcp_position = None
        self.move_queue = Queue()
        self.stop_event = threading.Event()

        # control period
        self.control_freq = 30
        self.control_period = 1 / self.control_freq

        self.cycle_times = deque(maxlen=100)
        self._is_moving = False

        self._move_start_time = None
        self._target_position = None

        # Publishers
        self.joint_state_publisher = JointStatePublisher(
            publisher_name=JAKA_JOINT_STATE_TOPIC
        )
        self.command_joint_state_publisher = JointStatePublisher(
            publisher_name=JAKA_COMMANDED_JOINT_STATE_TOPIC
        )
        self.ee_pose_publisher = FloatArrayPublisher(publisher_name=JAKA_EE_POSE_TOPIC)

        self.initialize_connection()

        # Start threads
        self.tcp_reader_thread = threading.Thread(target=self.read_tcp_position)
        self.movement_thread = threading.Thread(target=self.movement_handler)

        self.tcp_reader_thread.daemon = True
        self.movement_thread.daemon = True

        self.tcp_reader_thread.start()
        self.movement_thread.start()

    def initialize_connection(self):
        try:
            self.flexiv = flexivrdk.Robot(self.robot_sn)

            if self.flexiv.fault():
                self.logger.warn("Fault occurred on the connected robot")
                if not self.flexiv.ClearFault():
                    return 1
                self.logger.info("Fault cleared")

            self.flexiv.Enable()
            self.flexiv.SwitchMode(self.mode.NRT_CARTESIAN_MOTION_FORCE)

            while not self.flexiv.operational():
                time.sleep(1)

            self.tcp_position = self.flexiv.states().tcp_pose

        except Exception as e:
            self.logger.error(f"Failed to connect to Flexiv robot: {e}")

    def home_robot(self):
        self.flexiv.SwitchMode(self.mode.NRT_PRIMITIVE_EXECUTION)

        flexiv_home_js = FLEXIV_POSITIONS["home_js"]
        # flexiv_home_js = [j * 180 for j in flexiv_home_js]

        self.flexiv.ExecutePrimitive(
            f"MoveJ(target={' '.join(map(str, flexiv_home_js))}, jntVelScale=10)"
        )

        while (
            self._parse_pt_states(self.flexiv.primitive_states(), "reachedTarget")
            != "1"
        ):
            time.sleep(1)

        self.flexiv.SwitchMode(self.mode.NRT_CARTESIAN_MOTION_FORCE)

    def precise_sleep(self, duration):
        start = time.perf_counter()
        while time.perf_counter() - start < duration:
            pass

    def read_tcp_position(self):
        while not self.stop_event.is_set():
            try:
                t_start = time.monotonic()

                # read TCP position
                self.tcp_position = self.flexiv.states().tcp_pose
                self.publish_state()

                # control period
                elapsed = time.time() - t_start
                if elapsed < self.control_period:
                    time.sleep(self.control_period - elapsed)

                # get actual cycle time
                actual_cycle_time = time.time() - t_start
                self.cycle_times.append(actual_cycle_time)

            except Exception as e:
                self.logger.error(f"TCP position reading error: {e}")
                time.sleep(0.1)

    # def movement_handler(self):
    #     """Thread function for handling movement commands"""
    #     while not self.stop_event.is_set():
    #         try:
    #             if not self.move_queue.empty():
    #                 target_pose = self.move_queue.get()
    #                 self.flexiv.SendCartesianMotionForce(target_pose, [0]*6, self.vel)
    #                 time.sleep(1/200)
    #         except Exception as e:
    #             self.logger.error(f"Movement execution error: {e}")
    #             time.sleep(0.1)

    # def movement_handler(self):
    #     """优化为30Hz的运动处理"""
    #     while not self.stop_event.is_set():
    #         try:
    #             start_time = time.time()

    #             if not self.move_queue.empty() and not self._is_moving:
    #                 target_pose = self.move_queue.get()
    #                 if self.tcp_position is not None:

    #                     # 计算运动距离和时间
    #                     distance = np.linalg.norm(np.array(target_pose[:3]) - np.array(self.tcp_position[:3]))
    #                     move_time = max(distance / self.vel)  # 至少0.5秒

    #                     self._is_moving = True
    #                     self._move_start_time = time.time()
    #                     self._target_position = target_pose
    #                     self.flexiv.SendCartesianMotionForce(target_pose, [0]*6)

    #             # 检查运动是否完成
    #             if self._is_moving and self._check_motion_complete():
    #                 self._is_moving = False
    #                 self._target_position = None

    #             # 控制周期
    #             elapsed = time.time() - start_time
    #             if elapsed < self.control_period:
    #                 time.sleep(self.control_period - elapsed)

    #         except Exception as e:
    #             self._is_moving = False
    #             self.logger.error(f"Movement execution error: {e}")
    #             time.sleep(0.1)

    # def _check_motion_complete(self, tolerance=0.001):
    #     """检查运动是否完成"""
    #     if self.tcp_position is None or self._target_position is None:
    #         return False

    #     position_reached = np.allclose(
    #         self.tcp_position[:3],
    #         self._target_position[:3],
    #         atol=tolerance
    #     )

    #     timeout_reached = False
    #     if self._move_start_time is not None:
    #         current_time = time.time()
    #         move_duration = current_time - self._move_start_time
    #         distance = np.linalg.norm(
    #             np.array(self._target_position[:3]) -
    #             np.array(self.tcp_position[:3])
    #         )
    #         expected_time = max(distance / self.vel, 0.5)
    #         timeout_reached = move_duration > (expected_time + 1.0)  # 1秒余量

    #     return position_reached or timeout_reached

    def movement_handler(self):
        while not self.stop_event.is_set():
            try:
                start_time = time.perf_counter()

                if not self.move_queue.empty() and not self._is_moving:
                    target_pose = self.move_queue.get()
                    if self.tcp_position is not None:
                        self._is_moving = True
                        self._move_start_time = time.perf_counter()
                        self._target_position = list(target_pose)

                        # 确保使用正确的数据类型
                        force_torque = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

                        # 记录日志
                        self.logger.info(f"Moving to position: {target_pose[:3]}")
                        self.logger.info(f"Current position: {self.tcp_position[:3]}")

                        # 发送运动命令
                        try:
                            self.flexiv.SendCartesianMotionForce(
                                list(target_pose), force_torque
                            )
                        except Exception as e:
                            self.logger.error(f"SendCartesianMotionForce failed: {e}")
                            self._is_moving = False

                # 检查运动是否完成
                if self._is_moving:
                    if self._check_motion_complete():
                        self._is_moving = False
                        self._target_position = None
                        self.logger.info("Motion completed")
                    elif (
                        time.perf_counter() - self._move_start_time > self.max_wait_time
                    ):
                        self._is_moving = False
                        self._target_position = None
                        self.logger.warn("Motion timeout")

                # 精确控制循环时间
                elapsed = time.perf_counter() - start_time
                if elapsed < self.control_period:
                    time.sleep(self.control_period - elapsed)

            except Exception as e:
                self._is_moving = False
                self.logger.error(f"Movement execution error: {e}")
                time.sleep(0.1)

    def _check_motion_complete(self, tolerance=None):
        if tolerance is None:
            tolerance = self.position_tolerance

        if self.tcp_position is None or self._target_position is None:
            return False

        current_pos = np.array(self.tcp_position[:3])
        target_pos = np.array(self._target_position[:3])

        distance = np.linalg.norm(current_pos - target_pos)
        is_reached = distance <= tolerance

        if is_reached:
            self.logger.info(f"Position reached. Distance: {distance:.6f}")

        return is_reached

    def move_to_position(self, target_pose):
        """Queue a movement command"""
        self.logger.info(f"Queueing movement to: {target_pose[:3]}")
        self.move_queue.put(list(target_pose))
        self.publish_state(target_pose)

    def monitor_performance(self):
        """监控控制性能"""
        if len(self.cycle_times) > 0:
            mean_freq = 1.0 / np.mean(self.cycle_times)
            std_freq = np.std([1.0 / t for t in self.cycle_times])
            max_freq = 1.0 / min(self.cycle_times)
            min_freq = 1.0 / max(self.cycle_times)

            self.logger.info(
                f"""
                Performance Stats:
                    Mean Frequency: {mean_freq:.2f} Hz
                    Std Frequency: {std_freq:.2f} Hz
                    Min Frequency: {min_freq:.2f} Hz
                    Max Frequency: {max_freq:.2f} Hz
                """
            )

    # def move_to_position(self, target_arm_pose):
    #     """Queue a movement command"""
    #     self.move_queue.put(target_arm_pose)
    #     self.publish_state(target_arm_pose)

    def publish_state(self, input_cmd=None):
        if self.tcp_position:
            self.ee_pose_publisher.publish(self.tcp_position)
        if input_cmd is not None:
            self.command_joint_state_publisher.publish(input_cmd)

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

    def cleanup(self):
        """Cleanup function to properly stop all threads"""
        self.stop_event.set()
        self.tcp_reader_thread.join()
        self.movement_thread.join()


if __name__ == "__main__":
    rospy.init_node("flexiv_arm_controller")
    controller = FlexivArm()
    controller.home_robot()
    # try:
    #     initial_pose = controller.tcp_position

    #     for i in range(10):
    #         new_pose = initial_pose.copy()
    #         new_pose[2] += 0.1 * i / 10
    #         print('============')
    #         cprint(f"Target position: {new_pose[:3]}", 'red')
    #         controller.move_to_position(new_pose)
    #         while controller._is_moving:
    #             print(f"Current position: {controller.tcp_position[:3]}")
    #             time.sleep(0.1)
    #         print('============')
    #         cprint(f"Current position: {controller.tcp_position[:3]}", 'green')

    # except KeyboardInterrupt:
    #     print("Shutting down...")
    # finally:
    #     controller.cleanup()
    #     print("Process terminated.")

    try:
        # 等待初始化完成
        time.sleep(1.0)

        # 记录初始位置
        initial_pose = controller.tcp_position
        if initial_pose is None:
            raise RuntimeError("Failed to get initial pose")

        # 设置目标位置
        target_pose = initial_pose.copy()
        target_pose[2] += 0.1  # 向上移动10cm

        # 执行运动
        print("Starting movement...")
        controller.move_queue.put(target_pose)

        # 等待运动完成
        while controller._is_moving:
            print(f"Current position: {controller.tcp_position[:3]}")
            time.sleep(0.1)

        print("Movement completed!")

        # 显示性能统计
        controller.monitor_performance()

    except Exception as e:
        print(f"Error occurred: {e}")

    finally:
        controller.stop_event.set()
        time.sleep(0.5)
