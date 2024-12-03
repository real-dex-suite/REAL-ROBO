"""
This file starts a control server running on the real time PC connected to the franka robot.
Reference: https://rdk.flexiv.com/api/classflexiv_1_1rdk_1_1_robot.html#ad65e38f8694b28d0704d29806f5d6d57
"""

from flask import Flask, request, jsonify
import numpy as np
import time
import subprocess
from scipy.spatial.transform import Rotation as R
from absl import app, flags
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib_py"))
import flexivrdk
import time

# from utils import quat2eulerZYX
# from utils import parse_pt_states
# from utils import list2str
import spdlog  # pip install spdlog
from ipdb import set_trace


def quat2eulerZYX(quat, degree=False):
    """
    Convert quaternion to Euler angles with ZYX axis rotations.

    Parameters
    ----------
    quat : float list
        Quaternion input in [w,x,y,z] order.
    degree : bool
        Return values in degrees, otherwise in radians.

    Returns
    ----------
    float list
        Euler angles in [x,y,z] order, radian by default unless specified otherwise.
    """

    # Convert target quaternion to Euler ZYX using scipy package's 'xyz' extrinsic rotation
    # NOTE: scipy uses [x,y,z,w] order to represent quaternion
    eulerZYX = (
        R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        .as_euler("xyz", degrees=degree)
        .tolist()
    )

    return eulerZYX


def list2str(ls):
    """
    Convert a list to a string.

    Parameters
    ----------
    ls : list
        Source list of any size.

    Returns
    ----------
    str
        A string with format "ls[0] ls[1] ... ls[n] ", i.e. each value
        followed by a space, including the last one.
    """

    ret_str = ""
    for i in ls:
        ret_str += str(i) + " "
    return ret_str


def parse_pt_states(pt_states, parse_target):
    """
    Parse the value of a specified primitive state from the pt_states string list.

    Parameters
    ----------
    pt_states : str list
        Primitive states string list returned from Robot::getPrimitiveStates().
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


def print_green(text: str) -> None:
    GREEN = "\033[92m"  # ANSI 转义码，用于设置绿色文本
    RESET = "\033[0m"  # ANSI 转义码，用于重置颜色
    print(f"{GREEN}{text}{RESET}")


# FLAGS = flags.FLAGS
# flags.DEFINE_string(
#     "robot_ip", "172.16.0.2", "IP address of the franka robot's controller box"
# )
# flags.DEFINE_string(
#     "gripper_ip", "192.168.1.114", "IP address of the robotiq gripper if being used"
# )
# flags.DEFINE_string(
#     "gripper_type", "Robotiq", "Type of gripper to use: Robotiq, Franka, or None"
# )
# flags.DEFINE_list(
#     "reset_joint_target",
#     [0, 0, 0, -1.9, -0, 2, 0],
#     "Target joint angles for the robot to reset to",
# )


# Maximum contact wrench [fx, fy, fz, mx, my, mz] [N][Nm]
# reset_qpos = [0.6837236285209656, -0.11226988583803177, 0.2890230119228363, 0.0009086112258955836, -0.001942529808729887, 0.9999863505363464, -0.004769640974700451] # homing tcp pose
# reset_euler = quat2eulerZYX(reset_qpos[3:], degree=True)
# reset_qpos_euler = reset_qpos[:3] + reset_euler
RESET_JOINT_TARGET = [30, -45, 0, 90, 0, 40, 30]
MAX_CONTACT_WRENCH = [50.0, 50.0, 50.0, 15.0, 15.0, 15.0]


class FlexivServer:
    """Handles the starting and stopping of the impedance controller
    (as well as backup) joint recovery policy."""

    def __init__(self, robot_sn="Rizon4-062521"):  # initialize
        """activate the robot"""
        # Define alias
        self.logger = spdlog.ConsoleLogger("Example")
        self.mode = flexivrdk.Mode

        self.robot = flexivrdk.Robot(robot_sn)

        # Clear fault on the connected robot if any
        if self.robot.fault():
            self.logger.warn(
                "Fault occurred on the connected robot, trying to clear ..."
            )
            # Try to clear the fault
            if not self.robot.ClearFault():
                self.logger.error("Fault cannot be cleared, exiting ...")
                return 1
            self.logger.info("Fault on the connected robot is cleared")

        # Enable the robot, make sure the E-stop is released before enabling
        self.logger.info("Enabling robot ...")
        self.robot.Enable()

        # Wait for the robot to become operational
        while not self.robot.operational():
            time.sleep(1)

        self.logger.info("Robot is now operational")

        # Move robot to home pose
        self.logger.info("Moving to home pose")
        self.robot.SwitchMode(
            self.mode.NRT_PRIMITIVE_EXECUTION
        )  # default mode of the robot
        self.robot.ExecutePrimitive("Home()")
        # Wait for the primitive to finish
        while self.robot.busy():
            time.sleep(1)

        # IMPORTANT: must zero force/torque sensor offset for accurate force/torque measurement
        self.robot.ExecutePrimitive("ZeroFTSensor()")

        # to compute the jacobian
        self.model = flexivrdk.Model(self.robot)

    def clear(self):
        if self.robot.fault():
            print("detecting fault... clearing...")
            self.robot.ClearFault()
            time.sleep(2)
        if self.robot.fault():
            print("CANNOT CLEAR FAULT")
            return

        # self.robot.SwitchMode(self.mode.NRT_PRIMITIVE_EXECUTION)
        # self.robot.ExecutePrimitive("Home()")
        # # IMPORTANT: must zero force/torque sensor offset for accurate force/torque measurement
        # self.robot.ExecutePrimitive("ZeroFTSensor()")

        # self.start_impedance()

    def clear_force(self):
        self.robot.SwitchMode(self.mode.NRT_PRIMITIVE_EXECUTION)
        # IMPORTANT: must zero force/torque sensor offset for accurate force/torque measurement
        self.robot.ExecutePrimitive("ZeroFTSensor()")
        while self.robot.busy():
            time.sleep(1)
        self.start_impedance()

    # def output(self):
    #     import importlib.util
    #     import os

    #     # 模块文件的路径
    #     module_path = os.path.join('/home/wmingd/Projects/dualarm-serl/serl_robot_infra/franka_env/envs/peg_env', 'config.py')

    #     # 加载模块
    #     spec = importlib.util.spec_from_file_location("PegEnvConfig", module_path)
    #     config = importlib.util.module_from_spec(spec)
    #     spec.loader.exec_module(config)

    #     # 访问参数
    #     print(config.PegEnvConfig.PRECISION_PARAM)
    #     print(config.PegEnvConfig.COMPLIANCE_PARAM)

    def start_impedance(self, stiffness=0.08):
        self.robot.SwitchMode(self.mode.NRT_CARTESIAN_MOTION_FORCE)
        # self.robot.SwitchMode(self.mode.NRT_JOINT_IMPEDANCE)

        # new_Kx = np.multiply(self.robot.info().K_x_nom, stiffness)
        print("K_x_nom: ", self.robot.info().K_x_nom)
        # [10000.0, 10000.0, 10000.0, 2500.0, 2500.0, 2500.0]
        # initial_Kx = [800, 800, 800, 200, 200, 200]
        # new_Kx = [800, 800, 800, 50, 50, 50] # too large
        new_Kx = [500, 500, 500, 10, 10, 10]
        new_Zx = [0.6] * 6  # default 0.7
        # new_Zx = [0.7] * 6 # default 0.7
        # new_Kq = np.multiply(self.robot.info().K_q_nom, stiffness)
        self.robot.SetCartesianImpedance(new_Kx, new_Zx)
        # self.robot.SetJointImpedance(new_Kq)
        self.logger.info(
            f"Start impedance! Cartesian stiffness set to {new_Kx} {new_Zx}"
        )

    def stop_impedance(self):
        new_Kx = np.multiply(self.robot.info().K_x_nom, 1.0)
        self.robot.SetCartesianImpedance(new_Kx)

        # new_Kq = np.multiply(self.robot.info().K_q_nom, 1.0)
        # self.robot.SetJointImpedance(new_Kq)

        self.logger.info(f"Stop Impedance! Cartesian stiffness set to {new_Kx}")
        self.robot.SwitchMode(self.mode.NRT_PRIMITIVE_EXECUTION)

    def reset_joint(self):
        self.stop_impedance()
        try:
            self.logger.info("Moving to home pose")
            self.robot.SwitchMode(self.mode.NRT_PRIMITIVE_EXECUTION)
            self.robot.ExecutePrimitive("Home()")
        except:
            print("Homing failed!")

        # set_trace()

        # Wait for the primitive to finish
        while self.robot.busy():
            time.sleep(1)

        # clear errors
        self.clear()

        self.start_impedance()

    def move(self, pose_xyzw: list):
        """Moves to a pose: [x,y,z,qx,qy,qz,qw]"""
        assert len(pose_xyzw) == 7
        print("start moving !!! ")
        print(self.mode)
        # set_trace()
        quat_xyzw = pose_xyzw[3:]  # [qx, qy, qz, qw]
        quat_wxyz = np.concatenate([quat_xyzw[3:], quat_xyzw[0:3]])  # [qw, qx, qy, qz]
        pose_wxyz = np.concatenate(
            [pose_xyzw[:3], quat_wxyz]
        )  # [x, y, z, qw, qx, qy, qz]
        self.robot.SendCartesianMotionForce(
            pose_wxyz,
            max_linear_vel=0.5,
            max_angular_vel=1.0,
            max_linear_acc=2.0,
            max_angular_acc=5.0,
            # max_linear_vel=1,
            # max_angular_vel=12.0,
            # max_linear_acc=4.0,
            # max_angular_acc=10.0,
            # max_linear_vel=0.2,
            # max_angular_vel=5.0,
            # max_linear_acc=1.0,
            # max_angular_acc=2.0,
        )  # (position 3x1, rotation (quaternion) 4x1)

    def start_free_drive(self):
        self.robot.SwitchMode(self.mode.NRT_PRIMITIVE_EXECUTION)
        self.robot.ExecutePrimitive("FloatingCartesian()")

    def stop_free_drive(self):
        self.start_impedance()

    def get_state(self):
        robot_states = self.robot.states()
        # print("{")
        # print(f"q: {['%.2f' % i for i in robot.states().q]}",)
        # print(f"theta: {['%.2f' % i for i in robot.states().theta]}")
        # print(f"dq: {['%.2f' % i for i in robot.states().dq]}")
        # print(f"dtheta: {['%.2f' % i for i in robot.states().dtheta]}")
        # print(f"tau: {['%.2f' % i for i in robot.states().tau]}")
        # print(f"tau_des: {['%.2f' % i for i in robot.states().tau_des]}")
        # print(f"tau_dot: {['%.2f' % i for i in robot.states().tau_dot]}")
        # print(f"tau_ext: {['%.2f' % i for i in robot.states().tau_ext]}")
        # print(f"tcp_pose: {['%.2f' % i for i in robot.states().tcp_pose]}")
        # print(f"tcp_pose_d: {['%.2f' % i for i in robot.states().tcp_pose_des]}")
        # print(f"tcp_velocity: {['%.2f' % i for i in robot.states().tcp_vel]}")
        # print(f"flange_pose: {['%.2f' % i for i in robot.states().flange_pose]}")
        # print(f"FT_sensor_raw_reading: {['%.2f' % i for i in robot.states().ft_sensor_raw]}")
        # print(f"F_ext_tcp_frame: {['%.2f' % i for i in robot.states().ext_wrench_in_tcp]}")
        # print(f"F_ext_world_frame: {['%.2f' % i for i in robot.states().ext_wrench_in_world]}")
        # print("}", flush= True)
        """ translate robot_states to dict """
        tcp_quat_wxyz = robot_states.tcp_pose[3:]  # quaternion
        tcp_quat_xyzw = np.concatenate(
            [tcp_quat_wxyz[1:], tcp_quat_wxyz[0:1]]
        )  # quaternion
        tcp_pose_xyzw = np.concatenate(
            [robot_states.tcp_pose[:3], tcp_quat_xyzw]
        )  # [x, y, z, qx, qy, qz, qw]
        robot_states_dict = {
            "q": robot_states.q,
            "theta": robot_states.theta,
            "dq": robot_states.dq,
            "dtheta": robot_states.dtheta,
            "jacobian": self.model.J("flange"),
            "tau": robot_states.tau,
            "tau_des": robot_states.tau_des,
            "tau_dot": robot_states.tau_dot,
            "tau_ext": robot_states.tau_ext,
            # "tcp_pose": tcp_pose_xyzw, # wxyz -> xyzw
            "tcp_pose": robot_states.tcp_pose,
            "tcp_pose_d": robot_states.tcp_pose_des,
            "tcp_velocity": robot_states.tcp_vel,
            "flange_pose": robot_states.flange_pose,
            "FT_sensor_raw_reading": robot_states.ft_sensor_raw,
            "F_ext_tcp_frame": robot_states.ext_wrench_in_tcp,
            "F_ext_base_frame": robot_states.ext_wrench_in_world,
        }
        # from ipdb import set_trace; set_trace()
        return robot_states_dict


###############################################################################


def main(_):

    webapp = Flask(__name__)

    # from robot_servers.robotiq_gripper_server import RobotiqGripperServer

    # gripper_server = RobotiqGripperServer(pty_device='/dev/ttyUSB0') # fake gripper

    """Starts impedance controller"""

    robot_server = FlexivServer()

    robot_server.get_state()

    robot_server.start_impedance()

    # set_trace()

    # robot_server.output()

    # robot_server.reset_joint()

    # key
    # Route for Getting Pose
    @webapp.route("/getpos", methods=["POST"])
    def get_pos():
        robot_state = robot_server.get_state()

        return jsonify({"pose": np.array(robot_state["tcp_pose"]).tolist()})

    @webapp.route("/start_free_drive", methods=["POST"])
    def start_free_drive():
        robot_server.start_free_drive()
        return "Started Free Drive Mode"

    @webapp.route("/stop_free_drive", methods=["POST"])
    def stop_free_drive():
        robot_server.stop_free_drive()
        return "Stopped Free Drive Mode"

    @webapp.route("/getpos_euler", methods=["POST"])
    def get_pos_euler():
        r = R.from_quat(robot_server.pos[3:])
        euler = r.as_euler("xyz")
        return jsonify({"pose": np.concatenate([robot_server.pos[:3], euler]).tolist()})

    @webapp.route("/getvel", methods=["POST"])
    def get_vel():
        return jsonify({"vel": np.array(robot_server.vel).tolist()})

    @webapp.route("/getforce", methods=["POST"])
    def get_force():
        return jsonify({"force": np.array(robot_server.force).tolist()})

    @webapp.route("/gettorque", methods=["POST"])
    def get_torque():
        return jsonify({"torque": np.array(robot_server.torque).tolist()})

    @webapp.route("/getq", methods=["POST"])
    def get_q():
        return jsonify({"q": np.array(robot_server.q).tolist()})

    @webapp.route("/getdq", methods=["POST"])
    def get_dq():
        return jsonify({"dq": np.array(robot_server.dq).tolist()})

    @webapp.route("/getjacobian", methods=["POST"])
    def get_jacobian():
        return jsonify({"jacobian": np.array(robot_server.jacobian).tolist()})

    # Route for getting gripper distance
    @webapp.route("/get_gripper", methods=["POST"])
    def get_gripper():
        return jsonify({"gripper": gripper_server.gripper_pos})

    # Route for Running Joint Reset
    @webapp.route("/jointreset", methods=["POST"])
    def joint_reset():
        robot_server.clear()
        robot_server.reset_joint()
        return "Reset Joint"

    # Route for Activating the Gripper
    @webapp.route("/activate_gripper", methods=["POST"])
    def activate_gripper():
        print("activate gripper")
        gripper_server.activate_gripper()
        return "Activated"

    # Route for Resetting the Gripper. It will reset and activate the gripper
    @webapp.route("/reset_gripper", methods=["POST"])
    def reset_gripper():
        print("reset gripper")
        gripper_server.reset_gripper()
        return "Reset"

    # Route for Opening the Gripper
    @webapp.route("/open_gripper", methods=["POST"])
    def open():
        print("open")
        gripper_server.open()
        return "Opened"

    # Route for Closing the Gripper
    @webapp.route("/close_gripper", methods=["POST"])
    def close():
        print("close")
        gripper_server.close()
        return "Closed"

    # Route for moving the gripper
    @webapp.route("/move_gripper", methods=["POST"])
    def move_gripper():
        gripper_pos = request.json
        pos = np.clip(int(gripper_pos["gripper_pos"]), 0, 255)  # 0-255
        print(f"move gripper to {pos}")
        gripper_server.move(pos)
        return "Moved Gripper"

    # Route for Clearing Errors (Communcation constraints, etc.)
    @webapp.route("/clearerr", methods=["POST"])
    def clear_err():
        robot_server.clear()
        return "Clear"

    # Route for Clearing Errors (Communcation constraints, etc.)
    @webapp.route("/clearforce", methods=["POST"])
    def clear_force():
        robot_server.clear_force()
        return "Clear Force"

    # key
    # Route for Sending a pose command
    @webapp.route("/pose", methods=["POST"])
    def pose():
        pos = np.array(request.json["arr"])
        print("Moving to", pos)
        robot_server.move(pos)
        return "Moved"

    # key
    # Route for getting all state information
    @webapp.route("/getstate", methods=["POST"])
    def get_state():
        # robot_states_dict = {
        #     "q": robot_states.q,
        #     "theta": robot_states.theta,
        #     "dq": robot_states.dq,
        #     "dtheta": robot_states.dtheta,
        # "jacobian": self.model.J("flange"),
        #     "tau": robot_states.tau,
        #     "tau_des": robot_states.tau_des,
        #     "tau_dot": robot_states.tau_dot,
        #     "tau_ext": robot_states.tau_ext,
        #     "tcp_pose": robot_states.tcp_pose,
        #     "tcp_pose_d": robot_states.tcp_pose_des,
        #     "tcp_velocity": robot_states.tcp_vel,
        #     "flange_pose": robot_states.flange_pose,
        #     "FT_sensor_raw_reading": robot_states.ft_sensor_raw,
        #     "F_ext_tcp_frame": robot_states.ext_wrench_in_tcp,
        #     "F_ext_base_frame": robot_states.ext_wrench_in_world,
        # }
        robot_state = robot_server.get_state()
        return jsonify(
            {
                "pose": np.array(robot_state["tcp_pose"]).tolist(),
                "vel": np.array(robot_state["tcp_velocity"]).tolist(),
                "force": np.array(robot_state["F_ext_tcp_frame"][:3]).tolist(),
                "torque": np.array(robot_state["F_ext_tcp_frame"][3:]).tolist(),
                "q": np.array(robot_state["q"]).tolist(),
                "dq": np.array(robot_state["dq"]).tolist(),
                # "jacobian": np.array(np.zeros((6, 7))).tolist(),
                "jacobian": np.array(robot_state["jacobian"]).tolist(),
                "gripper_pos": [0],
            }
        )

    # Route for updating compliance parameters
    @webapp.route("/update_param", methods=["POST"])
    def update_param():
        return "Updated"

    webapp.run(host="0.0.0.0")


if __name__ == "__main__":
    app.run(main)
