# from holodex.robot.arm.franka_fr3.franka_env_wrapper import FrankaEnvWrapper
import rospy
from holodex.constants import *
import warnings
import time
from termcolor import cprint

warnings.filterwarnings(
    "ignore",
    message="Link .* is of type 'fixed' but set as active in the active_links_mask.*",
)
# load module according to hand type
if HAND_TYPE is not None:
    hand_module = __import__("holodex.robot.hand")
    KDLControl_module_name = f"{HAND_TYPE}KDLControl" 
    JointControl_module_name = f"{HAND_TYPE}JointControl" 
    Hand_module_name = f"{HAND_TYPE}Hand" 
    # get relevant classes
    KDLControl = (
        getattr(hand_module.robot, KDLControl_module_name)
    )
    JointControl = (
        getattr(hand_module.robot, JointControl_module_name)
    )
    Hand = getattr(hand_module.robot, Hand_module_name) 
    # load constants according to hand type
    hand_type = HAND_TYPE.lower() 
    JOINTS_PER_FINGER = (
        eval(f"{hand_type.upper()}_JOINTS_PER_FINGER") 
    )
    JOINT_OFFSETS = (
        eval(f"{hand_type.upper()}_JOINT_OFFSETS") 
    )
    
if ARM_TYPE is not None:
    # load module according to arm type
    arm_module = __import__("holodex.robot.arm")
    Arm_module_name = f"{ARM_TYPE}Arm"
    Arm = getattr(arm_module.robot, Arm_module_name)

class RobotController(object):
    def __init__(
        self,
        teleop,
        servo_mode=True,
        arm_control_mode="ik",
        hand_control_mode="joint",
        home=True,
        random_arm_home=False,
    ) -> None:
        if ARM_TYPE == "Flexiv":
            self.arm = Arm()
            cprint("Call Flexiv Arm", "red")
        elif ARM_TYPE == "Franka":
            from holodex.robot.arm.franka.franka_env_wrapper import FrankaEnvWrapper
            self.arm = FrankaEnvWrapper(control_mode="joint") # modify this 
            cprint("Call Franka Arm", "red")
        elif ARM_TYPE == "FrankaGenesis":
            from holodex.robot.arm.franka.franka_genesis_env_wrapper import FrankaGenesisEnvWrapper
            self.arm = FrankaGenesisEnvWrapper(control_mode="joint") # modify this 
            cprint("Call FrankaGenesis Arm", "red")
        else:
            self.arm = (
                Arm(
                    servo_mode=servo_mode,
                    teleop=teleop,
                    control_mode=arm_control_mode,
                    safety_moving_trans=JAKA_SAFE_MOVING_TRANS,
                    random_jaka_home=random_arm_home,
                )
                if ARM_TYPE is not None
                else None
            )

        self.arm_control_mode = arm_control_mode
        cprint(f"self.arm_control_mode: {self.arm_control_mode}", "red")

        if HAND_TYPE is not None:
            self.hand_control_mode = hand_control_mode
            self.hand = (
                Hand() 
            )  # TODO add different control mode for hand
            self.hand_KDLControl = KDLControl() 
            self.hand_JointControl = JointControl() 
            self.joints_per_finger = JOINTS_PER_FINGER
            self.joint_offsets = JOINT_OFFSETS
        else:
            self.hand = None
        self.teleop = teleop
        # self.home = home
        # if self.home is True:
        #     self.home_robot()
        # self.home_robot()
        
    def home_robot(self):
        if ARM_TYPE is not None:
            self.arm.home_robot()
        if HAND_TYPE is not None:
            self.hand.home_robot()

    def reset_robot(self):
        if ARM_TYPE is not None:
            self.arm.reset()
        if HAND_TYPE is not None:
            self.hand.reset()

    def get_arm_position(self):
        return self.arm.get_arm_position()

    def get_arm_velocity(self):
        return self.arm.get_arm_velocity()

    def get_arm_tcp_position(self):
        return self.arm.get_tcp_position()

    def get_arm_torque(self):
        return self.arm.get_arm_torque()

    def get_hand_position(self):
        return self.hand.get_hand_position()

    def get_hand_velocity(self):
        return self.hand.get_hand_velocity()

    def get_hand_torque(self):
        return self.hand.get_hand_torque()

    def move_hand(self, input_angles):
        if "LEAP" in HAND_TYPE.upper() and self.teleop:
            input_angles = np.array(input_angles)[
                [1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15]
            ]
        self.hand.move(input_angles)

    def move_arm(self, input_cmd: np.ndarray):
        self.arm.move(input_cmd)
        rospy.sleep(SLEEP_TIME)

    def move_arm_and_hand(self, input_cmd):
        '''
        First arm then hand.
        '''
        assert (
            self.arm is not None and self.hand is not None
        ), "Arm and hand are not initialized"
        # TODO simultaneous movement
        self.move_arm(input_cmd[: self.arm.dof])
        self.move_hand(input_cmd[self.arm.dof :])

    def move(self, input_cmd):
        '''
        Any movement, including arm and hand.
        '''
        if self.arm is not None:
            # TODO: add flexiv support
            self.move_arm(input_cmd[: self.arm.dof])
            if self.hand is not None:
                self.move_hand(input_cmd[self.arm.dof :])
        elif self.hand is not None:
            self.move_hand(input_cmd)
            rospy.sleep(SLEEP_TIME)

    def move_seperate(self, action: dict):
        # TODO: no interpolation
        if self.arm is not None:
            self.move_arm(action["arm"])
        if self.hand is not None:
            self.move_hand(action["hand"])
        rospy.sleep(1/5)
                
    def servo_move(self, action: dict, n_interpolations: int = 30):
        # hand interpolation
        if self.hand_control_mode == "joint":
            start_hand_pos = self.get_hand_position()
            end_hand_pos = action["hand"]

        # arm interpolation
        if self.arm_control_mode == "interpo_ik":
            start_arm_pos = self.get_arm_tcp_position()
            current_joint = self.get_arm_position()
            start_arm_pos = np.array(self.arm.compute_ik(current_joint, start_arm_pos))
            end_arm_pos = np.array(self.arm.compute_ik(current_joint, action["arm"]))
        elif self.arm_control_mode == "joint":
            start_arm_pos = self.get_arm_position()
            end_arm_pos = action["arm"]

        # make the robot move
        for step in range(1, n_interpolations + 1):
            interpolated_hand_pos = (
                start_hand_pos
                + (end_hand_pos - start_hand_pos) * step / n_interpolations
            )
            interpolated_arm_pos = (
                start_arm_pos + (end_arm_pos - start_arm_pos) * step / n_interpolations
            )

            self.move_arm(interpolated_arm_pos)
            self.move_hand(interpolated_hand_pos)
            rospy.sleep(SLEEP_TIME)

    def servo_move_self_interpo(self, action: dict, loop_time: int = 0.4):
        # hand interpolation
        if self.hand_control_mode == "joint":
            start_hand_pos = self.get_hand_position()
            end_hand_pos = action["hand"]

        # arm interpolation
        if self.arm_control_mode == "interpo_ik":
            start_arm_pos = self.get_arm_tcp_position()
            current_joint = self.get_arm_position()
            start_arm_pos = np.array(self.arm.compute_ik(current_joint, start_arm_pos))
            end_arm_pos = np.array(self.arm.compute_ik(current_joint, action["arm"]))
        elif self.arm_control_mode == "joint":
            start_arm_pos = self.get_arm_position()
            end_arm_pos = action["arm"]

        n_interpolations = int(loop_time / (SLEEP_TIME))
        hand_control_step = int(loop_time / (0.05))

        # make the robot move
        for step in range(1, n_interpolations + 1):
            interpolated_hand_pos = (
                start_hand_pos
                + (end_hand_pos - start_hand_pos) * step / n_interpolations
            )
            interpolated_arm_pos = (
                start_arm_pos + (end_arm_pos - start_arm_pos) * step / n_interpolations
            )
            # st = time.time()
            self.move_arm(interpolated_arm_pos)

            if step % hand_control_step == 0:
                self.move_hand(end_hand_pos)

            # print(f"Time taken: {time.time() - st}")
        # compute the difference between the desired and actual position
        hand_position = self.get_hand_position()
        hand_position_diff = np.linalg.norm(end_hand_pos - hand_position)
        # print(f"Hand position difference: {hand_position_diff}")
        arm_position = self.get_arm_tcp_position()
        arm_position_diff = np.linalg.norm(action["arm"][:3] - arm_position[:3])
        arm_orientation_diff = np.linalg.norm(action["arm"][3:] - arm_position[3:])
        
if __name__ == "__main__":
    rospy.init_node("test")
    robot = RobotController(teleop=False, random_arm_home=False, home=True)
    # robot.move_hand(np.zeros(16))
    # get  ee pose
    # print(robot.get_arm_tcp_position())

    # rospy.sleep(2)
    # tcp_position = robot.get_arm_tcp_position()
    # # tcp_position[0] += 0.15
    # # tcp_position[1] -= 0.05
    # tcp_position[2] += 0.05
    # rospy.sleep(2)

    # robot.move_arm(tcp_position)
    # rospy.sleep(2)

    # print(robot.get_arm_position())
    




    # robot.reset_robot()
    # robot.home_robot()

    # import time
    # while True:
    # x = robot.get_arm_tcp_position()
    # print(robot.get_arm_tcp_position())
    # hand_position = robot.get_hand_position()
    # arm_tcp = robot.get_arm_tcp_position()

    # print(hand_position)
    # print(arm_tcp)

    # hand_position[1] += 1.57
    # ##############################################################################################################
    # cur_hand_position = robot.get_hand_position()
    # target_hand_position = cur_hand_position.copy()
    # hand_position = np.zeros(16)
    # hand_position[13] = 1.57
    # while True:
    # #     hand_position = robot.get_hand_position()
    # #     cur_hand_position = hand_position.copy()
    # #     print(cur_hand_position[1], target_hand_position[1], target_hand_position[1]-cur_hand_position[1])
    # #     hand_position[1] += 0.1
    # #     target_hand_position = hand_position.copy()
    #     robot.move_hand(hand_position)
    #     time.sleep(0.1)

    ##############################################################################################################
    # arm_position[1] += 0.2
    # robot.move_arm(arm_position)

    #     current_hand_position = robot.get_hand_position()
    #     print(current_hand_position)
    #     current_arm_position = robot.get_arm_position()
    # print(current_arm_position)
    # # print difference
    # print("Hand position difference: ", current_hand_position - hand_position)
    # print("Arm position difference: ", current_arm_position - arm_position)

    # current_hand_position[1] -= 1.57
    # current_arm_position[1] -= 0.2
    # robot.move_hand(current_hand_position)
    # robot.move_arm(current_arm_position)
    # robot.move_arm_and_hand(np.concatenate((current_arm_position, current_hand_position)))

    ################################################################################################################
    # a = A()
