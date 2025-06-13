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
    
class RobotController(object):
    def __init__(
        self,
        teleop,
        arm_type="franka",
        simulator=None,
        gripper=None,
        gripper_init_state="open",
    ) -> None:
        self.arm_type = arm_type
        if arm_type == "flexiv":
            from holodex.robot.arm.flexiv.flexiv import FlexivArm
            self.arm = FlexivArm(gripper=gripper, gripper_init_state=gripper_init_state)
            cprint("Call FlexivArm", "red")
        elif arm_type == "franka":
            if simulator is not None:
                if simulator == "genesis":
                    from holodex.robot.arm.franka.franka_genesis_env_wrapper import FrankaGenesisEnvWrapper
                    self.arm = FrankaGenesisEnvWrapper(control_mode="joint", gripper=gripper, gripper_init_state=gripper_init_state) # modify this 
                    cprint("Call FrankaGenesisEnvWrapper", "red")
                else:
                    raise NotImplementedError(f"Robot controller under simulator {simulator} is not implemented.")
            else:
                from holodex.robot.arm.franka.franka_env_wrapper import FrankaEnvWrapper
                self.arm = FrankaEnvWrapper(control_mode="joint", gripper=gripper, gripper_init_state=gripper_init_state) # modify this 
                cprint("Call FrankaEnvWrapper", "red")
        elif arm_type == "jaka":
            from holodex.robot.arm.jaka.jaka import JakaArm
            self.arm = (
                JakaArm(
                    servo_mode=True,
                    teleop=teleop,
                    control_mode="joint",
                    safety_moving_trans=JAKA_SAFE_MOVING_TRANS,
                    random_jaka_home=False,
                    gripper=gripper,
                    gripper_init_state=gripper_init_state,
                )
            )
            cprint("Call JakaArm", "red")
        else:
            raise NotImplementedError("Unknown arm_type")

        if HAND_TYPE is not None:
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

    def home_robot(self):
        if self.arm_type is not None:
            self.arm.home_robot()
        if HAND_TYPE is not None:
            self.hand.home_robot()

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
