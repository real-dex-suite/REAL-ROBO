import rospy
from holodex.constants import *

# load module according to hand type
hand_module = __import__("holodex.robot.hand")
KDLControl_module_name = f'{HAND_TYPE}KDLControl'
JointControl_module_name = f'{HAND_TYPE}JointControl'
Hand_module_name = f'{HAND_TYPE}Hand'
# get relevant classes
KDLControl = getattr(hand_module.robot, KDLControl_module_name)
JointControl = getattr(hand_module.robot, JointControl_module_name)
Hand = getattr(hand_module.robot, Hand_module_name)

if ARM_TYPE is not None:
    # load module according to arm type
    arm_module = __import__("holodex.robot.arm")
    Arm_module_name = f'{ARM_TYPE}Arm'
    Arm = getattr(arm_module.robot, Arm_module_name)

# load constants according to hand type
hand_type = HAND_TYPE.lower()
JOINTS_PER_FINGER = eval(f'{hand_type.upper()}_JOINTS_PER_FINGER')
JOINT_OFFSETS = eval(f'{hand_type.upper()}_JOINT_OFFSETS')

class RobotController(object):
    def __init__(self, teleop, servo_mode=True, arm_control_mode="ik", hand_control_mode="joint") -> None:
        self.arm = Arm(servo_mode=servo_mode, teleop=teleop, control_mode=arm_control_mode, safety_moving_trans=JAKA_SAFE_MOVING_TRANS) if ARM_TYPE is not None else None

        self.hand = Hand() if HAND_TYPE is not None else None # TODO add different control mode for hand
        self.hand_KDLControl = KDLControl() if HAND_TYPE is not None else None
        self.hand_JointControl = JointControl() if HAND_TYPE is not None else None
        self.joints_per_finger = JOINTS_PER_FINGER
        self.joint_offsets = JOINT_OFFSETS
        self.teleop = teleop

        self.home_robot()
    
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
            input_angles = np.array(input_angles)[[1,0,2,3,5,4,6,7,9,8,10,11,12,13,14,15]]
        self.hand.move(input_angles)
    
    def move_arm(self, input_angles):
        self.arm.move(input_angles)
        rospy.sleep(SLEEP_TIME)

    def move_arm_and_hand(self, input_angles):
        assert self.arm is not None and self.hand is not None, "Arm and hand are not initialized"
        # TODO simultaneous movement
        self.move_arm(input_angles[:self.arm.dof])
        self.move_hand(input_angles[self.arm.dof:])
    
    def move(self, input_angles):
        if self.arm is not None:
            self.move_arm(input_angles[:self.arm.dof])
            if self.hand is not None:
                self.move_hand(input_angles[self.arm.dof:])
        elif self.hand is not None:
            self.move_hand(input_angles)

if __name__ == "__main__":
    rospy.init_node("test")
    robot = RobotController(teleop=False)
    robot.home_robot()
    # robot.reset_robot()
    # arm_position = robot.get_arm_position()
    # hand_position = robot.get_hand_position()
    # hand_position[1] += 1.57
    # robot.move_hand(hand_position)
    # arm_position[1] += 0.2
    # robot.move_arm(arm_position)

    # current_hand_position = robot.get_hand_position()
    # current_arm_position = robot.get_arm_position()
    # # print difference
    # print("Hand position difference: ", current_hand_position - hand_position)
    # print("Arm position difference: ", current_arm_position - arm_position)

    # current_hand_position[1] -= 1.57
    # current_arm_position[1] -= 0.2
    # robot.move_hand(current_hand_position)
    # robot.move_arm(current_arm_position)
    # robot.move_arm_and_hand(np.concatenate((current_arm_position, current_hand_position)))
