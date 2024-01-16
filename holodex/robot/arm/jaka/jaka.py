import numpy as np
import rospy
from sensor_msgs.msg import JointState
from holodex.constants import JAKA_JOINT_STATE_TOPIC, JAKA_POSITIONS
from jkrc import jkrc

class JakaArm(object):
    def __init__(self):
        rospy.init_node('jaka_arm_controller')

        self.robot = jkrc.RC("192.168.90.71")#返回一个机器人对象

        self.jaka_joint_state = None
        # TODO change to ros?
        # rospy.Subscriber(KINOVA_JOINT_STATE_TOPIC, JointState, self._callback_joint_state, queue_size = 1)
        self.move_mode = 0
        self.is_block = True
        self.speed = 0.1

    def _callback_joint_state(self):
        self.jaka_joint_state = self.robot.get_joint_position()[1]

    def get_arm_position(self):
        if self.jaka_joint_state is None:
            return None
        
        return np.array(self.jaka_joint_state, dtype = np.float32)

    def home_robot(self):
        self.robot.move(JAKA_POSITIONS['flat'])

    def reset(self):
        self.robot.move(JAKA_POSITIONS['flat'])

    def move(self, input_angles):
        self.robot.joint_move(input_angles, self.move_mode, self.is_block, self.speed)