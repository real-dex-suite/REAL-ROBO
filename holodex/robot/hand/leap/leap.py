import numpy as np
import rospy
from sensor_msgs.msg import JointState
from .leap_kdl import LeapKDL
from holodex.constants import *
from holodex.utils.files import get_yaml_data, get_path_in_package
from holodex.robot.hand.leap.controller import LeapNode # From https://github.com/NYU-robot-learning/DIME-Controllers
import holodex.robot.hand.leap.leap_hand_utils.leap_hand_utils as lhu

from copy import deepcopy as copy
try:
    from leap_hand.srv import *
except:
    pass
class LeapHand(object):
    def __init__(self):
        self.kdl_solver = LeapKDL()
        self.controller = LeapNode(LEAP_CMD_TYPE)
        self.joint_limit_config = get_yaml_data(get_path_in_package("robot/hand/leap/configs/leap_link_info.yaml"))['links_info']

        self.leap_joint_state = None
        self.leap_commanded_joint_state = None

        # rospy.wait_for_service('/leap_position')
        # self.leap_position = rospy.ServiceProxy('/leap_position', leap_position)
        # rospy.wait_for_service('/leap_velocity')
        # self.leap_velocity = rospy.ServiceProxy('/leap_velocity', leap_velocity)
        # rospy.wait_for_service('/leap_effort')
        # self.leap_effort = rospy.ServiceProxy('/leap_effort', leap_effort)

        rospy.Subscriber(LEAP_JOINT_STATE_TOPIC, JointState, self._callback_joint_state, queue_size = 1)
        rospy.Subscriber(LEAP_COMMANDED_JOINT_STATE_TOPIC, JointState, self._callback_commanded_joint_state, queue_size = 1)

    def _callback_joint_state(self, joint_state):
        self.leap_joint_state = joint_state

    def _callback_commanded_joint_state(self, joint_state):
        self.leap_commanded_joint_state = joint_state

    def get_hand_position(self):
        if self.leap_joint_state is None:
            return None
        
        if LEAP_CMD_TYPE == 'allegro':
            leap_hand_position = np.array(self.leap_joint_state.position, dtype = np.float32)
            leap_hand_position = lhu.LEAPhand_to_allegro(leap_hand_position, zeros=False)
        elif LEAP_CMD_TYPE == 'leap':
            leap_hand_position = np.array(self.leap_joint_state.position, dtype = np.float32)
            # TODO configure position type
        return leap_hand_position

    def get_hand_velocity(self):
        if self.leap_joint_state is None:
            return None

        return np.array(self.leap_joint_state.velocity, dtype = np.float32)

    def get_hand_torque(self):
        if self.leap_joint_state is None:
            return None

        return np.array(self.leap_joint_state.effort, dtype = np.float32)

    def get_commanded_joint_position(self):
        if self.leap_commanded_joint_state is None:
            return None

        return np.array(self.leap_commanded_joint_state.position, dtype = np.float32)

    def home_robot(self):
        self.move(LEAP_HOME_POSITION)

    def reset(self):
        self.move(LEAP_HOME_POSITION)

    # Moving the fingers to random positions
    def _get_finger_limits(self, finger_type):
        finger_min = np.array(self.joint_limit_config[finger_type]['joint_min'])
        finger_max = np.array(self.joint_limit_config[finger_type]['joint_max'])
        return finger_min, finger_max

    def _get_thumb_random_angles(self):
        thumb_low_limit, thumb_high_limit = self._get_finger_limits('thumb')

        random_angles = np.zeros((LEAP_JOINTS_PER_FINGER))
        for idx in range(LEAP_JOINTS_PER_FINGER - 1): # ignoring the base
            random_angles[idx + 1] = 0.5 * (thumb_low_limit[idx + 1] + (np.random.rand() * (thumb_high_limit[idx + 1] - thumb_low_limit[idx + 1])))

        return random_angles

    def _get_finger_random_angles(self, finger_type):
        if finger_type == 'thumb':
            return self._get_thumb_random_angles()

        finger_low_limit, finger_high_limit = self._get_finger_limits(finger_type)

        random_angles = np.zeros((LEAP_JOINTS_PER_FINGER))
        for idx in range(LEAP_JOINTS_PER_FINGER - 1): # ignoring the base
            random_angles[idx + 1] = 0.8 * (finger_low_limit[idx + 1] + (np.random.rand() * (finger_high_limit[idx + 1] - finger_low_limit[idx + 1])))

        random_angles[0] = -0.1 + (np.random.rand() * 0.2) # Base angle
        return random_angles

    def set_random_position(self):
        random_angles = []
        for finger_type in ['index', 'middle', 'ring', 'thumb']:
            random_angles.append(self._get_finger_random_angles(finger_type))

        target_angles = np.hstack(random_angles)
        self.move(target_angles)

    def get_fingertip_coords(self, joint_positions):
        index_coords = self.kdl_solver.finger_forward_kinematics('index', joint_positions[:4])[0]
        middle_coords = self.kdl_solver.finger_forward_kinematics('middle', joint_positions[4:8])[0]
        ring_coords = self.kdl_solver.finger_forward_kinematics('ring', joint_positions[8:12])[0]
        thumb_coords = self.kdl_solver.finger_forward_kinematics('thumb', joint_positions[12:16])[0]

        finger_tip_coords = np.hstack([index_coords, middle_coords, ring_coords, thumb_coords])
        return np.array(finger_tip_coords)

    def _get_joint_state_from_coord(self, index_tip_coord, middle_tip_coord, ring_tip_coord, thumb_tip_coord):
        current_joint_angles = list(self.get_hand_position())

        index_joint_angles = self.kdl_solver.finger_inverse_kinematics('index', index_tip_coord, current_joint_angles[0:4])
        middle_joint_angles = self.kdl_solver.finger_inverse_kinematics('middle', middle_tip_coord, current_joint_angles[4:8])
        ring_joint_angles = self.kdl_solver.finger_inverse_kinematics('ring', ring_tip_coord, current_joint_angles[8:12])
        thumb_joint_angles = self.kdl_solver.finger_inverse_kinematics('thumb', thumb_tip_coord, current_joint_angles[12:16])

        desired_joint_angles = copy(current_joint_angles)
        
        for idx in range(4):
            desired_joint_angles[idx] = index_joint_angles[idx]
            desired_joint_angles[4 + idx] = middle_joint_angles[idx]
            desired_joint_angles[8 + idx] = ring_joint_angles[idx]
            desired_joint_angles[12 + idx] = thumb_joint_angles[idx]

        return desired_joint_angles

    def move_to_coords(self, fingertip_coords):
        desired_angles = self._get_joint_state_from_coord(
            fingertip_coords[0:3],
            fingertip_coords[3:6],
            fingertip_coords[6:9],
            fingertip_coords[9:],
        )

        self.controller.move_hand(desired_angles)

    def move(self, angles):
        self.controller.move_hand(angles)