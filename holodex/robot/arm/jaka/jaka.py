import numpy as np
import rospy
from sensor_msgs.msg import JointState
# from holodex.constants import JAKA_JOINT_STATE_TOPIC, JAKA_POSITIONS
from jkrc import jkrc
from holodex.constants import JAKA_IP, JAKA_POSITIONS, JAKA_DOF, JAKA_JOINT_STATE_TOPIC, JAKA_COMMANDED_JOINT_STATE_TOPIC, JAKA_EE_POSE_TOPIC
from holodex.utils.network import JointStatePublisher, FloatArrayPublisher

class JakaArm(object):
    def __init__(self, servo_mode=True, control_mode="ik" ,teleop=False, safety_moving_trans = 100):
        # rospy.init_node('jaka_arm_controller')

        # Creating ROS Publishers
        self.joint_state_publisher = JointStatePublisher(publisher_name=JAKA_JOINT_STATE_TOPIC)
        self.command_joint_state_publisher = JointStatePublisher(publisher_name=JAKA_COMMANDED_JOINT_STATE_TOPIC)
        self.ee_pose_publisher = FloatArrayPublisher(publisher_name=JAKA_EE_POSE_TOPIC)

        self.robot = jkrc.RC(JAKA_IP)
        ret = self.robot.login()
        print("Robot logging: ", ret)

        ret = self.robot.power_on()
        print("Robot power on: ", ret)
        # if has collision, recover from collision
        success, collision = self.robot.is_in_collision()
        if collision:
            self.robot.collision_recover()
            self.robot.enable_robot()
        ret = self.robot.enable_robot()
        print("Robot enable: ", ret)
        self.jaka_joint_state = None
        # TODO change to ros?
        # rospy.Subscriber(KINOVA_JOINT_STATE_TOPIC, JointState, self._callback_joint_state, queue_size = 1)
        self.teleop = teleop

        self.move_mode = 0
        self.is_block = True
        self.speed = 10
        self.acc = 5
        self.tol = 0.1
        self.dof = JAKA_DOF
        self.safety_moving_trans = safety_moving_trans
        self.joint_vel_limit = 0.5 #TODO configureable
        self.joint_pos_limit = np.array([6.28, 2.09, 2.27, 6.28, 2.09, 6.28])

        self.servo_mode = servo_mode
        self.control_mode = control_mode
        if self.teleop:
            assert self.control_mode == "ik"

        self.robot.servo_move_enable(self.servo_mode)

        self.home_robot()

    def _callback_joint_state(self):
        self.jaka_joint_state = self.robot.get_joint_position()[1]

    def get_arm_position(self):
        # if self.jaka_joint_state is None:
        #     return None
        self._callback_joint_state()
        return np.array(self.jaka_joint_state, dtype = np.float32)
    
    def get_tcp_position(self):
        return np.array(self.robot.get_tcp_position()[1])
    
    def set_tcp_position(self, input_pose):
        self.robot.linear_move_extend(input_pose, self.move_mode, self.is_block, self.speed, self.acc, self.tol)

    def home_robot(self):
        self.move_joint(JAKA_POSITIONS['home'])

    def reset(self):
        self.move_joint(JAKA_POSITIONS['home'])

    def move_joint(self, input_angles):
        self.robot.joint_move(input_angles, self.move_mode, self.is_block, self.speed)
    
    def safety_check(self, target_arm_pose):
        current_arm_pose = self.get_tcp_position()
        if np.any(np.abs(target_arm_pose[:3] - current_arm_pose[:3]) > self.safety_moving_trans):
            print('Target pose is too far from current pose, arm will not moving')
            self.robot.motion_abort()
            return current_arm_pose
        else:
            return target_arm_pose
    
    def limit_joint_vel(self, target_joint):
        current_joint = self.get_arm_position()
        target_joint = current_joint + np.clip(target_joint - current_joint, -self.joint_vel_limit, self.joint_vel_limit)
        return target_joint
    
    def compute_ik(self, current_joint, cart_pose):
        joint_tuple = self.robot.kine_inverse(current_joint, cart_pose)
        if len(joint_tuple) > 1:
            return joint_tuple[1]
        else:
            print(joint_tuple)
            print('Inverse kinematics failed, arm will not moving')
            return current_joint
        
    def compute_joint(self, cart_pose):
        current_joint = self.get_arm_position()
        joint_tuple = self.robot.kine_inverse(current_joint, cart_pose)
        if len(joint_tuple) > 1:
            return joint_tuple[1]
        else:
            print(joint_tuple)
            print('Inverse kinematics failed, arm will not moving')
            return current_joint
    
    def limit_joint_pos(self, target_joint):
        target_joint = np.clip(target_joint, -self.joint_pos_limit, self.joint_pos_limit)
        return target_joint

    def publish_state(self, input_cmd = None):
        current_joint = self.get_arm_position()
        self.joint_state_publisher.publish(current_joint)

        current_ee_pose = self.get_tcp_position()
        self.ee_pose_publisher.publish(current_ee_pose)

        if input_cmd is not None:
            self.command_joint_state_publisher.publish(input_cmd)
    
    def move(self, input_cmd):
        if self.teleop:
            input_cmd = self.safety_check(input_cmd)
        # TODO add pose command
        if self.servo_mode:
            self.robot.servo_move_enable(True)
            if self.control_mode == "ik":
                input_cmd = self.compute_joint(input_cmd)
            if self.teleop:
                input_cmd = self.limit_joint_vel(input_cmd)
                input_cmd = self.limit_joint_pos(input_cmd)
            
                self.publish_state(input_cmd) #TODO maybe change to use ros

            self.robot.servo_j(input_cmd, self.move_mode)
        else:
            self.robot.joint_move(input_cmd, self.move_mode, self.is_block, self.speed)
        
if __name__ == '__main__':
    jaka = JakaArm()
    current_tcp_position = jaka.get_tcp_position()
    while True:
        # current_tcp_position[1]-=0.01
        print('target:', current_tcp_position)
        # jaka.move(current_tcp_position)
        jaka.publish_state()  