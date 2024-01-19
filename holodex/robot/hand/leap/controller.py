#!/usr/bin/env python3
import numpy as np
import rospy
from sensor_msgs.msg import JointState

from holodex.robot.hand.leap.leap_hand_utils.dynamixel_client import *
import holodex.robot.hand.leap.leap_hand_utils.leap_hand_utils as lhu
from leap_hand.srv import *#######################################################
"""This can control and query the LEAP Hand

I recommend you only query when necessary and below 90 samples a second.  Each of position, velociy and current costs one sample, so you can sample all three at 30 hz or one at 90hz.

#Allegro hand conventions:
#0.0 is the all the way out beginning pose, and it goes positive as the fingers close more and more
#http://wiki.wonikrobotics.com/AllegroHandWiki/index.php/Joint_Zeros_and_Directions_Setup_Guide I belive the black and white figure (not blue motors) is the zero position, and the + is the correct way around.  LEAP Hand in my videos start at zero position and that looks like that figure.

#LEAP hand conventions:
#180 is flat out for the index, middle, ring, fingers, and positive is closing more and more.

"""
########################################################
class LeapNode:
    def __init__(self, cmd_type, vel_limit=100):
        self.cmd_type = cmd_type

        self.cmd_allegro = rospy.Publisher("/leaphand_node/cmd_allegro", JointState, queue_size = 1) 
        self.cmd_leap = rospy.Publisher("/leaphand_node/cmd_leap", JointState, queue_size = 1)
        self.cmd_ones = rospy.Publisher("/leaphand_node/cmd_ones", JointState, queue_size = 1)
        
        self.angle_min, self.angle_max = lhu.LEAPsim_limits()
        self.vel_limit = vel_limit

        self.prev_pos = self.pos = self.curr_pos = lhu.allegro_to_LEAPhand(np.zeros(16))

    #Receive LEAP pose and directly control the robot
    def set_leap(self, pose):
        self.prev_pos = self.curr_pos
        pose = self.compute_pose_according_to_vel_limit(self.prev_pos, pose)
        self.curr_pos = np.array(pose)
        #Set the position of the hand when you're done
        stater = JointState()
        stater.position = self.curr_pos.copy()
        self.cmd_leap.publish(stater)
    #allegro compatibility
    def set_allegro(self, pose, clip=True):
        if clip:
            pose = np.clip(pose, self.angle_min, self.angle_max)
        # TODO vel limit weired, clean publish for diff func
        pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
        self.prev_pos = self.curr_pos
        pose = self.compute_pose_according_to_vel_limit(self.prev_pos, pose)
        self.curr_pos = np.array(pose)
        #Set the position of the hand when you're done
        stater = JointState()
        stater.position = self.curr_pos
        self.cmd_leap.publish(stater)
    
    #Sim compatibility, first read the sim value in range [-1,1] and then convert to leap
    def set_ones(self, pose):
        pose = lhu.sim_ones_to_LEAPhand(np.array(pose))
        self.prev_pos = self.curr_pos
        pose = self.compute_pose_according_to_vel_limit(self.prev_pos, pose)
        self.curr_pos = np.array(pose)
        #Set the position of the hand when you're done
        stater = JointState()
        stater.position = self.curr_pos
        self.cmd_leap.publish(stater)

    def compute_pose_according_to_vel_limit(self, prev_pos, curr_pos):
        # compute the velocity
        vel = curr_pos - prev_pos
        # clip the velocity
        vel = np.clip(vel, -self.vel_limit, self.vel_limit)
        # compute the position
        pose = prev_pos + vel
        return pose
    
    #compatibility
    def move_hand(self, pose, clip=True):
        if self.cmd_type == "allegro":
            self.set_allegro(pose, clip=clip)
        elif self.cmd_type == "leap": # TODO configure other cmd clip
            self.set_leap(pose)
        elif self.cmd_type == "ones":   
            self.set_ones(pose)

#init the node
def main(**kwargs):
    leap_hand = LeapNode()
    while True:
        # leap_hand.set_leap(np.ones(16)*3.14)
        joint_test = np.zeros(16)
        joint_test[0] = -1
        joint_test[1] = 1.57
        leap_hand.set_allegro(joint_test)
        print("Position: " + str(leap_hand.read_pos()))
        time.sleep(0.03)


if __name__ == "__main__":
    main()
