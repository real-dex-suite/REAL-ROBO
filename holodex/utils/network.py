import sys
import os
# append PATH: '/home/agibot/catkin_ws/devel/lib/python3/dist-packages', '/opt/ros/noetic/lib/python3/dist-packages'
# source 
# ros_setup_path = '/opt/ros/noetic/setup.bash'
# catkin_ws_path = '/home/agibot/catkin_ws/devel/setup.bash'

# os.system(f'source {ros_setup_path}')
# os.system(f'source {catkin_ws_path}')



import rospy
import zmq

from std_msgs.msg import Float64MultiArray, Bool
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge, CvBridgeError

def create_pull_socket(HOST, PORT):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.bind('tcp://{}:{}'.format(HOST, PORT))
    return socket

def create_push_socket(HOST, PORT):
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.bind('tcp://{}:{}'.format(HOST, PORT))
    return socket

def frequency_timer(frequency):
    return rospy.Rate(frequency)
    
# ROS Topic Pub/Sub classes
class FloatArrayPublisher(object):
    def __init__(self, publisher_name):
        # Initializing the publisher
        self.publisher = rospy.Publisher(publisher_name, Float64MultiArray, queue_size = 1)

    def publish(self, float_array, id=0):
        # use id to identify which serial number
        data_struct = Float64MultiArray()
        data_struct.data = float_array
        data_struct.layout.data_offset = id
        self.publisher.publish(data_struct)

class JointStatePublisher(object):
    def __init__(self, publisher_name):
        # Initializing the publisher
        self.publisher = rospy.Publisher(publisher_name, JointState, queue_size = 1)

    def publish(self, position):
        data_struct = JointState()
        data_struct.position = position
        self.publisher.publish(data_struct)

class ImagePublisher(object):
    def __init__(self, publisher_name, color_image = False):
        # Initializing the publisher
        self.publisher = rospy.Publisher(publisher_name, Image, queue_size = 1)
        
        # Initializing the cv bridge
        self.bridge = CvBridge()

        # Image type
        self.color_image = color_image

    def publish(self, image):
        try:
            if self.color_image:
                image = self.bridge.cv2_to_imgmsg(image, "bgr8")
            else:
                image = self.bridge.cv2_to_imgmsg(image)
        except CvBridgeError as e:
            print(e)

        self.publisher.publish(image)


class BoolPublisher(object):
    def __init__(self, publisher_name):
        # Initializing the publisher
        self.publisher = rospy.Publisher(publisher_name, Bool, queue_size = 1)

    def publish(self, bool):
        self.publisher.publish(Bool(bool))

class TactileSubscriber(object):
    def __init__(self, tactile_num):
        try:
            rospy.init_node(f'tactile_subscriber_{tactile_num}', disable_signals = True)
        except:
            pass
        
        self.tactile_data = None
        # Initializing the subscriber
        self.subscriber = rospy.Subscriber('/tactile_{}/raw_data'.format(tactile_num), Float64MultiArray, self._callback_tactile_raw_data, queue_size = 1)

    def _callback_tactile_raw_data(self, tactile_data):
        self.tactile_data = tactile_data.data

    def get_data(self):
        return self.tactile_data

class ImageSubscriber(object):
    def __init__(self, subscriber_name, node_name, color = True):
        try:
            rospy.init_node('{}'.format(node_name), disable_signals = True)
        except:
            pass

        self.color_image = color

        self.image = None
        self.bridge = CvBridge()
        rospy.Subscriber(subscriber_name, Image, self._callback_image, queue_size = 1)

    def _callback_image(self, image):
        try:
            if self.color_image:
                self.image = self.bridge.imgmsg_to_cv2(image, "bgr8")
            else:
                self.image = self.bridge.imgmsg_to_cv2(image, "passthrough")
        except CvBridgeError as e:
            print(e)

    def get_image(self):
        return self.image