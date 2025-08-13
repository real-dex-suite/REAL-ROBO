import zmq
import rospy
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
import re
from termcolor import cprint

# Modify these parameters to match your setup
IP_ADDRESS = "localhost"
PORT = 5555

DEBUG = True

class ZMQSubscriber:
    def __init__(self, ip_address=IP_ADDRESS, port=PORT):
        self.ip_address = ip_address
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self._setup_connection()

    def _setup_connection(self):
        self.socket.connect(f"tcp://{self.ip_address}:{self.port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        cprint(f"Subscribed to TCP://{self.ip_address}:{self.port}", "green")

    def _process_message(self, message):
        number_pattern = re.compile(r'-?\d+\.\d+|-?\d+')
        numbers = number_pattern.findall(message)
        trigger = numbers[0] if numbers else None
        ee_pose = [float(num) for num in numbers[1:]] if len(numbers) > 1 else []
        return trigger, ee_pose

    def _create_pose_message(self, ee_pose):
        pose_msg = Pose()
        pose_msg.position.x, pose_msg.position.y, pose_msg.position.z = ee_pose[:3]
        pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w = ee_pose[3:7]
        return pose_msg

    def _create_trigger_message(self, trigger):
        return Float64(data=float(trigger) if trigger else 1.0)

    def listen(self):
        pub_ee_pose = rospy.Publisher('vr/ee_pose', Pose, queue_size=10)
        pub_gripper = rospy.Publisher('vr/gripper', Float64, queue_size=10)
        
        rospy.init_node('zmq_listener', anonymous=True)
        while not rospy.is_shutdown():
            try:
                message = self.socket.recv_string()
                trigger, ee_pose = self._process_message(message)
                if DEBUG:
                    rospy.loginfo(f"Gripper: {trigger}, EE Pose: {ee_pose}")
                
                ee_pose_msg = self._create_pose_message(ee_pose)
                trigger_msg = self._create_trigger_message(trigger)
                
                pub_ee_pose.publish(ee_pose_msg)
                pub_gripper.publish(trigger_msg)
            except zmq.ZMQError as e:
                rospy.logerr(f"ZMQ Error receiving message: {str(e)}")
            except Exception as e:
                rospy.logerr(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    cprint("Initializing ZMQ Subscriber...", "cyan")
    subscriber = ZMQSubscriber()
    cprint("ZMQ Subscriber initialized. Starting to listen for messages...", "cyan")
    try:
        subscriber.listen()
    except KeyboardInterrupt:
        cprint("Subscriber stopped by user.", "yellow")
    except Exception as e:
        cprint(f"An unexpected error occurred: {str(e)}", "red")
    finally:
        cprint("Exiting ZMQ Subscriber.", "cyan")