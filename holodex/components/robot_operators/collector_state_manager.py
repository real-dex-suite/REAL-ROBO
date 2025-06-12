import rospy
from std_msgs.msg import Bool

class CollectorStateManager:
    def __init__(self):
        # Initialize state variables
        self.stop_move = False
        self.end_robot = False
        self.reset_robot = False

        # Set up ROS subscribers
        self._setup_subscribers()

    def _setup_subscribers(self):
        """Set up all ROS subscribers"""
        topics_callbacks = [
            ("/data_collector/reset_robot", Bool, self._callback_reset_robot),
            ("/data_collector/stop_move", Bool, self._callback_stop_move),
            ("/data_collector/end_robot", Bool, self._callback_end_robot),
        ]
        for topic, msg_type, callback in topics_callbacks:
            rospy.Subscriber(topic, msg_type, callback, queue_size=1)

    def _callback_end_robot(self, msg):
        """Callback function to set end_robot flag"""
        self.end_robot = msg.data

    def _callback_stop_move(self, msg):
        """Callback function to set stop_move flag"""
        self.stop_move = msg.data

    def _callback_reset_robot(self, msg):
        """Callback function to reset robot position"""
        self.reset_robot = msg.data