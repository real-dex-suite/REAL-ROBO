from holodex.components.robot_operators.robot import RobotController
from geometry_msgs.msg import PoseStamped
import rospy

if __name__ == "__main__":
    robot = RobotController(teleop=False,
                            simulator=None,
                            gripper="ctek",
                            gripper_init_state="close")

    pub = rospy.Publisher('/end_effector_pose', PoseStamped, queue_size=10)

    def timer_callback(event):
        tcp_pose = robot.get_arm_tcp_position()
        print(tcp_pose)
        
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "world"
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.pose.position.x = tcp_pose[0]
        pose_msg.pose.position.y = tcp_pose[1]
        pose_msg.pose.position.z = tcp_pose[2]
        pose_msg.pose.orientation.x = tcp_pose[3]
        pose_msg.pose.orientation.y = tcp_pose[4]
        pose_msg.pose.orientation.z = tcp_pose[5]
        pose_msg.pose.orientation.w = tcp_pose[6]

        pub.publish(pose_msg)

    timer = rospy.Timer(rospy.Duration(0.1), timer_callback) # 10Hz
    rospy.spin()


