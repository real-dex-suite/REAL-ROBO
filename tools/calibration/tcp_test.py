import argparse
from frankapy import FrankaArm
import rospy
from geometry_msgs.msg import PoseStamped

if __name__ == "__main__":
    pub = rospy.Publisher("/end_effector_pose", PoseStamped, queue_size=10)

    fa = FrankaArm(with_gripper=False)

    def timer_callback(event):
        tcp_pose = fa.get_pose()
        print("t", tcp_pose.translation)
        print("q", tcp_pose.quaternion)

        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "world"
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.pose.position.x = tcp_pose.translation[0]
        pose_msg.pose.position.y = tcp_pose.translation[1]
        pose_msg.pose.position.z = tcp_pose.translation[2]
        pose_msg.pose.orientation.w = tcp_pose.quaternion[0]
        pose_msg.pose.orientation.x = tcp_pose.quaternion[1]
        pose_msg.pose.orientation.y = tcp_pose.quaternion[2]
        pose_msg.pose.orientation.z = tcp_pose.quaternion[3]

        pub.publish(pose_msg)

    timer = rospy.Timer(rospy.Duration(0.5), timer_callback)  # 10Hz
    rospy.spin()
