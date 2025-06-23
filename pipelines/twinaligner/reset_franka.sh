task=${1:-"milk"}
unset ROS_DISTRO
source /opt/ros/noetic/local_setup.bash
cd dependencies/frankapy
source catkin_ws/devel/setup.bash
cd ../..
python tools/teleoperation/reset_franka_twinaligner.py --task $task