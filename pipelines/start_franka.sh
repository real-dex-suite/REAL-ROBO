unset ROS_DISTRO
source /opt/ros/noetic/local_setup.bash
cd dependencies/frankapy/
source catkin_ws/devel/setup.bash
bash bash_scripts/start_control_pc.sh