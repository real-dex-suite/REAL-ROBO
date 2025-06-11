unset ROS_DISTRO
source /opt/ros/noetic/local_setup.bash
cd dependencies/frankapy
source catkin_ws/devel/setup.bash
cd ../..
python tools/teleoperation/teleop.py --config-name=teleop_real_franka_pico