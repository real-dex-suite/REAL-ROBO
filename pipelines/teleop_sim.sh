unset ROS_DISTRO
source /opt/ros/noetic/local_setup.bash
conda activate real-robo
python tools/teleoperation/teleop.py --config-name=teleop_sim_franka_pico