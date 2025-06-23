#!/bin/bash

echo -e "\e[34m============================================"
echo "   Robot-Camera Calibration Script v1.0"
echo "   Authors: Jinzhou Li, Jiyao Zhang"
echo -e "============================================\e[0m"

# Launch ROS camera nodes in a new terminal
echo -e "\e[36mLaunching ROS camera nodes...\e[0m"
gnome-terminal -- bash -c "
echo 'Starting ROS camera nodes...';
source /opt/ros/noetic/setup.bash
roslaunch realsense2_camera rs_camera.launch;
exec bash"

# Wait for ROS nodes to initialize
echo -e "\e[36mWaiting for ROS nodes to initialize...\e[0m"
sleep 2

# open a new terminal to run the rqt_image_view
echo -e "\e[36mLaunching rqt_image_view...\e[0m"
gnome-terminal -- bash -c "
echo 'Starting rqt_image_view...';
source /opt/ros/noetic/setup.bash
rqt_image_view;
exec bash"

