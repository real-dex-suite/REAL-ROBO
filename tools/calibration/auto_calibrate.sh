#!/bin/bash

# Check if realsense node is running
if pgrep -f "realsense2_camera" > /dev/null; then
    echo "Killing existing realsense2_camera node..."
    rosnode kill /camera/realsense2_camera_manager
    sleep 2
fi

# Launch realsense camera node
echo "Starting realsense camera node..."
roslaunch realsense2_camera rs_camera.launch

# Wait for camera to initialize
sleep 5

# Launch auto calibration script
echo "Starting auto calibration..."
python3 tools/calibration/auto_calibrate.py \
    --marker_id 582 \
    --marker_size 0.078 \
    --image_topic /camera/color/image_raw \
    --camera_info_topic /camera/color/camera_info \
    --rotation_type quaternion

