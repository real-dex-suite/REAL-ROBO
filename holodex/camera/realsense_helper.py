# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# examples/python/reconstruction_system/sensors/realsense_helper.py

# pyrealsense2 is required.
# Please see instructions in https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python
import pyrealsense2 as rs
from ipdb import set_trace

def get_profiles(serial_number=None):
    ctx = rs.context()
    devices = ctx.query_devices()

    # Initialize lists to store profiles
    color_profiles = []
    depth_profiles = []

    for device in devices:
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)

        # If serial_number is provided, filter devices by serial number
        if serial_number and serial != serial_number:
            continue

        # print(f'Sensor: {name}, Serial: {serial}')
        # print('Supported video formats:')

        for sensor in device.query_sensors():
            for stream_profile in sensor.get_stream_profiles():
                stream_type = str(stream_profile.stream_type())

                if stream_type in ['stream.color', 'stream.depth']:
                    v_profile = stream_profile.as_video_stream_profile()
                    fmt = stream_profile.format()
                    w, h = v_profile.width(), v_profile.height()
                    fps = v_profile.fps()

                    video_type = stream_type.split('.')[-1]
                    print(f'  {video_type}: width={w}, height={h}, fps={fps}, fmt={fmt}')

                    if video_type == 'color':
                        color_profiles.append((w, h, fps, fmt))
                    else:
                        depth_profiles.append((w, h, fps, fmt))

    return color_profiles, depth_profiles


color_profiles, depth_profiles = get_profiles("f1230963")
# # # print and seperate line for each profile