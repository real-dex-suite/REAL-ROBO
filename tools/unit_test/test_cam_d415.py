import cv2
import pyrealsense2 as rs
import numpy as np

def get_device_by_serial(serial_number):
    # Create a RealSense context object
    context = rs.context()
    
    # Get the list of connected devices
    devices = context.query_devices()
    
    # Print the list of connected devices for debugging
    print("Connected devices:")
    for device in devices:
        print(f"Serial Number: {device.get_info(rs.camera_info.serial_number)}, Name: {device.get_info(rs.camera_info.name)}")
    
    # Search for the device with the specified serial number
    for device in devices:
        if device.get_info(rs.camera_info.serial_number) == serial_number:
            return device
    raise RuntimeError(f"Device with serial number {serial_number} not found.")


def main():
    CAMERA_SERIAL_NUMBER = '125322060991'
    device = get_device_by_serial(CAMERA_SERIAL_NUMBER)
    print(device.get_info(rs.camera_info.name))

if __name__ == "__main__":
    main()