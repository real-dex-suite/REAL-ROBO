import cv2
import pyrealsense2 as rs
import numpy as np

def initialize_l515(serial_number):
    # Create a RealSense context object
    context = rs.context()
    
    # Get the list of connected devices
    devices = context.query_devices()
    
    # Check if the device with the specified serial number is connected
    for device in devices:
        if device.get_info(rs.camera_info.serial_number) == serial_number:
            print(f"Device {serial_number} found.")
            return device
    
    raise RuntimeError(f"Device with serial number {serial_number} not found.")

def main():
    # Replace with your L515 camera serial number
    CAMERA_SERIAL_NUMBER = 'f1231617'  # Example serial number for L515

    # Initialize RealSense pipeline and configuration
    pipeline = rs.pipeline()
    config = rs.config()

    # Initialize the L515 camera
    try:
        device = initialize_l515(CAMERA_SERIAL_NUMBER)
        config.enable_device(CAMERA_SERIAL_NUMBER)
    except RuntimeError as e:
        print(e)
        return

    # Configure the pipeline to stream color and depth images
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # Start the pipeline with error handling
    try:
        pipeline.start(config)
    except Exception as e:
        print(f"Failed to start the pipeline: {e}")
        return

    try:
        while True:
            # Wait for a new set of frames
            frames = pipeline.wait_for_frames()
            
            # Get color and depth frames
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                print("Error: Failed to get frames.")
                continue
            
            # Convert the images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Apply a color map to depth image for better visualization
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            # Display the images using OpenCV
            cv2.imshow('Color Image', color_image)
            cv2.imshow('Depth Image', depth_colormap)
            
            # Exit the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Stop the pipeline
        pipeline.stop()
        # Close all OpenCV windows
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


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
