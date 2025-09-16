
## Configuration and Usage

### Configuration
Before using the Real-Robo Control framework, ensure that you have properly configured the necessary settings. The configuration files are located in the `configs` directory. These files allow you to customize various parameters such as robot control settings, data collection preferences, sensors,and more.

1. **Accessing Configuration Files:**
   - Go to the `configs` directory at the project's root.
   - Open the configuration file that matches your specific requirements.

#### Example 1: HAMER Configuration

The `hamer.yaml` file located in `configs/tracker/` is essential for setting up the HAMER tracker. HAMER is a 3D hand reconstruction model used for tracking hand keypoints. For more details, refer to the [HAMER repository](https://github.com/hwfan/hamer).

**Key Parameters:**
- **`type`**: Specifies the tracker type. It should be set to `HAMER`.
- **`visualize_graphs`**: A boolean option that controls the visualization of 3D and 2D plots during operation. Set this to `true` to enable visualization.

Additionally, modify the `configs/robot_camera_franka_dexhand.yaml` file to set the hand.

We are now not testing the hamer tracker with franka control. Please modify the `holodex/components/robot_operators/hamer.py`. refer to the `holodex/components/robot_operators/pico.py` for more details.

#### Example 2: Pico Configuration(TODO)