from multiprocessing import Process
from holodex.components import *
from holodex.viz import Hand2DVisualizer, Hand3DVisualizer, MPImageVisualizer
from holodex.viz.visualizer_3d import OculusLeftHandDirVisualizer
from holodex.camera.realsense_camera import reset_cameras, RealSenseRobotStream
from holodex.camera.camera_streamer import RobotCameraStreamer
from holodex.tactile.paxini_tactile import PaxiniTactileStream
from holodex.tactile_data.viewer import TactileDataViewer
from termcolor import cprint

def notify_process_start(notification_statement):
    cprint("***************************************************************", "green")
    cprint("     {}".format(notification_statement), "green")
    cprint("***************************************************************", "green")

def start_paxini_tactile_stream(serial_port, tactile_num, baudrate):
    notify_process_start("Starting Paxini Tactile Stream Process")
    tactile = PaxiniTactileStream(serial_port, tactile_num, baudrate)
    tactile.stream()

def start_robot_cam_stream(cam_serial_num, robot_cam_num):
    notify_process_start("Starting Robot Camera Stream {} Process".format(robot_cam_num))
    camera = RealSenseRobotStream(cam_serial_num, robot_cam_num)
    camera.stream()

def stream_cam_tcp(cam_num, host, port, cam_rotation_angle):
    notify_process_start("Starting Robot Camera TCP Stream Process")
    camera = RobotCameraStreamer(cam_num, host, port, cam_rotation_angle)
    camera.stream()

def start_mp_detector(options):
    notify_process_start("Starting Mediapipe Detection Process")
    detector = MPHandDetector(options['cam_serial_num'], options['resolution'], options['alpha'])
    detector.stream()

def start_lp_detector(options):
    notify_process_start("Starting Leapmotion Detection Process")
    detector = LPHandDetector()
    detector.stream()

def start_oculus_detector(options):
    notify_process_start("Starting OculusVR Detection Process")
    detector = OculusVRHandDetector(options['host'], options['keypoint_stream_port'])
    detector.stream()

def keypoint_transform(detector_type):
    notify_process_start("Starting Keypoint transformation Process")
    transformer = TransformHandCoords(detector_type)
    transformer.stream()

def plot_2d(detector_type, *args):
    notify_process_start("Starting 2D Hand Plotting Process")
    plotter = Hand2DVisualizer(detector_type, *args)
    plotter.stream()

def plot_3d(detector_type):
    notify_process_start("Starting 3D Hand Plotting Process")
    plotter = Hand3DVisualizer(detector_type)
    plotter.stream()

def plot_oculus_left_hand():
    notify_process_start("Starting Oculus Left Hand Direction Plotting Process")
    plotter = OculusLeftHandDirVisualizer()
    plotter.stream()

def viz_hand_stream(rotation_angle):
    notify_process_start("Starting Mediapipe Hand Prediction Image Stream Process")
    visualizer = MPImageVisualizer(rotation_angle)
    visualizer.stream()

def mp_teleop(teleop_configs):
    notify_process_start("Starting Teleoperation Process")
    teleop = MPDexArmTeleOp()
    teleop.move(teleop_configs['finger_configs'])

def lp_teleop(teleop_configs):
    notify_process_start("Starting Teleoperation Process")
    teleop = LPDexArmTeleOp(simulator=teleop_configs.get("simulator", None),
                            arm_type=teleop_configs.get("arm", "flexiv"))
    teleop.move(teleop_configs['finger_configs'])
    
def vr_teleop(teleop_configs):
    notify_process_start("Starting Teleoperation Process")
    teleop = VRDexArmTeleOp(simulator=teleop_configs.get("simulator", None),
                            arm_type=teleop_configs.get("arm", "jaka"))
    teleop.move(teleop_configs['finger_configs'])

def hamer_teleop(teleop_configs):
    notify_process_start("Starting Teleoperation Process")
    teleop = HamerDexArmTeleOp(simulator=teleop_configs.get("simulator", None),
                               arm_type=teleop_configs.get("arm", "flexiv"))
    teleop.move(teleop_configs['finger_configs'])

def hamer_gripper_teleop(teleop_configs):
    notify_process_start("Starting Teleoperation Process")
    teleop = HamerGripperDexArmTeleOp(simulator=teleop_configs.get("simulator", None), 
                                      gripper=teleop_configs.get("gripper", "panda"),
                                      arm_type=teleop_configs.get("arm", "franka"),
                                      gripper_init_state=teleop_configs.get("gripper_init_state", "open"))
    teleop.move()

def pico_teleop(teleop_configs):
    notify_process_start("Starting Teleoperation Process")
    teleop = PICODexArmTeleOp(simulator=teleop_configs.get("simulator", None), 
                              gripper=teleop_configs.get("gripper", "ctek"),
                              arm_type=teleop_configs.get("arm", "franka"),
                              gripper_init_state=teleop_configs.get("gripper_init_state", "open"))
    teleop.move()

def kb_teleop(teleop_configs):
    notify_process_start("Starting Teleoperation Process")
    teleop = KBArmTeleop(simulator=teleop_configs.get("simulator", None), 
                         gripper=teleop_configs.get("gripper", "panda"),
                         arm_type=teleop_configs.get("arm", "franka"),
                         gripper_init_state=teleop_configs.get("gripper", "open"))
    teleop.move()

def get_tactile_stream_processes(configs):
    tactile_processes = []
    if "tactile" in configs.keys():
        if configs['tactile']['type'] == 'paxini':
            if len(configs['tactile']['serial_port_number']) > 0:
                for idx, serial_port_number in enumerate(configs['tactile']['serial_port_number']):
                    tactile_num = int(serial_port_number[serial_port_number.find("USB")+3])
                    tactile_processes.append(
                        Process(target = start_paxini_tactile_stream, args = (serial_port_number, tactile_num+1, configs['tactile']['baudrate']))
                    )
    return tactile_processes

def get_camera_stream_processes(configs):
    robot_camera_processes = []
    robot_camera_stream_processes = []
    if "robot_cam_serial_numbers" in configs.keys():
        reset_cameras()
        for idx, cam_serial_num in enumerate(configs['robot_cam_serial_numbers']):
            robot_camera_processes.append(
                Process(target = start_robot_cam_stream, args = (cam_serial_num, idx + 1, ))
            )

    if 'tracker' in configs.keys(): # Since we use this for deployment as well
        if configs.tracker.type == 'VR':
            if configs.tracker['stream_robot_cam']:
                robot_camera_stream_processes.append(
                    Process(target = stream_cam_tcp, args = (
                        configs.tracker['stream_camera_num'], 
                        configs.tracker['host'], 
                        configs.tracker['robot_cam_stream_port'],
                        configs.tracker['stream_camera_rotation_angle'],
                    ))
                )

    return robot_camera_processes, robot_camera_stream_processes

def get_detector_processes(teleop_configs):
    if teleop_configs.tracker.type == 'MP':
        detection_process = Process(target = start_mp_detector, args = (teleop_configs.tracker, ))
        keypoint_transform_processes = [Process(target = keypoint_transform, args = ('MP', ))]
        
        plotter_processes = []
        if teleop_configs.tracker['visualize_graphs']:
            plotter_processes.append(Process(target = plot_2d, args = (teleop_configs.tracker.type, )))
            plotter_processes.append(Process(target = plot_3d, args = (teleop_configs.tracker.type, ))),
            
        if teleop_configs.tracker['visualize_pred_stream']:
            plotter_processes.append(Process(target = viz_hand_stream, args = (teleop_configs.tracker['pred_stream_rotation_angle'], )))
    
    elif teleop_configs.tracker.type == 'LP':
        detection_process = Process(target = start_lp_detector, args = (teleop_configs.tracker, teleop_configs.get("arm", "flexiv")))
        keypoint_transform_processes = [Process(target = keypoint_transform, args = ('LP', ))]
        
        plotter_processes = []
        if teleop_configs.tracker['visualize_graphs']:
            plotter_processes.append(Process(target = plot_3d, args = (teleop_configs.tracker.type, ))),

    elif teleop_configs.tracker.type == 'VR':
        detection_process = Process(target = start_oculus_detector, args = (teleop_configs.tracker, teleop_configs.get("arm", "flexiv")))
        keypoint_transform_processes = [
            Process(target = keypoint_transform, args = ('VR_RIGHT', )),
            Process(target = keypoint_transform, args = ('VR_LEFT', ))
        ]

        plotter_processes = []
        if teleop_configs.tracker['visualize_right_graphs']:
            plotter_processes.append(Process(target = plot_2d, args = ('VR_RIGHT', teleop_configs.tracker['host'], teleop_configs.tracker['plot_stream_port'], )))
            plotter_processes.append(Process(target = plot_3d, args = ('VR_RIGHT', )))
            
        if teleop_configs.tracker['visualize_left_graphs']:
            plotter_processes.append(Process(target = plot_oculus_left_hand))
    
    elif teleop_configs.tracker.type == 'HAMER':
        detection_process = None # outside for now
        keypoint_transform_processes = [Process(target = keypoint_transform, args = ('HAMER', teleop_configs.get("arm", "flexiv")))]
        
        plotter_processes = []
        if teleop_configs.tracker['visualize_graphs']:
            plotter_processes.append(Process(target = plot_3d, args = (teleop_configs.tracker.type, ))),
            
    else:
        detection_process = None # outside for now
        keypoint_transform_processes = []
        plotter_processes = []

    return detection_process, keypoint_transform_processes, plotter_processes

def get_teleop_process(teleop_configs):
    if teleop_configs.tracker.type == 'MP': # mediapipe
        teleop_process = Process(target = mp_teleop, args = (teleop_configs, ))
    elif teleop_configs.tracker.type == 'LP': # leapmotion
        teleop_process = Process(target = lp_teleop, args = (teleop_configs, ))
    elif teleop_configs.tracker.type == 'VR': # Oculus VR
        teleop_process = Process(target = vr_teleop, args = (teleop_configs, ))
    elif teleop_configs.tracker.type == 'HAMER': # HAMER detection, now pico VR
        teleop_process = Process(target = hamer_teleop, args = (teleop_configs, ))
    elif teleop_configs.tracker.type == 'HAMER_GRIPPER': # HAMER detection, now pico VR
        teleop_process = Process(target = hamer_gripper_teleop, args = (teleop_configs, ))
    elif teleop_configs.tracker.type == 'PICO': # HAMER detection, now pico VR
        teleop_process = Process(target = pico_teleop, args = (teleop_configs, ))
    elif teleop_configs.tracker.type == 'KB': # Keyboard
        teleop_process = Process(target = kb_teleop, args = (teleop_configs, ), daemon=False)

    return teleop_process

def start_tactile_visualizer():
    notify_process_start("Starting Tactile Visualizer")
    viewer = TactileDataViewer("2D")
    viewer.run()

def get_tactile_visualizer_process(configs):
    tactile_visualizer_process = None
    if "tactile" in configs.keys():
        if configs['tactile']['type'] == 'paxini':
            tactile_visualizer_process = Process(target = start_tactile_visualizer)
    return tactile_visualizer_process