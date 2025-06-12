import hydra
import time
from holodex.processes import get_camera_stream_processes, get_tactile_stream_processes, get_detector_processes, get_teleop_process, get_tactile_visualizer_process
import multiprocessing

@hydra.main(version_base = '1.2', config_path='../../configs', config_name='teleop_real_franka_pico')
def main(configs):    
    multiprocessing.set_start_method('spawn')
    
    # Obtaining all the robot streams
    robot_camera_processes, robot_camera_stream_processes = get_camera_stream_processes(configs)
    tactile_processes = get_tactile_stream_processes(configs)
    detection_process, keypoint_transform_processes, plotter_processes = get_detector_processes(configs)
    teleop_process = get_teleop_process(configs)
    tactile_visualizer_process = get_tactile_visualizer_process(configs)
    
    # Starting all the processes
    for process in tactile_processes:
        process.start()
        time.sleep(2)

    for process in robot_camera_processes:
        process.start()
        time.sleep(2)    

    if detection_process is not None:
        detection_process.start()

    for process in keypoint_transform_processes:
        process.start()

    for process in plotter_processes:
        process.start()
    
    if tactile_visualizer_process is not None:
        tactile_visualizer_process.start()

    if teleop_process is not None:
        # Teleop process
        time.sleep(2)
        teleop_process.start()

    # Joining all the processes
    for process in tactile_processes:
        process.join()
        
    for process in robot_camera_processes:
        process.join()

    for process in robot_camera_stream_processes:
        process.join()
        
    if detection_process is not None:
        detection_process.join()

    for process in keypoint_transform_processes:
        process.join()

    for process in plotter_processes:
        process.join()
        
    if tactile_visualizer_process is not None:
        tactile_visualizer_process.join()
    
    if teleop_process is not None:
        teleop_process.join()


 
if __name__ == '__main__':
    main()