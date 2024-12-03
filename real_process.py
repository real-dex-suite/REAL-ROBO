import hydra
import time
from processes import get_tactile_stream_processes, get_camera_stream_processes, get_detector_processes, get_teleop_process, get_tactile_visualizer_process

@hydra.main(version_base = '1.2', config_path='configs', config_name='teleop')
def main(configs):    
    # Obtaining all the robot streams
    tactile_processes = get_tactile_stream_processes(configs)
    robot_camera_processes, robot_camera_stream_processes = get_camera_stream_processes(configs)
    # detection_process, keypoint_transform_processes, plotter_processes = get_detector_processes(configs)
    teleop_process = get_teleop_process(configs)
    tactile_visualizer_process = get_tactile_visualizer_process()


    # Starting all the processes
    # tactile processes
    # for process in tactile_processes:
    #     process.start()
    #     time.sleep(2)
    # Camera processes
    for process in robot_camera_processes:
        process.start()
        time.sleep(2)


    # Joining all the processes
    # for process in tactile_processes:
    #     process.join()

    for process in robot_camera_processes:
        process.join()
    

 
if __name__ == '__main__':
    main()