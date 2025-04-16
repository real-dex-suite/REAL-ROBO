import hydra
import time
from processes import get_tactile_stream_processes, get_camera_stream_processes, get_detector_processes, get_teleop_process, get_tactile_visualizer_process
import multiprocessing

@hydra.main(version_base = '1.2', config_path='configs', config_name='teleop')
def main(configs):    
    # Obtaining all the robot streams
    # tactile_processes = get_tactile_stream_processes(configs)
    multiprocessing.set_start_method('spawn')
    
    robot_camera_processes, robot_camera_stream_processes = get_camera_stream_processes(configs)
    detection_process, keypoint_transform_processes, plotter_processes = get_detector_processes(configs)
    teleop_process = get_teleop_process(configs)
    # tactile_visualizer_process = get_tactile_visualizer_process()

    # Starting all the processes
    # tactile processes
    # for process in tactile_processes:
    #     process.start()
    #     time.sleep(2)
    for process in robot_camera_processes:
        process.start()
        time.sleep(2)

    for process in robot_camera_stream_processes:
        process.start()

    if configs.tracker.type != 'HAMER': # only for temporal
        # # Detection processes
        detection_process.start()

    for process in keypoint_transform_processes:
        process.start()

    for process in plotter_processes:
        process.start()
    

    # tactile_visualizer_process.start()

    time.sleep(2)
    # Teleop process
    teleop_process.start()

    # Joining all the processes
    # for process in tactile_processes:
    #     process.join()

    for process in robot_camera_processes:
        process.join()

    for process in robot_camera_stream_processes:
        process.join()

    if configs.tracker.type != 'HAMER': # only for temporal
        detection_process.join()

    for process in keypoint_transform_processes:
        process.join()

    for process in plotter_processes:
        process.join()

    # tactile_visualizer_process.join()
    teleop_process.join()


 
if __name__ == '__main__':
    main()