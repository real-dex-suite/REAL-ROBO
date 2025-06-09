import hydra
import time
from holodex.processes import get_tactile_stream_processes

@hydra.main(version_base = '1.2', config_path='configs', config_name='teleop')
def main(configs):    
    # Obtaining all the robot streams
    tactile_processes = get_tactile_stream_processes(configs)
    # tactile_visualizer_process = get_tactile_visualizer_process()

    # Starting all the processes
    # tactile processes
    for process in tactile_processes:
        process.start()
        time.sleep(2)

    # Joining all the processes
    for process in tactile_processes:
        process.join()
 
if __name__ == '__main__':
    main()