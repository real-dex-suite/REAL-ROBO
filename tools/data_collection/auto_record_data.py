import os
import hydra
from holodex.data import AutoDataCollector
from holodex.utils.files import make_dir
from datetime import datetime
from termcolor import cprint

@hydra.main(version_base = '1.2', config_path='../../configs', config_name='demo_record')
def main(configs):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    demo_path = os.path.join(os.getcwd(), configs['storage_path'], timestamp)
    make_dir(demo_path)

    cam_serial_numbers = configs.get("robot_cam_serial_numbers", None)
    num_cams = len(cam_serial_numbers) if cam_serial_numbers is not None else 0
    
    tactile_serial_numbers = configs.get("serial_port_number", None)
    num_tactiles = len(tactile_serial_numbers) if tactile_serial_numbers is not None else 0
    
    collector = AutoDataCollector(
        num_tactiles=num_tactiles,
        num_cams=num_cams,
        storage_root = demo_path,
        arm_type = configs["arm"],
        gripper = configs["gripper"],
        
    )

    cprint(f'Collecting {configs.demo_num} demos in {demo_path}', 'yellow')
    collector.extract()

if __name__ == '__main__':
    main()