import os
import hydra
from holodex.data.collector_franka import FrankaDataCollector
from holodex.utils.files import make_dir

@hydra.main(version_base='1.2', config_path='../../configs', config_name='demo_record')
def main(configs):
    demo_path = os.path.join(os.getcwd(), configs['storage_path'])
    make_dir(demo_path)

    collector = FrankaDataCollector(
        num_cams=configs['num_cams'], 
        keyboard_control=configs['keyboard_control'], 
        storage_path=demo_path,  # Corrected argument name
    )

    print(f'Recording data for demonstration: {configs.demo_num} with offset: {configs.offset}.')
    print(f'Storing the demonstration data in {demo_path}')
    collector.extract(offset=configs.offset)

if __name__ == '__main__':
    main()