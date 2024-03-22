import os
import hydra
from holodex.data import (
    FilterData, 
    TactileExtractor,
    ColorImageExtractor, 
    DepthImageExtractor, 
    StateExtractor, 
    ActionExtractor, 
)

from holodex.utils.files import *


@hydra.main(version_base = '1.2', config_path='configs', config_name='demo_extract')
def main(configs):
    if not configs.ssl_data:
        configs.target_path = os.path.join(configs.target_path, 'filtered') # When we sample and extract min_action data 
    else:
        configs.filter_path = configs.storage_path # Use the raw data as the filtered data
        configs.target_path = os.path.join(configs.target_path, 'ssl') # When we extract every datapoint for training the encoder

        # We extract only the image data
        configs.sample = False
        configs.states = False
        configs.actions = False

    make_dir(configs.target_path)
    make_dir(configs.filter_path)
    
    if not os.path.exists(configs.storage_path):
        print("No data available.")

    print(f"\nUsing hand-min-action distance to filter data: {configs['hand_min_action_distance']} {configs['hand_distance_unit']}")
    print(f"\nUsing arm-min-action distance to filter data: {configs['hand_min_action_distance']} {configs['arm_distance_unit']}")

    print("***************************************************************")
    print("     Starting Filtering and Extraction of Demonstrations!")
    print("***************************************************************")
         
    if configs.sample:
        print("\nFiltering demonstrations!")
        # If we don't want to filter the data by hand and arm distance, we can set
        # the hand_min_action_distance and arm_min_action_distance to 0 in
        # configs/demo_extract.yaml.
        hand_delta = configs['hand_min_action_distance']
        arm_delta = configs['arm_min_action_distance']
        data_filter = FilterData(data_path = configs.storage_path, hand_delta = hand_delta, arm_delta = arm_delta)     
        data_filter.filter(configs.filter_path)

    if configs.tactiles:
        print("\nExtracting tactiles!")
        extractor = TactileExtractor(configs.filter_path, extract_tactile_types=configs.tactile_data_types)
        tactiles_path = os.path.join(configs.target_path, 'tactiles')
        make_dir(tactiles_path)
        extractor.extract(tactiles_path)

    if configs.color_images:
        print("\nExtracting color images!")
        extractor = ColorImageExtractor(configs.filter_path, num_cams = configs.num_cams, image_size = configs.image_parameters.image_size, crop_sizes = configs.image_parameters.crop_sizes)
        images_path = os.path.join(configs.target_path, 'images')
        make_dir(images_path)
        extractor.extract(images_path)

    if configs.depth_images:
        print("\nExtracting depth images!")
        extractor = DepthImageExtractor(configs.filter_path, num_cams = configs.num_cams, image_size = configs.image_parameters.image_size, crop_sizes = configs.image_parameters.crop_sizes)
        images_path = os.path.join(configs.target_path, 'images')
        make_dir(images_path)
        extractor.extract(images_path)

    if configs.states:
        print("\nExtracting states!")
        extractor = StateExtractor(configs.filter_path, configs.state_data_types)
        states_path = os.path.join(configs.target_path, 'states')
        make_dir(states_path)
        extractor.extract(states_path)

    if configs.actions:
        print("\nExtracting actions!")
        extractor = ActionExtractor(configs.filter_path, configs.action_data_types)
        actions_path = os.path.join(configs.target_path, 'actions')
        make_dir(actions_path)
        extractor.extract(actions_path)

if __name__ == '__main__':
    main()