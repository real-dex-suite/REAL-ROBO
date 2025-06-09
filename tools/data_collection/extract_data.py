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
from holodex.tactile_data.tactile_image import TactileImage
from termcolor import cprint

import warnings

# temp fix for the warning
warnings.filterwarnings(
    "ignore",
    message="Link .* is of type 'fixed' but set as active in the active_links_mask.*",
    category=UserWarning
)

@hydra.main(version_base = '1.2', config_path='../../configs', config_name='demo_extract')
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
        cprint("No data available.", "red")

    cprint("\n***************************************************************", "green")
    assert configs['hand_distance_unit']=='cm' and configs['arm_distance_unit']=='cm', "Only cm is supported for distance units."
    print(f"Using hand-min-action distance to filter data: {configs['hand_min_action_distance']} {configs['hand_distance_unit']}")
    print(f"Using arm-min-action distance to filter data: {configs['hand_min_action_distance']} {configs['arm_distance_unit']}")
    cprint("***************************************************************", "green")
    
    cprint("\n***************************************************************", "green")
    cprint("     Starting Filtering and Extraction of Demonstrations!", "green", attrs=['bold'])
    cprint("***************************************************************", "green")
         
    if configs.sample:
        cprint("\nFiltering demonstrations!", "green", attrs=['bold'])
        # If we don't want to filter the data by hand and arm distance, we can set
        # the hand_min_action_distance and arm_min_action_distance to 0 in
        # configs/demo_extract.yaml.
        hand_delta = configs['hand_min_action_distance']
        arm_delta = configs['arm_min_action_distance']
        tactile_delta = configs['tactile_min_force_distance']
        data_filter = FilterData(data_path = configs.storage_path, hand_delta = hand_delta, arm_delta = arm_delta, tactile_delta = tactile_delta, play_data = configs.play_data, last_frame_save_number = configs.last_frame_save_number)     
        data_filter.filter(configs.filter_path)

    if configs.tactiles:
        cprint("\nExtracting tactiles!", "green", attrs=['bold'])
        extractor = TactileExtractor(configs.filter_path, extract_tactile_types=configs.tactile_data_types)
        tactiles_path = os.path.join(configs.target_path, 'tactiles')
        make_dir(tactiles_path)
        extractor.extract(tactiles_path)
        
        # TODO:t-dex
        if configs.tactile_image:
            cprint("\nExtracting tactile image representations!", "green", attrs=['bold'])
            tactile_image = TactileImage()
            if tactile_image:
                tactile_image.get_tactile_img(configs.tactile_img_representation, configs.target_path+"/tactiles", configs.target_path+"/tactiles")
            else:
                cprint("Failed to initialize TactileImage object.", "red")
        else:
            cprint("Tactile image extraction is disabled.", "yellow")
        
    if configs.color_images:
        cprint("\nExtracting color images!", "green", attrs=['bold'])
        extractor = ColorImageExtractor(configs.filter_path, num_cams = configs.num_cams, image_size = configs.image_parameters.image_size, crop_sizes = configs.image_parameters.crop_sizes, crop_image=configs.crop_image)
        images_path = os.path.join(configs.target_path, 'images')
        make_dir(images_path)
        extractor.extract(images_path)

    if configs.depth_images:
        cprint("\nExtracting depth images!", "green", attrs=['bold'])
        extractor = DepthImageExtractor(configs.filter_path, num_cams = configs.num_cams, image_size = configs.image_parameters.image_size, crop_sizes = configs.image_parameters.crop_sizes, extract_depth_types=configs.depth_data_types)
        images_path = os.path.join(configs.target_path, 'images')
        make_dir(images_path)
        extractor.extract(images_path)

    if configs.states:
        cprint("\nExtracting states!", "green", attrs=['bold'])
        extractor = StateExtractor(configs.filter_path, configs.state_data_types)
        states_path = os.path.join(configs.target_path, 'states')
        make_dir(states_path)
        extractor.extract(states_path)

    if configs.actions:
        cprint("\nExtracting actions!", "green", attrs=['bold'])
        extractor = ActionExtractor(configs.filter_path, configs.action_data_types)
        actions_path = os.path.join(configs.target_path, 'actions')
        make_dir(actions_path)
        extractor.extract(actions_path)

    cprint("\n***************************************************************", "green")
    cprint("     Filtering and Extraction of Demonstrations Completed!", "green", attrs=['bold'])
    cprint("***************************************************************", "green")

if __name__ == '__main__':
    main()