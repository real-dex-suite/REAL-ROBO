import hydra
import rospy
import time
import os

from holodex.components.robot_operators.replay_robot import (
    ReplayController,
    DataProcessor,
)
from holodex.data.replay_extractor import ColorImageExtractor


@hydra.main(version_base="1.2", config_path="configs", config_name="replay")
def main(configs):
    rospy.init_node("replay_node", anonymous=True)
    if configs.replay_robot:
        data_processor = DataProcessor(configs)
        # Load the data: "extracted" or "raw"
        arm_state_data, hand_state_data = data_processor.load_data(
            data_type="extracted"
        )
        print("Data loaded successfully!")
        print("Replaying robot motion...")

        replay_controller = ReplayController(
            configs, hand_state=hand_state_data, arm_state=arm_state_data
        )
        replay_controller.home_robot()
        rospy.sleep(1)

        if configs.replay_arm:
            replay_controller.replay_arm_motion()

        if configs.replay_hand:
            replay_controller.replay_hand_motion()

        if configs.replay_arm_and_hand:
            replay_controller.replay_arm_and_hand_motion()

    print("............................................")
    print("Don't stop now if you want to process the saved images.")

    if configs.extract_replay_images:
        extractor = ColorImageExtractor(configs)
        extractor.extract_images(configs.target_path)

    print("All Done")


if __name__ == "__main__":
    main()
