import torch
import random
import torchvision.transforms as T
import torch.nn.functional as F
from copy import deepcopy as copy
from PIL import Image
import numpy as np
import os
import cv2

class TactileImage:
    def __init__(self, tactile_image_size=224, shuffle_type=None):
        self.shuffle_type = shuffle_type
        self.size = tactile_image_size

        # TODO 1: check this range of tactile data
        self.transform = T.Compose(
            [
                T.Resize((224, 224)),
                # T.Lambda(lambda x: x.float()),  # convert to float tensor
                T.Lambda(tactile_clamp_transform),  # clamp the tactile data
                T.Lambda(tactile_scale_transform),  # normalize to [0, 1]
            ]
        )

        # #TODO 2: check this range of tactile data
        # self.transform = T.Compose([
        #     T.Resize((224, 224)),
        #     T.Lambda(lambda x: x.float()),  # convert to float tensor
        #     T.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))  # normalize to [0, 1]
        # ])


    def get_stacked_tactile_image(self, tactile_values):
        tactile_image = torch.FloatTensor(tactile_values)
        # Reshape the tactile image to (8, 5, 3, 3)
        tactile_image = tactile_image.view(8, 5, 3, 3)
        # Permute the dimensions to (8, 3, 5, 3)
        tactile_image = torch.permute(tactile_image, (0, 3, 1, 2))
        # Reshape the tactile image to (24, 5, 3)
        tactile_image = tactile_image.reshape(-1, 5, 3)

        return self.transform(tactile_image)

    def get_single_tactile_image(self, tactile_value):
        """Get the single tactile image.
        Args:
            tactile_value: A tensor of shape (360,).
        Returns:
            A tensor of shape (15, 15, 3).
        """

        if not isinstance(tactile_value, torch.Tensor):
            tactile_value = torch.tensor(tactile_value)

        tactile_image = tactile_value.view(8, 15, 3)

        # convert the tactile image to a PIL image for each sensor
        result = []
        for i in range(8):
            sensor_image = tactile_image[i]  # Shape (15, 3)

            sensor_image = self.transform_data(sensor_image)  # Shape (5, 3, 3)
            sensor_image = torch.permute(
                sensor_image, (2, 0, 1)
            )  # Shape (3, 5, 3) -> [[x],[y],[z]]

            padded_tensor = F.pad(
                sensor_image, (1, 1, 0, 0, 0, 0), mode="constant", value=0
            )
            processed_image = T.Resize((224, 224))(padded_tensor)

            transformed_image = self.transform(processed_image)
            result.append(transformed_image)

            # pil_image = T.ToPILImage()(transformed_image)  # save the image
            # image_path = f"sensor_image_{i + 1}.PNG"
            # pil_image.save(image_path)
        # return 8 sensor images
        return result

        # return transformed_image #TODO: integrate this with dataloader
        
    def get_whole_hand_tactile_image(
        self, tactile_value: torch.Tensor, shuffle_type: str, padding: bool = False # False
    ) -> torch.Tensor:
        """Get the whole hand tactile image.
        Args:
            tactile_value: A tensor of shape (360,).
            shuffle_type: A string indicating the shuffle type.
            padding: A boolean indicating whether to pad the tactile image for each sensor.
        Returns:
            A transformed image of shape (3, 224, 224).
        """
        if not isinstance(tactile_value, torch.Tensor):
            tactile_value = torch.tensor(tactile_value)

        tactile_image = tactile_value.view(8, 15, 3)
        # tactile_image = torch.flip(tactile_image, dims=[1])
        #TODO: check the order of the sensor to make sure it is correct in the real world
        
        # If padding is True, pad the tactile image to (5, 5, 3)
        if padding:
            padded_sensor_images = []
            for i in range(8):
                sensor_image = tactile_image[i]  # Shape (15, 3)
                sensor_image = sensor_image.view(5, 3, 3)
                # (5, 3, 3) to (5, 5, 3)
                padded_tensor = F.pad(
                    sensor_image, (0, 0, 1, 1, 0, 0), mode="constant", value=0
                )

            padded_sensor_images.append(padded_tensor)
            padded_tensor = torch.stack(padded_sensor_images)  # shape (8, 5, 5, 3)
            tactile_image = F.pad(
                padded_tensor, (0, 0, 0, 0, 0, 0, 0, 1), mode="constant", value=0
            )
        
        else:
            tactile_image = tactile_image.view(8, 5, 3, 3)
            # (8, 5, 3, 3) to (9, 5, 3, 3)
            tactile_image = F.pad(
                tactile_image, (0, 0, 0, 0, 0, 0, 0, 1), mode="constant", value=0
            )
            
            # tactile_image = self.reassign_data_point(tactile_image)
        # pad_idx = list(range(9)) # [0, 1, 2, 3, 4, 5, 6, 7, 8]
        pad_idx = [2, 3, 0, 4, 5, 1, 6, 7, 8]

        if shuffle_type == "pad":
            random.seed(10)
            random.shuffle(pad_idx)

        tactile_image = torch.cat(
            [
                torch.cat([tactile_image[pad_idx[i * 3 + j]] for j in range(3)], dim=0)
                for i in range(3)
            ],
            dim=1,
        )

        # print(tactile_image)

        if self.shuffle_type == "whole":
            copy_tactile_image = copy(tactile_image)
            sensor_idx = list(range(9 * 25))
            random.seed(10)
            random.shuffle(sensor_idx)
            for i in range(9):
                for j in range(5):
                    for k in range(5):
                        rand_id = sensor_idx[i * 25 + j * 5 + k]
                        rand_i = int(rand_id / 25)
                        rand_j = int((rand_id % 25) / 5)
                        rand_k = int((rand_id % 25) % 5)
                        tactile_image[i, j, k, :] = copy_tactile_image[
                            rand_i, rand_j, rand_k, :
                        ]

        tactile_image = torch.permute(tactile_image, (2, 0, 1))

        return self.transform(tactile_image) 
    # TODO: refactor this function and integrate with extract_data.py

    # why not just use the raw_data becasue we need to process the tensor data
    def get_tactile_img(self, repre_type: str, data_path: torch.Tensor, save_dir: str):
        """
        Extract and save tactile images to the corresponding directories.

        Args:
            repre_type: A string indicating the representation type.
            raw_data: A tensor of shape (n, 360) containing the raw tactile data.
            save_dir: A string indicating the base directory to save the generated images.
        """
        demo_list = sorted(
            os.listdir(data_path),
            key=lambda f: int("".join(filter(str.isdigit, f))),
        )

        for file_name in demo_list:
            print(file_name)
            if file_name.endswith(".pth"):
                folder_name = os.path.splitext(file_name)[0]
                folder_path = os.path.join(save_dir, folder_name)
                os.makedirs(folder_path, exist_ok=True)

                file_path = os.path.join(data_path, file_name)
                data = torch.load(file_path)
                data = data["raw_data"]

                if data is not None:
                    for i in range(data.size(0)):
                        tactile_value = data[i]

                        if repre_type == "none":
                            tactile_image = self.get_whole_hand_tactile_image(
                                tactile_value, "none"
                            )
                            os.makedirs(folder_path, exist_ok=True)
                            save_path = os.path.join(
                                folder_path, f"tactile_image_{i+1}.png"
                            )
                            pil_image = T.ToPILImage()(tactile_image)
                            pil_image.save(save_path)
                        
                        elif repre_type == "whole":
                            tactile_image = self.get_whole_hand_tactile_image(
                                tactile_value, "whole"
                            )
                            os.makedirs(folder_path, exist_ok=True)
                            save_path = os.path.join(
                                folder_path, f"whole_hand_tactile_image_{i+1}.png"
                            )
                            pil_image = T.ToPILImage()(tactile_image)
                            pil_image.save(save_path)

                        elif repre_type == "pad":
                            tactile_image = self.get_whole_hand_tactile_image(
                                tactile_value, "pad"
                            )
                            os.makedirs(folder_path, exist_ok=True)
                            save_path = os.path.join(
                                folder_path, f"pad_tactile_image_{i+1}.png"
                            )
                            pil_image = T.ToPILImage()(tactile_image)
                            pil_image.save(save_path)

                        elif repre_type == "single":
                            # this is kind different, we need to create one more folder for each step, becasue each step has 8 sensors
                            single_save_dir = os.path.join(
                                folder_path, f"single_tactile_image_{i+1}"
                            )
                            os.makedirs(single_save_dir, exist_ok=True)
                            # get the single tactile images
                            tactile_image = self.get_single_tactile_image(tactile_value)
                            # save the single tactile images
                            for j in range(8):
                                save_path = os.path.join(
                                    single_save_dir, f"sensor_image_{j+1}.png"
                                )
                                pil_image = T.ToPILImage()(tactile_image)
                                pil_image.save(save_path)
                        else:
                            raise ValueError("Invalid representation type.")


def tactile_scale_transform(image):
    print(TACTILE_PLAY_DATA_CLAMP_MIN, TACTILE_PLAY_DATA_CLAMP_MAX)
    image = (image - TACTILE_PLAY_DATA_CLAMP_MIN) / (
        TACTILE_PLAY_DATA_CLAMP_MAX - TACTILE_PLAY_DATA_CLAMP_MIN
    )
    return image


def tactile_clamp_transform(image):
    image = torch.clamp(
        image, min=TACTILE_PLAY_DATA_CLAMP_MIN, max=TACTILE_PLAY_DATA_CLAMP_MAX
    )
    return image


TACTILE_PLAY_DATA_CLAMP_MIN = -50
TACTILE_PLAY_DATA_CLAMP_MAX = 50
tactile_image_size = 224


if __name__ == "__main__":
    tactile_image = TactileImage()
    tactile_data = torch.randn(360)
    tactile_image = tactile_image.get_whole_hand_tactile_image(tactile_data, "whole", padding=True)
    cv2.imshow("tactile_image", tactile_image.cpu().numpy()*255)