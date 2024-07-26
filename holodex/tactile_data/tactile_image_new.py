import random
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from copy import deepcopy as copy
import matplotlib.pyplot as plt
import numpy as np


class TactileImageNew:
    def __init__(self, tactile_image_size=(224, 224), shuffle_type=None):
        self.size = tactile_image_size
        self.shuffle_type = shuffle_type
        self.pad_idx = [2, 3, 0, 4, 5, 1, 6, 7, 8]
        self.transform = T.Compose(
            [
                T.Resize(tactile_image_size),
                T.Lambda(tactile_clamp_transform),
                T.Lambda(tactile_scale_transform),
            ]
        )

    def get(self, type, tactile_values):
        return self.get_whole_hand_tactile_image(tactile_values)

    def get_whole_hand_tactile_image(
        self,
        tactile_value: torch.Tensor,
        padding: bool = False,
    ) -> torch.Tensor:
        if not isinstance(tactile_value, torch.Tensor):
            tactile_value = torch.tensor(tactile_value)

        tactile_image = tactile_value.view(8, 15, 3)  # (360) -> (8, 15, 3)

        if padding:
            padded_sensor_images = [
                F.pad(
                    sensor_image.view(5, 3, 3),
                    (0, 0, 1, 1, 0, 0),
                    mode="constant",
                    value=0,
                )
                for sensor_image in tactile_image
            ]
            tactile_image = F.pad(
                torch.stack(padded_sensor_images),
                (0, 0, 0, 0, 0, 0, 0, 1),
                mode="constant",
                value=0,
            )
        else:
            tactile_image = F.pad(
                tactile_image.view(8, 5, 3, 3),
                (0, 0, 0, 0, 0, 0, 0, 1),
                mode="constant",
                value=0,
            )

        tactile_image = torch.cat(
            [
                torch.cat(
                    [tactile_image[self.pad_idx[i * 3 + j]] for j in range(3)], dim=0
                )
                for i in range(3)
            ],
            dim=1,
        )
        tactile_image = torch.permute(tactile_image, (2, 0, 1))
        return self.transform(tactile_image)

def tactile_scale_transform(image):
    image = (image - TACTILE_PLAY_DATA_CLAMP_MIN) / (TACTILE_PLAY_DATA_CLAMP_MAX - TACTILE_PLAY_DATA_CLAMP_MIN)
    return image

def tactile_clamp_transform(image):
    image = torch.clamp(image, min=TACTILE_PLAY_DATA_CLAMP_MIN, max=TACTILE_PLAY_DATA_CLAMP_MAX)

TACTILE_PLAY_DATA_CLAMP_MIN = -60  # -1000
TACTILE_PLAY_DATA_CLAMP_MAX = 60  # 1000