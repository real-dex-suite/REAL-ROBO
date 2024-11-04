import cv2
import numpy as np
import os
from holodex.utils.files import *

class ConcatRGBTactile:
    def __init__(
        self,
        img_paths,
        tactile_path,
        prefix1,
        prefix2,
        num_images,
        interval,
        alpha,
        target_size,
        concat_image_save_path,
    ) -> None:
        self.img_paths = img_paths
        self.tactile_path = tactile_path
        self.prefix1 = prefix1
        self.prefix2 = prefix2
        self.num_images = num_images
        self.interval = interval
        self.alpha = alpha
        self.target_size = target_size
        self.concat_image_save_path = concat_image_save_path

    def load_images(self, folder_path, prefix):
        images = []
        for i in range(1, self.num_images, self.interval):
            filename = f"{prefix}{i}.PNG"
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                image = cv2.imread(file_path)
                if image is not None:
                    images.append(image)
                else:
                    print(f"Failed to load image: {file_path}")
            else:
                print(f"File does not exist: {file_path}")
        return images

    def concat_rgb_tactile(self, images1, images2):
        assert len(images1[0]) == len(images2)
        # if not images1 or not images2:
        #     raise ValueError("One or both image lists are empty.")

        num_images = min(len(images1[0]), len(images2))

        original_idx = []
        # read the original idx name of t

        for i in range(num_images):

            imgs = []
            for img in images1:
                imgs.append(cv2.resize(img[i], self.target_size))

            img2 = cv2.resize(images2[i], self.target_size)
            concat_img = np.hstack((*imgs, img2))
            # TODO write with original idx
            cv2.imwrite(os.path.join(self.concat_image_save_path, f"Image-Tactile-{i+1}.PNG"), concat_img)

    def run_comparison(self):
        imgs = []
        for img_path in self.img_paths:
            imgs.append(self.load_images(img_path, self.prefix1))
        tactile_img = self.load_images(self.tactile_path, self.prefix2)

        if len(imgs)>0 and tactile_img:
            diff_grid = self.concat_rgb_tactile(imgs, tactile_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Failed to load images for concat.")


def main():
    extracted_data_path = "/home/agibot/Projects/Real-Robo/expert_dataset/temp_dataset/extracted_data/filtered"
    ori_data_path = "/home/agibot/Projects/Real-Robo/expert_dataset/temp_dataset/recorded_data"
    image_names = ["camera_1_color_image", "camera_2_color_image"]

    for demonstration in os.listdir(ori_data_path):
        concat_image_save_path = os.path.join("/home/agibot/Projects/Real-Robo/expert_dataset/temp_dataset/concat_image", demonstration)
        make_dir(concat_image_save_path)

        img_paths = []
        for image_name in image_names:
            img_paths.append(os.path.join(extracted_data_path, "images", demonstration, image_name))

        tactile_path = os.path.join(extracted_data_path, "tactiles", demonstration)
        num_image = len(os.listdir(tactile_path))
        comparer = ConcatRGBTactile(
            img_paths,
            tactile_path,
            prefix1="",
            prefix2="",
            num_images=num_image,
            interval=1,
            alpha=0.7,
            target_size=(512, 512),
            concat_image_save_path=concat_image_save_path,
        )
        comparer.run_comparison()


if __name__ == "__main__":
    main()
