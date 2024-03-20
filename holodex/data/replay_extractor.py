import os
from PIL import Image

def make_dir(path):
    os.makedirs(path, exist_ok=True)

class ColorImageExtractor(object):
    def __init__(self, configs):
        self.data_path = configs.relpay_image_path
        self.num_cams = configs.num_cams
        self.image_size = configs.image_size
        self.crop_sizes = configs.crop_sizes

    def extract_images(self, target_path):
        # end with png
        image_files = [f for f in os.listdir(self.data_path) if f.startswith('replay_image_') and f.endswith('.png')]
        image_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        color_cam_image_paths = []
        for cam_num in range(self.num_cams):
            # camera_1_color_image
            color_cam_image_path = os.path.join(target_path, f'camera_{cam_num + 1}_color_image')
            color_cam_image_paths.append(color_cam_image_path)
            make_dir(color_cam_image_path)

        for image_file in image_files:
            image_path = os.path.join(self.data_path, image_file)
            image = Image.open(image_path)

            for cam_num in range(self.num_cams):
                if self.crop_sizes is not None:
                    cropped_image = image.crop(self.crop_sizes[cam_num])
                else:
                    cropped_image = image

                resized_image = cropped_image.resize((self.image_size, self.image_size), Image.ANTIALIAS)
                save_path = os.path.join(color_cam_image_paths[cam_num], image_file)
                resized_image.save(save_path)

        print(f"\nExtracted images saved in {target_path}")