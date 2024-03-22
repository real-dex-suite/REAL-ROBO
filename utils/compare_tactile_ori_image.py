import cv2
import numpy as np
import os


class CompareDifference:
    def __init__(
        self,
        img_path,
        tactile_path,
        prefix1,
        prefix2,
        num_images,
        interval,
        alpha,
        target_size,
    ) -> None:
        self.folder1_path = img_path
        self.folder2_path = tactile_path
        self.prefix1 = prefix1
        self.prefix2 = prefix2
        self.num_images = num_images
        self.interval = interval
        self.alpha = alpha
        self.target_size = target_size

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

    def create_diff_grid(self, images1, images2):
        assert len(images1) == len(images2)
        if not images1 or not images2:
            raise ValueError("One or both image lists are empty.")

        num_images = min(len(images1), len(images2))
        grid_size = int(np.ceil(np.sqrt(num_images)))
        grid_img = np.zeros(
            (self.target_size[0] * grid_size, self.target_size[1] * 2 * grid_size, 3),
            dtype=np.uint8,
        )

        for i in range(num_images):
            img1 = cv2.resize(images1[i], self.target_size)
            img2 = cv2.resize(images2[i], self.target_size)

            row, col = divmod(i, grid_size)
            grid_img[
                row * self.target_size[0] : (row + 1) * self.target_size[0],
                col * self.target_size[1] * 2 : (col + 1) * self.target_size[1] * 2,
                :,
            ] = np.hstack((img1, img2))

        return grid_img

    def run_comparison(self):
        images1 = self.load_images(self.folder1_path, self.prefix1)
        images2 = self.load_images(self.folder2_path, self.prefix2)

        if images1 and images2:
            diff_grid = self.create_diff_grid(images1, images2)
            # save the difference grid
            cv2.imwrite("tactileori_grid.png", diff_grid)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Failed to load images for comparison.")


def main():
    img_path = "/home/agibot/Projects/Real-Robo/expert_dataset/extracted_data/filtered/images/demonstration_1/camera_1_color_image"
    tactile_path = "/home/agibot/Projects/Real-Robo/expert_dataset/extracted_data/filtered/tactiles/demonstration_1"
    comparer = CompareDifference(
        img_path,
        tactile_path,
        prefix1="",
        prefix2="",
        num_images=200,
        interval=1,
        alpha=0.7,
        target_size=(512, 512),
    )
    comparer.run_comparison()


if __name__ == "__main__":
    main()
