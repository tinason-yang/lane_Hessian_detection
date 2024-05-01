import matplotlib.pyplot as plt
import yaml
import cv2
import os
import sys

sys.path.append('C:/Users/yangxiaohao/PycharmProjects/lane_Hessian_detection')
from utils.image_utils import scale_space
from utils.image_utils import grayscale


def load_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def Iterative(image, gaussian_kernel_sigma):
    grays_image = grayscale(image)
    gaussian_filter_result = scale_space(grays_image, gaussian_kernel_sigma)
    return gaussian_filter_result


if __name__ == "__main__":
    image = cv2.imread('../data/unprocessed_picture/path_to_image.jpg')
    config = load_config('../cfg/lane_Hessian_detection.yaml')
    save_path = "../data/sigma_Iterative_result"

    scale = config['gaussian_kernel_sigma']
    for i, scale in enumerate(scale):
        result = Iterative(image, gaussian_kernel_sigma=scale)
        file_name = f"sigma value is {scale}.jpg"
        full_path = os.path.join(save_path, file_name)
        cv2.imwrite(full_path, result)
        plt.Subplot(1, len(scale), i + 1)
        plt.imshow(result)
        plt.title(f'sigma = {scale}')
        plt.axis('off')
    plt.show()
