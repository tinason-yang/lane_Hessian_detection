import matplotlib.pyplot as plt
import yaml
import cv2
import os
import sys

sys.path.append('/')
from utils.image_utils import scale_space
from utils.image_utils import grayscale


def load_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def Iterative(image, gaussian_kernel_sigma, config_path):
    grays_image = grayscale(image)
    gaussian_filter_result = scale_space(grays_image, gaussian_kernel_sigma, config_path)
    return gaussian_filter_result


if __name__ == "__main__":
    image = cv2.imread('../data/unprocessed_picture/test1.jpg')
    config_path = '../cfg/lane_Hessian_detection.yaml'
    config = load_config('../cfg/lane_Hessian_detection.yaml')
    save_path = "../data/sigma_Iterative_result"

    scale = config['gaussian_kernel_sigma']
    # print(type(scale_all))
    length = len(scale)
    # print(length)
    # for i in range(length):
    for i, sigma in enumerate(scale):
        print(i)
        result = Iterative(image, gaussian_kernel_sigma=sigma, config_path=config_path)
        file_name = f"sigma value is {sigma}.jpg"
        full_path = os.path.join(save_path, file_name)
        cv2.imwrite(full_path, result)
        plt.subplot(2, length//2, i + 1)
        plt.imshow(result, cmap='gray')
        plt.title(f'sigma = {sigma}')
        plt.axis('off')
    plt.show()
