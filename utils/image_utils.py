import numpy as np
import cv2
from scipy.signal import convolve2d
import yaml
import sys
sys.path.append('C:/Users/yangxiaohao/PycharmProjects/lane_Hessian_detection')


def load_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def grayscale(image):
    if image is None:
        print("grayscale error")
        return None
    gray_image = np.zeros(image.shape)
    gray_image = 0.299 * image[:, :, 2] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 0]
    gray_image = gray_image.astype(np.float32)  # 由于归一到[0,1],选用float32类型

    return gray_image


def normalized(gray_image):
    normalized_data = gray_image / 255.0

    return normalized_data


def mask(image):
    height, width = image.shape[:2]
    image[:int(height * 0.3), :] = 0
    return image


def median_filter(normalized_data):
    config = load_config('./cfg/lane_Hessian_detection.yaml')
    kernel_size = config['median_filter_size']
    if kernel_size % 2 == 0:
        raise ValueError("滤波器的大小必须为大于一的奇数")

    edge = kernel_size // 2  # 向下取整
    height, width = normalized_data.shape
    filtered_image = np.zeros_like(normalized_data)
    filtered_image = cv2.medianBlur(normalized_data, kernel_size)

    # for i in range(edge, height - edge):
    #     for j in range(edge, width - edge):
    #         # 提取当前像素周围的核心区域
    #         window = normalized_data[i - edge:i + edge, j - edge:j + edge + 1]
    #         # 计算出中位数并赋值给输出图像
    #         filtered_image[i, j] = np.median(window)

    return filtered_image


def gaussian_kernel(sigma, config_path):
    config = load_config(config_path)
    size = config['gaussian_kernel_size']
    size = int(size) // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    g = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def scale_space(image, gaussian_kernel_sigma, config_path):
    """此步骤用于对图像进行高斯平滑以生成尺度空间."""
    kernel = gaussian_kernel(gaussian_kernel_sigma, config_path)
    print("0")
    return convolve2d(image, kernel, mode='same', boundary='fill', fillvalue=0).astype(np.float32)


def compute_hessian_matrix(scale_space_image, hessian_kernel_size):
    # 使用32位浮点数精度值进行计算
    # scale_space_image = scale_space_image.astype(np.float32)
    ddepth = cv2.CV_32F
    # 计算二阶导数
    dxx = cv2.Sobel(scale_space_image, ddepth, 2, 0, ksize=hessian_kernel_size)  # 2沿x轴的导数阶数
    dyy = cv2.Sobel(scale_space_image, ddepth, 0, 2, ksize=hessian_kernel_size)  # 2沿y轴的导数阶数
    dxy = cv2.Sobel(scale_space_image, ddepth, 1, 1, ksize=hessian_kernel_size)  # 1沿x轴的导数阶数，1沿y轴的导数阶数
    height, width = scale_space_image.shape
    eigenvalues = np.zeros((height, width, 2), dtype=np.float32)  # 存储两个特征值

    # 遍历每个像素点，计算其Hessian矩阵的特征值
    for y in range(height):
        for x in range(width):
            # 构造Hessian矩阵
            Hessian = np.array([
                [dxx[y, x], dxy[y, x]],
                [dxy[y, x], dyy[y, x]]
            ])
            # 计算特征值
            eigs = np.linalg.eigvals(Hessian)

            # 排序特征值，绝对值小的放前面
            eigs = sorted(eigs, key=abs)
            # print(eigs)
            eigenvalues[y, x, :] = eigs
        # print(eigenvalues)
    return eigenvalues


def max_lambda2(eigenvalues):
    max_lambda2 = np.max(eigenvalues[:, :, 1])
    return max_lambda2


def regularize_lambda(eigenvalues, max_lambda2, tau):
    # 初始化lambda_rou数组，与eigenvalues[:,:,1]维度相同
    lambda_rou = np.zeros_like(eigenvalues[:, :, 1], dtype=np.float32)
    for i in range(eigenvalues.shape[0]):
        for j in range(eigenvalues.shape[1]):
            lambda3 = eigenvalues[i, j, 1]
            if lambda3 >= tau * max_lambda2:
                lambda_rou[i, j] = lambda3
            elif lambda3 > 0:
                lambda_rou[i, j] = tau * max_lambda2
            # 其他情况下，lambda_rou初始化即为0，不作改变
    return lambda_rou


def enhance_filter(eigenvalues, lambda_rou):
    V_rou = np.zeros_like(eigenvalues[:, :, 1], dtype=np.float32)
    for i in range(eigenvalues.shape[0]):
        for j in range(eigenvalues.shape[1]):
            lambda2 = eigenvalues[i, j, 1]
            if lambda2 <= 0 and lambda_rou[i, j] <=0:
                V_rou[i, j] = 0
            elif lambda2 >= lambda_rou[i, j]/2 > 0:
                V_rou[i, j] = 1
            else:
                V_rou[i, j] = pow(lambda2, 2)*(lambda_rou[i, j] - lambda2) * pow(3/(lambda2 + lambda_rou[i, j]), 3)

    return V_rou
