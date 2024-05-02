import cv2
import yaml
import sys
import utils.image_utils as tools
sys.path.append('C:/Users/yangxiaohao/PycharmProjects/lane_Hessian_detection')


def load_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    image = cv2.imread('./data/unprocessed_picture/test2.jpg')
    config_path = "./cfg/lane_Hessian_detection.yaml"
    config = load_config(config_path)
    save_path = "./data/processed_picture"
    gk_sigma = config["gaussian_kernel_sigma"]
    hk_size = config["hessian_kernel_size"]
    tau = config["tau"]
    gray_image = tools.grayscale(image)
    normalized_data = tools.normalized(gray_image)
    masked_image = tools.mask(normalized_data)
    filtered_image = tools.median_filter(normalized_data)
    scale_space_image = tools.scale_space(filtered_image, gk_sigma[0], config_path)
    eigenvalues = tools.compute_hessian_matrix(scale_space_image, hk_size)
    max_lambda2 = tools.max_lambda2(eigenvalues)
    lambda_rou = tools.regularize_lambda(eigenvalues, max_lambda2, tau)
    V_rou = tools.enhance_filter(eigenvalues, lambda_rou)
    print(V_rou)
    # file_name = "./data/processed_picture/test1.jpg"
    # full_path = os.path.join(save_path, file_name)
    # cv2.imwrite(full_path, normalized_data)
    if True:
        # 创建一个窗口来显示图像
        cv2.imshow("车道线检测结果", V_rou)

        # 等待键盘输入
        while True:
            # 如果按下ESC键（键码是27），则退出循环
            if cv2.waitKey(1) & 0xFF == 27:
                break
