import cv2
import numpy as np
import sys
import yaml
import utils.image_utils as tools
import time
sys.path.append('C:/Users/yangxiaohao/PycharmProjects/lane_Hessian_detection')


def load_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def model(image, config_path):
    config = load_config(config_path)
    gk_sigma = config["gaussian_kernel_sigma"]
    hk_size = config["hessian_kernel_size"]
    tau = config["tau"]
    gray_image = tools.grayscale(image)
    normalized_data = tools.normalized(gray_image)
    # masked_image = tools.mask(normalized_data)
    filtered_image = tools.median_filter(normalized_data)
    scale_space_image = tools.scale_space(filtered_image, gk_sigma[0], config_path)
    eigenvalues = tools.compute_hessian_matrix(scale_space_image, hk_size)
    max_lambda2 = tools.max_lambda2(eigenvalues)
    lambda_rou = tools.regularize_lambda(eigenvalues, max_lambda2, tau)
    V_rou = tools.enhance_filter(eigenvalues, lambda_rou)
    # V_rou = tools.mask(V_rou)

    return V_rou


def hough_lane(V_rou):
    lines = cv2.HoughLinesP(V_rou, 0.1, np.pi/90, threshold=5, minLineLength=1, maxLineGap=1)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(V_rou, (x1, np.min((y1, y2))), (x2, np.max((y1, y2))), (0, 120, 0), 1)

    return V_rou


def detect(path, config_path, save_path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("open error")
        return
    else:
        print("正在输出检测画面")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            start_time = time.time()
            processed_frame = model(frame, config_path)  # 检测步骤
            end_time = time.time()
            process_time = end_time - start_time
            fps_value = int(1 / process_time*10)
            text = f'Processing Time: {process_time/10:.2f}s  FPS: {fps_value}'
            # combined_frame = cv2.hconcat([frame, processed_frame])
            # image_uint8 = cv2.normalize(processed_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            # processed_frame = hough_lane(image_uint8)
            cv2.putText(processed_frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("detect_result", processed_frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "./data/video/test.mp4"
    config_path = "./cfg/lane_Hessian_detection.yaml"
    save_path = "./data/processed_picture"
    detect(video_path, config_path, save_path)

