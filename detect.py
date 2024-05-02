import cv2
import numpy
import sys
sys.path.append('C:/Users/yangxiaohao/PycharmProjects/lane_Hessian_detection')


def model(frame):

    return frame


def detect(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("open error")
        return
    else:
        print("正在输出检测画面")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            processed_frame = model(frame)  # 检测步骤
            combined_frame = cv2.hconcat([frame, processed_frame])
            cv2.imshow("车道线检测结果", combined_frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = ""
    detect(video_path)
