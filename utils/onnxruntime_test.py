import onnxruntime
import matplotlib.pyplot as plt
import torch
import cv2
from torchvision import transforms
import numpy as np
import sys

sys.path.append('C:\Users\yangxiaohao\PycharmProjects\lane_Hessian_detection')
from utils.datasets import letterbox
from utils.general import non_max_suppression_export
from utils.plots import output_to_target, plot_skeleton_kpts

device = torch.device("cpu")

image = cv2.imread('./2.png')
image = letterbox(image, 960, stride=64, auto=True)[0]
image_ = image.copy()
image = transforms.ToTensor()(image)
image = torch.tensor(np.array([image.numpy()]))

print(image.shape)
sess = onnxruntime.InferenceSession('../export_onnx/lane_Hessian_detection.onnx')
# sess = onnxruntime.InferenceSession('../export_onnx/best.onnx',
#                                     providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider',
#                                                'CPUExecutionProvider'])

out = sess.run(['output'], {'images': image.numpy()})[0]
out = torch.from_numpy(out)

# output = non_max_suppression_kpt(out, 0.25, 0.65, nc=1, nkpt=17, kpt_label=True)
output = non_max_suppression_export(out, 0.25, 0.65, nc=1, kpt_label=True)
output = output_to_target(output)
nimg = image[0].permute(1, 2, 0) * 255
nimg = nimg.cpu().numpy().astype(np.uint8)
nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
for idx in range(output.shape[0]):
    plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

# matplotlib inline
plt.figure(figsize=(8, 8))
plt.axis('off')
plt.imshow(nimg)
plt.show()
plt.savefig("tmp")
