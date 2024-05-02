# lane_Hessian_detection
   (使用Hessian矩阵与多尺度滤波检测车道线的算法实现)

大连海事大学本科生毕业论文代码

## 数据集
https://github.com/TuSimple/tusimple-benchmark/issues/3

## 参数设置

- 超参数设置位于lane_Hessian_detection.yaml文件中

## 运行

- detect.py为视频流检测
- test.py为演示以及调试阶段用单张图片检测

## 图片以及视频保存

- 数据集位于unprocessed_picture文件夹中
- 不同大小的sigma处理后的效果图位于sigma_Iterative文件夹中
- 经过detect或者test处理后的图片以及视频位于processed_picture