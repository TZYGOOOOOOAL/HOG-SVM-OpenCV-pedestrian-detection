# HOG-SVM-OpenCV-pedestrian-detection
HOG+SVM 行人检测，计算机视觉实验

## 配置要求
VS2013（C++11及以上）
OpenCV 3.1.0 （高版本OpenCV加载SVM需更改代码）

## 数据 + 已训练好的模型
链接：https://pan.baidu.com/s/1gElUk5UOuSiI_HuvCuVL8Q 
提取码：oomg 

## 使用
1. 加载存在模型：
更改Config.h 中 SVM_MODEL_PATH 路径

2. 从头训练：
下载数据，将data文件夹复制到项目data文件夹（数据已处理好）
main函数 train()
