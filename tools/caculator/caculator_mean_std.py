# coding:utf-8
import os
import numpy as np
import cv2
from tqdm import tqdm  # python进度条工具
"""
    在训练前先运行该函数获得数据的均值和标准差
"""
means = [0,0,0]
stdevs = [0,0,0]

with tqdm(total = 6000) as load_bar:
    for n in range(6000):
        img = cv2.imread("/gdata1/zhuqi/Darkface_coco_MF/train/" + str(n+1) + '.png')
        for i in range(3):
            # 一个通道的均值和标准差
            means[i] += img[:, :, i].mean()
            stdevs[i] += img[:, :, i].std()
        load_bar.update(1) 

means = np.asarray(means) / 6000
stdevs = np.asarray(stdevs) / 6000

print(means) # [55.636, 51.555, 69.053]
print(stdevs) # [44.886, 48.516, 51.745]



  