import cv2
import numpy as np
import os
from tqdm import tqdm


def linear_enhance(img1, c, b):  # 亮度就是每个像素乘以c加上b
    rows, cols, channel = img1.shape
    blank = np.zeros([rows, cols, channel], img1.dtype)  # np.zeros(img1.shape, dtype=uint8)
    linear = cv2.addWeighted(img1, c, blank, 0, b)
    return linear


root = '/gdata1/zhuqi/DarkFace_coco_0.666/train/'
output_path = '/gdata1/zhuqi/DarkFace_coco_0.666/equalization_train/'

file_list = os.listdir(root)
file_list.sort(key=lambda x: int(x[:-4]))  # 以文件名倒数第四个数左边（.txt左边）的名称进行排序

with tqdm(total=len(file_list)) as load_bar:
    for i in range(len(file_list)):
        img = cv2.imread(root + file_list[i], 1)  # 1 3 4 5 7

        (b, g, r) = cv2.split(img)
        bH = cv2.equalizeHist(b)
        gH = cv2.equalizeHist(g)
        rH = cv2.equalizeHist(r)
        result = cv2.merge((bH, gH, rH))

        cv2.imwrite(output_path + file_list[i], result)
        load_bar.update(1)
