import cv2
import numpy as np
import os
from tqdm import tqdm

root = '/gdata1/zhuqi/DarkFace_coco_0.666/train/'
output_path = '/gdata1/zhuqi/DarkFace_coco_0.666/linear_train/'


def linear_enhance(img1, c, b): 
    rows, cols, channel = img1.shape
    blank = np.zeros([rows, cols, channel], img1.dtype) 
    linear = cv2.addWeighted(img1, c, blank, 0, b)
    return linear


file_list = os.listdir(root)
file_list.sort(key=lambda x: int(x[:-4]))

with tqdm(total=len(file_list)) as load_bar:
    for i in range(len(file_list)):
        img = cv2.imread(root + file_list[i], 1)  
        result = linear_enhance(img, 7, 25)  
        cv2.imwrite(output_path + file_list[i], result)
        load_bar.update(1)