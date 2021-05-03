from sklearn.model_selection import train_test_split
import numpy as np
import os
import random
import shutil
from tqdm import tqdm


lable_path = '/gdata1/zhuqi/DarkFace_MF_5500/train_lable/'
image_path = '/gdata1/zhuqi/Darkface_coco_MSRCR/train/'
#out_lable_path = '/gdata1/zhuqi/DarkFace_MSRCR_500/train_lable/'
out_image_path = '/gdata1/zhuqi/DarkFace_MSRCR_5500/train_new/'


# list = os.listdir(lable_path) 
# with tqdm(total=len(list)) as load_bar:
#     for name in list:
#         old_name = lable_path + name[:-4] + '.txt'
#         new_name = out_lable_path + name[:-4] + '.txt'
#         shutil.copyfile(old_name, new_name)
#         load_bar.update(1) 

list = os.listdir(lable_path) 
with tqdm(total=len(list)) as load_bar2:
    for name in list:
        old_name = image_path + name[:-4] + '.png'
        new_name = out_image_path + name[:-4] + '.png'
        shutil.copyfile(old_name, new_name)
        load_bar2.update(1) 