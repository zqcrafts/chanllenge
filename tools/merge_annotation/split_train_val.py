from sklearn.model_selection import train_test_split
import numpy as np
import os
import random
import shutil

train_ratio = 0.75
val_ratio = 1 - train_ratio

num = np.random.randint(100)  # 生成一个0~99的随机数
image_path = '/gdata1/zhuqi/Darkface_coco_MF/val'
lable_path = '/gdata1/zhuqi/DarkFace_coco_0.666/val_lable'

goal_dir = '/gdata1/zhuqi/DarkFace_coco_5500/'
goal_train = goal_dir + '/' + 'train' + '/'
goal_train_lable = goal_dir + '/' + 'train_lable' + '/'
goal_val = goal_dir + '/' + 'val' + '/'
goal_val_lable = goal_dir + '/' + 'val_lable' + '/'

os.mkdir(goal_dir)
os.mkdir(goal_val)
os.mkdir(goal_val_lable)
os.mkdir(goal_dir + '/' + 'annotations' + '/')

shutil.copytree(image_path, goal_train)  # copy目标文件夹必须不存在，此命令会自动新建一个文件夹
shutil.copytree(lable_path, goal_train_lable)

train_list = os.listdir(goal_train)  # 文件路径的子文件列表
random.shuffle(train_list)  # 列表重新随机排序
train_num = int(len(train_list) * train_ratio)  # 这个可以修改划分比例

for fname in train_list[train_num:]:
    shutil.move(goal_train + '/' + fname, goal_val + '/' + fname)

val_list = os.listdir(goal_val)
val_list.sort(key=lambda x:int(x[:-4]))
train_lable_list = os.listdir(goal_train_lable)
train_lable_list.sort(key=lambda x:int(x[:-4]))

for i in val_list:
    for lable in train_lable_list:
        if i[:-4] == lable[:-4]:
            shutil.move(goal_train_lable + '/' + lable, goal_val_lable + '/' + lable)
