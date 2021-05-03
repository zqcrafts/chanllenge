# txt2coco tools for CVPR UG2+ Chanllenge
# zqcraft 2021.3.23
"""
转换的文本格式如下：(name/w*h/num_bbox/bbox)不同图片之间以#断开
#
1.png
1080 720
3
740 402 780 443 1
508 425 529 452 1
465 351 470 358 1
#
2.png
1080 720
2
529 344 537 351 1
516 335 521 341 1
"""

import os
import cv2
import json
import argparse
from tqdm import tqdm  # python进度条工具

COCO_DICT = ['images', 'annotations', 'categories']
IMAGES_DICT = ['file_name', 'height', 'width', 'id']
# {'license': 1, 'file_name': '000000516316.jpg', 'coco_url': '',
# 'height': 480, 'width': 640, 'date_captured': '2013-11-18 18:15:05',
# 'flickr_url': '', 'id': 516316}
ANNOTATIONS_DICT = ['image_id', 'iscrowd', 'area', 'bbox', 'category_id', 'id']
# {'segmentation': [[]],
# 'area': 58488.148799999995, 'iscrowd': 0,
# 'image_id': 170893, 'bbox': [270.55, 80.55, 367.51, 393.7],
# 'category_id': 18, 'id': 9940}
CATEGORIES_DICT = ['id', 'name', 'supercategory']
# {'supercategory': 'person', 'id': 1, 'name': 'person'}
# {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'}
YOLO_CATEGORIES = ['face']

image_path = '/gdata1/zhuqi/DarkFace_MSRCR_5500/train/'
annotation_path = '/gdata1/zhuqi/DarkFace_MSRCR_5500/annotations/train_merge.txt'
save_path = '/gdata1/zhuqi/DarkFace_MSRCR_5500/annotations/train_annotations.json'

parser = argparse.ArgumentParser()  # 参数解析器实例化
parser.add_argument('--image_path', type=str, default=image_path)
parser.add_argument('--annotation_path', type=str, default=annotation_path)
parser.add_argument('--save', type=str, default=save_path)
args = parser.parse_args()


def load_image(path):
    img = cv2.imread(path)
    return img.shape[0], img.shape[1]


def generate_categories_dict(category):
    print('GENERATE_CATEGORIES_DICT...')
    return [{CATEGORIES_DICT[0]: category.index(x) + 1, CATEGORIES_DICT[1]: x, CATEGORIES_DICT[2]: x} for x in category]  # 类别标号从1开始


def generate_images_dict(imagelist, image_path, start_image_id):
    print('GENERATE_IMAGES_DICT...')
    images_dict = []
    with tqdm(total=len(imagelist)) as load_bar:  # 新建了一个长度一定的进度条
        for x in imagelist:
            dict = {IMAGES_DICT[0]: x,
                    IMAGES_DICT[1]: load_image(image_path + "/" + x)[0],
                    IMAGES_DICT[2]: load_image(image_path + "/" + x)[1],
                    IMAGES_DICT[3]: imagelist.index(x) + start_image_id}
            load_bar.update(1)  # 进度条走一位
            images_dict.append(dict)
    return images_dict


def generate_annotations_dict(annotation_path, start_id=0):
    print('GENERATE_ANNOTATIONS_DICT...')
    annotations_dict = []
    id = start_id
    image_id = -1
    txt = open(annotation_path, 'r', encoding='utf-8-sig')

    for line in txt.readlines():
        if line == '#\n':
            flag = 0
        else:
            if flag == 0:
                #image_name = int(line[:-5])
                image_id = image_id + 1
                flag = 1
            elif flag == 1:
                flag = 2
            elif flag == 2:
                flag = 3
            elif flag == 3:
                x_min = float(line.split(' ')[0])
                y_min = float(line.split(' ')[1])
                w = float(line.split(' ')[2])
                h = float(line.split(' ')[3])
                category_id = int(line.split(' ')[4])
                area = w * h
                bbox = [x_min, y_min, w, h]
                dict = {'id': id, 'image_id': image_id, 'iscrowd': 0, 'area': area, 'bbox': bbox,
                        'category_id': category_id}
                annotations_dict.append(dict)
                id = id + 1

    return annotations_dict


def load_json(path):
    with open(path, 'r') as f:
        json_dict = json.load(f)
        for i in json_dict:
            print(i)
        print(json_dict['annotations'])


def save_json(json_dict, path):
    print('SAVE_JSON...')
    with open(path, 'w') as f:
        json.dump(json_dict, f)
    print('SUCCESSFUL_SAVE_JSON:', path)


if __name__ == '__main__':
    image_path = args.image_path
    annotation_path = args.annotation_path
    save = args.save

    img_name = os.listdir(image_path)
    img_name.sort(key=lambda x: int(x[:-4]))

    categories_dict = generate_categories_dict(YOLO_CATEGORIES)
    images_dict = generate_images_dict(img_name, image_path, start_image_id=0)
    annotations_dict = generate_annotations_dict(annotation_path)


    #json_dict = []
    json_dict = ({COCO_DICT[0]: images_dict, COCO_DICT[1]: annotations_dict, COCO_DICT[2]: categories_dict})

    print('SUCCESSFUL_GENERATE_JSON')
    save_json(json_dict, save)

    # print('dd')
