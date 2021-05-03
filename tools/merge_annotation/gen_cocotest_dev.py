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

COCO_DICT = ['images', 'categories']
IMAGES_DICT = ['file_name', 'height', 'width', 'id']
# {'license': 1, 'file_name': '000000516316.jpg', 'coco_url': '',
# 'height': 480, 'width': 640, 'date_captured': '2013-11-18 18:15:05',
# 'flickr_url': '', 'id': 516316}

CATEGORIES_DICT = ['id', 'name', 'supercategory']
# {'supercategory': 'person', 'id': 1, 'name': 'person'}
# {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'}
DARKFACE_CATEGORIES = ['face']

image_path = '..\\..\\dataset\\DarkFace_coco_0.66\\test'
save_path = '..\\..\\dataset\\DarkFace_coco_0.66\\annotations\\test_dev.json'

parser = argparse.ArgumentParser()  # 参数解析器实例化
parser.add_argument('--image_path', type=str, default=image_path)
parser.add_argument('--save', type=str, default=save_path)
args = parser.parse_args()


def load_image(path):
    img = cv2.imread(path)
    return img.shape[0], img.shape[1]


def generate_categories_dict(category):
    print('GENERATE_CATEGORIES_DICT...')
    return [{CATEGORIES_DICT[0]: category.index(x) + 1, CATEGORIES_DICT[1]: x, CATEGORIES_DICT[2]: x} for x in category]  # 类别标号从1开始


def generate_images_dict(imagelist, start_image_id):
    print('GENERATE_IMAGES_DICT...')
    images_dict = []
    with tqdm(total=len(imagelist)) as load_bar:  # 新建了一个长度一定的进度条
        for x in imagelist:
            dict = {IMAGES_DICT[0]: x,
                    IMAGES_DICT[1]: 720,
                    IMAGES_DICT[2]: 1080,
                    IMAGES_DICT[3]: start_image_id}
            load_bar.update(1)  # 进度条走一位
            images_dict.append(dict)
            start_image_id = start_image_id + 1
    return images_dict


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
    save = args.save

    img_name = os.listdir(image_path)
    img_name.sort(key=lambda x: int(x[:-4]))

    categories_dict = generate_categories_dict(DARKFACE_CATEGORIES)
    images_dict = generate_images_dict(img_name, start_image_id=0)

    #json_dict = []
    json_dict = ({COCO_DICT[0]: images_dict,  COCO_DICT[1]: categories_dict})

    print('SUCCESSFUL_GENERATE_JSON')
    save_json(json_dict, save)

    # print('dd')
