import mmcv
import numpy as np
import os
import xml.etree.ElementTree as ET
import argparse
from nms import py_weighted_nms, nms
import tqdm
import time

parser = argparse.ArgumentParser(description='Convert MMDet prediction format to WiderFace one.')
parser.add_argument('config', default='widerface_cfg/tinaface.py', help='model config name')
args = parser.parse_args()

# prepare for output dir
model_name = args.config.split('/')[-1].split('.py')[0]
output_dir = 'widerface_evaluate/widerface_txt/'
os.makedirs(output_dir, exist_ok=True)

predict_name_list = [
    'result_500',
    'result_800',
    'result_1100',
    'result_1400',
    'result_1700'
]

def load_anno():
    anno_path = 'data/WIDERFace/val.txt'
    with open(anno_path, 'r') as f:
        anno_list = f.readlines()
    return anno_list

def load_pred():
    pred_list = []
    for pred_name in predict_name_list:  
        anno_path = 'work_dirs/%s/%s_nms.pkl' % (model_name, pred_name)
        det_result = mmcv.load(anno_path)
        pred_list.append(det_result)
    pred_data = []
    num_img = len(pred_list[0])
    for ii in range(num_img):
        pred_pair = []
        for tmp in range(len(pred_list)):
            pred_pair.append(pred_list[tmp][ii])
        pred_data.append(pred_pair)
    return pred_data

def parse_each_annotation(param):
    t_start = time.time()
    anno_name = param[0].strip(); dets = param[1]
    xml_folder_path = 'data/WIDERFace/WIDER_val/Annotations/'
    xml_path = xml_folder_path + anno_name + '.xml'
    tree = ET.parse(xml_path)
    root = tree.getroot()
    folder_name = root.find('folder').text
    os.makedirs(output_dir + folder_name, exist_ok=True)
    dets = np.vstack([det for det in dets])
    all_dets_shape = dets.shape
    # perform group NMS here
    dets = py_weighted_nms(dets, 0.45, 0.5)
    # dets = nms(dets, 0.45)
    save_path = output_dir + folder_name + '/' + anno_name + '.txt'
    bbox_list = []
    for ii in range(dets.shape[0]):
        box = dets[ii]
        x = int(box[0])
        y = int(box[1])
        w = int(box[2]) - int(box[0])
        h = int(box[3]) - int(box[1])
        confidence = str(box[4])
        line = f'{x} {y} {w} {h} {confidence}\n'
        bbox_list.append(line)
    with open(save_path, 'w') as f:
        f.write(anno_name + '.jpg' + '\n')
        f.write(str(len(bbox_list)) + '\n')
        for each_det in bbox_list:
            f.write(each_det)
    t_end = time.time()
    print("Processing ", all_dets_shape, " using %g s" % (t_end-t_start))
    # with open("tmp.log", 'a') as f:
    #     f.write("A\n")

def parse_annotation(anno_list, det_result):
    from multiprocessing import cpu_count
    from multiprocessing.pool import Pool
    pool = Pool(cpu_count() // 2)
    _ = list(tqdm.tqdm(pool.imap(parse_each_annotation, zip(anno_list, det_result)), total=len(anno_list)))
    pool.close()

anno_list = load_anno()
pred_list = load_pred()
parse_annotation(anno_list, pred_list)