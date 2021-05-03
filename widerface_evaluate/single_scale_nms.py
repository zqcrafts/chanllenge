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

def load_pred(pred_name):
    anno_path = 'work_dirs/%s/%s' % (model_name, pred_name)
    det_result = mmcv.load(anno_path)
    return det_result

def parse_each_annotation(dets):
    # perform group NMS here
    t_start = time.time()
    dets = dets[0]
    all_dets_shape = dets.shape
    dets = py_weighted_nms(dets, 0.45, 0.5)
    # dets = nms(dets, 0.45)
    t_end = time.time()
    print("Processing ", all_dets_shape, " using %g s" % (t_end-t_start))
    return dets

def parse_annotation(det_result):
    from multiprocessing import cpu_count
    from multiprocessing.pool import Pool
    pool = Pool(cpu_count() // 2)
    nms_dets = pool.map(parse_each_annotation, det_result)
    pool.close()
    return nms_dets

for pred_name in predict_name_list:
    pred_filename = '%s.pkl' % (pred_name)
    pred_dets = load_pred(pred_filename)
    nms_dets = parse_annotation(pred_dets)
    mmcv.dump(nms_dets, 'work_dirs/%s/%s_nms.pkl' % (model_name, pred_name))
