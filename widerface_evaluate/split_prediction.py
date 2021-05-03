import mmcv
import argparse

size = 10

parser = argparse.ArgumentParser(description='Convert MMDet prediction format to WiderFace one.')
parser.add_argument('config', default='widerface_cfg/tinaface.py', help='model config name')
args = parser.parse_args()

model_name = args.config.split('/')[-1].split('.py')[0]
predict_name_list = [
    'result_500',
    'result_800',
    'result_1100',
    'result_1400',
    'result_1700'
]

def split_data(pred_name, size):
    anno_path = 'work_dirs/%s/%s.pkl' % (model_name, pred_name)
    det_result = mmcv.load(anno_path)
    num_img = len(det_result)
    split_size = num_img // size + 1
    for ii in range(size):
        start = ii * split_size; end = min((ii + 1) * split_size, num_img)
        mmcv.dump(det_result[start:end], 'work_dirs/%s/%s_%d.pkl' % (model_name, pred_name, ii))

for pred_name in predict_name_list:
    split_data(pred_name, size)
