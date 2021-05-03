#!/bin/bash
cd /ghome/zhuqi/mmlab/mmdetection/

sudo ln -snf /home/worker/miniconda3/bin/python3  /usr/bin/python3
sudo python3 setup.py develop
sudo python3 tools/train.py configs/retinanet/retinanet_r50_fpn_1x_darkface_coco_0.66.py \
        --gpu 1 \
        --work-dir ./work_dirs/darkface_baseline
exit 0


