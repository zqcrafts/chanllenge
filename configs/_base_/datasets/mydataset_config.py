dataset_type = 'MyDataset'
data_root = '/ghome/zhuqi/mmlab/mmdetection/data/DarkFace_Train_2021'

dataset_A_train = dict(
    type='MyDataset',
    ann_file = 'annotation.txt',
    pipeline=train_pipeline
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    #dict(type='Resize', img_scale=(800, 500), keep_ratio=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]