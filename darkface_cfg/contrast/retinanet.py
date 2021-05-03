model = dict(
    type='RetinaNet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

# dataset settings
dataset_type = 'CocoDataset'
data_root = '/gdata1/zhuqi/DarkFace_coco_0.666/'
img_norm_cfg = dict(   
    mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile',to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=[1, 1]), 
    dict(type='RandomSquareCrop', crop_choice=[0.3, 0.45, 0.6, 0.8, 1.0]), 
    #dict(type='PhotoMetricDistortion', brightness_delta=32, contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=18),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
    dict(type='Normalize',**img_norm_cfg),  # **的作用是将字典拆成两个独立的形参传入函数
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),               
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1080, 720),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32, pad_val=0),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        ann_file=
        '/gdata1/zhuqi/DarkFace_coco_0.666/annotations/train_annotations.json',
        img_prefix=
        '/gdata1/zhuqi/DarkFace_coco_0.666/train/',
        pipeline=train_pipeline),
    val=dict(
        type='CocoDataset',
        ann_file=
        '/gdata1/zhuqi/DarkFace_coco_0.666/annotations/val_annotations.json',
        img_prefix=
        '/gdata1/zhuqi/DarkFace_coco_0.666/val/',
        pipeline=test_pipeline),
    test=dict(
        type='CocoDataset',
        ann_file=
        '/gdata1/zhuqi/DarkFace_coco_0.666/annotations/val_annotations.json',
        img_prefix=
        '/gdata1/zhuqi/DarkFace_coco_0.666/val/',
        pipeline=test_pipeline))


# optimizer
evaluation = dict(interval=1 , metric='bbox')
optimizer = dict(type='SGD', lr=2e-2, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineRestart',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    min_lr_ratio=0.01,
    periods=[30] * 21,
    restart_weights=[1] * 21,
    )
total_epochs = 30
checkpoint_config = dict(interval=6)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None #'/gdata2/zhuqi/work_dirs/tinaface/resume60/epoch_90.pth'
resume_from = None #'/gdata2/zhuqi/work_dirs/tinaface/resume60/epoch_90.pth'
workflow = [('train', 1)]
work_dir = '/gdata2/zhuqi/work_dirs/tinaface/contrast/retinanet'
gpu_ids = range(0, 1)