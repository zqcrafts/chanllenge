# contain GN \ resize640,640 \ Diouloss \ backbone \ inception 
norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)
#norm_cfg=dict(type='BN', requires_grad=True)
model = dict(
    type='RetinaNet', 
    pretrained='torchvision://resnet50',
    #pretrained = '/gdata2/zhuqi/work_dirs/tinaface/resume60/epoch_90.pth',
    backbone=dict(
        type='ResNet', 
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,  # 冻结的stage数量，即stage不更新参数，-1表示都更新参数
        norm_cfg=norm_cfg,  # 归一化方式
        #norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,  # 归一化的评价 Note: Effect on Batch Norm and its variants only.
        style='pytorch'),  # pytorch风格的网络，stride=2的3*3卷积层
    neck=[
        dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],  # 输入的各个stage通道数
            out_channels=256,  # 输出的每个尺度特征层通道数
            start_level=0,  # 从输入多尺度特征图第0个开始算，一共四个
            norm_cfg=norm_cfg,
            add_extra_convs='on_input',  # 额外的2个输出来源是由backbone输出的两个特征图
            num_outs=6,  # FPN输出特征图个数6，且通道数都是256，加了一层octave是为了检测小样本
            upsample_cfg=dict(mode='bilinear')  
        ),     
        dict(  
            type='Inception',
            in_channel=256,
            num_levels=6,
            norm_cfg=norm_cfg,
            share=True
        )
    ],
    bbox_head=dict(
        type='IoUAwareRetinaHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=2.5198420997897464,
            scales_per_octave=3,
            ratios=[1.3],
            strides=[4, 8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_iou=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        reg_decoded_bbox=True,
        loss_bbox=dict(type='DIoULoss', loss_weight=2.0)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',  # 最大iou分配器
            pos_iou_thr=0.35,  # 正样本阈值，
            neg_iou_thr=0.35,  # 负样本阈值，用于anchors匹配gt
            min_pos_iou=0.35,  # 正样本阈值下限，用于gt匹配anchors
            ignore_iof_thr=-1),  # bbox本身是否忽略，-1表示不忽略  
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,  # nms前每个输出层最多保留多少个框，-1不限制
        min_bbox_size=0,  # 过滤掉最小bbox的尺寸
        score_thr=0.05,  # 分数的阈值
        #nms=dict(type='nms', iou_threshold=0.5),  # nms方法和nms阈值 
        nms=dict(type='lb_nms', iou_threshold=0.45),  # tinaface使用lb_nms方法
        max_per_img=100))  # 最终输出的每张照片最多bbox个数

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
evaluation = dict(interval=1, metric='bbox')
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
total_epochs = 450
checkpoint_config = dict(interval=6)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None #'/gdata2/zhuqi/work_dirs/tinaface/resume60/epoch_90.pth'
resume_from = '/gdata2/zhuqi/work_dirs/tinaface/contrast/GN_Diou_Incep_iou/epoch_30.pth'
workflow = [('train', 1)]
work_dir = '/gdata2/zhuqi/work_dirs/tinaface/contrast/new_sota'
gpu_ids = range(0, 1)
