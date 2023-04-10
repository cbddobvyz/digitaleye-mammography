checkpoint_config = dict(interval=1, create_symlink=False)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/workspace/notebooks/mmdetection/checkpoints/dynamic_rcnn_r50_fpn_1x-62a3f276.pth'
resume_from = None
workflow = [('train', 1)]
custom_imports = dict(imports=['mammo_dataset'], allow_failed_imports=False)
dataset_type = 'MammoDataset'
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

classes = ('BIRADS45', 'BIRADS12')
work_dir = '/workspace/notebooks/Mammography_Benchmark_Article/KETEM_proven_pathology/model_results/4_Fold/results_dynamicrcnn'
# gpu_ids = range(0, 4)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=0,
    train=dict(
        type='MammoDataset',
        classes=('BIRADS45', 'BIRADS12'),
        ann_file=
        '/workspace/notebooks/Mammography_Benchmark_Article/KETEM_proven_pathology/4_Fold/annot_train.txt',
        img_prefix=
        '/workspace/notebooks/Mammography_Benchmark_Article/KETEM_proven_pathology/4_Fold/Train/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Resize',
                img_scale=[(1333, 640), (1333, 800)],
                multiscale_mode='range',
                keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                # mean=[123.675, 116.28, 103.53],
                # std=[58.395, 57.12, 57.375],
                mean=[0, 0, 0],
                std=[255, 255, 255],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='MammoDataset',
        ann_file=
        '/workspace/notebooks/Mammography_Benchmark_Article/KETEM_proven_pathology/4_Fold/annot_val.txt',
        img_prefix=
        '/workspace/notebooks/Mammography_Benchmark_Article/KETEM_proven_pathology/4_Fold/Val/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        # mean=[123.675, 116.28, 103.53],
                        # std=[58.395, 57.12, 57.375],
                        mean=[0, 0, 0],
                        std=[255, 255, 255],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('BIRADS45', 'BIRADS12')),
    test=dict(
        type='MammoDataset',
        ann_file=
        '/workspace/notebooks/Mammography_Benchmark_Article/KETEM_proven_pathology/annot_test.txt',
        img_prefix=
        '/workspace/notebooks/Mammography_Benchmark_Article/KETEM_proven_pathology/Test/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        # mean=[123.675, 116.28, 103.53],
                        # std=[58.395, 57.12, 57.375],
                        mean=[0, 0, 0],
                        std=[255, 255, 255],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    step=[8, 15])
evaluation = dict(
    interval=1,  
    metric='mAP') 
runner = dict(type='EpochBasedRunner', max_epochs=30)

_base_ = '/workspace/notebooks/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(frozen_stages=-1),
    roi_head=dict(
        type='DynamicRoIHead',
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=2,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    train_cfg=dict(
        rpn_proposal=dict(nms=dict(iou_threshold=0.85)),
        rcnn=dict(
            dynamic_rcnn=dict(
                iou_topk=75,
                beta_topk=10,
                update_iter_interval=100,
                initial_iou=0.4,
                initial_beta=1.0))),
    # test_cfg=dict(
    #     rpn=dict(
    #         nms=dict(type='nms', iou_threshold=0.85),
    #         ),
    #     rcnn=dict(
    #         score_thr=0.05,
    #         # nms=dict(type='nms', iou_threshold=0.1),
    #         nms=dict(type='soft_nms', iou_threshold=0.1, min_score=0.05),
    #         max_per_img=100))
    #         )
    test_cfg=dict(rpn=dict(nms=dict(iou_threshold=0.85))))