dataset_type = 'CULaneDataset'
data_root = '/home/fyj/zky/tusimple/culane/'
file_client_args = dict(backend='disk')
ori_img_w = 1640
ori_img_h = 590
img_w = 800
img_h = 320
cut_height = 270
train_pipeline = [
    dict(
        type='GenerateLaneLine',
        keys=['img', 'lane_line', 'seg'],
        img_info=(320, 800),
        num_points=72,
        max_lanes=4,
        meta_keys=['img_metas'],
        transforms=[
            dict(
                name='Resize',
                parameters=dict(size=dict(height=320, width=800)),
                p=1.0),
            dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.5),
            dict(name='ChannelShuffle', parameters=dict(p=1.0), p=0.1),
            dict(
                name='MultiplyAndAddToBrightness',
                parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
                p=0.6),
            dict(
                name='AddToHueAndSaturation',
                parameters=dict(value=(-10, 10)),
                p=0.7),
            dict(
                name='OneOf',
                transforms=[
                    dict(name='MotionBlur', parameters=dict(k=(3, 5))),
                    dict(name='MedianBlur', parameters=dict(k=(3, 5)))
                ],
                p=0.2),
            dict(
                name='Affine',
                parameters=dict(
                    translate_percent=dict(x=(-0.1, 0.1), y=(-0.1, 0.1)),
                    rotate=(-10, 10),
                    scale=(0.8, 1.2)),
                p=0.7),
            dict(
                name='Resize',
                parameters=dict(size=dict(height=320, width=800)),
                p=1.0)
        ]),
    dict(type='ToTensor_', keys=['img', 'lane_line', 'seg', 'img_metas'])
]
test_pipeline = [
    dict(
        type='GenerateLaneLine',
        keys=['img'],
        meta_keys=['img_metas'],
        img_info=(320, 800),
        num_points=72,
        max_lanes=4,
        transforms=[
            dict(
                name='Resize',
                parameters=dict(size=dict(height=320, width=800)),
                p=1.0)
        ],
        training=False),
    dict(type='ToTensor_', keys=['img', 'img_metas'])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='CULaneDataset',
        data_root='/home/fyj/zky/tusimple/culane/',
        split='train',
        cut_height=270,
        img_fo=(590, 1640),
        resize_img_info=(320, 800),
        pipeline=[
            dict(
                type='GenerateLaneLine',
                keys=['img', 'lane_line', 'seg'],
                img_info=(320, 800),
                num_points=72,
                max_lanes=4,
                meta_keys=['img_metas'],
                transforms=[
                    dict(
                        name='Resize',
                        parameters=dict(size=dict(height=320, width=800)),
                        p=1.0),
                    dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.5),
                    dict(name='ChannelShuffle', parameters=dict(p=1.0), p=0.1),
                    dict(
                        name='MultiplyAndAddToBrightness',
                        parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
                        p=0.6),
                    dict(
                        name='AddToHueAndSaturation',
                        parameters=dict(value=(-10, 10)),
                        p=0.7),
                    dict(
                        name='OneOf',
                        transforms=[
                            dict(name='MotionBlur', parameters=dict(k=(3, 5))),
                            dict(name='MedianBlur', parameters=dict(k=(3, 5)))
                        ],
                        p=0.2),
                    dict(
                        name='Affine',
                        parameters=dict(
                            translate_percent=dict(
                                x=(-0.1, 0.1), y=(-0.1, 0.1)),
                            rotate=(-10, 10),
                            scale=(0.8, 1.2)),
                        p=0.7),
                    dict(
                        name='Resize',
                        parameters=dict(size=dict(height=320, width=800)),
                        p=1.0)
                ]),
            dict(
                type='ToTensor_',
                keys=['img', 'lane_line', 'seg', 'img_metas'])
        ]),
    val=dict(
        type='CULaneDataset',
        data_root='/home/fyj/zky/tusimple/culane/',
        split='test',
        cut_height=270,
        img_fo=(590, 1640),
        resize_img_info=(320, 800),
        pipeline=[
            dict(
                type='GenerateLaneLine',
                keys=['img'],
                meta_keys=['img_metas'],
                img_info=(320, 800),
                num_points=72,
                max_lanes=4,
                transforms=[
                    dict(
                        name='Resize',
                        parameters=dict(size=dict(height=320, width=800)),
                        p=1.0)
                ],
                training=False),
            dict(type='ToTensor_', keys=['img', 'img_metas'])
        ]),
    test=dict(
        type='CULaneDataset',
        data_root='/home/fyj/zky/tusimple/culane/',
        split='test',
        cut_height=270,
        img_fo=(590, 1640),
        resize_img_info=(320, 800),
        pipeline=[
            dict(
                type='GenerateLaneLine',
                keys=['img'],
                meta_keys=['img_metas'],
                img_info=(320, 800),
                num_points=72,
                max_lanes=4,
                transforms=[
                    dict(
                        name='Resize',
                        parameters=dict(size=dict(height=320, width=800)),
                        p=1.0)
                ],
                training=False),
            dict(type='ToTensor_', keys=['img', 'img_metas'])
        ]))
work_dir = './output_dir'
evaluation = dict(interval=1, output_basedir='./output_dir', save_best='auto')
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [
    dict(
        type='ExpMomentumEMAHook',
        resume_from=None,
        momentum=0.0002,
        priority=49)
]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
num_classes = 4
num_points = 72
model = dict(
    type='DNLATR',
    num_queries=4,
    with_random_refpoints=False,
    num_patterns=0,
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=None,
        num_outs=1),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256, num_heads=8, dropout=0.0, batch_first=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.0,
                act_cfg=dict(type='PReLU')))),
    decoder=dict(
        num_layers=6,
        query_dim=3,
        query_scale_type='cond_elewise',
        with_modulated_hw_attn=True,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256,
                num_heads=8,
                attn_drop=0.0,
                proj_drop=0.0,
                cross_attn=False),
            cross_attn_cfg=dict(
                embed_dims=256,
                num_heads=8,
                attn_drop=0.0,
                proj_drop=0.0,
                cross_attn=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.0,
                act_cfg=dict(type='PReLU'))),
        return_intermediate=True),
    positional_encoding=dict(num_feats=128, temperature=20, normalize=True),
    head=dict(
        type='DNHead',
        num_classes=4,
        num_points=72,
        img_info=(320, 800),
        ori_img_info=(590, 1640),
        cut_height=270,
        assigner=dict(
            type='HungarianLaneAssigner',
            distance_cost=dict(type='Distance_cost', weight=3.0),
            cls_cost=dict(type='FocalLossCost')),
        loss_cls=dict(
            type='FocalLoss', gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_xyt=dict(type='SmoothL1Loss', loss_weight=0.2),
        loss_iou=dict(type='Line_iou', loss_weight=2.0),
        loss_seg=dict(
            type='CrossEntropyLoss', loss_weight=1.0, ignore_index=255),
        test_cfg=dict(conf_threshold=0.4)),
    train_cfg=None,
    test_cfg=None)
base_lr = 0.00025
interval = 1
optimizer = dict(
    type='AdamW',
    lr=0.00025,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(lr_mult=0.1, decay_mult=1.0))))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
max_epochs = 50
runner = dict(type='EpochBasedRunner', max_epochs=50)
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,
    num_last_epochs=1,
    min_lr_ratio=0.05)
auto_resume = False
gpu_ids = range(0, 2)
