_base_ = [
    './_base_/culane.py', './_base_/default_runtime.py'
]

num_classes = 4
num_points = 72

ori_img_w = 1640
ori_img_h = 590
img_w = 800
img_h = 320
cut_height = 270

ckpt_timm = "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet18-118f1556.pth"
model = dict(
    type='DNLATR',
    num_queries=192,
    left_prio=24,
    with_random_refpoints=False,
    num_patterns=0,
    max_lanes = num_classes,
    num_feat_layers = 3,
#    backbone=dict(
#        type='mmcls.TIMMBackbone',
#        model_name='swsl_resnet18',
#        features_only=True,
#        pretrained=True,
#        out_indices=(1, 2, 3, 4)),
    backbone=dict(
        type='ResNet',
        depth=34,
        num_stages=4,
        out_indices=(0,1,2,3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint="torchvision://resnet34")),
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
                embed_dims=256, num_heads=8, dropout=0., batch_first=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.,
                act_cfg=dict(type='PReLU')))),
    decoder=dict(
        num_layers=6,
        query_dim=3,
        query_scale_type='cond_elewise',
        num_points = num_points,
        with_modulated_hw_attn=True,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256,
                num_heads=8,
                attn_drop=0.,
                proj_drop=0.,
                cross_attn=False),
            cross_attn_cfg=dict(
                embed_dims=256,
                num_heads=8,
                attn_drop=0.,
                proj_drop=0.,
                cross_attn=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.,
                act_cfg=dict(type='PReLU'))),
        return_intermediate=True),
    positional_encoding=dict(num_feats=128, temperature=20, normalize=True),
    head=dict(
        type='DNHeadv2',
        num_classes=num_classes,
        num_points = num_points,
        img_info=(img_h,img_w),
        ori_img_info = (ori_img_h,ori_img_w),
        cut_height = cut_height,
        assigner = dict(type='One2ManyLaneAssigner',
                        distance_cost = dict(type="Distance_cost",weight=3.),
                        cls_cost = dict(type='FocalLossCost')),
        loss_cls=dict(
            type='FocalLoss_py',
            gamma=2.0,
            alpha=0.25,
            use_sigmoid=True,
            loss_weight=2.0),
        loss_xyt = dict(type='SmoothL1Loss',loss_weight = 0.3),
        loss_iou=dict(type='Line_iou', loss_weight=2.0),
        loss_seg = dict(type='CrossEntropyLoss',loss_weight=1.0,ignore_index=255,class_weight=[0.4,1.,1.,1.,1.]),
        test_cfg = dict(conf_threshold=0.5)),
     train_cfg = None,
     test_cfg = None
    # training and testing settings
    # train_cfg=dict(
    #     assigner=dict(
    #         type='HungarianAssigner',
    #         match_costs=[
    #             dict(type='FocalLossCost', weight=2., eps=1e-8),
    #             dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
    #             dict(type='IoUCost', iou_mode='giou', weight=2.0)
    #         ])),
    )

# optimizer
base_lr = 0.00025
interval = 1
eval_step = 1
optimizer = dict(
    type='AdamW',
    lr=base_lr, 
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)})
)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))

# learning policy
max_epochs = 20
runner = dict(
    type='EpochBasedRunner', max_epochs=max_epochs)

# learning rate
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    warmup='linear',
    warmup_iters=2777,
    warmup_ratio=0.01,
    min_lr=1e-08
     )


checkpoint_config = dict(interval=interval)
data = dict(samples_per_gpu=16)

custom_hooks = [
    dict(
        type='ExpMomentumEMAHook',
        resume_from=None,
        momentum=0.004,
        priority=49)
]

log_config = dict(interval=50)
auto_scale_lr = dict(base_batch_size=16, enable=False)
evaluation = dict(
    save_best='auto',
    interval=eval_step,
    metric='mAP')
