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
        keys=['img', 'lane_line', 'seg',],
        img_info = (img_h,img_w),
        num_points = 72,
        max_lanes = 4,
        meta_keys = ['img_metas'],
        transforms=[
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
            dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.5),
            dict(name='ChannelShuffle', parameters=dict(p=1.0), p=0.1),
            dict(name='MultiplyAndAddToBrightness',
                 parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
                 p=0.6),
            dict(name='AddToHueAndSaturation',
                 parameters=dict(value=(-10, 10)),
                 p=0.7),
            dict(name='OneOf',
                 transforms=[
                     dict(name='MotionBlur', parameters=dict(k=(3, 5))),
                     dict(name='MedianBlur', parameters=dict(k=(3, 5)))
                 ],
                 p=0.2),
            dict(name='Affine',
                 parameters=dict(translate_percent=dict(x=(-0.1, 0.1),
                                                        y=(-0.1, 0.1)),
                                 rotate=(-10, 10),
                                 scale=(0.8, 1.2)),
                 p=0.7),
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
        ]
    ),
    dict(type='ToTensor_', keys=['img', 'lane_line', 'seg', 'img_metas']),
]

test_pipeline = [
    dict(type='GenerateLaneLine',
         keys=['img',],
         meta_keys = ['img_metas'],
         img_info = (img_h,img_w),
         num_points = 72,
         max_lanes = 4,
         transforms=[
             dict(name='Resize',
                  parameters=dict(size=dict(height=img_h, width=img_w)),
                  p=1.0),
         ],
         training=False),
    dict(type='ToTensor_', keys=['img','img_metas']),
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        split = 'train',
        cut_height = 270,
        img_fo = (ori_img_h,ori_img_w),
        resize_img_info = (img_h,img_w),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        split = 'test',
        cut_height = 270,
        img_fo = (ori_img_h,ori_img_w),
        resize_img_info = (img_h,img_w),
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        split = 'test',
        cut_height = 270,
        img_fo = (ori_img_h,ori_img_w),
        resize_img_info = (img_h,img_w),
        pipeline=test_pipeline))
work_dir = './output_dir'
evaluation = dict(interval=1,output_basedir = work_dir,save_best = 'auto')