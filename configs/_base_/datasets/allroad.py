dataset_type = 'RoadDataset'
crop_size = (512, 512)


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

# 定义 mass 数据集配置
mass_data_root = '/home/cz/datasets/roaddataset/mass'
mass_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
mass_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1500, 1500), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

# 定义 deepglobe 数据集配置
deepglobe_data_root = '/home/cz/datasets/roaddataset/deepglobe'
deepglobe_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
deepglobe_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]



# 定义 chn6 数据集配置
chn6_data_root = '/home/cz/datasets/roaddataset/chn6'
chn6_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
chn6_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

# 定义GRset train_pipeline
grset_data_root = '/home/cz/datasets/roaddataset/Dataset_public'
grset_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

# 定义jilin train_pipeline
jilin_data_root = '/data1/datasets/zhengbo/jilin'
jilin_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

# 定义spacenet3 train和test pipeline
spacenet3_data_root = '/data1/datasets/zhengbo/spacenet3'
spacenet3_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
spacenet3_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1300, 1300), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]


# 定义spacenet5 train pipeline
spacenet5_data_root = '/data1/datasets/zhengbo/spacenet5'
spacenet5_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]



# 合并两个数据集的训练数据
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type=dataset_type,
                data_root=mass_data_root,
                data_prefix=dict(
                    img_path='images/train',
                    seg_map_path='annotations/train'),
                pipeline=mass_train_pipeline
            ),
            dict(
                type=dataset_type,
                data_root=deepglobe_data_root,
                data_prefix=dict(
                    img_path='images/train',
                    seg_map_path='annotations/train'),
                pipeline=deepglobe_train_pipeline
            ),
            dict(
                type=dataset_type,
                data_root=chn6_data_root,
                data_prefix=dict(
                    img_path='images/train',
                    seg_map_path='annotations/train'),
                pipeline=chn6_train_pipeline
            ),
            dict(
                type=dataset_type,
                data_root=grset_data_root,
                data_prefix=dict(
                    img_path='images/train',
                    seg_map_path='annotations/train'),
                pipeline=grset_train_pipeline
            ),
            dict(
                type=dataset_type,
                img_suffix='.png',
                data_root=jilin_data_root,
                data_prefix=dict(
                    img_path='images/train',
                    seg_map_path='annotations/train'),
                pipeline=jilin_train_pipeline
            ),
            # dict(
            #     type=dataset_type,
            #     data_root=spacenet3_data_root,
            #     data_prefix=dict(
            #         img_path='images/train',
            #         seg_map_path='annotations/train'),
            #     pipeline=spacenet3_train_pipeline
            # ),
            # dict(
            #     type=dataset_type,
            #     data_root=spacenet5_data_root,
            #     data_prefix=dict(
            #         img_path='images/train',
            #         seg_map_path='annotations/train'),
            #     pipeline=spacenet5_train_pipeline
            # )
        ]
    )
)


val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets = [
            # dict(
            #     type=dataset_type,
            #     data_root=mass_data_root,
            #     data_prefix=dict(img_path='images/val', seg_map_path='annotations/val'),
            #     pipeline=mass_test_pipeline,
            # ),
            dict(
                type=dataset_type,
                data_root=deepglobe_data_root,
                data_prefix=dict(img_path='images/val', seg_map_path='annotations/val'),
                pipeline=deepglobe_test_pipeline,
            ),
            # dict(
            #     type=dataset_type,
            #     data_root=chn6_data_root,
            #     data_prefix=dict(img_path='images/val', seg_map_path='annotations/val'),
            #     pipeline=chn6_test_pipeline,
            # ),
            # dict(
            #     type=dataset_type,
            #     data_root=spacenet3_data_root,
            #     data_prefix=dict(img_path='images/val', seg_map_path='annotations/val'),
            #     pipeline=spacenet3_test_pipeline,
            # ),
        ]
        ))


## deepglobe

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=deepglobe_data_root,
        data_prefix=dict(img_path='images/val', seg_map_path='annotations/val'),
        pipeline=deepglobe_test_pipeline))



## mass

# test_dataloader = dict(
#     batch_size=1, 
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=mass_data_root,
#         data_prefix=dict(img_path='images/test', seg_map_path='annotations/test'),
#         pipeline=mass_test_pipeline))


## chn6

# test_dataloader = dict(
#     batch_size=1,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=chn6_data_root,
#         data_prefix=dict(img_path='images/val', seg_map_path='annotations/val'),
#         pipeline=chn6_test_pipeline))


## spacenet3

# test_dataloader = dict(
#     batch_size=1,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=spacenet3_data_root,
#         data_prefix=dict(img_path='images/val', seg_map_path='annotations/val'),
#         pipeline=spacenet3_test_pipeline))



# img_ratios = [1.0, 1.25, 1.5, 1.75]
img_ratios = [1.0, 1.25, 1.5]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='vertical'),    # 垂直翻转
                # dict(type='RandomFlip', prob=1., direction='diagonal') 
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]


val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU','mFscore', 'mDice'],
    metric_items=['mIoU','mFscore', 'mDice']
)
test_evaluator = val_evaluator