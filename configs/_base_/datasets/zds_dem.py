# dataset settings
dataset_type = 'LavaDataset'
data_root = '/root/autodl-tmp/roaddataset/karst_datasets'
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomMultiResize',
        scale=(1024, 256),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomMultiCrop', crop_size=crop_size),#, cat_max_ratio=0.75),
    dict(type='RandomMultiFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    # dict(type='ConcatCDInput'),
    dict(type='PackMultiSegInputs')
]
test_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='MultiResize', scale=(256, 256), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackMultiSegInputs')
]



img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
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
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/train', seg_map_path='annotations/train', dem_path='dems/train'),
        img_suffix = '.tif',
        seg_map_suffix = '.tif',
        dem_suffix = '.tif',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/val', seg_map_path='annotations/val', dem_path='dems/val'),
        img_suffix = '.tif',
        seg_map_suffix = '.tif',
        dem_suffix = '.tif',
        pipeline=test_pipeline))
# test_dataloader = val_dataloader
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/test', seg_map_path='annotations/test', dem_path='dems/test'),
        img_suffix = '.tif',
        seg_map_suffix = '.tif',
        dem_suffix = '.tif',
        pipeline=test_pipeline))


val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU','mFscore', 'mDice'],
    metric_items=['mIoU','mFscore', 'mDice']
)

test_evaluator = val_evaluator
