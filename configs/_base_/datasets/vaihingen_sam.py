# dataset settings
dataset_type = 'ISPRSDataset'
data_root = '/data/datasets/rs/Vaihingen'
crop_size = (518, 518)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(518, 518),
        ratio_range=(0.75, 1.5),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.95),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [dict(type='RandomRotate', prob=1.0, degree=(0, 0),
                pad_val=0, seg_pad_val=255)],
            [dict(type='RandomRotate', prob=1.0, degree=(90, 90),
                pad_val=0, seg_pad_val=255)],
            [dict(type='RandomRotate', prob=1.0, degree=(180, 180),
                pad_val=0, seg_pad_val=255)],
            [dict(type='RandomRotate', prob=1.0, degree=(270, 270),
                pad_val=0, seg_pad_val=255)],
        ]
    ),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(560, 560), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]
# img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
img_ratios = [1.0]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale=(560,560), keep_ratio=True),
                dict(type='Resize', scale=(518,518), keep_ratio=True),
                dict(type='Resize', scale=(700,700), keep_ratio=True),
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='vertical'),
                dict(type='RandomFlip', prob=1., direction='diagonal')
            ], [dict(type='LoadAnnotations', reduce_zero_label=True)], [dict(type='PackSegInputs')]
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
            img_path='images/train', seg_map_path='annotations/train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/val', seg_map_path='annotations/val'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU','mFscore', 'mDice'],
    metric_items=['mIoU','mFscore', 'mDice'],
    selected_classes=[
        'impervious_surface',
        'building',
        'low_vegetation',
        'tree',
        'car',
    ],
)

test_evaluator = val_evaluator