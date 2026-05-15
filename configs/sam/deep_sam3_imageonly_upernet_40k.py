_base_ = [
    '../_base_/datasets/deepglobe.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]


crop_size = (560, 560)
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    # mean=[123.675, 116.28, 103.53],
    # std=[58.395, 57.12, 57.375],
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='SAM3',
        img_size=crop_size[0],
        precompute_resolution=crop_size[0],
        image_only=True,
        mask2former=False,
        num_class=2,
        ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[256, 256, 256],
        in_index=[0, 1, 2,],
        pool_scales=(1, 2, 3),
        channels=1024,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
    # test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)
    )

train_dataloader = dict(batch_size=4, num_workers=2)
val_dataloader = dict(batch_size=1, num_workers=1)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }
        ))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=40000,
        by_epoch=False,
    )
]


train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=40000),
)