_base_ = [
    '../_base_/datasets/loveda.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

crop_size = (560, 560)
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
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
        type='SAM3Vit',
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[1024, 1024, 1024, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))



train_dataloader = dict(batch_size=4, num_workers=4)
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
        end=20000,
        by_epoch=False,
    )
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=80000, save_best='mIoU'),
)