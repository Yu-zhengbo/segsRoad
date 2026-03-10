_base_ = [
    '../_base_/datasets/geotherm.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (32, 32)
data_preprocessor = dict(
    type='SegMultiDataPreProcessor',
    # mean=[123.675, 116.28, 103.53],
    # std=[58.395, 57.12, 57.375],
    dem_mean=[8.8619833e+00,  3.5408161e+01,  2.1858746e-02,  8.8520403e+00,
        -5.5325252e-01,  1.4163264e+02, -4.7869267e+00,  5.8981557e+00],
    dem_std=[6.60850525e+00, 3.47019882e+01, 2.09999345e-02, 8.41524315e+00,
        5.86972177e-01, 1.00210724e+02, 4.77712708e+01, 5.54549932e+00],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)


'''
model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64))
'''

model = dict(
    type='MultiEncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet18_v1c',
    pretrained_dem='open-mmlab://resnet18_v1c',
    fusion_mode='add',
    backbone=dict(
        type='ResNetV1c',
        in_channels=64,
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 1, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    backbone_dem=dict(
        type='ResNetV1c',
        in_channels=8,
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 1, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='FCNHead',
        in_channels=512,
        in_index=3,
        channels=128,
        num_convs=2,
        concat_input=True,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=2,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))



optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    # paramwise_cfg=dict(
    #     custom_keys={
    #         'pos_block': dict(decay_mult=0.),
    #         'norm': dict(decay_mult=0.),
    #         'head': dict(lr_mult=10.)
    #     })
    )

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
train_dataloader = dict(batch_size=16, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=1)
test_dataloader = val_dataloader
