_base_ = [
    '../_base_/datasets/deepglobe.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

# model settings
crop_size = (512, 512)
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=0,
    size=crop_size)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='MKUNet',
        pretrained='/home/cz/codes/segsRoad/work_dirs/mkunet/pretrainedOnAllWithoutSpace160k.pth',
        in_channels=3,
        channels=[16,32,64,96,160],
        depths=[2, 2, 2, 2, 2],
        kernel_sizes=[1,3,5],
        expansion_factor=2,
        ),
    decode_head=dict(
        type='MKHead',
        gag_kernel=3,
        expansion_factor=2,
        kernel_sizes=[1,3,5],
        in_channels=[16,32,64,96,160],
        in_index=[0,1,2,3,4],
        num_stages=[2,2,2,2,2],
        channels=16,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[1.0, 1.0])),
    # auxiliary_head=dict(
    #     type='FCNHead',
    #     in_channels=128,
    #     in_index=3,
    #     channels=64,
    #     num_convs=1,
    #     concat_input=False,
    #     dropout_ratio=0.1,
    #     num_classes=2,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    # test_cfg=dict(mode='slide', crop_size=256, stride=170))
    test_cfg=dict(mode='whole'))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0006, betas=(0.9, 0.999), weight_decay=0.01),
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
# train_dataloader = dict(batch_size=32, num_workers=16)
train_dataloader = dict(batch_size=8, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=1)
test_dataloader = val_dataloader
