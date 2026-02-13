_base_ = [
    '../_base_/models/dcan.py',
    '../_base_/datasets/mass.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
find_unused_parameters = True

crop_size = (512, 512)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=0,
    size=crop_size,)
    # test_cfg=dict(size_divisor=32))

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        embed_dims=[64, 128, 320, 512],
        depths=[3, 3, 12, 3],
        #init_cfg=None,
        init_cfg=dict(type='Pretrained', checkpoint='/home/cz/datasets/segsroad_model_weights/dsegnext/AttentionModuleK5D1259Cat_base.pth'),
        drop_path_rate=0.1),
    decode_head=dict(
        type='LightHamHead',
        in_channels=[128, 320, 512],
        in_index=[1, 2, 3],
        channels=512,
        ham_channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,class_weight=[1.0,3.0])),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
    #test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

# data
train_dataloader = dict(batch_size=12, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=1)
test_dataloader = val_dataloader
# evaluation = dict(interval=8000, metric='mIoU')
# checkpoint_config = dict(by_epoch=False, interval=8000)
# optimizer


# optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
#                  paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
#                                                  'norm': dict(decay_mult=0.),
#                                                  'head': dict(lr_mult=10.)
#                                                  }))

# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
#     dict(
#         type='PolyLR',
#         eta_min=0.0,
#         power=1.0,
#         begin=1500,
#         end=40000,
#         by_epoch=False,
#     )
# ]


optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={'pos_block': dict(decay_mult=0.),
                    'head': dict(decay_mult=10.),
                    'norm': dict(decay_mult=0.)}))

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

# train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=400)