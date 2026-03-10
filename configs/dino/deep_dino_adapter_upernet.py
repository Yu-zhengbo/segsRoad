_base_ = [
    '../_base_/datasets/deepglobe.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]


crop_size = (512, 512)
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    # mean=[123.675, 116.28, 103.53],
    # std=[58.395, 57.12, 57.375],
    mean = [109.65, 104.805, 75.48],
    std = [54.315, 39.78, 36.465],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='DINOAdapter',
        # model = 'vit_large_patch16_dinov3_qkvb.sat493m',
        model = 'vit_7b_patch16_dinov3.sat493m',
        embed_dim = 4096,
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        # interaction_indexes=[[0, 11], [12, 17], [18, 20], [21, 23]],
        freeze = True,
    ),
    decode_head=dict(
        type='UPerHead',
        # in_channels=[1024, 1024, 1024, 1024],
        in_channels=[4096, 4096, 4096, 4096],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=1024,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    
    
    # neck=dict(
    #     type='MLANeck',
    #     in_channels=[1024, 1024, 1024, 1024],
    #     out_channels=256,
    #     norm_cfg=norm_cfg,
    #     act_cfg=dict(type='ReLU'),
    # ),
    # decode_head=dict(
    #     type='SETRMLAHead',
    #     in_channels=(256, 256, 256, 256),
    #     channels=512,
    #     in_index=(0, 1, 2, 3),
    #     dropout_ratio=0,
    #     mla_channels=128,
    #     num_classes=2,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    
    # decode_head=dict(
    #     type='FCNHead',
    #     in_channels=1024,
    #     in_index=0,
    #     channels=512,
    #     num_convs=2,
    #     concat_input=True,
    #     dropout_ratio=0.1,
    #     num_classes=2,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=[
    #         dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    #         # dict(type='ConnectivityLoss', loss_weight=1.0, num_seeds=32, num_steps=10,
    #         #      downsample_scale=0.25)
    #         # dict(
    #         #     type='NeighborLoss',
    #         #     neigh_size=3,
    #         #     k = 1
    #         # )
    #     ]),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=4096,
        in_index=2,
        channels=1024,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
        ),
        # loss_decode = dict(type='ConnectivityLoss', loss_weight=0.4, num_seeds=32,
        #                    num_steps=10, downsample_scale=0.5)
        # ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))



train_dataloader = dict(batch_size=1, num_workers=1)
val_dataloader = dict(batch_size=1, num_workers=1)

optim_wrapper = dict(
    _delete_=True,
    # type='OptimWrapper',
    # type='AmpOptimWrapper',   # ✅ AMP
    # loss_scale='dynamic', 
    # accumulative_counts=6,
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
        end=80000,
        by_epoch=False,
    )
]



default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=80000),
)