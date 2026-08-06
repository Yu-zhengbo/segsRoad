_base_ = [
    '../_base_/datasets/chn6_sam.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]


randomness = dict(seed=1825026254) #2053899318 1489201871
crop_size = (560, 560)
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
        # type='SAM3Vit',
        type='SAM3WindowRegister',
        num_register_tokens=4,
        num_local_register_tokens=2,
        freeze_base=True,
    ),
    decode_head=dict(
        type='ConcatMFCADyPCADecoder',
        in_channels=1024,
        decoder_channels=384,
        num_layers=4,
        mfca_dct_size=7,
        mfca_frequency_branches=8,
        mfca_frequency_selection='top',
        mfca_reduction=16,
        
        concat_mfca_dct_size=7,
        concat_mfca_frequency_branches=8,
        concat_mfca_frequency_selection='top',
        concat_mfca_reduction=16,
        dropout_ratio=0.,
        num_classes=7,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        # loss_decode=[
        #     dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=False, loss_weight=1.0),
            # dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)
        # ]),
    # auxiliary_head=dict(
    #     type='FCNHead',
    #     in_channels=1024,
    #     in_index=2,
    #     channels=256,
    #     num_convs=1,
    #     concat_input=False,
    #     dropout_ratio=0.1,
    #     num_classes=7,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
    #     ),
        # loss_decode = dict(type='ConnectivityLoss', loss_weight=0.4, num_seeds=32,
        #                    num_steps=10, downsample_scale=0.5)),   
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=8, num_workers=8)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

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
# train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=1000)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=80000, save_best='mIoU'),
)
