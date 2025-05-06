_base_ = [
    '../_base_/datasets/deepglobe.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

crop_size = (512, 512)
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='VisionMambaSeg',
        img_size=512, 
        patch_size=16, 
        in_chans=3,
        embed_dim=384, 
        depth=24,
        out_indices=[5, 11, 17, 23],
        pretrained='./weights/vim_s_midclstok_ft_81p6acc.pth',
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        if_abs_pos_embed=True,
        if_rope=True,
        if_rope_residual=True,
        # if_bimamba=True,
        bimamba_type="v2",
        final_pool_type='all',
        if_cls_token=True,
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[384, 384, 384, 384],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=384,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
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
    test_cfg = dict(mode='slide', crop_size=crop_size, stride=(341, 341))
    )

find_unused_parameters = True

optimizer = dict(_delete_=True, 
                 type='AdamW', 
                 lr=1e-4, 
                 betas=(0.9, 0.999), 
                 weight_decay=0.05,
                 constructor='LayerDecayOptimizerConstructor', 
                 paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.92)
                )

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 4 GPUs with 8 images per GPU
# data=dict(samples_per_gpu=8, workers_per_gpu=16)

runner = dict(type='IterBasedRunnerAmp')

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=False,
)

train_dataloader = dict(batch_size=12, num_workers=6)
val_dataloader = dict(batch_size=1, num_workers=1)
