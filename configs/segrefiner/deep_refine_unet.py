_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/deep_segrefiner.py',
    '../_base_/schedules/schedule_40k.py'
]
crop_size = (512, 512)
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegMultiDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # dem_mean=[4303.082],
    # dem_std=[271.244],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=0,
    size=crop_size)
model = dict(
    type='SegRefiner',
    data_preprocessor=data_preprocessor,
    step=6,
    denoise_model=dict(
        type='DeformableHeadWithTime',
        in_channels=[4],
        channels=256,
        in_index=[0],
        dropout_ratio=0.,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        num_feature_levels=1,
        encoder=dict(
            type='DetrTransformerEncoder',
            num_layers=6,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                use_time_mlp=True,
                attn_cfgs=dict(
                    type='MultiScaleDeformableAttention',
                    embed_dims=256,
                    num_levels=1,
                    num_heads=8,
                    dropout=0.),
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=256,
                    feedforward_channels=1024,
                    ffn_drop=0.,
                    act_cfg=dict(type='GELU')),
                operation_order=('self_attn', 'norm', 'ffn', 'norm'))
        ),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)),
    # denoise_model=dict(
    #     type='DenoiseUNet',
    #     in_channels=4,
    #     out_channels=1,
    #     model_channels=128,
    #     num_res_blocks=2,
    #     num_heads=4,
    #     num_heads_upsample=-1,
    #     attention_strides=(16, 32),
    #     learn_time_embd=True,
    #     channel_mult = (1, 1, 2, 2, 4, 4),
    #     dropout=0.0,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    diffusion_cfg=dict(
        betas=dict(
            type='linear',
            start=0.8,
            stop=0,
            num_timesteps=6),
        diff_iter=False),
    # model training and testing settings
    test_cfg=dict()) 



optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.05),
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
# train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=100)
# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=3, num_workers=3)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader

# train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=500)