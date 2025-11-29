_base_ = [
    '../_base_/models/dcan.py',
    '../_base_/datasets/deepglobe.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='../data/pretrained_models/dsegNeXt/AttentionModuleK5D1259Cat_tiny.pth')),
    decode_head=dict(
        type='LightHamHead',
        in_channels=[64, 160, 256],
        in_index=[1, 2, 3],
        channels=256,
        ham_channels=256,
        ham_kwargs=dict(
            MD_R=16,
            SPATIAL=True,
            MD_S=1,
            MD_D=512,
            TRAIN_STEPS=6,
            EVAL_STEPS=7,
            INV_T=100,
            ETA=0.9,
            RAND_INIT=True
        ),
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    #test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(384, 384)))
    test_cfg=dict(mode='whole'))

data = dict(samples_per_gpu=8)


# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0016, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# fp16 settings
# optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# # fp16 placeholder
# fp16 = dict(loss_scale='dynamic')
#bash ./tools/dist_train.sh './myconfigs/van/b0/van.b0.512x512.OPM.20k.py' 2

#bash ./tools/dist_train.sh './myconfigs/segNeXt/tiny/segnext.tiny.512x512.deepglobe.50k.py' 2