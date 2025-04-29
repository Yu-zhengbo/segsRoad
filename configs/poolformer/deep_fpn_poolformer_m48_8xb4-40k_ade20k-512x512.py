_base_ = [
    '../_base_/models/fpn_poolformer_s12.py', 
    '../_base_/datasets/deepglobe.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/poolformer/poolformer-m48_3rdparty_32xb128_in1k_20220414-9378f3eb.pth'  # noqa

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
norm_cfg = dict(type='SyncBN', requires_grad=True)
# model settings
model = dict(
    data_preprocessor=data_preprocessor,
        backbone=dict(
        arch='m48',
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    neck=dict(in_channels=[96, 192, 384, 768]),
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
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
    # test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341))
    )

# optimizer
# optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, weight_decay=0.0001)
# optimizer_config = dict()
# # learning policy
# lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05))
# param_scheduler = [
#     dict(
#         type='PolyLR',
#         power=0.9,
#         begin=0,
#         end=40000,
#         eta_min=0.0,
#         by_epoch=False,
#     )
# ]


param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=80000,
        eta_min=0.0,
        by_epoch=False,
    )
]


train_dataloader = dict(batch_size=4,num_workers=4)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader