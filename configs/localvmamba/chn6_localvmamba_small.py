_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/chn6.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_25k.py'
]
crop_size = (512, 512)
# custom_imports = dict(imports=['model'])

directions = [
        ['h', 'v', 'v_flip', 'w7_flip'],
        ['h', 'h_flip', 'v', 'w7'],
        ['h_flip', 'v', 'v_flip', 'w7'],
        ['h_flip', 'v', 'w2_flip', 'w7'],
        ['h_flip', 'v', 'w2_flip', 'w7'],
        ['h_flip', 'v_flip', 'w2', 'w7_flip'],
        ['h_flip', 'v', 'v_flip', 'w7'],
        ['h', 'v', 'v_flip', 'w7'],
        ['h', 'v', 'v_flip', 'w7'],
        ['h', 'v', 'v_flip', 'w7_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'v', 'v_flip', 'w2'],
        ['v', 'v_flip', 'w2_flip', 'w7'],
        ['h', 'h_flip', 'v', 'w2'],
        ['h_flip', 'v', 'v_flip', 'w7_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'v', 'v_flip', 'w7_flip'],
        ['h', 'v', 'v_flip', 'w7_flip'],
        ['h', 'h_flip', 'v_flip', 'w7'],
        ['h_flip', 'v_flip', 'w2_flip', 'w7'],
        ['h_flip', 'v', 'v_flip', 'w7_flip'],
        ['v', 'v_flip', 'w7', 'w7_flip'],
        ['h', 'v', 'v_flip', 'w7_flip'],
        ['h_flip', 'v', 'v_flip', 'w2'],
        ['h', 'v', 'v_flip', 'w2_flip'],
        ['h', 'h_flip', 'v', 'w7'],
        ['h', 'h_flip', 'w7', 'w7_flip'],
        ['h', 'v_flip', 'w2', 'w2_flip'],
        ['h', 'v_flip', 'w2', 'w7'],
        ['h', 'v', 'v_flip', 'w7_flip'],
        ['h_flip', 'v', 'w2_flip', 'w7'],
        ['h_flip', 'v_flip', 'w7', 'w7_flip'],
        ['h', 'v', 'w7', 'w7_flip']
]

data_preprocessor = dict(size=crop_size)
# checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='Backbone_LocalVSSM',
        directions=directions,
        out_indices=(0, 1, 2, 3),
        pretrained='weights/local_vssm_small.ckpt',
        # pretrained="https://github.com/hunto/LocalMamba/releases/download/v1.0.0/local_vssm_tiny.ckpt",
        dims=96,
        depths=(2, 2, 27, 2),
        ssm_d_state=16,
        ssm_dt_rank="auto",
        ssm_ratio=2.0,
        mlp_ratio=0.0,
        downsample_version="v1",
        patchembed_version="v1",
        drop_path_rate=0.2,
    ),
    decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=2),
    auxiliary_head=dict(in_channels=384, num_classes=2))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=25000,
        by_epoch=False,
    )
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader