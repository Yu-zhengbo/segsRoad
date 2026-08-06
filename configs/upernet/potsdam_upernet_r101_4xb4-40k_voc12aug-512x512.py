_base_ = [
    '../_base_/models/upernet_r50.py',
    '../_base_/datasets/potsdam.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

crop_size = (512,512)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101),
    decode_head=dict(in_channels=[256, 512, 1024, 2048], num_classes=6),
    auxiliary_head=dict(in_channels=1024, num_classes=6))

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=40000,save_best='mIoU'),)