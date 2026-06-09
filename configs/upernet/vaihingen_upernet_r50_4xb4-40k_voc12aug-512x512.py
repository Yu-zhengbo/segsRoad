_base_ = [
    '../_base_/models/upernet_r50.py',
    '../_base_/datasets/vaihingen.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=6),
    auxiliary_head=dict(num_classes=6))

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=40000,save_best='mIoU'),)
