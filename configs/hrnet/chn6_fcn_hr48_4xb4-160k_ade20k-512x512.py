_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/chn6.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
# model = dict(
#     data_preprocessor=data_preprocessor, decode_head=dict(num_classes=150))


model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        num_classes=2,
        in_channels=[48, 96, 192, 384], channels=sum([48, 96, 192, 384])),
    # test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341))
    )

train_dataloader = dict(batch_size=4,num_workers=4)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader