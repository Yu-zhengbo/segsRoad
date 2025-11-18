_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/landslide500.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=2),
    # auxiliary_head=dict(num_classes=2),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
train_dataloader = dict(batch_size=6, num_workers=6)
val_dataloader = dict(batch_size=1, num_workers=1)
test_dataloader = val_dataloader
