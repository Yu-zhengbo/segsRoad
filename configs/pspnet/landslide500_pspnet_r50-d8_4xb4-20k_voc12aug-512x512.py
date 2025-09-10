_base_ = [
    '../_base_/models/pspnet_r50-d8.py',
    '../_base_/datasets/landslide500.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2))

train_dataloader = dict(batch_size=8, num_workers=4)
val_dataloader = dict(batch_size=1)