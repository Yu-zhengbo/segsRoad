_base_ = './potsdam_upernet_r50_4xb4-40k_voc12aug-512x512.py'
model = dict(pretrained='/data/openclaw/segsRoad/checkpoints/resnet152_v1d.pth', backbone=dict(depth=152))
