# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

# from mmseg.models.backbones.dswin import SwinBlock, DSwinTransformer
from mmseg.models.backbones.plainmamba import PlainMambaSeg
checkpoint_url = "https://huggingface.co/ChenhongyiYang/PlainMamba/resolve/main/l1.pth"

def test_swin_transformer():
    """Test Swin Transformer backbone."""

    # Test absolute position embedding
    temp = torch.randn((1, 3, 512, 512)).cuda()
    model = PlainMambaSeg(arch="L1",
        out_indices=(5, 11, 17, 23),
        drop_path_rate=0.1,
        final_norm=True,
        convert_syncbn=True,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_url,  prefix="backbone."),).cuda()#, use_abs_pos_embed=True)
    # model.init_weights()
    model(temp)

    # model.init_weights()
    model.train()
    out = model(temp)
    for o in out:
        print(o.shape)


if __name__ == '__main__':
    test_swin_transformer()