# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

# from mmseg.models.backbones.dswin import SwinBlock, DSwinTransformer
from mmseg.models.backbones.spiralmamba import DualMamba_tiny,DualMamba_small,DualMamba_base


def test_swin_transformer():
    """Test Swin Transformer backbone."""

    # Test absolute position embedding
    temp = torch.randn((1, 3, 512, 512)).cuda()
    # model = DAMamba_tiny(pretrained='/root/autodl-tmp/segsroad_model_weights/DAMamba-T.pth').cuda()#, use_abs_pos_embed=True)
    model = DualMamba_base(input_size=512,pretrained='/root/autodl-tmp/segsroad_model_weights/DAMamba-B.pth').cuda()#, use_abs_pos_embed=True)
    # model.init_weights()
    model(temp)

    # model.init_weights()
    model.train()
    out = model(temp)
    for o in out:
        print(o.shape)


if __name__ == '__main__':
    test_swin_transformer()