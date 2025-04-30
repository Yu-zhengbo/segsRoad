# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

# from mmseg.models.backbones.dswin import SwinBlock, DSwinTransformer
from mmseg.models.backbones.vmamba import Backbone_VSSM


def test_swin_transformer():
    """Test Swin Transformer backbone."""

    # Test absolute position embedding
    temp = torch.randn((1, 3, 512, 512)).cuda()
    model = Backbone_VSSM(        out_indices=(0, 1, 2, 3),
        pretrained="weights/upernet_vssm_4xb4-160k_ade20k-512x512_base_iter_160000.pth",
        # copied from classification/configs/vssm/vssm_base_224.yaml
        dims=128,
        depths=(2, 2, 15, 2),
        ssm_d_state=1,
        ssm_dt_rank="auto",
        ssm_ratio=2.0,
        ssm_conv=3,
        ssm_conv_bias=False,
        forward_type="v05_noz", # v3_noz,
        mlp_ratio=4.0,
        downsample_version="v3",
        patchembed_version="v2",
        drop_path_rate=0.6,
        norm_layer="ln2d",).cuda()#, use_abs_pos_embed=True)
    # model.init_weights()
    model(temp)

    # model.init_weights()
    model.train()
    out = model(temp)
    for o in out:
        print(o.shape)


if __name__ == '__main__':
    test_swin_transformer()