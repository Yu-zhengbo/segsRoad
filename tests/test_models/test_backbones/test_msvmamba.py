# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

# from mmseg.models.backbones.dswin import SwinBlock, DSwinTransformer
# from mmseg.models.backbones.vmamba import Backbone_VSSM
# from mmseg.models.backbones.deformable_vmamba import DeformableVMAMBA
from mmseg.models.backbones.msvmamba import MSVMamba


def test_swin_transformer():
    """Test Swin Transformer backbone."""
    # Test absolute position embedding
    temp = torch.randn((1, 3, 512, 512)).cuda()
    model = MSVMamba(
        out_indices=(0, 1, 2, 3),
        pretrained="/root/autodl-tmp/segsroad_model_weights/upernet_ms_vssm_4xb4-160k_ade20k-512x512_tiny.pth",
        dims=96,
        depths=(1, 2, 9, 2),
        ssm_d_state=1,
        ssm_dt_rank="auto",
        ssm_ratio=2.0,
        mlp_ratio=0.0,
        downsample_version="v1",
        patchembed_version="v1",
        sscore_type="multiscale_4scan_12",
        convFFN=True,
        add_se=True,
        ms_stage=[0, 1, 2, 3],
        ms_split=[1, 3],
        ffn_dropout=0.0,
        drop_path_rate=0.2,
    ).cuda()#, use_abs_pos_embed=True)
    # model.init_weights()
    model(temp)

    # model.init_weights()
    model.train()
    out = model(temp)
    for o in out:
        print(o.shape)


if __name__ == '__main__':
    test_swin_transformer()