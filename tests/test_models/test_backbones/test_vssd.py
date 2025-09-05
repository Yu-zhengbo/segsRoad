# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

# from mmseg.models.backbones.dswin import SwinBlock, DSwinTransformer
# from mmseg.models.backbones.vmamba import Backbone_VSSM
# from mmseg.models.backbones.deformable_vmamba import DeformableVMAMBA
from mmseg.models.backbones.vssd import Backbone_VMAMBA2


def test_swin_transformer():
    """Test Swin Transformer backbone."""
    # Test absolute position embedding
    temp = torch.randn((1, 3, 512, 512)).cuda()
    model = Backbone_VMAMBA2(
        out_indices=(0, 1, 2, 3),
        pretrained="change to the path of the pretrained model",
        embed_dim=64,
        depths=(3, 4, 21, 5),
        num_heads=(2, 4, 8, 16),
        simple_downsample=False,
        simple_patch_embed=False,
        ssd_expand=2,
        ssd_chunk_size=256,
        linear_attn_duality=True,
        attn_types=['mamba2', 'mamba2', 'mamba2', 'standard'],
        bidirection=False,
        drop_path_rate=0.4,
        d_state=64,
        ssd_positve_dA=True,
        key='model_ema'
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