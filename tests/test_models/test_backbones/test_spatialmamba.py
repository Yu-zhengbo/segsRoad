# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

# from mmseg.models.backbones.dswin import SwinBlock, DSwinTransformer
# from mmseg.models.backbones.vmamba import Backbone_VSSM
# from mmseg.models.backbones.deformable_vmamba import DeformableVMAMBA
from mmseg.models.backbones.spatialmamba import Backbone_SpatialMamba


def test_swin_transformer():
    """Test Swin Transformer backbone."""
    # Test absolute position embedding
    temp = torch.randn((1, 3, 512, 512)).cuda()
    model = Backbone_SpatialMamba(out_indices=(0, 1, 2, 3),
        pretrained="/root/autodl-tmp/segsroad_model_weights/upernet_spatialmamba_4xb4-160k_ade20k-512x512_base_iter_160000.pth",
        dims=96,
        d_state=1,
        depths=(2, 4, 21, 5),
        drop_path_rate=0.5).cuda()#, use_abs_pos_embed=True)
    # model.init_weights()
    model(temp)

    # model.init_weights()
    model.train()
    out = model(temp)
    for o in out:
        print(o.shape)


if __name__ == '__main__':
    test_swin_transformer()