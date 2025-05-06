# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

# from mmseg.models.backbones.dswin import SwinBlock, DSwinTransformer
from mmseg.models.backbones.localvmamba import Backbone_LocalVSSM


def test_swin_transformer():
    """Test Swin Transformer backbone."""
    directions = [
            ['h', 'h_flip', 'w7', 'w7_flip'],
            ['h_flip', 'v_flip', 'w2', 'w2_flip'],
            ['h_flip', 'v_flip', 'w2_flip', 'w7'],
            ['h_flip', 'v', 'v_flip', 'w2'],
            ['h', 'h_flip', 'v_flip', 'w2_flip'],
            ['h_flip', 'v_flip', 'w2', 'w2_flip'],
            ['h', 'w2_flip', 'w7', 'w7_flip'],
            ['h', 'h_flip', 'v', 'v_flip'],
            ['h', 'v_flip', 'w7', 'w7_flip'],
            ['h_flip', 'v', 'w2', 'w7_flip'],
            ['v', 'v_flip', 'w2', 'w7_flip'],
            ['h', 'h_flip', 'v_flip', 'w2_flip'],
            ['v_flip', 'w2_flip', 'w7', 'w7_flip'],
            ['h_flip', 'v_flip', 'w2_flip', 'w7_flip'],
            ['h_flip', 'v', 'w7', 'w7_flip'],
    ]
    # Test absolute position embedding
    temp = torch.randn((1, 3, 512, 512)).cuda()
    model = Backbone_LocalVSSM(        directions=directions,
        out_indices=(0, 1, 2, 3),
        # pretrained="https://github.com/hunto/LocalMamba/releases/download/v1.0.0/local_vssm_tiny.ckpt",
        dims=96,
        depths=(2, 2, 9, 2),
        ssm_d_state=16,
        ssm_dt_rank="auto",
        ssm_ratio=2.0,
        mlp_ratio=0.0,
        downsample_version="v1",
        patchembed_version="v1",
        drop_path_rate=0.1,).cuda()#, use_abs_pos_embed=True)
    # model.init_weights()
    model(temp)

    # model.init_weights()
    model.train()
    out = model(temp)
    for o in out:
        print(o.shape)


if __name__ == '__main__':
    test_swin_transformer()