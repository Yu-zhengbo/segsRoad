"""
Author: Chenhongyi Yang
"""
from typing import Sequence

import copy
import numpy as np
import torch
import torch.nn as nn

from mmcv.cnn import build_norm_layer
from mmengine.model.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
from mmengine.model import ModuleList, BaseModule

from mmseg.registry import MODELS
from mmengine.utils import to_2tuple

import torch.nn.functional as F
from mmseg.models.backbones.mamba.plainmamba.patch_embed import ConvPatchEmbed
from mmseg.models.backbones.mamba.plainmamba.plain_layer import PlainMambaLayer


def resize_pos_embed(pos_embed,
                     src_shape,
                     dst_shape,
                     mode='bicubic',
                     num_extra_tokens=1):
    """Resize pos_embed weights.

    Args:
        pos_embed (torch.Tensor): Position embedding weights with shape
            [1, L, C].
        src_shape (tuple): The resolution of downsampled origin training
            image, in format (H, W).
        dst_shape (tuple): The resolution of downsampled new training
            image, in format (H, W).
        mode (str): Algorithm used for upsampling. Choose one from 'nearest',
            'linear', 'bilinear', 'bicubic' and 'trilinear'.
            Defaults to 'bicubic'.
        num_extra_tokens (int): The number of extra tokens, such as cls_token.
            Defaults to 1.

    Returns:
        torch.Tensor: The resized pos_embed of shape [1, L_new, C]
    """
    if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1]:
        return pos_embed
    assert pos_embed.ndim == 3, 'shape of pos_embed must be [1, L, C]'
    _, L, C = pos_embed.shape
    src_h, src_w = src_shape
    assert L == src_h * src_w + num_extra_tokens, \
        f"The length of `pos_embed` ({L}) doesn't match the expected " \
        f'shape ({src_h}*{src_w}+{num_extra_tokens}). Please check the' \
        '`img_size` argument.'
    extra_tokens = pos_embed[:, :num_extra_tokens]

    src_weight = pos_embed[:, num_extra_tokens:]
    src_weight = src_weight.reshape(1, src_h, src_w, C).permute(0, 3, 1, 2)

    dst_weight = F.interpolate(
        src_weight, size=dst_shape, align_corners=False, mode=mode)
    dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)

    return torch.cat((extra_tokens, dst_weight), dim=1)


@MODELS.register_module()
class PlainMamba(BaseModule):
    arch_zoo = {
        'L1': {
            'patch_size': 16,
            'embed_dims': 192,
            'num_layers': 24,
            'num_convs_patch_embed': 1,
            'layers_with_dwconv': [0],  # useful for L1 model
            'layer_cfgs': {
                'use_rms_norm': False,
                'mamba_cfg': {
                    'd_state': 16,
                    'expand': 2,
                    'conv_size': 7,
                    'dt_init': "random",
                    'conv_bias': True,
                    'bias': True,
                    'default_hw_shape': (224 // 16, 224 // 16)
                }
            }
        },
        'L2': {
            'patch_size': 16,
            'embed_dims': 384,
            'num_layers': 24,
            'num_convs_patch_embed': 2,
            'layers_with_dwconv': [],
            'layer_cfgs': {
                'use_rms_norm': False,
                'mamba_cfg': {
                    'd_state': 16,
                    'expand': 2,
                    'conv_size': 7,
                    'dt_init': "random",
                    'conv_bias': True,
                    'bias': True,
                    'default_hw_shape': (224 // 16, 224 // 16)
                }
            }
        },
        'L3': {
            'patch_size': 16,
            'embed_dims': 448,
            'num_layers': 36,
            'num_convs_patch_embed': 2,
            'layers_with_dwconv': [],
            'layer_cfgs': {
                'use_rms_norm': False,
                'mamba_cfg': {
                    'd_state': 16,
                    'expand': 2,
                    'conv_size': 7,
                    'dt_init': "random",
                    'conv_bias': True,
                    'bias': True,
                    'default_hw_shape': (224 // 16, 224 // 16)
                }
            }
        },
    }

    def __init__(self,
                 img_size=224,
                 in_channels=3,
                 arch=None,
                 patch_size=16,
                 embed_dims=192,
                 num_layers=20,
                 num_convs_patch_embed=1,
                 with_pos_embed=True,
                 out_indices=-1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 interpolate_mode='bicubic',
                 layer_cfgs=dict(),
                 layers_with_dwconv=[],
                 init_cfg=None,
                 test_cfg=dict(),
                 convert_syncbn=False,
                 freeze_patch_embed=False,
                 **kwargs):
        super(PlainMamba, self).__init__(init_cfg)

        self.test_cfg = test_cfg

        self.img_size = to_2tuple(img_size)
        self.convert_syncbn = convert_syncbn

        self.arch = arch
        if self.arch is None:
            self.embed_dims = embed_dims
            self.num_layers = num_layers
            self.patch_size = patch_size
            self.num_convs_patch_embed = num_convs_patch_embed
            self.layers_with_dwconv = layers_with_dwconv
            _layer_cfgs = layer_cfgs
        else:
            assert self.arch in self.arch_zoo.keys()
            self.embed_dims = self.arch_zoo[self.arch]['embed_dims']
            self.num_layers = self.arch_zoo[self.arch]['num_layers']
            self.patch_size = self.arch_zoo[self.arch]['patch_size']
            self.num_convs_patch_embed = self.arch_zoo[self.arch]['num_convs_patch_embed']
            self.layers_with_dwconv = self.arch_zoo[self.arch]['layers_with_dwconv']
            _layer_cfgs = self.arch_zoo[self.arch]['layer_cfgs']

        self.with_pos_embed = with_pos_embed
        self.interpolate_mode = interpolate_mode
        self.freeze_patch_embed = freeze_patch_embed
        _drop_path_rate = drop_path_rate

        self.patch_embed = ConvPatchEmbed(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            num_convs=self.num_convs_patch_embed,
            patch_size=self.patch_size,
            stride=self.patch_size
        )

        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # Set position embedding
        if with_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dims))
            trunc_normal_(self.pos_embed, std=0.02)
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices

        # stochastic depth decay rule
        dpr = np.linspace(0, _drop_path_rate, self.num_layers)
        self.drop_path_rate = _drop_path_rate

        self.layer_cfgs = _layer_cfgs

        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [copy.deepcopy(_layer_cfgs) for _ in range(self.num_layers)]

        for i in range(self.num_layers):
            _layer_cfg_i = layer_cfgs[i]
            _layer_cfg_i.update({
                "embed_dims": self.embed_dims,
                "drop_path_rate": dpr[i]
            })
            if i in self.layers_with_dwconv:
                _layer_cfg_i.update({"with_dwconv": True})
            else:
                _layer_cfg_i.update({"with_dwconv": False})
            self.layers.append(
                PlainMambaLayer(**_layer_cfg_i)
            )

        self.final_norm = final_norm
        # assert final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, self.embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)

        for i in out_indices:
            if i != self.num_layers - 1:
                if norm_cfg is not None:
                    norm_layer = build_norm_layer(norm_cfg, self.embed_dims)[1]
                else:
                    norm_layer = nn.Identity()
                self.add_module(f'norm_layer{i}', norm_layer)
        self.load_state_dict_unstrict(init_cfg)
    def load_state_dict_unstrict(self, init_cfg):
        state_dict = init_cfg.get('checkpoint', None)
        if state_dict:
            state_dict = torch.load(state_dict, map_location='cpu')['state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict['.'.join(k.split('.')[1:])] = v
            missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
            print(f"Missing keys: {missing_keys}")
            print(f"Unexpected keys: {unexpected_keys}")
    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def init_weights(self):
        super(PlainMamba, self).init_weights()
        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            if self.with_pos_embed:
                trunc_normal_(self.pos_embed, std=0.02)
        self.load_state_dict_unstrict(self.init_cfg)
        self.set_freeze_patch_embed()

    def set_freeze_patch_embed(self):
        if self.freeze_patch_embed:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

    def forward(self, x):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        if self.with_pos_embed:
            pos_embed = resize_pos_embed(
                self.pos_embed,
                self.patch_resolution,
                patch_resolution,
                mode=self.interpolate_mode,
                num_extra_tokens=0
            )
            x = x + pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, hw_shape=patch_resolution)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

            if i in self.out_indices:
                B, _, C = x.shape
                patch_token = x.reshape(B, *patch_resolution, C)
                if i != self.num_layers - 1:
                    norm_layer = getattr(self, f'norm_layer{i}')
                    patch_token = norm_layer(patch_token)
                patch_token = patch_token.permute(0, 3, 1, 2)
                outs.append(patch_token)
        return x, tuple(outs)
