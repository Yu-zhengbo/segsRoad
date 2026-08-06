# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmseg.registry import MODELS
from mmseg.models.utils.wrappers import resize


def _pairwise_resize(x, size, mode='bilinear'):
    """Resize helper that keeps nearest/bilinear arguments valid."""
    if mode in ('linear', 'bilinear', 'bicubic', 'trilinear'):
        return resize(x, size=size, mode=mode, align_corners=False)
    return resize(x, size=size, mode=mode)


def _scaled_size(size, scale_factor):
    h, w = size
    return (max(int(round(h * scale_factor)), 1),
            max(int(round(w * scale_factor)), 1))


def _parse_in_channels(in_channels, num_inputs):
    if isinstance(in_channels, int):
        return in_channels
    assert len(in_channels) == num_inputs
    assert len(set(in_channels)) == 1
    return in_channels[0]


def _make_out_channels(out_channels, num_scales, out_channel_mode='uniform'):
    if isinstance(out_channels, int):
        assert out_channel_mode in ('uniform', 'pyramid')
        if out_channel_mode == 'uniform':
            return [out_channels for _ in range(num_scales)]
        return [out_channels * (2**i) for i in range(num_scales)]

    assert len(out_channels) == num_scales
    return list(out_channels)


class HomogeneousLayerInteraction(BaseModule):
    """Layer interaction for same-resolution, same-channel foundation features."""

    def __init__(self,
                 channels,
                 num_inputs=4,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.num_inputs = num_inputs
        self.layer_logits = nn.Parameter(torch.eye(num_inputs))
        self.layer_convs = nn.ModuleList([
            ConvModule(
                channels,
                channels,
                kernel_size=1,
                norm_cfg=None,
                act_cfg=None,
                inplace=False) for _ in range(num_inputs)
        ])
        self.layer_scale = nn.Parameter(torch.ones(num_inputs) * 1e-3)

    def forward(self, inputs):
        assert len(inputs) == self.num_inputs
        weights = torch.softmax(self.layer_logits, dim=1)
        conv_feats = [
            layer_conv(feat)
            for layer_conv, feat in zip(self.layer_convs, inputs)
        ]

        outs = []
        for i, feat in enumerate(inputs):
            mixed = 0
            for j, conv_feat in enumerate(conv_feats):
                mixed = mixed + weights[i, j] * conv_feat
            outs.append(feat + self.layer_scale[i] * mixed)
        return outs, weights


class AdaptiveLayerToScalePyramid(BaseModule):
    """Route homogeneous backbone layers to a multi-scale feature pyramid."""

    def __init__(self,
                 in_channels,
                 out_channels=256,
                 num_inputs=4,
                 scale_factors=(4, 2, 1, 0.5),
                 out_channel_mode='uniform',
                 routing='learnable',
                 upsample_mode='bilinear',
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg)
        assert routing in ('fixed', 'learnable', 'dynamic')
        self.num_inputs = num_inputs
        self.num_scales = len(scale_factors)
        self.scale_factors = scale_factors
        self.routing = routing
        self.upsample_mode = upsample_mode
        self.out_channels = _make_out_channels(
            out_channels, self.num_scales, out_channel_mode)
        route_channels = self.out_channels[0]

        self.input_projs = nn.ModuleList([
            nn.ModuleList([
                ConvModule(
                    in_channels,
                    scale_channels,
                    kernel_size=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False) for _ in range(num_inputs)
            ]) for scale_channels in self.out_channels
        ])
        self.output_convs = nn.ModuleList([
            ConvModule(
                scale_channels,
                scale_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False) for scale_channels in self.out_channels
        ])
        self.route_projs = nn.ModuleList([
            ConvModule(
                in_channels,
                route_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False) for _ in range(num_inputs)
        ])

        if routing == 'learnable':
            self.route_logits = nn.Parameter(
                torch.eye(self.num_scales, num_inputs) * 3.0)
        elif routing == 'dynamic':
            hidden_channels = max(route_channels // 4, 16)
            self.route_mlp = nn.Sequential(
                nn.Linear(route_channels * num_inputs, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, self.num_scales * num_inputs))

    def _fixed_weights(self, inputs):
        weights = inputs[0].new_zeros(self.num_scales, self.num_inputs)
        for i in range(self.num_scales):
            weights[i, min(i, self.num_inputs - 1)] = 1
        return weights

    def _route_weights(self, feats):
        if self.routing == 'fixed':
            return self._fixed_weights(feats)
        if self.routing == 'learnable':
            return torch.softmax(self.route_logits, dim=1)

        descriptors = [
            F.adaptive_avg_pool2d(feat, 1).flatten(1) for feat in feats
        ]
        descriptors = torch.cat(descriptors, dim=1)
        logits = self.route_mlp(descriptors)
        logits = logits.view(-1, self.num_scales, self.num_inputs)
        return torch.softmax(logits, dim=2)

    def forward(self, inputs):
        assert len(inputs) == self.num_inputs
        route_feats = [
            proj(x) for proj, x in zip(self.route_projs, inputs)
        ]
        route_weights = self._route_weights(route_feats)
        base_size = inputs[0].shape[-2:]

        outs = []
        for scale_idx, scale_factor in enumerate(self.scale_factors):
            target_size = _scaled_size(base_size, scale_factor)
            out = 0
            scale_feats = [
                proj(x)
                for proj, x in zip(self.input_projs[scale_idx], inputs)
            ]
            for layer_idx, feat in enumerate(scale_feats):
                resized = _pairwise_resize(
                    feat, target_size, mode=self.upsample_mode)
                if self.routing == 'dynamic':
                    weight = route_weights[:, scale_idx, layer_idx].view(
                        -1, 1, 1, 1)
                else:
                    weight = route_weights[scale_idx, layer_idx]
                out = out + resized * weight
            outs.append(self.output_convs[scale_idx](out))
        return outs, route_weights


class RemoteSensingDetailEnhancement(BaseModule):
    """Inject high-frequency detail into high-resolution pyramid levels."""

    def __init__(self,
                 channels,
                 p3_channels=None,
                 upsample_mode='bilinear',
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg)
        p2_channels = channels
        p3_channels = p3_channels or p2_channels
        self.upsample_mode = upsample_mode
        self.detail_extract = nn.Sequential(
            ConvModule(
                p2_channels,
                p2_channels,
                kernel_size=3,
                padding=1,
                groups=p2_channels,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False),
            ConvModule(
                p2_channels,
                p2_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False))
        self.detail_fuse = ConvModule(
            p2_channels,
            p2_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False)
        if p2_channels == p3_channels:
            self.p3_proj = nn.Identity()
        else:
            self.p3_proj = ConvModule(
                p2_channels,
                p3_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
        self.p2_scale = nn.Parameter(torch.tensor(1e-3))
        self.p3_scale = nn.Parameter(torch.tensor(1e-3))

    def forward(self, inputs):
        assert len(inputs) >= 2
        outs = list(inputs)
        high_freq = outs[0] - F.avg_pool2d(
            outs[0], kernel_size=3, stride=1, padding=1)
        detail = self.detail_fuse(self.detail_extract(high_freq))
        outs[0] = outs[0] + self.p2_scale * detail
        p3_detail = _pairwise_resize(
            detail, outs[1].shape[-2:], mode=self.upsample_mode)
        outs[1] = outs[1] + self.p3_scale * self.p3_proj(p3_detail)
        return outs


@MODELS.register_module()
class SimpleLayerToScaleNeck(BaseModule):
    """Simple fixed layer-to-scale baseline for homogeneous features."""

    def __init__(self,
                 in_channels,
                 out_channels=256,
                 num_inputs=4,
                 scale_factors=(4, 2, 1, 0.5),
                 out_channel_mode='uniform',
                 upsample_mode='bilinear',
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(init_cfg)
        in_channels = _parse_in_channels(in_channels, num_inputs)
        self.num_inputs = num_inputs
        self.scale_factors = scale_factors
        self.upsample_mode = upsample_mode
        self.out_channels = _make_out_channels(
            out_channels, len(scale_factors), out_channel_mode)
        self.proj_convs = nn.ModuleList([
            ConvModule(
                in_channels,
                scale_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False) for scale_channels in self.out_channels
        ])
        self.out_convs = nn.ModuleList([
            ConvModule(
                scale_channels,
                scale_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False) for scale_channels in self.out_channels
        ])

    def forward(self, inputs):
        assert len(inputs) == self.num_inputs
        base_size = inputs[0].shape[-2:]
        outs = []
        for i, scale_factor in enumerate(self.scale_factors):
            input_idx = min(i, self.num_inputs - 1)
            feat = self.proj_convs[i](inputs[input_idx])
            feat = _pairwise_resize(
                feat, _scaled_size(base_size, scale_factor),
                mode=self.upsample_mode)
            outs.append(self.out_convs[i](feat))
        return outs


@MODELS.register_module()
class FixedTopDownDINONeck(BaseModule):
    """Fixed top-down upsample + lateral-add baseline from DINONeck."""

    def __init__(self,
                 in_channels,
                 out_channels=256,
                 num_inputs=4,
                 out_channel_mode='uniform',
                 upsample_mode='bicubic',
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(init_cfg)
        in_channels = _parse_in_channels(in_channels, num_inputs)
        assert num_inputs == 4
        self.num_inputs = num_inputs
        self.upsample_mode = upsample_mode
        self.out_channels = _make_out_channels(
            out_channels, num_inputs, out_channel_mode)

        self.deep_proj = ConvModule(
            in_channels,
            self.out_channels[3],
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False)
        self.lateral_convs = nn.ModuleList([
            ConvModule(
                in_channels,
                self.out_channels[1],
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False),
            ConvModule(
                in_channels,
                self.out_channels[2],
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False),
            ConvModule(
                in_channels,
                self.out_channels[3],
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
        ])
        self.conv1 = ConvModule(
            self.out_channels[3],
            self.out_channels[3],
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False)
        self.conv2 = ConvModule(
            self.out_channels[3],
            self.out_channels[2],
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False)
        self.conv3 = ConvModule(
            self.out_channels[2],
            self.out_channels[1],
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False)
        self.conv4 = ConvModule(
            self.out_channels[1],
            self.out_channels[0],
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False)

    def forward(self, inputs):
        assert len(inputs) == self.num_inputs

        x0 = self.conv1(self.deep_proj(inputs[3]))

        x = _pairwise_resize(x0, inputs[-2].shape[-2:], self.upsample_mode)
        x = x + self.lateral_convs[2](inputs[2])
        x1 = self.conv2(x)

        x = _pairwise_resize(x1, inputs[1].shape[-2:], self.upsample_mode)
        x = x + self.lateral_convs[1](inputs[1])
        x2 = self.conv3(x)

        x = _pairwise_resize(x2, inputs[0].shape[-2:], self.upsample_mode)
        x = x + self.lateral_convs[0](inputs[0])
        x3 = self.conv4(x)

        return [x3, x2, x1, x0]


@MODELS.register_module()
class HFFNeck(BaseModule):
    """Homogeneous Foundation Feature Neck for SAM/DINO style backbones."""

    def __init__(self,
                 in_channels,
                 out_channels=256,
                 num_inputs=4,
                 scale_factors=(4, 2, 1, 0.5),
                 out_channel_mode='uniform',
                 use_hli=True,
                 use_alsp=True,
                 use_rsde=True,
                 use_topdown=False,
                 routing='learnable',
                 upsample_mode='bilinear',
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(init_cfg)
        assert routing in ('fixed', 'learnable', 'dynamic')
        in_channels = _parse_in_channels(in_channels, num_inputs)
        self.num_inputs = num_inputs
        self.use_hli = use_hli
        self.use_alsp = use_alsp
        self.use_rsde = use_rsde
        self.use_topdown = use_topdown
        self.upsample_mode = upsample_mode
        self.out_channels = _make_out_channels(
            out_channels, len(scale_factors), out_channel_mode)

        if use_hli:
            self.hli = HomogeneousLayerInteraction(in_channels, num_inputs)

        if use_alsp:
            self.pyramid = AdaptiveLayerToScalePyramid(
                in_channels=in_channels,
                out_channels=out_channels,
                num_inputs=num_inputs,
                scale_factors=scale_factors,
                out_channel_mode=out_channel_mode,
                routing=routing,
                upsample_mode=upsample_mode,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        elif use_topdown:
            self.pyramid = FixedTopDownDINONeck(
                in_channels=in_channels,
                out_channels=out_channels,
                num_inputs=num_inputs,
                out_channel_mode=out_channel_mode,
                upsample_mode=upsample_mode,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.pyramid = SimpleLayerToScaleNeck(
                in_channels=in_channels,
                out_channels=out_channels,
                num_inputs=num_inputs,
                scale_factors=scale_factors,
                out_channel_mode=out_channel_mode,
                upsample_mode=upsample_mode,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

        if use_rsde:
            self.rsde = RemoteSensingDetailEnhancement(
                self.out_channels[0],
                p3_channels=self.out_channels[1],
                upsample_mode=upsample_mode,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

    def forward(self, inputs, return_aux=False):
        assert len(inputs) == self.num_inputs
        aux = {}
        feats = list(inputs)
        base_size = feats[0].shape[-2:]
        # feats = [
        #     feat if feat.shape[-2:] == base_size else _pairwise_resize(
        #         feat, base_size, mode=self.upsample_mode)
        #     for feat in feats
        # ]

        feats = [
            _pairwise_resize(feat, (int(base_size[0]*2**(2-i)),int(base_size[1]*2**(2-i))), mode=self.upsample_mode) if i < 2 else feat
            for i,feat in enumerate(feats)
        ]

        if self.use_hli:
            feats, hli_weights = self.hli(feats)
            aux['hli_weights'] = hli_weights

        if self.use_alsp:
            outs, route_weights = self.pyramid(feats)
            aux['route_weights'] = route_weights
        else:
            outs = self.pyramid(feats)

        if self.use_rsde:
            outs = self.rsde(outs)

        if return_aux:
            return outs, aux
        return outs


# @MODELS.register_module()
# class DINONeck(BaseModule):
#     def __init__(self,
#                  in_channels,
#                  num_in=4,
#                  upsample='bicubic',
#                  init_cfg=dict(
#                      type='Xavier', layer='Conv2d', distribution='uniform')):
#         super().__init__(init_cfg)
        
#         if upsample == 'shuffle':
#             self.unsample = nn.PixelShuffle(2)
#         else:
#             self.upsample = nn.Upsample(scale_factor=2, mode=upsample)
        
#         out_channels = [in_channels for i in range(num_in)]
#         self.conv1 = ConvModule(in_channels,out_channels[0],kernel_size=3,padding=1,inplace=False,stride=2)
#         self.conv2 = ConvModule(out_channels[0],out_channels[1],kernel_size=3,padding=1,inplace=False)
#         self.conv3 = ConvModule(out_channels[1],out_channels[2],kernel_size=3,padding=1,inplace=False)
#         self.conv4 = ConvModule(out_channels[2],out_channels[3],kernel_size=3,padding=1,inplace=False)
        
#         self.inter_conv1 = ConvModule(in_channels,out_channels[0],kernel_size=1,inplace=False)
#         self.inter_conv2 = ConvModule(in_channels,out_channels[1],kernel_size=1,inplace=False)
#         self.inter_conv3 = ConvModule(in_channels,out_channels[2],kernel_size=1,inplace=False)
        
#         ## input 1024, 40, 40 * 4
#     def forward(self, inputs):
#         x0 = self.conv1(inputs[-1])
        
#         x = self.upsample(x0)
#         inter_fpn = self.inter_conv1(inputs[-2])
#         x = x + F.interpolate(inter_fpn, size=x.shape[-2:], mode="nearest")
#         x1 = self.conv2(x)
        
#         x = self.upsample(x1)
#         inter_fpn = self.inter_conv2(inputs[0])
#         x = x + F.interpolate(inter_fpn, size=x.shape[-2:], mode="nearest")
#         x2 = self.conv3(x)
        
#         x = self.upsample(x2)
#         inter_fpn = self.inter_conv3(inputs[1])
#         x = x + F.interpolate(inter_fpn, size=x.shape[-2:], mode="nearest")
#         x3 = self.conv4(x)

#         return [x3,x2,x1,x0]
    
if __name__ == "__main__":
    input = [torch.randn(2,1024,32,32) for i in range(4)]
    neck = HFFNeck(in_channels=1024,
                 out_channels=256,
                 num_inputs=4,
                 scale_factors=(4, 2, 1, 0.5),
                 out_channel_mode='uniform', #pyramid
                 use_hli=False, 
                 use_alsp=False,
                 use_rsde=False,
                 use_topdown=True,
                 routing='learnable',
                 upsample_mode='bilinear',
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform'))
    output = neck(input)
    for o in output:
        print(o.shape)
