# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import SELayer, resize
from .decode_head import BaseDecodeHead
from .psp_head import PPM


@MODELS.register_module()
class UPerHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output


@MODELS.register_module()
class UPerHeadForDINO(UPerHead):
    """UPerHead variant with progressive upsampling for DINO ViT features.

    DINO ViT intermediate features usually share the same patch resolution, so
    the standard UPerNet FPN path mostly fuses channels and layers without
    recovering spatial resolution inside the head. This variant keeps the
    original PPM + FPN fusion and adds a light decoder tail:
    ``upsample -> 3x3 conv`` repeated ``num_upsample_layers`` times.
    """

    def __init__(
        self,
        num_upsample_layers=4,
        use_fpn_concat=False,
        fpn_concat_se_ratio=16,
        use_concat=False,
        concat_se_ratio=16,
        use_se=False,
        se_ratio=16,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_upsample_layers = num_upsample_layers
        self.use_fpn_concat = use_fpn_concat
        self.use_concat = use_concat
        self.use_se = use_se
        self.fpn_concat_fuse = nn.ModuleList([
            nn.Sequential(
                SELayer(self.channels * 2, ratio=fpn_concat_se_ratio),
                ConvModule(
                    self.channels * 2,
                    self.channels,
                    1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            for _ in range(len(self.in_channels) - 1)
        ]) if use_fpn_concat else None
        self.concat_se = SELayer(
            len(self.in_channels) * self.channels,
            ratio=concat_se_ratio) if use_concat else None
        self.up_refine_convs = nn.ModuleList([
            ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            for _ in range(num_upsample_layers)
        ])
        self.up_refine_se = nn.ModuleList([
            SELayer(self.channels, ratio=se_ratio)
            for _ in range(num_upsample_layers)
        ]) if use_se else None

    def _forward_feature(self, inputs):
        inputs = self._transform_inputs(inputs)

        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        laterals.append(self.psp_forward(inputs))

        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            top_down = resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)
            if self.use_fpn_concat:
                laterals[i - 1] = self.fpn_concat_fuse[i - 1](
                    torch.cat([laterals[i - 1], top_down], dim=1))
            else:
                laterals[i - 1] = laterals[i - 1] + top_down

        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        fpn_outs = torch.cat(fpn_outs, dim=1)
        if self.use_concat:
            fpn_outs = self.concat_se(fpn_outs)
        feats = self.fpn_bottleneck(fpn_outs)

        for i, refine_conv in enumerate(self.up_refine_convs):
            feats = resize(
                feats,
                scale_factor=2,
                mode='bilinear',
                align_corners=self.align_corners)
            feats = refine_conv(feats)
            if self.use_se:
                feats = self.up_refine_se[i](feats)
        return feats
