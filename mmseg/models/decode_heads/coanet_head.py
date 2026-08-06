# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2021 Jie Mei et al.
#
# This implementation adapts the decoder in CoANet
# (https://github.com/mj129/CoANet) to the MMSegmentation DecodeHead API.
# The upstream project is GPL-3.0 and restricts use to non-commercial
# research; make sure that those terms are compatible with your use case.

"""CoANet decode head for road segmentation and connectivity supervision."""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead


def _norm(norm_cfg, channels: int) -> nn.Module:
    """Build a normalization layer compatible with an MMSeg config."""
    if norm_cfg is None:
        return nn.BatchNorm2d(channels)
    return build_norm_layer(norm_cfg, channels)[1]


class _ASPPBranch(nn.Sequential):
    def __init__(self, in_channels, channels, kernel_size, padding, dilation,
                 norm_cfg):
        super().__init__(
            nn.Conv2d(
                in_channels,
                channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False), _norm(norm_cfg, channels), nn.ReLU(inplace=True))


class StripConvBlock(nn.Module):
    """The four-direction strip convolution block used by CoANet."""

    def __init__(self, in_channels, out_channels, norm_cfg, upsample=False):
        super().__init__()
        if in_channels % 8:
            raise ValueError('in_channels must be divisible by 8, '
                             f'but got {in_channels}.')
        mid_channels = in_channels // 4
        branch_channels = in_channels // 8
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1), _norm(norm_cfg,
                                                            mid_channels),
            nn.ReLU(inplace=True))
        self.horizontal = nn.Conv2d(mid_channels, branch_channels,
                                    (1, 9), padding=(0, 4))
        self.vertical = nn.Conv2d(mid_channels, branch_channels,
                                  (9, 1), padding=(4, 0))
        self.trans_horizontal = nn.Conv2d(mid_channels,
                                          branch_channels,
                                          (9, 1),
                                          padding=(4, 0))
        self.trans_vertical = nn.Conv2d(mid_channels, branch_channels,
                                        (1, 9), padding=(0, 4))
        self.upsample = upsample
        # Four branches each produce ``in_channels // 8`` channels.
        self.fuse = nn.Sequential(_norm(norm_cfg, in_channels // 2),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(in_channels // 2, out_channels, 1),
                                  _norm(norm_cfg, out_channels),
                                  nn.ReLU(inplace=True))

    @staticmethod
    def _h_transform(x):
        n, c, h, w = x.shape
        x = F.pad(x, (0, w)).reshape(n, c, -1)[..., :-w]
        return x.reshape(n, c, h, 2 * w - 1)

    @staticmethod
    def _inv_h_transform(x):
        n, c, h, _ = x.shape
        x = F.pad(x.reshape(n, c, -1), (0, h)).reshape(n, c, h, 2 * h)
        return x[..., :h]

    @classmethod
    def _v_transform(cls, x):
        return cls._h_transform(x.transpose(2, 3)).transpose(2, 3)

    @classmethod
    def _inv_v_transform(cls, x):
        return cls._inv_h_transform(x.transpose(2, 3)).transpose(2, 3)

    def forward(self, x):
        x = self.reduce(x)
        x1 = self.horizontal(x)
        x2 = self.vertical(x)
        x3 = self._inv_h_transform(self.trans_horizontal(self._h_transform(x)))
        x4 = self._inv_v_transform(self.trans_vertical(self._v_transform(x)))
        x = torch.cat((x1, x2, x3, x4), dim=1)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear',
                              align_corners=True)
        return self.fuse(x)


class _SELayer(nn.Module):
    """Squeeze-and-excitation used in CoANet's connectivity branches."""

    def __init__(self, channels, reduction=3):
        super().__init__()
        hidden_channels = channels // reduction
        if hidden_channels < 1:
            raise ValueError('num_neighbor must be at least reduction.')
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels, hidden_channels, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden_channels, channels,
                                          bias=False), nn.Sigmoid())

    def forward(self, x):
        weights = self.fc(self.pool(x).flatten(1)).view(x.shape[0],
                                                        x.shape[1], 1, 1)
        return x * weights


def _connectivity_target(seg_label, step: int, foreground_index: int):
    """Build the nine CoANet connectivity labels from a semantic GT mask.

    This is the batched, on-device equivalent of ``get_con_1_batch`` and
    ``get_con_3_batch`` used by :class:`MFAPCSUPerHead`.  Ignore pixels and
    all non-road classes are treated as background, as in that implementation.
    """
    if seg_label.ndim != 4 or seg_label.shape[1] != 1:
        raise ValueError('seg_label must have shape [B, 1, H, W].')
    road = (seg_label == foreground_index).to(dtype=torch.float32)
    _, _, height, width = road.shape
    padded = F.pad(road, (step, step, step, step), value=0)
    targets = []
    for dy in (0, step, 2 * step):
        for dx in (0, step, 2 * step):
            neighbor = padded[:, :, dy:dy + height, dx:dx + width]
            targets.append(road * neighbor)
    return torch.cat(targets, dim=1)


@MODELS.register_module()
class CoANetHead(BaseDecodeHead):
    """CoANet segmentation head.

    Args:
        in_channels (Sequence[int]): Four ResNet feature dimensions.  The
            canonical CoANet setting is ``(256, 512, 1024, 2048)``.
        in_index (Sequence[int]): Feature indices from shallow to deep.
        aspp_dilations (Sequence[int]): Atrous rates used at the deepest
            feature level.  Use ``(1, 12, 24, 36)`` for output stride 8 and
            ``(1, 6, 12, 18)`` for output stride 16.
    """

    def __init__(self,
                 in_channels=(256, 512, 1024, 2048),
                 in_index=(0, 1, 2, 3),
                 aspp_dilations=(1, 6, 12, 18),
                 num_neighbor=9,
                 foreground_index=1,
                 connectivity_loss_weight=0.2,
                 connectivity_d0_weight=0.6,
                 connectivity_d1_weight=0.4,
                 restore_connectivity=False,
                 connect_threshold=0.9,
                 connect_d1_threshold=2.0,
                 restored_logit=20.0,
                 **kwargs):
        kwargs.setdefault('channels', 64)
        kwargs.setdefault('input_transform', 'multiple_select')
        super().__init__(in_channels=in_channels, in_index=in_index, **kwargs)
        self.num_neighbor = num_neighbor
        if num_neighbor != 9:
            raise ValueError('CoANet connectivity supervision requires '
                             'num_neighbor=9.')
        self.foreground_index = foreground_index
        self.connectivity_loss_weight = connectivity_loss_weight
        self.connectivity_d0_weight = connectivity_d0_weight
        self.connectivity_d1_weight = connectivity_d1_weight
        self.restore_connectivity = restore_connectivity
        self.connect_threshold = connect_threshold
        self.connect_d1_threshold = connect_d1_threshold
        self.restored_logit = restored_logit
        if len(in_channels) != 4:
            raise ValueError('CoANetHead requires exactly four input features.')
        if tuple(in_channels) != (256, 512, 1024, 2048):
            raise ValueError('CoANetHead currently follows the canonical '
                             'ResNet channel layout (256, 512, 1024, 2048).')

        norm_cfg = self.norm_cfg
        deep_channels = in_channels[-1]
        self.aspp = nn.ModuleList([
            _ASPPBranch(deep_channels, 256, 1 if rate == 1 else 3,
                        0 if rate == 1 else rate, rate, norm_cfg)
            for rate in aspp_dilations
        ])
        self.image_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        nn.Conv2d(deep_channels, 256, 1,
                                                  bias=False), _norm(norm_cfg,
                                                                     256),
                                        nn.ReLU(inplace=True))
        self.aspp_bottleneck = nn.Sequential(
            nn.Conv2d(256 * (len(aspp_dilations) + 1), 256, 1, bias=False),
            _norm(norm_cfg, 256), nn.ReLU(inplace=True), nn.Dropout(0.5))

        self.decoder4 = StripConvBlock(256, 256, norm_cfg)
        # With CoANet's official output-stride-8 setting, e2/e3/e4 are all
        # at /8 resolution, so this stage intentionally does not upsample.
        self.decoder3 = StripConvBlock(512, 128, norm_cfg)
        self.decoder2 = StripConvBlock(256, 64, norm_cfg, upsample=True)
        self.decoder1 = StripConvBlock(128, 64, norm_cfg, upsample=True)
        self.skip3 = nn.Sequential(nn.Conv2d(1024, 256, 1, bias=False),
                                   _norm(norm_cfg, 256), nn.ReLU(inplace=True))
        self.skip2 = nn.Sequential(nn.Conv2d(512, 128, 1, bias=False),
                                   _norm(norm_cfg, 128), nn.ReLU(inplace=True))
        self.skip1 = nn.Sequential(nn.Conv2d(256, 64, 1, bias=False),
                                   _norm(norm_cfg, 64), nn.ReLU(inplace=True))

        # These are the original CoANet output branches.  ``forward`` returns
        # only segmentation logits to honour BaseDecodeHead's MMSeg contract;
        # a connectivity-aware segmentor can call ``forward_with_connectivity``
        # to obtain the two auxiliary 9-channel maps.
        self.seg_branch = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(64, self.out_channels, 1))
        self.connect_branch = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(64, num_neighbor, 3,
                                                      padding=1))
        self.connect_branch_d1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, num_neighbor, 3, padding=3, dilation=3))
        self.connect_se = _SELayer(num_neighbor)
        self.connect_d1_se = _SELayer(num_neighbor)
        self.connectivity_loss = nn.BCEWithLogitsLoss()

        # CoANet has its own 3x3 + 1x1 segmentation branch instead of the
        # BaseDecodeHead 1x1 classifier.
        self.conv_seg = nn.Identity()

    def _aspp_forward(self, x):
        outputs = [branch(x) for branch in self.aspp]
        image_pool = self.image_pool(x)
        image_pool = F.interpolate(image_pool,
                                   size=x.shape[2:],
                                   mode='bilinear',
                                   align_corners=True)
        return self.aspp_bottleneck(torch.cat((*outputs, image_pool), dim=1))

    def _forward_features(self, inputs):
        e1, e2, e3, e4 = self._transform_inputs(inputs)
        e4 = self._aspp_forward(e4)
        d4 = torch.cat((self.decoder4(e4), self.skip3(e3)), dim=1)
        d3 = torch.cat((self.decoder3(d4), self.skip2(e2)), dim=1)
        d2 = torch.cat((self.decoder2(d3), self.skip1(e1)), dim=1)
        x = self.decoder1(d2)
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)
        return x

    def forward_with_connectivity(self, inputs):
        """Return CoANet's segmentation and two connectivity logit maps.

        Connectivity targets are not part of MMSegmentation's standard
        ``SegDataSample``.  This method exposes the complete original model
        for a downstream connectivity-aware loss/segmentor implementation.
        """
        x = self._forward_features(inputs)
        return (self.seg_branch(x), self.connect_se(self.connect_branch(x)),
                self.connect_d1_se(self.connect_branch_d1(x)))

    def _restore_connectivity_logits(self, seg_logits, connect_logits,
                                     connect_d1_logits):
        """Apply CoANet's inference-time connectivity union.

        The official implementation thresholds the road probability, sums the
        sigmoid-activated 9-channel connectivity maps, thresholds the two
        sums at 0.9 and 2.0, then takes their union.  Since MMSegmentation
        must receive logits (not a binary mask), the resulting union is
        encoded as ``+/-restored_logit``.  This works for both a one-logit
        binary head and a two-class softmax head.
        """
        if self.out_channels not in (1, 2):
            raise RuntimeError('Connectivity restoration supports only '
                               'one-logit or two-class CoANet outputs.')
        if self.out_channels == 2 and self.foreground_index not in (0, 1):
            raise RuntimeError('foreground_index must be 0 or 1 for a '
                               'two-class CoANet output.')
        threshold = self.threshold if self.threshold is not None else 0.5
        if self.out_channels == 1:
            seg_mask = seg_logits.sigmoid() >= threshold
        else:
            seg_mask = seg_logits.softmax(dim=1)[:, self.foreground_index:
                                                self.foreground_index + 1] \
                >= threshold
        connect_mask = connect_logits.sigmoid().sum(dim=1, keepdim=True) >= \
            self.connect_threshold
        connect_d1_mask = connect_d1_logits.sigmoid().sum(
            dim=1, keepdim=True) >= \
            self.connect_d1_threshold
        restored_mask = seg_mask | connect_mask | connect_d1_mask
        foreground_logits = torch.where(
            restored_mask, seg_logits.new_full((), self.restored_logit),
            seg_logits.new_full((), -self.restored_logit))
        if self.out_channels == 1:
            return foreground_logits

        # Emit a confident two-class logit map so MMSeg's argmax
        # post-processor exactly preserves the connectivity union.
        restored_logits = seg_logits.new_empty(seg_logits.shape)
        restored_logits[:, self.foreground_index] = foreground_logits[:, 0]
        restored_logits[:, 1 - self.foreground_index] = -foreground_logits[:,
                                                                            0]
        return restored_logits

    ## 仅在验证和测试中会被encode_deocde使用
    def forward(self, inputs):
        seg_logits, connect_logits, connect_d1_logits = \
            self.forward_with_connectivity(inputs)
        if self.restore_connectivity and not self.training:
            return self._restore_connectivity_logits(seg_logits, connect_logits,
                                                     connect_d1_logits)
        return seg_logits

    def _connectivity_loss(self, logits, target):
        logits = resize(logits,
                        size=target.shape[2:],
                        mode='bilinear',
                        align_corners=self.align_corners)
        return self.connectivity_loss(logits, target)

    def loss(self, inputs, batch_data_samples, train_cfg):
        """Compute segmentation and online-generated connectivity losses.

        The targets are derived from ``gt_sem_seg`` in the current batch, so
        no dataset transform or additional annotation file is required.
        ``connect`` uses a one-pixel neighbourhood and ``connect_d1`` uses
        the four-pixel neighbourhood from the existing MFA PCS head.
        """
        seg_logits, connect_logits, connect_d1_logits = \
            self.forward_with_connectivity(inputs)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        seg_label = self._stack_batch_gt(batch_data_samples)
        connect_target = _connectivity_target(seg_label, 1,
                                              self.foreground_index)
        connect_d1_target = _connectivity_target(seg_label, 4,
                                                 self.foreground_index)
        connect_loss = (
            self.connectivity_d0_weight *
            self._connectivity_loss(connect_logits, connect_target) +
            self.connectivity_d1_weight *
            self._connectivity_loss(connect_d1_logits, connect_d1_target))
        losses['loss_connectivity'] = (
            self.connectivity_loss_weight * connect_loss)
        return losses
