# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead
from .psp_head import PPM

import math
import torch.nn.functional as F
from mmseg.models.decode_heads.deformable_head_with_time_connect import get_con_1_batch,get_con_3_batch
from ..utils.se_layer import SELayer

def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiFrequencyChannelAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 dct_h, dct_w,
                 frequency_branches=16,
                 frequency_selection='top',
                 reduction=16):
        super().__init__()
        assert frequency_branches in [1, 2, 4, 8, 16, 32]

        self.in_channels = in_channels
        self.num_freq = frequency_branches
        self.dct_h = dct_h
        self.dct_w = dct_w

        freq_x, freq_y = get_freq_indices(f'{frequency_selection}{frequency_branches}')
        freq_x = [x * (dct_h // 7) for x in freq_x]
        freq_y = [y * (dct_w // 7) for y in freq_y]
        self.num_split = len(freq_x)
        self.register_buffer('dct_weights', self._build_dct_bank(freq_x, freq_y))

        mid_channels = in_channels // reduction
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x):
        pooled = self._resize_to_dct(x)
        avg_score, max_score, min_score = self._frequency_pool(pooled)
        attention = torch.sigmoid(
            self.fc(avg_score) + self.fc(max_score) + self.fc(min_score))
        return x * attention

    def _resize_to_dct(self, x):
        if x.shape[-2:] == (self.dct_h, self.dct_w):
            return x
        return F.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))

    def _frequency_pool(self, x):
        spectral = x.unsqueeze(1) * self.dct_weights.unsqueeze(0)
        avg_score = spectral.mean(dim=(-1, -2)).mean(dim=1)
        max_score = spectral.amax(dim=(-1, -2)).mean(dim=1)
        min_score = spectral.amin(dim=(-1, -2)).mean(dim=1)
        return tuple(item[:, :, None, None] for item in (avg_score, max_score, min_score))

    def _build_dct_bank(self, freq_x, freq_y):
        basis_x = self._dct_basis(self.dct_h, freq_x)
        basis_y = self._dct_basis(self.dct_w, freq_y)
        filters = basis_x[:, :, None] * basis_y[:, None, :]
        return filters[:, None].repeat(1, self.in_channels, 1, 1)

    @staticmethod
    def _dct_basis(length, frequencies):
        pos = torch.arange(length, dtype=torch.float64) + 0.5
        freq = torch.tensor(frequencies, dtype=torch.float64).unsqueeze(1)
        basis = torch.cos(math.pi * freq * pos / length) / math.sqrt(length)
        basis[freq.squeeze(1) != 0] *= math.sqrt(2)
        return basis.float()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        bank_key = prefix + 'dct_weights'
        legacy_keys = [prefix + f'dct_weight_{i}' for i in range(self.num_freq)]
        if bank_key not in state_dict and all(key in state_dict for key in legacy_keys):
            state_dict[bank_key] = torch.stack([state_dict.pop(key) for key in legacy_keys])
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

class MFMSAttentionBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 scale_branches=2,
                 frequency_branches=16,
                 frequency_selection='top',
                 block_repetition=1,
                 min_channel=64,
                 min_resolution=8,
                 groups=32):
        super(MFMSAttentionBlock, self).__init__()

        self.scale_branches = scale_branches
        self.frequency_branches = frequency_branches
        self.block_repetition = block_repetition
        self.min_channel = min_channel
        self.min_resolution = min_resolution

        self.multi_scale_branches = nn.ModuleList([])
        for scale_idx in range(scale_branches):
            inter_channel = in_channels // 2**scale_idx
            if inter_channel < self.min_channel: inter_channel = self.min_channel

            self.multi_scale_branches.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1 + scale_idx, dilation=1 + scale_idx, groups=groups, bias=False),
                nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, inter_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(inter_channel), nn.ReLU(inplace=True)
            ))

        c2wh = dict([(32, 112), (64, 56), (128, 28), (256, 14), (512, 7)])
        self.multi_frequency_branches = nn.ModuleList([])
        self.multi_frequency_branches_conv1 = nn.ModuleList([])
        self.multi_frequency_branches_conv2 = nn.ModuleList([])
        self.alpha_list = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(scale_branches)])
        self.beta_list = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(scale_branches)])

        for scale_idx in range(scale_branches):
            inter_channel = in_channels // 2**scale_idx
            if inter_channel < self.min_channel: inter_channel = self.min_channel

            if frequency_branches > 0:
                self.multi_frequency_branches.append(
                    nn.Sequential(
                        MultiFrequencyChannelAttention(inter_channel, c2wh[in_channels], c2wh[in_channels], frequency_branches, frequency_selection)))
            self.multi_frequency_branches_conv1.append(
                nn.Sequential(
                    nn.Conv2d(inter_channel, 1, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.Sigmoid()))
            self.multi_frequency_branches_conv2.append(
                nn.Sequential(
                    nn.Conv2d(inter_channel, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True)))

    def forward(self, x):
        feature_aggregation = 0
        for scale_idx in range(self.scale_branches):
            feature = F.avg_pool2d(x, kernel_size=2 ** scale_idx, stride=2 ** scale_idx, padding=0) if int(x.shape[2] // 2 ** scale_idx) >= self.min_resolution else x
            feature = self.multi_scale_branches[scale_idx](feature)
            if self.frequency_branches > 0:
                feature = self.multi_frequency_branches[scale_idx](feature)
            spatial_attention_map = self.multi_frequency_branches_conv1[scale_idx](feature)
            feature = self.multi_frequency_branches_conv2[scale_idx](feature * (1 - spatial_attention_map) * self.alpha_list[scale_idx] + feature * spatial_attention_map * self.beta_list[scale_idx])
            feature_aggregation += F.interpolate(feature, size=None, scale_factor=2**scale_idx, mode='bilinear', align_corners=None) if (x.shape[2] != feature.shape[2]) or (x.shape[3] != feature.shape[3]) else feature
        feature_aggregation /= self.scale_branches
        feature_aggregation += x

        return feature_aggregation

class UpsampleBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 skip_connection_channels,
                 scale_branches=2,
                 frequency_branches=16,
                 frequency_selection='top',
                 block_repetition=1,
                 min_channel=64,
                 min_resolution=8):
        super(UpsampleBlock, self).__init__()

        in_channels = in_channels + skip_connection_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

        self.attention_layer = MFMSAttentionBlock(out_channels, scale_branches, frequency_branches, frequency_selection, block_repetition, min_channel, min_resolution)

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x, skip_connection=None):
        x = F.interpolate(x, size=None, scale_factor=2, mode='bilinear', align_corners=None)

        x = torch.cat([x, skip_connection], dim=1)
        x = self.conv1(x)
        x = self.attention_layer(x)
        x = self.conv2(x)

        return x

@MODELS.register_module()
class MFAPCSUPerHead(BaseDecodeHead):
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
        self.mfa_convs = nn.ModuleList()
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
            mfa_conv = UpsampleBlock(in_channels=self.channels, out_channels=self.channels, skip_connection_channels=self.channels, scale_branches=2,
                          frequency_branches=16, frequency_selection='top', block_repetition=1, min_channel=64,
                          min_resolution=8)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.mfa_convs.append(mfa_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.connect_d0 = nn.Sequential(
            nn.Conv2d(self.channels, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 9, 3, padding=1, dilation=1),
            SELayer(9, ratio=3),
        )
        self.connect_d1 = nn.Sequential(
            nn.Conv2d(self.channels, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 9, 3, padding=3, dilation=3),
            SELayer(9, ratio=3),
        )
        self.con_loss = nn.BCEWithLogitsLoss()

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
            laterals[i - 1] = self.mfa_convs[i - 1](laterals[i],laterals[i - 1])
            # prev_shape = laterals[i - 1].shape[2:]
            # laterals[i - 1] = laterals[i - 1] + resize(
            #     laterals[i],
            #     size=prev_shape,
            #     mode='bilinear',
            #     align_corners=self.align_corners)

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

    def forward_train(self,inputs):
        output = self._forward_feature(inputs)
        con0 = self.connect_d0(output)
        con1 = self.connect_d1(output)
        output = self.cls_seg(output)
        return output,con0,con1

    def ConLoss(self,logit, target):
        logit = resize(
            input=logit,
            size=target.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        loss = self.con_loss(logit, target)
        return loss

    def loss(self, inputs, data_samples, train_cfg):
        seg_logits,con0,con1 = self.forward_train(inputs)
        losses = self.loss_by_feat(seg_logits, data_samples)

        gt_temp = self._stack_batch_gt(data_samples)
        d1 = get_con_1_batch(gt_temp)
        d3 = get_con_3_batch(gt_temp)
        con_loss0 = self.ConLoss(con0, d1) * 0.4 * 0.4
        con_loss1 = self.ConLoss(con1, d3) * 0.6 * 0.4
        losses['con_loss'] = con_loss0 + con_loss1
        return losses
