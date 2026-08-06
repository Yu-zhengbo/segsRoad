# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
import torch, torch.nn as nn, torch.nn.functional as F
from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import SELayer as SEModule
from mmseg.models.utils import DySample
from .mfa_pcs_uper_head import MultiFrequencyChannelAttention

class SeparableConvBlock(nn.Module):
    """SegFormer-style separable conv: depthwise→pointwise→BN→GELU."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x

@MODELS.register_module()
class PCADecoder(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """
    def __init__(self, in_channels=1024, num_classes=7, decoder_channels=384, num_layers=4, **kwargs):
        super().__init__(in_channels,channels=decoder_channels,num_classes=num_classes,**kwargs)
        # Deep+Wide projection: 1024→512→384
        self.linear_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 512, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.GELU(),
                nn.Conv2d(512, decoder_channels, 1, bias=False),
                nn.BatchNorm2d(decoder_channels),
                nn.GELU(),
            ) for _ in range(num_layers)
        ])
        # SE on concat(4×256=1024)
        self.concat_se = SEModule(decoder_channels * num_layers, ratio=16)
        # Fuse: 1024→256
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(decoder_channels * num_layers, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
        )
        # Refine blocks with SE
        self.up_refine1 = nn.Sequential(
            SeparableConvBlock(decoder_channels, decoder_channels), SEModule(decoder_channels))
        self.up_refine2 = nn.Sequential(
            SeparableConvBlock(decoder_channels, decoder_channels), SEModule(decoder_channels))
        self.up_refine3 = nn.Sequential(
            SeparableConvBlock(decoder_channels, decoder_channels), SEModule(decoder_channels))
        self.up_refine4 = nn.Sequential(
            SeparableConvBlock(decoder_channels, decoder_channels), SEModule(decoder_channels))
        self.conv_seg = nn.Conv2d(decoder_channels, num_classes, 1)

    def _forward_feature(self, feats):
        mlp_feats = [linear(feat) for feat, linear in zip(feats, self.linear_layers)]
        x = torch.cat(mlp_feats, dim=1)
        x = self.concat_se(x)          # SE after concat
        x = self.linear_fuse(x)
        for refine in [self.up_refine1, self.up_refine2, self.up_refine3, self.up_refine4]:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = refine(x)
        return x
    
    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
    
@MODELS.register_module()
class DyPCADecoder(PCADecoder):
    def __init__(self, in_channels=1024, num_classes=7, decoder_channels=384, num_layers=4, **kwargs):
        super().__init__(in_channels,decoder_channels=decoder_channels,num_classes=num_classes,num_layers=num_layers,**kwargs)
        self.dysample_list = nn.ModuleList([
            DySample(decoder_channels, scale=2, style='lp', groups=4, dyscope=False) for i in range(4)
        ])

    def _forward_feature(self, feats):
        mlp_feats = [linear(feat) for feat, linear in zip(feats, self.linear_layers)]
        x = torch.cat(mlp_feats, dim=1)
        x = self.concat_se(x)          # SE after concat
        x = self.linear_fuse(x)
        for index,refine in enumerate([self.up_refine1, self.up_refine2, self.up_refine3, self.up_refine4]):
            # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = self.dysample_list[index](x)
            x = refine(x)
        return x


@MODELS.register_module()
class MFCADyPCADecoder(DyPCADecoder):
    def __init__(self,
                 in_channels=1024,
                 num_classes=7,
                 decoder_channels=384,
                 num_layers=4,
                 mfca_dct_size=7,
                 mfca_frequency_branches=8,
                 mfca_frequency_selection='top',
                 mfca_reduction=16,
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            num_layers=num_layers,
            **kwargs)
        
        self.fuse_mfca = MultiFrequencyChannelAttention(
            decoder_channels,
            mfca_dct_size,
            mfca_dct_size,
            frequency_branches=mfca_frequency_branches,
            frequency_selection=mfca_frequency_selection,
            reduction=mfca_reduction)

    def _forward_feature(self, feats):
        mlp_feats = [linear(feat) for feat, linear in zip(feats, self.linear_layers)]
        x = torch.cat(mlp_feats, dim=1)
        x = self.concat_se(x)
        x = self.linear_fuse(x)
        x = self.fuse_mfca(x)
        for index,refine in enumerate([self.up_refine1, self.up_refine2, self.up_refine3, self.up_refine4]):
            x = self.dysample_list[index](x)
            x = refine(x)
        return x


@MODELS.register_module()
class ConcatMFCADyPCADecoder(MFCADyPCADecoder):
    def __init__(self,
                 in_channels=1024,
                 num_classes=7,
                 decoder_channels=384,
                 num_layers=4,
                 concat_mfca_dct_size=7,
                 concat_mfca_frequency_branches=4,
                 concat_mfca_frequency_selection='top',
                 concat_mfca_reduction=16,
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            num_layers=num_layers,
            **kwargs)
        concat_channels = decoder_channels * num_layers
        self.concat_se = MultiFrequencyChannelAttention(
            concat_channels,
            concat_mfca_dct_size,
            concat_mfca_dct_size,
            frequency_branches=concat_mfca_frequency_branches,
            frequency_selection=concat_mfca_frequency_selection,
            reduction=concat_mfca_reduction)
        self.fuse_mfca = nn.Identity()
        self.up_refine4 = nn.Identity()
    
@MODELS.register_module()
class DyShuffleDecoder(PCADecoder):
    def __init__(self, in_channels=1024, num_classes=7, decoder_channels=384, num_layers=4, **kwargs):
        super().__init__(in_channels,decoder_channels=decoder_channels,num_classes=num_classes,num_layers=num_layers,**kwargs)
        self.dysample_list = nn.ModuleList([
            DySample(decoder_channels, scale=2, style='lp', groups=4, dyscope=False) for i in range(4)
        ])
        
        self.concat_se = SEModule(decoder_channels, ratio=16)
        # Fuse: 1024→256
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
        )
        self.up_refine4 = nn.Identity()
        self.dysample_list = nn.ModuleList([
            DySample(decoder_channels, scale=2, style='lp', groups=4, dyscope=False) for i in range(3)
        ])

    def _forward_feature(self, feats):
        mlp_feats = [linear(feat) for feat, linear in zip(feats, self.linear_layers)]
        B, C, H, W = mlp_feats[0].shape
        # x = torch.cat(mlp_feats, dim=1)
        
        x_new = mlp_feats[0].new_zeros(B, C, 2*H, 2*W)
        x_new[:,:,::2,::2] = mlp_feats[0][:,:,:,:]
        x_new[:,:,::2,1::2] = mlp_feats[1][:,:,:,:]
        x_new[:,:,1::2,::2] = mlp_feats[2][:,:,:,:]
        x_new[:,:,1::2,1::2] = mlp_feats[3][:,:,:,:]
        
        x = self.concat_se(x_new)          # SE after concat
        x = self.linear_fuse(x)
        for index,refine in enumerate([self.up_refine1, self.up_refine2, self.up_refine3]):
            # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = self.dysample_list[index](x)
            x = refine(x)
        return x
    
