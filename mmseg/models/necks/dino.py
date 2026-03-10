# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmseg.registry import MODELS
from ..utils import resize


@MODELS.register_module()
class DINONeck(BaseModule):
    """Feature Pyramid Network.

    This neck is the implementation of `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: dict(mode='nearest').
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 num_in=4,
                 upsample='bicubic',
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(init_cfg)
        
        if upsample == 'shuffle':
            self.unsample = nn.PixelShuffle(2)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode=upsample)
        
        out_channels = [in_channels//2**i for i in range(num_in)]

        self.conv1 = ConvModule(in_channels,out_channels[0],kernel_size=3,padding=1,inplace=False)
        self.conv2 = ConvModule(out_channels[0],out_channels[0],kernel_size=3,padding=1,inplace=False)
        self.conv3 = ConvModule(out_channels[0],out_channels[1],kernel_size=3,padding=1,inplace=False)
        self.conv4 = ConvModule(out_channels[1],out_channels[2],kernel_size=3,padding=1,inplace=False)
        self.conv5 = ConvModule(out_channels[2],out_channels[3],kernel_size=3,padding=1,inplace=False)
        
        self.inter_conv1 = ConvModule(in_channels,out_channels[0],kernel_size=1,inplace=False)
        self.inter_conv2 = ConvModule(in_channels,out_channels[1],kernel_size=1,inplace=False)
        self.inter_conv3 = ConvModule(in_channels,out_channels[2],kernel_size=1,inplace=False)
        

    def forward(self, inputs):
        outs = [inputs[-1]]  # 1024, 1/16
        x = inputs[-1]
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.upsample(x)
        inter_fpn = self.inter_conv1(inputs[0])
        x = x + F.interpolate(inter_fpn, size=x.shape[-2:], mode="nearest")
        x = self.conv3(x)
        outs.append(x)  # 512, 1/8
        
        x = self.upsample(x)
        inter_fpn = self.inter_conv2(inputs[1])
        x = x + F.interpolate(inter_fpn, size=x.shape[-2:], mode="nearest")
        x = self.conv4(x)
        outs.append(x)  # 256, 1/4

        inter_fpn = self.inter_conv3(inputs[2])
        x = x + F.interpolate(inter_fpn, size=x.shape[-2:], mode="nearest")
        x = self.conv5(x)
        outs.append(x)  # 128, 1/4


        return tuple(outs[::-1])
