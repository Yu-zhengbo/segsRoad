"""
Some code is borrowed from timm: https://github.com/huggingface/pytorch-image-models
"""
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from functools import partial
import torch
import torch.nn as nn
import math
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import torch.nn.functional as F
from torch.nn.init import constant_
from einops import repeat
from fvcore.nn import flop_count, parameter_count
import copy
from mmseg.registry import MODELS
# from mmseg.utils import get_root_logger
import logging
# from mmcv.runner import load_checkpoint
from mmcv.cnn import build_norm_layer


# try:
#     from .ops_dcnv3.functions import DCNv3Function
# except:
#     from ops_dcnv3.functions import DCNv3Function
from mmseg.models.backbones.mamba.damamba.ops_dcnv3.functions import DCNv3Function
# try:
#     from .utils import selective_scan_state_flop_jit, selective_scan_fn
# except:
#     from utils import selective_scan_state_flop_jit, selective_scan_fn
from mmseg.models.backbones.mamba.damamba.utils import selective_scan_state_flop_jit, selective_scan_fn
from mmseg.models.backbones.swin import SwinBlock,ShiftWindowMSA,WindowMSA
from mmcv.cnn.bricks.transformer import build_dropout
from mmengine.utils import to_2tuple
import torch.utils.checkpoint as cp
class CrossWindowMSA(WindowMSA):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None):

        super().__init__(embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias,
                 qk_scale,
                 attn_drop_rate,
                 proj_drop_rate,
                 init_cfg=init_cfg)
        del self.qkv
        self.q = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.kv = nn.Linear(embed_dims, embed_dims * 2, bias=qkv_bias)

    def forward(self, x,x2, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape
        q = self.q(x2).reshape(B, N, 1, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(x).reshape(B, N, 2, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = q[0], kv[0], kv[1]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossShiftWindowMSA(ShiftWindowMSA):
    """Shifted Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None):
        super().__init__(embed_dims,
                 num_heads,
                 window_size,
                 shift_size,
                 qkv_bias,
                 qk_scale,
                 attn_drop_rate,
                 proj_drop_rate,
                 dropout_layer,
                 init_cfg)

        self.w_msa = CrossWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None)
        
    def forward_twice(self,query, B, C, H ,W):
        query = query.view(B, H, W, C)
        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]
        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                                                  attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size**2, C)
        return query_windows, attn_mask, H_pad, W_pad, pad_r, pad_b
    def forward(self, query,query2, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'

        query_windows, attn_mask, H_pad, W_pad, pad_r, pad_b = self.forward_twice(query, B, C, H, W)
        query2_windows,_, _, _, _, _ = self.forward_twice(query2, B, C, H, W)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows,query2_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop(x)
        return x


class CrossSwinBlock(SwinBlock):
    """"
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        window_size (int, optional): The local window scale. Default: 7.
        shift (bool, optional): whether to shift window or not. Default False.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 window_size=7,
                 shift=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):

        super().__init__(embed_dims,
                 num_heads,
                 feedforward_channels,
                 window_size,
                 shift,
                 qkv_bias,
                 qk_scale,
                 drop_rate,
                 attn_drop_rate,
                 drop_path_rate,
                 act_cfg,
                 norm_cfg,
                 with_cp,
                 init_cfg)

        self.attn = CrossShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None)
        self.other_norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

    def forward(self, x, x2, hw_shape):

        def _inner_forward(x, x2):
            identity = x
            x = self.norm1(x)
            x2 = self.other_norm1(x2)
            x = self.attn(x, x2, hw_shape)

            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x, x2)
        else:
            x = _inner_forward(x, x2)

        return x

class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


def my_build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()

    raise NotImplementedError(f'build_act_layer does not support {act_layer}')




class CenterFeatureScaleModule(nn.Module):
    def forward(self,
                query,
                center_feature_scale_proj_weight,
                center_feature_scale_proj_bias):
        center_feature_scale = F.linear(query,
                                        weight=center_feature_scale_proj_weight,
                                        bias=center_feature_scale_proj_bias).sigmoid()
        return center_feature_scale


class Dynamic_Adaptive_Scan(nn.Module):
    def __init__(
            self,
            channels=64,
            kernel_size=1,
            dw_kernel_size=3,
            stride=1,
            pad=0,
            dilation=1,
            group=1,
            offset_scale=1.0,
            act_layer='GELU',
            norm_layer='LN',
            center_feature_scale=False,
            remove_center=False,
    ):
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)

        if self.remove_center and self.kernel_size % 2 == 0:
            raise ValueError('remove_center is only compatible with odd kernel size.')

        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=dw_kernel_size,
                stride=1,
                padding=(dw_kernel_size - 1) // 2,
                groups=channels),
            my_build_norm_layer(
                channels,
                norm_layer,
                'channels_first',
                'channels_last'),
            build_act_layer(act_layer))
        self.offset = nn.Linear(
            channels,
            group * (kernel_size * kernel_size - remove_center) * 2)
        self._reset_parameters()

        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)


    def forward(self, input, x):
        N, _, H, W = input.shape
        x_proj = x
        x1 = input
        x1 = self.dw_conv(x1)
        offset = self.offset(x1)
        mask = torch.ones(N, H, W, self.group, device=x.device, dtype=x.dtype)
        x = DCNv3Function.apply(
            x, offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale,
            256,
            self.remove_center)

        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            # N, H, W, groups -> N, H, W, groups, 1 -> N, H, W, groups, _d_per_group -> N, H, W, channels
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale

        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,ffnconv=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features = out_features or in_features
        self.hidden_features = hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act1 = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        if ffnconv:
            self.conv = ResDWC(hidden_features, 3)
        else:
            self.conv = nn.Identity()
            print("FFN_False", in_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.conv(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ResDWC(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2, groups=dim)
        a = torch.zeros(kernel_size ** 2)
        a[4] = 1.
        self.conv_constant = nn.Parameter(a.reshape(1, 1, kernel_size, kernel_size))
        self.conv_constant.requires_grad = False
    def forward(self, x):
        return F.conv2d(x, self.conv.weight + self.conv_constant, self.conv.bias, stride=1,
                        padding=self.kernel_size // 2, groups=self.dim)  # equal to x + conv(x)


class MlpHead(nn.Module):
    """ MLP classification head
    """

    def __init__(self, dim, num_classes=1000):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = x.mean((2, 3))  # global average pooling
        x = self.fc(x)
        return x


class DASSM(nn.Module):
    def __init__(
        self,
        d_model,
        head_dim=16,
        d_state=1,
        d_conv=3,
        expand=1,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj =nn.Conv2d(self.d_model, self.d_inner, 1,bias=bias, **factory_kwargs)

        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state*2), bias=False, **factory_kwargs)
        self.x_proj_weight = nn.Parameter(self.x_proj.weight)
        del self.x_proj

        self.dt_projs = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
        self.dt_projs_weight = nn.Parameter(self.dt_projs.weight)
        self.dt_projs_bias = nn.Parameter(self.dt_projs.bias)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, dt_init)
        self.Ds = self.D_init(self.d_inner, dt_init)

        self.selective_scan = selective_scan_fn

        self.out_norm = LayerNorm(self.d_inner)
        self.out_proj = nn.Conv2d(self.d_inner, self.d_model, 1,bias=bias, **factory_kwargs)

        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        num_group=d_model//head_dim
        self.da_scan = Dynamic_Adaptive_Scan(channels=self.d_inner,group=num_group)
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, bias=True,**factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=bias, **factory_kwargs)

        if bias:
            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))

            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            dt_proj.bias._no_reinit = True

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        elif dt_init == "simple":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.randn((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.randn((d_inner)))
                dt_proj.bias._no_reinit = True
        elif dt_init == "zero":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.rand((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.rand((d_inner)))
                dt_proj.bias._no_reinit = True
        else:
            raise NotImplementedError

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, init, device=None):
        if init=="random" or "constant":
            # S4D real initialization
            A = repeat(
                torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=d_inner,
            ).contiguous()
            A_log = torch.log(A)
            A_log = nn.Parameter(A_log)
            A_log._no_weight_decay = True
        elif init=="simple":
            A_log = nn.Parameter(torch.randn((d_inner, d_state)))
        elif init=="zero":
            A_log = nn.Parameter(torch.zeros((d_inner, d_state)))
        else:
            raise NotImplementedError
        return A_log

    @staticmethod
    def D_init(d_inner, init="random", device=None):
        if init=="random" or "constant":
            # D "skip" parameter
            D = torch.ones(d_inner, device=device)
            D = nn.Parameter(D)
            D._no_weight_decay = True
        elif init == "simple" or "zero":
            D = nn.Parameter(torch.ones(d_inner))
        else:
            raise NotImplementedError
        return D

    def ssm(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W

        xs = x.view(B, -1, L)

        x_dbl = torch.matmul(self.x_proj_weight.view(1, -1, C), xs)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=1)
        dts = torch.matmul(self.dt_projs_weight.view(1, C, -1), dts)

        As = -torch.exp(self.A_logs)
        Ds = self.Ds
        dts = dts.contiguous()
        dt_projs_bias = self.dt_projs_bias

        h = self.selective_scan(
            xs, dts,
            As, Bs, None,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        )

        h=h.reshape(B,C,H*W)

        y = h * Cs
        y = y + xs * Ds.view(-1, 1)

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        B, C,H, W = x.shape
        input=x
        x = self.in_proj(x)
        x = self.act(self.conv2d(x))


        x=self.da_scan(input,x.permute(0, 2,3,1).contiguous())
        y = self.ssm(x)
        y=y.reshape(B, C,H, W)

        y = self.out_norm(y)
        y = self.out_proj(y)
        if self.dropout is not None:
            y = self.dropout(y)
        return y



class Block(nn.Module):
    def __init__(
            self,
            dim,
            token_mixer=nn.Identity,
            head_dim=24,
            mlp_layer=ConvFFN,
            mlp_ratio=4,
            act_layer=nn.GELU,
            drop_path=0.,
            ffnconv=True, index=None, layerscale=False,
            norm_cfg=dict(type='SyncBN', requires_grad=True)

    ):
        super().__init__()
        if isinstance(token_mixer, list):
            if index % 2 == 0:
                self.token_mixer = token_mixer[0](dim, head_dim=head_dim)
            elif index % 2 == 1:
                self.token_mixer = token_mixer[1](dim, head_dim=head_dim)
        else:
            self.token_mixer = token_mixer(dim, head_dim=head_dim)

        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer, ffnconv=ffnconv)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        layer_scale_init_value = 1e-6
        self.layerscale = layerscale
        if layerscale:
            print('layerscale', dim)
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.pos_embed = ResDWC(dim, 3)
    def forward(self, x):
        x = self.pos_embed(x)
        if self.layerscale == False:

            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))

        return x


class DAMambaStage(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            ds_stride=2,
            depth=2,
            drop_path_rates=None,
            token_mixer=nn.Identity,
            head_dim=24,
            act_layer=nn.GELU,
            norm_layer=None,
            mlp_ratio=4,
            ffnconv=True,
            layerscale=False,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
    ):
        super().__init__()
        self.grad_checkpointing = False
        if ds_stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chs, out_chs, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                build_norm_layer(norm_cfg, out_chs)[1],
            )
        else:
            self.downsample = nn.Identity()

        drop_path_rates = drop_path_rates or [0.] * depth
        stage_blocks = []

        for i in range(depth):
            stage_blocks.append(Block(
                dim=out_chs,
                drop_path=drop_path_rates[i],
                token_mixer=token_mixer,
                head_dim=head_dim,
                act_layer=act_layer,
                mlp_ratio=mlp_ratio,
                ffnconv=ffnconv, index=i, layerscale=layerscale
            ))
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class LayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        if self.training:
            x = x.permute(0, 2, 3, 1).contiguous()
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2).contiguous()
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2)
        return x




class CrossDAMamba(nn.Module):
    r"""
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: (3, 3, 9, 3)
        dims (tuple(int)): Feature dimension at each stage. Default: (96, 192, 384, 768)
        token_mixers: Token mixer function. Default: nn.Identity
        norm_layer: Normalziation layer. Default: nn.BatchNorm2d
        act_layer: Activation function for MLP. Default: nn.GELU
        mlp_ratios (int or tuple(int)): MLP ratios. Default: (4, 4, 4, 3)
        head_fn: classifier head
        drop_rate (float): Head dropout rate
        drop_path_rate (float): Stochastic depth rate. Default: 0.
    """

    def __init__(
            self,
            in_chans=3,
            other_chans=1,
            num_classes=1000,
            depths=(3, 3, 9, 3),
            dims=(96, 192, 384, 768),
            token_mixers=[DASSM, DASSM, DASSM, DASSM],
            head_dim=24,
            norm_layer=nn.BatchNorm2d,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            act_layer=nn.GELU,
            mlp_ratios=(4, 4, 4, 4),
            head_fn=MlpHead,
            drop_rate=0.,
            drop_path_rate=0.1,
            layerscale=[False,False,False,False],
            pretrained=None,
            **kwargs,
    ):
        super().__init__()
        print(depths,dims,layerscale)
        num_stage = len(depths)
        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * num_stage
        if not isinstance(mlp_ratios, (list, tuple)):
            mlp_ratios = [mlp_ratios] * num_stage
        self.ffnconvs = [True, True, True, True]
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0] // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            build_norm_layer(norm_cfg, dims[0] // 2)[1],
            nn.GELU(),

            nn.Conv2d(dims[0] // 2, dims[0] // 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            build_norm_layer(norm_cfg, dims[0] // 2)[1],
            nn.GELU(),

            nn.Conv2d(dims[0] // 2, dims[0], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            build_norm_layer(norm_cfg, dims[0])[1],
            nn.GELU(),

            nn.Conv2d(dims[0], dims[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            build_norm_layer(norm_cfg, dims[0])[1],
        )

        self.other_stem = nn.Sequential(
            nn.Conv2d(other_chans, dims[0] // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            build_norm_layer(norm_cfg, dims[0] // 2)[1],
            nn.GELU(),

            nn.Conv2d(dims[0] // 2, dims[0] // 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            build_norm_layer(norm_cfg, dims[0] // 2)[1],
            nn.GELU(),

            nn.Conv2d(dims[0] // 2, dims[0], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            build_norm_layer(norm_cfg, dims[0])[1],
            nn.GELU(),

            nn.Conv2d(dims[0], dims[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            build_norm_layer(norm_cfg, dims[0])[1],
        )

        self.stages = nn.Sequential()
        self.other_stages = nn.Sequential()
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        print(dp_rates)
        stages = []
        other_stages = []
        prev_chs = dims[0]
        self.cross = nn.ModuleList()
        num_heads = [2, 4, 8, 16]
        
        for i in range(num_stage):
            out_chs = dims[i]
            stages.append(DAMambaStage(
                prev_chs,
                out_chs,
                ds_stride=2 if i > 0 else 1,
                depth=depths[i],
                drop_path_rates=dp_rates[i],
                act_layer=act_layer,
                token_mixer=token_mixers[i],
                head_dim=head_dim,
                norm_layer=norm_layer,
                mlp_ratio=mlp_ratios[i],
                ffnconv=self.ffnconvs[i],
                layerscale=layerscale[i]

            ))
            norm = LayerNorm(dims[i], eps=1e-6)

            other_stages.append(DAMambaStage(
                prev_chs,
                out_chs,
                ds_stride=2 if i > 0 else 1,
                depth=depths[i],
                drop_path_rates=dp_rates[i],
                act_layer=act_layer,
                token_mixer=token_mixers[i],
                head_dim=head_dim,
                norm_layer=norm_layer,
                mlp_ratio=mlp_ratios[i],
                ffnconv=self.ffnconvs[i],
                layerscale=layerscale[i]

            ))
            other_norm = LayerNorm(dims[i], eps=1e-6)
            setattr(self, f"norm{i + 1}", norm)
            setattr(self, f"other_norm{i + 1}", other_norm)
            
            cross = CrossSwinBlock(out_chs,num_heads=num_heads[i],feedforward_channels=prev_chs*4, window_size=7,drop_path_rate=dp_rates[i][-1])
            self.cross.append(cross)
            prev_chs = out_chs
        self.stages = nn.Sequential(*stages)
        self.other_stages = nn.Sequential(*other_stages)
        self.num_features = prev_chs
        for n, m in self.named_modules():
            self._init_weights(m, n)
        if pretrained:
            self.init_weights(pretrained)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    def _init_weights(self, m: nn.Module, name: str = ''):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            # logger = get_root_logger()
            logger = logging.getLogger()
            state_dict = torch.load(pretrained, map_location='cpu')['model']
            other_state_dict = {'other_'+k:v for k,v in state_dict.items()}
            other_state_dict.pop('other_stem.0.weight')
            state_dict.update(other_state_dict)
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            print(f"Missing keys: {missing_keys}")
            print(f"Unexpected keys: {unexpected_keys}")

            # load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'norm'}

    def forward_features(self, x,x2):
        x = self.stem(x)
        x2 = self.other_stem(x2)
        outs = []
        for i, (stage, other_stage) in enumerate(zip(self.stages, self.other_stages)):
            x = stage(x)
            x2 = other_stage(x2)
            norm = getattr(self, f"norm{i + 1}")
            other_norm = getattr(self, f"other_norm{i + 1}")
            x = self.cross[i](x.flatten(2).permute(0, 2, 1),x2.flatten(2).permute(0, 2, 1),(x.shape[2],x.shape[3]))
            x = x.permute(0, 2, 1).view(x.shape[0], x.shape[2], int(x.shape[1]**0.5), int(x.shape[1]**0.5))
            x = norm(x)
            x2 = other_norm(x2)
            
            outs.append(x)
        # return x
        return outs

    def forward_head(self, x):
        x = self.head(x)
        return x

    def forward(self, x, x2):
        x = self.forward_features(x,x2)

        return x




@MODELS.register_module()
class CrossDAMamba_tiny(CrossDAMamba):
    def __init__(self, **kwargs):
        super().__init__(
            depths=[3, 4, 12, 5], dims=[80, 160, 320, 512], mlp_ratios=(4, 4, 3, 3),
            token_mixers = [DASSM, DASSM, DASSM, DASSM],
            head_dim = 16,
            drop_rate=0.0, drop_path_rate=0.3,
            pretrained=kwargs['pretrained'])

@MODELS.register_module()
class CrossDAMamba_small(CrossDAMamba):
    def __init__(self, **kwargs):
        super().__init__(
            depths=[4, 8, 20, 6], dims=[96, 192, 384, 512], mlp_ratios=(4, 4, 3, 3),
            token_mixers = [DASSM, DASSM, DASSM, DASSM],
            head_dim = 16,
            drop_rate=0.0, drop_path_rate=0.4,
            layerscale=[False,False,True,True],
            pretrained=kwargs['pretrained'])


@MODELS.register_module()
class CrossDAMamba_base(CrossDAMamba):
    def __init__(self, **kwargs):
        super().__init__(
            depths=[4, 8, 25, 8], dims=[112, 224, 448, 640], mlp_ratios=(4, 4, 3, 4),
            token_mixers = [DASSM, DASSM, DASSM, DASSM],
            head_dim = 16,
            drop_rate=0.0, drop_path_rate=0.6,
            layerscale=[False,False,True,True],
            pretrained=kwargs['pretrained'])