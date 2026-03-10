# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmengine.model import BaseModule
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from mmseg.registry import MODELS
import timm
import torch
import math
import os
from torch.nn.init import normal_
from timm.models.layers import DropPath, trunc_normal_
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_
from functools import partial
import torch.utils.checkpoint as cp
# from ops.modules import MSDeformAttn
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
# from torch.nn import Identity as MultiScaleDeformableAttention
from mmseg.utils.transformer import BaseTransformerLayer
from mmengine.config import ConfigDict
os.environ['HF_HUB_OFFLINE'] = '1'
# torch.autograd.set_detect_anomaly(True)

def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs(x):
    bs, c, h, w = x.shape
    spatial_shapes = torch.as_tensor([(h // 8, w // 8),
                                      (h // 16, w // 16),
                                      (h // 32, w // 32)],
                                     dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 16, w // 16)], x.device)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

    spatial_shapes = torch.as_tensor([(h // 16, w // 16)], dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 8, w // 8),
                                             (h // 16, w // 16),
                                             (h // 32, w // 32)], x.device)
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]

    return deform_inputs1, deform_inputs2


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21
        x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class Extractor(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        # self.attn = MultiScaleDeformableAttention(embed_dims=dim, num_levels=n_levels, num_heads=num_heads,
        #                          num_points=n_points, value_proj_ratio=deform_ratio)
        self.attn = BaseTransformerLayer(use_time_mlp=False, num_points=n_points, value_proj_ratio=deform_ratio,
                attn_cfgs=ConfigDict(
                    type='MultiScaleDeformableAttention',
                    embed_dims=dim,
                    num_levels=n_levels,
                    num_heads=num_heads,
                    dropout=0.),
                ffn_cfgs=ConfigDict(
                    type='FFN',
                    embed_dims=dim,
                    feedforward_channels=1024,
                    ffn_drop=0.,
                    act_cfg=ConfigDict(type='GELU')),
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))
        
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):

        def _inner_forward(query, feat):

            # attn = self.attn(self.query_norm(query), reference_points,
            #                  self.feat_norm(feat), spatial_shapes,
            #                  level_start_index, None)
            attn = self.attn(self.query_norm(query), self.feat_norm(feat), self.feat_norm(feat),
                             reference_points = reference_points,
                             spatial_shapes = spatial_shapes,
                             level_start_index = level_start_index)
            query = query + attn

            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query.permute(1,0,2)), H, W)).permute(1,0,2)
            return query

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class Injector(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        # self.attn = MultiScaleDeformableAttention(embed_dims=dim, num_levels=n_levels, num_heads=num_heads,
        #                          num_points=n_points, value_proj_ratio=deform_ratio)
        self.attn = BaseTransformerLayer(use_time_mlp=False, num_points=n_points, value_proj_ratio=deform_ratio,
                attn_cfgs=ConfigDict(
                    type='MultiScaleDeformableAttention',
                    embed_dims=dim,
                    num_levels=n_levels,
                    num_heads=num_heads,
                    dropout=0.),
                ffn_cfgs=ConfigDict(
                    type='FFN',
                    embed_dims=dim,
                    feedforward_channels=1024,
                    ffn_drop=0.,
                    act_cfg=ConfigDict(type='GELU')),
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):

        def _inner_forward(query, feat):

            # attn = self.attn(self.query_norm(query), reference_points,
            #                  self.feat_norm(feat), spatial_shapes,
            #                  level_start_index, None)

            attn = self.attn(self.query_norm(query), self.feat_norm(feat), self.feat_norm(feat),
                    reference_points = reference_points,
                    spatial_shapes = spatial_shapes,
                    level_start_index = level_start_index)
            return query + self.gamma * attn

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class InteractionBlock(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
                 deform_ratio=1.0, extra_extractor=False, with_cp=False):
        super().__init__()

        self.injector = Injector(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                 with_cp=with_cp)
        self.extractor = Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
                          with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                          drop=drop, drop_path=drop_path, with_cp=with_cp)
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W, rot_pos_embed):
        tail = self.injector(query=x[:,5:].permute(1,0,2), reference_points=deform_inputs1[0],
                          feat=c.permute(1,0,2), spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2]).permute(1,0,2)
        x = torch.cat([x[:, :5], tail], dim=1)
        for idx, blk in enumerate(blocks):
            x = blk(x, rope=rot_pos_embed)
        c = self.extractor(query=c.permute(1,0,2), reference_points=deform_inputs2[0],
                           feat=x[:,5:].permute(1,0,2), spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W).permute(1,0,2)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c.permute(1,0,2), reference_points=deform_inputs2[0],
                              feat=x[:,5:].permute(1,0,2), spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W).permute(1,0,2)
        return x, c
    
    

class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384, with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        self.stem = nn.Sequential(*[
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(2 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):

        def _inner_forward(x):
            c1 = self.stem(x)
            c2 = self.conv2(c1)
            c3 = self.conv3(c2)
            c4 = self.conv4(c3)
            c1 = self.fc1(c1)
            c2 = self.fc2(c2)
            c3 = self.fc3(c3)
            c4 = self.fc4(c4)

            bs, dim, _, _ = c1.shape
            # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
            c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
            c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
            c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

            return c1, c2, c3, c4

        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)
        return outs

class LoRA(nn.Module):
    """
    LoRA wrapper for a fused qkv Linear (out = 3*dim).
    Applies LoRA to Q and V only:
      Q slice: [0:dim]
      K slice: [dim:2*dim] (unchanged)
      V slice: [2*dim:3*dim]
    """

    def __init__(
        self,
        qkv: nn.Linear,
        linear_a_q: nn.Linear,
        linear_b_q: nn.Linear,
        linear_a_v: nn.Linear,
        linear_b_v: nn.Linear,
    ):
        super().__init__()
        if not isinstance(qkv, nn.Linear):
            raise TypeError("qkv must be nn.Linear")

        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v

        self.dim = qkv.in_features
        self.in_features = qkv.in_features
        self.out_features = qkv.out_features

        if self.out_features != 3 * self.dim:
            raise ValueError(f"qkv.out_features must be 3*dim, got {self.out_features} vs {3*self.dim}")

        # fuse 状态与缓存（用于 unfuse）
        self._fused = False
        self._cached_delta = None  # (dW_q, db_q, dW_v, db_v)

    def forward(self, x) -> torch.Tensor:
        # 如果已经 fuse，就直接走 qkv（不再额外算 LoRA）
        if self._fused:
            return self.qkv(x)

        qkv_out = self.qkv(x)  # (B, N, 3*dim)

        # LoRA 增量
        dq = self.linear_b_q(self.linear_a_q(x))  # (B,N,dim)
        dv = self.linear_b_v(self.linear_a_v(x))  # (B,N,dim)

        # 避免 inplace（训练更稳）
        q, k, v = qkv_out.split(self.dim, dim=-1)
        q = q + dq
        v = v + dv
        return torch.cat([q, k, v], dim=-1)

    @torch.no_grad()
    def _delta_W_b(self, A: nn.Linear, B: nn.Linear):
        """
        Compute delta weight and delta bias for composition: B(A(x))
        A: (dim -> r), B: (r -> dim)
        Returns:
          dW: (dim, dim)
          db: (dim,) or None if qkv has no bias
        """
        device = self.qkv.weight.device
        dtype = self.qkv.weight.dtype

        A_w = A.weight.to(device=device, dtype=dtype)  # (r, dim)
        B_w = B.weight.to(device=device, dtype=dtype)  # (dim, r)

        dW = B_w @ A_w  # (dim, dim)

        db = None
        if self.qkv.bias is not None:
            db = torch.zeros((self.dim,), device=device, dtype=dtype)
            # 你创建 A/B 都是 bias=False，所以通常这两句不会进，但保留通用性
            if A.bias is not None:
                db = db + (B_w @ A.bias.to(device=device, dtype=dtype))
            if B.bias is not None:
                db = db + B.bias.to(device=device, dtype=dtype)

        return dW, db

    @torch.no_grad()
    def fuse_(self, strict: bool = True):
        """
        Merge LoRA weights into qkv weights/bias in-place.
        After fuse, forward becomes just qkv(x).
        """
        if self._fused:
            return self

        # sanity checks（可选）
        if strict:
            if self.linear_a_q.in_features != self.dim or self.linear_a_v.in_features != self.dim:
                raise ValueError("A_q/A_v in_features must equal dim")
            if self.linear_b_q.out_features != self.dim or self.linear_b_v.out_features != self.dim:
                raise ValueError("B_q/B_v out_features must equal dim")

        dW_q, db_q = self._delta_W_b(self.linear_a_q, self.linear_b_q)
        dW_v, db_v = self._delta_W_b(self.linear_a_v, self.linear_b_v)

        # 写入 qkv 的 Q / V 段
        W = self.qkv.weight  # (3*dim, dim)
        W[: self.dim, :] += dW_q
        W[2 * self.dim : 3 * self.dim, :] += dW_v

        if self.qkv.bias is not None:
            if db_q is not None:
                self.qkv.bias[: self.dim] += db_q
            if db_v is not None:
                self.qkv.bias[2 * self.dim : 3 * self.dim] += db_v

        # 缓存 delta，方便 unfuse
        self._cached_delta = (dW_q, db_q, dW_v, db_v)
        self._fused = True
        return self

    @torch.no_grad()
    def unfuse_(self):
        """
        Revert a previous fuse_() by subtracting cached deltas.
        Only works if fuse_() was called and cache exists.
        """
        if not self._fused:
            return self
        if self._cached_delta is None:
            raise RuntimeError("No cached deltas to unfuse. Call fuse_() first.")

        dW_q, db_q, dW_v, db_v = self._cached_delta

        W = self.qkv.weight
        W[: self.dim, :] -= dW_q
        W[2 * self.dim : 3 * self.dim, :] -= dW_v

        if self.qkv.bias is not None:
            if db_q is not None:
                self.qkv.bias[: self.dim] -= db_q
            if db_v is not None:
                self.qkv.bias[2 * self.dim : 3 * self.dim] -= db_v

        self._fused = False
        self._cached_delta = None
        return self


@MODELS.register_module()
class DINOAdapter(nn.Module):
    def __init__(
        self,
        model='vit_large_patch16_dinov3_qkvb.sat493m',             # 'vit_7b_patch16_dinov3.sat493m',
        embed_dim=1024,
        init_values=1e-6,
        drop_path_rate=0.3,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        with_cp=True,  # set with_cp=True to save memory
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        with_cffn=True,
        add_vit_feature=True,
        freeze=True,
    ):
        super().__init__()
        self.eva = timm.create_model(
            model,
            pretrained=True,
            features_only=True,
        ).model
        
        
        self.freeze_backbone = freeze
        if freeze:
            for param in self.eva.parameters():
                param.requires_grad = False
            self.eva.eval()
        
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False)
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=drop_path_rate,
                             with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=True if i == len(interaction_indexes) - 1 else False,
                             with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)
        
    # def _init_deform_weights(self, m):
    #     if isinstance(m, MultiScaleDeformableAttention):
    #         m._reset_parameters()
    def _init_deform_weights(self, m):
        if not isinstance(m, MultiScaleDeformableAttention):
            return
        constant_(m.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(
            m.num_heads, dtype=torch.float32) * (2.0 * math.pi / m.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
                         m.num_heads, 1, 1, 2).repeat(1, m.num_levels, m.num_points, 1)
        for i in range(m.num_points):
            grid_init[:, :, i, :] *= i + 1

        with torch.no_grad():
            m.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(m.attention_weights.weight.data, 0.)
        constant_(m.attention_weights.bias.data, 0.)
        xavier_uniform_(m.value_proj.weight.data)
        constant_(m.value_proj.bias.data, 0.)
        xavier_uniform_(m.output_proj.weight.data)
        constant_(m.output_proj.bias.data, 0.)
            
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4
    
    def dino_forward(self, x):
        # forward pass
        B, _, height, width = x.shape
        x = self.eva.patch_embed(x)
        x, rot_pos_embed = self.eva._pos_embed(x)
        x = self.eva.norm_pre(x)
        
        intermediates = []
        
        for i, blk in enumerate(self.eva.blocks):
            x = blk(x, rope=rot_pos_embed)
            intermediates.append(x)
        
        intermediates = [y[:, self.eva.num_prefix_tokens:] for y in intermediates]
            # reshape to BCHW output format
        H, W = self.eva.patch_embed.dynamic_feat_size((height, width))
        intermediates = [y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]

        return intermediates
    
    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)
        
        # Patch Embedding forward
        x = self.eva.patch_embed(x)
        H, W = x.shape[1:3]
        bs, n, dim = x.shape[0], H*W, x.shape[-1]
        
        x, rot_pos_embed = self.eva._pos_embed(x)
        x = self.eva.norm_pre(x)
        
        # Interaction
        outs = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.eva.blocks[indexes[0]:indexes[-1] + 1],
                              deform_inputs1, deform_inputs2, H, W, rot_pos_embed)
            outs.append(x[:,5:].transpose(1, 2).view(bs, dim, H, W).contiguous())
        # return outs
        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]
    
    def init_weights(self):
        return
    
    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.eva.eval()  # 冻结时也常常希望关掉 dropout
            for p in self.eva.parameters():
                p.requires_grad = False



if __name__ == "__main__":
    # dinov3 = DinoV3ConvNeXt()     # 默认base模型
    dinov3 = DinoV3Vit()
    input = torch.randn(1, 3, 512, 512)
    # 打印每个特征图的shape，后续配置文件修改neck里的参数需要根据这个输出来。
    print([i.shape for i in dinov3(input)])

