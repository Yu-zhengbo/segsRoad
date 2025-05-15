# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
import warnings
from typing import Sequence
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer)
from mmseg.registry import MODELS
from mmengine.model import BaseModule,ModuleList

from mmengine import ConfigDict
from torch.nn.modules.utils import _pair as to_2tuple
from mmseg.models.backbones.damamba import DASSM,Block,LayerNorm,ResDWC,ConvFFN,DropPath
from mmseg.utils.diag_scan import DiagScanModule,FastDiagScan

class DASSM(DASSM):
    def __init__(self,scan, *args, **kwargs):
        super(DASSM, self).__init__(*args, **kwargs)
        self.scan = scan
        self.cache = {self.scan.H:self.scan}

    def ssm(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        if H not in self.cache:
            self.cache[H] = FastDiagScan(H, W).to(x.device)
            # self.scan = self.cache[H]
        x_rd, x_ld = self.cache[H](x)
        x = x.view(B,-1, L)
        x_inv = torch.flip(x,[-1])

        xs = torch.cat([x.unsqueeze(1), x_inv.unsqueeze(1), x_rd.unsqueeze(1), x_ld.unsqueeze(1)], dim=1)

        x_dbl = torch.matmul(self.x_proj_weight.view(1, -1, C), xs)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.matmul(self.dt_projs_weight.view(1, C, -1), dts)

        As = -torch.exp(self.A_logs)
        Ds = self.Ds
        dts = dts.contiguous()
        dt_projs_bias = self.dt_projs_bias

        # h = self.selective_scan(
        #     xs, dts,
        #     As, Bs, None,
        #     z=None,
        #     delta_bias=dt_projs_bias,
        #     delta_softplus=True,
        #     return_last_state=False,
        # )
        hs = []
        for i in range(4):
            h = self.selective_scan(
                xs[:,0,], dts[:,0],
                As, Bs[:,0], None,
                z=None,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
                return_last_state=False,
            )
            hs.append(h)
        h = torch.stack(hs, dim=1).squeeze(3)

        y = h * Cs
        y = y + xs * Ds.view(-1, 1)

        y_rd = self.cache[H].recover(y[:,2], 'rd')
        y_ld = self.cache[H].recover(y[:,3], 'ld')
        y_inv = torch.flip(y[:,1],[-1])
        y = y[:,0] + y_rd + y_ld + y_inv

        return y

@MODELS.register_module()
class MambaDDPBlock(nn.Module):
    def __init__(
        self,
        scan,
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
        self.token_mixer = token_mixer(scan, dim, head_dim=head_dim)

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

        # self.token_mixer = args['token_mixer'](scan, args['dim'],head_dim=args['head_dim'])
        self.time_mlp = nn.Sequential( # [2, 1024]
            nn.SiLU(),
            nn.Linear(1024, 512) # [2, 512]
        ) 

    def forward_mamba(self,x):
        x = self.pos_embed(x)
        if self.layerscale == False:

            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x
    

    def forward(self, query,time, query_pos=None):
        b,c,h,w = query.shape
        if query_pos is not None:
            query = query + query_pos
        query = self.forward_mamba(query)
        time = self.time_mlp(time) # [2, 1024] -> [2, 512]
        time = rearrange(time, 'b c -> 1 b c') # [2, 512] -> [1, 2, 512]
        scale, shift = time.chunk(2, dim=2) # [1, 2, 256] * 2
        # query = query.flatten(2).permute(2,0,1) # [2, 1024, 4, 4] -> [2, 1024, 16]
        query = rearrange(query, 'b c h w -> (h w) b c').contiguous() # [2, 16, 512]
        query = query * (scale + 1) + shift
        query = rearrange(query, '(h w) b c -> b c h w', h=h, w=w).contiguous() # [2, 16, 512] -> [2, 512, 4, 4]
        return query

@MODELS.register_module()
class MambaDDPSequence(BaseModule):
    def __init__(
            self,
            in_chs,
            depth=6,
            drop_path_rates=None,
            token_mixer='DASSM',
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
        # self.scan = DiagScanModule(128,128)
        self.scan = FastDiagScan(128,128)
        self.embed_dims = in_chs

        drop_path_rates = drop_path_rates or [0.] * depth
        self.stage_blocks = nn.ModuleList()
        if token_mixer == 'DASSM':
            token_mixer = DASSM
        for i in range(depth):
            self.stage_blocks.append(MambaDDPBlock(
                scan=self.scan,
                dim=in_chs,
                drop_path=drop_path_rates[i],
                token_mixer=token_mixer,
                head_dim=head_dim,
                act_layer=act_layer,
                mlp_ratio=mlp_ratio,
                ffnconv=ffnconv, index=i, layerscale=layerscale
            ))


    def forward(self, query,time, query_pos=None):
        for blk in self.stage_blocks:
            query = blk(query,time, query_pos=query_pos)
        return query