# Copyright (c) OpenMMLab. All rights reserved.
import warnings

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
os.environ['HF_HUB_OFFLINE'] = '1'

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
class DinoV3Vit(BaseModule):

    def __init__(
            self,
            model = 'vit_large_patch16_dinov3_qkvb.sat493m',
            pretrained = True,
            features_only = True,
            out_indices = (5, 11, 17, 23),
            freeze = True,
            use_lora = True,
            rank = 3,
        ):
        super(DinoV3Vit, self).__init__()

        self.dinov3 =timm.create_model(
            model,
            pretrained=pretrained,
            features_only=features_only,
            out_indices=out_indices,
        )
        
        self.freeze_backbone = freeze
        if freeze:
            for param in self.dinov3.parameters():
                param.requires_grad = False
            self.dinov3.eval()
        if use_lora:
            self.lora_As = []
            self.lora_Bs = []
            self.lora_blocks = []  # 记录哪些 block 被替换（可选）
            self._inject_lora_into_blocks(rank)
            self._init_lora_params()
    
    def init_weights(self):
        return
    
    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.dinov3.eval()  # 冻结时也常常希望关掉 dropout
            for p in self.dinov3.parameters():
                p.requires_grad = False
    
    def fuse(self):
        for m in self.dinov3.modules():
            if isinstance(m, LoRA):
                print('fuse')
                m.fuse_()
    
    def _make_lora_pair(self, dim: int, r: int):
        # 和你原版一致：A/B 都无 bias
        A = nn.Linear(dim, r, bias=False)
        B = nn.Linear(r, dim, bias=False)
        return A, B

    def _inject_lora_into_blocks(self, rank: int):
        # 默认对全部 blocks 注入；如果你想挑选层，可改成传入 indices
        for i, block in enumerate(self.dinov3.model.blocks):
            block.attn.qkv_bias_separate = True
            qkv = block.attn.qkv
            dim = qkv.in_features

            A_q, B_q = self._make_lora_pair(dim, rank)
            A_v, B_v = self._make_lora_pair(dim, rank)

            # 注册到 ModuleList（保证可训练/可保存）
            self.lora_As.extend([A_q, A_v])
            self.lora_Bs.extend([B_q, B_v])

            block.attn.qkv = LoRA(qkv, A_q, B_q, A_v, B_v)
            self.lora_blocks.append(i)

    def _init_lora_params(self):
        # A: kaiming，B: 0（与你原逻辑一致）
        for A in self.lora_As:
            nn.init.kaiming_uniform_(A.weight, a=math.sqrt(5))
        for B in self.lora_Bs:
            nn.init.zeros_(B.weight)

    def forward(self, x):  # should return a tuple
        outputs = self.dinov3(x)
        return outputs



if __name__ == "__main__":
    # dinov3 = DinoV3ConvNeXt()     # 默认base模型
    dinov3 = DinoV3Vit()
    input = torch.randn(1, 3, 512, 512)
    # 打印每个特征图的shape，后续配置文件修改neck里的参数需要根据这个输出来。
    print([i.shape for i in dinov3(input)])

