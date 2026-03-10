# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
try:
    from skimage.morphology import skeletonize
except Exception:
    skeletonize = None

def get_neighbor_diff_num(im: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    计算每个像素与其 k×k 邻域内“不同”的像素个数（等价你 TF+scipy 的逻辑）
    im: (H, W) 或 (B, 1, H, W) 的 0/1 二值图（uint8/bool/int 都行）
    return: 同 shape 的 float32
    """
    if im.dim() == 2:
        im = im.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    elif im.dim() == 3:
        im = im.unsqueeze(1)  # (B,1,H,W)
    assert im.dim() == 4 and im.size(1) == 1, f"im shape must be (B,1,H,W), got {im.shape}"

    im_f = im.float()
    B, _, H, W = im_f.shape

    # 全 1 卷积核
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=im.device, dtype=im_f.dtype)

    # TF 的 boundary='symm' 是“对称 padding”（reflect）
    pad = kernel_size // 2
    # reflect 要求 pad < size，通常 H,W>pad 都满足
    im_pad = F.pad(im_f, (pad, pad, pad, pad), mode="reflect")

    # 卷积得到邻域内 1 的个数（包含中心）
    conv = F.conv2d(im_pad, kernel)  # (B,1,H,W)

    # 等价 TF: num = where(im==0, conv, k*k - conv)
    kk = float(kernel_size * kernel_size)
    num = torch.where(im_f == 0.0, conv, kk - conv)

    return num.squeeze(1)  # (B,H,W)


def neighbor_loss(
    logits: torch.Tensor,
    label: torch.Tensor,
    neigh_size: int = 3,
    k: float = 1.0,
    ignore_index: int | None = None,
) -> torch.Tensor:
    """
    logits: (B,C,H,W)
    label : (B,H,W)  (long) 取值 [0..C-1]，可选 ignore_index
    neigh_size: NeighSize
    k: 你的 weight 系数，最终 weight = normalized*8*k + 1
    """
    assert logits.dim() == 4, f"logits must be (B,C,H,W), got {logits.shape}"
    B, C, H, W = logits.shape

    assert label.shape == (B, H, W), f"label shape must be (B,H,W), got {label.shape}"

    # 1) 离散预测图（不可导，和你 TF 的 argmax 一样）
    with torch.no_grad():
        pred_cls = logits.argmax(dim=1)  # (B,H,W) long

        # 你原 getNeighborDiffNum 假设 im 是 0/1 二值图
        # 如果是二分类：pred_cls 已经是 0/1
        # 如果多分类：这里默认把“非0类”都当成 1（你可以按需要改）
        if C > 2:
            im = (pred_cls != 0).to(torch.uint8)  # (B,H,W) 0/1
        else:
            im = pred_cls.to(torch.uint8)

        diff_num = get_neighbor_diff_num(im, kernel_size=neigh_size)  # (B,H,W) float

        weight = diff_num * float(k) + 1.0  # (B,H,W)

        # ignore_index：忽略处 weight 设 0（不贡献 loss）
        if ignore_index is not None:
            weight = weight.masked_fill(label == ignore_index, 0.0)

    # 3) 像素级 CE（不 reduce），再乘 weight
    ce = F.cross_entropy(
        logits, label.long(),
        reduction="none",
        ignore_index=(ignore_index if ignore_index is not None else -100),
    )  # (B,H,W)

    loss = ce * weight

    # 4) mean（忽略 ignore 的像素）
    if ignore_index is not None:
        valid = (label != ignore_index).float()
        # 避免除 0
        return loss.sum() / (valid.sum().clamp_min(1.0))
    else:
        return loss.mean()



@MODELS.register_module()
class NeighborLoss(nn.Module):
    def __init__(self,neigh_size: int = 3, k: float = 1.0, loss_name: str = 'neighbor_loss'):
        super().__init__()
        self.neigh_size = neigh_size
        self.k = k
        self._loss_name = loss_name
    
    def forward(self, pred, target, ignore_index=None, **kwargs):
        return neighbor_loss(
                            pred,
                            target,
                            self.neigh_size,
                            self.k,
                            ignore_index
                        )
    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
    

def _skeletonize_batch(bin_mask_bhw: torch.Tensor) -> torch.Tensor:
    """
    bin_mask_bhw: (B,H,W) uint8/bool/int，取值0/1
    return: (B,H,W) float32 0/1 on CPU
    """
    if skeletonize is None:
        raise ImportError(
            "skimage is required for skeletonize. Please `pip install scikit-image` "
            "or replace _skeletonize_batch with your own getSkeleton implementation."
        )

    bin_mask_bhw = bin_mask_bhw.detach().to("cpu")
    B, H, W = bin_mask_bhw.shape
    out = torch.zeros((B, H, W), dtype=torch.float32)

    for i in range(B):
        arr = bin_mask_bhw[i].numpy().astype(bool)
        sk = skeletonize(arr).astype("float32")
        out[i] = torch.from_numpy(sk)

    return out  # CPU float32


@MODELS.register_module()
class DirectionLoss(nn.Module):
    """
    PyTorch implementation of the MATLAB layer gl_fus forwardLoss.
    """
    def __init__(self, kernel_len: int = 5, k: float = 1.0, loss_name: str = 'direction_loss'):
        super().__init__()
        self.k = float(k)
        self.kernel_len = int(kernel_len)
        self._loss_name = loss_name
        L = self.kernel_len

        # 预先构建四种方向的核（注册为 buffer，自动跟随 device/dtype）
        # 水平 1xL
        self.register_buffer("k_h1", torch.ones(1, 1, 1, L))
        self.register_buffer("k_h10", torch.ones(1, 1, 1, L) * 10.0)

        # 垂直 Lx1
        self.register_buffer("k_v1", torch.ones(1, 1, L, 1))
        self.register_buffer("k_v10", torch.ones(1, 1, L, 1) * 10.0)

        # 对角 /
        diag = torch.eye(L).flip(1)  # 反对角线
        self.register_buffer("k_d1", diag.view(1, 1, L, L))
        self.register_buffer("k_d10", (diag * 10.0).view(1, 1, L, L))

        # 对角 \
        diag2 = torch.eye(L)
        self.register_buffer("k_b1", diag2.view(1, 1, L, L))
        self.register_buffer("k_b10", (diag2 * 10.0).view(1, 1, L, L))

    @staticmethod
    def _same_pad(kernel_len: tuple) -> tuple:
        # 你的 MATLAB 用 Padding='same'；这里要求 kernel_len 是奇数
        return (kernel_len[0] // 2, kernel_len[1]//2)

    def _direction_map(self, B_skel: torch.Tensor, k1: torch.Tensor, k10: torch.Tensor) -> torch.Tensor:
        """
        复刻 MATLAB 的每个方向逻辑：
          C = conv(B, ones)
          C(C!=2)=0; C(C==2)=1
          D = conv(C, 10s)
          D(D>10)=10; D(D==0)=1
        """
        pad = self._same_pad(k1.shape[-2:])
        C = F.conv2d(B_skel, k1, padding=pad)
        C = torch.where(C == 2.0, torch.ones_like(C), torch.zeros_like(C))
        D = F.conv2d(C, k10, padding=pad)
        D = torch.clamp(D, max=10.0)
        D = torch.where(D == 0.0, torch.ones_like(D), D)
        return D

    def forward(self, logits: torch.Tensor, target: torch.Tensor, ignore_index=None, **kwargs) -> torch.Tensor:
        assert logits.dim() == 4 and logits.size(1) == 2, f"logits must be (B,2,H,W), got {logits.shape}"
        B, _, H, W = logits.shape

        if target.dim() == 4 and target.size(1) == 1:
            target = target[:, 0]
        assert target.shape == (B, H, W), f"target must be (B,H,W), got {target.shape}"
        target = target.long()

        ce = F.cross_entropy(
            logits, target,
            reduction="none",
            ignore_index=(ignore_index if ignore_index is not None else -100)
        )  # (B,H,W)

        # ignore 的像素不应产生加权贡献（保持与 mmseg 一致）
        if ignore_index is not None:
            valid = (target != ignore_index).float()  # (B,H,W)
        else:
            valid = torch.ones_like(ce)

        # ---- 2) Z = fg prob，阈值化得到 A（0/1）
        with torch.no_grad():
            prob = torch.softmax(logits, dim=1)  # (B,2,H,W)
            Z = prob[:, 1]  # fg prob (B,H,W)
            A = (Z >= 0.5).to(torch.uint8)  # (B,H,W)

            # ---- 3) skeletonize（CPU）
            B_skel_cpu = _skeletonize_batch(A)  # (B,H,W) on CPU float32
            B_skel = B_skel_cpu.to(device=logits.device, dtype=logits.dtype)  # back to device
            B_skel = B_skel.unsqueeze(1)  # (B,1,H,W)

        # ---- 4) 四方向卷积得到 D/E/F/G
        Dh = self._direction_map(B_skel, self.k_h1.to(B_skel.dtype), self.k_h10.to(B_skel.dtype))
        Dv = self._direction_map(B_skel, self.k_v1.to(B_skel.dtype), self.k_v10.to(B_skel.dtype))
        Dd = self._direction_map(B_skel, self.k_d1.to(B_skel.dtype), self.k_d10.to(B_skel.dtype))
        Db = self._direction_map(B_skel, self.k_b1.to(B_skel.dtype), self.k_b10.to(B_skel.dtype))

        D = Dh + Dv + Dd + Db
        D = torch.where(D >= 10.0, torch.tensor(10.0, device=D.device, dtype=D.dtype), D)
        D = torch.where(D == 2.0, torch.tensor(1.0, device=D.device, dtype=D.dtype), D)  # MATLAB: D(D==2)=1
        D = D[:, 0]  # (B,H,W)

        # ---- 5) J = D .* L，然后对像素 mean，再对 batch mean
        J = D * ce * valid  # ignore 处为0

        # MATLAB 是 mean(J(:),'all')，但 ignore 需要按有效像素平均更合理
        denom = valid.sum(dim=(1, 2)).clamp_min(1.0)  # (B,)
        loss_per_img = J.sum(dim=(1, 2)) / denom      # (B,)

        return loss_per_img.mean() * self.k
    
    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
    
    
@MODELS.register_module()
class ConnectivityLoss(nn.Module):
    """
    可微连通性损失 (Differentiable Connectivity Loss)
    通过模拟基于 Min-MaxPool 的“火势蔓延”，惩罚拓扑断点和错误粘连。
    """

    def __init__(
        self,
        loss_weight=0.5,
        num_seeds=32,
        num_steps=15,
        downsample_scale=0.25,
        loss_name="connectivity_loss",
    ):
        super(ConnectivityLoss, self).__init__()
        self.loss_weight = loss_weight
        self.num_seeds = num_seeds  # 每次采样的火种数量（Batch中的每张图）
        self.num_steps = num_steps  # 膨胀/蔓延的步数 K
        self.downsample_scale = (
            downsample_scale  # 降采样倍数，防显存爆炸，且能增加单步感受野
        )
        self._loss_name = loss_name

    def forward(self, cls_score, valid_label, weight=None, ignore_index=255, **kwargs):
        """
        Args:
            cls_score (Tensor): 网络的原始输出 Logits，形状 (B, C, H, W)
            valid_label (Tensor): GT 标签，形状 (B, H, W)
        """
        B, C, H, W = cls_score.shape

        # 1. 提取道路预测概率 (B, 1, H, W)
        probs = F.softmax(cls_score, dim=1)
        road_prob = probs[:, 1:2, :, :]

        # 2. 提取道路 GT 掩码 (B, 1, H, W)
        gt_mask = valid_label.float().unsqueeze(1)

        # 3. 降采样 (工程优化：省显存，加速，变相扩大 K 步的物理范围)
        if self.downsample_scale != 1.0:
            road_prob = F.interpolate(
                road_prob,
                scale_factor=self.downsample_scale,
                mode="bilinear",
                align_corners=False,
            )
            gt_mask = F.interpolate(
                gt_mask, scale_factor=self.downsample_scale, mode="nearest"
            )

        B_d, _, H_d, W_d = road_prob.shape

        # 4. 随机播撒火种 (Seed Initialization)
        # 我们构建一个形状为 (B, num_seeds, H_d, W_d) 的状态图
        seeds = torch.zeros((B_d, self.num_seeds, H_d, W_d), device=cls_score.device)

        for b in range(B_d):
            # 找到这张图上真实的道路像素坐标
            road_indices = torch.nonzero(gt_mask[b, 0] >= 0.75, as_tuple=False)
            num_road_pixels = road_indices.shape[0]

            if num_road_pixels > 0:
                # 从道路像素中随机抽取 num_seeds 个点
                # 如果道路像素不够，就允许重复抽样 (replacement=True)
                replace = True if num_road_pixels < self.num_seeds else False
                rand_idx = torch.randint(
                    0, num_road_pixels, (self.num_seeds,), device=cls_score.device
                )
                sampled_coords = road_indices[rand_idx]

                # 点燃这些火种
                for i in range(self.num_seeds):
                    y, x = sampled_coords[i]
                    seeds[b, i, y, x] = 1.0

        # 5. 可微连通性传播 (Differentiable Fire Spread)
        S_pred = seeds.clone()
        S_gt = seeds.clone()

        # 注意：这里需要把 road_prob 和 gt_mask 扩展到和 S 一样的通道数，方便并行 Min 操作
        road_prob_expand = road_prob.expand(-1, self.num_seeds, -1, -1)
        gt_mask_expand = gt_mask.expand(-1, self.num_seeds, -1, -1)

        for _ in range(self.num_steps):
            # 向 8 邻域蔓延 (MaxPool 模拟膨胀)
            S_pred_dilated = F.max_pool2d(S_pred, kernel_size=7, stride=1, padding=3)
            S_gt_dilated = F.max_pool2d(S_gt, kernel_size=7, stride=1, padding=3)

            # 被预测概率/GT概率阻断 (Min 操作防止长距离梯度消失)
            S_pred = torch.min(road_prob_expand, S_pred_dilated)
            S_gt = torch.min(gt_mask_expand, S_gt_dilated)

            # 保证火种源头不灭（可选，但通常加上更稳定）
            S_pred = torch.max(S_pred, seeds)
            S_gt = torch.max(S_gt, seeds)

        # 6. 计算连通性损失
        # S_gt 是标准答案（非 0 即 1），S_pred 是网络的软可达性预测
        # 我们用 BCE Loss 惩罚它们之间的差异
        # detach() 是因为我们不更新 GT 的计算图
        # loss = F.binary_cross_entropy(S_pred, S_gt.detach(), reduction="mean")

        S_pred_f32 = S_pred.float()
        S_gt_f32 = S_gt.detach().float()

        # 2. 截断概率范围，防止出现绝对的 0 或 1 导致 log(0) 报 NaN
        eps = 1e-7
        S_pred_f32 = torch.clamp(S_pred_f32, min=eps, max=1.0 - eps)

        # 3. 手动计算 BCE Loss，绕过 PyTorch 的 autocast 拦截器
        bce_loss = -(S_gt_f32 * torch.log(S_pred_f32) + (1.0 - S_gt_f32) * torch.log(1.0 - S_pred_f32))
        loss = bce_loss.mean()

        return self.loss_weight * loss

    @property
    def loss_name(self):
        return self._loss_name
