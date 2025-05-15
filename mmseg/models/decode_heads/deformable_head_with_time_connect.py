import torch.nn as nn
from mmengine.model import BaseModule
import torch
import warnings
from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
# from mmseg.ops import resize
from ..utils import resize

try:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
except ImportError:
    warnings.warn(
        '`MultiScaleDeformableAttention` in MMCV has been moved to '
        '`mmcv.ops.multi_scale_deform_attn`, please update your MMCV')
    from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention
from mmcv.cnn.bricks.transformer import (build_transformer_layer_sequence,
                                         build_positional_encoding)
from torch.nn.init import normal_
from ..utils.se_layer import SELayer
import torch.nn.functional as F

def get_con_1_batch(png: torch.Tensor):
    """
    输入：png [B, 1, H, W]，数值为 0, 1, 255
    输出：dir_array [B, H, W, 9]，表示每个前景像素与其3x3邻域的连接状态
    """
    assert png.dim() == 4 and png.shape[1] == 1, "输入必须是 [B, 1, H, W]"

    B, _, H, W = png.shape
    device = png.device

    # 只保留等于1的像素，其它（包括255）都设为0
    binary_img = (png == 1).to(torch.float32)  # [B, 1, H, W]

    # padding两圈
    padded = F.pad(binary_img, (2, 2, 2, 2), mode='constant', value=0)  # [B, 1, H+4, W+4]

    # 输出初始化
    dir_array = torch.zeros(B, H, W, 9, device=device)

    # pad后偏移位置
    offsets = [
        (0, 0), (0, 2), (0, 4),
        (2, 0), (2, 2), (2, 4),
        (4, 0), (4, 2), (4, 4),
    ]

    for k, (dy, dx) in enumerate(offsets):
        patch = padded[:, :, dy:dy+H, dx:dx+W]  # [B, 1, H, W]
        dir_array[...,k] = (binary_img[:, 0] * patch[:, 0])  # [B, H, W]

    return dir_array.permute(0, 3, 1, 2)  # [B, 9, H, W]



def get_con_3_batch(png: torch.Tensor):
    """
    输入：png [B, 1, H, W]，值为 0, 1, 255
    输出：dir_array [B, H, W, 9]，表示9个方向的连接状态（步长为4）
    """
    assert png.dim() == 4 and png.shape[1] == 1, "输入必须是 [B, 1, H, W]"
    B, _, H, W = png.shape
    device = png.device

    # 只保留值为1的前景像素
    binary_img = (png == 1).to(torch.float32)

    # padding四圈
    padded = F.pad(binary_img, (4, 4, 4, 4), mode='constant', value=0)  # [B, 1, H+8, W+8]

    dir_array = torch.zeros(B, H, W, 9, device=device)

    # 以4为步长提取9个点位
    offsets = [
        (0, 0), (0, 4), (0, 8),
        (4, 0), (4, 4), (4, 8),
        (8, 0), (8, 4), (8, 8),
    ]

    for k, (dy, dx) in enumerate(offsets):
        patch = padded[:, :, dy:dy+H, dx:dx+W]  # [B, 1, H, W]
        dir_array[..., k] = binary_img[:, 0] * patch[:, 0]

    return dir_array.permute(0, 3, 1, 2)  # [B, 9, H, W]


@MODELS.register_module()
class DeformableHeadWithTimeConnect(BaseDecodeHead):
    """Implements the DeformableEncoder.
    Args:
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
    """
    
    def __init__(self,
                 num_feature_levels,
                 encoder,
                 positional_encoding,
                 **kwargs):
        
        super().__init__(input_transform='multiple_select', **kwargs)
    
        self.num_feature_levels = num_feature_levels
        self.encoder = build_transformer_layer_sequence(encoder)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.embed_dims = self.encoder.embed_dims
        
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
                                                 f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
                                                 f' and {num_feats}.'
        
        self.connect_d0 = nn.Sequential(
            nn.Conv2d(kwargs['channels'], 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 9, 3, padding=1, dilation=1),
            SELayer(9, ratio=3),
        )
        self.connect_d1 = nn.Sequential(
            nn.Conv2d(kwargs['channels'], 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 9, 3, padding=3, dilation=3),
            SELayer(9, ratio=3),
        )
        # self.level_embeds = nn.Parameter(
        #     torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.con_loss = nn.BCEWithLogitsLoss()
        self.init_weights()
    
    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        # normal_(self.level_embeds)
    def ConLoss(self,logit, target):
        logit = resize(
            input=logit,
            size=target.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        loss = self.con_loss(logit, target)
        return loss
    @staticmethod
    def get_reference_points(spatial_shapes, device):
        """Get the reference points used in decoder.
        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None]
        return reference_points
    

    def forward(self, inputs, times):
        
        mlvl_feats = inputs[-self.num_feature_levels:]
        
        feat_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            mask = torch.zeros((bs, h, w), device=feat.device, requires_grad=False)
            pos_embed = self.positional_encoding(mask)   #得到位置编码 B,256,128,128
            lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
        feat_flatten = torch.cat(feat_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
       
        memory = self.encoder(
            query=feat_flatten,
            time=times,
            query_pos=lvl_pos_embed_flatten,)

        out = self.conv_seg(memory)

        con0 = self.connect_d0(memory)
        con1 = self.connect_d1(memory)

        return out,con0,con1

    # @auto_fp16()
    # def forward(self, inputs, times):
        
    #     mlvl_feats = inputs[-self.num_feature_levels:]
        
    #     feat_flatten = []
    #     lvl_pos_embed_flatten = []
    #     spatial_shapes = []
    #     for lvl, feat in enumerate(mlvl_feats):
    #         bs, c, h, w = feat.shape
    #         spatial_shape = (h, w)
    #         spatial_shapes.append(spatial_shape)
    #         mask = torch.zeros((bs, h, w), device=feat.device, requires_grad=False)
    #         pos_embed = self.positional_encoding(mask)   #得到位置编码 B,256,128,128
    #         pos_embed = pos_embed.flatten(2).transpose(1, 2)
    #         feat = feat.flatten(2).transpose(1, 2)
    #         lvl_pos_embed = pos_embed
    #         lvl_pos_embed_flatten.append(lvl_pos_embed)
    #         feat_flatten.append(feat)
    #     feat_flatten = torch.cat(feat_flatten, 1)
    #     lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
    #     spatial_shapes = torch.as_tensor(
    #         spatial_shapes, dtype=torch.long, device=feat_flatten.device)
    #     level_start_index = torch.cat((spatial_shapes.new_zeros(
    #         (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        
    #     reference_points = self.get_reference_points(spatial_shapes, device=feat.device)   #得到每一个anchor中心点的坐标  B,128X128,1,2
    #     feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)  128x128,B,256
    #     lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims) 128x128,B,256
    #     memory = self.encoder(
    #         query=feat_flatten,
    #         key=None,
    #         value=None,
    #         time=times,
    #         query_pos=lvl_pos_embed_flatten,
    #         query_key_padding_mask=None,
    #         spatial_shapes=spatial_shapes,
    #         reference_points=reference_points,
    #         level_start_index=level_start_index)
    #     memory = memory.permute(1, 2, 0)
    #     memory = memory.reshape(bs, c, h, w).contiguous()
    #     out = self.conv_seg(memory)

    #     con0 = self.connect_d0(memory)
    #     con1 = self.connect_d1(memory)

    #     return out,con0,con1
    
    def forward_train(self, inputs, times, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits,con0,con1 = self(inputs, times)
        seg_logits = resize(seg_logits,size=gt_semantic_seg.shape[2:])
        losses = self.loss_by_feat(seg_logits, img_metas)


        gt_temp = self._stack_batch_gt(img_metas)
        d1 = get_con_1_batch(gt_temp)
        d3 = get_con_3_batch(gt_temp)
        con_loss0 = self.ConLoss(con0, d1) * 0.4 * 0.4
        con_loss1 = self.ConLoss(con1, d3) * 0.6 * 0.4
        losses['con_loss'] = con_loss0 + con_loss1
        return losses
    
    def forward_train_return_logits(self, inputs, times, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits,con0,con1 = self(inputs, times)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses, seg_logits

    def forward_test(self, inputs, times, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        out,con0,con1 = self(inputs, times)
        return out
        # con0 = torch.sigmoid(con0)
        # con1 = torch.sigmoid(con1)
        # con0_pred = torch.sum(con0, dim=1, keepdim=True)
        # con1_pred = torch.sum(con1, dim=1, keepdim=True)
        # con0_pred[con0_pred<0.9] = 0
        # con0_pred[con0_pred>=0.9] = 1
        # con1_pred[con1_pred<2] = 0
        # con1_pred[con1_pred>=2] = 1

        # out_pred = torch.argmax(out, dim=1)

        # out_pred = out_pred + con0_pred + con1_pred
        # out_pred[out_pred>=1] = 1
        # out_pred[out_pred<1] = 0

        

        # return out_pred.type(torch.LongTensor).to(out.device)[0]