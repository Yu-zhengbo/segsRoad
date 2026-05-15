import torch
import pkg_resources
import PIL
import numpy as np
from torch import nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmengine.model import BaseModule
from torchvision.transforms import v2
from iopath.common.file_io import g_pathmgr
from mmseg.models.backbones.tools.ops.modules import MSDeformAttn
from timm.models.layers import DropPath
import torch.utils.checkpoint as cp
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_
import math
from functools import partial
from mmseg.models.backbones.sam3.sam3.model.act_ckpt_utils import activation_ckpt_wrapper

# processor
from mmseg.models.backbones.sam3.sam3.model import box_ops
from mmseg.models.backbones.sam3.sam3.model.data_misc import FindStage, interpolate


# position embeding
from mmseg.models.backbones.sam3.sam3.model.position_encoding import PositionEmbeddingSine

# image encoder
from mmseg.models.backbones.sam3.sam3.model.vitdet import ViT,get_abs_pos
from mmseg.models.backbones.sam3.sam3.model.necks import Sam3DualViTDetNeck

# text encoder
from mmseg.models.backbones.sam3.sam3.model.text_encoder_ve import VETextEncoder
from mmseg.models.backbones.sam3.sam3.model.tokenizer_ve import SimpleTokenizer

# image+text encoder
from mmseg.models.backbones.sam3.sam3.model.vl_combiner import SAM3VLBackbone


#transformer encoder
from mmseg.models.backbones.sam3.sam3.model.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from mmseg.models.backbones.sam3.sam3.model.model_misc import (
    DotProductScoring,
    MLP,
    MultiheadAttentionWrapper as MultiheadAttention,
    TransformerWrapper,
)
# transformer decoder
from mmseg.models.backbones.sam3.sam3.model.decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerDecoderLayerv2,
    TransformerEncoderCrossAttention,
)

# segmentation head 
from mmseg.models.backbones.sam3.sam3.model.maskformer_segmentation import PixelDecoder, UniversalSegmentationHead

# geometry encoder
from mmseg.models.backbones.sam3.sam3.model.geometry_encoders import SequenceGeometryEncoder
from mmseg.models.backbones.sam3.sam3.model.sam3_image import Sam3Image, Sam3ImageOnVideoMultiGPU





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
    spatial_shapes = torch.as_tensor([(h // 7, w // 7),
                                      (h // 14, w // 14),
                                      (h // 28, w // 28)],
                                     dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 14, w // 14)], x.device)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]
    
    spatial_shapes = torch.as_tensor([(h // 14, w // 14)], dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 7, w // 7),
                                                   (h // 14, w // 14),
                                                   (h // 28, w // 28)], x.device)
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]
    
    return deform_inputs1, deform_inputs2


def deform_inputs_only_one(x, h, w):
    # bs, c, h, w = x.shape
    spatial_shapes = torch.as_tensor([(h // 7, w // 7),
                                      (h // 14, w // 14),
                                      (h // 28, w // 28)],
                                     dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 7, w // 7),
                                      (h // 14, w // 14),
                                      (h // 28, w // 28)], device=x.device)
    deform_inputs = [reference_points, spatial_shapes, level_start_index]
    
    return deform_inputs


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
    

class MultiDWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        dim1 = dim
        dim = dim // 2

        self.dwconv1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.dwconv2 = nn.Conv2d(dim, dim, 5, 1, 2, bias=True, groups=dim)

        self.dwconv3 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.dwconv4 = nn.Conv2d(dim, dim, 5, 1, 2, bias=True, groups=dim)

        self.dwconv5 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.dwconv6 = nn.Conv2d(dim, dim, 5, 1, 2, bias=True, groups=dim)

        self.act1 = nn.GELU()
        self.bn1 = nn.BatchNorm2d(dim1)

        self.act2 = nn.GELU()
        self.bn2 = nn.BatchNorm2d(dim1)

        self.act3 = nn.GELU()
        self.bn3 = nn.BatchNorm2d(dim1)

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21
        x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()
        
        x11, x12 = x1[:,:C//2,:,:], x1[:,C//2:,:,:]
        x11 = self.dwconv1(x11)  # BxCxHxW
        x12 = self.dwconv2(x12)
        x1 = torch.cat([x11, x12], dim=1)
        x1 = self.act1(self.bn1(x1)).flatten(2).transpose(1, 2)
        

        x21, x22 = x2[:,:C//2,:,:], x2[:,C//2:,:,:]
        x21 = self.dwconv3(x21)
        x22 = self.dwconv4(x22)
        x2 = torch.cat([x21, x22], dim=1)
        x2 = self.act2(self.bn2(x2)).flatten(2).transpose(1, 2)

        x31, x32 = x3[:,:C//2,:,:], x3[:,C//2:,:,:]
        x31 = self.dwconv5(x31)
        x32 = self.dwconv6(x32)
        x3 = torch.cat([x31, x32], dim=1)
        x3 = self.act3(self.bn3(x3)).flatten(2).transpose(1, 2)

        x = torch.cat([x1, x2, x3], dim=1)
        return x


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


class MRFP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = MultiDWConv(hidden_features)
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


class MultiscaleExtractor(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    
    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        
        def _inner_forward(query, feat):
            
            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            query = query + attn
            
            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))

            return query
        
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)
        
        return query


class CTI_toC(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False,
                 cnn_feature_interaction=True):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        # if with_cffn:
        #     self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
        #     self.ffn_norm = norm_layer(dim)
        #     self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.cnn_feature_interaction = cnn_feature_interaction
        if cnn_feature_interaction:
            self.cfinter = MultiscaleExtractor(dim=dim, n_levels=3, num_heads=num_heads,
                                n_points=n_points, norm_layer=norm_layer, 
                                deform_ratio=deform_ratio, with_cffn=with_cffn,
                                cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, 
                                with_cp=with_cp)
    
    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        
        def _inner_forward(query, feat, H, W):
            B, N, C = query.shape
            n = N // 21
            x1 = query[:, 0:16 * n, :].contiguous()
            x2 = query[:, 16 * n:20 * n, :].contiguous()
            x3 = query[:, 20 * n:, :].contiguous()
            x2 = x2 + feat
            query = torch.cat([x1, x2, x3], dim=1)

            # if self.with_cffn:
            #     query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W)) 

            if self.cnn_feature_interaction:               
                deform_input = deform_inputs_only_one(query, H*14, W*14)
                query = self.cfinter(query=self.query_norm(query), reference_points=deform_input[0],
                          feat=self.feat_norm(query), spatial_shapes=deform_input[1],
                          level_start_index=deform_input[2],
                          H=H, W=W)               
            
            return query
        
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat, H, W)
        else:
            query = _inner_forward(query, feat, H, W)
        
        return query

class Extractor_CTI(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False,
                 cnn_feature_interaction=True):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.cnn_feature_interaction = cnn_feature_interaction
        if cnn_feature_interaction:
            self.cfinter = MultiscaleExtractor(dim=dim, n_levels=3, num_heads=num_heads,
                                n_points=n_points, norm_layer=norm_layer, 
                                deform_ratio=deform_ratio, with_cffn=with_cffn,
                                cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, 
                                with_cp=with_cp)
    
    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        
        def _inner_forward(query, feat, H, W):
            B, N, C = query.shape
            n = N // 21
            x1 = query[:, 0:16 * n, :].contiguous()
            x2 = query[:, 16 * n:20 * n, :].contiguous()
            x3 = query[:, 20 * n:, :].contiguous()
            x2 = x2 + feat
            query = torch.cat([x1, x2, x3], dim=1)

            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W)) 

            if self.cnn_feature_interaction:               
                deform_input = deform_inputs_only_one(query, H*14, W*14)
                query = self.cfinter(query=self.query_norm(query), reference_points=deform_input[0],
                          feat=self.feat_norm(query), spatial_shapes=deform_input[1],
                          level_start_index=deform_input[2],
                          H=H, W=W)               
            
            return query
        
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat, H, W)
        else:
            query = _inner_forward(query, feat, H, W)
        
        return query



class CTI_toV(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False, drop=0., drop_path=0., cffn_ratio=0.25):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

        self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
        self.ffn_norm = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
       
    
    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        
        def _inner_forward(query, feat, H, W):
            B, N, C = feat.shape
            c1 = self.attn(self.query_norm(feat), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)

            c1 = c1 + self.drop_path(self.ffn(self.ffn_norm(c1), H, W)) 

            c_select1, c_select2, c_select3 = c1[:,:H*W*4, :], c1[:, H*W*4:H*W*4+H*W, :], c1[:, H*W*4+H*W:, :]
            c_select1 = F.interpolate(c_select1.permute(0,2,1).reshape(B, C, H*2, W*2), scale_factor=0.5, mode='bilinear', align_corners=False).flatten(2).permute(0,2,1)
            c_select3 = F.interpolate(c_select3.permute(0,2,1).reshape(B, C, H//2, W//2), scale_factor=2, mode='bilinear', align_corners=False).flatten(2).permute(0,2,1)
            # x = x + c_select1 + c_select2 + c_select3

            return query + self.gamma * (c_select1 + c_select2 + c_select3)
        
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat, H, W)
        else:
            query = _inner_forward(query, feat, H, W)
        
        return query


class CTIBlock(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
                 deform_ratio=1.0, extra_CTI=False, with_cp=False, 
                 use_CTI_toV=True, 
                 use_CTI_toC=True,
                 dim_ratio=6.0,
                 cnn_feature_interaction=False,
                 the_last=False):
        super().__init__()
        self.the_last = the_last
        if the_last:
            use_CTI_toC = False
            
        if use_CTI_toV:
            self.cti_tov = CTI_toV(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                 with_cp=with_cp, drop=drop, drop_path=drop_path, cffn_ratio=cffn_ratio)
        if use_CTI_toC:
            self.cti_toc = CTI_toC(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp,
                                   cnn_feature_interaction=cnn_feature_interaction)
        
        if extra_CTI:
            self.extra_CTIs = nn.Sequential(*[
                Extractor_CTI(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp,
                                   cnn_feature_interaction=cnn_feature_interaction)
                for _ in range(4)
            ])

        else:
            self.extra_CTIs = None
        
        self.use_CTI_toV = use_CTI_toV
        self.use_CTI_toC = use_CTI_toC

        self.mrfp = MRFP(dim, hidden_features=int(dim * dim_ratio))
        self.with_cp = with_cp
    
    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W):
        B, N, C = x.shape
        deform_inputs = deform_inputs_only_one(x, H*14, W*14)
        if self.use_CTI_toV:
            c = self.mrfp(c, H, W)
            c_select1, c_select2, c_select3 = c[:,:H*W*4, :], c[:, H*W*4:H*W*4+H*W, :], c[:, H*W*4+H*W:, :]
            c = torch.cat([c_select1, c_select2 + x, c_select3], dim=1)

            x = self.cti_tov(query=x, reference_points=deform_inputs[0],
                          feat=c, spatial_shapes=deform_inputs[1],
                          level_start_index=deform_inputs[2], H=H, W=W)
        x = x.reshape(B, H, W, C)
        for idx, blk in enumerate(blocks):
            if self.training and self.with_cp:
                x = cp.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = x.flatten(1,2)
        if self.use_CTI_toC:
            c = self.cti_toc(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W)
                           
        if self.extra_CTIs is not None:
            for cti in self.extra_CTIs:
                c = cti(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        return x, c


class CNN(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384):
        super().__init__()

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
        B,C,H,W = x.shape
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c1 = self.fc1(c1)
        c2 = self.fc2(c2)
        c3 = self.fc3(c3)
        c4 = self.fc4(c4)

        c2 = F.interpolate(c2,size=(H//7,W//7))
        c3 = F.interpolate(c3,size=(H//14,W//14))
        c4 = F.interpolate(c4,size=(H//28,W//28))
        
        bs, dim, _, _ = c1.shape
        # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
        c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
        c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
        c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

        return c1, c2, c3, c4


def _update_out(out, out_name, out_value, auxiliary=True, update_aux=True):
    out[out_name] = out_value[-1] if auxiliary else out_value
    if auxiliary and update_aux:
        if "aux_outputs" not in out:
            out["aux_outputs"] = [{} for _ in range(len(out_value) - 1)]
        assert len(out["aux_outputs"]) == len(out_value) - 1
        for aux_output, aux_value in zip(out["aux_outputs"], out_value[:-1]):
            aux_output[out_name] = aux_value

def _load_checkpoint(model, checkpoint_path):
    """Load model checkpoint from file."""
    with g_pathmgr.open(checkpoint_path, "rb") as f:
        ckpt = torch.load(f, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    sam3_image_ckpt = {
        k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k
    }
    if model.inst_interactive_predictor is not None:
        sam3_image_ckpt.update(
            {
                k.replace("tracker.", "inst_interactive_predictor.model."): v
                for k, v in ckpt.items()
                if "tracker" in k
            }
        )
        
    drop_keys = [k for k in sam3_image_ckpt.keys() if k.endswith("freqs_cis")]
    for k in drop_keys:
        sam3_image_ckpt.pop(k)

    missing_keys, _ = model.load_state_dict(sam3_image_ckpt, strict=False)
    if len(missing_keys) > 0:
        print(
            f"loaded {checkpoint_path} and found "
            f"missing and/or unexpected keys:\n{missing_keys=}"
        )
    return model

def _create_transformer_encoder() -> TransformerEncoderFusion:
    """Create transformer encoder with its layer."""
    encoder_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
    )

    encoder = TransformerEncoderFusion(
        layer=encoder_layer,
        num_layers=6,
        d_model=256,
        num_feature_levels=1,
        frozen=False,
        use_act_checkpoint=True,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )
    return encoder

def _create_transformer_decoder() -> TransformerDecoder:
    """Create transformer decoder with its layer."""
    decoder_layer = TransformerDecoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
        ),
        n_heads=8,
        use_text_cross_attention=True,
    )

    decoder = TransformerDecoder(
        layer=decoder_layer,
        num_layers=6,
        num_queries=200,
        return_intermediate=True,
        box_refine=True,
        num_o2m_queries=0,
        dac=True,
        boxRPB="log",
        d_model=256,
        frozen=False,
        interaction_layer=None,
        dac_use_selfatt_ln=True,
        resolution=1008,
        stride=14,
        use_act_checkpoint=True,
        presence_token=True,
    )
    return decoder

def _create_geometry_encoder():
    """Create geometry encoder with all its components."""
    # Create position encoding for geometry encoder
    geo_pos_enc = PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=None,
    )
    # Create geometry encoder layer
    geo_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
        pos_enc_at_cross_attn_queries=False,
        pos_enc_at_cross_attn_keys=True,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
    )

    # Create geometry encoder
    input_geometry_encoder = SequenceGeometryEncoder(
        pos_enc=geo_pos_enc,
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=True,
        points_pos_enc=True,
        boxes_direct_project=True,
        boxes_pool=True,
        boxes_pos_enc=True,
        d_model=256,
        num_layers=3,
        layer=geo_layer,
        use_act_ckpt=True,
        add_cls=True,
        add_post_encode_proj=True,
    )
    return input_geometry_encoder

@MODELS.register_module()
class SAM3Myself(BaseModule):
    def __init__(self,img_size=1008,
                 precompute_resolution=1008,
                 compile_mode=None,
                 enable_inst_interactivity=False,
                 eval_mode=True,
                 checkpoint_path='/home/cz/codes/githubs/sam3/checkpoints/sam3.pt',
                 bpe_path='/home/cz/codes/githubs/sam3/checkpoints/bpe_simple_vocab_16e6.txt.gz',
                 image_only=True,
                 mask2former=True,
                 num_class=2,
                 ## paramters of adapter
                 init_values=1e-6,
                 cffn_ratio=0.25,
                 drop_path_rate=0.1,
                 conv_inplane=64,
                 n_points=4,
                 embed_dim=1024,
                 deform_num_heads=8,
                 deform_ratio=0.5,
                 with_cp=True,  # set with_cp=True to save memory
                 interaction_indexes=[[0, 7], [8, 15], [16, 23], [24, 31]],
                 with_cffn=True,
                 add_vit_feature=True,
                 use_extra_CTI=False,
                 use_CTI_toV=True, 
                 use_CTI_toC=True,
                 cnn_feature_interaction=True,
                 dim_ratio=1.0,
                 norm_layer = partial(nn.LayerNorm, eps=1e-6),
                ):
                 
        super().__init__()
        position_encoding = PositionEmbeddingSine(
            num_pos_feats=256,
            normalize=True,
            scale=None,
            temperature=10000,
            precompute_resolution=precompute_resolution,
        )
        vit_backbone = ViT(
            img_size=img_size,
            pretrain_img_size=336,
            patch_size=14,
            embed_dim=1024,
            depth=32,
            num_heads=16,
            mlp_ratio=4.625,
            norm_layer="LayerNorm",
            drop_path_rate=0.1,
            qkv_bias=True,
            use_abs_pos=True,
            tile_abs_pos=True,
            global_att_blocks=(7, 15, 23, 31),
            rel_pos_blocks=(),
            use_rope=True,
            use_interp_rope=True,
            window_size=24,
            pretrain_use_cls_token=True,
            retain_cls_token=False,
            ln_pre=True,
            ln_post=False,
            return_interm_layers=False,
            bias_patch_embed=False,
            compile_mode=compile_mode,
        )
        
        vit_neck = Sam3DualViTDetNeck(
            position_encoding=position_encoding,
            d_model=256,
            scale_factors=[4.0, 2.0, 1.0, 0.5],
            trunk=vit_backbone,
            add_sam2_neck=enable_inst_interactivity,
        )
        
        # bpe_path = pkg_resources.resource_filename(
        #     "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"
        # )
        text_encoder = VETextEncoder(
            tokenizer=SimpleTokenizer(bpe_path=bpe_path),
            d_model=256,
            width=1024,
            heads=16,
            layers=24,
        )
        
        # image + text encoder
        backbone = SAM3VLBackbone(visual=vit_neck, text=text_encoder, scalp=1, act_ckpt_whole_vision_backbone=True, act_ckpt_whole_language_backbone=True)
        
        
        # transformer 
        transformer_encoder = _create_transformer_encoder()
        transformer_decoder = _create_transformer_decoder()
        transformer = TransformerWrapper(encoder=transformer_encoder, decoder=transformer_decoder, d_model=256)
        
        # dot product scoring 
        prompt_mlp = MLP(
            input_dim=256,
            hidden_dim=2048,
            output_dim=256,
            num_layers=2,
            dropout=0.1,
            residual=True,
            out_norm=nn.LayerNorm(256),
        )
        dot_prod_scoring = DotProductScoring(d_model=256, d_proj=256, prompt_mlp=prompt_mlp)
        
        # segmentation head
        pixel_decoder = PixelDecoder(
            num_upsampling_stages=3,
            interpolation_mode="nearest",
            hidden_dim=256,
            compile_mode=compile_mode,
        )
        cross_attend_prompt = MultiheadAttention(
            num_heads=8,
            dropout=0,
            embed_dim=256,
        )
        segmentation_head = UniversalSegmentationHead(
            hidden_dim=256,
            upsampling_stages=3,
            aux_masks=False,
            presence_head=False,
            dot_product_scorer=None,
            act_ckpt=True,
            cross_attend_prompt=cross_attend_prompt,
            pixel_decoder=pixel_decoder,
        )

        # geometry encoder
        input_geometry_encoder = _create_geometry_encoder()
        
        # track module
        inst_predictor = None
        
        common_params = {
            "backbone": backbone,
            "transformer": transformer,
            "input_geometry_encoder": input_geometry_encoder,
            "segmentation_head": segmentation_head,
            "num_feature_levels": 1,
            "o2m_mask_predict": True,
            "dot_prod_scoring": dot_prod_scoring,
            "use_instance_query": False,
            "multimask_output": True,
            "inst_interactive_predictor": inst_predictor,
        }

        matcher = None
        if not eval_mode:
            from mmseg.models.backbones.sam3.sam3.train.matcher import BinaryHungarianMatcherV2

            matcher = BinaryHungarianMatcherV2(
                focal=True,
                cost_class=2.0,
                cost_bbox=5.0,
                cost_giou=2.0,
                alpha=0.25,
                gamma=2,
                stable=False,
            )
        common_params["matcher"] = matcher
        self.model = Sam3Image(**common_params)
        _load_checkpoint(self.model, checkpoint_path)
        
        if eval_mode:
            self.model.eval()
        
        self.image_only = image_only
        self.mask2former = mask2former
        self.init_processor(resolution=img_size)
        if mask2former and not image_only:
            self.to_mask2former = nn.Linear(1, num_class + 1)
        self.freeze_model()
        
        
        ## adapter
        self.interaction_indexes = interaction_indexes
        self.use_CTI_toC = use_CTI_toC
        self.use_CTI_toV = use_CTI_toV
        self.add_vit_feature = add_vit_feature
        self.embed_dim = embed_dim
        
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = CNN(inplanes=conv_inplane, embed_dim=embed_dim)
        self.interactions = nn.Sequential(*[
            CTIBlock(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                            init_values=init_values, drop_path=drop_path_rate,
                            norm_layer=norm_layer, with_cffn=with_cffn,
                            cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                            use_CTI_toV=use_CTI_toV if isinstance(use_CTI_toV, bool) else use_CTI_toV[i],
                            use_CTI_toC=use_CTI_toC if isinstance(use_CTI_toC, bool) else use_CTI_toC[i],
                            dim_ratio=dim_ratio,
                            cnn_feature_interaction=cnn_feature_interaction if isinstance(cnn_feature_interaction, bool) else cnn_feature_interaction[i],
                            extra_CTI=((True if i == len(interaction_indexes) - 1 else False) and use_extra_CTI),
                            the_last=(i==len(interaction_indexes)-1),
                            with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])
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
        
    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

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
    
    def init_weights(self):
        pass
        
    def init_processor(self, resolution=1008,  confidence_threshold=0.5):
        self.resolution = resolution
        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(resolution, resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.confidence_threshold = confidence_threshold
        
    def forward(self,image, text='road'):
        text = [text] * image.shape[0]
        if self.image_only:
            state = self.forward_image(image)
            return state['backbone_out']['backbone_fpn']
        elif self.mask2former:
            return self.forward_image_text(image,text)
        else:
            return self.forward_image_text_without_mask2former(image,text)

    def forward_image(self, x, state=None):
        """Sets the image on which we want to do predictions."""
        state = {}
        state["original_height"] = 560
        state["original_width"] = 560
        
        
        deform_inputs1, deform_inputs2 = deform_inputs(x)
        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        ## VIT
        x = self.model.backbone.vision_backbone.trunk.patch_embed(x)
        bs, h, w, dim = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        s = 0

        if self.model.backbone.vision_backbone.trunk.pos_embed is not None:
            x = x + get_abs_pos(
                self.model.backbone.vision_backbone.trunk.pos_embed,
                self.model.backbone.vision_backbone.trunk.pretrain_use_cls_token,
                (h, w),
                self.model.backbone.vision_backbone.trunk.retain_cls_token,
                tiling=self.model.backbone.vision_backbone.trunk.tile_abs_pos,
            )

        x = self.model.backbone.vision_backbone.trunk.ln_pre(x)
        x = x.flatten(1,2)
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.model.backbone.vision_backbone.trunk.blocks[indexes[0]:indexes[-1] + 1],
                              deform_inputs1, deform_inputs2, h, w)
        x = x.reshape(bs,h,w,dim)
        x = self.model.backbone.vision_backbone.trunk.ln_post(x)
        x = x.permute(0, 3, 1, 2)

        sam3_features, sam3_pos = [], []
        sam2_features, sam2_pos = None, None
        if self.model.backbone.vision_backbone.sam2_convs is not None:
            sam2_features, sam2_pos = [], []
            
        for i in range(len(self.model.backbone.vision_backbone.convs)):
            sam3_x_out = self.model.backbone.vision_backbone.convs[i](x)
            sam3_pos_out = self.model.backbone.vision_backbone.position_encoding(sam3_x_out).to(sam3_x_out.dtype)
            sam3_features.append(sam3_x_out)
            sam3_pos.append(sam3_pos_out)

            if self.model.backbone.vision_backbone.sam2_convs is not None:
                sam2_x_out = self.model.backbone.vision_backbone.sam2_convs[i](x)
                sam2_pos_out = self.model.backbone.vision_backbone.position_encoding(sam2_x_out).to(sam2_x_out.dtype)
                sam2_features.append(sam2_x_out)
                sam2_pos.append(sam2_pos_out)
        
        if self.model.backbone.scalp > 0:
            # Discard the lowest resolution features
            sam3_features, sam3_pos = (
                sam3_features[: -self.model.backbone.scalp],
                sam3_pos[: -self.model.backbone.scalp],
            )
            if sam2_features is not None and sam2_pos is not None:
                sam2_features, sam2_pos = (
                    sam2_features[: -self.model.backbone.scalp],
                    sam2_pos[: -self.model.backbone.scalp],
                )

        sam2_output = None

        if sam2_features is not None and sam2_pos is not None:
            sam2_src = sam2_features[-1]
            sam2_output = {
                "vision_features": sam2_src,
                "vision_pos_enc": sam2_pos,
                "backbone_fpn": sam2_features,
            }

        sam3_src = sam3_features[-1]
        output = {
            "vision_features": sam3_src,
            "vision_pos_enc": sam3_pos,
            "backbone_fpn": sam3_features,
            "sam2_backbone_out": sam2_output,
        }
        
        state["backbone_out"] = output

        return state
    
    def forward_image_text(self,image,text):
        state = self.forward_image(image)
        self.find_stage = FindStage(
            img_ids=torch.arange(image.shape[0]),
            text_ids=torch.arange(image.shape[0]),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        out = self.set_text_prompt(text,state)
        return out
    
    def forward_image_text_without_mask2former(self,image,text):
        state = self.forward_image(image)
        self.find_stage = FindStage(
            img_ids=torch.arange(image.shape[0]),
            text_ids=torch.arange(image.shape[0]),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        out = self.set_text_prompt(text,state,mask2former=False)
        return out
    def set_text_prompt(self, prompt, state, mask2former=True):
        """Sets the text prompt and run the inference"""
        
        text_outputs = self.model.backbone.forward_text(prompt, device=self.model.device)
        # will erase the previous text prompt if any
        state["backbone_out"].update(text_outputs)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt(num_prompts=state['backbone_out']['vision_features'].shape[0])

        outputs = self.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=self.find_stage,
            geometric_prompt=state["geometric_prompt"],
            find_target=None,
            mask2former=mask2former
        )
        if mask2former is False:
            return outputs
        
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = self.to_mask2former(out_logits)
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)
        return [[out_probs], [out_masks], outputs]

    
    
    def forward_grounding(self,
                        backbone_out,
                        find_input,
                        find_target,
                        geometric_prompt,
                        mask2former=True):
        with torch.profiler.record_function("SAM3Image._encode_prompt"):
            prompt, prompt_mask, backbone_out = self._encode_prompt(
                backbone_out, find_input, geometric_prompt
            )
        # Run the encoder
        with torch.profiler.record_function("SAM3Image._run_encoder"):
            backbone_out, encoder_out, _ = self._run_encoder(
                backbone_out, find_input, prompt, prompt_mask
            )
        out = {
            "encoder_hidden_states": encoder_out["encoder_hidden_states"],
            "prev_encoder_out": {
                "encoder_out": encoder_out,
                "backbone_out": backbone_out,
            },
        }

        # Run the decoder
        with torch.profiler.record_function("SAM3Image._run_decoder"):
            out, hs = self._run_decoder(
                memory=out["encoder_hidden_states"],
                pos_embed=encoder_out["pos_embed"],
                src_mask=encoder_out["padding_mask"],
                out=out,
                prompt=prompt,
                prompt_mask=prompt_mask,
                encoder_out=encoder_out,
            )
            
                # Run segmentation heads
        with torch.profiler.record_function("SAM3Image._run_segmentation_heads"):
            backbone_fpn = self._run_segmentation_heads(
                out=out,
                backbone_out=backbone_out,
                img_ids=find_input.img_ids,
                vis_feat_sizes=encoder_out["vis_feat_sizes"],
                encoder_hidden_states=out["encoder_hidden_states"],
                prompt=prompt,
                prompt_mask=prompt_mask,
                hs=hs,
                mask2former=mask2former,
            )
        
        if mask2former is False:
            return backbone_fpn
        
        return out
    
    
    def _encode_prompt(
        self,
        backbone_out,
        find_input,
        geometric_prompt,
        visual_prompt_embed=None,
        visual_prompt_mask=None,
        encode_text=True,
        prev_mask_pred=None,
    ):
        # index text features (note that regardless of early or late fusion, the batch size of
        # `txt_feats` is always the number of *prompts* in the encoder)
        txt_ids = find_input.text_ids
        txt_feats = backbone_out["language_features"][:, txt_ids]
        txt_masks = backbone_out["language_mask"][txt_ids]
        
        feat_tuple = self._get_img_feats(backbone_out, find_input.img_ids)
        backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple

        # Encode geometry 这里point、box会和image feat做cross attention
        geo_feats, geo_masks = self.model.geometry_encoder(
            geo_prompt=geometric_prompt,
            img_feats=img_feats,
            img_sizes=vis_feat_sizes,
            img_pos_embeds=img_pos_embeds,
        )
        
        if visual_prompt_embed is None:
            visual_prompt_embed = torch.zeros(
                (0, *geo_feats.shape[1:]), device=geo_feats.device
            )
            visual_prompt_mask = torch.zeros(
                (*geo_masks.shape[:-1], 0),
                device=geo_masks.device,
                dtype=geo_masks.dtype,
            )
        
        prompt = torch.cat([txt_feats, geo_feats, visual_prompt_embed], dim=0)
        prompt_mask = torch.cat([txt_masks, geo_masks, visual_prompt_mask], dim=1)
        
        return prompt, prompt_mask, backbone_out
    
    
    def _run_encoder(
        self,
        backbone_out,
        find_input,
        prompt,
        prompt_mask,
        encoder_extra_kwargs = None,
    ):
        feat_tuple = self._get_img_feats(backbone_out, find_input.img_ids)
        backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple

        # Run the encoder img_feat先self attention，然后再和prompt做cross attention
        prompt_pos_embed = torch.zeros_like(prompt)
        # make a copy of the image feature lists since the encoder may modify these lists in-place
        memory = self.model.transformer.encoder(
            src=img_feats.copy(),
            src_key_padding_mask=None,
            src_pos=img_pos_embeds.copy(),
            prompt=prompt,
            prompt_pos=prompt_pos_embed,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=vis_feat_sizes,
            encoder_extra_kwargs=encoder_extra_kwargs,
        )
        encoder_out = {
            # encoded image features
            "encoder_hidden_states": memory["memory"],
            "pos_embed": memory["pos_embed"],
            "padding_mask": memory["padding_mask"],
            "level_start_index": memory["level_start_index"],
            "spatial_shapes": memory["spatial_shapes"],
            "valid_ratios": memory["valid_ratios"],
            "vis_feat_sizes": vis_feat_sizes,
            # encoded text features (or other prompts)
            "prompt_before_enc": prompt,
            "prompt_after_enc": memory.get("memory_text", prompt),
            "prompt_mask": prompt_mask,
        }
        return backbone_out, encoder_out, feat_tuple
    
    
    def _run_decoder(
        self,
        pos_embed,
        memory,
        src_mask,
        out,
        prompt,
        prompt_mask,
        encoder_out,
    ):
        bs = memory.shape[1]
        query_embed = self.model.transformer.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)

        apply_dac = self.model.transformer.decoder.dac and self.training
        
        # query_embed(200x256)先算sa，接着分别和prompt_feat和img_feat算ca
        hs, reference_boxes, dec_presence_out, dec_presence_feats = (
            self.model.transformer.decoder(
                tgt=tgt,
                memory=memory,
                memory_key_padding_mask=src_mask,
                pos=pos_embed,
                reference_boxes=None,
                level_start_index=encoder_out["level_start_index"],
                spatial_shapes=encoder_out["spatial_shapes"],
                valid_ratios=encoder_out["valid_ratios"],
                tgt_mask=None,
                memory_text=prompt,
                text_attention_mask=prompt_mask,
                apply_dac=apply_dac,
            )
        )
        hs = hs.transpose(1, 2)  # seq-first to batch-first
        reference_boxes = reference_boxes.transpose(1, 2)  # seq-first to batch-first
        if dec_presence_out is not None:
            # seq-first to batch-first
            dec_presence_out = dec_presence_out.transpose(1, 2)

        out["presence_feats"] = dec_presence_feats
        self.model._update_scores_and_boxes(
            out,
            hs,
            reference_boxes,
            prompt,
            prompt_mask,
            dec_presence_out=dec_presence_out,
        )
        return out, hs
    
    def _run_segmentation_heads(
        self,
        out,
        backbone_out,
        img_ids,
        vis_feat_sizes,
        encoder_hidden_states,
        prompt,
        prompt_mask,
        hs,
        mask2former=True
    ):
        apply_dac = self.model.transformer.decoder.dac and self.training
        if self.model.segmentation_head is not None:
            num_o2o = (hs.size(2) // 2) if apply_dac else hs.size(2)
            num_o2m = hs.size(2) - num_o2o
            obj_queries = hs if self.model.o2m_mask_predict else hs[:, :, :num_o2o]
            
            # encoder_hidden_states即img_feat，替换backbone_fpn中通分辨率的特征，
            # 并和obj_queries类似于mask2former的head做实例分割
            seg_head_outputs = activation_ckpt_wrapper(self.model.segmentation_head)(
                backbone_feats=backbone_out["backbone_fpn"],
                obj_queries=obj_queries,
                image_ids=img_ids,
                encoder_hidden_states=encoder_hidden_states,
                act_ckpt_enable=self.training and self.model.use_act_checkpoint_seg_head,
                prompt=prompt,
                prompt_mask=prompt_mask,
                mask2former=mask2former
            )
            if mask2former is False:
                return seg_head_outputs
            
            aux_masks = False  # self.aux_loss and self.segmentation_head.aux_masks
            for k, v in seg_head_outputs.items():
                if k in self.model.segmentation_head.instance_keys:
                    _update_out(out, k, v[:, :num_o2o], auxiliary=aux_masks)
                    if (
                        self.model.o2m_mask_predict and num_o2m > 0
                    ):  # handle o2m mask prediction
                        _update_out(
                            out, f"{k}_o2m", v[:, num_o2o:], auxiliary=aux_masks
                        )
                else:
                    out[k] = v
        else:
            backbone_out.pop("backbone_fpn", None)

        return None
    def _get_img_feats(self, backbone_out, img_ids):
        """Retrieve correct image features from backbone output."""
        if "backbone_fpn" in backbone_out:
            if "id_mapping" in backbone_out and backbone_out["id_mapping"] is not None:
                img_ids = backbone_out["id_mapping"][img_ids]
                # If this assert fails, it likely means we're requesting different img_ids (perhaps a different frame?)
                # We currently don't expect this to happen. We could technically trigger a recompute here,
                # but likely at the cost of a cpu<->gpu sync point, which would deteriorate perf
                torch._assert_async((img_ids >= 0).all())

            vis_feats = backbone_out["backbone_fpn"][-self.model.num_feature_levels :]
            vis_pos_enc = backbone_out["vision_pos_enc"][-self.model.num_feature_levels :]
            vis_feat_sizes = [x.shape[-2:] for x in vis_pos_enc]  # (H, W) shapes
            # index and flatten visual features NxCxHxW => HWxNxC (batch-first => seq-first)
            img_feats = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_feats]
            img_pos_embeds = [
                x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_pos_enc
            ]
            return backbone_out, img_feats, img_pos_embeds, vis_feat_sizes
        
    def train(self, mode: bool = True):
        # 先调用父类，保证外层模块状态正常
        super().train(mode)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        return self
    
    def freeze_model(self):
        # 1. 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
    
    def init_weights(self):
        pass
    
if __name__ == "__main__":
    device = torch.device('cuda:0')
    model = SAM3Myself(560,560,image_only=True,mask2former=False).to(device)
    model.eval()
    input = torch.randn(3,3,560,560).to(device)
    with torch.no_grad():
        output = model(input,['road','road','street'])
    for i in output:
        if isinstance(i,list):
            for _ in i:
                print(_.shape)
        else:
            print(i.shape)