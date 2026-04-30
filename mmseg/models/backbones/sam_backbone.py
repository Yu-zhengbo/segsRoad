import torch
from torch import nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmengine.model import BaseModule
from iopath.common.file_io import g_pathmgr
from mmseg.models.backbones.tools.ops.modules import MSDeformAttn
from timm.models.layers import DropPath
import torch.utils.checkpoint as cp
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_
from functools import partial
import math
# image encoder
from mmseg.models.backbones.sam3.sam3.model.vitdet import ViT,get_abs_pos,PatchEmbed


def _load_checkpoint(model, checkpoint_path):
    """Load model checkpoint from file."""
    with g_pathmgr.open(checkpoint_path, "rb") as f:
        ckpt = torch.load(f, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    sam3_image_ckpt = {
        k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k
    }
    if hasattr(model,'inst_interactive_predictor'):
        sam3_image_ckpt.update(
            {
                k.replace("tracker.", "inst_interactive_predictor.model."): v
                for k, v in ckpt.items()
                if "tracker" in k
            }
        )
        
    sam3_image_ckpt = {k.replace('backbone.vision_backbone.trunk.',''):v for k,v in sam3_image_ckpt.items()}
    drop_keys = [k for k in sam3_image_ckpt.keys() if k.endswith("freqs_cis")]
    for k in drop_keys:
        sam3_image_ckpt.pop(k)

    missing_keys, _ = model.load_state_dict(sam3_image_ckpt, strict=False)
    if len(missing_keys) > 0:
        print(
            f"loaded {checkpoint_path} and found "
            f"missing and/or unexpected keys:\n{missing_keys=}"
        )

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

class Resnet(nn.Module):
    def __init__(self, depth=50, embed_dim=384):
        super().__init__()

        norm_cfg = dict(type='SyncBN', requires_grad=True)
        backbone=dict(
            type='ResNetV1c',
            depth=depth,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            dilations=(1, 1, 2, 4),
            strides=(1, 2, 2, 2),
            norm_cfg=norm_cfg,
            pretrained=f'open-mmlab://resnet{depth}_v1c',
            norm_eval=False,
            style='pytorch',
            contract_dilation=True)
        self.model = MODELS.build(backbone)
        self.model.init_weights()
        if depth == 18:
            inplanes = 64
        elif depth == 50:
            inplanes = 256
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(8 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        B,C,H,W = x.shape
        c1, c2, c3, c4 = self.model(x)
        c1 = self.fc1(c1)
        c2 = self.fc2(c2)
        c3 = self.fc3(c3)
        c4 = self.fc4(c4)
        c1 = F.interpolate(c1, size=(H//7 * 2,W//7 * 2))
        c2 = F.interpolate(c2,size=(H//7,W//7))
        c3 = F.interpolate(c3,size=(H//14,W//14))
        c4 = F.interpolate(c4,size=(H//28,W//28))
        
        bs, dim, _, _ = c1.shape
        # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
        c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
        c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
        c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

        return c1, c2, c3, c4

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
        c1 = F.interpolate(c1, size=(H//7 * 2,W//7 * 2))
        c2 = F.interpolate(c2,size=(H//7,W//7))
        c3 = F.interpolate(c3,size=(H//14,W//14))
        c4 = F.interpolate(c4,size=(H//28,W//28))
        
        bs, dim, _, _ = c1.shape
        # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
        c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
        c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
        c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

        return c1, c2, c3, c4


@MODELS.register_module()
class SAM3Vit(BaseModule):
    def __init__(self,img_size=1008,
                 compile_mode=None,
                 eval_mode=True,
                 checkpoint_path='/home/cz/codes/githubs/sam3/checkpoints/sam3.pt',
                 ):
                 
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.model = ViT(
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
            return_interm_layers=True,
            bias_patch_embed=False,
            compile_mode=compile_mode,
        )
        _load_checkpoint(self.model, checkpoint_path)
        if eval_mode:
            self.model.eval()
        self.freeze_model()
       
    def forward(self,image):
        result = self.model(image)
        return result
    
    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        return self
    
    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
    
    def init_weights(self):
        _load_checkpoint(self.model, self.checkpoint_path)
        pass
    

class Adapter(nn.Module):
    def __init__(self, blk, rank=32) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, rank),
            nn.GELU(),
            nn.Linear(rank, dim),
            nn.GELU()
        )
        self.init_weights()

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net
    
    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.prompt_learn.apply(_init_weights)
        for param in self.block.parameters():
            param.requires_grad = False

@MODELS.register_module()
class SAM3VitUnetAdapter(BaseModule):
    def __init__(self,img_size=1008,
                 compile_mode=None,
                 eval_mode=True,
                 checkpoint_path='/home/cz/codes/githubs/sam3/checkpoints/sam3.pt',
                 use_act_checkpoint=True,
                 interaction_indexes=[7, 15, 23, 31],
                 ):
                 
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.use_act_checkpoint = use_act_checkpoint
        self.interaction_indexes = interaction_indexes
        
        self.model = ViT(
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
            return_interm_layers=True,
            bias_patch_embed=False,
            compile_mode=compile_mode,
        )
        _load_checkpoint(self.model, checkpoint_path)
        if eval_mode:
            self.model.eval()
        self.freeze_model()
       
        blocks = []
        for block in self.model.blocks:
            blocks.append(
                Adapter(block)
            )  
        self.model.blocks = nn.Sequential(
            *blocks
        )
        self.scale = dict(zip(interaction_indexes,[4,2,1,0.5]))
       
    def forward(self,x):
        x = self.model.patch_embed(x)
        bs, h, w, dim = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

        if self.model.pos_embed is not None:
            x = x + get_abs_pos(
                self.model.pos_embed,
                self.model.pretrain_use_cls_token,
                (h, w),
                self.model.retain_cls_token,
                tiling=self.model.tile_abs_pos,
            )
            
        x = self.model.ln_pre(x)

        outputs = []
        for i, blk in enumerate(self.model.blocks):
            if self.use_act_checkpoint and self.training:
                x = cp.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
            if i in self.interaction_indexes:
                x = self.model.ln_post(x)
                feats = x.reshape(bs,h,w,dim)
                feats = feats.permute(0, 3, 1, 2)
                scale = self.scale[i]
                feats = F.interpolate(feats, size=(int(h*scale), int(w*scale)), mode='bilinear')
                outputs.append(feats)
        return outputs
    
    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        return self
    
    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
    
    def init_weights(self):
        _load_checkpoint(self.model, self.checkpoint_path)
        pass


class PromptGenerator(nn.Module):
    def __init__(self, scale_factor, embed_dim, depth, input_type,
                 freq_nums, handcrafted_tune, embedding_tune, patch_size):
        """
        Args:
        """
        super(PromptGenerator, self).__init__()
        self.scale_factor = scale_factor
        self.embed_dim = embed_dim
        self.input_type = input_type
        self.freq_nums = freq_nums
        self.depth = depth
        self.handcrafted_tune = handcrafted_tune
        self.embedding_tune = embedding_tune

        self.shared_mlp = nn.Linear(self.embed_dim//self.scale_factor, self.embed_dim)
        self.embedding_generator = nn.Linear(self.embed_dim, self.embed_dim//self.scale_factor)
        
        for i in range(self.depth):
            lightweight_mlp = nn.Sequential(
                nn.Linear(self.embed_dim//self.scale_factor, self.embed_dim//self.scale_factor),
                nn.GELU()
            )
            setattr(self, 'lightweight_mlp_{}'.format(str(i)), lightweight_mlp)

        self.handcrafted_patchembed =  nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim//self.scale_factor,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=False,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_embeddings(self, x):
        x = x.flatten(1,2)
        return self.embedding_generator(x)

    def init_handcrafted(self, x):
        x = self.fft(x, self.freq_nums)
        return self.handcrafted_patchembed(x)

    def get_prompt(self, handcrafted_feature, embedding_feature):
        N, C, H, W = handcrafted_feature.shape
        handcrafted_feature = handcrafted_feature.view(N, C, H*W).permute(0, 2, 1)
        prompts = []
        for i in range(self.depth):
            lightweight_mlp = getattr(self, 'lightweight_mlp_{}'.format(str(i)))
            # prompt = proj_prompt(prompt)
            prompt = lightweight_mlp(handcrafted_feature + embedding_feature)
            prompts.append(self.shared_mlp(prompt))
        return prompts
    
    def fft(self, x, rate):
        # the smaller rate, the smoother; the larger rate, the darker
        # rate = 4, 8, 16, 32
        mask = torch.zeros(x.shape).to(x.device)
        w, h = x.shape[-2:]
        line = int((w * h * rate) ** .5 // 2)
        mask[:, :, w//2-line:w//2+line, h//2-line:h//2+line] = 1

        fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))
        # mask[fft.float() > self.freq_nums] = 1
        # high pass: 1-mask, low pass: mask
        fft = fft * (1 - mask)
        # fft = fft * mask
        fr = fft.real
        fi = fft.imag

        fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))
        inv = torch.fft.ifft2(fft_hires, norm="forward").real

        inv = torch.abs(inv)

        return inv

    # def forward(self, x):
    #     if self.input_type == 'laplacian':
    #         pyr_A = self.lap_pyramid.pyramid_decom(img=x, num=self.freq_nums)
    #         x = pyr_A[:-1]
    #         laplacian = x[0]
    #         for x_i in x[1:]:
    #             x_i = F.interpolate(x_i, size=(laplacian.size(2), laplacian.size(3)), mode='bilinear', align_corners=True)
    #             laplacian = torch.cat([laplacian, x_i], dim=1)
    #         x = laplacian
    #     elif self.input_type == 'fft':
    #         x = self.fft(x, self.freq_nums)
    #     elif self.input_type == 'all':
    #         x = self.prompt.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)

    #     # get prompting
    #     prompt = self.prompt_generator(x)

    #     if self.mode == 'input':
    #         prompt = self.proj(prompt)
    #         return prompt
    #     elif self.mode == 'stack':
    #         prompts = []
    #         for i in range(self.depth):
    #             proj = getattr(self, 'proj_{}'.format(str(i)))
    #             prompts.append(proj(prompt))
    #         return prompts
    #     elif self.mode == 'hierarchical':
    #         prompts = []
    #         for i in range(self.depth):
    #             proj_prompt = getattr(self, 'proj_prompt_{}'.format(str(i)))
    #             prompt = proj_prompt(prompt)
    #             prompts.append(self.proj_token(prompt))
    #         return prompts




@MODELS.register_module()
class SAM3VitFftAdapter(BaseModule):
    def __init__(self,img_size=1008,
                 compile_mode=None,
                 eval_mode=True,
                 checkpoint_path='/home/cz/codes/githubs/sam3/checkpoints/sam3.pt',
                 use_act_checkpoint=True,
                 interaction_indexes=[7, 15, 23, 31],
                 ):
                 
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.use_act_checkpoint = use_act_checkpoint
        self.interaction_indexes = interaction_indexes
        
        self.model = ViT(
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
            return_interm_layers=True,
            bias_patch_embed=False,
            compile_mode=compile_mode,
        )
        _load_checkpoint(self.model, checkpoint_path)
        if eval_mode:
            self.model.eval()
        self.freeze_model()
        
        self.prompt_generator = PromptGenerator(scale_factor=32, embed_dim=1024,depth=32,
                                                input_type='fft', freq_nums=0.25,
                                                handcrafted_tune=True, embedding_tune=True, patch_size=14)
        self.scale = dict(zip(interaction_indexes,[4,2,1,0.5]))
        
    def forward(self,x):
        inp = x
        x = self.model.patch_embed(x)
        bs, h, w, dim = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

        embedding_feature = self.prompt_generator.init_embeddings(x)
        handcrafted_feature = self.prompt_generator.init_handcrafted(inp)
        prompt = self.prompt_generator.get_prompt(handcrafted_feature, embedding_feature)

        if self.model.pos_embed is not None:
            x = x + get_abs_pos(
                self.model.pos_embed,
                self.model.pretrain_use_cls_token,
                (h, w),
                self.model.retain_cls_token,
                tiling=self.model.tile_abs_pos,
            )
            
        x = self.model.ln_pre(x)

        outputs = []
        for i, blk in enumerate(self.model.blocks):
            x = prompt[i].reshape(bs,h,w,dim) + x
            if self.use_act_checkpoint and self.training:
                x = cp.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
            if i in self.interaction_indexes:
                x = self.model.ln_post(x)
                feats = x.reshape(bs,h,w,dim)
                feats = feats.permute(0, 3, 1, 2)
                scale = self.scale[i]
                feats = F.interpolate(feats, size=(int(h*scale), int(w*scale)), mode='bilinear')
                outputs.append(feats)
        return outputs
    
    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        return self
    
    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
    
    def init_weights(self):
        _load_checkpoint(self.model, self.checkpoint_path)
        pass



@MODELS.register_module()
class SAM3VitComer(BaseModule):
    def __init__(self,img_size=1008,
                 compile_mode=None,
                 eval_mode=True,
                 checkpoint_path='/home/cz/codes/githubs/sam3/checkpoints/sam3.pt',
                 ## paramters of adapter
                 local_model='cnn',
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
        self.checkpoint_path = checkpoint_path
        self.model = ViT(
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
            return_interm_layers=True,
            bias_patch_embed=False,
            compile_mode=compile_mode,
        )
        _load_checkpoint(self.model, checkpoint_path)
        if eval_mode:
            self.model.eval()
        self.freeze_model()
       
        ## adapter
        self.interaction_indexes = interaction_indexes
        self.use_CTI_toC = use_CTI_toC
        self.use_CTI_toV = use_CTI_toV
        self.add_vit_feature = add_vit_feature
        self.embed_dim = embed_dim
        
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        if local_model == 'cnn':
            self.spm = CNN(inplanes=conv_inplane, embed_dim=embed_dim)
        elif local_model == 'res18':
            self.spm = Resnet(depth=18,embed_dim=embed_dim)
        elif local_model == 'res50':
            self.spm = Resnet(depth=50,embed_dim=embed_dim)
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
        # self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        # self.norm1 = nn.SyncBatchNorm(embed_dim)
        # self.norm2 = nn.SyncBatchNorm(embed_dim)
        # self.norm3 = nn.SyncBatchNorm(embed_dim)
        # self.norm4 = nn.SyncBatchNorm(embed_dim)

        # self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)
       
    def forward(self,x):
        deform_inputs1, deform_inputs2 = deform_inputs(x)
        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)
        
        x = self.model.patch_embed(x)
        bs, h, w, dim = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

        if self.model.pos_embed is not None:
            x = x + get_abs_pos(
                self.model.pos_embed,
                self.model.pretrain_use_cls_token,
                (h, w),
                self.model.retain_cls_token,
                tiling=self.model.tile_abs_pos,
            )
            
        x = self.model.ln_pre(x)
        x = x.flatten(1,2)
        outputs = []
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.model.blocks[indexes[0]:indexes[-1] + 1],
                            deform_inputs1, deform_inputs2, h, w)
            x = self.model.ln_post(x)
            feats = x.reshape(bs,h,w,dim)
            feats = feats.permute(0, 3, 1, 2)
            outputs.append(feats)
        return outputs
    
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, h * 2, w * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, h, w).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, h // 2, w // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outputs
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
    
    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        return self
    
    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
    
    def init_weights(self):
        _load_checkpoint(self.model, self.checkpoint_path)
        pass
    
    
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
    def train(self, mode: bool = True):
        # 先调用父类，保证外层模块状态正常
        super().train(mode)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        return self
    
    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4
    
    
if __name__ == "__main__":
    device = torch.device('cuda:0')
    # model = SAM3Vit(560).to(device)
    # model = SAM3VitComer(560).to(device)
    # model = SAM3VitUnetAdapter(560).to(device)
    model = SAM3VitFftAdapter(560).to(device)
    model.eval()
    input = torch.randn(3,3,560,560).to(device)
    with torch.no_grad():
        output = model(input)
    for i in output:
        print(i.shape)