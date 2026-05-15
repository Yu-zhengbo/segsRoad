# Copyright (c) OpenMMLab. All rights reserved.
from .beit import BEiT
from .bisenetv1 import BiSeNetV1
from .bisenetv2 import BiSeNetV2
from .cgnet import CGNet
from .ddrnet import DDRNet
from .erfnet import ERFNet
from .fast_scnn import FastSCNN
from .hrnet import HRNet
from .icnet import ICNet
from .mae import MAE
from .mit import MixVisionTransformer
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v3 import MobileNetV3
from .mscan import MSCAN
from .pidnet import PIDNet
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1c, ResNetV1d
from .resnext import ResNeXt
from .stdc import STDCContextPathNet, STDCNet
from .swin import SwinTransformer
from .timm_backbone import TIMMBackbone
from .twins import PCPVT, SVT
from .unet import UNet
from .vit import VisionTransformer
from .vpd import VPD
from .dswin import DSwinTransformer
from .vim import VisionMambaSeg
from .vmamba import Backbone_VSSM
from .plainmamba import PlainMambaSeg
from .localvmamba import Backbone_LocalVSSM
from .damamba import DAMamba_tiny, DAMamba_small, DAMamba_base
from .dmambaforroad import DAMambaForRoad
from .deformable_vmamba import DeformableVMAMBA
from .deformable_vmamba_unfold import DeformableAggregateVMAMBA
from .spatialmamba import Backbone_SpatialMamba
from .spiralmamba import DualMamba_base, DualMamba_small, DualMamba_tiny
from .diagmamba import DualDiagMamba_base
from .cross_damamba import CrossDAMamba_tiny, CrossDAMamba_small, CrossDAMamba_base
from .dcan import DCAN_Tiny, DCAN_Small, DCAN_Base
from .mkunet import MKUNet
from .dino import DinoV3Vit
from .dino_adapter import DINOAdapter
from .dino_comer import DINOComer
from .dino_myself import DINOAdapterMyself
from .dino_myself_v2 import DINOAdapterMyselfv2
from .sam import SAM3
from .sam_myself import SAM3Myself
from .sam_backbone import SAM3Vit, SAM3VitComer
# , DINOComer
__all__ = [
    'ResNet', 'ResNetV1c', 'ResNetV1d', 'ResNeXt', 'HRNet', 'FastSCNN',
    'ResNeSt', 'MobileNetV2', 'UNet', 'CGNet', 'MobileNetV3',
    'VisionTransformer', 'SwinTransformer', 'MixVisionTransformer',
    'BiSeNetV1', 'BiSeNetV2', 'ICNet', 'TIMMBackbone', 'ERFNet', 'PCPVT',
    'SVT', 'STDCNet', 'STDCContextPathNet', 'BEiT', 'MAE', 'PIDNet', 'MSCAN',
    'DDRNet', 'VPD', 'DSwinTransformer','VisionMambaSeg','Backbone_VSSM','PlainMambaSeg',
    'Backbone_LocalVSSM','DAMamba_tiny','DAMamba_small','DAMamba_base','DAMambaForRoad',
    'DeformableVMAMBA', 'DeformableAggregateVMAMBA', 'Backbone_SpatialMamba',
    'DualMamba_base', 'DualMamba_small', 'DualMamba_tiny', 'DualDiagMamba_base',
    'CrossDAMamba_tiny', 'CrossDAMamba_small', 'CrossDAMamba_base',
    'DCAN_Tiny', 'DCAN_Small', 'DCAN_Base', 'MKUNet','DinoV3Vit', 'DINOAdapter',
    'DINOComer','DINOAdapterMyself','DINOAdapterMyselfv2', 'SAM3', 'SAM3Myself',
    'SAM3Vit','SAM3VitComer'
]
