import torch
from torch import nn
import torch.nn.functional as F
from mmseg.models.decode_heads.mkunet_head import mk_irb_bottleneck
from mmseg.registry import MODELS


@MODELS.register_module()
class MKUNet(nn.Module):

    def __init__(self,  pretrained, in_channels=3, channels=[16,32,64,96,160], depths=[1, 1, 1, 1, 1], kernel_sizes=[1,3,5], expansion_factor=2, **kwargs):
        super().__init__()
        
        self.encoder1 = mk_irb_bottleneck(in_channels, channels[0], depths[0], 1, expansion_factor=expansion_factor, dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.encoder2 = mk_irb_bottleneck(channels[0], channels[1], depths[1], 1, expansion_factor=expansion_factor, dw_parallel=True, add=True, kernel_sizes=kernel_sizes)  
        self.encoder3 = mk_irb_bottleneck(channels[1], channels[2], depths[2], 1, expansion_factor=expansion_factor, dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.encoder4 = mk_irb_bottleneck(channels[2], channels[3], depths[3], 1, expansion_factor=expansion_factor, dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.encoder5 = mk_irb_bottleneck(channels[3], channels[4], depths[4], 1, expansion_factor=expansion_factor, dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.load_pretrained(ckpt=pretrained)
        
    def load_pretrained(self, ckpt=None, key="state_dict"):
        if ckpt is None:
            return
        try:
            _ckpt = torch.load(ckpt, map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")

            loaded_state_dict = _ckpt[key]
            new_state_dict = {}
            for k, v in loaded_state_dict.items():
                k = '.'.join(k.split('.')[1:])
                new_state_dict[k] = v
            k1,k2 = self.load_state_dict(new_state_dict, strict=False)
            print('miss keys:', k1)
            print('unexpected keys:', k2)
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

        
    def forward(self, x):

        # if x.shape[1]==1:
        #     x = x.repeat(1, 3, 1, 1)
        
        B = x.shape[0]
        ### Encoder
        ### Stage 1
        out = F.max_pool2d(self.encoder1(x),2,2)
        t1 = out
        ### Stage 2
        out = F.max_pool2d(self.encoder2(out),2,2)
        t2 = out
        ### Stage 3
        out = F.max_pool2d(self.encoder3(out),2,2)
        t3 = out

        ### Stage 4
        out = F.max_pool2d(self.encoder4(out),2,2)
        t4 = out

        ### Bottleneck
        t5 = F.max_pool2d(self.encoder5(out),2,2)


        return [t1, t2, t3, t4, t5]

if __name__ == "__main__":
    input = torch.randn(2,3,224,224)
    model = MKUNet(num_classes=2)
    output = model(input)
    
    for i in output:
        print(i.shape)
