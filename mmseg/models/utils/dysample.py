import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

@MODELS.register_module()
class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        self.scale = scale
        self.style = style
        self.groups = groups

        conv_channels = in_channels // scale ** 2 if style == 'pl' else in_channels
        out_channels = 2 * groups * (1 if style == 'pl' else scale ** 2)

        self.offset = nn.Conv2d(conv_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(conv_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        pos = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        pos = torch.stack(torch.meshgrid([pos, pos], indexing='ij')).transpose(1, 2)
        return pos.repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        b, _, h, w = offset.shape
        offset = offset.reshape(b, 2, -1, h, w)

        coords_h = torch.arange(h, dtype=x.dtype, device=x.device) + 0.5
        coords_w = torch.arange(w, dtype=x.dtype, device=x.device) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h], indexing='ij')).transpose(1, 2)
        coords = coords.unsqueeze(1).unsqueeze(0)

        normalizer = torch.tensor([w, h], dtype=x.dtype, device=x.device)
        normalizer = normalizer.reshape(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1

        coords = F.pixel_shuffle(coords.reshape(b, -1, h, w), self.scale)
        coords = coords.reshape(b, 2, -1, self.scale * h, self.scale * w)
        coords = coords.permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)

        x = x.reshape(b * self.groups, -1, h, w)
        output = F.grid_sample(
            x, coords, mode='bilinear', align_corners=False, padding_mode='border')
        return output.reshape(b, -1, self.scale * h, self.scale * w)

    def _learned_offset(self, x):
        offset = self.offset(x)
        if hasattr(self, 'scope'):
            return offset * self.scope(x).sigmoid() * 0.5
        return offset * 0.25

    def forward_lp(self, x):
        return self.sample(x, self._learned_offset(x) + self.init_pos)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        offset = F.pixel_unshuffle(self._learned_offset(x_), self.scale) + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)


if __name__ == '__main__':
    x = torch.rand(2, 1024, 40, 40)
    dys = DySample(1024, scale=2, style='lp', groups=4, dyscope=False)
    print(dys(x).shape)
