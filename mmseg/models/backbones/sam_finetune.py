import torch
from torch import nn
import argparse
import torch.utils.checkpoint as checkpoint

from mmseg.registry import MODELS
from mmseg.models.backbones.sam_backbone import SAM3Vit
from mmseg.models.backbones.sam3.sam3.model.vitdet import get_abs_pos


class LoRAQKV(nn.Module):
    """LoRA wrapper for SAM3 ViT fused qkv projection.

    The original qkv linear is kept frozen by default. LoRA updates are applied
    to selected q/k/v slices without changing the Attention forward code.
    """

    _TARGET_TO_SLICE = {
        'q': 0,
        'k': 1,
        'v': 2,
    }

    def __init__(
        self,
        qkv: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        targets=('q', 'v'),
    ):
        super().__init__()
        if not isinstance(qkv, nn.Linear):
            raise TypeError('qkv must be an nn.Linear module.')
        if rank <= 0:
            raise ValueError('rank must be positive.')

        self.qkv = qkv
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dim = qkv.in_features
        self.in_features = qkv.in_features
        self.out_features = qkv.out_features

        if self.out_features != self.dim * 3:
            raise ValueError(
                f'qkv.out_features must be 3 * in_features, got '
                f'{self.out_features} and {self.dim}.')

        self.targets = tuple(targets)
        invalid_targets = set(self.targets) - set(self._TARGET_TO_SLICE)
        if invalid_targets:
            raise ValueError(f'Unsupported LoRA targets: {invalid_targets}.')

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_a = nn.ModuleDict()
        self.lora_b = nn.ModuleDict()
        for target in self.targets:
            self.lora_a[target] = nn.Linear(self.dim, rank, bias=False)
            self.lora_b[target] = nn.Linear(rank, self.dim, bias=False)

        self._merged = False
        self._cached_delta = None
        self.reset_parameters()

    def reset_parameters(self):
        for target in self.targets:
            nn.init.kaiming_uniform_(self.lora_a[target].weight, a=5**0.5)
            nn.init.zeros_(self.lora_b[target].weight)

    def forward(self, x):
        if self._merged:
            return self.qkv(x)

        qkv = self.qkv(x)
        deltas = []
        dropped = self.dropout(x)
        for target in self.targets:
            delta = self.lora_b[target](self.lora_a[target](dropped))
            deltas.append((self._TARGET_TO_SLICE[target], delta * self.scaling))

        chunks = list(qkv.split(self.dim, dim=-1))
        for index, delta in deltas:
            chunks[index] = chunks[index] + delta
        return torch.cat(chunks, dim=-1)

    @torch.no_grad()
    def _delta_weight(self, target):
        delta = self.lora_b[target].weight @ self.lora_a[target].weight
        return delta * self.scaling

    @torch.no_grad()
    def fuse_lora_(self):
        if self._merged:
            return self

        cached_delta = []
        for target in self.targets:
            index = self._TARGET_TO_SLICE[target]
            delta = self._delta_weight(target).to(
                device=self.qkv.weight.device,
                dtype=self.qkv.weight.dtype,
            )
            start = index * self.dim
            end = (index + 1) * self.dim
            self.qkv.weight[start:end, :] += delta
            cached_delta.append((start, end, delta))

        self._cached_delta = cached_delta
        self._merged = True
        return self

    @torch.no_grad()
    def unfuse_lora_(self):
        if not self._merged:
            return self
        if self._cached_delta is None:
            raise RuntimeError('No cached LoRA delta. Call fuse_lora_() first.')

        for start, end, delta in self._cached_delta:
            self.qkv.weight[start:end, :] -= delta

        self._cached_delta = None
        self._merged = False
        return self


@MODELS.register_module()
class SAM3VitLoRA(SAM3Vit):
    """SAM3 ViT backbone with LoRA finetuning on attention qkv layers."""

    def __init__(
        self,
        img_size=1008,
        compile_mode=None,
        eval_mode=True,
        checkpoint_path='/home/cz/codes/githubs/sam3/checkpoints/sam3.pt',
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.0,
        lora_targets=('q', 'v'),
        lora_layers=None,
        freeze_base=True,
    ):
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_targets = tuple(lora_targets)
        self.lora_layers = lora_layers
        self.freeze_base = freeze_base
        super().__init__(
            img_size=img_size,
            compile_mode=compile_mode,
            eval_mode=eval_mode,
            checkpoint_path=checkpoint_path,
        )
        self._install_lora()
        self.freeze_model()

    def _resolve_lora_layers(self):
        num_blocks = len(self.model.blocks)
        if self.lora_layers is None:
            return set(range(num_blocks))
        return {
            layer if layer >= 0 else num_blocks + layer
            for layer in self.lora_layers
        }

    def _install_lora(self):
        lora_layers = self._resolve_lora_layers()
        for index, block in enumerate(self.model.blocks):
            if index not in lora_layers:
                continue
            block.attn.qkv = LoRAQKV(
                block.attn.qkv,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
                targets=self.lora_targets,
            )

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        if self.freeze_base:
            self.freeze_model()
            for module in self.modules():
                if isinstance(module, LoRAQKV):
                    module.train(mode)
        return self

    def freeze_model(self):
        if not self.freeze_base:
            for param in self.model.parameters():
                param.requires_grad = True
            return

        for name, param in self.model.named_parameters():
            param.requires_grad = 'lora_' in name
        self.model.eval()

    def init_weights(self):
        # The base class has already loaded checkpoint weights before LoRA is
        # attached. Reloading after wrapping qkv changes checkpoint key names.
        pass

    def fuse_lora_(self):
        for module in self.modules():
            if isinstance(module, LoRAQKV):
                module.fuse_lora_()
        return self

    def unfuse_lora_(self):
        for module in self.modules():
            if isinstance(module, LoRAQKV):
                module.unfuse_lora_()
        return self


@MODELS.register_module()
class SAM3Register(SAM3Vit):
    """SAM3 ViT backbone with learnable register tokens.

    The register tokens are extra latent tokens used as scratch space for
    global attention blocks. They are removed before returning dense feature
    maps, so downstream decode heads still receive only image patch features.
    """

    def __init__(
        self,
        img_size=1008,
        compile_mode=None,
        eval_mode=True,
        checkpoint_path='/home/cz/codes/githubs/sam3/checkpoints/sam3.pt',
        num_register_tokens=4,
        freeze_base=True,
    ):
        if num_register_tokens <= 0:
            raise ValueError('num_register_tokens must be positive.')

        self.num_register_tokens = num_register_tokens
        self.freeze_base = freeze_base
        super().__init__(
            img_size=img_size,
            compile_mode=compile_mode,
            eval_mode=eval_mode,
            checkpoint_path=checkpoint_path,
        )

        embed_dim = self.model.patch_embed.proj.out_channels
        self.register_tokens = nn.Parameter(
            torch.zeros(1, num_register_tokens, embed_dim))
        nn.init.trunc_normal_(self.register_tokens, std=0.02)
        self.freeze_model()

    def _forward_global_block_with_registers(self, block, x, registers):
        bs, h, w, dim = x.shape
        num_patch_tokens = h * w
        patch_tokens = x.reshape(bs, num_patch_tokens, dim)
        shortcut = torch.cat([patch_tokens, registers], dim=1)

        tokens = block.norm1(shortcut)
        attn = block.attn
        if attn.use_rel_pos:
            raise NotImplementedError(
                'SAM3Register does not support relative position attention '
                'with register tokens.')

        qkv = attn.qkv(tokens).reshape(
            bs, num_patch_tokens + self.num_register_tokens, 3,
            attn.num_heads, -1)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        q_patch, k_patch = attn._apply_rope(
            q[:, :, :num_patch_tokens],
            k[:, :, :num_patch_tokens],
        )
        q = torch.cat([q_patch, q[:, :, num_patch_tokens:]], dim=2)
        k = torch.cat([k_patch, k[:, :, num_patch_tokens:]], dim=2)

        tokens = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        tokens = tokens.view(
            bs, attn.num_heads, num_patch_tokens + self.num_register_tokens,
            -1)
        tokens = tokens.permute(0, 2, 1, 3).reshape(
            bs, num_patch_tokens + self.num_register_tokens, dim)
        tokens = attn.proj(tokens)

        tokens = shortcut + block.dropout(block.drop_path(block.ls1(tokens)))
        tokens = tokens + block.dropout(
            block.drop_path(block.ls2(block.mlp(block.norm2(tokens)))))

        x = tokens[:, :num_patch_tokens].reshape(bs, h, w, dim)
        registers = tokens[:, num_patch_tokens:]
        return x, registers

    def _forward_registers_mlp(self, block, registers):
        return registers + block.dropout(
            block.drop_path(block.ls2(block.mlp(block.norm2(registers)))))

    def forward(self, image):
        x = self.model.patch_embed(image)
        bs, h, w, _ = x.shape

        if self.model.pos_embed is not None:
            x = x + get_abs_pos(
                self.model.pos_embed,
                self.model.pretrain_use_cls_token,
                (h, w),
                self.model.retain_cls_token,
                tiling=self.model.tile_abs_pos,
            )

        x = self.model.ln_pre(x)
        registers = self.register_tokens.expand(bs, -1, -1)

        outputs = []
        for i, block in enumerate(self.model.blocks):
            if block.window_size > 0:
                if self.model.use_act_checkpoint and self.model.training:
                    x = checkpoint.checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)
                registers = self._forward_registers_mlp(block, registers)
            else:
                if self.model.use_act_checkpoint and self.model.training:
                    x, registers = checkpoint.checkpoint(
                        lambda x_, registers_: (
                            self._forward_global_block_with_registers(
                                block, x_, registers_)
                        ),
                        x,
                        registers,
                        use_reentrant=False,
                    )
                else:
                    x, registers = self._forward_global_block_with_registers(
                        block, x, registers)
                    # print(1)

            if (i == self.model.full_attn_ids[-1]) or (
                self.model.return_interm_layers
                and i in self.model.full_attn_ids
            ):
                feats = self.model.ln_post(
                    x) if i == self.model.full_attn_ids[-1] else x
                feats = feats.permute(0, 3, 1, 2)
                outputs.append(feats)

        return outputs

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        if self.freeze_base:
            self.freeze_model()
        return self

    def freeze_model(self):
        if not self.freeze_base:
            for param in self.model.parameters():
                param.requires_grad = True
            if hasattr(self, 'register_tokens'):
                self.register_tokens.requires_grad = True
            return

        for param in self.model.parameters():
            param.requires_grad = False
        if hasattr(self, 'register_tokens'):
            self.register_tokens.requires_grad = True
        self.model.eval()

    def init_weights(self):
        # The base class has already loaded checkpoint weights before register
        # tokens are attached. Avoid reloading with the extra parameter present.
        pass


def _linear_flops(batch_tokens, in_features, out_features):
    return 2 * batch_tokens * in_features * out_features


def _estimate_lora_qkv_flops(lora_qkv, input_shape):
    batch_tokens = 1
    for size in input_shape[:-1]:
        batch_tokens *= size

    base_flops = _linear_flops(
        batch_tokens,
        lora_qkv.in_features,
        lora_qkv.out_features,
    )
    per_target_flops = (
        _linear_flops(batch_tokens, lora_qkv.dim, lora_qkv.rank)
        + _linear_flops(batch_tokens, lora_qkv.rank, lora_qkv.dim)
    )
    lora_flops = per_target_flops * len(lora_qkv.targets)
    return base_flops, lora_flops


def _format_flops(flops):
    units = [('T', 10**12), ('G', 10**9), ('M', 10**6), ('K', 10**3)]
    for suffix, scale in units:
        if flops >= scale:
            return f'{flops / scale:.4f} {suffix}FLOPs'
    return f'{flops} FLOPs'


def _randomize_lora(module, std=0.02):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if 'lora_' in name:
                param.normal_(mean=0.0, std=std)


def main():
    parser = argparse.ArgumentParser(
        description='Test LoRA qkv output diff before/after fuse and estimate FLOPs.'
    )
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-tokens', type=int, default=64)
    parser.add_argument('--dim', type=int, default=1024)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=16.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--targets', nargs='+', default=['q', 'v'])
    parser.add_argument(
        '--num-lora-layers',
        type=int,
        default=1,
        help='Number of qkv layers with LoRA. Used only for total FLOPs report.',
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument(
        '--randomize-lora',
        action='store_true',
        help='Fill LoRA weights with non-zero random values before testing.',
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    qkv = nn.Linear(args.dim, args.dim * 3, bias=True).to(device)
    lora_qkv = LoRAQKV(
        qkv,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        targets=tuple(args.targets),
    ).to(device)
    lora_qkv.eval()
    if args.randomize_lora:
        _randomize_lora(lora_qkv)

    x = torch.randn(args.batch_size, args.num_tokens, args.dim, device=device)
    with torch.no_grad():
        y_before = lora_qkv(x)
        base_flops, lora_flops = _estimate_lora_qkv_flops(lora_qkv, tuple(x.shape))
        lora_qkv.fuse_lora_()
        y_after = lora_qkv(x)

    diff = (y_before - y_after).abs()
    print(f'input shape: {tuple(x.shape)}')
    print(f'targets: {tuple(args.targets)}, rank: {args.rank}, alpha: {args.alpha}')
    print(f'max abs diff after fuse: {diff.max().item():.8e}')
    print(f'mean abs diff after fuse: {diff.mean().item():.8e}')
    print(f'per qkv before fuse FLOPs: {_format_flops(base_flops + lora_flops)}')
    print(f'  base qkv FLOPs: {_format_flops(base_flops)}')
    print(f'  LoRA extra FLOPs: {_format_flops(lora_flops)}')
    print(f'per qkv after fuse FLOPs: {_format_flops(base_flops)}')
    print(f'per qkv saved FLOPs by fuse: {_format_flops(lora_flops)}')
    if args.num_lora_layers > 1:
        print(f'total LoRA qkv layers: {args.num_lora_layers}')
        print(
            'total before fuse FLOPs: '
            f'{_format_flops((base_flops + lora_flops) * args.num_lora_layers)}'
        )
        print(
            'total after fuse FLOPs: '
            f'{_format_flops(base_flops * args.num_lora_layers)}'
        )
        print(
            'total saved FLOPs by fuse: '
            f'{_format_flops(lora_flops * args.num_lora_layers)}'
        )


if __name__ == '__main__':
    # main()
    # backbone=SAM3VitLoRA(
    #     img_size=1008,
    #     lora_rank=8,
    #     lora_alpha=16,
    #     lora_dropout=0.0,
    #     lora_targets=('q', 'v'),
    # )
    backbone = SAM3Register(img_size=560)
    input = torch.randn(1,3,560,560)
    output = backbone(input)
    for i in output:
        print(i.shape)
