from functools import partial

import torch
from torch import nn
from torch.nn.init import normal_

from mmseg.registry import MODELS
from mmseg.models.backbones.dino import _load_unirefiner_checkpoint
from mmseg.models.backbones.dino_comer import (
    CTIBlock,
    DINOComer,
    deform_inputs,
    deform_inputs_only_one,
)


DINO_REFINED_CHECKPOINT_PATH = (
    '/data/openclaw/UniRefiner/outputs/dinov3/checkpoints/epoch_4.pt'
)


class RegisterCTIBlock(CTIBlock):
    """DINO Comer CTI block that keeps register tokens in the ViT stream."""

    def __init__(self, *args, num_prefix_tokens=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_prefix_tokens = num_prefix_tokens

    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W,
                rot_pos_embed):
        head = x[:, :self.num_prefix_tokens]
        x = x[:, self.num_prefix_tokens:]
        deform_inputs = deform_inputs_only_one(x, H * 16, W * 16)

        if self.use_CTI_toV:
            c = self.mrfp(c, H, W)
            c1 = c[:, :H * W * 4, :]
            c2 = c[:, H * W * 4:H * W * 4 + H * W, :]
            c3 = c[:, H * W * 4 + H * W:, :]
            c = torch.cat([c1, c2 + x, c3], dim=1)

            x = self.cti_tov(
                query=x,
                reference_points=deform_inputs[0],
                feat=c,
                spatial_shapes=deform_inputs[1],
                level_start_index=deform_inputs[2],
                H=H,
                W=W,
            )

        x = torch.cat([head, x], dim=1)
        for block in blocks:
            x = block(x, rope=rot_pos_embed)

        head = x[:, :self.num_prefix_tokens]
        x = x[:, self.num_prefix_tokens:]

        if self.use_CTI_toC:
            c = self.cti_toc(
                query=c,
                reference_points=deform_inputs2[0],
                feat=x,
                spatial_shapes=deform_inputs2[1],
                level_start_index=deform_inputs2[2],
                H=H,
                W=W,
            )

        if self.extra_CTIs is not None:
            for cti in self.extra_CTIs:
                c = cti(
                    query=c,
                    reference_points=deform_inputs2[0],
                    feat=x,
                    spatial_shapes=deform_inputs2[1],
                    level_start_index=deform_inputs2[2],
                    H=H,
                    W=W,
                )

        x = torch.cat([head, x], dim=1)
        return x, c


@MODELS.register_module()
class DINOv3RefinedRegisterComer(DINOComer):
    """DINOv3 Comer backbone with UniRefiner weights and register tokens."""

    def __init__(
        self,
        model='vit_large_patch16_dinov3_qkvb.sat493m',
        freeze=True,
        checkpoint=DINO_REFINED_CHECKPOINT_PATH,
        refined=True,
        checkpoint_lora_alpha=16.0,
        num_register_tokens=4,
        embed_dim=1024,
        init_values=1e-6,
        drop_path_rate=0.3,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        with_cp=True,
        interaction_indexes=((0, 5), (6, 11), (12, 17), (18, 23)),
        with_cffn=True,
        add_vit_feature=True,
        pretrain_size=512,
        use_extra_CTI=True,
        pretrained=None,
        use_CTI_toV=True,
        use_CTI_toC=True,
        cnn_feature_interaction=True,
        dim_ratio=6.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        *args,
        **kwargs,
    ):
        if num_register_tokens <= 0:
            raise ValueError('num_register_tokens must be positive.')

        self.num_register_tokens = num_register_tokens
        super().__init__(
            model=model,
            freeze=freeze,
            embed_dim=embed_dim,
            init_values=init_values,
            drop_path_rate=drop_path_rate,
            conv_inplane=conv_inplane,
            n_points=n_points,
            deform_num_heads=deform_num_heads,
            cffn_ratio=cffn_ratio,
            deform_ratio=deform_ratio,
            with_cp=with_cp,
            interaction_indexes=[list(x) for x in interaction_indexes],
            with_cffn=with_cffn,
            add_vit_feature=add_vit_feature,
            pretrain_size=pretrain_size,
            use_extra_CTI=use_extra_CTI,
            pretrained=pretrained,
            use_CTI_toV=use_CTI_toV,
            use_CTI_toC=use_CTI_toC,
            cnn_feature_interaction=cnn_feature_interaction,
            dim_ratio=dim_ratio,
            norm_layer=norm_layer,
            *args,
            **kwargs,
        )

        if refined and checkpoint is not None:
            _load_unirefiner_checkpoint(
                self.eva,
                checkpoint,
                target_prefix='',
                lora_alpha=checkpoint_lora_alpha,
            )

        self.num_original_prefix_tokens = self.eva.num_prefix_tokens
        self.num_total_prefix_tokens = (
            self.num_original_prefix_tokens + num_register_tokens)

        embed_dim = self.eva.num_features
        self.register_tokens = nn.Parameter(
            torch.zeros(1, num_register_tokens, embed_dim))
        nn.init.trunc_normal_(self.register_tokens, std=0.02)
        self._set_attention_prefix_tokens()

        self.interactions = nn.Sequential(*[
            RegisterCTIBlock(
                dim=embed_dim,
                num_heads=deform_num_heads,
                n_points=n_points,
                init_values=init_values,
                drop_path=drop_path_rate,
                norm_layer=norm_layer,
                with_cffn=with_cffn,
                cffn_ratio=cffn_ratio,
                deform_ratio=deform_ratio,
                use_CTI_toV=use_CTI_toV
                if isinstance(use_CTI_toV, bool) else use_CTI_toV[i],
                use_CTI_toC=use_CTI_toC
                if isinstance(use_CTI_toC, bool) else use_CTI_toC[i],
                dim_ratio=dim_ratio,
                cnn_feature_interaction=cnn_feature_interaction
                if isinstance(cnn_feature_interaction, bool)
                else cnn_feature_interaction[i],
                extra_CTI=(
                    (i == len(interaction_indexes) - 1) and use_extra_CTI),
                num_prefix_tokens=self.num_total_prefix_tokens,
            )
            for i in range(len(interaction_indexes))
        ])
        self.interactions.apply(self._init_weights)
        self.interactions.apply(self._init_deform_weights)
        normal_(self.level_embed)
        self.freeze_model()

    def _set_attention_prefix_tokens(self):
        for block in self.eva.blocks:
            if hasattr(block, 'attn') and hasattr(block.attn,
                                                 'num_prefix_tokens'):
                block.attn.num_prefix_tokens = self.num_total_prefix_tokens

    def _insert_register_tokens(self, x):
        batch_size = x.shape[0]
        registers = self.register_tokens.expand(batch_size, -1, -1)
        prefix = x[:, :self.num_original_prefix_tokens]
        patch_tokens = x[:, self.num_original_prefix_tokens:]
        return torch.cat([prefix, registers, patch_tokens], dim=1)

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        x = self.eva.patch_embed(x)
        H, W = x.shape[1:3]
        bs, dim = x.shape[0], x.shape[-1]

        x, rot_pos_embed = self.eva._pos_embed(x)
        x = self._insert_register_tokens(x)
        x = self.eva.norm_pre(x)

        outs = []
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(
                x,
                c,
                self.eva.blocks[indexes[0]:indexes[-1] + 1],
                deform_inputs1,
                deform_inputs2,
                H,
                W,
                rot_pos_embed,
            )
            patch_tokens = x[:, self.num_total_prefix_tokens:]
            outs.append(
                patch_tokens.transpose(1, 2).view(
                    bs, dim, H, W).contiguous())

        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = torch.nn.functional.interpolate(
                x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = torch.nn.functional.interpolate(
                x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = torch.nn.functional.interpolate(
                x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        return [self.norm1(c1), self.norm2(c2), self.norm3(c3),
                self.norm4(c4)]

    def train(self, mode=True):
        nn.Module.train(self, mode)
        if self.freeze_backbone:
            self.freeze_model()
        return self

    def freeze_model(self):
        if not self.freeze_backbone:
            for param in self.eva.parameters():
                param.requires_grad = True
            if hasattr(self, 'register_tokens'):
                self.register_tokens.requires_grad = True
            return

        for param in self.eva.parameters():
            param.requires_grad = False
        if hasattr(self, 'register_tokens'):
            self.register_tokens.requires_grad = True
        self.eva.eval()

    def init_weights(self):
        pass
