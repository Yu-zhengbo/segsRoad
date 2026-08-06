from functools import partial

import torch
from torch import nn
import torch.utils.checkpoint as cp
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmseg.models.backbones.sam3.sam3.model.vitdet import (
    get_abs_pos,
    window_partition,
    window_unpartition,
)
from mmseg.models.backbones.sam_backbone import (
    REFINED_SAM3_CHECKPOINT_PATH,
    SAM3VitComer,
    deform_inputs,
    deform_inputs_only_one,
)


@MODELS.register_module()
class SAM3RefinedWindowRegisterComer(SAM3VitComer):
    """SAM3 Comer backbone with refined weights and window/register tokens.

    The SAM ViT trunk is loaded through ``SAM3VitComer`` so the Comer modules
    stay identical to the existing implementation. The interaction loop is
    overridden because register tokens must be injected while each SAM block is
    executed inside a CTI stage.
    """

    def __init__(
        self,
        img_size=1008,
        compile_mode=None,
        eval_mode=True,
        checkpoint_path='/home/cz/codes/githubs/sam3/checkpoints/sam3.pt',
        refined=True,
        refined_weight=REFINED_SAM3_CHECKPOINT_PATH,
        num_register_tokens=4,
        num_local_register_tokens=4,
        freeze_base=True,
        local_model='cnn',
        init_values=1e-6,
        cffn_ratio=0.25,
        drop_path_rate=0.1,
        conv_inplane=64,
        n_points=4,
        embed_dim=1024,
        deform_num_heads=8,
        deform_ratio=0.5,
        with_cp=True,
        interaction_indexes=((0, 7), (8, 15), (16, 23), (24, 31)),
        with_cffn=True,
        add_vit_feature=True,
        use_extra_CTI=False,
        use_CTI_toV=True,
        use_CTI_toC=True,
        cnn_feature_interaction=True,
        dim_ratio=1.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        if num_register_tokens <= 0:
            raise ValueError('num_register_tokens must be positive.')
        if num_local_register_tokens <= 0:
            raise ValueError('num_local_register_tokens must be positive.')

        self.num_register_tokens = num_register_tokens
        self.num_local_register_tokens = num_local_register_tokens
        self.freeze_base = freeze_base

        if refined:
            checkpoint_path = refined_weight

        super().__init__(
            img_size=img_size,
            compile_mode=compile_mode,
            eval_mode=eval_mode,
            checkpoint_path=checkpoint_path,
            local_model=local_model,
            init_values=init_values,
            cffn_ratio=cffn_ratio,
            drop_path_rate=drop_path_rate,
            conv_inplane=conv_inplane,
            n_points=n_points,
            embed_dim=embed_dim,
            deform_num_heads=deform_num_heads,
            deform_ratio=deform_ratio,
            with_cp=with_cp,
            interaction_indexes=[list(x) for x in interaction_indexes],
            with_cffn=with_cffn,
            add_vit_feature=add_vit_feature,
            use_extra_CTI=use_extra_CTI,
            use_CTI_toV=use_CTI_toV,
            use_CTI_toC=use_CTI_toC,
            cnn_feature_interaction=cnn_feature_interaction,
            dim_ratio=dim_ratio,
            norm_layer=norm_layer,
        )

        embed_dim = self.model.patch_embed.proj.out_channels
        self.register_tokens = nn.Parameter(
            torch.zeros(1, num_register_tokens, embed_dim))
        self.window_block_indices = tuple(
            i for i, block in enumerate(self.model.blocks)
            if block.window_size > 0)
        self.window_block_index_map = {
            block_index: local_index
            for local_index, block_index in enumerate(self.window_block_indices)
        }
        self.local_register_tokens = nn.Parameter(
            torch.zeros(
                len(self.window_block_indices), 1, num_local_register_tokens,
                embed_dim))
        
        
        # self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        # self.norm1 = nn.SyncBatchNorm(embed_dim)
        # self.norm2 = nn.SyncBatchNorm(embed_dim)
        # self.norm3 = nn.SyncBatchNorm(embed_dim)
        # self.norm4 = nn.SyncBatchNorm(embed_dim)
        # self.up.apply(self._init_weights)
        
        nn.init.trunc_normal_(self.register_tokens, std=0.02)
        nn.init.trunc_normal_(self.local_register_tokens, std=0.02)
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
                'Register tokens are not supported with relative position '
                'attention in SAM3 global blocks.')

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
            bs, attn.num_heads,
            num_patch_tokens + self.num_register_tokens, -1)
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

    def _forward_window_block_with_registers(self, block, block_index, x):
        shortcut = x
        x = block.norm1(x)
        h, w = x.shape[1], x.shape[2]
        window_size = block.window_size
        windows, pad_hw = window_partition(x, window_size)

        num_windows, _, _, dim = windows.shape
        num_patch_tokens = window_size * window_size
        patch_tokens = windows.reshape(num_windows, num_patch_tokens, dim)
        local_index = self.window_block_index_map[block_index]
        local_registers = self.local_register_tokens[local_index].expand(
            num_windows, -1, -1)
        tokens = torch.cat([patch_tokens, local_registers], dim=1)

        attn = block.attn
        if attn.use_rel_pos:
            raise NotImplementedError(
                'Local register tokens are not supported with relative '
                'position attention in SAM3 window blocks.')

        qkv = attn.qkv(tokens).reshape(
            num_windows,
            num_patch_tokens + self.num_local_register_tokens,
            3,
            attn.num_heads,
            -1,
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        q_patch, k_patch = attn._apply_rope(
            q[:, :, :num_patch_tokens],
            k[:, :, :num_patch_tokens],
        )
        q = torch.cat([q_patch, q[:, :, num_patch_tokens:]], dim=2)
        k = torch.cat([k_patch, k[:, :, num_patch_tokens:]], dim=2)

        tokens = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        tokens = tokens.view(
            num_windows,
            attn.num_heads,
            num_patch_tokens + self.num_local_register_tokens,
            -1,
        )
        tokens = tokens.permute(0, 2, 1, 3).reshape(
            num_windows,
            num_patch_tokens + self.num_local_register_tokens,
            dim,
        )
        tokens = attn.proj(tokens)

        patch_tokens = tokens[:, :num_patch_tokens]
        windows = patch_tokens.reshape(
            num_windows, window_size, window_size, dim)
        x = window_unpartition(windows, window_size, pad_hw, (h, w))

        x = shortcut + block.dropout(block.drop_path(block.ls1(x)))
        x = x + block.dropout(
            block.drop_path(block.ls2(block.mlp(block.norm2(x)))))
        return x

    def _forward_block_with_registers(self, block, block_index, x, registers):
        if block.window_size > 0:
            if self.model.use_act_checkpoint and self.model.training:
                x = cp.checkpoint(
                    lambda x_: self._forward_window_block_with_registers(
                        block, block_index, x_),
                    x,
                    use_reentrant=False,
                )
            else:
                x = self._forward_window_block_with_registers(
                    block, block_index, x)
            registers = self._forward_registers_mlp(block, registers)
            return x, registers

        if self.model.use_act_checkpoint and self.model.training:
            return cp.checkpoint(
                lambda x_, registers_: (
                    self._forward_global_block_with_registers(
                        block, x_, registers_)
                ),
                x,
                registers,
                use_reentrant=False,
            )
        return self._forward_global_block_with_registers(block, x, registers)

    def _forward_interaction_with_registers(
        self,
        layer,
        x,
        c,
        block_indexes,
        deform_inputs1,
        deform_inputs2,
        h,
        w,
        registers,
    ):
        bs, _, dim = x.shape
        deform_inputs = deform_inputs_only_one(x, h * 14, w * 14)

        if layer.use_CTI_toV:
            c = layer.mrfp(c, h, w)
            c1 = c[:, :h * w * 4, :]
            c2 = c[:, h * w * 4:h * w * 4 + h * w, :]
            c3 = c[:, h * w * 4 + h * w:, :]
            c = torch.cat([c1, c2 + x, c3], dim=1)

            x = layer.cti_tov(
                query=x,
                reference_points=deform_inputs[0],
                feat=c,
                spatial_shapes=deform_inputs[1],
                level_start_index=deform_inputs[2],
                H=h,
                W=w,
            )

        x = x.reshape(bs, h, w, dim)
        for block_index in block_indexes:
            block = self.model.blocks[block_index]
            x, registers = self._forward_block_with_registers(
                block, block_index, x, registers)
        x = x.flatten(1, 2)

        if layer.use_CTI_toC:
            c = layer.cti_toc(
                query=c,
                reference_points=deform_inputs2[0],
                feat=x,
                spatial_shapes=deform_inputs2[1],
                level_start_index=deform_inputs2[2],
                H=h,
                W=w,
            )

        if layer.extra_CTIs is not None:
            for cti in layer.extra_CTIs:
                c = cti(
                    query=c,
                    reference_points=deform_inputs2[0],
                    feat=x,
                    spatial_shapes=deform_inputs2[1],
                    level_start_index=deform_inputs2[2],
                    H=h,
                    W=w,
                )

        return x, c, registers

    def forward(self, image):
        deform_inputs1, deform_inputs2 = deform_inputs(image)

        c1, c2, c3, c4 = self.spm(image)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        x = self.model.patch_embed(image)
        bs, h, w, dim = x.shape

        if self.model.pos_embed is not None:
            x = x + get_abs_pos(
                self.model.pos_embed,
                self.model.pretrain_use_cls_token,
                (h, w),
                self.model.retain_cls_token,
                tiling=self.model.tile_abs_pos,
            )

        x = self.model.ln_pre(x)
        x = x.flatten(1, 2)
        registers = self.register_tokens.expand(bs, -1, -1)

        outputs = []
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            block_indexes = range(indexes[0], indexes[-1] + 1)
            x, c, registers = self._forward_interaction_with_registers(
                layer,
                x,
                c,
                block_indexes,
                deform_inputs1,
                deform_inputs2,
                h,
                w,
                registers,
            )
            feats = self.model.ln_post(x).reshape(bs, h, w, dim)
            outputs.append(feats.permute(0, 3, 1, 2))

        return outputs
        # c2 = c[:, 0:c2.size(1), :]
        # c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        # c4 = c[:, c2.size(1) + c3.size(1):, :]

        # c2 = c2.transpose(1, 2).view(bs, dim, h * 2, w * 2).contiguous()
        # c3 = c3.transpose(1, 2).view(bs, dim, h, w).contiguous()
        # c4 = c4.transpose(1, 2).view(bs, dim, h // 2, w // 2).contiguous()
        # c1 = self.up(c2) + c1

        # if self.add_vit_feature:
        #     x1, x2, x3, x4 = outputs
        #     x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
        #     x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
        #     x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
        #     c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # # Final Norm
        # f1 = self.norm1(c1)
        # f2 = self.norm2(c2)
        # f3 = self.norm3(c3)
        # f4 = self.norm4(c4)
        # return [f1, f2, f3, f4]

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        if self.freeze_base:
            self.freeze_model()
        return self

    def freeze_model(self):
        if not getattr(self, 'freeze_base', True):
            for param in self.model.parameters():
                param.requires_grad = True
            if hasattr(self, 'register_tokens'):
                self.register_tokens.requires_grad = True
            if hasattr(self, 'local_register_tokens'):
                self.local_register_tokens.requires_grad = True
            return

        for param in self.model.parameters():
            param.requires_grad = False
        if hasattr(self, 'register_tokens'):
            self.register_tokens.requires_grad = True
        if hasattr(self, 'local_register_tokens'):
            self.local_register_tokens.requires_grad = True
        self.model.eval()

    def init_weights(self):
        pass


if __name__ == "__main__":
    input = torch.randn(1,3,560,560).cuda()
    model = SAM3RefinedWindowRegisterComer(img_size=1008,
        checkpoint_path='/home/cz/codes/githubs/sam3/checkpoints/sam3.pt',
        refined=True,
        refined_weight="/data/openclaw/UniRefiner/outputs_1/sam3pro/checkpoints/model_final.pt",
        num_register_tokens=4,
        num_local_register_tokens=2,
        freeze_base=True,
        local_model='cnn',
        interaction_indexes=((0, 7), (8, 15), (16, 23), (24, 31)),
    ).cuda()
    output = model(input)
    for o in output:
        print(o.shape)
