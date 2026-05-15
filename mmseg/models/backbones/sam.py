import torch
import pkg_resources
import PIL
import numpy as np
from torch import nn
from mmseg.registry import MODELS
from mmengine.model import BaseModule
from torchvision.transforms import v2
from iopath.common.file_io import g_pathmgr
from mmseg.models.backbones.sam3.sam3.model.act_ckpt_utils import activation_ckpt_wrapper

# processor
from mmseg.models.backbones.sam3.sam3.model import box_ops
from mmseg.models.backbones.sam3.sam3.model.data_misc import FindStage, interpolate


# position embeding
from mmseg.models.backbones.sam3.sam3.model.position_encoding import PositionEmbeddingSine

# image encoder
from mmseg.models.backbones.sam3.sam3.model.vitdet import ViT
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
class SAM3(BaseModule):
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
        
        
        # model = _create_sam3_model(
        #     backbone,
        #     transformer,
        #     input_geometry_encoder,
        #     segmentation_head,
        #     dot_prod_scoring,
        #     inst_predictor,
        #     eval_mode,
        # )
        
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
    
    def forward(self,image, text='road'):
        text = [text] * image.shape[0]
        if self.image_only:
            state = self.forward_image(image)
            return state['backbone_out']['backbone_fpn']
        elif self.mask2former:
            return self.forward_image_text(image,text)
        else:
            return self.forward_image_text_without_mask2former(image,text)

    def forward_image(self, image, state=None):
        """Sets the image on which we want to do predictions."""
        state = {}
        state["original_height"] = 560
        state["original_width"] = 560
        state["backbone_out"] = self.model.backbone.forward_image(image)
        return state
    
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
    model = SAM3(560,560,image_only=False,mask2former=True).to(device)
    model.eval()
    input = torch.randn(3,3,560,560).to(device)
    with torch.no_grad():
        output = model(input,['road','road','street'])
    for i in output:
        if isinstance(i,list):
            for _ in i:
                print(_.shape)