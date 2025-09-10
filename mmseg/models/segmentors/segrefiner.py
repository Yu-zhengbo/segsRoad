from collections import OrderedDict
import torch
import torch.distributed as dist
import numpy as np
from .base import BaseSegmentor
from mmseg.registry import MODELS
import os
import torch.nn.functional as F
import numpy as np
from mmcv.ops import nms
from .encoder_decoder import EncoderDecoder
from mmseg.structures import SegDataSample
from ..utils import resize
from mmengine.structures import PixelData


from .ddp import LearnedSinusoidalPosEmb
from torch import nn    


def uniform_sampler(num_steps, batch_size):
    all_indices = np.arange(num_steps)
    indices_np = np.random.choice(all_indices, size=(batch_size,))
    indices = torch.from_numpy(indices_np).long()
    return indices  

@MODELS.register_module()
class SegRefiner(EncoderDecoder):
    """Base class for detectors."""
    def __init__(self,
                 step,
                 denoise_model,
                 diffusion_cfg,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor = None,
                 loss_mask=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                 loss_texture=dict(type='TextureL1Loss', loss_weight=5.0),
                 init_cfg=None):
        super(SegRefiner, self).__init__(data_preprocessor=data_preprocessor,init_cfg=init_cfg)
        self.denoise_model = MODELS.build(denoise_model)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._diffusion_init(diffusion_cfg)
        self.align_corners = False
        # self.loss_mask = MODELS.build(loss_mask)
        # self.loss_texture = MODELS.build(loss_texture)
        self.num_classes = 2
        self.step = step
        self.denoise_model_name = denoise_model.get('type')
        if denoise_model.get('type') == "DeformableHeadWithTime":
            self.conv_patch = nn.Conv2d(4, 128, 4, 4, 0)
            learned_sinusoidal_dim = 16
            time_dim = 128 * 4  # 1024
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
            fourier_dim = learned_sinusoidal_dim + 1

            self.time_mlp = nn.Sequential(  # [2,]
                sinu_pos_emb,  # [2, 17]
                nn.Linear(fourier_dim, time_dim),  # [2, 1024]
                nn.GELU(),
                nn.Linear(time_dim, time_dim)  # [2, 1024]
            )
        
    
    def _diffusion_init(self, diffusion_cfg):
        self.diff_iter = diffusion_cfg['diff_iter']
        betas = diffusion_cfg['betas']
        self.eps = 1.e-6
        self.betas_cumprod = np.linspace(
            betas['start'], betas['stop'], 
            betas['num_timesteps'])
        betas_cumprod_prev = self.betas_cumprod[:-1]
        self.betas_cumprod_prev = np.insert(betas_cumprod_prev, 0, 1)
        self.betas = self.betas_cumprod / self.betas_cumprod_prev
        self.num_timesteps = self.betas_cumprod.shape[0]
    
    # def forward(self, img_metas, return_loss=True, **kwargs):

    #     if return_loss:
    #         return self.forward_train(**kwargs)
    #     else:
    #         if self.task == 'instance':
    #             return self.simple_test_instance(img_metas, **kwargs)
    #         elif self.task == 'semantic':
    #             return self.simple_test_semantic(img_metas, **kwargs)
    #         else:
    #             raise ValueError(f'unsupported task type: {self.task}')
    def forward(self,
            inputs,
            img2,
            data_samples = None,
            mode = 'tensor'):
        if mode == 'loss':
            return self.loss(inputs, img2, data_samples)
        elif mode == 'predict':
            return self.predict(inputs,img2, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs,img2, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def loss(self, inputs,img2, data_samples):
        # target, x_last, img, current_device = self.get_train_input(**kwargs)
        target = [
            data_sample.gt_sem_seg.data for data_sample in data_samples
        ]
        target = torch.stack(target, dim=0)
        current_device = target.device
        t = uniform_sampler(self.num_timesteps, inputs.shape[0]).to(current_device)
        x_t = self.q_sample(target, img2, t, current_device)
        z_t = torch.cat((inputs, x_t), dim=1)
        losses = self._decode_head_forward_train(z_t, t, data_samples)
        return losses
    
    def _decode_head_forward_train(self, inputs,time,
                                   data_samples) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        try:
            losses = self.denoise_model.loss(inputs, time,
                                            data_samples)
        except:
            inputs = self.conv_patch(inputs)
            gt_semantic_seg = torch.cat([data_samples[i].get('gt_sem_seg').data for i in range(len(data_samples))],dim=0).unsqueeze(1)
            input_times = self.time_mlp(time)
            losses = self.denoise_model.forward_train([inputs], input_times, data_samples, gt_semantic_seg,self.train_cfg)

        return losses

    def q_sample(self, x_start, x_last, t, current_device):
        q_ori_probs = torch.tensor(self.betas_cumprod, device=current_device)
        q_ori_probs = q_ori_probs[t]
        q_ori_probs = q_ori_probs.reshape(-1, 1, 1, 1)
        sample_noise = torch.rand(size=x_start.shape, device=current_device)
        transition_map = (sample_noise < q_ori_probs).float()
        sample = transition_map * x_start + (1 - transition_map) * x_last
        return sample
    
    def p_sample_loop(self, xs, indices, current_device, use_last_step=True):
        res, fine_probs = [], []
        for data in xs:
            x_last, img, cur_fine_probs = data
            if cur_fine_probs is None:
                cur_fine_probs = torch.zeros_like(x_last)
            x = x_last
            for i in indices:
                t = torch.tensor([i] * x.shape[0], device=current_device)
                last_step_flag = (use_last_step and i==indices[-1])
                model_input = torch.cat((img, x), dim=1)
                x, cur_fine_probs = self.p_sample(model_input, cur_fine_probs, t)

                if last_step_flag:
                    x =  x#.sigmoid()
                else:
                    sample_noise = torch.rand(size=x.shape, device=x.device)
                    fine_map = (sample_noise < cur_fine_probs).float()
                    pred_x_start = (x >= 0).float()
                    x = pred_x_start * fine_map + x_last * (1 - fine_map)
            res.append(x)
            fine_probs.append(cur_fine_probs)
        res = torch.cat(res, dim=0)
        fine_probs = torch.cat(fine_probs, dim=0)
        return res, fine_probs

    def p_sample(self, model_input, cur_fine_probs, t):
        if self.denoise_model_name == "DeformableHeadWithTime":
            inputs = self.conv_patch(model_input)
            input_times = self.time_mlp(t)
            pred_logits = self.denoise_model([inputs], input_times)
            pred_logits = F.interpolate(pred_logits, size=cur_fine_probs.shape[-2:])
        else:
            pred_logits = self.denoise_model(model_input, t)
        t = t[0].item()
        x_start_fine_probs = 2 * torch.abs(pred_logits.sigmoid() - 0.5)
        beta_cumprod = self.betas_cumprod[t]
        beta_cumprod_prev = self.betas_cumprod_prev[t]
        p_c_to_f = x_start_fine_probs * (beta_cumprod_prev - beta_cumprod) / (1 - x_start_fine_probs*beta_cumprod)
        cur_fine_probs = cur_fine_probs + (1 - cur_fine_probs) * p_c_to_f
        return pred_logits, cur_fine_probs
    
    def predict(self,
            inputs,
            img2,
            data_samples):
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, img2, batch_img_metas)

        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self,
                inputs,
                data_samples):
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(inputs)
        return self.decode_head.forward(x)

    def inference(self, img, coarse_masks, batch_img_metas):

        if coarse_masks.sum() <= 128:
            return torch.zeros_like(coarse_masks)
        ori_shape = batch_img_metas[0]['ori_shape']
        # if coarse_masks[0].masks.sum() <= 128:
        #     return [(np.zeros_like(coarse_masks[0].masks[0]), output_file)]
        current_device = img.device
        
        indices = list(range(self.num_timesteps))[::-1]
        global_indices = indices
        # global_indices = indices[:-1]
        # local_indices = [indices[-1]]

        # global_step
        # global_img, global_mask = self._get_global_input(img, coarse_masks, ori_shape, current_device)
        global_img, global_mask = img, coarse_masks
        model_size_mask, fine_probs = self.p_sample_loop([(global_mask, global_img, None)], 
                                                        global_indices, 
                                                        current_device, 
                                                        use_last_step=True)
        
        return model_size_mask
        # ori_size_mask = F.interpolate(model_size_mask, size=ori_shape)
        # ori_size_mask = (ori_size_mask >= 0.5).float()

        # if patch_imgs is None:
        #     return [ori_size_mask[0, 0].cpu().numpy()]


        # # local_step
        # patch_imgs, patch_masks, patch_fine_probs, patch_coors = \
        #     self.get_local_input(img, ori_size_mask, fine_probs, ori_shape)
        
        # batch_max = self.test_cfg.get('batch_max', 0)
        # num_ins = len(patch_imgs)
        # if num_ins <= batch_max:
        #     xs = [(patch_masks, patch_imgs, patch_fine_probs)]
        # else:
        #     xs = []
        #     for idx in range(0, num_ins, batch_max):
        #         end = min(num_ins, idx + batch_max)
        #         xs.append((patch_masks[idx: end], patch_imgs[idx:end], patch_fine_probs[idx:end]))

        # local_masks, _ = self.p_sample_loop(xs, 
        #                                     local_indices, 
        #                                     patch_imgs.device,
        #                                     use_last_step=True)
        
        # # local_masks = (local_masks >= 0.5).float()
        # # local_save(patch_imgs, local_masks, patch_masks, torch.zeros_like(local_masks), img_metas, 'local')
        
        # mask = self.paste_local_patch(local_masks, ori_size_mask, patch_coors)
        # return [(mask.cpu().numpy(), output_file)]

        
    def get_local_input(self, img, ori_size_mask, fine_probs, ori_shape):
        img_h, img_w = ori_shape
        ori_size_fine_probs = F.interpolate(fine_probs, ori_shape)
        fine_prob_thr = self.test_cfg.get('fine_prob_thr', 0.9)
        fine_prob_thr = fine_probs.max().item() * fine_prob_thr
        model_size = self.test_cfg.get('model_size', 512)
        low_cofidence_points = fine_probs < fine_prob_thr
        scores = fine_probs[low_cofidence_points]
        y_c, x_c = torch.where(low_cofidence_points.squeeze(0).squeeze(0))
        scale_factor_y, scale_factor_x = img_h / model_size, img_w / model_size
        y_c, x_c = (y_c * scale_factor_y).int(), (x_c * scale_factor_x).int()        
        scores = 1 - scores
        patch_coors = self._get_patch_coors(x_c, y_c, 0, 0, img_w, img_h, model_size, scores)
        return self.crop_patch(img, ori_size_mask, ori_size_fine_probs, patch_coors)
    
    def _get_patch_coors(self, x_c, y_c, X_1, Y_1, X_2, Y_2, patch_size, scores):
        y_1, y_2 = y_c - patch_size/2, y_c + patch_size/2 
        x_1, x_2 = x_c - patch_size/2, x_c + patch_size/2
        invalid_y = y_1 < Y_1
        y_1[invalid_y] = Y_1
        y_2[invalid_y] = Y_1 + patch_size
        invalid_y = y_2 > Y_2
        y_1[invalid_y] = Y_2 - patch_size
        y_2[invalid_y] = Y_2
        invalid_x = x_1 < X_1
        x_1[invalid_x] = X_1
        x_2[invalid_x] = X_1 + patch_size
        invalid_x = x_2 > X_2
        x_1[invalid_x] = X_2 - patch_size
        x_2[invalid_x] = X_2
        proposals = torch.stack((x_1, y_1, x_2, y_2), dim=-1)
        patch_coors, _ = nms(proposals, scores, iou_threshold=self.test_cfg.get('iou_thr', 0.2))
        return patch_coors.int()
    
    def crop_patch(self, img, mask, fine_probs, patch_coors):
        patch_imgs, patch_masks, patch_fine_probs, new_patch_coors = [], [], [], []
        for coor in patch_coors:
            patch_mask = mask[:, :, coor[1]:coor[3], coor[0]:coor[2]]
            if (patch_mask.any()) and (not patch_mask.all()):
                patch_imgs.append(img[:, :, coor[1]:coor[3], coor[0]:coor[2]])
                patch_fine_probs.append(fine_probs[:, :, coor[1]:coor[3], coor[0]:coor[2]])
                patch_masks.append(patch_mask)
                new_patch_coors.append(coor)
        if len(patch_imgs) == 0:
            return None, None, None, None
        patch_imgs = torch.cat(patch_imgs, dim=0)
        patch_masks = torch.cat(patch_masks, dim=0)
        patch_fine_probs = torch.cat(patch_fine_probs, dim=0)
        patch_masks = (patch_masks >= 0.5).float()
        return patch_imgs, patch_masks, patch_fine_probs, new_patch_coors
    
    def paste_local_patch(self, local_masks, mask, patch_coors):
        mask = mask.squeeze(0).squeeze(0)
        refined_mask = torch.zeros_like(mask)
        weight = torch.zeros_like(mask)
        local_masks = local_masks.squeeze(1)
        for local_mask, coor in zip(local_masks, patch_coors):
            refined_mask[coor[1]:coor[3], coor[0]:coor[2]] += local_mask
            weight[coor[1]:coor[3], coor[0]:coor[2]] += 1
        refined_area = (weight > 0).float()
        weight[weight == 0] = 1
        refined_mask = refined_mask / weight
        refined_mask = (refined_mask >= 0.5).float()
        return refined_area * refined_mask + (1 - refined_area) * mask

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
    
    def extract_feat(self, img):
        """Directly extract features from the backbone and neck."""
        raise NotImplementedError
    

    def postprocess_result(self,
                           seg_logits,
                           data_samples= None):
        """ Convert results list to `SegDataSample`.
        Args:
            seg_logits (Tensor): The segmentation results, seg_logits from
                model of each input image.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        batch_size, C, H, W = seg_logits.shape

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                img_meta = data_samples[i].metainfo
                # remove padding area
                if 'img_padding_size' not in img_meta:
                    padding_size = img_meta.get('padding_size', [0] * 4)
                else:
                    padding_size = img_meta['img_padding_size']
                padding_left, padding_right, padding_top, padding_bottom =\
                    padding_size
                # i_seg_logits shape is 1, C, H, W after remove padding
                i_seg_logits = seg_logits[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]

                flip = img_meta.get('flip', None)
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        i_seg_logits = i_seg_logits.flip(dims=(3, ))
                    else:
                        i_seg_logits = i_seg_logits.flip(dims=(2, ))

                # resize as original shape
                i_seg_logits = resize(
                    i_seg_logits,
                    size=img_meta['ori_shape'],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False).squeeze(0)
            else:
                i_seg_logits = seg_logits[i]

            if C > 1:
                i_seg_pred = i_seg_logits.argmax(dim=0, keepdim=True)
            else:
                i_seg_logits = i_seg_logits.sigmoid()
                i_seg_pred = (i_seg_logits >
                              0.3).to(i_seg_logits)
            data_samples[i].set_data({
                'seg_logits':
                PixelData(**{'data': i_seg_logits}),
                'pred_sem_seg':
                PixelData(**{'data': i_seg_pred})
            })

        return data_samples
