# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import math
from mmseg.registry import MODELS
from .utils import get_class_weight, weight_reduce_loss
import numpy
import gudhi as gd
from pylab import *
import torch


def compute_dgm_force(lh_dgm, gt_dgm, pers_thresh=0.03, pers_thresh_perfect=0.99, do_return_perfect=False):
    """
    Compute the persistent diagram of the image

    Args:
        lh_dgm: likelihood persistent diagram.
        gt_dgm: ground truth persistent diagram.
        pers_thresh: Persistent threshold, which also called dynamic value, which measure the difference.
        between the local maximum critical point value with its neighouboring minimum critical point value.
        The value smaller than the persistent threshold should be filtered. Default: 0.03
        pers_thresh_perfect: The distance difference between two critical points that can be considered as
        correct match. Default: 0.99
        do_return_perfect: Return the persistent point or not from the matching. Default: False

    Returns:
        force_list: The matching between the likelihood and ground truth persistent diagram
        idx_holes_to_fix: The index of persistent points that requires to fix in the following training process
        idx_holes_to_remove: The index of persistent points that require to remove for the following training
        process

    """
    lh_pers = abs(lh_dgm[:, 1] - lh_dgm[:, 0])
    if (gt_dgm.shape[0] == 0):
        gt_pers = None
        gt_n_holes = 0
    else:
        gt_pers = gt_dgm[:, 1] - gt_dgm[:, 0]
        gt_n_holes = gt_pers.size  # number of holes in gt

    if (gt_pers is None or gt_n_holes == 0):
        idx_holes_to_fix = list()
        idx_holes_to_remove = list(set(range(lh_pers.size)))
        idx_holes_perfect = list()
    else:
        # check to ensure that all gt dots have persistence 1
        tmp = gt_pers > pers_thresh_perfect

        # get "perfect holes" - holes which do not need to be fixed, i.e., find top
        # lh_n_holes_perfect indices
        # check to ensure that at least one dot has persistence 1 it is the hole
        # formed by the padded boundary
        # if no hole is ~1 (ie >.999) then just take all holes with max values
        tmp = lh_pers > pers_thresh_perfect  # old: assert tmp.sum() >= 1
        lh_pers_sorted_indices = np.argsort(lh_pers)[::-1]
        if np.sum(tmp) >= 1:
            lh_n_holes_perfect = tmp.sum()
            idx_holes_perfect = lh_pers_sorted_indices[:lh_n_holes_perfect]
        else:
            idx_holes_perfect = list()

        # find top gt_n_holes indices
        idx_holes_to_fix_or_perfect = lh_pers_sorted_indices[:gt_n_holes]

        # the difference is holes to be fixed to perfect
        idx_holes_to_fix = list(
            set(idx_holes_to_fix_or_perfect) - set(idx_holes_perfect))

        # remaining holes are all to be removed
        idx_holes_to_remove = lh_pers_sorted_indices[gt_n_holes:]

    # only select the ones whose persistence is large enough
    # set a threshold to remove meaningless persistence dots
    pers_thd = pers_thresh
    idx_valid = np.where(lh_pers > pers_thd)[0]
    idx_holes_to_remove = list(
        set(idx_holes_to_remove).intersection(set(idx_valid)))

    force_list = np.zeros(lh_dgm.shape)
    
    # push each hole-to-fix to (0,1)
    force_list[idx_holes_to_fix, 0] = 0 - lh_dgm[idx_holes_to_fix, 0]
    force_list[idx_holes_to_fix, 1] = 1 - lh_dgm[idx_holes_to_fix, 1]

    # push each hole-to-remove to (0,1)
    force_list[idx_holes_to_remove, 0] = lh_pers[idx_holes_to_remove] / \
                                         math.sqrt(2.0)
    force_list[idx_holes_to_remove, 1] = -lh_pers[idx_holes_to_remove] / \
                                         math.sqrt(2.0)

    if (do_return_perfect):
        return force_list, idx_holes_to_fix, idx_holes_to_remove, idx_holes_perfect

    return force_list, idx_holes_to_fix, idx_holes_to_remove

def getCriticalPoints(likelihood):
    """
    Compute the critical points of the image (Value range from 0 -> 1)

    Args:
        likelihood: Likelihood image from the output of the neural networks

    Returns:
        pd_lh:  persistence diagram.
        bcp_lh: Birth critical points.
        dcp_lh: Death critical points.
        Bool:   Skip the process if number of matching pairs is zero.

    """
    lh = 1 - likelihood
    lh_vector = np.asarray(lh).flatten()

    lh_cubic = gd.CubicalComplex(
        dimensions=[lh.shape[0], lh.shape[1]],
        top_dimensional_cells=lh_vector
    )

    Diag_lh = lh_cubic.persistence(homology_coeff_field=2, min_persistence=0)
    pairs_lh = lh_cubic.cofaces_of_persistence_pairs()

    # If the paris is 0, return False to skip
    if (len(pairs_lh[0])==0): return 0, 0, 0, False

    # return persistence diagram, birth/death critical points
    pd_lh = numpy.array([[lh_vector[pairs_lh[0][0][i][0]], lh_vector[pairs_lh[0][0][i][1]]] for i in range(len(pairs_lh[0][0]))])
    bcp_lh = numpy.array([[pairs_lh[0][0][i][0]//lh.shape[1], pairs_lh[0][0][i][0]%lh.shape[1]] for i in range(len(pairs_lh[0][0]))])
    dcp_lh = numpy.array([[pairs_lh[0][0][i][1]//lh.shape[1], pairs_lh[0][0][i][1]%lh.shape[1]] for i in range(len(pairs_lh[0][0]))])

    return pd_lh, bcp_lh, dcp_lh, True

def getTopoLoss(likelihood_tensor, gt_tensor, topo_size=100):
    """
    Calculate the topology loss of the predicted image and ground truth image 
    Warning: To make sure the topology loss is able to back-propagation, likelihood 
    tensor requires to clone before detach from GPUs. In the end, you can hook the
    likelihood tensor to GPUs device.

    Args:
        likelihood_tensor:   The likelihood pytorch tensor.
        gt_tensor        :   The groundtruth of pytorch tensor.
        topo_size        :   The size of the patch is used. Default: 100

    Returns:
        loss_topo        :   The topology loss value (tensor)

    """

    likelihood = torch.sigmoid(likelihood_tensor).clone()
    gt = gt_tensor.clone()

    likelihood = torch.squeeze(likelihood).cpu().detach().numpy()
    gt = torch.squeeze(gt).cpu().detach().numpy()

    topo_cp_weight_map = np.zeros(likelihood.shape)
    topo_cp_ref_map = np.zeros(likelihood.shape)

    for y in range(0, likelihood.shape[0], topo_size):
        for x in range(0, likelihood.shape[1], topo_size):

            lh_patch = likelihood[y:min(y + topo_size, likelihood.shape[0]),
                         x:min(x + topo_size, likelihood.shape[1])]
            gt_patch = gt[y:min(y + topo_size, gt.shape[0]),
                         x:min(x + topo_size, gt.shape[1])]

            if(np.min(lh_patch) == 1 or np.max(lh_patch) == 0): continue
            if(np.min(gt_patch) == 1 or np.max(gt_patch) == 0): continue

            # Get the critical points of predictions and ground truth
            pd_lh, bcp_lh, dcp_lh, pairs_lh_pa = getCriticalPoints(lh_patch)
            pd_gt, bcp_gt, dcp_gt, pairs_lh_gt = getCriticalPoints(gt_patch)

            # If the pairs not exist, continue for the next loop
            if not(pairs_lh_pa): continue
            if not(pairs_lh_gt): continue

            force_list, idx_holes_to_fix, idx_holes_to_remove = compute_dgm_force(pd_lh, pd_gt, pers_thresh=0.03)

            if (len(idx_holes_to_fix) > 0 or len(idx_holes_to_remove) > 0):
                for hole_indx in idx_holes_to_fix:
                    if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[0] and int(
                            bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) < likelihood.shape[1]):
                        topo_cp_weight_map[y + int(bcp_lh[hole_indx][0]), x + int(
                            bcp_lh[hole_indx][1])] = 1  # push birth to 0 i.e. min birth prob or likelihood
                        topo_cp_ref_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = 0
                    if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                        0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                            likelihood.shape[1]):
                        topo_cp_weight_map[y + int(dcp_lh[hole_indx][0]), x + int(
                            dcp_lh[hole_indx][1])] = 1  # push death to 1 i.e. max death prob or likelihood
                        topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = 1
                for hole_indx in idx_holes_to_remove:
                    if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[
                        0] and int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) <
                            likelihood.shape[1]):
                        topo_cp_weight_map[y + int(bcp_lh[hole_indx][0]), x + int(
                            bcp_lh[hole_indx][1])] = 1  # push birth to death  # push to diagonal
                        if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                            0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                                likelihood.shape[1]):
                            topo_cp_ref_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = \
                                lh_patch[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])]
                        else:
                            topo_cp_ref_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = 1
                    if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                        0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                            likelihood.shape[1]):
                        topo_cp_weight_map[y + int(dcp_lh[hole_indx][0]), x + int(
                            dcp_lh[hole_indx][1])] = 1  # push death to birth # push to diagonal
                        if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[
                            0] and int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) <
                                likelihood.shape[1]):
                            topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = \
                                lh_patch[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])]
                        else:
                            topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = 0

    topo_cp_weight_map = torch.tensor(topo_cp_weight_map, dtype=torch.float).cuda()
    topo_cp_ref_map = torch.tensor(topo_cp_ref_map, dtype=torch.float).cuda()

    # Measuring the MSE loss between predicted critical points and reference critical points
    loss_topo = (((likelihood_tensor * topo_cp_weight_map) - topo_cp_ref_map) ** 2).sum()
    return loss_topo


@MODELS.register_module()
class TopoLoss(nn.Module):
    def __init__(self,
                 topo_size=100,
                 pers_thresh=0.03,
                 lambda_max=0.01,
                 warmup_iters=20000,
                 ramp_iters=60000,
                 every_n_iters=5,
                 sample_ratio=0.5,
                 loss_name='loss_topo',
                 use_softmax=True):
        super().__init__()
        self.lambda_max = float(lambda_max)
        self.warmup_iters = int(warmup_iters)
        self.ramp_iters = int(ramp_iters)
        self.every_n_iters = int(every_n_iters)
        self.sample_ratio = float(sample_ratio)
        self._iter = 0
        self.topo_size = topo_size
        self._loss_name = loss_name
        self.use_softmax = use_softmax
        # self.topo = TopoLoss(topo_size=topo_size, pers_thresh=pers_thresh)

    def _lambda_now(self):
        t = self._iter
        if t < self.warmup_iters:
            return 0.0
        if self.ramp_iters <= 0:
            return self.lambda_max
        # 线性 ramp
        r = min(1.0, (t - self.warmup_iters) / float(self.ramp_iters))
        return self.lambda_max * r

    def forward(self, pred, target, ignore_index, **kwargs):
        self._iter += 1

        lam = self._lambda_now()
        if lam <= 0:
            return pred.sum() * 0.0  # no-op

        if self.every_n_iters > 1 and (self._iter % self.every_n_iters != 0):
            return pred.sum() * 0.0

        if self.use_softmax:
            pred = pred.softmax(dim=1)[:,1,:,:]

        # if target.dim() == 3:
        #     target = target.unsqueeze(1)
        # target = target.float()

        if self.sample_ratio < 1.0:
            B = pred.size(0)
            k = max(1, int(round(B * self.sample_ratio)))
            idx = torch.randperm(B, device=pred.device)[:k]
            pred = pred[idx]
            target = target[idx]
        
        

        valid_mask = (target != ignore_index).float()
        # 把 ignore 区域在 target 里当作背景（或任意常数）
        target_clean = target.clone()
        target_clean[target_clean == ignore_index] = 0.0
        # 同时在 pred 上 mask 掉（让这些位置不产生梯度）
        pred_clean = pred * valid_mask
        
        
        loss = 0
        for i in range(pred_clean.shape[0]):
            loss += getTopoLoss(pred_clean[i], target_clean[i], topo_size=self.topo_size)
        return loss * lam
    
    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name