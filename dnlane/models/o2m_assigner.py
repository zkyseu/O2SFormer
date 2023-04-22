import torch
from torch import Tensor
from scipy.optimize import linear_sum_assignment
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox import BaseAssigner
from mmdet.core.bbox.match_costs.builder import build_match_cost

from .losses.lane_iou import line_iou

def dynamic_k_assign(cost, pair_wise_ious,n_candidate_k=4,min_candidate = 2):
    """
    This file is borrowed from https://github.com/Turoad/CLRNet/blob/main/clrnet/models/utils/dynamic_assign.py
    Assign grouth truths with priors dynamically for SimOTA
    Args:
        cost: the assign cost.
        pair_wise_ious: iou of grouth truth and priors.
    Returns:
        prior_idx: the index of assigned prior.
        gt_idx: the corresponding ground truth index.
    """
    matching_matrix = torch.zeros_like(cost)
    ious_matrix = pair_wise_ious
    ious_matrix[ious_matrix < 0] = 0.
    topk_ious, _ = torch.topk(ious_matrix, n_candidate_k, dim=0)
    dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=min_candidate)
    num_gt = cost.shape[1]
    for gt_idx in range(num_gt):
        _, pos_idx = torch.topk(cost[:, gt_idx],
                                k=dynamic_ks[gt_idx].item(),
                                largest=False)
        matching_matrix[pos_idx, gt_idx] = 1.0
    del topk_ious, dynamic_ks, pos_idx

    matched_gt = matching_matrix.sum(1)
    if (matched_gt > 1).sum() > 0:
        _, cost_argmin = torch.min(cost[matched_gt > 1, :], dim=1)
        matching_matrix[matched_gt > 1, 0] *= 0.0
        matching_matrix[matched_gt > 1, cost_argmin] = 1.0

    prior_idx = matching_matrix.sum(1).nonzero()
    gt_idx = matching_matrix[prior_idx].argmax(-1)
    return prior_idx.flatten(), gt_idx.flatten()


@BBOX_ASSIGNERS.register_module()
class One2ManyLaneAssigner(BaseAssigner):
    def __init__(self,
                 distance_cost,
                 cls_cost):
        super().__init__()
        self.distance_cost = build_match_cost(distance_cost)
        self.cls_cost = build_match_cost(cls_cost)
    
    def assign(self,               
               predictions: Tensor,
               targets: Tensor,
               img_w,
               img_h,):
        hugpredictions = predictions.detach().clone()
        predictions = predictions.detach().clone()
        predictions[:, 3] *= (img_w - 1)
        predictions[:, 6:] *= (img_w - 1)

        # classification cost and distance cost
#        pair_wise_iou = line_iou(predictions[..., 6:].clone(), targets[..., 6:].clone(), img_w, aligned=False)
        cls_cost = self.cls_cost(predictions[:, :2], targets[:, 1].long())
        distance_cost = self.distance_cost(predictions,targets,img_w,img_h)
        cost = cls_cost+ distance_cost

        pair_wise_iou = line_iou(predictions[..., 6:].clone(), targets[..., 6:].clone(), img_w, aligned=False)
        ota_matched_row_inds, ota_matched_col_inds = dynamic_k_assign(cost, pair_wise_iou) # pred and target index
        cls_label = torch.unique(ota_matched_col_inds)
        final_match_row_inds = hugpredictions.new_zeros(targets.shape[0])#one2one assignment results
        final_match_col_inds = hugpredictions.new_zeros(targets.shape[0])
        for cls in cls_label:
            cls = cls.item()
            aux_cost = cost[:,cls].clone()
            unmatch_cls_col = ota_matched_row_inds[ota_matched_col_inds != cls]
            aux_cost[unmatch_cls_col] = 1000000000
            cls_col = ota_matched_col_inds==cls # label index
            cls_row = ota_matched_row_inds[cls_col] # pred index
            cost_cls = cost[cls_row][:,cls]
            assign_min_cost,indice = torch.min(cost_cls,dim = 0) # 二分图匹配
            index_coord = torch.nonzero((aux_cost==assign_min_cost).int())
            row_inds = index_coord[0][0]
            final_match_row_inds[cls] = row_inds
            final_match_col_inds[cls] = cls
        
        return ota_matched_row_inds,ota_matched_col_inds,final_match_row_inds,final_match_col_inds
            
