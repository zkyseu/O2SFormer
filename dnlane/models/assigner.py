import torch
from torch import Tensor
from scipy.optimize import linear_sum_assignment
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox import BaseAssigner,AssignResult
from mmdet.core.bbox.match_costs.builder import build_match_cost
from mmdet.core import multi_apply
        

@BBOX_ASSIGNERS.register_module()
class HungarianLaneAssigner(BaseAssigner):
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
        """
        get targets for single decoder layer
        Args:
            predictions: [num_query,78]
            targets: [num_gts,78]
        """
        predictions = predictions.detach().clone()
        predictions[:, 3] *= (img_w - 1)
        predictions[:, 6:] *= (img_w - 1)

        num_query = predictions.size(0)
        num_gts = targets.size(0)

        assigned_gt_inds = predictions.new_full((num_query,),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = predictions.new_full((num_query,),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_query == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        cls_cost = self.cls_cost(predictions[:, :2], targets[:, 1].long())
        distance_cost = self.distance_cost(predictions,targets,img_w,img_h)
        cost = cls_cost + distance_cost

        cost = cost.detach().cpu()
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost) #pred,gt
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            predictions.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            predictions.device)
        return matched_row_inds,matched_col_inds

#        gt_labels = predictions.new_ones(num_gts,).long()
#        assigned_gt_inds[:] = 0
#        # assign foregrounds based on matching results
#        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
#        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds,]
#
#        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)