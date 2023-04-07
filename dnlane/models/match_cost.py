import torch
from torch import Tensor
from mmdet.core.bbox.match_costs.builder import MATCH_COST
from mmdet.core.bbox.match_costs import FocalLossCost

@MATCH_COST.register_module()
class Distance_cost:
    def __init__(self,weight):
        self.weight = weight
        
    def point_distance(self,predictions: Tensor,targets: Tensor,img_w):
        """
        Args:
            prediction: [num_query, dim]
            targets: [num_targets, dim]
        """

        num_query = predictions.shape[0]
        num_targets = targets.shape[0]

        predictions = torch.repeat_interleave(predictions[:,6:],num_targets,dim=0)
        targets = torch.cat([targets]*num_query,dim=0)[:,6:]

        invalid_masks = (targets < 0) | (targets >= img_w)
        lengths = (~invalid_masks).sum(dim=-1)
        distances = torch.abs((targets - predictions))
        distances[invalid_masks] = 0.
        distances = distances.sum(dim=-1) / (lengths.float() + 1e-9)
        distances = distances.view(num_query,num_targets)

        return distances
    
    def __call__(self,
                 predictions: Tensor,
                 targets: Tensor,
                 img_w,
                 img_h,):
        num_query = predictions.size(0)
        num_gts = targets.size(0)

        distances_score = self.point_distance(predictions,targets,img_w)
        distances_score = 1 - (distances_score / torch.max(distances_score)
                           ) + 1e-2  # normalize the distance
        
        target_start_xys = targets[:, 2:4]  # num_targets, 2
        target_start_xys[..., 0] *= (img_h - 1)
        prediction_start_xys = predictions[:, 2:4]
        prediction_start_xys[..., 0] *= (img_h - 1)

        start_xys_score = torch.cdist(prediction_start_xys, target_start_xys,p=2).reshape(num_query, num_gts)
        start_xys_score = (1 - start_xys_score / torch.max(start_xys_score)) + 1e-2

        target_thetas = targets[..., 4].unsqueeze(-1)
        prediction_thetas = predictions[..., 4].unsqueeze(-1)

        theta_score = torch.cdist(prediction_thetas,target_thetas,p=1).reshape(num_query, num_gts) * 180
        theta_score = (1 - theta_score / torch.max(theta_score)) + 1e-2

        distance_cost = -(distances_score * start_xys_score * theta_score)**2*self.weight

        return distance_cost

@MATCH_COST.register_module()
class FocalIOULossCost(FocalLossCost):
    def _focal_loss_cost(self, cls_pred:Tensor, gt_labels:Tensor,pair_wise_iou):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                (num_query, num_class).
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
        Returns:
            torch.Tensor: cls_cost value with weight
        """
        cls_pred = cls_pred.sigmoid()
        pair_wise_iou = pair_wise_iou.clone()
        pair_wise_iou[pair_wise_iou<0] = 0.
#        print("pair_wise_iou shape:",pair_wise_iou)
#        print("cls pred shape:",cls_pred.shape)
#        print("gt label shape:",gt_labels)
#        neg_cost = -self.alpha*cls_pred.pow(self.gamma)*(1 - cls_pred + self.eps).log()
#        pos_cost = -pair_wise_iou*(pair_wise_iou*(cls_pred + self.eps).log()+ \
#                                   (1-pair_wise_iou)*(1 - cls_pred + self.eps).log())
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)

        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels] #[192,4]
        return cls_cost * self.weight

    def __call__(self, cls_pred, gt_labels,pair_wise_iou=None):
        """
        Args:
            cls_pred (Tensor): Predicted classfication logits.
            gt_labels (Tensor)): Labels.
        Returns:
            Tensor: Focal cost matrix with weight in shape\
                (num_query, num_gt).
        """
        if self.binary_input:
            return self._mask_focal_loss_cost(cls_pred, gt_labels)
        else:
            assert pair_wise_iou is not None,'pair_wise_iou should not be None'
            return self._focal_loss_cost(cls_pred, gt_labels,pair_wise_iou)    