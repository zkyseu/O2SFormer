from typing import Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmcv.cnn.bricks.transformer import FFN
from mmcv.runner import force_fp32
from mmengine.model import BaseModule
from mmdet.models.builder import build_loss
from mmdet.models.builder import HEADS
from mmdet.core import multi_apply
from mmdet.core.bbox.builder import build_assigner,build_sampler

from .transformer import inverse_sigmoid
from .transformer_utils import MLP
from .utils.general_utils import ConfigType
from .lane import Lane


class SegDecoder(nn.Module):
    '''
    Optionaly seg decoder
    '''
    def __init__(self,
                 image_height,
                 image_width,
                 num_class,
                 feat_channels=64,
                 refine_layers=1):
        super().__init__()
        self.dropout = nn.Dropout2d(0.1)
        self.conv = nn.Conv2d(feat_channels * refine_layers, num_class,
                              1)
        self.image_height = image_height
        self.image_width = image_width

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv(x)
        x = F.interpolate(x,
                          size=[self.image_height, self.image_width],
                          mode='bilinear',
                          align_corners=False)
        return x

@HEADS.register_module()
class DNHead(BaseModule):
    def __init__(self,
                 num_classes,
                 num_reg_fcs = 2,
                 num_points = 72,
                 seg_feat = 256,
                 sample_points=36,
                 assigner: ConfigType = dict(),
                 loss_cls: ConfigType = dict(
                    type='CrossEntropyLoss',
                    bg_cls_weight=0.1,
                    use_sigmoid=False,
                    loss_weight=1.0,
                    class_weight=1.0),
                 loss_xyt: ConfigType = dict(),
                 loss_iou: ConfigType = dict(),
                 loss_seg: ConfigType = dict(),
                 img_info = None,
                 ori_img_info = None,
                 cut_height = None,
                 samle_cfg = dict(type='PseudoSampler'),
                 test_cfg: ConfigType = dict(max_per_img=100),
                 init_cfg = None,
                 max_lanes = 4,
                 **kwargs):
        super().__init__(init_cfg)

        self.sample_points = sample_points
        self.n_strips = num_points - 1
        self.n_offsets = num_points
        self.seg_feat = seg_feat
        self.num_reg_fcs = num_reg_fcs
        self.img_h,self.img_w = img_info
        self.ori_img_h,self.ori_img_w = ori_img_info
        self.cut_height = cut_height
        self.test_cfg = test_cfg
        self.max_lanes = max_lanes

        self.register_buffer(name='sample_x_indexs', tensor=(torch.linspace(
            0, 1, steps=self.sample_points, dtype=torch.float32) *
                                self.n_strips).long())
        self.register_buffer(name='prior_feat_ys', tensor=torch.flip(
            (1 - self.sample_x_indexs.float() / self.n_strips), dims=[-1]))
        self.register_buffer(name='prior_ys', tensor=torch.linspace(1,
                                       0,
                                       steps=self.n_offsets,
                                       dtype=torch.float32))

        self.seg_decoder = SegDecoder(self.img_h, self.img_w,
                                      num_classes + 1,
                                      seg_feat,)
        self.cls_loss = build_loss(loss_cls)
        self.xyt_loss = build_loss(loss_xyt)
        self.iou_loss = build_loss(loss_iou)
        self.seg_loss = build_loss(loss_seg)

        self.assigner = build_assigner(assigner)
        self.sampler = build_sampler(samle_cfg,context=self)        
        self.cls_layers = nn.Linear(self.seg_feat, 2)
        # self.reg_ffn = FFN(
        #     self.seg_feat,
        #     self.seg_feat,
        #     self.num_reg_fcs,
        #     dict(type='ReLU', inplace=True),
        #     dropout=0.0,
        #     add_residual=False)
        self.fc_reg = nn.Linear(self.seg_feat,3) # start_x + start_y + theta
        self.reg_length = nn.Linear(
            self.seg_feat, 1)  # 1 length
        self.reg_layers = nn.Linear(
            self.seg_feat, self.n_offsets)  # n offsets
        self.cls_layers = nn.Linear(self.seg_feat, 2)

        self.init_weights()

    # function to init layer weights
    def init_weights(self):
        # initialize heads
        for m in self.cls_layers.parameters():
            nn.init.normal_(m, mean=0., std=1e-3)

        for m in self.reg_layers.parameters():
            nn.init.normal_(m, mean=0., std=1e-3)

        for m in self.fc_reg.parameters():
            nn.init.normal_(m, mean=0., std=1e-3)

        for m in self.reg_length.parameters():
            nn.init.normal_(m, mean=0., std=1e-3)

    def tran_tensor(self,t):
        return t.unsqueeze(-1).clone().repeat(1, 1, 1, self.n_offsets)

    def forward(self, hidden_states: Tensor,
                references: Tensor,
                seg_feature: Tensor = None,
                img_metas = None):
        num_decoder_layers,batch_size,num_query = hidden_states.shape[0],hidden_states.shape[1],hidden_states.shape[2]
        hidden_states = hidden_states[-1]
        predicition_start_point = references[-1] #start point coord
        layers_cls_scores = self.cls_layers(hidden_states)
        # references_before_sigmoid = inverse_sigmoid(references, eps=1e-3)
        # pred_start_coord = self.fc_reg(hidden_states) #start point coord
        pred_offset = self.reg_layers(hidden_states) # offest
        pred_length = self.reg_length(hidden_states) # length
        
        # We first obtain the start point location
        predicition = torch.zeros_like(torch.cat([layers_cls_scores,predicition_start_point,pred_length,pred_offset],dim=-1))
        # start_point_offsets = pred_start_coord
        # assert start_point_offsets.size(-1) == references_before_sigmoid.size(-1)
        # start_point_offsets += references_before_sigmoid
        # predicition_start_point = start_point_offsets.sigmoid()
        
        # To get the predicition
        predicition[...,:2] = layers_cls_scores #cls score
        predicition[...,2:4] = predicition_start_point[...,:2] #start point coord
        predicition[...,4] = predicition_start_point[...,2] # lane angle
        predicition[...,5] = pred_length[...,0] # length
        predicition[..., 6:] = (
             self.tran_tensor(predicition[..., 3]) * (self.img_w - 1) + \
            ((1 - self.prior_ys.repeat(batch_size, num_query, 1) - \
             self.tran_tensor(predicition[..., 2])) * self.img_h / \
                torch.tan(self.tran_tensor(predicition[..., 4]) * math.pi + 1e-5))) / (self.img_w - 1)

        predicition[...,6:] += pred_offset

        # Compute segmentation
        if self.training and seg_feature is not None:
            assert img_metas is not None
            resize_shape = img_metas[0]['img_metas']['image_shape']
            seg_feature = F.interpolate(seg_feature,size=resize_shape,mode='bilinear',align_corners=False)
            seg = self.seg_decoder(seg_feature)
            output = {'prediction': predicition.unsqueeze(0), 'seg': seg}
            return output
        
        return predicition
    
    @force_fp32(apply_to=('prediction','seg','seg_targets'))
    def loss(self,
             prediction:Tensor,
             targets:Tensor,
             seg_targets = None,
             batch_img_metas = None,
             seg = None):
        # we calculate three types loss
        predictions_layer = prediction[-1]
#        cls_loss = 0
#        reg_xytl_loss = 0
#        iou_loss = 0
#        num_layers = prediction.shape[0]
        num_layers = 1
#        prediction_list = prediction.tolist()

#        for predictions_layer in prediction: # 遍历所有num_layer
        predictions_layer = [pred for pred in predictions_layer]
        targets_list = [target for target in targets]
        cls_loss_list,reg_xytl_loss_list,iou_loss_list = multi_apply(self.loss_single_bs,predictions_layer,targets_list)
        cls_loss = sum(cls_loss_list)
        reg_xytl_loss = sum(reg_xytl_loss_list)
        iou_loss = sum(iou_loss_list)

        cls_loss /= (len(targets)*num_layers)
        reg_xytl_loss /= (len(targets)*num_layers)
        iou_loss /= (len(targets)*num_layers)

        if seg is not None:
            seg_loss = self.seg_loss(seg,seg_targets.long())
            return dict(cls_loss = cls_loss,reg_xytl_loss = reg_xytl_loss,iou_loss = iou_loss, seg_loss = seg_loss)
        else:
            return dict(cls_loss = cls_loss,reg_xytl_loss = reg_xytl_loss,iou_loss = iou_loss)

    def get_target(self,                          
                   predictions: Tensor,
                   target: Tensor):
        
        targets = [target for i in range(predictions.shape[0])]
        predictions = [pred for pred in predictions]

        cls_target_list, target_yxtl_list, reg_targets_list, target_starts_list, \
            reg_yxtl_list, reg_pred_list, predictions_starts_list= multi_apply(self.get_target_single,predictions,targets)
        
        # get target
        cls_target = torch.cat(cls_target_list,0)
        target_yxtl = torch.cat(target_yxtl_list,0)
        reg_targets = torch.cat(reg_targets_list,0)
        target_starts = torch.cat(target_starts_list,0)
        # get pred
        reg_yxtl = torch.cat(reg_yxtl_list,0)
        reg_pred = torch.cat(reg_pred_list,0)
        predictions_starts = torch.cat(predictions_starts_list,0)

        return cls_target,target_yxtl,reg_targets,target_starts,reg_yxtl,reg_pred,predictions_starts

    def loss_single_bs(self,predictions:Tensor,target:Tensor):
        cls_loss = 0
        reg_xytl_loss = 0
        iou_loss = 0
        target = target[target[:, 1] == 1]
#        print("target x:",target[:,2:6])
#            print("target=== shape:",target.shape) #[valid_lane,78]

#            print("predictions shape:",predictions.shape) #[4,4,78]
        if len(target) == 0:
            # If there are no targets, all predictions have to be negatives (i.e., 0 confidence)
            cls_target = predictions.new_zeros(predictions.shape[0],).long() #get the label
            cls_pred = predictions[:, :2]
            cls_pred = cls_pred.reshape(predictions.shape[0],cls_pred.shape[-1])
            cls_loss = cls_loss + self.cls_loss(cls_pred, cls_target).sum()
            return cls_loss,reg_xytl_loss,iou_loss
        
        # apply label assignment
        cls_target,target_yxtl,reg_targets,target_starts, \
            reg_yxtl,reg_pred,predictions_starts = self.get_target_single(predictions,target)
        cls_pred = predictions[:, :2].reshape(-1,2)

        reg_yxtl[:, 0] *= self.n_strips
        reg_yxtl[:, 1] *= (self.img_w - 1)
        reg_yxtl[:, 2] *= 180
        reg_yxtl[:, 3] *= self.n_strips
#        print("reg_yxtl:",reg_yxtl)

        # regression targets -> S coordinates (all transformed to absolute values)
        reg_pred *= (self.img_w - 1)

        # with torch.no_grad():
        #     target_yxtl[:, -1] -= (predictions_starts - target_starts
        #                             )  # reg length

        cls_loss = cls_loss + self.cls_loss(cls_pred, cls_target,reduction_override='sum') / target.shape[0]

#        print("target_yxtl:",target_yxtl)
#        print("reg_yxtl:",reg_yxtl)

        target_yxtl[:, 0] *= self.n_strips
        target_yxtl[:, 2] *= 180
#        print("target_yxtl:",target_yxtl)
#        print("start point loss:",F.smooth_l1_loss(
#                    reg_yxtl, target_yxtl,
#                    reduction='none').mean())
#        print("++++++++++++++++++++++++++++++")
        reg_xytl_loss = reg_xytl_loss + self.xyt_loss(reg_yxtl, target_yxtl)

        iou_loss = iou_loss + self.iou_loss(reg_pred, reg_targets,self.img_w)
        return cls_loss,reg_xytl_loss,iou_loss

    def get_target_single(self,
                          predictions: Tensor,
                          target: Tensor):
        
        num_preds = predictions.size(0)
        target_assign = target.detach().clone()

        with torch.no_grad():
             matched_row_inds, matched_col_inds = self.assigner.assign(predictions, target_assign, self.img_w, self.img_h)
#        print("target x second:",target[:,2])
#            assign_result = self.assigner.assign(predictions, target, self.img_w, self.img_h)
#            sampling_result = self.sampler.sample(assign_result, predictions, target)

#        pos_inds = sampling_result.pos_inds
#        neg_inds = sampling_result.neg_inds

        labels = target.new_full((num_preds,),0,dtype=torch.long)
        labels[matched_row_inds] = 1

#        target_yxtl = sampling_result.pos_gt_bboxes[:,2:6].clone()
        reg_yxtl = predictions[matched_row_inds, 2:6]
        target_yxtl = target[matched_col_inds, 2:6].clone()
        reg_pred = predictions[matched_row_inds, 6:]
        reg_targets = target[matched_col_inds, 6:].clone()
#        reg_targets = sampling_result.pos_gt_bboxes[:,6:].clone()
#        target_yxtl = target[sampling_result.pos_assigned_gt_inds, 2:6].clone()
#        reg_targets = target[sampling_result.pos_assigned_gt_inds, 6:].clone()
#        reg_yxtl = predictions[pos_inds, 2:6]
#        reg_pred = predictions[pos_inds, 6:]

        with torch.no_grad():
#            print("target x:",target[matched_col_inds,2])
            target_starts = (target[matched_col_inds, 2] * self.n_strips).round().long()
            predictions_starts = torch.clamp((predictions[matched_row_inds, 2] * \
                                    self.n_strips).round().long(), 0, self.n_strips)  # ensure the predictions starts is valid
#            print("predictions_starts:",predictions_starts)
#            print("target_starts:",target_starts)
            target_yxtl[:, -1] -= (predictions_starts - target_starts
                                        )  # reg length

        return (labels,target_yxtl,reg_targets,target_starts,reg_yxtl,reg_pred,predictions_starts)

    def predictions_to_pred(self, predictions):
        '''
        Convert predictions to internal Lane structure for evaluation.
        '''
        self.prior_ys = self.prior_ys.to(predictions.device)
        self.prior_ys = self.prior_ys.double()
        lanes = []
        for lane in predictions:
            lane_xs = lane[6:]  # normalized value
            start = min(max(0, int(round(lane[2].item() * self.n_strips))),
                        self.n_strips)
            length = int(round(lane[5].item()))
            end = start + length - 1
            end = min(end, len(self.prior_ys) - 1)
            # end = label_end
            # if the prediction does not start at the bottom of the image,
            # extend its prediction until the x is outside the image
            mask = ~((((lane_xs[:start] >= 0.) & (lane_xs[:start] <= 1.)
                       ).cpu().numpy()[::-1].cumprod()[::-1]).astype(np.bool))
            lane_xs[end + 1:] = -2
            lane_xs[:start][mask] = -2
            lane_ys = self.prior_ys[lane_xs >= 0]
            lane_xs = lane_xs[lane_xs >= 0]
            lane_xs = lane_xs.flip(0).double()
            lane_ys = lane_ys.flip(0)

            lane_ys = (lane_ys * (self.ori_img_h - self.cut_height) +
                       self.cut_height) / self.ori_img_h
            if len(lane_xs) <= 1:
                continue
            points = torch.stack(
                (lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)),
                dim=1).squeeze(2)
            lane = Lane(points=points.cpu().numpy(),
                        metadata={
                            'start_x': lane[3],
                            'start_y': lane[2],
                            'conf': lane[1]
                        })
            lanes.append(lane)
        return lanes

    def get_lanes(self, predictions, as_lanes=True):
        '''
        Convert model output to lanes.
        '''

#        print("pred shape:",predictions.shape)
        predictions = predictions[-1,...]
        decoded = []
        threshold = self.test_cfg.conf_threshold
        scores = F.softmax(predictions[..., :2],dim=-1)[..., 1]
        score_topk,score_indice = torch.topk(scores,self.max_lanes,dim=-1)
        predictions = predictions[score_indice]
        keep_inds = score_topk >= threshold # use thres to filter false preditction
        predictions = predictions[keep_inds]
#        scores = scores[keep_inds]

        if predictions.shape[0] == 0:
            decoded.append([])
            return decoded

        predictions[..., 5] = torch.round(predictions[..., 5] * self.n_strips)
        if as_lanes:
            pred = self.predictions_to_pred(predictions)
        else:
            pred = predictions
        decoded.append(pred)
        return decoded
