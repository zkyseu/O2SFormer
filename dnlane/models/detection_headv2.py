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

from .transformer_utils import MLP
from .utils.general_utils import ConfigType
from .lane import Lane
from .losses.lane_iou import line_iou

class MLP_(MLP):
    def forward(self, x: Tensor) -> Tensor:
        """Forward function of MLP.
        Args:
            x (Tensor): The input feature, has shape
                (num_queries, bs, input_dim).
        Returns:
            Tensor: The output feature, has shape
                (num_queries, bs, output_dim).
        """
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x))
        return x

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
class DNHeadv2(BaseModule):
    def __init__(self,
                 num_classes,
                 num_reg_fcs = 2,
                 num_points = 72,
                 seg_feat = 256,
                 seg_decoder_feat = sum([64,128,256,512]),
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
                 num_feat_layers = 4,
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
                                      seg_decoder_feat,)
        self.cls_loss = build_loss(loss_cls)
        self.xyt_loss = build_loss(loss_xyt)
        self.iou_loss = build_loss(loss_iou)
        self.seg_loss = build_loss(loss_seg)

        self.assigner = build_assigner(assigner)
        self.sampler = build_sampler(samle_cfg,context=self)        
        self.cls_layers = nn.Linear(self.seg_feat, 2)
        self.feat_layer = MLP_(self.seg_feat,self.seg_feat,self.seg_feat,num_feat_layers)
        self.fc_reg = MLP(self.seg_feat,self.seg_feat,3,num_feat_layers) # start_x + start_y + theta
        self.reg_length = nn.Linear(
            self.seg_feat, 1)  # 1 length
        self.reg_layers = nn.Linear(
            self.seg_feat, self.n_offsets)  # n offsets
        self.cls_layers = nn.Linear(self.seg_feat, 2)

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

    def tran_tensor(self,t:Tensor):
        return t.unsqueeze(-1).clone().repeat(1, 1, 1, self.n_offsets)

    def forward(self, hidden_states: Tensor,
                references: Tensor,
                offset_points: Tensor,
                seg_feature: Tensor = None,
                img_metas = None):
        num_decoder_layers,batch_size,num_query = hidden_states.shape[0],hidden_states.shape[1],hidden_states.shape[2]
        # predict start point and theta
        predicition_start_point = self.fc_reg(hidden_states)
        assert predicition_start_point.size(-1) == references.size(-1) 
        predicition_start_point += references
        
        # predict cls score,offest and length
        hidden_states = self.feat_layer(hidden_states)
        predicition_start_point = references #start point coord
        layers_cls_scores = self.cls_layers(hidden_states)
        pred_offset = self.reg_layers(hidden_states) # offest
        pred_offset += offset_points #update offset
        pred_length = self.reg_length(hidden_states) # length
        
        # We first obtain the start point location
        predicition = torch.zeros_like(torch.cat([layers_cls_scores,predicition_start_point,pred_length,pred_offset],dim=-1))
        
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
#            seg_feature = F.interpolate(seg_feature,size=resize_shape,mode='bilinear',align_corners=False)
            seg = self.seg_decoder(seg_feature)
            output = {'prediction': predicition, 'seg': seg}
            return output
        
        return predicition[-1]

    @force_fp32(apply_to=('prediction','targets','seg','seg_targets'))
    def loss(self,
             prediction:Tensor,
             targets:Tensor,
             seg_targets = None,
             seg = None):

        # we calculate three types loss
        cls_loss = 0
        reg_xytl_loss = torch.tensor(0.).to(prediction.device)
        iou_loss = torch.tensor(0.).to(prediction.device)
        num_layers = prediction.shape[0]

        # apply label assignment for each layer
        final_pred = prediction[-1].clone() # BxNQX78
        final_pred_list = [pred for pred in final_pred]
        targets_list = [target for target in targets]
        ota_matched_col_inds,ota_matched_row_inds,match_col_inds,match_row_inds,is_empty_list = multi_apply(self.gengeral_label_assign,final_pred_list,targets_list)
        line_iou_list = multi_apply(self.get_iou_sim,final_pred_list,targets_list,ota_matched_row_inds,ota_matched_col_inds,match_row_inds,is_empty_list)[0]
        
        for idx,predictions_layer in enumerate(prediction): # 遍历所有num_layer
            is_last = (idx == num_layers-1)
            scale_factor = ((num_layers-1-idx)/(num_layers-1))
            scale_factor_list = [scale_factor for pred in predictions_layer]
            predictions_layer = [pred for pred in predictions_layer] # 遍历所有batchsize
            is_last_list = [True if is_last else False for pred in predictions_layer]
            if not is_last:
                assigned_labels = multi_apply(self.get_assign_score,predictions_layer,ota_matched_row_inds,match_row_inds,is_last_list,is_empty_list,scale_factor_list,line_iou_list)[0]
                cls_loss_list,reg_xytl_loss_list,iou_loss_list = multi_apply(self.loss_single_bs,predictions_layer,targets_list,ota_matched_row_inds,ota_matched_col_inds,assigned_labels)
            else:
                assigned_labels = multi_apply(self.get_assign_score,predictions_layer,match_row_inds,match_row_inds,is_last_list,is_empty_list)[0]
                cls_loss_list,reg_xytl_loss_list,iou_loss_list = multi_apply(self.loss_single_bs,predictions_layer,targets_list,match_row_inds,match_col_inds,assigned_labels)

            cls_loss += sum(cls_loss_list)
            reg_xytl_loss += sum(reg_xytl_loss_list)
            iou_loss += sum(iou_loss_list)
            
        cls_loss /= (len(targets)*num_layers)
        reg_xytl_loss /= (len(targets)*num_layers)
        iou_loss /= (len(targets)*num_layers)

        if seg is not None:
            seg_loss = self.seg_loss(seg,seg_targets.long())
            return dict(cls_loss = cls_loss,reg_xytl_loss = reg_xytl_loss,iou_loss = iou_loss, seg_loss = seg_loss)
        else:
            return dict(cls_loss = cls_loss,reg_xytl_loss = reg_xytl_loss,iou_loss = iou_loss)

    def loss_single_bs(self,predictions:Tensor,target:Tensor,matched_row_inds,matched_col_inds,assign_score):
        """
        Compute the loss of single batch because loss will be NaN if we compute the whole batchsize directly.
        Chinese: 计算单个batch的损失, 因为直接计算整个batchsize会导致loss异常。
        """
        cls_loss = 0
        reg_xytl_loss = 0
        iou_loss = 0
        target = target[target[:, 1] == 1]        

        if len(target) == 0:
            # If there are no targets, all predictions have to be negatives (i.e., 0 confidence)
            cls_target = predictions.new_zeros(predictions.shape[0],).long() #get the label
            cls_pred = predictions[:, :2]
            cls_pred = cls_pred.reshape(predictions.shape[0],cls_pred.shape[-1])
            cls_loss = cls_loss + self.cls_loss(cls_pred, cls_target).sum()
            return cls_loss,reg_xytl_loss,iou_loss
        
        # apply label assignment
        matched_row_inds = matched_row_inds.long()
        matched_col_inds = matched_col_inds.long()
        cls_target,target_yxtl,reg_targets,target_starts, \
            reg_yxtl,reg_pred,predictions_starts = self.get_target_single(predictions,target,matched_row_inds,matched_col_inds,assign_score)
        cls_pred = predictions[:, :2].reshape(-1,2)        

        reg_yxtl[:, 0] *= self.n_strips
        reg_yxtl[:, 1] *= (self.img_w - 1)
        reg_yxtl[:, 2] *= 180
        reg_yxtl[:, 3] *= self.n_strips

        reg_pred *= (self.img_w - 1)

        # assign score
        one_hot_cls = assign_score.clone()
        one_hot_cls[one_hot_cls>0] = 1
        one_hot_cls = one_hot_cls.long()
        new_cls_target = F.one_hot(one_hot_cls, num_classes=cls_pred.shape[-1]).float()
        new_cls_target[matched_row_inds,1]= assign_score[matched_row_inds]
        
        cls_loss = cls_loss + self.cls_loss(cls_pred, new_cls_target,reduction_override='sum') / target.shape[0]

        target_yxtl[:, 0] *= self.n_strips
        target_yxtl[:, 2] *= 180

        reg_xytl_loss = reg_xytl_loss + self.xyt_loss(reg_yxtl, target_yxtl)

        iou_loss = iou_loss + self.iou_loss(reg_pred, reg_targets,self.img_w)
        return cls_loss,reg_xytl_loss,iou_loss

    @torch.no_grad()
    def get_iou_sim(self,predictions:Tensor,target:Tensor,ota_matched_row_inds,ota_matched_col_inds,match_row_inds,is_empty):
        if is_empty:
            return predictions.new_zeros((predictions.shape[0],))
        target = target.clone()
        target = target[target[:, 1] == 1]
        target = target[ota_matched_col_inds,6:]
        
        predictions = predictions.clone()
        reg_pred_line = predictions[ota_matched_row_inds,6:]*self.img_w
        line_iou_ = F.sigmoid(line_iou(reg_pred_line,target,self.img_w))

        return [line_iou_]

    @torch.no_grad()
    def get_assign_score(self,
                         prediction: Tensor,
                         ota_matched_row_inds,
                         match_row_inds,
                         is_last = False,
                         is_empty = False,
                         scale_factor = 1.,
                         line_iou = 1.):
        """
        Assign predicted score for each decoder layer
        Chinese: 为每个layer分配pred score
        """
        assigned_score = prediction.new_zeros(prediction.size(0),dtype=torch.float)
        match_row_inds = match_row_inds.long()
        if is_empty:
            return assigned_score
        if not is_last:
            ota_assigned_pred_score = F.sigmoid(prediction[..., :2])[ota_matched_row_inds, 1]
            max_score_,indice = ota_assigned_pred_score.max(-1)
            assigned_score[ota_matched_row_inds] = line_iou
            mask_assigned_score = assigned_score[match_row_inds].clone()
            ota_assigned_scale_pred_score = (ota_assigned_pred_score/max_score_)*scale_factor
            assigned_score[ota_matched_row_inds] *= ota_assigned_scale_pred_score
            assigned_score[match_row_inds] = mask_assigned_score 
        else:
            assigned_score[match_row_inds] = 1.
            
        return [assigned_score]


    @torch.no_grad()
    def gengeral_label_assign(self,
                     predictions: Tensor,
                     target: Tensor):
        """
        apply label assignment of the general layer

        return: ota_matched_row_inds,ota_matched_col_inds : label assignment for n-1 layers
                match_row_inds,match_col_inds: label assignment for final layer
        """
        target = target[target[:, 1] == 1]
        is_empty = False
        if len(target) == 0:
            is_empty = True
            ota_matched_row_inds = ota_matched_col_inds = match_row_inds = match_col_inds = predictions.new_zeros((predictions.shape[0],))
        else:
            target_assign = target.detach().clone()
            ota_matched_row_inds,ota_matched_col_inds,match_row_inds,match_col_inds = self.assigner.assign(
                predictions, target_assign, self.img_w, self.img_h
            )
        return ota_matched_col_inds,ota_matched_row_inds,match_col_inds,match_row_inds,is_empty

    def get_target_single(self,
                          predictions: Tensor,
                          target: Tensor,
                          matched_row_inds,
                          matched_col_inds,
                          labels):

        reg_yxtl = predictions[matched_row_inds, 2:6]
        target_yxtl = target[matched_col_inds, 2:6].clone()
        reg_pred = predictions[matched_row_inds, 6:]
        reg_targets = target[matched_col_inds, 6:].clone()        

        with torch.no_grad():
            target_starts = (target[matched_col_inds, 2] * self.n_strips).round().long()
            predictions_starts = torch.clamp((predictions[matched_row_inds, 2] * \
                                    self.n_strips).round().long(), 0, self.n_strips)  # ensure the predictions starts is valid
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
        predictions = predictions[0,...]
        decoded = []
        threshold = self.test_cfg.conf_threshold
        scores = F.sigmoid(predictions[..., :2])[..., 1]
        score_topk,score_indice = torch.topk(scores,self.max_lanes,dim=-1)
        predictions = predictions[score_indice]
        keep_inds = score_topk >= threshold # use thres to filter false preditction
        predictions = predictions[keep_inds]

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
