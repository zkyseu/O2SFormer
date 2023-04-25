# This file is mainly modified from DAB-DETR in mmdetection
from typing import Tuple,Dict
import cv2
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import mmcv
from mmdet.models.detectors import BaseDetector
from mmdet.models import build_backbone,build_head,build_neck
from mmdet.models.builder import MODELS
from mmengine.model import uniform_init

from .transformer import (DABDetrTransformerDecoder,SinePositionalEncoding,
                                  DABDetrTransformerEncoder, inverse_sigmoid)
from .utils.general_utils import COLORS

@MODELS.register_module()
class DNLATR(BaseDetector):
    def __init__(self,
                 backbone,
                 neck,
                 head,
                 encoder = None,
                 decoder = None,
                 num_queries = None,
                 offset_dim = None,
                 positional_encoding = None,
                 pretrain = None,
                 with_random_refpoints = False,
                 num_patterns = 0,
                 train_cfg = None,
                 test_cfg = None,
                 left_prio = 1,   
                 sample_y = range(589, 270, -8),              
                 **kwargs
                 ):
        super().__init__()

        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.bbox_head = build_head(head)

        self.encoder = encoder
        self.decoder = decoder
        self.positional_encoding = positional_encoding
        self.num_queries = num_queries
        self.num_patterns = num_patterns
        self.with_random_refpoints = with_random_refpoints
        self.left_prio = left_prio
        self.sample_y = sample_y

        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = DABDetrTransformerEncoder(**self.encoder)
        self.decoder = DABDetrTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_dim = self.decoder.query_dim
        self.n_offsets = self.bbox_head.n_offsets
        self.query_embedding = nn.Embedding(self.num_queries, self.query_dim)
        if self.num_patterns > 0:
            self.patterns = nn.Embedding(self.num_patterns, self.embed_dims)

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'
            

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights()
    
    def init_weights(self) -> None:
        """Initialize weights for query."""
        super(BaseDetector, self).init_weights()
        left_priors_nums = self.left_prio
        bottom_priors_nums = self.num_queries - 2*left_priors_nums
        assert bottom_priors_nums >0,"bottom_priors_nums should be greater than 0"
        strip_size = 0.5 / (left_priors_nums // 2 - 1)
        bottom_strip_size = 1 / (bottom_priors_nums // 4 + 1)
        for i in range(left_priors_nums):
            nn.init.constant_(self.query_embedding.weight[i, 0],
                              (i // 2) * strip_size)
            nn.init.constant_(self.query_embedding.weight[i, 1], 0.)
            nn.init.constant_(self.query_embedding.weight[i, 2],
                              0.16 if i % 2 == 0 else 0.32)
            
        for i in range(left_priors_nums,
                       left_priors_nums + bottom_priors_nums):
            nn.init.constant_(self.query_embedding.weight[i, 0], 0.)
            nn.init.constant_(self.query_embedding.weight[i, 1],
                              ((i - left_priors_nums) // 4 + 1) *
                              bottom_strip_size)
            nn.init.constant_(self.query_embedding.weight[i, 2],
                              0.2 * (i % 4 + 1))

        for i in range(left_priors_nums + bottom_priors_nums, self.num_queries):
            nn.init.constant_(
                self.query_embedding.weight[i, 0],
                ((i - left_priors_nums - bottom_priors_nums) // 2) *
                strip_size)
            nn.init.constant_(self.query_embedding.weight[i, 1], 1.)
            nn.init.constant_(self.query_embedding.weight[i, 2],
                              0.68 if i % 2 == 0 else 0.84)

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.
        Args:
            batch_inputs (Tensor): Image tensor, has shape (bs, dim, H, W).
        Returns:
            tuple[Tensor]: Tuple of feature maps from neck. Each feature map
            has shape (bs, dim, H, W).
        """
        x = self.backbone(batch_inputs)
        neck_input = [x[-1]]
        output = self.neck(neck_input)
        return output,x

    def pre_transformer(
            self,
            img_feats: Tuple[Tensor],
            img_metas) -> Tuple[Dict, Dict]:
        """Prepare the inputs of the Transformer.
        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.
        Args:
            img_feats (Tuple[Tensor]): Tuple of features output from the neck,
                has shape (bs, c, h, w).
        Returns:
            tuple[dict, dict]: The first dict contains the inputs of encoder
            and the second dict contains the inputs of decoder.
            - encoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_encoder()`, which includes 'feat', 'feat_mask',
              and 'feat_pos'.
            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'memory_mask',
              and 'memory_pos'.
        """

        feat = img_feats[-1]  # NOTE img_feats contains only one feature.
        batch_size, feat_dim, _, _ = feat.shape
        # construct binary masks which for the transformer.

        batch_input_shape = img_metas[0]['img_metas']['image_shape']
        img_shape_list = [sample['img_metas']['image_shape'] for sample in img_metas]

        input_img_h, input_img_w = batch_input_shape
        masks = feat.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w = img_shape_list[img_id]
            masks[img_id, :img_h, :img_w] = 0
        # NOTE following the official DETR repo, non-zero values represent
        # ignored positions, while zero values mean valid positions.

        masks = F.interpolate(
            masks.unsqueeze(1), size=feat.shape[-2:]).to(torch.bool).squeeze(1)
        # [batch_size, embed_dim, h, w]
        pos_embed = self.positional_encoding(masks)

        # use `view` instead of `flatten` for dynamically exporting to ONNX
        # [bs, c, h, w] -> [bs, h*w, c]
        feat = feat.view(batch_size, feat_dim, -1).permute(0, 2, 1)
        pos_embed = pos_embed.view(batch_size, feat_dim, -1).permute(0, 2, 1)
        # [bs, h, w] -> [bs, h*w]
        masks = masks.view(batch_size, -1)

        # prepare transformer_inputs_dict
        encoder_inputs_dict = dict(
            feat=feat, feat_mask=masks, feat_pos=pos_embed)
        decoder_inputs_dict = dict(memory_mask=masks, memory_pos=pos_embed)
        return encoder_inputs_dict, decoder_inputs_dict
    
    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor) -> Dict:
        """Forward with Transformer encoder.
        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.
        Args:
            feat (Tensor): Sequential features, has shape (bs, num_feat_points,
                dim).
            feat_mask (Tensor): ByteTensor, the padding mask of the features,
                has shape (bs, num_feat_points).
            feat_pos (Tensor): The positional embeddings of the features, has
                shape (bs, num_feat_points, dim).
        Returns:
            dict: The dictionary of encoder outputs, which includes the
            `memory` of the encoder output.
        """
        memory = self.encoder(
            query=feat, query_pos=feat_pos,
            key_padding_mask=feat_mask)  # for self_attn
        encoder_outputs_dict = dict(memory=memory)
        return encoder_outputs_dict
    
    def pre_decoder(self, memory: Tensor) -> Tuple[Dict, Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`.
        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
        Returns:
            tuple[dict, dict]: The first dict contains the inputs of decoder
            and the second dict contains the inputs of the bbox_head function.
            - decoder_inputs_dict (dict): The keyword args dictionary of
                `self.forward_decoder()`, which includes 'query', 'query_pos',
                'memory' and 'reg_branches'.
            - head_inputs_dict (dict): The keyword args dictionary of the
                bbox_head functions, which is usually empty, or includes
                `enc_outputs_class` and `enc_outputs_class` when the detector
                support 'two stage' or 'query selection' strategies.
        """
        batch_size = memory.size(0)
        query_pos = self.query_embedding.weight
        query_pos = query_pos.unsqueeze(0).repeat(batch_size, 1, 1)
        if self.num_patterns == 0:
            query = query_pos.new_zeros(batch_size, self.num_queries,
                                        self.embed_dims)
        else:
            query = self.patterns.weight[:, None, None, :]\
                .repeat(1, self.num_queries, batch_size, 1)\
                .view(-1, batch_size, self.embed_dims)\
                .permute(1, 0, 2)
            query_pos = query_pos.repeat(1, self.num_patterns, 1)
        offset_points = query.new_zeros((batch_size, self.num_queries, self.n_offsets),device=query.device)
        
        decoder_inputs_dict = dict(
            query_pos=query_pos, query=query, memory=memory,offset_points=offset_points)
        head_inputs_dict = dict()
        return decoder_inputs_dict, head_inputs_dict
    
    def forward_decoder(self, query: Tensor, query_pos: Tensor, memory: Tensor,
                        memory_mask: Tensor, memory_pos: Tensor, offset_points: Tensor) -> Dict:
        """Forward with Transformer decoder.
        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries, dim).
            query_pos (Tensor): The positional queries of decoder inputs,
                has shape (bs, num_queries, dim).
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            memory_pos (Tensor): The positional embeddings of memory, has
                shape (bs, num_feat_points, dim).
        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` and `references` of the decoder output.
        """

        offset_branch = nn.Sequential(self.bbox_head.feat_layer,self.bbox_head.reg_layers)
        hidden_states, references, offset_points = self.decoder(
            query=query,
            key=memory,
            query_pos=query_pos,
            key_pos=memory_pos,
            key_padding_mask=memory_mask,
            offset_points = offset_points,
            reg_branches=self.bbox_head.fc_reg,  # iterative refinement for anchor boxes,
            offset_branches=offset_branch   # iterative refinement for offset,
        )
        head_inputs_dict = dict(
            hidden_states=hidden_states, references=references, offset_points = offset_points)
        return head_inputs_dict
    
    def forward_transformer(self,
                            img_feats: Tuple[Tensor],
                            img_metas,
                            batch_feature : Tuple[Tensor] = None,) -> Dict:
        """Forward process of Transformer, which includes four steps:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'. We
        summarized the parameters flow of the existing DETR-like detector,
        which can be illustrated as follow:
        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                    feature map has shape (bs, dim, H, W).
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.
        Returns:
            dict: The dictionary of bbox_head function inputs, which always
            includes the `hidden_states` of the decoder output and may contain
            `references` including the initial and intermediate references.
        """
        batch_feature = list(batch_feature)
        seg_feature = torch.cat([
                F.interpolate(feature,
                              size=[
                                  batch_feature[0].shape[2],
                                  batch_feature[0].shape[3]
                              ],
                              mode='bilinear',
                              align_corners=False)
                for feature in batch_feature
            ],dim=1)        
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, img_metas)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(**encoder_outputs_dict)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        head_inputs_dict.update({"seg_feature":seg_feature})
        head_inputs_dict.update({"img_metas":img_metas})
        return head_inputs_dict

    def forward_train(self, img, img_metas = None, **kwargs):
        targets = kwargs['lane_line']
        seg_targets = kwargs['seg']
        img_feats,batch_feature = self.extract_feat(img)
        head_inputs_dict = self.forward_transformer(img_feats,img_metas,batch_feature)
        head_out = self.bbox_head(**head_inputs_dict)
        head_out.update({"targets":targets,"seg_targets":seg_targets})
        loss = self.bbox_head.loss(**head_out)
        return loss

    def train_step(self, data, optimizer):
        losses = self.forward_train(**data)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))
        return outputs

    def predict(self,img, img_metas = None,**kwargs):
        img = img.cuda()
        img_metas = img_metas.data[0]
        img_feats,batch_feature = self.extract_feat(img)
        head_inputs_dict = self.forward_transformer(img_feats,img_metas,batch_feature)   
        out_head = self.bbox_head(**head_inputs_dict)
        output = self.bbox_head.get_lanes(out_head)
        return output

    def forward_dummy(self, img):
        """Used for computing network flops.
        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        shape = img.shape[2:]
        img_metas = [{"img_metas":{"image_shape":shape}}]
        img_feats,batch_feature = self.extract_feat(img)
        head_inputs_dict = self.forward_transformer(img_feats,img_metas,batch_feature)  
        out_head = self.bbox_head(**head_inputs_dict)
#        output = self.bbox_head.get_lanes(out_head)
        return out_head

    def show_result(self,img, lanes, show=False, out_file=None, width=4):
        """
        Draw detection lane over image
        """
        img = mmcv.imread(img)
        lanes = lanes[0]
        lanes = [lane.to_array(self.sample_y,self.bbox_head.ori_img_w,self.bbox_head.ori_img_h) for lane in lanes]
        lanes_xys = []
        for _, lane in enumerate(lanes):
            xys = []
            for x, y in lane:
                if x <= 0 or y <= 0:
                    continue
                x, y = int(x), int(y)
                xys.append((x, y))
            lanes_xys.append(xys)
        lanes_xys = [xys for xys in lanes_xys if xys!=[]]

        for idx, xys in enumerate(lanes_xys):
            for i in range(1, len(xys)):
                cv2.line(img, xys[i - 1], xys[i], COLORS[idx], thickness=width)

        if show:
            cv2.imshow('view', img)
            cv2.waitKey(0)

        if out_file:
            cv2.imwrite(out_file, img)
    
    def simple_test(self, img, img_metas, **kwargs):
        return super().simple_test(img, img_metas, **kwargs)
    
    def aug_test(self, imgs, img_metas, **kwargs):
        return super().aug_test(imgs, img_metas, **kwargs)
