import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule,ModuleList
from mmcv.cnn import build_conv_layer, build_norm_layer, build_activation_layer
from mmcls.models.backbones.repvgg import RepVGGBlock
from mmdet.models.builder import NECKS

from .transformer import DetrTransformerEncoderLayer

def linear_init_(module):
    bound = 1 / math.sqrt(module.weight.shape[0])
    nn.init.uniform_(module.weight, -bound, bound)
    nn.init.uniform_(module.bias, -bound, bound)

class BaseConv(BaseModule):
    def __init__(self,                 
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 groups=1,
                 bias=False,
                 conv_cfg = None,
                 norm_cfg = None,
                 act = None):
        super().__init__()

        self.conv = build_conv_layer(conv_cfg,in_channels,out_channels,kernel_size = ksize,padding=(ksize - 1) // 2,groups = groups,stride = stride,bias = bias)
        self.norm= build_norm_layer(norm_cfg,out_channels)[1]

    def forward(self,x):
        x = self.norm(self.conv(x))
        y = x * F.sigmoid(x)
        return y        

class CSPRepLayer(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=False,
                 act=dict(type='SiLU'),
                 conv_cfg=None,
                 norm_cfg=None):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act, norm_cfg=norm_cfg ,conv_cfg=conv_cfg)
        self.conv2 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act, norm_cfg=norm_cfg ,conv_cfg=conv_cfg)
        self.bottlenecks = nn.Sequential(*[
            RepVGGBlock(
                hidden_channels, hidden_channels, act_cfg=act)
            for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = BaseConv(
                hidden_channels,
                out_channels,
                ksize=1,
                stride=1,
                bias=bias,
                act=act,
                norm_cfg=norm_cfg)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)

@NECKS.register_module()
class HybridEncoder(BaseModule):
    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 embed_dims = 256,
                 use_encoder_idx=[2],
                 num_levels=3,
                 num_encoder_layers=1,
                 encoder_layer=None,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act=dict(type='SiLU'),
                 in_dice=(1,2,3),
                 eval_size=None,
                 conv_cfg = None,
                 norm_cfg = None,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_size = eval_size
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.num_levels = num_levels
        self.in_dice = in_dice
        self.embed_dims = embed_dims

        # channel projection
        self.input_proj = ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    build_conv_layer(conv_cfg,in_channel, hidden_dim,kernel_size=1,bias = False),
                    build_norm_layer(norm_cfg,hidden_dim)[1]))
        # encoder transformer
        self.encoder = ModuleList([
            DetrTransformerEncoderLayer(**encoder_layer)
            for _ in range(len(use_encoder_idx))
        ])

        # top-down fpn
        self.lateral_convs = ModuleList()
        self.fpn_blocks = ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(
                BaseConv(
                    hidden_dim, hidden_dim, 1, 1, act=act,conv_cfg=conv_cfg,norm_cfg=norm_cfg,))
            self.fpn_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act=act,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    expansion=expansion))

        # bottom-up pan
        self.downsample_convs = ModuleList()
        self.pan_blocks = ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsample_convs.append(
                BaseConv(
                    hidden_dim, hidden_dim, 3, stride=2, act=act, conv_cfg=conv_cfg, norm_cfg=norm_cfg))
            self.pan_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act=act,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,                    
                    expansion=expansion))
        self.fuse_layer = build_conv_layer(conv_cfg,len(in_channels)*hidden_dim, hidden_dim,kernel_size=1)
        self.down_sample_fuse_conv = BaseConv(
                    hidden_dim, hidden_dim, 3, stride=2, act=act, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        self.down_sample_fuse_conv_2 = BaseConv(
                    hidden_dim, hidden_dim, 3, stride=2, act=act, conv_cfg=conv_cfg, norm_cfg=norm_cfg)

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_size[1] // stride, self.eval_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)

    def _get_encoder_input(self, feats):
        # get encoder inputs
        feat_flatten = []
        _,_,h,w = feats[1].shape
        for i, feat in enumerate(feats):
            if i == 0:
                feat = self.down_sample_fuse_conv(feat)
            if i == len(feats)-1:
                feat = F.interpolate(feat,size=(h,w),mode="nearest")   
            feat_flatten.append(feat)
#            _, _, h, w = feat.shape
#            # [b, c, h, w] -> [b, h*w, c]
#            feat_flatten.append(feat.flatten(2).permute([0, 2, 1]))

        # [b, l, c]
        feat_flatten = torch.cat(feat_flatten, 1)
        feat_flatten = self.fuse_layer(feat_flatten)
        feat_flatten = self.down_sample_fuse_conv_2(feat_flatten).flatten(2).permute([0, 2, 1])
#        feat_flatten = self.fuse_layer(feat_flatten).flatten(2).permute([0, 2, 1]
        return feat_flatten

    @staticmethod
    def build_2d_sincos_position_embedding(w,
                                           h,
                                           embed_dim=256,
                                           temperature=10000.):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @omega[None]
        out_h = grid_h.flatten()[..., None] @omega[None]

        return torch.cat(
            [
                torch.sin(out_w), torch.cos(out_w), torch.sin(out_h),
                torch.cos(out_h)
            ],
            dim=1)[None, :, :].cuda()

    def forward(self, feats, key_padding_mask=None):
        assert len(feats) == len(self.in_channels)
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(
                    [0, 2, 1])
                if self.training or self.eval_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None)
                memory = self.encoder[i](src_flatten, query_pos=pos_embed, key_padding_mask = key_padding_mask)
                proj_feats[enc_ind] = memory.permute([0, 2, 1]).reshape(
                    [-1, self.hidden_dim, h, w])

        # top-down fpn
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](
                feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = F.interpolate(
                feat_heigh, scale_factor=2., mode="nearest")
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                torch.cat(
                    [upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        # bottom-up pan
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.cat(
                [downsample_feat, feat_height], dim=1))
            outs.append(out)
        
        output = self._get_encoder_input(outs)

        return output