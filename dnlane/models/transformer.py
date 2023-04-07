from typing import Tuple,List
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN,MultiheadAttention
from mmengine.model import ModuleList
from mmengine.model import BaseModule
from .transformer_utils import coordinate_to_encoding,DetrTransformerDecoder,DetrTransformerDecoderLayer
from .utils.general_utils import ConfigType, OptMultiConfig,OptConfigType

def inverse_sigmoid(x: Tensor, eps: float = 1e-6) -> Tensor:
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the inverse.
        eps (float): EPS avoid numerical overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse function of sigmoid, has the same
        shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

class MLP(BaseModule):
    """Very simple multi-layer perceptron (also called FFN) with relu. Mostly
    used in DETR series detectors.
    Args:
        input_dim (int): Feature dim of the input tensor.
        hidden_dim (int): Feature dim of the hidden layer.
        output_dim (int): Feature dim of the output tensor.
        num_layers (int): Number of FFN layers. As the last
            layer of MLP only contains FFN (Linear).
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

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
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

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


class SinePositionalEncoding(BaseModule):
    """Position encoding with sine and cosine functions.
    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None
    """

    def __init__(self,
                 num_feats: int,
                 temperature: int = 10000,
                 normalize: bool = False,
                 scale: float = 2 * math.pi,
                 eps: float = 1e-6,
                 offset: float = 0.,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask: Tensor) -> Tensor:
        """Forward function for `SinePositionalEncoding`.
        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].
        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        # For convenience of exporting to ONNX, it's required to convert
        # `masks` from bool to int.
        mask = mask.to(torch.int)
        not_mask = 1 - mask  # logical_not
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = (y_embed + self.offset) / \
                      (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / \
                      (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        B, H, W = mask.size()
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self) -> str:
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'temperature={self.temperature}, '
        repr_str += f'normalize={self.normalize}, '
        repr_str += f'scale={self.scale}, '
        repr_str += f'eps={self.eps})'
        return repr_str

class DetrTransformerEncoderLayer(BaseModule):
    """Implements encoder layer in DETR transformer.
    Args:
        self_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for self
            attention.
        ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
            normalization layers. All the layers will share the same
            config. Defaults to `LN`.
        init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 self_attn_cfg: OptConfigType = dict(
                     embed_dims=256, num_heads=8, dropout=0.0),
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True)),
                 norm_cfg: OptConfigType = dict(type='LN'),
                 init_cfg: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.self_attn_cfg = self_attn_cfg
        if 'batch_first' not in self.self_attn_cfg:
            self.self_attn_cfg['batch_first'] = True
        else:
            assert self.self_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(2)
        ]
        self.norms = ModuleList(norms_list)

    def forward(self, query: Tensor, query_pos: Tensor,
                key_padding_mask: Tensor, **kwargs) -> Tensor:
        """Forward function of an encoder layer.
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `query`.
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor. has shape (bs, num_queries).
        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            key_padding_mask=key_padding_mask,
            **kwargs)
        query = self.norms[0](query)
        query = self.ffn(query)
        query = self.norms[1](query)

        return query

class DetrTransformerEncoder(BaseModule):
    """Encoder of DETR.
    Args:
        num_layers (int): Number of encoder layers.
        layer_cfg (:obj:`ConfigDict` or dict): the config of each encoder
            layer. All the layers will share the same config.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 num_layers: int,
                 layer_cfg: ConfigType,
                 init_cfg: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)
        self.num_layers = num_layers
        self.layer_cfg = layer_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims

    def forward(self, query: Tensor, query_pos: Tensor,
                key_padding_mask: Tensor, **kwargs) -> Tensor:
        """Forward function of encoder.
        Args:
            query (Tensor): Input queries of encoder, has shape
                (bs, num_queries, dim).
            query_pos (Tensor): The positional embeddings of the queries, has
                shape (bs, num_queries, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (bs, num_queries).
        Returns:
            Tensor: Has shape (bs, num_queries, dim) if `batch_first` is
            `True`, otherwise (num_queries, bs, dim).
        """
        for layer in self.layers:
            query = layer(query, query_pos, key_padding_mask, **kwargs)
        return query

class DABDetrTransformerEncoder(DetrTransformerEncoder):
    """Encoder of DAB-DETR."""

    def _init_layers(self):
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        embed_dims = self.layers[0].embed_dims
        self.embed_dims = embed_dims
        self.query_scale = MLP(embed_dims, embed_dims, embed_dims, 2)

    def forward(self, query: Tensor, query_pos: Tensor,
                key_padding_mask: Tensor, **kwargs):
        """Forward function of encoder.
        Args:
            query (Tensor): Input queries of encoder, has shape
                (bs, num_queries, dim).
            query_pos (Tensor): The positional embeddings of the queries, has
                shape (bs, num_feat_points, dim).
            key_padding_mask (Tensor): ByteTensor, the key padding mask
                of the queries, has shape (bs, num_feat_points).
        Returns:
            Tensor: With shape (num_queries, bs, dim).
        """

        for layer in self.layers:
            pos_scales = self.query_scale(query)
            query = layer(
                query,
                query_pos=query_pos * pos_scales,
                key_padding_mask=key_padding_mask,
                **kwargs)

        return query

class ConditionalAttention(BaseModule):
    """A wrapper of conditional attention, dropout and residual connection.
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop: A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        cross_attn (bool): Whether the attention module is for cross attention.
            Default: False
        keep_query_pos (bool): Whether to transform query_pos before cross
            attention.
            Default: False.
        batch_first (bool): When it is True, Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default: True.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 cross_attn: bool = False,
                 keep_query_pos: bool = False,
                 batch_first: bool = True,
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)

        assert batch_first is True, 'Set `batch_first`\
        to False is NOT supported in ConditionalAttention. \
        First dimension of all DETRs in mmdet is `batch`, \
        please set `batch_first` to True.'

        self.cross_attn = cross_attn
        self.keep_query_pos = keep_query_pos
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self._init_layers()

    def _init_layers(self):
        """Initialize layers for qkv projection."""
        embed_dims = self.embed_dims
        self.qcontent_proj = nn.Linear(embed_dims, embed_dims)
        self.qpos_proj = nn.Linear(embed_dims, embed_dims)
        self.kcontent_proj = nn.Linear(embed_dims, embed_dims)
        self.kpos_proj = nn.Linear(embed_dims, embed_dims)
        self.v_proj = nn.Linear(embed_dims, embed_dims)
        if self.cross_attn:
            self.qpos_sine_proj = nn.Linear(embed_dims, embed_dims)
        self.out_proj = nn.Linear(embed_dims, embed_dims)

        nn.init.constant_(self.out_proj.bias, 0.)

    def forward_attn(self,
                     query: Tensor,
                     key: Tensor,
                     value: Tensor,
                     attn_mask: Tensor = None,
                     key_padding_mask: Tensor = None) -> Tuple[Tensor]:
        """Forward process for `ConditionalAttention`.
        Args:
            query (Tensor): The input query with shape [bs, num_queries,
                embed_dims].
            key (Tensor): The key tensor with shape [bs, num_keys,
                embed_dims].
                If None, the `query` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.
        Returns:
            Tuple[Tensor]: Attention outputs of shape :math:`(N, L, E)`,
            where :math:`N` is the batch size, :math:`L` is the target
            sequence length , and :math:`E` is the embedding dimension
            `embed_dim`. Attention weights per head of shape :math:`
            (num_heads, L, S)`. where :math:`N` is batch size, :math:`L`
            is target sequence length, and :math:`S` is the source sequence
            length.
        """
        assert key.size(1) == value.size(1), \
            f'{"key, value must have the same sequence length"}'
        assert query.size(0) == key.size(0) == value.size(0), \
            f'{"batch size must be equal for query, key, value"}'
        assert query.size(2) == key.size(2), \
            f'{"q_dims, k_dims must be equal"}'
        assert value.size(2) == self.embed_dims, \
            f'{"v_dims must be equal to embed_dims"}'

        bs, tgt_len, hidden_dims = query.size()
        _, src_len, _ = key.size()
        head_dims = hidden_dims // self.num_heads
        v_head_dims = self.embed_dims // self.num_heads
        assert head_dims * self.num_heads == hidden_dims, \
            f'{"hidden_dims must be divisible by num_heads"}'
        scaling = float(head_dims)**-0.5

        q = query * scaling
        k = key
        v = value

        if attn_mask is not None:
            assert attn_mask.dtype == torch.float32 or \
                   attn_mask.dtype == torch.float64 or \
                   attn_mask.dtype == torch.float16 or \
                   attn_mask.dtype == torch.uint8 or \
                   attn_mask.dtype == torch.bool, \
                   'Only float, byte, and bool types are supported for \
                    attn_mask'

            if attn_mask.dtype == torch.uint8:
                warnings.warn('Byte tensor for attn_mask is deprecated.\
                     Use bool tensor instead.')
                attn_mask = attn_mask.to(torch.bool)
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(1), key.size(1)]:
                    raise RuntimeError(
                        'The size of the 2D attn_mask is not correct.')
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [
                        bs * self.num_heads,
                        query.size(1),
                        key.size(1)
                ]:
                    raise RuntimeError(
                        'The size of the 3D attn_mask is not correct.')
            else:
                raise RuntimeError(
                    "attn_mask's dimension {} is not supported".format(
                        attn_mask.dim()))
        # attn_mask's dim is 3 now.

        if key_padding_mask is not None and key_padding_mask.dtype == int:
            key_padding_mask = key_padding_mask.to(torch.bool)

        q = q.contiguous().view(bs, tgt_len, self.num_heads,
                                head_dims).permute(0, 2, 1, 3).flatten(0, 1)
        if k is not None:
            k = k.contiguous().view(bs, src_len, self.num_heads,
                                    head_dims).permute(0, 2, 1,
                                                       3).flatten(0, 1)
        if v is not None:
            v = v.contiguous().view(bs, src_len, self.num_heads,
                                    v_head_dims).permute(0, 2, 1,
                                                         3).flatten(0, 1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bs
            assert key_padding_mask.size(1) == src_len

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [
            bs * self.num_heads, tgt_len, src_len
        ]

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(
                bs, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(
                bs * self.num_heads, tgt_len, src_len)

        attn_output_weights = F.softmax(
            attn_output_weights -
            attn_output_weights.max(dim=-1, keepdim=True)[0],
            dim=-1)
        attn_output_weights = self.attn_drop(attn_output_weights)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(
            attn_output.size()) == [bs * self.num_heads, tgt_len, v_head_dims]
        attn_output = attn_output.view(bs, self.num_heads, tgt_len,
                                       v_head_dims).permute(0, 2, 1,
                                                            3).flatten(2)
        attn_output = self.out_proj(attn_output)

        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bs, self.num_heads,
                                                       tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / self.num_heads

    def forward(self,
                query: Tensor,
                key: Tensor,
                query_pos: Tensor = None,
                ref_sine_embed: Tensor = None,
                key_pos: Tensor = None,
                attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                is_first: bool = False) -> Tensor:
        """Forward function for `ConditionalAttention`.
        Args:
            query (Tensor): The input query with shape [bs, num_queries,
                embed_dims].
            key (Tensor): The key tensor with shape [bs, num_keys,
                embed_dims].
                If None, the `query` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query in self
                attention, with the same shape as `x`. If not None, it will
                be added to `x` before forward function.
                Defaults to None.
            query_sine_embed (Tensor): The positional encoding for query in
                cross attention, with the same shape as `x`. If not None, it
                will be added to `x` before forward function.
                Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.
            is_first (bool): A indicator to tell whether the current layer
                is the first layer of the decoder.
                Defaults to False.
        Returns:
            Tensor: forwarded results with shape
            [bs, num_queries, embed_dims].
        """

        if self.cross_attn:
            q_content = self.qcontent_proj(query)
            k_content = self.kcontent_proj(key)
            v = self.v_proj(key)

            bs, nq, c = q_content.size()
            _, hw, _ = k_content.size()

            k_pos = self.kpos_proj(key_pos)
            if is_first or self.keep_query_pos:
                q_pos = self.qpos_proj(query_pos)
                q = q_content + q_pos
                k = k_content + k_pos
            else:
                q = q_content
                k = k_content
            q = q.view(bs, nq, self.num_heads, c // self.num_heads)
            query_sine_embed = self.qpos_sine_proj(ref_sine_embed)
            query_sine_embed = query_sine_embed.view(bs, nq, self.num_heads,
                                                     c // self.num_heads)
            q = torch.cat([q, query_sine_embed], dim=3).view(bs, nq, 2 * c)
            k = k.view(bs, hw, self.num_heads, c // self.num_heads)
            k_pos = k_pos.view(bs, hw, self.num_heads, c // self.num_heads)
            k = torch.cat([k, k_pos], dim=3).view(bs, hw, 2 * c)
            ca_output = self.forward_attn(
                query=q,
                key=k,
                value=v,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask)[0]
            query = query + self.proj_drop(ca_output)
        else:
            q_content = self.qcontent_proj(query)
            q_pos = self.qpos_proj(query_pos)
            k_content = self.kcontent_proj(query)
            k_pos = self.kpos_proj(query_pos)
            v = self.v_proj(query)
            q = q_content if q_pos is None else q_content + q_pos
            k = k_content if k_pos is None else k_content + k_pos
            sa_output = self.forward_attn(
                query=q,
                key=k,
                value=v,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask)[0]
            query = query + self.proj_drop(sa_output)

        return query


class DABDetrTransformerDecoderLayer(DetrTransformerDecoderLayer):
    """Implements decoder layer in DAB-DETR transformer."""

    def _init_layers(self):
        """Initialize self-attention, cross-attention, FFN, normalization and
        others."""
        self.self_attn = ConditionalAttention(**self.self_attn_cfg)
        self.cross_attn = ConditionalAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)
        self.keep_query_pos = self.cross_attn.keep_query_pos

    def forward(self,
                query: Tensor,
                key: Tensor,
                query_pos: Tensor,
                key_pos: Tensor,
                ref_sine_embed: Tensor = None,
                self_attn_masks: Tensor = None,
                cross_attn_masks: Tensor = None,
                key_padding_mask: Tensor = None,
                is_first: bool = False,
                **kwargs) -> Tensor:
        """
        Args:
            query (Tensor): The input query with shape [bs, num_queries,
                dim].
            key (Tensor): The key tensor with shape [bs, num_keys,
                dim].
            query_pos (Tensor): The positional encoding for query in self
                attention, with the same shape as `x`.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`.
            ref_sine_embed (Tensor): The positional encoding for query in
                cross attention, with the same shape as `x`.
                Defaults to None.
            self_attn_masks (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_masks (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.
            is_first (bool): A indicator to tell whether the current layer
                is the first layer of the decoder.
                Defaults to False.
        Returns:
            Tensor: forwarded results with shape
            [bs, num_queries, dim].
        """

        query = self.self_attn(
            query=query,
            key=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_masks,
            **kwargs)
        query = self.norms[0](query)
        query = self.cross_attn(
            query=query,
            key=key,
            query_pos=query_pos,
            key_pos=key_pos,
            ref_sine_embed=ref_sine_embed,
            attn_mask=cross_attn_masks,
            key_padding_mask=key_padding_mask,
            is_first=is_first,
            **kwargs)
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)

        return query

class DABDetrTransformerDecoder(DetrTransformerDecoder):
    """Decoder of DAB-DETR.
    Args:
        query_dim (int): The last dimension of query pos,
            4 for anchor format, 2 for point format.
            Defaults to 4.
        query_scale_type (str): Type of transformation applied
            to content query. Defaults to `cond_elewise`.
        with_modulated_hw_attn (bool): Whether to inject h&w info
            during cross conditional attention. Defaults to True.
    """

    def __init__(self,
                 *args,
                 query_dim: int = 4,
                 query_scale_type: str = 'cond_elewise',
                 with_modulated_hw_attn: bool = True,
                 num_points = 72,
                 **kwargs):

        self.query_dim = query_dim
        self.query_scale_type = query_scale_type
        self.with_modulated_hw_attn = with_modulated_hw_attn
        self.num_points = num_points

        super().__init__(*args, **kwargs)

    def _init_layers(self):
        """Initialize decoder layers and other layers."""
        assert self.query_dim in [2, 3, 4], \
            f'{"dab-detr only supports anchor prior or reference point prior"}'
        assert self.query_scale_type in [
            'cond_elewise', 'cond_scalar', 'fix_elewise'
        ]

        self.layers = ModuleList([
            DABDetrTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])

        embed_dims = self.layers[0].embed_dims
        self.embed_dims = embed_dims

        self.post_norm = build_norm_layer(self.post_norm_cfg, embed_dims)[1]
        if self.query_scale_type == 'cond_elewise':
            self.query_scale = MLP(embed_dims, embed_dims, embed_dims, 2)
        elif self.query_scale_type == 'cond_scalar':
            self.query_scale = MLP(embed_dims, embed_dims, 1, 2)
        elif self.query_scale_type == 'fix_elewise':
            self.query_scale = nn.Embedding(self.num_layers, embed_dims)
        else:
            raise NotImplementedError('Unknown query_scale_type: {}'.format(
                self.query_scale_type))

        self.ref_point_head = MLP((self.query_dim + 1) * (embed_dims // 2), embed_dims,
                                  embed_dims, 2)

        if self.with_modulated_hw_attn and self.query_dim == 4:
            self.ref_anchor_head = MLP(embed_dims, embed_dims, 2, 2)

        self.keep_query_pos = self.layers[0].keep_query_pos
        if not self.keep_query_pos:
            for layer_id in range(self.num_layers - 1):
                self.layers[layer_id + 1].cross_attn.qpos_proj = None
        
        self.offset_map = MLP(self.num_points,(self.num_points+1)//2,1,2)

    def forward(self,
                query: Tensor,
                key: Tensor,
                query_pos: Tensor,
                key_pos: Tensor,
                reg_branches: nn.Module,
                offset_branches:nn.Module,
                offset_points:Tensor,
                key_padding_mask: Tensor = None,
                **kwargs) -> List[Tensor]:
        """Forward function of decoder.
        Args:
            query (Tensor): The input query with shape (bs, num_queries, dim).
            key (Tensor): The input key with shape (bs, num_keys, dim).
            query_pos (Tensor): The positional encoding for `query`, with the
                same shape as `query`.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`.
            reg_branches (nn.Module): The regression branch for dynamically
                updating references in each layer.
            key_padding_mask (Tensor): ByteTensor with shape (bs, num_keys).
                Defaults to `None`.
        Returns:
            List[Tensor]: forwarded results with shape (num_decoder_layers,
            bs, num_queries, dim) if `return_intermediate` is True, otherwise
            with shape (1, bs, num_queries, dim). references with shape
            (num_decoder_layers, bs, num_queries, 2/4).
        """
        output = query
        unsigmoid_references = query_pos.clone()
        reference_points = unsigmoid_references
        offset_map = self.offset_map(offset_points)
        query_pos_wo = torch.cat([query_pos,offset_map],dim=-1)
        
        intermediate_reference_points = [reference_points]
        intermediate_offest_points = [offset_points]
#        intermediate_reference_points = []
        intermediate = []
        for layer_id, layer in enumerate(self.layers):
            obj_center = query_pos_wo
            ref_sine_embed = coordinate_to_encoding(
                coord_tensor=obj_center, num_feats=self.embed_dims // 2)
            query_pos = self.ref_point_head(
                ref_sine_embed)  # [bs, nq, 2c] -> [bs, nq, c]
            # For the first decoder layer, do not apply transformation
            if self.query_scale_type != 'fix_elewise':
                if layer_id == 0:
                    pos_transformation = 1
                else:
                    pos_transformation = self.query_scale(output)
            else:
                pos_transformation = self.query_scale.weight[layer_id]
            # apply transformation
            ref_sine_embed = ref_sine_embed[
                ..., :self.embed_dims] * pos_transformation

            output = layer(
                output,
                key,
                query_pos=query_pos,
                ref_sine_embed=ref_sine_embed,
                key_pos=key_pos,
                key_padding_mask=key_padding_mask,
                is_first=(layer_id == 0),
                **kwargs)
            # iter update
            tmp_reg_preds = reg_branches(output)
            tmp_reg_preds[..., :self.query_dim] += reference_points
            new_reference_points = tmp_reg_preds[
                ..., :self.query_dim]
            if layer_id != self.num_layers - 1:
                intermediate_reference_points.append(new_reference_points)
            query_pos_wo[...,:self.query_dim] = new_reference_points.detach()

            # iter update offest
            tmp_offset_points = offset_branches(output)
            tmp_offset_points += offset_points
            if layer_id != self.num_layers - 1:
                intermediate_offest_points.append(tmp_offset_points)
            offset_points = tmp_offset_points.detach()
            query_pos_wo[...,self.query_dim:] = self.offset_map(tmp_offset_points) # map offset to an query


            if self.return_intermediate:
                intermediate.append(self.post_norm(output))

        output = self.post_norm(output)

        if self.return_intermediate:
            return [
                torch.stack(intermediate),
                torch.stack(intermediate_reference_points),
                torch.stack(intermediate_offest_points)
            ]
        else:
            return [
                output.unsqueeze(0),
                torch.stack(intermediate_reference_points),
                torch.stack(intermediate_offest_points)
            ]
