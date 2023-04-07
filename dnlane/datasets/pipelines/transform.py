import torch
import numpy as np
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.formatting import to_tensor

def LaneTransforms(img_h, img_w):
    return [
        dict(name='Resize',
             parameters=dict(size=dict(height=img_h, width=img_w)),
             p=1.0),
        dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.5),
        dict(name='Affine',
             parameters=dict(translate_percent=dict(x=(-0.1, 0.1),
                                                    y=(-0.1, 0.1)),
                             rotate=(-10, 10),
                             scale=(0.8, 1.2)),
             p=0.7),
        dict(name='Resize',
             parameters=dict(size=dict(height=img_h, width=img_w)),
             p=1.0),
    ]


@PIPELINES.register_module()
class ToTensor_(object):
    """Convert some results to :obj:`torch.Tensor` by given keys.
    Args:
        keys (Sequence[str]): Keys that need to be converted to Tensor.
    """
    def __init__(self, keys=['img', 'mask'], cfg=None):
        self.keys = keys

    def __call__(self, sample):
        data = {}
        if len(sample['img'].shape) < 3:
            sample['img'] = np.expand_dims(sample['img'], -1)
        for key in self.keys:
            if key == 'img_metas' or key == 'gt_masks' or key == 'lane_line' or key=="batch_data_samples":
                data[key] = sample[key]
                continue
            data[key] = to_tensor(sample[key])
        data['img'] = data['img'].permute(2, 0, 1)
        return data

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'