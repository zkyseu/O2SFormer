import glob
import os

import cv2
import numpy as np

from mmcv.utils.logging import get_logger
from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.pipelines import Compose
from mmdet.datasets.builder import DATASETS
from mmcv.parallel import DataContainer as DC
from . import culane_metric
from .test_data_element import DetDataSample
from ..models.utils.general_utils import path_join

logger = get_logger('mmcv')

LIST_FILE = {
    'train': 'list/train_gt.txt',
    'val': 'list/test.txt',
    'test': 'list/test.txt',
} 

CATEGORYS = {
    'normal': 'list/test_split/test0_normal.txt',
    'crowd': 'list/test_split/test1_crowd.txt',
    'hlight': 'list/test_split/test2_hlight.txt',
    'shadow': 'list/test_split/test3_shadow.txt',
    'noline': 'list/test_split/test4_noline.txt',
    'arrow': 'list/test_split/test5_arrow.txt',
    'curve': 'list/test_split/test6_curve.txt',
    'cross': 'list/test_split/test7_cross.txt',
    'night': 'list/test_split/test8_night.txt',
}

@DATASETS.register_module()
class CULaneDataset(CustomDataset):
    CLASSES = ['lane1','lane2','lane3','lane4']
    def __init__(self,
                 data_root, 
                 split,
                 pipeline,
                 cut_height = 160,
                 img_fo = None,
                 resize_img_info = None,
                 test_mode = False,
                 ):

        self.data_root = data_root
        self.list_path = path_join(data_root, LIST_FILE[split])
        self.pipeline = Compose(pipeline)
        self.cut_height = cut_height
        self.ori_img_h,self.ori_img_w = img_fo
        self.test_mode = test_mode
        self.img_h,self.img_w = resize_img_info

        self.parser_datalist()
        self._set_group_flag()

    def parser_datalist(self,):
        self.data_infos = []
        with open(self.list_path) as list_file:
            for line in list_file:
                infos = self.load_annotation(line.split())
                self.data_infos.append(infos)

    def _set_group_flag(self,):
        self.flag = np.zeros(len(self), dtype=np.uint8)
        if self.ori_img_w/self.ori_img_h>1:
            self.flag[:] = 1

    def load_annotation(self, line):
        infos = {}
        img_line = line[0]
        img_line = img_line[1 if img_line[0] == '/' else 0::]
        img_path = os.path.join(self.data_root, img_line)
        infos['img_name'] = img_line 
        infos['img_path'] = img_path
        if len(line) > 1:
            mask_line = line[1]
            mask_line = mask_line[1 if mask_line[0] == '/' else 0::]
            mask_path = os.path.join(self.data_root, mask_line)
            infos['mask_path'] = mask_path

        if len(line) > 2:
            exist_list = [int(l) for l in line[2:]]
            infos['lane_exist'] = np.array(exist_list)

        anno_path = img_path[:-3] + 'lines.txt'  # remove sufix jpg and add lines.txt
        with open(anno_path, 'r') as anno_file:
            data = [list(map(float, line.split())) for line in anno_file.readlines()]
        lanes = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2) if lane[i] >= 0 and lane[i + 1] >= 0]
                 for lane in data]
        lanes = [list(set(lane)) for lane in lanes]  # remove duplicated points
        lanes = [lane for lane in lanes if len(lane) > 3]  # remove lanes with less than 2 points

        lanes = [sorted(lane, key=lambda x: x[1]) for lane in lanes]  # sort by y
        infos['lanes'] = lanes

        return infos

    def __len__(self):
        return len(self.data_infos)

    def prepare_train_img(self, idx):
        #load image
        data_info = self.data_infos[idx]
        img = cv2.imread(data_info['img_path'])
        img = img[self.cut_height:, :, :]

        results = data_info.copy()
        results.update({'img':img})
        #load label
        label = cv2.imread(data_info['mask_path'], cv2.IMREAD_UNCHANGED)
        if len(label.shape) > 2:
            label = label[:, :, 0]
        label = label.squeeze()
        label = label[self.cut_height:, :]
        results.update({'mask': label})
        if self.cut_height != 0:
            new_lanes = []
            for i in data_info['lanes']:
                lanes = []
                for p in i:
                    lanes.append((p[0], p[1] - self.cut_height))
                new_lanes.append(lanes)
            results.update({'lanes': new_lanes})
        
             
        img_infos = {'img_metas':{'full_img_path': data_info['img_path'],
        'img_name': data_info['img_name'],'image_shape':(self.img_h,self.img_w)}}  
        results.update(img_infos)

        return self.pipeline(results)

    def prepare_test_img(self, idx):
        #load image
        data_info = self.data_infos[idx]
        img = cv2.imread(data_info['img_path'])
        img = img[self.cut_height:, :, :]

        results = data_info.copy()
        results.update({'img':img})        
             
        img_infos = {'img_metas':{'full_img_path': data_info['img_path'],
        'img_name': data_info['img_name'],'image_shape':(self.img_h,self.img_w)}}  
        results.update(img_infos)

        return self.pipeline(results)

    def get_prediction_string(self, pred):
        ys = np.arange(270, 590, 8) / self.ori_img_h
        out = []
        for lane in pred:
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * self.ori_img_w
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * self.ori_img_h
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            lane_str = ' '.join([
                '{:.5f} {:.5f}'.format(x, y) for x, y in zip(lane_xs, lane_ys)
            ])
            if lane_str != '':
                out.append(lane_str)

        return '\n'.join(out)

    def evaluate(self, predictions, output_basedir,**kwargs):
        logger.info("Generating evaluation results")
        for idx, pred in enumerate(predictions):
            output_dir = os.path.join(
                output_basedir,
                os.path.dirname(self.data_infos[idx]['img_name']))
            output_filename = os.path.basename(
                self.data_infos[idx]['img_name'])[:-3] + 'lines.txt'
            os.makedirs(output_dir, exist_ok=True)
            output = self.get_prediction_string(pred)

            with open(os.path.join(output_dir, output_filename),
                      'w') as out_file:
                out_file.write(output)

        for cate, cate_file in CATEGORYS.items():
            result = culane_metric.eval_predictions(output_basedir,
                                                    self.data_root,
                                                    os.path.join(self.data_root, cate_file),
                                                    iou_thresholds=[0.5],
                                                    official=True)
            logger.info(f"cate:{cate_file}")
            logger.info(f"result is :{result}")

        result = culane_metric.eval_predictions(output_basedir,
                                                self.data_root,
                                                self.list_path,
                                                iou_thresholds=np.linspace(0.5, 0.95, 10),
                                                official=True)
        logger.info(f"F1 score is {result[0.5]['F1']}")
        return {"mAP":result[0.5]['F1']}
