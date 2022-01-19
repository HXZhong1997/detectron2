# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import os,json, logging
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union
from fvcore.common.timer import Timer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager


__all__ = ["get_change_pairs_dict"]

logger = logging.getLogger(__name__)

def get_change_pairs_dict(json_name):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        json_name: json path to change pairs
    """
    timer = Timer()
    with open(json_name) as f:
        dicts = json.load(f)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_name, timer.seconds()))
    dicts_new = []

    for i in range(len(dicts['before'])):

        tmp_before = {
            'image': dicts['before'][i].pop('image'),
            'annotations': [
                {'bbox': dicts['before'][i].pop('bbox')},
                {'bbox_mode':BoxMode.XYWH_ABS},
            ],
        }
        tmp_after = {
            'image': dicts['after'][i].pop('image'),
            'annotations': [
                {'bbox': dicts['after'][i].pop('bbox')},
                {'bbox_mode':BoxMode.XYWH_ABS},
            ],
        }
        dicts_new.append({'before': tmp_before, 'after': tmp_after})
        # dicts['before'][i]['annotations'] = [
        #     {'bbox': dicts['before'][i].pop('bbox')},
        #     {'bbox_mode':BoxMode.XYWH_ABS},
        # ]
        # dicts['after'][i]['annotations'] = [
        #     {'bbox': dicts['after'][i].pop('bbox')},
        #     {'bbox_mode':BoxMode.XYWH_ABS},
        # ]
        # dicts['before'][i].pop('category_id')
        # dicts['after'][i].pop('category_id')
    logger.info("Loaded {} pairs of images.".format(len(dicts_new)))
    return dicts_new


DatasetCatalog.register('change_pairs', lambda: get_change_pairs_dict('/data1/zhonghaoxiang/dataset/OVIS/ovis_occlusion.json'))

