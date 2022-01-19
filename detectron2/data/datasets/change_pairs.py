# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import os,json
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager


__all__ = ["load_voc_instances", "register_pascal_voc"]

def get_change_pairs_dict(json_name):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        json_name: json path to change pairs
    """

    with open(json_name) as f:
        dicts = json.load(f)
    
    for i in range(len(dicts['before'])):
        dicts['before'][i]['annotations'] = [
            {'bbox': dicts['before'][i].pop('bbox')},
            {'bbox_mode':BoxMode.XYWH_ABS},
        ]
        dicts['after'][i]['annotations'] = [
            {'bbox': dicts['after'][i].pop('bbox')},
            {'bbox_mode':BoxMode.XYWH_ABS},
        ]
        dicts['before'][i].pop('category_id')
        dicts['after'][i].pop('category_id')
    return dicts


DatasetCatalog.register('change_pairs', lambda: get_change_pairs_dict('/data1/zhonghaoxiang/dataset/OVIS/ovis_occlusion.json'))

