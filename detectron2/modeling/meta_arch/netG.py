# Copyright (c) Facebook, Inc. and its affiliates.
#from ctypes.wintypes import LARGE_INTEGER
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn


import detectron2.utils.comm as comm
from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.modeling.roi_heads import fast_rcnn
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY

__all__ = ["GeneralizedRCNN", "ProposalNetwork"]


@META_ARCH_REGISTRY.register()
class NetG(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        in_channels,
        mask_type,
        pixel_mean,
        pixel_std,
        input_format,
        mode,
        version,
    ):
        """
        Args:
            in_channels:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            
        """
        super().__init__()
        self.mode = mode
        self.version = version
        if self.version == 'version2' or self.version == 'version3':
            self.netG = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(
                    num_features=in_channels,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=3,
                        padding=1,
                        groups=in_channels,
                ),
                nn.Conv2d(in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=1,
                        padding=0,
                ),
                nn.BatchNorm2d(
                    num_features=in_channels,
                ),
                nn.Sigmoid(),
            )
        else:
            self.netG = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(
                    num_features=in_channels,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=3,
                        padding=1,
                        groups=in_channels,
                ),
                nn.Conv2d(in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=1,
                        padding=0,
                ),
                nn.BatchNorm2d(
                    num_features=in_channels,
                ),
                nn.ReLU(inplace=True),
            )

        # if self.mode == 'channel':
        #     self.criterion = nn.SmoothL1Loss()
        # else:
        if self.version == 'version2' or self.version == 'version3':
            self.criterion=nn.BCELoss()
        else:
            self.criterion = nn.L1Loss()
        self.mask_type = mask_type

        self.input_format = input_format
        
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        
        return {
            "in_channels": cfg.NET_G.IN_CHANNELS,
            "mask_type": cfg.NET_G.MASK_TYPE,
            "input_format": cfg.INPUT.FORMAT,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "mode": cfg.NET_G.G_MODE,
            "version": cfg.NET_G.VERSION,
        }

    @property
    def device(self):
        return self.pixel_mean.device


    def forward(self, batched_inputs, faster_rcnn=None):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            assert isinstance(batched_inputs, torch.Tensor), "infer only when input is the feature tensor"
            self.netG.eval()
            with torch.no_grad():
                mask_ret = self.netG(batched_inputs)
                # if self.mode == 'spatial':
                #     mask_ret = torch.mean(mask_ret,dim=1,keepdim=True)
                # if self.mode == 'channel':
                #     mask_ret = torch.mean(mask_ret,dim=(2,3),keepdim=True)
            return mask_ret
        
        assert faster_rcnn is not None
        distributed = comm.get_world_size() > 1
        if distributed:
            faster_rcnn = faster_rcnn.module
        faster_rcnn.eval()

        images_before, images_after = self.preprocess_image(batched_inputs, faster_rcnn.backbone.size_divisibility)

        # infer roi feature with faster-rcnn
        with torch.no_grad():
            feats_before = faster_rcnn.backbone(images_before.tensor)
            feats_after = faster_rcnn.backbone(images_after.tensor)
            #TODO: Use faster_rcnn roi head to predict feature of gt box.

        gt_instances_before = [x["instances_before"].to(self.device) for x in batched_inputs]
        gt_instances_after = [x["instances_after"].to(self.device) for x in batched_inputs]

        #roi_feats_before, _ = faster_rcnn.roi_heads(images_before, feats_before,)
        
        def _get_box_feats(faster_rcnn, feats, gt):
            with torch.no_grad():
                features = [feats[f] for f in faster_rcnn.roi_heads.box_in_features]
                box_features = faster_rcnn.roi_heads.box_pooler(features, [x.gt_boxes for x in gt])
                # box_features = faster_rcnn.roi_heads.box_head(box_features)
            return box_features
        
        box_feats_before = _get_box_feats(faster_rcnn, feats_before, gt_instances_before) # tensor, proposal_num * channel * w * h (n*256*7*7)  
        box_feats_after = _get_box_feats(faster_rcnn, feats_after, gt_instances_after) 
        def _generate_mask(feats_before,feats_after):
            if self.version == 'version2':
                m = feats_after/(feats_before+1e-9)
                m=m.cpu().flatten()
                m=torch.abs(m)
                _, idx = m.topk(k=int(0.1*len(m)),largest=False)
                idx = np.unravel_index(idx,feats_before.shape)
                mask = torch.ones_like(feats_before)
                mask[idx]=0
                return mask
            
            if self.version == 'version3':
                m = feats_after/(feats_before+1e-9)
                m=m.cpu().reshape(feats_after.size(0),-1)
                m=torch.abs(m)
                mask = torch.ones_like(feats_before)
                _, idx = m.topk(dim=1,k=int(0.1*m.size(1)),largest=False)

                for i in range(idx.size(0)):
                    idx_ = np.unravel_index(idx[i],feats_after[0].shape)
                    mask[i][idx_] = 0
                return mask
            
            if self.mode == 'spatial':
                feats_before = torch.mean(feats_before, dim=1, keepdim=True)
                feats_after = torch.mean(feats_after, dim=1, keepdim=True)
            if self.mode == 'channel':
                feats_before = torch.mean(feats_before,dim=(2,3),keepdim=True) #torch.nn.functional.adaptive_avg_pool2d(feats_before,(1,1))
                feats_after = torch.mean(feats_after,dim=(2,3),keepdim=True) #torch.nn.functional.adaptive_avg_pool2d(feats_after,(1,1))
            with torch.no_grad():
                if self.mask_type=='icassp':
                    mask = torch.ones_like(feats_before)
                    idx_before = torch.where(feats_before==0)
                    idx_after = torch.where(feats_after==0)
                    mask[idx_after]=0
                    mask[idx_before]=1
                elif self.mask_type=='residual':
                    mask = nn.ReLU(inplace=True)(feats_after-feats_before)
                else:
                    raise NotImplementedError
            return mask

        target = _generate_mask(box_feats_before, box_feats_after)
        
        
        pred = self.netG(box_feats_before)
        if self.mode == 'spatial':
            pred = torch.mean(pred,dim=1,keepdim=True)
        if self.mode == 'channel':
            pred = torch.mean(pred,dim=(2,3),keepdim=True)

        loss = self.criterion(pred, target)
        #print('target shape:{}, pred shape:{}'.format(target.shape,pred.shape))
        #print('loss:', loss)
        losses = {'loss_g': loss}
        return losses

    
    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]], size_divisibility):
        """
        Normalize, pad and batch the input images.
        """
        images_before = [x["image_before"].to(self.device) for x in batched_inputs]
        images_before = [(x - self.pixel_mean) / self.pixel_std for x in images_before]
        images_before = ImageList.from_tensors(images_before, size_divisibility)

        images_after = [x["image_before"].to(self.device) for x in batched_inputs]
        images_after = [(x - self.pixel_mean) / self.pixel_std for x in images_after]
        images_after = ImageList.from_tensors(images_after, size_divisibility)
        return images_before, images_after

