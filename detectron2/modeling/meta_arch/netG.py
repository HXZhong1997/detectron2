# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

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
        in_channels: int,
        mask_type: str,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
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

        self.criterion = nn.L1Loss()
        self.mask_type = mask_type

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

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
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs: tuple(List[Dict[str, torch.Tensor]]), faster_rcnn: nn.Module = None):
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
            return mask_ret
        
        assert faster_rcnn is not None
        faster_rcnn.eval()

        images_before, images_after = self.preprocess_image(batched_inputs)

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
                box_features = faster_rcnn.roi_heads.box_head(box_features)
            return box_features
        
        box_feats_before = _get_box_feats(faster_rcnn, feats_before, gt_instances_before) # tensor, proposal_num * channel * w * h (n*256*7*7)  
        box_feats_after = _get_box_feats(faster_rcnn, feats_after, gt_instances_after) 
        
        def _generate_mask(feats_before,feats_after):
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

        loss = self.criterion(pred, target)

        losses = {'loss_g': loss}
        return losses

    

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images_before = [x["image_before"].to(self.device) for x in batched_inputs]
        images_before = [(x - self.pixel_mean) / self.pixel_std for x in images_before]
        images_before = ImageList.from_tensors(images_before, self.backbone.size_divisibility)

        images_after = [x["image_before"].to(self.device) for x in batched_inputs]
        images_after = [(x - self.pixel_mean) / self.pixel_std for x in images_after]
        images_after = ImageList.from_tensors(images_after, self.backbone.size_divisibility)
        return images_before, images_after


