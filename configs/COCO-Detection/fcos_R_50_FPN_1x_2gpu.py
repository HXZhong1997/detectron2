from ..common.optim import SGD as optimizer
from ..common.coco_schedule import lr_multiplier_1x as lr_multiplier
from ..common.data.coco import dataloader
from ..common.models.fcos import model
from ..common.train import train

dataloader.train.mapper.use_instance_mask = False
dataloader.train.total_batch_size = 4
optimizer.lr = 0.0025 #0.01


model.backbone.bottom_up.freeze_at = 2
train.init_checkpoint = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
train.output_dir = "./output/fcos_2gpu"
