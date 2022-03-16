#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

from dataclasses import dataclass
import logging
import os,sys,time
import numpy as np
import random
from pyexpat import features
from collections import OrderedDict
import torch, argparse
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage

from detectron2.data.dataset_mapper_pairs import DatasetMapperChangePairs
logger = logging.getLogger("detectron2")


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results

def do_update_g(model_det, model_g, cfg_det, data):
        distributed = comm.get_world_size() > 1
        if "instances" in data[0]:
            gt_instances = [x["instances"].to(model_det.device) for x in data]
        else:
            gt_instances = None
        if distributed:
            images = model_det.module.preprocess_image(data)
            with torch.no_grad():
                features_orig = model_det.module.backbone(images.tensor)
                proposals, _ = model_det.module.proposal_generator(images, features_orig, None)
                proposals = model_det.module.roi_heads.label_and_sample_proposals(proposals, gt_instances)
                features = [features_orig[f] for f in model_det.module.roi_heads.box_in_features]
                box_features = model_det.module.roi_heads.box_pooler(features, [x.proposal_boxes for x in proposals])
        else:
            images = model_det.preprocess_image(data)
            with torch.no_grad():
                features_orig = model_det.backbone(images.tensor)
                proposals, _ = model_det.proposal_generator(images, features_orig, None)
                proposals = model_det.roi_heads.label_and_sample_proposals(proposals, gt_instances)
                features = [features_orig[f] for f in model_det.roi_heads.box_in_features]
                box_features = model_det.roi_heads.box_pooler(features, [x.proposal_boxes for x in proposals])
        
        masks_slide = []
        win_width=3
        win_overlap=1

        for i in range(3):
            for j in range(3):
                mask_ = torch.ones_like(box_features[0,0,:,:])
                x_start = i*win_width-i*win_overlap
                x_end = x_start+win_width
                y_start = i*win_width-i*win_overlap
                y_end = y_start+win_width

                mask_[x_start:x_end,y_start:y_end]=0
                masks_slide.append(mask_)            
                    
        #random perturbation on mask

        for mask_ in masks_slide:
            
            with torch.no_grad():
                if cfg_det.NET_G.UPDATE_MODE == 'icassp' or cfg_det.NET_G.UPDATE_MODE == 'loss-cls':
                    model_det.train()
                    if distributed:
                        _, detector_loss_g = model_det.module.roi_heads(images,features_orig,proposals,gt_instances,model_g,mask_)
                    else:
                        _, detector_loss_g = model_det.roi_heads(images,features_orig,proposals,gt_instances,model_g,mask_)
                    if cfg_det.NET_G.UPDATE_MODE == 'loss-cls':
                        scores=detector_loss_g['loss_cls'].cpu()
                    else:
                        scores=sum(detector_loss_g.values()).cpu()
                    model_det.eval()
                else: 
                    if cfg_det.NET_G.MASK_TYPE == 'icassp':
                        box_features_ = box_features * mask_
                    elif cfg_det.NET_G.MASK_TYPE == 'residual':
                        box_features_ = box_features + mask_
                    else:
                        raise NotImplementedError
                    
                    if distributed:
                        box_features_ = model_det.module.roi_heads.box_head(box_features_)
                        predictions = model_det.module.roi_heads.box_predictor(box_features_)
                        pred_instances = model_det.module.roi_heads.box_predictor.inference(predictions,proposals)
                    else:
                        box_features_ = model_det.roi_heads.box_head(box_features_)
                        predictions = model_det.roi_heads.box_predictor(box_features_)
                        pred_instances = model_det.roi_heads.box_predictor.inference(predictions,proposals)
                    
                    scores=[]

                    for it in pred_instances[0]:
                        if len(it.scores) == 0: #TODO: no such field
                            scores.append(torch.tensor(0,dtype=box_features_.dtype))
                        elif cfg_det.NET_G.UPDATE_MODE=='confuse':
                            idx = torch.argmin(torch.abs(it.scores-0.5))
                            scores.append(it.scores[idx].cpu())
                        else:
                            scores.append(it.scores.max().cpu()) #batchsize * 1
            
            max_scores.append(scores)
            masks_pert.append(mask_)

        #TODO: select target mask and update netG
        def select_mask(max_scores):
            '''
                max_scores: list(np.array(batchsize,1))        
            '''
            if cfg_det.NET_G.UPDATE_MODE == 'icassp':
                return np.argmax(max_scores)
            elif cfg_det.NET_G.UPDATE_MODE == 'minmax':
                max_scores = [np.max(it) for it in max_scores]
                return np.argmin(max_scores)
            elif cfg_det.NET_G.UPDATE_MODE == 'minmin':
                min_score = [np.min(it) for it in max_scores]
                return np.argmin(min_score)
            elif cfg_det.NET_G.UPDATE_MODE == 'minmean':
                mean_score = [np.mean(it) for it in max_scores]
                return np.argmin(mean_score)
            elif cfg_det.NET_G.UPDATE_MODE == 'confuse':
                confuse = [np.abs(np.array(it)-0.5).mean() for it in max_scores]
                return np.argmin(confuse)
            else:
                raise NotImplementedError

        idx = select_mask(max_scores)
        mask_tar = masks_pert[idx]

        del masks_pert, max_scores

        # 'Updating net G ...'
        model_g.train(True)

        with torch.enable_grad():
            if distributed:
                mask_pre = model_g.module.netG(box_features)
            else:
                mask_pre = model_g.netG(box_features)
            if cfg_det.NET_G.G_MODE == 'spatial':
                mask_pre = torch.mean(mask_pre,dim=1,keepdim=True)
            if cfg_det.NET_G.G_MODE == 'channel':
                mask_pre = torch.mean(mask_pre,dim=(2,3),keepdim=True)

        loss = nn.L1Loss()(mask_pre,mask_tar)

        return {'loss_g':loss}
        


def do_train(cfg_g, model_g, cfg_det, model_det, resume=False):
    model_det.train()
    model_g.eval()

    optimizer = build_optimizer(cfg_det, model_det)
    scheduler = build_lr_scheduler(cfg_det, optimizer)

    optimizer_g = build_optimizer(cfg_g, model_g)
    scheduler_g = torch.optim.lr_scheduler.StepLR(
        optimizer_g,
        step_size= int(((cfg_det.SOLVER.MAX_ITER - cfg_det.NET_G.UPDATE_START)/cfg_det.NET_G.UPDATE_INTERVAL)*cfg_det.NET_G.G_STEP),
        gamma=1.0,
        )

    checkpointer_g = DetectionCheckpointer(
        model_g, cfg_g.OUTPUT_DIR,
    )
    # checkpointer_g.load(os.path.join(cfg_g.OUTPUT_DIR,'model_final.pth'))

    checkpointer = DetectionCheckpointer(
        model_det, cfg_det.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg_det.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg_det.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg_det.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg_det.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    data_loader = build_detection_train_loader(cfg_det)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            #pretrain
            if iteration == cfg_det.NET_G.PRETRAIN_START:
                model_det.eval()
                model_g.train()
            if iteration +1 == cfg_det.NET_G.UPDATE_START:
                model_g.eval()
                model_det.train()
            
            if cfg_det.NET_G.PRETRAIN_START <= iteration <= cfg_det.NET_G.UPDATE_START:
                tic = time.time()
                model_det.eval()
                model_g.train()
                loss_dict_g = do_update_g(model_det,model_g,cfg_det,data)
                losses = sum(loss_dict_g.values())

                assert torch.isfinite(losses).all(), loss_dict_g

                loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict_g).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

                optimizer_g.zero_grad()
                losses.backward()
                optimizer_g.step()
                scheduler_g.step()

                logger.info('Net-G Updated at iter {}  #{}  loss_g: {:.4f}   sec: {:.5f}'.format(iteration, i, loss_dict_reduced['loss_g'], time.time()-tic))
                continue


            if (
                iteration > cfg_det.NET_G.UPDATE_START and 
                (iteration + 1)%cfg_det.NET_G.UPDATE_INTERVAL==0 and
                cfg_det.NET_G.UPDATE_MODE != 'no'
            ):
                for i in range(cfg_det.NET_G.UPDATE_TIMES):
                #TODO: update netG
                    tic = time.time()
                    model_det.eval()
                    model_g.train()
                    loss_dict_g = do_update_g(model_det,model_g,cfg_det,data)
                    losses = sum(loss_dict_g.values())

                    assert torch.isfinite(losses).all(), loss_dict_g

                    loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict_g).items()}
                    losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                    if comm.is_main_process():
                        storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

                    optimizer_g.zero_grad()
                    losses.backward()
                    optimizer_g.step()
                    scheduler_g.step()
                
                    model_g.eval()
                    model_det.train()
                    logger.info('Net-G Updated at iter {}  #{}  loss_g: {:.4f}   sec: {:.5f}'.format(iteration, i, loss_dict_reduced['loss_g'], time.time()-tic))
            if (
                iteration > cfg_det.NET_G.START_ITER and 
                (iteration + 1)%cfg_det.NET_G.INTERVAL == 0
            ):
                loss_dict = model_det(data, model_g)
            else:
                loss_dict = model_det(data)

            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg_det.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg_det.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                do_test(cfg_det, model_det)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg_det = get_cfg()
    cfg_det.merge_from_file(args.config_det)
    cfg_det.merge_from_list(args.opts)
    cfg_det.freeze()

    cfg_g = get_cfg()
    cfg_g.merge_from_file(args.config_file)
    #cfg_g.merge_from_list(args.opts)
    cfg_g.freeze()

    default_setup(
        cfg_det, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg_det, cfg_g


def main(args):
    cfg_det,cfg_g = setup(args)

    model_det = build_model(cfg_det)
    logger.info("Detector Model:\n{}".format(model_det))

    model_g = build_model(cfg_g)
    logger.info("Net-G Model:\n{}".format(model_g))
    
    distributed = comm.get_world_size() > 1
    if distributed:
        model_det = DistributedDataParallel(
            model_det, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )
        model_g = DistributedDataParallel(
            model_g, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg_g, model_g, cfg_det, model_det, resume=args.resume)
    return do_test(cfg_det,model_det)

def my_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

Change some config options:
    $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--config-det", default="", metavar="FILE", help="path to config file of detector")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == "__main__":
    args = my_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
