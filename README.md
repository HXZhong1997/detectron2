# Code for Faster-rcnn of Chap04 of Haoxiang's Thesis 

Based on Detectron2.


## Step1: Preparation

Prepare dataset `VOC2007`, `VOC2012`, `OVIS` in `./datasets`

Download [`ovis_occlusion_no2severe.json`](https://pan.quark.cn/s/6cde99d07622) to `./datasets/OVIS`

Download [`output/faster-rcnn/model_final.pth`](https://pan.quark.cn/s/6cde99d07622) and put it at that path.

Prepare the environment as in [README of Detectron](#below-are-the-original-readme-file-of-detecron2)

## Step2: Train mask generation network (net g).

```
python tools/plain_train_G.py \
--config-file configs/Net-G-no2severe.yaml \
--config-det configs/COCO-Detection/faster_rcnn_R_50_FPN_1x_trained.yaml \
--num-gpus 2 \
OUTPUT_DIR output/netG_lr1e-2-no2svr
```

## Step3: Train Faster RCNN with net g together with the update strategy

```
python tools/plain_train_net_wG.py \
--config-file configs/faster-rcnn-g/Net-G-lr1e-2-no2svr.yaml \
--config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN.yaml \
--num-gpus 4 \
OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-no2svr-ui20x5-d3-icassp-gs005-clip \
NET_G.INTERVAL 20 \
NET_G.START_ITER 10000 \
NET_G.UPDATE_START 10000 \
NET_G.UPDATE_INTERVAL 20 \
NET_G.UPDATE_MODE 'icassp' \
NET_G.ONLY_G True \
NET_G.UPDATE_TIMES 5 \
NET_G.DROP 0.3 \
NET_G.MASK_CLIP True \
NET_G.G_STEP 0.05 
```

## Below are the original README file of Detecron2

<img src=".github/Detectron2-Logo-Horz.svg" width="300" >

Detectron2 is Facebook AI Research's next generation library
that provides state-of-the-art detection and segmentation algorithms.
It is the successor of
[Detectron](https://github.com/facebookresearch/Detectron/)
and [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/).
It supports a number of computer vision research projects and production applications in Facebook.

<div align="center">
  <img src="https://user-images.githubusercontent.com/1381301/66535560-d3422200-eace-11e9-9123-5535d469db19.png"/>
</div>

### What's New
* Includes new capabilities such as panoptic segmentation, Densepose, Cascade R-CNN, rotated bounding boxes, PointRend,
  DeepLab, etc.
* Used as a library to support building [research projects](projects/) on top of it.
* Models can be exported to TorchScript format or Caffe2 format for deployment.
* It [trains much faster](https://detectron2.readthedocs.io/notes/benchmarks.html).

See our [blog post](https://ai.facebook.com/blog/-detectron2-a-pytorch-based-modular-object-detection-library-/)
to see more demos and learn about detectron2.

## Installation

See [installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).

## Getting Started

See [Getting Started with Detectron2](https://detectron2.readthedocs.io/tutorials/getting_started.html),
and the [Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
to learn about basic usage.

Learn more at our [documentation](https://detectron2.readthedocs.org).
And see [projects/](projects/) for some projects that are built on top of detectron2.

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [Detectron2 Model Zoo](MODEL_ZOO.md).

## License

Detectron2 is released under the [Apache 2.0 license](LICENSE).

## Citing Detectron2

If you use Detectron2 in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```BibTeX
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```
