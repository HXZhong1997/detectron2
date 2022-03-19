#python tools/plain_train_net.py --config-file configs/PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/faster-rcnn SOLVER.CHECKPOINT_PERIOD 2000
#python tools/plain_train_net_wG.py --resume --config-file configs/faster-rcnn-g/Net-G-lr1e-2.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-s10k-i10-clip  NET_G.INTERVAL 10  NET_G.START_ITER 10000  NET_G.ONLY_G True NET_G.MASK_CLIP True

#python tools/plain_train_net_wG.py --resume --config-file configs/faster-rcnn-g/Net-G-lr1e-2.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-s10k-i25-clip  NET_G.INTERVAL 25  NET_G.START_ITER 10000  NET_G.ONLY_G True NET_G.MASK_CLIP True


CUDA_VISIBLE_DEVICES=$1 python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr1e-2-no2svr.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN_B8.yaml --num-gpus 1 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-no2svr-ui10x4-gs005-clip-minmax  NET_G.INTERVAL 10  NET_G.START_ITER 10000  NET_G.ONLY_G True NET_G.MASK_CLIP True NET_G.UPDATE_START 10000 NET_G.UPDATE_INTERVAL 10 NET_G.UPDATE_TIMES 4 NET_G.UPDATE_MODE 'minmax' NET_G.DROP 0.3 NET_G.G_STEP 0.05

CUDA_VISIBLE_DEVICES=$1 python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr1e-2-no2svr.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN_B8.yaml --num-gpus 1 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-no2svr-ui10x4-gs005-clip-minmin  NET_G.INTERVAL 10  NET_G.START_ITER 10000  NET_G.ONLY_G True NET_G.MASK_CLIP True NET_G.UPDATE_START 10000 NET_G.UPDATE_INTERVAL 10 NET_G.UPDATE_TIMES 4 NET_G.UPDATE_MODE 'minmin' NET_G.DROP 0.3 NET_G.G_STEP 0.05

CUDA_VISIBLE_DEVICES=$1 python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr1e-2-no2svr.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN_B8.yaml --num-gpus 1 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-no2svr-ui10x4-gs005-clip-minmean  NET_G.INTERVAL 10  NET_G.START_ITER 10000  NET_G.ONLY_G True NET_G.MASK_CLIP True NET_G.UPDATE_START 10000 NET_G.UPDATE_INTERVAL 10 NET_G.UPDATE_TIMES 4 NET_G.UPDATE_MODE 'minmean' NET_G.DROP 0.3 NET_G.G_STEP 0.05

CUDA_VISIBLE_DEVICES=$1 python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr1e-2-no2svr.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN_B8.yaml --num-gpus 1 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-no2svr-ui10x4-gs005-clip-confuse  NET_G.INTERVAL 10  NET_G.START_ITER 10000  NET_G.ONLY_G True NET_G.MASK_CLIP True NET_G.UPDATE_START 10000 NET_G.UPDATE_INTERVAL 10 NET_G.UPDATE_TIMES 4 NET_G.UPDATE_MODE 'confuse' NET_G.DROP 0.3 NET_G.G_STEP 0.05

#CUDA_VISIBLE_DEVICES=$1 python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr1e-2-svr.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN_B8.yaml --num-gpus 1 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-ch-svr-s10k-i20-clip  NET_G.INTERVAL 20  NET_G.START_ITER 10000  NET_G.ONLY_G True NET_G.MASK_CLIP True NET_G.G_MODE 'channel'

#CUDA_VISIBLE_DEVICES=$1 python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr1e-2-no2svr.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN_B8.yaml --num-gpus 1 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-ch-no2svr-s10k-i40-clip  NET_G.INTERVAL 40  NET_G.START_ITER 10000  NET_G.ONLY_G True NET_G.MASK_CLIP True NET_G.G_MODE 'channel'

#CUDA_VISIBLE_DEVICES=$1 python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr1e-2-svr.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN_B8.yaml --num-gpus 1 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-ch-svr-s10k-i40-clip  NET_G.INTERVAL 40  NET_G.START_ITER 10000  NET_G.ONLY_G True NET_G.MASK_CLIP True NET_G.G_MODE 'channel'


#CUDA_VISIBLE_DEVICES=$1 python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-sp-lr1e-2-svr.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN_B8.yaml --num-gpus 1 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-sp-svr-s10k-i10-clip  NET_G.INTERVAL 10  NET_G.START_ITER 10000  NET_G.ONLY_G True NET_G.MASK_CLIP True NET_G.G_MODE 'spatial'

#CUDA_VISIBLE_DEVICES=$1 python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr1e-2-no2svr.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN_B8.yaml --num-gpus 1 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-sp-no2svr-s10k-i10-clip  NET_G.INTERVAL 10  NET_G.START_ITER 10000  NET_G.ONLY_G True NET_G.MASK_CLIP True NET_G.G_MODE 'spatial'

#CUDA_VISIBLE_DEVICES=$1 python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-ch-lr1e-2-svr.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN_B8.yaml --num-gpus 1 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-ch-svr-s10k-i10-clip  NET_G.INTERVAL 10  NET_G.START_ITER 10000  NET_G.ONLY_G True NET_G.MASK_CLIP True NET_G.G_MODE 'channel'

#CUDA_VISIBLE_DEVICES=$1 python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-ch-lr1e-2-no2svr.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN_B8.yaml --num-gpus 1 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-ch-no2svr-s10k-i10-clip  NET_G.INTERVAL 10  NET_G.START_ITER 10000  NET_G.ONLY_G True NET_G.MASK_CLIP True NET_G.G_MODE 'channel'

#python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr1e-2.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-s10k-i75-clip  NET_G.INTERVAL 75  NET_G.START_ITER 10000  NET_G.ONLY_G True NET_G.MASK_CLIP True

#python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr1e-2.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-s10k-i100-clip  NET_G.INTERVAL 100  NET_G.START_ITER 10000  NET_G.ONLY_G True NET_G.MASK_CLIP True

#python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr5e-3.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/fasterrcnn-glr5e_3-i50  NET_G.INTERVAL 50  NET_G.START_ITER 12000 

#python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr5e-3.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/fasterrcnn-glr5e_3-i100  NET_G.INTERVAL 100  NET_G.START_ITER 12000
