#python tools/plain_train_net.py --config-file configs/PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/faster-rcnn SOLVER.CHECKPOINT_PERIOD 2000

#python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr1e-2.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-ui25-conf2  NET_G.INTERVAL 25  NET_G.START_ITER 12000 NET_G.UPDATE_START 12000 NET_G.UPDATE_INTERVAL 25 NET_G.UPDATE_MODE 'confuse'

#python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr1e-2.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-ui50-conf2  NET_G.INTERVAL 50  NET_G.START_ITER 12000 NET_G.UPDATE_START 12000 NET_G.UPDATE_INTERVAL 50 NET_G.UPDATE_MODE 'confuse'

#python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr1e-2.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-ui75-conf2  NET_G.INTERVAL 75  NET_G.START_ITER 12000 NET_G.UPDATE_START 12000 NET_G.UPDATE_INTERVAL 75 NET_G.UPDATE_MODE 'confuse'

#python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr1e-2.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-ui100-conf2  NET_G.INTERVAL 100  NET_G.START_ITER 12000 NET_G.UPDATE_START 12000 NET_G.UPDATE_INTERVAL 100 NET_G.UPDATE_MODE 'confuse'

#python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr1e-2.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-ui25-mean2  NET_G.INTERVAL 25  NET_G.START_ITER 12000 NET_G.UPDATE_START 12000 NET_G.UPDATE_INTERVAL 25 NET_G.UPDATE_MODE 'minmean'
#python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr1e-2.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-s10k-i75  NET_G.INTERVAL 75  NET_G.START_ITER 10000  NET_G.ONLY_G True
python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr1e-2.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-ui10x4-d3-icassp-clip  NET_G.INTERVAL 10  NET_G.START_ITER 10000 NET_G.UPDATE_START 10000 NET_G.UPDATE_INTERVAL 10 NET_G.UPDATE_MODE 'icassp' NET_G.ONLY_G True NET_G.UPDATE_TIMES 4 NET_G.DROP 0.3 NET_G.MASK_CLIP True

python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr1e-2.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-ui25x10-d3-icassp-clip  NET_G.INTERVAL 25  NET_G.START_ITER 10000 NET_G.UPDATE_START 10000 NET_G.UPDATE_INTERVAL 25 NET_G.UPDATE_MODE 'icassp' NET_G.ONLY_G True NET_G.UPDATE_TIMES 10 NET_G.DROP 0.3 NET_G.MASK_CLIP True

python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr1e-2.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-ui50x20-d3-icassp-clip  NET_G.INTERVAL 50  NET_G.START_ITER 10000 NET_G.UPDATE_START 10000 NET_G.UPDATE_INTERVAL 50 NET_G.UPDATE_MODE 'icassp' NET_G.ONLY_G True NET_G.UPDATE_TIMES 20 NET_G.DROP 0.3 NET_G.MASK_CLIP True

python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr1e-2.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-ui75x30-d3-icassp-clip  NET_G.INTERVAL 75  NET_G.START_ITER 10000 NET_G.UPDATE_START 10000 NET_G.UPDATE_INTERVAL 75 NET_G.UPDATE_MODE 'icassp' NET_G.ONLY_G True NET_G.UPDATE_TIMES 30 NET_G.DROP 0.3 NET_G.MASK_CLIP True

python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr1e-2.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-ui100x40-d3-icassp-clip  NET_G.INTERVAL 100  NET_G.START_ITER 10000 NET_G.UPDATE_START 10000 NET_G.UPDATE_INTERVAL 100 NET_G.UPDATE_MODE 'icassp' NET_G.ONLY_G True NET_G.UPDATE_TIMES 40 NET_G.DROP 0.3 NET_G.MASK_CLIP True

#python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr1e-2.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-i25-2  NET_G.INTERVAL 25  NET_G.START_ITER 12000

#python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr1e-2.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-i75-2  NET_G.INTERVAL 75  NET_G.START_ITER 12000
#python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr1e-2.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-ui75-mean2  NET_G.INTERVAL 75  NET_G.START_ITER 12000 NET_G.UPDATE_START 12000 NET_G.UPDATE_INTERVAL 75 NET_G.UPDATE_MODE 'minmean'

#python tools/plain_train_net_wG.py --resume --config-file configs/faster-rcnn-g/Net-G-lr1e-2.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-ui100-mean2  NET_G.INTERVAL 100  NET_G.START_ITER 12000 NET_G.UPDATE_START 12000 NET_G.UPDATE_INTERVAL 100 NET_G.UPDATE_MODE 'minmean'

#python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr1e-2.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-ui25-2  NET_G.INTERVAL 25  NET_G.START_ITER 12000 NET_G.UPDATE_START 12000 NET_G.UPDATE_INTERVAL 25 NET_G.UPDATE_MODE 'icassp'

#python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr1e-2.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-ui50-2  NET_G.INTERVAL 50  NET_G.START_ITER 12000 NET_G.UPDATE_START 12000 NET_G.UPDATE_INTERVAL 50 NET_G.UPDATE_MODE 'icassp'

#python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr1e-2.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-ui75-2  NET_G.INTERVAL 75  NET_G.START_ITER 12000 NET_G.UPDATE_START 12000 NET_G.UPDATE_INTERVAL 75 NET_G.UPDATE_MODE 'icassp'

#python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr1e-2.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-ui100-2  NET_G.INTERVAL 100  NET_G.START_ITER 12000 NET_G.UPDATE_START 12000 NET_G.UPDATE_INTERVAL 100 NET_G.UPDATE_MODE 'icassp'

#python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr5e-3.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/fasterrcnn-glr5e_3-i50  NET_G.INTERVAL 50  NET_G.START_ITER 12000 

#python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr5e-3.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/fasterrcnn-glr5e_3-i100  NET_G.INTERVAL 100  NET_G.START_ITER 12000
