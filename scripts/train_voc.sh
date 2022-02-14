#python tools/plain_train_net.py --config-file configs/PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/faster-rcnn SOLVER.CHECKPOINT_PERIOD 2000

python tools/plain_train_net_wG.py --resume --config-file configs/faster-rcnn-g/Net-G-lr1e-2.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-s10k-i25  NET_G.INTERVAL 25  NET_G.START_ITER 10000  NET_G.ONLY_G True

python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr1e-2.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-s10k-i50  NET_G.INTERVAL 50  NET_G.START_ITER 10000  NET_G.ONLY_G True

python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr1e-2.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-s10k-i75  NET_G.INTERVAL 75  NET_G.START_ITER 10000  NET_G.ONLY_G True

python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr1e-2.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/fasterrcnn-glr1e_2-s10k-i200  NET_G.INTERVAL 100  NET_G.START_ITER 10000  NET_G.ONLY_G True

#python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr5e-3.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/fasterrcnn-glr5e_3-i50  NET_G.INTERVAL 50  NET_G.START_ITER 12000 

#python tools/plain_train_net_wG.py --config-file configs/faster-rcnn-g/Net-G-lr5e-3.yaml --config-det configs/faster-rcnn-g/voc_faster_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR output/voc/fasterrcnn-glr5e_3-i100  NET_G.INTERVAL 100  NET_G.START_ITER 12000
