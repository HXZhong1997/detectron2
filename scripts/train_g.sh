python tools/plain_train_G.py --config-file configs/Net-G-no2severe.yaml --config-det configs/COCO-Detection/faster_rcnn_R_50_FPN_1x_trained.yaml --num-gpus 2 OUTPUT_DIR output/netG_ch_lr1e-2-no2svr NET_G.MODE 'channel'

python tools/plain_train_G.py --config-file configs/Net-G-no2severe.yaml --config-det configs/COCO-Detection/faster_rcnn_R_50_FPN_1x_trained.yaml --num-gpus 2 OUTPUT_DIR output/netG_sp_lr1e-2-no2svr NET_G.MODE 'spatial'

python tools/plain_train_G.py --config-file configs/Net-G-no2severe.yaml --config-det configs/COCO-Detection/faster_rcnn_R_50_FPN_1x_trained.yaml --num-gpus 2 OUTPUT_DIR output/netG_lr1e-2-no2svr


python tools/plain_train_G.py --resume --config-file configs/Net-G-severe.yaml --config-det configs/COCO-Detection/faster_rcnn_R_50_FPN_1x_trained.yaml --num-gpus 2 OUTPUT_DIR output/netG_ch_lr1e-2-svr NET_G.MODE 'channel'

python tools/plain_train_G.py --config-file configs/Net-G-severe.yaml --config-det configs/COCO-Detection/faster_rcnn_R_50_FPN_1x_trained.yaml --num-gpus 2 OUTPUT_DIR output/netG_sp_lr1e-2-svr NET_G.MODE 'spatial'

python tools/plain_train_G.py --config-file configs/Net-G-severe.yaml --config-det configs/COCO-Detection/faster_rcnn_R_50_FPN_1x_trained.yaml --num-gpus 2 OUTPUT_DIR output/netG_lr1e-2-svr 

#python tools/plain_train_G.py --config-file configs/Net-G-no2severe.yaml --config-det configs/COCO-Detection/faster_rcnn_R_50_FPN_1x_trained.yaml --num-gpus 2 OUTPUT_DIR output/netG_ch_lr1e-2-no2svr NET_G.MODE 'channel'

#python tools/plain_train_G.py --config-file configs/Net-G-no2severe.yaml --config-det configs/COCO-Detection/faster_rcnn_R_50_FPN_1x_trained.yaml --num-gpus 2 OUTPUT_DIR output/netG_sp_lr1e-2-no2svr NET_G.MODE 'spatial'

#python tools/plain_train_G.py --config-file configs/Net-G-no2severe.yaml --config-det configs/COCO-Detection/faster_rcnn_R_50_FPN_1x_trained.yaml --num-gpus 2 OUTPUT_DIR output/netG_lr1e-2-no2svr


python tools/plain_train_G.py --resume --config-file configs/Net-G.yaml --config-det configs/COCO-Detection/faster_rcnn_R_50_FPN_1x_trained.yaml --num-gpus 2 OUTPUT_DIR output/netG_ch_lr1e-2 NET_G.MODE 'channel'

#python tools/plain_train_G.py --config-file configs/Net-G.yaml --config-det configs/COCO-Detection/faster_rcnn_R_50_FPN_1x_trained.yaml --num-gpus 2 OUTPUT_DIR output/netG_sp_lr1e-2 NET_G.MODE 'spatial'


#python tools/plain_train_G.py --config-file configs/Net-G.yaml --config-det configs/COCO-Detection/faster_rcnn_R_50_FPN_1x_trained.yaml --num-gpus 2 OUTPUT_DIR output/netG_ch_lr1e-2 NET_G.MODE 'channel'

#python tools/plain_train_G.py --config-file configs/Net-G.yaml --config-det configs/COCO-Detection/faster_rcnn_R_50_FPN_1x_trained.yaml --num-gpus 2 OUTPUT_DIR output/netG_sp_lr1e-2 NET_G.MODE 'spatial'
#rm output/netG_lr1e-2/model_00{0-4}*

#python tools/plain_train_G.py --config-file configs/Net-G.yaml --config-det configs/COCO-Detection/faster_rcnn_R_50_FPN_1x_trained.yaml --num-gpus 4 OUTPUT_DIR output/netG_res_lr5e-3  SOLVER.BASE_LR 0.005 NET_G.MASK_TYPE 'residual'

#rm output/netG_lr1e-3/model_00{0-4}*

