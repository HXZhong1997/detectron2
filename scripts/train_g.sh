python tools/plain_train_G.py --config-file configs/Net-G.yaml --config-det configs/COCO-Detection/faster_rcnn_R_50_FPN_1x_trained.yaml --num-gpus 4 OUTPUT_DIR output/netG_lr1e-2 
#rm output/netG_lr1e-2/model_00{0-4}*

python tools/plain_train_G.py --config-file configs/Net-G.yaml --config-det configs/COCO-Detection/faster_rcnn_R_50_FPN_1x_trained.yaml --num-gpus 4 OUTPUT_DIR output/netG_lr5e-3  SOLVER.BASE_LR 0.005

#rm output/netG_lr1e-3/model_00{0-4}*

