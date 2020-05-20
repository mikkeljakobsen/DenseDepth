#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
#VOID RGB
#python evaluate.py \
#--dataset void \
#--channels 3 \
#--use-cpu \
#--save \
#--model-name void-rgb \
#--use-void-1500 \
#--model /home/mikkel/final_models/void_rgb/1589053397-n12104-e20-bs4-lr0.0001-void-nyu/model.h5
#
#python evaluate.py \
#--dataset void \
#--channels 3 \
#--use-cpu \
#--use-median-scaling \
#--save \
#--model-name void-rgb-global-gt-scaling \
#--use-void-1500 \
#--model /home/mikkel/final_models/void_rgb/1589053397-n12104-e20-bs4-lr0.0001-void-nyu/model.h5

python evaluate.py \
--dataset void \
--channels 3 \
--use-cpu \
--use-median-scaling \
--use-sparse-depth-scaling \
--save \
--model-name void-rgb-global-iD-scaling \
--use-void-1500 \
--model /home/mikkel/final_models/void_rgb/1589053397-n12104-e20-bs4-lr0.0001-void-nyu/model.h5

python evaluate.py \
--dataset void \
--channels 3 \
--use-cpu \
--use-median-scaling \
--use-sparse-depth-scaling \
--use-scaling-array \
--model-name void-rgb-local-iD-scaling \
--use-void-1500 \
--model /home/mikkel/final_models/void_rgb/1589053397-n12104-e20-bs4-lr0.0001-void-nyu/model.h5

#python evaluate.py \
#--model /home/mikkel/final_models//model.h5 \
#--dataset nyu \
#--channels 3 \
#--use-cpu \
#--dont-interpolate \
#--use-void-1500 \
#--use-median-scaling \
#--use-sparse-depth-scaling \
#--use-scaling-array \
#--path /home/mikkel/data/pico_sommerhus \
#--gt-divider 1000.0 \
#--save
