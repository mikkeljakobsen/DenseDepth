#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python evaluate.py \
--model /home/mikkel/final_models/nyu/1589616487-n12672-e20-bs4-lr0.0001-nyu-half-features/model.h5 \
--dataset nyu \
--use-cpu

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
