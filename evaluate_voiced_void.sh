#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
#NYU dataset
#python evaluate.py \
#--model /home/mikkel/final_models/nyu/1589616487-n12672-e20-bs4-lr0.0001-nyu-half-features/model.h5 \
#--dataset nyu \
#--use-cpu

#python evaluate.py \
#--dataset nyu \
#--use-cpu \
#--model /home/mikkel/final_models/nyu/1589474496-n12672-e20-bs4-lr0.0001-nyu-all-features/model.h5

#python evaluate.py \
#--dataset nyu \
#--use-cpu \
#--model /home/mikkel/final_models/nyu/1587572185-n6336-e20-bs8-lr0.0001-nyu-resnet50/model.h5

#VOID RGB
python evaluate.py \
--dataset void \
--channels 3 \
--use-cpu \
--use-void-1500 \
--model /home/mikkel/final_models/void_rgb/1589053397-n12104-e20-bs4-lr0.0001-void-nyu/model.h5

python evaluate.py \
--dataset void \
--channels 3 \
--use-cpu \
--use-void-1500 \
--model /home/mikkel/final_models/void_rgb/1586925390-n12104-e5-bs4-lr0.0001-nyu-void/model.h5

python evaluate.py \
--dataset void \
--channels 3 \
--use-cpu \
--use-void-1500 \
--model /home/mikkel/final_models/void_rgb/densedepth_void_e20-bs4-lr0.001/model.h5

python evaluate.py \
--dataset void \
--channels 3 \
--use-cpu \
--use-void-1500 \
--model /home/mikkel/final_models/void_rgb/1589360045-n12104-e20-bs4-lr0.0001-void-only-all-features/model.h5

VOID RGBD 150
python evaluate.py \
--dataset void \
--channels 4 \
--use-cpu \
--dont-interpolate \
--use-void-1500 \
--model /home/mikkel/final_models/void_rgbd-150/1589287561-n12104-e15-bs4-lr0.0001-void-1500-weigted-early-fusion-imagenet-3+1-sD/model.h5

python evaluate.py \
--dataset void \
--channels 4 \
--use-cpu \
--dont-interpolate \
--use-void-1500 \
--model /home/mikkel/final_models/void_rgbd-150/1589204737-n12104-e10-bs4-lr0.0001-void-1500-weigted-early-fusion-imagenet-3+1-sD-without-depthnorm/model.h5

python evaluate.py \
--dataset void \
--channels 4 \
--use-cpu \
--use-void-1500 \
--model /home/mikkel/final_models/void_rgbd-150/1588955269-n24207-e10-bs2-lr0.0001-void-1500-late-fusion-imagenet-3+1-iD/model.h5

python evaluate.py \
--dataset void \
--channels 4 \
--use-cpu \
--use-void-1500 \
--model /home/mikkel/final_models/void_rgbd-150/1588876366-n12104-e20-bs4-lr0.0001-void-1500-early-fusion-imagenet-3+1-iD/model.h5

python evaluate.py \
--dataset void \
--channels 4 \
--use-cpu \
--use-void-1500 \
--model /home/mikkel/final_models/void_rgbd-150/1588839176-n12104-e9-bs4-lr0.0001-void-early-fusion-imagenet-3+1-iD-without-depthnorm/model.h5

python evaluate.py \
--dataset void \
--channels 5 \
--use-cpu \
--dont-interpolate \
--use-void-1500 \
--model /home/mikkel/final_models/void_rgbd-150/1588698490-n24207-e9-bs2-lr0.0001-void-two-branch-end-fusion-imagenet-3+2-sD-without-depthnorm/model.h5

python evaluate.py \
--dataset void \
--channels 4 \
--use-cpu \
--use-void-1500 \
--model /home/mikkel/final_models/void_rgbd-150/1588180820-n24207-e6-bs2-lr0.0001-void-two-branch-imagenet-iD-with-depthnorm/model.h5

python evaluate.py \
--dataset void \
--channels 4 \
--use-cpu \
--use-void-1500 \
--model /home/mikkel/final_models/void_rgbd-150/1588085278-n12104-e9-bs4-lr0.0001-void-4channel-imagenet-iD-with-depthnorm-early-fusion/model.h5

python evaluate.py \
--dataset void \
--channels 5 \
--use-cpu \
--use-void-1500 \
--model /home/mikkel/final_models/void_rgbd-150/1587845588-n12104-e20-bs4-lr0.0001-void-5channel-without-depthnorm/model.h5

python evaluate.py \
--dataset void \
--channels 5 \
--use-cpu \
--use-void-1500 \
--model /home/mikkel/final_models/void_rgbd-150/1587726473-n12104-e20-bs4-lr0.0001-void-5channel-3rd-run/model.h5

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
