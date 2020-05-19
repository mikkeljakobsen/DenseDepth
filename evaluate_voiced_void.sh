#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python evaluate.py \
--model /home/mikkel/final_models/1589616487-n12672-e20-bs4-lr0.0001-nyu-half-features/model.h5 \
--dataset nyu \
--use-cpu
