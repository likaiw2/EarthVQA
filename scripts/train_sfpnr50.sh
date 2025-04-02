#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:`pwd`
NUM_GPUS=1

# config_path='sfpnr50'
# model_dir='./log/sfpnr50'

config_path='sfpnr_on_cityscapes'
model_dir='./log/sfpnr_on_cityscapes/'

# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 29500 train_lovedav2_seg.py \
#     --config_path=${config_path} \
#     --model_dir=${model_dir} \

torchrun --nproc_per_node=${NUM_GPUS} --master_port 29500 train_lovedav2_seg.py \
    --config_path=${config_path} \
    --model_dir=${model_dir} \

# CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port 29501 train_lovedav2_seg.py \
#     --config_path=${config_path} \
#     --model_dir=${model_dir}