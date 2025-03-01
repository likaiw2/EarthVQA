#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:`pwd`
config_path='sfpnr50'
model_dir='./log/sfpnr50'
NUM_GPUS=1

# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 29500 train_lovedav2_seg.py \
#     --config_path=${config_path} \
#     --model_dir=${model_dir} \


python -m debugpy --listen 5678 --wait-for-client -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port 29500 train_lovedav2_seg.py \
    --config_path=sfpnr50 \
    --model_dir=./log/sfpnr50