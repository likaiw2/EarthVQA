#!/usr/bin/env bash

# export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=$PYTHONPATH:`pwd`
config_path='soba'
model_dir='./log/soba'
NUM_GPUS=2

python -m torch.distributed.launch\
    --nproc_per_node=${NUM_GPUS} \
    --master_port 29500 train_earthvqa.py \
    --config_path=${config_path} \
    --model_dir=${model_dir} \
    learning_rate.params.max_iters 40000 \
    train.num_iters 40000

s