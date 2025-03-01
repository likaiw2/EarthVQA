#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:`pwd`
config_path='soba'
model_dir='./log/soba'
NUM_GPUS=1

# python -m torch.distributed.launch \
#     --nproc_per_node=${NUM_GPUS} \
#     --master_port $RANDOM train_earthvqa.py \
#     --config_path=${config_path} \
#     --model_dir=${model_dir} \
#     learning_rate.params.max_iters 40000 \
#     train.num_iters 40000


python -m debugpy \
    --listen 5678 \
    --wait-for-client -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    --master_port 29500 train_earthvqa.py \
    --config_path=sfpnr50 \
    --model_dir=./log/sfpnr50 \
    learning_rate.params.max_iters 40000 \
    train.num_iters 40000