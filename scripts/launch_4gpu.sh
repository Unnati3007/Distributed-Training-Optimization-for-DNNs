#!/bin/bash
# launch_4gpu.sh — Launch DDP training on 4 GPUs (single node)

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun \
    --nproc_per_node=4 \
    --master_addr=localhost \
    --master_port=29500 \
    src/train_ddp.py \
    --data-dir /data/imagenet \
    --epochs 90 \
    --batch-size 256 \
    --lr 0.1

echo "Training complete."
