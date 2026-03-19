#!/bin/bash
# 服务器分布式训练脚本 - 使用 torchrun

export MASTER_ADDR="localhost"
export MASTER_PORT="29500"

# 使用 torchrun 启动
torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    main.py \
    --dataset_name trento \
    --device cuda \
    --with_svd 1 \
    --svd_k 8 \
    --epochs 50 \
    --batch_size 64 \
    --output_dir ./output/server_distributed \
    --log_path ./logs/server_distributed.log \
    --log_epoch 5