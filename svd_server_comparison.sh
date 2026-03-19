#!/bin/bash
# 对比不同SVD参数的性能

# 测试不同k值
for k in 0 2 4 8 16 32
do
    echo ""
    echo "=================================================="
    echo "测试 SVD k=$k"
    echo "=================================================="

    if [ $k -eq 0 ]; then
        # k=0 表示不使用SVD
        python -m torch.distributed.launch \
            --nproc_per_node=2 \
            --nnodes=1 \
            main.py \
            --dataset_name trento \
            --device cuda \
            --with_svd 0 \
            --epochs 10 \
            --batch_size 64 \
            --output_dir ./output/svd_k${k} \
            --log_path ./logs/svd_k${k}.log
    else
        # 使用SVD
        python -m torch.distributed.launch \
            --nproc_per_node=2 \
            --nnodes=1 \
            main.py \
            --dataset_name trento \
            --device cuda \
            --with_svd 1 \
            --svd_k $k \
            --epochs 10 \
            --batch_size 64 \
            --output_dir ./output/svd_k${k} \
            --log_path ./logs/svd_k${k}.log
    fi
done

# 收集结果
echo ""
echo "=================================================="
echo "SVD性能对比结果"
echo "=================================================="
echo "k值 | 准确率 | 训练时间 | 压缩比"
echo "------------------------------------"

for k in 0 2 4 8 16 32
do
    if [ -f "./logs/svd_k${k}.log" ]; then
        acc=$(grep "Test_OA" ./logs/svd_k${k}.log | tail -1 | awk '{print $NF}')
        echo "k=$k | $acc"
    fi
done