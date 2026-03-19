"""
真正的分布式训练实现
使用多进程模拟多节点训练
"""
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import scipy.io as scio
import os
import sys
import time

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import Fed_Fusion, Fed_Fusion_Loss
from config import dataset_config
from py_utils import random_mini_batches_standardtwoModality
from torch.utils.data import DataLoader


def setup(rank, world_size):
    """设置分布式环境"""
    # 使用文件系统初始化，避免网络问题
    init_method = f"file:///{os.path.abspath('.')}/distributed_init"
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size
    )
    print(f"[Rank {rank}] 分布式初始化成功")


def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()


def load_distributed_data(rank, world_size, dataset_dir):
    """
    为不同rank加载不同的数据分区
    模拟真实的分布式数据分布
    """
    print(f"[Rank {rank}] 加载数据分区...")

    # 加载完整数据
    hsi_data = scio.loadmat(os.path.join(dataset_dir, 'TrTe/HSI_TrSet.mat'))['Data']
    lidar_data = scio.loadmat(os.path.join(dataset_dir, 'TrTe/LiDAR_TrSet.mat'))['Data']
    labels = scio.loadmat(os.path.join(dataset_dir, 'TrTe/Y_train.mat'))['Data']

    # 将数据分区给不同rank
    total_samples = len(hsi_data)
    samples_per_rank = total_samples // world_size
    start_idx = rank * samples_per_rank
    end_idx = start_idx + samples_per_rank if rank < world_size - 1 else total_samples

    # 每个rank获取不同的数据分区
    rank_hsi = hsi_data[start_idx:end_idx]
    rank_lidar = lidar_data[start_idx:end_idx]
    rank_labels = labels[start_idx:end_idx]

    print(f"[Rank {rank}] 数据分区: {start_idx}:{end_idx} (共{samples_per_rank}样本)")
    print(f"[Rank {rank}] HSI: {rank_hsi.shape}, LiDAR: {rank_lidar.shape}")

    return rank_hsi, rank_lidar, rank_labels


def train_distributed(rank, world_size, args):
    """分布式训练函数"""
    print(f"\n{'=' * 60}")
    print(f"分布式训练 - Rank {rank}/{world_size}")
    print(f"{'=' * 60}")

    # 设置分布式
    setup(rank, world_size)

    # 设置设备
    torch.cuda.set_device(rank) if torch.cuda.is_available() else None
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print(f"[Rank {rank}] 使用设备: {device}")

    # 获取数据集配置
    config = dataset_config[args.dataset]
    dataset_dir = config['dataset_dir']

    # 加载数据分区
    hsi_data, lidar_data, labels = load_distributed_data(rank, world_size, dataset_dir)

    # 转换为tensor
    hsi_tensor = torch.tensor(hsi_data, dtype=torch.float32).to(device)
    lidar_tensor = torch.tensor(lidar_data, dtype=torch.float32).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).to(device)

    # 创建数据加载器
    train_dataset = random_mini_batches_standardtwoModality(
        hsi_tensor, lidar_tensor, labels_tensor
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # Windows上设为0避免问题
    )

    # 创建模型并包装为DDP
    model = Fed_Fusion(
        config['hsi_n_feature'],
        config['lidar_n_feature'],
        config['num_class']
    ).to(device)

    # 使用DistributedDataParallel
    ddp_model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)

    # 优化器和损失函数
    optimizer = optim.Adam(
        ddp_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    criterion = Fed_Fusion_Loss(
        torch.ones(config['num_class']).to(device)
    ).to(device)

    print(f"[Rank {rank}] 开始分布式训练...")
    print(f"[Rank {rank}] 总epochs: {args.epochs}, 批次大小: {args.batch_size}")

    # 训练循环
    for epoch in range(args.epochs):
        ddp_model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0

        for batch_idx, (hsi_batch, lidar_batch, labels_batch) in enumerate(train_loader):
            hsi_batch = hsi_batch.to(device)
            lidar_batch = lidar_batch.to(device)
            labels_batch = labels_batch.to(device)
            target = torch.argmax(labels_batch, dim=1)

            # 前向传播
            outputs = ddp_model(hsi_batch, lidar_batch, with_svd=args.with_svd)
            loss = criterion(outputs, target)

            # 反向传播（DDP自动处理梯度同步）
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计（只在rank 0打印）
            epoch_loss += loss.item()
            pred = torch.argmax(outputs[0], dim=1)
            epoch_correct += (pred == target).sum().item()
            epoch_total += target.size(0)

            if rank == 0 and batch_idx % 10 == 0:
                print(f"[Rank 0] Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # 同步所有rank的统计信息
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100 * epoch_correct / epoch_total

        # 收集所有rank的指标
        loss_tensor = torch.tensor([avg_loss]).to(device)
        acc_tensor = torch.tensor([accuracy]).to(device)

        # 使用all_reduce计算平均值
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)

        global_avg_loss = loss_tensor.item() / world_size
        global_avg_acc = acc_tensor.item() / world_size

        if rank == 0:
            print(f"\n[全局统计] Epoch {epoch + 1}/{args.epochs}:")
            print(f"  平均损失: {global_avg_loss:.4f}")
            print(f"  平均准确率: {global_avg_acc:.2f}%")

        # 同步点：确保所有rank完成这个epoch
        dist.barrier()

    # 保存模型（只在rank 0保存）
    if rank == 0:
        model_path = f"./output/distributed_model_epochs{args.epochs}.pth"
        torch.save(ddp_model.module.state_dict(), model_path)
        print(f"\n[Rank 0] ✅ 分布式模型已保存: {model_path}")

    # 清理
    cleanup()
    print(f"[Rank {rank}] 训练完成")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='分布式训练')
    parser.add_argument('--dataset', type=str, default='trento', help='数据集')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='权重衰减')
    parser.add_argument('--with_svd', type=int, default=0, help='是否使用SVD')
    parser.add_argument('--world_size', type=int, default=2, help='进程数')

    args = parser.parse_args()

    print("=" * 60)
    print("FedFusion 分布式训练")
    print("=" * 60)
    print(f"数据集: {args.dataset}")
    print(f"进程数: {args.world_size}")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print("=" * 60)

    # 创建输出目录
    os.makedirs("./output", exist_ok=True)

    # 清理可能存在的初始化文件
    init_file = "./distributed_init"
    if os.path.exists(init_file):
        os.remove(init_file)

    # 启动分布式训练
    try:
        mp.spawn(
            train_distributed,
            args=(args.world_size, args),
            nprocs=args.world_size,
            join=True
        )
        print("\n✅ 分布式训练完成！")

    except Exception as e:
        print(f"\n❌ 分布式训练失败: {e}")
        print("\n💡 可能的解决方案:")
        print("1. 确保没有其他进程在使用distributed_init文件")
        print("2. 尝试减少world_size")
        print("3. 在Linux环境下运行以获得更好的兼容性")


if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    main()