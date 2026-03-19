"""
HSI客户端 - 专门处理高光谱数据
基于 torch.distributed 的分布式训练节点
"""
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import scipy.io as scio
import os
import sys
import argparse
import time

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import Fed_Fusion, Fed_Fusion_Loss
from config import dataset_config
from py_utils import random_mini_batches_standardtwoModality


# def setup(rank, world_size, port=29500):
#     """设置分布式环境 - 支持单机模式"""
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = str(port)
#     os.environ['RANK'] = str(rank)
#     os.environ['WORLD_SIZE'] = str(world_size)
#     os.environ['LOCAL_RANK'] = str(rank)
#
#     try:
#         # 尝试初始化分布式
#         dist.init_process_group("gloo", rank=rank, world_size=world_size)
#         print(f"[HSI Client] Rank {rank}/{world_size} 分布式初始化成功")
#         return True
#     except Exception as e:
#         print(f"[HSI Client] 分布式初始化失败: {e}")
#         print("  使用单机模式继续训练...")
#         return False

# 在客户端文件中，移除降级代码
def setup(rank, world_size, port=29500):
    """真正的分布式设置"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)

    # 直接初始化，不try-catch
    dist.init_process_group(
        backend="gloo" if not torch.cuda.is_available() else "nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size
    )
    print(f"[Rank {rank}] 分布式初始化成功")
    return True


def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()


def load_client_data(dataset_dir, client_type='hsi', data_ratio=0.5):
    """
    加载客户端数据 - HSI客户端只加载HSI数据
    data_ratio: 客户端拥有的数据比例 (0-1)
    """
    print(f"[HSI Client] 加载 {client_type.upper()} 数据 (比例: {data_ratio})...")

    # 加载完整数据
    hsi_data = scio.loadmat(os.path.join(dataset_dir, 'TrTe/HSI_TrSet.mat'))['Data']
    lidar_data = scio.loadmat(os.path.join(dataset_dir, 'TrTe/LiDAR_TrSet.mat'))['Data']
    labels = scio.loadmat(os.path.join(dataset_dir, 'TrTe/Y_train.mat'))['Data']

    # 数据划分：每个客户端只获取部分数据
    total_samples = len(hsi_data)
    client_samples = int(total_samples * data_ratio)

    # 模拟数据分布：HSI客户端有完整的HSI数据，但没有LiDAR数据
    if client_type == 'hsi':
        # HSI数据：前50%
        client_hsi = hsi_data[:client_samples]
        # LiDAR数据：零（模拟数据隐私）
        client_lidar = np.zeros_like(lidar_data[:client_samples])
    else:  # lidar
        # HSI数据：零
        client_hsi = np.zeros_like(hsi_data[:client_samples])
        # LiDAR数据：后50%
        client_lidar = lidar_data[-client_samples:]

    client_labels = labels[:client_samples]

    print(f"[HSI Client] 数据加载完成:")
    print(f"  HSI: {client_hsi.shape}")
    print(f"  LiDAR: {client_lidar.shape}")
    print(f"  Labels: {client_labels.shape}")

    return client_hsi, client_lidar, client_labels


def train_client(rank, world_size, args):
    """HSI客户端训练函数"""
    print(f"\n{'=' * 60}")
    print(f"HSI Client - Rank {rank}/{world_size}")
    print(f"{'=' * 60}")

    # 设置分布式
    setup(rank, world_size, port=args.port)

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[HSI Client] 使用设备: {device}")

    # 获取数据集配置
    config = dataset_config[args.dataset]
    dataset_dir = config['dataset_dir']

    # 加载数据
    hsi_data, lidar_data, labels = load_client_data(
        dataset_dir,
        client_type='hsi',
        data_ratio=args.data_ratio
    )

    # 转换为tensor
    hsi_tensor = torch.tensor(hsi_data, dtype=torch.float32)
    lidar_tensor = torch.tensor(lidar_data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    # 创建数据加载器
    from torch.utils.data import DataLoader
    train_dataset = random_mini_batches_standardtwoModality(
        hsi_tensor, lidar_tensor, labels_tensor
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    # 创建模型
    model = Fed_Fusion(
        config['hsi_n_feature'],
        config['lidar_n_feature'],
        config['num_class']
    ).to(device)

    # 优化器和损失函数
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    criterion = Fed_Fusion_Loss(
        torch.ones(config['num_class']).to(device)
    ).to(device)

    print(f"[HSI Client] 开始联邦训练...")
    print(f"  本地epochs: {args.local_epochs}")
    print(f"  批次大小: {args.batch_size}")

    # 联邦训练循环
    for round in range(args.fed_rounds):
        if rank == 0:
            print(f"\n[HSI Client] 联邦轮次 {round + 1}/{args.fed_rounds}")

        model.train()
        round_loss = 0
        round_correct = 0
        round_total = 0

        # 本地训练多个epoch
        for epoch in range(args.local_epochs):
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0

            for batch_idx, (hsi_batch, lidar_batch, labels_batch) in enumerate(train_loader):
                hsi_batch = hsi_batch.to(device)
                lidar_batch = lidar_batch.to(device)
                labels_batch = labels_batch.to(device)
                target = torch.argmax(labels_batch, dim=1)

                # 前向传播
                outputs = model(hsi_batch, lidar_batch, with_svd=args.with_svd)
                loss = criterion(outputs, target)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()

                # 分布式梯度同步（模拟联邦学习的参数交换）
                # 分布式梯度同步（模拟联邦学习的参数交换）
                try:
                    if dist.is_initialized():
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                # 只同步共享层，模拟联邦学习的部分参数共享
                                shared_layers = ['cross_a', 'cross_b', 'conv5', 'bn5', 'conv6', 'bn6', 'conv7']
                                if any(shared in name for shared in shared_layers):
                                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG)
                except:
                    pass  # 如果分布式不可用，跳过同步

                optimizer.step()

                # 统计
                epoch_loss += loss.item()
                pred = torch.argmax(outputs[0], dim=1)
                epoch_correct += (pred == target).sum().item()
                epoch_total += target.size(0)

            # 本地epoch统计
            avg_epoch_loss = epoch_loss / len(train_loader)
            epoch_accuracy = 100 * epoch_correct / epoch_total

            if rank == 0 and (epoch + 1) % 2 == 0:
                print(f"[HSI Client]  本地Epoch {epoch + 1}/{args.local_epochs}: "
                      f"Loss: {avg_epoch_loss:.4f}, Acc: {epoch_accuracy:.2f}%")

            round_loss += epoch_loss
            round_correct += epoch_correct
            round_total += epoch_total

        # 联邦轮次统计
        avg_round_loss = round_loss / (args.local_epochs * len(train_loader))
        round_accuracy = 100 * round_correct / round_total

        if rank == 0:
            print(f"[HSI Client] 联邦轮次 {round + 1} 完成: "
                  f"平均Loss: {avg_round_loss:.4f}, 平均Acc: {round_accuracy:.2f}%")

        # 模拟联邦聚合：同步模型参数
        try:
            if args.sync_interval > 0 and (round + 1) % args.sync_interval == 0:
                if rank == 0:
                    print(f"[HSI Client] 同步模型参数...")

                # 同步所有参数（如果分布式可用）
                if dist.is_initialized():
                    for param in model.parameters():
                        dist.all_reduce(param.data, op=dist.ReduceOp.AVG)
                else:
                    print("[HSI Client] 单机模式，跳过参数同步")
        except:
            print("[HSI Client] 参数同步失败，继续训练")

    # 保存客户端模型
    if rank == 0:
        model_path = f"./output/client_hsi_model_round{args.fed_rounds}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"\n[HSI Client] ✅ 客户端模型已保存: {model_path}")

    # 清理
    def cleanup():
        """清理分布式环境"""
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
                print("[HSI Client] 分布式环境已清理")
        except:
            pass  # 如果分布式未初始化，忽略错误


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='HSI联邦学习客户端')
    parser.add_argument('--dataset', type=str, default='trento',
                        help='数据集名称')
    parser.add_argument('--fed_rounds', type=int, default=10,
                        help='联邦训练轮数')
    parser.add_argument('--local_epochs', type=int, default=5,
                        help='每轮本地训练epoch数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='权重衰减')
    parser.add_argument('--data_ratio', type=float, default=0.5,
                        help='客户端数据比例 (0-1)')
    parser.add_argument('--with_svd', type=int, default=0,
                        help='是否使用SVD压缩')
    parser.add_argument('--sync_interval', type=int, default=2,
                        help='参数同步间隔轮数')
    parser.add_argument('--world_size', type=int, default=2,
                        help='总进程数')
    parser.add_argument('--port', type=int, default=29500,
                        help='分布式端口')

    args = parser.parse_args()

    print("=" * 60)
    print("HSI联邦学习客户端启动")
    print("=" * 60)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    # 创建输出目录
    os.makedirs("./output", exist_ok=True)

    # 启动分布式训练
    try:
        mp.spawn(
            train_client,
            args=(args.world_size, args),
            nprocs=args.world_size,
            join=True
        )
        print("\n✅ HSI客户端训练完成！")
    except Exception as e:
        print(f"\n❌ HSI客户端训练失败: {e}")


if __name__ == "__main__":
    main()