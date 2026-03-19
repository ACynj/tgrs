"""
轻量级分布式功能测试
验证基本的分布式通信是否工作
"""
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os


def test_all_reduce(rank, world_size):
    """测试all_reduce操作"""
    print(f"[Rank {rank}] 启动...")

    # 初始化
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # 测试all_reduce
    tensor = torch.tensor([rank + 1.0])  # rank 0: [1.0], rank 1: [2.0]
    print(f"[Rank {rank}] 初始值: {tensor.item()}")

    # 求和
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"[Rank {rank}] 求和后: {tensor.item()} (应为3.0)")

    # 求平均
    tensor = torch.tensor([rank + 1.0])
    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    print(f"[Rank {rank}] 平均后: {tensor.item()} (应为1.5)")

    dist.destroy_process_group()
    print(f"[Rank {rank}] 测试完成")


def test_model_parallel(rank, world_size):
    """测试模型并行"""
    print(f"\n[Rank {rank}] 测试模型并行...")

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # 创建简单模型
    model = torch.nn.Linear(10, 5)

    # 模拟参数同步
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.AVG)

    print(f"[Rank {rank}] 参数同步完成")
    dist.destroy_process_group()


def main_test():
    """主测试函数"""
    print("=" * 60)
    print("分布式功能测试")
    print("=" * 60)

    world_size = 2

    print("\n1. 测试all_reduce通信...")
    try:
        mp.spawn(
            test_all_reduce,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
        print("✅ all_reduce测试通过")
    except Exception as e:
        print(f"❌ all_reduce测试失败: {e}")

    print("\n2. 测试模型并行...")
    try:
        mp.spawn(
            test_model_parallel,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
        print("✅ 模型并行测试通过")
    except Exception as e:
        print(f"❌ 模型并行测试失败: {e}")

    print("\n" + "=" * 60)
    print("分布式测试完成")
    print("=" * 60)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main_test()