import torch
import numpy as np
import os

# 设置环境变量（避免分布式错误）
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

# 禁用可能的分布式初始化错误
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['LOCAL_RANK'] = '0'

print("=" * 50)
print("FedFusion 模型测试")
print("=" * 50)

# 1. 导入模型
try:
    from model import Fed_Fusion

    print("✅ 成功导入 Fed_Fusion 模型")
except Exception as e:
    print(f"❌ 导入模型失败: {e}")
    exit(1)

# 2. 获取trento数据集参数
from config import dataset_config

dataset_name = 'trento'
config = dataset_config[dataset_name]

hsi_n_feature = config['hsi_n_feature']  # 63
lidar_n_feature = config['lidar_n_feature']  # 1
num_class = config['num_class']  # 6

print(f"\n📊 数据集参数:")
print(f"  HSI特征维度: {hsi_n_feature}")
print(f"  LiDAR特征维度: {lidar_n_feature}")
print(f"  类别数: {num_class}")

# 3. 创建模型（CPU模式）
device = torch.device('cpu')
print(f"\n🖥️  使用设备: {device}")

try:
    model = Fed_Fusion(hsi_n_feature, lidar_n_feature, num_class).to(device)
    print("✅ 模型创建成功")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

except Exception as e:
    print(f"❌ 模型创建失败: {e}")
    exit(1)

# 4. 测试前向传播
print(f"\n🧪 测试前向传播...")
batch_size = 2

# 创建模拟输入（与预处理后的数据形状匹配）
# 注意：预处理后每个样本是7x7的patch
x1 = torch.randn(batch_size, hsi_n_feature * 7 * 7).to(device)  # HSI: [2, 63*49]
x2 = torch.randn(batch_size, lidar_n_feature * 7 * 7).to(device)  # LiDAR: [2, 1*49]

print(f"  输入形状:")
print(f"    HSI: {x1.shape}")
print(f"    LiDAR: {x2.shape}")

try:
    # 重塑为模型期望的格式 [batch, channels, height, width]
    x1_reshaped = x1.view(-1, hsi_n_feature, 7, 7)
    x2_reshaped = x2.view(-1, lidar_n_feature, 7, 7)

    print(f"  重塑后形状:")
    print(f"    HSI: {x1_reshaped.shape}")
    print(f"    LiDAR: {x2_reshaped.shape}")

    # 前向传播（关闭SVD，关闭分布式）
    with torch.no_grad():
        outputs = model(x1_reshaped, x2_reshaped, with_svd=False)

    print(f"\n✅ 前向传播成功!")
    print(f"  主输出 (fusion1): {outputs[0].shape}")
    print(f"  辅助输出1 (fusion2): {outputs[1].shape}")
    print(f"  辅助输出2 (fusion3): {outputs[2].shape}")
    print(f"  L2正则项: {outputs[3]}")

    # 检查输出是否合理
    if outputs[0].shape[0] == batch_size and outputs[0].shape[1] == num_class:
        print(f"  输出维度正确: [batch_size={batch_size}, num_classes={num_class}]")
    else:
        print(f"⚠️  输出维度异常: {outputs[0].shape}")

except Exception as e:
    print(f"❌ 前向传播失败: {e}")
    print("\n可能需要修改 model.py 以支持单机模式...")

print("\n" + "=" * 50)
print("模型测试完成!")
print("=" * 50)