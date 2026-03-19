"""
测试两个客户端保存的模型
"""
import torch
import numpy as np
from model import Fed_Fusion
from config import dataset_config

print("=" * 60)
print("测试客户端模型")
print("=" * 60)

# 加载配置
dataset_name = 'trento'
config = dataset_config[dataset_name]

# 测试HSI客户端模型
print("\n1. 测试HSI客户端模型...")
try:
    model_hsi = Fed_Fusion(
        config['hsi_n_feature'],
        config['lidar_n_feature'],
        config['num_class']
    )
    model_hsi.load_state_dict(torch.load('./output/client_hsi_model_round1.pth', map_location='cpu'))
    print("✅ HSI客户端模型加载成功")

    # 测试推理
    x1 = torch.randn(2, config['hsi_n_feature'] * 7 * 7).view(-1, config['hsi_n_feature'], 7, 7)
    x2 = torch.zeros(2, config['lidar_n_feature'] * 7 * 7).view(-1, config['lidar_n_feature'], 7, 7)  # LiDAR数据为零

    with torch.no_grad():
        outputs = model_hsi(x1, x2, with_svd=False)
    print(f"  输出形状: {outputs[0].shape}")

except Exception as e:
    print(f"❌ HSI客户端模型加载失败: {e}")

# 测试LiDAR客户端模型
print("\n2. 测试LiDAR客户端模型...")
try:
    model_lidar = Fed_Fusion(
        config['hsi_n_feature'],
        config['lidar_n_feature'],
        config['num_class']
    )
    model_lidar.load_state_dict(torch.load('./output/client_lidar_model_round1.pth', map_location='cpu'))
    print("✅ LiDAR客户端模型加载成功")

    # 测试推理
    x1 = torch.zeros(2, config['hsi_n_feature'] * 7 * 7).view(-1, config['hsi_n_feature'], 7, 7)  # HSI数据为零
    x2 = torch.randn(2, config['lidar_n_feature'] * 7 * 7).view(-1, config['lidar_n_feature'], 7, 7)

    with torch.no_grad():
        outputs = model_lidar(x1, x2, with_svd=False)
    print(f"  输出形状: {outputs[0].shape}")

except Exception as e:
    print(f"❌ LiDAR客户端模型加载失败: {e}")

print("\n" + "=" * 60)
print("模型测试完成")
print("=" * 60)