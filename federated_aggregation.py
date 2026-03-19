"""
联邦平均聚合算法
将HSI和LiDAR客户端的模型聚合为全局模型
"""
import torch
import numpy as np
from model import Fed_Fusion
from config import dataset_config


def load_client_models():
    """加载两个客户端模型"""
    print("=" * 60)
    print("联邦模型聚合")
    print("=" * 60)

    # 配置
    dataset_name = 'trento'
    config = dataset_config[dataset_name]

    # 创建模型结构
    print("创建模型结构...")
    model_hsi = Fed_Fusion(
        config['hsi_n_feature'],
        config['lidar_n_feature'],
        config['num_class']
    )

    model_lidar = Fed_Fusion(
        config['hsi_n_feature'],
        config['lidar_n_feature'],
        config['num_class']
    )

    # 加载客户端模型
    print("\n加载客户端模型...")
    try:
        model_hsi.load_state_dict(torch.load('./output/client_hsi_model_round2.pth', map_location='cpu'))
        print("✅ HSI客户端模型加载成功")
    except Exception as e:
        print(f"❌ HSI客户端模型加载失败: {e}")
        return None, None

    try:
        model_lidar.load_state_dict(torch.load('./output/client_lidar_model_round2.pth', map_location='cpu'))
        print("✅ LiDAR客户端模型加载成功")
    except Exception as e:
        print(f"❌ LiDAR客户端模型加载失败: {e}")
        return None, None

    return model_hsi, model_lidar


def federated_average(model_hsi, model_lidar, weights=None):
    """
    联邦平均聚合
    weights: 客户端权重列表 [w_hsi, w_lidar]，默认为[0.5, 0.5]
    """
    if weights is None:
        weights = [0.5, 0.5]  # 默认等权重

    print(f"\n执行联邦平均聚合 (权重: HSI={weights[0]}, LiDAR={weights[1]})...")

    # 创建全局模型
    dataset_name = 'trento'
    config = dataset_config[dataset_name]
    global_model = Fed_Fusion(
        config['hsi_n_feature'],
        config['lidar_n_feature'],
        config['num_class']
    )

    global_dict = global_model.state_dict()
    hsi_dict = model_hsi.state_dict()
    lidar_dict = model_lidar.state_dict()

    # 联邦平均：weighted average
    for key in global_dict.keys():
        global_dict[key] = weights[0] * hsi_dict[key] + weights[1] * lidar_dict[key]

    global_model.load_state_dict(global_dict)

    print("✅ 联邦平均聚合完成")
    return global_model


def test_federated_model(global_model, test_mode='both'):
    """
    测试联邦聚合模型在不同模态下的性能
    test_mode: 'hsi_only', 'lidar_only', 'both'
    """
    print(f"\n测试联邦模型 (模式: {test_mode})...")

    # 配置
    dataset_name = 'trento'
    config = dataset_config[dataset_name]

    # 创建测试输入
    batch_size = 4

    if test_mode == 'hsi_only':
        # 只有HSI数据（模拟HSI客户端）
        x1 = torch.randn(batch_size, config['hsi_n_feature'] * 7 * 7).view(-1, config['hsi_n_feature'], 7, 7)
        x2 = torch.zeros(batch_size, config['lidar_n_feature'] * 7 * 7).view(-1, config['lidar_n_feature'], 7, 7)
    elif test_mode == 'lidar_only':
        # 只有LiDAR数据（模拟LiDAR客户端）
        x1 = torch.zeros(batch_size, config['hsi_n_feature'] * 7 * 7).view(-1, config['hsi_n_feature'], 7, 7)
        x2 = torch.randn(batch_size, config['lidar_n_feature'] * 7 * 7).view(-1, config['lidar_n_feature'], 7, 7)
    else:  # 'both'
        # 两种数据都有（模拟集中式训练）
        x1 = torch.randn(batch_size, config['hsi_n_feature'] * 7 * 7).view(-1, config['hsi_n_feature'], 7, 7)
        x2 = torch.randn(batch_size, config['lidar_n_feature'] * 7 * 7).view(-1, config['lidar_n_feature'], 7, 7)

    # 前向传播
    with torch.no_grad():
        outputs = global_model(x1, x2, with_svd=False)

    print(f"  输出形状: {outputs[0].shape}")
    print(f"  预测值范围: [{outputs[0].min():.4f}, {outputs[0].max():.4f}]")

    # 模拟分类结果
    predictions = torch.argmax(outputs[0], dim=1)
    print(f"  预测类别: {predictions.tolist()}")

    return outputs


def compare_models(model_hsi, model_lidar, global_model):
    """比较三个模型的参数差异"""
    print("\n" + "=" * 60)
    print("模型参数比较")
    print("=" * 60)

    # 计算参数差异
    hsi_params = dict(model_hsi.named_parameters())
    lidar_params = dict(model_lidar.named_parameters())
    global_params = dict(global_model.named_parameters())

    print("\n关键层参数差异分析:")

    key_layers = ['conv1_a.weight', 'conv1_b.weight', 'cross_a.conv.weight', 'conv7.weight']

    for layer in key_layers:
        if layer in hsi_params and layer in lidar_params:
            hsi_val = hsi_params[layer].data.numpy().flatten()
            lidar_val = lidar_params[layer].data.numpy().flatten()
            global_val = global_params[layer].data.numpy().flatten()

            # 计算差异
            hsi_lidar_diff = np.mean(np.abs(hsi_val - lidar_val))
            hsi_global_diff = np.mean(np.abs(hsi_val - global_val))
            lidar_global_diff = np.mean(np.abs(lidar_val - global_val))

            print(f"\n{layer}:")
            print(f"  HSI-LiDAR差异: {hsi_lidar_diff:.6f}")
            print(f"  HSI-全局差异: {hsi_global_diff:.6f}")
            print(f"  LiDAR-全局差异: {lidar_global_diff:.6f}")


def save_global_model(global_model, filename='global_model_federated.pth'):
    """保存联邦聚合模型"""
    model_path = f"./output/{filename}"
    torch.save(global_model.state_dict(), model_path)
    print(f"\n✅ 联邦全局模型已保存: {model_path}")
    return model_path


def main():
    """主函数"""

    # 1. 加载客户端模型
    model_hsi, model_lidar = load_client_models()
    if model_hsi is None or model_lidar is None:
        print("❌ 无法加载客户端模型")
        return

    # 2. 联邦平均聚合
    print("\n" + "=" * 60)
    print("联邦平均聚合")
    print("=" * 60)

    # 方案1：等权重聚合
    print("\n方案1: 等权重聚合 (HSI: 0.5, LiDAR: 0.5)")
    global_model_equal = federated_average(model_hsi, model_lidar, weights=[0.5, 0.5])

    # 方案2：根据准确率加权
    print("\n方案2: 根据准确率加权聚合")
    # HSI: 96.14%, LiDAR: 93.72%
    acc_hsi = 96.14
    acc_lidar = 93.72
    total_acc = acc_hsi + acc_lidar
    weight_hsi = acc_hsi / total_acc
    weight_lidar = acc_lidar / total_acc
    print(f"  权重计算: HSI={weight_hsi:.3f}, LiDAR={weight_lidar:.3f}")
    global_model_weighted = federated_average(model_hsi, model_lidar,
                                              weights=[weight_hsi, weight_lidar])

    # 3. 测试联邦模型
    print("\n" + "=" * 60)
    print("联邦模型测试")
    print("=" * 60)

    print("\n1. 测试等权重聚合模型:")
    test_federated_model(global_model_equal, test_mode='both')
    test_federated_model(global_model_equal, test_mode='hsi_only')
    test_federated_model(global_model_equal, test_mode='lidar_only')

    print("\n2. 测试加权聚合模型:")
    test_federated_model(global_model_weighted, test_mode='both')

    # 4. 比较模型差异
    compare_models(model_hsi, model_lidar, global_model_equal)

    # 5. 保存模型
    save_global_model(global_model_equal, 'global_model_equal_weights.pth')
    save_global_model(global_model_weighted, 'global_model_weighted.pth')

    print("\n" + "=" * 60)
    print("联邦聚合完成！")
    print("=" * 60)
    print("\n📊 总结:")
    print(f"  1. HSI客户端准确率: {acc_hsi:.2f}%")
    print(f"  2. LiDAR客户端准确率: {acc_lidar:.2f}%")
    print(f"  3. 等权重聚合模型已保存")
    print(f"  4. 加权聚合模型已保存")
    print(f"\n💡 下一步: 可以使用全局模型进行推理或继续联邦训练")


if __name__ == "__main__":
    main()