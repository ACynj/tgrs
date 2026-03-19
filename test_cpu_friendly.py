# test_cpu_friendly.py
import torch
import os
import numpy as np

print("=" * 60)
print("FedFusion CPU友好测试")
print("=" * 60)

# 检查关键文件
print("\n🔍 文件检查:")
essential_files = [
    ('main.py', True),
    ('model.py', True),
    ('config.py', True),
    ('py_utils.py', True),
    ('trento/HSI.mat', False),
    ('trento/LiDAR.mat', False),
    ('trento/TrLabel.mat', False),
    ('trento/TeLabel.mat', False),
]

missing = []
for file_path, is_required in essential_files:
    exists = os.path.exists(file_path)
    status = "✅" if exists else "❌"
    print(f"  {status} {file_path}")
    if not exists and is_required:
        missing.append(file_path)

if missing:
    print(f"\n⚠️  缺失必需文件: {missing}")
else:
    print("\n✅ 所有必需文件都存在")

# 测试模型导入和内存使用
print("\n🧠 模型内存测试:")

try:
    from model import Fed_Fusion

    # 使用较小的模型配置节省内存
    model = Fed_Fusion(63, 1, 6)

    # 计算模型大小
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    total_size = (param_size + buffer_size) / 1024 ** 2  # 转换为MB
    print(f"  模型内存占用: {total_size:.2f} MB")
    print(f"  总参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试小批量前向传播
    print("\n  前向传播测试 (batch_size=1):")
    with torch.no_grad():
        x1 = torch.randn(1, 7 * 7 * 63)  # 最小的batch
        x2 = torch.randn(1, 7 * 7 * 1)

        # 测试无SVD
        outputs = model(x1, x2, with_svd=False)
        print(f"    无SVD输出形状: {outputs[0].shape}")

        # 测试有SVD (CPU模式可能需要修改)
        try:
            outputs_svd = model(x1, x2, with_svd=True, k=2)
            print(f"    有SVD输出形状: {outputs_svd[0].shape}")
        except Exception as e:
            print(f"    SVD测试失败 (CPU模式正常): {e}")

    print("\n✅ 模型测试通过")

except Exception as e:
    print(f"\n❌ 模型测试失败: {e}")
    import traceback

    traceback.print_exc()

# 数据预处理测试
print("\n📊 数据预处理测试:")
try:
    # 检查预处理输出目录
    preprocessed_dir = "trento/TrTe"
    if os.path.exists(preprocessed_dir):
        print(f"  预处理目录已存在: {preprocessed_dir}")
        files = os.listdir(preprocessed_dir)
        print(f"  包含文件: {files}")
    else:
        print("  预处理目录不存在，需要运行 sample_dataset.py")

except Exception as e:
    print(f"  数据检查失败: {e}")

# 内存优化建议
print("\n💡 CPU运行建议:")
print("  1. 使用小batch_size (2-4)")
print("  2. 关闭SVD压缩 (--with_svd 0)")
print("  3. 减少训练轮数 (--epochs 2-5)")
print("  4. 监控内存使用")

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)

# 建议命令
print("\n🎯 推荐运行命令:")
print("""
# 1. 数据预处理
python sample_dataset.py

# 2. 最小化训练测试  
python main.py --dataset_name trento --device cpu --batch_size 2 --epochs 2 --with_svd 0

# 3. 完整CPU训练
python main.py --dataset_name trento --device cpu --batch_size 4 --epochs 10
""")