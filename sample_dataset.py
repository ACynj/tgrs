import os
import h5py
import numpy as np
import scipy.io as scio
from py_utils import sampling, generate_cube
from py_utils import mapset_normalization
from config import dataset_config

# 设置数据集
dataset_name = 'houston13'

if dataset_name in dataset_config:
    config = dataset_config[dataset_name]
    dataset_dir = config['dataset_dir']
    concate_pixel = config['concate_pixel']
    class_labels = config['class_labels']
else:
    raise ValueError("Unknown dataset")

# 注意：这里用实际存在的类别数 7，不是配置文件中的 15
num_class = 7  # Houston13 实际只有 7 个类别有样本

print(f"处理 {dataset_name} 数据集...")
print(f"数据目录: {dataset_dir}")
print(f"实际类别数: {num_class}")

# 读取 Houston13.mat (HSI 和 LiDAR 数据)
print("\n读取 Houston13.mat...")
with h5py.File(os.path.join(dataset_dir, 'Houston13.mat'), 'r') as f:
    print("文件中的键:", list(f.keys()))

    # 读取 HSI 数据 (形状: 48, 954, 210)
    hsi_data = np.array(f['ori_data']).transpose(2, 1, 0)  # 转置为 (210, 954, 48)
    print(f"HSI 数据形状 (转置后): {hsi_data.shape}")

    # LiDAR 数据 - 从 HSI 中取第一个波段作为替代
    lidar_data = hsi_data[:, :, :1]  # 取第一个波段作为 LiDAR
    print(f"LiDAR 数据形状: {lidar_data.shape}")

# 读取 Houston13_7gt.mat (ground truth)
print("\n读取 Houston13_7gt.mat...")
with h5py.File(os.path.join(dataset_dir, 'Houston13_7gt.mat'), 'r') as f:
    print("文件中的键:", list(f.keys()))

    # 读取 ground truth
    gt = np.array(f['map']).transpose(1, 0)  # 转置为 (210, 954)
    print(f"GT 形状: {gt.shape}")

    # 统计实际存在的类别
    unique_labels = np.unique(gt)
    print(f"GT 所有唯一值: {unique_labels}")

    # 只保留 1-7 的类别，过滤掉其他
    valid_labels = [l for l in unique_labels if 1 <= l <= 7]
    print(f"有效类别 (1-7): {valid_labels}")

# 创建输出目录
os.makedirs(os.path.join(dataset_dir, 'TrTe'), exist_ok=True)
f_log = open(os.path.join(dataset_dir, 'TrTe/info.txt'), 'w')

# 数据标准化
HSI_MapSet = hsi_data  # (210, 954, 48)
LiDAR_MapSet = lidar_data  # (210, 954, 1)

HSI_MapSet, row, col, _ = mapset_normalization(
    mapset=HSI_MapSet,
    concate_pixel=concate_pixel,
    type='HSI'
)
LiDAR_MapSet, row, col, _ = mapset_normalization(
    mapset=LiDAR_MapSet,
    concate_pixel=concate_pixel,
    type='LiDAR',
    n_feature=1
)

print("\n类别分布:")
for label in range(1, 8):
    count = np.sum(gt == label)
    print(f"  类别 {label}: {count} 个像素")

# 划分训练集和测试集 (80% 训练, 20% 测试)
np.random.seed(42)

TrLabel = np.zeros_like(gt)
TeLabel = np.zeros_like(gt)

for label in range(1, 8):  # 只处理 1-7 类
    idx = np.where(gt == label)
    n_samples = len(idx[0])
    n_train = int(n_samples * 0.8)  # 80% 训练

    perm = np.random.permutation(n_samples)
    train_idx = (idx[0][perm[:n_train]], idx[1][perm[:n_train]])
    test_idx = (idx[0][perm[n_train:]], idx[1][perm[n_train:]])

    TrLabel[train_idx] = label
    TeLabel[test_idx] = label

print("\n训练集类别分布:", file=f_log)
for i in range(1, 8):
    count = np.sum(TrLabel == i)
    print(f"  类别 {i}: {count} 个样本", file=f_log)

print("\n测试集类别分布:", file=f_log)
for i in range(1, 8):
    count = np.sum(TeLabel == i)
    print(f"  类别 {i}: {count} 个样本", file=f_log)

patch_size = 7

# 将标签展平用于 sampling
TrLabel_flat = TrLabel.reshape(-1)
TeLabel_flat = TeLabel.reshape(-1)

train_idx, test_idx = sampling(TrLabel_flat, TeLabel_flat)

print(f"\n训练集索引数量: {len(train_idx)}", file=f_log)
print(f"测试集索引数量: {len(test_idx)}", file=f_log)

# 生成训练和测试数据
HSI_TrSet, Y_train = generate_cube(
    train_idx, HSI_MapSet, TrLabel_flat,
    patch_size, row, col, num_class=num_class, shuffle=False
)
HSI_TeSet, Y_test = generate_cube(
    test_idx, HSI_MapSet, TeLabel_flat,
    patch_size, row, col, num_class=num_class, shuffle=False
)
LiDAR_TrSet, _ = generate_cube(
    train_idx, LiDAR_MapSet, TrLabel_flat,
    patch_size, row, col, num_class=num_class, shuffle=False
)
LiDAR_TeSet, _ = generate_cube(
    test_idx, LiDAR_MapSet, TeLabel_flat,
    patch_size, row, col, num_class=num_class, shuffle=False
)

# 保存处理后的数据
scio.savemat(os.path.join(dataset_dir, 'TrTe/HSI_TrSet.mat'), {'Data': HSI_TrSet.astype(np.float32)})
scio.savemat(os.path.join(dataset_dir, 'TrTe/LiDAR_TrSet.mat'), {'Data': LiDAR_TrSet.astype(np.float32)})
scio.savemat(os.path.join(dataset_dir, 'TrTe/Y_train.mat'), {'Data': Y_train})
scio.savemat(os.path.join(dataset_dir, 'TrTe/HSI_TeSet.mat'), {'Data': HSI_TeSet.astype(np.float32)})
scio.savemat(os.path.join(dataset_dir, 'TrTe/LiDAR_TeSet.mat'), {'Data': LiDAR_TeSet.astype(np.float32)})
scio.savemat(os.path.join(dataset_dir, 'TrTe/Y_test.mat'), {'Data': Y_test})

print(f"\nHSI_TrSet.shape = {HSI_TrSet.shape}", file=f_log)
print(f"HSI_TeSet.shape = {HSI_TeSet.shape}", file=f_log)
print(f"LiDAR_TrSet.shape = {LiDAR_TrSet.shape}", file=f_log)
print(f"LiDAR_TeSet.shape = {LiDAR_TeSet.shape}", file=f_log)
print(f"Y_train.shape = {Y_train.shape}", file=f_log)
print(f"Y_test.shape = {Y_test.shape}", file=f_log)

print("\n✅ 数据预处理完成!")
print(f"预处理结果保存在: {dataset_dir}/TrTe/")
f_log.close()