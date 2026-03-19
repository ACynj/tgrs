import os
import numpy as np
import scipy.io as scio
from py_utils import sampling, generate_cube
from py_utils import mapset_normalization
from config import dataset_config

# 设置数据集
dataset_name = 'houston13'
config = dataset_config[dataset_name]
dataset_dir = '/T20170026/zmx/FedFusion-TGRS/DATASET/newhuston'  # 新数据目录
concate_pixel = config['concate_pixel']
num_class = 15  # 完整 15 类

print(f"处理 Houston 完整数据集...")
print(f"数据目录: {dataset_dir}")

# 加载数据
print("\n读取 houston.mat...")
data = scio.loadmat(os.path.join(dataset_dir, 'houston.mat'))
HSI_MapSet = data['houston']  # (349, 1905, 144)
print(f"HSI 形状: {HSI_MapSet.shape}")

# Houston 的 LiDAR 数据需要单独下载？这里先用 HSI 的第一个波段模拟
LiDAR_MapSet = HSI_MapSet[:, :, :1]  # 取第一个波段作为 LiDAR
print(f"LiDAR 形状: {LiDAR_MapSet.shape}")

# 加载标签
print("\n读取 houston_gt.mat...")
gt_data = scio.loadmat(os.path.join(dataset_dir, 'houston_gt.mat'))
TrLabel = gt_data['houston_gt_tr']  # 训练标签
TeLabel = gt_data['houston_gt_te']  # 测试标签
print(f"训练标签形状: {TrLabel.shape}")
print(f"测试标签形状: {TeLabel.shape}")

# 统计类别分布
print("\n训练集类别分布:")
for i in range(1, 16):
    count = np.sum(TrLabel == i)
    if count > 0:
        print(f"  类别 {i}: {count} 个像素")

print("\n测试集类别分布:")
for i in range(1, 16):
    count = np.sum(TeLabel == i)
    if count > 0:
        print(f"  类别 {i}: {count} 个像素")

# 创建输出目录
os.makedirs(os.path.join(dataset_dir, 'TrTe'), exist_ok=True)
f_log = open(os.path.join(dataset_dir, 'TrTe/info.txt'), 'w')

# 数据标准化
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

print(f"\n标准化后 HSI 形状: {HSI_MapSet.shape}")
print(f"标准化后 LiDAR 形状: {LiDAR_MapSet.shape}", file=f_log)

patch_size = 7
TrLabel_flat = TrLabel.reshape(-1)
TeLabel_flat = TeLabel.reshape(-1)

train_idx, test_idx = sampling(TrLabel_flat, TeLabel_flat)

print(f"\n训练集索引数量: {len(train_idx)}", file=f_log)
print(f"测试集索引数量: {len(test_idx)}", file=f_log)

# 生成训练和测试数据
HSI_TrSet, Y_train = generate_cube(
    train_idx, HSI_MapSet, TrLabel_flat,
    patch_size, row, col, num_class=num_class, shuffle=False, augment=True  # 添加 augment=True
)
HSI_TeSet, Y_test = generate_cube(
    test_idx, HSI_MapSet, TeLabel_flat,
    patch_size, row, col, num_class=num_class, shuffle=False, augment=False  # 测试不用增强
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
out_dir = os.path.join(dataset_dir, 'TrTe')
scio.savemat(os.path.join(out_dir, 'HSI_TrSet.mat'), {'Data': HSI_TrSet.astype(np.float32)})
scio.savemat(os.path.join(out_dir, 'LiDAR_TrSet.mat'), {'Data': LiDAR_TrSet.astype(np.float32)})
scio.savemat(os.path.join(out_dir, 'Y_train.mat'), {'Data': Y_train})
scio.savemat(os.path.join(out_dir, 'HSI_TeSet.mat'), {'Data': HSI_TeSet.astype(np.float32)})
scio.savemat(os.path.join(out_dir, 'LiDAR_TeSet.mat'), {'Data': LiDAR_TeSet.astype(np.float32)})
scio.savemat(os.path.join(out_dir, 'Y_test.mat'), {'Data': Y_test})

print(f"\nHSI_TrSet.shape = {HSI_TrSet.shape}", file=f_log)
print(f"HSI_TeSet.shape = {HSI_TeSet.shape}", file=f_log)
print(f"LiDAR_TrSet.shape = {LiDAR_TrSet.shape}", file=f_log)
print(f"LiDAR_TeSet.shape = {LiDAR_TeSet.shape}", file=f_log)
print(f"Y_train.shape = {Y_train.shape}", file=f_log)
print(f"Y_test.shape = {Y_test.shape}", file=f_log)

print("\n✅ 数据预处理完成!")
print(f"预处理结果保存在: {out_dir}")
f_log.close()