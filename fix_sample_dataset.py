import numpy as np
import scipy.io as scio
import os
from py_utils import sampling, generate_cube, mapset_normalization
from config import dataset_config

# TODO: 这里修改数据集
dataset_name = 'trento'

if dataset_name in dataset_config:
    config = dataset_config[dataset_name]
    dataset_dir = config['dataset_dir']
    concate_pixel = config['concate_pixel']
    class_labels = config['class_labels']
else:
    raise ValueError("Unknown dataset")

num_class = max(class_labels.keys())

print(f"处理 {dataset_name} 数据集...")
print(f"数据目录: {dataset_dir}")

# 加载数据（使用正确的变量名）
HSI_MapSet = scio.loadmat(os.path.join(dataset_dir, f'HSI.mat'))['HSI_Trento']
# Trento 数据集文件名为 Lidar.mat（小写 d），变量名为 Lidar_Trento
LiDAR_MapSet = scio.loadmat(os.path.join(dataset_dir, f'Lidar.mat'))['Lidar_Trento']
TrLabel = scio.loadmat(os.path.join(dataset_dir, f'TrLabel.mat'))['TrLabel']
TeLabel = scio.loadmat(os.path.join(dataset_dir, f'TeLabel.mat'))['TeLabel']

print(f"✅ 数据加载成功!")
print(f"  HSI_Trento.shape = {HSI_MapSet.shape}")
print(f"  Lidar_Trento.shape = {LiDAR_MapSet.shape}")
print(f"  TrLabel.shape = {TrLabel.shape}")
print(f"  TeLabel.shape = {TeLabel.shape}")

# LiDAR数据需要扩展维度 (166, 600) -> (166, 600, 1)
if LiDAR_MapSet.ndim == 2:
    LiDAR_MapSet = LiDAR_MapSet[:, :, np.newaxis]
    print(f"  LiDAR扩展维度后: {LiDAR_MapSet.shape}")

os.makedirs(os.path.join(dataset_dir, f'TrTe'), exist_ok = True)
f = open(os.path.join(dataset_dir, f'TrTe/info.txt'), 'w')

print('HSI_MapSet.shape = ', HSI_MapSet.shape, file = f)
print('LiDAR_MapSet.shape = ', LiDAR_MapSet.shape, file = f)
print('TrLabel.shape = ', TrLabel.shape, file = f)
print('TeLabel.shape = ', TeLabel.shape, file = f)
print("Unique values in TrLabel:", np.unique(TrLabel), file = f)
print("Unique values in TeLabel:", np.unique(TeLabel), file = f)

# 检查标签数据格式
TrLabel = TrLabel.astype(np.int8)
TeLabel = TeLabel.astype(np.int8)

# HSI数据标准化
HSI_MapSet, row, col, _ = mapset_normalization(
    mapset=HSI_MapSet,
    concate_pixel=concate_pixel,
    type='HSI'
)

# LiDAR数据标准化 - 注意：n_feature=1会导致维度压缩
LiDAR_MapSet, row, col, _ = mapset_normalization(
    mapset=LiDAR_MapSet,
    concate_pixel=concate_pixel,
    type='LiDAR',
    n_feature=1
)

# ========== 关键修复：确保标准化后LiDAR是3D ==========
if LiDAR_MapSet.ndim == 2:
    LiDAR_MapSet = LiDAR_MapSet[:, :, np.newaxis]
    print(f"标准化后LiDAR维度扩展: {LiDAR_MapSet.shape}")
    print(f"标准化后LiDAR维度扩展: {LiDAR_MapSet.shape}", file=f)

print(f"标准化后 HSI形状: {HSI_MapSet.shape}", file=f)
print(f"标准化后 LiDAR形状: {LiDAR_MapSet.shape}", file=f)

TrLabel = TrLabel.reshape(row * col)
TeLabel = TeLabel.reshape(row * col)

Trlabel_flat = TrLabel.flatten()
Telabel_flat = TeLabel.flatten()

undefined_value = min(np.unique(Trlabel_flat))

# 统计每个类别的数量,忽略未标注的像素（-1）
Tr_counts = np.bincount(Trlabel_flat[Trlabel_flat != undefined_value])
Te_counts = np.bincount(Telabel_flat[Telabel_flat != undefined_value])

# 打印统计结果
print("TrLabel 中每个类别的数量:", file = f)
for i in range(1, len(Tr_counts)):
    print(f"类别 {i}: {Tr_counts[i]} 个", file = f)

print("\nTelabel 中每个类别的数量:", file = f)
for i in range(1, len(Te_counts)):
    print(f"类别 {i}: {Te_counts[i]} 个", file = f)

if dataset_name == 'houston13':
    f.close()
    print('houston13 can not be sampled, because we only have LiDAR.map with 1 band')
    exit()

patch_size = 7
train_idx, test_idx = sampling(TrLabel, TeLabel)
# 打印生成的索引
print("训练集索引数量:", len(train_idx), file = f)
print("测试集索引数量:", len(test_idx), file = f)

HSI_TrSet, Y_train = generate_cube(train_idx, HSI_MapSet, TrLabel.reshape(row * col), patch_size, row, col, num_class = num_class, shuffle=False)
HSI_TeSet, Y_test = generate_cube(test_idx, HSI_MapSet, TeLabel.reshape(row * col), patch_size, row, col, num_class = num_class, shuffle=False)
LiDAR_TrSet, _ = generate_cube(train_idx, LiDAR_MapSet, TrLabel.reshape(row * col), patch_size, row, col, num_class = num_class, shuffle=False)
LiDAR_TeSet, _ = generate_cube(test_idx, LiDAR_MapSet, TeLabel.reshape(row * col), patch_size, row, col, num_class = num_class, shuffle=False)

scio.savemat(os.path.join(dataset_dir, f'TrTe/HSI_TrSet.mat'), {'Data': HSI_TrSet.astype(np.float32)})
scio.savemat(os.path.join(dataset_dir, f'TrTe/LiDAR_TrSet.mat'), {'Data': LiDAR_TrSet.astype(np.float32)})
scio.savemat(os.path.join(dataset_dir, f'TrTe/Y_train.mat'), {'Data': Y_train})
scio.savemat(os.path.join(dataset_dir, f'TrTe/HSI_TeSet.mat'), {'Data': HSI_TeSet.astype(np.float32)})
scio.savemat(os.path.join(dataset_dir, f'TrTe/LiDAR_TeSet.mat'), {'Data': LiDAR_TeSet.astype(np.float32)})
scio.savemat(os.path.join(dataset_dir, f'TrTe/Y_test.mat'), {'Data': Y_test})

print('HSI_TrSet.shape = ', HSI_TrSet.shape, file = f)
print('HSI_TeSet.shape = ', HSI_TeSet.shape, file = f)
print('LiDAR_TrSet.shape = ', LiDAR_TrSet.shape, file = f)
print('LiDAR_TeSet.shape = ', LiDAR_TeSet.shape, file = f)
print('Y_train.shape = ', Y_train.shape, file = f)
print('Y_test.shape = ', Y_test.shape, file = f)

print("\n✅ 数据预处理完成!")
print(f"预处理结果保存在: {dataset_dir}/TrTe/")
print(f"详细信息见: {dataset_dir}/TrTe/info.txt")
f.close()