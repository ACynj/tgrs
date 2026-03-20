"""
Houston 2013 数据集预处理脚本
- 支持真实 LiDAR 加载（缺失时回退到 HSI 第一波段并警告）
- LiDAR 官方下载: http://hyperspectral.ee.uh.edu/?page_id=459
- 常见文件名: Houston_lidar.mat (349x1905 高程)
"""
import os
import argparse
import numpy as np
import scipy.io as scio
from py_utils import sampling, generate_cube
from py_utils import mapset_normalization
from config import dataset_config

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--split_mode', type=str, default='fixed',
                    choices=['fixed', 'random_95_5'],
                    help='fixed: 使用 houston_gt_tr/te 固定划分; random_95_5: 论文 95%% 训练 5%% 验证')
args = parser.parse_args()

# 设置数据集
dataset_name = 'houston13'
config = dataset_config[dataset_name]
dataset_dir = config['dataset_dir']  # 使用 config 相对路径
concate_pixel = config['concate_pixel']
num_class = 15  # 完整 15 类

print(f"处理 Houston 完整数据集...")
print(f"数据目录: {dataset_dir}")
print(f"划分模式: {args.split_mode}")

# 加载 HSI
print("\n读取 houston.mat...")
data = scio.loadmat(os.path.join(dataset_dir, 'houston.mat'))
HSI_MapSet = data['houston']  # (349, 1905, 144)
print(f"HSI 形状: {HSI_MapSet.shape}")

# 加载 LiDAR：依次尝试多个可能的文件与变量名
LiDAR_MapSet = None
lidar_candidates = [
    ('houston_lidar.mat', ['houston_lidar', 'lidar', 'Lidar_Houston']),
    ('lidar.mat', ['lidar', 'Lidar_Houston', 'Houston_lidar']),
    ('Houston_lidar.mat', ['Houston_lidar', 'lidar', 'houston_lidar']),
]
for fname, var_names in lidar_candidates:
    fpath = os.path.join(dataset_dir, fname)
    if os.path.exists(fpath):
        mat = scio.loadmat(fpath)
        for v in var_names:
            if v in mat and not v.startswith('__'):
                LiDAR_MapSet = mat[v]
                print(f"✅ 加载真实 LiDAR: {fname} 变量 '{v}', 形状 {LiDAR_MapSet.shape}")
                break
        if LiDAR_MapSet is not None:
            break

if LiDAR_MapSet is None:
    LiDAR_MapSet = HSI_MapSet[:, :, :1]
    print("⚠️  WARNING: 未找到真实 LiDAR 文件，使用 HSI 第一波段替代。")
    print("   论文效果需真实 LiDAR 高程数据，请从 http://hyperspectral.ee.uh.edu/?page_id=459 下载")
    print("   将 Houston_lidar.mat 或 lidar.mat 放入 DATASET/newhuston/ 后重新运行")
else:
    # 兼容 (1, H, W) 格式，转为 (H, W)
    if LiDAR_MapSet.ndim == 3 and LiDAR_MapSet.shape[0] == 1:
        LiDAR_MapSet = LiDAR_MapSet.squeeze(0)
        print(f"  LiDAR 从 (1,H,W) 转为 (H,W): {LiDAR_MapSet.shape}")
    if LiDAR_MapSet.ndim == 2:
        LiDAR_MapSet = LiDAR_MapSet[:, :, np.newaxis]
        print(f"  LiDAR 扩展维度后: {LiDAR_MapSet.shape}")

# 加载标签
print("\n读取 houston_gt.mat...")
gt_data = scio.loadmat(os.path.join(dataset_dir, 'houston_gt.mat'))
TrLabel = gt_data['houston_gt_tr']
TeLabel = gt_data['houston_gt_te']
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

if LiDAR_MapSet.ndim == 2:
    LiDAR_MapSet = LiDAR_MapSet[:, :, np.newaxis]

print(f"\n标准化后 HSI 形状: {HSI_MapSet.shape}")
print(f"标准化后 LiDAR 形状: {LiDAR_MapSet.shape}", file=f_log)

patch_size = 7
TrLabel_flat = TrLabel.reshape(-1)
TeLabel_flat = TeLabel.reshape(-1)

if args.split_mode == 'random_95_5':
    # 论文 95% 训练 / 5% 验证：合并 Tr+Te 后按类别分层随机划分
    all_idx = []
    all_labels = []
    for i in range(row * col):
        if TrLabel_flat[i] > 0:
            all_idx.append(i)
            all_labels.append(TrLabel_flat[i])
        elif TeLabel_flat[i] > 0:
            all_idx.append(i)
            all_labels.append(TeLabel_flat[i])
    all_idx = np.array(all_idx)
    all_labels = np.array(all_labels)

    train_idx = []
    test_idx = []
    np.random.seed(42)
    for c in range(1, num_class + 1):
        mask = all_labels == c
        class_idx = all_idx[mask]
        np.random.shuffle(class_idx)
        n = len(class_idx)
        n_train = max(1, int(0.95 * n))
        train_idx.extend(class_idx[:n_train])
        test_idx.extend(class_idx[n_train:])

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    label_map = np.zeros(row * col, dtype=np.int8)
    for i in range(len(all_idx)):
        label_map[all_idx[i]] = all_labels[i]
    TrLabel_flat = np.zeros(row * col, dtype=np.int8)
    TeLabel_flat = np.zeros(row * col, dtype=np.int8)
    for idx in train_idx:
        TrLabel_flat[idx] = label_map[idx]
    for idx in test_idx:
        TeLabel_flat[idx] = label_map[idx]

    print(f"random_95_5: 训练 {len(train_idx)}, 测试 {len(test_idx)}")
else:
    train_idx, test_idx = sampling(TrLabel_flat, TeLabel_flat)

print(f"\n训练集索引数量: {len(train_idx)}", file=f_log)
print(f"测试集索引数量: {len(test_idx)}", file=f_log)

# 生成训练和测试数据
HSI_TrSet, Y_train = generate_cube(
    train_idx, HSI_MapSet, TrLabel_flat,
    patch_size, row, col, num_class=num_class, shuffle=False, augment=True
)
HSI_TeSet, Y_test = generate_cube(
    test_idx, HSI_MapSet, TeLabel_flat,
    patch_size, row, col, num_class=num_class, shuffle=False, augment=False
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
