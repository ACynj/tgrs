"""
使用 rs-fusion-datasets 下载 Houston 2013 真实 LiDAR (DSM) 数据
安装: pip install rs-fusion-datasets
"""
import os
import numpy as np
import scipy.io as scio
from config import dataset_config

try:
    from rs_fusion_datasets import fetch_houston2013
except ImportError:
    print("请先安装: pip install rs-fusion-datasets")
    raise

dataset_dir = dataset_config['houston13']['dataset_dir']
os.makedirs(dataset_dir, exist_ok=True)

print("正在从 rs-fusion-datasets 下载 Houston 2013 数据...")
hsi, dsm, train_label, test_label, info = fetch_houston2013()

print(f"HSI 形状: {hsi.shape}")
print(f"DSM (LiDAR) 形状: {dsm.shape}")

# 保存 LiDAR 为 houston_lidar.mat，变量名 houston_lidar
if hasattr(dsm, 'toarray'):
    dsm_arr = dsm.toarray()
elif hasattr(dsm, 'todense'):
    dsm_arr = np.asarray(dsm.todense())
else:
    dsm_arr = np.asarray(dsm)
dsm_arr = np.asarray(dsm_arr, dtype=np.float32)

lidar_path = os.path.join(dataset_dir, 'houston_lidar.mat')
scio.savemat(lidar_path, {'houston_lidar': dsm_arr})
print(f"✅ LiDAR 已保存: {lidar_path}")

# 若不存在 houston.mat / houston_gt.mat，一并保存
houston_path = os.path.join(dataset_dir, 'houston.mat')
if not os.path.exists(houston_path):
    hsi_arr = hsi.toarray() if hasattr(hsi, 'toarray') else np.asarray(hsi)
    hsi_arr = np.asarray(hsi_arr, dtype=np.float32)
    scio.savemat(houston_path, {'houston': hsi_arr})
    print(f"✅ HSI 已保存: {houston_path}")

gt_path = os.path.join(dataset_dir, 'houston_gt.mat')
if not os.path.exists(gt_path):
    tr = train_label.todense() if hasattr(train_label, 'todense') else train_label
    te = test_label.todense() if hasattr(test_label, 'todense') else test_label
    tr = np.asarray(tr).astype(np.int8)
    te = np.asarray(te).astype(np.int8)
    scio.savemat(gt_path, {'houston_gt_tr': tr, 'houston_gt_te': te})
    print(f"✅ 标签已保存: {gt_path}")

print("\n完成。请运行: python sample_dataset_houston_full.py --split_mode fixed")
