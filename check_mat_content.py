# check_mat_content.py
import scipy.io as scio
import os

# 检查HSI.mat
print("=== 检查 HSI.mat ===")
hsi_path = './DATASET/trento/HSI.mat'
if os.path.exists(hsi_path):
    print(f"✅ 文件存在: {hsi_path}")

    # 加载MAT文件（会看到所有内容）
    mat_data = scio.loadmat(hsi_path)

    print("📦 文件中的变量:")
    for key in mat_data.keys():
        value = mat_data[key]
        if not key.startswith('__'):  # 非MATLAB内部变量
            print(f"  🔑 变量名: '{key}'")
            if hasattr(value, 'shape'):
                print(f"     形状: {value.shape}")
                print(f"     数据类型: {value.dtype}")
        else:
            print(f"  ⚙️  MATLAB内部变量: {key}")
else:
    print(f"❌ 文件不存在: {hsi_path}")

print("\n" + "=" * 40 + "\n")

# 检查LiDAR.mat
print("=== 检查 LiDAR.mat ===")
lidar_path = './DATASET/trento/LiDAR.mat'
if os.path.exists(lidar_path):
    print(f"✅ 文件存在: {lidar_path}")

    mat_data = scio.loadmat(lidar_path)

    print("📦 文件中的变量:")
    for key in mat_data.keys():
        value = mat_data[key]
        if not key.startswith('__'):
            print(f"  🔑 变量名: '{key}'")
            if hasattr(value, 'shape'):
                print(f"     形状: {value.shape}")
                print(f"     数据类型: {value.dtype}")
else:
    print(f"❌ 文件不存在: {lidar_path}")