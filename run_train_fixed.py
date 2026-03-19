import os
import subprocess
import sys

print("=" * 60)
print("FedFusion 训练脚本 (使用默认参数)")
print("=" * 60)

# 设置环境变量
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['LOCAL_RANK'] = '0'

# 创建输出目录
output_dir = './output/simple_test'
log_dir = './logs'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# 使用main.py支持的参数
cmd = [
    sys.executable, 'main.py',
    '--dataset_name', 'trento',
    '--device', 'cpu',
    '--with_svd', '0',
    '--inference_only', '0',
    '--model_arch', 'Fed_Fusion',
    '--mode', 'MML',
    '--output_dir', output_dir,
    '--log_path', os.path.join(log_dir, 'simple_test.log'),
    '--log_epoch', '1'
]

print("运行命令:")
print(" ".join(cmd))
print("\n" + "=" * 60)
print("开始训练...")
print("=" * 60)

# 运行训练
try:
    # 直接运行，不捕获输出，可以看到实时进度
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n✅ 训练成功完成!")
        print(f"结果保存在: {output_dir}")
    else:
        print("\n❌ 训练失败!")

except KeyboardInterrupt:
    print("\n\n⚠️ 训练被用户中断")
except Exception as e:
    print(f"\n❌ 训练出错: {e}")