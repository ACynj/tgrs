import os
import subprocess
import sys

print("=" * 60)
print("FedFusion 简化训练脚本")
print("=" * 60)

# 设置环境变量（避免分布式错误）
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['LOCAL_RANK'] = '0'

# 训练参数
config = {
    'dataset_name': 'trento',
    'device': 'cpu',
    'batch_size': 4,  # 小batch_size节省内存
    'epochs': 2,  # 只训练2个epoch测试
    'with_svd': 0,  # 关闭SVD压缩
    'inference_only': 0,  # 进行训练
    'model_arch': 'Fed_Fusion',
    'mode': 'MML',  # 多模态学习
    'output_dir': './output/simple_test',
    'log_path': './logs/simple_test.log',
    'log_epoch': 1  # 每个epoch都输出日志
}

# 创建输出目录
os.makedirs(config['output_dir'], exist_ok=True)
os.makedirs(os.path.dirname(config['log_path']), exist_ok=True)

# 构建命令
cmd = [
    sys.executable, 'main.py',
    '--dataset_name', config['dataset_name'],
    '--device', config['device'],
    '--batch_size', str(config['batch_size']),
    '--epochs', str(config['epochs']),
    '--with_svd', str(config['with_svd']),
    '--inference_only', str(config['inference_only']),
    '--model_arch', config['model_arch'],
    '--mode', config['mode'],
    '--output_dir', config['output_dir'],
    '--log_path', config['log_path'],
    '--log_epoch', str(config['log_epoch'])
]

print("运行命令:")
print(" ".join(cmd))
print("\n" + "=" * 60)
print("开始训练...")
print("=" * 60)

# 运行训练
try:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True
    )

    print("✅ 训练成功完成!")
    print("\n训练输出:")
    print("-" * 40)

    # 只显示最后50行输出，避免太多信息
    lines = result.stdout.split('\n')
    for line in lines[-50:]:
        if line.strip():
            print(line)

    print("\n" + "=" * 60)
    print("训练完成！结果保存在:", config['output_dir'])

except subprocess.CalledProcessError as e:
    print("❌ 训练失败!")
    print("\n错误信息:")
    print("-" * 40)
    print(e.stderr)

    print("\n标准输出:")
    print("-" * 40)
    print(e.stdout)

    sys.exit(1)