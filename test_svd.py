"""
测试SVD压缩功能
"""
import os
import subprocess
import sys
import time


def run_with_svd(svd_k=4, epochs=5):
    """运行带SVD压缩的训练"""
    print(f"\n🔧 测试SVD压缩 (k={svd_k}, epochs={epochs})")
    print("=" * 60)

    # 设置环境变量
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['LOCAL_RANK'] = '0'

    # 临时修改main.py中的epochs值
    print("临时修改main.py中的训练参数...")

    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 找到并修改epochs和batch_size
    import re

    # 修改epochs (从200改为指定值)
    content = re.sub(r'epochs\s*=\s*\d+', f'epochs = {epochs}', content)

    # 修改batch_size为较小值（节省内存）
    content = re.sub(r'batch_size\s*=\s*\d+', 'batch_size = 16', content)

    # 写入临时文件
    temp_file = 'main_svd_test.py'
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(content)

    # 构建命令
    cmd = [
        sys.executable, temp_file,
        '--dataset_name', 'trento',
        '--device', 'cpu',
        '--with_svd', '1',
        '--svd_k', str(svd_k),
        '--inference_only', '0',
        '--model_arch', 'Fed_Fusion',
        '--mode', 'MML',
        '--output_dir', f'./output/svd_test_k{svd_k}',
        '--log_path', f'./logs/svd_test_k{svd_k}.log',
        '--log_epoch', '1'
    ]

    print(f"运行命令: {' '.join(cmd)}")
    print("\n开始训练...")

    start_time = time.time()

    try:
        # 运行训练
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        elapsed_time = time.time() - start_time

        print(f"\n✅ SVD测试完成! 耗时: {elapsed_time:.1f}秒")

        # 分析输出
        print("\n📊 训练输出摘要:")
        print("-" * 40)

        # 提取关键信息
        lines = result.stdout.split('\n')
        for line in lines:
            if 'epoch' in line.lower() and 'accuracy' in line.lower():
                print(line)
            if 'svd' in line.lower():
                print(line)
            if 'error' in line.lower() or 'fail' in line.lower():
                print(f"⚠️  {line}")

        return True, elapsed_time

    except subprocess.CalledProcessError as e:
        print(f"\n❌ SVD测试失败!")
        print(f"错误: {e.stderr}")
        return False, 0

    finally:
        # 清理临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"清理临时文件: {temp_file}")


def compare_svd_performance():
    """比较不同SVD设置的性能"""
    print("=" * 60)
    print("SVD压缩性能对比测试")
    print("=" * 60)

    test_cases = [
        {'name': '无SVD (baseline)', 'with_svd': 0, 'svd_k': 0, 'epochs': 3},
        {'name': 'SVD k=2', 'with_svd': 1, 'svd_k': 2, 'epochs': 3},
        {'name': 'SVD k=4', 'with_svd': 1, 'svd_k': 4, 'epochs': 3},
        {'name': 'SVD k=8', 'with_svd': 1, 'svd_k': 8, 'epochs': 3},
    ]

    results = []

    for test in test_cases:
        print(f"\n🔍 测试: {test['name']}")

        # 临时修改main.py
        with open('main.py', 'r', encoding='utf-8') as f:
            content = f.read()

        import re
        content = re.sub(r'epochs\s*=\s*\d+', f'epochs = {test["epochs"]}', content)
        content = re.sub(r'batch_size\s*=\s*\d+', 'batch_size = 16', content)

        temp_file = f'main_temp_{test["svd_k"]}.py'
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(content)

        # 运行测试
        cmd = [
            sys.executable, temp_file,
            '--dataset_name', 'trento',
            '--device', 'cpu',
            '--with_svd', str(test['with_svd']),
            '--svd_k', str(test['svd_k']),
            '--inference_only', '0',
            '--output_dir', f'./output/svd_compare',
            '--log_epoch', '1'
        ]

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10分钟超时
            )

            elapsed_time = time.time() - start_time

            # 提取准确率
            accuracy = 0
            for line in result.stdout.split('\n'):
                if 'Test_OA:' in line:
                    parts = line.split('Test_OA:')
                    if len(parts) > 1:
                        acc_str = parts[1].split(',')[0].strip()
                        try:
                            accuracy = float(acc_str)
                        except:
                            pass

            results.append({
                'name': test['name'],
                'success': result.returncode == 0,
                'time': elapsed_time,
                'accuracy': accuracy,
                'output': result.stdout[-500:]  # 最后500字符
            })

            print(f"  状态: {'✅ 成功' if result.returncode == 0 else '❌ 失败'}")
            print(f"  时间: {elapsed_time:.1f}秒")
            print(f"  准确率: {accuracy:.4f}")

        except subprocess.TimeoutExpired:
            print(f"  ⏰ 超时!")
            results.append({
                'name': test['name'],
                'success': False,
                'time': 600,
                'accuracy': 0,
                'output': 'Timeout'
            })
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            results.append({
                'name': test['name'],
                'success': False,
                'time': 0,
                'accuracy': 0,
                'output': str(e)
            })
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    # 打印比较结果
    print("\n" + "=" * 60)
    print("SVD性能对比结果")
    print("=" * 60)
    print(f"{'测试案例':<15} {'状态':<8} {'时间(秒)':<10} {'准确率':<10}")
    print("-" * 60)

    for res in results:
        status = '✅' if res['success'] else '❌'
        print(f"{res['name']:<15} {status:<8} {res['time']:<10.1f} {res['accuracy']:<10.4f}")

    return results


def test_svd_in_client():
    """在客户端中测试SVD"""
    print("\n" + "=" * 60)
    print("在客户端中测试SVD")
    print("=" * 60)

    # 测试HSI客户端带SVD
    print("\n1. 测试HSI客户端带SVD...")
    cmd_hsi = [
        sys.executable, 'client_hsi.py',
        '--dataset', 'trento',
        '--fed_rounds', '1',
        '--local_epochs', '1',
        '--batch_size', '8',
        '--world_size', '1',
        '--with_svd', '1',
        '--svd_k', '4'
    ]

    try:
        result = subprocess.run(cmd_hsi, capture_output=True, text=True, timeout=300)
        print(f"HSI客户端SVD测试: {'✅ 成功' if result.returncode == 0 else '❌ 失败'}")
        if 'svd' in result.stdout.lower():
            print("  检测到SVD相关输出")
    except Exception as e:
        print(f"HSI客户端SVD测试: ❌ 错误 - {e}")


def main():
    """主函数"""
    print("=" * 60)
    print("FedFusion SVD压缩模块测试")
    print("=" * 60)

    # 创建输出目录
    os.makedirs('./output/svd_test', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)

    choice = input(
        "\n选择测试模式:\n1. 单次SVD测试 (k=8, 5 epochs)\n2. 对比测试 (不同k值)\n3. 客户端SVD测试\n选择 (1/2/3): ").strip()

    if choice == '1':
        # 单次测试
        success, elapsed = run_with_svd(svd_k=8, epochs=5)

        if success:
            print(f"\n🎉 SVD测试成功完成!")
            print(f"总耗时: {elapsed:.1f}秒")
            print(f"结果保存在: ./output/svd_test_k8/")
        else:
            print("\n⚠️ SVD测试失败，可能需要检查代码")

    elif choice == '2':
        # 对比测试
        results = compare_svd_performance()

        # 保存结果
        import json
        with open('./output/svd_comparison.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n详细结果已保存: ./output/svd_comparison.json")

    elif choice == '3':
        # 客户端SVD测试
        test_svd_in_client()
    else:
        print("无效选择")


if __name__ == "__main__":
    main()