#!/usr/bin/env python3
"""
收集SVD测试结果
"""
import os
import re
import pandas as pd


def collect_results():
    """收集所有SVD测试结果"""
    results = []

    for k in [0, 2, 4, 8, 16, 32]:
        log_file = f"./logs/svd_k{k}.log"
        if not os.path.exists(log_file):
            continue

        with open(log_file, 'r') as f:
            content = f.read()

        # 提取准确率
        acc_matches = re.findall(r'Test_OA:\s*([0-9.]+)', content)
        if acc_matches:
            final_acc = float(acc_matches[-1])
        else:
            final_acc = 0

        # 提取训练时间
        time_match = re.search(r'Trining time:\s*([0-9.]+)\s*s', content)
        train_time = float(time_match.group(1)) if time_match else 0

        # 提取压缩信息
        compress_match = re.search(r'SVD压缩率:\s*([0-9.]+)x', content)
        compress_ratio = float(compress_match.group(1)) if compress_match else 1.0

        results.append({
            'k': k,
            'accuracy': final_acc * 100,  # 转为百分比
            'time': train_time,
            'compress_ratio': compress_ratio
        })

    # 打印表格
    print("\n" + "=" * 60)
    print("SVD性能对比结果")
    print("=" * 60)
    print(f"{'k值':<6} {'准确率(%)':<12} {'训练时间(s)':<12} {'压缩比':<10}")
    print("-" * 50)

    for res in results:
        print(f"{res['k']:<6} {res['accuracy']:<12.2f} {res['time']:<12.1f} {res['compress_ratio']:<10.1f}x")

    # 保存到CSV
    df = pd.DataFrame(results)
    df.to_csv('./output/svd_comparison.csv', index=False)
    print(f"\n结果已保存到: ./output/svd_comparison.csv")


if __name__ == "__main__":
    collect_results()