import numpy as np
import torch.nn.functional as F
import scipy.io as scio
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR, StepLR, LambdaLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from py_utils import same_seeds, generate_batch, random_mini_batches_standardtwoModality
from py_utils import create_excel, log_and_print, mapset_normalization, result_cal, save_result_excel
import seaborn as sns
import argparse
import logging
import os
import datetime
import time
from config import dataset_config, custom_colors
from model import Fed_Fusion, Fed_Fusion_Loss, FocalLoss
import torch.distributed as dist
from matplotlib.colors import ListedColormap


# Argument parser for command-line options
def command_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_arch', type=str, default='Fed_Fusion', help="Choose the model to use.")
    parser.add_argument('--mode', type=str, default='MML', choices=['MML', 'CML-LiDAR', 'CML-HSI'],
                        help="Choose the model to use.")
    parser.add_argument('--inference_only', type=int, default='0')
    parser.add_argument('--with_svd', type=int, default='0', help="Use svd or not.")
    parser.add_argument('--svd_k', type=int, default='4', help="Only valid when with_svd is True.")
    parser.add_argument('--dataset_name', type=str, default='houston13', help="Dataset name.")
    parser.add_argument('--device', type=str, default='cuda', help="cpu or cuda")
    parser.add_argument('--output_dir', type=str, default='./output', help="Base directory to save outputs.")
    parser.add_argument('--exp_name', type=str, default=None, help="Experiment name (optional)")
    parser.add_argument('--save_excel', type=int, default=1, help="Set to True to save results in an Excel file.")
    parser.add_argument('--save_png', type=int, default=1, help="Set to True to save PNG images.")
    parser.add_argument('--log_path', type=str, default=None,
                        help="Path to save the log file (will be auto-generated if None)")
    parser.add_argument('--log_epoch', type=int, default=50, help="epochs interval to log.")

    # 新增分布式训练参数
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for distributed training')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='Weight decay')

    args = parser.parse_args()
    return args


def load_dataset(batch_size):
    HSI_TrSet = scio.loadmat(os.path.join(dataset_dir, f'TrTe/HSI_TrSet.mat'))['Data']
    HSI_TeSet = scio.loadmat(os.path.join(dataset_dir, f'TrTe/HSI_TeSet.mat'))['Data']
    LiDAR_TrSet = scio.loadmat(os.path.join(dataset_dir, f'TrTe/LiDAR_TrSet.mat'))['Data']
    LiDAR_TeSet = scio.loadmat(os.path.join(dataset_dir, f'TrTe/LiDAR_TeSet.mat'))['Data']
    Y_train = scio.loadmat(os.path.join(dataset_dir, f'TrTe/Y_train.mat'))['Data']
    Y_test = scio.loadmat(os.path.join(dataset_dir, f'TrTe/Y_test.mat'))['Data']

    log_and_print('HSI_TrSet.shape = ', HSI_TrSet.shape)
    log_and_print('HSI_TeSet.shape = ', HSI_TeSet.shape)
    log_and_print('LiDAR_TrSet.shape = ', LiDAR_TrSet.shape)
    log_and_print('LiDAR_TeSet.shape = ', LiDAR_TeSet.shape)
    log_and_print('Y_train.shape = ', Y_train.shape)
    log_and_print('Y_test.shape = ', Y_test.shape)

    trainset1 = HSI_TrSet
    trainset2 = LiDAR_TrSet
    if mode == 'MML':
        valset1 = HSI_TeSet
        valset2 = LiDAR_TeSet

    if mode == 'CML-LiDAR':
        valset1 = np.zeros_like(HSI_TeSet)
        valset2 = LiDAR_TeSet

    if mode == 'CML-HSI':
        valset1 = HSI_TeSet
        valset2 = np.zeros_like(LiDAR_TeSet)

    X1_train = torch.tensor(trainset1).to(device)
    X2_train = torch.tensor(trainset2).to(device)
    X1_test = torch.tensor(valset1).to(device)
    X2_test = torch.tensor(valset2).to(device)

    val_dataset = random_mini_batches_standardtwoModality(X1_test, X2_test, Y_test)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True)
    if not with_dist:
        train_dataset = random_mini_batches_standardtwoModality(X1_train, X2_train, Y_train)
    else:
        if dist.get_rank() == 0:
            train_dataset = random_mini_batches_standardtwoModality(X1_train, torch.zeros_like(X2_train), Y_train)
        else:
            train_dataset = random_mini_batches_standardtwoModality(torch.zeros_like(X1_train), X2_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader


def train(model, optimizer, epochs, train_loader, test_loader, criterion, with_dist=False, local_rank=0):
    best_test_acc = 0
    best_epoch = 0
    best_model_path = os.path.join(model_dir, f"{base_name}_best.pth")

    if with_dist is False:
        best_test_result = [float(0)]
    else:
        try:
            best_test_result = [float(0)] * dist.get_world_size()
        except:
            best_test_result = [float(0)]
            with_dist = False

    for epoch in range(epochs + 1):
        model.train()
        num_correct = 0
        num_total = 0

        for batch_idx, (x1, x2, target) in enumerate(train_loader):
            x1 = x1.to(device, dtype=torch.float32)
            x2 = x2.to(device, dtype=torch.float32)
            target = torch.argmax(target, dim=1).to(device)

            outputs = model(x1, x2, with_svd=with_svd, k=svd_k)
            output1, _, _, _ = outputs

            loss = criterion(output1, target)  # 注意：这里只传 output1

            optimizer.zero_grad()
            loss.backward()

            shared_layers = ['cross_a', 'cross_b', 'conv5', 'bn5', 'conv6', 'bn6', 'conv7']
            if with_dist is True:
                try:
                    for name, param in model.named_parameters():
                        if param.grad is None:
                            param.grad = torch.zeros_like(param.data)
                        if any(shared in name for shared in shared_layers):
                            dist.all_reduce(param.grad.data / 2, op=dist.ReduceOp.AVG)
                        else:
                            dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG)
                except Exception as e:
                    print(f"⚠️  分布式同步失败，继续单机训练: {e}")
                    with_dist = False
            optimizer.step()

            pred_classes = torch.argmax(output1, dim=1)
            num_correct += torch.sum(pred_classes == target)
            num_total += target.size(0)

        train_acc = num_correct.item() / num_total
        train_loss = loss.item()
        scheduler.step()

        test_loss, test_acc, per_class_acc, test_aa, k = test(model, test_loader, criterion)

        # ========== 保存最佳模型 ==========
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch

            # 保存最佳模型（同时保存优化器状态，方便继续训练）
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'train_loss': train_loss,
                'test_aa': test_aa,
                'kappa': k,
                'per_class_acc': per_class_acc,
            }, best_model_path)

            if local_rank == 0:
                print(f"\n🏆 新的最佳模型! epoch {epoch}, Test_OA: {test_acc:.4f}, AA: {test_aa:.4f}, Kappa: {k:.4f}")
                print(f"💾 已保存到: {best_model_path}")

        # ========== 定期保存检查点（每10个epoch） ==========
        if epoch % 10 == 0 and epoch > 0:
            checkpoint_path = os.path.join(model_dir, f"{base_name}_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            if local_rank == 0:
                print(f"💾 已保存检查点: epoch {epoch}")

        if epoch % log_epoch == 0:
            if local_rank == 0:
                log_and_print(
                    "epoch %i: Train_loss: %f, Test_loss: %f, Train_OA: %f, Test_OA: %f, Test_AA: %f, Kappa:%.2f" % (
                        epoch, train_loss, test_loss, train_acc, test_acc, test_aa, k))
                if save_excel is True:
                    save_result_excel(workbook, worksheet, excel_path, epoch, svd_k, test_acc, per_class_acc, test_aa,
                                      k)

    # 训练结束后打印总结
    if local_rank == 0:
        print("\n" + "=" * 60)
        print("🏁 训练完成!")
        print(f"📊 最佳模型: epoch {best_epoch}, Test_OA: {best_test_acc:.4f}")
        print(f"💾 最佳模型路径: {best_model_path}")
        print("=" * 60)


def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for (x1, x2, target) in test_loader:
            x1 = x1.to(device, dtype=torch.float32)
            x2 = x2.to(device, dtype=torch.float32)
            target = torch.argmax(target, dim=1).to(device)

            outputs = model(x1, x2, with_svd=with_svd)
            predictions, _, _, _ = outputs

            loss = criterion(predictions, target)

            test_loss += loss.item() * target.size(0)
            all_targets.append(target)
            all_preds.append(torch.argmax(predictions, dim=1))

        all_targets = torch.cat(all_targets).cpu().numpy()
        all_preds = torch.cat(all_preds).cpu().numpy()

        c = confusion_matrix(all_targets, all_preds)
        test_acc, per_class_acc, aa, k = result_cal(torch.from_numpy(c))
        test_loss = test_loss / len(test_loader.dataset)
    return test_loss, test_acc, per_class_acc, aa, k


def inference_all(only_valid=False):
    model.eval()

    # ===== 根据数据集名称加载数据 =====
    if dataset_name == 'houston13':
        # Houston13 版本 (newhuston 目录)
        try:
            # 加载 ground truth
            gt_data = scio.loadmat(os.path.join(dataset_dir, 'houston_gt.mat'))
            # houston_gt.mat 中有 'houston_gt_tr' 和 'houston_gt_te'
            # 对于推理，我们使用测试集标签或合并两者
            if 'houston_gt_te' in gt_data:
                ground_truth = gt_data['houston_gt_te'].astype(np.int8)
                print("✅ 加载 ground truth: houston_gt.mat, 使用测试集标签")
            elif 'houston_gt_tr' in gt_data:
                ground_truth = gt_data['houston_gt_tr'].astype(np.int8)
                print("✅ 加载 ground truth: houston_gt.mat, 使用训练集标签")
            else:
                # 取第一个非内部变量
                for key in gt_data.keys():
                    if not key.startswith('__'):
                        ground_truth = gt_data[key].astype(np.int8)
                        print(f"✅ 加载 ground truth: houston_gt.mat, 变量: {key}")
                        break

            # 加载 HSI 和 LiDAR 数据
            data = scio.loadmat(os.path.join(dataset_dir, 'houston.mat'))
            HSI_MapSet = data['houston']  # (349, 1905, 144)
            print(f"✅ 加载 HSI 数据: houston.mat, 形状 {HSI_MapSet.shape}")

            # LiDAR 数据 - 从 HSI 取第一个波段作为替代
            LiDAR_MapSet = HSI_MapSet[:, :, :1]  # (349, 1905, 1)
            print(f"✅ 使用 HSI 第一个波段作为 LiDAR 数据")

        except Exception as e:
            print(f"❌ 加载 Houston13 数据失败: {e}")
            raise

    else:  # trento 版本
        try:
            ground_truth = scio.loadmat(os.path.join(dataset_dir, 'gt.mat'))['GT_Trento'].astype(np.int8)
            HSI_MapSet = scio.loadmat(os.path.join(dataset_dir, 'HSI.mat'))['HSI_Trento']
            LiDAR_MapSet = scio.loadmat(os.path.join(dataset_dir, 'Lidar.mat'))['Lidar_Trento']
            print("✅ 加载 Trento 数据成功")
        except Exception as e:
            print(f"❌ 加载 Trento 数据失败: {e}")
            raise

    # ===== 数据标准化 =====
    print("\n开始数据标准化...")
    HSI_MapSet, row, col, n_feature = mapset_normalization(
        mapset=HSI_MapSet,
        concate_pixel=concate_pixel,
        n_feature=hsi_n_feature,
        type='HSI'
    )
    LiDAR_MapSet, row, col, n_feature = mapset_normalization(
        mapset=LiDAR_MapSet,
        concate_pixel=concate_pixel,
        n_feature=lidar_n_feature,
        type='LiDAR'
    )
    print(f"标准化完成: 图像大小 {row} x {col}")

    # ===== 生成预测 =====
    print("\n开始生成预测...")
    drawall_idx = np.array([j for j, x in enumerate(ground_truth.reshape(row * col).ravel().tolist())])
    print(f"总像素数: {row * col}, 有效索引数: {len(drawall_idx)}")

    drawmap_loder1 = generate_batch(drawall_idx, HSI_MapSet, ground_truth.reshape(row * col), batch_size, patch_size,
                                    row, col, num_class=num_class, shuffle=False, only_valid=only_valid)
    drawmap_loder2 = generate_batch(drawall_idx, LiDAR_MapSet, ground_truth.reshape(row * col), batch_size, patch_size,
                                    row, col, num_class=num_class, shuffle=False, only_valid=only_valid)

    model.eval()

    all_preds = []
    all_labels = []
    pred_test = np.full(row * col, min(np.unique(ground_truth)))
    valid_idx = []
    total_processed = 0

    label = ground_truth.reshape(row * col)

    # 计算总批次数
    total_batches = (len(drawall_idx) + batch_size - 1) // batch_size
    print(f"总批次数: {total_batches}")
    batch_count = 0
    start_time = time.time()

    with torch.no_grad():
        for one, two in zip(drawmap_loder1, drawmap_loder2):
            batch_count += 1
            if batch_count % 10 == 0 or batch_count == 1:
                elapsed = time.time() - start_time
                print(f"处理批次 {batch_count}/{total_batches} (已用时间: {elapsed:.1f}秒)")

            data1 = one[0]
            data2 = two[0]
            valid_idx = two[1]
            if len(valid_idx) == 0:
                continue

            # 转换为 tensor
            data1 = torch.tensor(data1)
            data2 = torch.tensor(data2)
            x1 = data1.to(torch.float32).to(device)
            x2 = data2.to(torch.float32).to(device)

            # 模型推理
            outputs = model(x1, x2, with_svd=with_svd)
            predictions, _, _, _ = outputs
            predictions = predictions.to(device)

            if len(predictions.shape) == 1:
                predictions = predictions.unsqueeze(0)
            pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()
            pred_classes = pred_classes + 1

            pred_test[valid_idx] = pred_classes

            all_preds.extend(pred_classes)
            all_labels.extend(label[valid_idx])
            total_processed += len(valid_idx)

            # 清理显存
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    print(f"\n处理完成，共处理 {total_processed} 个像素，总用时: {elapsed:.1f}秒")
    pred_test = pred_test.reshape(row, col)
    print(f"预测结果形状: {pred_test.shape}")
    print(f"预测类别: {np.unique(pred_test)}")

    return pred_test, all_preds, all_labels


def output_visual(pred_test):
    log_and_print('Drawing map')

    # 打印调试信息
    unique_values = np.unique(pred_test)
    print(f"预测结果中的唯一值: {unique_values}")

    # 根据数据集名称设置类别名称
    if dataset_name == 'houston13':
        class_names = {
            1: 'Healthy grass',
            2: 'Stressed grass',
            3: 'Synthetic grass',
            4: 'Trees',
            5: 'Soil',
            6: 'Water',
            7: 'Residential',
            8: 'Commercial',
            9: 'Road',
            10: 'Highway',
            11: 'Railway',
            12: 'Parking Lot 1',
            13: 'Parking Lot 2',
            14: 'Tennis Court',
            15: 'Running Track'
        }
        num_classes = 15
        # Houston2013 需要15种颜色
        paper_colors = [
            [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255],
            [0, 255, 255], [128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0],
            [128, 0, 128], [0, 128, 128], [255, 128, 0], [255, 0, 128], [128, 255, 0]
        ]
    else:  # trento
        class_names = {
            1: 'Apples',
            2: 'Buildings',
            3: 'Ground',
            4: 'Woods',
            5: 'Vineyard',
            6: 'Road'
        }
        num_classes = 6
        paper_colors = [
            [255, 0, 0], [0, 255, 0], [0, 0, 255],
            [255, 255, 0], [255, 0, 255], [0, 255, 255]
        ]

    # 转换为0-1范围
    rgb_colors = np.array(paper_colors[:num_classes]) / 255.0

    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    # 创建颜色映射
    cmap = mcolors.ListedColormap(rgb_colors)

    # 设置边界
    bounds = [i - 0.5 for i in range(1, num_classes + 2)]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 8))

    # 显示图像
    im = ax.imshow(pred_test, cmap=cmap, norm=norm, interpolation='nearest')

    # 设置坐标轴
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Classification Map', fontsize=14, fontweight='bold')

    # 创建自定义图例
    legend_elements = []
    for i in range(1, num_classes + 1):
        if i in class_names:
            legend_elements.append(
                mpatches.Patch(color=rgb_colors[i - 1], label=class_names[i])
            )

    # 添加图例，放在图外右侧
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1),
              loc='upper left', fontsize=10, title='Class Labels')

    plt.tight_layout()

    # 保存高分辨率图片
    plt.savefig(map_fig_path, bbox_inches='tight', dpi=600, facecolor='white')
    plt.close()

    print(f"✅ 图片已保存: {map_fig_path}")
    print(f"📊 图片大小: {pred_test.shape}")


same_seeds(2)

args = command_parser()
model_class = globals()[args.model_arch]
mode = args.mode
dataset_name = args.dataset_name.lower()
inference_only = bool(args.inference_only)
with_svd = bool(args.with_svd)
svd_k = args.svd_k if with_svd else 0
save_excel = bool(args.save_excel)
save_png = bool(args.save_png)
log_epoch = args.log_epoch

# ===== 自动创建实验目录 =====
base_output_dir = args.output_dir

# 生成实验名称
if args.exp_name:
    experiment_name = args.exp_name
else:
    # 自动生成带时间戳和关键参数的实验名
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"exp_{timestamp}_svd{svd_k}_bs{args.batch_size}_ep{args.epochs}"

# 创建实验目录
exp_dir = os.path.join(base_output_dir, experiment_name)
os.makedirs(exp_dir, exist_ok=True)

# 在实验目录下创建子目录
subdirs = ['models', 'logs', 'figures']
for subdir in subdirs:
    os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)

# 设置各个输出路径
output_dir = exp_dir  # 覆盖原来的 output_dir
log_path = args.log_path if args.log_path else os.path.join(exp_dir, 'logs', 'training.log')
excel_path = os.path.join(exp_dir, 'results.xlsx')
map_fig_path = os.path.join(exp_dir, 'figures',
                            f"{args.model_arch}_{with_svd}_map_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
model_dir = os.path.join(exp_dir, 'models')  # 模型保存目录

print("=" * 60)
print(f"📁 实验目录: {exp_dir}")
print(f"📁 子目录:")
print(f"  - 模型: {model_dir}")
print(f"  - 日志: {os.path.join(exp_dir, 'logs')}")
print(f"  - 图片: {os.path.join(exp_dir, 'figures')}")
print(f"📊 结果文件: {excel_path}")
print("=" * 60)

# Setup logging
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(message)s')
# Log all input arguments
logging.info("Parsed Input Arguments:")
for arg, value in vars(args).items():
    log_and_print(f"{arg}: {value}")

with_dist = bool(f"{args.model_arch}" == "Fed_Fusion")
base_name = f"{args.model_arch}"

patch_size = 7
if dataset_name in dataset_config:
    config = dataset_config[dataset_name]
    dataset_dir = config['dataset_dir']
    hsi_n_feature = config['hsi_n_feature']
    lidar_n_feature = config['lidar_n_feature']
    concate_pixel = config['concate_pixel']
    class_labels = config['class_labels']
else:
    raise NameError(f"Dataset {dataset_name} not recognized")

cm_labels = list(class_labels.values())[1:]
num_class = max(class_labels.keys())
workbook, worksheet = create_excel(excel_path, mode, cm_labels)

palette = {0: (0, 0, 0)}
for k in range(len(cm_labels)):
    palette[k + 1] = tuple(custom_colors[k])
colormap = [np.array(color) / 255.0 for color in palette.values()]

local_rank = int(os.environ.get('RANK', 0))

# 强制设置环境变量，确保分布式能正确初始化
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
os.environ['RANK'] = str(local_rank)
os.environ['LOCAL_RANK'] = str(local_rank)
os.environ['WORLD_SIZE'] = '1'

# 在分布式初始化部分，确保 with_dist 被正确设置
with_dist = bool(f"{args.model_arch}" == "Fed_Fusion")

# 服务器版 - 真正的分布式（无打印版本）
if with_dist is True:
    if args.device == 'cuda':
        # 获取实际可用的GPU数量
        num_gpus = torch.cuda.device_count()

        # 如果local_rank超出GPU数量，取模
        if local_rank >= num_gpus:
            effective_gpu = local_rank % num_gpus
        else:
            effective_gpu = local_rank

        torch.cuda.set_device(effective_gpu)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=local_rank,
            world_size=int(os.environ.get('WORLD_SIZE', 2))
        )
        device = torch.device(f"cuda:{effective_gpu}")
    else:
        # CPU分布式 - 使用gloo后端
        dist.init_process_group(
            backend="gloo",
            init_method="env://",
            rank=local_rank,
            world_size=int(os.environ.get('WORLD_SIZE', 2))
        )
        device = torch.device("cpu")
else:
    device = torch.device(args.device)

model = model_class(hsi_n_feature, lidar_n_feature, num_class).to(device)

# ===== 加载已训练好的模型（继续训练） =====
if not inference_only:
    # 指定要加载的模型路径 - 使用第50轮的模型
    resume_path = './output/exp_20260319_225814_svd8_bs32_ep200/models/Fed_Fusion_epoch_50.pth'

    if os.path.exists(resume_path):
        print(f"📂 加载模型继续训练: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)

        # 加载模型权重
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            # 也可以选择加载优化器状态
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            print(f"✅ 模型加载成功！从 epoch {start_epoch} 继续训练")
        else:
            model.load_state_dict(checkpoint)
            start_epoch = 50
            print(f"✅ 模型加载成功！从 epoch 50 继续训练")
    else:
        print(f"⚠️ 未找到模型文件: {resume_path}, 从头开始训练")
        start_epoch = 0

# 更新权重路径到实验目录
if with_svd:
    weight_path = os.path.join(model_dir, f"{base_name}_svd_rank{local_rank}.pth")
else:
    weight_path = os.path.join(model_dir, f"{base_name}_rank{local_rank}.pth")

batch_size = args.batch_size
epochs = args.epochs
base_lr = args.lr
weight_decay = args.weight_decay

# 改为 SGD with momentum
# optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=weight_decay)

optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
log_and_print("epoches = {0}, batch size = {1}, base learning rate = {2}, weight decay = {3}".format(epochs, batch_size,
                                                                                                     base_lr,
                                                                                                     weight_decay))

# 加载数据集
train_loader, test_loader = load_dataset(batch_size)

# ===== 使用普通交叉熵损失（无权重） =====
criterion = nn.CrossEntropyLoss().to(device)
print(f"✅ 使用普通交叉熵损失")
# ====================================

if inference_only is False:
    start_time = datetime.datetime.now()
    train(model, optimizer, epochs, train_loader, test_loader, criterion, with_dist=with_dist, local_rank=local_rank)
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    log_and_print(f"Trining time: {elapsed_time.total_seconds():.2f} s")

pred_test, _, _ = inference_all()

if local_rank == 0 and save_png is True:
    output_visual(pred_test)

if with_dist:
    try:
        dist.destroy_process_group()
        print("✅ 分布式进程组已销毁")
    except:
        pass