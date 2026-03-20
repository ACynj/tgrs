# FedFusion 论文与实现对照表

## 一致项 ✓

| 项目 | 论文 | 当前实现 | 状态 |
|------|------|----------|------|
| 损失函数 | CE + MSE + L2 | Fed_Fusion_Loss | ✓ |
| 优化器 | Adam | Adam | ✓ |
| 学习率 | 0.001 | 0.001 | ✓ |
| StepLR | step_size=60, gamma=0.5 | step_size=60, gamma=0.5 | ✓ |
| batch_size | 64 (Houston/Trento) | 64 | ✓ |
| epochs | 300 | 300 | ✓ |
| L2 正则 | weight_decay | 0.001 | ✓ |
| patch_size | 7×7 | 7 | ✓ |
| SVD K | 2 最优 | svd_k=2 | ✓ |
| 网络结构 | 双分支 CNN + Cross Fusion | Fed_Fusion | ✓ |
| 评估指标 | OA, AA, κ, CA | result_cal | ✓ |
| HSI 波段 | Houston 144 | 144 | ✓ |
| LiDAR | 单波段高程 | 1 通道 | ✓ |
| 数据划分 (Houston) | 95% 训练 / 5% 验证 | random_95_5 (14270/759) | ✓ |

---

## 不一致项 ✗

### 1. 联邦学习 client 数量（影响大）

| 论文 | 当前实现 |
|------|----------|
| 16 clients (8 HSI + 8 LiDAR) | 单进程 (WORLD_SIZE=1) |
| 梯度在 client 间聚合 | 无联邦聚合，等价 local-fusion |

**说明**：需 `torchrun --nproc_per_node=2` 或更多进程才能模拟联邦。当前默认单进程运行。

---

### 2. MUUFL 专用设置

| 论文 | 当前实现 |
|------|----------|
| batch_size=128 | 固定 64 |
| ReduceLROnPlateau | 固定 StepLR |

**说明**：MUUFL 需单独配置，当前未区分。

---

### 3. 数据来源

| 论文 | 当前实现 |
|------|----------|
| 官方 Houston 竞赛数据 | rs-fusion-datasets 下载 |

**说明**：预处理与标签格式可能略有差异。

---

## 总结

**已对齐**：损失、优化器、学习率策略、网络结构、SVD、评估指标等核心实现。

**未对齐**：
1. **16 client 联邦**：当前为单 client
2. **MUUFL**：batch_size 与学习率策略未按论文单独设置

若要尽量贴近论文，建议：
- 使用 `torchrun --nproc_per_node=2 main.py ...` 进行 2-client 联邦训练（若资源允许）
