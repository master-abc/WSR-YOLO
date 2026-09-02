# 计算预算校准（2026-08-15）

本文件记录用于冻结训练协议的工程探针。它们不是论文结果，不参与精度表，也不能替代训练后 checkpoint 的正式延迟测量。

## 环境与条件

- GPU：NVIDIA GeForce RTX 3060 Laptop GPU，6GB
- PyTorch 2.5.1+cu121，Ultralytics 8.4.50，AMP
- DsPCBSD+ 固定训练划分的 10%（739 张），完整 validation（821 张）
- 640 像素，1 epoch，Windows，`workers=0`

## 观测

| 模型/实现 | batch | 总耗时（秒） | 备注 |
|---|---:|---:|---|
| YOLO11s | 8 | 154.32 | RAM cache，仅用于寻找可用批量 |
| WSR-YOLO11s，显式邻域 gather | 8 | 224.33 | 优化前直接基准 |
| WSR-YOLO11s，共享深度卷积邻域 | 8 | 196.31 | 数学等价重写，下降约 12.5% |

batch 32 虽未立即 OOM，但在首个 epoch 前没有产生进度或结果，已作为无效探针排除。不会把它写入协议。数学等价重写另有 unfold 对照、路由预算和梯度测试。

独立 batch-1 FP32 前向（50 次 warm-up、200 次重复）测得 YOLO11s 58.78ms、WSR 59.16ms，均值比约 1.006。该测量只证明延迟门禁在训练前可行；论文最终数值必须来自三个 pilot checkpoint 和冻结正式 checkpoint。

## 冻结决策

- 正式控制实验：完整训练集，最多 80 epoch，batch 8，nominal batch 64，patience 15。
- pilot 与验证集消融：固定 35% 训练子集，最多 30 epoch，patience 10；基线与候选预算完全一致。
- 分辨率保持 640，不通过给 WSR 单独降分辨率换取速度。
- 所有中断恢复必须通过提交、协议、数据和选择决策哈希校验。
