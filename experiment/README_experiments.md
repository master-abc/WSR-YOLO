# DWGSA-YOLO 实验目录

本目录包含论文 "DWGSA-YOLO: Discrete Wavelet Geometry-prior Sparse Attention for PCB Defect Detection" 的全部实验。

## 实验概览

| 实验 | 科学问题 | 证明目标 |
|------|----------|----------|
| exp1 | DWGSA vs SOTA 注意力机制 | 领域特化设计优于通用注意力 |
| exp2 | 各组件贡献多少？ | 每个组件不可替代 |
| exp3 | 是否跨数据集泛化？ | 方法通用，非过拟合 |
| exp4 | 注意力是否聚焦缺陷？ | 可视化验证设计动机 |
| exp5 | 噪声环境下是否鲁棒？ | 小波变换抗噪优势 |

## 快速开始

```bash
# 1. 生成 smoke test 用的迷你数据集
python experiment/create_mini_dataset.py

# 2. 运行各实验的 smoke test（验证功能正确性，2 epochs）
python experiment/exp1/run.py --smoke
python experiment/exp2/run.py --smoke
python experiment/exp3/run.py --smoke
python experiment/exp4/run.py --smoke

# 3. 完整实验（需要真实数据集 + GPU）
python experiment/exp1/run.py --full
python experiment/exp2/run.py --full
python experiment/exp3/run.py --full
python experiment/exp4/run.py --full --dwgsa-weights experiment/exp1/runs/dwgsa_yolo11m/weights/best.pt
```

## 数据集

- **DeepPCB**: 1,500张, 6类 (主实验 Exp1/Exp2) — `python experiment/data/download_deeppcb.py`
- **DefectDet**: 268张, 5类 (泛化实验 Exp3) — `python experiment/data/convert_defectdet.py`
- **PKU_PCB**: 图像级分类标注 (仅参考，无精确bbox) — 已放置于 `experiment/datasets/`
- **Mini Datasets**: 合成数据, 用于 smoke test — `python experiment/create_mini_dataset.py`

## 实验详细说明

### Exp1: SOTA 对比
验证 DWGSA-YOLO 相比 CBAM/EMA/SimAM/CoordAtt/FDSA 的优越性。
核心论点：PCB 缺陷的物理特性（高频异常+几何分布）需要领域特化的注意力设计。

### Exp2: 消融实验
系统性移除 DWGSA 的各组件，量化贡献：
- Wavelet Branch: 高频特征提取
- Geometry Sparse Attn: 结构引导的稀疏注意力
- Adaptive Fusion: 物理信号驱动的门控
- 2-Level DWT: 多分辨率分解

### Exp3: 跨数据集泛化
在DefectDet（5类，268张）和PKU_PCB（6类，693张）上验证方法的通用性。
核心论点：在不同来源、不同类别的数据集上都能保持优势，证明方法通用而非过拟合。

### Exp4: 可视化分析
GradCAM、小波子带、几何先验 mask 的可视化，直观展示模块工作机制。

### Exp5: 噪声鲁棒性实验
测试在不同噪声水平（σ=0, 3, 6, 9, 15）下的性能，验证小波变换对噪声的鲁棒性。
核心论点：小波分解可以将噪声隔离到高频子带，保护低频结构信息，因此DWGSA在噪声环境下性能下降更小。
