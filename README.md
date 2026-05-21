# DWGSA-YOLO: Discrete Wavelet Geometry-prior Sparse Attention for PCB Defect Detection

基于 Haar 小波分解和几何先验稀疏注意力的 PCB 缺陷检测方法，集成于 YOLO11m 架构。

## 环境安装

### 1. 创建 Conda 环境

```bash
conda create -n dwgsa python=3.10 -y
conda activate dwgsa
```

### 2. 安装 PyTorch (CUDA 12.1)

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

其他 CUDA 版本请参考 [PyTorch 官网](https://pytorch.org/get-started/locally/)。

### 3. 安装依赖

```bash
cd DWGSA-YOLO
pip install -r requirements.txt
```

### 4. 验证安装

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
python -c "from algorithm.register import register_custom_modules; register_custom_modules()"
```

## 项目结构

```
DWGSA-YOLO/
├── algorithm/                  # 核心算法模块
│   ├── dwgsa.py               # DWGSA 主模块 + 消融变体
│   ├── fdsa.py                # FDSA 对比模块 + EMA + SimAM
│   ├── cbam.py                # CBAM 对比模块
│   ├── coordatt.py            # CoordAtt 对比模块
│   └── register.py            # Ultralytics 注册入口
├── experiment/
│   ├── configs/               # 所有配置文件
│   │   ├── experiment.yaml    # 统一实验配置（超参数、方法列表）
│   │   ├── deeppcb.yaml       # DeepPCB 数据集配置
│   │   ├── defectdet.yaml     # DefectDet 数据集配置
│   │   └── *.yaml             # 各模型架构配置
│   ├── datasets/              # 数据集（需自行放置）
│   │   ├── DeepPCB/           # 主数据集 (6类, 1500张)
│   │   └── DefectDet_YOLO/    # 泛化数据集 (5类, 268张)
│   ├── exp1/run.py            # 实验1: SOTA 对比
│   ├── exp2/run.py            # 实验2: 消融实验
│   ├── exp3/run.py            # 实验3: 跨数据集泛化
│   ├── exp4/run.py            # 实验4: 可视化分析
│   └── exp5/run.py            # 实验5: 噪声鲁棒性
└── requirements.txt
```

## 数据集准备

确保数据集放置在 `experiment/datasets/` 下，目录结构如下：

```
experiment/datasets/
├── DeepPCB/
│   ├── images/
│   │   ├── train/    (850 张)
│   │   ├── val/      (150 张)
│   │   └── test/     (500 张)
│   └── labels/       (YOLO 格式 txt)
│       ├── train/
│       ├── val/
│       └── test/
└── DefectDet_YOLO/
    ├── images/
    │   ├── train/    (214 张)
    │   └── val/      (54 张)
    └── labels/
        ├── train/
        └── val/
```

## 运行实验

所有实验参数集中在 `experiment/configs/experiment.yaml` 管理，无需命令行参数。

### 硬件要求

- GPU 显存 >= 12GB（默认 batch=16, imgsz=640）
- 如显存不足，修改 `experiment.yaml` 中 `train.batch` 为 8

### 实验1: SOTA 对比

对比 7 个方法：YOLO11m baseline、CBAM、EMA、SimAM、CoordAtt、FDSA、DWGSA-YOLO。

```bash
python experiment/exp1/run.py
```

- 输出: `experiment/exp1/results.json` + 对比图表
- 预计时间: 7 × 300 epochs ≈ 35-50 小时 (单卡 RTX 3090)

### 实验2: 消融实验

验证各组件贡献：Wavelet Branch、Sparse Attention、Geometry Prior、Adaptive Fusion、Multi-level DWT。

```bash
python experiment/exp2/run.py
```

- 输出: `experiment/exp2/results.json` + 消融图表
- 预计时间: 7 × 300 epochs ≈ 35-50 小时

### 实验3: 跨数据集泛化

在 DefectDet 和 PKU_PCB 上验证泛化能力（Baseline vs DWGSA-YOLO）。

```bash
python experiment/exp3/run.py
```

- 输出: `experiment/exp3/results.json` + 泛化对比图
- 预计时间: 4 × 300 epochs ≈ 20-30 小时

### 实验4: 可视化分析

生成 GradCAM、小波子带、几何先验 mask 可视化。需先完成实验1。

训练完成后，编辑 `experiment/configs/experiment.yaml` 填入权重路径：

```yaml
exp4:
  baseline_weights: "experiment/exp1/runs/yolo11m_baseline/weights/best.pt"
  dwgsa_weights: "experiment/exp1/runs/dwgsa_yolo11m/weights/best.pt"
```

然后运行：

```bash
python experiment/exp4/run.py
```

- 输出: `experiment/exp4/figures/*.png`

### 实验5: 噪声鲁棒性

测试在不同噪声水平下的性能，验证小波变换的抗噪能力。

实验策略：在clean数据上训练模型，在不同噪声水平的测试集上评估。

噪声水平基于真实PCB图像分析：
- Clean: σ=0 (无噪声)
- Low: σ=3 (真实工业成像水平)
- Medium: σ=6 (2x真实水平)
- High: σ=9 (3x真实水平)
- Very High: σ=15 (5x真实水平)

```bash
python experiment/exp5/run.py
```

- 输出: `experiment/exp5/results.json` + 鲁棒性对比图表
- 预计时间: 2 methods × 300 epochs 训练 + 10次评估 ≈ 10-15 小时

## 配置说明

核心超参数位于 `experiment/configs/experiment.yaml`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| epochs | 300 | 训练轮数 |
| batch | 16 | 批大小（显存不足时改为 8） |
| imgsz | 640 | 输入分辨率 |
| optimizer | SGD | 优化器 |
| lr0 | 0.01 | 初始学习率 |
| cos_lr | true | 余弦退火调度 |
| patience | 0 | 禁用 early stopping，确保公平对比 |
| seed | 42 | 随机种子（可复现） |

### 修改 GPU 设备

```yaml
hardware:
  device: "0"        # 单卡
  # device: "0,1"    # 双卡（需相应调整 batch）
```

## 实验完成后

1. 结果保存在各实验目录的 `results.json`
2. 训练曲线和权重保存在 `runs/` 子目录
3. 将数值填入 `main.tex` 对应表格即可

## Citation

```bibtex
@article{dwgsa2025,
  title={DWGSA-YOLO: Discrete Wavelet Geometry-prior Sparse Attention for High-Fidelity PCB Defect Detection},
  author={Luo, Lihua and Liang, JieFeng and Tan, SiJin},
  year={2025}
}
```
