# CCF-B 目标实验轨道

这套代码把原来的“单次 smoke test + 空表格”重构为可审计、可复现的论文实验协议。它不会保证论文一定录用，也不会在训练完成前制造 SOTA 结论。版本化论文源码是 `paper/main.tex`；工作区上层的 `main_ccfb.tex` 只是便于原路径编译的包装文件。

当前完成度、阻断项和投稿门槛见 [`STATUS.md`](STATUS.md)。运行本轨道前安装独立依赖：

```powershell
pip install -r experiment\paper_b\requirements-paper-b.txt
```

## 核心研究问题

新方法称为 Wavelet-Conditioned Top-k Sparse Routing（WSR）。与旧版 soft mask 不同，WSR 只 gather 排名前 `rho * H * W` 的 P3 token，执行 LayerNorm+MLP 后 scatter 回原特征图，因此精炼算子的计算量随 `rho` 线性变化。默认 `rho=12.5%`。

论文需要同时回答三个问题：

1. 同一 YOLO11s 训练预算下，WSR 是否稳定优于基线？
2. 与 2024--2026 年公开可运行检测器相比，精度、延迟和模型规模是否有竞争力？
3. 路由是否真的集中于缺陷，而不是仅产生一张好看的注意力图？

## 数据协议

| 数据集 | train | val | test | 用途 |
|---|---:|---:|---:|---|
| DsPCBSD+ | 7,387 | 821 | 2,051 | 主工业数据集；官方 val 锁定为最终 test |
| DeepPCB | 850 | 150 | 500 | 经典基准、鲁棒性和跨域实验 |
| DefectDet | 188 | 40 | 40 | 仅探索；缺少 sequence/template 分组，正式轨道默认禁用 |
| PCB-IND | 待官方数据 | 待按板级分组 | 待锁定 | 2026 外部工业验证；默认禁用 |

数据划分种子固定为 2026，与模型种子分离。感知审计显示公开 PCB 数据中存在相似板图候选，但没有发现跨划分 SHA-256 精确重复；由于原始发布未提供完整 board/lot ID，本项目只能声称“已审计并披露”，不能声称已证明板级独立。本地 `PKU_PCB` 的框几乎全部是 `0.5 0.5 0.8 0.8` 伪框，审计会拒绝它。PCB-IND 只有在取得官方数据、确认许可并能按 board/lot 分组后才启用，不能用随机图像划分占位。

```powershell
cd DWGSA-YOLO
python -m experiment.paper_b.prepare_dspcbsd
python -m experiment.paper_b.run prepare --dataset dspcbsd_plus --coco
python -m experiment.paper_b.run prepare --dataset deeppcb --coco
python -m experiment.paper_b.run audit
```

## 正确的运行顺序

### 1. 只在验证集完成 pilot、消融和结构选择

```powershell
python -m experiment.paper_b.pilot plan
python -m experiment.paper_b.pilot train --device 0
python -m experiment.paper_b.pilot diagnose --device 0
python -m experiment.paper_b.pilot benchmark --device 0
python -m experiment.paper_b.pilot evaluate
python -m experiment.paper_b.pilot freeze

# 完整论文消融仍只使用 val
python -m experiment.paper_b.ablation materialize
python -m experiment.paper_b.ablation train --variant route_p3_12p5 --seed 13 --device 0
```

pilot 对 YOLO11s 验证集基线和 P3/12.5% 候选各运行种子 13、42、3407，并在 `val` 上自动检查平均 AP50:95 增益、路由富集和同硬件延迟。两者都使用固定 35% 训练子集、30 epoch 上限和完整验证集；只有三项全部通过，`freeze` 才会写出可提交 Git 的选择决策。完整消融也只评价 `val`，绝不读取 test。

### 2. 同架构控制主实验

```powershell
python -m experiment.paper_b.run plan
python -m experiment.paper_b.run train --dataset dspcbsd_plus --model yolo11s --seed 13 --device 0
python -m experiment.paper_b.run train --dataset dspcbsd_plus --model wsr_yolo11s_p3_r25 --seed 13 --device 0
```

基线和本文方法使用七个配对种子：13、42、3407、4703、8391、9475、10501。七对使精确双侧 Wilcoxon 检验有可能达到 `p<0.05`；五对时理论最小 p 值为 0.0625。中断后可加 `--resume`，已存在标准结果时默认跳过。

正式运行前必须先提交当前代码，使 Git 工作区保持干净，并完成包含 SHA-256 与感知近重复检查的全量数据审计。WSR 插入 P3 后会改变后续 Ultralytics 层号，脚本会显式重映射 YOLO11s 预训练参数；若目标参数覆盖率低于 99%，训练立即拒绝启动。正式协议为最多 80 epoch、`batch=8, nbs=64`、patience 15；不得只给 WSR 单独降低分辨率或训练预算。实测依据见 [`COMPUTE_BUDGET.md`](COMPUTE_BUDGET.md)。

### 3. 近期 SOTA 轨道

主表只接收官方可运行实现：YOLOv10-M、YOLO11-M、YOLOv12-M、YOLO26-M、RF-DETR-M、RT-DETRv2-S、D-FINE-M 和 DEIM-D-FINE-M。RT-DETRv4 依赖 DINOv3 教师且 DEIMv2 训练成本较高，先列为延后可运行项；资源允许时再补，不能用论文报告值冒充本地同划分重训。PCB-MMF、PCB-FS、Structure-Guided PCB Detection、UniPCB、SCP-DETR 和 LSDM-PCB 仅进入独立的领域方法表，并清楚标注数据、划分和“reported only”。

```powershell
python -m experiment.paper_b.run prepare --dataset dspcbsd_plus --coco
python -m experiment.paper_b.external bootstrap --repos-root D:\paper_b_repos
python -m experiment.paper_b.external materialize --dataset dspcbsd_plus --repos-root D:\paper_b_repos --device 0
```

每个官方仓库应使用独立环境。生成的 `commands.ps1` 不会自动选择测试权重：必须按验证集选择 checkpoint，再手动替换 `<BEST_VALIDATION_CHECKPOINT>` 执行一次 test-only，禁止按 test AP 选权重。

所有框架必须先导出原始 COCO detection JSON，再统一由 `pycocotools.COCOeval` 计算 AP；禁止从 DETR 日志手工抄 AP。官方 COCO 汇总采用每图最多 100 个检测，导出阶段可保留 300 个候选。单点 precision/recall/F1 没有预注册的验证集阈值，因此不进入主表。

### 4. 机制、鲁棒性和迁移实验

```powershell
# 路由是否富集在真实框内
python -m experiment.paper_b.mechanism_diagnostics --weights PATH\best.pt --data experiment\paper_b\generated\datasets\deeppcb\dataset.yaml --output route.json

# 八类退化、五档强度；建议先用 DeepPCB 控制存储量
python -m experiment.paper_b.corruptions experiment\paper_b\generated\datasets\deeppcb\dataset.yaml experiment\paper_b\generated\robustness\deeppcb

# LL/HF/方向子带像素级反事实
python -m experiment.paper_b.frequency_interventions experiment\paper_b\generated\datasets\deeppcb\dataset.yaml experiment\paper_b\generated\frequency\deeppcb

# 用同一冻结权重评测上述目录
python -m experiment.paper_b.evaluate_suite --weights PATH\best.pt --suite-root experiment\paper_b\generated\frequency\deeppcb --output frequency_result.json --model wsr_yolo11s_p3_r25 --dataset deeppcb --seed 13

# 少样本曲线
python -m experiment.paper_b.few_shot experiment\paper_b\generated\datasets\dspcbsd_plus\dataset.yaml experiment\paper_b\generated\few_shot\dspcbsd_plus

# 共享五类上的真正零样本跨域
python -m experiment.paper_b.cross_domain remap experiment\paper_b\generated\datasets\deeppcb\dataset.yaml experiment\paper_b\generated\cross_domain\deeppcb_common5
python -m experiment.paper_b.cross_domain remap experiment\paper_b\generated\datasets\dspcbsd_plus\dataset.yaml experiment\paper_b\generated\cross_domain\dspcbsd_common5
python -m experiment.paper_b.cross_domain pair experiment\paper_b\generated\cross_domain\deeppcb_common5\dataset.yaml experiment\paper_b\generated\cross_domain\dspcbsd_common5\dataset.yaml experiment\paper_b\generated\cross_domain\deep_to_dsp
```

若重新取得 DeepPCB 官方 template 图像，可用 `false_positive.py` 报告 defect-free board FPR 和 FPPI。只有人工确认全部输入均无缺陷时，该指标才有效。

### 5. 效率与统计

```powershell
python -m experiment.paper_b.benchmark --weights PATH\best.pt --data experiment\paper_b\generated\datasets\deeppcb\dataset.yaml --output speed_fp16.json --half
python -m experiment.paper_b.freeze_results --root experiment\paper_b\generated\runs
python -m experiment.paper_b.stats --root experiment\paper_b\frozen_results --output experiment\paper_b\generated\tables
```

统计脚本输出 CSV、Markdown、LaTeX、95% t 区间、配对 Wilcoxon、Holm 校正和配对 Cohen's dz。`paper/main.tex` 只读取自动生成的 LaTeX 表，避免手工抄错数字。

## 投稿前硬性检查

- 主结果至少完成 DsPCBSD+ 的 7 对控制实验和 3 次 SOTA 重复；
- 不把不同划分、分辨率或论文报告值混进同一排名；
- 消融不使用测试集选择结构；
- 同时报告 AP50:95、AP75、逐类 AP、方差、区间、延迟和显存；
- 所有主表结果来自同一个 COCOeval，保留预测 JSON 哈希，且预训练参数覆盖率不低于 99%；
- 正式结果只能从干净 Git 提交运行，并通过 `freeze_results` 固化到可追踪目录；
- 频域反事实与路由富集必须支持方法机制，否则收缩论文主张；
- 所有可视化必须来自训练后的冻结 checkpoint；
- 未跑出的数字保持空白，绝不写“state of the art”。
