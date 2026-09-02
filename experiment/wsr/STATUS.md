# 论文实验状态（2026-08-27）

## 状态

论文实验已经完成。远端确认队列为 `complete`，全预算组件结果为 27/27、失败 0；修订阶段的机制、延迟、定性和负样本实验也均为 `complete`。当前没有论文训练或评估进程在运行。

## 冻结结论

- DsPCBSD+ 正式测试：WSR-YOLO11s 46.69±0.72 AP，YOLO11s 46.54±0.36 AP；+0.155 AP，双侧 Wilcoxon `p=0.8125`。
- DeepPCB 正式测试：68.81±4.12 对 64.61±1.91 AP；+4.196 AP，精确检验 `p=0.25`，同时伴随更高的无缺陷模板误报。
- 第一阶段误报缓解已完成：仅在种子 13 的验证集上从 10%/25% 训练模板负样本中选择 25%，随后固定方案复现到种子 42 和 3407。三种子正样本 AP 从 68.81±4.12 提升到 72.04±0.87；在置信度 0.25 下，无缺陷模板板级误报率从 53.5±6.0% 降到 47.1±1.6%，FPPI 从 1.360 降到 1.079，F1 从 0.887 提升到 0.897。
- 第二阶段从第一阶段权重出发，加入全部 850 张训练模板负样本，以 `lr0=3e-4`、无 mosaic 微调 3 epoch；方案在种子 13 的正负混合验证集确定后，原样复现到种子 42 和 3407。相对第一阶段，AP 从 72.04±0.87 变为 72.31±0.72；置信度 0.25 下板级误报率从 47.1±1.6% 降到 32.7±0.3%，FPPI 从 1.079±0.061 降到 0.662±0.059，precision 从 0.842 升到 0.885，recall 从 0.960 降到 0.951，F1 从 0.897 升到 0.917。
- 三个种子分别只用 150 张验证模板选择满足 5% 板级误报上限的全局阈值，再一次性评估隔离的 500 张测试模板。测试板级误报率为 10.9±5.1%，FPPI 为 0.146±0.080，正样本 precision/recall/F1 为 0.975±0.011/0.887±0.035/0.929±0.014；各种子测试误报率跨度为 5.0%–14.4%，因此仍不是生产保证。
- 第三阶段仅用种子 13 的训练负样本分数锁定最高 25%（213/850）难负样本，并在有效训练列表中重复 3 次；其他种子复用同一列表，以 `lr0=1e-4`、无 mosaic 微调 3 epoch。测试 AP 从 72.31±0.72 变为 72.44±0.73；置信度 0.25 下板级误报率降至 27.5±1.9%，FPPI 降至 0.514±0.067，precision/recall/F1 为 0.899±0.005/0.946±0.002/0.922±0.003。
- 第三阶段的单模型 1% 验证校准在测试上为 6.2%/2.4%/1.8%（均值 3.47%），未稳定达到目标；验证零误报校准虽使测试均值降至 0.4%，但平均召回只有 0.411，不能作为可用方案。
- 验证集选择的三 checkpoint 全一致策略（每模型置信度至少 0.65、同类 IoU 至少 0.3）在 150 张验证模板上为 0/150；冻结后在 500 张测试模板上为 3/500，即板级误报率 0.6%（精确 95% CI 0.12%–1.74%）、FPPI 0.006，正样本 precision/recall/F1 为 0.993/0.813/0.894。该策略需要三次推理，而且一致性方法族是在查看单模型测试结果后才加入，因此只能标为探索性结果，必须用全新 holdout 复核后才能作部署声明。
- 三阶段缓解均使用额外配对模板监督和优化，只能作为事后补救实验；种子 13 参与方案选择，真正的选择外复现只有两个种子。以上 DeepPCB 方案也不能解决 DsPCBSD+ 图 3 中的误报。
- DsPCBSD+ 种子 13 使用仅在验证集选择的“逐类阈值 + 同类重叠抑制（IoU 0.3）”，可把测试误检从 1,222 降到 742（-39.3%），F1 从 0.756 提升到 0.774，但漏检从 862 增至 1,041。同样处理后的 YOLO11s F1 为 0.782，因此这不是 WSR 优势；图 3 仍保留统一阈值以完整展示原始错误与权衡。
- 三轮 WSRStable 验证优化没有候选通过联合精度—机制—延迟门槛，因此没有选择新模型，也没有重新评估锁定测试集。
- 27 个全预算验证组件任务显示完整 WSR 相对重复基线 +1.030 AP，但参数匹配卷积、等权融合和去除 HF 对照达到相当或更高均值；该结果不能证明完整 WSR 组合具有独立精度贡献。
- 同 GPU 配对计时显示 WSR 模型前向延迟为基线的 1.18–1.21 倍，不支持实际加速声明。

## 结果位置

- 正式冻结结果：`frozen_results/`
- 全预算组件种子级结果：`generated/revision_results/confirmatory/`
- 全预算组件汇总：`generated/revision_results/confirmatory_summary.json`
- 全预算实验源代码归档：`experiment/wsr/archive/confirmatory_source_63e4911.bundle`
- 稳定性优化审计：`selection/optimization_summary.md`
- 第一阶段负样本缓解三种子汇总：`generated/negative_aware/mitigation_aggregate.json`
- 第二阶段固定阈值/校准阈值汇总：`generated/negative_aware_local/stage2_aggregate.json`、`generated/negative_aware_local/stage2_calibrated_aggregate.json`
- 第三阶段固定阈值/校准阈值汇总：`generated/negative_aware_local/stage3_hard25_r3_test_aggregate.json`、`generated/negative_aware_local/stage3_hard25_r3_calibrated_01_aggregate.json`
- 探索性一致性策略与测试：`generated/negative_aware_local/stage3_consensus_validation_policy_zero.json`、`generated/negative_aware_local/stage3_consensus_zero_test.json`
- 验证集阈值与操作点结果：`generated/operating_point/`
- 论文派生证据：`../../paper/figures/derived_summary.json`

## 复现实验约束

- 所有正式数字使用统一 COCOeval。
- 正式与确认实验必须来自干净、可解析的 Git 提交。
- 确认实验固定使用验证集，所有结果必须记录 `test_evaluated=false`。
- 不得根据确认实验重新选择模型或重新打开正式测试集。
- 未取得 PCB-IND 的 board/lot 安全划分前，不把它加入当前正式结论。
