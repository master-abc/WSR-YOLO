# 论文项目状态（2026-08-15）

## 当前结论

代码与实验协议已经完成关键整改，但论文尚未“实验完成”。目前只有 2-epoch smoke 结果，不能用于论文数值、SOTA 声明或投稿结论。正式主实验开始前必须先把本轨道全部加入 Git 并提交；运行器会拒绝脏工作区。

## 已通过的门禁

- WSR 插层后的 YOLO11s 权重按层号重映射；P3/P4/P5/P3+P4 实测覆盖率为 99.81%/99.81%/99.40%/99.65%，门槛为 99%。
- 所有框架先导出 COCO prediction JSON，再统一使用 `pycocotools.COCOeval`。
- 三套本地数据无跨划分 SHA-256 精确重复；近重复候选已记录在各数据集 `audit.json`。
- 正式结果记录代码、协议、数据、checkpoint 与预测哈希，并只能从干净 Git 提交运行。
- 结果冻结器拒绝 smoke、旧 schema、脏提交和不合格的预训练覆盖。
- 8 个核心测试、Python 编译、统计表生成和论文编译均已通过。
- 本机结构诊断中，WSR/YOLO11s batch-1 FP32 前向均值为 59.16/58.78 ms，比例约 1.006，低于 pilot 上限 1.20；该未训练诊断只用于预算，不进入论文表。
- 6GB GPU 吞吐探针后，正式协议冻结为 YOLO11s、最多 80 epoch、batch 8、patience 15；pilot/消融使用固定 35% 训练子集和 30 epoch 上限。

## 正式实验设计

- 主数据集：DsPCBSD+，基线与 WSR 使用 7 个配对种子。
- 次数据集：DeepPCB，使用 3 个配对种子，并明确其人工缺陷属性。
- DefectDet：因样本小且缺少 sequence/template 分组，默认退出正式主表。
- PCB-IND：取得官方数据并验证 board/lot 安全划分后再作为 2026 外部工业测试。
- 控制实验总计 20 次；`pilot.py` 会先执行 DsPCBSD+ 验证集 6 次配对训练、3 次机制诊断和 2 次延迟基准，再自动决定是否运行全部正式矩阵。
- 当前可运行比较组包括 YOLOv10-M、YOLO11-M、YOLOv12-M、YOLO26-M、RF-DETR-M、RT-DETRv2-S、D-FINE-M 和 DEIM-D-FINE-M；RT-DETRv4/DEIMv2 为资源允许时的延后组。

## Pilot 继续/停止标准

三种子验证集必须同时满足：平均 AP50:95 增益至少 0.01、路由富集大于 1.0、同硬件延迟比不高于 1.20。任何一项失败，都应先修改方法或收缩论文主张，不进入 7-seed 正式测试。

## 投稿判断

- 现在：不具备 CCF-C 投稿所需的正式结果，更不具备 CCF-B 证据。
- 完成 20 次控制实验、统一 SOTA 重训、机制与鲁棒性分析后：若增益稳定且统计显著，可形成 CCF-C 候选。
- CCF-B：除上述条件外，还需要在 PCB-IND 或另一真实工业外部集上验证泛化，并对 YOLO26/RF-DETR 等 2026 强基线保持有说服力的精度—延迟权衡。

## 下一步

1. 审查变更后执行 `git add` 和 `git commit`。
2. 运行三种子验证集消融与 pilot，不读取最终 test。
3. 达到 pilot 门槛后运行 `python -m experiment.paper_b.run plan` 中的 20 次控制实验。
4. 运行统一 SOTA、机制、鲁棒性和跨域实验。
5. 用 `freeze_results` 固化结果，再由 `stats` 生成论文表格。
