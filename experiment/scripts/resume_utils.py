"""
断点续跑工具模块（experiment 公用）。

提供两级 resume 能力：
    1. 调度级：load_existing_results / is_method_completed —— 跳过已成功完成的方法
    2. 训练级：resolve_resume —— 当 runs/<name>/weights/last.pt 存在但训练未完成时
                                 返回该路径并把 model 切换为该 ckpt 启动，让 Ultralytics
                                 自动从断点恢复 (epoch、optimizer state、scheduler)

设计原则：
    - 训练完成的判据：results.json 中存在该 name 且无 "error" 字段 且 best.pt 存在
    - 训练中断的判据：runs/<name>/weights/last.pt 存在但 best.pt 缺失 或 results.json 无记录
    - 中断/失败的方法会被重新启动；如有 last.pt 则从断点继续，否则从 0 开始
    - 每次 run_single 成功后增量写回 results.json，避免后续中断丢已得结果
"""

from pathlib import Path
import json
from typing import Optional


def load_existing_results(results_file: Path) -> dict:
    """加载已存在的 results.json，返回 {name: result_dict}。

    若文件不存在或格式无效，返回空字典（首次运行场景）。
    """
    if not results_file.exists():
        return {}
    try:
        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        results_list = data.get("results", []) if isinstance(data, dict) else data
        return {r["name"]: r for r in results_list if "name" in r}
    except (json.JSONDecodeError, KeyError, OSError):
        return {}


def is_method_completed(name: str, existing: dict, runs_dir: Path) -> bool:
    """判断指定方法是否已成功完成训练。

    需同时满足：
        - results.json 有 name 记录且无 "error" 字段
        - runs/<name>/weights/best.pt 存在
    """
    if name not in existing:
        return False
    if "error" in existing[name]:
        return False
    best_pt = runs_dir / name / "weights" / "best.pt"
    return best_pt.exists()


def resolve_resume(runs_dir: Path, name: str) -> Optional[Path]:
    """检测是否需要从 last.pt 恢复训练。

    返回值:
        - Path: last.pt 存在且 best.pt 不存在（说明上次训练未完成） → 从该 ckpt 恢复
        - None: 没有 last.pt 或已有 best.pt（应该走调度级跳过） → 全新训练
    """
    last_pt = runs_dir / name / "weights" / "last.pt"
    best_pt = runs_dir / name / "weights" / "best.pt"
    if last_pt.exists() and not best_pt.exists():
        return last_pt
    return None


def save_results_incremental(results_file: Path, results_list: list, experiment_meta: dict) -> None:
    """增量保存 results.json（每完成一个方法就写一次）。

    experiment_meta 形如 {"experiment": "exp1_sota_comparison", "config": {...}}，
    timestamp 会自动追加为当前时间。
    """
    from datetime import datetime

    results_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **experiment_meta,
        "timestamp": datetime.now().isoformat(),
        "results": results_list,
    }
    tmp = results_file.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(results_file)
