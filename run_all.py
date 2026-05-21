"""
DWGSA-YOLO 入口脚本。

顺序运行 exp1-exp5 的 run.py。每个子实验都有自己的 resume 机制，
中断后重跑会自动跳过已完成方法并从 last.pt 续训未完成方法。

所有配置（包括运行模式 mode: full/smoke）从 experiment/configs/experiment.yaml 读取。

Usage:
    python run_all.py   # 运行全部实验，使用 experiment.yaml 中的配置
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

EXP_SCRIPTS = {
    "exp1": PROJECT_ROOT / "experiment" / "exp1" / "run.py",
    "exp2": PROJECT_ROOT / "experiment" / "exp2" / "run.py",
    "exp3": PROJECT_ROOT / "experiment" / "exp3" / "run.py",
    "exp4": PROJECT_ROOT / "experiment" / "exp4" / "run.py",
    "exp5": PROJECT_ROOT / "experiment" / "exp5" / "run.py",
}


def main():
    targets = list(EXP_SCRIPTS.keys())

    for name in targets:
        script = EXP_SCRIPTS[name]
        cmd = [sys.executable, str(script)]
        print(f"\n{'='*70}")
        print(f"  Running {name}: {script}")
        print(f"{'='*70}")
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if result.returncode != 0:
            print(f"\n[FAILED] {name} returned exit code {result.returncode}")
            sys.exit(result.returncode)

    print(f"\n{'='*70}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
