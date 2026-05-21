"""
DWGSA-YOLO 入口脚本。

顺序运行 exp1/exp2/exp3/exp4 的 run.py。每个子实验都有自己的 resume 机制，
中断后重跑会自动跳过已完成方法并从 last.pt 续训未完成方法。

Usage:
    python run_all.py              # 运行全部 4 个实验（完整训练）
    python run_all.py --smoke      # smoke test（每实验 2 epochs）
    python run_all.py --only exp1  # 只跑 exp1

注意：
- 这是推荐入口。experiment/scripts/train_multigpu.py 是旧版多进程调度脚本，
  缺少 resume / FPS 测量，仅在需要并行多 GPU 实验时使用。
"""

import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(description="DWGSA-YOLO suite runner")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test for every experiment (2 epochs, batch=2)")
    parser.add_argument("--only", choices=list(EXP_SCRIPTS.keys()) + ["all"], default="all",
                        help="Run only one experiment (default: all in order)")
    return parser.parse_args()


def main():
    args = parse_args()
    targets = list(EXP_SCRIPTS.keys()) if args.only == "all" else [args.only]

    for name in targets:
        script = EXP_SCRIPTS[name]
        cmd = [sys.executable, str(script)]
        if args.smoke:
            cmd.append("--smoke")
        print(f"\n{'='*70}")
        print(f"  Running {name}: {script}")
        print(f"  Command: {' '.join(cmd)}")
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
