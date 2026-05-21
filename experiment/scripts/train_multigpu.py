"""
DWGSA-YOLO 多 GPU 并行训练调度脚本（旧版入口）。

⚠️ 推荐用法：直接运行 `python run_all.py`（在项目根目录），它会顺序调用
   experiment/expN/run.py，自带断点续跑、增量保存、FPS 测量。

本脚本的适用场景：你有 >=2 张 GPU 且希望多个实验并行训练，可接受没有 resume 机制。
所有参数从 experiment/configs/experiment.yaml 读取。
"""

import sys
import subprocess
import time
import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # DWGSA-YOLO/
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "experiment" / "configs" / "experiment.yaml"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_train_script(exp, gpu_ids, data_cfg, project_dir, train_args, batch_override=None):
    """生成单个实验的训练 Python 脚本。"""
    model_cfg = exp["config"]
    pretrained = exp.get("pretrained", None)
    device_str = ",".join(str(g) for g in gpu_ids)
    batch = batch_override or train_args["batch"] * len(gpu_ids)

    root_str = str(PROJECT_ROOT).replace("\\", "/")
    configs_dir = str(PROJECT_ROOT / "experiment" / "configs").replace("\\", "/")
    data_path = str(PROJECT_ROOT / "experiment" / "configs" / data_cfg).replace("\\", "/")
    project_path = str(PROJECT_ROOT / project_dir).replace("\\", "/")

    # 所有自定义 YAML 配置都需要注册自定义模块
    model_path_str = pretrained if pretrained else str(PROJECT_ROOT / "experiment" / "configs" / model_cfg).replace("\\", "/")
    register_block = (
        f'import sys; sys.path.insert(0, "{root_str}")\n'
        f'from algorithm.register import register_custom_modules\n'
        f'register_custom_modules()\n'
    )

    exp_name = exp["name"]
    exp_desc = exp["desc"]
    result_path = f"{root_str}/results/{exp_name}_result.json"

    ta = train_args
    script = f'''{register_block}
from ultralytics import YOLO
import json, os

model = YOLO("{model_path_str}")
results = model.train(
    data="{data_path}",
    device="{device_str}",
    batch={batch},
    project="{project_path}",
    name="{exp_name}",
    epochs={ta["epochs"]},
    imgsz={ta["imgsz"]},
    optimizer="{ta["optimizer"]}",
    lr0={ta["lr0"]},
    momentum={ta["momentum"]},
    weight_decay={ta["weight_decay"]},
    warmup_epochs={ta["warmup_epochs"]},
    warmup_momentum={ta["warmup_momentum"]},
    warmup_bias_lr={ta["warmup_bias_lr"]},
    cos_lr={ta["cos_lr"]},
    amp={ta["amp"]},
    mosaic={ta["mosaic"]},
    mixup={ta["mixup"]},
    hsv_h={ta["hsv_h"]},
    hsv_s={ta["hsv_s"]},
    hsv_v={ta["hsv_v"]},
    flipud={ta["flipud"]},
    fliplr={ta["fliplr"]},
    verbose=True,
    seed={ta["seed"]},
    exist_ok=True,
    workers={ta["workers"]},
)

metrics = model.val(data="{data_path}", split="test")
result = {{
    "name": "{exp_desc}",
    "config": "{exp_name}",
    "map50": float(metrics.box.map50),
    "map50_95": float(metrics.box.map),
    "precision": float(metrics.box.mp),
    "recall": float(metrics.box.mr),
    "f1": float(2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr + 1e-8)),
}}
result_file = "{result_path}"
os.makedirs(os.path.dirname(result_file), exist_ok=True)
with open(result_file, "w") as f:
    json.dump(result, f, indent=2)
print(f"[DONE] {exp_desc}: mAP@.5={{metrics.box.map50:.4f}}")
'''
    return script


def run_parallel(experiments, gpu_ids, gpus_per_exp, data_cfg, project_dir, train_args):
    """并行运行一组实验，按 GPU 容量分批调度。"""
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    max_concurrent = len(gpu_ids) // gpus_per_exp
    if max_concurrent < 1:
        max_concurrent = 1

    pending = list(enumerate(experiments))
    active = []  # [(exp, proc, gpu_slot_index)]

    def assign_gpus(slot_idx):
        start = slot_idx * gpus_per_exp
        return gpu_ids[start:start + gpus_per_exp]

    print(f"  Max concurrent: {max_concurrent} (GPUs: {len(gpu_ids)}, per_exp: {gpus_per_exp})")

    while pending or active:
        # 启动新实验填满空闲 slot
        while pending and len(active) < max_concurrent:
            i, exp = pending.pop(0)
            slot_idx = len(active)
            # 找一个空闲 slot
            used_slots = {s for _, _, s in active}
            for s in range(max_concurrent):
                if s not in used_slots:
                    slot_idx = s
                    break

            assigned = assign_gpus(slot_idx)
            exp_data = exp.get("data", data_cfg)
            script = generate_train_script(exp, assigned, exp_data, project_dir, train_args)
            script_file = log_dir / f"train_{exp['name']}.py"
            script_file.write_text(script, encoding="utf-8")

            log_file = log_dir / f"{exp['name']}.log"
            print(f"  [LAUNCH] {exp['desc']:<35} GPU {assigned} -> {log_file.name}")

            with open(log_file, "w") as lf:
                proc = subprocess.Popen(
                    [sys.executable, str(script_file)],
                    stdout=lf, stderr=subprocess.STDOUT,
                    cwd=str(PROJECT_ROOT),
                )
            active.append((exp, proc, slot_idx))

        # 检查已完成的实验
        for item in active[:]:
            exp, proc, slot_idx = item
            retcode = proc.poll()
            if retcode is not None:
                status = "OK" if retcode == 0 else f"FAIL(exit={retcode})"
                print(f"  [{status}] {exp['desc']}")
                active.remove(item)

        if active:
            time.sleep(30)


def run_sequential(experiments, gpu_ids, gpus_per_exp, data_cfg, project_dir, train_args):
    """顺序运行一组实验。"""
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    assigned = gpu_ids[:gpus_per_exp]

    for i, exp in enumerate(experiments, 1):
        exp_data = exp.get("data", data_cfg)
        print(f"\n  [{i}/{len(experiments)}] {exp['desc']} on GPU {assigned}")

        script = generate_train_script(exp, assigned, exp_data, project_dir, train_args)
        script_file = log_dir / f"train_{exp['name']}.py"
        script_file.write_text(script, encoding="utf-8")

        result = subprocess.run(
            [sys.executable, str(script_file)],
            cwd=str(PROJECT_ROOT),
        )
        status = "OK" if result.returncode == 0 else "FAILED"
        print(f"  [{status}] {exp['desc']}")


def print_results():
    """汇总打印所有实验结果。"""
    results_dir = PROJECT_ROOT / "results"
    if not results_dir.exists():
        return

    print("\n" + "=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)
    print(f"  {'Model':<35} {'mAP@.5':>8} {'mAP@.5:.95':>12} {'P':>6} {'R':>6} {'F1':>6}")
    print("  " + "-" * 73)

    for f in sorted(results_dir.glob("*_result.json")):
        with open(f) as fp:
            r = json.load(fp)
            print(f"  {r['name']:<35} {r['map50']:>8.4f} {r['map50_95']:>12.4f} "
                  f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r.get('f1', 0):>6.3f}")


def run_phase(phase_cfg, hw_cfg, train_args, default_data, phase_name):
    """运行单个 phase 的所有实验。"""
    experiments = phase_cfg["experiments"]
    project_dir = phase_cfg["project"]
    sequential = phase_cfg.get("sequential", hw_cfg.get("sequential", False))

    gpu_ids = list(range(hw_cfg.get("num_gpus", 1)))
    gpus_per_exp = hw_cfg.get("gpus_per_exp", 1)

    print(f"\n{'=' * 70}")
    print(f"  {phase_name} ({len(experiments)} experiments)")
    print(f"{'=' * 70}")

    if sequential:
        run_sequential(experiments, gpu_ids, gpus_per_exp, default_data, project_dir, train_args)
    else:
        run_parallel(experiments, gpu_ids, gpus_per_exp, default_data, project_dir, train_args)


def main():
    cfg = load_config()

    train_args = cfg["train"]
    hw_cfg = cfg["hardware"]
    default_data = cfg.get("default_data", cfg.get("datasets", {}).get("deeppcb", "deeppcb.yaml"))

    print("=" * 70)
    print("  DWGSA-YOLO Experiment Suite")
    print(f"  Config: {CONFIG_PATH}")
    print(f"  GPUs: {hw_cfg.get('num_gpus', 1)} | {hw_cfg.get('gpus_per_exp', 1)} per experiment")
    print(f"  Epochs: {train_args['epochs']} | Batch: {train_args['batch']} | ImgSz: {train_args['imgsz']}")
    print("=" * 70)

    # 按顺序执行所有实验
    datasets_cfg = cfg.get("datasets", {})
    configs_dir = PROJECT_ROOT / "experiment" / "configs"

    if "exp1" in cfg:
        exp1 = cfg["exp1"]
        data_cfg = datasets_cfg.get(exp1.get("data", "deeppcb"), default_data)
        phase_cfg = {
            "experiments": exp1["methods"],
            "project": f"experiment/{exp1['output_dir']}/runs",
        }
        run_phase(phase_cfg, hw_cfg, train_args, data_cfg, "Exp1: SOTA Comparison")

    if "exp2" in cfg:
        exp2 = cfg["exp2"]
        data_cfg = datasets_cfg.get(exp2.get("data", "deeppcb"), default_data)
        phase_cfg = {
            "experiments": exp2["variants"],
            "project": f"experiment/{exp2['output_dir']}/runs",
        }
        run_phase(phase_cfg, hw_cfg, train_args, data_cfg, "Exp2: Ablation Study")

    if "exp3" in cfg:
        exp3 = cfg["exp3"]
        exp3_experiments = []
        for e in exp3["experiments"]:
            resolved = dict(e)
            resolved["data"] = datasets_cfg.get(e.get("data", "deeppcb"), default_data)
            exp3_experiments.append(resolved)
        phase_cfg = {
            "experiments": exp3_experiments,
            "project": f"experiment/{exp3['output_dir']}/runs",
        }
        run_phase(phase_cfg, hw_cfg, train_args, default_data, "Exp3: Cross-Dataset Generalization")

    print_results()


if __name__ == "__main__":
    main()
