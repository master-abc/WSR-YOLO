"""
将自定义模块注册到 Ultralytics 框架。

通过 monkey-patch 方式注入，无需修改 ultralytics 源码。
使用前调用 register_custom_modules() 即可在 YAML 配置中使用自定义模块名。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithm.fdsa import (
    FDSA, FDSAFreqOnly, FDSASpatOnly, FDSANoGate, FDSANoFreqLearn,
    EMA, SimAM,
)
from algorithm.dwgsa import (
    DWGSA, DWGSAWaveOnly, DWGSASparseOnly,
    DWGSANoGeoPrior, DWGSANoAdaptive, DWGSASingleLevel, DWGSARouter, WSR,
)
from algorithm.cbam import CBAM
from algorithm.coordatt import CoordAtt


def register_custom_modules():
    """注册自定义模块到 Ultralytics，使 YAML 配置文件可以直接引用模块名。"""
    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules

    custom_modules = {
        # DWGSA 主模块及消融变体
        "DWGSA": DWGSA,
        "DWGSAWaveOnly": DWGSAWaveOnly,
        "DWGSASparseOnly": DWGSASparseOnly,
        "DWGSANoGeoPrior": DWGSANoGeoPrior,
        "DWGSANoAdaptive": DWGSANoAdaptive,
        "DWGSASingleLevel": DWGSASingleLevel,
        "DWGSARouter": DWGSARouter,
        "WSR": WSR,
        # FDSA（前代模块，保留用于对比）
        "FDSA": FDSA,
        "FDSAFreqOnly": FDSAFreqOnly,
        "FDSASpatOnly": FDSASpatOnly,
        "FDSANoGate": FDSANoGate,
        "FDSANoFreqLearn": FDSANoFreqLearn,
        # 对比方法
        "EMA": EMA,
        "SimAM": SimAM,
        "CBAM": CBAM,
        "CoordAtt": CoordAtt,
    }

    for name, module_cls in custom_modules.items():
        setattr(modules, name, module_cls)
        setattr(tasks, name, module_cls)

    # 注入到 parse_model 的全局命名空间
    if hasattr(tasks.parse_model, "__wrapped__"):
        orig_func = tasks.parse_model.__wrapped__
    else:
        orig_func = tasks.parse_model

    for name, cls in custom_modules.items():
        orig_func.__globals__[name] = cls

    print(f"[DWGSA-YOLO] Registered {len(custom_modules)} custom modules")
    return custom_modules
