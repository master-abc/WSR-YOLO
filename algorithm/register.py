"""Register the WSR modules with Ultralytics without modifying Ultralytics."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithm.dwgsa import (
    DWGSARouter,
    MatchedConvResidual,
    ScaleOnlyControl,
    WSR,
    WSRStable,
)


def register_custom_modules():
    """Expose the current WSR modules to the Ultralytics YAML parser."""
    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules

    custom_modules = {
        "DWGSARouter": DWGSARouter,
        "WSR": WSR,
        "WSRStable": WSRStable,
        "MatchedConvResidual": MatchedConvResidual,
        "ScaleOnlyControl": ScaleOnlyControl,
    }

    for name, module_cls in custom_modules.items():
        setattr(modules, name, module_cls)
        setattr(tasks, name, module_cls)

    # Inject names into parse_model's global namespace.
    if hasattr(tasks.parse_model, "__wrapped__"):
        orig_func = tasks.parse_model.__wrapped__
    else:
        orig_func = tasks.parse_model

    for name, cls in custom_modules.items():
        orig_func.__globals__[name] = cls

    print(f"[WSR-YOLO] Registered {len(custom_modules)} custom modules")
    return custom_modules
