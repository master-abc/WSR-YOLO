from __future__ import annotations

import re
from pathlib import Path
from typing import Any

try:
    from .common import sha256_file
except ImportError:
    from common import sha256_file


LAYER_KEY = re.compile(r"^model\.(\d+)(\..+)$")
ROUTER_CLASS_NAMES = {
    "DWGSARouter",
    "WSR",
    "WSRStable",
    "MatchedConvResidual",
    "ScaleOnlyControl",
}


def inserted_router_layers(model: Any) -> list[int]:
    """Return target graph indices occupied by newly inserted WSR blocks."""

    layers = getattr(model, "model", None)
    if layers is None:
        return []
    return [
        index
        for index, layer in enumerate(layers)
        if layer.__class__.__name__ in ROUTER_CLASS_NAMES
    ]


def shift_source_layer(source_index: int, inserted_target_layers: list[int]) -> int:
    """Map a baseline layer index into a graph containing inserted layers."""

    target_index = source_index
    for insertion in sorted(inserted_target_layers):
        if target_index >= insertion:
            target_index += 1
    return target_index


def remap_source_key(key: str, inserted_target_layers: list[int]) -> str:
    match = LAYER_KEY.match(key)
    if match is None:
        return key
    target_index = shift_source_layer(int(match.group(1)), inserted_target_layers)
    return f"model.{target_index}{match.group(2)}"


def transfer_pretrained(
    target_yolo: Any,
    pretrained: str | Path,
    yolo_class: Any,
    minimum_parameter_fraction: float = 0.99,
) -> dict[str, Any]:
    """Load baseline weights after graph insertion and return an auditable report.

    Ultralytics normally matches state-dict keys literally. Inserting WSR at P3
    shifts every later ``model.N`` key, which silently discards most pretrained
    weights. This loader remaps baseline layer numbers around inserted WSR
    modules, then rejects a run when too little of the target model was reused.
    Dataset-specific classification logits are allowed to remain unmatched.
    """

    pretrained = str(pretrained)
    source_yolo = yolo_class(pretrained)
    source_state = source_yolo.model.state_dict()
    target_state = target_yolo.model.state_dict()
    inserted_layers = inserted_router_layers(target_yolo.model)

    compatible: dict[str, Any] = {}
    shape_mismatches: list[dict[str, Any]] = []
    remapped_missing: list[str] = []
    for source_key, value in source_state.items():
        target_key = remap_source_key(source_key, inserted_layers)
        if target_key not in target_state:
            remapped_missing.append(target_key)
            continue
        if tuple(value.shape) != tuple(target_state[target_key].shape):
            shape_mismatches.append(
                {
                    "source": source_key,
                    "target": target_key,
                    "source_shape": list(value.shape),
                    "target_shape": list(target_state[target_key].shape),
                }
            )
            continue
        compatible[target_key] = value

    target_yolo.model.load_state_dict(compatible, strict=False)
    loaded_parameters = sum(target_state[key].numel() for key in compatible)
    target_parameters = sum(value.numel() for value in target_state.values())
    loaded_fraction = loaded_parameters / max(target_parameters, 1)
    missing_target = sorted(set(target_state) - set(compatible))

    report: dict[str, Any] = {
        "loader": "layer_index_remap_v1",
        "pretrained": pretrained,
        "inserted_target_layers": inserted_layers,
        "source_tensors": len(source_state),
        "target_tensors": len(target_state),
        "loaded_tensors": len(compatible),
        "loaded_parameters": loaded_parameters,
        "target_parameters": target_parameters,
        "loaded_parameter_fraction": loaded_fraction,
        "minimum_parameter_fraction": minimum_parameter_fraction,
        "shape_mismatches": shape_mismatches,
        "missing_target_tensor_count": len(missing_target),
        "missing_target_tensors": missing_target,
        "remapped_missing_tensor_count": len(remapped_missing),
    }
    pretrained_path = Path(pretrained)
    if pretrained_path.is_file():
        report["pretrained_sha256"] = sha256_file(pretrained_path)

    if loaded_fraction < minimum_parameter_fraction:
        raise RuntimeError(
            "Pretrained transfer fairness check failed: "
            f"loaded {loaded_fraction:.2%} of target parameters, expected at least "
            f"{minimum_parameter_fraction:.2%}. Inserted layers={inserted_layers}."
        )
    return report

