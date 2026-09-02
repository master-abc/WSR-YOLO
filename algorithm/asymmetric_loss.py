"""Classification losses for false-alarm-aware detector refinement.

The standard Ultralytics detection loss applies BCE to every class/anchor pair.
On PCB images this produces an extreme imbalance: millions of already-correct
background logits can dominate the relatively small number of defect logits.
The asymmetric focal wrapper keeps the full positive gradient while focusing
the negative gradient on high-confidence (hard) false alarms.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricFocalBCE(nn.Module):
    """Elementwise BCE with separate focusing exponents for positives/negatives.

    The returned tensor deliberately has the same shape as ``logits`` because
    :class:`ultralytics.utils.loss.v8DetectionLoss` performs its own assignment,
    class weighting, normalization, and reduction.
    """

    def __init__(
        self,
        gamma_negative: float = 2.0,
        gamma_positive: float = 0.0,
        negative_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if gamma_negative < 0.0 or gamma_positive < 0.0:
            raise ValueError("Focusing exponents must be non-negative")
        if negative_weight <= 0.0:
            raise ValueError("negative_weight must be positive")
        self.gamma_negative = float(gamma_negative)
        self.gamma_positive = float(gamma_positive)
        self.negative_weight = float(negative_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(
                f"logits and targets must have identical shapes: "
                f"{tuple(logits.shape)} != {tuple(targets.shape)}"
            )
        probability = logits.sigmoid()
        positive = targets > 0.0
        positive_modulator = (1.0 - probability).pow(self.gamma_positive)
        negative_modulator = (
            probability.pow(self.gamma_negative) * self.negative_weight
        )
        modulator = torch.where(positive, positive_modulator, negative_modulator)
        return F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        ) * modulator
