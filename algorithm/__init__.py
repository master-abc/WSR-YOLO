"""DWGSA-YOLO Algorithm Package."""

from algorithm.fdsa import (
    FDSA, FDSAFreqOnly, FDSASpatOnly, FDSANoGate, FDSANoFreqLearn,
    EMA, SimAM,
)
from algorithm.cbam import CBAM
from algorithm.coordatt import CoordAtt
from algorithm.dwgsa import (
    DWGSA, DWGSAWaveOnly, DWGSASparseOnly,
    DWGSANoGeoPrior, DWGSANoAdaptive, DWGSASingleLevel,
    DWGSARouter, WSR, WSRStable, MatchedConvResidual, ScaleOnlyControl,
)
from algorithm.register import register_custom_modules

__all__ = [
    # DWGSA (proposed)
    "DWGSA",
    "DWGSAWaveOnly",
    "DWGSASparseOnly",
    "DWGSANoGeoPrior",
    "DWGSANoAdaptive",
    "DWGSASingleLevel",
    "DWGSARouter",
    "WSR",
    "WSRStable",
    "MatchedConvResidual",
    "ScaleOnlyControl",
    # FDSA (previous)
    "FDSA",
    "FDSAFreqOnly",
    "FDSASpatOnly",
    "FDSANoGate",
    "FDSANoFreqLearn",
    # Comparisons
    "EMA",
    "SimAM",
    "CBAM",
    "CoordAtt",
    "register_custom_modules",
]
