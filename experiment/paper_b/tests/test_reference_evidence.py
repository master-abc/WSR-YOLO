import numpy as np

from experiment.paper_b.reference_evidence import (
    box_change_evidence,
    box_structural_change_evidence,
)
from experiment.paper_b.single_model_operating_point import negative_metrics


def test_box_change_evidence_reads_context_channel_separation():
    encoded = np.full((20, 20, 3), 100, dtype=np.uint8)
    encoded[5:15, 5:15, 1] = 130
    encoded[5:15, 5:15, 2] = 70
    assert box_change_evidence(encoded, [5, 5, 15, 15]) == 30.0
    assert box_change_evidence(encoded, [0, 0, 4, 4]) == 0.0


def test_negative_metrics_applies_reference_evidence_gate():
    audit = {
        "per_image": [
            {
                "detections": [
                    {"class_id": 0, "score": 0.9, "change_evidence": 2.0},
                    {"class_id": 1, "score": 0.8, "change_evidence": 12.0},
                ]
            }
        ]
    }
    metrics = negative_metrics(audit, {0: 0.5, 1: 0.5}, 5.0)
    assert metrics["false_positives"] == 1


def test_structural_evidence_rejects_thin_edges_and_keeps_blob():
    encoded = np.full((30, 30, 3), 100, dtype=np.uint8)
    encoded[:, 10, 1] = 140
    encoded[:, 10, 2] = 60
    assert box_structural_change_evidence(encoded, [0, 0, 30, 30]) == 0.0
    encoded[12:18, 12:18, 1] = 140
    encoded[12:18, 12:18, 2] = 60
    assert box_structural_change_evidence(encoded, [0, 0, 30, 30]) >= 4.0
