import json

import cv2
import numpy as np
import pytest

from experiment.paper_b.single_model_inference import encode_pair, load_frozen_policy


def _policy(tmp_path, **model_input):
    payload = {
        "test_evaluated_during_selection": False,
        "model_input": {
            "encoding": "context",
            "single_model_forward_pass": True,
            "noise_floor": 3.0,
            "gain": 4.0,
            **model_input,
        },
    }
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_policy_requires_frozen_input_contract(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps({"test_evaluated_during_selection": False}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="model_input"):
        load_frozen_policy(path)


def test_encode_pair_uses_frozen_context_parameters(tmp_path):
    candidate = np.full((16, 16), 100, dtype=np.uint8)
    candidate[4:8, 5:9] = 130
    reference = np.full((16, 16), 100, dtype=np.uint8)
    candidate_path = tmp_path / "candidate.png"
    reference_path = tmp_path / "reference.png"
    assert cv2.imwrite(str(candidate_path), candidate)
    assert cv2.imwrite(str(reference_path), reference)
    policy = load_frozen_policy(_policy(tmp_path))
    encoded = encode_pair(candidate_path, reference_path, policy)
    assert encoded.shape == (16, 16, 3)
    assert not np.array_equal(encoded[:, :, 1], encoded[:, :, 2])
