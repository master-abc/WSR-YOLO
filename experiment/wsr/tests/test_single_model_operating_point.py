from __future__ import annotations

import json
import unittest

from experiment.wsr.single_model_operating_point import (
    choose_checkpoint,
    negative_metrics,
)


class SingleModelOperatingPointTest(unittest.TestCase):
    def test_negative_metrics_apply_classwise_thresholds(self) -> None:
        audit = {
            "per_image": [
                {
                    "detections": [
                        {"class_id": 0, "score": 0.4},
                        {"class_id": 1, "score": 0.8},
                    ]
                },
                {"detections": [{"class_id": 0, "score": 0.7}]},
            ]
        }
        metrics = negative_metrics(audit, {0: 0.6, 1: 0.9})
        self.assertEqual(metrics["alarmed_boards"], 1)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["board_false_positive_rate"], 0.5)

    def test_choose_checkpoint_uses_validation_f1(self) -> None:
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index, (f1, board_fpr) in enumerate(((0.91, 0.0), (0.94, 0.01))):
                path = root / f"policy_{index}.json"
                path.write_text(
                    json.dumps(
                        {
                            "test_evaluated_during_selection": False,
                            "maximum_board_fpr": 0.01,
                            "weights": f"weights_{index}.pt",
                            "weights_sha256": str(index),
                            "positive_validation": {
                                "overall": {
                                    "f1": f1,
                                    "recall": f1,
                                    "precision": f1,
                                }
                            },
                            "negative_validation": {
                                "board_false_positive_rate": board_fpr
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                paths.append(path)
            result = choose_checkpoint(paths, root / "frozen.json")
            self.assertEqual(result["weights"], "weights_1.pt")
            self.assertEqual(len(result["checkpoint_selection"]["candidates"]), 2)


if __name__ == "__main__":
    unittest.main()
