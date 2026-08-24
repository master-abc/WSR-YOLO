from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from algorithm.dwgsa import DWGSARouter, WSRStable
from experiment.paper_b.corruptions import CORRUPTIONS, corrupt
from experiment.paper_b.ablation import subset_coco_annotations, validation_run_directory
from experiment.paper_b.coco_evaluator import evaluate_coco_predictions, predict_yolo_to_coco
from experiment.paper_b.external_detr import prediction_rows
from experiment.paper_b.external_ultralytics import normalize_prediction_image_ids
from experiment.paper_b.frequency_interventions import haar_filters, intervene
from experiment.paper_b.pilot import benchmark_path, diagnostics_path, evaluate_gate, result_path
from experiment.paper_b.pretrained import remap_source_key, transfer_pretrained
from experiment.paper_b.common import sha256_file


PROJECT_DIR = Path(__file__).resolve().parents[3]


class PaperBCoreTests(unittest.TestCase):
    def test_inserted_layer_key_remapping(self):
        self.assertEqual(remap_source_key("model.4.cv.weight", [5]), "model.4.cv.weight")
        self.assertEqual(remap_source_key("model.5.cv.weight", [5]), "model.6.cv.weight")
        self.assertEqual(remap_source_key("model.7.cv.weight", [5, 8]), "model.9.cv.weight")

    @unittest.skipUnless((PROJECT_DIR / "yolo11m.pt").exists(), "local YOLO11m weights unavailable")
    def test_wsr_reuses_nearly_all_pretrained_parameters(self):
        from algorithm.register import register_custom_modules

        register_custom_modules()
        from ultralytics import YOLO

        target = YOLO(str(PROJECT_DIR / "experiment" / "configs" / "dwgsa_router_yolo11m_p3.yaml"))
        report = transfer_pretrained(target, PROJECT_DIR / "yolo11m.pt", YOLO, 0.99)
        self.assertEqual(report["inserted_target_layers"], [5])
        self.assertGreaterEqual(report["loaded_parameter_fraction"], 0.99)

    def test_unified_coco_evaluator_perfect_prediction(self):
        try:
            import pycocotools  # noqa: F401
        except ImportError:
            self.skipTest("pycocotools is not installed")
        annotation = {
            "info": {},
            "licenses": [],
            "images": [{"id": 1, "file_name": "one.png", "width": 32, "height": 32}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 0,
                    "bbox": [4.0, 5.0, 10.0, 11.0],
                    "area": 110.0,
                    "iscrowd": 0,
                }
            ],
            "categories": [{"id": 0, "name": "defect"}],
        }
        prediction = [
            {
                "image_id": 1,
                "category_id": 0,
                "bbox": [4.0, 5.0, 10.0, 11.0],
                "score": 0.99,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = root / "annotations.json"
            predictions = root / "predictions.json"
            annotations.write_text(json.dumps(annotation), encoding="utf-8")
            predictions.write_text(json.dumps(prediction), encoding="utf-8")
            report = evaluate_coco_predictions(annotations, predictions)
        self.assertAlmostEqual(report["metrics"]["map50_95"], 1.0, places=6)

    def test_yolo_prediction_export_explicitly_chunks_path_lists(self):
        class Boxes:
            def __init__(self):
                self.xyxy = torch.tensor([[1.0, 2.0, 5.0, 8.0]])
                self.conf = torch.tensor([0.9])
                self.cls = torch.tensor([0.0])

            def __len__(self):
                return 1

        class FakeYolo:
            def __init__(self):
                self.calls = []

            def predict(self, source, **kwargs):
                self.calls.append((list(source), kwargs["batch"]))
                return iter(SimpleNamespace(boxes=Boxes()) for _ in source)

        annotation = {
            "images": [
                {"id": value, "file_name": f"{value}.png", "width": 16, "height": 16}
                for value in range(1, 6)
            ],
            "annotations": [],
            "categories": [{"id": 7, "name": "defect"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = root / "annotations.json"
            predictions = root / "predictions.json"
            annotations.write_text(json.dumps(annotation), encoding="utf-8")
            for image in annotation["images"]:
                (root / image["file_name"]).write_bytes(b"fixture")
            yolo = FakeYolo()
            predict_yolo_to_coco(
                yolo, annotations, root, predictions, 640, 2, "cpu", 0.001, 0.7, 300
            )
            payload = json.loads(predictions.read_text(encoding="utf-8"))
        self.assertEqual([len(paths) for paths, _ in yolo.calls], [2, 2, 1])
        self.assertEqual([batch for _, batch in yolo.calls], [2, 2, 1])
        self.assertEqual([item["image_id"] for item in payload], [1, 2, 3, 4, 5])
        self.assertEqual({item["category_id"] for item in payload}, {7})

    def test_detr_evaluator_predictions_are_normalized_to_coco_rows(self):
        evaluator = SimpleNamespace(
            coco_eval={
                "bbox": SimpleNamespace(
                    cocoDt=SimpleNamespace(
                        dataset={
                            "annotations": [
                                {
                                    "id": 99,
                                    "image_id": np.int64(3),
                                    "category_id": np.int64(7),
                                    "bbox": [np.float32(1), np.float32(2), np.float32(4), np.float32(5)],
                                    "score": np.float32(0.75),
                                }
                            ]
                        }
                    )
                )
            }
        )
        self.assertEqual(
            prediction_rows(evaluator),
            [{"image_id": 3, "category_id": 7, "bbox": [1.0, 2.0, 4.0, 5.0], "score": 0.75}],
        )

    def test_validator_filename_ids_map_to_locked_coco_ids(self):
        annotations = {
            "images": [
                {"id": 17, "file_name": "board_001.jpg"},
                {"id": 18, "file_name": "00000018_original_board.jpg"},
            ],
            "annotations": [],
            "categories": [{"id": 0, "name": "defect"}],
        }
        predictions = [
            {"image_id": "board_001", "category_id": 0, "bbox": [1, 2, 3, 4], "score": 0.5},
            {"image_id": "original_board", "category_id": 0, "bbox": [2, 3, 4, 5], "score": 0.6},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation_path = root / "annotations.json"
            prediction_path = root / "predictions.json"
            annotation_path.write_text(json.dumps(annotations), encoding="utf-8")
            prediction_path.write_text(json.dumps(predictions), encoding="utf-8")
            changed = normalize_prediction_image_ids(annotation_path, prediction_path)
            observed = json.loads(prediction_path.read_text(encoding="utf-8"))
        self.assertTrue(changed)
        self.assertEqual(observed[0]["image_id"], 17)
        self.assertEqual(observed[1]["image_id"], 18)

    def test_smoke_coco_subset_never_adds_images(self):
        payload = {
            "images": [{"id": value, "file_name": f"{value}.jpg"} for value in (3, 1, 2)],
            "annotations": [
                {"id": value, "image_id": value, "category_id": 0} for value in (1, 2, 3)
            ],
            "categories": [{"id": 0, "name": "defect"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "full.json"
            target = root / "smoke.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            subset_coco_annotations(source, target, 2)
            observed = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual([item["id"] for item in observed["images"]], [1, 2])
        self.assertEqual({item["image_id"] for item in observed["annotations"]}, {1, 2})

    def test_haar_analysis_synthesis_is_exact(self):
        value = torch.rand(1, 3, 33, 35)
        padded = F.pad(value, (0, 1, 0, 1), mode="reflect")
        weight = haar_filters(3, padded.dtype, padded.device)
        bands = F.conv2d(padded, weight, stride=2, groups=3)
        restored = F.conv_transpose2d(bands, weight, stride=2, groups=3)[:, :, :33, :35]
        self.assertLess(float((value - restored).abs().max()), 1e-5)

    def test_route_budget_and_router_gradient(self):
        module = DWGSARouter(64, route_ratio=0.125).enable_diagnostics().train()
        value = torch.randn(2, 64, 20, 20, requires_grad=True)
        output = module(value)
        self.assertEqual(output.shape, value.shape)
        self.assertTrue(torch.allclose(module.last_route_mask.mean((1, 2, 3)), torch.tensor([0.125, 0.125])))
        torch.manual_seed(9)
        (output * torch.randn_like(output)).mean().backward()
        gradient = module.context_router.router[-1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(int(torch.count_nonzero(gradient)), 0)
        neighborhood_gradient = module.sparse_refine.neighborhood_logits.grad
        self.assertIsNotNone(neighborhood_gradient)
        self.assertGreater(float(neighborhood_gradient.abs().sum()), 0.0)

    def test_depthwise_neighborhood_matches_explicit_unfold(self):
        torch.manual_seed(17)
        value = torch.randn(2, 5, 7, 9)
        logits = torch.randn(9, requires_grad=True)
        weights = logits.softmax(0)
        padded = F.pad(value, (1, 1, 1, 1), mode="replicate")
        neighborhoods = F.unfold(padded, kernel_size=3).view(2, 5, 9, 7, 9)
        explicit = (neighborhoods * weights.view(1, 1, 9, 1, 1)).sum(2)
        kernel = weights.view(1, 1, 3, 3).expand(5, 1, 3, 3).contiguous()
        depthwise = F.conv2d(padded, kernel, groups=5)
        self.assertTrue(torch.allclose(depthwise, explicit, atol=1e-6, rtol=1e-6))

    def test_stable_wsr_starts_near_identity_and_preserves_unrouted_tokens(self):
        torch.manual_seed(23)
        module = WSRStable(64, route_ratio=0.125, residual_init=0.1).enable_diagnostics().eval()
        value = torch.randn(2, 64, 20, 20)
        output = module(value)
        relative_change = (output - value).abs().mean() / value.abs().mean()
        self.assertLess(float(relative_change), 0.1)

        sparse_change = (output[:, 32:] - value[:, 32:]).abs().sum(1, keepdim=True)
        unselected = module.last_route_mask == 0
        self.assertEqual(int(torch.count_nonzero(sparse_change[unselected])), 0)

    def test_stable_router_is_uniform_scale_invariant(self):
        torch.manual_seed(29)
        module = WSRStable(64, route_ratio=0.125).enable_diagnostics().eval()
        value = torch.rand(2, 64, 20, 20) + 0.1
        module(value)
        first_mask = module.last_route_mask.clone()
        module(value * 3.0)
        self.assertTrue(torch.equal(first_mask, module.last_route_mask))

    def test_stable_wsr_accepts_asymmetric_residual_initialization(self):
        module = WSRStable(
            64,
            route_ratio=0.125,
            residual_init=0.35,
            sparse_residual_init=0.10,
        )
        expected = torch.tensor([0.35, 0.10], dtype=module.residual_scale.dtype)
        self.assertTrue(torch.equal(module.residual_scale.detach(), expected))

    def test_counterfactuals_and_corruptions_preserve_shape(self):
        image = np.random.default_rng(7).integers(0, 256, (33, 35, 3), dtype=np.uint8)
        for mode in ("low_only", "high_only", "remove_lh", "remove_hl", "remove_hh"):
            result = intervene(image, mode)
            self.assertEqual(result.shape, image.shape)
            self.assertEqual(result.dtype, np.uint8)
        for kind in CORRUPTIONS:
            result = corrupt(image, kind, 3, np.random.default_rng(11))
            self.assertEqual(result.shape, image.shape)
            self.assertEqual(result.dtype, np.uint8)

    def test_validation_only_pilot_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_file = root / "paper_b.yaml"
            protocol_file.write_text("protocol_version: 2\n", encoding="utf-8")
            protocol = {
                "_path": protocol_file,
                "_output_root": root / "generated",
                "pilot_gate": {
                    "dataset": "fixture",
                    "seeds": [13, 42, 3407],
                    "baseline": "baseline",
                    "candidate": "candidate",
                    "formal_model": "formal_candidate",
                    "validation_only": True,
                    "mechanism_split": "val",
                    "latency_split": "val",
                    "latency_seed": 13,
                    "minimum_mean_ap50_95_gain": 0.01,
                    "minimum_route_enrichment": 1.0,
                    "maximum_latency_ratio": 1.2,
                },
            }
            for seed in protocol["pilot_gate"]["seeds"]:
                for model, metric in (("baseline", 0.30), ("candidate", 0.315)):
                    output = result_path(protocol, model, seed)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    weights = output.parent / "best.pt"
                    weights.write_bytes(f"{model}-{seed}".encode())
                    output.write_text(
                        json.dumps(
                            {
                                "schema_version": 2,
                                "track": "ablation_validation_only",
                                "smoke": False,
                                "model": model,
                                "seed": seed,
                                "budget_profile": "pilot",
                                "selection_split": "val",
                                "test_evaluated": False,
                                "weights_sha256": sha256_file(weights),
                                "architecture_definition_sha256": "selected-architecture",
                                "metrics": {"map50_95": metric},
                                "environment_at_start": {"git_dirty": False, "git_commit": "abc123"},
                                "pretrained_transfer": {
                                    "loaded_parameter_fraction": 1.0,
                                    "minimum_parameter_fraction": 0.99,
                                },
                                "protocol_sha256": sha256_file(protocol_file),
                            }
                        ),
                        encoding="utf-8",
                    )
                candidate = json.loads(
                    result_path(protocol, "candidate", seed).read_text(encoding="utf-8")
                )
                diagnostic = diagnostics_path(protocol, seed)
                diagnostic.write_text(
                    json.dumps(
                        {
                            "selection_split": "val",
                            "weights_sha256": candidate["weights_sha256"],
                            "summary": {"route_enrichment": {"mean": 1.5}},
                        }
                    ),
                    encoding="utf-8",
                )
            for model, latency in (("baseline", 10.0), ("candidate", 11.0)):
                result = json.loads(
                    result_path(protocol, model, 13).read_text(encoding="utf-8")
                )
                benchmark = benchmark_path(protocol, model)
                benchmark.write_text(
                    json.dumps(
                        {
                            "selection_split": "val",
                            "weights_sha256": result["weights_sha256"],
                            "model_only": {"mean_ms": latency},
                        }
                    ),
                    encoding="utf-8",
                )
            with patch(
                "experiment.paper_b.pilot.environment_snapshot",
                return_value={"git_dirty": False, "git_commit": "abc123"},
            ):
                decision = evaluate_gate(protocol)
            self.assertEqual(decision["status"], "PASS")
            self.assertFalse(decision["test_evaluated"])
            self.assertAlmostEqual(decision["measurements"]["mean_map50_95_gain"], 0.015)

    def test_pilot_results_are_isolated_from_ablation_results(self):
        protocol = {
            "_output_root": Path("generated"),
            "pilot_gate": {
                "dataset": "fixture",
                "validation_only": True,
                "mechanism_split": "val",
                "latency_split": "val",
            },
        }
        path = result_path(protocol, "candidate", 13)
        self.assertEqual(
            path,
            Path("generated/runs/pilot/fixture/candidate/seed_13/ablation_result.json"),
        )

    def test_ablation_results_are_isolated_by_protocol_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_file = root / "paper_b.yaml"
            protocol_file.write_text("protocol_version: 2\n", encoding="utf-8")
            protocol = {
                "_path": protocol_file,
                "_output_root": root / "generated",
                "ablation_track": {"dataset": "fixture"},
            }
            path = validation_run_directory(
                protocol, "candidate", 13, False, "ablation"
            )
            self.assertEqual(path.parts[-5], "ablation")
            self.assertEqual(path.parts[-2:], ("candidate", "seed_13"))
            self.assertEqual(len(path.parts[-4]), 12)
if __name__ == "__main__":
    unittest.main()
