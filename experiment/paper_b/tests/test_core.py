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

from algorithm.dwgsa import DWGSARouter, MatchedConvResidual, ScaleOnlyControl, WSRStable
from experiment.paper_b.corruptions import CORRUPTIONS, corrupt
from experiment.paper_b.ablation import subset_coco_annotations, validation_run_directory
from experiment.paper_b.aggregate_operating_points import aggregate as aggregate_operating_points
from experiment.paper_b.analyze_false_positive import analyze as analyze_false_positives
from experiment.paper_b.coco_evaluator import evaluate_coco_predictions, predict_yolo_to_coco
from experiment.paper_b.confirmatory_summary import MODELS as CONFIRMATORY_MODELS
from experiment.paper_b.confirmatory_summary import SEEDS as CONFIRMATORY_SEEDS
from experiment.paper_b.confirmatory_summary import summarize as summarize_confirmatory
from experiment.paper_b.consensus_ensemble import consensus_detections
from experiment.paper_b.external_detr import prediction_rows
from experiment.paper_b.external_ultralytics import normalize_prediction_image_ids
from experiment.paper_b.frequency_interventions import haar_filters, intervene
from experiment.paper_b.mitigation_summary import _negative_sample_key
from experiment.paper_b.operating_point import operational_metrics, select_threshold
from experiment.paper_b.pilot import benchmark_path, diagnostics_path, evaluate_gate, result_path
from experiment.paper_b.pretrained import remap_source_key
from experiment.paper_b.prepare_hard_negatives import prepare as prepare_hard_negatives
from experiment.paper_b.reference_verifier import box_change_score, sample_key
from experiment.paper_b.validation_postprocess import suppress_overlaps
from experiment.paper_b.common import sha256_file
from experiment.paper_b.stats import format_cell, format_latex_cell, latex_escape


class PaperBCoreTests(unittest.TestCase):
    def test_hard_negative_preparation_ranks_training_inputs_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            positive_list = root / "positive.txt"
            negative_list = root / "negative.txt"
            positive_list.write_text("positive.jpg\n", encoding="utf-8")
            negatives = [root / f"negative_{index}.jpg" for index in range(4)]
            negative_list.write_text(
                "\n".join(str(path) for path in negatives) + "\n", encoding="utf-8"
            )
            data = root / "dataset.yaml"
            data.write_text(
                f"train:\n  - {positive_list}\n  - {negative_list}\nval: unused.txt\n",
                encoding="utf-8",
            )
            audit = root / "audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "per_image": [
                            {"image": str(path), "scores": [score]}
                            for path, score in zip(negatives, (0.1, 0.9, 0.3, 0.8))
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = prepare_hard_negatives(data, audit, root / "derived", 0.5, 3)
            rows = Path(result["negative_list"]).read_text(encoding="utf-8").splitlines()

            self.assertEqual(result["hard_images"], 2)
            self.assertEqual(result["effective_negative_samples"], 8)
            self.assertEqual(rows.count(str(negatives[1].resolve())), 3)
            self.assertEqual(rows.count(str(negatives[3].resolve())), 3)
            self.assertFalse(result["test_images_used_for_selection"])

    def test_consensus_requires_distinct_models_same_class_and_location(self):
        model_rows = [
            {
                (1, 1): [
                    {"score": 0.9, "box": [0, 0, 10, 10]},
                    {"score": 0.8, "box": [1, 1, 11, 11]},
                ],
                (1, 2): [{"score": 0.9, "box": [0, 0, 10, 10]}],
            },
            {(1, 1): [{"score": 0.85, "box": [1, 1, 11, 11]}]},
            {(1, 1): [{"score": 0.95, "box": [30, 30, 40, 40]}]},
        ]

        fused = consensus_detections(
            model_rows,
            base_confidence=0.5,
            match_iou=0.3,
            minimum_votes=2,
        )

        self.assertEqual(set(fused), {(1, 1)})
        self.assertEqual(len(fused[(1, 1)]), 1)
        self.assertEqual(fused[(1, 1)][0]["votes"], 2)

    def test_deeppcb_negative_keys_match_across_hosts_and_suffixes(self):
        self.assertEqual(
            _negative_sample_key("/remote/test/00041000_temp.jpg"),
            _negative_sample_key(r"C:\local\test\00041000_negative.jpg"),
        )
        self.assertEqual(sample_key("00041000_test.jpg"), "00041000")
        self.assertEqual(sample_key("00041000_temp.jpg"), "00041000")

    def test_reference_verifier_box_score_is_bounded(self):
        difference = np.zeros((12, 12), dtype=np.float32)
        self.assertEqual(box_change_score(difference, [3, 3, 4, 4]), 0.0)
        difference[3:7, 3:7] = 255.0
        score = box_change_score(difference, [3, 3, 4, 4])
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_operating_point_aggregate_preserves_seed_pairing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for seed, threshold, fpr in ((42, 0.6, 0.12), (13, 0.7, 0.08)):
                record = {
                    "selection": {"selected_threshold": threshold},
                    "negative_calibration": {"board_false_positive_rate": 0.04},
                    "negative_holdout": {
                        "calibrated": {
                            "board_false_positive_rate": fpr,
                            "false_positives_per_image": fpr * 2,
                        }
                    },
                    "positive_test": {
                        "calibrated": {
                            "overall": {
                                "precision": 0.9,
                                "recall": 0.8,
                                "f1": 0.847,
                            }
                        }
                    },
                }
                path = root / f"seed{seed}.json"
                path.write_text(json.dumps(record), encoding="utf-8")
                paths.append(path)
            result = aggregate_operating_points(paths, [42, 13], root / "out.json")
            self.assertEqual(result["seeds"], [13, 42])
            self.assertAlmostEqual(result["summary"]["holdout_board_fpr"]["mean"], 0.10)

    def test_postprocess_suppression_can_be_class_aware(self):
        predictions = [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
            {"image_id": 1, "category_id": 1, "bbox": [1, 1, 10, 10], "score": 0.8},
            {"image_id": 1, "category_id": 2, "bbox": [1, 1, 10, 10], "score": 0.7},
        ]
        classwise = suppress_overlaps(predictions, 0.5, class_agnostic=False)
        agnostic = suppress_overlaps(predictions, 0.5, class_agnostic=True)
        self.assertEqual(len(classwise), 2)
        self.assertEqual(len(agnostic), 1)

    def test_operating_point_calibration_is_discrete_and_class_aware(self):
        negative_rows = {
            "a": [0.90],
            "b": [0.70, 0.20],
            "c": [0.40],
            "d": [],
        }
        threshold = select_threshold(negative_rows, 0.25)
        self.assertGreater(threshold, 0.70)
        self.assertLessEqual(threshold, 0.90)

        annotations = {
            "categories": [{"id": 1, "name": "defect"}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 10,
                    "category_id": 1,
                    "bbox": [0, 0, 10, 10],
                    "iscrowd": 0,
                }
            ],
        }
        predictions = [
            {"image_id": 10, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
            {"image_id": 10, "category_id": 1, "bbox": [20, 20, 5, 5], "score": 0.8},
            {"image_id": 10, "category_id": 2, "bbox": [0, 0, 10, 10], "score": 0.95},
        ]
        metrics = operational_metrics(annotations, predictions, 0.75)
        self.assertEqual(metrics["overall"]["tp"], 1)
        self.assertEqual(metrics["overall"]["fp"], 2)
        self.assertEqual(metrics["overall"]["fn"], 0)

    def test_confirmatory_summary_validates_complete_paired_design(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for model_index, model in enumerate(CONFIRMATORY_MODELS):
                for seed_index, seed in enumerate(CONFIRMATORY_SEEDS):
                    record = {
                        "model": model,
                        "seed": seed,
                        "budget_profile": "confirmatory",
                        "selection_split": "val",
                        "test_evaluated": False,
                        "protocol_sha256": "one-protocol",
                        "environment_at_start": {"git_commit": "one-commit"},
                        "metrics": {
                            "map50_95": 0.4 + model_index * 0.001 + seed_index * 0.0001
                        },
                        "complexity": {
                            "parameters": 100 + model_index,
                            "gflops": 20.0 + model_index * 0.01,
                        },
                        "pretrained_transfer": {"loaded_parameter_fraction": 1.0},
                    }
                    (root / f"{model}__seed_{seed}.json").write_text(
                        json.dumps(record), encoding="utf-8"
                    )
            output = root / "summary.json"
            summary = summarize_confirmatory(root, output)
            proposed = summary["models"]["confirm_wsr_p3_r25"]
            self.assertEqual(len(summary["models"]), 9)
            self.assertAlmostEqual(
                proposed["paired_delta_vs_yolo11s_points"]["mean"], 0.1
            )
            self.assertEqual(proposed["added_parameters_vs_yolo11s"], 1)
            self.assertTrue(output.is_file())

    def test_negative_input_analysis_is_paired_by_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_path = root / "baseline.json"
            candidate_path = root / "candidate.json"
            output_path = root / "paired.json"
            common = {
                "image_manifest_sha256": "same-manifest",
                "metrics": {"0.1": {}},
            }
            baseline_path.write_text(
                json.dumps(
                    {
                        **common,
                        "per_image": [
                            {"image": "a", "scores": []},
                            {"image": "b", "scores": [0.2]},
                            {"image": "c", "scores": []},
                            {"image": "d", "scores": []},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            candidate_path.write_text(
                json.dumps(
                    {
                        **common,
                        "per_image": [
                            {"image": "a", "scores": [0.2]},
                            {"image": "b", "scores": [0.2, 0.3]},
                            {"image": "c", "scores": []},
                            {"image": "d", "scores": []},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            row = analyze_false_positives(
                baseline_path, candidate_path, output_path
            )["rows"][0]
            self.assertEqual(row["candidate_only_positive_boards"], 1)
            self.assertEqual(row["baseline_only_positive_boards"], 0)
            self.assertAlmostEqual(row["mean_paired_fp_difference"], 0.5)
            self.assertTrue(output_path.is_file())

    def test_statistics_tables_are_latex_safe(self):
        summary = {"mean": 0.4669, "std": 0.0072}
        self.assertEqual(format_cell(summary), "46.69 +/- 0.72")
        self.assertEqual(format_latex_cell(summary), "46.69 $\\pm$ 0.72")
        self.assertEqual(latex_escape("dspcbsd_plus"), r"dspcbsd\_plus")

    def test_inserted_layer_key_remapping(self):
        self.assertEqual(remap_source_key("model.4.cv.weight", [5]), "model.4.cv.weight")
        self.assertEqual(remap_source_key("model.5.cv.weight", [5]), "model.6.cv.weight")
        self.assertEqual(remap_source_key("model.7.cv.weight", [5, 8]), "model.9.cv.weight")

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

    def test_random_router_preserves_budget_and_changes_selection(self):
        torch.manual_seed(31)
        module = DWGSARouter(64, route_ratio=0.25, random_router=True).enable_diagnostics().eval()
        value = torch.randn(1, 64, 20, 20)
        module(value)
        first = module.last_route_mask.clone()
        module(value)
        second = module.last_route_mask.clone()
        self.assertEqual(float(first.mean()), 0.25)
        self.assertEqual(float(second.mean()), 0.25)
        self.assertFalse(torch.equal(first, second))

    def test_fairness_controls_preserve_shape_and_parameter_budget(self):
        value = torch.randn(2, 256, 20, 20)
        matched = MatchedConvResidual(256)
        scale = ScaleOnlyControl(256)
        self.assertEqual(matched(value).shape, value.shape)
        self.assertEqual(scale(value).shape, value.shape)
        parameters = sum(parameter.numel() for parameter in matched.parameters())
        self.assertLess(abs(parameters - 13342) / 13342, 0.02)

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
