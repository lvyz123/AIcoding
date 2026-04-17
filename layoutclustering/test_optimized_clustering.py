#!/usr/bin/env python3
"""Tests for lithography-behavior coverage clustering."""

from __future__ import annotations

import json
import shutil
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import gdstk
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import layout_clustering_optimized as optimized
from layout_utils import MarkerRecord, ExactCluster, _canonical_bitmap_hash, _write_oas_library


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, image=np.asarray(image, dtype=np.float32))


def _record(marker_id: str, bitmap: np.ndarray, exact_cluster_id: int = -1) -> MarkerRecord:
    clip_hash, _ = _canonical_bitmap_hash(bitmap)
    h, w = bitmap.shape
    return MarkerRecord(
        marker_id=marker_id,
        source_path="unit.oas",
        source_name="unit.oas",
        marker_bbox=(0.0, 0.0, 0.1, 0.1),
        marker_center=(0.05, 0.05),
        clip_bbox=(0.0, 0.0, float(w), float(h)),
        expanded_bbox=(0.0, 0.0, float(w), float(h)),
        clip_bbox_q=(0, 0, int(w), int(h)),
        expanded_bbox_q=(0, 0, int(w), int(h)),
        marker_bbox_q=(0, 0, 1, 1),
        shift_limits_px={"x": (0, 0), "y": (0, 0)},
        clip_bitmap=np.ascontiguousarray(bitmap, dtype=bool),
        expanded_bitmap=np.ascontiguousarray(bitmap, dtype=bool),
        clip_hash=clip_hash,
        expanded_hash=clip_hash,
        clip_area=float(np.count_nonzero(bitmap)),
        exact_cluster_id=int(exact_cluster_id),
    )


def _bitmap_square(offset: int = 0) -> np.ndarray:
    bitmap = np.zeros((16, 16), dtype=bool)
    bitmap[4 + offset:10 + offset, 4:10] = True
    return bitmap


def _make_oas(path: Path) -> None:
    lib = gdstk.Library()
    cell = gdstk.Cell("TOP")
    centers = [(0.0, 0.0), (3.0, 0.0), (6.0, 0.0)]
    for idx, (cx, cy) in enumerate(centers):
        cell.add(gdstk.rectangle((cx - 0.02, cy - 0.02), (cx + 0.02, cy + 0.02), layer=999, datatype=0))
        if idx < 2:
            cell.add(gdstk.rectangle((cx + 0.10, cy + 0.10), (cx + 0.25, cy + 0.25), layer=1, datatype=0))
        else:
            cell.add(gdstk.rectangle((cx + 0.25, cy + 0.10), (cx + 0.40, cy + 0.25), layer=1, datatype=0))
    lib.add(cell)
    _write_oas_library(lib, str(path))


class OptimizedBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = SCRIPT_DIR / "test_outputs" / "_behavior_unit"
        shutil.rmtree(self.temp_root, ignore_errors=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def _manifest_and_features(self, marker_ids, *, include_epe=False):
        manifest = self.temp_root / "behavior.jsonl"
        rows = []
        features = []
        images = []
        base = np.zeros((16, 16), dtype=np.float32)
        base[4:10, 4:10] = 1.0
        shifted = np.zeros((16, 16), dtype=np.float32)
        shifted[8:14, 4:10] = 1.0
        for idx, marker_id in enumerate(marker_ids):
            image = base if idx < len(marker_ids) - 1 else shifted
            image_path = self.temp_root / f"{marker_id}_aerial.npz"
            _write_image(image_path, image)
            row = {
                "sample_id": marker_id,
                "source_path": "unit.oas",
                "marker_id": marker_id,
                "clip_bbox": [0.0, 0.0, 1.0, 1.0],
                "aerial_npz": str(image_path),
                "risk_score": 2.0 if idx == len(marker_ids) - 1 else 0.0,
            }
            if include_epe:
                epe_path = self.temp_root / f"{marker_id}_epe.npz"
                _write_image(epe_path, image * 0.5)
                row["epe_npz"] = str(epe_path)
            rows.append(row)
            images.append(image)
            if idx < len(marker_ids) - 1:
                features.append([1.0, 0.0])
            else:
                features.append([-1.0, 0.0])
        with manifest.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        feature_npz = self.temp_root / "features.npz"
        np.savez_compressed(feature_npz, sample_ids=np.asarray(marker_ids, dtype=str), features=np.asarray(features, dtype=np.float32))
        return manifest, feature_npz, images

    def test_manifest_and_feature_loading(self):
        manifest, feature_npz, _ = self._manifest_and_features(["m0", "m1"])

        samples = optimized._load_behavior_manifest(manifest)
        features = optimized._load_feature_npz(feature_npz)

        self.assertEqual(sorted(samples), ["m0", "m1"])
        self.assertEqual(features["m0"].shape, (2,))
        self.assertEqual(optimized._available_verification_channels(samples), ["aerial"])

    def test_runner_requires_manifest_features_to_match(self):
        manifest, _, _ = self._manifest_and_features(["m0", "m1"])
        bad_feature_npz = self.temp_root / "bad_features.npz"
        np.savez_compressed(
            bad_feature_npz,
            sample_ids=np.asarray(["m0"], dtype=str),
            features=np.asarray([[1.0, 0.0]], dtype=np.float32),
        )

        with self.assertRaisesRegex(ValueError, "Feature NPZ missing"):
            optimized.OptimizedMainlineRunner(
                config={
                    "marker_layer": "999/0",
                    "behavior_manifest": str(manifest),
                    "feature_npz": str(bad_feature_npz),
                },
                temp_dir=self.temp_root / "bad_run",
            )

    def test_optional_channel_must_be_dataset_level(self):
        manifest, _, _ = self._manifest_and_features(["m0", "m1"])
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        epe_path = self.temp_root / "m0_epe.npz"
        _write_image(epe_path, np.zeros((16, 16), dtype=np.float32))
        rows[0]["epe_npz"] = str(epe_path)
        with manifest.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

        with self.assertRaisesRegex(ValueError, "Optional channel epe"):
            optimized._load_behavior_manifest(manifest)

    def test_ann_topk_graph_returns_neighbors(self):
        features = np.asarray([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0]], dtype=np.float32)
        neighbors, distances = optimized._ann_topk_graph(features, 2)

        self.assertEqual(neighbors.shape, (3, 2))
        self.assertEqual(distances.shape, (3, 2))
        self.assertTrue(np.all(neighbors[:, 0] == np.arange(3)))

    def test_weighted_facility_and_kcenter_selection(self):
        features = np.asarray([[1.0, 0.0], [0.95, 0.05], [-1.0, 0.0]], dtype=np.float32)
        neighbors, distances = optimized._ann_topk_graph(features, 2)
        tau = optimized._similarity_tau(distances)
        _, reverse = optimized._sparse_similarity(neighbors, distances, tau)
        selected, score, _ = optimized._weighted_facility_location(
            np.asarray([1.0, 1.0, 5.0], dtype=np.float64),
            reverse,
            coverage_target=0.90,
            min_gain=1e-6,
        )

        self.assertIn(2, selected)
        self.assertGreater(score, 0.0)

    def test_ssim_behavior_verification_and_weight_normalization(self):
        manifest, _, _ = self._manifest_and_features(["m0", "m1", "m2"], include_epe=True)
        samples = optimized._load_behavior_manifest(manifest)
        result = optimized._behavior_verification(
            samples["m0"],
            samples["m1"],
            channels=["aerial", "epe"],
            threshold=0.08,
        )

        self.assertTrue(result.passed)
        self.assertIn("aerial", result.channel_distances)
        self.assertIn("epe", result.channel_distances)
        self.assertAlmostEqual(sum(optimized._normalized_behavior_weights(["aerial", "epe"]).values()), 1.0)

    def test_full_marker_behavior_pipeline(self):
        oas_path = self.temp_root / "unit.oas"
        _make_oas(oas_path)
        marker_ids = ["unit__marker_000000", "unit__marker_000002"]
        manifest, feature_npz, _ = self._manifest_and_features(marker_ids)
        runner = optimized.OptimizedMainlineRunner(
            config={
                "marker_layer": "999/0",
                "clip_size_um": 1.0,
                "behavior_manifest": str(manifest),
                "feature_npz": str(feature_npz),
                "ann_top_k": 2,
                "coverage_target": 0.95,
                "facility_min_gain": 1e-6,
                "behavior_verification_threshold": 0.05,
                "high_risk_quantile": 0.90,
            },
            temp_dir=self.temp_root / "run",
        )

        result = runner.run(str(oas_path))

        self.assertEqual(result["pipeline_mode"], optimized.PIPELINE_MODE)
        self.assertEqual(result["marker_count"], 3)
        self.assertEqual(result["exact_cluster_count"], 2)
        self.assertIn("behavior_stats", result)
        self.assertGreaterEqual(result["selected_representative_count"], 1)
        payload = json.dumps(result, default=optimized._json_default)
        for legacy_word in ("HDBSCAN", "hdbscan", "ILP", "closed_loop", "fft", "auto_marker"):
            self.assertNotIn(legacy_word, payload)

    def test_review_diff_channel_output_is_optional(self):
        oas_path = self.temp_root / "unit.oas"
        _make_oas(oas_path)
        marker_ids = ["unit__marker_000000", "unit__marker_000002"]
        manifest, feature_npz, _ = self._manifest_and_features(marker_ids)
        runner = optimized.OptimizedMainlineRunner(
            config={
                "marker_layer": "999/0",
                "clip_size_um": 1.0,
                "behavior_manifest": str(manifest),
                "feature_npz": str(feature_npz),
                "ann_top_k": 2,
            },
            temp_dir=self.temp_root / "run_review",
        )
        result = runner.run(str(oas_path))
        review_dir = self.temp_root / "review"
        info = optimized._export_review(result, str(review_dir), diff_channels=("aerial",))

        self.assertTrue((review_dir / "representative_files.txt").exists())
        self.assertGreaterEqual(info["diff_file_count"], 1)

    def test_cli_parses_behavior_args_and_layer_ops(self):
        parser = optimized._build_parser()
        args = parser.parse_args(
            [
                "input.oas",
                "--marker-layer",
                "999/0",
                "--behavior-manifest",
                "behavior.jsonl",
                "--feature-npz",
                "features.npz",
                "--ann-top-k",
                "32",
                "--export-diff-channels",
                "aerial,pv",
                "--register-op",
                "1/0",
                "2/0",
                "subtract",
                "10/0",
            ]
        )

        self.assertEqual(args.ann_top_k, 32)
        self.assertEqual(optimized._parse_diff_channels(args.export_diff_channels), ("aerial", "pv"))
        self.assertTrue(bool(args.apply_layer_ops or args.register_op))

    def test_layer_ops_processor_writes_boolean_result_layer(self):
        processor = optimized._make_layer_processor([["1/0", "2/0", "intersect", "10/0"]])
        lib = gdstk.Library()
        cell = gdstk.Cell("TOP")
        cell.add(gdstk.rectangle((0.0, 0.0), (2.0, 2.0), layer=1, datatype=0))
        cell.add(gdstk.rectangle((1.0, 1.0), (3.0, 3.0), layer=2, datatype=0))
        lib.add(cell)

        processor.apply_layer_operations(lib)
        result_polygons = [
            poly
            for poly in cell.get_polygons(apply_repetitions=True, include_paths=True, depth=None)
            if (int(poly.layer), int(poly.datatype)) == (10, 0)
        ]
        payload = optimized._layer_operation_payload(processor)

        self.assertEqual(payload[0]["operation"], "intersect")
        self.assertGreaterEqual(len(result_polygons), 1)

    def test_start_banner_prints_chinese_layer_ops(self):
        parser = optimized._build_parser()
        args = parser.parse_args(
            [
                "input.oas",
                "--marker-layer",
                "999/0",
                "--behavior-manifest",
                "behavior.jsonl",
                "--feature-npz",
                "features.npz",
                "--register-op",
                "1/0",
                "2/0",
                "subtract",
                "10/0",
            ]
        )
        processor = optimized._make_layer_processor(args.register_op)
        output = StringIO()

        with redirect_stdout(output):
            optimized._print_start_banner(
                "测试启动",
                args,
                apply_layer_operations=True,
                layer_ops=optimized._layer_operation_payload(processor),
            )

        text = output.getvalue()
        self.assertIn("层操作启用: 是", text)
        self.assertIn("规则 1: 1/0 subtract 2/0 -> 10/0", text)


if __name__ == "__main__":
    unittest.main()
