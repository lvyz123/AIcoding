#!/usr/bin/env python3
"""Tests for no-training handcrafted feature extraction."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import unittest
from pathlib import Path

import gdstk
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import feature_extractor_handcraft as handcraft
import layout_clustering_optimized_notrain as notrain
from layout_utils import DEFAULT_PIXEL_SIZE_NM, _write_oas_library


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, image=np.asarray(image, dtype=np.float32))


def _make_aerial(kind: str) -> np.ndarray:
    image = np.zeros((32, 32), dtype=np.float32)
    if kind == "vertical":
        image[5:27, 13:19] = 1.0
    elif kind == "two":
        image[7:13, 7:25] = 1.0
        image[19:25, 7:25] = 1.0
    else:
        image[10:22, 10:22] = 1.0
    return image


def _make_oas(path: Path) -> None:
    lib = gdstk.Library()
    cell = gdstk.Cell("TOP")
    centers = [(0.0, 0.0), (3.0, 0.0)]
    for idx, (cx, cy) in enumerate(centers):
        cell.add(gdstk.rectangle((cx - 0.02, cy - 0.02), (cx + 0.02, cy + 0.02), layer=999, datatype=0))
        if idx == 0:
            cell.add(gdstk.rectangle((cx - 0.25, cy - 0.06), (cx - 0.05, cy + 0.06), layer=1, datatype=0))
            cell.add(gdstk.rectangle((cx + 0.05, cy - 0.06), (cx + 0.25, cy + 0.06), layer=1, datatype=0))
        else:
            cell.add(gdstk.rectangle((cx - 0.06, cy - 0.25), (cx + 0.06, cy - 0.05), layer=1, datatype=0))
            cell.add(gdstk.rectangle((cx - 0.06, cy + 0.05), (cx + 0.06, cy + 0.25), layer=1, datatype=0))
    lib.add(cell)
    _write_oas_library(lib, str(path))


def _manifest(path: Path, marker_ids, *, same_aerial: bool = False, include_pv: bool = False) -> Path:
    manifest = path / "behavior.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for idx, marker_id in enumerate(marker_ids):
            aerial = _make_aerial("square" if same_aerial else ("square" if idx == 0 else "vertical"))
            aerial_path = path / f"{marker_id}_aerial.npz"
            _write_image(aerial_path, aerial)
            row = {
                "sample_id": marker_id,
                "source_path": "unit.oas",
                "marker_id": marker_id,
                "clip_bbox": [-0.5, -0.5, 0.5, 0.5],
                "aerial_npz": str(aerial_path),
            }
            if include_pv:
                pv_path = path / f"{marker_id}_pv.npz"
                _write_image(pv_path, aerial * (idx + 1))
                row["pv_npz"] = str(pv_path)
            handle.write(json.dumps(row) + "\n")
    return manifest


class HandcraftedFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = SCRIPT_DIR / "test_outputs" / "_handcraft_unit"
        shutil.rmtree(self.temp_root, ignore_errors=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_aerial_features_change_with_image(self):
        a = handcraft._aerial_feature_block(_make_aerial("square"))
        b = handcraft._aerial_feature_block(_make_aerial("vertical"))

        self.assertEqual(a.shape, b.shape)
        self.assertGreater(float(np.linalg.norm(a - b)), 0.1)

    def test_encode_outputs_l2_normalized_features_and_metadata(self):
        oas_path = self.temp_root / "unit.oas"
        _make_oas(oas_path)
        marker_ids = ["unit__marker_000000", "unit__marker_000001"]
        manifest = _manifest(self.temp_root, marker_ids, include_pv=True)
        features_out = self.temp_root / "features.npz"
        metadata_out = self.temp_root / "features.meta.json"

        metadata = handcraft.encode_handcrafted_features(
            input_path=str(oas_path),
            marker_layer="999/0",
            behavior_manifest=str(manifest),
            features_out=features_out,
            metadata_out=metadata_out,
        )

        with np.load(features_out, allow_pickle=False) as data:
            self.assertEqual(data["sample_ids"].tolist(), marker_ids)
            features = np.asarray(data["features"], dtype=np.float32)
        self.assertTrue(np.all(np.isfinite(features)))
        self.assertTrue(np.allclose(np.linalg.norm(features, axis=1), 1.0, atol=1e-5))
        block_names = [block["name"] for block in metadata["block_metadata"]]
        self.assertIn("optional_behavior_stats", block_names)
        self.assertIn("layout_wl_graph", block_names)
        self.assertTrue(metadata_out.exists())

    def test_wl_and_layout_features_different_for_topology_change(self):
        oas_path = self.temp_root / "unit.oas"
        _make_oas(oas_path)
        marker_ids = ["unit__marker_000000", "unit__marker_000001"]
        manifest = _manifest(self.temp_root, marker_ids, same_aerial=True)
        features_out = self.temp_root / "features_same_aerial.npz"

        handcraft.encode_handcrafted_features(
            input_path=str(oas_path),
            marker_layer="999/0",
            behavior_manifest=str(manifest),
            features_out=features_out,
        )

        with np.load(features_out, allow_pickle=False) as data:
            features = np.asarray(data["features"], dtype=np.float32)
        self.assertGreater(float(np.linalg.norm(features[0] - features[1])), 0.01)

    def test_notrain_pipeline_runs_without_feature_npz_argument(self):
        oas_path = self.temp_root / "unit.oas"
        _make_oas(oas_path)
        marker_ids = ["unit__marker_000000", "unit__marker_000001"]
        manifest = _manifest(self.temp_root, marker_ids)
        output = self.temp_root / "notrain.json"
        args = argparse.Namespace(
            input_path=str(oas_path),
            output=str(output),
            format="json",
            marker_layer="999/0",
            clip_size=1.0,
            behavior_manifest=str(manifest),
            ann_top_k=2,
            coverage_target=0.95,
            facility_min_gain=1e-6,
            behavior_verification_threshold=0.08,
            high_risk_quantile=0.90,
            export_diff_channels="",
            review_dir=None,
            export_cluster_review_dir=None,
            apply_layer_ops=False,
            register_op=None,
        )

        result = notrain.run_notrain(args)

        self.assertEqual(result["pipeline_mode"], notrain.PIPELINE_MODE)
        self.assertEqual(result["feature_source"], "handcraft")
        self.assertIn("feature_metadata", result)
        self.assertEqual(result["marker_count"], 2)
        self.assertTrue(Path(result["handcraft_feature_npz"]).exists())

    def test_notrain_parser_has_no_feature_npz(self):
        help_text = notrain._build_parser().format_help()

        self.assertNotIn("--feature-npz", help_text)
        self.assertIn("--behavior-manifest", help_text)
        self.assertEqual(DEFAULT_PIXEL_SIZE_NM, 10)

    def test_notrain_source_is_decoupled_from_optimized_entrypoint(self):
        source = Path(notrain.__file__).read_text(encoding="utf-8")

        self.assertNotIn("from layout_clustering_optimized import", source)
        self.assertNotIn("OptimizedMainlineRunner", source)
        self.assertIn("NotrainOptimizedRunner", source)


if __name__ == "__main__":
    unittest.main()
