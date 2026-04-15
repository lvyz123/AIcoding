#!/usr/bin/env python3
"""Regression tests for HDBSCAN window geometry helpers."""

from __future__ import annotations

import sys
import shutil
import unittest
from pathlib import Path

import gdstk
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import layout_clustering_hdbscan as hdbscan_mod


def _points_from(polygons):
    return np.vstack([np.asarray(poly.points, dtype=np.float64) for poly in polygons])


class HDBSCANWindowGeometryRegressionTests(unittest.TestCase):
    def test_parse_marker_layer_spec(self):
        self.assertEqual(hdbscan_mod._parse_layer_spec("999/0"), (999, 0))
        self.assertEqual(hdbscan_mod._parse_layer_spec(" 12 / 34 "), (12, 34))
        with self.assertRaises(ValueError):
            hdbscan_mod._parse_layer_spec("999")
        with self.assertRaises(ValueError):
            hdbscan_mod._parse_layer_spec("abc/0")

    def test_marker_layer_is_excluded_from_pattern_index(self):
        lib = gdstk.Library()
        cell = gdstk.Cell("TOP")
        cell.add(gdstk.rectangle((0.0, 0.0), (1.0, 1.0), layer=1, datatype=0))
        cell.add(gdstk.rectangle((0.45, 0.45), (0.55, 0.55), layer=999, datatype=0))
        lib.add(cell)

        spatial_index, indexed_elements, layout_bbox, marker_polygons = hdbscan_mod._build_layout_spatial_index(
            lib,
            marker_layer=(999, 0),
        )

        self.assertIsNotNone(spatial_index)
        self.assertEqual(len(indexed_elements), 1)
        self.assertEqual(len(marker_polygons), 1)
        self.assertEqual(hdbscan_mod._element_layer_datatype(indexed_elements[0]["element"]), (1, 0))
        self.assertEqual(hdbscan_mod._element_layer_datatype(marker_polygons[0]), (999, 0))
        self.assertEqual(layout_bbox, (0.0, 0.0, 1.0, 1.0))

    def test_marker_candidate_uses_every_marker_bbox_center(self):
        lib = gdstk.Library()
        cell = gdstk.Cell("TOP")
        cell.add(gdstk.rectangle((0.4, 0.4), (0.6, 0.6), layer=1, datatype=0))
        cell.add(gdstk.rectangle((0.45, 0.45), (0.55, 0.55), layer=999, datatype=0))
        cell.add(gdstk.rectangle((5.0, 5.0), (5.1, 5.1), layer=999, datatype=0))
        lib.add(cell)

        spatial_index, indexed_elements, _, marker_polygons = hdbscan_mod._build_layout_spatial_index(
            lib,
            marker_layer=(999, 0),
        )
        candidates, meta = hdbscan_mod._select_marker_candidate_centers(
            marker_polygons,
            spatial_index,
            indexed_elements,
            window_size_um=0.5,
            marker_layer=(999, 0),
            enable_clip_shifting=False,
        )

        self.assertEqual(len(candidates), 2)
        self.assertEqual(meta["marker_count"], 2)
        self.assertEqual(meta["marker_candidate_count"], 2)
        self.assertEqual(meta["marker_skipped_invalid_count"], 0)
        self.assertEqual(candidates[0]["seed_kind"], "marker")
        self.assertEqual(candidates[0]["marker_layer"], "999/0")
        self.assertEqual(candidates[0]["seed_center"], (0.5, 0.5))
        self.assertEqual(candidates[0]["center"], (0.5, 0.5))
        self.assertAlmostEqual(candidates[1]["seed_center"][0], 5.05)
        self.assertAlmostEqual(candidates[1]["seed_center"][1], 5.05)
        self.assertAlmostEqual(candidates[1]["center"][0], 5.05)
        self.assertAlmostEqual(candidates[1]["center"][1], 5.05)

    def test_window_xor_ratio_matches_translated_local_patterns(self):
        polygons_a = [gdstk.rectangle((10.0, 10.0), (11.0, 11.0), layer=1, datatype=0)]
        polygons_b = [gdstk.rectangle((20.0, 20.0), (21.0, 21.0), layer=1, datatype=0)]

        record_a = {
            "normalized_polygons": hdbscan_mod._normalize_polygons_to_local_bbox(
                polygons_a,
                (10.0, 10.0, 12.0, 12.0),
            ),
            "window_area": 4.0,
        }
        record_b = {
            "normalized_polygons": hdbscan_mod._normalize_polygons_to_local_bbox(
                polygons_b,
                (20.0, 20.0, 22.0, 22.0),
            ),
            "window_area": 4.0,
        }

        self.assertAlmostEqual(hdbscan_mod._window_xor_ratio(record_a, record_b), 0.0)

    def test_shift_candidate_normalizes_against_shifted_outer_bbox(self):
        lib = gdstk.Library()
        cell = gdstk.Cell("TOP")
        cell.add(gdstk.rectangle((0.8, 0.8), (1.2, 1.2), layer=1, datatype=0))
        lib.add(cell)

        spatial_index, indexed_elements, _ = hdbscan_mod._build_layout_spatial_index(lib)
        origin_record = hdbscan_mod._build_candidate_window_record(
            {
                "elem_id": 0,
                "center": (1.0, 1.0),
                "seed_center": (1.0, 1.0),
                "center_shift": (0.0, 0.0),
                "seed_kind": "element",
            },
            spatial_index,
            indexed_elements,
            source_name="unit",
            source_window_id="origin",
            window_size_um=1.0,
            context_width_um=0.5,
            max_elements_per_window=16,
            quant_step_um=0.001,
            signature_bins=4,
        )
        self.assertIsNotNone(origin_record)

        shifted_outer_bbox = hdbscan_mod._make_centered_bbox((1.2, 1.0), 2.0, 2.0)
        local_element_ids = list(spatial_index.intersection(shifted_outer_bbox))
        shifted_record = hdbscan_mod._build_shift_candidate_record(
            0,
            origin_record,
            (1.2, 1.0),
            indexed_elements,
            local_element_ids,
            window_size_um=1.0,
            context_width_um=0.5,
            max_elements_per_window=16,
            quant_step_um=0.001,
            signature_bins=4,
        )
        self.assertIsNotNone(shifted_record)

        shifted_abs_points = _points_from(shifted_record["outer_polygons"])
        shifted_norm_points = _points_from(shifted_record["normalized_polygons"])
        outer_bbox = shifted_record["outer_bbox"]

        self.assertAlmostEqual(
            float(np.min(shifted_norm_points[:, 0])),
            float(np.min(shifted_abs_points[:, 0]) - outer_bbox[0]),
        )
        self.assertAlmostEqual(
            float(np.min(shifted_norm_points[:, 1])),
            float(np.min(shifted_abs_points[:, 1]) - outer_bbox[1]),
        )

        origin_norm_points = _points_from(origin_record["normalized_polygons"])
        self.assertNotAlmostEqual(
            float(np.min(origin_norm_points[:, 0])),
            float(np.min(shifted_norm_points[:, 0])),
        )

    def test_shift_candidate_recuts_fixed_layout_from_local_element_pool(self):
        lib = gdstk.Library()
        cell = gdstk.Cell("TOP")
        cell.add(gdstk.rectangle((0.8, 0.8), (1.2, 1.2), layer=1, datatype=0))
        cell.add(gdstk.rectangle((2.05, 0.9), (2.15, 1.1), layer=1, datatype=0))
        lib.add(cell)

        spatial_index, indexed_elements, _ = hdbscan_mod._build_layout_spatial_index(lib)
        origin_record = hdbscan_mod._build_candidate_window_record(
            {
                "elem_id": 0,
                "center": (1.0, 1.0),
                "seed_center": (1.0, 1.0),
                "center_shift": (0.0, 0.0),
                "seed_kind": "element",
            },
            spatial_index,
            indexed_elements,
            source_name="unit",
            source_window_id="origin",
            window_size_um=1.0,
            context_width_um=0.5,
            max_elements_per_window=16,
            quant_step_um=0.001,
            signature_bins=4,
        )
        self.assertIsNotNone(origin_record)

        shifted_outer_bbox = hdbscan_mod._make_centered_bbox((1.3, 1.0), 2.0, 2.0)
        local_element_ids = list(spatial_index.intersection(shifted_outer_bbox))
        shifted_record = hdbscan_mod._build_shift_candidate_record(
            0,
            origin_record,
            (1.3, 1.0),
            indexed_elements,
            local_element_ids,
            window_size_um=1.0,
            context_width_um=0.5,
            max_elements_per_window=16,
            quant_step_um=0.001,
            signature_bins=4,
        )
        self.assertIsNotNone(shifted_record)

        origin_abs_points = _points_from(origin_record["outer_polygons"])
        shifted_abs_points = _points_from(shifted_record["outer_polygons"])

        self.assertLess(float(np.max(origin_abs_points[:, 0])), 2.0)
        self.assertGreater(float(np.max(shifted_abs_points[:, 0])), 2.0)
        self.assertGreater(len(shifted_record["outer_polygons"]), len(origin_record["outer_polygons"]))

    def test_shift_cover_merges_clip_position_redundancy(self):
        lib = gdstk.Library()
        cell = gdstk.Cell("TOP")
        cell.add(gdstk.rectangle((0.8, 0.8), (1.2, 1.2), layer=1, datatype=0))
        cell.add(gdstk.rectangle((2.05, 0.9), (2.15, 1.1), layer=1, datatype=0))
        lib.add(cell)

        spatial_index, indexed_elements, _ = hdbscan_mod._build_layout_spatial_index(lib)
        common_kwargs = {
            "spatial_index": spatial_index,
            "indexed_elements": indexed_elements,
            "source_name": "unit",
            "window_size_um": 1.0,
            "context_width_um": 0.5,
            "max_elements_per_window": 16,
            "quant_step_um": 0.001,
            "signature_bins": 4,
        }
        origin_record = hdbscan_mod._build_candidate_window_record(
            {
                "elem_id": 0,
                "center": (1.0, 1.0),
                "seed_center": (1.0, 1.0),
                "center_shift": (0.0, 0.0),
                "seed_kind": "element",
            },
            source_window_id="origin",
            **common_kwargs,
        )
        target_record = hdbscan_mod._build_candidate_window_record(
            {
                "elem_id": 0,
                "center": (1.3, 1.0),
                "seed_center": (1.0, 1.0),
                "center_shift": (0.3, 0.0),
                "seed_kind": "element",
            },
            source_window_id="target",
            **common_kwargs,
        )
        self.assertIsNotNone(origin_record)
        self.assertIsNotNone(target_record)

        merged_records = hdbscan_mod._compress_window_records_with_shift_cover(
            [origin_record, target_record],
            spatial_index,
            indexed_elements,
            window_size_um=1.0,
            context_width_um=0.5,
            max_elements_per_window=16,
            quant_step_um=0.001,
            signature_bins=4,
            similarity_threshold=0.96,
            shift_directions=("right",),
            neighbor_limit=16,
        )

        self.assertEqual(len(merged_records), 1)
        merged_sample = merged_records[0]["sample"]
        self.assertEqual(merged_sample.duplicate_count, 2)
        self.assertEqual(set(merged_records[0]["covered_window_ids"]), {"origin", "target"})
        self.assertTrue(merged_records[0]["generated_by_pattern_shift"])

    def test_shift_candidate_matching_uses_signature_gate_before_xor(self):
        candidate = {
            "pattern_hash": "candidate",
            "invariants": np.ones(9, dtype=np.float64),
            "signature": np.zeros(16, dtype=np.float32),
            "generated_by_shift": True,
            "normalized_polygons": [gdstk.rectangle((0.0, 0.0), (1.0, 1.0))],
            "window_area": 1.0,
        }
        target = {
            "pattern_hash": "target",
            "invariants": np.ones(9, dtype=np.float64),
            "signature": np.ones(16, dtype=np.float32),
            "normalized_polygons": [gdstk.rectangle((0.0, 0.0), (1.0, 1.0))],
            "window_area": 1.0,
        }

        self.assertFalse(
            hdbscan_mod._candidate_matches_window_record(
                candidate,
                target,
                signature_floor=0.5,
                area_match_ratio=0.92,
                invariant_dist_limit=0.12,
            )
        )

    def test_zero_signature_vectors_are_not_similarity_evidence(self):
        self.assertEqual(
            hdbscan_mod._cosine_similarity_1d(np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32)),
            0.0,
        )
        self.assertAlmostEqual(
            hdbscan_mod._cosine_similarity_1d(np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32)),
            1.0,
        )

    def test_export_cluster_review_remaps_result_paths(self):
        temp_root = SCRIPT_DIR / "test_outputs" / "_review_remap_regression"
        shutil.rmtree(temp_root, ignore_errors=True)
        try:
            src_dir = temp_root / "src"
            src_dir.mkdir(parents=True, exist_ok=True)
            src_a = src_dir / "window_a.oas"
            src_b = src_dir / "window_b.oas"
            src_a.write_bytes(b"window-a")
            src_b.write_bytes(b"window-b")

            pipeline = hdbscan_mod.LayoutClusteringPipeline({"apply_layer_operations": False})
            pipeline.filepaths = [str(src_a), str(src_b)]
            pipeline.sample_infos = [None, None]
            pipeline.features = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
            pipeline.clusters = [[0, 1]]
            pipeline.representatives = [1]
            pipeline.active_feature_names = ["f0", "f1"]

            review_dir = temp_root / "review"
            review_info = pipeline.export_cluster_review(str(review_dir))
            result = pipeline.get_results()

            self.assertEqual(review_info["path_remap_count"], 2)
            self.assertEqual(result["cluster_review"]["path_remap_count"], 2)
            self.assertTrue(all(str(review_dir.resolve()) in path for path in result["file_list"]))

            cluster = result["clusters"][0]
            result_paths = list(cluster["sample_files"]) + [cluster["representative_file"]]
            self.assertTrue(all(Path(path).exists() for path in result_paths))
            self.assertTrue(Path(cluster["representative_file"]).name.startswith("REP__"))

            rep_txt = Path(result["cluster_review"]["representative_files_txt"])
            rep_lines = rep_txt.read_text(encoding="utf-8").splitlines()
            self.assertEqual(rep_lines, [cluster["representative_file"]])
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
