#!/usr/bin/env python3
"""Unit tests for the raster-first marker-driven mainline pipeline."""

from __future__ import annotations

import shutil
import sys
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

import gdstk
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from layout_clustering_clip_shifting import LayoutClusteringPipeline
from layout_utils import _write_oas_library
from mainline import (
    CandidateClip,
    MainlineRunner,
    _bitmap_acc_match,
    _bitmap_ecc_match,
    _canonical_bitmap_hash,
    _collect_boundary_positions,
    _collect_shift_values_px,
)


TEST_OUTPUT_ROOT = SCRIPT_DIR / "test_outputs"
HOTSPOT_LAYER = 999
HOTSPOT_DATATYPE = 0
PATTERN_LAYER = 1
PATTERN_DATATYPE = 0
PIXEL_SIZE_NM = 10
PIXEL_SIZE_UM = PIXEL_SIZE_NM / 1000.0


def _add_pattern(cell: gdstk.Cell, origin_x: float, origin_y: float) -> None:
    cell.add(
        gdstk.rectangle(
            (origin_x - 0.35, origin_y - 0.18),
            (origin_x + 0.35, origin_y + 0.18),
            layer=PATTERN_LAYER,
            datatype=PATTERN_DATATYPE,
        )
    )
    cell.add(
        gdstk.rectangle(
            (origin_x - 0.08, origin_y - 0.42),
            (origin_x + 0.08, origin_y + 0.42),
            layer=PATTERN_LAYER,
            datatype=PATTERN_DATATYPE,
        )
    )


def _add_marker(cell: gdstk.Cell, origin_x: float, origin_y: float) -> None:
    cell.add(
        gdstk.rectangle(
            (origin_x - 0.05, origin_y - 0.05),
            (origin_x + 0.05, origin_y + 0.05),
            layer=HOTSPOT_LAYER,
            datatype=HOTSPOT_DATATYPE,
        )
    )


def _write_simple_layout(filepath: Path, *, hierarchical: bool = False) -> None:
    lib = gdstk.Library()
    top = gdstk.Cell("TOP")
    lib.add(top)

    if hierarchical:
        leaf = gdstk.Cell("LEAF")
        _add_pattern(leaf, 0.0, 0.0)
        lib.add(leaf)
        top.add(gdstk.Reference(leaf, (0.0, 0.0)))
        top.add(gdstk.Reference(leaf, (4.0, 0.0)))
    else:
        _add_pattern(top, 0.0, 0.0)
        _add_pattern(top, 4.0, 0.0)

    _add_marker(top, 0.0, 0.0)
    _add_marker(top, 4.0, 0.0)
    _write_oas_library(lib, str(filepath))


def _make_config(**overrides):
    config = {
        "clip_size_um": 1.6,
        "hotspot_layer": f"{HOTSPOT_LAYER}/{HOTSPOT_DATATYPE}",
        "matching_mode": "ecc",
        "solver": "auto",
        "geometry_mode": "exact",
        "pixel_size_nm": PIXEL_SIZE_NM,
        "area_match_ratio": 0.96,
        "edge_tolerance_um": 0.02,
        "max_elements_per_window": 64,
        "clip_shift_directions": "left,right,up,down",
        "clip_shift_boundary_tolerance_um": 0.02,
        "apply_layer_operations": False,
    }
    config.update(overrides)
    return config


def _make_runner(**overrides) -> MainlineRunner:
    return MainlineRunner(config=_make_config(**overrides), temp_dir=TEST_OUTPUT_ROOT / "runner_tmp")


def _make_candidate(candidate_id: str, origin_id: int, bitmap: np.ndarray, *, shift_direction: str = "base") -> CandidateClip:
    bitmap = np.ascontiguousarray(bitmap, dtype=bool)
    clip_hash, _ = _canonical_bitmap_hash(bitmap)
    height, width = bitmap.shape
    return CandidateClip(
        candidate_id=candidate_id,
        origin_exact_cluster_id=int(origin_id),
        center=(0.0, 0.0),
        clip_bbox=(0.0, 0.0, float(width) * PIXEL_SIZE_UM, float(height) * PIXEL_SIZE_UM),
        clip_bbox_q=(0, 0, int(width), int(height)),
        clip_bitmap=bitmap,
        clip_hash=clip_hash,
        shift_direction=shift_direction,
        shift_distance_um=0.0 if shift_direction == "base" else PIXEL_SIZE_UM,
        coverage={int(origin_id)},
        source_marker_id=f"marker_{int(origin_id)}",
    )


class MainlinePipelineTests(unittest.TestCase):
    @contextmanager
    def _temp_dir(self):
        TEST_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        temp_dir = TEST_OUTPUT_ROOT / f"mainline_test_{uuid.uuid4().hex[:8]}"
        temp_dir.mkdir(parents=True, exist_ok=False)
        try:
            yield str(temp_dir)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _run_pipeline(self, layout_path: Path, **config_overrides):
        pipeline = LayoutClusteringPipeline(_make_config(**config_overrides))
        try:
            return pipeline.run_pipeline(str(layout_path))
        finally:
            pipeline.cleanup()

    def test_mainline_requires_hotspot_layer(self):
        with self._temp_dir() as temp_dir:
            layout_path = Path(temp_dir) / "simple.oas"
            _write_simple_layout(layout_path)
            pipeline = LayoutClusteringPipeline(_make_config(hotspot_layer=None))
            try:
                with self.assertRaisesRegex(ValueError, "hotspot-layer"):
                    pipeline.run_pipeline(str(layout_path))
            finally:
                pipeline.cleanup()

    def test_mainline_merges_identical_expanded_clips(self):
        with self._temp_dir() as temp_dir:
            layout_path = Path(temp_dir) / "simple.oas"
            _write_simple_layout(layout_path)
            result = self._run_pipeline(layout_path)
            self.assertEqual(result["pipeline_mode"], "mainline")
            self.assertEqual(result["total_clusters"], 1)
            self.assertEqual(result["result_summary"]["exact_cluster_count"], 1)
            self.assertEqual(result["cluster_sizes"], [2])

    def test_mainline_outputs_required_fields(self):
        with self._temp_dir() as temp_dir:
            layout_path = Path(temp_dir) / "simple.oas"
            _write_simple_layout(layout_path)
            result = self._run_pipeline(layout_path)
            cluster = result["clusters"][0]
            for key in (
                "pipeline_mode",
                "marker_id",
                "exact_cluster_id",
                "matching_mode",
                "selected_candidate_id",
                "selected_shift_direction",
                "selected_shift_distance_um",
                "solver_used",
            ):
                self.assertIn(key, cluster)
            self.assertEqual(result["pixel_size_nm"], PIXEL_SIZE_NM)

    def test_mainline_fast_mode_runs(self):
        with self._temp_dir() as temp_dir:
            layout_path = Path(temp_dir) / "simple.oas"
            _write_simple_layout(layout_path)
            result = self._run_pipeline(layout_path, geometry_mode="fast", max_elements_per_window=8)
            self.assertEqual(result["geometry_mode"], "fast")
            self.assertEqual(result["total_clusters"], 1)

    def test_mainline_flattens_hierarchy(self):
        with self._temp_dir() as temp_dir:
            layout_path = Path(temp_dir) / "hierarchical.oas"
            _write_simple_layout(layout_path, hierarchical=True)
            result = self._run_pipeline(layout_path)
            self.assertEqual(result["total_clusters"], 1)
            self.assertEqual(result["cluster_sizes"], [2])

    def test_mainline_supports_chinese_paths(self):
        with self._temp_dir() as temp_dir:
            root = Path(temp_dir) / "中文路径验证"
            layout_path = root / "输入目录_版图" / "样例版图_中文文件名.oas"
            result_path = root / "输出目录_结果" / "聚类结果_中文.json"
            review_dir = root / "输出目录_结果" / "聚类review_中文"
            _write_simple_layout(layout_path)

            pipeline = LayoutClusteringPipeline(_make_config())
            try:
                result = pipeline.run_pipeline(str(layout_path))
                pipeline.save_results(str(result_path), "json")
                review = pipeline.export_cluster_review(str(review_dir))
            finally:
                pipeline.cleanup()

            self.assertTrue(layout_path.exists())
            self.assertTrue(result_path.exists())
            self.assertTrue(review_dir.exists())
            self.assertEqual(result["total_clusters"], 1)
            self.assertEqual(result["cluster_sizes"], [2])
            self.assertEqual(review["missing_file_count"], 0)


class RasterHelperTests(unittest.TestCase):
    def test_raster_hash_is_stable_for_same_shape(self):
        rect = np.zeros((8, 8), dtype=bool)
        rect[2:6, 1:7] = True
        rotated = rect.copy()
        hash_a, _ = _canonical_bitmap_hash(rect)
        hash_b, _ = _canonical_bitmap_hash(rotated)
        self.assertEqual(hash_a, hash_b)

    def test_raster_hash_respects_dihedral_symmetry(self):
        l_shape = np.zeros((8, 8), dtype=bool)
        l_shape[1:7, 1:3] = True
        l_shape[5:7, 1:7] = True
        mirrored = np.fliplr(l_shape)
        hash_a, _ = _canonical_bitmap_hash(l_shape)
        hash_b, _ = _canonical_bitmap_hash(mirrored)
        self.assertEqual(hash_a, hash_b)

    def test_shift_candidates_are_single_axis(self):
        expanded = np.zeros((10, 14), dtype=bool)
        expanded[2:8, 3:5] = True
        expanded[2:8, 9:11] = True
        boundaries = _collect_boundary_positions(np.any(expanded, axis=0))
        shifts = _collect_shift_values_px(boundaries, 4, 8, (-3, 3), 1, max_count=8)
        self.assertTrue(all(isinstance(value, int) for value in shifts))
        self.assertIn(0, shifts)
        self.assertTrue(all(-3 <= value <= 3 for value in shifts))

    def test_acc_bitmap_matches_small_geometry_equivalent_case(self):
        bitmap_a = np.zeros((12, 12), dtype=bool)
        bitmap_b = np.zeros((12, 12), dtype=bool)
        bitmap_a[2:10, 2:10] = True
        bitmap_b[2:10, 2:9] = True
        self.assertTrue(_bitmap_acc_match(bitmap_a, bitmap_a, 0.99))
        self.assertFalse(_bitmap_acc_match(bitmap_a, bitmap_b, 0.99))

    def test_ecc_bitmap_matches_small_geometry_equivalent_case(self):
        bitmap_a = np.zeros((12, 12), dtype=bool)
        bitmap_b = np.zeros((12, 12), dtype=bool)
        bitmap_a[2:10, 2:10] = True
        bitmap_b[2:10, 3:11] = True
        self.assertTrue(_bitmap_ecc_match(bitmap_a, bitmap_b, 0.02, PIXEL_SIZE_UM))

    def test_expanded_bitmap_slice_matches_manual_shift_slice(self):
        expanded = np.zeros((10, 12), dtype=bool)
        expanded[2:8, 3:7] = True
        base_slice = np.ascontiguousarray(expanded[2:8, 3:7], dtype=bool)
        shifted_slice = np.ascontiguousarray(expanded[2:8, 4:8], dtype=bool)
        manual = np.zeros_like(base_slice)
        manual[:, :-1] = base_slice[:, 1:]
        self.assertTrue(np.array_equal(shifted_slice[:, :-1], manual[:, :-1]))


class CandidateCoverageTests(unittest.TestCase):
    def test_pairwise_shifted_candidates_create_cross_coverage(self):
        base_a = np.zeros((10, 10), dtype=bool)
        base_a[2:6, 1:5] = True
        base_b = np.zeros((10, 10), dtype=bool)
        base_b[1:4, 5:8] = True
        base_b[6:8, 6:9] = True
        shifted_match = np.zeros((10, 10), dtype=bool)
        shifted_match[2:6, 3:7] = True

        candidates = [
            _make_candidate("c1_base", 1, base_a),
            _make_candidate("c1_shift", 1, shifted_match, shift_direction="right"),
            _make_candidate("c2_base", 2, base_b),
            _make_candidate("c2_shift", 2, shifted_match, shift_direction="left"),
        ]

        runner = _make_runner(matching_mode="acc", area_match_ratio=1.0)
        runner._evaluate_candidate_coverage_raster(candidates, [])

        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        self.assertEqual(by_id["c1_base"].coverage, {1})
        self.assertEqual(by_id["c2_base"].coverage, {2})
        self.assertEqual(by_id["c1_shift"].coverage, {1, 2})
        self.assertEqual(by_id["c2_shift"].coverage, {1, 2})

    def test_pairwise_candidate_coverage_uses_acc_for_non_identical_groups(self):
        bitmap_a = np.zeros((10, 10), dtype=bool)
        bitmap_b = np.zeros((10, 10), dtype=bool)
        bitmap_a[2:8, 2:8] = True
        bitmap_b[2:8, 2:8] = True
        bitmap_b[7, 7] = False

        candidates = [
            _make_candidate("c1", 1, bitmap_a),
            _make_candidate("c2", 2, bitmap_b),
        ]

        runner = _make_runner(matching_mode="acc", area_match_ratio=0.98)
        runner._evaluate_candidate_coverage_raster(candidates, [])

        self.assertEqual(candidates[0].coverage, {1, 2})
        self.assertEqual(candidates[1].coverage, {1, 2})

    def test_pairwise_candidate_coverage_uses_ecc_for_non_identical_groups(self):
        bitmap_a = np.zeros((12, 12), dtype=bool)
        bitmap_b = np.zeros((12, 12), dtype=bool)
        bitmap_a[2:10, 2:8] = True
        bitmap_b[2:10, 3:9] = True

        candidates = [
            _make_candidate("c1", 1, bitmap_a),
            _make_candidate("c2", 2, bitmap_b),
        ]

        runner = _make_runner(matching_mode="ecc", edge_tolerance_um=PIXEL_SIZE_UM)
        runner._evaluate_candidate_coverage_raster(candidates, [])

        self.assertEqual(candidates[0].coverage, {1, 2})
        self.assertEqual(candidates[1].coverage, {1, 2})


def main() -> int:
    suite = unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(MainlinePipelineTests))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(RasterHelperTests))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(CandidateCoverageTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
