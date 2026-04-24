#!/usr/bin/env python3
"""Tests for the uniform-grid optimized v1 clustering pipeline."""

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
REPO_ROOT = SCRIPT_DIR.parent
SAMPLE_LAYOUT = REPO_ROOT / "layoutgenerator" / "out_oas" / "sample_layout_001.oas"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import layout_clustering_optimized_v1 as optimized
import mainline
from layer_operations import LayerOperationProcessor
from layout_utils import _write_oas_library
from mainline import CandidateClip, ExactCluster, MainlineRunner, MarkerRecord, _canonical_bitmap_hash, _query_candidate_ids


def _record(seed_id: str, bitmap: np.ndarray, *, seed_weight: int) -> MarkerRecord:
    """构造一个最小可用的 marker record，用于 coverage / cluster 单元测试。"""

    clip_hash, _ = _canonical_bitmap_hash(bitmap)
    height, width = bitmap.shape
    return MarkerRecord(
        marker_id=seed_id,
        source_path="unit.oas",
        source_name="unit.oas",
        marker_bbox=(0.0, 0.0, 0.05, 0.05),
        marker_center=(0.025, 0.025),
        clip_bbox=(0.0, 0.0, float(width), float(height)),
        expanded_bbox=(0.0, 0.0, float(width), float(height)),
        clip_bbox_q=(0, 0, int(width), int(height)),
        expanded_bbox_q=(0, 0, int(width), int(height)),
        marker_bbox_q=(0, 0, 1, 1),
        shift_limits_px={"x": (0, 0), "y": (0, 0)},
        clip_bitmap=np.ascontiguousarray(bitmap, dtype=bool),
        expanded_bitmap=np.ascontiguousarray(bitmap, dtype=bool),
        clip_hash=clip_hash,
        expanded_hash=clip_hash,
        clip_area=float(np.count_nonzero(bitmap)),
        seed_weight=int(seed_weight),
    )


def _candidate(
    candidate_id: str,
    bitmap: np.ndarray,
    *,
    origin_exact_cluster_id: int,
    shift_direction: str,
    coverage: set[int] | None = None,
) -> CandidateClip:
    """构造一个最小可用的 candidate，用于 chunked coverage 单元测试。"""

    clip_hash, _ = _canonical_bitmap_hash(bitmap)
    height, width = bitmap.shape
    return CandidateClip(
        candidate_id=candidate_id,
        origin_exact_cluster_id=int(origin_exact_cluster_id),
        center=(float(width) * 0.5, float(height) * 0.5),
        clip_bbox=(0.0, 0.0, float(width), float(height)),
        clip_bbox_q=(0, 0, int(width), int(height)),
        clip_bitmap=np.ascontiguousarray(bitmap, dtype=bool),
        clip_hash=clip_hash,
        shift_direction=shift_direction,
        shift_distance_um=0.0 if shift_direction == "base" else 0.02,
        coverage=set(coverage) if coverage is not None else ({int(origin_exact_cluster_id)} if shift_direction == "base" else set()),
        source_marker_id=f"seed_{origin_exact_cluster_id}",
    )


def _write_oas(path: Path, polygons: list[gdstk.Polygon]) -> None:
    """把给定 polygon 列表写成最小 OAS fixture。"""

    lib = gdstk.Library()
    cell = gdstk.Cell("TOP")
    for poly in polygons:
        cell.add(poly)
    lib.add(cell)
    _write_oas_library(lib, str(path))


def _write_repeated_tile_oas(path: Path) -> None:
    """生成一份重复 tile 小样本，方便测试 coarse bucketing 的 weight 累计。"""

    lib = gdstk.Library()
    cell = gdstk.Cell("TOP")
    repeated = [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (6.0, 0.0)]
    for cx, cy in repeated:
        cell.add(gdstk.rectangle((cx + 0.05, cy + 0.05), (cx + 0.20, cy + 0.22), layer=1, datatype=0))
        cell.add(gdstk.rectangle((cx + 0.30, cy + 0.02), (cx + 0.42, cy + 0.18), layer=1, datatype=0))
    lib.add(cell)
    _write_oas_library(lib, str(path))


class OptimizedGridV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = SCRIPT_DIR / "test_outputs" / "_optimized_v1"
        shutil.rmtree(self.temp_root, ignore_errors=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def _make_runner(self, **overrides: object) -> optimized.OptimizedMainlineRunner:
        """创建一个统一配置的 v1 runner。"""

        config = {
            "clip_size_um": 1.0,
            "geometry_match_mode": "ecc",
            "area_match_ratio": 0.96,
            "edge_tolerance_um": 0.02,
            "pixel_size_nm": 10,
            "apply_layer_operations": False,
        }
        config.update(overrides)
        return optimized.OptimizedMainlineRunner(
            config=config,
            temp_dir=self.temp_root / f"run_{len(list(self.temp_root.glob('run_*'))):03d}",
        )

    def test_parser_removes_seed_strategy(self) -> None:
        help_text = optimized._build_parser().format_help()
        self.assertNotIn("--seed-strategy", help_text)
        self.assertIn("--clip-size", help_text)

    def test_uniform_grid_generates_only_occupied_cells(self) -> None:
        input_oas = self.temp_root / "occupied_grid_only.oas"
        _write_oas(
            input_oas,
            [
                gdstk.rectangle((0.00, 0.00), (0.20, 0.20), layer=1, datatype=0),
                gdstk.rectangle((1.10, 0.00), (1.30, 0.20), layer=1, datatype=0),
            ],
        )
        runner = self._make_runner()
        layout_index = runner._prepare_layout(input_oas)
        _, stats = optimized._build_uniform_grid_seed_candidates(layout_index, clip_size_um=1.0)
        self.assertEqual(stats["seed_strategy"], "uniform_grid")
        self.assertEqual(stats["grid_step_ratio"], optimized.GRID_STEP_RATIO)
        self.assertEqual(stats["grid_seed_count"], 2)
        self.assertLessEqual(stats["bucketed_seed_count"], stats["grid_seed_count"])

    def test_uniform_grid_keeps_tiny_feature_anchor_cell(self) -> None:
        input_oas = self.temp_root / "tiny_feature_anchor_cell.oas"
        _write_oas(
            input_oas,
            [gdstk.rectangle((0.01, 0.01), (0.03, 0.03), layer=1, datatype=0)],
        )
        runner = self._make_runner()
        layout_index = runner._prepare_layout(input_oas)
        _, stats = optimized._build_uniform_grid_seed_candidates(layout_index, clip_size_um=1.0)
        self.assertEqual(stats["grid_seed_count"], 1)
        self.assertEqual(stats["bucketed_seed_count"], 1)

    def test_grid_seed_bbox_equals_grid_cell_bbox_and_shift_limit(self) -> None:
        input_oas = self.temp_root / "grid_seed_bbox.oas"
        _write_oas(
            input_oas,
            [gdstk.rectangle((0.00, 0.00), (0.20, 0.20), layer=1, datatype=0)],
        )
        runner = self._make_runner()
        with redirect_stdout(StringIO()):
            records = runner._collect_marker_records_for_file(input_oas)
        self.assertGreater(len(records), 0)
        record = records[0]
        auto_seed = dict(record.match_cache.get("auto_seed", {}))
        self.assertEqual(auto_seed["seed_bbox"], auto_seed["grid_cell_bbox"])
        self.assertEqual(auto_seed["seed_bbox"], list(record.marker_bbox))
        expected_half_step_px = int(round((runner.clip_size_um * optimized.GRID_STEP_RATIO * 0.5) / runner.pixel_size_um))
        self.assertEqual(abs(int(record.shift_limits_px["x"][0])), expected_half_step_px)
        self.assertEqual(abs(int(record.shift_limits_px["x"][1])), expected_half_step_px)
        self.assertEqual(abs(int(record.shift_limits_px["y"][0])), expected_half_step_px)
        self.assertEqual(abs(int(record.shift_limits_px["y"][1])), expected_half_step_px)

    def test_repeated_tiles_accumulate_bucket_weight(self) -> None:
        input_oas = self.temp_root / "repeated_tiles.oas"
        _write_repeated_tile_oas(input_oas)
        runner = self._make_runner()
        with redirect_stdout(StringIO()):
            records = runner._collect_marker_records_for_file(input_oas)
        self.assertGreater(len(records), 0)
        self.assertTrue(any(record.seed_weight > 1 for record in records))

    def test_layer_operations_only_keep_result_layer(self) -> None:
        input_oas = self.temp_root / "layer_ops_result_only.oas"
        _write_oas(
            input_oas,
            [
                gdstk.rectangle((0.00, 0.00), (0.40, 0.20), layer=10, datatype=0),
                gdstk.rectangle((0.18, -0.02), (0.30, 0.22), layer=11, datatype=0),
                gdstk.rectangle((1.00, 1.00), (1.10, 1.10), layer=12, datatype=0),
            ],
        )
        processor = LayerOperationProcessor()
        processor.register_operation_rule("10/0", "subtract", "11/0", "13/0")
        runner = optimized.OptimizedMainlineRunner(
            config={
                "clip_size_um": 1.0,
                "geometry_match_mode": "ecc",
                "area_match_ratio": 0.96,
                "edge_tolerance_um": 0.02,
                "pixel_size_nm": 10,
                "apply_layer_operations": True,
            },
            temp_dir=self.temp_root / "run_layer_ops",
            layer_processor=processor,
        )
        layout_index = runner._prepare_layout(input_oas)
        pattern_layers = {(int(item["layer"]), int(item["datatype"])) for item in layout_index.indexed_elements}
        self.assertIn((13, 0), pattern_layers)
        self.assertIn((12, 0), pattern_layers)
        self.assertNotIn((10, 0), pattern_layers)
        self.assertNotIn((11, 0), pattern_layers)
        summary = runner._effective_layer_summary()
        self.assertIn("13/0", summary["effective_clustering_layers"])
        self.assertIn("10/0", summary["excluded_helper_layers"])

    def test_spatial_query_matches_reference_bbox_filter(self) -> None:
        input_oas = self.temp_root / "spatial_query_equivalence.oas"
        _write_oas(
            input_oas,
            [
                gdstk.rectangle((0.00, 0.00), (0.10, 0.10), layer=1, datatype=0),
                gdstk.rectangle((0.08, 0.08), (0.18, 0.18), layer=1, datatype=0),
                gdstk.rectangle((0.30, 0.30), (0.45, 0.45), layer=1, datatype=0),
            ],
        )
        runner = MainlineRunner(
            config={
                "clip_size_um": 1.0,
                "hotspot_layer": "999/0",
                "matching_mode": "ecc",
                "solver": "greedy",
                "geometry_mode": "exact",
                "pixel_size_nm": 10,
                "edge_tolerance_um": 0.02,
                "apply_layer_operations": False,
            },
            temp_dir=self.temp_root / "run_spatial_query",
        )
        layout_index = runner._prepare_layout(input_oas)
        query_bbox = (0.05, 0.05, 0.20, 0.20)
        actual = set(
            _query_candidate_ids(
                layout_index,
                query_bbox,
                geometry_mode="exact",
                max_elements=None,
                center_xy=(0.125, 0.125),
            )
        )
        expected = {
            idx
            for idx, _ in enumerate(layout_index.indexed_elements)
            if (
                float(layout_index.bbox_x1[idx]) > query_bbox[0]
                and float(layout_index.bbox_x0[idx]) < query_bbox[2]
                and float(layout_index.bbox_y1[idx]) > query_bbox[1]
                and float(layout_index.bbox_y0[idx]) < query_bbox[3]
            )
        }
        self.assertEqual(actual, expected)

    def test_chunked_coverage_matches_small_reference(self) -> None:
        bitmap_left = np.zeros((12, 12), dtype=bool)
        bitmap_left[4:8, 2:6] = True
        bitmap_right = np.zeros((12, 12), dtype=bool)
        bitmap_right[4:8, 6:10] = True

        runner = self._make_runner(geometry_match_mode="acc")
        exact_a = ExactCluster(0, _record("a0", bitmap_left, seed_weight=1), [_record("a0", bitmap_left, seed_weight=1)])
        exact_b = ExactCluster(1, _record("b0", bitmap_right, seed_weight=1), [_record("b0", bitmap_right, seed_weight=1)])
        cand_a_base = _candidate("cand_a_base", bitmap_left, origin_exact_cluster_id=0, shift_direction="base")
        cand_b_base = _candidate("cand_b_base", bitmap_right, origin_exact_cluster_id=1, shift_direction="base")
        cand_b_shift = _candidate("cand_b_shift", bitmap_left, origin_exact_cluster_id=1, shift_direction="left")

        old_budget = mainline.COVERAGE_CHUNK_BYTE_BUDGET
        try:
            mainline.COVERAGE_CHUNK_BYTE_BUDGET = 1
            runner.prefilter_stats = optimized._empty_prefilter_stats()
            runner._evaluate_candidate_coverage([cand_a_base, cand_b_base, cand_b_shift], [exact_a, exact_b])
        finally:
            mainline.COVERAGE_CHUNK_BYTE_BUDGET = old_budget

        self.assertIn(1, cand_a_base.coverage)
        self.assertIn(0, cand_b_shift.coverage)

    def test_exact_hash_coverage_does_not_build_geometry_cache(self) -> None:
        bitmap_left = np.zeros((12, 12), dtype=bool)
        bitmap_left[4:8, 2:6] = True
        bitmap_right = np.fliplr(bitmap_left)

        runner = self._make_runner(geometry_match_mode="ecc")
        exact_a = ExactCluster(0, _record("a0", bitmap_left, seed_weight=1), [_record("a0", bitmap_left, seed_weight=1)])
        exact_b = ExactCluster(1, _record("b0", bitmap_right, seed_weight=1), [_record("b0", bitmap_right, seed_weight=1)])
        cand_a = _candidate("cand_a", bitmap_left, origin_exact_cluster_id=0, shift_direction="base")
        cand_b = _candidate("cand_b", bitmap_right, origin_exact_cluster_id=1, shift_direction="base")

        runner.prefilter_stats = optimized._empty_prefilter_stats()
        with redirect_stdout(StringIO()):
            runner._evaluate_candidate_coverage([cand_a, cand_b], [exact_a, exact_b])

        self.assertIn(1, cand_a.coverage)
        self.assertIn(0, cand_b.coverage)
        self.assertGreater(runner.prefilter_stats["exact_hash_pass"], 0)
        self.assertEqual(runner.coverage_debug_stats["geometry_cache_group_count"], 0)
        self.assertEqual(runner.coverage_debug_stats["geometry_pair_count"], 0)

    def test_lazy_geometry_cache_only_for_geometry_candidates(self) -> None:
        bitmap_source = np.zeros((16, 16), dtype=bool)
        bitmap_source[4:12, 4:12] = True
        bitmap_shifted = bitmap_source.copy()
        bitmap_shifted[6, 6] = False
        bitmap_dense = np.ones((16, 16), dtype=bool)

        runner = self._make_runner(geometry_match_mode="ecc")
        exact_a = ExactCluster(0, _record("a0", bitmap_source, seed_weight=1), [_record("a0", bitmap_source, seed_weight=1)])
        exact_b = ExactCluster(1, _record("b0", bitmap_shifted, seed_weight=1), [_record("b0", bitmap_shifted, seed_weight=1)])
        exact_c = ExactCluster(2, _record("c0", bitmap_dense, seed_weight=1), [_record("c0", bitmap_dense, seed_weight=1)])
        cand_a = _candidate("cand_a", bitmap_source, origin_exact_cluster_id=0, shift_direction="base")
        cand_b = _candidate("cand_b", bitmap_shifted, origin_exact_cluster_id=1, shift_direction="base")
        cand_c = _candidate("cand_c", bitmap_dense, origin_exact_cluster_id=2, shift_direction="base")

        runner.prefilter_stats = optimized._empty_prefilter_stats()
        with redirect_stdout(StringIO()):
            runner._evaluate_candidate_coverage([cand_a, cand_b, cand_c], [exact_a, exact_b, exact_c])

        self.assertIn(1, cand_a.coverage)
        self.assertGreater(runner.coverage_debug_stats["geometry_pair_count"], 0)
        self.assertGreater(runner.coverage_debug_stats["geometry_cache_group_count"], 0)
        self.assertLess(runner.coverage_debug_stats["geometry_cache_group_count"], 3)
        self.assertGreaterEqual(runner.prefilter_stats["cheap_reject"], 0)

    def test_output_uses_uniform_grid_fields(self) -> None:
        input_oas = self.temp_root / "uniform_grid_output.oas"
        _write_oas(
            input_oas,
            [gdstk.rectangle((0.00, 0.00), (0.20, 0.20), layer=1, datatype=0)],
        )
        runner = self._make_runner()
        with redirect_stdout(StringIO()):
            result = runner.run(str(input_oas))
        self.assertEqual(result["pipeline_mode"], optimized.PIPELINE_MODE)
        self.assertEqual(result["seed_strategy"], "uniform_grid")
        self.assertIn("grid_step_ratio", result)
        self.assertIn("grid_step_um", result)
        self.assertIn("grid_seed_count", result)
        self.assertNotIn("contact_pair_seed_count", result)
        self.assertNotIn("drc_component_seed_count", result)
        self.assertGreater(result["grid_seed_count"], 0)
        self.assertIn("cheap_reject", result["prefilter_stats"])
        self.assertIn("coverage_detail_seconds", result)
        self.assertIn("coverage_debug_stats", result)
        self.assertIn("result_detail_seconds", result)
        self.assertIn("final_verification_detail_seconds", result)
        for key in (
            "geometry_dilated_cache_group_count",
            "geometry_donut_cache_group_count",
            "shortlist_subgroup_count",
            "shortlist_exact_subgroup_count",
            "shortlist_hnsw_subgroup_count",
            "shortlist_max_subgroup_size",
        ):
            self.assertIn(key, result["coverage_debug_stats"])
            self.assertGreaterEqual(int(result["coverage_debug_stats"][key]), 0)
        for value in result["coverage_detail_seconds"].values():
            self.assertGreaterEqual(float(value), 0.0)
        for value in result["result_detail_seconds"].values():
            self.assertGreaterEqual(float(value), 0.0)
        for value in result["final_verification_detail_seconds"].values():
            self.assertGreaterEqual(float(value), 0.0)
        self.assertTrue(all("export_representative_metadata" in cluster for cluster in result["clusters"]))

    def test_default_run_does_not_materialize_clip_files(self) -> None:
        input_oas = self.temp_root / "no_materialize.oas"
        _write_oas(
            input_oas,
            [gdstk.rectangle((0.00, 0.00), (0.20, 0.20), layer=1, datatype=0)],
        )
        runner = self._make_runner()
        with redirect_stdout(StringIO()):
            result = runner.run(str(input_oas))
        self.assertFalse(result["materialized_outputs"])
        self.assertEqual(result["total_samples"], len(result["file_metadata"]))
        self.assertEqual(result["total_files"], 0)
        self.assertEqual(result["file_list"], [])
        self.assertFalse((runner.temp_dir / "samples").exists())
        self.assertFalse((runner.temp_dir / "representatives").exists())
        for cluster in result["clusters"]:
            self.assertEqual(cluster["sample_files"], [])
            self.assertIsNone(cluster["representative_file"])

    def test_review_mode_materializes_clip_files(self) -> None:
        input_oas = self.temp_root / "materialize.oas"
        _write_oas(
            input_oas,
            [gdstk.rectangle((0.00, 0.00), (0.20, 0.20), layer=1, datatype=0)],
        )
        runner = self._make_runner(materialize_outputs=True)
        with redirect_stdout(StringIO()):
            result = runner.run(str(input_oas))
        self.assertTrue(result["materialized_outputs"])
        self.assertGreater(result["total_files"], 0)
        self.assertTrue((runner.temp_dir / "samples").exists())
        self.assertTrue((runner.temp_dir / "representatives").exists())
        self.assertTrue(all(Path(path).exists() for path in result["file_list"]))
        representative_files = [cluster["representative_file"] for cluster in result["clusters"]]
        self.assertTrue(all(Path(path).exists() for path in representative_files if path))

    def test_sample_layout_001_validation(self) -> None:
        if not SAMPLE_LAYOUT.exists():
            self.skipTest(f"missing sample layout: {SAMPLE_LAYOUT}")

        output_path = self.temp_root / "sample_layout_001_v1_result.json"
        runner = optimized.OptimizedMainlineRunner(
            config={
                "clip_size_um": 1.35,
                "geometry_match_mode": "ecc",
                "area_match_ratio": 0.96,
                "edge_tolerance_um": 0.02,
                "pixel_size_nm": 10,
                "apply_layer_operations": False,
            },
            temp_dir=self.temp_root / "run_sample_layout_001",
        )

        captured = StringIO()
        with redirect_stdout(captured):
            result = runner.run(str(SAMPLE_LAYOUT))
            optimized._save_results(result, str(output_path), "json")

        self.assertTrue(output_path.exists())
        loaded = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["pipeline_mode"], optimized.PIPELINE_MODE)
        self.assertEqual(loaded["seed_strategy"], "uniform_grid")
        self.assertGreater(loaded["grid_seed_count"], 0)
        self.assertGreater(loaded["bucketed_seed_count"], 0)
        self.assertGreater(loaded["exact_cluster_count"], 0)
        self.assertGreater(loaded["candidate_count"], 0)
        self.assertGreater(loaded["total_clusters"], 0)
        self.assertFalse(loaded["materialized_outputs"])
        self.assertEqual(loaded["total_samples"], len(loaded["file_metadata"]))
        self.assertEqual(loaded["total_files"], 0)
        self.assertIn("effective_clustering_layers", loaded)
        self.assertIn("excluded_helper_layers", loaded)
        self.assertNotIn("contact_pair_seed_count", loaded)
        self.assertNotIn("drc_component_seed_count", loaded)


if __name__ == "__main__":
    unittest.main(verbosity=2)
