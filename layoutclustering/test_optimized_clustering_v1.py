#!/usr/bin/env python3
"""Tests for the uniform-grid optimized v1 clustering pipeline."""

from __future__ import annotations

import csv
import os
import shutil
import sys
import unittest
import uuid
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import gdstk
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SAMPLE_LAYOUT = REPO_ROOT / "layoutgenerator" / "out_oas" / "sample_layout_001.oas"
CLUSTER_ASSIGNMENTS_TEMPLATE = SCRIPT_DIR / "cluster_assignments.csv"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import layout_clustering_optimized_v1 as optimized
import mainline
from layer_operations import LayerOperationProcessor
from layout_utils import _write_oas_library
from mainline import CandidateClip, ExactCluster, MainlineRunner, MarkerRecord, _canonical_bitmap_hash, _query_candidate_ids


def _record(seed_id: str, bitmap: np.ndarray, *, seed_weight: int, seed_type: str | None = None) -> MarkerRecord:
    """构造一个最小可用的 marker record，用于 coverage / cluster 单元测试。"""

    clip_hash, _ = _canonical_bitmap_hash(bitmap)
    height, width = bitmap.shape
    record = MarkerRecord(
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
    if seed_type is not None:
        record.match_cache["auto_seed"] = {"seed_type": str(seed_type)}
    return record


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


def _coverage_groups(runner: optimized.OptimizedMainlineRunner, candidates: list[CandidateClip]) -> list[optimized.CoverageCandidateGroup]:
    """把单测 candidate 显式合并成当前 coverage group 输入。"""

    group_buckets = {}
    ordered_groups = []
    for candidate in candidates:
        runner._merge_coverage_candidate(
            group_buckets,
            ordered_groups,
            candidate,
            retain_materialized_candidates=True,
        )
    return ordered_groups


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


def _make_shiftable_exact_cluster() -> ExactCluster:
    """构造同时具备 x/y systematic shift 空间的 exact cluster。"""

    expanded = np.zeros((8, 8), dtype=bool)
    expanded[1:5, 1:4] = True
    expanded[4:7, 4:6] = True
    clip_bbox_q = (2, 2, 6, 6)
    clip_bitmap = np.ascontiguousarray(expanded[2:6, 2:6], dtype=bool)
    clip_hash, _ = _canonical_bitmap_hash(clip_bitmap)
    expanded_hash, _ = _canonical_bitmap_hash(expanded)
    record = MarkerRecord(
        marker_id="marker_shiftable",
        source_path="synthetic.oas",
        source_name="synthetic.oas",
        marker_bbox=(0.02, 0.02, 0.06, 0.06),
        marker_center=(0.04, 0.04),
        clip_bbox=(0.02, 0.02, 0.06, 0.06),
        expanded_bbox=(0.0, 0.0, 0.08, 0.08),
        clip_bbox_q=clip_bbox_q,
        expanded_bbox_q=(0, 0, 8, 8),
        marker_bbox_q=clip_bbox_q,
        shift_limits_px={"x": (-2, 2), "y": (-2, 2)},
        clip_bitmap=clip_bitmap,
        expanded_bitmap=expanded,
        clip_hash=clip_hash,
        expanded_hash=expanded_hash,
        clip_area=float(np.count_nonzero(clip_bitmap)) * 0.0001,
        seed_weight=1,
        exact_cluster_id=0,
        match_cache={},
    )
    return ExactCluster(0, record, [record])


def _make_duplicate_shift_exact_cluster() -> ExactCluster:
    """构造所有 shift slice 都相同的 exact cluster，用于测试前置去重。"""

    expanded = np.ones((8, 8), dtype=bool)
    clip_bbox_q = (2, 2, 6, 6)
    clip_bitmap = np.ascontiguousarray(expanded[2:6, 2:6], dtype=bool)
    clip_hash, _ = _canonical_bitmap_hash(clip_bitmap)
    expanded_hash, _ = _canonical_bitmap_hash(expanded)
    record = MarkerRecord(
        marker_id="marker_duplicate_shift",
        source_path="synthetic.oas",
        source_name="synthetic.oas",
        marker_bbox=(0.02, 0.02, 0.06, 0.06),
        marker_center=(0.04, 0.04),
        clip_bbox=(0.02, 0.02, 0.06, 0.06),
        expanded_bbox=(0.0, 0.0, 0.08, 0.08),
        clip_bbox_q=clip_bbox_q,
        expanded_bbox_q=(0, 0, 8, 8),
        marker_bbox_q=clip_bbox_q,
        shift_limits_px={"x": (-2, 2), "y": (-2, 2)},
        clip_bitmap=clip_bitmap,
        expanded_bitmap=expanded,
        clip_hash=clip_hash,
        expanded_hash=expanded_hash,
        clip_area=float(np.count_nonzero(clip_bitmap)) * 0.0001,
        seed_weight=1,
        exact_cluster_id=0,
        match_cache={},
    )
    return ExactCluster(0, record, [record])


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
        parser = optimized._build_parser()
        help_text = parser.format_help()
        self.assertNotIn("--seed-strategy", help_text)
        for removed_option in (
            "--format",
            "--graph-invariant-limit",
            "--graph-topology-threshold",
            "--graph-signature-threshold",
            "--strict-invariant-limit",
            "--strict-topology-threshold",
            "--strict-signature-threshold",
            "--coverage-shortlist-max-targets",
        ):
            self.assertNotIn(removed_option, help_text)
        self.assertIn("--clip-size", help_text)
        self.assertIn("--compute-quality-metrics", help_text)
        args = parser.parse_args(["input.oas"])
        self.assertEqual(args.output, "clustering_results.csv")
        self.assertFalse(args.compute_quality_metrics)
        metrics_args = parser.parse_args(["input.oas", "--compute-quality-metrics"])
        self.assertTrue(metrics_args.compute_quality_metrics)
        with self.assertRaises(SystemExit), redirect_stderr(StringIO()):
            parser.parse_args(["input.oas", "--output", "clustering_results.json"])
        with self.assertRaises(SystemExit), redirect_stderr(StringIO()):
            parser.parse_args(["input.oas", "--output", "clustering_results.txt"])
        csv_args = parser.parse_args(["input.oas", "--output", "custom_result.CSV"])
        self.assertEqual(csv_args.output, "custom_result.CSV")
        for removed_option in (
            "--format",
            "--graph-signature-threshold",
            "--strict-invariant-limit",
            "--coverage-shortlist-max-targets",
        ):
            with self.assertRaises(SystemExit), redirect_stderr(StringIO()):
                parser.parse_args(["input.oas", removed_option, "1"])

    def test_geometry_driven_array_seed_reduces_regular_grid(self) -> None:
        input_oas = self.temp_root / "array_seed.oas"
        shapes = []
        for ix in range(6):
            for iy in range(6):
                x0 = 0.1 + ix * 0.6
                y0 = 0.1 + iy * 0.6
                shapes.append(gdstk.rectangle((x0, y0), (x0 + 0.18, y0 + 0.18), layer=1, datatype=0))
        _write_oas(input_oas, shapes)
        runner = self._make_runner()
        layout_index = runner._prepare_layout(input_oas)
        seeds, stats = optimized._build_geometry_driven_seed_candidates(layout_index, clip_size_um=1.0)
        coverage_audit = optimized._build_seed_coverage_audit(
            layout_index,
            optimized._layout_bbox(layout_index),
            float(stats["grid_step_um"]),
            1.0,
            seeds,
        )
        self.assertEqual(stats["seed_strategy"], "geometry_driven")
        self.assertEqual(stats["grid_step_ratio"], optimized.GRID_STEP_RATIO)
        self.assertGreater(stats["array_group_count"], 0)
        self.assertGreater(stats["array_seed_count"], 0)
        self.assertGreater(stats["array_spacing_seed_count"], 0)
        self.assertGreater(stats["array_spacing_group_count"], 0)
        self.assertGreater(stats["array_spacing_weight_total"], 0)
        self.assertIn("array_representative", stats["seed_type_counts"])
        self.assertIn("array_spacing", stats["seed_type_counts"])
        self.assertIn("seed_audit", stats)
        self.assertGreater(stats["seed_audit"]["array_group_count"], 0)
        self.assertGreater(stats["seed_audit"]["array_groups"][0]["spacing_representative_count"], 0)
        self.assertEqual(stats["long_shape_count"], 0)
        self.assertEqual(stats["residual_element_count"], 0)
        self.assertLess(len(seeds), len(shapes))
        self.assertTrue(any(seed.seed_type == optimized.SEED_TYPE_ARRAY for seed in seeds))
        self.assertTrue(any(seed.seed_type == optimized.SEED_TYPE_ARRAY_SPACE for seed in seeds))
        self.assertGreaterEqual(stats["seed_weight_total"], len(shapes))
        for key in (
            "target_edge_length_coverage_ratio",
            "target_polygon_area_coverage_ratio",
            "weighted_pattern_type_coverage_ratio",
        ):
            self.assertGreaterEqual(float(coverage_audit[key]), 0.0)
            self.assertLessEqual(float(coverage_audit[key]), 1.0)
        self.assertNotIn("clip_window_union_coverage_ratio", coverage_audit)
        self.assertNotIn("clip_window_uncovered_occupied_grid_ratio", coverage_audit)

    def test_pattern_coverage_ignores_tiny_grid_touch_noise(self) -> None:
        """真实 polygon coverage 不应被旧 grid touch 噪声影响。"""

        input_oas = self.temp_root / "tiny_touch_coverage.oas"
        _write_oas(
            input_oas,
            [gdstk.rectangle((1.90, 1.90), (1.91, 1.91), layer=1, datatype=0)],
        )
        runner = self._make_runner()
        layout_index = runner._prepare_layout(input_oas)
        seeds = [
            optimized.GridSeedCandidate(
                center=(0.5, 0.5),
                seed_bbox=(0.0, 0.0, 1.0, 1.0),
                grid_ix=0,
                grid_iy=0,
                seed_type=optimized.SEED_TYPE_RESIDUAL,
            )
        ]
        coverage_audit = optimized._build_seed_coverage_audit(
            layout_index,
            optimized._layout_bbox(layout_index),
            0.5,
            1.0,
            seeds,
        )
        self.assertEqual(float(coverage_audit["target_edge_length_coverage_ratio"]), 0.0)
        self.assertEqual(float(coverage_audit["target_polygon_area_coverage_ratio"]), 0.0)
        self.assertEqual(float(coverage_audit["weighted_pattern_type_coverage_ratio"]), 0.0)
        self.assertNotIn("clip_window_union_coverage_ratio", coverage_audit)

    def test_pattern_coverage_deduplicates_overlapping_clip_windows(self) -> None:
        """重叠 clip window 不应重复累计边长或面积 coverage。"""

        input_oas = self.temp_root / "overlap_coverage.oas"
        _write_oas(input_oas, [gdstk.rectangle((0.0, 0.0), (1.0, 1.0), layer=1, datatype=0)])
        runner = self._make_runner()
        layout_index = runner._prepare_layout(input_oas)
        seeds = [
            optimized.GridSeedCandidate((0.5, 0.5), (0.0, 0.0, 1.0, 1.0), 0, 0),
            optimized.GridSeedCandidate((0.6, 0.5), (0.1, 0.0, 1.1, 1.0), 0, 0),
        ]
        coverage_audit = optimized._build_seed_coverage_audit(
            layout_index,
            optimized._layout_bbox(layout_index),
            0.5,
            1.0,
            seeds,
        )
        self.assertAlmostEqual(float(coverage_audit["target_edge_length_coverage_ratio"]), 1.0, places=6)
        self.assertAlmostEqual(float(coverage_audit["target_polygon_area_coverage_ratio"]), 1.0, places=6)

    def test_pattern_coverage_reports_partial_polygon_coverage(self) -> None:
        """半覆盖 polygon 应输出可解释的边长和面积覆盖比例。"""

        input_oas = self.temp_root / "partial_coverage.oas"
        _write_oas(input_oas, [gdstk.rectangle((0.0, 0.0), (2.0, 1.0), layer=1, datatype=0)])
        runner = self._make_runner()
        layout_index = runner._prepare_layout(input_oas)
        seeds = [optimized.GridSeedCandidate((0.5, 0.5), (0.0, 0.0, 1.0, 1.0), 0, 0)]
        coverage_audit = optimized._build_seed_coverage_audit(
            layout_index,
            optimized._layout_bbox(layout_index),
            0.5,
            1.0,
            seeds,
        )
        self.assertAlmostEqual(float(coverage_audit["target_edge_length_coverage_ratio"]), 0.5, places=6)
        self.assertAlmostEqual(float(coverage_audit["target_polygon_area_coverage_ratio"]), 0.5, places=6)

    def test_weighted_pattern_type_coverage_uses_edge_weight(self) -> None:
        """轻量 pattern-type coverage 应按 type 的边长权重汇总。"""

        input_oas = self.temp_root / "type_coverage.oas"
        _write_oas(
            input_oas,
            [
                gdstk.rectangle((0.0, 0.0), (1.0, 1.0), layer=1, datatype=0),
                gdstk.rectangle((10.0, 0.0), (12.0, 1.0), layer=1, datatype=0),
            ],
        )
        runner = self._make_runner()
        layout_index = runner._prepare_layout(input_oas)
        seeds = [optimized.GridSeedCandidate((0.5, 0.5), (0.0, 0.0, 1.0, 1.0), 0, 0)]
        coverage_audit = optimized._build_seed_coverage_audit(
            layout_index,
            optimized._layout_bbox(layout_index),
            0.5,
            1.0,
            seeds,
        )
        self.assertAlmostEqual(float(coverage_audit["weighted_pattern_type_coverage_ratio"]), 0.4, places=6)

    def test_residual_local_grid_keeps_isolated_feature(self) -> None:
        input_oas = self.temp_root / "residual_seed.oas"
        _write_oas(
            input_oas,
            [gdstk.rectangle((0.01, 0.01), (0.03, 0.03), layer=1, datatype=0)],
        )
        runner = self._make_runner()
        layout_index = runner._prepare_layout(input_oas)
        seeds, stats = optimized._build_geometry_driven_seed_candidates(layout_index, clip_size_um=1.0)
        self.assertEqual(stats["seed_strategy"], "geometry_driven")
        self.assertEqual(stats["grid_seed_count"], 1)
        self.assertEqual(stats["bucketed_seed_count"], 1)
        self.assertEqual(stats["array_group_count"], 0)
        self.assertEqual(stats["long_shape_count"], 0)
        self.assertEqual(stats["residual_element_count"], 1)
        self.assertEqual(stats["residual_seed_count"], 1)
        self.assertEqual(len(seeds), 1)
        self.assertTrue(all(seed.seed_type == optimized.SEED_TYPE_RESIDUAL for seed in seeds))

    def test_seed_bbox_matches_marker_bbox_and_shift_limit(self) -> None:
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
        self.assertNotIn("grid_cell_bbox", auto_seed)
        self.assertEqual(auto_seed["seed_bbox"], list(record.marker_bbox))
        self.assertIn(auto_seed["seed_type"], {optimized.SEED_TYPE_ARRAY, optimized.SEED_TYPE_ARRAY_SPACE, optimized.SEED_TYPE_LONG, optimized.SEED_TYPE_RESIDUAL})
        expected_half_step_px = int(round((runner.clip_size_um * optimized.GRID_STEP_RATIO * 0.5) / runner.pixel_size_um))
        self.assertEqual(abs(int(record.shift_limits_px["x"][0])), expected_half_step_px)
        self.assertEqual(abs(int(record.shift_limits_px["x"][1])), expected_half_step_px)
        self.assertEqual(abs(int(record.shift_limits_px["y"][0])), expected_half_step_px)
        self.assertEqual(abs(int(record.shift_limits_px["y"][1])), expected_half_step_px)

    def test_array_spacing_seed_keeps_separate_dedupe_slot(self) -> None:
        center_seed = optimized.GridSeedCandidate((0.0, 0.0), (-0.1, -0.1, 0.1, 0.1), 3, 4, 2, optimized.SEED_TYPE_ARRAY)
        spacing_seed = optimized.GridSeedCandidate(
            (0.0, 0.0),
            (-0.1, -0.1, 0.1, 0.1),
            3,
            4,
            5,
            optimized.SEED_TYPE_ARRAY_SPACE,
        )
        duplicate_spacing = optimized.GridSeedCandidate(
            (0.0, 0.0),
            (-0.1, -0.1, 0.1, 0.1),
            3,
            4,
            7,
            optimized.SEED_TYPE_ARRAY_SPACE,
        )
        deduped = optimized._dedupe_geometry_seeds([center_seed, spacing_seed, duplicate_spacing])
        self.assertEqual(len(deduped), 2)
        type_counts = {seed.seed_type: seed.bucket_weight for seed in deduped}
        self.assertEqual(type_counts[optimized.SEED_TYPE_ARRAY], 2)
        self.assertEqual(type_counts[optimized.SEED_TYPE_ARRAY_SPACE], 12)

    def test_long_shape_path_seed_is_one_dimensional(self) -> None:
        input_oas = self.temp_root / "long_seed.oas"
        _write_oas(
            input_oas,
            [
                gdstk.rectangle((0.0, 0.0), (20.0, 0.4), layer=1, datatype=0),
                gdstk.rectangle((9.8, -2.0), (10.2, 2.0), layer=1, datatype=0),
            ],
        )
        runner = self._make_runner()
        layout_index = runner._prepare_layout(input_oas)
        seeds, stats = optimized._build_geometry_driven_seed_candidates(layout_index, clip_size_um=1.0)
        self.assertEqual(stats["seed_strategy"], "geometry_driven")
        self.assertEqual(stats["long_shape_count"], 2)
        self.assertGreater(stats["long_shape_seed_count"], 0)
        self.assertEqual(stats["residual_element_count"], 0)
        self.assertLess(len(seeds), 80)
        self.assertTrue(all(seed.seed_type == optimized.SEED_TYPE_LONG for seed in seeds))

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

    def test_candidate_generation_adds_bounded_diagonal_shifts(self) -> None:
        cluster = _make_shiftable_exact_cluster()
        runner = self._make_runner()
        candidates = runner._generate_candidates_for_cluster(cluster)
        directions = {str(candidate.shift_direction) for candidate in candidates}
        diagonal_candidates = [candidate for candidate in candidates if str(candidate.shift_direction).startswith("diag_")]
        self.assertIn("base", directions)
        self.assertTrue(any(direction in directions for direction in ("left", "right")))
        self.assertTrue(any(direction in directions for direction in ("up", "down")))
        self.assertGreater(len(diagonal_candidates), 0)
        self.assertLessEqual(len(diagonal_candidates), optimized.DIAGONAL_SHIFT_MAX_COUNT)
        for candidate in diagonal_candidates:
            self.assertNotEqual(candidate.clip_bbox_q[0], cluster.representative.clip_bbox_q[0])
            self.assertNotEqual(candidate.clip_bbox_q[1], cluster.representative.clip_bbox_q[1])
            self.assertGreater(candidate.shift_distance_um, 0.0)
        summary = optimized._candidate_shift_summary(candidates)
        self.assertEqual(summary["diagonal_candidate_count"], len(diagonal_candidates))
        self.assertGreater(summary["max_shift_distance_um"], 0.0)

    def test_packed_expanded_bitmap_preserves_candidate_generation(self) -> None:
        original_cluster = _make_shiftable_exact_cluster()
        packed_cluster = _make_shiftable_exact_cluster()
        runner = self._make_runner()
        original_candidates = runner._generate_candidates_for_cluster(original_cluster)
        self.assertTrue(optimized._pack_marker_expanded_bitmap(packed_cluster.representative))
        self.assertIsNone(packed_cluster.representative.expanded_bitmap)

        packed_candidates = runner._generate_candidates_for_cluster(packed_cluster)

        self.assertEqual(
            [(candidate.clip_hash, candidate.shift_direction, candidate.clip_bbox_q) for candidate in packed_candidates],
            [(candidate.clip_hash, candidate.shift_direction, candidate.clip_bbox_q) for candidate in original_candidates],
        )

    def test_early_duplicate_shift_keeps_base_candidate(self) -> None:
        runner = self._make_runner()
        cluster = _make_duplicate_shift_exact_cluster()

        candidates = runner._generate_candidates_for_cluster(cluster)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].shift_direction, "base")

    def test_candidate_bitmap_interning_shares_equal_bitmaps(self) -> None:
        bitmap = np.zeros((8, 8), dtype=bool)
        bitmap[2:6, 2:6] = True
        runner = self._make_runner()
        exact_a = ExactCluster(0, _record("a0", bitmap, seed_weight=1), [_record("a0", bitmap, seed_weight=1)])
        exact_b = ExactCluster(1, _record("b0", bitmap, seed_weight=1), [_record("b0", bitmap, seed_weight=1)])

        cand_a = runner._build_candidate_clip(exact_a, exact_a.representative.clip_bbox, exact_a.representative.clip_bbox_q, bitmap.copy(), "base", 0.0, 0)
        cand_b = runner._build_candidate_clip(exact_b, exact_b.representative.clip_bbox, exact_b.representative.clip_bbox_q, bitmap.copy(), "base", 0.0, 0)

        self.assertIs(cand_a.clip_bitmap, cand_b.clip_bitmap)
        self.assertNotEqual(cand_a.candidate_id, cand_b.candidate_id)
        self.assertEqual(cand_a.coverage, (0,))
        self.assertEqual(cand_b.coverage, (1,))

    def test_digest_key_groups_same_bitmap_without_raw_key(self) -> None:
        bitmap = np.zeros((8, 8), dtype=bool)
        bitmap[2:6, 2:6] = True
        runner = self._make_runner()
        cand_a = _candidate("cand_a", bitmap, origin_exact_cluster_id=0, shift_direction="base")
        cand_b = _candidate("cand_b", bitmap.copy(), origin_exact_cluster_id=1, shift_direction="base")

        bundle = next(iter(runner._build_candidate_match_bundles(_coverage_groups(runner, [cand_a, cand_b]), 0).values()))

        self.assertEqual(len(bundle["candidate_groups"]), 1)
        self.assertEqual(bundle["candidate_groups"][0].logical_candidate_count, 2)

    def test_digest_collision_does_not_merge_different_bitmaps(self) -> None:
        bitmap_a = np.zeros((8, 8), dtype=bool)
        bitmap_a[2:6, 2:6] = True
        bitmap_b = np.zeros((8, 8), dtype=bool)
        bitmap_b[1:3, 1:7] = True
        runner = self._make_runner()
        cand_a = _candidate("cand_a", bitmap_a, origin_exact_cluster_id=0, shift_direction="base")
        cand_b = _candidate("cand_b", bitmap_b, origin_exact_cluster_id=1, shift_direction="base")
        original_digest = optimized._strict_bitmap_digest

        def constant_digest(packed, shape):
            """强制制造 digest collision，验证逐像素比较兜底。"""

            del packed, shape
            return b"\x01" * optimized.STRICT_BITMAP_DIGEST_SIZE

        try:
            optimized._strict_bitmap_digest = constant_digest
            bundle = next(iter(runner._build_candidate_match_bundles(_coverage_groups(runner, [cand_a, cand_b]), 0).values()))
        finally:
            optimized._strict_bitmap_digest = original_digest

        self.assertEqual(len(bundle["candidate_groups"]), 2)
        self.assertEqual([group.logical_candidate_count for group in bundle["candidate_groups"]], [1, 1])

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
            runner._evaluate_candidate_coverage(_coverage_groups(runner, [cand_a_base, cand_b_base, cand_b_shift]), [exact_a, exact_b])
        finally:
            mainline.COVERAGE_CHUNK_BYTE_BUDGET = old_budget

        self.assertIn(1, cand_a_base.coverage)
        self.assertIn(0, cand_b_shift.coverage)

    def test_final_verification_accepts_shift_witness_match(self) -> None:
        """final verification 应通过 target shift witness，而不是只比较 target base。"""

        bitmap_left = np.zeros((12, 12), dtype=bool)
        bitmap_left[4:8, 2:6] = True
        bitmap_base = np.zeros((12, 12), dtype=bool)
        bitmap_base[4:8, 6:11] = True
        exact_b = ExactCluster(1, _record("b0", bitmap_base, seed_weight=1), [_record("b0", bitmap_base, seed_weight=1)])
        selected = _candidate("selected_left", bitmap_left, origin_exact_cluster_id=0, shift_direction="left", coverage={1})
        target_base = _candidate("target_base", bitmap_base, origin_exact_cluster_id=1, shift_direction="base")
        target_shift = _candidate("target_shift", bitmap_left, origin_exact_cluster_id=1, shift_direction="left")
        runner = self._make_runner(geometry_match_mode="acc")
        runner._base_candidate_by_exact_id[1] = target_base
        runner._target_witness_candidates = lambda exact_cluster: ([target_base, target_shift], target_base)

        units = runner._verified_cluster_units([selected], [exact_b])

        self.assertEqual(units, [(selected, [exact_b])])
        self.assertEqual(runner.final_verification_stats["verified_pass"], 1)
        self.assertEqual(runner.final_verification_stats["witness_attempted"], 1)
        self.assertEqual(runner.final_verification_stats["witness_verified_pass"], 1)
        self.assertEqual(runner.final_verification_breakdown["pass_reason_counts"]["exact_hash"], 1)
        self.assertEqual(runner.final_verification_breakdown["witness_shift_direction_counts"]["left"], 1)
        self.assertEqual(runner.final_verification_stats["singleton_created"], 0)

    def test_final_verification_rejects_when_all_witnesses_fail(self) -> None:
        """所有 target witnesses 都失败时应输出细分几何原因并创建 singleton。"""

        bitmap_left = np.zeros((12, 12), dtype=bool)
        bitmap_left[4:8, 2:6] = True
        bitmap_base = np.zeros((12, 12), dtype=bool)
        bitmap_base[4:8, 6:11] = True
        exact_b = ExactCluster(1, _record("b0", bitmap_base, seed_weight=1), [_record("b0", bitmap_base, seed_weight=1)])
        selected = _candidate("selected_left", bitmap_left, origin_exact_cluster_id=0, shift_direction="left", coverage={1})
        target_base = _candidate("target_base", bitmap_base, origin_exact_cluster_id=1, shift_direction="base")
        runner = self._make_runner(geometry_match_mode="acc")
        runner._base_candidate_by_exact_id[1] = target_base
        runner._target_witness_candidates = lambda exact_cluster: ([target_base], target_base)

        units = runner._verified_cluster_units([selected], [exact_b])

        self.assertEqual(units, [(target_base, [exact_b])])
        self.assertEqual(runner.final_verification_stats["verified_reject"], 1)
        self.assertEqual(runner.final_verification_stats["singleton_created"], 1)
        self.assertEqual(runner.final_verification_breakdown["reject_reason_counts"]["geometry_acc_xor"], 1)
        self.assertNotIn("geometry", runner.final_verification_breakdown["reject_reason_counts"])

    def test_bucketed_coverage_matches_small_reference(self) -> None:
        bitmap_left = np.zeros((12, 12), dtype=bool)
        bitmap_left[4:8, 2:6] = True
        bitmap_right = np.zeros((12, 12), dtype=bool)
        bitmap_right[4:8, 6:10] = True
        exact_a = ExactCluster(0, _record("a0", bitmap_left, seed_weight=1), [_record("a0", bitmap_left, seed_weight=1)])
        exact_b = ExactCluster(1, _record("b0", bitmap_right, seed_weight=1), [_record("b0", bitmap_right, seed_weight=1)])

        def run_once(force_bucketed: bool) -> tuple[tuple[int, ...], tuple[int, ...], dict[str, int]]:
            """运行一次 coverage，返回关键 coverage 与统计。"""

            runner = self._make_runner(geometry_match_mode="acc")
            cand_a = _candidate("cand_a", bitmap_left, origin_exact_cluster_id=0, shift_direction="base")
            cand_b = _candidate("cand_b", bitmap_right, origin_exact_cluster_id=1, shift_direction="base")
            cand_shift = _candidate("cand_shift", bitmap_left, origin_exact_cluster_id=1, shift_direction="left")
            old_threshold = optimized.COVERAGE_BUCKETED_GROUP_THRESHOLD
            try:
                if force_bucketed:
                    optimized.COVERAGE_BUCKETED_GROUP_THRESHOLD = 1
                runner.prefilter_stats = optimized._empty_prefilter_stats()
                with redirect_stdout(StringIO()):
                    runner._evaluate_candidate_coverage(_coverage_groups(runner, [cand_a, cand_b, cand_shift]), [exact_a, exact_b])
            finally:
                optimized.COVERAGE_BUCKETED_GROUP_THRESHOLD = old_threshold
            return tuple(cand_a.coverage), tuple(cand_shift.coverage), dict(runner.coverage_debug_stats)

        normal_a, normal_shift, _ = run_once(False)
        bucketed_a, bucketed_shift, bucket_stats = run_once(True)

        self.assertEqual(bucketed_a, normal_a)
        self.assertEqual(bucketed_shift, normal_shift)
        self.assertGreater(bucket_stats["bucketed_coverage_bundle_count"], 0)
        self.assertGreater(bucket_stats["coverage_fill_bin_count"], 0)
        self.assertNotIn("coverage_density_bin_count", bucket_stats)

    def test_bucketed_coverage_uses_fill_only_windows_by_default(self) -> None:
        """强制走 bucketed coverage，并验证默认 fill-only window 不降低 coverage。"""

        bitmap_solid = np.zeros((12, 12), dtype=bool)
        bitmap_solid[1:3, 1:3] = True
        bitmap_sparse = np.zeros((12, 12), dtype=bool)
        bitmap_sparse[1, 1] = True
        bitmap_sparse[1, 10] = True
        bitmap_sparse[10, 1] = True
        bitmap_sparse[10, 10] = True
        exact_a = ExactCluster(0, _record("a0", bitmap_solid, seed_weight=1), [_record("a0", bitmap_solid, seed_weight=1)])
        exact_b = ExactCluster(1, _record("b0", bitmap_sparse, seed_weight=1), [_record("b0", bitmap_sparse, seed_weight=1)])

        def run_once(force_bucketed: bool) -> tuple[tuple[int, ...], tuple[int, ...], dict[str, int]]:
            """运行一次 coverage，返回两个候选的 coverage 与内部统计。"""

            runner = self._make_runner(geometry_match_mode="acc")
            cand_solid = _candidate("cand_solid", bitmap_solid, origin_exact_cluster_id=0, shift_direction="base")
            cand_sparse = _candidate("cand_sparse", bitmap_sparse, origin_exact_cluster_id=1, shift_direction="base")
            old_threshold = optimized.COVERAGE_BUCKETED_GROUP_THRESHOLD
            try:
                if force_bucketed:
                    optimized.COVERAGE_BUCKETED_GROUP_THRESHOLD = 1
                runner.prefilter_stats = optimized._empty_prefilter_stats()
                with redirect_stdout(StringIO()):
                    runner._evaluate_candidate_coverage(_coverage_groups(runner, [cand_solid, cand_sparse]), [exact_a, exact_b])
            finally:
                optimized.COVERAGE_BUCKETED_GROUP_THRESHOLD = old_threshold
            return tuple(cand_solid.coverage), tuple(cand_sparse.coverage), dict(runner.coverage_debug_stats)

        normal_solid, normal_sparse, _ = run_once(False)
        bucketed_solid, bucketed_sparse, bucket_stats = run_once(True)

        self.assertEqual(bucketed_solid, normal_solid)
        self.assertEqual(bucketed_sparse, normal_sparse)
        self.assertGreater(bucket_stats["coverage_fill_bin_count"], 0)
        self.assertNotIn("coverage_density_bin_count", bucket_stats)
        self.assertNotIn("descriptor_lru_live_peak_count", bucket_stats)

    def test_same_hash_exact_pass_crosses_fill_bins_without_geometry_cache(self) -> None:
        bitmap_sparse = np.zeros((12, 12), dtype=bool)
        bitmap_sparse[1, 1] = True
        bitmap_sparse[1, 10] = True
        bitmap_sparse[10, 1] = True
        bitmap_sparse[10, 10] = True
        bitmap_dense = np.ones((12, 12), dtype=bool)
        runner = self._make_runner(geometry_match_mode="ecc")
        cand_sparse = _candidate("cand_sparse", bitmap_sparse, origin_exact_cluster_id=0, shift_direction="base")
        cand_dense = _candidate("cand_dense", bitmap_dense, origin_exact_cluster_id=1, shift_direction="base")
        cand_sparse.clip_hash = "forced_same_hash"
        cand_dense.clip_hash = "forced_same_hash"
        exact_a = ExactCluster(0, _record("a0", bitmap_sparse, seed_weight=1), [_record("a0", bitmap_sparse, seed_weight=1)])
        exact_b = ExactCluster(1, _record("b0", bitmap_dense, seed_weight=1), [_record("b0", bitmap_dense, seed_weight=1)])
        old_threshold = optimized.COVERAGE_BUCKETED_GROUP_THRESHOLD
        try:
            optimized.COVERAGE_BUCKETED_GROUP_THRESHOLD = 1
            runner.prefilter_stats = optimized._empty_prefilter_stats()
            with redirect_stdout(StringIO()):
                runner._evaluate_candidate_coverage(_coverage_groups(runner, [cand_sparse, cand_dense]), [exact_a, exact_b])
        finally:
            optimized.COVERAGE_BUCKETED_GROUP_THRESHOLD = old_threshold

        self.assertIn(1, cand_sparse.coverage)
        self.assertIn(0, cand_dense.coverage)
        self.assertGreater(runner.prefilter_stats["exact_hash_pass"], 0)
        self.assertEqual(runner.coverage_debug_stats["geometry_cache_group_count"], 0)
        self.assertLess(
            runner.coverage_debug_stats["max_bucket_window_group_count"],
            runner.coverage_debug_stats["max_bundle_group_count"],
        )
        self.assertNotIn("coverage_density_bin_count", runner.coverage_debug_stats)

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
            runner._evaluate_candidate_coverage(_coverage_groups(runner, [cand_a, cand_b]), [exact_a, exact_b])

        self.assertIn(1, cand_a.coverage)
        self.assertIn(0, cand_b.coverage)
        self.assertGreater(runner.prefilter_stats["exact_hash_pass"], 0)
        self.assertEqual(runner.coverage_debug_stats["full_descriptor_cache_group_count"], 0)
        self.assertEqual(runner.coverage_debug_stats["geometry_cache_group_count"], 0)
        self.assertEqual(runner.coverage_debug_stats["geometry_cache_release_count"], 0)
        self.assertEqual(runner.coverage_debug_stats["geometry_pair_count"], 0)
        self.assertNotIn("optimized_graph_descriptor", cand_a.match_cache)
        self.assertNotIn("optimized_graph_descriptor", cand_b.match_cache)

    def test_lazy_full_prefilter_cache_only_for_survivors(self) -> None:
        bitmap_source = np.zeros((16, 16), dtype=bool)
        bitmap_source[4:12, 4:12] = True
        bitmap_shifted = bitmap_source.copy()
        bitmap_shifted[6, 6] = False
        bitmap_dense = np.ones((16, 16), dtype=bool)

        runner = self._make_runner(geometry_match_mode="ecc")
        cand_a = _candidate("cand_a", bitmap_source, origin_exact_cluster_id=0, shift_direction="base")
        cand_b = _candidate("cand_b", bitmap_shifted, origin_exact_cluster_id=1, shift_direction="base")
        cand_c = _candidate("cand_c", bitmap_dense, origin_exact_cluster_id=2, shift_direction="base")

        runner.prefilter_stats = optimized._empty_prefilter_stats()
        runner.coverage_detail_seconds = optimized._empty_coverage_detail_seconds()
        runner.coverage_debug_stats = optimized._empty_coverage_debug_stats()
        bundle = next(iter(runner._build_candidate_match_bundles(_coverage_groups(runner, [cand_a, cand_b, cand_c]), 2).values()))
        shortlist_index = runner._build_bundle_shortlist_index(bundle)
        self.assertNotIn("descriptors", shortlist_index)
        self.assertNotIn("signature_embeddings", shortlist_index)
        self.assertEqual(shortlist_index["cheap_invariants"].shape[0], 3)
        kept = runner._batch_prefilter(bundle, shortlist_index, 0, np.asarray([1, 2], dtype=np.int64))

        self.assertEqual(kept.tolist(), [1])
        self.assertEqual(runner.coverage_debug_stats["full_descriptor_cache_group_count"], 2)
        self.assertGreaterEqual(runner.coverage_debug_stats["lazy_signature_embedding_group_count"], 0)
        self.assertEqual(set(bundle["full_descriptor_cache_by_idx"].keys()), {0, 1})
        self.assertNotIn("optimized_graph_descriptor", cand_a.match_cache)
        self.assertNotIn("optimized_graph_descriptor", cand_b.match_cache)
        self.assertNotIn("optimized_graph_descriptor", cand_c.match_cache)

    def test_full_descriptor_cache_is_window_local_and_released_per_source(self) -> None:
        """验证 full descriptor 使用 window-local cache，并在 source prefilter 后释放。"""

        bitmap = np.zeros((16, 16), dtype=bool)
        bitmap[4:12, 4:12] = True
        bitmap_shifted = bitmap.copy()
        bitmap_shifted[6, 6] = False
        runner = self._make_runner(geometry_match_mode="ecc")
        cand_a = _candidate("cand_a", bitmap, origin_exact_cluster_id=0, shift_direction="base")
        cand_b = _candidate("cand_b", bitmap_shifted, origin_exact_cluster_id=1, shift_direction="base")
        exact_a = ExactCluster(0, _record("a0", bitmap, seed_weight=1), [_record("a0", bitmap, seed_weight=1)])
        exact_b = ExactCluster(1, _record("b0", bitmap_shifted, seed_weight=1), [_record("b0", bitmap_shifted, seed_weight=1)])
        release_snapshots: list[tuple[int, int]] = []
        original_release = runner._release_bundle_full_descriptor_cache

        def recording_release(bundle: dict[str, object]) -> None:
            """记录释放前后的 full descriptor cache 大小。"""

            before = len(bundle.get("full_descriptor_cache_by_idx", {}))
            original_release(bundle)
            after = len(bundle.get("full_descriptor_cache_by_idx", {}))
            release_snapshots.append((before, after))

        runner._release_bundle_full_descriptor_cache = recording_release
        runner.prefilter_stats = optimized._empty_prefilter_stats()
        with redirect_stdout(StringIO()):
            runner._evaluate_candidate_coverage(_coverage_groups(runner, [cand_a, cand_b]), [exact_a, exact_b])

        self.assertTrue(any(before > 0 and after == 0 for before, after in release_snapshots))
        self.assertGreater(runner.coverage_debug_stats["full_descriptor_cache_group_count"], 0)
        self.assertNotIn("full_descriptor_lru_hit", runner.coverage_debug_stats)

    def test_lazy_signature_embedding_shortlist_matches_reference(self) -> None:
        bitmap_a = np.zeros((16, 16), dtype=bool)
        bitmap_a[4:12, 4:12] = True
        bitmap_b = bitmap_a.copy()
        bitmap_b[6, 6] = False
        bitmap_c = bitmap_a.copy()
        bitmap_c[9, 9] = False
        runner = self._make_runner(geometry_match_mode="ecc")
        cand_a = _candidate("cand_a", bitmap_a, origin_exact_cluster_id=0, shift_direction="base")
        cand_b = _candidate("cand_b", bitmap_b, origin_exact_cluster_id=1, shift_direction="base")
        cand_c = _candidate("cand_c", bitmap_c, origin_exact_cluster_id=2, shift_direction="base")
        bundle = next(iter(runner._build_candidate_match_bundles(_coverage_groups(runner, [cand_a, cand_b, cand_c]), 2).values()))
        shortlist_index = runner._build_bundle_shortlist_index(bundle)
        subgroup_id = int(shortlist_index["source_subgroup_ids"][0])
        subgroup_key = shortlist_index["subgroup_keys"][subgroup_id]
        group_indices = np.asarray(shortlist_index["subgroup_members"][subgroup_key], dtype=np.int32)
        self.assertGreaterEqual(int(group_indices.size), 2)

        runner.coverage_shortlist_max_targets = 1
        payload = runner._ensure_shortlist_payload(shortlist_index, subgroup_key)
        group_vectors = np.asarray(
            [
                optimized._signature_embedding(
                    optimized._cheap_bitmap_descriptor(bundle["representatives"][int(group_idx)].clip_bitmap)
                )
                for group_idx in group_indices.tolist()
            ],
            dtype=np.float32,
        )
        expected_labels = group_indices[
            optimized._exact_cosine_topk_labels(
                group_vectors,
                min(int(runner.coverage_shortlist_max_targets) + 1, int(group_indices.size)),
            )
        ].astype(np.int32, copy=False)

        np.testing.assert_array_equal(np.asarray(payload["mapped_labels"], dtype=np.int32), expected_labels)
        self.assertGreater(runner.coverage_debug_stats["lazy_signature_embedding_group_count"], 0)
        self.assertGreater(runner.coverage_debug_stats["signature_embedding_live_peak_count"], 0)

    def test_compact_candidate_group_keeps_direction_and_origin_stats(self) -> None:
        bitmap = np.zeros((10, 10), dtype=bool)
        bitmap[3:7, 3:7] = True
        runner = self._make_runner()
        cand_a = _candidate("cand_a", bitmap, origin_exact_cluster_id=0, shift_direction="base")
        cand_b = _candidate("cand_b", bitmap.copy(), origin_exact_cluster_id=1, shift_direction="left")

        groups = _coverage_groups(runner, [cand_a, cand_b])

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].logical_candidate_count, 2)
        self.assertEqual(set(int(value) for value in groups[0].origin_ids.tolist()), {0, 1})
        self.assertEqual(groups[0].direction_counts["base"], 1)
        self.assertEqual(groups[0].direction_counts["left"], 1)

    def test_base_candidates_survive_group_merge_for_singleton_fallback(self) -> None:
        bitmap = np.zeros((10, 10), dtype=bool)
        bitmap[3:7, 3:7] = True
        runner = self._make_runner()
        exact_a = ExactCluster(0, _record("a0", bitmap, seed_weight=1), [_record("a0", bitmap, seed_weight=1)])
        exact_b = ExactCluster(1, _record("b0", bitmap, seed_weight=1), [_record("b0", bitmap, seed_weight=1)])

        groups, candidate_count, _ = runner._build_global_coverage_candidate_groups([exact_a, exact_b])

        self.assertGreater(candidate_count, 0)
        self.assertEqual(len(groups), 1)
        self.assertIn(0, runner._base_candidate_by_exact_id)
        self.assertIn(1, runner._base_candidate_by_exact_id)
        self.assertEqual(runner._base_candidate_by_exact_id[0].shift_direction, "base")
        self.assertEqual(runner._base_candidate_by_exact_id[1].shift_direction, "base")

    def test_greedy_cover_prefers_base_candidate_when_coverage_same(self) -> None:
        bitmap = np.zeros((10, 10), dtype=bool)
        bitmap[3:7, 3:7] = True
        rec_a = _record("a0", bitmap, seed_weight=1)
        rec_b = _record("b0", bitmap, seed_weight=1)
        rec_a.match_cache["auto_seed"] = {"seed_type": optimized.SEED_TYPE_ARRAY}
        rec_b.match_cache["auto_seed"] = {"seed_type": optimized.SEED_TYPE_ARRAY}
        exact_a = ExactCluster(0, rec_a, [rec_a])
        exact_b = ExactCluster(1, rec_b, [rec_b])
        runner = self._make_runner()

        base_candidate = _candidate(
            "cand_base",
            bitmap,
            origin_exact_cluster_id=0,
            shift_direction="base",
            coverage=(0, 1),
        )
        shift_candidate = _candidate(
            "cand_shift",
            bitmap,
            origin_exact_cluster_id=1,
            shift_direction="right",
            coverage=(0, 1),
        )

        selected = runner._greedy_cover(_coverage_groups(runner, [shift_candidate, base_candidate]), [exact_a, exact_b])

        self.assertEqual([candidate.candidate_id for candidate in selected], ["cand_base"])

    def test_greedy_cover_uses_0429_tiebreak_without_seed_family(self) -> None:
        bitmap_candidate = np.zeros((10, 10), dtype=bool)
        bitmap_candidate[2:5, 2:5] = True
        bitmap_target = np.zeros((10, 10), dtype=bool)
        bitmap_target[5:8, 5:8] = True
        rec_array = _record("array0", bitmap_target, seed_weight=1)
        rec_long = _record("long0", bitmap_target, seed_weight=1)
        rec_target = _record("target0", bitmap_target, seed_weight=1)
        rec_array.match_cache["auto_seed"] = {"seed_type": optimized.SEED_TYPE_ARRAY}
        rec_long.match_cache["auto_seed"] = {"seed_type": optimized.SEED_TYPE_LONG}
        rec_target.match_cache["auto_seed"] = {"seed_type": optimized.SEED_TYPE_LONG}
        exact_clusters = [
            ExactCluster(0, rec_array, [rec_array]),
            ExactCluster(1, rec_long, [rec_long]),
            ExactCluster(2, rec_target, [rec_target]),
        ]
        runner = self._make_runner()

        low_distance_diag = _candidate(
            "cand_low_diag",
            bitmap_candidate,
            origin_exact_cluster_id=0,
            shift_direction="diag_ne",
            coverage=(0, 1, 2),
        )
        low_distance_diag.shift_distance_um = 0.01
        same_family_axis = _candidate(
            "cand_same_family_axis",
            bitmap_candidate,
            origin_exact_cluster_id=1,
            shift_direction="right",
            coverage=(0, 1, 2),
        )
        same_family_axis.shift_distance_um = 0.08

        selected = runner._greedy_cover(_coverage_groups(runner, [same_family_axis, low_distance_diag]), exact_clusters)

        self.assertEqual([candidate.candidate_id for candidate in selected], ["cand_low_diag"])

    def test_greedy_cover_prefers_higher_opc_center_score_before_shift_distance(self) -> None:
        good_bitmap = np.zeros((10, 10), dtype=bool)
        good_bitmap[3:7, 3:7] = True
        bad_bitmap = np.zeros((10, 10), dtype=bool)
        bad_bitmap[0:3, 0:3] = True
        rec_a = _record("score_a", good_bitmap, seed_weight=1)
        rec_b = _record("score_b", good_bitmap, seed_weight=1)
        exact_clusters = [
            ExactCluster(0, rec_a, [rec_a]),
            ExactCluster(1, rec_b, [rec_b]),
        ]
        runner = self._make_runner()
        good_candidate = _candidate(
            "cand_good_center",
            good_bitmap,
            origin_exact_cluster_id=0,
            shift_direction="right",
            coverage=(0, 1),
        )
        good_candidate.shift_distance_um = 0.08
        bad_candidate = _candidate(
            "cand_bad_center",
            bad_bitmap,
            origin_exact_cluster_id=1,
            shift_direction="right",
            coverage=(0, 1),
        )
        bad_candidate.shift_distance_um = 0.01

        selected = runner._greedy_cover(_coverage_groups(runner, [bad_candidate, good_candidate]), exact_clusters)

        self.assertEqual([candidate.candidate_id for candidate in selected], ["cand_good_center"])

    def test_greedy_cover_prefers_higher_confidence_proxy_before_origin_id(self) -> None:
        bitmap_exact = np.zeros((10, 10), dtype=bool)
        bitmap_exact[2:6, 2:6] = True
        bitmap_relaxed = np.zeros((10, 10), dtype=bool)
        bitmap_relaxed[3:7, 3:7] = True
        rec_a = _record("conf_a", bitmap_exact, seed_weight=1)
        rec_b = _record("conf_b", bitmap_exact, seed_weight=1)
        exact_clusters = [
            ExactCluster(0, rec_a, [rec_a]),
            ExactCluster(1, rec_b, [rec_b]),
        ]
        runner = self._make_runner()
        exact_hash_candidate = _candidate(
            "cand_exact_hash",
            bitmap_exact,
            origin_exact_cluster_id=1,
            shift_direction="right",
            coverage=(0, 1),
        )
        exact_hash_candidate.shift_distance_um = 0.02
        relaxed_candidate = _candidate(
            "cand_relaxed_hash",
            bitmap_relaxed,
            origin_exact_cluster_id=1,
            shift_direction="right",
            coverage=(0, 1),
        )
        relaxed_candidate.shift_distance_um = 0.02

        with mock.patch.object(optimized, "_candidate_representative_score", return_value=0.5):
            selected = runner._greedy_cover(_coverage_groups(runner, [relaxed_candidate, exact_hash_candidate]), exact_clusters)

        self.assertEqual([candidate.candidate_id for candidate in selected], ["cand_exact_hash"])

    def test_assign_exact_clusters_uses_0429_tiebreak_without_seed_family(self) -> None:
        bitmap_candidate = np.zeros((10, 10), dtype=bool)
        bitmap_candidate[2:5, 2:5] = True
        bitmap_target = np.zeros((10, 10), dtype=bool)
        bitmap_target[5:8, 5:8] = True
        rec_array = _record("array0", bitmap_target, seed_weight=1)
        rec_long = _record("long0", bitmap_target, seed_weight=1)
        rec_target = _record("target0", bitmap_target, seed_weight=1)
        rec_array.match_cache["auto_seed"] = {"seed_type": optimized.SEED_TYPE_ARRAY}
        rec_long.match_cache["auto_seed"] = {"seed_type": optimized.SEED_TYPE_LONG}
        rec_target.match_cache["auto_seed"] = {"seed_type": optimized.SEED_TYPE_LONG}
        exact_array = ExactCluster(0, rec_array, [rec_array])
        exact_long = ExactCluster(1, rec_long, [rec_long])
        exact_target = ExactCluster(2, rec_target, [rec_target])
        runner = self._make_runner()

        low_distance_diag = _candidate(
            "cand_low_diag",
            bitmap_candidate,
            origin_exact_cluster_id=0,
            shift_direction="diag_ne",
            coverage=(0, 2),
        )
        low_distance_diag.shift_distance_um = 0.01
        same_family_axis = _candidate(
            "cand_same_family_axis",
            bitmap_candidate,
            origin_exact_cluster_id=1,
            shift_direction="right",
            coverage=(1, 2),
        )
        same_family_axis.shift_distance_um = 0.08

        assignments = runner._assign_exact_clusters(
            [same_family_axis, low_distance_diag],
            [exact_array, exact_long, exact_target],
        )

        assigned_to_low = [int(cluster.exact_cluster_id) for cluster in assignments["cand_low_diag"]]
        assigned_to_same_family = [int(cluster.exact_cluster_id) for cluster in assignments["cand_same_family_axis"]]
        self.assertIn(2, assigned_to_low)
        self.assertNotIn(2, assigned_to_same_family)

    def test_final_verification_breakdown_tracks_reject_reason_and_seed_type(self) -> None:
        bitmap = np.zeros((10, 10), dtype=bool)
        bitmap[3:7, 3:7] = True
        rec_a = _record("a0", bitmap, seed_weight=1)
        rec_b = _record("b0", bitmap, seed_weight=1)
        rec_a.match_cache["auto_seed"] = {"seed_type": optimized.SEED_TYPE_ARRAY}
        rec_b.match_cache["auto_seed"] = {"seed_type": optimized.SEED_TYPE_LONG}
        exact_a = ExactCluster(0, rec_a, [rec_a])
        exact_b = ExactCluster(1, rec_b, [rec_b])
        runner = self._make_runner()

        selected = _candidate(
            "cand_diag",
            bitmap,
            origin_exact_cluster_id=0,
            shift_direction="diag_ne",
            coverage=(0, 1),
        )
        runner._base_candidate_by_exact_id = {
            0: _candidate("cand_base_a", bitmap, origin_exact_cluster_id=0, shift_direction="base"),
            1: _candidate("cand_base_b", bitmap, origin_exact_cluster_id=1, shift_direction="base"),
        }

        def fake_match(candidate: CandidateClip, exact_cluster: ExactCluster, *, strict: bool) -> tuple[bool, str, str]:
            del candidate, strict
            if int(exact_cluster.exact_cluster_id) == 0:
                return False, "geometry_acc_xor", "none"
            return False, "graph_topology", "none"

        runner._candidate_matches_exact = fake_match  # type: ignore[method-assign]
        units = runner._verified_cluster_units([selected], [exact_a, exact_b])

        self.assertEqual(len(units), 2)
        breakdown = runner.final_verification_breakdown
        self.assertEqual(breakdown["reject_reason_counts"]["geometry_acc_xor"], 1)
        self.assertNotIn("geometry", breakdown["reject_reason_counts"])
        self.assertEqual(breakdown["reject_reason_counts"]["graph_topology"], 1)
        self.assertEqual(breakdown["reject_shift_direction_counts"]["diag_ne"], 2)
        self.assertEqual(breakdown["reject_origin_seed_type_counts"][optimized.SEED_TYPE_ARRAY], 2)
        self.assertEqual(breakdown["reject_target_seed_type_counts"][optimized.SEED_TYPE_ARRAY], 1)
        self.assertEqual(breakdown["reject_target_seed_type_counts"][optimized.SEED_TYPE_LONG], 1)

    def test_quality_metrics_samples_each_cluster_independently(self) -> None:
        """质量指标应按 cluster 分层采样，不让第一个失败簇吃掉全部预算。"""

        bitmap = np.zeros((8, 8), dtype=bool)
        bitmap[2:6, 2:6] = True
        exact_clusters = [
            ExactCluster(idx, _record(f"m{idx}", bitmap, seed_weight=1), [_record(f"m{idx}", bitmap, seed_weight=1)])
            for idx in range(5)
        ]
        runner = self._make_runner(compute_quality_metrics=True)
        runner._base_candidate_by_exact_id = {
            idx: _candidate(f"base_{idx}", bitmap, origin_exact_cluster_id=idx, shift_direction="base")
            for idx in range(5)
        }

        called_candidate_ids: list[str] = []
        strict_call_count = 0

        def fake_match(candidate: CandidateClip, exact_cluster: ExactCluster, *, strict: bool) -> tuple[bool, str, str]:
            nonlocal strict_call_count
            called_candidate_ids.append(str(candidate.candidate_id))
            if strict:
                strict_call_count += 1
            return True, "unit_test", "base"

        runner._candidate_matches_exact = fake_match  # type: ignore[method-assign]
        cluster_units = [
            (_candidate("cluster_bad", bitmap, origin_exact_cluster_id=0, shift_direction="base"), exact_clusters[0:2]),
            (_candidate("cluster_good", bitmap, origin_exact_cluster_id=2, shift_direction="base"), exact_clusters[2:4]),
            (_candidate("cluster_single", bitmap, origin_exact_cluster_id=4, shift_direction="base"), exact_clusters[4:5]),
        ]
        with mock.patch.object(optimized, "QUALITY_PER_CLUSTER_INTRA_PAIR_LIMIT", 1):
            metrics = runner._build_quality_metrics(
                cluster_units,
                exact_clusters,
                [],
                [candidate for candidate, _ in cluster_units],
            )

        self.assertAlmostEqual(metrics["representative_visual_pass_ratio"], 1.0)
        self.assertAlmostEqual(metrics["representative_visual_weighted_pass_ratio"], 1.0)
        self.assertAlmostEqual(metrics["representative_visual_reject_weight_ratio"], 0.0)
        self.assertEqual(metrics["pairwise_geometry_sampled_cluster_count"], 2)
        self.assertEqual(metrics["pairwise_geometry_no_pair_cluster_count"], 1)
        self.assertEqual(metrics["pairwise_geometry_sampled_pair_count"], 2)
        self.assertAlmostEqual(metrics["visual_purity_score"], 1.0)
        self.assertAlmostEqual(metrics["pairwise_geometry_purity"], 1.0)
        self.assertEqual(metrics["cluster_quality_by_index"][0]["pairwise_geometry_sample_status"], "sampled")
        self.assertEqual(metrics["cluster_quality_by_index"][2]["pairwise_geometry_sample_status"], "no_pair")
        self.assertAlmostEqual(metrics["raw_coverage_graph_recall"], 1.0)
        self.assertEqual(strict_call_count, 0)
        self.assertTrue(any(candidate_id.startswith("base_") for candidate_id in called_candidate_ids))
        self.assertTrue(any(candidate_id.startswith("cluster_") for candidate_id in called_candidate_ids))

    def test_raw_coverage_graph_recall_dedupes_edges(self) -> None:
        """raw coverage graph recall 应复用已有 coverage 边并去重重复边。"""

        bitmap = np.zeros((8, 8), dtype=bool)
        bitmap[2:6, 2:6] = True
        duplicate_bitmap = np.zeros((8, 8), dtype=bool)
        duplicate_bitmap[1:5, 1:5] = True
        exact_clusters = [
            ExactCluster(idx, _record(f"s{idx}", bitmap, seed_weight=1), [_record(f"s{idx}", bitmap, seed_weight=1)])
            for idx in range(3)
        ]
        runner = self._make_runner(compute_quality_metrics=True)
        runner._base_candidate_by_exact_id = {
            idx: _candidate(f"base_single_{idx}", bitmap, origin_exact_cluster_id=idx, shift_direction="base")
            for idx in range(3)
        }

        def fake_match(candidate: CandidateClip, exact_cluster: ExactCluster, *, strict: bool) -> tuple[bool, str, str]:
            del candidate, exact_cluster, strict
            return True, "unit_test_visual", "base"

        runner._candidate_matches_exact = fake_match  # type: ignore[method-assign]
        cluster_units = [
            (_candidate("cluster_merged", bitmap, origin_exact_cluster_id=0, shift_direction="base"), exact_clusters[0:2]),
            (_candidate("cluster_single", bitmap, origin_exact_cluster_id=2, shift_direction="base"), exact_clusters[2:3]),
        ]
        edge_candidate = _candidate("edge_candidate", bitmap, origin_exact_cluster_id=0, shift_direction="east", coverage={0, 1, 2})
        duplicate_edge = _candidate("duplicate_edge", duplicate_bitmap, origin_exact_cluster_id=0, shift_direction="north", coverage={2})
        candidate_groups = _coverage_groups(runner, [edge_candidate, duplicate_edge])

        metrics = runner._build_quality_metrics(
            cluster_units,
            exact_clusters,
            candidate_groups,
            [candidate for candidate, _ in cluster_units],
        )

        self.assertEqual(metrics["raw_coverage_graph_edge_count"], 2)
        self.assertEqual(metrics["raw_coverage_graph_cross_cluster_edge_count"], 1)
        self.assertAlmostEqual(metrics["raw_coverage_graph_cross_cluster_edge_weight_ratio"], 0.5)
        self.assertAlmostEqual(metrics["raw_coverage_graph_recall"], 0.5)
        self.assertEqual(metrics["review_merge_candidate_edge_count"], 1)
        self.assertAlmostEqual(metrics["review_merge_candidate_weight_ratio"], 0.5)
        self.assertAlmostEqual(metrics["singleton_trusted_mergeable_weight_ratio"], 1.0)
        self.assertAlmostEqual(metrics["trusted_fragmentation_recall"], 1.0)
        for old_key in (
            "fragmentation_sampled_pair_count",
            "fragmentation_mergeable_pair_count",
            "fragmentation_recall_proxy",
            "singleton_mergeable_ratio",
            "coverage_graph_fragmentation_recall",
        ):
            self.assertNotIn(old_key, metrics)

    def test_actionable_overmerge_review_counts_only_low_shift_low_pairwise(self) -> None:
        """actionable over-merge review 只统计低 shift 且低 pairwise 的 cluster。"""

        bitmap = np.zeros((8, 8), dtype=bool)
        bitmap[2:6, 2:6] = True
        exact_clusters = [
            ExactCluster(idx, _record(f"act{idx}", bitmap, seed_weight=1), [_record(f"act{idx}", bitmap, seed_weight=1)])
            for idx in range(6)
        ]
        runner = self._make_runner(compute_quality_metrics=True)
        runner._base_candidate_by_exact_id = {
            idx: _candidate(f"base_act_{idx}", bitmap, origin_exact_cluster_id=idx, shift_direction="base")
            for idx in range(6)
        }

        def fake_match(candidate: CandidateClip, exact_cluster: ExactCluster, *, strict: bool) -> tuple[bool, str, str]:
            del candidate, exact_cluster, strict
            return True, "unit_test_visual", "base"

        def fake_pair(
            left_exact_id: int,
            right_exact_id: int,
            exact_by_id: dict[int, ExactCluster],
            *,
            strict: bool,
        ) -> bool | None:
            del exact_by_id, strict
            pair = frozenset({int(left_exact_id), int(right_exact_id)})
            if pair in {frozenset({0, 1}), frozenset({2, 3})}:
                return False
            return True

        runner._candidate_matches_exact = fake_match  # type: ignore[method-assign]
        runner._exact_pair_matches = fake_pair  # type: ignore[method-assign]
        low_shift_low_pairwise = _candidate(
            "low_shift_low_pairwise",
            bitmap,
            origin_exact_cluster_id=0,
            shift_direction="base",
        )
        high_shift_low_pairwise = _candidate(
            "high_shift_low_pairwise",
            bitmap,
            origin_exact_cluster_id=2,
            shift_direction="east",
        )
        high_shift_low_pairwise.shift_distance_um = 0.25
        low_shift_high_pairwise = _candidate(
            "low_shift_high_pairwise",
            bitmap,
            origin_exact_cluster_id=4,
            shift_direction="base",
        )
        cluster_units = [
            (low_shift_low_pairwise, exact_clusters[0:2]),
            (high_shift_low_pairwise, exact_clusters[2:4]),
            (low_shift_high_pairwise, exact_clusters[4:6]),
        ]

        metrics = runner._build_quality_metrics(
            cluster_units,
            exact_clusters,
            [],
            [candidate for candidate, _ in cluster_units],
        )

        self.assertEqual(metrics["low_shift_low_pairwise_review_cluster_count"], 1)
        self.assertAlmostEqual(metrics["low_shift_low_pairwise_review_weight_ratio"], 2.0 / 6.0)
        self.assertAlmostEqual(metrics["pairwise_review_sampled_weight_ratio"], 1.0)

    def test_fragmentation_metrics_split_trusted_rejected_and_review_edges(self) -> None:
        """fragmentation 新口径应区分 verified、gate rejected 与待 review 的 coverage 边。"""

        bitmaps: list[np.ndarray] = []
        for idx in range(4):
            bitmap = np.zeros((8, 8), dtype=bool)
            bitmap[1 + idx : 3 + idx, 1:3] = True
            bitmaps.append(bitmap)
        exact_clusters = [
            ExactCluster(idx, _record(f"frag{idx}", bitmaps[idx], seed_weight=1), [_record(f"frag{idx}", bitmaps[idx], seed_weight=1)])
            for idx in range(4)
        ]
        runner = self._make_runner(compute_quality_metrics=True)
        runner._base_candidate_by_exact_id = {
            idx: _candidate(f"base_frag_{idx}", bitmaps[idx], origin_exact_cluster_id=idx, shift_direction="base")
            for idx in range(4)
        }

        def fake_match(candidate: CandidateClip, exact_cluster: ExactCluster, *, strict: bool) -> tuple[bool, str, str]:
            del candidate, exact_cluster, strict
            return True, "unit_test_visual", "base"

        runner._candidate_matches_exact = fake_match  # type: ignore[method-assign]
        trusted_selected = _candidate(
            "trusted_selected",
            bitmaps[0],
            origin_exact_cluster_id=0,
            shift_direction="base",
            coverage={0, 1},
        )
        rejected_bridge = _candidate(
            "rejected_bridge",
            bitmaps[1],
            origin_exact_cluster_id=0,
            shift_direction="east",
            coverage={0, 2},
        )
        review_bridge = _candidate(
            "review_bridge",
            bitmaps[2],
            origin_exact_cluster_id=0,
            shift_direction="west",
            coverage={0, 3},
        )
        runner._greedy_rejected_candidate_ids = {"rejected_bridge"}
        cluster_units = [
            (trusted_selected, exact_clusters[0:2]),
            (runner._base_candidate_by_exact_id[2], exact_clusters[2:3]),
            (runner._base_candidate_by_exact_id[3], exact_clusters[3:4]),
        ]

        metrics = runner._build_quality_metrics(
            cluster_units,
            exact_clusters,
            _coverage_groups(runner, [trusted_selected, rejected_bridge, review_bridge]),
            [trusted_selected, runner._base_candidate_by_exact_id[2], runner._base_candidate_by_exact_id[3]],
        )

        self.assertEqual(metrics["raw_coverage_graph_edge_count"], 3)
        self.assertEqual(metrics["raw_coverage_graph_cross_cluster_edge_count"], 2)
        self.assertAlmostEqual(metrics["raw_coverage_graph_recall"], 1.0 / 3.0)
        self.assertEqual(metrics["trusted_fragmentation_edge_count"], 1)
        self.assertEqual(metrics["trusted_fragmentation_cross_edge_count"], 0)
        self.assertAlmostEqual(metrics["trusted_fragmentation_recall"], 1.0)
        self.assertEqual(metrics["gate_rejected_edge_count"], 1)
        self.assertAlmostEqual(metrics["gate_rejected_edge_weight_ratio"], 1.0 / 3.0)
        self.assertEqual(metrics["review_merge_candidate_edge_count"], 1)
        self.assertAlmostEqual(metrics["review_merge_candidate_weight_ratio"], 1.0 / 3.0)
        self.assertEqual(metrics["singleton_trusted_mergeable_edge_count"], 1)
        self.assertAlmostEqual(metrics["singleton_trusted_mergeable_weight_ratio"], 0.5)

    def test_review_merge_candidates_are_tiered_for_safe_merge(self) -> None:
        """review merge 候选应分 high/medium/low，并且只作为 safe merge 内部证据。"""

        def bitmap_for(idx: int) -> np.ndarray:
            bitmap = np.zeros((8, 8), dtype=bool)
            x0 = 1 + (idx % 3)
            y0 = 1 + (idx // 3)
            bitmap[y0 : y0 + 2, x0 : x0 + 2] = True
            return bitmap

        seed_types = [
            optimized.SEED_TYPE_RESIDUAL,
            optimized.SEED_TYPE_RESIDUAL,
            optimized.SEED_TYPE_ARRAY,
            optimized.SEED_TYPE_RESIDUAL,
            optimized.SEED_TYPE_RESIDUAL,
            optimized.SEED_TYPE_RESIDUAL,
            optimized.SEED_TYPE_RESIDUAL,
        ]
        bitmaps = [bitmap_for(idx) for idx in range(len(seed_types))]
        exact_clusters = []
        for idx, seed_type in enumerate(seed_types):
            record = _record(f"review{idx}", bitmaps[idx], seed_weight=1, seed_type=seed_type)
            exact_clusters.append(ExactCluster(idx, record, [record]))

        runner = self._make_runner(compute_quality_metrics=True)
        runner._base_candidate_by_exact_id = {
            idx: _candidate(f"base_review_{idx}", bitmaps[idx], origin_exact_cluster_id=idx, shift_direction="base")
            for idx in range(len(exact_clusters))
        }

        def fake_match(candidate: CandidateClip, exact_cluster: ExactCluster, *, strict: bool) -> tuple[bool, str, str]:
            if not strict:
                return True, "unit_test_visual", "base"
            if int(candidate.origin_exact_cluster_id) == 0 and int(exact_cluster.exact_cluster_id) == 1:
                return True, "unit_test_strict", "base"
            return False, "unit_test_reject", "none"

        runner._candidate_matches_exact = fake_match  # type: ignore[method-assign]
        selected = _candidate("selected_same_cluster", bitmaps[0], origin_exact_cluster_id=0, shift_direction="base", coverage={0, 4})
        high = _candidate("high_review", bitmaps[1], origin_exact_cluster_id=0, shift_direction="east", coverage={0, 1})
        medium = _candidate("medium_review", bitmaps[2], origin_exact_cluster_id=0, shift_direction="north", coverage={0, 3})
        low_seed = _candidate("low_seed_review", bitmaps[3], origin_exact_cluster_id=0, shift_direction="west", coverage={0, 2})
        low_shift = _candidate("low_shift_review", bitmaps[4], origin_exact_cluster_id=0, shift_direction="south", coverage={0, 5})
        rejected = _candidate("rejected_review", bitmaps[5], origin_exact_cluster_id=0, shift_direction="diag_ne", coverage={0, 6})
        high.shift_distance_um = 0.05
        medium.shift_distance_um = 0.15
        low_seed.shift_distance_um = 0.05
        low_shift.shift_distance_um = 0.25
        rejected.shift_distance_um = 0.05
        runner._greedy_rejected_candidate_ids = {"rejected_review"}

        cluster_units = [
            (selected, [exact_clusters[0], exact_clusters[4]]),
            (runner._base_candidate_by_exact_id[1], [exact_clusters[1]]),
            (runner._base_candidate_by_exact_id[2], [exact_clusters[2]]),
            (runner._base_candidate_by_exact_id[3], [exact_clusters[3]]),
            (runner._base_candidate_by_exact_id[5], [exact_clusters[5]]),
            (runner._base_candidate_by_exact_id[6], [exact_clusters[6]]),
        ]

        metrics = runner._build_quality_metrics(
            cluster_units,
            exact_clusters,
            _coverage_groups(runner, [selected, high, medium, low_seed, low_shift, rejected]),
            [candidate for candidate, _ in cluster_units],
        )

        self.assertEqual(metrics["review_merge_candidate_edge_count"], 4)
        self.assertEqual(metrics["gate_rejected_edge_count"], 1)
        self.assertAlmostEqual(metrics["review_merge_candidate_weight_ratio"], 4.0 / 6.0)
        self.assertAlmostEqual(metrics["high_conf_review_merge_weight_ratio"], 1.0 / 6.0)
        self.assertAlmostEqual(metrics["medium_conf_review_merge_weight_ratio"], 1.0 / 6.0)
        self.assertAlmostEqual(metrics["low_conf_review_merge_weight_ratio"], 2.0 / 6.0)
        self.assertAlmostEqual(metrics["high_conf_singleton_mergeable_weight_ratio"], 1.0 / 6.0)
        rows = metrics["review_merge_candidate_rows"]
        self.assertEqual([row["confidence_tier"] for row in rows], ["high", "medium", "low", "low"])
        self.assertNotIn("selected_same_cluster", {row["candidate_id"] for row in rows})
        self.assertNotIn("rejected_review", {row["candidate_id"] for row in rows})
        self.assertEqual(rows[0]["source_cluster_id"], 1)
        self.assertEqual(rows[0]["target_cluster_id"], 2)
        self.assertTrue(rows[0]["target_is_singleton"])
        pair_rows = metrics["review_merge_cluster_pair_rows"]
        self.assertEqual(len(pair_rows), 4)
        self.assertEqual(pair_rows[0]["source_cluster_id"], 1)
        self.assertEqual(pair_rows[0]["target_cluster_id"], 2)
        self.assertEqual(pair_rows[0]["pair_edge_weight_sum"], 1)
        self.assertEqual(pair_rows[0]["pair_review_bucket"], "safe_recall_candidate")
        self.assertEqual([len(assigned) for _, assigned in cluster_units], [2, 1, 1, 1, 1, 1])

    def test_review_merge_candidate_rows_are_bounded_for_low_memory(self) -> None:
        """review merge 候选行应按 tier 配额截断，避免 high tier 独占内部证据。"""

        rows: list[dict[str, object]] = []
        with (
            mock.patch.object(optimized, "REVIEW_MERGE_CANDIDATE_TOP_N", 6),
            mock.patch.object(optimized, "REVIEW_MERGE_CANDIDATE_TIER_QUOTAS", {"high": 2, "medium": 2, "low": 2}),
        ):
            for idx in range(10):
                tier = ("high", "medium", "low")[idx % 3]
                optimized._append_bounded_review_merge_row(
                    rows,
                    {
                        "source_exact_cluster_id": idx,
                        "target_exact_cluster_id": idx + 100,
                        "source_cluster_id": idx,
                        "target_cluster_id": idx + 100,
                        "candidate_id": f"review_{idx}",
                        "edge_weight": idx + 1,
                        "candidate_seed_type": optimized.SEED_TYPE_RESIDUAL,
                        "target_seed_type": optimized.SEED_TYPE_RESIDUAL,
                        "shift_direction": "east",
                        "shift_distance_um": 0.01,
                        "confidence_tier": tier,
                        "confidence_reason": "same_seed_small_shift",
                        "target_is_singleton": False,
                        "source_cluster_exact_count": 1,
                        "target_cluster_exact_count": 1,
                    },
                )
                self.assertLessEqual(len(rows), 12)

            top_rows = optimized._tier_balanced_review_merge_candidate_rows(rows)

        self.assertEqual(len(top_rows), 6)
        self.assertEqual(Counter(str(row["confidence_tier"]) for row in top_rows), Counter({"high": 2, "medium": 2, "low": 2}))
        self.assertEqual([int(row["edge_weight"]) for row in top_rows if row["confidence_tier"] == "high"], [10, 7])
        self.assertEqual([int(row["edge_weight"]) for row in top_rows if row["confidence_tier"] == "medium"], [8, 5])
        self.assertEqual([int(row["edge_weight"]) for row in top_rows if row["confidence_tier"] == "low"], [9, 6])

    def test_review_merge_cluster_pair_rows_bucket_endpoint_quality(self) -> None:
        """review merge pair 应聚合重复行，并按端点质量输出 review bucket。"""

        review_rows = [
            {
                "source_cluster_id": 1,
                "target_cluster_id": 2,
                "candidate_id": "safe_a",
                "edge_weight": 10,
                "confidence_tier": "high",
                "target_is_singleton": True,
            },
            {
                "source_cluster_id": 2,
                "target_cluster_id": 1,
                "candidate_id": "safe_b",
                "edge_weight": 5,
                "confidence_tier": "high",
                "target_is_singleton": False,
            },
            {
                "source_cluster_id": 1,
                "target_cluster_id": 3,
                "candidate_id": "overmerge",
                "edge_weight": 7,
                "confidence_tier": "high",
                "target_is_singleton": False,
            },
            {
                "source_cluster_id": 1,
                "target_cluster_id": 4,
                "candidate_id": "high_shift",
                "edge_weight": 6,
                "confidence_tier": "high",
                "target_is_singleton": False,
            },
            {
                "source_cluster_id": 1,
                "target_cluster_id": 5,
                "candidate_id": "medium_only",
                "edge_weight": 4,
                "confidence_tier": "medium",
                "target_is_singleton": False,
            },
        ]
        endpoint_quality = {
            1: {
                "cluster_weight": 100,
                "exact_cluster_count": 4,
                "shift_distance_um": 0.0,
                "pairwise_geometry_purity": 0.95,
                "pairwise_geometry_sample_status": "sampled",
                "pairwise_geometry_sampled_pair_count": 3,
                "overmerge_reason": "ok",
            },
            2: {
                "cluster_weight": 20,
                "exact_cluster_count": 1,
                "shift_distance_um": 0.0,
                "pairwise_geometry_purity": None,
                "pairwise_geometry_sample_status": "no_pair",
                "pairwise_geometry_sampled_pair_count": 0,
                "overmerge_reason": "ok",
            },
            3: {
                "cluster_weight": 30,
                "exact_cluster_count": 3,
                "shift_distance_um": 0.0,
                "pairwise_geometry_purity": 0.30,
                "pairwise_geometry_sample_status": "sampled",
                "pairwise_geometry_sampled_pair_count": 2,
                "overmerge_reason": "low_pairwise_geometry_purity",
            },
            4: {
                "cluster_weight": 40,
                "exact_cluster_count": 2,
                "shift_distance_um": 0.25,
                "pairwise_geometry_purity": 0.90,
                "pairwise_geometry_sample_status": "sampled",
                "pairwise_geometry_sampled_pair_count": 1,
                "overmerge_reason": "ok",
            },
            5: {
                "cluster_weight": 50,
                "exact_cluster_count": 1,
                "shift_distance_um": 0.0,
                "pairwise_geometry_purity": None,
                "pairwise_geometry_sample_status": "no_pair",
                "pairwise_geometry_sampled_pair_count": 0,
                "overmerge_reason": "ok",
            },
        }

        pair_rows = optimized._review_merge_cluster_pair_rows(review_rows, endpoint_quality)
        pair_by_target = {int(row["target_cluster_id"]): row for row in pair_rows}

        self.assertEqual(pair_by_target[2]["pair_edge_weight_sum"], 15)
        self.assertEqual(pair_by_target[2]["row_count"], 2)
        self.assertEqual(pair_by_target[2]["unique_candidate_count"], 2)
        self.assertEqual(pair_by_target[2]["singleton_target_row_count"], 1)
        self.assertEqual(pair_by_target[2]["pair_review_bucket"], "safe_recall_candidate")
        self.assertEqual(pair_by_target[3]["pair_review_bucket"], "overmerge_touching_candidate")
        self.assertEqual(pair_by_target[4]["pair_review_bucket"], "high_shift_touching_candidate")
        self.assertEqual(pair_by_target[5]["pair_review_bucket"], "low_confidence_candidate")

    def test_safe_recall_merge_accepts_full_union_verified_pair(self) -> None:
        """safe recall merge 只在 candidate 覆盖并验证完整 union 时合并。"""

        bitmap = np.zeros((8, 8), dtype=bool)
        bitmap[2:6, 2:6] = True
        exact_clusters = [
            ExactCluster(idx, _record(f"safe{idx}", bitmap, seed_weight=1), [_record(f"safe{idx}", bitmap, seed_weight=1)])
            for idx in range(2)
        ]
        runner = self._make_runner(compute_quality_metrics=True)
        base0 = _candidate("base_safe_0", bitmap, origin_exact_cluster_id=0, shift_direction="base")
        base1 = _candidate("base_safe_1", bitmap, origin_exact_cluster_id=1, shift_direction="base")
        merge_candidate = _candidate("safe_merge", bitmap, origin_exact_cluster_id=0, shift_direction="east", coverage={0, 1})
        runner._base_candidate_by_exact_id = {0: base0, 1: base1}

        def fake_cached(candidate: CandidateClip, exact_cluster: ExactCluster, *, strict: bool) -> tuple[bool, str, str]:
            del exact_cluster
            return (str(candidate.candidate_id) == "safe_merge" and bool(strict), "unit_test", "base")

        runner._cached_candidate_matches_exact_result = fake_cached  # type: ignore[method-assign]
        quality_metrics = {
            "review_merge_candidate_rows": [
                {
                    "source_cluster_id": 1,
                    "target_cluster_id": 2,
                    "candidate_id": "safe_merge",
                    "confidence_tier": "high",
                    "target_exact_cluster_id": 1,
                    "edge_weight": 10,
                }
            ],
            "review_merge_cluster_pair_rows": [
                {
                    "source_cluster_id": 1,
                    "target_cluster_id": 2,
                    "pair_review_bucket": "safe_recall_candidate",
                    "source_overmerge_reason": "ok",
                    "target_overmerge_reason": "ok",
                }
            ],
        }

        merged_units, metrics = runner._apply_safe_recall_merge(
            [(base0, [exact_clusters[0]]), (base1, [exact_clusters[1]])],
            quality_metrics,
            _coverage_groups(runner, [merge_candidate]),
        )

        self.assertEqual(len(merged_units), 1)
        self.assertEqual(str(merged_units[0][0].candidate_id), "safe_merge")
        self.assertEqual([int(cluster.exact_cluster_id) for cluster in merged_units[0][1]], [0, 1])
        self.assertEqual(metrics["safe_recall_merge_attempted_pair_count"], 1)
        self.assertEqual(metrics["safe_recall_merge_merged_pair_count"], 1)
        self.assertEqual(metrics["safe_recall_merge_cluster_reduction"], 1)

    def test_safe_recall_merge_rejects_partial_coverage_and_risky_endpoint(self) -> None:
        """coverage 不完整或端点有 overmerge reason 时不做 safe merge。"""

        bitmap = np.zeros((8, 8), dtype=bool)
        bitmap[2:6, 2:6] = True
        exact_clusters = [
            ExactCluster(idx, _record(f"reject{idx}", bitmap, seed_weight=1), [_record(f"reject{idx}", bitmap, seed_weight=1)])
            for idx in range(2)
        ]
        runner = self._make_runner(compute_quality_metrics=True)
        base0 = _candidate("base_reject_0", bitmap, origin_exact_cluster_id=0, shift_direction="base")
        base1 = _candidate("base_reject_1", bitmap, origin_exact_cluster_id=1, shift_direction="base")
        partial_candidate = _candidate("partial_merge", bitmap, origin_exact_cluster_id=0, shift_direction="east", coverage={0})
        runner._base_candidate_by_exact_id = {0: base0, 1: base1}
        runner._cached_candidate_matches_exact_result = (  # type: ignore[method-assign]
            lambda candidate, exact_cluster, *, strict: (True, "unit_test", "base")
        )
        quality_metrics = {
            "review_merge_candidate_rows": [
                {
                    "source_cluster_id": 1,
                    "target_cluster_id": 2,
                    "candidate_id": "partial_merge",
                    "confidence_tier": "high",
                    "target_exact_cluster_id": 1,
                    "edge_weight": 10,
                }
            ],
            "review_merge_cluster_pair_rows": [
                {
                    "source_cluster_id": 1,
                    "target_cluster_id": 2,
                    "pair_review_bucket": "safe_recall_candidate",
                    "source_overmerge_reason": "ok",
                    "target_overmerge_reason": "ok",
                }
            ],
        }

        merged_units, metrics = runner._apply_safe_recall_merge(
            [(base0, [exact_clusters[0]]), (base1, [exact_clusters[1]])],
            quality_metrics,
            _coverage_groups(runner, [partial_candidate]),
        )

        self.assertEqual(len(merged_units), 2)
        self.assertEqual(metrics["safe_recall_merge_merged_pair_count"], 0)
        self.assertEqual(metrics["safe_recall_merge_reject_reason_counts"]["candidate_missing_union_coverage"], 1)

        risky_metrics = {
            "review_merge_candidate_rows": quality_metrics["review_merge_candidate_rows"],
            "review_merge_cluster_pair_rows": [
                {
                    "source_cluster_id": 1,
                    "target_cluster_id": 2,
                    "pair_review_bucket": "safe_recall_candidate",
                    "source_overmerge_reason": "low_pairwise_geometry_purity",
                    "target_overmerge_reason": "ok",
                }
            ],
        }
        risky_candidate = _candidate("partial_merge", bitmap, origin_exact_cluster_id=0, shift_direction="east", coverage={0})
        merged_units, metrics = runner._apply_safe_recall_merge(
            [(base0, [exact_clusters[0]]), (base1, [exact_clusters[1]])],
            risky_metrics,
            _coverage_groups(runner, [risky_candidate]),
        )
        self.assertEqual(len(merged_units), 2)
        self.assertEqual(metrics["safe_recall_merge_attempted_pair_count"], 0)
        self.assertEqual(metrics["safe_recall_merge_reject_reason_counts"]["endpoint_overmerge_reason"], 1)

    def test_safe_recall_merge_keeps_disjoint_pairs_only(self) -> None:
        """第一版 safe merge 不做链式 component 合并，避免大 component 风险。"""

        bitmap = np.zeros((8, 8), dtype=bool)
        bitmap[2:6, 2:6] = True
        exact_clusters = [
            ExactCluster(idx, _record(f"chain{idx}", bitmap, seed_weight=1), [_record(f"chain{idx}", bitmap, seed_weight=1)])
            for idx in range(3)
        ]
        runner = self._make_runner(compute_quality_metrics=True)
        bases = [
            _candidate(f"base_chain_{idx}", bitmap, origin_exact_cluster_id=idx, shift_direction="base")
            for idx in range(3)
        ]
        merge_ab = _candidate("merge_ab", bitmap, origin_exact_cluster_id=0, shift_direction="east", coverage={0, 1})
        merge_bc = _candidate("merge_bc", bitmap, origin_exact_cluster_id=1, shift_direction="east", coverage={1, 2})
        runner._base_candidate_by_exact_id = {idx: bases[idx] for idx in range(3)}
        runner._cached_candidate_matches_exact_result = (  # type: ignore[method-assign]
            lambda candidate, exact_cluster, *, strict: (True, "unit_test", "base")
        )
        quality_metrics = {
            "review_merge_candidate_rows": [
                {
                    "source_cluster_id": 1,
                    "target_cluster_id": 2,
                    "candidate_id": "merge_ab",
                    "confidence_tier": "high",
                    "target_exact_cluster_id": 1,
                    "edge_weight": 10,
                },
                {
                    "source_cluster_id": 2,
                    "target_cluster_id": 3,
                    "candidate_id": "merge_bc",
                    "confidence_tier": "high",
                    "target_exact_cluster_id": 2,
                    "edge_weight": 9,
                },
            ],
            "review_merge_cluster_pair_rows": [
                {
                    "source_cluster_id": 1,
                    "target_cluster_id": 2,
                    "pair_review_bucket": "safe_recall_candidate",
                    "source_overmerge_reason": "ok",
                    "target_overmerge_reason": "ok",
                },
                {
                    "source_cluster_id": 2,
                    "target_cluster_id": 3,
                    "pair_review_bucket": "safe_recall_candidate",
                    "source_overmerge_reason": "ok",
                    "target_overmerge_reason": "ok",
                },
            ],
        }

        merged_units, metrics = runner._apply_safe_recall_merge(
            [(bases[idx], [exact_clusters[idx]]) for idx in range(3)],
            quality_metrics,
            _coverage_groups(runner, [merge_ab, merge_bc]),
        )

        self.assertEqual(len(merged_units), 2)
        self.assertEqual(metrics["safe_recall_merge_merged_pair_count"], 1)
        self.assertEqual(metrics["safe_recall_merge_cluster_reduction"], 1)
        self.assertEqual(metrics["safe_recall_merge_reject_reason_counts"]["cluster_already_merged"], 1)

    def test_empty_exception_message_keeps_exception_type(self) -> None:
        """空消息异常应打印异常类型，方便定位厂内大版图失败。"""

        self.assertEqual(optimized._format_exception_message(MemoryError()), "MemoryError")
        self.assertEqual(optimized._format_exception_message(AssertionError()), "AssertionError")
        self.assertEqual(optimized._format_exception_message(ValueError("bad config")), "bad config")

    def test_exact_pair_match_cache_reuses_strict_witness_result(self) -> None:
        """重复 strict pair 判断应复用缓存，避免 greedy / 质量指标重复算 ECC。"""

        bitmap = np.zeros((8, 8), dtype=bool)
        bitmap[2:6, 2:6] = True
        exact_clusters = [
            ExactCluster(idx, _record(f"cache{idx}", bitmap, seed_weight=1), [_record(f"cache{idx}", bitmap, seed_weight=1)])
            for idx in range(2)
        ]
        exact_by_id = {idx: cluster for idx, cluster in enumerate(exact_clusters)}
        runner = self._make_runner()
        runner._base_candidate_by_exact_id = {
            idx: _candidate(f"base_cache_{idx}", bitmap, origin_exact_cluster_id=idx, shift_direction="base")
            for idx in range(2)
        }
        call_count = 0

        def fake_match(candidate: CandidateClip, exact_cluster: ExactCluster, *, strict: bool) -> tuple[bool, str, str]:
            nonlocal call_count
            del candidate, exact_cluster, strict
            call_count += 1
            return True, "unit_test_cache", "base"

        runner._candidate_matches_exact = fake_match  # type: ignore[method-assign]

        self.assertTrue(runner._exact_pair_matches_strict(0, 1, exact_by_id))
        self.assertTrue(runner._exact_pair_matches_strict(0, 1, exact_by_id))
        self.assertEqual(call_count, 1)
        self.assertEqual(runner.coverage_debug_stats["exact_pair_cache_miss_count"], 1)
        self.assertEqual(runner.coverage_debug_stats["exact_pair_cache_hit_count"], 1)

    def test_result_match_caches_are_bounded_for_low_memory(self) -> None:
        """result 阶段匹配缓存应有 LRU 上限，避免大版图质量评估无界增长。"""

        bitmap = np.zeros((8, 8), dtype=bool)
        bitmap[2:6, 2:6] = True
        exact_clusters = [
            ExactCluster(idx, _record(f"bound{idx}", bitmap, seed_weight=1), [_record(f"bound{idx}", bitmap, seed_weight=1)])
            for idx in range(4)
        ]
        exact_by_id = {idx: cluster for idx, cluster in enumerate(exact_clusters)}
        runner = self._make_runner()
        runner._base_candidate_by_exact_id = {
            idx: _candidate(f"base_bound_{idx}", bitmap, origin_exact_cluster_id=idx, shift_direction="base")
            for idx in range(4)
        }

        def fake_match(candidate: CandidateClip, exact_cluster: ExactCluster, *, strict: bool) -> tuple[bool, str, str]:
            del candidate, exact_cluster, strict
            return True, "unit_test_bounded", "base"

        runner._candidate_matches_exact = fake_match  # type: ignore[method-assign]
        with mock.patch.object(optimized, "RESULT_CANDIDATE_EXACT_CACHE_MAX_ENTRIES", 2):
            selected = _candidate("bounded_selected", bitmap, origin_exact_cluster_id=0, shift_direction="base")
            for exact_cluster in exact_clusters[:3]:
                runner._cached_candidate_matches_exact_result(selected, exact_cluster, strict=True)
        self.assertLessEqual(len(runner._candidate_exact_match_cache), 2)
        self.assertGreaterEqual(int(runner.coverage_debug_stats["candidate_exact_cache_evict_count"]), 1)

        with mock.patch.object(optimized, "RESULT_EXACT_PAIR_CACHE_MAX_ENTRIES", 2):
            for left_id, right_id in ((0, 1), (1, 2), (2, 3)):
                runner._exact_pair_matches_strict(left_id, right_id, exact_by_id)
        self.assertLessEqual(len(runner._exact_pair_match_cache), 2)
        self.assertGreaterEqual(int(runner.coverage_debug_stats["exact_pair_cache_evict_count"]), 1)

    def test_selected_candidate_payload_is_repaired_before_final_verification(self) -> None:
        """final verification 前应能从 coverage group 修复 selected candidate 的 packed bitmap。"""

        bitmap = np.zeros((8, 8), dtype=bool)
        bitmap[2:6, 2:6] = True
        runner = self._make_runner()
        candidate = _candidate("repair_selected", bitmap, origin_exact_cluster_id=0, shift_direction="east", coverage={0})
        candidate_groups = _coverage_groups(runner, [candidate])
        candidate.clip_bitmap = None
        candidate.match_cache.pop(optimized.PACKED_CANDIDATE_CLIP_BITMAP_KEY, None)
        candidate.match_cache.pop(optimized.PACKED_CANDIDATE_CLIP_SHAPE_KEY, None)

        runner._ensure_selected_candidate_payloads([candidate], candidate_groups)

        self.assertIn(optimized.PACKED_CANDIDATE_CLIP_BITMAP_KEY, candidate.match_cache)
        self.assertTrue(np.array_equal(optimized._candidate_clip_bitmap(candidate), bitmap))
        self.assertEqual(runner.coverage_debug_stats["selected_candidate_payload_repair_count"], 1)

    def test_selected_base_payload_repaired_by_clip_signature(self) -> None:
        """selected base 不在 group best 上时，也应能按 clip signature 恢复 packed bitmap。"""

        bitmap = np.zeros((8, 8), dtype=bool)
        bitmap[2:6, 2:6] = True
        runner = self._make_runner()
        best_candidate = _candidate("best_same_bitmap", bitmap, origin_exact_cluster_id=0, shift_direction="east", coverage={0})
        selected_base = _candidate("selected_base_same_bitmap", bitmap, origin_exact_cluster_id=0, shift_direction="base", coverage={0})
        candidate_groups = _coverage_groups(runner, [best_candidate])
        selected_base.clip_bitmap = None
        selected_base.match_cache.pop(optimized.PACKED_CANDIDATE_CLIP_BITMAP_KEY, None)
        selected_base.match_cache.pop(optimized.PACKED_CANDIDATE_CLIP_SHAPE_KEY, None)

        runner._ensure_selected_candidate_payloads([selected_base], candidate_groups)

        self.assertIn(optimized.PACKED_CANDIDATE_CLIP_BITMAP_KEY, selected_base.match_cache)
        self.assertTrue(np.array_equal(optimized._candidate_clip_bitmap(selected_base), bitmap))

    def test_safe_recall_merge_reject_keeps_base_candidate_payload(self) -> None:
        """safe merge 拒绝非 selected base candidate 时不能删除其 packed bitmap。"""

        bitmap = np.zeros((8, 8), dtype=bool)
        bitmap[2:6, 2:6] = True
        exact_clusters = [
            ExactCluster(idx, _record(f"base_keep{idx}", bitmap, seed_weight=1), [_record(f"base_keep{idx}", bitmap, seed_weight=1)])
            for idx in range(2)
        ]
        runner = self._make_runner(compute_quality_metrics=True)
        selected0 = _candidate("selected_keep_0", bitmap, origin_exact_cluster_id=0, shift_direction="east", coverage={0})
        selected1 = _candidate("selected_keep_1", bitmap, origin_exact_cluster_id=1, shift_direction="east", coverage={1})
        base_candidate = _candidate("base_keep_payload", bitmap, origin_exact_cluster_id=0, shift_direction="base", coverage={0})
        candidate_groups = _coverage_groups(runner, [base_candidate])
        group = candidate_groups[0]
        runner._park_candidate_group_bitmap(base_candidate, group.packed_clip_bitmap, group.clip_bitmap_shape)
        quality_metrics = {
            "review_merge_candidate_rows": [
                {
                    "source_cluster_id": 1,
                    "target_cluster_id": 2,
                    "candidate_id": "base_keep_payload",
                    "confidence_tier": "high",
                    "target_exact_cluster_id": 1,
                    "edge_weight": 10,
                }
            ],
            "review_merge_cluster_pair_rows": [
                {
                    "source_cluster_id": 1,
                    "target_cluster_id": 2,
                    "pair_review_bucket": "safe_recall_candidate",
                    "source_overmerge_reason": "ok",
                    "target_overmerge_reason": "ok",
                }
            ],
        }

        merged_units, metrics = runner._apply_safe_recall_merge(
            [(selected0, [exact_clusters[0]]), (selected1, [exact_clusters[1]])],
            quality_metrics,
            candidate_groups,
        )

        self.assertEqual(len(merged_units), 2)
        self.assertEqual(metrics["safe_recall_merge_merged_pair_count"], 0)
        self.assertIn(optimized.PACKED_CANDIDATE_CLIP_BITMAP_KEY, base_candidate.match_cache)
        self.assertTrue(np.array_equal(optimized._candidate_clip_bitmap(base_candidate), bitmap))

    def test_final_verification_reuses_cached_candidate_exact_result(self) -> None:
        """final verification 写入 candidate-exact 缓存，后续重复检查应命中。"""

        bitmap = np.zeros((8, 8), dtype=bool)
        bitmap[2:6, 2:6] = True
        exact_cluster = ExactCluster(0, _record("fv_cache", bitmap, seed_weight=1), [_record("fv_cache", bitmap, seed_weight=1)])
        selected = _candidate("fv_selected", bitmap, origin_exact_cluster_id=0, shift_direction="base", coverage={0})
        runner = self._make_runner()
        call_count = 0

        def fake_match(candidate: CandidateClip, exact_cluster: ExactCluster, *, strict: bool) -> tuple[bool, str, str]:
            nonlocal call_count
            del candidate, exact_cluster, strict
            call_count += 1
            return True, "unit_test_cached", "base"

        runner._candidate_matches_exact = fake_match  # type: ignore[method-assign]
        units = runner._verified_cluster_units([selected], [exact_cluster])
        cached = runner._cached_candidate_matches_exact_result(selected, exact_cluster, strict=True)

        self.assertEqual(units, [(selected, [exact_cluster])])
        self.assertEqual(cached, (True, "unit_test_cached", "base"))
        self.assertEqual(call_count, 1)
        self.assertEqual(runner.coverage_debug_stats["candidate_exact_cache_miss_count"], 1)
        self.assertEqual(runner.coverage_debug_stats["candidate_exact_cache_hit_count"], 1)

    def test_target_witness_pool_reuses_exact_generated_witnesses(self) -> None:
        """同一 exact cluster 的 target witness candidates 应按 exact-id 复用。"""

        bitmap = np.zeros((8, 8), dtype=bool)
        bitmap[2:6, 2:6] = True
        exact_cluster = ExactCluster(0, _record("tw_cache", bitmap, seed_weight=1), [_record("tw_cache", bitmap, seed_weight=1)])
        candidate = _candidate("tw_selected", bitmap, origin_exact_cluster_id=0, shift_direction="base", coverage={0})
        target_base = _candidate("tw_target", bitmap, origin_exact_cluster_id=0, shift_direction="base")
        runner = self._make_runner()
        build_count = 0

        def fake_witnesses(exact_cluster: ExactCluster) -> tuple[list[CandidateClip], CandidateClip | None]:
            nonlocal build_count
            del exact_cluster
            build_count += 1
            return [target_base], target_base

        runner._target_witness_candidates = fake_witnesses  # type: ignore[method-assign]
        runner._candidate_bitmap_pool[(1, 1, b"unit")] = [bitmap]

        self.assertTrue(runner._candidate_matches_exact(candidate, exact_cluster, strict=True)[0])
        self.assertTrue(runner._candidate_matches_exact(candidate, exact_cluster, strict=True)[0])
        self.assertEqual(build_count, 1)
        self.assertEqual(runner.coverage_debug_stats["target_witness_cache_miss_count"], 1)
        self.assertEqual(runner.coverage_debug_stats["target_witness_cache_hit_count"], 1)
        self.assertEqual(runner._candidate_bitmap_pool, {})

    def test_greedy_purity_gate_rejects_bridge_candidate(self) -> None:
        """高风险 bridge candidate 的 strict pair 全失败时应被 set-cover gate 拦截。"""

        bitmaps = []
        for idx in range(4):
            bitmap = np.zeros((8, 8), dtype=bool)
            bitmap[1 + idx : 3 + idx, 1 + idx : 3 + idx] = True
            bitmaps.append(bitmap)
        exact_clusters = [
            ExactCluster(idx, _record(f"g{idx}", bitmaps[idx], seed_weight=1), [_record(f"g{idx}", bitmaps[idx], seed_weight=1)])
            for idx in range(4)
        ]
        runner = self._make_runner()
        base_candidates = [
            _candidate(f"base_{idx}", bitmaps[idx], origin_exact_cluster_id=idx, shift_direction="base")
            for idx in range(4)
        ]
        runner._base_candidate_by_exact_id = {idx: candidate for idx, candidate in enumerate(base_candidates)}
        bridge_bitmap = np.ones((8, 8), dtype=bool)
        bridge = _candidate(
            "bridge",
            bridge_bitmap,
            origin_exact_cluster_id=0,
            shift_direction="right",
            coverage={0, 1, 2, 3},
        )

        def fake_match(candidate: CandidateClip, exact_cluster: ExactCluster, *, strict: bool) -> tuple[bool, str, str]:
            del candidate, exact_cluster, strict
            return False, "unit_test_bridge", "none"

        runner._candidate_matches_exact = fake_match  # type: ignore[method-assign]
        with (
            mock.patch.object(optimized, "GREEDY_PURITY_GATE_EXACT_TRIGGER", 4),
            mock.patch.object(optimized, "GREEDY_PURITY_GATE_STAGE1_SAMPLE_PAIRS", 3),
        ):
            selected = runner._greedy_cover(_coverage_groups(runner, [bridge, *base_candidates]), exact_clusters)

        self.assertNotIn("bridge", {candidate.candidate_id for candidate in selected})
        self.assertEqual({candidate.candidate_id for candidate in selected}, {f"base_{idx}" for idx in range(4)})
        self.assertEqual(runner.coverage_debug_stats["greedy_purity_gate_reject_count"], 1)
        self.assertEqual(runner.coverage_debug_stats["greedy_purity_gate_stage1_reject_count"], 1)
        self.assertLessEqual(runner.coverage_debug_stats["greedy_purity_gate_sampled_pair_count"], 3)

    def test_greedy_purity_gate_rejects_default_high_shift_bridge(self) -> None:
        """默认阈值下 high-shift 小型 bridge candidate 也应触发 purity gate。"""

        bitmaps = []
        for idx in range(10):
            bitmap = np.zeros((8, 8), dtype=bool)
            bitmap[(idx % 5) : (idx % 5) + 2, (idx // 5) : (idx // 5) + 2] = True
            bitmaps.append(bitmap)
        exact_clusters = [
            ExactCluster(idx, _record(f"hs{idx}", bitmaps[idx], seed_weight=1), [_record(f"hs{idx}", bitmaps[idx], seed_weight=1)])
            for idx in range(10)
        ]
        runner = self._make_runner()
        base_candidates = [
            _candidate(f"base_hs_{idx}", bitmaps[idx], origin_exact_cluster_id=idx, shift_direction="base")
            for idx in range(10)
        ]
        runner._base_candidate_by_exact_id = {idx: candidate for idx, candidate in enumerate(base_candidates)}
        bridge = _candidate(
            "high_shift_bridge",
            np.ones((8, 8), dtype=bool),
            origin_exact_cluster_id=0,
            shift_direction="right",
            coverage=set(range(10)),
        )
        bridge.shift_distance_um = 0.20

        def fake_match(candidate: CandidateClip, exact_cluster: ExactCluster, *, strict: bool) -> tuple[bool, str, str]:
            del candidate, exact_cluster, strict
            return False, "unit_test_bridge", "none"

        runner._candidate_matches_exact = fake_match  # type: ignore[method-assign]
        selected = runner._greedy_cover(_coverage_groups(runner, [bridge, *base_candidates]), exact_clusters)

        self.assertNotIn("high_shift_bridge", {candidate.candidate_id for candidate in selected})
        self.assertEqual(runner.coverage_debug_stats["greedy_purity_gate_reject_count"], 1)

    def test_donut_degenerate_coverage_requires_strict_graph(self) -> None:
        """donut 退化匹配应记录统计，并在关闭 auto-pass 时受 strict graph 约束。"""

        bitmap_source = np.zeros((8, 8), dtype=bool)
        bitmap_source[2:6, 2:6] = True
        bitmap_target = bitmap_source.copy()
        bitmap_target[0, 0] = True
        runner = self._make_runner(geometry_match_mode="ecc")
        cand_source = _candidate("cand_source", bitmap_source, origin_exact_cluster_id=0, shift_direction="base")
        cand_target = _candidate("cand_target", bitmap_target, origin_exact_cluster_id=1, shift_direction="base")
        bundle = next(iter(runner._build_candidate_match_bundles(_coverage_groups(runner, [cand_source, cand_target]), 1).values()))
        source_idx = next(idx for idx, candidate in enumerate(bundle["representatives"]) if candidate.candidate_id == "cand_source")
        target_idx = next(idx for idx, candidate in enumerate(bundle["representatives"]) if candidate.candidate_id == "cand_target")
        runner.coverage_detail_seconds = optimized._empty_coverage_detail_seconds()
        runner.coverage_debug_stats = optimized._empty_coverage_debug_stats()

        def fake_values(bundle_arg: dict[str, object], indices: np.ndarray, tol_px: int, key: str) -> np.ndarray:
            del bundle_arg, tol_px
            if key == "dilated_area_px":
                return np.full(int(len(indices)), 100, dtype=np.int64)
            if key == "donut_area_px":
                return np.zeros(int(len(indices)), dtype=np.int64)
            raise AssertionError(key)

        def fake_matrix(bundle_arg: dict[str, object], indices: np.ndarray, tol_px: int, key: str) -> np.ndarray:
            del bundle_arg, tol_px, key
            return np.zeros((int(len(indices)), 1), dtype=np.uint8)

        with (
            mock.patch.object(runner, "_bundle_geometry_values", side_effect=fake_values),
            mock.patch.object(runner, "_bundle_geometry_matrix", side_effect=fake_matrix),
            mock.patch.object(optimized, "_graph_descriptor_passes_with_thresholds", return_value=(False, "signature")),
        ):
            matched = runner._ecc_positive_tolerance_chunk_matches(
                bundle,
                int(source_idx),
                np.asarray([target_idx], dtype=np.int32),
                1,
                1.0,
                1.0,
                100,
                np.zeros(1, dtype=np.uint8),
                np.zeros(1, dtype=np.uint8),
            )

        self.assertEqual(matched.size, 0)
        self.assertEqual(runner.coverage_debug_stats["donut_auto_pass_pair_count"], 1)
        self.assertEqual(runner.coverage_debug_stats["donut_degenerate_strict_graph_reject_count"], 1)

    def test_long_shape_cross_seed_guard_requires_high_signature(self) -> None:
        """long_shape_path 跨 seed type 覆盖时应额外要求更高 signature 相似度。"""

        bitmap_long = np.zeros((10, 10), dtype=bool)
        bitmap_long[4:6, 1:9] = True
        bitmap_array = np.zeros((10, 10), dtype=bool)
        bitmap_array[2:8, 2:8] = True
        runner = self._make_runner()
        cand_long = _candidate("cand_long", bitmap_long, origin_exact_cluster_id=0, shift_direction="base")
        cand_array = _candidate("cand_array", bitmap_array, origin_exact_cluster_id=1, shift_direction="base")
        cand_long.match_cache["origin_seed_type"] = optimized.SEED_TYPE_LONG
        cand_array.match_cache["origin_seed_type"] = optimized.SEED_TYPE_ARRAY
        bundle = next(iter(runner._build_candidate_match_bundles(_coverage_groups(runner, [cand_long, cand_array]), 1).values()))
        source_idx = next(idx for idx, candidate in enumerate(bundle["representatives"]) if candidate.candidate_id == "cand_long")
        target_idx = next(idx for idx, candidate in enumerate(bundle["representatives"]) if candidate.candidate_id == "cand_array")
        runner.coverage_detail_seconds = optimized._empty_coverage_detail_seconds()
        runner.coverage_debug_stats = optimized._empty_coverage_debug_stats()

        with (
            mock.patch.object(optimized, "_graph_descriptor_passes_with_thresholds", return_value=(True, "pass")),
            mock.patch.object(optimized, "_signature_similarity", return_value=0.50),
        ):
            rejected = runner._apply_long_shape_cross_seed_guard(
                bundle,
                int(source_idx),
                np.asarray([target_idx], dtype=np.int32),
            )
        self.assertEqual(rejected.size, 0)
        self.assertEqual(runner.coverage_debug_stats["long_shape_cross_seed_guard_reject_count"], 1)

        runner.coverage_debug_stats = optimized._empty_coverage_debug_stats()
        with (
            mock.patch.object(optimized, "_graph_descriptor_passes_with_thresholds", return_value=(True, "pass")),
            mock.patch.object(optimized, "_signature_similarity", return_value=0.95),
        ):
            accepted = runner._apply_long_shape_cross_seed_guard(
                bundle,
                int(source_idx),
                np.asarray([target_idx], dtype=np.int32),
            )
        self.assertEqual(accepted.tolist(), [int(target_idx)])
        self.assertEqual(runner.coverage_debug_stats["long_shape_cross_seed_guard_pass_count"], 1)

    def test_removed_strict_override_config_is_rejected(self) -> None:
        """旧 strict graph 调参键应显式失败，避免静默忽略旧配置。"""

        with self.assertRaisesRegex(ValueError, "strict_signature_threshold"):
            self._make_runner(strict_signature_threshold=0.0)

    def test_full_prefilter_rejects_before_geometry_cache(self) -> None:
        bitmap_source = np.zeros((16, 16), dtype=bool)
        bitmap_source[4:12, 4:12] = True
        bitmap_target = bitmap_source.copy()
        bitmap_target[6, 6] = False

        runner = self._make_runner(geometry_match_mode="ecc")
        cand_a = _candidate("cand_a", bitmap_source, origin_exact_cluster_id=0, shift_direction="base")
        cand_b = _candidate("cand_b", bitmap_target, origin_exact_cluster_id=1, shift_direction="base")
        source_desc = optimized._bitmap_descriptor(cand_a.clip_bitmap)

        def fake_full_descriptor(bundle, group_idx):
            """稳定制造 topology mismatch，验证 geometry cache 前的 full prefilter。"""

            del bundle
            if int(group_idx) == 0:
                return source_desc
            return optimized.GraphDescriptor(
                invariants=np.asarray(source_desc.invariants, dtype=np.float64),
                topology=np.asarray(source_desc.topology, dtype=np.float64) + 10.0,
                signature_grid=np.asarray(source_desc.signature_grid, dtype=np.float32),
                signature_proj_x=np.asarray(source_desc.signature_proj_x, dtype=np.float32),
                signature_proj_y=np.asarray(source_desc.signature_proj_y, dtype=np.float32),
            )

        runner.prefilter_stats = optimized._empty_prefilter_stats()
        runner.coverage_detail_seconds = optimized._empty_coverage_detail_seconds()
        runner.coverage_debug_stats = optimized._empty_coverage_debug_stats()
        runner._bundle_full_descriptor = fake_full_descriptor
        bundle = next(iter(runner._build_candidate_match_bundles(_coverage_groups(runner, [cand_a, cand_b]), 2).values()))
        shortlist_index = runner._build_bundle_shortlist_index(bundle)
        kept = runner._batch_prefilter(bundle, shortlist_index, 0, np.asarray([1], dtype=np.int64))

        self.assertEqual(kept.size, 0)
        self.assertGreater(runner.prefilter_stats["topology_reject"], 0)
        self.assertGreater(runner.prefilter_stats["full_prefilter_reject"], 0)
        self.assertEqual(runner.coverage_debug_stats["geometry_cache_group_count"], 0)

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
            runner._evaluate_candidate_coverage(_coverage_groups(runner, [cand_a, cand_b, cand_c]), [exact_a, exact_b, exact_c])

        self.assertIsInstance(cand_a.coverage, tuple)
        self.assertIn(1, cand_a.coverage)
        self.assertGreater(runner.coverage_debug_stats["geometry_pair_count"], 0)
        self.assertGreater(runner.coverage_debug_stats["geometry_cache_group_count"], 0)
        self.assertLess(runner.coverage_debug_stats["geometry_cache_group_count"], 3)
        self.assertEqual(runner.coverage_debug_stats["geometry_cache_live_after_bundle_count"], 0)
        self.assertGreater(runner.coverage_debug_stats["geometry_cache_release_count"], 0)
        self.assertGreaterEqual(runner.prefilter_stats["cheap_reject"], 0)

    def test_mega_pair_tracker_low_memory_mode_preserves_coverage(self) -> None:
        bitmap_left = np.zeros((12, 12), dtype=bool)
        bitmap_left[4:8, 2:6] = True
        bitmap_right = np.zeros((12, 12), dtype=bool)
        bitmap_right[4:8, 6:10] = True

        exact_a = ExactCluster(0, _record("a0", bitmap_left, seed_weight=1), [_record("a0", bitmap_left, seed_weight=1)])
        exact_b = ExactCluster(1, _record("b0", bitmap_right, seed_weight=1), [_record("b0", bitmap_right, seed_weight=1)])
        old_threshold = optimized.MEGA_BUNDLE_PAIR_TRACKER_DISABLE_THRESHOLD
        try:
            optimized.MEGA_BUNDLE_PAIR_TRACKER_DISABLE_THRESHOLD = 1
            runner = self._make_runner(geometry_match_mode="acc")
            cand_a = _candidate("cand_a", bitmap_left, origin_exact_cluster_id=0, shift_direction="base")
            cand_b = _candidate("cand_b", bitmap_right, origin_exact_cluster_id=1, shift_direction="base")
            cand_shift = _candidate("cand_shift", bitmap_left, origin_exact_cluster_id=1, shift_direction="left")
            runner.prefilter_stats = optimized._empty_prefilter_stats()
            with redirect_stdout(StringIO()):
                runner._evaluate_candidate_coverage(_coverage_groups(runner, [cand_a, cand_b, cand_shift]), [exact_a, exact_b])
        finally:
            optimized.MEGA_BUNDLE_PAIR_TRACKER_DISABLE_THRESHOLD = old_threshold

        self.assertIn(1, cand_a.coverage)
        self.assertIn(0, cand_shift.coverage)
        self.assertEqual(runner.coverage_debug_stats["pair_tracker_mode"], "source_unique")
        self.assertGreater(runner.coverage_debug_stats["pair_tracker_disabled_bundle_count"], 0)

    def test_ecc_chunk_exception_releases_bundle_geometry_cache(self) -> None:
        bitmap_source = np.zeros((16, 16), dtype=bool)
        bitmap_source[4:12, 4:12] = True
        bitmap_target = bitmap_source.copy()
        bitmap_target[6, 6] = False

        runner = self._make_runner(geometry_match_mode="ecc")
        exact_a = ExactCluster(0, _record("a0", bitmap_source, seed_weight=1), [_record("a0", bitmap_source, seed_weight=1)])
        exact_b = ExactCluster(1, _record("b0", bitmap_target, seed_weight=1), [_record("b0", bitmap_target, seed_weight=1)])
        cand_a = _candidate("cand_a", bitmap_source, origin_exact_cluster_id=0, shift_direction="base")
        cand_b = _candidate("cand_b", bitmap_target, origin_exact_cluster_id=1, shift_direction="base")
        captured: dict[str, object] = {}
        original_build = runner._build_candidate_match_bundles

        def capture_bundle(candidates, tol_px):
            """捕获 coverage 内部 bundle，便于异常后检查缓存是否残留。"""

            bundles = original_build(candidates, tol_px)
            captured["bundle"] = next(iter(bundles.values()))
            return bundles

        def keep_all_targets(bundle, shortlist_index, source_idx, target_indices):
            """绕过 prefilter，确保测试稳定进入 ECC chunk 路径。"""

            del bundle, shortlist_index, source_idx
            return target_indices

        def raise_after_target_cache(
            bundle,
            source_idx,
            target_chunk,
            tol_px,
            source_area,
            source_area_limit,
            source_dilated_area,
            source_packed,
            source_packed_dilated,
        ):
            """模拟 target cache 已构建后发生异常的 chunk 路径。"""

            del source_idx, source_area, source_area_limit, source_dilated_area, source_packed, source_packed_dilated
            for target_idx in np.asarray(target_chunk, dtype=np.int32).tolist():
                runner._bundle_geometry_cache(bundle, int(target_idx), int(tol_px), level="donut")
            raise RuntimeError("forced chunk failure")

        runner._build_candidate_match_bundles = capture_bundle
        runner._batch_prefilter = keep_all_targets
        runner._ecc_positive_tolerance_chunk_matches = raise_after_target_cache
        runner.prefilter_stats = optimized._empty_prefilter_stats()

        with self.assertRaisesRegex(RuntimeError, "forced chunk failure"):
            runner._evaluate_candidate_coverage(_coverage_groups(runner, [cand_a, cand_b]), [exact_a, exact_b])

        bundle = captured["bundle"]
        self.assertEqual(bundle["geometry_cache_by_idx"], {})
        self.assertGreaterEqual(runner.coverage_debug_stats["geometry_cache_release_count"], 2)

    def test_slim_raster_payload_clone_does_not_share_match_cache(self) -> None:
        bitmap = np.zeros((8, 8), dtype=bool)
        bitmap[2:6, 2:6] = True
        cached_record = _record("cached", bitmap, seed_weight=3)
        cached_record.match_cache["optimized_cheap_descriptor"] = optimized._cheap_descriptor(cached_record)
        cached_record.match_cache["auto_seed"] = {"seed_type": "old"}
        payload = optimized._raster_payload_from_record(cached_record)
        runner = self._make_runner()
        seed = optimized.GridSeedCandidate(
            center=(0.5, 0.5),
            seed_bbox=(0.0, 0.0, 1.0, 1.0),
            grid_ix=2,
            grid_iy=3,
            bucket_weight=5,
            seed_type=optimized.SEED_TYPE_RESIDUAL,
        )

        cloned = runner._clone_cached_record(payload, Path("clone.oas"), 7, seed)
        runner._apply_seed_metadata(cloned, Path("clone.oas"), 7, seed)

        self.assertIs(cloned.clip_bitmap, cached_record.clip_bitmap)
        self.assertIs(cloned.expanded_bitmap, cached_record.expanded_bitmap)
        self.assertEqual(cloned.clip_hash, cached_record.clip_hash)
        self.assertEqual(cloned.expanded_hash, cached_record.expanded_hash)
        self.assertEqual(cloned.clip_area, cached_record.clip_area)
        self.assertIsNot(cloned.match_cache, cached_record.match_cache)
        self.assertNotEqual(cloned.match_cache["auto_seed"], cached_record.match_cache["auto_seed"])
        self.assertIn("optimized_cheap_descriptor", cloned.match_cache)
        self.assertNotIn("optimized_graph_descriptor", cloned.match_cache)

    def test_export_rerank_uses_cached_scores_after_clip_release(self) -> None:
        bitmap_a = np.zeros((12, 12), dtype=bool)
        bitmap_a[3:9, 3:9] = True
        bitmap_b = np.zeros((12, 12), dtype=bool)
        bitmap_b[2:10, 4:8] = True
        rec_a = _record("a0", bitmap_a, seed_weight=1)
        rec_b = _record("b0", bitmap_b, seed_weight=3)

        original_member, original_scores = optimized._rerank_export_representative([rec_a, rec_b])
        for record in (rec_a, rec_b):
            optimized._ensure_export_rerank_cache(record, include_distance=False)
            self.assertTrue(optimized._pack_marker_clip_bitmap(record))
            record.clip_bitmap = None

        cached_member, cached_scores = optimized._rerank_export_representative([rec_a, rec_b])

        self.assertEqual(cached_member.marker_id, original_member.marker_id)
        for key in ("score", "medoid_score", "worst_case_score", "distance_worst_case_score", "weight_score"):
            self.assertAlmostEqual(float(cached_scores[key]), float(original_scores[key]), places=6)

    def test_online_exact_grouping_matches_reference(self) -> None:
        bitmap_a = np.zeros((8, 8), dtype=bool)
        bitmap_a[2:6, 2:6] = True
        bitmap_b = np.zeros((8, 8), dtype=bool)
        bitmap_b[1:5, 1:4] = True
        runner = self._make_runner()

        online_records = [
            _record("a0", bitmap_a, seed_weight=1),
            _record("a1", bitmap_a.copy(), seed_weight=2),
            _record("b0", bitmap_b, seed_weight=1),
        ]
        reference_records = [
            _record("a0_ref", bitmap_a, seed_weight=1),
            _record("a1_ref", bitmap_a.copy(), seed_weight=2),
            _record("b0_ref", bitmap_b, seed_weight=1),
        ]

        marker_records: list[MarkerRecord] = []
        exact_clusters: list[ExactCluster] = []
        exact_index_by_key: dict[tuple[str, str], int] = {}
        for record in online_records:
            runner._register_online_exact_record(record, marker_records, exact_clusters, exact_index_by_key)
        reference_clusters = runner._group_exact_clusters(reference_records)

        self.assertEqual(len(exact_clusters), len(reference_clusters))
        self.assertEqual(
            [sorted(int(member.seed_weight) for member in cluster.members) for cluster in exact_clusters],
            [sorted(int(member.seed_weight) for member in cluster.members) for cluster in reference_clusters],
        )
        self.assertEqual(
            [cluster.representative.clip_hash for cluster in exact_clusters],
            [cluster.representative.clip_hash for cluster in reference_clusters],
        )

    def test_online_exact_lightens_nonrepresentative_without_review(self) -> None:
        bitmap = np.zeros((8, 8), dtype=bool)
        bitmap[2:6, 2:6] = True
        runner = self._make_runner()
        rec_a = _record("a0", bitmap, seed_weight=1)
        rec_b = _record("a1", bitmap.copy(), seed_weight=3)
        marker_records: list[MarkerRecord] = []
        exact_clusters: list[ExactCluster] = []
        exact_index_by_key: dict[tuple[str, str], int] = {}

        runner._register_online_exact_record(rec_a, marker_records, exact_clusters, exact_index_by_key)
        runner._register_online_exact_record(rec_b, marker_records, exact_clusters, exact_index_by_key)

        self.assertEqual(len(exact_clusters), 1)
        self.assertIsNotNone(rec_a.clip_bitmap)
        self.assertIsNone(rec_b.clip_bitmap)
        self.assertIsNone(rec_b.expanded_bitmap)
        self.assertIn(optimized.EXPORT_CHEAP_FEATURE_KEY, rec_b.match_cache)
        self.assertIn(optimized.EXPORT_WORST_SCORE_KEY, rec_b.match_cache)
        self.assertIn(optimized.EXPORT_DISTANCE_SCORE_KEY, rec_b.match_cache)

    def test_online_exact_keeps_member_clip_with_review_dir(self) -> None:
        bitmap = np.zeros((8, 8), dtype=bool)
        bitmap[2:6, 2:6] = True
        runner = self._make_runner(materialize_outputs=True)
        rec_a = _record("a0", bitmap, seed_weight=1)
        rec_b = _record("a1", bitmap.copy(), seed_weight=2)
        marker_records: list[MarkerRecord] = []
        exact_clusters: list[ExactCluster] = []
        exact_index_by_key: dict[tuple[str, str], int] = {}

        runner._register_online_exact_record(rec_a, marker_records, exact_clusters, exact_index_by_key)
        runner._register_online_exact_record(rec_b, marker_records, exact_clusters, exact_index_by_key)

        self.assertEqual(len(exact_clusters), 1)
        self.assertIsNotNone(rec_b.clip_bitmap)
        self.assertIsNone(rec_b.expanded_bitmap)

    def test_packed_candidate_group_bitmap_roundtrip_and_window_release(self) -> None:
        bitmap_a = np.zeros((12, 12), dtype=bool)
        bitmap_a[3:9, 3:9] = True
        bitmap_b = np.zeros((12, 12), dtype=bool)
        bitmap_b[3:9, 4:10] = True
        runner = self._make_runner()
        exact_a = ExactCluster(0, _record("a0", bitmap_a, seed_weight=1), [_record("a0", bitmap_a, seed_weight=1)])
        exact_b = ExactCluster(1, _record("b0", bitmap_b, seed_weight=1), [_record("b0", bitmap_b, seed_weight=1)])

        candidate_groups, _, _ = runner._build_global_coverage_candidate_groups([exact_a, exact_b])
        self.assertTrue(all(group.best_candidate.clip_bitmap is None for group in candidate_groups))
        self.assertTrue(all(group.packed_clip_bitmap.size > 0 for group in candidate_groups))

        bundle = next(iter(runner._build_candidate_match_bundles(candidate_groups, 2).values()))
        shortlist_index = runner._build_bundle_shortlist_index(bundle)
        self.assertEqual(shortlist_index["bundle"], bundle)
        desc = runner._bundle_full_descriptor(bundle, 0)
        geom = runner._bundle_geometry_cache(bundle, 0, 2, level="donut")

        self.assertGreater(len(bundle["bitmap_cache_by_idx"]), 0)
        self.assertIsInstance(desc, optimized.GraphDescriptor)
        self.assertIn("packed_donut", geom)
        runner._release_bundle_bitmap_cache(bundle)
        self.assertEqual(bundle["bitmap_cache_by_idx"], {})

    def test_release_helpers_trim_bitmap_lifetimes(self) -> None:
        bitmap = np.zeros((8, 8), dtype=bool)
        bitmap[2:6, 2:6] = True
        rec_a = _record("a0", bitmap, seed_weight=1)
        rec_b = _record("b0", bitmap, seed_weight=1)
        runner = self._make_runner()

        exact_clusters = runner._group_exact_clusters([rec_a, rec_b])
        runner._release_marker_records_after_exact_cluster([rec_a, rec_b], exact_clusters)
        self.assertIsNotNone(exact_clusters[0].representative.expanded_bitmap)
        self.assertIsNone(rec_b.expanded_bitmap)

        shift_cluster = _make_shiftable_exact_cluster()
        candidates = runner._generate_candidates_for_cluster(shift_cluster)
        runner._release_representative_expanded_bitmaps([shift_cluster])
        self.assertIsNone(shift_cluster.representative.expanded_bitmap)

        base_candidate = next(candidate for candidate in candidates if candidate.shift_direction == "base")
        runner._release_unselected_candidates(candidates, {base_candidate.candidate_id})
        self.assertIsNotNone(base_candidate.clip_bitmap)
        self.assertTrue(any(candidate.clip_bitmap is None for candidate in candidates if candidate.shift_direction != "base"))

    def test_candidate_generation_uses_packed_representative_clip(self) -> None:
        """result 前可压缩 representative clip，candidate 生成再按需解包。"""

        shift_cluster = _make_shiftable_exact_cluster()
        runner = self._make_runner()
        optimized._pack_marker_expanded_bitmap(shift_cluster.representative)
        runner._pack_representative_clip_bitmaps_for_result_stage([shift_cluster])

        self.assertIsNone(shift_cluster.representative.clip_bitmap)
        self.assertIn(optimized.PACKED_CLIP_BITMAP_KEY, shift_cluster.representative.match_cache)
        candidates = runner._generate_candidates_for_cluster(shift_cluster)

        self.assertTrue(candidates)
        self.assertTrue(any(candidate.shift_direction == "base" for candidate in candidates))

    def test_save_results_writes_csv_with_template_columns(self) -> None:
        input_oas = self.temp_root / "csv_save.oas"
        _write_oas(
            input_oas,
            [gdstk.rectangle((0.00, 0.00), (0.20, 0.20), layer=1, datatype=0)],
        )
        runner = self._make_runner()
        captured = StringIO()
        with redirect_stdout(captured):
            result = runner.run(str(input_oas))
        logs = captured.getvalue()
        self.assertIn("exact hash 聚合完成", logs)
        self.assertIn("candidate 生成完成", logs)
        self.assertIn("最终 cluster 数", logs)
        self.assertNotIn("内存 RSS", logs)
        self.assertNotIn("memory debug", logs)
        self.assertNotIn("coverage 几何统计", logs)

        self.assertIn("__csv_state", result)
        self.assertIn("__cluster_csv_state", result)
        self.assertEqual(result["clusters"], [])
        self.assertEqual(result["file_metadata"], [])
        self.assertEqual(result["result_csv_row_count"], result["exact_cluster_count"])
        self.assertEqual(result["cluster_representative_csv_row_count"], result["total_clusters"])
        self.assertGreaterEqual(result["result_csv_release_count"], 0)
        self.assertNotIn("quality_metrics", result)

        output_path = self.temp_root / "csv_result.csv"
        with redirect_stdout(StringIO()):
            optimized._save_results(result, str(output_path))
        cluster_output_path = self.temp_root / "csv_result_cluster_representatives.csv"
        metrics_output_path = self.temp_root / "csv_result_quality_metrics.json"
        suspect_output_path = self.temp_root / "csv_result_overmerge_suspects.csv"
        review_merge_output_path = self.temp_root / "csv_result_review_merge_candidates.csv"
        review_merge_pair_output_path = self.temp_root / "csv_result_review_merge_cluster_pairs.csv"
        with output_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        self.assertEqual(reader.fieldnames, optimized.CSV_OUTPUT_COLUMNS)
        self.assertEqual(reader.fieldnames[:5], ["groupID", "cluster_id", "center_x_um", "center_y_um", "clip_size"])
        self.assertEqual(len(reader.fieldnames), 8)
        self.assertEqual(len(rows), result["exact_cluster_count"])
        self.assertEqual(int(rows[0]["groupID"]), 1)
        self.assertEqual(int(rows[0]["cluster_id"]), 1)
        self.assertIn("center_x_um", rows[0])
        self.assertIn("center_y_um", rows[0])
        self.assertIn("group_weight", rows[0])
        self.assertIn("risk_score", rows[0])
        self.assertNotIn("exact_cluster_id", rows[0])
        self.assertNotIn("representative_score", rows[0])
        self.assertNotIn("opc_center_score", rows[0])
        self.assertEqual(sorted(int(row["risk_rank"]) for row in rows), list(range(1, len(rows) + 1)))
        self.assertEqual(float(rows[0]["clip_size"]), float(runner.clip_size_um))
        self.assertTrue(cluster_output_path.exists())
        with cluster_output_path.open("r", encoding="utf-8", newline="") as handle:
            cluster_reader = csv.DictReader(handle)
            cluster_rows = list(cluster_reader)
        self.assertEqual(cluster_reader.fieldnames, optimized.CLUSTER_REPRESENTATIVE_CSV_COLUMNS)
        self.assertEqual(len(cluster_rows), result["total_clusters"])
        self.assertFalse(metrics_output_path.exists())
        self.assertFalse(suspect_output_path.exists())
        self.assertFalse(review_merge_output_path.exists())
        self.assertFalse(review_merge_pair_output_path.exists())
        with self.assertRaises(ValueError):
            optimized._save_results(result, str(self.temp_root / "csv_result.json"))

    def test_default_result_build_skips_member_annotation(self) -> None:
        """无 review 目录时不应给每个 member 写 selected_candidate_info 大 dict。"""

        input_oas = self.temp_root / "no_annotation_default.oas"
        _write_oas(
            input_oas,
            [gdstk.rectangle((0.00, 0.00), (0.20, 0.20), layer=1, datatype=0)],
        )
        runner = self._make_runner()
        with mock.patch.object(
            runner,
            "_annotate_cluster_member_selection",
            side_effect=AssertionError("default run should stay streaming"),
        ):
            with redirect_stdout(StringIO()):
                result = runner.run(str(input_oas))
        self.assertFalse(result["materialized_outputs"])

    def test_exact_cluster_review_row_uses_marker_center(self) -> None:
        bitmap = np.zeros((8, 8), dtype=bool)
        bitmap[2:6, 2:6] = True
        record = _record("review_center", bitmap, seed_weight=3)
        record.marker_center = (3.25, 4.75)
        exact_cluster = ExactCluster(4, record, [record])

        row = optimized._csv_exact_cluster_review_row(
            group_id=5,
            cluster_id=2,
            exact_cluster=exact_cluster,
            clip_size_um=1.35,
            group_weight=3,
            risk_score=0.42,
            risk_rank=1,
        )

        self.assertEqual(list(row.keys()), optimized.CSV_OUTPUT_COLUMNS)
        self.assertEqual(row["groupID"], 5)
        self.assertEqual(row["cluster_id"], 2)
        self.assertEqual(row["center_x_um"], 3.25)
        self.assertEqual(row["center_y_um"], 4.75)
        self.assertEqual(row["group_weight"], 3)
        self.assertNotIn("exact_cluster_id", row)

    def test_main_logs_are_compact_and_csv_only(self) -> None:
        """主入口只打印关键阶段信息，并固定输出 CSV。"""

        input_oas = self.temp_root / "main_log.oas"
        output_path = self.temp_root / "main_log_result.csv"
        _write_oas(
            input_oas,
            [gdstk.rectangle((0.00, 0.00), (0.20, 0.20), layer=1, datatype=0)],
        )
        temp_runs = SCRIPT_DIR / "_temp_runs"
        temp_runs_existed = temp_runs.exists()
        before_runs = set(temp_runs.iterdir()) if temp_runs_existed else set()
        old_argv = sys.argv[:]
        captured = StringIO()
        try:
            sys.argv = [
                "layout_clustering_optimized_v1.py",
                str(input_oas),
                "--output",
                str(output_path),
                "--compute-quality-metrics",
            ]
            with redirect_stdout(captured):
                exit_code = optimized.main()
        finally:
            sys.argv = old_argv
            after_runs = set(temp_runs.iterdir()) if temp_runs.exists() else set()
            for path in after_runs.difference(before_runs):
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
            if not temp_runs_existed and temp_runs.exists():
                temp_runs.rmdir()

        logs = captured.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertTrue(output_path.exists())
        self.assertIn("representative quality:", logs)
        self.assertIn("representative_visual_purity=", logs)
        self.assertIn("weighted_representative_visual_purity=", logs)
        self.assertIn("low_representative_quality_weight_ratio=", logs)
        self.assertIn("actionable overmerge review:", logs)
        self.assertIn("low_shift_low_pairwise_review_weight_ratio=", logs)
        self.assertIn("low_shift_low_pairwise_review_cluster_count=", logs)
        self.assertIn("pairwise_review_sampled_weight_ratio=", logs)
        self.assertIn("fragmentation/review merge audit:", logs)
        self.assertIn("raw_recall=", logs)
        self.assertIn("trusted_recall=", logs)
        self.assertIn("gate_rejected_edge_weight_ratio=", logs)
        self.assertIn("review_merge_candidate_weight_ratio=", logs)
        self.assertIn("singleton_trusted_mergeable_weight_ratio=", logs)
        self.assertIn("safe recall merge:", logs)
        self.assertIn("cluster_reduction=", logs)
        self.assertIn("rejects=", logs)
        self.assertNotIn("visual quality:", logs)
        self.assertNotIn("weighted_visual_purity=", logs)
        self.assertNotIn("low_visual_purity_weight_ratio=", logs)
        self.assertNotIn("fragmentation audit:", logs)
        self.assertNotIn("\nreview merge audit:", logs)
        self.assertNotIn("overmerge/review risk:", logs)
        self.assertNotIn("pairwise_geometry_purity=", logs)
        self.assertNotIn("low_pairwise_geometry_weight_ratio=", logs)
        self.assertNotIn("high_shift_low_pairwise_artifact", logs)
        self.assertNotIn("low_shift_low_pairwise_top_cluster_ids=", logs)
        self.assertNotIn("dry_run_", logs)
        self.assertNotIn("pair_dry_run_", logs)
        self.assertIn("clip size", logs)
        self.assertIn("exact cluster 数", logs)
        self.assertIn("candidate group 数", logs)
        self.assertIn("最终 cluster 数", logs)
        self.assertIn("pattern coverage:", logs)
        self.assertIn("target_edge_length_coverage_ratio=", logs)
        self.assertIn("target_polygon_area_coverage_ratio=", logs)
        self.assertIn("weighted_pattern_type_coverage_ratio=", logs)
        self.assertNotIn("seed coverage audit:", logs)
        self.assertNotIn("clip_window_union_coverage_ratio=", logs)
        self.assertNotIn("seed_density_p95=", logs)
        self.assertNotIn("occupied_grid_cell_count=", logs)
        self.assertNotIn("seeded_occupied_grid_cell_count=", logs)
        self.assertNotIn("empty_occupied_grid_ratio=", logs)
        self.assertNotIn("clip_window_covered_occupied_grid_cell_count=", logs)
        self.assertNotIn("clip_window_uncovered_occupied_grid_ratio=", logs)
        self.assertNotIn("seed_density_p50=", logs)
        self.assertNotIn("seed_density_max=", logs)
        self.assertNotIn("score summary:", logs)
        self.assertNotIn("seed type distribution [", logs)
        self.assertNotIn("overmerge risk:", logs)
        self.assertNotIn("fragmentation recall:", logs)
        self.assertNotIn("coverage_graph_recall=", logs)
        self.assertNotIn("coverage precondition:", logs)
        self.assertNotIn("result build timing:", logs)
        self.assertNotIn("final verification timing:", logs)
        self.assertNotIn("coverage diagnostics:", logs)
        self.assertNotIn("post split:", logs)
        self.assertNotIn("内存 RSS", logs)
        self.assertNotIn("memory debug", logs)
        self.assertNotIn("tuning diagnostics", logs)
        self.assertNotIn("final verification reject details", logs)
        self.assertNotIn("quality metrics:", logs)
        self.assertNotIn("recall_proxy", logs)
        self.assertNotIn("fragmentation_recall=", logs)

    def test_quality_metrics_flag_extends_representative_csv_without_json(self) -> None:
        """仅在显式开启时计算并保存分组质量指标。"""

        input_oas = self.temp_root / "quality_metrics.oas"
        output_stem = f"quality_metrics_result_csv_only_{uuid.uuid4().hex}"
        output_path = self.temp_root / f"{output_stem}.csv"
        _write_oas(
            input_oas,
            [gdstk.rectangle((0.00, 0.00), (0.20, 0.20), layer=1, datatype=0)],
        )
        runner = self._make_runner(compute_quality_metrics=True)
        metrics_path = self.temp_root / f"{output_stem}_quality_metrics.json"
        cluster_path = self.temp_root / f"{output_stem}_cluster_representatives.csv"
        suspect_path = self.temp_root / f"{output_stem}_overmerge_suspects.csv"
        review_merge_path = self.temp_root / f"{output_stem}_review_merge_candidates.csv"
        review_merge_pair_path = self.temp_root / f"{output_stem}_review_merge_cluster_pairs.csv"

        with redirect_stdout(StringIO()):
            result = runner.run(str(input_oas))
            optimized._save_results(result, str(output_path))

        self.assertFalse(metrics_path.exists())
        self.assertTrue(cluster_path.exists())
        self.assertFalse(suspect_path.exists())
        self.assertFalse(review_merge_path.exists())
        self.assertFalse(review_merge_pair_path.exists())
        self.assertTrue(result["quality_metrics_enabled"])
        self.assertNotIn("post_split_stats", result)
        self.assertNotIn("post_split_stats", result["result_summary"])
        self.assertIn("quality_metrics", result)
        self.assertNotIn("review_merge_candidate_rows", result["quality_metrics"])
        self.assertNotIn("review_merge_cluster_pair_rows", result["quality_metrics"])
        for key in (
            "singleton_ratio",
            "singleton_weight_ratio",
            "verified_pass_ratio",
            "merged_repeat_weight_ratio",
            "representative_visual_pass_ratio",
            "representative_visual_weighted_pass_ratio",
            "representative_visual_reject_weight_ratio",
            "representative_visual_purity",
            "weighted_representative_visual_purity",
            "low_representative_quality_weight_ratio",
            "visual_purity_score",
            "weighted_visual_purity",
            "low_visual_purity_weight_ratio",
            "pairwise_geometry_fail_rate",
            "pairwise_geometry_purity",
            "weighted_pairwise_geometry_purity",
            "low_pairwise_geometry_weight_ratio",
            "low_shift_low_pairwise_review_weight_ratio",
            "low_shift_low_pairwise_review_cluster_count",
            "pairwise_review_sampled_weight_ratio",
            "pairwise_geometry_unknown_cluster_count",
            "pairwise_geometry_no_pair_cluster_count",
            "raw_coverage_graph_recall",
            "raw_coverage_graph_cross_cluster_edge_weight_ratio",
            "trusted_fragmentation_recall",
            "gate_rejected_edge_weight_ratio",
            "review_merge_candidate_weight_ratio",
            "high_conf_review_merge_weight_ratio",
            "medium_conf_review_merge_weight_ratio",
            "low_conf_review_merge_weight_ratio",
            "high_conf_singleton_mergeable_weight_ratio",
            "safe_recall_merge_enabled",
            "safe_recall_merge_candidate_pair_count",
            "safe_recall_merge_attempted_pair_count",
            "safe_recall_merge_merged_pair_count",
            "safe_recall_merge_cluster_reduction",
            "safe_recall_merge_checked_exact_count",
            "safe_recall_merge_reject_reason_counts",
            "singleton_trusted_mergeable_weight_ratio",
        ):
            self.assertIn(key, result["quality_metrics"])
        for old_key in (
            "fragmentation_sampled_pair_count",
            "fragmentation_mergeable_pair_count",
            "fragmentation_recall_proxy",
            "singleton_mergeable_ratio",
            "coverage_graph_fragmentation_recall",
            "coverage_graph_cross_cluster_edge_weight_ratio",
            "coverage_graph_singleton_mergeable_weight_ratio",
            "strict_pair_fail_rate",
            "strict_transitivity_purity",
            "weighted_strict_transitivity_purity",
            "strict_low_purity_weight_ratio",
            "strict_fail_count",
            "strict_sampled_pair_count",
            "strict_sample_status",
        ):
            self.assertNotIn(old_key, result["quality_metrics"])
        with cluster_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        self.assertEqual(reader.fieldnames, optimized._cluster_representative_csv_columns(True))
        self.assertEqual(len(rows), result["total_clusters"])
        self.assertIn("representative_visual_pass_ratio", rows[0])
        self.assertIn("representative_visual_checked_count", rows[0])
        self.assertIn("pairwise_geometry_purity", rows[0])
        self.assertIn("pairwise_geometry_fail_count", rows[0])
        self.assertIn("pairwise_geometry_sampled_pair_count", rows[0])
        self.assertNotIn("strict_transitivity_purity", rows[0])
        self.assertNotIn("strict_fail_count", rows[0])
        self.assertNotIn("strict_sampled_pair_count", rows[0])
        self.assertNotIn("strict_sample_status", rows[0])
        self.assertNotIn("purity_proxy", rows[0])
        self.assertNotIn("visual_purity_proxy", rows[0])

    def test_removed_tuning_config_is_rejected(self) -> None:
        """旧 graph/coverage 调参键应显式失败，不再作为兼容残余保留。"""

        with self.assertRaisesRegex(ValueError, "coverage_shortlist_max_targets"):
            self._make_runner(
                graph_signature_threshold=0.70,
                strict_invariant_limit=0.24,
                strict_topology_threshold=5.0,
                strict_signature_threshold=0.78,
                coverage_shortlist_max_targets=96,
            )

    def test_output_uses_geometry_driven_fields(self) -> None:
        input_oas = self.temp_root / "geometry_driven_output.oas"
        _write_oas(
            input_oas,
            [gdstk.rectangle((0.00, 0.00), (0.20, 0.20), layer=1, datatype=0)],
        )
        runner = self._make_runner()
        with redirect_stdout(StringIO()):
            result = runner.run(str(input_oas))
        self.assertEqual(result["pipeline_mode"], optimized.PIPELINE_MODE)
        self.assertEqual(result["seed_strategy"], "geometry_driven")
        self.assertIn("grid_step_ratio", result)
        self.assertIn("grid_step_um", result)
        self.assertIn("grid_seed_count", result)
        self.assertNotIn("contact_pair_seed_count", result)
        self.assertNotIn("drc_component_seed_count", result)
        self.assertGreater(result["grid_seed_count"], 0)
        self.assertIn("seed_type_counts", result)
        self.assertIn("seed_audit", result)
        self.assertIn("residual_local_grid", result["seed_type_counts"])
        self.assertIn("candidate_direction_counts", result)
        self.assertIn("candidate_group_count", result)
        self.assertIn("diagonal_candidate_count", result)
        self.assertIn("selected_diagonal_candidate_count", result)
        self.assertIn("cheap_reject", result["prefilter_stats"])
        self.assertIn("full_prefilter_reject", result["prefilter_stats"])
        self.assertIn("coverage_detail_seconds", result)
        self.assertIn("coverage_debug_stats", result)
        self.assertIn("result_detail_seconds", result)
        self.assertIn("final_verification_detail_seconds", result)
        self.assertIn("result_detail_seconds", result["result_summary"])
        self.assertIn("final_verification_detail_seconds", result["result_summary"])
        self.assertIn("quality_metrics", result["result_detail_seconds"])
        self.assertIn("target_metrics", result)
        self.assertIn("candidate_direction_conversion", result)
        self.assertIn("final_verification_breakdown", result)
        self.assertIn("witness_attempted", result["final_verification_stats"])
        self.assertIn("witness_verified_pass", result["final_verification_stats"])
        self.assertIn("pass_reason_counts", result["final_verification_breakdown"])
        self.assertIn("witness_shift_direction_counts", result["final_verification_breakdown"])
        self.assertNotIn("tuning_diagnostics", result)
        self.assertNotIn("relaxed_only_pass_count", result)
        self.assertNotIn("candidate_object_avoided_count", result)
        self.assertEqual(result["config"]["strict_invariant_limit"], optimized.STRICT_INVARIANT_LIMIT)
        self.assertEqual(result["config"]["strict_topology_threshold"], optimized.STRICT_TOPOLOGY_THRESHOLD)
        self.assertEqual(result["config"]["strict_signature_threshold"], optimized.STRICT_SIGNATURE_THRESHOLD)
        self.assertEqual(result["config"]["coverage_shortlist_max_targets"], optimized.COVERAGE_SHORTLIST_MAX_TARGETS)
        self.assertGreaterEqual(int(result["diagonal_candidate_count"]), 0)
        self.assertGreaterEqual(int(result["selected_diagonal_candidate_count"]), 0)
        self.assertEqual(result["target_metrics"]["coverage_sample_count"], result["total_samples"])
        self.assertEqual(result["target_metrics"]["covered_exact_cluster_count"], result["exact_cluster_count"])
        self.assertGreaterEqual(float(result["target_metrics"]["verified_pass_ratio"]), 0.0)
        self.assertLessEqual(float(result["target_metrics"]["verified_pass_ratio"]), 1.0)
        for key in (
            "geometry_dilated_cache_group_count",
            "geometry_donut_cache_group_count",
            "geometry_cache_live_peak_count",
            "geometry_cache_release_count",
            "geometry_cache_live_after_bundle_count",
            "full_descriptor_cache_group_count",
            "full_prefilter_probe_pair_count",
            "full_prefilter_probe_reject_count",
            "full_prefilter_disabled_bundle_count",
            "shortlist_subgroup_count",
            "shortlist_exact_subgroup_count",
            "shortlist_hnsw_subgroup_count",
            "shortlist_max_subgroup_size",
            "shortlist_payload_peak_count",
            "shortlist_payload_release_count",
            "lazy_signature_embedding_group_count",
            "signature_embedding_live_peak_count",
            "pair_tracker_disabled_bundle_count",
            "pair_tracker_row_count",
            "bucketed_coverage_bundle_count",
            "coverage_fill_bin_count",
            "max_fill_bin_group_count",
            "max_bucket_window_group_count",
            "bucketed_source_group_count",
            "bucketed_target_group_count",
            "window_bitmap_live_peak_count",
            "candidate_exact_cache_hit_count",
            "candidate_exact_cache_miss_count",
            "target_witness_cache_hit_count",
            "target_witness_cache_miss_count",
            "target_witness_cache_evict_count",
        ):
            self.assertIn(key, result["coverage_debug_stats"])
            self.assertGreaterEqual(int(result["coverage_debug_stats"][key]), 0)
        self.assertIn("pair_tracker_mode", result["coverage_debug_stats"])
        for key in (
            "shortlist_payload_build",
            "shortlist_payload_release",
            "geometry_cache_release",
            "bucket_index_build",
            "bucket_window_index",
            "bucket_window_release",
            "greedy_set_cover",
        ):
            self.assertIn(key, result["coverage_detail_seconds"])
            self.assertGreaterEqual(float(result["coverage_detail_seconds"][key]), 0.0)
        self.assertIn("pre_raster_payload_cache_count", result)
        self.assertIn("exact_bitmap_payload_cache_count", result)
        self.assertGreaterEqual(int(result["pre_raster_payload_cache_count"]), 0)
        self.assertGreaterEqual(int(result["exact_bitmap_payload_cache_count"]), 0)
        self.assertNotIn("memory_debug", result)
        for key in ("result_csv_row_count", "result_csv_release_count"):
            self.assertIn(key, result)
            self.assertGreaterEqual(int(result[key]), 0)
        self.assertEqual(int(result["result_csv_row_count"]), int(result["exact_cluster_count"]))
        self.assertEqual(result["result_csv_columns"], optimized.CSV_OUTPUT_COLUMNS)
        self.assertEqual(result["cluster_representative_csv_columns"], optimized._cluster_representative_csv_columns(False))
        self.assertEqual(int(result["cluster_representative_csv_row_count"]), int(result["total_clusters"]))
        self.assertNotIn("review_merge_candidate_csv_columns", result)
        self.assertNotIn("review_merge_candidate_csv_row_count", result)
        self.assertNotIn("review_merge_cluster_pair_csv_columns", result)
        self.assertNotIn("review_merge_cluster_pair_csv_row_count", result)
        self.assertNotIn("overmerge_suspect_csv_columns", result)
        self.assertNotIn("overmerge_suspect_csv_row_count", result)
        self.assertFalse(result["quality_metrics_enabled"])
        self.assertNotIn("quality_metrics", result)
        self.assertIn("seed_coverage_audit", result)
        self.assertIn("target_edge_length_coverage_ratio", result["seed_coverage_audit"])
        self.assertIn("target_polygon_area_coverage_ratio", result["seed_coverage_audit"])
        self.assertIn("weighted_pattern_type_coverage_ratio", result["seed_coverage_audit"])
        self.assertNotIn("clip_window_union_coverage_ratio", result["seed_coverage_audit"])
        self.assertNotIn("clip_window_uncovered_occupied_grid_ratio", result["seed_coverage_audit"])
        self.assertIn("seed_type_distribution", result)
        self.assertIn("score_summary", result)
        for value in result["coverage_detail_seconds"].values():
            self.assertGreaterEqual(float(value), 0.0)
        for value in result["result_detail_seconds"].values():
            self.assertGreaterEqual(float(value), 0.0)
        for value in result["final_verification_detail_seconds"].values():
            self.assertGreaterEqual(float(value), 0.0)
        for direction, generated_count in result["candidate_direction_counts"].items():
            self.assertIn(direction, result["candidate_direction_conversion"])
            self.assertEqual(
                int(result["candidate_direction_conversion"][direction]["generated_count"]),
                int(generated_count),
            )
        self.assertIn("reject_reason_counts", result["final_verification_breakdown"])
        self.assertNotIn("geometry", result["final_verification_breakdown"]["reject_reason_counts"])
        self.assertEqual(result["clusters"], [])
        self.assertEqual(result["file_metadata"], [])

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
        self.assertEqual(result["exact_cluster_count"], result["result_csv_row_count"])
        self.assertEqual(result["total_files"], 0)
        self.assertEqual(result["file_list"], [])
        self.assertEqual(result["file_metadata"], [])
        self.assertEqual(result["clusters"], [])
        self.assertFalse((runner.temp_dir / "samples").exists())
        self.assertFalse((runner.temp_dir / "representatives").exists())

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
        review_dir = self.temp_root / "review_export"
        info = optimized._export_review(result, str(review_dir))
        self.assertTrue(info["exported"])
        self.assertTrue((review_dir / "representative_files.txt").exists())
        self.assertGreaterEqual(info["exported_file_count"], 1)

    def test_sample_layout_001_validation(self) -> None:
        if os.environ.get("RUN_LAYOUT_REAL_SAMPLE") != "1":
            self.skipTest("set RUN_LAYOUT_REAL_SAMPLE=1 to enable heavy real-sample regression")
        if not SAMPLE_LAYOUT.exists():
            self.skipTest(f"missing sample layout: {SAMPLE_LAYOUT}")

        output_path = self.temp_root / "sample_layout_001_v1_result.csv"
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
            optimized._save_results(result, str(output_path))

        self.assertTrue(output_path.exists())
        with output_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertGreater(result["grid_seed_count"], 0)
        self.assertGreater(result["bucketed_seed_count"], 0)
        self.assertGreater(result["exact_cluster_count"], 0)
        self.assertGreater(result["candidate_count"], 0)
        self.assertGreater(result["total_clusters"], 0)
        self.assertFalse(result["materialized_outputs"])
        self.assertEqual(len(rows), result["candidate_group_count"])
        self.assertEqual(result["total_files"], 0)
        self.assertIn("effective_clustering_layers", result)
        self.assertIn("excluded_helper_layers", result)
        self.assertNotIn("contact_pair_seed_count", result)
        self.assertNotIn("drc_component_seed_count", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
