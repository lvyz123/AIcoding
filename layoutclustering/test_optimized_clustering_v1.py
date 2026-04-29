#!/usr/bin/env python3
"""Tests for the uniform-grid optimized v1 clustering pipeline."""

from __future__ import annotations

import json
import os
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
        help_text = optimized._build_parser().format_help()
        self.assertNotIn("--seed-strategy", help_text)
        self.assertIn("--clip-size", help_text)

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
        seeds, stats = optimized._build_uniform_grid_seed_candidates(layout_index, clip_size_um=1.0)
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

    def test_residual_local_grid_keeps_isolated_feature(self) -> None:
        input_oas = self.temp_root / "residual_seed.oas"
        _write_oas(
            input_oas,
            [gdstk.rectangle((0.01, 0.01), (0.03, 0.03), layer=1, datatype=0)],
        )
        runner = self._make_runner()
        layout_index = runner._prepare_layout(input_oas)
        seeds, stats = optimized._build_uniform_grid_seed_candidates(layout_index, clip_size_um=1.0)
        self.assertEqual(stats["seed_strategy"], "geometry_driven")
        self.assertEqual(stats["grid_seed_count"], 1)
        self.assertEqual(stats["bucketed_seed_count"], 1)
        self.assertEqual(stats["array_group_count"], 0)
        self.assertEqual(stats["long_shape_count"], 0)
        self.assertEqual(stats["residual_element_count"], 1)
        self.assertEqual(stats["residual_seed_count"], 1)
        self.assertEqual(len(seeds), 1)
        self.assertTrue(all(seed.seed_type == optimized.SEED_TYPE_RESIDUAL for seed in seeds))

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
        seeds, stats = optimized._build_uniform_grid_seed_candidates(layout_index, clip_size_um=1.0)
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
        self.assertGreater(runner.memory_debug["early_duplicate_shift_candidate_count"], 0)

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
        self.assertEqual(runner.memory_debug["candidate_bitmap_pool_unique_count"], 1)
        self.assertEqual(runner.memory_debug["candidate_bitmap_pool_hit_count"], 1)

    def test_digest_key_groups_same_bitmap_without_raw_key(self) -> None:
        bitmap = np.zeros((8, 8), dtype=bool)
        bitmap[2:6, 2:6] = True
        runner = self._make_runner()
        cand_a = _candidate("cand_a", bitmap, origin_exact_cluster_id=0, shift_direction="base")
        cand_b = _candidate("cand_b", bitmap.copy(), origin_exact_cluster_id=1, shift_direction="base")

        bundle = next(iter(runner._build_candidate_match_bundles([cand_a, cand_b], 0).values()))

        self.assertEqual(len(bundle["candidate_groups"]), 1)
        self.assertEqual(len(bundle["candidate_groups"][0]), 2)
        self.assertEqual(runner.memory_debug["strict_digest_key_count"], 1)
        self.assertGreater(float(runner.memory_debug["strict_key_bytes_avoided_estimate_mb"]), 0.0)

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
            bundle = next(iter(runner._build_candidate_match_bundles([cand_a, cand_b], 0).values()))
        finally:
            optimized._strict_bitmap_digest = original_digest

        self.assertEqual(len(bundle["candidate_groups"]), 2)
        self.assertGreater(runner.memory_debug["strict_digest_collision_count"], 0)

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
                    runner._evaluate_candidate_coverage([cand_a, cand_b, cand_shift], [exact_a, exact_b])
            finally:
                optimized.COVERAGE_BUCKETED_GROUP_THRESHOLD = old_threshold
            return tuple(cand_a.coverage), tuple(cand_shift.coverage), dict(runner.coverage_debug_stats)

        normal_a, normal_shift, _ = run_once(False)
        bucketed_a, bucketed_shift, bucket_stats = run_once(True)

        self.assertEqual(bucketed_a, normal_a)
        self.assertEqual(bucketed_shift, normal_shift)
        self.assertGreater(bucket_stats["bucketed_coverage_bundle_count"], 0)
        self.assertGreater(bucket_stats["coverage_fill_bin_count"], 0)

    def test_same_hash_exact_pass_crosses_fill_bins_without_geometry_cache(self) -> None:
        bitmap_sparse = np.zeros((12, 12), dtype=bool)
        bitmap_sparse[1:3, 1:3] = True
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
                runner._evaluate_candidate_coverage([cand_sparse, cand_dense], [exact_a, exact_b])
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
        bundle = next(iter(runner._build_candidate_match_bundles([cand_a, cand_b, cand_c], 2).values()))
        shortlist_index = runner._build_bundle_shortlist_index(bundle)
        self.assertNotIn("descriptors", shortlist_index)
        self.assertEqual(shortlist_index["cheap_invariants"].shape[0], 3)
        self.assertEqual(shortlist_index["signature_embeddings"].shape[0], 3)
        kept = runner._batch_prefilter(bundle, shortlist_index, 0, np.asarray([1, 2], dtype=np.int64))

        self.assertEqual(kept.tolist(), [1])
        self.assertEqual(runner.coverage_debug_stats["full_descriptor_cache_group_count"], 2)
        self.assertEqual(set(bundle["full_descriptor_cache_by_idx"].keys()), {0, 1})
        self.assertNotIn("optimized_graph_descriptor", cand_a.match_cache)
        self.assertNotIn("optimized_graph_descriptor", cand_b.match_cache)
        self.assertNotIn("optimized_graph_descriptor", cand_c.match_cache)

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
        bundle = next(iter(runner._build_candidate_match_bundles([cand_a, cand_b], 2).values()))
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
            runner._evaluate_candidate_coverage([cand_a, cand_b, cand_c], [exact_a, exact_b, exact_c])

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
                runner._evaluate_candidate_coverage([cand_a, cand_b, cand_shift], [exact_a, exact_b])
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
            runner._evaluate_candidate_coverage([cand_a, cand_b], [exact_a, exact_b])

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

    def test_stream_save_results_preserves_schema(self) -> None:
        input_oas = self.temp_root / "stream_save.oas"
        _write_oas(
            input_oas,
            [gdstk.rectangle((0.00, 0.00), (0.20, 0.20), layer=1, datatype=0)],
        )
        runner = self._make_runner()
        original = optimized._should_stream_result
        optimized._should_stream_result = lambda sample_count, cluster_count, materialize_outputs: True
        try:
            with redirect_stdout(StringIO()):
                result = runner.run(str(input_oas))
        finally:
            optimized._should_stream_result = original

        self.assertIn("__stream_state", result)
        self.assertEqual(result["clusters"], [])
        self.assertEqual(result["file_metadata"], [])

        output_path = self.temp_root / "stream_result.json"
        with redirect_stdout(StringIO()):
            optimized._save_results(result, str(output_path), "json")
        loaded = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["total_samples"], len(loaded["file_metadata"]))
        self.assertEqual(loaded["total_clusters"], len(loaded["clusters"]))
        self.assertTrue(all("seed_type" in sample for sample in loaded["file_metadata"]))
        self.assertTrue(all("sample_metadata" in cluster for cluster in loaded["clusters"]))

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
        self.assertIn("diagonal_candidate_count", result)
        self.assertIn("selected_diagonal_candidate_count", result)
        self.assertIn("cheap_reject", result["prefilter_stats"])
        self.assertIn("full_prefilter_reject", result["prefilter_stats"])
        self.assertIn("coverage_detail_seconds", result)
        self.assertIn("coverage_debug_stats", result)
        self.assertIn("result_detail_seconds", result)
        self.assertIn("final_verification_detail_seconds", result)
        self.assertGreaterEqual(int(result["diagonal_candidate_count"]), 0)
        self.assertGreaterEqual(int(result["selected_diagonal_candidate_count"]), 0)
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
            "pair_tracker_disabled_bundle_count",
            "pair_tracker_row_count",
            "bucketed_coverage_bundle_count",
            "coverage_fill_bin_count",
            "max_fill_bin_group_count",
            "max_bucket_window_group_count",
            "bucketed_source_group_count",
            "bucketed_target_group_count",
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
        ):
            self.assertIn(key, result["coverage_detail_seconds"])
            self.assertGreaterEqual(float(result["coverage_detail_seconds"][key]), 0.0)
        self.assertIn("pre_raster_payload_cache_count", result)
        self.assertIn("exact_bitmap_payload_cache_count", result)
        self.assertGreaterEqual(int(result["pre_raster_payload_cache_count"]), 0)
        self.assertGreaterEqual(int(result["exact_bitmap_payload_cache_count"]), 0)
        self.assertIn("memory_debug", result)
        for key in (
            "rss_collect_markers_mb",
            "rss_exact_cluster_mb",
            "rss_candidate_generation_mb",
            "rss_coverage_eval_mb",
            "rss_set_cover_mb",
            "rss_result_build_mb",
            "rss_peak_estimate_mb",
            "released_marker_expanded_count",
            "released_marker_clip_count",
            "released_candidate_clip_count",
            "released_cache_owner_count",
            "pre_raster_payload_cache_count",
            "exact_bitmap_payload_cache_count",
            "packed_marker_expanded_count",
            "unpacked_marker_expanded_count",
            "packed_marker_clip_count",
            "candidate_bitmap_pool_unique_count",
            "candidate_bitmap_pool_hit_count",
            "released_candidate_list_ref_count",
            "strict_digest_key_count",
            "strict_digest_collision_count",
            "strict_key_bytes_avoided_estimate_mb",
            "early_duplicate_shift_candidate_count",
        ):
            self.assertIn(key, result["memory_debug"])
        for value in result["coverage_detail_seconds"].values():
            self.assertGreaterEqual(float(value), 0.0)
        for value in result["result_detail_seconds"].values():
            self.assertGreaterEqual(float(value), 0.0)
        for value in result["final_verification_detail_seconds"].values():
            self.assertGreaterEqual(float(value), 0.0)
        self.assertTrue(all("export_representative_metadata" in cluster for cluster in result["clusters"]))
        self.assertTrue(all("seed_type" in sample for cluster in result["clusters"] for sample in cluster["sample_metadata"]))
        self.assertTrue(
            all(sample["grid_cell_bbox"] == sample["seed_bbox"] for cluster in result["clusters"] for sample in cluster["sample_metadata"])
        )
        for cluster in result["clusters"]:
            self.assertIn("distance_worst_case_score", cluster["export_representative_metadata"])
            self.assertGreaterEqual(float(cluster["export_representative_metadata"]["distance_worst_case_score"]), 0.0)

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
        if os.environ.get("RUN_LAYOUT_REAL_SAMPLE") != "1":
            self.skipTest("set RUN_LAYOUT_REAL_SAMPLE=1 to enable heavy real-sample regression")
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
        self.assertEqual(loaded["seed_strategy"], "geometry_driven")
        self.assertGreater(loaded["grid_seed_count"], 0)
        self.assertGreater(loaded["bucketed_seed_count"], 0)
        self.assertGreater(loaded["exact_cluster_count"], 0)
        self.assertGreater(loaded["candidate_count"], 0)
        self.assertGreater(loaded["total_clusters"], 0)
        self.assertIn("seed_type_counts", loaded)
        self.assertIn("seed_audit", loaded)
        self.assertIn("diagonal_candidate_count", loaded)
        self.assertFalse(loaded["materialized_outputs"])
        self.assertEqual(loaded["total_samples"], len(loaded["file_metadata"]))
        self.assertEqual(loaded["total_files"], 0)
        self.assertIn("effective_clustering_layers", loaded)
        self.assertIn("excluded_helper_layers", loaded)
        self.assertNotIn("contact_pair_seed_count", loaded)
        self.assertNotIn("drc_component_seed_count", loaded)


if __name__ == "__main__":
    unittest.main(verbosity=2)
