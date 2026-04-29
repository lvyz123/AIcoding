#!/usr/bin/env python3
"""Tests for the Python 3.6 compatible LSF v2 clustering pipeline."""

import ast
import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

import gdstk
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import layout_clustering_optimized_v2_lsf as v2_lsf
from layout_utils_lsf import _write_oas_library
from mainline_lsf import CandidateClip
from mainline_lsf import ExactCluster
from mainline_lsf import GridSeedCandidate
from mainline_lsf import MarkerRecord
from mainline_lsf import _canonical_bitmap_hash
from mainline_lsf import _dedupe_geometry_seeds
from mainline_lsf import add_candidates_to_candidate_bundle_accumulator
from mainline_lsf import build_uniform_grid_seed_candidates
from mainline_lsf import candidate_shift_summary
from mainline_lsf import create_candidate_bundle_accumulator
from mainline_lsf import evaluate_candidate_coverage
from mainline_lsf import generate_candidates_for_cluster
from mainline_lsf import prepare_layout
from mainline_lsf import load_candidate_bundle_buckets_for_candidates
from mainline_lsf import load_coverage_shard_metadata
from mainline_lsf import load_shard_records
from mainline_lsf import save_candidate_bundle_index
from mainline_lsf import save_candidate_bundle_index_from_accumulator


LSF_FILES = [
    SCRIPT_DIR / "layout_clustering_optimized_v2_lsf.py",
    SCRIPT_DIR / "mainline_lsf.py",
    SCRIPT_DIR / "layout_utils_lsf.py",
    SCRIPT_DIR / "layer_operations_lsf.py",
]


def _write_oas(path, polygons):
    """写出最小 OAS fixture。"""

    lib = gdstk.Library()
    cell = gdstk.Cell("TOP")
    for poly in polygons:
        cell.add(poly)
    lib.add(cell)
    _write_oas_library(lib, str(path))


def _make_candidate(candidate_id, origin_exact_cluster_id, bitmap, shift_direction="base"):
    """构造 coverage 单测使用的最小 CandidateClip。"""

    clip_hash, _ = _canonical_bitmap_hash(bitmap)
    coverage = set([int(origin_exact_cluster_id)]) if str(shift_direction) == "base" else set()
    return CandidateClip(
        candidate_id=str(candidate_id),
        origin_exact_cluster_id=int(origin_exact_cluster_id),
        origin_exact_key="exact_%s" % int(origin_exact_cluster_id),
        center=(0.0, 0.0),
        clip_bbox=(0.0, 0.0, 1.0, 1.0),
        clip_bbox_q=(0, 0, int(bitmap.shape[1]), int(bitmap.shape[0])),
        clip_bitmap=bitmap,
        clip_hash=str(clip_hash),
        shift_direction=str(shift_direction),
        shift_distance_um=0.0,
        coverage=coverage,
        source_marker_id="marker_%s" % int(origin_exact_cluster_id),
    )


def _make_shiftable_exact_cluster():
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
        metadata={},
    )
    return ExactCluster(0, "exact_shiftable", record, [record])


def _make_dummy_exact_cluster(cluster_id, area_px):
    """构造 source shard 规划测试使用的最小 exact cluster。"""

    bitmap = np.zeros((10, 10), dtype=bool)
    bitmap.reshape(-1)[: int(area_px)] = True
    representative = type("Representative", (object,), {})()
    representative.clip_bitmap = bitmap
    cluster = type("Cluster", (object,), {})()
    cluster.exact_cluster_id = int(cluster_id)
    cluster.representative = representative
    return cluster


class OptimizedV2LsfTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = SCRIPT_DIR / "test_outputs" / "_optimized_v2_lsf"
        shutil.rmtree(str(self.temp_root), ignore_errors=True)
        self.temp_root.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(str(self.temp_root), ignore_errors=True)

    def test_lsf_scripts_do_not_import_old_pipeline(self):
        """LSF 新脚本不能依赖旧版脚本。"""

        forbidden_modules = {
            "mainline",
            "layout_utils",
            "layer_operations",
            "layout_clustering_optimized_v1",
        }
        for path in LSF_FILES:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertNotIn(alias.name, forbidden_modules, str(path))
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn(node.module, forbidden_modules, str(path))

    def test_lsf_scripts_parse_as_python36(self):
        """LSF 新脚本需要保持 Python 3.6 语法可解析。"""

        for path in LSF_FILES:
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path), feature_version=(3, 6))

    def test_legacy_bitmap_prefilter_removed(self):
        """v2_lsf coverage 不再保留旧的 bitmap/XOR prefilter 路径。"""

        source = (SCRIPT_DIR / "mainline_lsf.py").read_text(encoding="utf-8")
        tree = ast.parse(source, filename="mainline_lsf.py")
        function_name_list = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        function_names = set(function_name_list)
        self.assertNotIn("_bitmap_descriptor", function_names)
        self.assertNotIn("_cheap_coverage_prefilter", function_names)
        self.assertNotIn("_xor_coverage_prefilter", function_names)
        self.assertEqual(function_name_list.count("evaluate_candidate_coverage"), 1)

    def test_exact_hash_direct_skips_descriptor_and_geometry_cache(self):
        """exact hash 直通应覆盖同 hash origin，且不触发 full descriptor/geometry cache。"""

        bitmap = np.zeros((12, 12), dtype=bool)
        bitmap[2:8, 3:9] = True
        candidates = [
            _make_candidate("cand_000000_000", 0, bitmap.copy()),
            _make_candidate("cand_000001_000", 1, bitmap.copy()),
        ]
        stats = evaluate_candidate_coverage(
            candidates,
            [],
            {
                "geometry_match_mode": "ecc",
                "area_match_ratio": 0.96,
                "edge_tolerance_um": 0.02,
                "pixel_size_nm": 20,
            },
        )
        self.assertEqual(candidates[0].coverage, set([0, 1]))
        self.assertEqual(candidates[1].coverage, set([0, 1]))
        self.assertGreater(stats["exact_hash_pairs"], 0)
        self.assertEqual(stats["geometry_pair_count"], 0)
        self.assertEqual(stats["full_descriptor_cache_group_count"], 0)
        self.assertEqual(stats["geometry_cache_group_count"], 0)
        self.assertIn("coverage_detail_seconds", stats)
        self.assertTrue(all(value >= 0.0 for value in stats["coverage_detail_seconds"].values()))

    def test_candidate_bundle_fill_bucket_loads_only_neighbor_bins(self):
        """大 shape candidate bundle 应按 fill 子桶加载，减少 coverage shard 目标集合。"""

        candidates = []
        shape = (16, 16)
        pixel_count = int(shape[0] * shape[1])
        for idx in range(100):
            bitmap = np.zeros(shape, dtype=bool)
            if idx < 50:
                area = 8 + (idx % 12)
            else:
                area = 180 + (idx % 24)
            rng = np.random.RandomState(idx)
            bitmap.reshape(-1)[rng.permutation(pixel_count)[:area]] = True
            candidates.append(_make_candidate(idx, idx, bitmap))

        bundle_index = save_candidate_bundle_index(
            candidates,
            self.temp_root / "candidate_bundle_split",
            {"pipeline_mode": v2_lsf.PIPELINE_MODE, "stage": "unit-test"},
        )
        self.assertEqual(bundle_index["bucket_split_mode"], "shape_fill_bin")
        self.assertGreater(bundle_index["bucket_count"], bundle_index["shape_bucket_count"])
        shape_item = next(iter(bundle_index["shape_buckets"].values()))
        first_bucket = next(iter(shape_item["buckets"].values()))
        arrays = np.load(first_bucket["output_npz"], allow_pickle=False)
        self.assertIn("packed_bitmaps", arrays.files)
        self.assertIn("cheap_invariants", arrays.files)
        self.assertIn("cheap_signature_vectors", arrays.files)
        self.assertIn("cheap_subgroup_keys", arrays.files)

        target_bundles, load_stats = load_candidate_bundle_buckets_for_candidates(bundle_index, [candidates[0]])
        self.assertEqual(load_stats["shape_count_loaded"], 1)
        self.assertLess(load_stats["bucket_count_loaded"], bundle_index["bucket_count"])
        self.assertLess(load_stats["candidate_group_count_loaded"], bundle_index["candidate_group_count"])
        loaded_group_total = sum(len(bundle["candidate_groups"]) for bundle in target_bundles.values())
        self.assertEqual(loaded_group_total, load_stats["candidate_group_count_loaded"])
        loaded_bundle = next(iter(target_bundles.values()))
        self.assertIn("precomputed_cheap_invariants", loaded_bundle)
        self.assertIn("precomputed_packed_bitmaps", loaded_bundle)

    def test_chunked_candidate_bundle_matches_eager_bundle_counts(self):
        """chunked bundle 写出应与 eager bundle 保持核心计数一致。"""

        candidates = []
        for idx in range(12):
            bitmap = np.zeros((8, 8), dtype=bool)
            bitmap[1:4, 1:4] = True
            if idx % 3 == 0:
                bitmap[4:6, 4:6] = True
            if idx % 4 == 0:
                bitmap = np.rot90(bitmap)
            candidates.append(_make_candidate(idx, idx % 5, bitmap, "base" if idx % 2 == 0 else "left"))

        eager_index = save_candidate_bundle_index(
            candidates,
            self.temp_root / "candidate_bundle_eager",
            {"pipeline_mode": v2_lsf.PIPELINE_MODE, "stage": "unit-test-eager"},
        )
        accumulator = create_candidate_bundle_accumulator()
        add_candidates_to_candidate_bundle_accumulator(accumulator, candidates[:5])
        add_candidates_to_candidate_bundle_accumulator(accumulator, candidates[5:9])
        add_candidates_to_candidate_bundle_accumulator(accumulator, candidates[9:])
        chunked_index = save_candidate_bundle_index_from_accumulator(
            accumulator,
            self.temp_root / "candidate_bundle_chunked",
            {"pipeline_mode": v2_lsf.PIPELINE_MODE, "stage": "unit-test-chunked"},
        )

        self.assertEqual(chunked_index["candidate_count"], eager_index["candidate_count"])
        self.assertEqual(chunked_index["candidate_group_count"], eager_index["candidate_group_count"])
        self.assertEqual(chunked_index["bucket_count"], eager_index["bucket_count"])
        self.assertEqual(chunked_index["shape_bucket_count"], eager_index["shape_bucket_count"])

    def test_prepare_generates_manifest_and_shards(self):
        """prepare 阶段应生成 manifest、seed 文件和 shard 命令。"""

        input_oas = self.temp_root / "prepare.oas"
        _write_oas(
            input_oas,
            [
                gdstk.rectangle((0.05, 0.05), (0.20, 0.20), layer=1, datatype=0),
                gdstk.rectangle((1.05, 0.05), (1.20, 0.20), layer=1, datatype=0),
            ],
        )
        work_dir = self.temp_root / "work_prepare"
        config = {
            "clip_size_um": 1.0,
            "geometry_match_mode": "ecc",
            "area_match_ratio": 0.96,
            "edge_tolerance_um": 0.02,
            "pixel_size_nm": 20,
        }
        manifest = v2_lsf.prepare_stage(str(input_oas), str(work_dir), config, [], shard_count=1, shard_size=1)
        manifest_path = work_dir / "manifest.json"
        self.assertTrue(manifest_path.exists())
        self.assertTrue(Path(manifest["seed_file"]).exists())
        self.assertIn("seed_audit", manifest)
        self.assertTrue(Path(manifest["seed_audit"]["output_json"]).exists())
        self.assertIn("spatial_index_stats", manifest)
        self.assertGreaterEqual(manifest["spatial_index_stats"]["max_bin_load"], 0)
        self.assertIn("seed_type_counts", manifest["seed_stats"])
        self.assertIn("array_spacing_seed_count", manifest["seed_stats"])
        self.assertGreaterEqual(manifest["seed_stats"]["array_spacing_seed_count"], 0)
        self.assertGreater(manifest["shard_count"], 0)
        self.assertTrue(all("run-shard" in shard["command"] for shard in manifest["shards"]))
        self.assertTrue(all("halo_bbox" in shard for shard in manifest["shards"]))
        self.assertIn("input_file_bytes", manifest)
        self.assertGreater(manifest["input_file_bytes"], 0)
        self.assertEqual(manifest["tile_cache_mode"], "per_shard_halo_oas_v1")
        self.assertIn("tile_oas_total_bytes", manifest)
        self.assertGreater(manifest["tile_oas_total_bytes"], 0)
        self.assertIn("tile_oas_total_element_count", manifest)
        self.assertGreaterEqual(manifest["tile_oas_total_element_count"], 0)
        for shard in manifest["shards"]:
            self.assertEqual(shard["tile_cache_mode"], "per_shard_halo_oas_v1")
            self.assertTrue(Path(shard["tile_oas"]).exists())
            self.assertGreater(shard["tile_oas_bytes"], 0)
            self.assertGreaterEqual(shard["tile_element_count"], 0)
        self.assertIn("lsf_wrapper", manifest)
        self.assertIn("run_shards", manifest["lsf_wrapper"])
        run_shards = manifest["lsf_wrapper"]["run_shards"]
        self.assertEqual(run_shards["command_count"], manifest["shard_count"])
        self.assertTrue(Path(run_shards["command_file"]).exists())
        self.assertTrue(Path(run_shards["bsub_template"]).exists())

    def test_merge_stage_rejects_large_central_exact_cluster_count(self):
        """大样本不应误走集中式 merge。"""

        input_oas = self.temp_root / "merge_limit.oas"
        _write_oas(
            input_oas,
            [
                gdstk.rectangle((0.05, 0.05), (0.24, 0.24), layer=1, datatype=0),
                gdstk.rectangle((1.05, 0.05), (1.24, 0.24), layer=1, datatype=0),
                gdstk.rectangle((2.05, 0.05), (2.24, 0.24), layer=1, datatype=0),
            ],
        )
        work_dir = self.temp_root / "work_merge_limit"
        config = {
            "clip_size_um": 1.0,
            "geometry_match_mode": "ecc",
            "area_match_ratio": 0.96,
            "edge_tolerance_um": 0.02,
            "pixel_size_nm": 20,
        }
        manifest = v2_lsf.prepare_stage(str(input_oas), str(work_dir), config, [], shard_count=1, shard_size=4)
        manifest_path = work_dir / "manifest.json"
        for shard in manifest["shards"]:
            v2_lsf.run_shard_stage(str(manifest_path), int(shard["shard_id"]))
        with mock.patch.object(v2_lsf, "CENTRAL_MERGE_EXACT_CLUSTER_LIMIT", 0):
            with self.assertRaises(RuntimeError) as ctx:
                v2_lsf.merge_stage(str(manifest_path), str(self.temp_root / "merge_limit.json"))
        self.assertIn("prepare-coverage", str(ctx.exception))

    def test_coverage_source_shards_are_grouped_by_fill_bin(self):
        """coverage source shards 应优先按 fill-bin 分组。"""

        clusters = [
            _make_dummy_exact_cluster(0, 4),
            _make_dummy_exact_cluster(1, 4),
            _make_dummy_exact_cluster(2, 4),
            _make_dummy_exact_cluster(3, 20),
            _make_dummy_exact_cluster(4, 20),
            _make_dummy_exact_cluster(5, 60),
        ]
        clusters = sorted(
            clusters,
            key=lambda cluster: (
                int(v2_lsf.coverage_fill_bin_for_bitmap(cluster.representative.clip_bitmap)),
                int(cluster.exact_cluster_id),
            ),
        )
        specs = v2_lsf._coverage_source_shard_specs(clusters, 2)
        self.assertEqual([spec["end"] - spec["start"] for spec in specs], [2, 1, 2, 1])
        self.assertTrue(all(spec["source_fill_bin_count"] == 1 for spec in specs))
        self.assertEqual(len(set(spec["source_fill_bin_values"][0] for spec in specs)), 3)

    def test_grid_step_ratio_is_fixed_to_v1_default(self):
        """grid_step_ratio 应固定保持 v1 主线默认值 0.5。"""

        input_oas = self.temp_root / "grid_ratio.oas"
        _write_oas(
            input_oas,
            [
                gdstk.rectangle((0.05, 0.05), (0.22, 0.22), layer=1, datatype=0),
                gdstk.rectangle((1.05, 0.05), (1.22, 0.22), layer=1, datatype=0),
                gdstk.rectangle((2.05, 0.05), (2.22, 0.22), layer=1, datatype=0),
            ],
        )
        layout_index = prepare_layout(str(input_oas), None, False)
        default_seeds, default_stats = build_uniform_grid_seed_candidates(layout_index, 1.0)
        self.assertEqual(default_stats["seed_strategy"], "geometry_driven")
        self.assertEqual(default_stats["grid_step_ratio"], 0.5)
        self.assertEqual(default_stats["grid_step_um"], 0.5)
        self.assertGreater(len(default_seeds), 0)

        manifest = v2_lsf.prepare_stage(
            str(input_oas),
            str(self.temp_root / "work_grid_ratio"),
            {
                "clip_size_um": 1.0,
                "geometry_match_mode": "ecc",
                "area_match_ratio": 0.96,
                "edge_tolerance_um": 0.02,
                "pixel_size_nm": 20,
            },
            [],
            shard_count=1,
            shard_size=10,
        )
        self.assertEqual(manifest["seed_stats"]["grid_step_ratio"], 0.5)
        self.assertEqual(manifest["config"]["grid_step_ratio"], 0.5)

    def test_seed_json_preserves_seed_type(self):
        """seed JSON 应保留 seed_type，同时兼容旧 payload。"""

        seed = GridSeedCandidate((1.0, 2.0), (0.5, 1.5, 1.5, 2.5), 3, 4, 7, "array_representative")
        restored = GridSeedCandidate.from_json(seed.to_json())
        self.assertEqual(restored.seed_type, "array_representative")
        self.assertEqual(restored.bucket_weight, 7)
        legacy = GridSeedCandidate.from_json(
            {
                "center": [0.0, 0.0],
                "seed_bbox": [-0.5, -0.5, 0.5, 0.5],
                "grid_ix": 0,
                "grid_iy": 0,
                "bucket_weight": 1,
            }
        )
        self.assertEqual(legacy.seed_type, "residual_local_grid")

    def test_candidate_generation_adds_bounded_diagonal_shifts(self):
        """systematic shift 应包含少量 diagonal 候选，并保持诊断统计可读。"""

        cluster = _make_shiftable_exact_cluster()
        candidates = generate_candidates_for_cluster(
            cluster,
            {
                "pixel_size_nm": 10,
                "edge_tolerance_um": 0.01,
                "max_shift_count": 4,
            },
        )
        directions = set(str(candidate.shift_direction) for candidate in candidates)
        diagonal_candidates = [candidate for candidate in candidates if str(candidate.shift_direction).startswith("diag_")]
        self.assertIn("base", directions)
        self.assertTrue(any(direction in directions for direction in ("left", "right")))
        self.assertTrue(any(direction in directions for direction in ("up", "down")))
        self.assertGreater(len(diagonal_candidates), 0)
        self.assertLessEqual(len(diagonal_candidates), 2)
        for candidate in diagonal_candidates:
            self.assertNotEqual(candidate.clip_bbox_q[0], cluster.representative.clip_bbox_q[0])
            self.assertNotEqual(candidate.clip_bbox_q[1], cluster.representative.clip_bbox_q[1])
            self.assertGreater(candidate.shift_distance_um, 0.0)
        summary = candidate_shift_summary(candidates)
        self.assertEqual(summary["diagonal_candidate_count"], len(diagonal_candidates))
        self.assertGreater(summary["max_shift_distance_um"], 0.0)

    def test_array_representative_seed_reduces_regular_grid(self):
        """规则二维阵列应生成中心代表和间距代表，且数量受控。"""

        input_oas = self.temp_root / "array_seed.oas"
        shapes = []
        for ix in range(6):
            for iy in range(6):
                x0 = 0.1 + ix * 0.6
                y0 = 0.1 + iy * 0.6
                shapes.append(gdstk.rectangle((x0, y0), (x0 + 0.18, y0 + 0.18), layer=1, datatype=0))
        _write_oas(input_oas, shapes)
        layout_index = prepare_layout(str(input_oas), None, False)
        seeds, stats = build_uniform_grid_seed_candidates(layout_index, 1.0)
        self.assertEqual(stats["seed_strategy"], "geometry_driven")
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
        self.assertTrue(any(seed.seed_type == "array_representative" for seed in seeds))
        self.assertTrue(any(seed.seed_type == "array_spacing" for seed in seeds))
        self.assertGreaterEqual(stats["seed_weight_total"], len(shapes))

    def test_array_spacing_seed_keeps_separate_dedupe_slot(self):
        """同一 anchor 下普通 seed 和 array_spacing seed 应能各保留一个。"""

        center_seed = GridSeedCandidate((0.0, 0.0), (-0.1, -0.1, 0.1, 0.1), 3, 4, 2, "array_representative")
        spacing_seed = GridSeedCandidate((0.0, 0.0), (-0.1, -0.1, 0.1, 0.1), 3, 4, 5, "array_spacing")
        duplicate_spacing = GridSeedCandidate((0.0, 0.0), (-0.1, -0.1, 0.1, 0.1), 3, 4, 7, "array_spacing")
        deduped = _dedupe_geometry_seeds([center_seed, spacing_seed, duplicate_spacing])
        self.assertEqual(len(deduped), 2)
        type_counts = dict((seed.seed_type, seed.bucket_weight) for seed in deduped)
        self.assertEqual(type_counts["array_representative"], 2)
        self.assertEqual(type_counts["array_spacing"], 12)

    def test_long_shape_path_seed_is_one_dimensional(self):
        """长条图形应只生成一维路径 seed，避免二维 bbox 网格爆炸。"""

        input_oas = self.temp_root / "long_seed.oas"
        _write_oas(
            input_oas,
            [
                gdstk.rectangle((0.0, 0.0), (20.0, 0.4), layer=1, datatype=0),
                gdstk.rectangle((9.8, -2.0), (10.2, 2.0), layer=1, datatype=0),
            ],
        )
        layout_index = prepare_layout(str(input_oas), None, False)
        seeds, stats = build_uniform_grid_seed_candidates(layout_index, 1.0)
        self.assertEqual(stats["long_shape_count"], 2)
        self.assertGreater(stats["long_shape_seed_count"], 0)
        self.assertEqual(stats["residual_element_count"], 0)
        self.assertLess(len(seeds), 80)
        self.assertTrue(all(seed.seed_type == "long_shape_path" for seed in seeds))

    def test_simple_spatial_index_matches_bbox_query(self):
        """网格空间索引应返回与朴素 bbox 相交判断一致的元素 id。"""

        input_oas = self.temp_root / "spatial_index.oas"
        _write_oas(
            input_oas,
            [
                gdstk.rectangle((0.00, 0.00), (0.20, 0.20), layer=1, datatype=0),
                gdstk.rectangle((0.50, 0.00), (0.70, 0.20), layer=1, datatype=0),
                gdstk.rectangle((1.00, 0.00), (1.20, 0.20), layer=1, datatype=0),
            ],
        )
        layout_index = prepare_layout(str(input_oas), None, False)
        query_bbox = (0.10, -0.05, 0.80, 0.25)
        actual = [int(idx) for idx in layout_index.spatial_index.intersection(query_bbox)]
        expected = []
        for idx, item in enumerate(layout_index.indexed_elements):
            bbox = item["bbox"]
            if bbox[2] > query_bbox[0] and bbox[0] < query_bbox[2] and bbox[3] > query_bbox[1] and bbox[1] < query_bbox[3]:
                expected.append(int(idx))
        self.assertEqual(actual, expected)

    def test_run_stage_rejects_manifest_config_drift(self):
        """run stage 应拒绝 config/register-op 已漂移的 manifest。"""

        input_oas = self.temp_root / "manifest_drift.oas"
        _write_oas(
            input_oas,
            [
                gdstk.rectangle((0.05, 0.05), (0.20, 0.20), layer=1, datatype=0),
            ],
        )
        work_dir = self.temp_root / "work_manifest_drift"
        config = {
            "clip_size_um": 1.0,
            "geometry_match_mode": "ecc",
            "area_match_ratio": 0.96,
            "edge_tolerance_um": 0.02,
            "pixel_size_nm": 20,
        }
        manifest = v2_lsf.prepare_stage(str(input_oas), str(work_dir), config, [], shard_count=1, shard_size=10)
        manifest_path = work_dir / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["config"]["clip_size_um"] = 1.1
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            v2_lsf.run_shard_stage(str(manifest_path), int(manifest["shards"][0]["shard_id"]))

    def test_run_shard_uses_tile_oas_without_source_oas(self):
        """run-shard 应优先读取 prepare 生成的 tile OAS。"""

        input_oas = self.temp_root / "tile_source.oas"
        _write_oas(
            input_oas,
            [
                gdstk.rectangle((0.05, 0.05), (0.24, 0.24), layer=1, datatype=0),
                gdstk.rectangle((1.05, 0.05), (1.24, 0.24), layer=1, datatype=0),
            ],
        )
        work_dir = self.temp_root / "work_tile_source"
        config = {
            "clip_size_um": 1.0,
            "geometry_match_mode": "ecc",
            "area_match_ratio": 0.96,
            "edge_tolerance_um": 0.02,
            "pixel_size_nm": 20,
        }
        manifest = v2_lsf.prepare_stage(str(input_oas), str(work_dir), config, [], shard_count=1, shard_size=10)
        manifest_path = work_dir / "manifest.json"
        backup_path = input_oas.with_suffix(".bak")
        input_oas.rename(backup_path)

        summary = v2_lsf.run_shard_stage(str(manifest_path), int(manifest["shards"][0]["shard_id"]))

        self.assertEqual(summary["layout_load_mode"], "tile_oas")
        self.assertFalse(summary["layout_apply_layer_operations"])
        self.assertTrue(Path(summary["tile_oas"]).exists())
        self.assertGreater(summary["tile_oas_bytes"], 0)
        self.assertGreaterEqual(summary["marker_count"], 0)
        self.assertTrue(Path(manifest["shards"][0]["output_json"]).exists())

    def test_run_shard_and_merge_outputs_result(self):
        """run-shard 产物应可被 merge 汇总成 compact result。"""

        input_oas = self.temp_root / "merge.oas"
        _write_oas(
            input_oas,
            [
                gdstk.rectangle((0.05, 0.05), (0.22, 0.22), layer=1, datatype=0),
                gdstk.rectangle((1.05, 0.05), (1.22, 0.22), layer=1, datatype=0),
                gdstk.rectangle((2.05, 0.05), (2.22, 0.22), layer=1, datatype=0),
            ],
        )
        work_dir = self.temp_root / "work_merge"
        output = self.temp_root / "merge_result.json"
        config = {
            "clip_size_um": 1.0,
            "geometry_match_mode": "ecc",
            "area_match_ratio": 0.96,
            "edge_tolerance_um": 0.02,
            "pixel_size_nm": 20,
        }
        manifest = v2_lsf.prepare_stage(str(input_oas), str(work_dir), config, [], shard_count=1, shard_size=1)
        manifest_path = work_dir / "manifest.json"
        for shard in manifest["shards"]:
            summary = v2_lsf.run_shard_stage(str(manifest_path), int(shard["shard_id"]))
            self.assertGreaterEqual(summary["marker_count"], 0)
            self.assertTrue(Path(shard["output_json"]).exists())
            self.assertTrue(Path(shard["output_npz"]).exists())
            self.assertIn("candidate_summaries", summary)
            self.assertEqual(summary["shard_payload_mode"], "marker_records_only")
            self.assertEqual(summary["local_candidate_count"], 0)
            self.assertTrue(summary["local_coverage_debug_stats"]["skipped"])
            self.assertIn("spatial_index_stats", summary)
            self.assertIn("query_candidate_count_stats", summary)
            self.assertGreaterEqual(summary["query_candidate_count_stats"]["query_candidate_count_max"], 0)
            self.assertEqual(summary["layout_load_mode"], "tile_oas")
            self.assertFalse(summary["layout_apply_layer_operations"])
        result = v2_lsf.merge_stage(str(manifest_path), str(output))
        self.assertTrue(output.exists())
        self.assertEqual(result["pipeline_mode"], v2_lsf.PIPELINE_MODE)
        self.assertEqual(result["seed_strategy"], "geometry_driven")
        self.assertGreater(result["marker_count"], 0)
        self.assertGreater(result["exact_cluster_count"], 0)
        self.assertGreater(result["candidate_count"], 0)
        self.assertGreater(result["total_clusters"], 0)
        self.assertTrue(all("distance_worst_case_score" in cluster for cluster in result["clusters"]))
        self.assertTrue(all(cluster["distance_worst_case_score"] >= 0.0 for cluster in result["clusters"]))
        self.assertIn("lsf_manifest", result)

    def test_distributed_coverage_matches_central_merge(self):
        """coverage 分片流程应与集中 merge 保持核心结果一致。"""

        input_oas = self.temp_root / "coverage.oas"
        _write_oas(
            input_oas,
            [
                gdstk.rectangle((0.05, 0.05), (0.24, 0.24), layer=1, datatype=0),
                gdstk.rectangle((1.05, 0.05), (1.24, 0.24), layer=1, datatype=0),
                gdstk.rectangle((2.05, 0.05), (2.24, 0.24), layer=1, datatype=0),
                gdstk.rectangle((3.05, 0.05), (3.22, 0.22), layer=1, datatype=0),
            ],
        )
        work_dir = self.temp_root / "work_coverage"
        baseline_output = self.temp_root / "coverage_baseline.json"
        distributed_output = self.temp_root / "coverage_distributed.json"
        config = {
            "clip_size_um": 1.0,
            "geometry_match_mode": "ecc",
            "area_match_ratio": 0.96,
            "edge_tolerance_um": 0.02,
            "pixel_size_nm": 20,
        }
        manifest = v2_lsf.prepare_stage(str(input_oas), str(work_dir), config, [], shard_count=2, shard_size=1)
        manifest_path = work_dir / "manifest.json"
        for shard in manifest["shards"]:
            v2_lsf.run_shard_stage(str(manifest_path), int(shard["shard_id"]))

        baseline = v2_lsf.merge_stage(str(manifest_path), str(baseline_output))
        manifest = v2_lsf.prepare_coverage_stage(str(manifest_path), coverage_shard_count=2, coverage_shard_size=1)
        self.assertGreater(manifest["coverage_shard_count"], 0)
        self.assertTrue(Path(manifest["exact_index"]["output_json"]).exists())
        self.assertTrue(Path(manifest["exact_index"]["output_npz"]).exists())
        self.assertGreater(manifest["exact_target_buckets"]["bucket_count"], 0)
        self.assertIn("candidate_bundle_index", manifest)
        self.assertGreater(manifest["candidate_bundle_index"]["candidate_group_count"], 0)
        coverage_plan = manifest["coverage_plan"]
        coverage_timing = coverage_plan["timing_seconds"]
        for timing_key in (
            "prepare_coverage_marker_load",
            "prepare_coverage_exact_cluster",
            "prepare_coverage_source_sort",
            "prepare_coverage_exact_index_write",
            "prepare_coverage_target_bucket_write",
            "prepare_coverage_candidate_generation",
            "prepare_coverage_candidate_bundle_write",
            "prepare_coverage_source_shard_write",
            "prepare_coverage",
        ):
            self.assertIn(timing_key, coverage_timing)
            self.assertGreaterEqual(coverage_timing[timing_key], 0.0)
        self.assertIn("candidate_bundle_bucket_count", coverage_plan)
        self.assertIn("candidate_bundle_split_mode", coverage_plan)
        self.assertIn("candidate_chunk_size", coverage_plan)
        self.assertEqual(coverage_plan["coverage_source_partition_mode"], "fill_bin_grouped")
        self.assertIn("coverage_source_fill_bin_group_count", coverage_plan)
        self.assertGreaterEqual(coverage_plan["coverage_source_fill_bin_group_count"], 0)
        self.assertIn("max_source_fill_bin_count_per_shard", coverage_plan)
        self.assertGreaterEqual(coverage_plan["max_source_fill_bin_count_per_shard"], 0)
        self.assertIn("candidate_direction_counts", coverage_plan)
        self.assertIn("diagonal_candidate_count", coverage_plan)
        self.assertGreaterEqual(coverage_plan["diagonal_candidate_count"], 0)
        self.assertIn("max_shift_distance_um", coverage_plan)
        self.assertGreaterEqual(coverage_plan["max_shift_distance_um"], 0.0)
        self.assertIn("input_file_bytes", coverage_plan)
        self.assertGreater(coverage_plan["input_file_bytes"], 0)
        self.assertIn("lsf_wrapper", manifest)
        self.assertIn("run_coverage_shards", manifest["lsf_wrapper"])
        self.assertIn("merge_coverage", manifest["lsf_wrapper"])
        run_coverage_wrapper = manifest["lsf_wrapper"]["run_coverage_shards"]
        self.assertEqual(run_coverage_wrapper["command_count"], manifest["coverage_shard_count"])
        self.assertTrue(Path(run_coverage_wrapper["command_file"]).exists())
        self.assertTrue(Path(run_coverage_wrapper["bsub_template"]).exists())
        self.assertTrue(Path(manifest["lsf_wrapper"]["merge_coverage"]["command_file"]).exists())
        self.assertTrue(all("run-coverage-shard" in shard["command"] for shard in manifest["coverage_shards"]))
        self.assertTrue(all("source_index_json" in shard for shard in manifest["coverage_shards"]))
        self.assertTrue(all("source_fill_bin_count" in shard for shard in manifest["coverage_shards"]))
        self.assertTrue(all(shard["source_fill_bin_count"] <= 1 for shard in manifest["coverage_shards"]))
        for shard in manifest["shards"]:
            Path(shard["output_npz"]).unlink()
        Path(manifest["exact_index"]["output_json"]).unlink()
        Path(manifest["exact_index"]["output_npz"]).unlink()
        for coverage_shard in manifest["coverage_shards"]:
            summary = v2_lsf.run_coverage_shard_stage(str(manifest_path), int(coverage_shard["coverage_shard_id"]))
            self.assertTrue(Path(coverage_shard["output_json"]).exists())
            self.assertTrue(Path(coverage_shard["output_npz"]).exists())
            self.assertGreaterEqual(summary["candidate_count"], 0)
            self.assertIn("candidate_direction_counts", summary)
            self.assertIn("diagonal_candidate_count", summary)
            self.assertGreaterEqual(summary["diagonal_candidate_count"], 0)
            self.assertGreater(summary["target_bucket_count_loaded"], 0)
            self.assertIn("target_candidate_group_load_ratio", summary)
            self.assertGreaterEqual(summary["target_candidate_group_load_ratio"], 0.0)
            self.assertLessEqual(summary["target_candidate_group_load_ratio"], 1.0)
            self.assertIn("target_load_warning", summary)
            self.assertIn("source_fill_bin_count", summary)
            self.assertGreaterEqual(summary["source_fill_bin_count"], 0)
            self.assertEqual(summary["source_fill_bin_count"], coverage_shard["source_fill_bin_count"])
            self.assertIn("candidate_fill_bin_count", summary)
            self.assertGreaterEqual(summary["candidate_fill_bin_count"], summary["source_fill_bin_count"])
            raw_payload = json.loads(Path(coverage_shard["output_json"]).read_text(encoding="utf-8"))
            self.assertEqual(raw_payload["coverage_storage"], "npz_offsets_v1")
            self.assertEqual(raw_payload["source_fill_bin_count"], coverage_shard["source_fill_bin_count"])
            self.assertTrue(all("coverage" not in candidate for candidate in raw_payload["candidates"]))
            metadata_candidates, _ = load_coverage_shard_metadata(
                coverage_shard["output_json"],
                coverage_shard["output_npz"],
            )
            self.assertTrue(all(candidate.clip_bitmap is None for candidate in metadata_candidates))
            self.assertTrue(any(candidate.coverage for candidate in metadata_candidates))

        distributed = v2_lsf.merge_coverage_stage(str(manifest_path), str(distributed_output))
        self.assertEqual(distributed["total_clusters"], baseline["total_clusters"])
        self.assertEqual(distributed["exact_cluster_count"], baseline["exact_cluster_count"])
        self.assertEqual(distributed["candidate_count"], baseline["candidate_count"])
        self.assertEqual(distributed["selected_candidate_count"], baseline["selected_candidate_count"])
        self.assertIn("candidate_direction_counts", distributed)
        self.assertIn("diagonal_candidate_count", distributed)
        self.assertGreaterEqual(distributed["diagonal_candidate_count"], 0)
        self.assertIn("selected_diagonal_candidate_count", distributed)
        self.assertGreaterEqual(distributed["selected_diagonal_candidate_count"], 0)
        self.assertTrue(all("distance_worst_case_score" in cluster for cluster in distributed["clusters"]))
        self.assertTrue(all(cluster["distance_worst_case_score"] >= 0.0 for cluster in distributed["clusters"]))
        self.assertEqual(
            distributed["coverage_debug_stats"]["geometry_pair_count"],
            baseline["coverage_debug_stats"]["geometry_pair_count"],
        )
        self.assertIn("exact_hash_pairs", distributed["coverage_debug_stats"])
        self.assertIn("cheap_reject", distributed["coverage_debug_stats"])
        self.assertIn("full_prefilter_reject", distributed["coverage_debug_stats"])
        self.assertIn("coverage_detail_seconds", distributed)
        self.assertTrue(all(value >= 0.0 for value in distributed["coverage_detail_seconds"].values()))
        self.assertEqual(distributed["coverage_debug_stats"]["candidate_bitmap_preload_count"], 0)
        self.assertEqual(distributed["coverage_debug_stats"]["candidate_object_preload_count"], 0)
        self.assertGreater(distributed["coverage_debug_stats"]["coverage_csr_edge_count"], 0)
        self.assertEqual(
            distributed["coverage_debug_stats"]["selected_bitmap_load_count"],
            distributed["selected_candidate_count"],
        )
        self.assertEqual(distributed["lsf_manifest"]["coverage_shard_count"], manifest["coverage_shard_count"])
        self.assertTrue(distributed_output.exists())

        inspect_output = self.temp_root / "coverage_inspect.json"
        inspection = v2_lsf.inspect_workdir_stage(str(manifest_path), str(inspect_output))
        self.assertTrue(inspect_output.exists())
        self.assertEqual(inspection["coverage_shards"]["candidate_count"], distributed["candidate_count"])
        self.assertEqual(
            inspection["coverage_shards"]["coverage_value_count"],
            distributed["coverage_debug_stats"]["coverage_csr_edge_count"],
        )
        self.assertIn("target_candidate_group_load_ratio_avg", inspection["coverage_shards"])
        self.assertGreaterEqual(inspection["coverage_shards"]["target_candidate_group_load_ratio_avg"], 0.0)
        self.assertIn("target_candidate_group_load_ratio_max", inspection["coverage_shards"])
        self.assertGreaterEqual(inspection["coverage_shards"]["target_candidate_group_load_ratio_max"], 0.0)
        self.assertIn("candidate_fill_bin_count_max", inspection["coverage_shards"])
        self.assertGreaterEqual(inspection["coverage_shards"]["candidate_fill_bin_count_max"], 0)
        self.assertIn("tile_oas_bytes", inspection["shards"])
        self.assertGreater(inspection["shards"]["tile_oas_bytes"], 0)
        self.assertIn("lsf_wrapper", inspection)
        self.assertIn("run_coverage_shards", inspection["lsf_wrapper"])
        self.assertGreater(inspection["coverage_shards"]["npz_zip_uncompressed_bytes"], 0)
        self.assertGreater(len(inspection["largest_files"]), 0)

    def test_run_local_small_sample(self):
        """run-local 应能顺序模拟完整 LSF 流程。"""

        input_oas = self.temp_root / "local.oas"
        _write_oas(
            input_oas,
            [
                gdstk.rectangle((0.05, 0.05), (0.25, 0.25), layer=1, datatype=0),
                gdstk.rectangle((0.55, 0.05), (0.75, 0.25), layer=1, datatype=0),
            ],
        )
        work_dir = self.temp_root / "work_local"
        output = self.temp_root / "local_result.json"
        config = {
            "clip_size_um": 1.0,
            "geometry_match_mode": "ecc",
            "area_match_ratio": 0.96,
            "edge_tolerance_um": 0.02,
            "pixel_size_nm": 20,
        }
        result = v2_lsf.run_local_stage(str(input_oas), str(work_dir), str(output), config, [], shard_count=2, shard_size=1)
        loaded = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(loaded["pipeline_mode"], v2_lsf.PIPELINE_MODE)
        self.assertEqual(result["total_clusters"], loaded["total_clusters"])
        self.assertGreaterEqual(loaded["selected_candidate_count"], 1)
        self.assertTrue(all("distance_worst_case_score" in cluster for cluster in loaded["clusters"]))
        self.assertTrue(all(cluster["distance_worst_case_score"] >= 0.0 for cluster in loaded["clusters"]))
        self.assertNotIn("contact_pair_seed_count", loaded)
        self.assertNotIn("drc_component_seed_count", loaded)
        distributed_output = self.temp_root / "local_distributed_result.json"
        distributed = v2_lsf.run_local_stage(
            str(input_oas),
            str(self.temp_root / "work_local_distributed"),
            str(distributed_output),
            config,
            [],
            shard_count=2,
            shard_size=1,
            distributed_coverage=True,
            coverage_shard_count=2,
            coverage_shard_size=1,
        )
        self.assertEqual(distributed["total_clusters"], loaded["total_clusters"])
        self.assertGreater(distributed["lsf_manifest"]["coverage_shard_count"], 0)

    def test_layer_operation_lsf_path(self):
        """LSF layer operation 应保留 result layer 并排除 helper-only layer。"""

        input_oas = self.temp_root / "layer_ops.oas"
        _write_oas(
            input_oas,
            [
                gdstk.rectangle((0.00, 0.00), (0.40, 0.20), layer=10, datatype=0),
                gdstk.rectangle((0.18, -0.02), (0.30, 0.22), layer=11, datatype=0),
            ],
        )
        work_dir = self.temp_root / "work_layer_ops"
        output = self.temp_root / "layer_ops_result.json"
        config = {
            "clip_size_um": 1.0,
            "geometry_match_mode": "ecc",
            "area_match_ratio": 0.96,
            "edge_tolerance_um": 0.02,
            "pixel_size_nm": 20,
            "apply_layer_operations": True,
        }
        result = v2_lsf.run_local_stage(
            str(input_oas),
            str(work_dir),
            str(output),
            config,
            [["10/0", "11/0", "subtract", "13/0"]],
            shard_count=1,
            shard_size=10,
        )
        self.assertGreater(result["marker_count"], 0)
        self.assertGreater(result["total_clusters"], 0)
        manifest = json.loads((work_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["apply_layer_operations"])
        self.assertEqual(manifest["register_ops"], [["10/0", "11/0", "subtract", "13/0"]])
        self.assertEqual(manifest["effective_pattern_layers"], [[13, 0]])
        self.assertIn([10, 0], manifest["excluded_helper_layers"])
        self.assertIn([11, 0], manifest["excluded_helper_layers"])
        records, payload = load_shard_records(
            manifest["shards"][0]["output_json"],
            manifest["shards"][0]["output_npz"],
        )
        self.assertGreater(len(records), 0)
        self.assertTrue(payload["apply_layer_operations"])
        self.assertEqual(payload["registered_layer_operations"], [["10/0", "11/0", "subtract", "13/0"]])
        self.assertEqual(payload["effective_pattern_layers"], [[13, 0]])
        self.assertGreaterEqual(payload["layout_element_count"], payload["halo_filtered_element_count"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
