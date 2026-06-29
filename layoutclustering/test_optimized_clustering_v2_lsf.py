#!/usr/bin/env python3
"""Tests for the Python 3.12 LSF v2 clustering pipeline."""

import ast
import csv
import json
import shutil
import sys
import unittest
from unittest import mock
from pathlib import Path

import gdstk
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import layout_clustering_optimized_v2_lsf as v2_lsf
import mainline_lsf
from layout_utils_lsf import _write_oas_library
from mainline_lsf import CandidateClip
from mainline_lsf import ExactCluster
from mainline_lsf import GridSeedCandidate
from mainline_lsf import MarkerRecord
from mainline_lsf import _canonical_bitmap_hash
from mainline_lsf import _candidate_matches_exact
from mainline_lsf import _candidate_matches_witness
from mainline_lsf import _dedupe_geometry_seeds
from mainline_lsf import add_candidates_to_candidate_bundle_accumulator
from mainline_lsf import build_geometry_driven_seed_candidates
from mainline_lsf import candidate_shift_summary
from mainline_lsf import create_candidate_bundle_accumulator
from mainline_lsf import evaluate_candidate_coverage
from mainline_lsf import generate_candidates_for_cluster
from mainline_lsf import prepare_layout
from mainline_lsf import load_candidate_bundle_buckets_for_candidates
from mainline_lsf import load_coverage_shard_csr_metadata
from mainline_lsf import load_shard_records
from mainline_lsf import save_candidate_bundle_index
from mainline_lsf import save_candidate_bundle_index_from_accumulator


LSF_FILES = [
    SCRIPT_DIR / "layout_clustering_optimized_v2_lsf.py",
    SCRIPT_DIR / "mainline_lsf.py",
    SCRIPT_DIR / "layout_utils_lsf.py",
    SCRIPT_DIR / "layer_operations_lsf.py",
]

PACKAGE_LIST_FILE = SCRIPT_DIR / "Python 3.12 supporting packages.txt"


def _write_oas(path, polygons):
    """写出最小 OAS fixture。"""

    lib = gdstk.Library()
    cell = gdstk.Cell("TOP")
    for poly in polygons:
        cell.add(poly)
    lib.add(cell)
    _write_oas_library(lib, str(path))


def _make_candidate(candidate_id, origin_exact_cluster_id, bitmap, shift_direction="base", origin_seed_type="unknown"):
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
        origin_seed_type=str(origin_seed_type),
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


def _make_exact_cluster(cluster_id, bitmap, seed_type=mainline_lsf.SEED_TYPE_ARRAY):
    """构造 quality/safe-recall 单测使用的 exact cluster。"""

    clip_bitmap = np.ascontiguousarray(bitmap, dtype=bool)
    clip_hash, _ = _canonical_bitmap_hash(clip_bitmap)
    record = MarkerRecord(
        marker_id="marker_%s" % int(cluster_id),
        source_path="synthetic.oas",
        source_name="synthetic.oas",
        marker_bbox=(0.0, 0.0, 1.0, 1.0),
        marker_center=(0.5 + float(cluster_id), 0.5),
        clip_bbox=(0.0, 0.0, 1.0, 1.0),
        expanded_bbox=(0.0, 0.0, 1.0, 1.0),
        clip_bbox_q=(0, 0, int(clip_bitmap.shape[1]), int(clip_bitmap.shape[0])),
        expanded_bbox_q=(0, 0, int(clip_bitmap.shape[1]), int(clip_bitmap.shape[0])),
        marker_bbox_q=(0, 0, int(clip_bitmap.shape[1]), int(clip_bitmap.shape[0])),
        shift_limits_px={"x": (0, 0), "y": (0, 0)},
        clip_bitmap=clip_bitmap,
        expanded_bitmap=clip_bitmap,
        clip_hash=clip_hash,
        expanded_hash=clip_hash,
        clip_area=float(np.count_nonzero(clip_bitmap)),
        seed_weight=1,
        exact_cluster_id=int(cluster_id),
        metadata={"seed_type": str(seed_type)},
    )
    return ExactCluster(int(cluster_id), "exact_%s" % int(cluster_id), record, [record])


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


def _read_csv_rows(path):
    """读取测试输出 CSV。"""

    with Path(str(path)).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class OptimizedV2LsfTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = SCRIPT_DIR / "test_outputs" / "_optimized_v2_lsf"
        shutil.rmtree(str(self.temp_root), ignore_errors=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)

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

    def test_lsf_scripts_parse_as_python312(self):
        """LSF 脚本需要在目标 Python 3.12 语法下可解析。"""

        for path in LSF_FILES:
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path), feature_version=(3, 12))

    def test_lsf_runtime_imports_are_in_python312_package_list(self):
        """v2_lsf 第三方运行依赖必须来自 Python 3.12 环境清单。"""

        package_names = set()
        for raw_line in PACKAGE_LIST_FILE.read_text(encoding="utf-8").splitlines():
            parts = raw_line.split()
            if len(parts) >= 2 and not raw_line.startswith("Python") and not raw_line.startswith("Package"):
                package_names.add(parts[0].lower().replace("_", "-"))

        module_to_package = {
            "gdstk": "gdstk",
            "numpy": "numpy",
            "scipy": "scipy",
            "sklearn": "scikit-learn",
        }
        local_modules = {path.stem for path in LSF_FILES}
        imported_roots = set()
        for path in LSF_FILES:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 12))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
                if isinstance(node, ast.ImportFrom) and node.module:
                    imported_roots.add(node.module.split(".", 1)[0])

        external_roots = {
            root
            for root in imported_roots
            if root not in sys.stdlib_module_names and root not in local_modules and not root.startswith("_")
        }
        missing = sorted(
            root
            for root in external_roots
            if module_to_package.get(root, root).lower().replace("_", "-") not in package_names
        )
        self.assertEqual([], missing)

    def test_final_output_cli_is_csv_only(self):
        """最终主输出 CLI 不再接受旧 JSON/TXT 或 --format 入口。"""

        parser = v2_lsf.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["run-local", "input.oas", "--work-dir", "work", "--output", "result.json"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["run-local", "input.oas", "--work-dir", "work", "--output", "result.csv", "--format", "json"])
        args = parser.parse_args(
            ["run-local", "input.oas", "--work-dir", "work", "--output", "result.csv", "--compute-quality-metrics"]
        )
        self.assertTrue(args.compute_quality_metrics)
        merge_args = parser.parse_args(
            ["merge", "--manifest", "manifest.json", "--output", "result.csv", "--compute-quality-metrics"]
        )
        self.assertTrue(merge_args.compute_quality_metrics)
        coverage_args = parser.parse_args(
            ["merge-coverage", "--manifest", "manifest.json", "--output", "result.csv", "--compute-quality-metrics"]
        )
        self.assertTrue(coverage_args.compute_quality_metrics)

    def test_config_json_rejects_removed_keys(self):
        """config JSON 不再接受旧方案字段或未知字段。"""

        parser = v2_lsf.build_parser()
        removed_config = self.temp_root / "removed_config.json"
        removed_config.write_text(
            json.dumps({"clip_size_um": 1.0, "strict_signature_threshold": 0.7, "grid_step_ratio": 0.7}, ensure_ascii=False),
            encoding="utf-8",
        )
        removed_args = parser.parse_args(
            [
                "run-local",
                "input.oas",
                "--config",
                str(removed_config),
                "--work-dir",
                "work",
                "--output",
                "result.csv",
            ]
        )
        with self.assertRaisesRegex(ValueError, "strict_signature_threshold"):
            v2_lsf._config_payload(removed_args)

        unknown_config = self.temp_root / "unknown_config.json"
        unknown_config.write_text(json.dumps({"clip_size_um": 1.0, "unused_knob": True}, ensure_ascii=False), encoding="utf-8")
        unknown_args = parser.parse_args(
            [
                "prepare",
                "input.oas",
                "--config",
                str(unknown_config),
                "--work-dir",
                "work",
            ]
        )
        with self.assertRaisesRegex(ValueError, "unused_knob"):
            v2_lsf._config_payload(unknown_args)

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

    def test_legacy_candidate_group_csv_writer_removed(self):
        """v2_lsf 主 CSV 只保留 exact cluster review writer，不保留旧 candidate group writer。"""

        source = (SCRIPT_DIR / "mainline_lsf.py").read_text(encoding="utf-8")
        tree = ast.parse(source, filename="mainline_lsf.py")
        function_names = set(node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        self.assertNotIn("_csv_candidate_group_row", function_names)
        self.assertNotIn("_candidate_group_representatives_from_candidates", function_names)
        self.assertNotIn("_candidate_group_representatives_from_index", function_names)
        self.assertIn("write_result_csv", function_names)
        self.assertNotIn("write_result_csv_exact_review", function_names)
        self.assertNotIn("candidate group 粒度写成 5 列主 CSV", source)

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

    def test_donut_degenerate_coverage_requires_strict_graph(self):
        """donut 退化匹配不应直接 auto-pass，应进入 strict graph 过滤并记录统计。"""

        bitmap_source = np.zeros((8, 8), dtype=bool)
        bitmap_source[2:6, 2:6] = True
        bitmap_target = bitmap_source.copy()
        bitmap_target[0, 0] = True
        cand_source = _make_candidate("cand_source", 0, bitmap_source, origin_seed_type=mainline_lsf.SEED_TYPE_RESIDUAL)
        cand_target = _make_candidate("cand_target", 1, bitmap_target, origin_seed_type=mainline_lsf.SEED_TYPE_RESIDUAL)
        bundle = next(iter(mainline_lsf.build_candidate_match_bundles([cand_source, cand_target]).values()))
        source_idx = next(idx for idx, candidate in enumerate(bundle["representatives"]) if candidate.candidate_id == "cand_source")
        target_idx = next(idx for idx, candidate in enumerate(bundle["representatives"]) if candidate.candidate_id == "cand_target")
        detail_seconds = mainline_lsf._empty_coverage_detail_seconds()
        debug_stats = mainline_lsf._empty_coverage_debug_stats()

        def fake_values(bundle_arg, indices, tol_px, key, detail_arg, debug_arg):
            del bundle_arg, tol_px, detail_arg, debug_arg
            if key == "dilated_area_px":
                return np.full(int(len(indices)), 100, dtype=np.int64)
            if key == "donut_area_px":
                return np.zeros(int(len(indices)), dtype=np.int64)
            raise AssertionError(key)

        def fake_matrix(bundle_arg, indices, tol_px, key, detail_arg, debug_arg):
            del bundle_arg, tol_px, detail_arg, debug_arg
            width = int(np.packbits(np.zeros((8, 8), dtype=np.uint8).reshape(-1)).size)
            if key == "packed_dilated":
                return np.full((int(len(indices)), width), 255, dtype=np.uint8)
            if key == "packed":
                return np.zeros((int(len(indices)), width), dtype=np.uint8)
            raise AssertionError(key)

        with mock.patch.object(mainline_lsf, "_bundle_geometry_values", side_effect=fake_values):
            with mock.patch.object(mainline_lsf, "_bundle_geometry_matrix", side_effect=fake_matrix):
                with mock.patch.object(mainline_lsf, "_strict_graph_gate_reason", return_value=(False, "graph_signature")):
                    matched = mainline_lsf._matched_target_indices(
                        bundle["candidate_groups"][int(source_idx)],
                        bundle,
                        np.asarray([int(target_idx)], dtype=np.int64),
                        {
                            "geometry_match_mode": "ecc",
                            "area_match_ratio": 0.96,
                            "edge_tolerance_um": 0.02,
                            "pixel_size_nm": 20,
                        },
                        1,
                        detail_seconds,
                        debug_stats,
                    )

        self.assertEqual(matched.size, 0)
        self.assertEqual(debug_stats["donut_auto_pass_pair_count"], 1)
        self.assertEqual(debug_stats["donut_degenerate_strict_graph_reject_count"], 1)

    def test_long_shape_cross_seed_guard_requires_high_signature(self):
        """long_shape_path 跨 seed type 覆盖时应额外要求更高 signature 相似度。"""

        bitmap_long = np.zeros((10, 10), dtype=bool)
        bitmap_long[4:6, 1:9] = True
        bitmap_array = np.zeros((10, 10), dtype=bool)
        bitmap_array[2:8, 2:8] = True
        cand_long = _make_candidate("cand_long", 0, bitmap_long, origin_seed_type=mainline_lsf.SEED_TYPE_LONG)
        cand_array = _make_candidate("cand_array", 1, bitmap_array, origin_seed_type=mainline_lsf.SEED_TYPE_ARRAY)
        bundle = next(iter(mainline_lsf.build_candidate_match_bundles([cand_long, cand_array]).values()))
        source_idx = next(idx for idx, candidate in enumerate(bundle["representatives"]) if candidate.candidate_id == "cand_long")
        target_idx = next(idx for idx, candidate in enumerate(bundle["representatives"]) if candidate.candidate_id == "cand_array")
        detail_seconds = mainline_lsf._empty_coverage_detail_seconds()
        debug_stats = mainline_lsf._empty_coverage_debug_stats()

        with mock.patch.object(mainline_lsf, "_strict_graph_gate_reason", return_value=(True, "pass")):
            with mock.patch.object(mainline_lsf, "_graph_signature_similarity", return_value=0.50):
                rejected = mainline_lsf._apply_long_shape_cross_seed_guard(
                    bundle["candidate_groups"][int(source_idx)],
                    bundle,
                    np.asarray([int(target_idx)], dtype=np.int64),
                    detail_seconds,
                    debug_stats,
                )
        self.assertEqual(rejected.size, 0)
        self.assertEqual(debug_stats["long_shape_cross_seed_guard_reject_count"], 1)

        debug_stats = mainline_lsf._empty_coverage_debug_stats()
        with mock.patch.object(mainline_lsf, "_strict_graph_gate_reason", return_value=(True, "pass")):
            with mock.patch.object(mainline_lsf, "_graph_signature_similarity", return_value=0.95):
                accepted = mainline_lsf._apply_long_shape_cross_seed_guard(
                    bundle["candidate_groups"][int(source_idx)],
                    bundle,
                    np.asarray([int(target_idx)], dtype=np.int64),
                    detail_seconds,
                    debug_stats,
                )
        self.assertEqual(accepted.tolist(), [int(target_idx)])
        self.assertEqual(debug_stats["long_shape_cross_seed_guard_pass_count"], 1)

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
        try:
            self.assertIn("packed_bitmaps", arrays.files)
            self.assertIn("cheap_invariants", arrays.files)
            self.assertIn("cheap_signature_vectors", arrays.files)
            self.assertIn("cheap_subgroup_keys", arrays.files)
        finally:
            arrays.close()

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
        self.assertIn("seed_coverage_audit", manifest)
        self.assertIn("target_edge_length_coverage_ratio", manifest["seed_coverage_audit"])
        self.assertIn("target_polygon_area_coverage_ratio", manifest["seed_coverage_audit"])
        self.assertIn("weighted_pattern_type_coverage_ratio", manifest["seed_coverage_audit"])
        self.assertNotIn("clip_window_union_coverage_ratio", manifest["seed_coverage_audit"])
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

    def test_seed_coverage_audit_uses_target_pattern_metrics(self):
        """coverage audit 应按 target 边长、面积和 pattern type 权重统计。"""

        input_oas = self.temp_root / "coverage_audit.oas"
        _write_oas(
            input_oas,
            [
                gdstk.rectangle((0.0, 0.0), (1.0, 1.0), layer=1, datatype=0),
                gdstk.rectangle((2.0, 0.0), (3.0, 1.0), layer=1, datatype=0),
                gdstk.rectangle((4.0, 0.0), (4.4, 0.2), layer=1, datatype=0),
            ],
        )
        layout_index = prepare_layout(str(input_oas))
        seeds = [
            GridSeedCandidate((0.5, 0.5), (0.0, 0.0, 1.0, 1.0), 0, 0, seed_type="unit"),
            GridSeedCandidate((0.6, 0.5), (0.1, 0.0, 1.1, 1.0), 1, 0, seed_type="overlap"),
            GridSeedCandidate((2.0, 0.5), (1.5, 0.0, 2.5, 1.0), 2, 0, seed_type="partial"),
            GridSeedCandidate((3.5, 0.5), (3.0, 0.0, 4.0, 1.0), 3, 0, seed_type="tiny_touch"),
        ]
        audit = mainline_lsf._seed_coverage_audit(layout_index, seeds, 1.0, 0.5)

        self.assertIn("target_edge_length_coverage_ratio", audit)
        self.assertIn("target_polygon_area_coverage_ratio", audit)
        self.assertIn("weighted_pattern_type_coverage_ratio", audit)
        self.assertNotIn("clip_window_union_coverage_ratio", audit)
        self.assertLessEqual(audit["target_polygon_area_covered"], audit["target_polygon_area_total"])
        self.assertGreater(audit["target_polygon_area_coverage_ratio"], 0.0)
        self.assertLess(audit["target_polygon_area_coverage_ratio"], 1.0)
        self.assertGreaterEqual(audit["covered_pattern_type_count"], 1)

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
        previous_limit = v2_lsf.CENTRAL_MERGE_EXACT_CLUSTER_LIMIT
        try:
            v2_lsf.CENTRAL_MERGE_EXACT_CLUSTER_LIMIT = 0
            with self.assertRaises(RuntimeError) as ctx:
                v2_lsf.merge_stage(str(manifest_path), str(self.temp_root / "merge_limit.csv"))
        finally:
            v2_lsf.CENTRAL_MERGE_EXACT_CLUSTER_LIMIT = previous_limit
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
        default_seeds, default_stats = build_geometry_driven_seed_candidates(layout_index, 1.0)
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

    def test_seed_json_requires_current_fields(self):
        """seed JSON 必须携带当前 geometry-driven 字段。"""

        seed = GridSeedCandidate((1.0, 2.0), (0.5, 1.5, 1.5, 2.5), 3, 4, 7, "array_representative")
        restored = GridSeedCandidate.from_json(seed.to_json())
        self.assertEqual(restored.seed_type, "array_representative")
        self.assertEqual(restored.bucket_weight, 7)
        payload = seed.to_json()
        del payload["seed_type"]
        with self.assertRaises(KeyError):
            GridSeedCandidate.from_json(payload)

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

    def test_final_verification_accepts_shift_witness_match(self):
        """final verification 应匹配 target shift witness，而不是只比较 target base。"""

        cluster = _make_shiftable_exact_cluster()
        config = {"pixel_size_nm": 10, "edge_tolerance_um": 0.0, "geometry_match_mode": "ecc"}
        witnesses = generate_candidates_for_cluster(cluster, config)
        base = [candidate for candidate in witnesses if str(candidate.shift_direction) == "base"][0]
        shifted = [candidate for candidate in witnesses if str(candidate.shift_direction) != "base" and candidate.clip_hash != base.clip_hash][0]
        selected = _make_candidate("selected_shift", 99, np.ascontiguousarray(shifted.clip_bitmap, dtype=bool), "right")
        matched, reason, witness_direction = _candidate_matches_exact(selected, cluster, config, {})

        self.assertTrue(matched)
        self.assertEqual(reason, "exact_hash")
        self.assertEqual(witness_direction, shifted.shift_direction)

    def test_final_verification_reject_reason_is_detailed(self):
        """所有 witness 都失败时应输出细分几何原因，不再输出粗粒度 geometry。"""

        cluster = _make_shiftable_exact_cluster()
        empty = np.zeros_like(cluster.representative.clip_bitmap, dtype=bool)
        selected = _make_candidate("selected_empty", 99, empty, "right")
        config = {
            "pixel_size_nm": 10,
            "edge_tolerance_um": 0.01,
            "geometry_match_mode": "acc",
            "area_match_ratio": 0.99,
        }
        matched, reason, _ = _candidate_matches_exact(selected, cluster, config, {})

        self.assertFalse(matched)
        self.assertEqual(reason, "geometry_acc_xor")
        self.assertNotEqual(reason, "geometry")

    def test_final_verification_strict_graph_gate_rejects_after_geometry(self):
        """geometry 通过后仍必须通过内部 strict graph gate。"""

        candidate_bitmap = np.zeros((8, 8), dtype=bool)
        candidate_bitmap[0:2, 0:2] = True
        witness_bitmap = np.zeros((8, 8), dtype=bool)
        witness_bitmap[2:7, 2:7] = True
        candidate = _make_candidate("selected_graph", 0, candidate_bitmap, "right")
        witness = _make_candidate("witness_graph", 1, witness_bitmap, "base")
        matched, reason = _candidate_matches_witness(
            candidate,
            witness,
            {"geometry_match_mode": "acc", "area_match_ratio": 0.0},
            {},
        )

        self.assertFalse(matched)
        self.assertIn(reason, ("graph_invariant", "graph_topology", "graph_signature"))

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
        seeds, stats = build_geometry_driven_seed_candidates(layout_index, 1.0)
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
        seeds, stats = build_geometry_driven_seed_candidates(layout_index, 1.0)
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
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["input_path"] = str(input_oas.with_suffix(".missing"))
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

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
        output = self.temp_root / "merge_result.csv"
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
        rows = _read_csv_rows(output)
        self.assertEqual(list(rows[0].keys()), v2_lsf.CSV_OUTPUT_COLUMNS)
        self.assertEqual(len(rows), result["exact_cluster_count"])
        self.assertNotIn("output_format", result)
        self.assertEqual(result["result_csv_row_count"], result["exact_cluster_count"])
        self.assertEqual(result["result_csv_columns"], v2_lsf.CSV_OUTPUT_COLUMNS)
        self.assertEqual(
            v2_lsf.CSV_OUTPUT_COLUMNS,
            ["groupID", "cluster_id", "center_x_um", "center_y_um", "clip_size", "group_weight", "risk_score", "risk_rank"],
        )
        self.assertEqual([int(row["groupID"]) for row in rows], list(range(1, len(rows) + 1)))
        self.assertTrue(all(int(row["cluster_id"]) >= 1 for row in rows))
        self.assertTrue(all(float(row["clip_size"]) == 1.0 for row in rows))
        representative_csv = Path(result["cluster_representative_csv_path"])
        self.assertTrue(representative_csv.exists())
        representative_rows = _read_csv_rows(representative_csv)
        self.assertEqual(len(representative_rows), result["total_clusters"])
        self.assertEqual(len(result["cluster_representative_csv_columns"]), len(representative_rows[0].keys()))
        self.assertEqual(result["pipeline_mode"], v2_lsf.PIPELINE_MODE)
        self.assertEqual(result["seed_strategy"], "geometry_driven")
        self.assertGreater(result["marker_count"], 0)
        self.assertGreater(result["exact_cluster_count"], 0)
        self.assertGreater(result["candidate_count"], 0)
        self.assertGreater(result["total_clusters"], 0)
        self.assertTrue(all("distance_worst_case_score" in cluster for cluster in result["clusters"]))
        self.assertTrue(all(cluster["distance_worst_case_score"] >= 0.0 for cluster in result["clusters"]))
        self.assertIn("final_verification_breakdown", result)
        self.assertIn("reject_reason_counts", result["final_verification_breakdown"])
        self.assertNotIn("geometry", result["final_verification_breakdown"]["reject_reason_counts"])
        self.assertIn("witness_attempted", result["final_verification_stats"])
        self.assertIn("witness_verified_pass", result["final_verification_stats"])
        self.assertIn("lsf_manifest", result)
        self.assertNotIn("max_rss_mb", result["lsf_manifest"])
        payload = v2_lsf._final_stage_output_payload(str(output), result)
        self.assertIn("final_verification_reject_reason_counts", payload)
        self.assertIn("candidate_group_count", payload)
        self.assertIn("cluster_representative_csv", payload)
        self.assertIn("seed_coverage_audit", payload)
        self.assertNotIn("max_rss_mb", payload)
        with self.assertRaises(ValueError):
            v2_lsf.merge_stage(str(manifest_path), str(self.temp_root / "merge_result.json"))

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
        baseline_output = self.temp_root / "coverage_baseline.csv"
        distributed_output = self.temp_root / "coverage_distributed.csv"
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
        self.assertTrue(baseline_output.exists())
        manifest = v2_lsf.prepare_coverage_stage(str(manifest_path), coverage_shard_count=2, coverage_shard_size=1)
        self.assertGreater(manifest["coverage_shard_count"], 0)
        self.assertTrue(Path(manifest["exact_index"]["output_json"]).exists())
        self.assertTrue(Path(manifest["exact_index"]["output_npz"]).exists())
        self.assertIn("exact_member_index", manifest)
        self.assertTrue(Path(manifest["exact_member_index"]["output_json"]).exists())
        self.assertEqual(manifest["exact_member_index"]["member_count"], baseline["marker_count"])
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
            "prepare_coverage_exact_member_index_write",
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
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for shard in manifest_payload["shards"]:
            shard["output_npz"] = str(Path(shard["output_npz"]).with_suffix(".missing"))
        manifest_payload["exact_index"]["output_json"] = str(Path(manifest_payload["exact_index"]["output_json"]).with_suffix(".missing"))
        manifest_payload["exact_index"]["output_npz"] = str(Path(manifest_payload["exact_index"]["output_npz"]).with_suffix(".missing"))
        manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest = manifest_payload
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
            shard_metadata, coverage_offsets, coverage_values, _ = load_coverage_shard_csr_metadata(
                coverage_shard["output_json"],
                coverage_shard["output_npz"],
            )
            self.assertEqual(len(shard_metadata) + 1, int(coverage_offsets.size))
            self.assertGreater(int(coverage_values.size), 0)
            bad_npz = self.temp_root / ("coverage_without_csr_%s.npz" % coverage_shard["coverage_shard_id"])
            np.savez_compressed(str(bad_npz), candidate_bitmaps=np.zeros((0, 0, 0), dtype=bool))
            with self.assertRaises(RuntimeError):
                load_coverage_shard_csr_metadata(coverage_shard["output_json"], bad_npz)

        broken_manifest = json.loads(json.dumps(manifest))
        broken_manifest["exact_member_index"]["output_json"] = str(
            Path(broken_manifest["exact_member_index"]["output_json"]).with_suffix(".missing")
        )
        manifest_path.write_text(json.dumps(broken_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            v2_lsf.merge_coverage_stage(str(manifest_path), str(self.temp_root / "missing_exact_members.csv"))
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        distributed = v2_lsf.merge_coverage_stage(str(manifest_path), str(distributed_output))
        distributed_rows = _read_csv_rows(distributed_output)
        self.assertEqual(list(distributed_rows[0].keys()), v2_lsf.CSV_OUTPUT_COLUMNS)
        self.assertEqual(len(distributed_rows), distributed["exact_cluster_count"])
        self.assertNotIn("output_format", distributed)
        self.assertEqual(distributed["result_csv_row_count"], distributed["exact_cluster_count"])
        self.assertEqual(distributed["candidate_group_count"], manifest["candidate_bundle_index"]["candidate_group_count"])
        self.assertEqual([int(row["groupID"]) for row in distributed_rows], list(range(1, len(distributed_rows) + 1)))
        self.assertTrue(all(int(row["cluster_id"]) >= 1 for row in distributed_rows))
        representative_csv = Path(distributed["cluster_representative_csv_path"])
        self.assertTrue(representative_csv.exists())
        representative_rows = _read_csv_rows(representative_csv)
        self.assertEqual(len(representative_rows), distributed["total_clusters"])
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
        self.assertIn("result_csv", distributed["lsf_manifest"])
        self.assertNotIn("max_rss_mb", distributed["lsf_manifest"])
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
        self.assertEqual(inspection["exact_member_index"]["member_count"], distributed["marker_count"])
        self.assertTrue(inspection["result_csv"]["exists"])
        self.assertGreater(inspection["result_csv"]["bytes"], 0)
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
        output = self.temp_root / "local_result.csv"
        config = {
            "clip_size_um": 1.0,
            "geometry_match_mode": "ecc",
            "area_match_ratio": 0.96,
            "edge_tolerance_um": 0.02,
            "pixel_size_nm": 20,
        }
        result = v2_lsf.run_local_stage(str(input_oas), str(work_dir), str(output), config, [], shard_count=2, shard_size=1)
        rows = _read_csv_rows(output)
        self.assertEqual(result["pipeline_mode"], v2_lsf.PIPELINE_MODE)
        self.assertEqual(len(rows), result["exact_cluster_count"])
        self.assertGreaterEqual(result["selected_candidate_count"], 1)
        self.assertTrue(Path(result["cluster_representative_csv_path"]).exists())
        self.assertTrue(all("distance_worst_case_score" in cluster for cluster in result["clusters"]))
        self.assertTrue(all(cluster["distance_worst_case_score"] >= 0.0 for cluster in result["clusters"]))
        self.assertNotIn("contact_pair_seed_count", result)
        self.assertNotIn("drc_component_seed_count", result)
        distributed_output = self.temp_root / "local_distributed_result.csv"
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
        self.assertEqual(distributed["total_clusters"], result["total_clusters"])
        self.assertGreater(distributed["lsf_manifest"]["coverage_shard_count"], 0)

    def test_quality_metrics_extend_representative_csv(self):
        """开启 quality metrics 时应补充全局指标和 representative CSV 质量列。"""

        input_oas = self.temp_root / "quality.oas"
        _write_oas(
            input_oas,
            [
                gdstk.rectangle((0.05, 0.05), (0.25, 0.25), layer=1, datatype=0),
                gdstk.rectangle((0.55, 0.05), (0.75, 0.25), layer=1, datatype=0),
                gdstk.rectangle((1.05, 0.05), (1.25, 0.25), layer=1, datatype=0),
            ],
        )
        work_dir = self.temp_root / "work_quality"
        output = self.temp_root / "quality_result.csv"
        config = {
            "clip_size_um": 1.0,
            "geometry_match_mode": "ecc",
            "area_match_ratio": 0.96,
            "edge_tolerance_um": 0.02,
            "pixel_size_nm": 20,
            "compute_quality_metrics": True,
        }
        result = v2_lsf.run_local_stage(str(input_oas), str(work_dir), str(output), config, [], shard_count=1, shard_size=1)
        self.assertTrue(result["quality_metrics_enabled"])
        self.assertIn("quality_metrics", result)
        self.assertIn("representative_visual_purity", result["quality_metrics"])
        self.assertIn("weighted_representative_visual_purity", result["quality_metrics"])
        self.assertIn("pairwise_geometry_purity", result["quality_metrics"])
        self.assertIn("raw_coverage_graph_recall", result["quality_metrics"])
        self.assertIn("trusted_fragmentation_recall", result["quality_metrics"])
        self.assertIn("review_merge_candidate_weight_ratio", result["quality_metrics"])
        self.assertNotIn("safe_recall_merge_cluster_reduction", result["quality_metrics"])
        self.assertFalse(any(str(key).startswith("singleton_absorption_") for key in result["quality_metrics"]))
        self.assertFalse(any(str(key).startswith("singleton_microcluster_") for key in result["quality_metrics"]))
        self.assertNotIn("sampled_purity_score", result["quality_metrics"])
        self.assertNotIn("overmerge_suspect_count", result["quality_metrics"])
        self.assertNotIn("member_to_member_transitivity_purity", result["quality_metrics"])
        self.assertNotIn("recall_proxy_score", result["quality_metrics"])
        representative_rows = _read_csv_rows(result["cluster_representative_csv_path"])
        self.assertIn("representative_visual_pass_ratio", representative_rows[0])
        self.assertIn("representative_visual_fail_count", representative_rows[0])
        self.assertIn("representative_visual_checked_count", representative_rows[0])
        self.assertIn("representative_visual_sample_status", representative_rows[0])
        self.assertIn("pairwise_geometry_purity", representative_rows[0])
        self.assertIn("pairwise_geometry_fail_count", representative_rows[0])
        self.assertIn("pairwise_geometry_sampled_pair_count", representative_rows[0])
        self.assertIn("pairwise_geometry_sample_status", representative_rows[0])
        self.assertIn("overmerge_score", representative_rows[0])
        self.assertIn("overmerge_reason", representative_rows[0])
        self.assertNotIn("purity_proxy", representative_rows[0])
        self.assertNotIn("intra_cluster_fail_count", representative_rows[0])
        self.assertNotIn("intra_cluster_sampled_pair_count", representative_rows[0])
        self.assertNotIn("purity_sample_status", representative_rows[0])

    def test_greedy_cover_prefers_exact_count_before_weight(self):
        """set cover 先减少 cluster 数，再用权重做次级排序。"""

        bitmap = np.zeros((8, 8), dtype=bool)
        bitmap[2:6, 2:6] = True
        exact_clusters = [_make_exact_cluster(0, bitmap), _make_exact_cluster(1, bitmap), _make_exact_cluster(2, bitmap)]
        exact_clusters[0].weight_sum = 100
        heavy_candidate = _make_candidate("heavy", 0, bitmap)
        heavy_candidate.coverage = set([0])
        small_pair_candidate = _make_candidate("small_pair", 1, bitmap)
        small_pair_candidate.coverage = set([1, 2])

        selected = mainline_lsf.greedy_cover([heavy_candidate, small_pair_candidate], exact_clusters, {})

        self.assertEqual([candidate.candidate_id for candidate in selected], ["small_pair", "heavy"])

    def test_singleton_absorption_review_edge_absorbs_target_singleton(self):
        """review edge 的 high/medium 候选可以把 singleton 吸收到 non-singleton target。"""

        bitmap = np.zeros((8, 8), dtype=bool)
        bitmap[2:6, 2:6] = True
        exact_clusters = [_make_exact_cluster(idx, bitmap) for idx in range(3)]
        target = _make_candidate("target", 0, bitmap, origin_seed_type=mainline_lsf.SEED_TYPE_ARRAY)
        singleton = _make_candidate("singleton", 2, bitmap, origin_seed_type=mainline_lsf.SEED_TYPE_ARRAY)
        merged_units, metrics = mainline_lsf._apply_singleton_absorption_exact_review(
            [(target, [exact_clusters[0], exact_clusters[1]]), (singleton, [exact_clusters[2]])],
            {
                "singleton_total_count": 1,
                "singleton_absorption_candidate_rows": [
                    {
                        "source_cluster_id": 1,
                        "target_cluster_id": 2,
                        "candidate_id": "review_edge",
                        "confidence_tier": "high",
                        "candidate_source": "review_edge",
                        "singleton_cluster_id": 2,
                        "absorb_cluster_id": 1,
                        "singleton_exact_cluster_id": 2,
                        "source_exact_cluster_id": 0,
                        "target_exact_cluster_id": 2,
                        "edge_weight": 10,
                    }
                ],
                "cluster_quality_by_index": {
                    0: {
                        "representative_visual_pass_ratio": 1.0,
                        "representative_visual_checked_count": 2,
                        "representative_seed_type": mainline_lsf.SEED_TYPE_ARRAY,
                        "shift_distance_um": 0.0,
                        "overmerge_score": 0.0,
                    }
                },
            },
            {"clip_size_um": 1.0, "geometry_match_mode": "ecc", "edge_tolerance_um": 0.0, "pixel_size_nm": 125},
        )

        self.assertEqual(len(merged_units), 1)
        self.assertEqual([cluster.exact_cluster_id for cluster in merged_units[0][1]], [0, 1, 2])
        self.assertEqual(metrics["singleton_absorption_merged_count"], 1)
        self.assertEqual(metrics["singleton_absorption_merged_by_source"], {"review_edge": 1})

    def test_singleton_absorption_source_side_singleton(self):
        """singleton 在 source 端时也应能识别并吸收到 non-singleton target。"""

        bitmap = np.zeros((8, 8), dtype=bool)
        bitmap[2:6, 2:6] = True
        exact_clusters = [_make_exact_cluster(idx, bitmap) for idx in range(3)]
        target = _make_candidate("target", 0, bitmap, origin_seed_type=mainline_lsf.SEED_TYPE_ARRAY)
        singleton = _make_candidate("singleton", 2, bitmap, origin_seed_type=mainline_lsf.SEED_TYPE_ARRAY)
        merged_units, metrics = mainline_lsf._apply_singleton_absorption_exact_review(
            [(target, [exact_clusters[0], exact_clusters[1]]), (singleton, [exact_clusters[2]])],
            {
                "review_merge_candidate_rows": [
                    {
                        "source_cluster_id": 2,
                        "target_cluster_id": 1,
                        "candidate_id": "review_source_singleton",
                        "confidence_tier": "medium",
                        "candidate_source": "review_edge",
                        "source_exact_cluster_id": 2,
                        "target_exact_cluster_id": 0,
                        "edge_weight": 8,
                    }
                ],
                "cluster_quality_by_index": {
                    0: {
                        "representative_visual_pass_ratio": 1.0,
                        "representative_visual_checked_count": 2,
                        "representative_seed_type": mainline_lsf.SEED_TYPE_ARRAY,
                        "shift_distance_um": 0.0,
                        "overmerge_score": 0.0,
                    }
                },
            },
            {"clip_size_um": 1.0, "geometry_match_mode": "ecc", "edge_tolerance_um": 0.0, "pixel_size_nm": 125},
        )

        self.assertEqual(len(merged_units), 1)
        self.assertEqual(metrics["singleton_absorption_merged_by_source"], {"review_edge": 1})

    def test_singleton_absorption_strong_agreement_rescue(self):
        """无 review edge 时，descriptor 与 graph 双命中的 strong agreement 可进入吸收。"""

        bitmap = np.zeros((8, 8), dtype=bool)
        bitmap[2:6, 2:6] = True
        exact_clusters = [_make_exact_cluster(idx, bitmap) for idx in range(3)]
        target = _make_candidate("target", 0, bitmap, origin_seed_type=mainline_lsf.SEED_TYPE_ARRAY)
        singleton = _make_candidate("singleton", 2, bitmap, origin_seed_type=mainline_lsf.SEED_TYPE_ARRAY)
        merged_units, metrics = mainline_lsf._apply_singleton_absorption_exact_review(
            [(target, [exact_clusters[0], exact_clusters[1]]), (singleton, [exact_clusters[2]])],
            {"cluster_quality_by_index": {}},
            {"clip_size_um": 1.0, "geometry_match_mode": "ecc", "edge_tolerance_um": 0.0, "pixel_size_nm": 125},
        )

        self.assertEqual(len(merged_units), 1)
        self.assertEqual(metrics["singleton_absorption_merged_by_source"], {"strong_descriptor_graph_agreement": 1})

    def test_singleton_absorption_context_graph_is_strict_only(self):
        """context-close graph 候选 strict 失败后不触发 normal fallback。"""

        bitmap = np.zeros((8, 8), dtype=bool)
        bitmap[2:6, 2:6] = True
        exact_clusters = [_make_exact_cluster(idx, bitmap) for idx in range(3)]
        target = _make_candidate("target", 0, bitmap, origin_seed_type=mainline_lsf.SEED_TYPE_ARRAY)
        singleton = _make_candidate("singleton", 2, bitmap, origin_seed_type=mainline_lsf.SEED_TYPE_ARRAY)

        def fake_descriptor_vector(candidate):
            """让 descriptor rescue 不命中，只保留 graph/context close。"""

            if str(candidate.candidate_id) == "target":
                return np.asarray([1.0, 0.0], dtype=np.float32)
            return np.asarray([0.0, 1.0], dtype=np.float32)

        with mock.patch.object(mainline_lsf, "_normalized_cheap_feature_vector", side_effect=fake_descriptor_vector), mock.patch.object(
            mainline_lsf,
            "_candidate_matches_exact",
            return_value=(False, "strict_reject", "none"),
        ), mock.patch.object(mainline_lsf, "_candidate_matches_exact_normal", return_value=(True, "normal_pass", "base")) as normal_match:
            merged_units, metrics = mainline_lsf._apply_singleton_absorption_exact_review(
                [(target, [exact_clusters[0], exact_clusters[1]]), (singleton, [exact_clusters[2]])],
                {"cluster_quality_by_index": {}},
                {"clip_size_um": 1.0, "geometry_match_mode": "ecc", "edge_tolerance_um": 0.0, "pixel_size_nm": 125},
            )

        self.assertEqual(len(merged_units), 2)
        self.assertEqual(metrics["singleton_absorption_attempted_by_source"], {"graph_rescue_context_close": 1})
        self.assertEqual(metrics["singleton_absorption_merged_count"], 0)
        self.assertEqual(
            metrics["singleton_absorption_reject_reason_by_source"],
            {"graph_rescue_context_close": {"strict_verification_reject": 1}},
        )
        normal_match.assert_not_called()

    def test_singleton_microcluster_clique_and_pair_fallback(self):
        """strict-only microcluster 先合 size-3 clique，broken clique 再允许 pair fallback。"""

        bitmap = np.zeros((8, 8), dtype=bool)
        bitmap[2:6, 2:6] = True
        exact_clusters = [_make_exact_cluster(idx, bitmap) for idx in range(3)]
        candidates = [_make_candidate("base_%d" % idx, idx, bitmap, origin_seed_type=mainline_lsf.SEED_TYPE_ARRAY) for idx in range(3)]
        units = [(candidates[idx], [exact_clusters[idx]]) for idx in range(3)]
        merged_units, metrics = mainline_lsf._apply_singleton_absorption_exact_review(
            units,
            {"cluster_quality_by_index": {}},
            {"clip_size_um": 1.0, "geometry_match_mode": "ecc", "edge_tolerance_um": 0.0, "pixel_size_nm": 125},
        )
        self.assertEqual(len(merged_units), 1)
        self.assertEqual(metrics["singleton_microcluster_clique_merged_count"], 2)
        self.assertEqual(metrics["singleton_microcluster_pair_merged_count"], 0)

        def strict_pair_side_effect(candidate, exact_cluster, config, descriptor_cache):
            """模拟 A-C strict fail，A-B/B-C strict pass。"""

            if str(candidate.candidate_id) == "base_0" and int(exact_cluster.exact_cluster_id) == 2:
                return False, "strict_reject", "none"
            return True, "exact_hash", "base"

        with mock.patch.object(mainline_lsf, "_candidate_matches_exact", side_effect=strict_pair_side_effect):
            broken_units, broken_metrics = mainline_lsf._apply_singleton_absorption_exact_review(
                units,
                {"cluster_quality_by_index": {}},
                {"clip_size_um": 1.0, "geometry_match_mode": "ecc", "edge_tolerance_um": 0.0, "pixel_size_nm": 125},
            )
        self.assertEqual(len(broken_units), 2)
        self.assertEqual(broken_metrics["singleton_microcluster_clique_merged_count"], 0)
        self.assertEqual(broken_metrics["singleton_microcluster_pair_merged_count"], 1)
        self.assertEqual(
            broken_metrics["singleton_microcluster_clique_reject_reason_counts"],
            {"strict_verification_reject": 1},
        )

    def test_public_quality_metrics_hide_singleton_absorption_details(self):
        """public quality_metrics 与 final payload 不暴露旧 safe recall 或 singleton 内部诊断字段。"""

        bitmap = np.zeros((8, 8), dtype=bool)
        bitmap[2:6, 2:6] = True
        exact_clusters = [_make_exact_cluster(0, bitmap), _make_exact_cluster(1, bitmap)]
        base0 = _make_candidate("base_0", 0, bitmap, origin_seed_type=mainline_lsf.SEED_TYPE_ARRAY)
        base1 = _make_candidate("base_1", 1, bitmap, origin_seed_type=mainline_lsf.SEED_TYPE_ARRAY)
        result = mainline_lsf.build_compact_result_exact_review(
            None,
            exact_clusters,
            [base0, base1],
            [base0, base1],
            {"candidate_group_count": 2},
            {
                "clip_size_um": 1.0,
                "geometry_match_mode": "ecc",
                "edge_tolerance_um": 0.0,
                "pixel_size_nm": 125,
                "compute_quality_metrics": True,
            },
            {},
        )
        self.assertEqual(result["total_clusters"], 1)
        self.assertFalse(any(str(key).startswith("safe_recall_merge_") for key in result["quality_metrics"]))
        self.assertFalse(any(str(key).startswith("singleton_absorption_") for key in result["quality_metrics"]))
        self.assertFalse(any(str(key).startswith("singleton_microcluster_") for key in result["quality_metrics"]))
        payload = v2_lsf._final_stage_output_payload(str(self.temp_root / "result.csv"), result)
        self.assertFalse(any(str(key).startswith("safe_recall_merge_") for key in payload["quality_metrics"]))
        self.assertFalse(any(str(key).startswith("singleton_absorption_") for key in payload["quality_metrics"]))
        self.assertFalse(any(str(key).startswith("singleton_microcluster_") for key in payload["quality_metrics"]))

    def test_distributed_candidate_bitmap_loader_restores_candidate(self):
        """分布式 evidence 仍可按 candidate_bitmap_locations 恢复 candidate bitmap。"""

        bitmap = np.zeros((8, 8), dtype=bool)
        bitmap[2:6, 2:6] = True
        candidate = _make_candidate("dist_candidate", 0, bitmap, origin_seed_type=mainline_lsf.SEED_TYPE_ARRAY)
        metadata = mainline_lsf.candidate_metadata(candidate, include_coverage=False)
        npz_path = self.temp_root / "candidate_bitmaps.npz"
        np.savez_compressed(str(npz_path), candidate_bitmaps=np.asarray([candidate.clip_bitmap], dtype=bool))
        restored = mainline_lsf._candidate_clip_from_evidence(
            {"item": metadata, "coverage": set([0])},
            {str(candidate.candidate_id): {"npz_path": str(npz_path), "bitmap_index": 0}},
        )

        self.assertTrue(np.array_equal(restored.clip_bitmap, bitmap))
        self.assertEqual(restored.coverage, set([0]))

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
        output = self.temp_root / "layer_ops_result.csv"
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
        self.assertTrue(all("grid_cell_bbox" not in record.metadata for record in records))
        self.assertTrue(payload["apply_layer_operations"])
        self.assertEqual(payload["registered_layer_operations"], [["10/0", "11/0", "subtract", "13/0"]])
        self.assertEqual(payload["effective_pattern_layers"], [[13, 0]])
        self.assertGreaterEqual(payload["layout_element_count"], payload["halo_filtered_element_count"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
