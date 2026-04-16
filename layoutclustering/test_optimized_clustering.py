#!/usr/bin/env python3
"""Unit tests for the optimized marker-driven clustering path."""

from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import layout_clustering_optimized as optimized
from mainline import _canonical_bitmap_hash


def _bitmap_square(size=20, start=5, end=13):
    bitmap = np.zeros((size, size), dtype=bool)
    bitmap[start:end, start:end] = True
    return bitmap


def _record(marker_id: str, bitmap: np.ndarray, exact_cluster_id: int = -1) -> optimized.MarkerRecord:
    clip_hash, _ = _canonical_bitmap_hash(bitmap)
    height, width = bitmap.shape
    record = optimized.MarkerRecord(
        marker_id=marker_id,
        source_path="unit.oas",
        source_name="unit.oas",
        marker_bbox=(0.0, 0.0, 1.0, 1.0),
        marker_center=(0.5, 0.5),
        clip_bbox=(0.0, 0.0, float(width), float(height)),
        expanded_bbox=(0.0, 0.0, float(width), float(height)),
        clip_bbox_q=(0, 0, int(width), int(height)),
        expanded_bbox_q=(0, 0, int(width), int(height)),
        marker_bbox_q=(0, 0, int(width), int(height)),
        shift_limits_px={"x": (0, 0), "y": (0, 0)},
        clip_bitmap=np.ascontiguousarray(bitmap, dtype=bool),
        expanded_bitmap=np.ascontiguousarray(bitmap, dtype=bool),
        clip_hash=clip_hash,
        expanded_hash=clip_hash,
        clip_area=float(np.count_nonzero(bitmap)),
        exact_cluster_id=int(exact_cluster_id),
    )
    return record


def _candidate(candidate_id: str, bitmap: np.ndarray, origin_exact_cluster_id: int = 0, *, direction="base"):
    clip_hash, _ = _canonical_bitmap_hash(bitmap)
    height, width = bitmap.shape
    return optimized.CandidateClip(
        candidate_id=candidate_id,
        origin_exact_cluster_id=int(origin_exact_cluster_id),
        center=(0.5 * float(width), 0.5 * float(height)),
        clip_bbox=(0.0, 0.0, float(width), float(height)),
        clip_bbox_q=(0, 0, int(width), int(height)),
        clip_bitmap=np.ascontiguousarray(bitmap, dtype=bool),
        clip_hash=clip_hash,
        shift_direction=str(direction),
        shift_distance_um=0.0 if direction == "base" else 0.02,
        coverage={int(origin_exact_cluster_id)} if direction == "base" else set(),
        source_marker_id=f"marker_{origin_exact_cluster_id}",
    )


class OptimizedClusteringTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = SCRIPT_DIR / "test_outputs" / "_optimized_unit"
        shutil.rmtree(self.temp_root, ignore_errors=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def _runner(self, mode="acc"):
        return optimized.OptimizedMainlineRunner(
            config={
                "marker_layer": "999/0",
                "geometry_match_mode": mode,
                "pixel_size_nm": 10,
                "edge_tolerance_um": 0.02,
                "area_match_ratio": 0.96,
            },
            temp_dir=self.temp_root,
        )

    def test_exact_hash_fast_path_skips_prefilter_and_geometry(self):
        runner = self._runner("acc")
        bitmap = _bitmap_square()
        exact = optimized.ExactCluster(0, _record("m0", bitmap, 0), [_record("m0", bitmap, 0)])
        candidate = _candidate("c0", bitmap, 0)
        stats = optimized._empty_prefilter_stats()

        original_prefilter = optimized._graph_prefilter_passes
        original_geometry = runner._geometry_passes
        try:
            optimized._graph_prefilter_passes = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("prefilter called"))
            runner._geometry_passes = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("geometry called"))
            self.assertTrue(runner._candidate_matches_exact(candidate, exact, strict=False, stats=stats))
        finally:
            optimized._graph_prefilter_passes = original_prefilter
            runner._geometry_passes = original_geometry
        self.assertEqual(stats["exact_hash_pass"], 1)

    def test_prefilter_rejections_do_not_enter_geometry_gate(self):
        runner = self._runner("acc")
        exact = optimized.ExactCluster(0, _record("m0", _bitmap_square(start=3, end=11), 0), [])
        candidate = _candidate("c0", _bitmap_square(start=8, end=16), 0, direction="right")

        original_prefilter = optimized._graph_prefilter_passes
        original_geometry = runner._geometry_passes
        try:
            runner._geometry_passes = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("geometry called"))
            for reason in ("invariant", "topology", "signature"):
                stats = optimized._empty_prefilter_stats()
                optimized._graph_prefilter_passes = lambda *args, _reason=reason, **kwargs: (False, _reason)
                self.assertFalse(runner._candidate_matches_exact(candidate, exact, strict=False, stats=stats))
                self.assertEqual(stats[f"{reason}_reject"], 1)
        finally:
            optimized._graph_prefilter_passes = original_prefilter
            runner._geometry_passes = original_geometry

    def test_acc_gate_accepts_same_bitmap_and_rejects_obvious_area_delta(self):
        runner = self._runner("acc")
        same = _bitmap_square()
        missing = same.copy()
        missing[5:11, 5:11] = False
        self.assertTrue(runner._geometry_passes(_candidate("same", same), _record("same", same)))
        self.assertFalse(runner._geometry_passes(_candidate("missing", missing), _record("same", same)))

    def test_ecc_gate_accepts_small_edge_shift_and_rejects_large_or_missing(self):
        runner = self._runner("ecc")
        base = _bitmap_square(size=24, start=7, end=15)
        small_shift = np.zeros_like(base)
        small_shift[8:16, 7:15] = True
        large_shift = np.zeros_like(base)
        large_shift[13:21, 7:15] = True
        missing = base.copy()
        missing[7:13, 7:13] = False

        self.assertTrue(runner._geometry_passes(_candidate("small", small_shift), _record("base", base)))
        self.assertFalse(runner._geometry_passes(_candidate("large", large_shift), _record("base", base)))
        self.assertFalse(runner._geometry_passes(_candidate("missing", missing), _record("base", base)))

    def test_greedy_set_cover_covers_all_exact_clusters(self):
        runner = self._runner("acc")
        clusters = [
            optimized.ExactCluster(idx, _record(f"m{idx}", _bitmap_square(start=4 + idx, end=10 + idx), idx), [])
            for idx in range(3)
        ]
        c0 = _candidate("c0", _bitmap_square(), 0)
        c1 = _candidate("c1", _bitmap_square(start=7, end=15), 1)
        c2 = _candidate("c2", _bitmap_square(start=8, end=16), 2)
        c0.coverage = {0, 1}
        c1.coverage = {1}
        c2.coverage = {2}
        selected = runner._greedy_cover([c0, c1, c2], clusters)
        covered = set().union(*(candidate.coverage for candidate in selected))
        self.assertEqual(covered, {0, 1, 2})
        self.assertEqual([candidate.candidate_id for candidate in selected], ["c0", "c2"])

    def test_final_verification_rejects_bad_member_and_creates_singleton(self):
        runner = self._runner("acc")
        bitmap0 = _bitmap_square()
        bitmap1 = _bitmap_square(start=9, end=17)
        exact0 = optimized.ExactCluster(0, _record("m0", bitmap0, 0), [_record("m0", bitmap0, 0)])
        exact1 = optimized.ExactCluster(1, _record("m1", bitmap1, 1), [_record("m1", bitmap1, 1)])
        c0 = _candidate("c0", bitmap0, 0)
        c0.coverage = {0, 1}
        base1 = _candidate("base1", bitmap1, 1)
        runner._base_candidate_by_exact_id = {0: c0, 1: base1}

        original_match = runner._candidate_matches_exact
        try:
            def fake_match(candidate, exact_cluster, *, strict, stats=None):
                del stats
                if not strict:
                    return True
                return candidate.candidate_id == "c0" and exact_cluster.exact_cluster_id == 0

            runner._candidate_matches_exact = fake_match
            units = runner._verified_cluster_units([c0], [exact0, exact1])
        finally:
            runner._candidate_matches_exact = original_match

        self.assertEqual([(unit[0].candidate_id, [c.exact_cluster_id for c in unit[1]]) for unit in units], [("base1", [1]), ("c0", [0])])
        self.assertEqual(runner.final_verification_stats["verified_pass"], 1)
        self.assertEqual(runner.final_verification_stats["verified_reject"], 1)
        self.assertEqual(runner.final_verification_stats["singleton_created"], 1)

    def test_output_metadata_is_optimized_only(self):
        runner = self._runner("acc")
        bitmap = _bitmap_square()
        record = _record("m0", bitmap, 0)
        exact = optimized.ExactCluster(0, record, [record])
        candidate = _candidate("c0", bitmap, 0)
        candidate.coverage = {0}
        runner._base_candidate_by_exact_id = {0: candidate}
        result = runner._build_results(
            [record],
            [exact],
            [candidate],
            "greedy",
            runtime_summary={"total": 0.0},
            candidate_count=1,
        )
        self.assertEqual(result["pipeline_mode"], "optimized")
        for key in (
            "geometry_match_mode",
            "pixel_size_nm",
            "area_match_ratio",
            "edge_tolerance_um",
            "marker_count",
            "exact_cluster_count",
            "candidate_count",
            "selected_candidate_count",
            "total_clusters",
            "cluster_sizes",
            "max_shift_distance_um",
            "prefilter_stats",
            "final_verification_stats",
            "clusters",
            "file_list",
            "file_metadata",
        ):
            self.assertIn(key, result)
        payload = json.dumps(result, default=optimized._json_default)
        for legacy_word in ("HDBSCAN", "hdbscan", "ILP", "closed_loop", "fft"):
            self.assertNotIn(legacy_word, payload)


if __name__ == "__main__":
    unittest.main()
