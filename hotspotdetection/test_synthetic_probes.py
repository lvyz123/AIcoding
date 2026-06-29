#!/usr/bin/env python3
"""Synthetic weak-pattern probes，用于保护 MP/AF/AP 审查能力。"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path

import gdstk
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import matchability_audit as match_audit
import review_evidence_audit as evidence_audit
import recipe_site_selector as selector
from layout_utils import _write_oas_library


def _write_image(path: Path) -> None:
    """写出测试用主行为图像。"""
    image = np.zeros((32, 32), dtype=np.float32)
    image[8:24, 14:18] = 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, image=image)


def _add_marker(cell: gdstk.Cell, cx: float, cy: float) -> None:
    """添加 synthetic hotspot marker。"""
    cell.add(gdstk.rectangle((cx - 0.02, cy - 0.02), (cx + 0.02, cy + 0.02), layer=999, datatype=0))


def _add_line_pair(cell: gdstk.Cell, cx: float, cy: float) -> None:
    """添加 narrow-spacing line pair。"""
    gap = 0.06
    half_gap = gap * 0.5
    cell.add(gdstk.rectangle((cx - half_gap - 0.05, cy - 0.12), (cx - half_gap, cy + 0.12), layer=1, datatype=0))
    cell.add(gdstk.rectangle((cx + half_gap, cy - 0.12), (cx + half_gap + 0.05, cy + 0.12), layer=1, datatype=0))


def _add_l_shape(cell: gdstk.Cell, cx: float, cy: float) -> None:
    """添加局部唯一 L-shape AP probe。"""
    cell.add(gdstk.rectangle((cx - 0.10, cy - 0.10), (cx - 0.04, cy + 0.12), layer=1, datatype=0))
    cell.add(gdstk.rectangle((cx - 0.10, cy - 0.10), (cx + 0.12, cy - 0.04), layer=1, datatype=0))


def _make_oas(path: Path, mode: str) -> None:
    """生成小型 synthetic OAS probe。"""
    lib = gdstk.Library()
    cell = gdstk.Cell("TOP")
    _add_marker(cell, 0.0, 0.0)
    _add_line_pair(cell, 0.0, 0.0)
    if mode == "corner_ap":
        _add_l_shape(cell, 0.0, 0.6)
        _add_line_pair(cell, 0.6, 0.0)
    elif mode == "periodic_ap":
        for x in (-0.6, 0.0, 0.6):
            for y in (-0.6, 0.0, 0.6):
                if abs(x) < 1e-9 and abs(y) < 1e-9:
                    continue
                _add_line_pair(cell, x, y)
    lib.add(cell)
    _write_oas_library(lib, str(path))


def _make_manifest(root: Path) -> Path:
    """生成只含一个 marker 的 behavior manifest。"""
    aerial_path = root / "unit__marker_000000_aerial.npz"
    _write_image(aerial_path)
    manifest = root / "behavior.jsonl"
    row = {
        "sample_id": "unit__marker_000000",
        "source_path": "unit.oas",
        "marker_id": "unit__marker_000000",
        "clip_bbox": [-0.2, -0.2, 0.2, 0.2],
        "aerial_npz": str(aerial_path),
        "risk_score": 1.0,
    }
    manifest.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def _args(root: Path, oas_path: Path, manifest: Path) -> argparse.Namespace:
    """构造 recipe selector synthetic probe 参数。"""
    return argparse.Namespace(
        input_path=str(oas_path),
        marker_layer="999/0",
        behavior_manifest=str(manifest),
        output_dir=str(root / "recipe_out"),
        clip_size=0.4,
        mp_template_size_um=None,
        af_template_size_um=None,
        ap_template_size_um=None,
        max_sites=1,
        mp_coverage_target=0.95,
        mp_search_radius_um=0.8,
        mp_candidates_per_marker=2,
        max_care_area_instances_per_family=50,
        min_feature_um=None,
        af_search_radius_um=0.8,
        sem_image_shift_limit_um=None,
        ap_search_radius_um=0.8,
        candidate_step_um=0.2,
        min_site_distance_um=0.3,
        recursive_input=False,
        apply_layer_ops=False,
        register_op=None,
    )


class SyntheticProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = SCRIPT_DIR / "test_outputs" / f"_synthetic_probe_{uuid.uuid4().hex[:8]}"
        shutil.rmtree(self.temp_root, ignore_errors=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def _run_probe(self, mode: str) -> dict:
        """运行单个 synthetic OAS probe。"""
        oas_path = self.temp_root / "unit.oas"
        _make_oas(oas_path, mode)
        manifest = _make_manifest(self.temp_root)
        return selector.run_recipe_selector(_args(self.temp_root, oas_path, manifest))

    def test_corner_ap_probe_records_matchability(self):
        result = self._run_probe("corner_ap")
        details = next(item for item in result["site_details"] if item.get("ap_candidate"))
        components = details["ap_candidate"]["components"]

        self.assertIn("layout_matchability_score", components)
        self.assertGreater(components["keypoint_count"], 0)
        self.assertGreater(components["layout_matchability_score"], 0.20)
        self.assertLess(components["periodicity_penalty"], 0.95)

    def test_periodic_ap_probe_records_periodicity_without_relaxing_gate(self):
        result = self._run_probe("periodic_ap")
        details = result["site_details"][0]
        site = details["site"]
        ap_candidate = details["ap_candidate"]
        ap_components = ap_candidate["components"]

        self.assertIn("layout_matchability_score", ap_components)
        self.assertGreaterEqual(ap_components["periodicity_penalty"], 0.20)
        if "no_unique_ap" in site["reject_reason"]:
            self.assertFalse(ap_candidate["accepted"])
        else:
            self.assertEqual(site["recipe_status"], "selected")
            self.assertTrue(ap_candidate["accepted"])
            self.assertTrue(all(ap_candidate["acceptance_checks"].values()))

    def test_keypoint_poor_bitmap_has_low_matchability(self):
        bitmap = np.zeros((32, 32), dtype=bool)

        audit = match_audit.compute_ap_matchability(bitmap, descriptor_margin=0.1, nearest_similarity=0.9, peak_count=6)

        self.assertLess(audit["layout_matchability_score"], 0.45)
        self.assertGreater(audit["periodicity_penalty"], 0.30)

    def test_af_hotspot_core_probe_keeps_hard_reject(self):
        bitmap = np.zeros((16, 16), dtype=bool)
        bitmap[3:13, 7:9] = True
        bitmap[7:9, 3:13] = True
        candidate = selector.WindowCandidate(0.4, 0.0, 0.4, {"clip_bitmap": bitmap, "clip_bbox": [0.2, -0.2, 0.6, 0.2]})

        selected = selector._select_af_candidate([candidate], mp_bitmap=bitmap, radius_um=1.0)

        self.assertIs(selected, candidate)
        self.assertEqual(candidate.reject_reason, "too_hotspot_like")
        self.assertIn("layout_matchability_score", candidate.components)

    def test_graph_context_distinguishes_line_pair_and_corner(self):
        line_pair = np.zeros((32, 32), dtype=bool)
        line_pair[6:26, 9:12] = True
        line_pair[6:26, 20:23] = True
        corner = np.zeros((32, 32), dtype=bool)
        corner[8:25, 8:12] = True
        corner[21:25, 8:25] = True

        line_graph = evidence_audit.compute_graph_context(line_pair)
        corner_graph = evidence_audit.compute_graph_context(corner)

        self.assertGreaterEqual(line_graph["graph_node_count"], 2)
        self.assertEqual(corner_graph["graph_node_count"], 1)
        self.assertNotEqual(line_graph["graph_feature_vector"], corner_graph["graph_feature_vector"])

    def test_evidence_contradiction_marks_high_priority_low_evidence(self):
        candidate = {
            "mp_hotspot_score": 0.05,
            "behavior_risk": 0.0,
            "care_area_match_score": 0.1,
            "care_area_homogeneity_score": 0.1,
            "mp_localization_confidence": 0.0,
            "metrology_priority_score": 0.8,
            "recipe_waste_penalty": 0.1,
            "mp_discovery_components": {},
        }

        audit = evidence_audit.compute_evidence_contradiction_audit(
            candidate,
            graph_context={"graph_env_complexity": 0.0},
            ring_context={},
            memory_prior={},
        )

        self.assertLess(audit["defect_evidence_proxy_score"], 0.40)
        self.assertIn("high_priority_low_evidence", audit["static_contradiction_tags"])

    def test_taxonomy_audit_separates_tnsb_and_htc_extremes(self):
        strong_evidence = {"defect_evidence_proxy_score": 0.85}
        tnsb = evidence_audit.compute_pattern_taxonomy_audit(
            {
                "pattern_novelty": 0.95,
                "care_area_match_score": 0.2,
                "care_area_homogeneity_score": 0.2,
                "recipe_waste_penalty": 0.1,
            },
            graph_context={"mp_graph_rarity": 0.95, "care_area_graph_similarity": 0.0, "graph_nearest_similarity": 0.0},
            evidence_audit=strong_evidence,
            memory_prior={"memory_prior_confidence": 0.0, "memory_nearest_similarity": 0.0, "memory_waste_prior": 0.5},
        )
        htc = evidence_audit.compute_pattern_taxonomy_audit(
            {
                "pattern_novelty": 0.05,
                "care_area_match_score": 0.95,
                "care_area_homogeneity_score": 0.95,
                "recipe_waste_penalty": 0.9,
            },
            graph_context={"mp_graph_rarity": 0.05, "care_area_graph_similarity": 0.95, "graph_nearest_similarity": 0.95},
            evidence_audit={"defect_evidence_proxy_score": 0.10},
            memory_prior={"memory_prior_confidence": 1.0, "memory_nearest_similarity": 0.95, "memory_waste_prior": 0.9},
        )

        self.assertEqual(tnsb["pattern_taxonomy_class"], "tnsb_like")
        self.assertEqual(htc["pattern_taxonomy_class"], "htc_like")

    def test_expected_feasibility_proxy_is_bounded(self):
        audit = evidence_audit.compute_expected_feasibility_audit(
            {
                "recipe_waste_penalty": 0.2,
                "mp_localization_confidence": 0.7,
                "mp_verified": True,
            },
            graph_context={"graph_env_complexity": 0.6, "mp_graph_rarity": 0.8},
            memory_prior={
                "memory_prior_confidence": 0.8,
                "memory_af_success_prior": 0.75,
                "memory_ap_success_prior": 0.65,
                "memory_ap_duplicate_prior": 0.1,
            },
        )

        self.assertTrue(0.0 <= audit["expected_af_pass_proxy"] <= 1.0)
        self.assertTrue(0.0 <= audit["expected_ap_pass_proxy"] <= 1.0)
        self.assertGreater(audit["expected_recipe_feasibility_proxy"], 0.5)


if __name__ == "__main__":
    unittest.main()
