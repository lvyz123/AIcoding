#!/usr/bin/env python3
"""CD-SEM hotspot recipe selector 主流程回归测试。"""

from __future__ import annotations

import argparse
import csv
import json
import os
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

import recipe_site_selector as selector
import mp_candidate_generator as mp_gen
import care_area_generator as care_gen
import metrology_context as met_ctx
import ring_context as ring_ctx
import pattern_memory as pat_mem
import subset_objective_selection as subset_sel
from layout_utils import LayoutIndex, _write_oas_library


def _write_image(path: Path) -> None:
    """写出测试用 aerial NPZ。"""
    image = np.zeros((32, 32), dtype=np.float32)
    image[8:24, 14:18] = 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, image=image)


def _add_marker(cell: gdstk.Cell, cx: float, cy: float) -> None:
    """添加测试用 hotspot marker。"""
    cell.add(gdstk.rectangle((cx - 0.02, cy - 0.02), (cx + 0.02, cy + 0.02), layer=999, datatype=0))


def _add_line_pair_gap(cell: gdstk.Cell, cx: float, cy: float, gap: float = 0.06) -> None:
    """添加可调 gap 的双线 pattern，用于 care-area look-alike 测试。"""
    half_gap = float(gap) * 0.5
    cell.add(gdstk.rectangle((cx - half_gap - 0.05, cy - 0.12), (cx - half_gap, cy + 0.12), layer=1, datatype=0))
    cell.add(gdstk.rectangle((cx + half_gap, cy - 0.12), (cx + half_gap + 0.05, cy + 0.12), layer=1, datatype=0))


def _add_line_pair(cell: gdstk.Cell, cx: float, cy: float) -> None:
    """添加可用于 MP/AF 相似匹配的双线 pattern。"""
    _add_line_pair_gap(cell, cx, cy, gap=0.06)


def _add_l_shape(cell: gdstk.Cell, cx: float, cy: float) -> None:
    """添加局部唯一 L 形 addressing pattern。"""
    cell.add(gdstk.rectangle((cx - 0.10, cy - 0.10), (cx - 0.04, cy + 0.12), layer=1, datatype=0))
    cell.add(gdstk.rectangle((cx - 0.10, cy - 0.10), (cx + 0.12, cy - 0.04), layer=1, datatype=0))


def _make_oas(path: Path, mode: str) -> None:
    """按测试模式生成小型 synthetic OAS。"""
    lib = gdstk.Library()
    cell = gdstk.Cell("TOP")
    _add_marker(cell, 0.0, 0.0)
    if mode == "multi_mp":
        _add_line_pair(cell, -0.55, 0.0)
        _add_line_pair(cell, 0.0, 0.0)
        _add_line_pair(cell, 0.55, 0.0)
        _add_l_shape(cell, 0.0, 0.6)
    elif mode == "care_area_expand":
        _add_line_pair(cell, 0.0, 0.0)
        _add_line_pair(cell, 1.2, 0.0)
        _add_line_pair(cell, -1.2, 0.0)
        _add_l_shape(cell, 0.0, 0.6)
    elif mode == "care_area_homogeneity":
        _add_line_pair(cell, 0.0, 0.0)
        _add_line_pair(cell, 1.2, 0.0)
        _add_line_pair_gap(cell, 1.9, 0.0, gap=0.28)
        _add_l_shape(cell, 0.0, 0.6)
    elif mode == "offset_mp":
        _add_line_pair(cell, 0.4, 0.0)
        _add_line_pair(cell, 0.8, 0.0)
        _add_l_shape(cell, 0.4, 0.6)
    elif mode != "fallback_mp":
        _add_line_pair(cell, 0.0, 0.0)
    if mode in {"normal", "periodic_ap"}:
        _add_line_pair(cell, 0.6, 0.0)
    if mode == "normal":
        _add_l_shape(cell, 0.0, 0.6)
    if mode == "periodic_ap":
        for x in (-0.6, 0.0, 0.6):
            for y in (-0.6, 0.0, 0.6):
                if abs(x) < 1e-9 and abs(y) < 1e-9:
                    continue
                _add_line_pair(cell, x, y)
    lib.add(cell)
    _write_oas_library(lib, str(path))


def _make_manifest(root: Path, marker_id: str = "unit__marker_000000", risk_score: float = 1.0) -> Path:
    """生成只包含一个 marker 的 behavior manifest。"""
    aerial_path = root / f"{marker_id}_aerial.npz"
    _write_image(aerial_path)
    manifest = root / "behavior.jsonl"
    row = {
        "sample_id": marker_id,
        "source_path": "unit.oas",
        "marker_id": marker_id,
        "clip_bbox": [-0.2, -0.2, 0.2, 0.2],
        "aerial_npz": str(aerial_path),
        "risk_score": float(risk_score),
    }
    manifest.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def _memory_candidate(candidate_id: str, *, care_type: str = "spacing", mp_type: str = "critical_spacing_anchor") -> dict:
    """构造 pattern memory 单测用 compact candidate。"""
    ring = {
        "ring_radii_um": [0.10, 0.20, 0.35, 0.50],
        "ring_density_profile": [0.10, 0.70, 0.20, 0.80],
        "ring_edge_crossing_profile": [0.20, 0.60, 0.10, 0.70],
        "ring_asymmetry_profile": [0.10, 0.40, 0.20, 0.50],
        "ring_proxy_score": [0.20, 0.70, 0.15, 0.80],
    }
    return {
        "mp_candidate_id": str(candidate_id),
        "source_marker_id": "unit__marker_000000",
        "hotspot_cluster_id": 1,
        "care_area_family_id": f"fam_{care_type}",
        "care_area_instance_id": f"inst_{candidate_id}",
        "care_area_type": str(care_type),
        "care_area_match_score": 0.9,
        "care_area_homogeneity_score": 0.8,
        "metrology_context_group_id": f"{care_type}__high",
        "metrology_priority_class": "high",
        "metrology_priority_score": 0.8,
        "site_reliability_risk": 0.2,
        "recipe_waste_penalty": 0.2,
        "pool_status": "candidate",
        "pool_reject_reason": "",
        "mp_candidate_rank": 0,
        "mp_candidate_type": str(mp_type),
        "mp_verified": True,
        "mp_reject_reason": "",
        "mp_hotspot_score": 0.8,
        "mp_priority_score": 0.7,
        "mp_selection_gain": 0.6,
        "mp_x_um": 0.0,
        "mp_y_um": 0.0,
        "clip_bbox": [-0.2, -0.2, 0.2, 0.2],
        "bitmap_fingerprint": [0.1, 0.2, 0.3, 0.4],
        "ring_context": ring,
    }


def _memory_row(candidate_id: str, *, status: str, reason: str = "", af: bool = False, ap: bool = False, duplicate: bool = False) -> dict:
    """构造 pattern memory 单测用 recipe row。"""
    return {
        "mp_candidate_id": str(candidate_id),
        "recipe_status": str(status),
        "reject_reason": str(reason),
        "af_oas": "af.oas" if af else "",
        "af_reject_reason": "" if af else ("low_similarity" if "no_safe_af" in reason else ""),
        "ap_oas": "ap.oas" if ap else "",
        "ap_reject_reason": "" if ap else ("low_uniqueness" if "no_unique_ap" in reason else ""),
        "ap_global_duplicate": bool(duplicate),
        "ap_global_duplicate_with": "site_0000" if duplicate else "",
    }


def _args(
    root: Path,
    oas_path: Path,
    manifest: Path,
    *,
    max_sites: int = 1,
    mp_candidates_per_marker: int = 1,
) -> argparse.Namespace:
    """构造 recipe selector 测试参数。"""
    return argparse.Namespace(
        input_path=str(oas_path),
        marker_layer="999/0",
        behavior_manifest=str(manifest),
        output_dir=str(root / "recipe_out"),
        clip_size=0.4,
        mp_template_size_um=None,
        af_template_size_um=None,
        ap_template_size_um=None,
        max_sites=max_sites,
        mp_coverage_target=0.95,
        mp_search_radius_um=0.8,
        mp_candidates_per_marker=mp_candidates_per_marker,
        max_care_area_instances_per_family=80,
        min_feature_um=None,
        af_search_radius_um=0.8,
        sem_image_shift_limit_um=None,
        ap_search_radius_um=0.8,
        candidate_step_um=0.2,
        min_site_distance_um=0.3,
        recursive_input=False,
        apply_layer_ops=False,
        register_op=None,
        skip_pattern_memory_store_append=False,
    )


class RecipeSiteSelectorTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = SCRIPT_DIR / "test_outputs" / f"_recipe_site_selector_{uuid.uuid4().hex[:8]}"
        shutil.rmtree(self.temp_root, ignore_errors=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def _run_case(
        self,
        mode: str,
        *,
        risk_score: float = 1.0,
        max_sites: int = 1,
        mp_candidates_per_marker: int = 1,
        arg_overrides: dict | None = None,
    ) -> dict:
        oas_path = self.temp_root / "unit.oas"
        _make_oas(oas_path, mode)
        manifest = _make_manifest(self.temp_root, risk_score=risk_score)
        args = _args(
            self.temp_root,
            oas_path,
            manifest,
            max_sites=max_sites,
            mp_candidates_per_marker=mp_candidates_per_marker,
        )
        for key, value in (arg_overrides or {}).items():
            setattr(args, key, value)
        return selector.run_recipe_selector(args)

    def test_mp_selection_smoke_outputs_recipe_files(self):
        result = self._run_case("normal")

        self.assertTrue(Path(result["outputs"]["recipe_sites_csv"]).exists())
        self.assertTrue(Path(result["outputs"]["recipe_sites_json"]).exists())
        self.assertGreaterEqual(result["summary"]["total_output_rows"], 1)
        with Path(result["outputs"]["recipe_sites_csv"]).open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_marker_id"], "unit__marker_000000")
        self.assertIn("mp_hotspot_score", rows[0])
        self.assertIn("mp_verified", rows[0])
        self.assertIn("mp_reject_reason", rows[0])
        self.assertIn("mp_candidate_id", rows[0])
        self.assertIn("care_area_family_id", rows[0])
        self.assertIn("metrology_priority_score", rows[0])
        self.assertIn("metrology_context_group_id", rows[0])
        self.assertIn("af_reject_reason", rows[0])
        self.assertIn("ap_reject_reason", rows[0])
        self.assertIn("af_acceptance_checks_json", rows[0])
        self.assertIn("ap_acceptance_checks_json", rows[0])
        self.assertTrue(rows[0]["care_area_family_id"])
        self.assertTrue(Path(result["outputs"]["mp_candidate_pool_json"]).exists())
        self.assertTrue(Path(result["outputs"]["care_area_groups_json"]).exists())
        self.assertTrue(Path(result["outputs"]["metrology_context_audit_json"]).exists())
        self.assertTrue(Path(result["outputs"]["subset_objective_audit_json"]).exists())
        self.assertIn("subset_objective_score", result["summary"])
        self.assertIn("selected_subset_objective_by_category", result["summary"])

    def test_seed_family_extraction_writes_care_area_group(self):
        result = self._run_case("normal")

        self.assertGreaterEqual(result["summary"]["care_area_family_count"], 1)
        with Path(result["outputs"]["care_area_groups_json"]).open("r", encoding="utf-8") as handle:
            groups = json.load(handle)
        self.assertGreaterEqual(groups["care_area_family_count"], 1)
        self.assertIn(groups["families"][0]["care_area_type"], {"spacing", "line_end", "corner_jog", "density_transition"})
        self.assertEqual(groups["families"][0]["seed_marker_id"], "unit__marker_000000")

    def test_metrology_context_fields_propagate_to_reviews(self):
        result = self._run_case("normal")
        fields = {
            "metrology_priority_score",
            "metrology_priority_class",
            "site_reliability_risk",
            "recipe_waste_penalty",
            "metrology_context_group_id",
            "selection_profile_id",
        }

        with Path(result["outputs"]["care_area_groups_json"]).open("r", encoding="utf-8") as handle:
            groups = json.load(handle)
        with Path(result["outputs"]["mp_candidate_pool_json"]).open("r", encoding="utf-8") as handle:
            pool = json.load(handle)
        site_rows = [row for row in result["sites"] if str(row["site_id"]).startswith("site_")]
        site_summary_path = Path(result["outputs"]["recipe_review_dir"]) / site_rows[0]["site_id"] / "site_summary.json"
        with site_summary_path.open("r", encoding="utf-8") as handle:
            site_summary = json.load(handle)

        self.assertTrue(fields.issubset(groups["families"][0].keys()))
        self.assertTrue(fields.issubset(groups["families"][0]["instances"][0].keys()))
        self.assertTrue(fields.issubset(pool[0].keys()))
        self.assertTrue(fields.issubset(site_rows[0].keys()))
        self.assertTrue(fields.issubset(site_summary["mp_candidate"].keys()))

    def test_metrology_context_audit_matches_recipe_summary(self):
        result = self._run_case("care_area_expand", max_sites=2, mp_candidates_per_marker=3)

        with Path(result["outputs"]["metrology_context_audit_json"]).open("r", encoding="utf-8") as handle:
            audit = json.load(handle)

        self.assertIn("by_metrology_priority_class", audit)
        self.assertIn("by_care_area_type", audit)
        self.assertIn("by_metrology_context_group", audit)
        self.assertEqual(
            audit["summary"]["metrology_context_group_count"],
            result["summary"]["metrology_context_group_count"],
        )
        self.assertEqual(
            audit["summary"]["selected_metrology_context_group_count"],
            result["summary"]["selected_metrology_context_group_count"],
        )
        self.assertEqual(
            audit["summary"]["selected_by_metrology_context_group"],
            result["summary"]["selected_by_metrology_context_group"],
        )
        self.assertEqual(
            sum(audit["summary"]["selected_by_metrology_priority_class"].values()),
            result["summary"]["selected_recipe_site_count"],
        )

    def test_ring_context_feature_smoke(self):
        bitmap = np.zeros((32, 32), dtype=bool)
        bitmap[8:24, 14:18] = True

        context = ring_ctx.compute_ring_context(bitmap, pixel_size_um=0.02)

        self.assertEqual(len(context["ring_density_profile"]), len(ring_ctx.DEFAULT_RING_RADII_UM))
        self.assertEqual(len(context["ring_edge_crossing_profile"]), len(ring_ctx.DEFAULT_RING_RADII_UM))
        self.assertEqual(len(context["ring_asymmetry_profile"]), len(ring_ctx.DEFAULT_RING_RADII_UM))
        self.assertEqual(len(context["ring_pattern_code"]), len(ring_ctx.DEFAULT_RING_RADII_UM))
        self.assertTrue(all(0.0 <= float(value) <= 1.0 for value in context["ring_density_profile"]))
        self.assertTrue(all(0.0 <= float(value) <= 1.0 for value in context["ring_edge_crossing_profile"]))
        self.assertTrue(all(isinstance(value, int) for value in context["ring_pattern_code"]))
        self.assertIn("ring_selected_radii_um", context)

    def test_ring_context_dp_selects_spaced_radii(self):
        selected = ring_ctx.select_nonredundant_radii(
            [0.10, 0.15, 0.20, 0.45],
            [0.9, 0.8, 0.7, 0.6],
            max_count=2,
            min_spacing_um=0.20,
        )

        radii = selected["selected_radii_um"]
        self.assertEqual(len(radii), 2)
        self.assertGreaterEqual(abs(float(radii[1]) - float(radii[0])), 0.20)
        self.assertIn(0, selected["selected_indices"])

    def test_ring_context_audit_writes_pool_and_site_summary(self):
        result = self._run_case("normal")

        with Path(result["outputs"]["mp_candidate_pool_json"]).open("r", encoding="utf-8") as handle:
            pool = json.load(handle)
        site_rows = [row for row in result["sites"] if str(row["site_id"]).startswith("site_")]
        site_summary_path = Path(result["outputs"]["recipe_review_dir"]) / site_rows[0]["site_id"] / "site_summary.json"
        with site_summary_path.open("r", encoding="utf-8") as handle:
            site_summary = json.load(handle)

        self.assertIn("ring_context", pool[0])
        self.assertIn("ring_density_profile", pool[0]["ring_context"])
        self.assertIn("bitmap_fingerprint", pool[0])
        self.assertNotIn("ring_context", pool[0]["mp_risk_components"])
        self.assertIn("ring_context", site_summary["mp_candidate"])
        self.assertIn("ring_edge_crossing_profile", site_summary["mp_candidate"]["ring_context"])

    def test_recipe_sites_json_uses_compact_pool_and_capped_source_refs(self):
        result = self._run_case("care_area_expand", max_sites=2, mp_candidates_per_marker=3)

        with Path(result["outputs"]["recipe_sites_json"]).open("r", encoding="utf-8") as handle:
            recipe_json = json.load(handle)
        with Path(result["outputs"]["mp_candidate_pool_json"]).open("r", encoding="utf-8") as handle:
            full_pool = json.load(handle)

        self.assertIn("mp_candidate_pool", recipe_json)
        self.assertGreaterEqual(len(recipe_json["mp_candidate_pool"]), 1)
        self.assertNotIn("bitmap_fingerprint", recipe_json["mp_candidate_pool"][0])
        self.assertNotIn("ring_context", recipe_json["mp_candidate_pool"][0])
        self.assertIn("bitmap_fingerprint", full_pool[0])
        self.assertIn("ring_context", full_pool[0])

        site_rows = [row for row in result["sites"] if str(row["site_id"]).startswith("site_")]
        site_summary_path = Path(result["outputs"]["recipe_review_dir"]) / site_rows[0]["site_id"] / "site_summary.json"
        with site_summary_path.open("r", encoding="utf-8") as handle:
            site_summary = json.load(handle)
        mp_summary = site_summary["mp_candidate"]
        self.assertIn("source_marker_candidate_total_count", mp_summary)
        self.assertIn("source_marker_candidate_status_counts", mp_summary)
        self.assertLessEqual(len(mp_summary["source_marker_top_candidates"]), 10)
        self.assertNotIn("bitmap_fingerprint", mp_summary["source_marker_top_candidates"][0])

    def test_source_marker_candidate_index_is_written(self):
        result = self._run_case("care_area_expand", max_sites=2, mp_candidates_per_marker=3)

        index_path = Path(result["outputs"]["source_marker_candidate_index_json"])
        self.assertTrue(index_path.exists())
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)

        self.assertGreaterEqual(index["source_marker_count"], 1)
        marker_index = index["markers"]["unit__marker_000000"]
        self.assertGreaterEqual(marker_index["total_count"], 1)
        self.assertIn("status_counts", marker_index)
        self.assertLessEqual(len(marker_index["top_candidates"]), 10)
        self.assertIn("mp_candidate_id", marker_index["top_candidates"][0])

    def test_skip_pattern_memory_store_append_keeps_per_run_export(self):
        result = self._run_case("normal", arg_overrides={"skip_pattern_memory_store_append": True})

        self.assertTrue(result["summary"]["pattern_memory_store_append_skipped"])
        self.assertEqual(result["summary"]["pattern_memory_store_record_count"], 0)
        self.assertEqual(result["summary"]["pattern_memory_store_added_record_count"], 0)
        self.assertTrue(Path(result["outputs"]["pattern_memory_records_jsonl"]).exists())
        self.assertTrue(Path(result["outputs"]["pattern_memory_vectors_npz"]).exists())

    def test_care_area_prescore_limits_use_lightweight_formula(self):
        result = self._run_case(
            "care_area_expand",
            max_sites=1,
            mp_candidates_per_marker=1,
            arg_overrides={"max_care_area_instances_per_family": 30, "skip_pattern_memory_store_append": True},
        )

        with Path(result["outputs"]["care_area_groups_json"]).open("r", encoding="utf-8") as handle:
            groups = json.load(handle)
        audits = [family["anchor_table_audit"] for family in groups["families"] if family.get("anchor_table_audit")]

        self.assertTrue(audits)
        self.assertEqual(audits[0]["pre_score_limit"], 180)
        self.assertEqual(audits[0]["pre_score_pre_nms_match_limit"], 60)

    def test_pattern_memory_export_is_compact_and_summarized(self):
        result = self._run_case("normal")
        records_path = Path(result["outputs"]["pattern_memory_records_jsonl"])
        vectors_path = Path(result["outputs"]["pattern_memory_vectors_npz"])

        self.assertTrue(records_path.exists())
        self.assertTrue(vectors_path.exists())
        self.assertEqual(result["summary"]["pattern_memory_record_count"], result["summary"]["mp_candidate_pool_count"])
        self.assertGreater(result["summary"]["pattern_memory_estimated_disk_bytes"], 0)
        with records_path.open("r", encoding="utf-8") as handle:
            record = json.loads(handle.readline())
        self.assertNotIn("clip_bitmap", record)
        self.assertIn("vector_index", record)
        self.assertIn("ring_context", record)
        with np.load(vectors_path) as vectors_npz:
            vectors = vectors_npz["vectors"]
            candidate_ids = vectors_npz["candidate_ids"]
        self.assertEqual(vectors.shape[0], result["summary"]["pattern_memory_record_count"])
        self.assertEqual(candidate_ids.shape[0], result["summary"]["pattern_memory_record_count"])
        self.assertGreater(vectors.shape[1], 0)
        self.assertTrue(Path(result["outputs"]["pattern_memory_store_manifest"]).exists())
        self.assertTrue(Path(result["outputs"]["pattern_memory_store_memory_audit_json"]).exists())
        self.assertTrue(Path(result["outputs"]["pattern_memory_store_ring_outcome_audit_json"]).exists())
        self.assertGreaterEqual(result["summary"]["pattern_memory_store_record_count"], result["summary"]["pattern_memory_record_count"])

    def test_pattern_memory_store_appends_and_deduplicates_export(self):
        export_dir = self.temp_root / "export_a"
        store_dir = self.temp_root / "memory_store"
        candidates = [_memory_candidate("cand_a"), _memory_candidate("cand_b", care_type="line_end")]
        rows = [
            _memory_row("cand_a", status="selected", af=True, ap=True),
            _memory_row("cand_b", status="rejected", reason="no_safe_af"),
        ]
        pat_mem.export_pattern_memory(mp_candidate_pool=candidates, rows=rows, output_dir=export_dir)

        first = pat_mem.append_pattern_memory_export(export_dir=export_dir, store_dir=store_dir)
        second = pat_mem.append_pattern_memory_export(export_dir=export_dir, store_dir=store_dir)

        self.assertEqual(first["record_count"], 2)
        self.assertEqual(first["added_record_count"], 2)
        self.assertEqual(second["record_count"], 2)
        self.assertEqual(second["added_record_count"], 0)
        self.assertEqual(second["duplicate_skipped_count"], 2)
        with (store_dir / "records.jsonl").open("r", encoding="utf-8") as handle:
            record = json.loads(handle.readline())
        self.assertNotIn("clip_bitmap", record)
        self.assertIn("vector_hash", record)
        self.assertTrue((store_dir / "manifest.json").exists())
        self.assertTrue((store_dir / "vectors.npz").exists())

    def test_memory_prior_empty_store_is_neutral(self):
        audit = pat_mem.build_memory_prior_audit([_memory_candidate("query")], store_dir=self.temp_root / "missing_store")
        prior = audit["by_candidate_id"]["query"]

        self.assertEqual(prior["memory_neighbor_count"], 0)
        self.assertEqual(prior["memory_recipe_success_prior"], 0.5)
        self.assertEqual(prior["memory_af_success_prior"], 0.5)
        self.assertEqual(prior["memory_prior_confidence"], 0.0)

    def test_memory_prior_uses_nearest_historical_outcomes(self):
        export_dir = self.temp_root / "export_prior"
        store_dir = self.temp_root / "memory_store_prior"
        candidates = [_memory_candidate("hist_selected"), _memory_candidate("hist_rejected")]
        rows = [
            _memory_row("hist_selected", status="selected", af=True, ap=True),
            _memory_row("hist_rejected", status="rejected", reason="no_unique_ap", af=True, ap=False),
        ]
        pat_mem.export_pattern_memory(mp_candidate_pool=candidates, rows=rows, output_dir=export_dir)
        pat_mem.append_pattern_memory_export(export_dir=export_dir, store_dir=store_dir)

        audit = pat_mem.build_memory_prior_audit([_memory_candidate("query")], store_dir=store_dir, top_k=5, min_similarity=0.70)
        prior = audit["by_candidate_id"]["query"]

        self.assertEqual(prior["memory_neighbor_count"], 2)
        self.assertGreater(prior["memory_nearest_similarity"], 0.99)
        self.assertGreater(prior["memory_prior_confidence"], 0.0)
        self.assertAlmostEqual(prior["memory_recipe_success_prior"], 0.5)
        self.assertAlmostEqual(prior["memory_ap_success_prior"], 0.5)

    def test_memory_prior_does_not_mutate_mp_scoring_fields(self):
        result = self._run_case("normal")
        with Path(result["outputs"]["mp_candidate_pool_json"]).open("r", encoding="utf-8") as handle:
            candidate = json.load(handle)[0]

        self.assertIn("memory_prior_audit", candidate)
        self.assertIn("memory_prior_candidate_count", result["summary"])
        self.assertNotIn("memory_recipe_success_prior", candidate["mp_risk_components"])
        self.assertNotIn("memory_prior_confidence", candidate["mp_risk_components"])

    def test_review_evidence_audit_is_written_without_scoring_mutation(self):
        result = self._run_case("normal")
        with Path(result["outputs"]["mp_candidate_pool_json"]).open("r", encoding="utf-8") as handle:
            candidate = json.load(handle)[0]
        summary_path = Path(result["outputs"]["review_evidence_audit_json"])

        self.assertTrue(summary_path.exists())
        self.assertIn("graph_context_audit", candidate)
        self.assertIn("evidence_contradiction_audit", candidate)
        self.assertIn("pattern_taxonomy_audit", candidate)
        self.assertIn("expected_feasibility_audit", candidate)
        self.assertIn("review_taxonomy_by_class", result["summary"])
        self.assertIn("expected_recipe_feasibility_proxy", candidate["expected_feasibility_audit"])
        self.assertNotIn("graph_env_complexity", candidate["mp_risk_components"])
        self.assertNotIn("defect_evidence_proxy_score", candidate["mp_risk_components"])
        self.assertNotIn("expected_recipe_feasibility_proxy", candidate["mp_risk_components"])
        details = result["site_details"][0]["mp_candidate"]
        self.assertIn("graph_context_audit", details)
        self.assertIn("evidence_contradiction_audit", details)

    def test_subset_objective_fields_are_written_to_reviews(self):
        result = self._run_case("normal")
        with Path(result["outputs"]["mp_candidate_pool_json"]).open("r", encoding="utf-8") as handle:
            candidate = json.load(handle)[0]
        audit_path = Path(result["outputs"]["subset_objective_audit_json"])

        self.assertTrue(audit_path.exists())
        self.assertIn("subset_objective_components", candidate)
        self.assertIn("subset_objective_target_bins", candidate)
        self.assertIn("subset_objective_status", candidate)
        self.assertIn("subset_objective_marginal_gain", candidate)
        self.assertIn("objective_candidate_value", candidate["subset_objective_components"])
        with audit_path.open("r", encoding="utf-8") as handle:
            audit = json.load(handle)
        self.assertIn("target_distribution", audit)
        self.assertIn("selected_marginal_gain_trace", audit)
        details = result["site_details"][0]["mp_candidate"]
        self.assertIn("subset_objective_components", details)
        self.assertIn("subset_objective_target_bins", details)

    def test_pattern_memory_audit_stats_are_smoothed_by_group(self):
        export_dir = self.temp_root / "export_b"
        store_dir = self.temp_root / "memory_store_b"
        candidates = [
            _memory_candidate("cand_selected", care_type="spacing", mp_type="critical_spacing_anchor"),
            _memory_candidate("cand_af", care_type="spacing", mp_type="critical_spacing_anchor"),
            _memory_candidate("cand_ap", care_type="line_end", mp_type="fragment_line_end_anchor"),
            _memory_candidate("cand_dup", care_type="line_end", mp_type="fragment_line_end_anchor"),
        ]
        rows = [
            _memory_row("cand_selected", status="selected", af=True, ap=True),
            _memory_row("cand_af", status="rejected", reason="no_safe_af"),
            _memory_row("cand_ap", status="rejected", reason="no_unique_ap"),
            _memory_row("cand_dup", status="rejected", reason="ap_global_duplicate", af=True, ap=True, duplicate=True),
        ]
        pat_mem.export_pattern_memory(mp_candidate_pool=candidates, rows=rows, output_dir=export_dir)
        pat_mem.append_pattern_memory_export(export_dir=export_dir, store_dir=store_dir)

        with (store_dir / "memory_audit.json").open("r", encoding="utf-8") as handle:
            audit = json.load(handle)

        self.assertEqual(audit["summary"]["record_count"], 4)
        self.assertEqual(audit["by_care_area_type"]["spacing"]["record_count"], 2)
        self.assertEqual(audit["by_care_area_type"]["spacing"]["selected_count"], 1)
        self.assertEqual(audit["by_care_area_type"]["spacing"]["af_fail_count"], 1)
        self.assertIn("no_safe_af", audit["by_care_area_type"]["spacing"]["reject_reasons"])
        self.assertGreater(audit["by_care_area_type"]["spacing"]["recipe_success_prior"], 0.0)
        self.assertLess(audit["by_care_area_type"]["spacing"]["recipe_success_prior"], 1.0)
        self.assertEqual(audit["by_mp_candidate_type"]["fragment_line_end_anchor"]["ap_duplicate_count"], 1)

    def test_ring_outcome_audit_records_radius_mi_and_dp_selection(self):
        records = []
        for index in range(6):
            candidate = _memory_candidate(f"cand_ring_{index}", care_type="spacing")
            candidate["ring_context"]["ring_density_profile"] = [0.10, 0.90 if index < 3 else 0.20, 0.30, 0.80 if index % 2 else 0.10]
            outcome = _memory_row(
                f"cand_ring_{index}",
                status="selected" if index < 3 else "rejected",
                reason="" if index < 3 else "no_unique_ap",
                af=index < 3,
                ap=index < 3,
            )
            export_dir = self.temp_root / f"ring_export_{index}"
            pat_mem.export_pattern_memory(mp_candidate_pool=[candidate], rows=[outcome], output_dir=export_dir)
            with (export_dir / "records.jsonl").open("r", encoding="utf-8") as handle:
                records.append(json.loads(handle.readline()))

        audit = pat_mem.build_ring_outcome_audit(records)

        self.assertEqual(audit["summary"]["record_count"], 6)
        self.assertGreaterEqual(len(audit["radii"]), 4)
        self.assertTrue(all("outcome_mi_proxy" in item for item in audit["radii"]))
        selected = audit["summary"]["selected_radii_um"]
        for left, right in zip(selected, selected[1:]):
            self.assertGreaterEqual(abs(float(right) - float(left)), 0.20)
        self.assertTrue(any(item["mean_mi_proxy"] > 0.0 for item in audit["radii"]))

    def test_lookalike_expansion_adds_far_care_area_instances(self):
        result = self._run_case("care_area_expand", max_sites=2, mp_candidates_per_marker=3)

        with Path(result["outputs"]["care_area_groups_json"]).open("r", encoding="utf-8") as handle:
            groups = json.load(handle)
        centers = [
            instance["center_um"]
            for family in groups["families"]
            for instance in family["instances"]
            if instance["care_area_match_score"] >= 0.78
        ]
        audits = [family["anchor_table_audit"] for family in groups["families"]]
        self.assertTrue(any(abs(float(center[0])) > 0.8 for center in centers))
        self.assertGreaterEqual(result["summary"]["care_area_instance_count"], 2)
        self.assertTrue(all("pre_score_candidate_count" in audit for audit in audits))
        self.assertTrue(all(audit["instantiated_anchor_count"] <= audit["pre_score_candidate_count"] for audit in audits))
        self.assertTrue(all("pre_score_match_count" in audit for audit in audits))
        self.assertTrue(all("pre_score_reject_reasons" in audit for audit in audits))
        self.assertTrue(all("pre_score_final_instance_count" in audit for audit in audits))
        self.assertTrue(all("pre_score_fallback_anchor_count" in audit for audit in audits))
        self.assertTrue(all("pre_score_match_rate_by_source" in audit for audit in audits))
        self.assertTrue(all("pre_score_early_stop_final_shortfall" in audit for audit in audits))
        self.assertTrue(all("pre_score_final_instance_count_by_source" in audit for audit in audits))
        self.assertTrue(all("pre_score_final_instance_rate_by_source" in audit for audit in audits))

    def test_homogeneity_rejects_low_match_instance(self):
        result = self._run_case("care_area_homogeneity", max_sites=1, mp_candidates_per_marker=3)

        with Path(result["outputs"]["care_area_groups_json"]).open("r", encoding="utf-8") as handle:
            groups = json.load(handle)
        self.assertTrue(any(family["rejected_instance_count"] > 0 for family in groups["families"]))
        self.assertTrue(all(instance["care_area_match_score"] >= 0.78 for family in groups["families"] for instance in family["instances"]))
        for family in groups["families"]:
            self.assertEqual(family["rejected_instance_count"], sum(family["instance_reject_reasons"].values()))

    def test_expanded_rarity_inherits_seed_semantics(self):
        bitmap = np.zeros((16, 16), dtype=bool)
        bitmap[:, 7:9] = True
        window = {"center": (0.0, 0.0), "clip_bitmap": bitmap, "clip_bbox": [-0.2, -0.2, 0.2, 0.2]}
        seed = mp_gen.MPCandidate(
            0.0,
            0.0,
            0.0,
            "critical_spacing_anchor",
            ["critical_spacing_anchor"],
            window,
            score=0.9,
            components={"local_rarity": 0.8, "critical_geometry_score": 1.0, "proposal_voting": 1.0},
            verified=True,
        )
        instances = [
            argparse.Namespace(
                is_seed_instance=False,
                care_area_type="spacing",
                window=window,
                signature=[1.0 for _ in range(10)],
                match_score=0.9,
                homogeneity_score=1.0,
                signature_quality=1.0,
                anchor_type_match=1.0,
                bitmap_fingerprint_similarity=0.1,
            ),
            argparse.Namespace(
                is_seed_instance=False,
                care_area_type="spacing",
                window=window,
                signature=[1.0 for _ in range(10)],
                match_score=0.9,
                homogeneity_score=1.0,
                signature_quality=1.0,
                anchor_type_match=1.0,
                bitmap_fingerprint_similarity=0.9,
            ),
        ]
        family = argparse.Namespace(
            instances=instances,
            cluster_size=1,
            seed_candidate=seed,
            care_area_type="spacing",
            behavior_risk=0.0,
            homogeneity_score=1.0,
            signature=[1.0 for _ in range(10)],
        )

        care_gen._refresh_metrology_contexts(family)

        rarity_values = [instance.metrology_context_components["pattern_rarity"] for instance in instances]
        self.assertAlmostEqual(rarity_values[0], rarity_values[1])
        self.assertAlmostEqual(rarity_values[0], 0.72)

    def test_care_area_bitmap_similarity_uses_weighted_components(self):
        left = np.zeros((16, 16), dtype=bool)
        right = np.zeros((16, 16), dtype=bool)
        left[:, 1:3] = True
        right[:, 12:14] = True

        parts = care_gen._bitmap_similarity_parts(left, right, np.asarray([1.0, 0.0]), np.asarray([1.0, 0.0]))

        self.assertAlmostEqual(parts["bitmap_shifted_iou"], 0.0)
        self.assertAlmostEqual(parts["bitmap_fingerprint_similarity"], 1.0)
        self.assertLess(parts["bitmap_similarity"], parts["bitmap_fingerprint_similarity"])

    def test_metrology_context_scores_high_weak_pattern(self):
        bitmap = np.zeros((16, 16), dtype=bool)
        bitmap[:, 7:9] = True

        context = met_ctx.compute_metrology_context(
            care_area_type="spacing",
            bitmap=bitmap,
            components={"critical_geometry_score": 1.0, "proposal_voting": 1.0},
            inherited_behavior_risk=1.0,
            family_representativeness=1.0,
            pattern_rarity=0.8,
            mp_localization_confidence=1.0,
            family_homogeneity=1.0,
            signature_quality=1.0,
            mp_verified=True,
        )

        self.assertGreaterEqual(context.metrology_priority_score, 0.66)
        self.assertEqual(context.metrology_priority_class, "high")
        self.assertTrue(context.metrology_context_group_id.startswith("spacing__"))

    def test_metrology_context_marks_low_reliability(self):
        bitmap = np.ones((16, 16), dtype=bool)

        context = met_ctx.compute_metrology_context(
            care_area_type="spacing",
            bitmap=bitmap,
            components={"critical_geometry_score": 0.1},
            inherited_behavior_risk=0.0,
            family_representativeness=0.2,
            pattern_rarity=0.0,
            mp_localization_confidence=0.0,
            family_homogeneity=0.0,
            signature_quality=0.0,
            mp_verified=False,
        )

        self.assertGreaterEqual(context.site_reliability_risk, 0.75)
        self.assertAlmostEqual(context.recipe_waste_penalty, context.site_reliability_risk)

    def test_metrology_geometry_prior_does_not_saturate(self):
        bitmap = np.zeros((16, 16), dtype=bool)
        bitmap[:, 7:9] = True

        weak = met_ctx.compute_metrology_context(
            care_area_type="spacing",
            bitmap=bitmap,
            components={},
            inherited_behavior_risk=0.0,
            family_representativeness=0.0,
            pattern_rarity=0.0,
            mp_localization_confidence=0.0,
        )
        strong = met_ctx.compute_metrology_context(
            care_area_type="spacing",
            bitmap=bitmap,
            components={"critical_geometry_score": 1.0},
            inherited_behavior_risk=0.0,
            family_representativeness=0.0,
            pattern_rarity=0.0,
            mp_localization_confidence=0.0,
        )

        self.assertLess(weak.components["hotspot_geometry_risk"], 1.0)
        self.assertGreater(strong.components["hotspot_geometry_risk"], weak.components["hotspot_geometry_risk"])

    def test_signature_sparse_penalty_increases_reliability_risk(self):
        bitmap = np.zeros((16, 16), dtype=bool)
        bitmap[:, 7:9] = True
        kwargs = {
            "care_area_type": "spacing",
            "bitmap": bitmap,
            "components": {"critical_geometry_score": 1.0},
            "inherited_behavior_risk": 1.0,
            "family_representativeness": 1.0,
            "pattern_rarity": 1.0,
            "mp_localization_confidence": 1.0,
            "family_homogeneity": 1.0,
            "mp_verified": True,
        }

        reliable = met_ctx.compute_metrology_context(**kwargs, signature_quality=1.0)
        sparse = met_ctx.compute_metrology_context(**kwargs, signature_quality=0.0)

        self.assertGreater(sparse.site_reliability_risk, reliable.site_reliability_risk + 0.08)

    def test_candidate_level_context_differs_within_same_instance(self):
        bitmap = np.zeros((16, 16), dtype=bool)
        bitmap[:, 7:9] = True
        family = argparse.Namespace(care_area_type="spacing")
        instance = argparse.Namespace(homogeneity_score=1.0, signature_quality=1.0)
        weak = mp_gen.MPCandidate(
            0.0,
            0.0,
            0.0,
            "critical_spacing_anchor",
            ["critical_spacing_anchor"],
            {"center": (0.0, 0.0), "clip_bitmap": bitmap, "clip_bbox": [-0.2, -0.2, 0.2, 0.2]},
            score=0.4,
            components={"critical_geometry_score": 0.1, "proposal_voting": 0.1, "local_rarity": 0.1},
            verified=True,
        )
        strong = mp_gen.MPCandidate(
            0.0,
            0.0,
            0.0,
            "critical_spacing_anchor",
            ["critical_spacing_anchor"],
            {"center": (0.0, 0.0), "clip_bitmap": bitmap, "clip_bbox": [-0.2, -0.2, 0.2, 0.2]},
            score=0.9,
            components={"critical_geometry_score": 1.0, "proposal_voting": 1.0, "local_rarity": 1.0},
            verified=True,
        )

        weak_summary, _, _, _ = selector._mp_candidate_metrology_summary(
            family=family,
            instance=instance,
            mp_candidate=weak,
            effective_behavior_risk=1.0,
            coverage_weight=1,
        )
        strong_summary, _, _, _ = selector._mp_candidate_metrology_summary(
            family=family,
            instance=instance,
            mp_candidate=strong,
            effective_behavior_risk=1.0,
            coverage_weight=1,
        )

        self.assertGreater(strong_summary["metrology_priority_score"], weak_summary["metrology_priority_score"])

    def test_lightweight_instance_reuses_instance_context(self):
        instance = argparse.Namespace(
            metrology_priority_score=0.42,
            metrology_priority_class="mid",
            site_reliability_risk=0.23,
            recipe_waste_penalty=0.23,
            metrology_context_group_id="spacing__mid",
            selection_profile_id="metrology_profile__spacing__mid",
            metrology_context_components={
                "pattern_rarity": 0.31,
                "mp_localization_confidence": 0.62,
                "family_representativeness": 0.73,
            },
        )

        summary, rarity, localization, representativeness = selector._lightweight_instance_metrology_summary(
            instance=instance,
            coverage_weight=5,
        )

        self.assertEqual(summary["metrology_priority_score"], 0.42)
        self.assertEqual(summary["metrology_context_group_id"], "spacing__mid")
        self.assertAlmostEqual(rarity, 0.31)
        self.assertAlmostEqual(localization, 0.62)
        self.assertAlmostEqual(representativeness, 0.73)

    def test_care_area_anchor_type_match_penalizes_subtype_mismatch(self):
        exact = care_gen._anchor_type_match("fragment_facing_pair_anchor", "fragment_facing_pair_anchor")
        fragment_to_spacing = care_gen._anchor_type_match("fragment_facing_pair_anchor", "critical_spacing_anchor")
        different_type = care_gen._anchor_type_match("fragment_facing_pair_anchor", "fragment_line_end_anchor")

        self.assertEqual(exact, 1.0)
        self.assertGreater(fragment_to_spacing, 0.0)
        self.assertLess(fragment_to_spacing, exact)
        self.assertEqual(different_type, 0.0)

    def test_care_area_anchor_cap_audit_is_recorded(self):
        count = 2001
        indexed = [{"bbox": (float(index), 0.0, float(index) + 0.01, 0.01), "element": None} for index in range(count)]
        layout_index = LayoutIndex(
            indexed_elements=indexed,
            bbox_x0=np.asarray([item["bbox"][0] for item in indexed], dtype=np.float64),
            bbox_y0=np.asarray([item["bbox"][1] for item in indexed], dtype=np.float64),
            bbox_x1=np.asarray([item["bbox"][2] for item in indexed], dtype=np.float64),
            bbox_y1=np.asarray([item["bbox"][3] for item in indexed], dtype=np.float64),
            marker_polygons=[],
        )

        _, audit = care_gen._build_seeded_anchor_table(
            layout_index=layout_index,
            target_types=set(),
            local_radius_um=0.4,
            step_um=0.2,
            pixel_size_um=0.01,
            min_feature_um=None,
        )

        self.assertTrue(audit["anchor_table_cap_hit"])
        self.assertEqual(audit["total_layout_element_count"], count)
        self.assertEqual(audit["processed_element_count"], care_gen.MAX_ANCHOR_ELEMENTS)
        self.assertIn("tile_coverage_ratio", audit)
        self.assertGreater(audit["tile_coverage_ratio"], 0.0)
        self.assertFalse(audit["seed_proximity_weighted"])
        self.assertEqual(audit["seed_center_count"], 0)
        self.assertFalse(audit["seed_distance_precomputed"])

    def test_care_area_anchor_cap_records_seed_proximity(self):
        count = 2001
        indexed = [{"bbox": (float(index), 0.0, float(index) + 0.01, 0.01), "element": None} for index in range(count)]
        layout_index = LayoutIndex(
            indexed_elements=indexed,
            bbox_x0=np.asarray([item["bbox"][0] for item in indexed], dtype=np.float64),
            bbox_y0=np.asarray([item["bbox"][1] for item in indexed], dtype=np.float64),
            bbox_x1=np.asarray([item["bbox"][2] for item in indexed], dtype=np.float64),
            bbox_y1=np.asarray([item["bbox"][3] for item in indexed], dtype=np.float64),
            marker_polygons=[],
        )

        _, audit = care_gen._build_seeded_anchor_table(
            layout_index=layout_index,
            target_types=set(),
            local_radius_um=0.4,
            step_um=0.2,
            pixel_size_um=0.01,
            min_feature_um=None,
            seed_centers=[(1999.0, 0.0)],
        )

        self.assertTrue(audit["seed_proximity_weighted"])
        self.assertEqual(audit["seed_center_count"], 1)
        self.assertTrue(audit["seed_distance_precomputed"])

    def test_prescored_anchor_selection_keeps_tile_exploration(self):
        anchors = [
            (
                1.0 - index * 0.01,
                {"x": float(index) * 0.01, "y": 0.0, "sources": ["critical_spacing_anchor"], "metrics": {}},
            )
            for index in range(10)
        ]
        anchors.append((0.10, {"x": 10.0, "y": 10.0, "sources": ["critical_spacing_anchor"], "metrics": {}}))

        selected, audit, source_by_key = care_gen._select_prescored_anchors(
            anchors,
            instantiate_cap=5,
            layout_bbox=(0.0, 0.0, 10.0, 10.0),
        )

        self.assertEqual(len(selected), 5)
        self.assertTrue(audit["pre_score_stratified"])
        self.assertGreater(audit["pre_score_tile_anchor_count"], 0)
        self.assertIn("pre_score_fallback_anchor_count", audit)
        self.assertEqual(len(source_by_key), len(selected))
        self.assertTrue(any(float(anchor["x"]) >= 10.0 for _, anchor in selected))

    def test_batch_prescore_matches_single_anchor_score(self):
        bitmap = np.zeros((16, 16), dtype=bool)
        bitmap[:, 7:9] = True
        window = {"center": (0.0, 0.0), "clip_bitmap": bitmap, "clip_bbox": [-0.2, -0.2, 0.2, 0.2]}
        seed = mp_gen.MPCandidate(
            0.0,
            0.0,
            0.0,
            "critical_spacing_anchor",
            ["critical_spacing_anchor"],
            window,
            score=0.9,
            components={"fragment_facing_pair_score": 1.0, "proposal_voting": 1.0},
            verified=True,
        )
        family = care_gen.CareAreaFamily(
            family_id="cafam",
            family_rank=0,
            seed_marker_id="marker",
            cluster_id=0,
            source_path="unit.oas",
            marker_ids=["marker"],
            representative_metadata={},
            cluster={},
            seed_center=(0.0, 0.0),
            seed_candidate=seed,
            seed_discovery=None,
            care_area_type="spacing",
            behavior_risk=1.0,
            cluster_size=1,
            fingerprint=np.zeros((8,), dtype=np.float32),
            signature=care_gen._signature_from_components(seed.components, gap_norm_um=0.1),
            signature_gap_norm_um=0.1,
        )
        anchors = {
            (0, 0): {"x": 0.0, "y": 0.0, "sources": ["critical_spacing_anchor"], "metrics": {"fragment_facing_pair_score": 1.0}},
            (1, 0): {"x": 1.0, "y": 0.0, "sources": ["fragment_line_end_anchor"], "metrics": {"fragment_line_end_score": 1.0}},
        }

        batch = care_gen._prescore_typed_anchors(family, anchors)

        expected_score = (
            0.45 * care_gen._anchor_type_match(family.seed_candidate.candidate_type, "critical_spacing_anchor")
            + 0.35
            * care_gen._signature_similarity(
                family.signature,
                care_gen._signature_from_components(anchors[(0, 0)]["metrics"], gap_norm_um=family.signature_gap_norm_um),
            )
            + 0.20 * care_gen._anchor_source_strength(["critical_spacing_anchor"])
        )

        self.assertEqual(len(batch), 1)
        self.assertAlmostEqual(batch[0][0], expected_score)

    def test_density_transition_signature_records_density_metrics(self):
        signature = care_gen._signature_from_components(
            {"density_transition_score": 0.8, "density_local": 0.35},
            gap_norm_um=0.30,
        )

        self.assertAlmostEqual(signature[-2], 0.8)
        self.assertAlmostEqual(signature[-1], 0.35)

    def test_sparse_signature_fallback_accepts_strong_bitmap_match(self):
        rect = gdstk.rectangle((-0.05, -0.05), (0.05, 0.05), layer=1, datatype=0)
        indexed = [{"bbox": (-0.05, -0.05, 0.05, 0.05), "element": rect}]
        layout_index = LayoutIndex(
            indexed_elements=indexed,
            bbox_x0=np.asarray([-0.05], dtype=np.float64),
            bbox_y0=np.asarray([-0.05], dtype=np.float64),
            bbox_x1=np.asarray([0.05], dtype=np.float64),
            bbox_y1=np.asarray([0.05], dtype=np.float64),
            marker_polygons=[],
        )
        window = care_gen.rasterize_centered_window(layout_index, (0.0, 0.0), 0.4, 0.02)
        seed = mp_gen.MPCandidate(
            0.0,
            0.0,
            0.0,
            "critical_spacing_anchor",
            ["critical_spacing_anchor"],
            window,
            score=0.9,
            verified=True,
        )
        discovery = mp_gen.MPDiscoveryResult(seed, [seed], 1, 1, 0, 0, 0, "unit", True)
        family = care_gen.CareAreaFamily(
            family_id="family_sparse",
            family_rank=0,
            seed_marker_id="marker_0",
            cluster_id=0,
            source_path="unit.oas",
            marker_ids=["marker_0"],
            representative_metadata={},
            cluster={},
            seed_center=(0.0, 0.0),
            seed_candidate=seed,
            seed_discovery=discovery,
            care_area_type="spacing",
            behavior_risk=0.5,
            cluster_size=1,
            fingerprint=care_gen.bitmap_fingerprint(window["clip_bitmap"]),
            signature=[1.0 for _ in range(10)],
            signature_gap_norm_um=0.30,
        )

        instance, reason = care_gen._instance_from_anchor(
            family,
            layout_index=layout_index,
            raw_anchor={"x": 0.0, "y": 0.0, "sources": ["critical_spacing_anchor"], "metrics": {}},
            window_size_um=0.4,
            pixel_size_um=0.02,
        )

        self.assertEqual(reason, "")
        self.assertIsNotNone(instance)
        self.assertLess(instance.signature_quality, 0.50)
        self.assertGreaterEqual(instance.match_score, care_gen.CARE_AREA_MATCH_THRESHOLD)

    def test_signature_prefilter_rejects_mismatch_before_rasterize(self):
        bitmap = np.zeros((16, 16), dtype=bool)
        bitmap[:, 7:9] = True
        window = {"center": (0.0, 0.0), "clip_bitmap": bitmap, "clip_bbox": [-0.2, -0.2, 0.2, 0.2]}
        seed = mp_gen.MPCandidate(
            0.0,
            0.0,
            0.0,
            "critical_spacing_anchor",
            ["critical_spacing_anchor"],
            window,
            score=0.9,
            verified=True,
        )
        discovery = mp_gen.MPDiscoveryResult(seed, [seed], 1, 1, 0, 0, 0, "unit", True)
        family = care_gen.CareAreaFamily(
            family_id="family_signature_gate",
            family_rank=0,
            seed_marker_id="marker_0",
            cluster_id=0,
            source_path="unit.oas",
            marker_ids=["marker_0"],
            representative_metadata={},
            cluster={},
            seed_center=(0.0, 0.0),
            seed_candidate=seed,
            seed_discovery=discovery,
            care_area_type="spacing",
            behavior_risk=0.5,
            cluster_size=1,
            fingerprint=care_gen.bitmap_fingerprint(bitmap),
            signature=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            signature_gap_norm_um=0.30,
        )
        layout_index = LayoutIndex(
            indexed_elements=[],
            bbox_x0=np.asarray([], dtype=np.float64),
            bbox_y0=np.asarray([], dtype=np.float64),
            bbox_x1=np.asarray([], dtype=np.float64),
            bbox_y1=np.asarray([], dtype=np.float64),
            marker_polygons=[],
        )

        instance, reason = care_gen._instance_from_anchor(
            family,
            layout_index=layout_index,
            raw_anchor={
                "x": 0.0,
                "y": 0.0,
                "sources": ["critical_spacing_anchor"],
                "metrics": {
                    "internal_facing_distance_um": 0.30,
                    "external_facing_distance_um": 0.30,
                    "layout_complexity": 1.0,
                    "proposal_voting": 1.0,
                    "density_transition_score": 1.0,
                    "density_local": 1.0,
                },
            },
            window_size_um=0.4,
            pixel_size_um=0.02,
        )

        self.assertIsNone(instance)
        self.assertEqual(reason, "signature_gate_reject")

    def test_family_merge_keeps_max_behavior_risk(self):
        bitmap = np.zeros((16, 16), dtype=bool)
        bitmap[:, 7:9] = True
        window = {"center": (0.0, 0.0), "clip_bitmap": bitmap, "clip_bbox": [-0.2, -0.2, 0.2, 0.2]}

        def family(family_id: str, marker_id: str, risk: float, score: float) -> care_gen.CareAreaFamily:
            seed = mp_gen.MPCandidate(
                0.0,
                0.0,
                0.0,
                "critical_spacing_anchor",
                ["critical_spacing_anchor"],
                window,
                score=score,
                verified=True,
            )
            discovery = mp_gen.MPDiscoveryResult(seed, [seed], 1, 1, 0, 0, 0, "unit", True)
            return care_gen.CareAreaFamily(
                family_id=family_id,
                family_rank=0,
                seed_marker_id=marker_id,
                cluster_id=0,
                source_path="unit.oas",
                marker_ids=[marker_id],
                representative_metadata={},
                cluster={},
                seed_center=(0.0, 0.0),
                seed_candidate=seed,
                seed_discovery=discovery,
                care_area_type="spacing",
                behavior_risk=risk,
                cluster_size=1,
                fingerprint=care_gen.bitmap_fingerprint(bitmap),
                signature=[1.0 for _ in range(10)],
                signature_gap_norm_um=0.30,
                seed_behavior_risk=risk,
                merged_behavior_risk_values=[risk],
                merged_cluster_ids=[0],
            )

        merged = care_gen._merge_duplicate_families(
            [family("family_low", "marker_low", 0.2, 0.9), family("family_high", "marker_high", 0.9, 0.8)],
            radius_um=0.2,
        )

        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged[0].behavior_risk, 0.9)
        self.assertEqual(set(merged[0].marker_ids), {"marker_low", "marker_high"})
        self.assertIn("family_high", merged[0].merged_seed_family_ids)

    def test_effective_behavior_risk_records_care_area_attenuation_factor(self):
        result = self._run_case("care_area_expand", max_sites=2, mp_candidates_per_marker=2)

        with Path(result["outputs"]["mp_candidate_pool_json"]).open("r", encoding="utf-8") as handle:
            pool = json.load(handle)
        expanded = [item for item in pool if not str(item["care_area_instance_id"]).endswith("inst_0000")]

        self.assertTrue(expanded)
        for item in expanded:
            components = item["mp_risk_components"]
            self.assertLessEqual(components["effective_behavior_risk"], components["seed_behavior_risk"])
            self.assertLessEqual(components["risk_attenuation_factor"], 1.0)

    def test_expanded_instances_use_lightweight_rank0_mp(self):
        result = self._run_case("care_area_expand", max_sites=2, mp_candidates_per_marker=2)

        with Path(result["outputs"]["mp_candidate_pool_json"]).open("r", encoding="utf-8") as handle:
            pool = json.load(handle)
        expanded = [item for item in pool if not str(item["care_area_instance_id"]).endswith("inst_0000")]

        self.assertTrue(expanded)
        self.assertTrue(any(item["mp_rule_coverage_audit"].get("care_area_lightweight_instance") for item in expanded))
        self.assertTrue(all(int(item["mp_candidate_rank"]) == 0 for item in expanded))

    def test_lightweight_instance_rejects_uniform_bitmap(self):
        bitmap = np.ones((16, 16), dtype=bool)
        window = {"center": (0.0, 0.0), "clip_bitmap": bitmap, "clip_bbox": [-0.2, -0.2, 0.2, 0.2]}
        seed = mp_gen.MPCandidate(
            0.0,
            0.0,
            0.0,
            "critical_spacing_anchor",
            ["critical_spacing_anchor"],
            window,
            score=0.9,
            verified=True,
            verification_components={"density": 0.5},
        )
        discovery = mp_gen.MPDiscoveryResult(seed, [seed], 1, 1, 0, 0, 0, "unit", True)
        family = care_gen.CareAreaFamily(
            family_id="cafam_unit",
            family_rank=0,
            seed_marker_id="marker_0",
            cluster_id=0,
            source_path="unit.oas",
            marker_ids=["marker_0"],
            representative_metadata={},
            cluster={},
            seed_center=(0.0, 0.0),
            seed_candidate=seed,
            seed_discovery=discovery,
            care_area_type="spacing",
            behavior_risk=0.8,
            cluster_size=1,
            fingerprint=care_gen.bitmap_fingerprint(bitmap),
            signature=[1.0 for _ in range(10)],
            signature_gap_norm_um=0.30,
        )
        instance = care_gen.CareAreaInstance(
            instance_id="cafam_unit__inst_0001",
            family_id="cafam_unit",
            instance_rank=1,
            source_path="unit.oas",
            center=(0.4, 0.0),
            bbox=(-0.2, -0.2, 0.2, 0.2),
            window=window,
            care_area_type="spacing",
            match_score=0.9,
            bitmap_similarity=0.9,
            bitmap_shifted_iou=0.9,
            bitmap_fingerprint_similarity=0.9,
            fragment_signature_similarity=0.9,
            anchor_type_match=1.0,
            homogeneity_score=0.9,
            raw_sources=["critical_spacing_anchor"],
            signature=[1.0 for _ in range(10)],
        )

        result = selector._lightweight_discovery_from_instance(
            family,
            instance,
            behavior_risk=0.8,
            behavior_risk_enabled=True,
        )

        self.assertFalse(result.selected_candidate.verified)
        self.assertEqual(result.selected_candidate.verification_reason, "uniform_bitmap")
        self.assertAlmostEqual(result.selected_candidate.verification_components["density"], 1.0)

    def test_selected_expanded_instance_is_refined_after_selection(self):
        result = self._run_case("care_area_expand", max_sites=2, mp_candidates_per_marker=2)

        refined_rows = [
            row
            for row in result["sites"]
            if str(row["site_id"]).startswith("site_")
            and json.loads(row["mp_discovery_components_json"]).get("post_selection_refine") == 1.0
        ]

        self.assertGreaterEqual(result["summary"]["selected_expanded_mp_refine_attempted_count"], 1)
        self.assertGreaterEqual(result["summary"]["selected_expanded_mp_refined_count"], 1)
        self.assertTrue(refined_rows)
        self.assertTrue(all("refine_shift_um" in json.loads(row["mp_discovery_components_json"]) for row in refined_rows))
        self.assertTrue(all("post_refine_priority_stale" in json.loads(row["mp_risk_components_json"]) for row in refined_rows))

    def test_refine_failure_rejects_selected_lightweight_site(self):
        oas_path = self.temp_root / "unit.oas"
        _make_oas(oas_path, "care_area_expand")
        manifest = _make_manifest(self.temp_root)
        args = _args(self.temp_root, oas_path, manifest)
        window_cache = selector.RecipeWindowCache(
            marker_layer=args.marker_layer,
            clip_size_um=args.clip_size,
            output_dir=Path(args.output_dir),
            apply_layer_operations=False,
            layer_processor=None,
            recursive_input=False,
        )
        window = window_cache.window(oas_path, (1.2, 0.0), args.clip_size)
        candidate = mp_gen.MPCandidate(
            1.2,
            0.0,
            1.2,
            "critical_spacing_anchor",
            ["critical_spacing_anchor"],
            window,
            score=0.8,
            components={"care_area_lightweight_instance": 1.0, "local_rarity": 1.0},
            verified=True,
            verification_components={"density": 0.2},
        )
        discovery = mp_gen.MPDiscoveryResult(
            selected_candidate=candidate,
            top_candidates=[candidate],
            raw_candidate_count=1,
            rasterized_candidate_count=1,
            empty_rejected_count=0,
            nms_rejected_count=0,
            verification_rejected_count=0,
            mp_discovery_reason="care_area_instance_rank0",
            behavior_risk_enabled=True,
            rule_coverage_audit={"care_area_lightweight_instance": True},
        )
        info = selector.MPPoolCandidateInfo(
            cluster_id=0,
            representative_marker_id="unit__marker_000000",
            marker_ids=["unit__marker_000000"],
            representative_metadata={"source_path": str(oas_path)},
            cluster={},
            mp_candidate_id="cafam_unit__inst_0001__mpcand_000",
            mp_candidate_rank=0,
            mp_window=window,
            source_marker_center=(0.0, 0.0),
            mp_candidate_type="critical_spacing_anchor",
            mp_hotspot_score=0.8,
            mp_verified=True,
            mp_reject_reason="",
            mp_verification_components={"density": 0.2},
            mp_discovery_components={"care_area_lightweight_instance": 1.0},
            mp_discovery=discovery,
            raw_components={
                "mp_hotspot_score": 0.8,
                "behavior_risk": 0.8,
                "effective_behavior_risk": 0.8,
                "pattern_rarity": 1.0,
                "cluster_coverage": 1.0,
                "care_area_signature_quality": 1.0,
            },
            mp_priority_score=0.8,
            mp_selection_gain=0.8,
            pool_status="selected",
            care_area_family_id="cafam_unit",
            care_area_instance_id="cafam_unit__inst_0001",
            care_area_type="spacing",
            care_area_match_score=1.0,
            care_area_homogeneity_score=1.0,
            care_area_instance_count=2,
            care_area_seed_marker_id="unit__marker_000000",
            care_area_instance_bbox=[-0.2, -0.2, 0.2, 0.2],
            metrology_context_group_id="spacing__high",
            selection_profile_id="metrology_profile__spacing__high",
        )

        def failing_discovery(**kwargs):
            failed = mp_gen.MPCandidate(
                1.2,
                0.0,
                0.0,
                "local_grid_probe",
                ["local_grid_probe"],
                kwargs["marker_window"],
                score=0.0,
                components={},
                accepted=False,
                reject_reason="sparse_bitmap",
                verified=False,
                verification_reason="sparse_bitmap",
                verification_components={"density": 0.0},
            )
            return mp_gen.MPDiscoveryResult(failed, [failed], 1, 1, 0, 0, 1, "unit_fail", True)

        original = selector.discover_mp_candidates
        selector.discover_mp_candidates = failing_discovery
        try:
            row = selector._construct_selected_site(
                info,
                site_index=0,
                output_dir=Path(args.output_dir),
                review_dir=Path(args.output_dir) / "recipe_review",
                window_cache=window_cache,
                args=args,
                source_marker_candidates=[info],
            )
        finally:
            selector.discover_mp_candidates = original

        self.assertEqual(row["recipe_status"], "rejected")
        self.assertEqual(row["reject_reason"], "post_selection_refine_failed")
        self.assertFalse(row["mp_verified"])
        self.assertEqual(row["mp_reject_reason"], "post_selection_refine_failed")
        self.assertEqual(info.pool_status, "rejected")
        self.assertEqual(info.pool_reject_reason, "post_selection_refine_failed")
        self.assertEqual(info.raw_components["pre_refine_pool_status"], "selected")

    def test_seed_instance_behavior_risk_is_not_attenuated(self):
        result = self._run_case("care_area_expand", max_sites=1, mp_candidates_per_marker=2)

        with Path(result["outputs"]["mp_candidate_pool_json"]).open("r", encoding="utf-8") as handle:
            pool = json.load(handle)
        seed_items = [item for item in pool if str(item["care_area_instance_id"]).endswith("inst_0000")]
        self.assertTrue(seed_items)
        components = seed_items[0]["mp_risk_components"]
        self.assertAlmostEqual(components["effective_behavior_risk"], components["seed_behavior_risk"])
        self.assertAlmostEqual(components["risk_attenuation_factor"], 1.0)

    def test_singleton_family_reports_zero_expansion_confidence(self):
        result = self._run_case("normal", arg_overrides={"max_care_area_instances_per_family": 1})

        with Path(result["outputs"]["care_area_groups_json"]).open("r", encoding="utf-8") as handle:
            groups = json.load(handle)
        self.assertTrue(groups["families"])
        self.assertTrue(all(family["is_singleton_family"] for family in groups["families"]))
        self.assertTrue(all(family["care_area_expansion_confidence"] == 0.0 for family in groups["families"]))

    def test_topk_pool_selects_multiple_mps_from_same_marker(self):
        result = self._run_case("multi_mp", max_sites=2, mp_candidates_per_marker=4)

        site_rows = [row for row in result["sites"] if str(row["site_id"]).startswith("site_")]
        self.assertEqual(len(site_rows), 2)
        self.assertEqual({row["source_marker_id"] for row in site_rows}, {"unit__marker_000000"})
        self.assertEqual(len({row["mp_candidate_id"] for row in site_rows}), 2)
        self.assertEqual(result["summary"]["selected_mp_candidate_count"], 2)

    def test_budget_cap_marks_unselected_pool_candidates(self):
        result = self._run_case("multi_mp", max_sites=1, mp_candidates_per_marker=4)

        with Path(result["outputs"]["mp_candidate_pool_json"]).open("r", encoding="utf-8") as handle:
            pool = json.load(handle)
        self.assertEqual(sum(1 for item in pool if item["pool_status"] == "selected"), 1)
        self.assertTrue(any(item["pool_reject_reason"] == "mp_pool_over_budget" for item in pool))

    def test_rejected_member_rows_inherit_context_fields(self):
        bitmap = np.zeros((16, 16), dtype=bool)
        bitmap[:, 7:9] = True
        window = {"center": (0.0, 0.0), "clip_bitmap": bitmap, "clip_bbox": [-0.2, -0.2, 0.2, 0.2]}
        info = selector.MPPoolCandidateInfo(
            cluster_id=0,
            representative_marker_id="marker_selected",
            marker_ids=["marker_selected", "marker_covered"],
            representative_metadata={"source_path": "unit.oas", "marker_center": [0.0, 0.0], "clip_bbox": [-0.2, -0.2, 0.2, 0.2]},
            cluster={},
            mp_candidate_id="candidate",
            mp_candidate_rank=0,
            mp_window=window,
            source_marker_center=(0.0, 0.0),
            mp_candidate_type="critical_spacing_anchor",
            mp_hotspot_score=0.8,
            mp_verified=True,
            mp_reject_reason="",
            mp_verification_components={},
            mp_discovery_components={},
            mp_discovery=None,
            raw_components={"mp_hotspot_score": 0.8},
            mp_priority_score=0.7,
            care_area_family_id="cafam_0000",
            care_area_instance_id="cafam_0000__inst_0000",
            care_area_type="spacing",
            care_area_match_score=1.0,
            care_area_homogeneity_score=1.0,
            care_area_instance_count=2,
            care_area_seed_marker_id="marker_selected",
            care_area_instance_bbox=[-0.2, -0.2, 0.2, 0.2],
            metrology_priority_score=0.8,
            metrology_priority_class="high",
            site_reliability_risk=0.1,
            recipe_waste_penalty=0.1,
            metrology_context_group_id="spacing__high",
            selection_profile_id="metrology_profile__spacing__high",
            pool_status="selected",
        )

        rows = selector._rejected_member_rows(
            [info],
            selected_infos=[info],
            metadata_by_marker={"marker_covered": {"marker_center": [0.5, 0.0], "clip_bbox": [0.3, -0.2, 0.7, 0.2]}},
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["reject_reason"], "covered_by_representative")
        self.assertEqual(rows[0]["care_area_family_id"], "cafam_0000")
        self.assertEqual(rows[0]["metrology_context_group_id"], "spacing__high")
        self.assertEqual(rows[0]["metrology_priority_class"], "high")

    def test_failed_selected_info_does_not_cover_cluster_members(self):
        bitmap = np.zeros((16, 16), dtype=bool)
        bitmap[:, 7:9] = True
        window = {"center": (0.0, 0.0), "clip_bitmap": bitmap, "clip_bbox": [-0.2, -0.2, 0.2, 0.2]}
        info = selector.MPPoolCandidateInfo(
            cluster_id=0,
            representative_marker_id="marker_selected",
            marker_ids=["marker_selected", "marker_covered"],
            representative_metadata={"source_path": "unit.oas", "marker_center": [0.0, 0.0], "clip_bbox": [-0.2, -0.2, 0.2, 0.2]},
            cluster={},
            mp_candidate_id="candidate",
            mp_candidate_rank=0,
            mp_window=window,
            source_marker_center=(0.0, 0.0),
            mp_candidate_type="critical_spacing_anchor",
            mp_hotspot_score=0.8,
            mp_verified=False,
            mp_reject_reason="post_selection_refine_failed",
            mp_verification_components={},
            mp_discovery_components={},
            mp_discovery=None,
            raw_components={"mp_hotspot_score": 0.8},
            mp_priority_score=0.7,
            pool_status="rejected",
            pool_reject_reason="post_selection_refine_failed",
        )

        rows = selector._rejected_member_rows(
            [info],
            selected_infos=[],
            constructed_infos=[info],
            metadata_by_marker={"marker_covered": {"marker_center": [0.5, 0.0], "clip_bbox": [0.3, -0.2, 0.7, 0.2]}},
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_marker_id"], "marker_covered")
        self.assertEqual(rows[0]["reject_reason"], "over_budget")

    def test_global_pool_duplicate_suppression_marks_lower_score_candidate(self):
        bitmap = np.zeros((16, 16), dtype=bool)
        bitmap[:, 7:9] = True
        window_a = {"center": (0.0, 0.0), "clip_bitmap": bitmap, "clip_bbox": [-0.2, -0.2, 0.2, 0.2]}
        window_b = {"center": (0.05, 0.0), "clip_bitmap": bitmap, "clip_bbox": [-0.15, -0.2, 0.25, 0.2]}

        def info(candidate_id: str, score: float, window: dict) -> selector.MPPoolCandidateInfo:
            return selector.MPPoolCandidateInfo(
                cluster_id=0,
                representative_marker_id="unit__marker_000000",
                marker_ids=["unit__marker_000000"],
                representative_metadata={"source_path": "unit.oas"},
                cluster={},
                mp_candidate_id=candidate_id,
                mp_candidate_rank=0,
                mp_window=window,
                source_marker_center=(0.0, 0.0),
                mp_candidate_type="critical_spacing_anchor",
                mp_hotspot_score=score,
                mp_verified=True,
                mp_reject_reason="",
                mp_verification_components={"density": 0.2},
                mp_discovery_components={"local_rarity": 1.0, "pattern_rarity": 1.0},
                mp_discovery=None,
                raw_components={"mp_hotspot_score": score, "behavior_risk": 1.0, "pattern_rarity": 1.0, "cluster_coverage": 1.0},
                mp_priority_score=score,
            )

        candidates = [info("a", 0.9, window_a), info("b", 0.8, window_b)]
        selected = selector._select_mp_candidate_pool(candidates, max_sites=2, duplicate_radius_um=0.10)

        self.assertEqual([item.mp_candidate_id for item in selected], ["a"])
        self.assertEqual(candidates[1].pool_reject_reason, "mp_pool_duplicate")

    def test_selection_duplicate_suppression_keeps_partial_similarity_candidate(self):
        left = np.zeros((16, 16), dtype=bool)
        left[:, 7:9] = True
        right = left.copy()
        for y, x in np.argwhere(right)[:7]:
            right[y, x] = False
        window_a = {"center": (0.0, 0.0), "clip_bitmap": left, "clip_bbox": [-0.2, -0.2, 0.2, 0.2]}
        window_b = {"center": (0.05, 0.0), "clip_bitmap": right, "clip_bbox": [-0.15, -0.2, 0.25, 0.2]}

        def info(candidate_id: str, score: float, window: dict) -> selector.MPPoolCandidateInfo:
            return selector.MPPoolCandidateInfo(
                cluster_id=0,
                representative_marker_id="unit__marker_000000",
                marker_ids=["unit__marker_000000"],
                representative_metadata={"source_path": "unit.oas"},
                cluster={},
                mp_candidate_id=candidate_id,
                mp_candidate_rank=0,
                mp_window=window,
                source_marker_center=(0.0, 0.0),
                mp_candidate_type="critical_spacing_anchor",
                mp_hotspot_score=score,
                mp_verified=True,
                mp_reject_reason="",
                mp_verification_components={},
                mp_discovery_components={"local_rarity": 1.0},
                mp_discovery=None,
                raw_components={"mp_hotspot_score": score, "behavior_risk": 1.0, "pattern_rarity": 1.0, "cluster_coverage": 1.0},
                mp_priority_score=score,
            )

        candidates = [info("a", 0.9, window_a), info("b", 0.8, window_b)]
        similarity = selector._pool_duplicate_similarity(candidates[0], candidates[1])
        selected = selector._select_mp_candidate_pool(candidates, max_sites=2, duplicate_radius_um=0.10)

        self.assertGreaterEqual(similarity, 0.88)
        self.assertLess(similarity, 0.92)
        self.assertEqual([item.mp_candidate_id for item in selected], ["a", "b"])

    def test_pool_duplicate_similarity_requires_shifted_support(self):
        left = np.zeros((16, 16), dtype=bool)
        right = np.zeros((16, 16), dtype=bool)
        left[:, 1:3] = True
        right[:, 12:14] = True
        window_a = {"center": (0.0, 0.0), "clip_bitmap": left, "clip_bbox": [-0.2, -0.2, 0.2, 0.2]}
        window_b = {"center": (0.05, 0.0), "clip_bitmap": right, "clip_bbox": [-0.15, -0.2, 0.25, 0.2]}

        def info(candidate_id: str, window: dict) -> selector.MPPoolCandidateInfo:
            return selector.MPPoolCandidateInfo(
                cluster_id=0,
                representative_marker_id="unit__marker_000000",
                marker_ids=["unit__marker_000000"],
                representative_metadata={"source_path": "unit.oas"},
                cluster={},
                mp_candidate_id=candidate_id,
                mp_candidate_rank=0,
                mp_window=window,
                source_marker_center=(0.0, 0.0),
                mp_candidate_type="critical_spacing_anchor",
                mp_hotspot_score=0.8,
                mp_verified=True,
                mp_reject_reason="",
                mp_verification_components={},
                mp_discovery_components={"local_rarity": 1.0},
                mp_discovery=None,
                raw_components={"mp_hotspot_score": 0.8, "behavior_risk": 1.0, "pattern_rarity": 1.0, "cluster_coverage": 1.0},
                mp_priority_score=0.8,
            )

        candidates = [info("a", window_a), info("b", window_b)]
        similarity = selector._pool_duplicate_similarity(candidates[0], candidates[1])
        selected = selector._select_mp_candidate_pool(candidates, max_sites=2, duplicate_radius_um=0.10)

        self.assertLess(similarity, 0.88)
        self.assertEqual(len(selected), 2)

    def test_mp_pool_pre_dedup_runs_before_rarity_scoring(self):
        bitmap = np.zeros((16, 16), dtype=bool)
        bitmap[:, 7:9] = True
        window_a = {"center": (0.0, 0.0), "clip_bitmap": bitmap, "clip_bbox": [-0.2, -0.2, 0.2, 0.2]}
        window_b = {"center": (0.05, 0.0), "clip_bitmap": bitmap, "clip_bbox": [-0.15, -0.2, 0.25, 0.2]}

        def info(candidate_id: str, score: float, window: dict) -> selector.MPPoolCandidateInfo:
            return selector.MPPoolCandidateInfo(
                cluster_id=0,
                representative_marker_id="unit__marker_000000",
                marker_ids=["unit__marker_000000"],
                representative_metadata={"source_path": "unit.oas"},
                cluster={},
                mp_candidate_id=candidate_id,
                mp_candidate_rank=0,
                mp_window=window,
                source_marker_center=(0.0, 0.0),
                mp_candidate_type="critical_spacing_anchor",
                mp_hotspot_score=score,
                mp_verified=True,
                mp_reject_reason="",
                mp_verification_components={},
                mp_discovery_components={"local_rarity": 1.0},
                mp_discovery=None,
                raw_components={"mp_hotspot_score": score, "behavior_risk": 1.0, "pattern_rarity": 1.0, "cluster_coverage": 1.0},
            )

        candidates = [info("a", 0.9, window_a), info("b", 0.8, window_b)]
        rejected = selector._pre_dedup_mp_candidate_pool(candidates, duplicate_radius_um=0.10)
        selector._score_mp_candidates(candidates)

        self.assertEqual(rejected, 1)
        self.assertEqual(candidates[1].pool_reject_reason, "mp_pool_preduplicate")
        self.assertAlmostEqual(candidates[1].score_components["global_candidate_rarity"], 0.0)

    def test_budget_selection_prefers_care_area_family_diversity(self):
        bitmap = np.zeros((16, 16), dtype=bool)
        bitmap[:, 7:9] = True
        window_a = {"center": (0.0, 0.0), "clip_bitmap": bitmap, "clip_bbox": [-0.2, -0.2, 0.2, 0.2]}
        window_b = {"center": (1.0, 0.0), "clip_bitmap": bitmap, "clip_bbox": [0.8, -0.2, 1.2, 0.2]}
        window_c = {"center": (2.0, 0.0), "clip_bitmap": bitmap, "clip_bbox": [1.8, -0.2, 2.2, 0.2]}

        def info(candidate_id: str, family_id: str, care_type: str, score: float, window: dict) -> selector.MPPoolCandidateInfo:
            return selector.MPPoolCandidateInfo(
                cluster_id=0,
                representative_marker_id="unit__marker_000000",
                marker_ids=["unit__marker_000000"],
                representative_metadata={"source_path": "unit.oas"},
                cluster={},
                mp_candidate_id=candidate_id,
                mp_candidate_rank=0,
                mp_window=window,
                source_marker_center=(0.0, 0.0),
                mp_candidate_type="critical_spacing_anchor",
                mp_hotspot_score=score,
                mp_verified=True,
                mp_reject_reason="",
                mp_verification_components={"density": 0.2},
                mp_discovery_components={"local_rarity": 1.0, "pattern_rarity": 1.0},
                mp_discovery=None,
                raw_components={"mp_hotspot_score": score, "behavior_risk": 1.0, "pattern_rarity": 1.0, "cluster_coverage": 1.0},
                mp_priority_score=score,
                care_area_family_id=family_id,
                care_area_instance_id=f"{family_id}__inst_0000",
                care_area_type=care_type,
                care_area_match_score=1.0,
                care_area_homogeneity_score=1.0,
                care_area_instance_count=1,
                care_area_seed_marker_id="unit__marker_000000",
            )

        candidates = [
            info("a1", "family_a", "spacing", 0.95, window_a),
            info("a2", "family_a", "spacing", 0.90, window_b),
            info("b1", "family_b", "line_end", 0.78, window_c),
        ]
        selected = selector._select_mp_candidate_pool(candidates, max_sites=2, duplicate_radius_um=0.10)

        self.assertEqual({item.care_area_family_id for item in selected}, {"family_a", "family_b"})

    def test_budget_selection_prefers_metrology_context_diversity(self):
        bitmap = np.zeros((16, 16), dtype=bool)
        bitmap[:, 7:9] = True
        window_a = {"center": (0.0, 0.0), "clip_bitmap": bitmap, "clip_bbox": [-0.2, -0.2, 0.2, 0.2]}
        window_b = {"center": (1.0, 0.0), "clip_bitmap": bitmap, "clip_bbox": [0.8, -0.2, 1.2, 0.2]}
        window_c = {"center": (2.0, 0.0), "clip_bitmap": bitmap, "clip_bbox": [1.8, -0.2, 2.2, 0.2]}

        def info(candidate_id: str, context_group: str, score: float, window: dict) -> selector.MPPoolCandidateInfo:
            priority_class = context_group.split("__")[-1]
            return selector.MPPoolCandidateInfo(
                cluster_id=0,
                representative_marker_id="unit__marker_000000",
                marker_ids=["unit__marker_000000"],
                representative_metadata={"source_path": "unit.oas"},
                cluster={},
                mp_candidate_id=candidate_id,
                mp_candidate_rank=0,
                mp_window=window,
                source_marker_center=(0.0, 0.0),
                mp_candidate_type="critical_spacing_anchor",
                mp_hotspot_score=score,
                mp_verified=True,
                mp_reject_reason="",
                mp_verification_components={"density": 0.2},
                mp_discovery_components={"local_rarity": 1.0, "pattern_rarity": 1.0},
                mp_discovery=None,
                raw_components={"mp_hotspot_score": score, "behavior_risk": 1.0, "pattern_rarity": 1.0, "cluster_coverage": 1.0},
                mp_priority_score=score,
                care_area_family_id="family_a",
                care_area_instance_id=f"{candidate_id}__inst_0000",
                care_area_type="spacing",
                care_area_match_score=1.0,
                care_area_homogeneity_score=1.0,
                care_area_instance_count=3,
                care_area_seed_marker_id="unit__marker_000000",
                metrology_priority_score=score,
                metrology_priority_class=priority_class,
                site_reliability_risk=0.0,
                recipe_waste_penalty=0.0,
                metrology_context_group_id=context_group,
                selection_profile_id=f"metrology_profile__{context_group}",
            )

        candidates = [
            info("high_a", "spacing__high", 0.95, window_a),
            info("high_b", "spacing__high", 0.92, window_b),
            info("mid_a", "spacing__mid", 0.88, window_c),
        ]
        selected = selector._select_mp_candidate_pool(candidates, max_sites=2, duplicate_radius_um=0.10)

        self.assertEqual([item.mp_candidate_id for item in selected], ["high_a", "mid_a"])
        self.assertEqual({item.metrology_context_group_id for item in selected}, {"spacing__high", "spacing__mid"})

    def test_objective_selection_records_diminishing_return_components(self):
        bitmap = np.zeros((16, 16), dtype=bool)
        bitmap[:, 7:9] = True
        windows = [
            {"center": (float(index), 0.0), "clip_bitmap": bitmap, "clip_bbox": [float(index) - 0.2, -0.2, float(index) + 0.2, 0.2]}
            for index in range(3)
        ]

        def info(candidate_id: str, family_id: str, score: float, window: dict) -> selector.MPPoolCandidateInfo:
            return selector.MPPoolCandidateInfo(
                cluster_id=0,
                representative_marker_id="unit__marker_000000",
                marker_ids=["unit__marker_000000"],
                representative_metadata={"source_path": "unit.oas"},
                cluster={},
                mp_candidate_id=candidate_id,
                mp_candidate_rank=0,
                mp_window=window,
                source_marker_center=(0.0, 0.0),
                mp_candidate_type="critical_spacing_anchor",
                mp_hotspot_score=score,
                mp_verified=True,
                mp_reject_reason="",
                mp_verification_components={},
                mp_discovery_components={"local_rarity": 1.0},
                mp_discovery=None,
                raw_components={"mp_hotspot_score": score, "behavior_risk": 1.0, "pattern_rarity": 1.0, "cluster_coverage": 1.0},
                mp_priority_score=score,
                care_area_family_id=family_id,
                care_area_type="spacing",
                metrology_context_group_id="spacing__high",
            )

        candidates = [
            info("a1", "family_a", 0.95, windows[0]),
            info("a2", "family_a", 0.94, windows[1]),
            info("b1", "family_b", 0.90, windows[2]),
        ]
        selected = selector._select_mp_candidate_pool(candidates, max_sites=2, duplicate_radius_um=0.10)

        self.assertEqual([item.mp_candidate_id for item in selected], ["a1", "b1"])
        self.assertIn("family_coverage_gain", candidates[1].subset_objective_components)
        self.assertLess(
            candidates[1].subset_objective_components["family_coverage_gain"],
            candidates[2].subset_objective_components["family_coverage_gain"],
        )
        self.assertEqual(candidates[1].subset_objective_status, "mp_pool_over_budget")
        self.assertEqual(candidates[1].pool_reject_reason, "mp_pool_over_budget")

    def test_objective_selection_penalizes_htc_waste(self):
        bitmap = np.zeros((16, 16), dtype=bool)
        bitmap[:, 7:9] = True
        window_a = {"center": (0.0, 0.0), "clip_bitmap": bitmap, "clip_bbox": [-0.2, -0.2, 0.2, 0.2]}
        window_b = {"center": (1.0, 0.0), "clip_bitmap": bitmap, "clip_bbox": [0.8, -0.2, 1.2, 0.2]}

        def info(candidate_id: str, priority: float, waste: float, taxonomy: str, htc_score: float, window: dict) -> selector.MPPoolCandidateInfo:
            candidate = selector.MPPoolCandidateInfo(
                cluster_id=0,
                representative_marker_id=candidate_id,
                marker_ids=[candidate_id],
                representative_metadata={"source_path": "unit.oas"},
                cluster={},
                mp_candidate_id=candidate_id,
                mp_candidate_rank=0,
                mp_window=window,
                source_marker_center=(0.0, 0.0),
                mp_candidate_type="critical_spacing_anchor",
                mp_hotspot_score=priority,
                mp_verified=True,
                mp_reject_reason="",
                mp_verification_components={},
                mp_discovery_components={"local_rarity": 1.0},
                mp_discovery=None,
                raw_components={"mp_hotspot_score": priority, "behavior_risk": 1.0, "pattern_rarity": 1.0, "cluster_coverage": 1.0},
                mp_priority_score=priority,
                care_area_family_id=candidate_id,
                care_area_type="spacing",
                metrology_priority_score=priority,
                recipe_waste_penalty=waste,
                metrology_context_group_id="spacing__high",
            )
            candidate.evidence_contradiction_audit = {"defect_evidence_proxy_score": priority}
            candidate.expected_feasibility_audit = {"expected_recipe_feasibility_proxy": 1.0 - 0.5 * waste}
            candidate.pattern_taxonomy_audit = {"pattern_taxonomy_class": taxonomy, "htc_like_score": htc_score}
            return candidate

        candidates = [
            info("htc_high_priority", 0.99, 1.0, "htc_like", 1.0, window_a),
            info("tnsb_lower_priority", 0.82, 0.0, "tnsb_like", 0.0, window_b),
        ]
        selected = selector._select_mp_candidate_pool(candidates, max_sites=1, duplicate_radius_um=0.10)

        self.assertEqual([item.mp_candidate_id for item in selected], ["tnsb_lower_priority"])
        self.assertGreater(candidates[0].subset_objective_components["htc_waste_penalty"], 0.9)

    def test_objective_selection_keeps_no_hard_quota_behavior(self):
        bitmap = np.zeros((16, 16), dtype=bool)
        bitmap[:, 7:9] = True

        def info(candidate_id: str, x: float) -> selector.MPPoolCandidateInfo:
            return selector.MPPoolCandidateInfo(
                cluster_id=0,
                representative_marker_id=candidate_id,
                marker_ids=[candidate_id],
                representative_metadata={"source_path": "unit.oas"},
                cluster={},
                mp_candidate_id=candidate_id,
                mp_candidate_rank=0,
                mp_window={"center": (x, 0.0), "clip_bitmap": bitmap, "clip_bbox": [x - 0.2, -0.2, x + 0.2, 0.2]},
                source_marker_center=(0.0, 0.0),
                mp_candidate_type="critical_spacing_anchor",
                mp_hotspot_score=0.8,
                mp_verified=True,
                mp_reject_reason="",
                mp_verification_components={},
                mp_discovery_components={"local_rarity": 1.0},
                mp_discovery=None,
                raw_components={"mp_hotspot_score": 0.8, "behavior_risk": 1.0, "pattern_rarity": 1.0, "cluster_coverage": 1.0},
                mp_priority_score=0.8,
                care_area_family_id="only_family",
                care_area_type="spacing",
                metrology_context_group_id="spacing__high",
            )

        candidates = [info("a", 0.0), info("b", 1.0), info("c", 2.0)]
        selected = selector._select_mp_candidate_pool(candidates, max_sites=2, duplicate_radius_um=0.10)

        self.assertEqual(len(selected), 2)
        self.assertTrue(all(item.care_area_family_id == "only_family" for item in selected))

    def test_objective_target_distribution_uses_weighted_pool_share(self):
        candidates = [
            {
                "mp_candidate_id": "spacing_high",
                "pool_status": "candidate",
                "mp_verified": True,
                "mp_hotspot_score": 1.0,
                "mp_priority_score": 1.0,
                "care_area_type": "spacing",
                "care_area_family_id": "family_a",
                "metrology_context_group_id": "spacing__high",
                "metrology_priority_score": 1.0,
                "recipe_waste_penalty": 0.0,
                "score_components": {"effective_behavior_risk": 1.0, "pattern_novelty": 1.0},
                "evidence_contradiction_audit": {"defect_evidence_proxy_score": 1.0},
                "expected_feasibility_audit": {"expected_recipe_feasibility_proxy": 1.0},
                "pattern_taxonomy_audit": {"pattern_taxonomy_class": "tnsb_like", "htc_like_score": 0.0},
            },
            {
                "mp_candidate_id": "line_low",
                "pool_status": "candidate",
                "mp_verified": True,
                "mp_hotspot_score": 0.2,
                "mp_priority_score": 0.2,
                "care_area_type": "line_end",
                "care_area_family_id": "family_b",
                "metrology_context_group_id": "line_end__mid",
                "metrology_priority_score": 0.2,
                "recipe_waste_penalty": 0.0,
                "score_components": {"effective_behavior_risk": 0.2, "pattern_novelty": 0.2},
                "evidence_contradiction_audit": {"defect_evidence_proxy_score": 0.2},
                "expected_feasibility_audit": {"expected_recipe_feasibility_proxy": 1.0},
                "pattern_taxonomy_audit": {"pattern_taxonomy_class": "ambiguous", "htc_like_score": 0.0},
            },
        ]
        meta = subset_sel.prepare_candidates(candidates, max_sites=4)
        targets = meta["target_distribution"]["care_area_type"]

        self.assertGreater(targets["spacing"]["weighted_pool_share"], targets["line_end"]["weighted_pool_share"])
        self.assertGreaterEqual(targets["spacing"]["target_count"], targets["line_end"]["target_count"])

    def test_subset_objective_audit_writes_gaps(self):
        candidates = [
            {"mp_candidate_id": "spacing", "pool_status": "candidate", "mp_verified": True, "mp_hotspot_score": 0.9, "mp_priority_score": 0.9, "care_area_type": "spacing", "care_area_family_id": "family_a", "metrology_context_group_id": "spacing__high", "metrology_priority_score": 0.9, "recipe_waste_penalty": 0.0, "score_components": {"effective_behavior_risk": 1.0, "pattern_novelty": 1.0}, "evidence_contradiction_audit": {"defect_evidence_proxy_score": 0.9}, "expected_feasibility_audit": {"expected_recipe_feasibility_proxy": 1.0}, "pattern_taxonomy_audit": {"pattern_taxonomy_class": "tnsb_like", "htc_like_score": 0.0}},
            {"mp_candidate_id": "line", "pool_status": "candidate", "mp_verified": True, "mp_hotspot_score": 0.8, "mp_priority_score": 0.8, "care_area_type": "line_end", "care_area_family_id": "family_b", "metrology_context_group_id": "line_end__high", "metrology_priority_score": 0.8, "recipe_waste_penalty": 0.0, "score_components": {"effective_behavior_risk": 1.0, "pattern_novelty": 1.0}, "evidence_contradiction_audit": {"defect_evidence_proxy_score": 0.8}, "expected_feasibility_audit": {"expected_recipe_feasibility_proxy": 1.0}, "pattern_taxonomy_audit": {"pattern_taxonomy_class": "tnsb_like", "htc_like_score": 0.0}},
        ]
        meta = subset_sel.prepare_candidates(candidates, max_sites=2)
        candidates[0]["pool_status"] = "selected"
        candidates[1]["pool_status"] = "rejected"
        candidates[1]["pool_reject_reason"] = "mp_pool_over_budget"
        audit = subset_sel.build_subset_objective_audit(
            mp_candidate_pool=candidates,
            site_details=[],
            target_distribution=meta["target_distribution"],
        )

        self.assertGreater(audit["summary"]["subset_objective_gap_count"], 0)
        self.assertTrue(any(gap["category"] == "care_area_type" and gap["bin"] == "line_end" for gap in audit["coverage_gaps"]))

    def test_subset_objective_audit_marks_high_risk_non_executable(self):
        audit = subset_sel.build_subset_objective_audit(
            mp_candidate_pool=[],
            site_details=[
                {
                    "site": {"site_id": "site_0000", "recipe_status": "rejected", "reject_reason": "no_safe_af"},
                    "mp_candidate": {
                        "mp_candidate_id": "cand_high",
                        "subset_objective_components": {
                            "objective_risk_score": 0.8,
                            "objective_feasibility_score": 0.7,
                            "objective_candidate_value": 0.68,
                        },
                        "subset_objective_target_bins": {"care_area_type": "spacing"},
                        "subset_objective_marginal_gain": 0.6,
                    },
                }
            ],
        )

        self.assertEqual(audit["summary"]["subset_objective_high_risk_non_executable_count"], 1)
        self.assertEqual(audit["high_risk_non_executable_candidates"][0]["mp_candidate_id"], "cand_high")

    def test_invalid_mp_pool_candidates_do_not_consume_budget(self):
        result = self._run_case("fallback_mp")

        self.assertEqual(result["summary"]["selected_mp_candidate_count"], 0)
        self.assertEqual(result["summary"]["eligible_mp_candidate_count"], 0)
        self.assertEqual(result["summary"]["rejected_care_area_seed_count"], 1)
        with Path(result["outputs"]["mp_candidate_pool_json"]).open("r", encoding="utf-8") as handle:
            pool = json.load(handle)
        self.assertEqual(pool, [])
        self.assertTrue(any(row["reject_reason"] == "no_care_area_family" for row in result["sites"]))

    def test_mp_offset_discovery_moves_from_marker_center(self):
        result = self._run_case("offset_mp")

        selected = [row for row in result["sites"] if row["source_marker_id"] == "unit__marker_000000"][0]
        self.assertGreater(float(selected["mp_source_marker_distance_um"]), 0.2)
        self.assertGreater(float(selected["mp_x_um"]), 0.2)
        self.assertIn(
            selected["mp_candidate_type"],
            {
                "fragment_facing_pair_anchor",
                "critical_spacing_anchor",
                "fragment_line_end_anchor",
                "line_end_anchor",
                "fragment_corner_anchor",
                "corner_or_jog_anchor",
                "density_transition_anchor",
            },
        )
        self.assertGreater(float(selected["mp_hotspot_score"]), 0.0)
        self.assertEqual(selected["mp_verified"], True)

    def test_fragment_facing_pair_anchor_is_generated_from_edge_fragments(self):
        raw = {}
        elements = [
            {"bbox": (-0.08, -0.12, -0.03, 0.12)},
            {"bbox": (0.03, -0.12, 0.08, 0.12)},
        ]

        mp_gen._add_fragment_anchors(raw, elements=elements, center_xy=(0.0, 0.0), radius_um=0.8, step_um=0.2)

        pair_candidates = [item for item in raw.values() if "fragment_facing_pair_anchor" in item["sources"]]
        self.assertTrue(pair_candidates)
        best = min(pair_candidates, key=lambda item: abs(item["x"]) + abs(item["y"]))
        self.assertAlmostEqual(best["x"], 0.0, places=6)
        self.assertAlmostEqual(best["y"], 0.0, places=6)
        self.assertGreater(best["metrics"]["fragment_facing_pair_score"], 0.0)
        self.assertIn("internal_facing_distance_um", best["metrics"])

    def test_fragment_line_end_anchor_is_generated_from_long_wire_end(self):
        raw = {}
        elements = [{"bbox": (-0.30, -0.025, 0.30, 0.025)}]

        mp_gen._add_fragment_anchors(raw, elements=elements, center_xy=(0.0, 0.0), radius_um=0.8, step_um=0.2)

        line_end_candidates = [item for item in raw.values() if "fragment_line_end_anchor" in item["sources"]]
        self.assertEqual(len(line_end_candidates), 2)
        self.assertTrue(all(item["metrics"]["fragment_line_end_score"] == 1.0 for item in line_end_candidates))

    def test_fragment_corner_context_is_recorded(self):
        raw = {}
        elements = [{"bbox": (-0.05, -0.05, 0.05, 0.05)}]

        mp_gen._add_fragment_anchors(raw, elements=elements, center_xy=(0.0, 0.0), radius_um=0.8, step_um=0.2)

        corner_candidates = [item for item in raw.values() if "fragment_corner_anchor" in item["sources"]]
        self.assertEqual(len(corner_candidates), 4)
        self.assertTrue(all(item["metrics"]["fragment_corner_score"] == 1.0 for item in corner_candidates))
        self.assertTrue(all(item["metrics"]["fragment_context_score"] > 0.0 for item in corner_candidates))

    def test_spatial_proximity_voting_boosts_supported_anchor(self):
        bitmap = np.zeros((16, 16), dtype=bool)
        bitmap[:, 7:9] = True
        window = {"clip_bitmap": bitmap, "clip_bbox": [-0.2, -0.2, 0.2, 0.2]}
        supported = mp_gen.MPCandidate(0.0, 0.0, 0.0, "fragment_facing_pair_anchor", ["fragment_facing_pair_anchor"], window)
        neighbor = mp_gen.MPCandidate(0.05, 0.0, 0.05, "fragment_corner_anchor", ["fragment_corner_anchor"], window)
        isolated = mp_gen.MPCandidate(0.5, 0.0, 0.5, "fragment_line_end_anchor", ["fragment_line_end_anchor"], window)

        mp_gen._annotate_spatial_voting([supported, neighbor, isolated], support_radius_um=0.1)

        self.assertGreater(supported.proposal_metrics["spatial_proposal_voting"], isolated.proposal_metrics["spatial_proposal_voting"])
        self.assertGreaterEqual(supported.proposal_metrics["supporting_anchor_count"], 2.0)

    def test_rule_coverage_audit_is_written_to_pool_review(self):
        normal_result = self._run_case("normal")
        with Path(normal_result["outputs"]["mp_candidate_pool_json"]).open("r", encoding="utf-8") as handle:
            normal_pool = json.load(handle)
        self.assertTrue(normal_pool[0]["mp_rule_coverage_audit"]["semantic_marker_covered"])

        fallback_root = self.temp_root / "fallback_rule_audit"
        fallback_root.mkdir(parents=True, exist_ok=True)
        fallback_oas = fallback_root / "unit.oas"
        _make_oas(fallback_oas, "fallback_mp")
        fallback_manifest = _make_manifest(fallback_root)
        fallback_result = selector.run_recipe_selector(_args(fallback_root, fallback_oas, fallback_manifest))
        with Path(fallback_result["outputs"]["care_area_groups_json"]).open("r", encoding="utf-8") as handle:
            fallback_groups = json.load(handle)
        self.assertEqual(fallback_groups["rejected_seed_count"], 1)
        self.assertEqual(fallback_groups["rejected_seeds"][0]["reject_reason"], "no_care_area_family")

    def test_mp_fallback_uses_marker_center_without_geometry_anchor(self):
        result = self._run_case("fallback_mp")

        self.assertEqual(result["summary"]["selected_mp_candidate_count"], 0)
        self.assertEqual(result["summary"]["invalid_mp_candidate_count"], 0)
        self.assertEqual(result["summary"]["rejected_care_area_seed_count"], 1)
        with Path(result["outputs"]["mp_candidate_pool_json"]).open("r", encoding="utf-8") as handle:
            pool = json.load(handle)
        self.assertEqual(pool, [])
        self.assertTrue(any(row["reject_reason"] == "no_care_area_family" for row in result["sites"]))

    def test_mp_nms_records_suppressed_candidates(self):
        bitmap = np.zeros((16, 16), dtype=bool)
        bitmap[:, 7:9] = True
        window = {"clip_bitmap": bitmap, "clip_bbox": [-0.2, -0.2, 0.2, 0.2]}
        first = mp_gen.MPCandidate(0.00, 0.00, 0.00, "critical_spacing_anchor", ["critical_spacing_anchor"], window, score=0.9)
        second = mp_gen.MPCandidate(0.05, 0.00, 0.05, "critical_spacing_anchor", ["critical_spacing_anchor"], window, score=0.8)
        far = mp_gen.MPCandidate(0.30, 0.00, 0.30, "critical_spacing_anchor", ["critical_spacing_anchor"], window, score=0.7)

        rejected = mp_gen._apply_nms([first, second, far], nms_radius_um=0.10)

        self.assertEqual(rejected, 1)
        self.assertTrue(first.accepted)
        self.assertFalse(second.accepted)
        self.assertTrue(far.accepted)

    def test_behavior_zero_redistributes_mp_discovery_weight(self):
        result = self._run_case("normal", risk_score=0.0)

        row = result["sites"][0]
        components = json.loads(row["mp_discovery_components_json"])
        self.assertEqual(float(components["behavior_weight_redistributed"]), 1.0)
        self.assertIn("proposal_voting", components)
        self.assertIn("local_rarity", components)
        self.assertIn("core_defect_proxy_score", components)
        self.assertIn("context_support_score", components)
        self.assertGreater(float(row["mp_selection_gain"]), 0.0)

    def test_global_candidate_rarity_is_recorded_for_budget_scoring(self):
        bitmap = np.zeros((16, 16), dtype=bool)
        bitmap[:, 7:9] = True
        window = {"center": (0.0, 0.0), "clip_bitmap": bitmap, "clip_bbox": [-0.2, -0.2, 0.2, 0.2]}

        def info(candidate_id: str) -> selector.MPPoolCandidateInfo:
            return selector.MPPoolCandidateInfo(
                cluster_id=0,
                representative_marker_id=candidate_id,
                marker_ids=[candidate_id],
                representative_metadata={"source_path": "unit.oas"},
                cluster={},
                mp_candidate_id=candidate_id,
                mp_candidate_rank=0,
                mp_window=window,
                source_marker_center=(0.0, 0.0),
                mp_candidate_type="critical_spacing_anchor",
                mp_hotspot_score=0.8,
                mp_verified=True,
                mp_reject_reason="",
                mp_verification_components={},
                mp_discovery_components={"local_rarity": 1.0},
                mp_discovery=None,
                raw_components={"mp_hotspot_score": 0.8, "behavior_risk": 1.0, "pattern_rarity": 1.0, "cluster_coverage": 1.0},
            )

        candidates = [info("a"), info("b")]
        selector._score_mp_candidates(candidates)

        self.assertIn("global_candidate_rarity", candidates[0].score_components)
        self.assertLessEqual(candidates[0].score_components["global_candidate_rarity"], 0.05)

    def test_global_candidate_rarity_ignores_invalid_candidates(self):
        bitmap = np.zeros((16, 16), dtype=bool)
        bitmap[:, 7:9] = True
        window = {"center": (0.0, 0.0), "clip_bitmap": bitmap, "clip_bbox": [-0.2, -0.2, 0.2, 0.2]}

        def info(candidate_id: str, verified: bool) -> selector.MPPoolCandidateInfo:
            return selector.MPPoolCandidateInfo(
                cluster_id=0,
                representative_marker_id=candidate_id,
                marker_ids=[candidate_id],
                representative_metadata={"source_path": "unit.oas"},
                cluster={},
                mp_candidate_id=candidate_id,
                mp_candidate_rank=0,
                mp_window=window,
                source_marker_center=(0.0, 0.0),
                mp_candidate_type="critical_spacing_anchor",
                mp_hotspot_score=0.8,
                mp_verified=verified,
                mp_reject_reason="" if verified else "sparse_bitmap",
                mp_verification_components={},
                mp_discovery_components={"local_rarity": 1.0},
                mp_discovery=None,
                raw_components={"mp_hotspot_score": 0.8, "behavior_risk": 1.0, "pattern_rarity": 1.0, "cluster_coverage": 1.0},
            )

        candidates = [info("valid", True), info("invalid", False)]
        selector._score_mp_candidates(candidates)

        self.assertAlmostEqual(candidates[0].score_components["global_candidate_rarity"], 1.0)
        self.assertAlmostEqual(candidates[1].score_components["global_candidate_rarity"], 0.0)
        self.assertAlmostEqual(candidates[1].mp_priority_score, 0.0)

    def test_behavior_risk_zero_flag_can_use_family_level_semantics(self):
        bitmap = np.zeros((16, 16), dtype=bool)
        bitmap[:, 7:9] = True
        window = {"center": (0.0, 0.0), "clip_bitmap": bitmap, "clip_bbox": [-0.2, -0.2, 0.2, 0.2]}
        candidate = selector.MPPoolCandidateInfo(
            cluster_id=0,
            representative_marker_id="unit__marker_000000",
            marker_ids=["unit__marker_000000"],
            representative_metadata={"source_path": "unit.oas"},
            cluster={},
            mp_candidate_id="unit",
            mp_candidate_rank=0,
            mp_window=window,
            source_marker_center=(0.0, 0.0),
            mp_candidate_type="critical_spacing_anchor",
            mp_hotspot_score=0.8,
            mp_verified=True,
            mp_reject_reason="",
            mp_verification_components={},
            mp_discovery_components={"local_rarity": 1.0},
            mp_discovery=None,
            raw_components={
                "mp_hotspot_score": 0.8,
                "behavior_risk": 0.0,
                "pattern_rarity": 1.0,
                "cluster_coverage": 1.0,
            },
        )

        selector._score_mp_candidates([candidate], all_risk_zero=False)

        self.assertAlmostEqual(candidate.score_components["risk_weight_redistributed"], 0.0)

    def test_mp_priority_does_not_double_count_metrology_priority_score(self):
        bitmap = np.zeros((16, 16), dtype=bool)
        bitmap[:, 7:9] = True
        window = {"center": (0.0, 0.0), "clip_bitmap": bitmap, "clip_bbox": [-0.2, -0.2, 0.2, 0.2]}

        def info(candidate_id: str, metrology_score: float, localization: float, waste: float) -> selector.MPPoolCandidateInfo:
            return selector.MPPoolCandidateInfo(
                cluster_id=0,
                representative_marker_id=candidate_id,
                marker_ids=[candidate_id],
                representative_metadata={"source_path": "unit.oas"},
                cluster={},
                mp_candidate_id=candidate_id,
                mp_candidate_rank=0,
                mp_window=window,
                source_marker_center=(0.0, 0.0),
                mp_candidate_type="critical_spacing_anchor",
                mp_hotspot_score=0.8,
                mp_verified=True,
                mp_reject_reason="",
                mp_verification_components={},
                mp_discovery_components={"local_rarity": 1.0},
                mp_discovery=None,
                metrology_priority_score=metrology_score,
                recipe_waste_penalty=waste,
                raw_components={
                    "mp_hotspot_score": 0.8,
                    "behavior_risk": 1.0,
                    "pattern_rarity": 1.0,
                    "cluster_coverage": 1.0,
                    "metrology_priority_score": metrology_score,
                    "mp_localization_confidence": localization,
                    "recipe_waste_penalty": waste,
                },
            )

        same_decision_signal = [info("low_context", 0.0, 1.0, 0.0), info("high_context", 1.0, 1.0, 0.0)]
        selector._score_mp_candidates(same_decision_signal)
        self.assertAlmostEqual(same_decision_signal[0].mp_priority_score, same_decision_signal[1].mp_priority_score)

        different_decision_signal = [info("localized", 0.0, 1.0, 0.0), info("weak_localization", 1.0, 0.0, 0.0)]
        selector._score_mp_candidates(different_decision_signal)
        self.assertGreater(different_decision_signal[0].mp_priority_score, different_decision_signal[1].mp_priority_score)

    def test_high_recipe_waste_soft_demotes_priority(self):
        bitmap = np.zeros((16, 16), dtype=bool)
        bitmap[:, 7:9] = True
        window = {"center": (0.0, 0.0), "clip_bitmap": bitmap, "clip_bbox": [-0.2, -0.2, 0.2, 0.2]}

        def info(candidate_id: str, waste: float, metrology_score: float) -> selector.MPPoolCandidateInfo:
            return selector.MPPoolCandidateInfo(
                cluster_id=0,
                representative_marker_id=candidate_id,
                marker_ids=[candidate_id],
                representative_metadata={"source_path": "unit.oas"},
                cluster={},
                mp_candidate_id=candidate_id,
                mp_candidate_rank=0,
                mp_window=window,
                source_marker_center=(0.0, 0.0),
                mp_candidate_type="critical_spacing_anchor",
                mp_hotspot_score=0.8,
                mp_verified=True,
                mp_reject_reason="",
                mp_verification_components={},
                mp_discovery_components={"local_rarity": 1.0},
                mp_discovery=None,
                metrology_priority_score=metrology_score,
                recipe_waste_penalty=waste,
                raw_components={
                    "mp_hotspot_score": 0.8,
                    "behavior_risk": 1.0,
                    "pattern_rarity": 1.0,
                    "cluster_coverage": 1.0,
                    "mp_localization_confidence": 1.0,
                    "metrology_priority_score": metrology_score,
                    "recipe_waste_penalty": waste,
                },
            )

        candidates = [info("ok", 0.2, 0.6), info("waste", 0.9, 0.6)]
        selector._score_mp_candidates(candidates)

        self.assertEqual(candidates[1].pool_status, "candidate")
        self.assertEqual(candidates[1].score_components["recipe_waste_soft_demoted"], 1.0)
        self.assertAlmostEqual(candidates[1].score_components["recipe_waste_demotion_factor"], 0.8)
        self.assertLess(candidates[1].mp_priority_score, candidates[0].mp_priority_score)

        moderate = [info("base", 0.2, 0.6), info("moderate_waste", 0.55, 0.6)]
        selector._score_mp_candidates(moderate)
        self.assertEqual(moderate[1].score_components["recipe_waste_soft_demoted"], 1.0)
        self.assertLess(moderate[1].score_components["recipe_waste_demotion_factor"], 1.0)

        high_priority = [info("base", 0.2, 0.9), info("high_priority_waste", 0.9, 0.9)]
        selector._score_mp_candidates(high_priority)
        self.assertLess(high_priority[1].score_components["recipe_waste_demotion_factor"], 1.0)
        self.assertGreater(high_priority[1].score_components["recipe_waste_demotion_factor"], candidates[1].score_components["recipe_waste_demotion_factor"])

    def test_priority_normalization_ignores_invalid_candidates(self):
        bitmap = np.zeros((16, 16), dtype=bool)
        bitmap[:, 7:9] = True
        window = {"center": (0.0, 0.0), "clip_bitmap": bitmap, "clip_bbox": [-0.2, -0.2, 0.2, 0.2]}

        def info(candidate_id: str, verified: bool, hotspot_score: float) -> selector.MPPoolCandidateInfo:
            return selector.MPPoolCandidateInfo(
                cluster_id=0,
                representative_marker_id=candidate_id,
                marker_ids=[candidate_id],
                representative_metadata={"source_path": "unit.oas"},
                cluster={},
                mp_candidate_id=candidate_id,
                mp_candidate_rank=0,
                mp_window=window,
                source_marker_center=(0.0, 0.0),
                mp_candidate_type="critical_spacing_anchor",
                mp_hotspot_score=hotspot_score,
                mp_verified=verified,
                mp_reject_reason="" if verified else "sparse_bitmap",
                mp_verification_components={},
                mp_discovery_components={"local_rarity": 1.0},
                mp_discovery=None,
                raw_components={"mp_hotspot_score": hotspot_score, "behavior_risk": 1.0, "pattern_rarity": 1.0, "cluster_coverage": 1.0},
            )

        candidates = [info("valid_low", True, 0.2), info("valid_high", True, 0.4), info("invalid_extreme", False, 100.0)]
        selector._score_mp_candidates(candidates)

        self.assertAlmostEqual(candidates[0].score_components["mp_hotspot_score"], 0.0)
        self.assertAlmostEqual(candidates[1].score_components["mp_hotspot_score"], 1.0)
        self.assertAlmostEqual(candidates[2].score_components["mp_hotspot_score"], 0.0)

    def test_mp_verification_rejects_sparse_bitmap(self):
        bitmap = np.zeros((16, 16), dtype=bool)
        bitmap[8, 8] = True
        candidate = mp_gen.MPCandidate(
            0.0,
            0.0,
            0.0,
            "critical_spacing_anchor",
            ["critical_spacing_anchor"],
            {"clip_bitmap": bitmap, "clip_bbox": [-0.2, -0.2, 0.2, 0.2]},
            components={"geometry_core_score": 0.9},
        )

        mp_gen._verify_mp_candidate(candidate, search_radius_um=0.8)

        self.assertFalse(candidate.verified)
        self.assertEqual(candidate.verification_reason, "sparse_bitmap")

    def test_density_transition_anchor_is_not_polygon_center_alias(self):
        raw = {}
        mp_gen._add_geometry_anchors(
            raw,
            elements=[{"bbox": (-0.05, -0.05, 0.05, 0.05)}],
            center_xy=(0.0, 0.0),
            radius_um=0.8,
            step_um=0.2,
        )

        sources = set()
        for item in raw.values():
            sources.update(item["sources"])
        self.assertNotIn("density_transition_anchor", sources)
        self.assertIn("corner_or_jog_anchor", sources)

    def test_recipe_level_ap_global_duplicate_rejects_lower_score_site(self):
        row_a = {
            "site_id": "site_0000",
            "recipe_status": "selected",
            "reject_reason": "",
            "ap_score": 0.9,
            "_details": {"ap_candidate": {"accepted": True, "fingerprint": [1.0, 0.0]}},
        }
        row_b = {
            "site_id": "site_0001",
            "recipe_status": "selected",
            "reject_reason": "",
            "ap_score": 0.8,
            "_details": {"ap_candidate": {"accepted": True, "fingerprint": [1.0, 0.0]}},
        }

        count = selector._apply_global_ap_uniqueness([row_a, row_b], duplicate_threshold=0.92)

        self.assertEqual(count, 1)
        self.assertEqual(row_b["recipe_status"], "rejected")
        self.assertEqual(row_b["ap_global_duplicate_with"], "site_0000")
        self.assertIn("ap_global_duplicate", row_b["reject_reason"])
        self.assertEqual(row_b["_details"]["site"]["recipe_status"], "rejected")
        selector._write_site_summaries([row_a, row_b], self.temp_root)
        with (self.temp_root / "site_0001" / "site_summary.json").open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        self.assertEqual(summary["site"]["recipe_status"], "rejected")
        self.assertTrue(summary["site"]["ap_global_duplicate"])

    def test_min_feature_um_expands_critical_spacing_gap(self):
        raw_without_hint = {}
        raw_with_hint = {}
        elements = [
            {"bbox": (-0.20, -0.10, -0.10, 0.10)},
            {"bbox": (0.10, -0.10, 0.20, 0.10)},
        ]

        mp_gen._add_pair_anchors(raw_without_hint, elements=elements, center_xy=(0.0, 0.0), radius_um=0.8, step_um=0.02)
        mp_gen._add_pair_anchors(
            raw_with_hint,
            elements=elements,
            center_xy=(0.0, 0.0),
            radius_um=0.8,
            step_um=0.02,
            min_feature_um=0.20,
        )

        self.assertFalse(any("critical_spacing_anchor" in item["sources"] for item in raw_without_hint.values()))
        self.assertTrue(any("critical_spacing_anchor" in item["sources"] for item in raw_with_hint.values()))

    def test_af_image_shift_limit_filters_search_candidates(self):
        bitmap = np.ones((4, 4), dtype=bool)

        class Cache:
            def window(self, source_path, center_xy, clip_size_um):
                return {"center": center_xy, "clip_bitmap": bitmap, "clip_bbox": [-0.1, -0.1, 0.1, 0.1]}

        candidates = selector._build_search_candidates(
            Cache(),
            source_path="unit.oas",
            center_xy=(0.0, 0.0),
            clip_size_um=0.2,
            radius_um=1.0,
            step_um=0.25,
            min_distance_um=0.0,
            max_distance_um=0.30,
        )

        self.assertTrue(candidates)
        self.assertLessEqual(max(candidate.distance_um for candidate in candidates), 0.30 + 1e-9)

    def test_ap_local_template_peak_proxy_is_recorded(self):
        bitmap_a = np.zeros((16, 16), dtype=bool)
        bitmap_b = np.zeros((16, 16), dtype=bool)
        bitmap_a[3:7, 3:7] = True
        bitmap_b[9:13, 9:13] = True
        candidates = [
            selector.WindowCandidate(0.0, 0.0, 0.4, {"clip_bitmap": bitmap_a, "clip_bbox": [-0.2, -0.2, 0.2, 0.2]}),
            selector.WindowCandidate(0.6, 0.0, 0.6, {"clip_bitmap": bitmap_b, "clip_bbox": [0.4, -0.2, 0.8, 0.2]}),
        ]

        selector._select_ap_candidate(candidates, radius_um=1.0, ignore_radius_um=0.1)

        self.assertIn("template_peak_margin", candidates[0].components)
        self.assertIn("template_peak_ratio", candidates[0].components)
        self.assertLessEqual(candidates[0].components["template_peak_ratio"], 100.0)

    def test_ap_richness_gate_rejects_uniform_candidate(self):
        bitmap = np.ones((16, 16), dtype=bool)
        candidate = selector.WindowCandidate(0.0, 0.0, 0.4, {"clip_bitmap": bitmap, "clip_bbox": [-0.2, -0.2, 0.2, 0.2]})

        selected = selector._select_ap_candidate([candidate], radius_um=1.0, ignore_radius_um=0.1)

        self.assertIs(selected, candidate)
        self.assertFalse(candidate.accepted)
        self.assertEqual(candidate.components["pattern_density"], 1.0)
        self.assertEqual(candidate.reject_reason, "density_out_of_range")
        self.assertFalse(candidate.acceptance_checks["density_ok"])

    def test_af_reject_reason_is_recorded(self):
        mp_bitmap = np.zeros((16, 16), dtype=bool)
        mp_bitmap[:, 7:9] = True
        candidate_bitmap = np.zeros((16, 16), dtype=bool)
        candidate_bitmap[7:9, :] = True
        candidate = selector.WindowCandidate(0.4, 0.0, 0.4, {"clip_bitmap": candidate_bitmap, "clip_bbox": [0.2, -0.2, 0.6, 0.2]})

        selected = selector._select_af_candidate([candidate], mp_bitmap=mp_bitmap, radius_um=1.0)

        self.assertIs(selected, candidate)
        self.assertFalse(candidate.accepted)
        self.assertIn(candidate.reject_reason, {"low_similarity", "low_af_score"})
        self.assertIn("score_ok", candidate.acceptance_checks)

    def test_af_accepts_high_structure_focus_candidate(self):
        bitmap = np.zeros((16, 16), dtype=bool)
        bitmap[3:13, 7:9] = True
        candidate = selector.WindowCandidate(0.4, 0.0, 0.4, {"clip_bitmap": bitmap, "clip_bbox": [0.2, -0.2, 0.6, 0.2]})

        selected = selector._select_af_candidate([candidate], mp_bitmap=bitmap, radius_um=1.0)

        self.assertIs(selected, candidate)
        self.assertTrue(candidate.accepted)
        self.assertIn("hotspot_core_risk", candidate.components)
        self.assertLess(candidate.components["hotspot_core_risk"], 0.85)

    def test_af_marks_hotspot_core_like_candidate(self):
        bitmap = np.zeros((16, 16), dtype=bool)
        bitmap[3:13, 7:9] = True
        bitmap[7:9, 3:13] = True
        candidate = selector.WindowCandidate(0.4, 0.0, 0.4, {"clip_bitmap": bitmap, "clip_bbox": [0.2, -0.2, 0.6, 0.2]})

        selected = selector._select_af_candidate([candidate], mp_bitmap=bitmap, radius_um=1.0)

        self.assertIs(selected, candidate)
        self.assertFalse(candidate.accepted)
        self.assertGreaterEqual(candidate.components["hotspot_core_risk"], 0.85)
        self.assertEqual(candidate.reject_reason, "too_hotspot_like")
        self.assertFalse(candidate.acceptance_checks["hotspot_core_safe"])
        self.assertLess(candidate.score, 0.94)

    def test_template_size_overrides_are_recorded(self):
        result = self._run_case(
            "normal",
            arg_overrides={"mp_template_size_um": 0.5, "af_template_size_um": 0.6, "ap_template_size_um": 0.7},
        )
        row = [item for item in result["sites"] if str(item["site_id"]).startswith("site_")][0]

        self.assertAlmostEqual(float(row["mp_template_size_um"]), 0.5)
        self.assertAlmostEqual(float(row["af_template_size_um"]), 0.6)
        self.assertAlmostEqual(float(row["ap_template_size_um"]), 0.7)
        if row["af_similarity"] != "":
            self.assertGreater(float(row["af_similarity"]), 0.0)

    def test_min_feature_um_expands_critical_spacing_gap(self):
        raw_without_hint = {}
        raw_with_hint = {}
        elements = [
            {"bbox": (-0.20, -0.10, -0.10, 0.10)},
            {"bbox": (0.10, -0.10, 0.20, 0.10)},
        ]

        mp_gen._add_pair_anchors(raw_without_hint, elements=elements, center_xy=(0.0, 0.0), radius_um=0.8, step_um=0.02)
        mp_gen._add_pair_anchors(
            raw_with_hint,
            elements=elements,
            center_xy=(0.0, 0.0),
            radius_um=0.8,
            step_um=0.02,
            min_feature_um=0.20,
        )

        self.assertFalse(any("critical_spacing_anchor" in item["sources"] for item in raw_without_hint.values()))
        self.assertTrue(any("critical_spacing_anchor" in item["sources"] for item in raw_with_hint.values()))

    def test_af_image_shift_limit_filters_search_candidates(self):
        bitmap = np.ones((4, 4), dtype=bool)

        class Cache:
            def window(self, source_path, center_xy, clip_size_um):
                return {"center": center_xy, "clip_bitmap": bitmap, "clip_bbox": [-0.1, -0.1, 0.1, 0.1]}

        candidates = selector._build_search_candidates(
            Cache(),
            source_path="unit.oas",
            center_xy=(0.0, 0.0),
            clip_size_um=0.2,
            radius_um=1.0,
            step_um=0.25,
            min_distance_um=0.0,
            max_distance_um=0.30,
        )

        self.assertTrue(candidates)
        self.assertLessEqual(max(candidate.distance_um for candidate in candidates), 0.30 + 1e-9)

    def test_ap_local_template_peak_proxy_is_recorded(self):
        bitmap_a = np.zeros((16, 16), dtype=bool)
        bitmap_b = np.zeros((16, 16), dtype=bool)
        bitmap_a[3:7, 3:7] = True
        bitmap_b[9:13, 9:13] = True
        candidates = [
            selector.WindowCandidate(0.0, 0.0, 0.4, {"clip_bitmap": bitmap_a, "clip_bbox": [-0.2, -0.2, 0.2, 0.2]}),
            selector.WindowCandidate(0.6, 0.0, 0.6, {"clip_bitmap": bitmap_b, "clip_bbox": [0.4, -0.2, 0.8, 0.2]}),
        ]

        selector._select_ap_candidate(candidates, radius_um=1.0, ignore_radius_um=0.1)

        self.assertIn("template_peak_margin", candidates[0].components)
        self.assertIn("template_peak_ratio", candidates[0].components)

    def test_template_size_overrides_are_recorded(self):
        result = self._run_case(
            "normal",
            arg_overrides={"mp_template_size_um": 0.5, "af_template_size_um": 0.6, "ap_template_size_um": 0.7},
        )
        row = [item for item in result["sites"] if str(item["site_id"]).startswith("site_")][0]

        self.assertAlmostEqual(float(row["mp_template_size_um"]), 0.5)
        self.assertAlmostEqual(float(row["af_template_size_um"]), 0.6)
        self.assertAlmostEqual(float(row["ap_template_size_um"]), 0.7)

    def test_af_non_overlap_and_ap_unique_are_selected(self):
        result = self._run_case("normal")

        selected = [row for row in result["sites"] if row["source_marker_id"] == "unit__marker_000000"][0]
        self.assertEqual(selected["recipe_status"], "selected")
        self.assertGreaterEqual(float(selected["af_distance_um"]), 0.3)
        self.assertLessEqual(int(selected["ap_peak_count"]), 3)
        self.assertTrue(Path(selected["mp_oas"]).exists())
        self.assertTrue(Path(selected["af_oas"]).exists())
        self.assertTrue(Path(selected["ap_oas"]).exists())

    def test_af_reject_when_no_safe_focus_pattern_exists(self):
        result = self._run_case("no_af")

        row = result["sites"][0]
        self.assertEqual(row["recipe_status"], "rejected")
        self.assertIn("no_safe_af", row["reject_reason"])
        self.assertTrue(row["af_reject_reason"])

    def test_ap_periodic_pattern_is_rejected(self):
        result = self._run_case("periodic_ap")

        row = result["sites"][0]
        self.assertEqual(row["recipe_status"], "rejected")
        self.assertIn("no_unique_ap", row["reject_reason"])
        self.assertTrue(row["ap_reject_reason"])

    @unittest.skipUnless(os.environ.get("RUN_SLOW_SMOKE") == "1", "设置 RUN_SLOW_SMOKE=1 后才运行真实 clip_for_lyu smoke")
    def test_real_clip_for_lyu_smoke_outputs_compact_review(self):
        oas_path = SCRIPT_DIR / "clip_for_lyu.oas"
        manifest = SCRIPT_DIR / "semsim_v11_smoke_clip_for_lyu" / "behavior.jsonl"
        if not oas_path.exists() or not manifest.exists():
            self.skipTest("缺少 clip_for_lyu.oas 或对应 behavior manifest")

        args = argparse.Namespace(
            input_path=str(oas_path),
            marker_layer="12530/2",
            behavior_manifest=str(manifest),
            output_dir=str(self.temp_root / "clip_for_lyu_recipe_selector_optimized"),
            clip_size=1.35,
            mp_template_size_um=None,
            af_template_size_um=None,
            ap_template_size_um=None,
            max_sites=20,
            mp_coverage_target=0.985,
            mp_search_radius_um=0.8,
            mp_candidates_per_marker=5,
            max_care_area_instances_per_family=80,
            min_feature_um=None,
            af_search_radius_um=3.0,
            sem_image_shift_limit_um=None,
            ap_search_radius_um=5.0,
            candidate_step_um=0.2,
            min_site_distance_um=0.5,
            recursive_input=False,
            apply_layer_ops=False,
            register_op=None,
            skip_pattern_memory_store_append=True,
        )

        result = selector.run_recipe_selector(args)

        self.assertGreater(result["summary"]["selected_recipe_site_count"], 0)
        self.assertTrue(Path(result["outputs"]["recipe_sites_csv"]).exists())
        self.assertTrue(Path(result["outputs"]["recipe_sites_json"]).exists())
        self.assertTrue(Path(result["outputs"]["mp_candidate_pool_json"]).exists())
        self.assertTrue(Path(result["outputs"]["subset_objective_audit_json"]).exists())
        self.assertTrue(Path(result["outputs"]["source_marker_candidate_index_json"]).exists())
        self.assertTrue(result["summary"]["pattern_memory_store_append_skipped"])
        self.assertLess(Path(result["outputs"]["recipe_sites_json"]).stat().st_size, 15 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
