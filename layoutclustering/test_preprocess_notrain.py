#!/usr/bin/env python3
"""Tests for preprocess_notrain.py."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import unittest
from pathlib import Path

import gdstk
import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import layout_clustering_optimized_notrain as notrain
import preprocess_notrain
from layout_utils import _write_oas_library


def _make_oas(path: Path, marker_count: int = 3) -> None:
    """生成带 marker layer 的小型 OAS，用于预处理测试。"""
    lib = gdstk.Library()
    cell = gdstk.Cell("TOP")
    for idx in range(marker_count):
        cx = float(idx) * 3.0
        cy = 0.0
        cell.add(gdstk.rectangle((cx - 0.02, cy - 0.02), (cx + 0.02, cy + 0.02), layer=999, datatype=0))
        cell.add(gdstk.rectangle((cx - 0.20, cy - 0.06), (cx + 0.20, cy + 0.06), layer=1, datatype=0))
    lib.add(cell)
    _write_oas_library(lib, str(path))


def _read_jsonl(path: Path):
    """读取 JSONL 测试文件。"""
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class PreprocessNotrainTests(unittest.TestCase):
    """验证 no-train 预处理脚本的 marker 对齐和图像转换行为。"""

    def setUp(self):
        self.temp_root = SCRIPT_DIR / "test_outputs" / "_preprocess_notrain"
        shutil.rmtree(self.temp_root, ignore_errors=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_preprocess_skips_missing_and_takes_first_duplicate(self):
        oas_path = self.temp_root / "unit.oas"
        aerial_dir = self.temp_root / "aerial"
        output_dir = self.temp_root / "notrain_inputs"
        aerial_dir.mkdir()
        _make_oas(oas_path, marker_count=3)

        rgb = np.asarray(
            [
                [[0, 0, 0], [255, 0, 0]],
                [[0, 255, 0], [0, 0, 255]],
            ],
            dtype=np.uint8,
        )
        Image.fromarray(rgb, mode="RGB").save(aerial_dir / "unit__marker_000000.png")
        first = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        second = np.asarray([[9.0, 9.0], [9.0, 9.0]], dtype=np.float32)
        np.save(aerial_dir / "marker_000001_a.npy", first)
        np.save(aerial_dir / "marker_000001_b.npy", second)

        summary = preprocess_notrain.preprocess_notrain(
            input_path=str(oas_path),
            marker_layer="999/0",
            aerial_dir=str(aerial_dir),
            output_dir=str(output_dir),
            clip_size_um=1.0,
            normalize=False,
        )

        self.assertEqual(summary["marker_count"], 3)
        self.assertEqual(summary["matched_marker_count"], 2)
        self.assertEqual(summary["skipped_missing_aerial_count"], 1)
        self.assertEqual(summary["duplicate_aerial_marker_count"], 1)
        self.assertEqual(summary["written_npz_count"], 2)
        self.assertIn("unit__marker_000002", summary["missing_marker_preview"])

        rows = _read_jsonl(output_dir / "behavior.jsonl")
        self.assertEqual([row["marker_id"] for row in rows], ["unit__marker_000000", "unit__marker_000001"])
        self.assertEqual(rows[0]["aerial_npz"], "aerial_npz/unit__marker_000000.npz")
        with np.load(output_dir / rows[0]["aerial_npz"], allow_pickle=False) as data:
            rgb_image = np.asarray(data["image"])
        self.assertEqual(rgb_image.ndim, 2)
        self.assertEqual(rgb_image.dtype, np.float32)
        with np.load(output_dir / rows[1]["aerial_npz"], allow_pickle=False) as data:
            chosen = np.asarray(data["image"])
        self.assertTrue(np.allclose(chosen, first))

    def test_npz_input_and_default_normalization(self):
        oas_path = self.temp_root / "unit.oas"
        aerial_dir = self.temp_root / "aerial_npz_src"
        output_dir = self.temp_root / "notrain_inputs_npz"
        aerial_dir.mkdir()
        _make_oas(oas_path, marker_count=1)
        np.savez_compressed(aerial_dir / "000000.npz", image=np.asarray([[2.0, 4.0], [6.0, 10.0]], dtype=np.float32))

        summary = preprocess_notrain.preprocess_notrain(
            input_path=str(oas_path),
            marker_layer="999/0",
            aerial_dir=str(aerial_dir),
            output_dir=str(output_dir),
            clip_size_um=1.0,
        )

        self.assertEqual(summary["written_npz_count"], 1)
        row = _read_jsonl(output_dir / "behavior.jsonl")[0]
        with np.load(output_dir / row["aerial_npz"], allow_pickle=False) as data:
            image = np.asarray(data["image"])
        self.assertAlmostEqual(float(np.min(image)), 0.0)
        self.assertAlmostEqual(float(np.max(image)), 1.0)

    def test_notrain_accepts_preprocess_output_directory(self):
        oas_path = self.temp_root / "unit.oas"
        aerial_dir = self.temp_root / "aerial_for_notrain"
        output_dir = self.temp_root / "notrain_inputs_for_run"
        aerial_dir.mkdir()
        _make_oas(oas_path, marker_count=3)
        np.save(aerial_dir / "marker_000000.npy", np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))
        np.save(aerial_dir / "marker_000001.npy", np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))

        preprocess_notrain.preprocess_notrain(
            input_path=str(oas_path),
            marker_layer="999/0",
            aerial_dir=str(aerial_dir),
            output_dir=str(output_dir),
            clip_size_um=1.0,
            normalize=False,
        )
        args = argparse.Namespace(
            input_path=str(oas_path),
            output=str(self.temp_root / "notrain_result.json"),
            format="json",
            marker_layer="999/0",
            clip_size=1.0,
            behavior_manifest=str(output_dir),
            ann_top_k=2,
            coverage_target=0.95,
            facility_min_gain=1e-6,
            behavior_verification_threshold=0.08,
            high_risk_quantile=0.90,
            export_diff_channels="",
            review_dir=None,
            export_cluster_review_dir=None,
            apply_layer_ops=False,
            register_op=None,
        )

        result = notrain.run_notrain(args)

        self.assertEqual(result["input_marker_count"], 3)
        self.assertEqual(result["marker_count"], 2)
        self.assertEqual(result["skipped_missing_behavior_count"], 1)
        self.assertEqual(result["pipeline_mode"], notrain.PIPELINE_MODE)


if __name__ == "__main__":
    unittest.main()
