#!/usr/bin/env python3
"""handcrafted FV 与 no-train backend 的恢复回归测试。"""

from __future__ import annotations

import argparse
import json
import shutil
import unittest
import uuid
from pathlib import Path

import gdstk
import numpy as np

import feature_extractor_handcraft as features
import hotspot_recipe_notrain_backend as backend


SCRIPT_DIR = Path(__file__).resolve().parent


def _write_oas(path: Path) -> None:
    """写出一个包含单个 marker 和简单 line-pair pattern 的 OAS。"""
    lib = gdstk.Library()
    cell = gdstk.Cell("TOP")
    cell.add(gdstk.rectangle((-0.08, -0.20), (-0.03, 0.20), layer=1, datatype=0))
    cell.add(gdstk.rectangle((0.03, -0.20), (0.08, 0.20), layer=1, datatype=0))
    cell.add(gdstk.rectangle((-0.20, -0.20), (0.20, 0.20), layer=999, datatype=0))
    lib.add(cell)
    lib.write_oas(str(path))


class HandcraftFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = SCRIPT_DIR / "test_outputs" / f"_handcraft_{uuid.uuid4().hex[:8]}"
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_aerial_features_change_with_image(self) -> None:
        """不同 CD-SEM/aerial 图像应产生不同 feature。"""
        layout = np.zeros((32, 32), dtype=bool)
        layout[:, 14:18] = True
        image_a = np.zeros((32, 32), dtype=np.float32)
        image_b = np.eye(32, dtype=np.float32)
        vec_a = features.extract_handcrafted_feature(layout, image_a)
        vec_b = features.extract_handcrafted_feature(layout, image_b)
        self.assertEqual(vec_a.shape, vec_b.shape)
        self.assertGreater(float(np.linalg.norm(vec_a - vec_b)), 1e-3)

    def test_manifest_rejects_optional_behavior_channels(self) -> None:
        """当前主线不再接受 EPE/PV/NILS/resist 旧分支。"""
        with self.assertRaises(ValueError):
            features.validate_behavior_row({"aerial_npz": "a.npz", "epe_npz": "legacy.npz"})

    def test_notrain_pipeline_runs_without_feature_npz_argument(self) -> None:
        """no-train backend 可直接从 OAS + behavior manifest 生成 clusters。"""
        oas = self.root / "unit.oas"
        _write_oas(oas)
        aerial = self.root / "unit__marker_000000_aerial.npz"
        np.savez_compressed(aerial, image=np.ones((32, 32), dtype=np.float32))
        manifest = self.root / "behavior.jsonl"
        manifest.write_text(
            json.dumps(
                {
                    "sample_id": "unit__marker_000000",
                    "source_path": str(oas),
                    "marker_id": "unit__marker_000000",
                    "aerial_npz": str(aerial),
                    "risk_score": 1.0,
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        args = argparse.Namespace(
            input_path=str(oas),
            marker_layer="999/0",
            clip_size=0.4,
            behavior_manifest=str(manifest),
            output_dir=str(self.root / "backend_out"),
            recursive_input=False,
            apply_layer_ops=False,
            register_op=None,
        )

        result = backend.run_notrain_mp_selection(args)

        self.assertEqual(result["result_summary"]["selected_cluster_count"], 1)
        self.assertEqual(result["clusters"][0]["marker_id"], "unit__marker_000000")
        self.assertGreater(result["result_summary"]["feature_dimension"], 0)


if __name__ == "__main__":
    unittest.main()
