#!/usr/bin/env python3
"""behavior manifest 预处理脚本的恢复回归测试。"""

from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

import gdstk
import numpy as np

import hotspot_recipe_notrain_backend as backend
from preprocess_behavior_inputs import preprocess_behavior_inputs


SCRIPT_DIR = Path(__file__).resolve().parent


def _write_oas(path: Path) -> None:
    """写出一个包含单个 marker 的最小 OAS。"""
    lib = gdstk.Library()
    cell = gdstk.Cell("TOP")
    cell.add(gdstk.rectangle((-0.08, -0.20), (-0.03, 0.20), layer=1, datatype=0))
    cell.add(gdstk.rectangle((0.03, -0.20), (0.08, 0.20), layer=1, datatype=0))
    cell.add(gdstk.rectangle((-0.20, -0.20), (0.20, 0.20), layer=999, datatype=0))
    lib.add(cell)
    lib.write_oas(str(path))


class PreprocessBehaviorInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = SCRIPT_DIR / "test_outputs" / f"_preprocess_{uuid.uuid4().hex[:8]}"
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_npz_input_and_default_normalization(self) -> None:
        """NPZ 图像输入会被转换为 manifest 中的 aerial_npz 记录。"""
        oas = self.root / "unit.oas"
        _write_oas(oas)
        image_dir = self.root / "images"
        image_dir.mkdir()
        np.savez_compressed(image_dir / "unit__marker_000000.npz", image=np.arange(16, dtype=np.float32).reshape(4, 4))

        result = preprocess_behavior_inputs(
            oas,
            marker_layer="999/0",
            image_dir=image_dir,
            output_dir=self.root / "behavior_inputs",
            clip_size_um=0.4,
            default_risk_score=0.25,
        )

        manifest = Path(result["behavior_manifest"])
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(result["matched_count"], 1)
        self.assertEqual(rows[0]["marker_id"], "unit__marker_000000")
        self.assertIn("aerial_npz", rows[0])
        self.assertAlmostEqual(float(rows[0]["risk_score"]), 0.25)
        loaded = np.load(rows[0]["aerial_npz"])["image"]
        self.assertGreaterEqual(float(np.min(loaded)), 0.0)
        self.assertLessEqual(float(np.max(loaded)), 1.0)

    def test_notrain_accepts_preprocess_output_directory(self) -> None:
        """backend 可以直接接受 preprocess 输出目录作为 behavior manifest。"""
        oas = self.root / "unit.oas"
        _write_oas(oas)
        image_dir = self.root / "images"
        image_dir.mkdir()
        np.save(image_dir / "unit__marker_000000.npy", np.ones((8, 8), dtype=np.float32))
        output_dir = self.root / "behavior_inputs"
        preprocess_behavior_inputs(
            oas,
            marker_layer="999/0",
            image_dir=image_dir,
            output_dir=output_dir,
            clip_size_um=0.4,
        )
        args = type(
            "Args",
            (),
            {
                "input_path": str(oas),
                "marker_layer": "999/0",
                "clip_size": 0.4,
                "behavior_manifest": str(output_dir),
                "output_dir": str(self.root / "backend_out"),
                "recursive_input": False,
                "apply_layer_ops": False,
                "register_op": None,
            },
        )()

        result = backend.run_notrain_mp_selection(args)

        self.assertEqual(result["result_summary"]["selected_cluster_count"], 1)


if __name__ == "__main__":
    unittest.main()
