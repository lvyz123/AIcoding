#!/usr/bin/env python3
"""semsimulator 第一版回归测试。"""

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
REPO_ROOT = SCRIPT_DIR.parent
HOTSPOT_DIR = REPO_ROOT / "hotspotdetection"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HOTSPOT_DIR) not in sys.path:
    sys.path.insert(0, str(HOTSPOT_DIR))

from semsimulator import sem_simulator  # noqa: E402
import hotspot_recipe_notrain_backend as mp_backend  # noqa: E402
from layout_utils import _write_oas_library  # noqa: E402


def _add_marker(cell: gdstk.Cell, cx: float, cy: float) -> None:
    """添加测试用 marker 小方块。"""
    cell.add(gdstk.rectangle((cx - 0.02, cy - 0.02), (cx + 0.02, cy + 0.02), layer=999, datatype=0))


def _add_signal_pattern(cell: gdstk.Cell, cx: float, cy: float, *, variant: int) -> None:
    """添加参与仿真的主 pattern layer 图形。"""
    if int(variant) % 2 == 0:
        cell.add(gdstk.rectangle((cx - 0.18, cy - 0.04), (cx + 0.18, cy + 0.04), layer=1, datatype=0))
        cell.add(gdstk.rectangle((cx - 0.03, cy - 0.20), (cx + 0.03, cy + 0.20), layer=1, datatype=0))
    else:
        cell.add(gdstk.rectangle((cx - 0.20, cy - 0.16), (cx - 0.12, cy + 0.16), layer=1, datatype=0))
        cell.add(gdstk.rectangle((cx + 0.12, cy - 0.16), (cx + 0.20, cy + 0.16), layer=1, datatype=0))


def _add_decoy_pattern(cell: gdstk.Cell, cx: float, cy: float) -> None:
    """添加不应参与指定 pattern layer 仿真的干扰图形。"""
    cell.add(gdstk.rectangle((cx - 0.36, cy - 0.36), (cx + 0.36, cy + 0.36), layer=2, datatype=0))


def _make_oas(
    path: Path,
    *,
    marker_count: int = 2,
    include_decoy: bool = True,
    signal_indices: set[int] | None = None,
) -> None:
    """生成带 marker、主 pattern 和可选 decoy layer 的小型 OAS。"""
    lib = gdstk.Library()
    cell = gdstk.Cell("TOP")
    active_signal_indices = set(range(int(marker_count))) if signal_indices is None else set(signal_indices)
    for idx in range(int(marker_count)):
        cx = float(idx) * 2.0
        cy = 0.0
        _add_marker(cell, cx, cy)
        if idx in active_signal_indices:
            _add_signal_pattern(cell, cx, cy, variant=idx)
        if include_decoy:
            _add_decoy_pattern(cell, cx, cy)
    lib.add(cell)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_oas_library(lib, str(path))


def _make_layer_ops_oas(path: Path) -> None:
    """生成 layer operation 单测用 OAS。"""
    lib = gdstk.Library()
    cell = gdstk.Cell("TOP")
    _add_marker(cell, 0.0, 0.0)
    cell.add(gdstk.rectangle((-0.22, -0.08), (0.22, 0.08), layer=1, datatype=0))
    cell.add(gdstk.rectangle((-0.04, -0.10), (0.04, 0.10), layer=2, datatype=0))
    lib.add(cell)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_oas_library(lib, str(path))


def _read_jsonl(path: Path):
    """读取 JSONL 测试文件。"""
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_first_image(output_dir: Path) -> np.ndarray:
    """读取输出目录中第一行 manifest 对应的仿真图像。"""
    row = _read_jsonl(output_dir / "behavior.jsonl")[0]
    with np.load(output_dir / row["aerial_npz"], allow_pickle=False) as data:
        return np.asarray(data["image"], dtype=np.float32)


def _load_first_npz_array(output_dir: Path, key: str) -> np.ndarray:
    """读取输出目录中第一行 manifest 对应 NPZ 的指定数组。"""
    row = _read_jsonl(output_dir / "behavior.jsonl")[0]
    with np.load(output_dir / row["aerial_npz"], allow_pickle=False) as data:
        return np.asarray(data[key])


class SemSimulatorTests(unittest.TestCase):
    """验证 SEM 仿真器输出结构、可复现性和下游兼容性。"""

    def setUp(self):
        """创建单测临时目录。"""
        self.temp_root = SCRIPT_DIR / "test_outputs" / f"_sem_simulator_{uuid.uuid4().hex[:8]}"
        shutil.rmtree(self.temp_root, ignore_errors=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """清理单测临时目录。"""
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_outputs_npz_png_manifest_and_summary(self):
        """多 marker 运行后应写出 NPZ、PNG、manifest 和 summary。"""
        oas_path = self.temp_root / "unit.oas"
        output_dir = self.temp_root / "sim_behavior"
        _make_oas(oas_path, marker_count=3)

        summary = sem_simulator.run_sem_simulation(
            input_path=str(oas_path),
            marker_layer="999/0",
            pattern_layers=["1/0"],
            output_dir=str(output_dir),
            clip_size_um=1.0,
            pixel_size_nm=20,
            seed=7,
        )

        self.assertEqual(summary["marker_count"], 3)
        self.assertEqual(summary["written_npz_count"], 3)
        self.assertEqual(summary["written_png_count"], 3)
        self.assertTrue((output_dir / "simulation_summary.json").exists())
        self.assertTrue((output_dir / "simulation_quality_audit.json").exists())
        self.assertEqual(summary["pipeline_mode"], sem_simulator.SIMULATION_MODEL)

        rows = _read_jsonl(output_dir / "behavior.jsonl")
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["risk_score"], 0.0)
        self.assertIn("marker_center", rows[0])
        self.assertIn("aerial_png", rows[0])
        self.assertEqual(rows[0]["pixel_size_nm"], 20)
        self.assertEqual(rows[0]["clip_size_um"], 1.0)
        self.assertEqual(rows[0]["simulation_model"], sem_simulator.SIMULATION_MODEL)
        self.assertEqual(rows[0]["model_profile"], "nominal")
        self.assertIn("pattern_pixel_ratio", rows[0])
        self.assertIn("simulation_low_structure", rows[0])

        npz_path = output_dir / rows[0]["aerial_npz"]
        png_path = output_dir / rows[0]["aerial_png"]
        self.assertTrue(npz_path.exists())
        self.assertTrue(png_path.exists())
        self.assertGreater(png_path.stat().st_size, 0)
        with np.load(npz_path, allow_pickle=False) as data:
            image = np.asarray(data["image"])
            self.assertIn("layout_bitmap", data.files)
            self.assertIn("edge_response", data.files)
            self.assertIn("raw_image", data.files)
            self.assertIn("normalization_mode", data.files)
        self.assertEqual(image.shape, (50, 50))
        self.assertEqual(image.dtype, np.float32)
        self.assertGreaterEqual(float(np.min(image)), 0.0)
        self.assertLessEqual(float(np.max(image)), 1.0)

    def test_pattern_layer_filter_ignores_unselected_decoy_layer(self):
        """只指定主 pattern layer 时，decoy layer 不应改变输出图像。"""
        with_decoy = self.temp_root / "with_decoy" / "unit.oas"
        without_decoy = self.temp_root / "without_decoy" / "unit.oas"
        _make_oas(with_decoy, marker_count=1, include_decoy=True)
        _make_oas(without_decoy, marker_count=1, include_decoy=False)

        out_a = self.temp_root / "out_with_decoy"
        out_b = self.temp_root / "out_without_decoy"
        for oas_path, output_dir in ((with_decoy, out_a), (without_decoy, out_b)):
            sem_simulator.run_sem_simulation(
                input_path=str(oas_path),
                marker_layer="999/0",
                pattern_layers=["1/0"],
                output_dir=str(output_dir),
                clip_size_um=1.0,
                pixel_size_nm=20,
                seed=11,
            )

        self.assertTrue(np.allclose(_load_first_image(out_a), _load_first_image(out_b)))

    def test_seed_controls_marker_noise_reproducibly(self):
        """相同 seed 应完全复现，不同 seed 应改变非空图像噪声。"""
        oas_path = self.temp_root / "unit.oas"
        _make_oas(oas_path, marker_count=1)
        out_a = self.temp_root / "seed_a"
        out_b = self.temp_root / "seed_b"
        out_c = self.temp_root / "seed_c"

        for output_dir, seed in ((out_a, 123), (out_b, 123), (out_c, 456)):
            sem_simulator.run_sem_simulation(
                input_path=str(oas_path),
                marker_layer="999/0",
                pattern_layers=["1/0"],
                output_dir=str(output_dir),
                clip_size_um=1.0,
                pixel_size_nm=20,
                seed=seed,
                write_png=False,
            )

        image_a = _load_first_image(out_a)
        image_b = _load_first_image(out_b)
        image_c = _load_first_image(out_c)
        self.assertTrue(np.allclose(image_a, image_b))
        self.assertFalse(np.allclose(image_a, image_c))

    def test_risk_score_csv_overrides_default_score(self):
        """risk_score CSV 应按 marker_id 覆盖默认风险分数。"""
        oas_path = self.temp_root / "unit.oas"
        output_dir = self.temp_root / "risk_behavior"
        risk_csv = self.temp_root / "risk.csv"
        _make_oas(oas_path, marker_count=2)
        risk_csv.write_text("marker_id,risk_score\nunit__marker_000001,0.8\n", encoding="utf-8")

        summary = sem_simulator.run_sem_simulation(
            input_path=str(oas_path),
            marker_layer="999/0",
            pattern_layers=["1/0"],
            output_dir=str(output_dir),
            clip_size_um=1.0,
            pixel_size_nm=20,
            risk_score=0.25,
            risk_score_csv=str(risk_csv),
        )

        rows = _read_jsonl(output_dir / "behavior.jsonl")
        self.assertEqual(summary["risk_score_override_hit_count"], 1)
        self.assertAlmostEqual(float(rows[0]["risk_score"]), 0.25)
        self.assertAlmostEqual(float(rows[1]["risk_score"]), 0.8)

    def test_cd_bias_override_changes_signal_width_response(self):
        """正负 CD bias 应改变固定归一化下的图像响应强度。"""
        oas_path = self.temp_root / "unit.oas"
        _make_oas(oas_path, marker_count=1)
        out_positive = self.temp_root / "cd_positive"
        out_negative = self.temp_root / "cd_negative"
        for output_dir, cd_bias in ((out_positive, 20.0), (out_negative, -20.0)):
            sem_simulator.run_sem_simulation(
                input_path=str(oas_path),
                marker_layer="999/0",
                pattern_layers=["1/0"],
                output_dir=str(output_dir),
                clip_size_um=1.0,
                pixel_size_nm=20,
                model_profile="clean",
                noise_scale=0.0,
                cd_bias_nm=cd_bias,
                normalization_mode="fixed",
                write_png=False,
            )

        self.assertGreater(float(np.mean(_load_first_image(out_positive))), float(np.mean(_load_first_image(out_negative))))

    def test_model_profiles_increase_empty_window_noise(self):
        """clean/nominal/stress profile 的空窗口 raw noise 应逐步增大。"""
        oas_path = self.temp_root / "empty.oas"
        _make_oas(oas_path, marker_count=1, signal_indices=set(), include_decoy=False)
        values = []
        for profile in ("clean", "nominal", "stress"):
            output_dir = self.temp_root / f"profile_{profile}"
            sem_simulator.run_sem_simulation(
                input_path=str(oas_path),
                marker_layer="999/0",
                pattern_layers=["1/0"],
                output_dir=str(output_dir),
                clip_size_um=1.0,
                pixel_size_nm=20,
                model_profile=profile,
                seed=3,
                write_png=False,
            )
            values.append(float(np.std(_load_first_npz_array(output_dir, "raw_image"))))

        self.assertLess(values[0], values[1])
        self.assertLess(values[1], values[2])

    def test_empty_window_is_marked_and_not_contrast_stretched(self):
        """空窗口应被标记为低结构，且不会被 per-image normalize 拉满对比度。"""
        oas_path = self.temp_root / "empty.oas"
        output_dir = self.temp_root / "empty_behavior"
        _make_oas(oas_path, marker_count=1, signal_indices=set(), include_decoy=False)

        sem_simulator.run_sem_simulation(
            input_path=str(oas_path),
            marker_layer="999/0",
            pattern_layers=["1/0"],
            output_dir=str(output_dir),
            clip_size_um=1.0,
            pixel_size_nm=20,
            seed=5,
            write_png=False,
        )

        row = _read_jsonl(output_dir / "behavior.jsonl")[0]
        image = _load_first_image(output_dir)
        self.assertTrue(row["simulation_empty"])
        self.assertTrue(row["simulation_low_structure"])
        self.assertLess(float(np.std(image)), 0.02)
        self.assertLess(float(np.max(image) - np.min(image)), 0.10)

    def test_layout_proxy_risk_scores_structure_above_empty(self):
        """layout-proxy risk 应让结构窗口风险高于空窗口。"""
        oas_path = self.temp_root / "mixed.oas"
        output_dir = self.temp_root / "proxy_risk"
        _make_oas(oas_path, marker_count=2, signal_indices={1}, include_decoy=False)

        sem_simulator.run_sem_simulation(
            input_path=str(oas_path),
            marker_layer="999/0",
            pattern_layers=["1/0"],
            output_dir=str(output_dir),
            clip_size_um=1.0,
            pixel_size_nm=20,
            risk_mode="layout-proxy",
            write_png=False,
        )

        rows = _read_jsonl(output_dir / "behavior.jsonl")
        self.assertEqual(rows[0]["risk_score_source"], "layout_sem_proxy")
        self.assertLess(float(rows[0]["risk_score"]), float(rows[1]["risk_score"]))
        self.assertEqual(float(rows[0]["risk_score"]), 0.0)
        self.assertIn("edge_density_score", rows[1]["risk_components"])

    def test_risk_score_csv_overrides_layout_proxy(self):
        """risk_score CSV 应优先于 layout-proxy 自动风险。"""
        oas_path = self.temp_root / "mixed.oas"
        output_dir = self.temp_root / "proxy_with_csv"
        risk_csv = self.temp_root / "proxy_risk.csv"
        _make_oas(oas_path, marker_count=2, signal_indices={1}, include_decoy=False)
        risk_csv.write_text("marker_id,risk_score\nmixed__marker_000001,0.05\n", encoding="utf-8")

        sem_simulator.run_sem_simulation(
            input_path=str(oas_path),
            marker_layer="999/0",
            pattern_layers=["1/0"],
            output_dir=str(output_dir),
            clip_size_um=1.0,
            pixel_size_nm=20,
            risk_mode="layout-proxy",
            risk_score_csv=str(risk_csv),
            write_png=False,
        )

        rows = _read_jsonl(output_dir / "behavior.jsonl")
        self.assertEqual(rows[1]["risk_score_source"], "csv")
        self.assertAlmostEqual(float(rows[1]["risk_score"]), 0.05)

    def test_fixed_normalization_preserves_empty_and_structured_mean_gap(self):
        """fixed normalization 应保留空窗口和结构窗口的强度差异。"""
        oas_path = self.temp_root / "mixed.oas"
        output_dir = self.temp_root / "fixed_norm"
        _make_oas(oas_path, marker_count=2, signal_indices={1}, include_decoy=False)

        sem_simulator.run_sem_simulation(
            input_path=str(oas_path),
            marker_layer="999/0",
            pattern_layers=["1/0"],
            output_dir=str(output_dir),
            clip_size_um=1.0,
            pixel_size_nm=20,
            normalization_mode="fixed",
            model_profile="clean",
            noise_scale=0.0,
            write_png=False,
        )

        rows = _read_jsonl(output_dir / "behavior.jsonl")
        self.assertEqual(rows[0]["normalization_mode"], "fixed")
        self.assertGreater(float(rows[1]["image_mean"]) - float(rows[0]["image_mean"]), 0.03)

    def test_layer_operations_can_generate_effective_pattern_layer(self):
        """启用 layer ops 后，derived result layer 应能参与仿真。"""
        oas_path = self.temp_root / "layer_ops.oas"
        out_no_ops = self.temp_root / "layer_ops_disabled"
        out_ops = self.temp_root / "layer_ops_enabled"
        _make_layer_ops_oas(oas_path)

        sem_simulator.run_sem_simulation(
            input_path=str(oas_path),
            marker_layer="999/0",
            pattern_layers=["3/0"],
            output_dir=str(out_no_ops),
            clip_size_um=1.0,
            pixel_size_nm=20,
            write_png=False,
        )
        sem_simulator.run_sem_simulation(
            input_path=str(oas_path),
            marker_layer="999/0",
            pattern_layers=["3/0"],
            output_dir=str(out_ops),
            clip_size_um=1.0,
            pixel_size_nm=20,
            apply_layer_ops=True,
            register_op=[["1/0", "2/0", "subtract", "3/0"]],
            write_png=False,
        )

        row_no_ops = _read_jsonl(out_no_ops / "behavior.jsonl")[0]
        row_ops = _read_jsonl(out_ops / "behavior.jsonl")[0]
        self.assertTrue(row_no_ops["simulation_empty"])
        self.assertFalse(row_ops["simulation_empty"])
        self.assertGreater(float(row_ops["pattern_pixel_ratio"]), 0.0)

    def test_notrain_backend_accepts_simulator_output_directory(self):
        """仿真输出目录应能直接作为 no-train backend 的 behavior_manifest。"""
        oas_path = self.temp_root / "unit.oas"
        output_dir = self.temp_root / "backend_behavior"
        _make_oas(oas_path, marker_count=2)
        sem_simulator.run_sem_simulation(
            input_path=str(oas_path),
            marker_layer="999/0",
            pattern_layers=["1/0"],
            output_dir=str(output_dir),
            clip_size_um=1.0,
            pixel_size_nm=20,
            risk_score=1.0,
            write_png=False,
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
            behavior_verification_threshold=0.30,
            local_residual_threshold=0.30,
            similarity_tau=None,
            similarity_tau_min=0.10,
            verification_shift_px=1,
            risk_weight_scale=1.0,
            recursive_input=False,
            high_risk_quantile=0.90,
            review_dir=None,
            export_cluster_review_dir=None,
            apply_layer_ops=False,
            register_op=None,
        )

        result = mp_backend.run_notrain_mp_selection(args)

        self.assertEqual(result["result_summary"]["input_marker_count"], 2)
        self.assertEqual(result["result_summary"]["selected_cluster_count"], 2)
        self.assertEqual(result["pipeline_mode"], "hotspot_recipe_notrain_backend_v1")


if __name__ == "__main__":
    unittest.main()
