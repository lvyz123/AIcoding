#!/usr/bin/env python3
"""Programmatic example for the optimized behavior clustering entrypoint."""

from __future__ import annotations

import argparse
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "example_outputs"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import layout_clustering_optimized as optimized


def build_config(args: argparse.Namespace, diff_channels: Tuple[str, ...]) -> Dict[str, Any]:
    """把示例 CLI 参数转换成 OptimizedMainlineRunner 配置。"""
    return {
        "marker_layer": str(args.marker_layer),
        "clip_size_um": float(args.clip_size),
        "behavior_manifest": str(args.behavior_manifest),
        "feature_npz": str(args.feature_npz),
        "ann_top_k": int(args.ann_top_k),
        "coverage_target": float(args.coverage_target),
        "facility_min_gain": float(args.facility_min_gain),
        "behavior_verification_threshold": float(args.behavior_verification_threshold),
        "high_risk_quantile": float(args.high_risk_quantile),
        "export_diff_channels": diff_channels,
        "apply_layer_operations": bool(args.apply_layer_ops or args.register_op),
    }


def run_optimized_example(args: argparse.Namespace) -> Tuple[Dict[str, Any], Path]:
    """以程序化方式运行 optimized behavior clustering，并保存结果。"""
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(args.input).stem}_optimized_example.{args.format}"
    temp_dir = output_dir / f"_temp_optimized_{uuid.uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=False)

    layer_processor = optimized._make_layer_processor(args.register_op or [])
    diff_channels = optimized._parse_diff_channels(args.export_diff_channels)
    runner = optimized.OptimizedMainlineRunner(
        config=build_config(args, diff_channels),
        temp_dir=temp_dir,
        layer_processor=layer_processor if (args.apply_layer_ops or args.register_op) else None,
    )
    try:
        result = runner.run(str(args.input))
        if args.review_dir:
            optimized._export_review(result, str(Path(args.review_dir).resolve()), diff_channels=diff_channels)
        optimized._save_results(result, str(output_path), str(args.format))
        return result, output_path
    finally:
        if args.cleanup_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)


def print_result_summary(result: Dict[str, Any], output_path: Path) -> None:
    """打印示例运行的关键统计。"""
    print(f"total_clusters={result.get('total_clusters', 0)}")
    print(f"total_samples={result.get('total_samples', 0)}")
    print(f"selected_representatives={result.get('selected_representative_count', 0)}")
    print(f"output={output_path}")


def main() -> int:
    """示例入口：解析参数并调用当前 optimized 行为聚类 runner。"""
    parser = argparse.ArgumentParser(description="optimized behavior clustering 的程序化调用示例")
    parser.add_argument("input", help="输入 OASIS 文件或目录")
    parser.add_argument("--marker-layer", required=True, help="marker 层，格式如 999/0")
    parser.add_argument("--behavior-manifest", required=True, help="behavior JSONL manifest 路径")
    parser.add_argument("--feature-npz", required=True, help="AutoEncoder encode 生成的 features.npz")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="结果输出目录")
    parser.add_argument("--format", choices=["json", "txt"], default="json", help="输出格式")
    parser.add_argument("--clip-size", type=float, default=1.35, help="marker-centered clip 边长，单位 um")
    parser.add_argument("--ann-top-k", type=int, default=64, help="ANN top-K graph 的近邻数")
    parser.add_argument("--coverage-target", type=float, default=0.985, help="coverage 目标")
    parser.add_argument("--facility-min-gain", type=float, default=1e-4, help="facility selection 最小边际收益")
    parser.add_argument("--behavior-verification-threshold", type=float, default=0.08, help="behavior verification 阈值")
    parser.add_argument("--high-risk-quantile", type=float, default=0.90, help="high-risk 分位数")
    parser.add_argument("--export-diff-channels", default="", help="可选 diff channels: aerial,resist,pv")
    parser.add_argument("--review-dir", default=None, help="可选 review 输出目录")
    parser.add_argument("--apply-layer-ops", action="store_true", help="聚类前应用注册的 layer operations")
    parser.add_argument(
        "--register-op",
        action="append",
        nargs=4,
        metavar=("SOURCE_LAYER", "TARGET_LAYER", "OPERATION", "RESULT_LAYER"),
        help="注册层操作规则，例如 --register-op 1/0 2/0 subtract 10/0",
    )
    parser.add_argument("--cleanup-temp", action="store_true", help="运行结束后删除临时 sample/representative 目录")
    args = parser.parse_args()

    result, output_path = run_optimized_example(args)
    print_result_summary(result, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
