#!/usr/bin/env python3
"""Small programmatic example for the current mainline pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "example_outputs"
DEFAULT_SAMPLE_CANDIDATES = (
    WORKSPACE_ROOT / "layoutgenerator" / "out_oas" / "sample_layout_000.oas",
    WORKSPACE_ROOT / "layoutgenerator" / "out_oas" / "sample_layout_001.oas",
)

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import layout_clustering_clip_shifting as mainline_mod


def resolve_default_input() -> Path:
    for candidate in DEFAULT_SAMPLE_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("未找到默认示例版图，请通过 --input 指定 .oas 文件。")


def build_config(hotspot_layer: str) -> Dict[str, Any]:
    return {
        "clip_size_um": 1.35,
        "hotspot_layer": hotspot_layer,
        "matching_mode": "ecc",
        "solver": "auto",
        "geometry_mode": "exact",
        "area_match_ratio": 0.96,
        "edge_tolerance_um": 0.02,
        "max_elements_per_window": 256,
        "clip_shift_directions": "left,right,up,down",
        "clip_shift_boundary_tolerance_um": 0.02,
        "apply_layer_operations": False,
    }


def run_pipeline_example(
    input_path: Path,
    output_dir: Path,
    hotspot_layer: str,
    export_review: bool = False,
) -> Tuple[Dict[str, Any], Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_mainline_example.json"
    review_dir = output_dir / f"{input_path.stem}_mainline_review"

    pipeline = mainline_mod.LayoutClusteringPipeline(build_config(hotspot_layer))
    try:
        result = pipeline.run_pipeline(str(input_path))
        if export_review:
            pipeline.export_cluster_review(str(review_dir))
        pipeline.save_results(str(output_path), format="json")
        return result, output_path
    finally:
        pipeline.cleanup()


def print_result_summary(result: Dict[str, Any], output_path: Path) -> None:
    summary = result.get("result_summary", {})
    print(f"total_files={result.get('total_files', 0)}")
    print(f"total_clusters={result.get('total_clusters', 0)}")
    print(f"cluster_sizes={summary.get('cluster_sizes', [])[:10]}")
    print(f"output={output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="当前 mainline 的程序化调用示例")
    parser.add_argument("--input", type=str, default=None, help="输入 OASIS 文件或目录")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="结果输出目录")
    parser.add_argument("--hotspot-layer", type=str, required=True, help="marker 层号，格式如 999/0")
    parser.add_argument("--export-review", action="store_true", help="导出 cluster review 目录")
    args = parser.parse_args()

    input_path = Path(args.input).resolve() if args.input else resolve_default_input()
    if not input_path.exists():
        print(f"输入路径不存在: {input_path}")
        return 1

    result, output_path = run_pipeline_example(
        input_path=input_path,
        output_dir=Path(args.output_dir).resolve(),
        hotspot_layer=args.hotspot_layer,
        export_review=bool(args.export_review),
    )
    print_result_summary(result, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
