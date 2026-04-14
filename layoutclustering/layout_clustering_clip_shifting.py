#!/usr/bin/env python3
"""Raster 主线的薄壳入口。

这里不承载算法本体，主要负责:
1. 解析 CLI 参数
2. 组装主线配置
3. 调用 MainlineRunner
4. 保存结果并导出 review 目录
"""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from layer_operations import LayerOperationProcessor
from mainline import MainlineRunner


class LayoutClusteringPipeline:
    """主线算法的轻量包装器。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = dict(config or {})
        self.apply_layer_operations = bool(self.config.get("apply_layer_operations", False))
        self.layer_processor = LayerOperationProcessor()

        self.mainline_result: Dict[str, Any] = {}
        self.filepaths = []
        self.sample_infos = []
        self.clusters = []
        self.representatives = []
        self.cluster_review_info: Dict[str, Any] = {}
        self._owned_temp_dir: Optional[Path] = None

    def _create_run_temp_dir(self) -> Path:
        if self._owned_temp_dir is not None:
            shutil.rmtree(self._owned_temp_dir, ignore_errors=True)
        root = Path(__file__).resolve().parent / "_temp_runs"
        root.mkdir(parents=True, exist_ok=True)
        self._owned_temp_dir = root / f"layout_clustering_mainline_{uuid.uuid4().hex[:8]}"
        self._owned_temp_dir.mkdir(parents=True, exist_ok=False)
        return self._owned_temp_dir

    def run_mainline(self, input_path: str) -> Dict[str, Any]:
        """直接调用唯一保留的 mainline。"""
        runner = MainlineRunner(
            config=self.config,
            temp_dir=self._create_run_temp_dir(),
            layer_processor=self.layer_processor if self.apply_layer_operations else None,
        )
        result = runner.run(input_path)
        self.mainline_result = dict(result)
        self.filepaths = list(result.get("file_list", []))
        self.sample_infos = list(result.get("file_metadata", []))
        self.clusters = list(result.get("clusters", []))
        self.representatives = [
            cluster.get("representative_file")
            for cluster in self.clusters
            if cluster.get("representative_file")
        ]
        self.cluster_review_info = dict(result.get("cluster_review", {}))
        return dict(self.mainline_result)

    def run_pipeline(self, input_path: str, split_large_layout: bool = False) -> Dict[str, Any]:
        del split_large_layout
        return self.run_mainline(input_path)

    def get_results(self) -> Dict[str, Any]:
        return dict(self.mainline_result)

    def save_results(self, output_path: str, format: str = "json") -> None:
        result = self.get_results()
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with output.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2, ensure_ascii=False)
            print(f"结果已保存到: {output}")
            return

        if format != "txt":
            raise ValueError(f"不支持的格式: {format}")

        with output.open("w", encoding="utf-8") as handle:
            handle.write("Layout Clustering Mainline Result\n")
            handle.write("=" * 40 + "\n\n")
            handle.write(f"总文件数: {int(result.get('total_files', 0))}\n")
            handle.write(f"总聚类数: {int(result.get('total_clusters', 0))}\n")
            handle.write(f"总样本数: {int(result.get('total_samples', 0))}\n")
            handle.write(f"匹配模式: {result.get('matching_mode', 'n/a')}\n")
            handle.write(f"求解器: {result.get('solver_used', 'n/a')}\n")
            handle.write(f"几何模式: {result.get('geometry_mode', 'n/a')}\n")
            handle.write(f"像素尺寸(nm): {result.get('pixel_size_nm', 'n/a')}\n\n")
            handle.write("聚类大小分布:\n")
            for cluster_id, size in enumerate(result.get("cluster_sizes", [])):
                handle.write(f"  cluster {cluster_id}: {int(size)}\n")
            handle.write("\n详细聚类信息:\n")
            for cluster in result.get("clusters", []):
                handle.write(f"\ncluster {int(cluster['cluster_id'])} (size={int(cluster['size'])}):\n")
                handle.write(f"  representative_file: {Path(str(cluster.get('representative_file', ''))).name}\n")
                handle.write(f"  candidate_id: {cluster.get('selected_candidate_id')}\n")
                handle.write(f"  shift_direction: {cluster.get('selected_shift_direction')}\n")
                handle.write(f"  shift_distance_um: {cluster.get('selected_shift_distance_um')}\n")
        print(f"结果已保存到: {output}")

    def export_cluster_review(self, output_dir: str) -> Dict[str, Any]:
        result = self.get_results()
        clusters = result.get("clusters", [])
        if not clusters:
            self.cluster_review_info = {}
            return {}

        review_root = Path(output_dir)
        review_root.mkdir(parents=True, exist_ok=True)
        representative_files = []
        exported_file_count = 0
        missing_files = []

        for cluster in clusters:
            cluster_id = int(cluster["cluster_id"])
            cluster_size = int(cluster["size"])
            cluster_dir = review_root / f"cluster_{cluster_id:04d}_size_{cluster_size:04d}"
            cluster_dir.mkdir(parents=True, exist_ok=True)

            representative_path = str(cluster.get("representative_file", ""))
            representative_files.append(representative_path)
            rep_copied = False

            for member_idx, src in enumerate(cluster.get("sample_files", [])):
                src_path = Path(src)
                if not src_path.exists():
                    missing_files.append(str(src_path))
                    continue
                prefix = "REP__" if str(src_path) == representative_path else "sample__"
                dest_name = f"{prefix}{member_idx:04d}__{src_path.name}"
                shutil.copy2(src_path, cluster_dir / dest_name)
                exported_file_count += 1
                if str(src_path) == representative_path:
                    rep_copied = True

            if representative_path and not rep_copied:
                rep_src = Path(representative_path)
                if rep_src.exists():
                    shutil.copy2(rep_src, cluster_dir / f"REP__selected__{rep_src.name}")
                    exported_file_count += 1
                else:
                    missing_files.append(str(rep_src))

        with (review_root / "representative_files.txt").open("w", encoding="utf-8") as handle:
            for filepath in representative_files:
                handle.write(f"{filepath}\n")

        self.cluster_review_info = {
            "exported": True,
            "review_dir": str(review_root),
            "cluster_count": int(len(clusters)),
            "exported_file_count": int(exported_file_count),
            "representative_file_count": int(len(representative_files)),
            "missing_file_count": int(len(missing_files)),
        }
        if missing_files:
            self.cluster_review_info["missing_files_preview"] = missing_files[:10]
        self.mainline_result["cluster_review"] = dict(self.cluster_review_info)
        print(f"cluster review 目录已导出到: {review_root}")
        return dict(self.cluster_review_info)

    def cleanup(self) -> None:
        if self._owned_temp_dir is not None:
            temp_root = self._owned_temp_dir.parent
            shutil.rmtree(self._owned_temp_dir, ignore_errors=True)
            self._owned_temp_dir = None
            try:
                temp_root.rmdir()
            except OSError:
                pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Marker-driven layout clustering mainline (raster-first, follow Chen 2017)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n\n"
            "1) Basic\n"
            "python layout_clustering_clip_shifting.py ./design.oas --hotspot-layer 999/0 --output results.json\n\n"
            "2) ACC matching\n"
            "python layout_clustering_clip_shifting.py ./design_dir --hotspot-layer 999/0 --matching-mode acc\n\n"
            "3) Fast raster mode\n"
            "python layout_clustering_clip_shifting.py ./design.oas --hotspot-layer 999/0 --geometry-mode fast --max-elements-per-window 512\n"
        ),
    )

    io_group = parser.add_argument_group("输入输出参数")
    mainline_group = parser.add_argument_group("主线参数")
    advanced_group = parser.add_argument_group("高级参数")

    io_group.add_argument("input", help="输入 OASIS 文件或目录路径")
    io_group.add_argument("--output", "-o", default="clustering_results.json", help="输出文件路径")
    io_group.add_argument("--format", "-f", default="json", choices=["json", "txt"], help="输出格式")
    io_group.add_argument("--export-cluster-review-dir", default="./output_clusters", help="导出 review 目录")

    mainline_group.add_argument("--hotspot-layer", required=True, help="marker 层号, 格式 layer/datatype, 例如 999/0")
    mainline_group.add_argument("--clip-size", type=float, default=1.35, help="clip 边长 (um)")
    mainline_group.add_argument("--matching-mode", choices=["acc", "ecc"], default="ecc", help="匹配判据")
    mainline_group.add_argument("--solver", choices=["greedy", "ilp", "auto"], default="auto", help="集合覆盖求解器")
    mainline_group.add_argument("--geometry-mode", choices=["exact", "fast"], default="exact", help="raster 主线运行模式")
    mainline_group.add_argument("--pixel-size-nm", type=int, default=10, help="raster 像素尺寸 (nm)")
    mainline_group.add_argument("--area-match-ratio", type=float, default=0.96, help="ACC 面积匹配阈值")
    mainline_group.add_argument("--edge-tolerance-um", type=float, default=0.02, help="ECC 边界容差")
    mainline_group.add_argument("--max-elements-per-window", type=int, default=256, help="fast 模式窗口元素上限")
    mainline_group.add_argument("--clip-shift-directions", default="left,right,up,down", help="允许的单方向平移集合")
    mainline_group.add_argument("--clip-shift-boundary-tol-um", type=float, default=0.02, help="systematic shift 边界容差")

    advanced_group.add_argument("--apply-layer-ops", action="store_true", help="应用层操作后再运行主线")
    advanced_group.add_argument(
        "--register-op",
        nargs=4,
        metavar=("SOURCE_LAYER", "TARGET_LAYER", "OPERATION", "RESULT_LAYER"),
        help="注册层操作规则: SOURCE_LAYER TARGET_LAYER OPERATION RESULT_LAYER",
    )
    return parser


def _build_config(args: argparse.Namespace) -> Dict[str, Any]:
    """把 CLI 参数整理成 MainlineRunner 需要的配置字典。"""
    return {
        "apply_layer_operations": bool(args.apply_layer_ops),
        "clip_size_um": float(args.clip_size),
        "max_elements_per_window": int(args.max_elements_per_window),
        "hotspot_layer": str(args.hotspot_layer),
        "matching_mode": str(args.matching_mode),
        "solver": str(args.solver),
        "geometry_mode": str(args.geometry_mode),
        "pixel_size_nm": int(args.pixel_size_nm),
        "area_match_ratio": float(args.area_match_ratio),
        "edge_tolerance_um": float(args.edge_tolerance_um),
        "clip_shift_directions": str(args.clip_shift_directions),
        "clip_shift_boundary_tolerance_um": float(args.clip_shift_boundary_tol_um),
    }


def main() -> int:
    """命令行入口。"""
    parser = _build_parser()
    args = parser.parse_args()

    print("=" * 60)
    print("Layout Clustering Mainline")
    print("=" * 60)

    pipeline = None
    try:
        pipeline = LayoutClusteringPipeline(_build_config(args))
        if args.register_op:
            source_layer, target_layer, operation, result_layer = args.register_op
            pipeline.layer_processor.register_operation_rule(
                tuple(map(int, source_layer.split("/"))),
                operation,
                tuple(map(int, target_layer.split("/"))),
                tuple(map(int, result_layer.split("/"))),
            )

        result = pipeline.run_pipeline(args.input)
        pipeline.save_results(args.output, args.format)
        if args.export_cluster_review_dir:
            pipeline.export_cluster_review(args.export_cluster_review_dir)

        print(f"总聚类数: {result.get('total_clusters', 0)}")
        print(f"总样本数: {result.get('total_samples', 0)}")
        print(f"像素尺寸(nm): {result.get('pixel_size_nm', 'n/a')}")
        return 0
    except Exception as exc:
        print(f"运行失败: {exc}")
        return 1
    finally:
        if pipeline is not None:
            pipeline.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
