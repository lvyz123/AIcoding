#!/usr/bin/env python3
"""
Optimized uniform-grid layout clustering v1.

中文整体说明：
1. 这个版本是面向 pattern grouping 新任务目标的 v1 主线版，seed 生成不再使用
   pair / pseudo-DRC 等“局部关系优先”的启发式路线，而是直接在整张版图上做
   uniform grid sampling。这样做的目的，是把主目标重新对齐为 Recall 优先：
   先尽量避免系统性漏抓，再交给后半段验证链路做去重和压缩。
2. uniform grid 的步长固定为 clip size 的 50%，也就是相邻窗口在 x / y 两个方向
   上各自重叠一半。脚本不会逐个 grid cell 去查询空间索引，而是把每个几何元素的
   bbox 反投影到它所覆盖的 grid cells，上屏为 occupied cells，再只为 occupied cells
   生成 synthetic seed。这一做法既能覆盖全版图，又能避开 pair 近邻搜索的高成本。
3. 每个 grid seed 的 synthetic marker bbox 直接等于 grid cell bbox 本身。
   这样 clip window 仍然保持固定大小，而 expanded window 的 shift limit 会自然等于
   半个 grid step，语义上与 50% overlap 的滑窗主线一致，不再混入 pair/drc 那种
   “源元素 bbox 决定 marker 大小”的额外假设。
4. grid seed 生成后仍会先走 coarse bucketing，把局部环境高度相似的窗口先合并成
   一个 seed bucket，并把 bucket weight 作为后续 exact hash / set cover 的权重来源。
   这一步只负责削减重复，不改变 recall-first 的整体采样语义。
5. 后半段流程尽量复用已经验证过的稳定主线：exact hash 合并完全重复窗口，
   graph descriptor prefilter 剪掉明显不可能匹配的 candidate 对，再用 ACC / ECC
   做最终几何 gate；coverage 边构建、lazy-heap greedy set cover 和 final verification
   都沿用当前优化版的实现。
6. final verification 仍然坚持“失败就拆回 singleton”的解释性策略，不做跨 cluster
   自动修补。换句话说，v1 的思路是：前半段先尽量多抓，后半段再严格验证，保证
   最终 cluster 的 representative-member 关系仍然容易人工理解和 review。

设计原则：
- 主线唯一：layer-op 过滤 -> uniform grid seed -> coarse bucketing -> shift-cover -> verified clustering。
- v1 不再保留 pair/drc seed strategy 及其 CLI、日志、结果字段，避免旧路线残留干扰。
- 后半段尽量不动，只替换 seed 生成语义，便于隔离变量、观察 recall-first 主线的真实效果。
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import heapq
import json
import math
import shutil
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import gdstk
import hnswlib
import numpy as np
from scipy import ndimage

from layer_operations import LayerOperationProcessor
from mainline import (
    DEFAULT_PIXEL_SIZE_NM,
    ECC_DONUT_OVERLAP_RATIO,
    ECC_RESIDUAL_RATIO,
    CandidateClip,
    ExactCluster,
    LayoutIndex,
    MainlineRunner,
    MarkerRecord,
    _bbox_center,
    _bitmap_exact_key,
    _bitcount_sum_rows,
    _chunk_indices_by_row_width,
    _make_centered_bbox,
    _make_sample_filename,
    _materialize_clip_bitmap,
    _raster_window_spec,
)


GRAPH_INVARIANT_LIMIT = 0.22
GRAPH_TOPOLOGY_THRESHOLD = 6.5
GRAPH_SIGNATURE_THRESHOLD = 0.74
STRICT_INVARIANT_LIMIT = 0.20
STRICT_TOPOLOGY_THRESHOLD = 3.0
STRICT_SIGNATURE_THRESHOLD = 0.84

DUMMY_MARKER_LAYER = "65535/65535"
PIPELINE_MODE = "optimized_v1"
SEED_MODE = "uniform_grid_shift"
GRID_STEP_RATIO = 0.5
GRID_BUCKET_QUANT_UM = 0.08
GRID_MAX_DESCRIPTOR_NEIGHBORS = 256
COVERAGE_SHORTLIST_MAX_TARGETS = 64
COVERAGE_EXACT_SHORTLIST_MAX_GROUPS = 512
PRE_RASTER_FINGERPRINT_QUANT_PX = 2
_POOL_EDGE_CACHE: Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]] = {}
_COVERAGE_STRUCTURE_CACHE: Dict[int, np.ndarray] = {}


@dataclass(frozen=True)
class GridSeedCandidate:
    """uniform grid 阶段的 seed 记录。"""

    center: Tuple[float, float]
    seed_bbox: Tuple[float, float, float, float]
    grid_ix: int
    grid_iy: int
    bucket_weight: int = 1


@dataclass(frozen=True)
class GraphDescriptor:
    """clip bitmap 提取出的轻量图形描述符，用于 prefilter。"""

    invariants: np.ndarray
    topology: np.ndarray
    signature_grid: np.ndarray
    signature_proj_x: np.ndarray
    signature_proj_y: np.ndarray


@dataclass(frozen=True)
class CheapDescriptor:
    """coverage shortlist 使用的低成本 bitmap 描述符。"""

    invariants: np.ndarray
    signature_grid: np.ndarray
    signature_proj_x: np.ndarray
    signature_proj_y: np.ndarray
    area_px: int


def _empty_prefilter_stats() -> Dict[str, int]:
    """返回 prefilter / geometry gate 阶段的统计计数器。"""

    return {
        "exact_hash_pass": 0,
        "cheap_reject": 0,
        "invariant_reject": 0,
        "topology_reject": 0,
        "signature_reject": 0,
        "geometry_reject": 0,
        "geometry_pass": 0,
    }


def _empty_verification_stats() -> Dict[str, int]:
    """返回 final verification 阶段的统计计数器。"""

    return {
        "verified_pass": 0,
        "verified_reject": 0,
        "singleton_created": 0,
    }


def _empty_verification_detail_seconds() -> Dict[str, float]:
    """返回 final verification 内部分阶段耗时统计。"""

    return {
        "assignment": 0.0,
        "geometry": 0.0,
        "graph_prefilter": 0.0,
    }


def _empty_coverage_detail_seconds() -> Dict[str, float]:
    """返回 coverage 内部分阶段耗时统计。"""

    return {
        "light_bundle_build": 0.0,
        "shortlist_index": 0.0,
        "prefilter": 0.0,
        "geometry_cache": 0.0,
        "geometry_match": 0.0,
    }


def _empty_coverage_debug_stats() -> Dict[str, int]:
    """返回 coverage 内部规模统计。"""

    return {
        "bundle_count": 0,
        "max_bundle_group_count": 0,
        "candidate_group_count": 0,
        "geometry_pair_count": 0,
        "geometry_cache_group_count": 0,
        "geometry_dilated_cache_group_count": 0,
        "geometry_donut_cache_group_count": 0,
        "shortlist_subgroup_count": 0,
        "shortlist_exact_subgroup_count": 0,
        "shortlist_hnsw_subgroup_count": 0,
        "shortlist_max_subgroup_size": 0,
    }


def _empty_result_detail_seconds() -> Dict[str, float]:
    """返回结果组装内部分阶段耗时统计。"""

    return {
        "sample_metadata": 0.0,
        "sample_materialize": 0.0,
        "final_verification": 0.0,
        "cluster_output": 0.0,
        "representative_materialize": 0.0,
    }


def _layer_spec_text(layer_spec: Any) -> str:
    """把层规格统一格式化为 `layer/datatype` 文本。"""

    if isinstance(layer_spec, str):
        return str(layer_spec).strip()
    if isinstance(layer_spec, Sequence) and len(layer_spec) >= 2:
        return f"{int(layer_spec[0])}/{int(layer_spec[1])}"
    return str(layer_spec)


def _layer_operation_payload(layer_processor: Any) -> List[Dict[str, str]]:
    """从 layer processor 提取可写入结果 JSON 的规则列表。"""

    rules = list(getattr(layer_processor, "operation_rules", []) or [])
    payload = []
    for rule in rules:
        payload.append(
            {
                "source_layer": _layer_spec_text(rule.get("source_layer")),
                "target_layer": _layer_spec_text(rule.get("target_layer")),
                "operation": str(rule.get("operation", "")),
                "result_layer": _layer_spec_text(rule.get("result_layer")),
            }
        )
    return payload


def _make_layer_processor(register_ops: Sequence[Sequence[str]] | None) -> LayerOperationProcessor:
    """根据 CLI 传入的 `--register-op` 构建层操作处理器。"""

    processor = LayerOperationProcessor()
    for source_layer, target_layer, operation, result_layer in register_ops or []:
        processor.register_operation_rule(source_layer, operation, target_layer, result_layer)
    return processor


def _layout_bbox(layout_index: LayoutIndex) -> Tuple[float, float, float, float]:
    """从 pattern elements 统计整张版图的外包框。"""

    if len(layout_index.indexed_elements) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        float(np.min(layout_index.bbox_x0)),
        float(np.min(layout_index.bbox_y0)),
        float(np.max(layout_index.bbox_x1)),
        float(np.max(layout_index.bbox_y1)),
    )


def _grid_index_range(
    layout_origin: float,
    bbox_min: float,
    bbox_max: float,
    grid_step_um: float,
) -> Tuple[int, int]:
    """把元素 bbox 反投影到覆盖它的 grid 索引区间。"""

    step = max(float(grid_step_um), 1e-9)
    start = int(math.floor((float(bbox_min) - float(layout_origin)) / step + 1e-9))
    end = int(math.floor((float(bbox_max) - float(layout_origin)) / step - 1e-9))
    if end < start:
        end = start
    return start, end


def _grid_center_index_range(
    layout_origin: float,
    bbox_min: float,
    bbox_max: float,
    grid_step_um: float,
) -> Tuple[int, int] | None:
    """返回 cell 中心落在元素 bbox 内的 grid 索引区间。"""

    step = max(float(grid_step_um), 1e-9)
    start = int(math.ceil((float(bbox_min) - float(layout_origin)) / step - 0.5 - 1e-9))
    end = int(math.floor((float(bbox_max) - float(layout_origin)) / step - 0.5 - 1e-9))
    if end < start:
        return None
    return start, end


def _grid_anchor_index(
    layout_origin: float,
    coord: float,
    grid_step_um: float,
) -> int:
    """返回某个坐标所属的 anchor grid 索引。"""

    step = max(float(grid_step_um), 1e-9)
    return int(math.floor((float(coord) - float(layout_origin)) / step + 1e-9))


def _grid_cell_bbox(
    layout_bbox: Tuple[float, float, float, float],
    grid_ix: int,
    grid_iy: int,
    grid_step_um: float,
) -> Tuple[float, float, float, float]:
    """根据 grid 索引返回 cell 的物理 bbox。"""

    origin_x = float(layout_bbox[0])
    origin_y = float(layout_bbox[1])
    step = float(grid_step_um)
    x0 = origin_x + int(grid_ix) * step
    y0 = origin_y + int(grid_iy) * step
    return (x0, y0, x0 + step, y0 + step)


def _seed_bucket_outer_side(clip_size_um: float, grid_step_um: float) -> float:
    """返回 coarse bucket 描述符所使用的外层观察窗口边长。"""

    return float(clip_size_um) + float(grid_step_um)


def _seed_bucket_key(
    layout_index: LayoutIndex,
    candidate: GridSeedCandidate,
    *,
    clip_size_um: float,
    grid_step_um: float,
    quant_step_um: float = GRID_BUCKET_QUANT_UM,
) -> str:
    """为单个 seed 候选生成 coarse bucket key。"""

    outer_side = _seed_bucket_outer_side(float(clip_size_um), float(grid_step_um))
    outer_bbox = _make_centered_bbox(candidate.center, outer_side, outer_side)
    return _coarse_window_descriptor(
        candidate.center,
        outer_bbox,
        layout_index,
        quant_step_um=float(quant_step_um),
    )


def _accumulate_seed_bucket(
    buckets: Dict[str, GridSeedCandidate],
    layout_index: LayoutIndex,
    candidate: GridSeedCandidate,
    *,
    clip_size_um: float,
    grid_step_um: float,
    quant_step_um: float = GRID_BUCKET_QUANT_UM,
) -> None:
    """把单个 seed 流式并入 coarse bucket，只保留桶代表和累计权重。"""

    group_key = _seed_bucket_key(
        layout_index,
        candidate,
        clip_size_um=float(clip_size_um),
        grid_step_um=float(grid_step_um),
        quant_step_um=float(quant_step_um),
    )
    current = buckets.get(group_key)
    if current is None:
        buckets[group_key] = candidate
        return
    buckets[group_key] = replace(current, bucket_weight=int(current.bucket_weight) + int(candidate.bucket_weight))


def _build_uniform_grid_seed_candidates(
    layout_index: LayoutIndex,
    *,
    clip_size_um: float,
) -> Tuple[List[GridSeedCandidate], Dict[str, int]]:
    """按 uniform grid 采样并直接完成 coarse bucketing。"""

    grid_step_um = float(clip_size_um) * float(GRID_STEP_RATIO)
    if len(layout_index.indexed_elements) == 0:
        return [], {
            "seed_strategy": "uniform_grid",
            "grid_step_ratio": float(GRID_STEP_RATIO),
            "grid_step_um": float(grid_step_um),
            "grid_seed_count": 0,
            "initial_seed_count": 0,
            "bucketed_seed_count": 0,
            "seed_bucket_merged_count": 0,
        }

    layout_bbox = _layout_bbox(layout_index)
    center_covered_cells: set[Tuple[int, int]] = set()
    anchor_cells: set[Tuple[int, int]] = set()
    for item in layout_index.indexed_elements:
        bbox = tuple(float(v) for v in item["bbox"])
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        center_ix_range = _grid_center_index_range(layout_bbox[0], bbox[0], bbox[2], grid_step_um)
        center_iy_range = _grid_center_index_range(layout_bbox[1], bbox[1], bbox[3], grid_step_um)
        if center_ix_range is not None and center_iy_range is not None:
            for grid_ix in range(int(center_ix_range[0]), int(center_ix_range[1]) + 1):
                for grid_iy in range(int(center_iy_range[0]), int(center_iy_range[1]) + 1):
                    center_covered_cells.add((int(grid_ix), int(grid_iy)))
        anchor_cells.add(
            (
                _grid_anchor_index(layout_bbox[0], 0.5 * (bbox[0] + bbox[2]), grid_step_um),
                _grid_anchor_index(layout_bbox[1], 0.5 * (bbox[1] + bbox[3]), grid_step_um),
            )
        )

    eligible_cells = center_covered_cells | anchor_cells

    bucketed_candidates: Dict[str, GridSeedCandidate] = {}
    for grid_ix, grid_iy in sorted(eligible_cells):
        seed_bbox = _grid_cell_bbox(layout_bbox, int(grid_ix), int(grid_iy), grid_step_um)
        _accumulate_seed_bucket(
            bucketed_candidates,
            layout_index,
            GridSeedCandidate(
                center=_bbox_center(seed_bbox),
                seed_bbox=seed_bbox,
                grid_ix=int(grid_ix),
                grid_iy=int(grid_iy),
            ),
            clip_size_um=float(clip_size_um),
            grid_step_um=float(grid_step_um),
        )

    grid_seed_count = int(len(eligible_cells))
    bucketed = list(bucketed_candidates.values())
    return bucketed, {
        "seed_strategy": "uniform_grid",
        "grid_step_ratio": float(GRID_STEP_RATIO),
        "grid_step_um": float(grid_step_um),
        "grid_seed_count": int(grid_seed_count),
        "initial_seed_count": int(grid_seed_count),
        "bucketed_seed_count": int(len(bucketed)),
        "seed_bucket_merged_count": int(max(0, grid_seed_count - len(bucketed))),
    }


def _coarse_window_descriptor(
    center_xy: Tuple[float, float],
    outer_bbox: Tuple[float, float, float, float],
    layout_index: LayoutIndex,
    *,
    quant_step_um: float = GRID_BUCKET_QUANT_UM,
    max_neighbors: int = GRID_MAX_DESCRIPTOR_NEIGHBORS,
) -> str:
    """为某个 seed 周围邻域生成轻量级粗分桶描述符。"""

    cx, cy = (float(center_xy[0]), float(center_xy[1]))
    neighbor_ids = list(layout_index.spatial_index.intersection(tuple(float(v) for v in outer_bbox)))
    if not neighbor_ids:
        return "empty"

    q = max(float(quant_step_um), 1e-6)
    ox0, oy0, ox1, oy1 = (float(v) for v in outer_bbox)
    max_radius_x = max((ox1 - ox0) * 0.5, q)
    max_radius_y = max((oy1 - oy0) * 0.5, q)
    parts: List[Tuple[int, int, int, int, int, int, Tuple[int, int, int, int]]] = []
    for elem_id in neighbor_ids[: max(1, int(max_neighbors))]:
        item = layout_index.indexed_elements[int(elem_id)]
        min_x, min_y, max_x, max_y = (float(v) for v in item["bbox"])
        rel = (
            int(round((min_x - cx) / q)),
            int(round((min_y - cy) / q)),
            int(round((max_x - cx) / q)),
            int(round((max_y - cy) / q)),
        )
        bbox_w = int(round((max_x - min_x) / q))
        bbox_h = int(round((max_y - min_y) / q))
        nx = int(round((((min_x + max_x) * 0.5) - cx) / max_radius_x * 16.0))
        ny = int(round((((min_y + max_y) * 0.5) - cy) / max_radius_y * 16.0))
        parts.append((int(item["layer"]), int(item["datatype"]), bbox_w, bbox_h, nx, ny, rel))

    parts.sort()
    payload = "|".join(
        f"{layer}:{datatype}:{bw}:{bh}:{nx}:{ny}:{r0}:{r1}:{r2}:{r3}"
        for layer, datatype, bw, bh, nx, ny, (r0, r1, r2, r3) in parts
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _review_dir_from_args(args: argparse.Namespace) -> str | None:
    """统一解析 review 目录参数，并兼容旧别名。"""

    review_dir = getattr(args, "review_dir", None)
    legacy_dir = getattr(args, "export_cluster_review_dir", None)
    if review_dir and legacy_dir:
        print(f"同时指定 --review-dir 和 --export-cluster-review-dir，使用 --review-dir: {review_dir}")
        return str(review_dir)
    return str(review_dir or legacy_dir) if (review_dir or legacy_dir) else None


def _print_start_banner(title: str, args: argparse.Namespace, *, apply_layer_operations: bool, layer_ops: Sequence[Dict[str, str]]) -> None:
    """打印脚本启动时的中文阶段摘要。"""

    print("=" * 60)
    print(title)
    print("=" * 60)
    print(f"输入路径: {args.input_path}")
    print(f"输出路径: {args.output}")
    print("seed 策略: uniform_grid")
    print(f"grid step ratio: {GRID_STEP_RATIO:.2f}")
    print(f"grid step: {float(args.clip_size) * GRID_STEP_RATIO:.4f} um")
    print(f"几何匹配模式: {args.geometry_match_mode}")
    print(f"像素尺寸: {args.pixel_size_nm} nm")
    print(f"层操作启用: {'是' if apply_layer_operations else '否'}")
    if layer_ops:
        print(f"层操作规则数: {len(layer_ops)}")
        for idx, rule in enumerate(layer_ops, start=1):
            print(
                f"  规则 {idx}: {rule['source_layer']} {rule['operation']} "
                f"{rule['target_layer']} -> {rule['result_layer']}"
            )
    print("=" * 60)


def _write_text_summary(result: Dict[str, Any], output_path: Path) -> None:
    """把 JSON 结果压缩成适合快速查看的 TXT 摘要。"""

    summary = result.get("result_summary", {})
    config = result.get("config", {})
    layer_ops = result.get("layer_operations", [])
    cluster_sizes = list(result.get("cluster_sizes", []))
    top_sizes = sorted((int(v) for v in cluster_sizes), reverse=True)[:10]

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("Optimized v1 Layout 聚类结果摘要\n")
        handle.write("=" * 40 + "\n")
        handle.write(f"pipeline mode: {result.get('pipeline_mode')}\n")
        handle.write(f"seed strategy: {result.get('seed_strategy')}\n")
        handle.write(f"grid step ratio: {result.get('grid_step_ratio')}\n")
        handle.write(f"grid step um: {result.get('grid_step_um')}\n")
        handle.write(f"geometry match mode: {result.get('geometry_match_mode')}\n")
        handle.write(f"pixel size nm: {result.get('pixel_size_nm')}\n")
        handle.write(f"area match ratio: {result.get('area_match_ratio')}\n")
        handle.write(f"edge tolerance um: {result.get('edge_tolerance_um')}\n")
        handle.write("\n统计:\n")
        handle.write(f"  seed/sample 数: {result.get('marker_count')} / {result.get('total_samples')}\n")
        handle.write(
            f"  raw/bucketed/merged grid seed 数: {result.get('grid_seed_count')} / "
            f"{result.get('bucketed_seed_count')} / {result.get('seed_bucket_merged_count')}\n"
        )
        handle.write(
            f"  pre-raster cache hit/miss: {result.get('pre_raster_cache_hit', 0)} / "
            f"{result.get('pre_raster_cache_miss', 0)}\n"
        )
        handle.write(
            f"  exact bitmap cache hit/miss: {result.get('exact_bitmap_cache_hit', 0)} / "
            f"{result.get('exact_bitmap_cache_miss', 0)}\n"
        )
        handle.write(f"  exact cluster 数: {result.get('exact_cluster_count')}\n")
        handle.write(f"  candidate 数: {result.get('candidate_count')}\n")
        handle.write(f"  selected candidate 数: {result.get('selected_candidate_count')}\n")
        handle.write(f"  final cluster 数: {result.get('total_clusters')}\n")
        handle.write(f"  selected candidate 方向分布: {result.get('selected_candidate_direction_counts')}\n")
        handle.write(f"  final cluster 方向分布: {result.get('final_cluster_direction_counts')}\n")
        handle.write(f"  最大 shift 距离(um): {result.get('max_shift_distance_um')}\n")
        handle.write(f"  top cluster sizes: {top_sizes}\n")
        handle.write("\n层操作:\n")
        handle.write(f"  apply_layer_operations: {result.get('apply_layer_operations')}\n")
        handle.write(f"  layer_operation_count: {result.get('layer_operation_count')}\n")
        handle.write(f"  effective clustering layers: {result.get('effective_clustering_layers', [])}\n")
        handle.write(f"  excluded helper layers: {result.get('excluded_helper_layers', [])}\n")
        if layer_ops:
            for idx, rule in enumerate(layer_ops, start=1):
                handle.write(
                    f"  {idx}. {rule['source_layer']} {rule['operation']} "
                    f"{rule['target_layer']} -> {rule['result_layer']}\n"
                )
        handle.write("\nprefilter stats:\n")
        handle.write(json.dumps(result.get("prefilter_stats", {}), ensure_ascii=False, indent=2))
        handle.write("\ncoverage detail seconds:\n")
        handle.write(json.dumps(result.get("coverage_detail_seconds", {}), ensure_ascii=False, indent=2))
        handle.write("\ncoverage debug stats:\n")
        handle.write(json.dumps(result.get("coverage_debug_stats", {}), ensure_ascii=False, indent=2))
        handle.write("\nresult detail seconds:\n")
        handle.write(json.dumps(result.get("result_detail_seconds", {}), ensure_ascii=False, indent=2))
        handle.write("\nfinal verification stats:\n")
        handle.write(json.dumps(result.get("final_verification_stats", {}), ensure_ascii=False, indent=2))
        handle.write("\nfinal verification detail seconds:\n")
        handle.write(json.dumps(result.get("final_verification_detail_seconds", {}), ensure_ascii=False, indent=2))
        handle.write("\nconfig:\n")
        handle.write(json.dumps(config or summary.get("config", {}), ensure_ascii=False, indent=2, default=_json_default))
        handle.write("\n")


def _save_results(result: Dict[str, Any], output_path: str, output_format: str) -> None:
    """根据输出格式把结果保存为 JSON 或 TXT 文件。"""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if str(output_format).lower() == "txt":
        _write_text_summary(result, output)
    else:
        with output.open("w", encoding="utf-8") as handle:
            sample_count = int(result.get("total_samples") or result.get("marker_count") or 0)
            cluster_count = int(result.get("total_clusters") or 0)
            if sample_count >= 1000 or cluster_count >= 1000:
                json.dump(result, handle, ensure_ascii=False, separators=(",", ":"), default=_json_default)
            else:
                json.dump(result, handle, indent=2, ensure_ascii=False, default=_json_default)
    print(f"结果已保存到: {output}")


def _pool_bitmap(bitmap: np.ndarray, bins: int = 10) -> np.ndarray:
    """把 bitmap 池化成固定大小的密度网格。"""

    src_bool = np.asarray(bitmap, dtype=bool)
    pooled = np.zeros((bins, bins), dtype=np.float32)
    if src_bool.ndim != 2 or src_bool.size == 0:
        return pooled

    h, w = (int(src_bool.shape[0]), int(src_bool.shape[1]))
    if not np.any(src_bool):
        return pooled
    if h < bins or w < bins:
        active = np.argwhere(src_bool)
        if active.size == 0:
            return pooled
        row_edges = np.linspace(0, h, bins + 1, dtype=np.int32)
        col_edges = np.linspace(0, w, bins + 1, dtype=np.int32)
        row_bins = np.searchsorted(row_edges[1:], active[:, 0], side="right")
        col_bins = np.searchsorted(col_edges[1:], active[:, 1], side="right")
        counts = np.zeros((bins, bins), dtype=np.float32)
        np.add.at(counts, (row_bins, col_bins), 1.0)
    else:
        cache_key = (h, w, int(bins))
        cached_edges = _POOL_EDGE_CACHE.get(cache_key)
        if cached_edges is None:
            cached_edges = (
                np.linspace(0, h, bins + 1, dtype=np.int32),
                np.linspace(0, w, bins + 1, dtype=np.int32),
            )
            _POOL_EDGE_CACHE[cache_key] = cached_edges
        row_edges, col_edges = cached_edges
        src_float = src_bool.astype(np.float32, copy=False)
        counts = np.add.reduceat(src_float, row_edges[:-1], axis=0)
        counts = np.add.reduceat(counts, col_edges[:-1], axis=1)[:bins, :bins].astype(np.float32, copy=False)

    row_sizes = np.maximum(np.diff(row_edges).astype(np.float32, copy=False), 1.0)
    col_sizes = np.maximum(np.diff(col_edges).astype(np.float32, copy=False), 1.0)
    pooled = counts / (row_sizes[:, None] * col_sizes[None, :])
    total = float(np.sum(pooled))
    if total > 0.0:
        pooled /= total
    return pooled


def _bitmap_descriptor(bitmap: np.ndarray) -> GraphDescriptor:
    """从 bitmap 提取 invariant、topology 和 signature 描述符。"""

    mask = np.asarray(bitmap, dtype=bool)
    h, w = mask.shape
    total_px = max(int(mask.size), 1)
    active = np.argwhere(mask)
    fill_ratio = float(active.shape[0]) / float(total_px)

    if active.size == 0:
        invariants = np.zeros(8, dtype=np.float64)
        topology = np.zeros(8, dtype=np.float64)
    else:
        rows = active[:, 0].astype(np.float64)
        cols = active[:, 1].astype(np.float64)
        bbox_h = float(np.max(rows) - np.min(rows) + 1.0)
        bbox_w = float(np.max(cols) - np.min(cols) + 1.0)
        bbox_long = max(bbox_w, bbox_h)
        bbox_short = min(bbox_w, bbox_h)
        span = max(float(h), float(w), 1.0)
        extent_area = max(bbox_w * bbox_h, 1.0)
        centroid = np.asarray([float(np.mean(rows)), float(np.mean(cols))])
        radii = np.linalg.norm(active.astype(np.float64) - centroid[None, :], axis=1)

        structure = np.ones((3, 3), dtype=bool)
        labels, component_count = ndimage.label(mask, structure=structure)
        if component_count > 0:
            component_sizes = np.bincount(labels.reshape(-1))[1:].astype(np.float64)
        else:
            component_sizes = np.empty(0, dtype=np.float64)
        logs = np.log1p(component_sizes) if component_sizes.size else np.zeros(1, dtype=np.float64)

        invariants = np.asarray(
            [
                math.log1p(float(component_count)),
                fill_ratio,
                bbox_long / span,
                bbox_short / span,
                bbox_short / max(bbox_long, 1.0),
                float(active.shape[0]) / extent_area,
                float(np.mean(radii)) / span,
                float(np.std(radii)) / span,
            ],
            dtype=np.float64,
        )
        topology = np.asarray(
            [
                math.log1p(float(component_count)),
                float(np.max(logs)),
                float(np.mean(logs)),
                float(np.std(logs)),
                bbox_long / span,
                bbox_short / span,
                fill_ratio,
                float(active.shape[0]) / extent_area,
            ],
            dtype=np.float64,
        )

    pooled = _pool_bitmap(mask, bins=10)
    return GraphDescriptor(
        invariants=invariants,
        topology=topology,
        signature_grid=pooled.reshape(-1).astype(np.float32),
        signature_proj_x=np.sum(pooled, axis=1, dtype=np.float32),
        signature_proj_y=np.sum(pooled, axis=0, dtype=np.float32),
    )


def _cheap_bitmap_descriptor(bitmap: np.ndarray) -> CheapDescriptor:
    """提取不依赖连通域标记的 cheap bitmap 描述符。"""

    mask = np.asarray(bitmap, dtype=bool)
    h, w = mask.shape
    total_px = max(int(mask.size), 1)
    active_count = int(np.count_nonzero(mask))
    fill_ratio = float(active_count) / float(total_px)

    invariants = np.zeros(8, dtype=np.float32)
    if active_count > 0:
        occupied_rows = np.flatnonzero(np.any(mask, axis=1))
        occupied_cols = np.flatnonzero(np.any(mask, axis=0))
        bbox_h = float(occupied_rows[-1] - occupied_rows[0] + 1.0)
        bbox_w = float(occupied_cols[-1] - occupied_cols[0] + 1.0)
        bbox_long = max(bbox_w, bbox_h)
        bbox_short = min(bbox_w, bbox_h)
        span = max(float(h), float(w), 1.0)
        extent_area = max(bbox_w * bbox_h, 1.0)
        invariants = np.asarray(
            [
                0.0,
                fill_ratio,
                bbox_long / span,
                bbox_short / span,
                bbox_short / max(bbox_long, 1.0),
                float(active_count) / extent_area,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

    pooled = _pool_bitmap(mask, bins=10)
    return CheapDescriptor(
        invariants=invariants,
        signature_grid=pooled.reshape(-1).astype(np.float32),
        signature_proj_x=np.sum(pooled, axis=1, dtype=np.float32),
        signature_proj_y=np.sum(pooled, axis=0, dtype=np.float32),
        area_px=int(active_count),
    )


def _descriptor(owner: Any) -> GraphDescriptor:
    """从缓存中获取图描述符；若不存在则现场计算。"""

    descriptor = owner.match_cache.get("optimized_graph_descriptor")
    if descriptor is None:
        descriptor = _bitmap_descriptor(owner.clip_bitmap)
        owner.match_cache["optimized_graph_descriptor"] = descriptor
    return descriptor


def _cheap_descriptor(owner: Any) -> CheapDescriptor:
    """从缓存中获取 cheap 描述符；若不存在则现场计算。"""

    descriptor = owner.match_cache.get("optimized_cheap_descriptor")
    if descriptor is None:
        descriptor = _cheap_bitmap_descriptor(owner.clip_bitmap)
        owner.match_cache["optimized_cheap_descriptor"] = descriptor
    return descriptor


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """计算两个一维向量的余弦相似度。"""

    a = np.asarray(vec_a, dtype=np.float64).ravel()
    b = np.asarray(vec_b, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.dot(a, b) / denom)


def _signature_similarity(desc_a: GraphDescriptor, desc_b: GraphDescriptor) -> float:
    """组合 grid 与投影特征，得到 signature 相似度。"""

    return float(
        0.6 * _cosine_similarity(desc_a.signature_grid, desc_b.signature_grid)
        + 0.2 * _cosine_similarity(desc_a.signature_proj_x, desc_b.signature_proj_x)
        + 0.2 * _cosine_similarity(desc_a.signature_proj_y, desc_b.signature_proj_y)
    )


def _signature_embedding(desc: GraphDescriptor) -> np.ndarray:
    """把 signature 特征拼成归一化向量，供 coverage shortlist 排序。"""

    vector = np.concatenate(
        [
            np.asarray(desc.signature_grid, dtype=np.float32),
            0.5 * np.asarray(desc.signature_proj_x, dtype=np.float32),
            0.5 * np.asarray(desc.signature_proj_y, dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector = vector / norm
    return vector


def _normalized_matrix(rows: Sequence[np.ndarray]) -> np.ndarray:
    """把一组向量堆叠成按行 L2 归一化的矩阵。"""

    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.ndim != 2:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms <= 1e-12, 1.0, norms)
    return matrix / norms


def _coverage_subgroup_key(desc: GraphDescriptor) -> Tuple[int, int, int, int, int]:
    """根据轻量描述符生成 coverage ANN 子分组 key。"""

    invariants = np.asarray(desc.invariants, dtype=np.float32)
    fill_ratio = float(invariants[1]) if invariants.size > 1 else 0.0
    component_log = float(invariants[0]) if invariants.size else 0.0
    signature_grid = np.asarray(desc.signature_grid, dtype=np.float32)
    peak_grid = int(np.argmax(signature_grid)) if signature_grid.size and np.any(signature_grid > 0.0) else -1
    peak_x = (
        int(np.argmax(np.asarray(desc.signature_proj_x, dtype=np.float32)))
        if np.any(np.asarray(desc.signature_proj_x, dtype=np.float32) > 0.0)
        else -1
    )
    peak_y = (
        int(np.argmax(np.asarray(desc.signature_proj_y, dtype=np.float32)))
        if np.any(np.asarray(desc.signature_proj_y, dtype=np.float32) > 0.0)
        else -1
    )
    return (
        int(round(fill_ratio * 40.0)),
        int(round(component_log * 4.0)),
        peak_grid,
        peak_x,
        peak_y,
    )


def _coverage_cheap_subgroup_key(desc: CheapDescriptor) -> Tuple[int, int, int, int, int]:
    """根据 cheap 描述符生成 coverage ANN 子分组 key。"""

    invariants = np.asarray(desc.invariants, dtype=np.float32)
    fill_ratio = float(invariants[1]) if invariants.size > 1 else 0.0
    bbox_long = float(invariants[2]) if invariants.size > 2 else 0.0
    bbox_short = float(invariants[3]) if invariants.size > 3 else 0.0
    signature_grid = np.asarray(desc.signature_grid, dtype=np.float32)
    peak_grid = int(np.argmax(signature_grid)) if signature_grid.size and np.any(signature_grid > 0.0) else -1
    peak_x = (
        int(np.argmax(np.asarray(desc.signature_proj_x, dtype=np.float32)))
        if np.any(np.asarray(desc.signature_proj_x, dtype=np.float32) > 0.0)
        else -1
    )
    peak_y = (
        int(np.argmax(np.asarray(desc.signature_proj_y, dtype=np.float32)))
        if np.any(np.asarray(desc.signature_proj_y, dtype=np.float32) > 0.0)
        else -1
    )
    return (
        int(round(fill_ratio * 40.0)),
        int(round(max(bbox_long, bbox_short) * 10.0)),
        int(round(min(bbox_long, bbox_short) * 10.0)),
        peak_grid,
        (peak_x * 31 + peak_y) if peak_x >= 0 and peak_y >= 0 else -1,
    )


def _cheap_feature_vector(owner: Any) -> np.ndarray:
    """把 cheap 描述符转成 representative 重排使用的一维特征。"""

    desc = _cheap_descriptor(owner)
    return np.concatenate(
        [
            np.asarray(desc.invariants, dtype=np.float32),
            np.asarray(desc.signature_grid, dtype=np.float32),
            0.5 * np.asarray(desc.signature_proj_x, dtype=np.float32),
            0.5 * np.asarray(desc.signature_proj_y, dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def _worst_case_proxy(bitmap: np.ndarray) -> float:
    """用边缘密度和填充稀疏度估计 OPC 建模中的 worst-case 倾向。"""

    mask = np.asarray(bitmap, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        return 0.0
    edge_count = int(np.count_nonzero(mask[:, 1:] != mask[:, :-1])) + int(np.count_nonzero(mask[1:, :] != mask[:-1, :]))
    active_count = max(int(np.count_nonzero(mask)), 1)
    fill_ratio = float(active_count) / float(max(mask.size, 1))
    edge_density = float(edge_count) / float(active_count)
    return float(edge_density * (1.0 - min(fill_ratio, 1.0)))


def _rerank_export_representative(cluster_members: Sequence[MarkerRecord]) -> Tuple[MarkerRecord, Dict[str, float]]:
    """在 cluster 内选择更适合导出的 representative sample。"""

    members = list(cluster_members)
    if not members:
        raise ValueError("cluster_members must not be empty")
    if len(members) == 1:
        return members[0], {
            "score": 1.0,
            "medoid_score": 1.0,
            "worst_case_score": float(_worst_case_proxy(members[0].clip_bitmap)),
            "weight_score": float(math.log1p(max(1, int(members[0].seed_weight)))),
        }

    features = np.asarray([_cheap_feature_vector(member) for member in members], dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms <= 1e-12, 1.0, norms)
    normalized = features / norms
    centroid = np.mean(normalized, axis=0)
    centroid_norm = float(np.linalg.norm(centroid))
    if centroid_norm > 1e-12:
        centroid = centroid / centroid_norm
    medoid_scores = normalized @ centroid
    medoid_scores = (medoid_scores + 1.0) * 0.5

    worst_scores = np.asarray([_worst_case_proxy(member.clip_bitmap) for member in members], dtype=np.float32)
    worst_max = float(np.max(worst_scores)) if worst_scores.size else 0.0
    if worst_max > 1e-12:
        worst_norm = worst_scores / worst_max
    else:
        worst_norm = np.zeros_like(worst_scores)

    weight_scores = np.asarray([math.log1p(max(1, int(member.seed_weight))) for member in members], dtype=np.float32)
    weight_max = float(np.max(weight_scores)) if weight_scores.size else 0.0
    if weight_max > 1e-12:
        weight_norm = weight_scores / weight_max
    else:
        weight_norm = np.zeros_like(weight_scores)

    scores = 0.45 * medoid_scores + 0.35 * worst_norm + 0.20 * weight_norm
    best_idx = int(np.argmax(scores))
    return members[best_idx], {
        "score": float(scores[best_idx]),
        "medoid_score": float(medoid_scores[best_idx]),
        "worst_case_score": float(worst_scores[best_idx]),
        "weight_score": float(weight_scores[best_idx]),
    }


def _pre_raster_fingerprint(
    layout_index: LayoutIndex,
    expanded_bbox: Tuple[float, float, float, float],
    *,
    quant_um: float,
) -> str:
    """用 expanded window 内的相对元素布局生成栅格化前指纹。"""

    q = max(float(quant_um), 1e-9)
    ex0, ey0, ex1, ey1 = (float(value) for value in expanded_bbox)
    parts: List[Tuple[int, int, int, int, int, int]] = []
    for elem_id in layout_index.spatial_index.intersection((ex0, ey0, ex1, ey1)):
        idx = int(elem_id)
        if not (
            float(layout_index.bbox_x1[idx]) > ex0
            and float(layout_index.bbox_x0[idx]) < ex1
            and float(layout_index.bbox_y1[idx]) > ey0
            and float(layout_index.bbox_y0[idx]) < ey1
        ):
            continue
        item = layout_index.indexed_elements[idx]
        bx0, by0, bx1, by1 = (float(value) for value in item["bbox"])
        parts.append(
            (
                int(round((bx0 - ex0) / q)),
                int(round((by0 - ey0) / q)),
                int(round((bx1 - ex0) / q)),
                int(round((by1 - ey0) / q)),
                int(item["layer"]),
                int(item["datatype"]),
            )
        )
    if not parts:
        return "empty"
    parts.sort()
    digest = hashlib.sha1()
    for part in parts:
        digest.update(("%d,%d,%d,%d,%d,%d;" % part).encode("ascii"))
    return digest.hexdigest()


def _invariant_distance(desc_a: GraphDescriptor, desc_b: GraphDescriptor) -> Tuple[float, bool]:
    """计算 invariant 相对误差距离，并标记是否命中关键维度大偏差。"""

    a = np.asarray(desc_a.invariants, dtype=np.float64)
    b = np.asarray(desc_b.invariants, dtype=np.float64)
    floors = np.asarray([0.25, 0.02, 0.02, 0.02, 0.03, 0.03, 0.02, 0.02], dtype=np.float64)
    weights = np.asarray([0.08, 0.24, 0.10, 0.08, 0.18, 0.14, 0.10, 0.08], dtype=np.float64)
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), floors)
    errs = np.minimum(np.abs(a - b) / denom, 1.0)
    critical = bool(errs[1] > 0.45 or errs[4] > 0.45 or errs[5] > 0.45)
    return float(np.dot(errs, weights)), critical


def _topology_distance(desc_a: GraphDescriptor, desc_b: GraphDescriptor) -> float:
    """计算 topology 特征之间的欧氏距离。"""

    a = np.asarray(desc_a.topology, dtype=np.float64)
    b = np.asarray(desc_b.topology, dtype=np.float64)
    return float(np.linalg.norm(a - b))


def _graph_prefilter_passes(candidate: Any, target: Any, *, strict: bool) -> Tuple[bool, str]:
    """执行 invariant / topology / signature 的统一图特征预筛选。"""

    desc_a = _descriptor(candidate)
    desc_b = _descriptor(target)
    invariant_limit = STRICT_INVARIANT_LIMIT if strict else GRAPH_INVARIANT_LIMIT
    topology_threshold = STRICT_TOPOLOGY_THRESHOLD if strict else GRAPH_TOPOLOGY_THRESHOLD
    signature_threshold = STRICT_SIGNATURE_THRESHOLD if strict else GRAPH_SIGNATURE_THRESHOLD

    invariant_score, critical = _invariant_distance(desc_a, desc_b)
    if critical or invariant_score > invariant_limit:
        return False, "invariant"
    if _topology_distance(desc_a, desc_b) > topology_threshold:
        return False, "topology"
    if _signature_similarity(desc_a, desc_b) < signature_threshold:
        return False, "signature"
    return True, "pass"


def _coverage_structure(tol_px: int) -> np.ndarray:
    """返回 coverage ECC 使用的方形结构元，并按容差像素缓存。"""

    tol = int(tol_px)
    cached = _COVERAGE_STRUCTURE_CACHE.get(tol)
    if cached is None:
        cached = np.ones((2 * tol + 1, 2 * tol + 1), dtype=bool)
        _COVERAGE_STRUCTURE_CACHE[tol] = cached
    return cached


def _init_coverage_geometry_cache(owner: Any) -> Dict[str, Any]:
    """初始化 coverage 几何缓存，只生成 packed bitmap 和面积。"""

    bitmap = np.ascontiguousarray(owner.clip_bitmap.astype(bool, copy=False))
    return {
        "bitmap": bitmap,
        "packed": np.packbits(bitmap.reshape(-1)),
        "area_px": int(np.count_nonzero(bitmap)),
    }


def _extend_coverage_dilated_cache(cache: Dict[str, Any], tol_px: int) -> None:
    """把 coverage 几何缓存扩展到膨胀层，不计算 donut。"""

    if int(tol_px) <= 0 or "packed_dilated" in cache:
        return
    structure = _coverage_structure(int(tol_px))
    dilated = ndimage.binary_dilation(cache["bitmap"], structure=structure)
    cache["dilated"] = np.ascontiguousarray(dilated, dtype=bool)
    cache["packed_dilated"] = np.packbits(cache["dilated"].reshape(-1))
    cache["dilated_area_px"] = int(np.count_nonzero(cache["dilated"]))


def _extend_coverage_donut_cache(cache: Dict[str, Any], tol_px: int) -> None:
    """把 coverage 几何缓存扩展到 donut 层，供最终 ECC overlap 使用。"""

    if int(tol_px) <= 0 or "packed_donut" in cache:
        return
    _extend_coverage_dilated_cache(cache, int(tol_px))
    structure = _coverage_structure(int(tol_px))
    eroded = ndimage.binary_erosion(cache["bitmap"], structure=structure, border_value=0)
    donut = cache["dilated"] & ~eroded
    cache["packed_donut"] = np.packbits(np.ascontiguousarray(donut, dtype=bool).reshape(-1))
    cache["donut_area_px"] = int(np.count_nonzero(donut))


def _exact_cosine_topk_labels(vectors: np.ndarray, k: int) -> np.ndarray:
    """对中小 subgroup 执行精确 cosine top-k，避免 HNSW 建索引开销。"""

    group_vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    group_count = int(group_vectors.shape[0])
    if group_count == 0:
        return np.empty((0, 0), dtype=np.int64)
    k = min(int(k), group_count)
    norms = np.linalg.norm(group_vectors, axis=1, keepdims=True)
    normalized = group_vectors / np.maximum(norms, 1e-6)
    similarities = normalized @ normalized.T
    if group_count <= k:
        labels = np.tile(np.arange(group_count, dtype=np.int64), (group_count, 1))
    else:
        labels = np.argpartition(-similarities, kth=k - 1, axis=1)[:, :k].astype(np.int64, copy=False)
    row_indices = np.arange(group_count)[:, None]
    order = np.argsort(-similarities[row_indices, labels], axis=1)
    return labels[row_indices, order]


def _ecc_cache(owner: Any, tol_px: int) -> Dict[str, Any]:
    """缓存 ECC 匹配所需的膨胀/环带中间结果。"""

    key = f"optimized_ecc_tol_{int(tol_px)}"
    cache = owner.match_cache.get(key)
    if cache is not None:
        return cache

    bitmap = np.ascontiguousarray(owner.clip_bitmap.astype(bool, copy=False))
    cache = {
        "bitmap": bitmap,
        "area": int(np.count_nonzero(bitmap)),
    }
    if tol_px > 0:
        structure = _coverage_structure(int(tol_px))
        dilated = ndimage.binary_dilation(bitmap, structure=structure)
        eroded = ndimage.binary_erosion(bitmap, structure=structure, border_value=0)
        cache["dilated"] = np.ascontiguousarray(dilated, dtype=bool)
        cache["donut"] = np.ascontiguousarray(dilated & ~eroded, dtype=bool)
        cache["donut_area"] = int(np.count_nonzero(cache["donut"]))
    owner.match_cache[key] = cache
    return cache


def _ecc_match_cached(candidate: CandidateClip, target: MarkerRecord, edge_tolerance_um: float, pixel_size_um: float) -> bool:
    """使用缓存后的位图形态学结果执行 ECC 几何匹配。"""

    if candidate.clip_bitmap.shape != target.clip_bitmap.shape:
        return False
    if not candidate.clip_bitmap.any() and not target.clip_bitmap.any():
        return True
    if not candidate.clip_bitmap.any() or not target.clip_bitmap.any():
        return False

    tol_px = max(0, int(math.ceil(float(edge_tolerance_um) / max(float(pixel_size_um), 1e-12) - 1e-12)))
    if tol_px <= 0:
        return bool(np.array_equal(candidate.clip_bitmap, target.clip_bitmap))

    cand = _ecc_cache(candidate, tol_px)
    tgt = _ecc_cache(target, tol_px)
    cand_area = max(float(cand["area"]), 1.0)
    tgt_area = max(float(tgt["area"]), 1.0)
    residual_cand = np.count_nonzero(cand["bitmap"] & ~tgt["dilated"]) / cand_area
    residual_tgt = np.count_nonzero(tgt["bitmap"] & ~cand["dilated"]) / tgt_area
    if residual_cand > ECC_RESIDUAL_RATIO or residual_tgt > ECC_RESIDUAL_RATIO:
        return False
    if int(cand["donut_area"]) == 0 or int(tgt["donut_area"]) == 0:
        return True
    overlap = int(np.count_nonzero(cand["donut"] & tgt["donut"]))
    denom = max(min(int(cand["donut_area"]), int(tgt["donut_area"])), 1)
    return float(overlap / denom) >= ECC_DONUT_OVERLAP_RATIO


class OptimizedMainlineRunner(MainlineRunner):
    """uniform-grid optimized v1 主运行器，负责串起完整聚类流程。"""

    def __init__(self, *, config: Dict[str, Any], temp_dir: Path, layer_processor: Any | None = None):
        """初始化 runner，并把 optimized 配置映射到主线 backend。"""

        clean_config = {
            "apply_layer_operations": bool(config.get("apply_layer_operations", False)),
            "clip_size_um": float(config.get("clip_size_um", 1.35)),
            "hotspot_layer": DUMMY_MARKER_LAYER,
            "matching_mode": str(config.get("geometry_match_mode", "ecc")),
            "solver": "greedy",
            "geometry_mode": "exact",
            "pixel_size_nm": int(config.get("pixel_size_nm", DEFAULT_PIXEL_SIZE_NM)),
            "area_match_ratio": float(config.get("area_match_ratio", 0.96)),
            "edge_tolerance_um": float(config.get("edge_tolerance_um", 0.02)),
            "clip_shift_directions": "left,right,up,down",
            "clip_shift_boundary_tolerance_um": float(config.get("edge_tolerance_um", 0.02)),
        }
        super().__init__(config=clean_config, temp_dir=temp_dir, layer_processor=layer_processor)
        self.prefilter_stats = _empty_prefilter_stats()
        self.final_verification_stats = _empty_verification_stats()
        self.final_verification_detail_seconds = _empty_verification_detail_seconds()
        self.coverage_detail_seconds = _empty_coverage_detail_seconds()
        self.coverage_debug_stats = _empty_coverage_debug_stats()
        self.result_detail_seconds = _empty_result_detail_seconds()
        self._base_candidate_by_exact_id: Dict[int, CandidateClip] = {}
        self._seed_stats_by_source: Dict[str, Dict[str, int]] = {}
        self.materialize_outputs = bool(config.get("materialize_outputs", False))

    def _log(self, message: str) -> None:
        """统一中文过程日志输出入口。"""

        print(message)

    def _layer_operations(self) -> List[Dict[str, str]]:
        """返回当前启用的 layer operation 规则列表。"""

        return _layer_operation_payload(self.layer_processor)

    def _effective_layer_summary(self) -> Dict[str, List[str]]:
        """返回当前 layer-op 过滤后真正参与聚类的层摘要。"""

        return {
            "effective_clustering_layers": [
                f"{int(layer)}/{int(datatype)}" for layer, datatype in getattr(self, "effective_pattern_layers", [])
            ],
            "excluded_helper_layers": [
                f"{int(layer)}/{int(datatype)}" for layer, datatype in getattr(self, "excluded_helper_layers", [])
            ],
        }

    def _seed_raster_spec(self, candidate: GridSeedCandidate) -> Dict[str, Any]:
        """根据 grid seed bbox 计算父类 marker raster 所需窗口参数。"""

        marker_bbox = tuple(float(value) for value in candidate.seed_bbox)
        marker_center = _bbox_center(marker_bbox)
        return _raster_window_spec(marker_bbox, marker_center, self.clip_size_um, self.pixel_size_um)

    def _clone_cached_record(
        self,
        cached: MarkerRecord,
        filepath: Path,
        marker_index: int,
        candidate: GridSeedCandidate,
    ) -> MarkerRecord:
        """复用已栅格化 record 的位图和哈希，并替换当前 seed 的身份字段。"""

        marker_bbox = tuple(float(value) for value in candidate.seed_bbox)
        marker_center = _bbox_center(marker_bbox)
        raster_spec = self._seed_raster_spec(candidate)
        return replace(
            cached,
            marker_id=f"{filepath.stem}__seed_{int(marker_index):06d}",
            source_path=str(filepath),
            source_name=filepath.name,
            marker_bbox=marker_bbox,
            marker_center=marker_center,
            clip_bbox=raster_spec["clip_bbox"],
            expanded_bbox=raster_spec["expanded_bbox"],
            clip_bbox_q=raster_spec["clip_bbox_q"],
            expanded_bbox_q=raster_spec["expanded_bbox_q"],
            marker_bbox_q=raster_spec["marker_bbox_q"],
            shift_limits_px=raster_spec["shift_limits_px"],
            seed_weight=int(candidate.bucket_weight),
            exact_cluster_id=-1,
            match_cache=dict(cached.match_cache),
        )

    def _apply_seed_metadata(self, record: MarkerRecord, filepath: Path, marker_index: int, candidate: GridSeedCandidate) -> None:
        """把当前 grid seed 的样本身份和 metadata 写入 record。"""

        record.marker_id = f"{filepath.stem}__seed_{int(marker_index):06d}"
        record.seed_weight = int(candidate.bucket_weight)
        record.exact_cluster_id = -1
        record.match_cache["auto_seed"] = {
            "seed_bbox": list(candidate.seed_bbox),
            "grid_ix": int(candidate.grid_ix),
            "grid_iy": int(candidate.grid_iy),
            "grid_cell_bbox": list(candidate.seed_bbox),
            "bucket_weight": int(candidate.bucket_weight),
        }

    def _collect_marker_records_for_file(self, filepath: Path) -> List[MarkerRecord]:
        """对单个 OAS 文件执行 uniform grid seed、分桶并构建窗口记录。"""

        layout_index = self._prepare_layout(filepath)
        if self.apply_layer_operations:
            self._log(
                f"文件 {filepath.name}: 有效聚类层 {len(layout_index.effective_pattern_layers)} 个, "
                f"排除 helper 层 {len(layout_index.excluded_helper_layers)} 个"
            )
        bucketed_candidates, seed_stats = _build_uniform_grid_seed_candidates(
            layout_index,
            clip_size_um=float(self.clip_size_um),
        )
        self._seed_stats_by_source[str(filepath)] = dict(seed_stats)
        self._log(
            f"文件 {filepath.name}: raw grid seed {seed_stats['grid_seed_count']}, "
            f"分桶后 seed {seed_stats['bucketed_seed_count']}, pattern 元素 {len(layout_index.indexed_elements)}"
        )

        records: List[MarkerRecord] = []
        pre_raster_cache: Dict[str, MarkerRecord] = {}
        exact_bitmap_cache: Dict[Tuple[str, str], MarkerRecord] = {}
        cache_stats = {
            "pre_raster_cache_hit": 0,
            "pre_raster_cache_miss": 0,
            "exact_bitmap_cache_hit": 0,
            "exact_bitmap_cache_miss": 0,
        }
        for marker_index, candidate in enumerate(bucketed_candidates):
            raster_spec = self._seed_raster_spec(candidate)
            pre_key = _pre_raster_fingerprint(
                layout_index,
                tuple(float(value) for value in raster_spec["expanded_bbox"]),
                quant_um=float(self.pixel_size_um) * float(PRE_RASTER_FINGERPRINT_QUANT_PX),
            )
            cached_pre_record = pre_raster_cache.get(pre_key)
            if cached_pre_record is not None:
                cache_stats["pre_raster_cache_hit"] += 1
                record = self._clone_cached_record(cached_pre_record, filepath, marker_index, candidate)
                self._apply_seed_metadata(record, filepath, marker_index, candidate)
                records.append(record)
                continue
            cache_stats["pre_raster_cache_miss"] += 1

            marker_poly = gdstk.rectangle(
                (float(candidate.seed_bbox[0]), float(candidate.seed_bbox[1])),
                (float(candidate.seed_bbox[2]), float(candidate.seed_bbox[3])),
                layer=0,
                datatype=0,
            )
            record = self._build_marker_record(filepath, marker_index, marker_poly, layout_index)
            if record is None:
                continue
            self._apply_seed_metadata(record, filepath, marker_index, candidate)
            exact_key = (str(record.clip_hash), str(record.expanded_hash))
            cached = exact_bitmap_cache.get(exact_key)
            if cached is None:
                cache_stats["exact_bitmap_cache_miss"] += 1
                exact_bitmap_cache[exact_key] = record
            else:
                cache_stats["exact_bitmap_cache_hit"] += 1
                record.clip_bitmap = cached.clip_bitmap
                record.expanded_bitmap = cached.expanded_bitmap
                if "optimized_graph_descriptor" in cached.match_cache:
                    record.match_cache["optimized_graph_descriptor"] = cached.match_cache["optimized_graph_descriptor"]
                if "optimized_cheap_descriptor" in cached.match_cache:
                    record.match_cache["optimized_cheap_descriptor"] = cached.match_cache["optimized_cheap_descriptor"]
            pre_raster_cache[pre_key] = record
            records.append(record)

        self._seed_stats_by_source[str(filepath)].update(cache_stats)
        self._log(f"文件 {filepath.name}: 生成 grid seed 窗口 {len(records)} 个")
        self._log(
            f"文件 {filepath.name}: pre-raster cache hit/miss "
            f"{cache_stats['pre_raster_cache_hit']}/{cache_stats['pre_raster_cache_miss']}, "
            f"exact bitmap cache hit/miss "
            f"{cache_stats['exact_bitmap_cache_hit']}/{cache_stats['exact_bitmap_cache_miss']}"
        )
        return records

    def run(self, input_path: str) -> Dict[str, Any]:
        """执行 uniform-grid optimized v1 主流程并返回最终结果字典。"""

        started_at = time.perf_counter()
        input_files = self._discover_input_files(input_path)
        if not input_files:
            raise ValueError("No .oas files found")

        self._seed_stats_by_source = {}
        self._log(f"发现输入 OAS 文件数: {len(input_files)}")
        self._log("开始收集 uniform grid seed 窗口...")
        marker_started = time.perf_counter()
        marker_records: List[MarkerRecord] = []
        for filepath in input_files:
            if self.apply_layer_operations:
                self._log(f" 对文件 {filepath.name} 应用层操作...")
            marker_records.extend(self._collect_marker_records_for_file(filepath))
        marker_elapsed = time.perf_counter() - marker_started

        if not marker_records:
            raise ValueError("No uniform grid seeds produced from layout geometry")
        total_grid_seeds = sum(int(stats.get("grid_seed_count", 0)) for stats in self._seed_stats_by_source.values())
        total_bucketed = sum(int(stats.get("bucketed_seed_count", 0)) for stats in self._seed_stats_by_source.values())
        self._log(
            f"uniform grid seed 收集完成: raw grid seed {total_grid_seeds}, "
            f"分桶后 seed {total_bucketed}, 样本 {len(marker_records)}"
        )

        self._log("开始 exact hash 聚合...")
        dedup_started = time.perf_counter()
        exact_clusters = self._group_exact_clusters(marker_records)
        dedup_elapsed = time.perf_counter() - dedup_started
        self._log(f"exact hash 聚合完成: {len(marker_records)} -> {len(exact_clusters)}")

        self._log("开始生成 systematic shift candidates...")
        candidate_started = time.perf_counter()
        all_candidates: List[CandidateClip] = []
        for exact_cluster in exact_clusters:
            all_candidates.extend(self._generate_candidates_for_cluster(exact_cluster))
        candidate_elapsed = time.perf_counter() - candidate_started
        self._log(f"candidate 生成完成: {len(all_candidates)} 个")

        self._log("开始构建 verified coverage edges...")
        coverage_started = time.perf_counter()
        self.prefilter_stats = _empty_prefilter_stats()
        self._evaluate_candidate_coverage(all_candidates, exact_clusters)
        coverage_elapsed = time.perf_counter() - coverage_started
        self._log(f"coverage 构建完成: {self.prefilter_stats}")

        self._log("开始 greedy set cover...")
        cover_started = time.perf_counter()
        selected_candidates = self._greedy_cover(all_candidates, exact_clusters)
        cover_elapsed = time.perf_counter() - cover_started
        selected_direction_counts = dict(Counter(str(candidate.shift_direction) for candidate in selected_candidates))
        self._log(
            f"set cover 完成: selected candidates={len(selected_candidates)}, "
            f"方向分布={selected_direction_counts}"
        )

        if self.materialize_outputs:
            self._log("开始物化样本、representative，并执行 final verification...")
        else:
            self._log("开始构建结果并执行 final verification；未指定 review 目录，跳过 clip 物化...")
        runtime_summary = {
            "collect_markers": round(marker_elapsed, 6),
            "exact_cluster": round(dedup_elapsed, 6),
            "candidate_generation": round(candidate_elapsed, 6),
            "coverage_eval": round(coverage_elapsed, 6),
            "set_cover": round(cover_elapsed, 6),
        }
        result_started = time.perf_counter()
        result = self._build_results(
            marker_records,
            exact_clusters,
            selected_candidates,
            "greedy",
            runtime_summary=runtime_summary,
            candidate_count=int(len(all_candidates)),
        )
        runtime_summary["result_build"] = round(time.perf_counter() - result_started, 6)
        runtime_summary["total"] = round(time.perf_counter() - started_at, 6)
        result["result_summary"]["timing_seconds"] = dict(runtime_summary)
        self._log(f"final verification 完成: {self.final_verification_stats}")
        self._log(f"final cluster 方向分布: {result.get('final_cluster_direction_counts', {})}")
        if self.materialize_outputs:
            self._log(f"样本目录: {self.temp_dir / 'samples'}")
            self._log(f"representative 目录: {self.temp_dir / 'representatives'}")
        return result

    def _build_marker_record(self, filepath: Path, marker_index: int, marker_poly: Any, layout_index: Any) -> MarkerRecord | None:
        """构建 marker record，图描述符延迟到 coverage / verification 阶段计算。"""

        return super()._build_marker_record(filepath, marker_index, marker_poly, layout_index)

    def _build_candidate_clip(
        self,
        cluster: ExactCluster,
        clip_bbox: Tuple[float, float, float, float],
        clip_bbox_q: Tuple[int, int, int, int],
        bitmap: np.ndarray,
        shift_direction: str,
        shift_distance_um: float,
        candidate_index: int,
    ) -> CandidateClip:
        """构建候选窗口对象，并保留 coverage 初值。"""

        candidate = super()._build_candidate_clip(
            cluster,
            clip_bbox,
            clip_bbox_q,
            bitmap,
            shift_direction,
            shift_distance_um,
            candidate_index,
        )
        candidate.coverage = {int(cluster.exact_cluster_id)} if str(shift_direction) == "base" else set()
        return candidate

    def _generate_candidates_for_cluster(self, cluster: ExactCluster) -> List[CandidateClip]:
        """为一个 exact cluster 生成 base 与四方向 systematic shift candidates。"""

        candidates = super()._generate_candidates_for_cluster(cluster)
        base = next((candidate for candidate in candidates if candidate.shift_direction == "base"), None)
        if base is not None:
            self._base_candidate_by_exact_id[int(cluster.exact_cluster_id)] = base
        return candidates

    def _record_exact_hash_match(self, source_candidate: CandidateClip, target_candidate: CandidateClip) -> None:
        """记录 exact hash 直通匹配次数。"""

        del source_candidate, target_candidate
        self.prefilter_stats["exact_hash_pass"] += 1

    def _record_geometry_result(
        self,
        source_candidate: CandidateClip,
        target_candidate: CandidateClip,
        matched: bool,
    ) -> None:
        """记录 ACC/ECC 几何判定结果。"""

        del source_candidate, target_candidate
        if matched:
            self.prefilter_stats["geometry_pass"] += 1
        else:
            self.prefilter_stats["geometry_reject"] += 1

    def _build_candidate_match_bundles(
        self,
        candidates: Sequence[CandidateClip],
        tol_px: int,
    ) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """按严格 bitmap key 构建轻量 coverage bundle，不预先生成 ECC cache。"""

        del tol_px
        grouped: Dict[Tuple[int, int, bytes], Dict[str, Any]] = {}
        for candidate in candidates:
            key = _bitmap_exact_key(candidate.clip_bitmap)
            bucket = grouped.get(key)
            if bucket is None:
                bucket = {"representative": candidate, "candidates": [], "origin_ids": set()}
                grouped[key] = bucket
            bucket["candidates"].append(candidate)
            bucket["origin_ids"].add(int(candidate.origin_exact_cluster_id))

        bundles: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for bucket in grouped.values():
            representative_candidate = bucket["representative"]
            shape = tuple(int(value) for value in representative_candidate.clip_bitmap.shape)
            bundle = bundles.setdefault(
                shape,
                {
                    "areas": [],
                    "hashes": [],
                    "origin_ids": [],
                    "candidate_groups": [],
                    "representatives": [],
                    "clip_pixels": int(shape[0]) * int(shape[1]),
                    "geometry_cache_by_idx": {},
                },
            )
            bundle["areas"].append(int(np.count_nonzero(representative_candidate.clip_bitmap)))
            bundle["hashes"].append(str(representative_candidate.clip_hash))
            bundle["origin_ids"].append(tuple(sorted(bucket["origin_ids"])))
            bundle["candidate_groups"].append(tuple(bucket["candidates"]))
            bundle["representatives"].append(representative_candidate)

        for bundle in bundles.values():
            bundle["areas"] = np.asarray(bundle["areas"], dtype=np.int64)
            bundle["hashes_np"] = np.asarray(bundle["hashes"])
            hash_to_indices: Dict[str, List[int]] = {}
            for idx, clip_hash in enumerate(bundle["hashes"]):
                hash_to_indices.setdefault(clip_hash, []).append(idx)
            bundle["hash_to_indices"] = hash_to_indices
        return bundles

    def _bundle_geometry_cache(
        self,
        bundle: Dict[str, Any],
        group_idx: int,
        tol_px: int,
        level: str = "packed",
    ) -> Dict[str, Any]:
        """按需获取某个 bundle group 的分层几何缓存。"""

        cache_by_idx = bundle.setdefault("geometry_cache_by_idx", {})
        idx = int(group_idx)
        cached = cache_by_idx.get(idx)
        if cached is None:
            started = time.perf_counter()
            cached = _init_coverage_geometry_cache(bundle["representatives"][idx])
            cache_by_idx[idx] = cached
            self.coverage_detail_seconds["geometry_cache"] += time.perf_counter() - started
            self.coverage_debug_stats["geometry_cache_group_count"] += 1

        if level in {"dilated", "donut"} and int(tol_px) > 0 and "packed_dilated" not in cached:
            started = time.perf_counter()
            _extend_coverage_dilated_cache(cached, int(tol_px))
            self.coverage_detail_seconds["geometry_cache"] += time.perf_counter() - started
            self.coverage_debug_stats["geometry_dilated_cache_group_count"] += 1

        if level == "donut" and int(tol_px) > 0 and "packed_donut" not in cached:
            started = time.perf_counter()
            _extend_coverage_donut_cache(cached, int(tol_px))
            self.coverage_detail_seconds["geometry_cache"] += time.perf_counter() - started
            self.coverage_debug_stats["geometry_donut_cache_group_count"] += 1
        return cached

    def _bundle_geometry_level_for_key(self, key: str) -> str:
        """根据缓存字段名推导所需的 coverage 几何缓存层级。"""

        if key in {"packed_donut", "donut_area_px"}:
            return "donut"
        if key in {"packed_dilated", "dilated_area_px"}:
            return "dilated"
        return "packed"

    def _bundle_geometry_matrix(
        self,
        bundle: Dict[str, Any],
        indices: np.ndarray,
        tol_px: int,
        key: str,
    ) -> np.ndarray:
        """按需把若干 group 的几何缓存字段堆叠成矩阵。"""

        level = self._bundle_geometry_level_for_key(str(key))
        rows = [
            np.asarray(self._bundle_geometry_cache(bundle, int(group_idx), int(tol_px), level=level)[key], dtype=np.uint8)
            for group_idx in np.asarray(indices, dtype=np.int64).tolist()
        ]
        if not rows:
            return np.empty((0, 0), dtype=np.uint8)
        return np.stack(rows, axis=0)

    def _bundle_geometry_values(
        self,
        bundle: Dict[str, Any],
        indices: np.ndarray,
        tol_px: int,
        key: str,
    ) -> np.ndarray:
        """按需读取若干 group 的几何缓存标量字段。"""

        level = self._bundle_geometry_level_for_key(str(key))
        values = [
            int(self._bundle_geometry_cache(bundle, int(group_idx), int(tol_px), level=level)[key])
            for group_idx in np.asarray(indices, dtype=np.int64).tolist()
        ]
        return np.asarray(values, dtype=np.int64)

    def _build_bundle_shortlist_index(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        """为一个 bitmap-shape bundle 预构建 cheap 分组 ANN shortlist。"""

        descriptors = [_cheap_descriptor(representative) for representative in bundle["representatives"]]
        signature_vectors = np.asarray([_signature_embedding(desc) for desc in descriptors], dtype=np.float32)
        group_count = int(signature_vectors.shape[0])
        neighbor_labels = np.full(
            (group_count, min(int(COVERAGE_SHORTLIST_MAX_TARGETS) + 1, max(group_count, 1))),
            -1,
            dtype=np.int64,
        )

        subgroups: Dict[Tuple[int, int, int, int, int], List[int]] = {}
        for idx, desc in enumerate(descriptors):
            subgroups.setdefault(_coverage_cheap_subgroup_key(desc), []).append(int(idx))

        self.coverage_debug_stats["shortlist_subgroup_count"] += int(len(subgroups))
        self.coverage_debug_stats["shortlist_max_subgroup_size"] = max(
            int(self.coverage_debug_stats.get("shortlist_max_subgroup_size", 0)),
            max((len(indices) for indices in subgroups.values()), default=0),
        )

        for indices in subgroups.values():
            if len(indices) <= 1:
                continue
            group_indices = np.asarray(indices, dtype=np.int64)
            k = min(int(COVERAGE_SHORTLIST_MAX_TARGETS) + 1, int(group_indices.size))
            if group_indices.size <= k:
                for global_idx in group_indices.tolist():
                    neighbor_labels[int(global_idx), :k] = group_indices[:k]
                continue

            group_vectors = np.ascontiguousarray(signature_vectors[group_indices], dtype=np.float32)
            if int(group_indices.size) <= int(COVERAGE_EXACT_SHORTLIST_MAX_GROUPS):
                self.coverage_debug_stats["shortlist_exact_subgroup_count"] += 1
                local_labels = _exact_cosine_topk_labels(group_vectors, k)
            else:
                self.coverage_debug_stats["shortlist_hnsw_subgroup_count"] += 1
                index = hnswlib.Index(space="cosine", dim=int(group_vectors.shape[1]))
                index.init_index(max_elements=int(group_indices.size), ef_construction=max(64, k * 2), M=12)
                index.add_items(group_vectors, np.arange(int(group_indices.size), dtype=np.int64))
                index.set_ef(max(64, k * 2))
                local_labels, _ = index.knn_query(group_vectors, k=k)
            mapped_labels = group_indices[np.asarray(local_labels, dtype=np.int64)]
            for row_idx, global_idx in enumerate(group_indices.tolist()):
                neighbor_labels[int(global_idx), :k] = mapped_labels[row_idx, :k]

        return {
            "neighbor_labels": np.asarray(neighbor_labels, dtype=np.int64),
            "cheap_invariant_matrix": np.asarray([desc.invariants for desc in descriptors], dtype=np.float32),
            "cheap_signature_grid_matrix": _normalized_matrix([desc.signature_grid for desc in descriptors]),
            "cheap_signature_proj_x_matrix": _normalized_matrix([desc.signature_proj_x for desc in descriptors]),
            "cheap_signature_proj_y_matrix": _normalized_matrix([desc.signature_proj_y for desc in descriptors]),
        }

    def _shortlist_target_indices(
        self,
        bundle: Dict[str, Any],
        shortlist_index: Dict[str, Any],
        source_idx: int,
    ) -> np.ndarray:
        """返回某个 source group 的 ANN shortlist 目标组索引。"""

        del bundle
        labels = np.asarray(shortlist_index["neighbor_labels"][int(source_idx)], dtype=np.int64)
        if labels.size == 0:
            return np.asarray([], dtype=np.int64)
        return labels[(labels >= 0) & (labels != int(source_idx))]

    def _batch_prefilter(
        self,
        bundle: Dict[str, Any],
        shortlist_index: Dict[str, Any],
        source_idx: int,
        target_indices: np.ndarray,
    ) -> np.ndarray:
        """对一个 source 的 shortlist targets 执行 cheap prefilter。"""

        if target_indices.size == 0:
            return target_indices

        cheap_inv_mat = np.asarray(shortlist_index["cheap_invariant_matrix"], dtype=np.float32)
        source_cheap = cheap_inv_mat[int(source_idx)]
        target_cheap = cheap_inv_mat[target_indices]
        cheap_floors = np.asarray([0.02, 0.03, 0.03], dtype=np.float32)
        cheap_source = source_cheap[[1, 4, 5]]
        cheap_target = target_cheap[:, [1, 4, 5]]
        cheap_denom = np.maximum(np.maximum(np.abs(cheap_source)[None, :], np.abs(cheap_target)), cheap_floors[None, :])
        cheap_errs = np.abs(cheap_target - cheap_source[None, :]) / cheap_denom
        cheap_ok = np.all(cheap_errs <= 0.45, axis=1)
        self.prefilter_stats["cheap_reject"] += int(np.count_nonzero(~cheap_ok))
        target_indices = target_indices[cheap_ok]
        if target_indices.size == 0:
            return target_indices

        return target_indices

    def _evaluate_candidate_coverage(self, candidates: Sequence[CandidateClip], exact_clusters: Sequence[ExactCluster]) -> None:
        """用 grid-aware shortlist + ACC/ECC 构建 candidate 覆盖关系。"""

        del exact_clusters
        tol_px = max(0, int(math.ceil(float(self.edge_tolerance_um) / max(float(self.pixel_size_um), 1e-12) - 1e-12)))
        self.coverage_detail_seconds = _empty_coverage_detail_seconds()
        self.coverage_debug_stats = _empty_coverage_debug_stats()

        light_bundle_started = time.perf_counter()
        bundles = self._build_candidate_match_bundles(candidates, tol_px)
        self.coverage_detail_seconds["light_bundle_build"] += time.perf_counter() - light_bundle_started
        bundle_sizes = [len(bundle["candidate_groups"]) for bundle in bundles.values()]
        self.coverage_debug_stats["bundle_count"] = int(len(bundles))
        self.coverage_debug_stats["max_bundle_group_count"] = int(max(bundle_sizes, default=0))
        self.coverage_debug_stats["candidate_group_count"] = int(sum(bundle_sizes))
        self._log(
            "coverage 轻量 bundle: "
            f"bundle={self.coverage_debug_stats['bundle_count']}, "
            f"最大 group={self.coverage_debug_stats['max_bundle_group_count']}, "
            f"总 group={self.coverage_debug_stats['candidate_group_count']}"
        )

        for candidate in candidates:
            candidate.coverage = set(candidate.coverage)

        for bundle in bundles.values():
            shortlist_started = time.perf_counter()
            shortlist_index = self._build_bundle_shortlist_index(bundle)
            self.coverage_detail_seconds["shortlist_index"] += time.perf_counter() - shortlist_started
            group_count = len(bundle["candidate_groups"])
            coverage_by_group: List[set[int]] = []
            for grouped_candidates in bundle["candidate_groups"]:
                group_coverage: set[int] = set()
                for candidate in grouped_candidates:
                    group_coverage.update(int(value) for value in candidate.coverage)
                coverage_by_group.append(group_coverage)

            for same_hash_indices in bundle["hash_to_indices"].values():
                if len(same_hash_indices) <= 1:
                    continue
                hash_origin_ids: set[int] = set()
                for idx in same_hash_indices:
                    hash_origin_ids.update(bundle["origin_ids"][idx])
                for idx in same_hash_indices:
                    coverage_by_group[int(idx)].update(hash_origin_ids)
                for left in range(len(same_hash_indices) - 1):
                    source_idx = int(same_hash_indices[left])
                    source_candidate = bundle["representatives"][source_idx]
                    for right in range(left + 1, len(same_hash_indices)):
                        target_idx = int(same_hash_indices[right])
                        target_candidate = bundle["representatives"][target_idx]
                        self._record_exact_hash_match(source_candidate, target_candidate)

            ratio_limit = max(0.0, 1.0 - float(self.area_match_ratio))
            compared_pairs: set[Tuple[int, int]] = set()
            for source_idx in range(max(0, group_count - 1)):
                source_candidate = bundle["representatives"][source_idx]
                target_indices = self._shortlist_target_indices(bundle, shortlist_index, source_idx)
                if target_indices.size == 0:
                    continue

                target_indices = target_indices[bundle["hashes_np"][target_indices] != bundle["hashes_np"][source_idx]]
                if target_indices.size == 0:
                    continue

                fresh_targets: List[int] = []
                for target_idx in target_indices.tolist():
                    pair_key = (min(int(source_idx), int(target_idx)), max(int(source_idx), int(target_idx)))
                    if pair_key in compared_pairs:
                        continue
                    compared_pairs.add(pair_key)
                    fresh_targets.append(int(target_idx))
                if not fresh_targets:
                    continue
                target_indices = np.asarray(fresh_targets, dtype=np.int64)

                prefilter_started = time.perf_counter()
                target_indices = self._batch_prefilter(bundle, shortlist_index, source_idx, target_indices)
                self.coverage_detail_seconds["prefilter"] += time.perf_counter() - prefilter_started
                if target_indices.size == 0:
                    continue
                self.coverage_debug_stats["geometry_pair_count"] += int(target_indices.size)

                source_cache = self._bundle_geometry_cache(bundle, int(source_idx), tol_px)
                source_packed = np.asarray(source_cache["packed"], dtype=np.uint8)
                packed_row_bytes = max(int(source_packed.size), 1)
                matched_chunks: List[np.ndarray] = []
                if self.matching_mode == "acc":
                    clip_pixels = max(float(bundle["clip_pixels"]), 1.0)
                    for target_chunk in _chunk_indices_by_row_width(target_indices, packed_row_bytes):
                        target_packed = self._bundle_geometry_matrix(bundle, target_chunk, tol_px, "packed")
                        geometry_started = time.perf_counter()
                        xor_rows = _bitcount_sum_rows(
                            np.bitwise_xor(target_packed, source_packed[None, :])
                        )
                        matched_chunks.append(target_chunk[(xor_rows / clip_pixels) <= ratio_limit])
                        self.coverage_detail_seconds["geometry_match"] += time.perf_counter() - geometry_started
                else:
                    if tol_px <= 0:
                        for target_chunk in _chunk_indices_by_row_width(target_indices, packed_row_bytes):
                            target_packed = self._bundle_geometry_matrix(bundle, target_chunk, tol_px, "packed")
                            geometry_started = time.perf_counter()
                            exact_equal = np.all(
                                target_packed == source_packed[None, :],
                                axis=1,
                            )
                            matched_chunks.append(target_chunk[exact_equal])
                            self.coverage_detail_seconds["geometry_match"] += time.perf_counter() - geometry_started
                    else:
                        source_area = float(bundle["areas"][source_idx])
                        source_area_limit = ECC_RESIDUAL_RATIO * max(source_area, 1.0)
                        source_dilated_cache = self._bundle_geometry_cache(
                            bundle,
                            int(source_idx),
                            tol_px,
                            level="dilated",
                        )
                        source_dilated_area = int(source_dilated_cache["dilated_area_px"])
                        source_packed_dilated = np.asarray(source_dilated_cache["packed_dilated"], dtype=np.uint8)
                        for target_chunk in _chunk_indices_by_row_width(target_indices, packed_row_bytes * 4):
                            geometry_started = time.perf_counter()
                            target_areas = bundle["areas"][target_chunk].astype(np.float64)
                            target_area_limits = ECC_RESIDUAL_RATIO * np.maximum(
                                target_areas,
                                1.0,
                            )
                            area_candidate_indices = target_chunk[
                                target_areas <= float(source_dilated_area) + target_area_limits
                            ]
                            self.coverage_detail_seconds["geometry_match"] += time.perf_counter() - geometry_started
                            if not area_candidate_indices.size:
                                continue

                            target_dilated_areas = self._bundle_geometry_values(
                                bundle,
                                area_candidate_indices,
                                tol_px,
                                "dilated_area_px",
                            )
                            geometry_started = time.perf_counter()
                            overlap_indices = area_candidate_indices[
                                source_area <= target_dilated_areas.astype(np.float64) + source_area_limit
                            ]
                            self.coverage_detail_seconds["geometry_match"] += time.perf_counter() - geometry_started

                            if overlap_indices.size:
                                target_packed_dilated = self._bundle_geometry_matrix(
                                    bundle,
                                    overlap_indices,
                                    tol_px,
                                    "packed_dilated",
                                )
                                geometry_started = time.perf_counter()
                                residual_source_counts = _bitcount_sum_rows(
                                    np.bitwise_and(
                                        source_packed[None, :],
                                        np.bitwise_not(target_packed_dilated),
                                    )
                                )
                                overlap_indices = overlap_indices[residual_source_counts <= source_area_limit]
                                self.coverage_detail_seconds["geometry_match"] += time.perf_counter() - geometry_started

                            if overlap_indices.size:
                                overlap_target_limits = ECC_RESIDUAL_RATIO * np.maximum(
                                    bundle["areas"][overlap_indices].astype(np.float64),
                                    1.0,
                                )
                                target_packed = self._bundle_geometry_matrix(bundle, overlap_indices, tol_px, "packed")
                                geometry_started = time.perf_counter()
                                residual_target_counts = _bitcount_sum_rows(
                                    np.bitwise_and(
                                        target_packed,
                                        np.bitwise_not(source_packed_dilated[None, :]),
                                    )
                                )
                                overlap_indices = overlap_indices[residual_target_counts <= overlap_target_limits]
                                self.coverage_detail_seconds["geometry_match"] += time.perf_counter() - geometry_started

                            if not overlap_indices.size:
                                continue

                            source_donut_cache = self._bundle_geometry_cache(
                                bundle,
                                int(source_idx),
                                tol_px,
                                level="donut",
                            )
                            source_donut_area = int(source_donut_cache["donut_area_px"])
                            source_packed_donut = np.asarray(source_donut_cache["packed_donut"], dtype=np.uint8)
                            target_donut_areas = self._bundle_geometry_values(bundle, overlap_indices, tol_px, "donut_area_px")
                            geometry_started = time.perf_counter()
                            auto_true = (source_donut_area == 0) | (target_donut_areas == 0)
                            if np.any(auto_true):
                                matched_chunks.append(overlap_indices[auto_true])

                            overlap_indices = overlap_indices[~auto_true]
                            target_donut_areas = target_donut_areas[~auto_true]
                            self.coverage_detail_seconds["geometry_match"] += time.perf_counter() - geometry_started
                            if overlap_indices.size:
                                target_packed_donut = self._bundle_geometry_matrix(
                                    bundle,
                                    overlap_indices,
                                    tol_px,
                                    "packed_donut",
                                )
                                geometry_started = time.perf_counter()
                                overlap_counts = _bitcount_sum_rows(
                                    np.bitwise_and(
                                        target_packed_donut,
                                        source_packed_donut[None, :],
                                    )
                                )
                                overlap_denominator = np.maximum(
                                    np.minimum(target_donut_areas, source_donut_area).astype(np.float64),
                                    1.0,
                                )
                                overlap_ok = (overlap_counts / overlap_denominator) >= ECC_DONUT_OVERLAP_RATIO
                                matched_chunks.append(overlap_indices[overlap_ok])
                                self.coverage_detail_seconds["geometry_match"] += time.perf_counter() - geometry_started

                non_empty_chunks = [chunk for chunk in matched_chunks if chunk.size]
                matched_indices = (
                    np.concatenate(non_empty_chunks, axis=0) if non_empty_chunks else np.asarray([], dtype=np.int64)
                )

                matched_set = {int(target_idx) for target_idx in matched_indices.tolist()}
                for target_idx in target_indices.tolist():
                    self._record_geometry_result(
                        source_candidate,
                        bundle["representatives"][int(target_idx)],
                        int(target_idx) in matched_set,
                    )
                for target_idx in matched_indices:
                    target_idx = int(target_idx)
                    coverage_by_group[source_idx].update(bundle["origin_ids"][target_idx])
                    coverage_by_group[target_idx].update(bundle["origin_ids"][source_idx])

            for group_idx, grouped_candidates in enumerate(bundle["candidate_groups"]):
                for candidate in grouped_candidates:
                    candidate.coverage = set(coverage_by_group[group_idx])

        self._log(
            "coverage 几何统计: "
            f"pair={self.coverage_debug_stats['geometry_pair_count']}, "
            f"cache group={self.coverage_debug_stats['geometry_cache_group_count']}, "
            f"dilated={self.coverage_debug_stats['geometry_dilated_cache_group_count']}, "
            f"donut={self.coverage_debug_stats['geometry_donut_cache_group_count']}, "
            f"detail={{{', '.join(f'{k}: {round(v, 3)}' for k, v in self.coverage_detail_seconds.items())}}}"
        )

    def _greedy_cover(self, candidates: Sequence[CandidateClip], exact_clusters: Sequence[ExactCluster]) -> List[CandidateClip]:
        """按 coverage、cluster 数与 shift 代价执行 lazy-heap greedy set cover。"""

        uncovered = {int(cluster.exact_cluster_id) for cluster in exact_clusters}
        weights = {int(cluster.exact_cluster_id): int(cluster.weight) for cluster in exact_clusters}
        selected: List[CandidateClip] = []
        selected_ids: set[str] = set()
        candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
        heap: List[Tuple[Tuple[Any, ...], str]] = []

        def _priority(candidate: CandidateClip) -> Tuple[Any, ...]:
            covered_now = set(candidate.coverage) & uncovered
            return (
                -sum(weights[cid] for cid in covered_now),
                -len(covered_now),
                -1 if candidate.shift_direction == "base" else 0,
                abs(candidate.shift_distance_um),
                int(candidate.origin_exact_cluster_id),
                candidate.candidate_id,
            )

        for candidate in candidates:
            heapq.heappush(heap, (_priority(candidate), candidate.candidate_id))

        while uncovered:
            best: CandidateClip | None = None
            covered_now: set[int] = set()
            while heap:
                saved_priority, candidate_id = heapq.heappop(heap)
                if candidate_id in selected_ids:
                    continue
                candidate = candidate_by_id[candidate_id]
                current_priority = _priority(candidate)
                if current_priority != saved_priority:
                    heapq.heappush(heap, (current_priority, candidate_id))
                    continue
                current_covered = set(candidate.coverage) & uncovered
                if current_covered:
                    best = candidate
                    covered_now = current_covered
                    break
            if best is None:
                missing = min(uncovered)
                best = self._base_candidate_by_exact_id[missing]
                covered_now = set(best.coverage) & uncovered
                if not covered_now:
                    covered_now = {missing}
            selected.append(best)
            selected_ids.add(best.candidate_id)
            uncovered -= covered_now
        return selected

    def _geometry_passes(self, candidate: CandidateClip, target: MarkerRecord) -> bool:
        """根据当前模式执行 ACC 或 ECC 最终几何判定。"""

        if candidate.clip_bitmap.shape != target.clip_bitmap.shape:
            return False
        if self.matching_mode == "acc":
            xor_ratio = float(np.count_nonzero(candidate.clip_bitmap ^ target.clip_bitmap)) / float(
                max(candidate.clip_bitmap.size, 1)
            )
            return bool(xor_ratio <= max(0.0, 1.0 - float(self.area_match_ratio)) + 1e-12)
        return _ecc_match_cached(candidate, target, self.edge_tolerance_um, self.pixel_size_um)

    def _assign_exact_clusters(
        self,
        selected_candidates: Sequence[CandidateClip],
        exact_clusters: Sequence[ExactCluster],
    ) -> Dict[str, List[ExactCluster]]:
        """把 exact clusters 分配给最终代表，按 coverage 反向索引避免全量扫描。"""

        assignments = {candidate.candidate_id: [] for candidate in selected_candidates}
        exact_by_id = {int(cluster.exact_cluster_id): cluster for cluster in exact_clusters}
        best_candidate_by_cluster: Dict[int, CandidateClip] = {}
        best_key_by_cluster: Dict[int, Tuple[Any, ...]] = {}

        for candidate in selected_candidates:
            for cluster_id in candidate.coverage:
                exact_cluster = exact_by_id.get(int(cluster_id))
                if exact_cluster is None:
                    continue
                key = (
                    0 if candidate.clip_hash == exact_cluster.representative.clip_hash else 1,
                    0 if candidate.origin_exact_cluster_id == exact_cluster.exact_cluster_id else 1,
                    0 if candidate.shift_direction == "base" else 1,
                    abs(candidate.shift_distance_um),
                    candidate.candidate_id,
                )
                previous_key = best_key_by_cluster.get(int(cluster_id))
                if previous_key is None or key < previous_key:
                    best_key_by_cluster[int(cluster_id)] = key
                    best_candidate_by_cluster[int(cluster_id)] = candidate

        for exact_cluster in exact_clusters:
            cluster_id = int(exact_cluster.exact_cluster_id)
            candidate = best_candidate_by_cluster.get(cluster_id)
            if candidate is None:
                raise RuntimeError(f"No selected candidate covers exact cluster {cluster_id}")
            assignments[candidate.candidate_id].append(exact_cluster)
        return assignments

    def _candidate_matches_exact(
        self,
        candidate: CandidateClip,
        exact_cluster: ExactCluster,
        *,
        strict: bool,
    ) -> bool:
        """判断 candidate 是否能够覆盖某个 exact cluster。"""

        target = exact_cluster.representative
        if candidate.clip_hash == target.clip_hash:
            return True

        geometry_started = time.perf_counter()
        geometry_ok = self._geometry_passes(candidate, target)
        self.final_verification_detail_seconds["geometry"] += time.perf_counter() - geometry_started
        if not geometry_ok:
            return False

        prefilter_started = time.perf_counter()
        prefilter_ok, reason = _graph_prefilter_passes(candidate, target, strict=strict)
        self.final_verification_detail_seconds["graph_prefilter"] += time.perf_counter() - prefilter_started
        del reason
        if not prefilter_ok:
            return False
        return True

    def _verified_cluster_units(
        self,
        selected_candidates: Sequence[CandidateClip],
        exact_clusters: Sequence[ExactCluster],
    ) -> List[Tuple[CandidateClip, List[ExactCluster]]]:
        """对选中的 representative 做最终验证，并把失败项拆成 singleton。"""

        self.final_verification_detail_seconds = _empty_verification_detail_seconds()
        assignment_started = time.perf_counter()
        assignments = self._assign_exact_clusters(selected_candidates, exact_clusters)
        self.final_verification_detail_seconds["assignment"] += time.perf_counter() - assignment_started
        self.final_verification_stats = _empty_verification_stats()
        units: List[Tuple[CandidateClip, List[ExactCluster]]] = []

        for candidate in selected_candidates:
            accepted: List[ExactCluster] = []
            for exact_cluster in assignments.get(candidate.candidate_id, []):
                if self._candidate_matches_exact(candidate, exact_cluster, strict=True):
                    accepted.append(exact_cluster)
                    self.final_verification_stats["verified_pass"] += 1
                else:
                    self.final_verification_stats["verified_reject"] += 1
                    self.final_verification_stats["singleton_created"] += 1
                    base = self._base_candidate_by_exact_id[int(exact_cluster.exact_cluster_id)]
                    units.append((base, [exact_cluster]))
            if accepted:
                units.append((candidate, accepted))
        return units

    def _sample_metadata(self, record: MarkerRecord) -> Dict[str, Any]:
        """生成单个 sample 的输出 metadata。"""

        auto_seed = dict(record.match_cache.get("auto_seed", {}) or {})
        return {
            "pipeline_mode": PIPELINE_MODE,
            "marker_id": str(record.marker_id),
            "exact_cluster_id": int(record.exact_cluster_id),
            "geometry_match_mode": str(self.matching_mode),
            "source_path": str(record.source_path),
            "source_name": str(record.source_name),
            "marker_bbox": list(record.marker_bbox),
            "marker_center": list(record.marker_center),
            "clip_bbox": list(record.clip_bbox),
            "selected_candidate_id": None,
            "selected_shift_direction": None,
            "selected_shift_distance_um": None,
            "seed_weight": int(record.seed_weight),
            "seed_bbox": auto_seed.get("seed_bbox"),
            "grid_ix": auto_seed.get("grid_ix"),
            "grid_iy": auto_seed.get("grid_iy"),
            "grid_cell_bbox": auto_seed.get("grid_cell_bbox"),
        }

    def _build_results(
        self,
        marker_records: Sequence[MarkerRecord],
        exact_clusters: Sequence[ExactCluster],
        selected_candidates: Sequence[CandidateClip],
        solver_used: str,
        runtime_summary: Dict[str, float],
        candidate_count: int,
    ) -> Dict[str, Any]:
        """物化 sample/representative clip，并组装最终结果 JSON。"""

        del solver_used
        self.result_detail_seconds = _empty_result_detail_seconds()
        sample_dir = self.temp_dir / "samples"
        representative_dir = self.temp_dir / "representatives"
        materialize_outputs = bool(self.materialize_outputs)
        if materialize_outputs:
            sample_dir.mkdir(parents=True, exist_ok=True)
            representative_dir.mkdir(parents=True, exist_ok=True)

        ordered_records = list(sorted(marker_records, key=lambda item: (item.source_name, item.marker_id)))
        sample_file_map: Dict[str, str] = {}
        sample_index_map: Dict[str, int] = {}
        file_list: List[str] = []
        file_metadata: List[Dict[str, Any]] = []

        sample_started = time.perf_counter()
        for sample_index, record in enumerate(ordered_records):
            if materialize_outputs:
                sample_path = sample_dir / _make_sample_filename("sample", record.source_name, sample_index)
                materialize_started = time.perf_counter()
                sample_file = _materialize_clip_bitmap(
                    record.clip_bitmap,
                    record.clip_bbox,
                    record.marker_id,
                    sample_path,
                    self.pixel_size_um,
                )
                self.result_detail_seconds["sample_materialize"] += time.perf_counter() - materialize_started
                sample_file_map[record.marker_id] = sample_file
                file_list.append(sample_file)
            sample_index_map[record.marker_id] = int(sample_index)
            file_metadata.append(self._sample_metadata(record))
        self.result_detail_seconds["sample_metadata"] += time.perf_counter() - sample_started

        verification_started = time.perf_counter()
        cluster_units = self._verified_cluster_units(selected_candidates, exact_clusters)
        self.result_detail_seconds["final_verification"] += time.perf_counter() - verification_started
        clusters_output: List[Dict[str, Any]] = []

        cluster_output_started = time.perf_counter()
        for cluster_index, (candidate, assigned_exact_clusters) in enumerate(cluster_units):
            cluster_members = list(
                sorted(
                    (member for exact_cluster in assigned_exact_clusters for member in exact_cluster.members),
                    key=lambda item: (item.source_name, item.marker_id),
                )
            )
            if not cluster_members:
                continue

            export_member, export_scores = _rerank_export_representative(cluster_members)
            representative_file = None
            if materialize_outputs:
                rep_path = representative_dir / _make_sample_filename("rep", cluster_members[0].source_name, cluster_index)
                rep_materialize_started = time.perf_counter()
                representative_file = _materialize_clip_bitmap(
                    candidate.clip_bitmap,
                    candidate.clip_bbox,
                    candidate.candidate_id,
                    rep_path,
                    self.pixel_size_um,
                )
                self.result_detail_seconds["representative_materialize"] += time.perf_counter() - rep_materialize_started
            sample_indices = [sample_index_map[member.marker_id] for member in cluster_members]
            sample_files = [sample_file_map[member.marker_id] for member in cluster_members] if materialize_outputs else []
            export_sample_index = sample_index_map[export_member.marker_id]
            export_representative_file = sample_file_map.get(export_member.marker_id) if materialize_outputs else None
            sample_metadata = []
            for member in cluster_members:
                sample_index = sample_index_map[member.marker_id]
                metadata = dict(file_metadata[sample_index])
                metadata.update(
                    {
                        "selected_candidate_id": str(candidate.candidate_id),
                        "selected_shift_direction": str(candidate.shift_direction),
                        "selected_shift_distance_um": float(candidate.shift_distance_um),
                    }
                )
                file_metadata[sample_index] = dict(metadata)
                sample_metadata.append(metadata)

            exact_cluster_ids = [int(exact_cluster.exact_cluster_id) for exact_cluster in assigned_exact_clusters]
            clusters_output.append(
                {
                    "cluster_id": int(cluster_index),
                    "pipeline_mode": PIPELINE_MODE,
                    "size": int(len(cluster_members)),
                    "sample_indices": sample_indices,
                    "sample_files": sample_files,
                    "sample_metadata": sample_metadata,
                    "representative_file": representative_file,
                    "cover_representative_file": representative_file,
                    "export_representative_file": export_representative_file,
                    "representative_metadata": {
                        "pipeline_mode": PIPELINE_MODE,
                        "marker_id": str(candidate.source_marker_id),
                        "exact_cluster_id": int(candidate.origin_exact_cluster_id),
                        "geometry_match_mode": str(self.matching_mode),
                        "selected_candidate_id": str(candidate.candidate_id),
                        "selected_shift_direction": str(candidate.shift_direction),
                        "selected_shift_distance_um": float(candidate.shift_distance_um),
                        "coverage_exact_cluster_ids": exact_cluster_ids,
                    },
                    "cover_representative_metadata": {
                        "pipeline_mode": PIPELINE_MODE,
                        "marker_id": str(candidate.source_marker_id),
                        "exact_cluster_id": int(candidate.origin_exact_cluster_id),
                        "geometry_match_mode": str(self.matching_mode),
                        "selected_candidate_id": str(candidate.candidate_id),
                        "selected_shift_direction": str(candidate.shift_direction),
                        "selected_shift_distance_um": float(candidate.shift_distance_um),
                        "coverage_exact_cluster_ids": exact_cluster_ids,
                    },
                    "export_representative_metadata": {
                        "pipeline_mode": PIPELINE_MODE,
                        "marker_id": str(export_member.marker_id),
                        "exact_cluster_id": int(export_member.exact_cluster_id),
                        "sample_index": int(export_sample_index),
                        "source_name": str(export_member.source_name),
                        "seed_weight": int(export_member.seed_weight),
                        "score": float(export_scores["score"]),
                        "medoid_score": float(export_scores["medoid_score"]),
                        "worst_case_score": float(export_scores["worst_case_score"]),
                        "weight_score": float(export_scores["weight_score"]),
                    },
                    "marker_id": str(candidate.source_marker_id),
                    "exact_cluster_id": int(candidate.origin_exact_cluster_id),
                    "marker_ids": [str(member.marker_id) for member in cluster_members],
                    "exact_cluster_ids": exact_cluster_ids,
                    "geometry_match_mode": str(self.matching_mode),
                    "selected_candidate_id": str(candidate.candidate_id),
                    "selected_shift_direction": str(candidate.shift_direction),
                    "selected_shift_distance_um": float(candidate.shift_distance_um),
                }
            )
        self.result_detail_seconds["cluster_output"] += time.perf_counter() - cluster_output_started

        cluster_sizes = [int(cluster["size"]) for cluster in clusters_output]
        max_shift = max((float(cluster["selected_shift_distance_um"]) for cluster in clusters_output), default=0.0)
        layer_ops = self._layer_operations()
        layer_summary = self._effective_layer_summary()
        apply_layer_operations = bool(self.apply_layer_operations)
        seed_strategy = "uniform_grid"
        grid_step_ratio = float(GRID_STEP_RATIO)
        grid_step_um = float(self.clip_size_um) * grid_step_ratio
        grid_seed_count = sum(int(stats.get("grid_seed_count", 0)) for stats in self._seed_stats_by_source.values())
        initial_seed_count = sum(int(stats.get("initial_seed_count", 0)) for stats in self._seed_stats_by_source.values())
        bucketed_seed_count = sum(int(stats.get("bucketed_seed_count", 0)) for stats in self._seed_stats_by_source.values())
        seed_bucket_merged_count = sum(
            int(stats.get("seed_bucket_merged_count", 0)) for stats in self._seed_stats_by_source.values()
        )
        pre_raster_cache_hit = sum(int(stats.get("pre_raster_cache_hit", 0)) for stats in self._seed_stats_by_source.values())
        pre_raster_cache_miss = sum(int(stats.get("pre_raster_cache_miss", 0)) for stats in self._seed_stats_by_source.values())
        exact_bitmap_cache_hit = sum(int(stats.get("exact_bitmap_cache_hit", 0)) for stats in self._seed_stats_by_source.values())
        exact_bitmap_cache_miss = sum(int(stats.get("exact_bitmap_cache_miss", 0)) for stats in self._seed_stats_by_source.values())
        selected_candidate_direction_counts = {
            str(direction): int(count)
            for direction, count in sorted(Counter(str(candidate.shift_direction) for candidate in selected_candidates).items())
        }
        final_cluster_direction_counts = {
            str(direction): int(count)
            for direction, count in sorted(Counter(str(cluster["selected_shift_direction"]) for cluster in clusters_output).items())
        }
        result = {
            "pipeline_mode": PIPELINE_MODE,
            "seed_mode": SEED_MODE,
            "seed_strategy": seed_strategy,
            "grid_step_ratio": float(grid_step_ratio),
            "grid_step_um": float(grid_step_um),
            "geometry_match_mode": str(self.matching_mode),
            "pixel_size_nm": int(self.pixel_size_nm),
            "area_match_ratio": float(self.area_match_ratio),
            "edge_tolerance_um": float(self.edge_tolerance_um),
            "apply_layer_operations": apply_layer_operations,
            "layer_operation_count": int(len(layer_ops)),
            "layer_operations": layer_ops,
            "effective_clustering_layers": list(layer_summary["effective_clustering_layers"]),
            "excluded_helper_layers": list(layer_summary["excluded_helper_layers"]),
            "marker_count": int(len(marker_records)),
            "grid_seed_count": int(grid_seed_count),
            "initial_seed_count": int(initial_seed_count),
            "bucketed_seed_count": int(bucketed_seed_count),
            "seed_bucket_merged_count": int(seed_bucket_merged_count),
            "pre_raster_cache_hit": int(pre_raster_cache_hit),
            "pre_raster_cache_miss": int(pre_raster_cache_miss),
            "exact_bitmap_cache_hit": int(exact_bitmap_cache_hit),
            "exact_bitmap_cache_miss": int(exact_bitmap_cache_miss),
            "exact_cluster_count": int(len(exact_clusters)),
            "candidate_count": int(candidate_count),
            "selected_candidate_count": int(len(selected_candidates)),
            "selected_candidate_direction_counts": selected_candidate_direction_counts,
            "total_clusters": int(len(clusters_output)),
            "total_samples": int(len(file_metadata)),
            "total_files": int(len(file_list)),
            "materialized_outputs": bool(materialize_outputs),
            "cluster_sizes": cluster_sizes,
            "final_cluster_direction_counts": final_cluster_direction_counts,
            "max_shift_distance_um": float(max_shift),
            "prefilter_stats": dict(self.prefilter_stats),
            "coverage_detail_seconds": dict(self.coverage_detail_seconds),
            "coverage_debug_stats": dict(self.coverage_debug_stats),
            "result_detail_seconds": dict(self.result_detail_seconds),
            "final_verification_stats": dict(self.final_verification_stats),
            "final_verification_detail_seconds": dict(self.final_verification_detail_seconds),
            "clusters": clusters_output,
            "file_list": file_list,
            "file_metadata": file_metadata,
            "result_summary": {
                "pipeline_mode": PIPELINE_MODE,
                "seed_mode": SEED_MODE,
                "seed_strategy": seed_strategy,
                "grid_step_ratio": float(grid_step_ratio),
                "grid_step_um": float(grid_step_um),
                "geometry_match_mode": str(self.matching_mode),
                "pixel_size_nm": int(self.pixel_size_nm),
                "area_match_ratio": float(self.area_match_ratio),
                "edge_tolerance_um": float(self.edge_tolerance_um),
                "apply_layer_operations": apply_layer_operations,
                "layer_operation_count": int(len(layer_ops)),
                "layer_operations": layer_ops,
                "effective_clustering_layers": list(layer_summary["effective_clustering_layers"]),
                "excluded_helper_layers": list(layer_summary["excluded_helper_layers"]),
                "marker_count": int(len(marker_records)),
                "grid_seed_count": int(grid_seed_count),
                "initial_seed_count": int(initial_seed_count),
                "bucketed_seed_count": int(bucketed_seed_count),
                "seed_bucket_merged_count": int(seed_bucket_merged_count),
                "pre_raster_cache_hit": int(pre_raster_cache_hit),
                "pre_raster_cache_miss": int(pre_raster_cache_miss),
                "exact_bitmap_cache_hit": int(exact_bitmap_cache_hit),
                "exact_bitmap_cache_miss": int(exact_bitmap_cache_miss),
                "exact_cluster_count": int(len(exact_clusters)),
                "candidate_count": int(candidate_count),
                "selected_candidate_count": int(len(selected_candidates)),
                "selected_candidate_direction_counts": selected_candidate_direction_counts,
                "total_clusters": int(len(clusters_output)),
                "total_samples": int(len(file_metadata)),
                "total_files": int(len(file_list)),
                "materialized_outputs": bool(materialize_outputs),
                "cluster_sizes": cluster_sizes,
                "final_cluster_direction_counts": final_cluster_direction_counts,
                "max_shift_distance_um": float(max_shift),
                "prefilter_stats": dict(self.prefilter_stats),
                "coverage_detail_seconds": dict(self.coverage_detail_seconds),
                "coverage_debug_stats": dict(self.coverage_debug_stats),
                "result_detail_seconds": dict(self.result_detail_seconds),
                "final_verification_stats": dict(self.final_verification_stats),
                "final_verification_detail_seconds": dict(self.final_verification_detail_seconds),
                "timing_seconds": dict(runtime_summary),
            },
            "config": {
                "seed_mode": SEED_MODE,
                "seed_strategy": seed_strategy,
                "grid_step_ratio": float(grid_step_ratio),
                "grid_step_um": float(grid_step_um),
                "clip_size": float(self.clip_size_um),
                "geometry_match_mode": str(self.matching_mode),
                "area_match_ratio": float(self.area_match_ratio),
                "edge_tolerance_um": float(self.edge_tolerance_um),
                "pixel_size_nm": int(self.pixel_size_nm),
                "apply_layer_operations": apply_layer_operations,
                "layer_operation_count": int(len(layer_ops)),
                "layer_operations": layer_ops,
                "effective_clustering_layers": list(layer_summary["effective_clustering_layers"]),
                "excluded_helper_layers": list(layer_summary["excluded_helper_layers"]),
                "materialized_outputs": bool(materialize_outputs),
                "grid_bucket_quant_um": float(GRID_BUCKET_QUANT_UM),
                "graph_invariant_limit": GRAPH_INVARIANT_LIMIT,
                "graph_topology_threshold": GRAPH_TOPOLOGY_THRESHOLD,
                "graph_signature_threshold": GRAPH_SIGNATURE_THRESHOLD,
                "strict_invariant_limit": STRICT_INVARIANT_LIMIT,
                "strict_topology_threshold": STRICT_TOPOLOGY_THRESHOLD,
                "strict_signature_threshold": STRICT_SIGNATURE_THRESHOLD,
            },
            "cluster_review": {},
        }
        result["result_summary"]["config"] = dict(result["config"])
        return result


def _json_default(value: Any) -> Any:
    """把 numpy / Path 等对象转换成 JSON 可序列化类型。"""

    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _export_review(result: Dict[str, Any], review_dir: str) -> Dict[str, Any]:
    """导出 review 目录，复制 representative 与 member clip 文件。"""

    review_root = Path(review_dir)
    review_root.mkdir(parents=True, exist_ok=True)
    representative_files: List[str] = []
    exported_file_count = 0
    missing_files: List[str] = []

    for cluster in result.get("clusters", []):
        cluster_id = int(cluster["cluster_id"])
        cluster_size = int(cluster["size"])
        cluster_dir = review_root / f"cluster_{cluster_id:04d}_size_{cluster_size:04d}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        representative_path = str(cluster.get("representative_file") or "")
        representative_files.append(representative_path)
        if representative_path:
            rep_src = Path(representative_path)
            if rep_src.exists():
                shutil.copy2(rep_src, cluster_dir / f"REP__selected__{rep_src.name}")
                exported_file_count += 1
            else:
                missing_files.append(str(rep_src))

        for member_idx, src in enumerate(cluster.get("sample_files", [])):
            src_path = Path(src)
            if src_path.exists():
                shutil.copy2(src_path, cluster_dir / f"sample__{member_idx:04d}__{src_path.name}")
                exported_file_count += 1
            else:
                missing_files.append(str(src_path))

    with (review_root / "representative_files.txt").open("w", encoding="utf-8") as handle:
        for filepath in representative_files:
            handle.write(f"{filepath}\n")

    info = {
        "exported": True,
        "review_dir": str(review_root),
        "cluster_count": int(len(result.get("clusters", []))),
        "exported_file_count": int(exported_file_count),
        "representative_file_count": int(len(representative_files)),
        "missing_file_count": int(len(missing_files)),
    }
    if missing_files:
        info["missing_files_preview"] = missing_files[:10]
    result["cluster_review"] = dict(info)
    return info


def _build_parser() -> argparse.ArgumentParser:
    """构建当前 optimized v1 uniform-grid 版本的命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="Optimized uniform-grid layout clustering v1")
    parser.add_argument("input_path", help="Input OASIS file or directory")
    parser.add_argument("--output", "-o", default="clustering_optimized_v1_results.json", help="Output JSON path")
    parser.add_argument("--format", "-f", choices=["json", "txt"], default="json", help="Output format")
    parser.add_argument("--clip-size", type=float, default=1.35, help="Clip side length in um")
    parser.add_argument("--geometry-match-mode", choices=["acc", "ecc"], default="ecc", help="Final geometry gate")
    parser.add_argument("--area-match-ratio", type=float, default=0.96, help="ACC area match threshold")
    parser.add_argument("--edge-tolerance-um", type=float, default=0.02, help="ECC edge tolerance in um")
    parser.add_argument("--pixel-size-nm", type=int, default=DEFAULT_PIXEL_SIZE_NM, help="Raster pixel size in nm")
    parser.add_argument("--review-dir", default=None, help="Optional review directory")
    parser.add_argument("--export-cluster-review-dir", default=None, help="Compatibility alias for --review-dir")
    parser.add_argument("--apply-layer-ops", action="store_true", help="Apply registered boolean layer operations before clustering")
    parser.add_argument(
        "--register-op",
        action="append",
        nargs=4,
        metavar=("SOURCE_LAYER", "TARGET_LAYER", "OPERATION", "RESULT_LAYER"),
        help="Register a layer operation rule, e.g. --register-op 1/0 2/0 subtract 10/0",
    )
    return parser


def main() -> int:
    """
    命令行主入口：解析参数、创建 runner、执行聚类并保存结果。

    使用说明：
    1. 最基本的运行方式：
       python layout_clustering_optimized_v1.py input.oas ^
         --output results.json
    2. 如果需要导出人工 review 目录：
       python layout_clustering_optimized_v1.py input.oas ^
         --output results.json ^
         --review-dir review_out
    3. 如果版图在进入聚类前需要先做层间布尔运算，可配合：
       --apply-layer-ops
       --register-op SOURCE_LAYER TARGET_LAYER OPERATION RESULT_LAYER

    注意点：
    - 这个版本只使用 uniform grid seed 主线，grid 步长固定为 clip size 的 50%，不再保留 pair/drc 分支。
    - seed 会先做粗分桶，再进入 exact hash / graph prefilter / ACC-ECC / shift-cover 主线；
      因此 `marker_count` 在输出里表示最终保留下来的 synthetic seed 样本数。
    - final verification 失败的 exact cluster 不会跨 cluster 重新分配，而是直接退回 singleton，
      这是为了保证结果容易解释和人工 review。
    - 指定 `--review-dir` 后会复制 sample 和 representative clip；大版图运行时请预留足够磁盘空间。
    """

    parser = _build_parser()
    args = parser.parse_args()
    review_dir = _review_dir_from_args(args)
    temp_root = Path(__file__).resolve().parent / "_temp_runs"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"layout_clustering_optimized_v1_{uuid.uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=False)

    register_ops = args.register_op or []
    apply_layer_operations = bool(args.apply_layer_ops or register_ops)
    try:
        layer_processor = _make_layer_processor(register_ops)
    except Exception as exc:
        print(f"运行失败: {exc}")
        return 1
    layer_ops = _layer_operation_payload(layer_processor)
    _print_start_banner(
        "Optimized v1 Layout 聚类分析",
        args,
        apply_layer_operations=apply_layer_operations,
        layer_ops=layer_ops,
    )

    config = {
        "clip_size_um": float(args.clip_size),
        "geometry_match_mode": str(args.geometry_match_mode),
        "area_match_ratio": float(args.area_match_ratio),
        "edge_tolerance_um": float(args.edge_tolerance_um),
        "pixel_size_nm": int(args.pixel_size_nm),
        "apply_layer_operations": apply_layer_operations,
        "materialize_outputs": bool(review_dir),
    }
    try:
        runner = OptimizedMainlineRunner(
            config=config,
            temp_dir=temp_dir,
            layer_processor=layer_processor if apply_layer_operations else None,
        )
        result = runner.run(str(args.input_path))
        if review_dir:
            info = _export_review(result, str(review_dir))
            print(f"cluster review 目录已导出到: {info.get('review_dir', review_dir)}")
        _save_results(result, str(args.output), str(args.format))
        print(f"最终 cluster 数: {result.get('total_clusters', 0)}")
        print(f"最终 seed/sample 数: {result.get('marker_count', 0)} / {result.get('total_samples', 0)}")
        print(
            f"raw/bucketed/merged grid seed 数: {result.get('grid_seed_count', 0)} / "
            f"{result.get('bucketed_seed_count', 0)} / {result.get('seed_bucket_merged_count', 0)}"
        )
        print(f"selected candidate 方向分布: {result.get('selected_candidate_direction_counts', {})}")
        print(f"final cluster 方向分布: {result.get('final_cluster_direction_counts', {})}")
        print(f"final verification: {result.get('final_verification_stats', {})}")
        return 0
    except Exception as exc:
        print(f"运行失败: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

