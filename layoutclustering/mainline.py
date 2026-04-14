#!/usr/bin/env python3
"""Raster-first marker-driven mainline.

中文说明:
1. 主线严格围绕 marker 驱动:
   marker -> exact cluster -> systematic shift candidate -> ACC/ECC -> set cover
2. 为了性能，默认使用 Manhattan 假设下的位图/像素匹配，不再在热路径中依赖几何布尔运算。
3. 当前文件只保留 raster 主线，不再保留几何回退分支。
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import gdstk
import numpy as np
from scipy import ndimage
from scipy.optimize import Bounds, LinearConstraint, milp

from layout_utils import (
    _bbox_center,
    _bbox_intersection,
    _element_layer_datatype,
    _make_centered_bbox,
    _polygon_vertices_array,
    _read_oas_only_library,
    _safe_bbox_tuple,
    _write_oas_library,
)


AUTO_MILP_MAX_CANDIDATES = 24
ECC_DONUT_OVERLAP_RATIO = 0.20
ECC_RESIDUAL_RATIO = 1e-3
DEFAULT_PIXEL_SIZE_NM = 10
DEFAULT_OUTPUT_LAYER = 1
DEFAULT_OUTPUT_DATATYPE = 0
BIT_COUNT_TABLE = np.array([bin(value).count("1") for value in range(256)], dtype=np.uint8)


@dataclass
class MarkerRecord:
    """单个 marker 的主记录。

    这里缓存了 clip / expanded window 的位图、量化后的窗口坐标以及后续匹配需要的哈希。
    """

    marker_id: str
    source_path: str
    source_name: str
    marker_bbox: Tuple[float, float, float, float]
    marker_center: Tuple[float, float]
    clip_bbox: Tuple[float, float, float, float]
    expanded_bbox: Tuple[float, float, float, float]
    clip_bbox_q: Tuple[int, int, int, int]
    expanded_bbox_q: Tuple[int, int, int, int]
    marker_bbox_q: Tuple[int, int, int, int]
    shift_limits_px: Dict[str, Tuple[int, int]]
    clip_bitmap: np.ndarray
    expanded_bitmap: np.ndarray
    clip_hash: str
    expanded_hash: str
    clip_area: float
    exact_cluster_id: int = -1
    match_cache: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExactCluster:
    """双重 exact hash 合并后的精确簇。"""

    exact_cluster_id: int
    representative: MarkerRecord
    members: List[MarkerRecord]

    @property
    def weight(self) -> int:
        return len(self.members)


@dataclass
class CandidateClip:
    """由某个 exact cluster representative 派生出的候选 clip。"""

    candidate_id: str
    origin_exact_cluster_id: int
    center: Tuple[float, float]
    clip_bbox: Tuple[float, float, float, float]
    clip_bbox_q: Tuple[int, int, int, int]
    clip_bitmap: np.ndarray
    clip_hash: str
    shift_direction: str
    shift_distance_um: float
    coverage: Set[int]
    source_marker_id: str
    match_cache: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LayoutIndex:
    """布局查询缓存。

    用紧凑的 bbox 数组替代超大 R-tree，降低整图预处理开销。
    """

    indexed_elements: List[Dict[str, Any]]
    bbox_x0: np.ndarray
    bbox_y0: np.ndarray
    bbox_x1: np.ndarray
    bbox_y1: np.ndarray
    marker_polygons: List[gdstk.Polygon]


def _rank_candidate_ids(
    candidate_ids: Sequence[int],
    indexed_elements: Sequence[Dict[str, Any]],
    bbox: Tuple[float, float, float, float],
    center_xy: Tuple[float, float],
) -> List[int]:
    cx, cy = center_xy

    def _score(elem_id: int) -> Tuple[float, float, int]:
        item_bbox = indexed_elements[int(elem_id)]["bbox"]
        inter = _bbox_intersection(item_bbox, bbox)
        overlap = 0.0 if inter is None else max(0.0, (inter[2] - inter[0]) * (inter[3] - inter[1]))
        icx, icy = _bbox_center(item_bbox)
        dist2 = (icx - cx) ** 2 + (icy - cy) ** 2
        return (-overlap, dist2, int(elem_id))

    return sorted((int(value) for value in candidate_ids), key=_score)


def _make_output_polygons(
    bitmap: np.ndarray,
    bbox: Tuple[float, float, float, float],
    pixel_size_um: float,
    *,
    layer: int = DEFAULT_OUTPUT_LAYER,
    datatype: int = DEFAULT_OUTPUT_DATATYPE,
) -> List[gdstk.Polygon]:
    """把位图重新压缩成矩形 polygon，主要用于结果 OAS 物化。"""

    height, width = bitmap.shape
    if height == 0 or width == 0:
        return []

    active: Dict[Tuple[int, int], Tuple[int, int]] = {}
    rectangles: List[Tuple[int, int, int, int]] = []
    for row in range(height):
        # 逐行提取 run-length，再把同宽度区间沿 y 方向合并成更大的矩形。
        runs: List[Tuple[int, int]] = []
        current_start: Optional[int] = None
        for col in range(width):
            if bitmap[row, col]:
                if current_start is None:
                    current_start = col
            elif current_start is not None:
                runs.append((current_start, col))
                current_start = None
        if current_start is not None:
            runs.append((current_start, width))

        next_active: Dict[Tuple[int, int], Tuple[int, int]] = {}
        run_set = set(runs)
        for run in runs:
            if run in active:
                next_active[run] = (active[run][0], row + 1)
            else:
                next_active[run] = (row, row + 1)
        for run, span in active.items():
            if run not in run_set:
                rectangles.append((run[0], span[0], run[1], span[1]))
        active = next_active

    for run, span in active.items():
        rectangles.append((run[0], span[0], run[1], span[1]))

    polygons: List[gdstk.Polygon] = []
    for x0, y0, x1, y1 in rectangles:
        polygons.append(
            gdstk.rectangle(
                (bbox[0] + x0 * pixel_size_um, bbox[1] + y0 * pixel_size_um),
                (bbox[0] + x1 * pixel_size_um, bbox[1] + y1 * pixel_size_um),
                layer=int(layer),
                datatype=int(datatype),
            )
        )
    return polygons


def _materialize_clip_bitmap(
    bitmap: np.ndarray,
    bbox: Tuple[float, float, float, float],
    sample_id: str,
    output_path: Path,
    pixel_size_um: float,
) -> str:
    lib = gdstk.Library()
    cell = gdstk.Cell(str(sample_id))
    polygons = _make_output_polygons(bitmap, bbox, pixel_size_um)
    if polygons:
        cell.add(*polygons)
    lib.add(cell)
    _write_oas_library(lib, str(output_path))
    return str(output_path)


def _pack_bitmap(bitmap: np.ndarray) -> np.ndarray:
    flat = np.ascontiguousarray(bitmap.reshape(-1).astype(np.uint8, copy=False))
    return np.packbits(flat)


def _bitcount_bytes(byte_values: np.ndarray) -> int:
    if byte_values.size == 0:
        return 0
    return int(BIT_COUNT_TABLE[byte_values].sum(dtype=np.int64))


def _bitmap_transforms(bitmap: np.ndarray) -> Tuple[np.ndarray, ...]:
    transpose = bitmap.T
    return (
        bitmap,
        np.fliplr(bitmap),
        np.flipud(bitmap),
        np.flipud(np.fliplr(bitmap)),
        transpose,
        np.fliplr(transpose),
        np.flipud(transpose),
        np.flipud(np.fliplr(transpose)),
    )


def _canonical_bitmap_payload(bitmap: np.ndarray) -> bytes:
    """对位图做 8 向对称归一化，生成稳定 payload 供 exact hash 使用。"""

    if bitmap.size == 0 or not np.any(bitmap):
        return b"empty"

    payloads: List[bytes] = []
    for transformed in _bitmap_transforms(bitmap):
        contig = np.ascontiguousarray(transformed.astype(np.uint8, copy=False))
        packed = np.packbits(contig.reshape(-1))
        payloads.append(f"{contig.shape[0]}x{contig.shape[1]}:".encode("ascii") + packed.tobytes())
    return min(payloads)


def _canonical_bitmap_hash(bitmap: np.ndarray) -> Tuple[str, bytes]:
    payload = _canonical_bitmap_payload(bitmap)
    return hashlib.sha256(payload).hexdigest(), payload


def _normalize_shift_directions(shift_directions: Optional[Iterable[str]]) -> Tuple[str, ...]:
    if shift_directions is None:
        return ("left", "right", "up", "down")
    if isinstance(shift_directions, str):
        items = [value.strip().lower() for value in shift_directions.split(",") if value.strip()]
    else:
        items = [str(value).strip().lower() for value in shift_directions if str(value).strip()]
    valid = [value for value in items if value in {"left", "right", "up", "down"}]
    return tuple(valid) if valid else ("left", "right", "up", "down")


def _parse_layer_spec(layer_spec: str) -> Tuple[int, int]:
    try:
        layer_str, datatype_str = str(layer_spec).split("/", 1)
        return int(layer_str.strip()), int(datatype_str.strip())
    except Exception as exc:
        raise ValueError(f"Invalid hotspot_layer '{layer_spec}', expected '<layer>/<datatype>'") from exc


def _make_sample_filename(prefix: str, source_name: str, index_value: int) -> str:
    stem = "".join(ch if ch.isascii() and (ch.isalnum() or ch in "-_") else "_" for ch in str(source_name))
    stem = stem.strip("._-") or "layout"
    return f"{prefix}_{stem}_{int(index_value):06d}.oas"


def _window_pixels(window_um: float, pixel_size_um: float) -> int:
    return max(1, int(math.ceil(float(window_um) / max(float(pixel_size_um), 1e-12) - 1e-12)))


def _raster_window_spec(
    marker_bbox: Tuple[float, float, float, float],
    marker_center: Tuple[float, float],
    clip_size_um: float,
    pixel_size_um: float,
) -> Dict[str, Any]:
    """根据 marker 位置生成 clip / expanded window 的物理坐标和像素坐标。

    这里采用“保守外扩”的量化方式，让窄线条在离散化后尽量不丢失。
    """

    cx, cy = marker_center
    clip_width_px = _window_pixels(clip_size_um, pixel_size_um)
    clip_height_px = clip_width_px
    clip_width_um = clip_width_px * pixel_size_um
    clip_height_um = clip_height_px * pixel_size_um
    clip_bbox = _make_centered_bbox(marker_center, clip_width_um, clip_height_um)

    left_extra_px = _window_pixels(max(0.0, cx - marker_bbox[0]), pixel_size_um)
    right_extra_px = _window_pixels(max(0.0, marker_bbox[2] - cx), pixel_size_um)
    bottom_extra_px = _window_pixels(max(0.0, cy - marker_bbox[1]), pixel_size_um)
    top_extra_px = _window_pixels(max(0.0, marker_bbox[3] - cy), pixel_size_um)

    expanded_bbox = (
        clip_bbox[0] - left_extra_px * pixel_size_um,
        clip_bbox[1] - bottom_extra_px * pixel_size_um,
        clip_bbox[2] + right_extra_px * pixel_size_um,
        clip_bbox[3] + top_extra_px * pixel_size_um,
    )
    width_px = clip_width_px + left_extra_px + right_extra_px
    height_px = clip_height_px + bottom_extra_px + top_extra_px
    clip_bbox_q = (
        int(left_extra_px),
        int(bottom_extra_px),
        int(left_extra_px + clip_width_px),
        int(bottom_extra_px + clip_height_px),
    )
    marker_bbox_q = (
        max(0, int(math.floor((marker_bbox[0] - expanded_bbox[0]) / pixel_size_um + 1e-9))),
        max(0, int(math.floor((marker_bbox[1] - expanded_bbox[1]) / pixel_size_um + 1e-9))),
        min(width_px, int(math.ceil((marker_bbox[2] - expanded_bbox[0]) / pixel_size_um - 1e-9))),
        min(height_px, int(math.ceil((marker_bbox[3] - expanded_bbox[1]) / pixel_size_um - 1e-9))),
    )
    return {
        "clip_bbox": clip_bbox,
        "expanded_bbox": expanded_bbox,
        "clip_bbox_q": clip_bbox_q,
        "expanded_bbox_q": (0, 0, int(width_px), int(height_px)),
        "marker_bbox_q": marker_bbox_q,
        "shape": (int(height_px), int(width_px)),
        "shift_limits_px": {
            "x": (-int(left_extra_px), int(right_extra_px)),
            "y": (-int(bottom_extra_px), int(top_extra_px)),
        },
    }


def _query_candidate_ids(
    layout_index: LayoutIndex,
    bbox: Tuple[float, float, float, float],
    *,
    geometry_mode: str,
    max_elements: Optional[int],
    center_xy: Tuple[float, float],
) -> List[int]:
    """查询落入目标窗口的候选图元。

    fast 模式下只在这里做一次粗裁剪，避免后面 shift 阶段重复筛选。
    """

    bbox_x0, bbox_y0, bbox_x1, bbox_y1 = bbox
    mask = (
        (layout_index.bbox_x1 > float(bbox_x0))
        & (layout_index.bbox_x0 < float(bbox_x1))
        & (layout_index.bbox_y1 > float(bbox_y0))
        & (layout_index.bbox_y0 < float(bbox_y1))
    )
    candidate_ids = np.flatnonzero(mask).tolist()
    if geometry_mode == "fast" and max_elements is not None and len(candidate_ids) > int(max_elements):
        candidate_ids = _rank_candidate_ids(candidate_ids, layout_index.indexed_elements, bbox, center_xy)[: int(max_elements)]
    return [int(value) for value in candidate_ids]


def _polygon_strip_spans(
    points: np.ndarray,
    bbox: Tuple[float, float, float, float],
) -> List[Tuple[float, float, float, float]]:
    """把曼哈顿 polygon 分解成一组水平 strip spans，便于直接写入位图。"""

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or len(pts) < 3:
        return []
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) < 3:
        return []

    y_levels = sorted({float(value) for value in pts[:, 1].tolist()})
    spans: List[Tuple[float, float, float, float]] = []
    if len(y_levels) < 2:
        return spans

    for y0, y1 in zip(y_levels[:-1], y_levels[1:]):
        strip_y0 = max(float(y0), float(bbox[1]))
        strip_y1 = min(float(y1), float(bbox[3]))
        if strip_y0 >= strip_y1:
            continue
        y_mid = 0.5 * (strip_y0 + strip_y1)
        xs: List[float] = []
        for start, end in zip(pts, np.roll(pts, -1, axis=0)):
            x0, y0_edge = float(start[0]), float(start[1])
            x1, y1_edge = float(end[0]), float(end[1])
            if abs(y0_edge - y1_edge) <= 1e-12:
                continue
            ymin = min(y0_edge, y1_edge)
            ymax = max(y0_edge, y1_edge)
            if not (ymin <= y_mid < ymax):
                continue
            if abs(x0 - x1) <= 1e-12:
                xs.append(x0)
            else:
                ratio = (y_mid - y0_edge) / (y1_edge - y0_edge)
                xs.append(x0 + ratio * (x1 - x0))
        xs.sort()
        for idx in range(0, len(xs) - 1, 2):
            span_x0 = max(float(xs[idx]), float(bbox[0]))
            span_x1 = min(float(xs[idx + 1]), float(bbox[2]))
            if span_x0 < span_x1:
                spans.append((span_x0, strip_y0, span_x1, strip_y1))
    return spans


def _fill_bitmap_from_elements(
    indexed_elements: Sequence[Dict[str, Any]],
    candidate_ids: Sequence[int],
    bbox: Tuple[float, float, float, float],
    shape: Tuple[int, int],
    pixel_size_um: float,
) -> np.ndarray:
    """把 expanded window 内的图元一次性栅格化成位图。

    后续 base clip 和所有 shifted clip 都从这张位图切片得到，避免重复裁剪。
    """

    height, width = int(shape[0]), int(shape[1])
    bitmap = np.zeros((height, width), dtype=bool)
    if height <= 0 or width <= 0:
        return bitmap

    bbox_x0, bbox_y0 = float(bbox[0]), float(bbox[1])
    for elem_id in candidate_ids:
        item = indexed_elements[int(elem_id)]
        if _bbox_intersection(item["bbox"], bbox) is None:
            continue
        points = _polygon_vertices_array(item["element"])
        if points is None:
            continue
        for span_x0, span_y0, span_x1, span_y1 in _polygon_strip_spans(points, bbox):
            x0 = max(0, int(math.floor((span_x0 - bbox_x0) / pixel_size_um + 1e-9)))
            x1 = min(width, int(math.ceil((span_x1 - bbox_x0) / pixel_size_um - 1e-9)))
            y0 = max(0, int(math.floor((span_y0 - bbox_y0) / pixel_size_um + 1e-9)))
            y1 = min(height, int(math.ceil((span_y1 - bbox_y0) / pixel_size_um - 1e-9)))
            if x0 < x1 and y0 < y1:
                bitmap[y0:y1, x0:x1] = True
    return bitmap


def _slice_bitmap(bitmap: np.ndarray, bbox_q: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox_q
    return np.ascontiguousarray(bitmap[y0:y1, x0:x1], dtype=bool)


def _collect_boundary_positions(axis_mask: np.ndarray) -> List[int]:
    """从 occupancy 掩码中提取边界跳变位置，用于生成 systematic shift 候选。"""

    padded = np.concatenate((np.array([False], dtype=bool), axis_mask.astype(bool), np.array([False], dtype=bool)))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return sorted({int(value) for value in starts.tolist() + ends.tolist()})


def _score_shift_px(
    shift_px: int,
    base_low: int,
    base_high: int,
    boundary_positions: Sequence[int],
    tolerance_px: int,
) -> Tuple[int, int, int]:
    left_boundary = base_low + int(shift_px)
    right_boundary = base_high + int(shift_px)
    if not boundary_positions:
        return 0, 0, -abs(int(shift_px))
    best_gap = min(
        min(abs(int(edge) - left_boundary), abs(int(edge) - right_boundary))
        for edge in boundary_positions
    )
    touch_count = sum(
        1
        for edge in boundary_positions
        if min(abs(int(edge) - left_boundary), abs(int(edge) - right_boundary)) <= tolerance_px
    )
    return touch_count, -int(best_gap), -abs(int(shift_px))


def _collect_shift_values_px(
    boundary_positions: Sequence[int],
    base_low: int,
    base_high: int,
    shift_interval: Tuple[int, int],
    tolerance_px: int,
    *,
    max_count: int,
) -> List[int]:
    """根据边界贴合程度和位移大小，对单轴 shift 候选进行排序裁剪。"""

    low, high = shift_interval
    values = {0, int(low), int(high)}
    for edge in boundary_positions:
        shift_left = int(edge) - int(base_low)
        shift_right = int(edge) - int(base_high)
        if low <= shift_left <= high:
            values.add(int(shift_left))
        if low <= shift_right <= high:
            values.add(int(shift_right))

    ranked = []
    for shift_value in values:
        ranked.append((_score_shift_px(shift_value, base_low, base_high, boundary_positions, tolerance_px), int(shift_value)))
    ranked.sort(key=lambda item: (-item[0][0], -item[0][1], abs(item[1]), item[1]))
    return [shift for _, shift in ranked[: max(1, int(max_count))]]


def _ensure_bitmap_cache(cache_owner: Any, pixel_size_um: float, tol_px: int) -> Dict[str, Any]:
    """缓存候选/目标 clip 的 packed bits 和形态学结果，避免重复计算。"""

    cache_key = f"tol_{int(tol_px)}"
    cache = cache_owner.match_cache.get(cache_key)
    if cache is not None:
        return cache

    bitmap = np.ascontiguousarray(cache_owner.clip_bitmap.astype(bool, copy=False))
    cache = {
        "bitmap": bitmap,
        "packed": _pack_bitmap(bitmap),
        "area_px": int(np.count_nonzero(bitmap)),
    }
    if tol_px > 0:
        structure = np.ones((2 * tol_px + 1, 2 * tol_px + 1), dtype=bool)
        dilated = ndimage.binary_dilation(bitmap, structure=structure)
        eroded = ndimage.binary_erosion(bitmap, structure=structure, border_value=0)
        cache["dilated"] = np.ascontiguousarray(dilated, dtype=bool)
        cache["eroded"] = np.ascontiguousarray(eroded, dtype=bool)
        cache["donut"] = np.ascontiguousarray(dilated & ~eroded, dtype=bool)
        cache["packed_dilated"] = _pack_bitmap(cache["dilated"])
        cache["packed_donut"] = _pack_bitmap(cache["donut"])
        cache["dilated_area_px"] = int(np.count_nonzero(cache["dilated"]))
        cache["donut_area_px"] = int(np.count_nonzero(cache["donut"]))
    cache_owner.match_cache[cache_key] = cache
    return cache


def _bitmap_exact_key(bitmap: np.ndarray) -> Tuple[int, int, bytes]:
    """生成严格区分方向的位图 key，用于零精度损失的去重/缓存。"""

    packed = _pack_bitmap(bitmap)
    return int(bitmap.shape[0]), int(bitmap.shape[1]), packed.tobytes()


def _bitcount_sum_rows(byte_matrix: np.ndarray) -> np.ndarray:
    """对二维 uint8 矩阵逐行做 popcount 求和。"""

    if byte_matrix.size == 0:
        return np.zeros((byte_matrix.shape[0],), dtype=np.int64)
    return BIT_COUNT_TABLE[byte_matrix].sum(axis=1, dtype=np.int64)


def _bitmap_acc_match(bitmap_a: np.ndarray, bitmap_b: np.ndarray, area_match_ratio: float) -> bool:
    """ACC: 使用像素级 XOR 比例衡量面积相似度。"""

    if bitmap_a.shape != bitmap_b.shape:
        return False
    # 论文公式使用 XOR 面积 / clip window 面积，而不是 / 图形覆盖面积。
    xor_ratio = _bitcount_bytes(np.bitwise_xor(_pack_bitmap(bitmap_a), _pack_bitmap(bitmap_b))) / max(
        float(bitmap_b.size),
        1.0,
    )
    return bool(xor_ratio <= max(0.0, 1.0 - float(area_match_ratio)))


def _bitmap_ecc_match(
    bitmap_a: np.ndarray,
    bitmap_b: np.ndarray,
    edge_tolerance_um: float,
    pixel_size_um: float,
) -> bool:
    """ECC: 使用位图形态学近似论文里的 edge continuity / donut overlap 判据。"""

    if bitmap_a.shape != bitmap_b.shape:
        return False
    if not bitmap_a.any() and not bitmap_b.any():
        return True
    if not bitmap_a.any() or not bitmap_b.any():
        return False

    tol_px = max(0, int(math.ceil(float(edge_tolerance_um) / max(float(pixel_size_um), 1e-12) - 1e-12)))
    if tol_px <= 0:
        return bool(np.array_equal(bitmap_a, bitmap_b))

    # 用统一结构元做膨胀/腐蚀，得到容差范围和 donut 区域。
    structure = np.ones((2 * tol_px + 1, 2 * tol_px + 1), dtype=bool)
    dilated_a = ndimage.binary_dilation(bitmap_a, structure=structure)
    dilated_b = ndimage.binary_dilation(bitmap_b, structure=structure)
    area_a = max(float(np.count_nonzero(bitmap_a)), 1.0)
    area_b = max(float(np.count_nonzero(bitmap_b)), 1.0)

    # 两两候选匹配是无向关系，两边残差分别按自身面积归一化，避免比较顺序影响结果。
    residual_a = np.count_nonzero(bitmap_a & ~dilated_b) / area_a
    residual_b = np.count_nonzero(bitmap_b & ~dilated_a) / area_b
    if residual_a > ECC_RESIDUAL_RATIO or residual_b > ECC_RESIDUAL_RATIO:
        return False

    eroded_a = ndimage.binary_erosion(bitmap_a, structure=structure, border_value=0)
    eroded_b = ndimage.binary_erosion(bitmap_b, structure=structure, border_value=0)
    donut_a = dilated_a & ~eroded_a
    donut_b = dilated_b & ~eroded_b
    donut_area_a = int(np.count_nonzero(donut_a))
    donut_area_b = int(np.count_nonzero(donut_b))
    if donut_area_a == 0 or donut_area_b == 0:
        return True
    overlap = int(np.count_nonzero(donut_a & donut_b))
    denom = max(min(donut_area_a, donut_area_b), 1)
    return float(overlap / denom) >= ECC_DONUT_OVERLAP_RATIO


class MainlineRunner:
    """主线执行器。

    负责把 OASIS 输入转换成 marker records，并执行:
    1. exact clustering
    2. systematic shift candidate generation
    3. candidate coverage evaluation
    4. set cover 求解
    5. 结果物化
    """

    def __init__(self, *, config: Dict[str, Any], temp_dir: Path, layer_processor: Optional[Any] = None):
        self.config = dict(config)
        self.temp_dir = Path(temp_dir)
        self.layer_processor = layer_processor
        self.clip_size_um = float(self.config.get("clip_size_um", 1.35))
        self.hotspot_layer = self.config.get("hotspot_layer")
        self.matching_mode = str(self.config.get("matching_mode", "ecc")).strip().lower()
        self.solver = str(self.config.get("solver", "auto")).strip().lower()
        self.geometry_mode = str(self.config.get("geometry_mode", "exact")).strip().lower()
        self.area_match_ratio = float(self.config.get("area_match_ratio", 0.96))
        self.edge_tolerance_um = float(self.config.get("edge_tolerance_um", 0.02))
        self.max_elements_per_window = None
        if self.geometry_mode == "fast":
            self.max_elements_per_window = int(self.config.get("max_elements_per_window", 256))
        self.shift_directions = _normalize_shift_directions(self.config.get("clip_shift_directions"))
        self.shift_boundary_tolerance_um = float(
            self.config.get("clip_shift_boundary_tolerance_um", self.edge_tolerance_um)
        )
        self.apply_layer_operations = bool(self.config.get("apply_layer_operations", False))
        self.pixel_size_nm = int(self.config.get("pixel_size_nm", DEFAULT_PIXEL_SIZE_NM))
        self.pixel_size_um = float(self.pixel_size_nm) / 1000.0

        if not self.hotspot_layer:
            raise ValueError("mainline mode requires --hotspot-layer")
        if self.matching_mode not in {"acc", "ecc"}:
            raise ValueError(f"Unsupported matching_mode: {self.matching_mode}")
        if self.solver not in {"auto", "greedy", "ilp"}:
            raise ValueError(f"Unsupported solver: {self.solver}")
        if self.geometry_mode not in {"exact", "fast"}:
            raise ValueError(f"Unsupported geometry_mode: {self.geometry_mode}")
        if self.pixel_size_um <= 0.0:
            raise ValueError("pixel_size_nm must be positive")

        self.hotspot_layer_tuple = _parse_layer_spec(str(self.hotspot_layer))

    def _discover_input_files(self, input_path: str) -> List[Path]:
        path = Path(input_path)
        if path.is_file():
            return [path]
        return sorted([item for item in path.glob("*.oas") if item.is_file()])

    def _prepare_layout(self, filepath: Path) -> LayoutIndex:
        """读取并 flatten 布局，把 pattern 与 marker 图形拆分出来，并建立空间索引。"""

        lib = _read_oas_only_library(str(filepath))
        if self.apply_layer_operations and self.layer_processor is not None:
            lib = self.layer_processor.apply_layer_operations(lib)

        top_cells = list(lib.top_level()) or list(lib.cells)
        pattern_polygons: List[gdstk.Polygon] = []
        marker_polygons: List[gdstk.Polygon] = []

        for top_cell in top_cells:
            # 直接让 gdstk 按层级展开并返回真实 polygon，避免 copy()+flatten()+再次复制 120w 级图元。
            polygons = top_cell.get_polygons(apply_repetitions=True, include_paths=True, depth=None)
            for poly in polygons:
                layer, datatype = _element_layer_datatype(poly)
                target = marker_polygons if (int(layer), int(datatype)) == self.hotspot_layer_tuple else pattern_polygons
                target.append(poly)

        indexed_elements: List[Dict[str, Any]] = []
        bbox_x0: List[float] = []
        bbox_y0: List[float] = []
        bbox_x1: List[float] = []
        bbox_y1: List[float] = []
        for poly in pattern_polygons:
            bbox = _safe_bbox_tuple(poly.bounding_box())
            if bbox is None:
                continue
            layer, datatype = _element_layer_datatype(poly)
            indexed_elements.append(
                {
                    "element": poly,
                    "bbox": bbox,
                    "layer": int(layer),
                    "datatype": int(datatype),
                }
            )
            bbox_x0.append(float(bbox[0]))
            bbox_y0.append(float(bbox[1]))
            bbox_x1.append(float(bbox[2]))
            bbox_y1.append(float(bbox[3]))
        return LayoutIndex(
            indexed_elements=indexed_elements,
            bbox_x0=np.asarray(bbox_x0, dtype=np.float64),
            bbox_y0=np.asarray(bbox_y0, dtype=np.float64),
            bbox_x1=np.asarray(bbox_x1, dtype=np.float64),
            bbox_y1=np.asarray(bbox_y1, dtype=np.float64),
            marker_polygons=marker_polygons,
        )

    def _build_marker_record(
        self,
        filepath: Path,
        marker_index: int,
        marker_poly: gdstk.Polygon,
        layout_index: LayoutIndex,
    ) -> Optional[MarkerRecord]:
        """围绕单个 marker 构建完整的位图记录。

        关键步骤:
        1. 计算 clip / expanded window
        2. 对 expanded window 做一次性空间查询
        3. 栅格化 expanded window
        4. 从 expanded bitmap 切出 base clip
        5. 生成 exact clustering 用的双重 hash
        """

        marker_bbox = _safe_bbox_tuple(marker_poly.bounding_box())
        if marker_bbox is None:
            return None

        marker_center = _bbox_center(marker_bbox)
        raster_spec = _raster_window_spec(marker_bbox, marker_center, self.clip_size_um, self.pixel_size_um)
        expanded_bbox = raster_spec["expanded_bbox"]
        candidate_ids = _query_candidate_ids(
            layout_index,
            expanded_bbox,
            geometry_mode=self.geometry_mode,
            max_elements=self.max_elements_per_window,
            center_xy=marker_center,
        )
        expanded_bitmap = _fill_bitmap_from_elements(
            layout_index.indexed_elements,
            candidate_ids,
            expanded_bbox,
            raster_spec["shape"],
            self.pixel_size_um,
        )
        clip_bitmap = _slice_bitmap(expanded_bitmap, raster_spec["clip_bbox_q"])
        clip_hash, _ = _canonical_bitmap_hash(clip_bitmap)
        expanded_hash, _ = _canonical_bitmap_hash(expanded_bitmap)

        return MarkerRecord(
            marker_id=f"{filepath.stem}__marker_{int(marker_index):06d}",
            source_path=str(filepath),
            source_name=filepath.name,
            marker_bbox=marker_bbox,
            marker_center=marker_center,
            clip_bbox=raster_spec["clip_bbox"],
            expanded_bbox=expanded_bbox,
            clip_bbox_q=raster_spec["clip_bbox_q"],
            expanded_bbox_q=raster_spec["expanded_bbox_q"],
            marker_bbox_q=raster_spec["marker_bbox_q"],
            shift_limits_px=raster_spec["shift_limits_px"],
            clip_bitmap=clip_bitmap,
            expanded_bitmap=expanded_bitmap,
            clip_hash=clip_hash,
            expanded_hash=expanded_hash,
            clip_area=float(np.count_nonzero(clip_bitmap)) * (self.pixel_size_um ** 2),
        )

    def _collect_marker_records_for_file(self, filepath: Path) -> List[MarkerRecord]:
        layout_index = self._prepare_layout(filepath)
        records: List[MarkerRecord] = []
        for marker_index, marker_poly in enumerate(layout_index.marker_polygons):
            record = self._build_marker_record(filepath, marker_index, marker_poly, layout_index)
            if record is not None:
                records.append(record)
        return records

    def _group_exact_clusters(self, marker_records: Sequence[MarkerRecord]) -> List[ExactCluster]:
        """用 clip hash + expanded hash 做双重 exact clustering。"""

        buckets: Dict[Tuple[str, str], List[MarkerRecord]] = {}
        for record in marker_records:
            buckets.setdefault((record.clip_hash, record.expanded_hash), []).append(record)

        exact_clusters: List[ExactCluster] = []
        for cluster_id, members in enumerate(sorted(buckets.values(), key=lambda items: (items[0].source_name, items[0].marker_id))):
            for member in members:
                member.exact_cluster_id = int(cluster_id)
            exact_clusters.append(
                ExactCluster(
                    exact_cluster_id=int(cluster_id),
                    representative=members[0],
                    members=list(members),
                )
            )
        return exact_clusters

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
        """把一个位图切片封装成候选 clip 对象。"""

        clip_hash, _ = _canonical_bitmap_hash(bitmap)
        return CandidateClip(
            candidate_id=f"cand_{cluster.exact_cluster_id:06d}_{candidate_index:03d}",
            origin_exact_cluster_id=int(cluster.exact_cluster_id),
            center=_bbox_center(clip_bbox),
            clip_bbox=clip_bbox,
            clip_bbox_q=clip_bbox_q,
            clip_bitmap=np.ascontiguousarray(bitmap, dtype=bool),
            clip_hash=clip_hash,
            shift_direction=str(shift_direction),
            shift_distance_um=float(shift_distance_um),
            coverage={int(cluster.exact_cluster_id)},
            source_marker_id=str(cluster.representative.marker_id),
        )

    def _generate_candidates_for_cluster(self, cluster: ExactCluster) -> List[CandidateClip]:
        """为某个 exact cluster 生成 base + systematic shift 候选。

        注意这里只允许单方向平移，不会组合出对角平移。
        """

        rep = cluster.representative
        candidates: List[CandidateClip] = [
            self._build_candidate_clip(
                cluster,
                rep.clip_bbox,
                rep.clip_bbox_q,
                rep.clip_bitmap,
                "base",
                0.0,
                0,
            )
        ]

        base_x0, base_y0, base_x1, base_y1 = rep.clip_bbox_q
        tolerance_px = max(0, int(math.ceil(self.shift_boundary_tolerance_um / max(self.pixel_size_um, 1e-12) - 1e-12)))
        max_shift_count = 8 if self.geometry_mode == "fast" else 12

        if {"left", "right"} & set(self.shift_directions):
            # x 方向候选直接从 expanded bitmap 的列边界跳变推导。
            occupied_cols = np.any(rep.expanded_bitmap, axis=0)
            x_boundaries = _collect_boundary_positions(occupied_cols)
            x_interval = list(rep.shift_limits_px["x"])
            if "left" not in self.shift_directions:
                x_interval[0] = max(x_interval[0], 0)
            if "right" not in self.shift_directions:
                x_interval[1] = min(x_interval[1], 0)
            for shift_px in _collect_shift_values_px(
                x_boundaries,
                base_x0,
                base_x1,
                (int(x_interval[0]), int(x_interval[1])),
                tolerance_px,
                max_count=max_shift_count,
            ):
                if shift_px == 0:
                    continue
                clip_bbox_q = (base_x0 + shift_px, base_y0, base_x1 + shift_px, base_y1)
                bitmap = _slice_bitmap(rep.expanded_bitmap, clip_bbox_q)
                shift_um = float(shift_px) * self.pixel_size_um
                clip_bbox = (
                    rep.clip_bbox[0] + shift_um,
                    rep.clip_bbox[1],
                    rep.clip_bbox[2] + shift_um,
                    rep.clip_bbox[3],
                )
                candidates.append(
                    self._build_candidate_clip(
                        cluster,
                        clip_bbox,
                        clip_bbox_q,
                        bitmap,
                        "right" if shift_px > 0 else "left",
                        abs(shift_um),
                        len(candidates),
                    )
                )

        if {"up", "down"} & set(self.shift_directions):
            # y 方向候选同理，保持单轴独立生成。
            occupied_rows = np.any(rep.expanded_bitmap, axis=1)
            y_boundaries = _collect_boundary_positions(occupied_rows)
            y_interval = list(rep.shift_limits_px["y"])
            if "down" not in self.shift_directions:
                y_interval[0] = max(y_interval[0], 0)
            if "up" not in self.shift_directions:
                y_interval[1] = min(y_interval[1], 0)
            for shift_px in _collect_shift_values_px(
                y_boundaries,
                base_y0,
                base_y1,
                (int(y_interval[0]), int(y_interval[1])),
                tolerance_px,
                max_count=max_shift_count,
            ):
                if shift_px == 0:
                    continue
                clip_bbox_q = (base_x0, base_y0 + shift_px, base_x1, base_y1 + shift_px)
                bitmap = _slice_bitmap(rep.expanded_bitmap, clip_bbox_q)
                shift_um = float(shift_px) * self.pixel_size_um
                clip_bbox = (
                    rep.clip_bbox[0],
                    rep.clip_bbox[1] + shift_um,
                    rep.clip_bbox[2],
                    rep.clip_bbox[3] + shift_um,
                )
                candidates.append(
                    self._build_candidate_clip(
                        cluster,
                        clip_bbox,
                        clip_bbox_q,
                        bitmap,
                        "up" if shift_px > 0 else "down",
                        abs(shift_um),
                        len(candidates),
                    )
                )

        # 不同 shift 可能收敛到同一个几何结果，这里按 hash 去重并保留代价更低的候选。
        deduped: Dict[str, CandidateClip] = {}
        for candidate in candidates:
            current = deduped.get(candidate.clip_hash)
            if current is None:
                deduped[candidate.clip_hash] = candidate
                continue
            current_cost = (0 if current.shift_direction == "base" else 1, abs(current.shift_distance_um), current.candidate_id)
            candidate_cost = (0 if candidate.shift_direction == "base" else 1, abs(candidate.shift_distance_um), candidate.candidate_id)
            if candidate_cost < current_cost:
                deduped[candidate.clip_hash] = candidate
        return list(sorted(deduped.values(), key=lambda item: item.candidate_id))

    def _build_candidate_match_bundles(
        self,
        candidates: Sequence[CandidateClip],
        tol_px: int,
    ) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """按候选 clip 位图聚合匹配单元。

        Chen 2017 的覆盖关系来自 candidate clip 之间的两两匹配。这里先按严格位图
        key 合并完全相同的候选，再按 shape 打包，后续即可在唯一位图组级别做批量
        ACC/ECC 比较；这只是语义等价加速，不会跳过任何候选对。
        """

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
            cache = _ensure_bitmap_cache(representative_candidate, self.pixel_size_um, tol_px)
            bundle = bundles.setdefault(
                shape,
                {
                    "areas": [],
                    "hashes": [],
                    "origin_ids": [],
                    "candidate_groups": [],
                    "packed": [],
                    "packed_dilated": [],
                    "packed_donut": [],
                    "dilated_areas": [],
                    "donut_areas": [],
                    "clip_pixels": int(shape[0]) * int(shape[1]),
                },
            )
            bundle["areas"].append(int(cache["area_px"]))
            bundle["hashes"].append(str(representative_candidate.clip_hash))
            bundle["origin_ids"].append(tuple(sorted(bucket["origin_ids"])))
            bundle["candidate_groups"].append(tuple(bucket["candidates"]))
            bundle["packed"].append(np.asarray(cache["packed"], dtype=np.uint8))
            if tol_px > 0:
                bundle["packed_dilated"].append(np.asarray(cache["packed_dilated"], dtype=np.uint8))
                bundle["packed_donut"].append(np.asarray(cache["packed_donut"], dtype=np.uint8))
                bundle["dilated_areas"].append(int(cache["dilated_area_px"]))
                bundle["donut_areas"].append(int(cache["donut_area_px"]))

        for bundle in bundles.values():
            bundle["areas"] = np.asarray(bundle["areas"], dtype=np.int64)
            bundle["packed"] = np.stack(bundle["packed"], axis=0)
            bundle["hashes_np"] = np.asarray(bundle["hashes"])
            hash_to_indices: Dict[str, List[int]] = {}
            for idx, clip_hash in enumerate(bundle["hashes"]):
                hash_to_indices.setdefault(clip_hash, []).append(idx)
            bundle["hash_to_indices"] = hash_to_indices
            if tol_px > 0:
                bundle["packed_dilated"] = np.stack(bundle["packed_dilated"], axis=0)
                bundle["packed_donut"] = np.stack(bundle["packed_donut"], axis=0)
                bundle["dilated_areas"] = np.asarray(bundle["dilated_areas"], dtype=np.int64)
                bundle["donut_areas"] = np.asarray(bundle["donut_areas"], dtype=np.int64)
            else:
                bundle["packed_dilated"] = None
                bundle["packed_donut"] = None
                bundle["dilated_areas"] = None
                bundle["donut_areas"] = None
        return bundles

    def _evaluate_candidate_coverage_raster(
        self,
        candidates: Sequence[CandidateClip],
        exact_clusters: Sequence[ExactCluster],
    ) -> None:
        """按论文主线建立 candidate clip 两两匹配后的覆盖关系。"""

        tol_px = max(0, int(math.ceil(float(self.edge_tolerance_um) / max(float(self.pixel_size_um), 1e-12) - 1e-12)))
        bundles = self._build_candidate_match_bundles(candidates, tol_px)

        # 自覆盖: C_ij 一定可以代表自己的源 exact cluster K_i。
        for candidate in candidates:
            candidate.coverage = {int(candidate.origin_exact_cluster_id)}

        for bundle in bundles.values():
            group_count = len(bundle["candidate_groups"])
            coverage_by_group: List[Set[int]] = [set(origin_ids) for origin_ids in bundle["origin_ids"]]

            # 严格位图组内已完全相同；相同 canonical hash 的组则对应 mirror/rotation exact match。
            for same_hash_indices in bundle["hash_to_indices"].values():
                if len(same_hash_indices) <= 1:
                    continue
                hash_origin_ids: Set[int] = set()
                for idx in same_hash_indices:
                    hash_origin_ids.update(bundle["origin_ids"][idx])
                for idx in same_hash_indices:
                    coverage_by_group[int(idx)].update(hash_origin_ids)

            all_indices = np.arange(group_count, dtype=np.int64)
            ratio_limit = max(0.0, 1.0 - float(self.area_match_ratio))
            for source_idx in range(max(0, group_count - 1)):
                target_indices = all_indices[source_idx + 1 :]
                if target_indices.size == 0:
                    continue

                # same hash 已作为 exact/symmetry match 处理，这里只比较剩余候选对。
                target_indices = target_indices[bundle["hashes_np"][target_indices] != bundle["hashes_np"][source_idx]]
                if target_indices.size == 0:
                    continue

                if self.matching_mode == "acc":
                    xor_rows = _bitcount_sum_rows(
                        np.bitwise_xor(bundle["packed"][target_indices], bundle["packed"][source_idx][None, :])
                    )
                    clip_pixels = max(float(bundle["clip_pixels"]), 1.0)
                    matched_indices = target_indices[(xor_rows / clip_pixels) <= ratio_limit]
                else:
                    if tol_px <= 0:
                        exact_equal = np.all(bundle["packed"][target_indices] == bundle["packed"][source_idx][None, :], axis=1)
                        matched_indices = target_indices[exact_equal]
                    else:
                        source_area = float(bundle["areas"][source_idx])
                        source_area_limit = ECC_RESIDUAL_RATIO * max(source_area, 1.0)
                        target_area_limits = ECC_RESIDUAL_RATIO * np.maximum(
                            bundle["areas"][target_indices].astype(np.float64),
                            1.0,
                        )
                        overlap_indices = target_indices[
                            (source_area <= bundle["dilated_areas"][target_indices].astype(np.float64) + source_area_limit)
                            & (
                                bundle["areas"][target_indices].astype(np.float64)
                                <= float(bundle["dilated_areas"][source_idx]) + target_area_limits
                            )
                        ]

                        if overlap_indices.size:
                            residual_source_counts = _bitcount_sum_rows(
                                np.bitwise_and(
                                    bundle["packed"][source_idx][None, :],
                                    np.bitwise_not(bundle["packed_dilated"][overlap_indices]),
                                )
                            )
                            overlap_indices = overlap_indices[residual_source_counts <= source_area_limit]

                        if overlap_indices.size:
                            overlap_target_limits = ECC_RESIDUAL_RATIO * np.maximum(
                                bundle["areas"][overlap_indices].astype(np.float64),
                                1.0,
                            )
                            residual_target_counts = _bitcount_sum_rows(
                                np.bitwise_and(
                                    bundle["packed"][overlap_indices],
                                    np.bitwise_not(bundle["packed_dilated"][source_idx][None, :]),
                                )
                            )
                            overlap_indices = overlap_indices[residual_target_counts <= overlap_target_limits]

                        if overlap_indices.size:
                            source_donut_area = int(bundle["donut_areas"][source_idx])
                            auto_true = (source_donut_area == 0) | (bundle["donut_areas"][overlap_indices] == 0)
                            matched_indices = overlap_indices[auto_true]

                            overlap_indices = overlap_indices[~auto_true]
                            if overlap_indices.size:
                                overlap_counts = _bitcount_sum_rows(
                                    np.bitwise_and(
                                        bundle["packed_donut"][overlap_indices],
                                        bundle["packed_donut"][source_idx][None, :],
                                    )
                                )
                                overlap_denominator = np.maximum(
                                    np.minimum(bundle["donut_areas"][overlap_indices], source_donut_area).astype(np.float64),
                                    1.0,
                                )
                                overlap_ok = (overlap_counts / overlap_denominator) >= ECC_DONUT_OVERLAP_RATIO
                                matched_indices = np.concatenate([matched_indices, overlap_indices[overlap_ok]])
                        else:
                            matched_indices = np.asarray([], dtype=np.int64)

                for target_idx in matched_indices:
                    target_idx = int(target_idx)
                    coverage_by_group[source_idx].update(bundle["origin_ids"][target_idx])
                    coverage_by_group[target_idx].update(bundle["origin_ids"][source_idx])

            for group_idx, grouped_candidates in enumerate(bundle["candidate_groups"]):
                for candidate in grouped_candidates:
                    candidate.coverage = set(coverage_by_group[group_idx])

    def _evaluate_candidate_coverage(self, candidates: Sequence[CandidateClip], exact_clusters: Sequence[ExactCluster]) -> None:
        """构建 candidate -> exact cluster 的覆盖关系。"""

        self._evaluate_candidate_coverage_raster(candidates, exact_clusters)

    def _select_solver(self, candidate_count: int) -> str:
        if self.solver != "auto":
            return self.solver
        return "ilp" if int(candidate_count) <= AUTO_MILP_MAX_CANDIDATES else "greedy"

    def _greedy_cover(self, candidates: Sequence[CandidateClip], exact_clusters: Sequence[ExactCluster]) -> List[CandidateClip]:
        """大规模时的吞吐优先求解器。"""

        uncovered = {int(cluster.exact_cluster_id) for cluster in exact_clusters}
        weights = {int(cluster.exact_cluster_id): int(cluster.weight) for cluster in exact_clusters}
        selected: List[CandidateClip] = []

        while uncovered:
            # 优先选择能覆盖更多未覆盖簇、且位移更小的候选。
            ranked = sorted(
                candidates,
                key=lambda candidate: (
                    -sum(weights[cid] for cid in (candidate.coverage & uncovered)),
                    -len(candidate.coverage & uncovered),
                    0 if candidate.shift_direction == "base" else 1,
                    abs(candidate.shift_distance_um),
                    candidate.candidate_id,
                ),
            )
            best = ranked[0]
            covered_now = set(best.coverage) & uncovered
            if not covered_now:
                missing_cluster_id = min(uncovered)
                fallback = next((candidate for candidate in candidates if candidate.origin_exact_cluster_id == missing_cluster_id), None)
                if fallback is None:
                    raise RuntimeError(f"Unable to cover exact cluster {missing_cluster_id}")
                best = fallback
                covered_now = set(best.coverage) & uncovered
            selected.append(best)
            uncovered -= covered_now
        return selected

    def _milp_cover(self, candidates: Sequence[CandidateClip], exact_clusters: Sequence[ExactCluster]) -> List[CandidateClip]:
        """小规模时的精确/近精确 set cover 求解器。

        目标函数首先最小化候选数量，其次轻微惩罚非 base shift 和较大的位移。
        """

        if not candidates:
            return []

        cluster_ids = [int(cluster.exact_cluster_id) for cluster in exact_clusters]
        candidate_count = len(candidates)
        # 每一行对应一个 exact cluster，每一列对应一个 candidate。
        coverage_matrix = np.zeros((len(cluster_ids), candidate_count), dtype=float)
        for row_index, cluster_id in enumerate(cluster_ids):
            for col_index, candidate in enumerate(candidates):
                coverage_matrix[row_index, col_index] = 1.0 if cluster_id in candidate.coverage else 0.0

        if np.any(np.sum(coverage_matrix, axis=1) <= 0.0):
            return self._greedy_cover(candidates, exact_clusters)

        objective = []
        for candidate_index, candidate in enumerate(candidates):
            objective.append(
                1_000_000.0
                + abs(float(candidate.shift_distance_um)) * 1_000.0
                + (10.0 if candidate.shift_direction != "base" else 0.0)
                + candidate_index * 1e-3
            )

        try:
            result = milp(
                c=np.asarray(objective, dtype=float),
                integrality=np.ones(candidate_count, dtype=np.int32),
                bounds=Bounds(np.zeros(candidate_count, dtype=float), np.ones(candidate_count, dtype=float)),
                constraints=LinearConstraint(
                    coverage_matrix,
                    np.ones(len(cluster_ids), dtype=float),
                    np.full(len(cluster_ids), np.inf),
                ),
            )
        except Exception:
            return self._greedy_cover(candidates, exact_clusters)

        if not getattr(result, "success", False) or getattr(result, "x", None) is None:
            return self._greedy_cover(candidates, exact_clusters)

        selected_indices = [idx for idx, value in enumerate(np.asarray(result.x)) if float(value) >= 0.5]
        if not selected_indices:
            return self._greedy_cover(candidates, exact_clusters)
        return [candidates[idx] for idx in selected_indices]

    def _exact_cover(self, candidates: Sequence[CandidateClip], exact_clusters: Sequence[ExactCluster]) -> List[CandidateClip]:
        return self._milp_cover(candidates, exact_clusters)

    def _assign_exact_clusters(
        self,
        selected_candidates: Sequence[CandidateClip],
        exact_clusters: Sequence[ExactCluster],
    ) -> Dict[str, List[ExactCluster]]:
        """把 exact clusters 分配给最终入选的 representative candidates。"""

        assignments = {candidate.candidate_id: [] for candidate in selected_candidates}
        for exact_cluster in exact_clusters:
            eligible = [candidate for candidate in selected_candidates if exact_cluster.exact_cluster_id in candidate.coverage]
            if not eligible:
                raise RuntimeError(f"No selected candidate covers exact cluster {exact_cluster.exact_cluster_id}")
            eligible.sort(
                key=lambda candidate: (
                    0 if candidate.clip_hash == exact_cluster.representative.clip_hash else 1,
                    0 if candidate.origin_exact_cluster_id == exact_cluster.exact_cluster_id else 1,
                    0 if candidate.shift_direction == "base" else 1,
                    abs(candidate.shift_distance_um),
                    candidate.candidate_id,
                )
            )
            assignments[eligible[0].candidate_id].append(exact_cluster)
        return assignments

    def _sample_metadata(self, record: MarkerRecord) -> Dict[str, Any]:
        return {
            "pipeline_mode": "mainline",
            "marker_id": str(record.marker_id),
            "exact_cluster_id": int(record.exact_cluster_id),
            "matching_mode": str(self.matching_mode),
            "source_path": str(record.source_path),
            "source_name": str(record.source_name),
            "marker_bbox": list(record.marker_bbox),
            "marker_center": list(record.marker_center),
            "clip_bbox": list(record.clip_bbox),
            "expanded_bbox": list(record.expanded_bbox),
            "selected_candidate_id": None,
            "selected_shift_direction": None,
            "selected_shift_distance_um": None,
            "solver_used": None,
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
        """把最终 representative / samples 物化成 OAS，并组织统一输出结构。"""

        sample_dir = self.temp_dir / "samples"
        representative_dir = self.temp_dir / "representatives"
        sample_dir.mkdir(parents=True, exist_ok=True)
        representative_dir.mkdir(parents=True, exist_ok=True)

        ordered_records = list(sorted(marker_records, key=lambda item: (item.source_name, item.marker_id)))
        sample_file_map: Dict[str, str] = {}
        sample_index_map: Dict[str, int] = {}
        file_list: List[str] = []
        file_metadata: List[Dict[str, Any]] = []

        for sample_index, record in enumerate(ordered_records):
            # 每个 marker record 都会输出一个 sample clip，便于 review 和回溯。
            sample_path = sample_dir / _make_sample_filename("sample", record.source_name, sample_index)
            sample_file = _materialize_clip_bitmap(record.clip_bitmap, record.clip_bbox, record.marker_id, sample_path, self.pixel_size_um)
            sample_file_map[record.marker_id] = sample_file
            sample_index_map[record.marker_id] = int(sample_index)
            file_list.append(sample_file)
            file_metadata.append(self._sample_metadata(record))

        assignments = self._assign_exact_clusters(selected_candidates, exact_clusters)
        clusters_output: List[Dict[str, Any]] = []

        for cluster_index, candidate in enumerate(selected_candidates):
            # representative 直接来自 set cover 选中的 candidate，而不是额外 medoid 重选。
            assigned_exact_clusters = assignments.get(candidate.candidate_id, [])
            cluster_members = list(
                sorted(
                    (member for exact_cluster in assigned_exact_clusters for member in exact_cluster.members),
                    key=lambda item: (item.source_name, item.marker_id),
                )
            )
            if not cluster_members:
                continue

            representative_path = representative_dir / _make_sample_filename("rep", cluster_members[0].source_name, cluster_index)
            representative_file = _materialize_clip_bitmap(
                candidate.clip_bitmap,
                candidate.clip_bbox,
                candidate.candidate_id,
                representative_path,
                self.pixel_size_um,
            )
            sample_indices = [sample_index_map[member.marker_id] for member in cluster_members]
            sample_files = [sample_file_map[member.marker_id] for member in cluster_members]
            sample_metadata = []
            for member in cluster_members:
                sample_index = sample_index_map[member.marker_id]
                metadata = dict(file_metadata[sample_index])
                metadata.update(
                    {
                        "selected_candidate_id": str(candidate.candidate_id),
                        "selected_shift_direction": str(candidate.shift_direction),
                        "selected_shift_distance_um": float(candidate.shift_distance_um),
                        "solver_used": str(solver_used),
                    }
                )
                file_metadata[sample_index] = dict(metadata)
                sample_metadata.append(metadata)

            exact_cluster_ids = [int(exact_cluster.exact_cluster_id) for exact_cluster in assigned_exact_clusters]
            marker_ids = [str(member.marker_id) for member in cluster_members]
            clusters_output.append(
                {
                    "cluster_id": int(cluster_index),
                    "pipeline_mode": "mainline",
                    "size": int(len(cluster_members)),
                    "sample_indices": sample_indices,
                    "sample_files": sample_files,
                    "sample_metadata": sample_metadata,
                    "representative_index": None,
                    "representative_file": representative_file,
                    "representative_metadata": {
                        "pipeline_mode": "mainline",
                        "marker_id": str(candidate.source_marker_id),
                        "exact_cluster_id": int(candidate.origin_exact_cluster_id),
                        "matching_mode": str(self.matching_mode),
                        "selected_candidate_id": str(candidate.candidate_id),
                        "selected_shift_direction": str(candidate.shift_direction),
                        "selected_shift_distance_um": float(candidate.shift_distance_um),
                        "solver_used": str(solver_used),
                        "coverage_exact_cluster_ids": exact_cluster_ids,
                    },
                    "marker_id": str(candidate.source_marker_id),
                    "exact_cluster_id": int(candidate.origin_exact_cluster_id),
                    "marker_ids": marker_ids,
                    "exact_cluster_ids": exact_cluster_ids,
                    "matching_mode": str(self.matching_mode),
                    "selected_candidate_id": str(candidate.candidate_id),
                    "selected_shift_direction": str(candidate.shift_direction),
                    "selected_shift_distance_um": float(candidate.shift_distance_um),
                    "solver_used": str(solver_used),
                    "cluster_mean_features": [],
                    "cluster_std_features": [],
                }
            )

        cluster_sizes = [int(cluster["size"]) for cluster in clusters_output]
        return {
            "pipeline_mode": "mainline",
            "matching_mode": str(self.matching_mode),
            "solver_used": str(solver_used),
            "geometry_mode": str(self.geometry_mode),
            "pixel_size_nm": int(self.pixel_size_nm),
            "total_files": int(len(file_list)),
            "total_clusters": int(len(clusters_output)),
            "total_samples": int(len(file_list)),
            "cluster_sizes": cluster_sizes,
            "result_summary": {
                "pipeline_mode": "mainline",
                "matching_mode": str(self.matching_mode),
                "solver_used": str(solver_used),
                "geometry_mode": str(self.geometry_mode),
                "pixel_size_nm": int(self.pixel_size_nm),
                "cluster_sizes": cluster_sizes,
                "exact_cluster_count": int(len(exact_clusters)),
                "selected_candidate_count": int(len(selected_candidates)),
                "candidate_count": int(candidate_count),
                "sample_count": int(len(file_list)),
                "marker_count": int(len(marker_records)),
                "timing_seconds": dict(runtime_summary),
            },
            "clusters": clusters_output,
            "file_list": file_list,
            "file_metadata": file_metadata,
            "feature_names": [],
            "feature_space": {},
            "cluster_review": {},
            "config": dict(self.config),
            "pattern_dedup": {
                "mode": "exact_clip_hash_raster",
                "raw_center_count": int(len(marker_records)),
                "unique_window_count": int(len(exact_clusters)),
                "compression_ratio": float(len(exact_clusters)) / float(max(1, len(marker_records))),
            },
            "litho_sampling": {
                "enabled": False,
                "reason": "mainline_raster_only",
                "full_count": int(len(file_list)),
                "selected_count": int(len(file_list)),
            },
            "pattern_coverage": {
                "enabled": False,
                "reason": "mainline_cover_selects_candidates_directly",
            },
        }

    def run(self, input_path: str) -> Dict[str, Any]:
        """主执行流程。

        整体步骤:
        1. 发现输入文件并收集 marker records
        2. exact clustering
        3. 生成每个 exact cluster 的 systematic shift candidates
        4. 评估 candidate coverage
        5. 求解 set cover，得到最终 representative 集合
        6. 物化输出并附带各阶段耗时
        """

        started_at = time.perf_counter()
        input_files = self._discover_input_files(input_path)
        if not input_files:
            raise ValueError("No .oas files found")

        # Step 1: 从每个输入文件中抽取 marker records。
        marker_started = time.perf_counter()
        marker_records: List[MarkerRecord] = []
        for filepath in input_files:
            marker_records.extend(self._collect_marker_records_for_file(filepath))
        marker_elapsed = time.perf_counter() - marker_started

        if not marker_records:
            raise ValueError("No hotspot markers found on the configured hotspot layer")

        # Step 2: 对 marker records 做双重 exact clustering。
        dedup_started = time.perf_counter()
        exact_clusters = self._group_exact_clusters(marker_records)
        dedup_elapsed = time.perf_counter() - dedup_started

        # Step 3: 以每个 exact cluster representative 为原点生成 shift 候选。
        candidate_started = time.perf_counter()
        all_candidates: List[CandidateClip] = []
        for exact_cluster in exact_clusters:
            all_candidates.extend(self._generate_candidates_for_cluster(exact_cluster))
        candidate_elapsed = time.perf_counter() - candidate_started

        # Step 4: 建立 candidate 与 exact cluster 的覆盖关系。
        coverage_started = time.perf_counter()
        self._evaluate_candidate_coverage(all_candidates, exact_clusters)
        coverage_elapsed = time.perf_counter() - coverage_started

        # Step 5: 根据规模自动选择 ILP 或 greedy 做 set cover。
        solver_used = self._select_solver(len(all_candidates))
        cover_started = time.perf_counter()
        if solver_used == "greedy":
            selected_candidates = self._greedy_cover(all_candidates, exact_clusters)
        else:
            selected_candidates = self._exact_cover(all_candidates, exact_clusters)
        cover_elapsed = time.perf_counter() - cover_started

        # Step 6: 写出 sample / representative，并整理统一结果结构。
        return self._build_results(
            marker_records,
            exact_clusters,
            selected_candidates,
            solver_used,
            runtime_summary={
                "collect_markers": round(marker_elapsed, 6),
                "exact_cluster": round(dedup_elapsed, 6),
                "candidate_generation": round(candidate_elapsed, 6),
                "coverage_eval": round(coverage_elapsed, 6),
                "set_cover": round(cover_elapsed, 6),
                "total": round(time.perf_counter() - started_at, 6),
            },
            candidate_count=int(len(all_candidates)),
        )
