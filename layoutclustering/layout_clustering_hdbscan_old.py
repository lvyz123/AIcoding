#!/usr/bin/env python3
"""
半导体Layout中心窗口聚类分析工具
输入：OASIS 文件集合
输出：聚类结果及每个聚类的代表性中心窗口
使用全样本HDBSCAN聚类算法
支持层操作（相减、合并等）

本优化版与原始 layout_clustering.py 的主要区别：
1. 不再用固定网格切 clip，而是先挑选若干中心点，再构造中心矩形和外围上下文窗口。
2. 特征提取时把中心矩形内部与外围上下文分开计算，并给予不同权重。
3. 默认使用“高性能近似几何去重”而不是最重的逐窗口精确布尔裁剪。
4. 将 clip shifting 的“边界对齐”思路合入候选中心生成阶段，并在窗口压缩阶段固定保留 shift-cover 压缩。

为什么默认采用近似方案：
- 在大尺寸 OAS/GDS 上，单个窗口常会与数千个原始 polygon 相交。
- 如果每个候选中心都做精确几何裁剪和增强哈希，通常会导致总耗时过长。
- 因此当前默认策略会先做空间分桶、轻量邻域签名预去重，并限制每个窗口保留的局部几何元素数，
  从而在保留“中心窗口聚类”整体思路的同时，把流程压到可运行范围。

当前 shift-cover 的接入方式：
- 候选中心阶段只做 center micro-shift，不在候选池展开大量 shifted clip。
- 窗口压缩阶段仍固定保留内部的 shift-cover 压缩，用于减少高度可覆盖的重复窗口。
- 这样既保留 paper 的关键思想，又把运行时间控制在工程可接受范围内。

当前默认性能策略：
- candidate_bin_size_um = max(10um, outer_window_size * 2)
- max_elements_per_window = 256
- enable_coarse_prefilter = True
- enable_clip_shifting = True
- clip_shift_neighbor_limit = 128
- clip_shift_boundary_tolerance_um = 0.02

如果更关心几何保真度而不是吞吐量，可以适当减小 --candidate-bin-size-um、
增大 --max-elements-per-window，或调整 --sample-similarity-threshold / --hash-precision-nm 做对比验证。

工程说明：
- 在部分 Windows 环境下，并行特征提取可能因进程池权限问题失败。
- 当前脚本已经加入自动回退逻辑：如果并行创建失败，会退回串行特征提取继续执行。
"""

import atexit
import itertools
import os
import sys
import math
import heapq
from collections import Counter
from contextlib import contextmanager
import numpy as np
import json
import hashlib
import shutil
import uuid
from dataclasses import dataclass, field

# 防止子进程输出干扰
if os.environ.get("LC_SILENT_CHILD", "0") == "1":
    _devnull = open(os.devnull, "w")
    atexit.register(_devnull.close)
    sys.stdout = _devnull
    sys.stderr = _devnull

from rtree import index

from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import gdstk
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.preprocessing import RobustScaler, normalize
from sklearn.feature_selection import VarianceThreshold
from scipy.ndimage import rotate as ndimage_rotate

try:
    from skimage.transform import radon as skimage_radon
except Exception:
    skimage_radon = None

FEATURE_BLOCK_NAMES = ("base", "spatial", "shape", "layer", "pattern", "radon")
DEFAULT_INNER_BLOCK_WEIGHTS = {name: 1.0 for name in FEATURE_BLOCK_NAMES}
DEFAULT_OUTER_BLOCK_WEIGHTS = {
    "base": 0.10,
    "spatial": 0.20,
    "shape": 0.15,
    "layer": 0.35,
    "pattern": 0.30,
    "radon": 0.25,
}
DEFAULT_SEED_KIND = "element"
DEFAULT_POSTMERGE_OUTLIER_THRESHOLD = 0.90
MAX_INTERPRET_SIMILARITY_SAMPLES = 200


def _ascii_safe_token(value: str, fallback: str = "layout") -> str:
    raw = str(value)
    keep = []
    for ch in raw:
        if ch.isascii() and (ch.isalnum() or ch in ("-", "_")):
            keep.append(ch)
        elif ch.isascii() and ch in (" ", ".", "+"):
            keep.append("_")
    token = "".join(keep).strip("._-")
    if not token:
        token = fallback
    token = token[:48]
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"{token}_{digest}"


def _make_ascii_temp_output_path(temp_dir: Path, *, prefix: str, source_path: str,
                                 suffix: str = ".oas", index: Optional[int] = None) -> Path:
    stem = Path(str(source_path)).stem
    stem_token = _ascii_safe_token(stem, fallback="layout")
    filename_parts = [prefix, stem_token]
    if index is not None:
        filename_parts.append(f"{int(index):06d}")
    filename = "_".join(filename_parts) + str(suffix)
    return Path(temp_dir) / filename


def _resolve_fs_path(path_value: Any) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


@contextmanager
def _pushd(directory: Path):
    previous = Path.cwd()
    target = _resolve_fs_path(directory)
    os.chdir(str(target))
    try:
        yield
    finally:
        os.chdir(str(previous))


def _read_oas_only_library(filepath: str):
    ext = os.path.splitext(filepath)[1].lower()
    if ext != ".oas":
        raise ValueError(f"当前版本仅支持 OASIS (.oas) 输入: {filepath}")
    path = _resolve_fs_path(filepath)
    try:
        with _pushd(path.parent):
            return gdstk.read_oas(path.name)
    except OSError as e:
        raise OSError(f"Error opening input file: {path}") from e


def _write_oas_library(lib: gdstk.Library, filepath: str) -> None:
    path = _resolve_fs_path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with _pushd(path.parent):
            lib.write_oas(path.name)
    except OSError as e:
        raise OSError(f"Error opening output file: {path}") from e


def _ensure_oas_input_path(filepath: str) -> None:
    if os.path.splitext(filepath)[1].lower() != ".oas":
        raise ValueError(f"当前版本仅支持 OASIS (.oas) 输入: {filepath}")


def _normalize_feature_block_weights(weights, default_weights=None):
    defaults = dict(default_weights or DEFAULT_OUTER_BLOCK_WEIGHTS)
    normalized = {str(k): float(v) for k, v in defaults.items()}
    if weights is None:
        return normalized

    if isinstance(weights, dict):
        items = weights.items()
    else:
        items = []
        for chunk in str(weights).split(","):
            if "=" not in chunk:
                continue
            key, value = chunk.split("=", 1)
            items.append((key.strip(), value.strip()))

    for key, value in items:
        key = str(key).strip().lower()
        if key not in normalized:
            continue
        try:
            normalized[key] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _normalize_seed_kind(seed_kind: Optional[str]) -> str:
    value = str(seed_kind or DEFAULT_SEED_KIND).strip().lower()
    return value or DEFAULT_SEED_KIND


def _normalize_seed_kind_votes(seed_kind_votes: Optional[Dict[str, int]]) -> Counter:
    votes = Counter()
    for key, value in dict(seed_kind_votes or {}).items():
        try:
            amount = int(value)
        except (TypeError, ValueError):
            continue
        if amount <= 0:
            continue
        votes[_normalize_seed_kind(key)] += amount
    return votes


def _resolve_seed_kind(seed_kind_votes: Optional[Dict[str, int]], fallback: Optional[str] = None) -> str:
    votes = _normalize_seed_kind_votes(seed_kind_votes)
    if not votes:
        return _normalize_seed_kind(fallback)
    ranked = sorted(votes.items(), key=lambda item: (-int(item[1]), str(item[0])))
    if len(ranked) > 1 and int(ranked[0][1]) == int(ranked[1][1]):
        return "mixed"
    return _normalize_seed_kind(ranked[0][0])


def _seed_kind_vote_counter(seed_kind: Optional[str], amount: int = 1) -> Counter:
    votes = Counter()
    votes[_normalize_seed_kind(seed_kind)] = max(1, int(amount))
    return votes


def extract_radon_features(occ_grid: np.ndarray, n_angles: int = 8) -> np.ndarray:
    """对占据栅格做 Radon 风格投影，提取结构方向特征。"""
    occ_grid = np.asarray(occ_grid, dtype=np.float32)
    if occ_grid.ndim != 2 or occ_grid.size == 0:
        return np.zeros(n_angles, dtype=np.float32)

    if float(np.max(occ_grid)) > 0.0:
        occ_grid = occ_grid / float(np.max(occ_grid))

    angles = np.linspace(0.0, 180.0, int(n_angles), endpoint=False)
    if skimage_radon is not None:
        sinogram = skimage_radon(occ_grid, theta=angles, circle=True)
        return np.asarray([np.std(sinogram[:, i]) for i in range(len(angles))], dtype=np.float32)

    # 无 skimage 时退回到旋转投影，避免新增依赖阻塞主流程。
    features = []
    for angle in angles:
        rotated = ndimage_rotate(
            occ_grid,
            angle=float(angle),
            reshape=False,
            order=1,
            mode='constant',
            cval=0.0,
        )
        projection = np.sum(rotated, axis=0)
        features.append(float(np.std(projection)))
    return np.asarray(features, dtype=np.float32)



def _polygon_vertices_array(polygon):
    if isinstance(polygon, gdstk.Polygon):
        return np.asarray(polygon.points, dtype=np.float64)
    return np.asarray(polygon, dtype=np.float64)


def _closed_polygon_edge_lengths(vertices):
    vertices = np.asarray(vertices, dtype=np.float64)
    if len(vertices) < 2:
        return np.empty(0, dtype=np.float64)
    closed = np.vstack((vertices, vertices[0]))
    deltas = np.diff(closed, axis=0)
    return np.linalg.norm(deltas, axis=1)


def polygon_perimeter(polygon):
    """计算多边形周长"""
    vertices = _polygon_vertices_array(polygon)
    return float(np.sum(_closed_polygon_edge_lengths(vertices)))



@dataclass
class LayoutWindowSample:
    """
    中心窗口样本的元数据。

    `center` 是最终用于提取窗口的中心；
    `seed_center` 是原始候选 seed 的中心；
    `center_shift` 记录 clip shifting 风格的中心微调量。
    """

    sample_id: str
    source_name: str
    center: Tuple[float, float]
    inner_bbox: Tuple[float, float, float, float]
    outer_bbox: Tuple[float, float, float, float]
    pattern_hash: str
    invariant_key: Tuple[int, ...]
    duplicate_count: int = 1
    raw_instance_ids: List[int] = field(default_factory=list)
    raw_instance_centers: List[Tuple[float, float]] = field(default_factory=list)
    seed_center: Optional[Tuple[float, float]] = None
    center_shift: Optional[Tuple[float, float]] = None
    seed_kind: str = DEFAULT_SEED_KIND

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_name": self.source_name,
            "center": [float(self.center[0]), float(self.center[1])],
            "seed_center": [float(self.seed_center[0]), float(self.seed_center[1])] if self.seed_center is not None else None,
            "center_shift": [float(self.center_shift[0]), float(self.center_shift[1])] if self.center_shift is not None else None,
            "inner_bbox": [float(v) for v in self.inner_bbox],
            "outer_bbox": [float(v) for v in self.outer_bbox],
            "pattern_hash": self.pattern_hash,
            "invariant_key": [int(v) for v in self.invariant_key],
            "duplicate_count": int(self.duplicate_count),
            "raw_instance_ids": [int(v) for v in self.raw_instance_ids],
            "raw_instance_centers": [[float(x), float(y)] for x, y in self.raw_instance_centers],
            "seed_kind": _normalize_seed_kind(self.seed_kind),
        }


def _safe_bbox_tuple(bbox):
    if bbox is None:
        return None
    try:
        (min_x, min_y), (max_x, max_y) = bbox
        coords = [float(min_x), float(min_y), float(max_x), float(max_y)]
    except Exception:
        return None
    if not all(math.isfinite(v) for v in coords):
        return None
    if coords[0] > coords[2] or coords[1] > coords[3]:
        return None
    return tuple(coords)


def _make_centered_bbox(center_xy, width_um, height_um):
    cx, cy = center_xy
    half_w = float(width_um) / 2.0
    half_h = float(height_um) / 2.0
    return (
        float(cx - half_w),
        float(cy - half_h),
        float(cx + half_w),
        float(cy + half_h),
    )


def _bbox_center(bbox):
    return (
        float((bbox[0] + bbox[2]) / 2.0),
        float((bbox[1] + bbox[3]) / 2.0),
    )


def _geometry_bbox(element):
    """兼容 polygon/path/FlexPath/RobustPath 的 bbox 获取。"""
    if element is None:
        return None

    bbox_fn = getattr(element, "bounding_box", None)
    if callable(bbox_fn):
        try:
            bbox = _safe_bbox_tuple(bbox_fn())
            if bbox is not None:
                return bbox
        except Exception:
            pass

    polygons = None
    to_polygons = getattr(element, "to_polygons", None)
    if callable(to_polygons):
        try:
            polygons = list(to_polygons())
        except Exception:
            polygons = None

    if not polygons:
        polygonset = getattr(element, "to_polygonset", None)
        if callable(polygonset):
            try:
                polyset = polygonset()
                if polyset is not None:
                    polygons = list(getattr(polyset, "polygons", polyset))
            except Exception:
                polygons = None

    if not polygons:
        return None

    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")
    found = False
    for poly in polygons:
        points = np.asarray(getattr(poly, "points", poly), dtype=np.float64)
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 2:
            continue
        found = True
        min_x = min(min_x, float(np.min(points[:, 0])))
        min_y = min(min_y, float(np.min(points[:, 1])))
        max_x = max(max_x, float(np.max(points[:, 0])))
        max_y = max(max_y, float(np.max(points[:, 1])))

    if not found:
        return None
    return (min_x, min_y, max_x, max_y)


def _build_layout_spatial_index(lib):
    """为整版图建立R树索引，并返回可复用的几何元素列表"""
    spatial_index = index.Index()
    indexed_elements = []
    element_counter = 0

    for cell in lib.cells:
        for poly in cell.polygons:
            bbox = _geometry_bbox(poly)
            if bbox is None:
                continue
            spatial_index.insert(element_counter, bbox)
            indexed_elements.append({
                "element": poly,
                "type": "polygon",
                "bbox": bbox,
                "cell_name": cell.name,
            })
            element_counter += 1

        for path in cell.paths:
            bbox = _geometry_bbox(path)
            if bbox is None:
                continue
            spatial_index.insert(element_counter, bbox)
            indexed_elements.append({
                "element": path,
                "type": "path",
                "bbox": bbox,
                "cell_name": cell.name,
            })
            element_counter += 1

    if element_counter == 0:
        return None, [], None
    return spatial_index, indexed_elements, _safe_bbox_tuple(spatial_index.bounds)


def _element_layer_datatype(element):
    return int(getattr(element, "layer", 0)), int(getattr(element, "datatype", 0))


def _bbox_intersection(a, b):
    min_x = max(float(a[0]), float(b[0]))
    min_y = max(float(a[1]), float(b[1]))
    max_x = min(float(a[2]), float(b[2]))
    max_y = min(float(a[3]), float(b[3]))
    if min_x >= max_x or min_y >= max_y:
        return None
    return (min_x, min_y, max_x, max_y)


def _rect_polygon_from_bbox(bbox, layer=0, datatype=0):
    poly = gdstk.rectangle((bbox[0], bbox[1]), (bbox[2], bbox[3]), layer=int(layer), datatype=int(datatype))
    return poly


def _approx_clip_indexed_elements_to_bbox(spatial_index, indexed_elements, bbox, center_xy=None, max_elements=None):
    """使用 bbox proxy 近似裁剪 R 树命中的元素到指定矩形区域，返回 polygon 列表。"""
    polygons = []
    candidate_ids = _select_relevant_element_ids(
        spatial_index,
        indexed_elements,
        bbox,
        center_xy=center_xy,
        max_elements=max_elements,
    )

    for elem_id in candidate_ids:
        item = indexed_elements[elem_id]
        clipped_bbox = _bbox_intersection(item["bbox"], bbox)
        if clipped_bbox is None:
            continue
        layer, datatype = _element_layer_datatype(item["element"])
        polygons.append(_rect_polygon_from_bbox(clipped_bbox, layer=layer, datatype=datatype))
    return polygons


def _approx_clip_polygons_with_bbox(polygons, bbox, operation):
    clipped = []
    for poly in polygons:
        poly_bbox_raw = poly.bounding_box()
        poly_bbox = _safe_bbox_tuple(poly_bbox_raw)
        if poly_bbox is None:
            continue
        if operation == "and":
            clipped_bbox = _bbox_intersection(poly_bbox, bbox)
            if clipped_bbox is None:
                continue
            clipped.append(_rect_polygon_from_bbox(clipped_bbox, layer=getattr(poly, "layer", 0), datatype=getattr(poly, "datatype", 0)))
        elif operation == "not":
            inner = _bbox_intersection(poly_bbox, bbox)
            if inner is None:
                clipped.append(_rect_polygon_from_bbox(poly_bbox, layer=getattr(poly, "layer", 0), datatype=getattr(poly, "datatype", 0)))
                continue
            x0, y0, x1, y1 = poly_bbox
            ix0, iy0, ix1, iy1 = inner
            layer = getattr(poly, "layer", 0)
            datatype = getattr(poly, "datatype", 0)
            pieces = [
                (x0, y0, ix0, y1),
                (ix1, y0, x1, y1),
                (ix0, y0, ix1, iy0),
                (ix0, iy1, ix1, y1),
            ]
            for piece in pieces:
                if piece[0] < piece[2] and piece[1] < piece[3]:
                    clipped.append(_rect_polygon_from_bbox(piece, layer=layer, datatype=datatype))
    return clipped


def _polygon_area_from_points(points):
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _apply_dihedral_transform(points: np.ndarray, transform_id: int) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    if transform_id == 0:
        tx, ty = x, y
    elif transform_id == 1:
        tx, ty = -y, x
    elif transform_id == 2:
        tx, ty = -x, -y
    elif transform_id == 3:
        tx, ty = y, -x
    elif transform_id == 4:
        tx, ty = -x, y
    elif transform_id == 5:
        tx, ty = -y, -x
    elif transform_id == 6:
        tx, ty = x, -y
    else:
        tx, ty = y, x
    return np.column_stack((tx, ty))


def _encode_point_sequence(point_seq):
    n = len(point_seq)
    if n == 0:
        return ""
    parts = []
    for i in range(n):
        x1, y1 = point_seq[i]
        x2, y2 = point_seq[(i + 1) % n]
        parts.append(f"{x1},{y1};{x2 - x1},{y2 - y1}")
    return "|".join(parts)


def _canonicalize_polygon_points(points_q: np.ndarray) -> str:
    seq = [tuple(int(v) for v in row) for row in points_q.tolist()]
    if not seq:
        return ""
    candidates = []
    for current in (seq, list(reversed(seq))):
        for shift in range(len(current)):
            rotated = current[shift:] + current[:shift]
            candidates.append(_encode_point_sequence(rotated))
    return min(candidates)


def _enhanced_window_hash(polygons, quant_step_um=0.005):
    point_sets = [np.asarray(poly.points, dtype=np.float64) for poly in polygons if poly is not None and len(poly.points) >= 3]
    if not point_sets:
        return hashlib.sha256(b"empty").hexdigest(), "empty"

    payloads = []
    for transform_id in range(8):
        transformed_sets = [_apply_dihedral_transform(points, transform_id) for points in point_sets]
        all_points = np.vstack(transformed_sets)
        shift = np.min(all_points, axis=0)
        encoded = []
        for points in transformed_sets:
            quantized = np.rint((points - shift) / quant_step_um).astype(np.int64)
            canonical = _canonicalize_polygon_points(quantized)
            area_q = int(round(_polygon_area_from_points(quantized)))
            encoded.append((area_q, len(quantized), canonical))
        encoded.sort(key=lambda item: (item[0], item[1], item[2]))
        payloads.append("#".join(item[2] for item in encoded))

    best_payload = min(payloads)
    return hashlib.sha256(best_payload.encode("utf-8")).hexdigest(), best_payload


def _compute_window_invariants(polygons):
    point_sets = []
    total_area = 0.0
    total_perimeter = 0.0
    vertex_count = 0

    for poly in polygons:
        if poly is None or len(poly.points) < 3:
            continue
        pts = np.asarray(poly.points, dtype=np.float64)
        point_sets.append(pts)
        total_area += abs(float(poly.area()))
        total_perimeter += float(polygon_perimeter(pts))
        vertex_count += int(len(pts))

    if not point_sets:
        return np.zeros(9, dtype=np.float64)

    all_points = np.vstack(point_sets)
    min_coord = np.min(all_points, axis=0)
    max_coord = np.max(all_points, axis=0)
    width = float(max_coord[0] - min_coord[0])
    height = float(max_coord[1] - min_coord[1])
    bbox_long = max(width, height)
    bbox_short = min(width, height)
    centroid = np.mean(all_points, axis=0)
    radii = np.linalg.norm(all_points - centroid, axis=1)
    density = total_area / max(width * height, 1e-12)

    return np.array([
        float(len(point_sets)),
        float(total_area),
        float(total_perimeter),
        float(bbox_long),
        float(bbox_short),
        float(vertex_count),
        float(np.mean(radii)) if len(radii) > 0 else 0.0,
        float(np.std(radii)) if len(radii) > 0 else 0.0,
        float(density),
    ], dtype=np.float64)


def _quantize_window_invariants(invariants, quant_step_um):
    q = max(float(quant_step_um), 1e-6)
    steps = np.array([
        1.0,
        q * q,
        q * 4.0,
        q * 2.0,
        q * 2.0,
        1.0,
        q,
        q,
        0.02,
    ], dtype=np.float64)
    return tuple(np.rint(np.asarray(invariants, dtype=np.float64) / steps).astype(np.int64).tolist())


def _stable_invariant_bucket(invariants, relative_tolerance=0.08):
    """
    为几何去重构建稳定的粗分桶键。

    注意这里故意不直接复用 hash_precision_nm 对应的绝对量化步长：
    当窗口经过 boolean 裁剪后，面积/周长/bbox 会带有微小浮点扰动；
    如果仍然用极细的绝对步长做分桶，会把本应进入“相似窗口比较”阶段的样本
    提前拆散到不同桶里，导致去重失效。
    """
    inv = np.asarray(invariants, dtype=np.float64)
    rel = max(float(relative_tolerance), 1e-6)
    floors = np.array([
        1.0,
        5e-4,
        5e-2,
        2e-2,
        2e-2,
        1.0,
        1e-2,
        1e-2,
        5e-2,
    ], dtype=np.float64)
    scales = np.maximum(np.abs(inv) * rel, floors)
    return tuple(np.rint(inv / scales).astype(np.int64).tolist())


def _neighboring_stable_bucket_keys(bucket_key, leading_dims=3):
    base_key = tuple(int(v) for v in bucket_key)
    if not base_key:
        return [tuple()]

    dims = min(max(int(leading_dims), 0), len(base_key))
    if dims == 0:
        return [base_key]

    keys = []
    seen = set()
    for offsets in itertools.product((-1, 0, 1), repeat=dims):
        candidate = list(base_key)
        for dim, offset in enumerate(offsets):
            candidate[dim] += int(offset)
        candidate_key = tuple(candidate)
        if candidate_key not in seen:
            seen.add(candidate_key)
            keys.append(candidate_key)
    return keys


def _invariant_relative_distance(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-9)
    return float(np.max(np.abs(a - b) / denom))


def _window_signature(polygons, bbox, n_bins=12):
    point_sets = [np.asarray(poly.points, dtype=np.float64) for poly in polygons if poly is not None and len(poly.points) > 0]
    if not point_sets:
        return np.zeros(n_bins * n_bins, dtype=np.float32)

    min_x, min_y, max_x, max_y = bbox
    if max_x <= min_x or max_y <= min_y:
        return np.zeros(n_bins * n_bins, dtype=np.float32)

    all_points = np.vstack(point_sets)
    hist, _, _ = np.histogram2d(
        all_points[:, 0],
        all_points[:, 1],
        bins=(n_bins, n_bins),
        range=[[min_x, max_x], [min_y, max_y]],
    )
    total = float(np.sum(hist))
    if total > 0:
        hist = hist / total
    return hist.astype(np.float32).flatten()


def _cosine_similarity_1d(vec_a, vec_b):
    a = np.asarray(vec_a, dtype=np.float64).ravel()
    b = np.asarray(vec_b, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _coarse_window_descriptor(center_xy, outer_bbox, spatial_index, indexed_elements,
                              quant_step_um=0.02, max_neighbors=256):
    """
    使用邻域元素的相对bbox构建轻量级几何描述符。

    这一步不做boolean裁剪，只用于在精确窗口提取前做快速预去重。
    """
    cx, cy = center_xy
    neighbor_ids = list(spatial_index.intersection(outer_bbox))
    if not neighbor_ids:
        return "empty"

    parts = []
    q = max(float(quant_step_um), 1e-6)
    ox0, oy0, ox1, oy1 = outer_bbox
    max_radius_x = max((ox1 - ox0) / 2.0, q)
    max_radius_y = max((oy1 - oy0) / 2.0, q)

    for elem_id in neighbor_ids[:max_neighbors]:
        item = indexed_elements[elem_id]
        min_x, min_y, max_x, max_y = item["bbox"]
        layer, datatype = _element_layer_datatype(item["element"])
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
        parts.append((layer, datatype, bbox_w, bbox_h, nx, ny, rel))

    parts.sort()
    payload = "|".join(
        f"{layer}:{datatype}:{bw}:{bh}:{nx}:{ny}:{r0}:{r1}:{r2}:{r3}"
        for layer, datatype, bw, bh, nx, ny, (r0, r1, r2, r3) in parts
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _select_relevant_element_ids(spatial_index, indexed_elements, bbox, center_xy=None, max_elements=None):
    candidate_ids = list(spatial_index.intersection(bbox))
    if max_elements is None or len(candidate_ids) <= int(max_elements):
        return candidate_ids

    cx, cy = center_xy if center_xy is not None else _bbox_center(bbox)

    def _score(elem_id):
        item_bbox = indexed_elements[elem_id]["bbox"]
        inter = _bbox_intersection(item_bbox, bbox)
        overlap = 0.0 if inter is None else max(0.0, (inter[2] - inter[0]) * (inter[3] - inter[1]))
        icx, icy = _bbox_center(item_bbox)
        dist2 = (icx - cx) ** 2 + (icy - cy) ** 2
        return (-overlap, dist2, elem_id)

    return sorted(candidate_ids, key=_score)[:int(max_elements)]


def _select_candidate_element_ids(indexed_elements, bin_size_um, window_size_um=None):
    """按空间粗分桶为中心点挑选代表元素，优先保留更适合局部窗口的几何。"""
    if not indexed_elements:
        return []

    step = max(float(bin_size_um), 1e-6)
    target_span = max(float(window_size_um) if window_size_um is not None else step, 1e-6)
    bin_map = {}
    for elem_id, item in enumerate(indexed_elements):
        cx, cy = _bbox_center(item["bbox"])
        key = (int(math.floor(cx / step)), int(math.floor(cy / step)))
        bbox = item["bbox"]
        width = max(0.0, float(bbox[2] - bbox[0]))
        height = max(0.0, float(bbox[3] - bbox[1]))
        area = width * height
        max_span = max(width, height)
        min_span = min(width, height)
        oversize = max(0.0, max_span - target_span)
        fit_delta = abs(max_span - target_span * 0.5)
        compactness = min_span / max(max_span, 1e-6)
        score = (
            -oversize,
            -fit_delta,
            compactness,
            area,
            -elem_id,
        )
        current = bin_map.get(key)
        if current is None or score > current[0]:
            bin_map[key] = (score, elem_id)

    selected = sorted(v[1] for v in bin_map.values())
    return selected


def _bbox_gap_distance(bbox_a, bbox_b):
    dx = max(float(bbox_a[0]) - float(bbox_b[2]), float(bbox_b[0]) - float(bbox_a[2]), 0.0)
    dy = max(float(bbox_a[1]) - float(bbox_b[3]), float(bbox_b[1]) - float(bbox_a[3]), 0.0)
    return float(math.hypot(dx, dy))


def _generate_relation_seed_specs(seed_ids, indexed_elements, spatial_index,
                                  window_size_um, gap_threshold_um=0.08,
                                  relation_seed_ratio=0.2):
    if not seed_ids:
        return []

    budget = int(math.ceil(len(seed_ids) * max(0.0, float(relation_seed_ratio)) * 0.5))
    if budget <= 0:
        return []

    scan_side = max(float(window_size_um) * 2.0, float(gap_threshold_um) * 8.0, 1e-6)
    pair_records = []
    seen_pairs = set()

    for elem_id in seed_ids:
        seed_bbox = indexed_elements[elem_id]["bbox"]
        seed_center = _bbox_center(seed_bbox)
        scan_bbox = _make_centered_bbox(seed_center, scan_side, scan_side)
        neighbor_ids = _select_relevant_element_ids(
            spatial_index,
            indexed_elements,
            scan_bbox,
            center_xy=seed_center,
            max_elements=24,
        )
        best_record = None
        best_pair_key = None
        for nid in neighbor_ids:
            if nid == elem_id:
                continue
            pair_key = tuple(sorted((int(elem_id), int(nid))))
            if pair_key in seen_pairs:
                continue
            neighbor_bbox = indexed_elements[nid]["bbox"]
            gap = _bbox_gap_distance(seed_bbox, neighbor_bbox)
            if gap > float(gap_threshold_um):
                continue
            other_center = _bbox_center(neighbor_bbox)
            center_dist = math.hypot(seed_center[0] - other_center[0], seed_center[1] - other_center[1])
            union_bbox = (
                min(float(seed_bbox[0]), float(neighbor_bbox[0])),
                min(float(seed_bbox[1]), float(neighbor_bbox[1])),
                max(float(seed_bbox[2]), float(neighbor_bbox[2])),
                max(float(seed_bbox[3]), float(neighbor_bbox[3])),
            )
            midpoint = ((seed_center[0] + other_center[0]) * 0.5, (seed_center[1] + other_center[1]) * 0.5)
            score = (-(gap), -(center_dist), -pair_key[1])
            record = {
                "elem_id": int(elem_id),
                "seed_center": (float(midpoint[0]), float(midpoint[1])),
                "seed_bbox": union_bbox,
                "score": score,
            }
            if best_record is None or record["score"] > best_record["score"]:
                best_record = record
                best_pair_key = pair_key
        if best_record is not None:
            seen_pairs.add(best_pair_key)
            pair_records.append(best_record)

    pair_records.sort(key=lambda item: item["score"], reverse=True)
    return pair_records[:budget]


def _generate_hotspot_seed_specs(seed_ids, indexed_elements, window_size_um,
                                 relation_seed_ratio=0.2):
    if not seed_ids:
        return []

    budget = int(math.ceil(len(seed_ids) * max(0.0, float(relation_seed_ratio)) * 0.5))
    if budget <= 0:
        return []

    hotspot_records = []
    for elem_id in seed_ids:
        bbox = indexed_elements[elem_id]["bbox"]
        cx, cy = _bbox_center(bbox)
        width = max(0.0, float(bbox[2] - bbox[0]))
        height = max(0.0, float(bbox[3] - bbox[1]))
        max_span = max(width, height)
        min_span = min(width, height)
        if max_span <= max(float(window_size_um) * 0.4, 1e-6):
            continue

        aspect_ratio = max_span / max(min_span, 1e-6)
        hotspot_points = []
        if aspect_ratio >= 3.0:
            if width >= height:
                hotspot_points = [(float(bbox[0]), cy), (float(bbox[2]), cy)]
            else:
                hotspot_points = [(cx, float(bbox[1])), (cx, float(bbox[3]))]
        else:
            hotspot_points = [
                (float(bbox[0]), float(bbox[1])),
                (float(bbox[0]), float(bbox[3])),
                (float(bbox[2]), float(bbox[1])),
                (float(bbox[2]), float(bbox[3])),
            ]

        score = (aspect_ratio, max_span / max(float(window_size_um), 1e-6), -int(elem_id))
        for hotspot_center in hotspot_points[:2 if aspect_ratio >= 3.0 else 4]:
            hotspot_records.append({
                "elem_id": int(elem_id),
                "seed_center": (float(hotspot_center[0]), float(hotspot_center[1])),
                "seed_bbox": bbox,
                "score": score,
            })

    hotspot_records.sort(key=lambda item: item["score"], reverse=True)
    selected = []
    seen_keys = set()
    quant = max(float(window_size_um) * 0.05, 1e-6)
    for record in hotspot_records:
        key = (
            int(round(record["seed_center"][0] / quant)),
            int(round(record["seed_center"][1] / quant)),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        selected.append(record)
        if len(selected) >= budget:
            break
    return selected


def _dedup_candidate_entries(candidate_entries, quant_step_um=0.01):
    if not candidate_entries:
        return []
    deduped = []
    seen = set()
    q = max(float(quant_step_um), 1e-6)
    for candidate in candidate_entries:
        center = candidate["center"]
        key = (int(round(center[0] / q)), int(round(center[1] / q)))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _normalize_shift_directions(shift_directions):
    if shift_directions is None:
        return ("left", "right", "up", "down")
    if isinstance(shift_directions, str):
        items = [v.strip().lower() for v in shift_directions.split(",") if v.strip()]
    else:
        items = [str(v).strip().lower() for v in shift_directions if str(v).strip()]
    valid = [v for v in items if v in {"left", "right", "up", "down"}]
    return tuple(valid) if valid else ("left", "right", "up", "down")


def _axis_shift_interval(seed_bbox, seed_center, half_span, axis, shift_directions):
    cx, cy = seed_center
    dirs = set(_normalize_shift_directions(shift_directions))
    if axis == "x":
        low = float(seed_bbox[2] - (cx + half_span))
        high = float(seed_bbox[0] - (cx - half_span))
        if "left" not in dirs:
            low = max(low, 0.0)
        if "right" not in dirs:
            high = min(high, 0.0)
    else:
        low = float(seed_bbox[3] - (cy + half_span))
        high = float(seed_bbox[1] - (cy - half_span))
        if "down" not in dirs:
            low = max(low, 0.0)
        if "up" not in dirs:
            high = min(high, 0.0)
    if low > high:
        return None
    return low, high


def _collect_axis_shift_values(center_coord, half_span, edge_values, shift_interval, max_count=12):
    if shift_interval is None:
        return [0.0]

    low, high = shift_interval
    values = {0.0, float(low), float(high)}
    left_boundary = center_coord - half_span
    right_boundary = center_coord + half_span

    for edge in edge_values:
        s1 = float(edge - left_boundary)
        s2 = float(edge - right_boundary)
        if low <= s1 <= high:
            values.add(round(s1, 6))
        if low <= s2 <= high:
            values.add(round(s2, 6))

    ordered = sorted(values, key=lambda v: (abs(v), v))
    if max_count is not None and len(ordered) > int(max_count):
        return ordered[:int(max_count)]
    return ordered


def _score_axis_shift(center_coord, shift_value, half_span, edge_values, tolerance_um):
    left_boundary = center_coord + shift_value - half_span
    right_boundary = center_coord + shift_value + half_span
    touch_count = 0
    best_gap = float("inf")
    for edge in edge_values:
        gap = min(abs(edge - left_boundary), abs(edge - right_boundary))
        if gap <= tolerance_um:
            touch_count += 1
        if gap < best_gap:
            best_gap = gap
    if not edge_values:
        best_gap = float("inf")
    return touch_count, -best_gap, -abs(shift_value)


def _choose_axis_shift(center_coord, half_span, edge_values, shift_values, tolerance_um):
    if not shift_values:
        return 0.0

    zero_score = _score_axis_shift(center_coord, 0.0, half_span, edge_values, tolerance_um)
    best_shift = max(
        shift_values,
        key=lambda s: _score_axis_shift(center_coord, s, half_span, edge_values, tolerance_um),
    )
    best_score = _score_axis_shift(center_coord, best_shift, half_span, edge_values, tolerance_um)
    zero_touch, zero_neg_gap, _ = zero_score
    best_touch, best_neg_gap, _ = best_score
    zero_gap = -zero_neg_gap
    best_gap = -best_neg_gap

    if abs(best_shift) <= 1e-9:
        return 0.0
    if best_touch <= 0:
        return 0.0
    if best_touch < zero_touch:
        return 0.0
    if best_touch == zero_touch and best_gap >= zero_gap - max(float(tolerance_um) * 0.5, 1e-6):
        return 0.0
    return float(best_shift)


def _build_shift_anchor_bbox(seed_bbox, seed_center, window_size_um, anchor_scale=0.5):
    """
    为 clip shifting 构造局部 anchor bbox。

    直接使用完整 seed bbox 时，若元素跨度远大于窗口尺寸，平移范围会退化为 0。
    这里退回到 seed 中心附近的局部 marker，尽量贴近论文里 marker/clip 的关系。
    """
    sx = max(0.0, float(seed_bbox[2] - seed_bbox[0]))
    sy = max(0.0, float(seed_bbox[3] - seed_bbox[1]))
    target_span = max(float(window_size_um) * float(anchor_scale), float(window_size_um) * 0.1, 1e-6)
    anchor_w = min(sx, target_span) if sx > 0 else target_span
    anchor_h = min(sy, target_span) if sy > 0 else target_span
    return _make_centered_bbox(seed_center, anchor_w, anchor_h)


def _compute_shifted_center_for_seed(spatial_index, indexed_elements, elem_id,
                                     window_size_um, shift_directions,
                                     neighbor_limit=128, boundary_tolerance_um=0.02,
                                     max_shift_values_per_axis=12,
                                     seed_center_override=None,
                                     seed_bbox_override=None):
    seed_item = indexed_elements[elem_id]
    seed_bbox = seed_bbox_override if seed_bbox_override is not None else seed_item["bbox"]
    seed_center = tuple(float(v) for v in (
        seed_center_override if seed_center_override is not None else _bbox_center(seed_bbox)
    ))
    anchor_bbox = _build_shift_anchor_bbox(seed_bbox, seed_center, window_size_um)
    half_span = float(window_size_um) / 2.0

    scan_side = max(float(window_size_um) * 2.0, 1e-6)
    scan_bbox = _make_centered_bbox(seed_center, scan_side, scan_side)
    neighbor_ids = _select_relevant_element_ids(
        spatial_index,
        indexed_elements,
        scan_bbox,
        center_xy=seed_center,
        max_elements=max(8, int(neighbor_limit)),
    )

    x_edges = []
    y_edges = []
    for nid in neighbor_ids:
        bbox = indexed_elements[nid]["bbox"]
        x_edges.extend([float(bbox[0]), float(bbox[2])])
        y_edges.extend([float(bbox[1]), float(bbox[3])])

    x_interval = _axis_shift_interval(anchor_bbox, seed_center, half_span, "x", shift_directions)
    y_interval = _axis_shift_interval(anchor_bbox, seed_center, half_span, "y", shift_directions)

    x_shifts = _collect_axis_shift_values(
        seed_center[0],
        half_span,
        x_edges,
        x_interval,
        max_count=max_shift_values_per_axis,
    )
    y_shifts = _collect_axis_shift_values(
        seed_center[1],
        half_span,
        y_edges,
        y_interval,
        max_count=max_shift_values_per_axis,
    )

    best_dx = 0.0
    best_dy = 0.0
    if x_shifts:
        best_dx = _choose_axis_shift(
            seed_center[0],
            half_span,
            x_edges,
            x_shifts,
            boundary_tolerance_um,
        )
    if y_shifts:
        best_dy = _choose_axis_shift(
            seed_center[1],
            half_span,
            y_edges,
            y_shifts,
            boundary_tolerance_um,
        )

    shifted_center = (float(seed_center[0] + best_dx), float(seed_center[1] + best_dy))
    return {
        "elem_id": int(elem_id),
        "seed_center": (float(seed_center[0]), float(seed_center[1])),
        "center": shifted_center,
        "center_shift": (float(best_dx), float(best_dy)),
    }


def _select_candidate_centers(indexed_elements, spatial_index, window_size_um,
                              candidate_bin_size_um=None,
                              relation_seed_ratio=0.2,
                              relation_gap_threshold_um=0.08,
                              enable_clip_shifting=True,
                              clip_shift_directions=("left", "right", "up", "down"),
                              clip_shift_neighbor_limit=128,
                              clip_shift_boundary_tolerance_um=0.02):
    outer_candidate_bin = float(candidate_bin_size_um) if candidate_bin_size_um is not None else max(float(window_size_um) * 2.0, 10.0, 1e-6)
    element_seed_ids = _select_candidate_element_ids(
        indexed_elements,
        outer_candidate_bin,
        window_size_um=window_size_um,
    )
    relation_seed_specs = _generate_relation_seed_specs(
        element_seed_ids,
        indexed_elements,
        spatial_index,
        window_size_um=window_size_um,
        gap_threshold_um=relation_gap_threshold_um,
        relation_seed_ratio=relation_seed_ratio,
    )
    hotspot_seed_specs = _generate_hotspot_seed_specs(
        element_seed_ids,
        indexed_elements,
        window_size_um=window_size_um,
        relation_seed_ratio=relation_seed_ratio,
    )

    if not enable_clip_shifting:
        candidates = [
            {
                "elem_id": int(elem_id),
                "seed_center": _bbox_center(indexed_elements[elem_id]["bbox"]),
                "center": _bbox_center(indexed_elements[elem_id]["bbox"]),
                "center_shift": (0.0, 0.0),
                "seed_kind": "element",
            }
            for elem_id in element_seed_ids
        ]
        for spec in relation_seed_specs:
            center = tuple(float(v) for v in spec["seed_center"])
            candidates.append({
                "elem_id": int(spec["elem_id"]),
                "seed_center": center,
                "center": center,
                "center_shift": (0.0, 0.0),
                "seed_kind": "relation",
            })
        for spec in hotspot_seed_specs:
            center = tuple(float(v) for v in spec["seed_center"])
            candidates.append({
                "elem_id": int(spec["elem_id"]),
                "seed_center": center,
                "center": center,
                "center_shift": (0.0, 0.0),
                "seed_kind": "hotspot",
            })
        candidates = _dedup_candidate_entries(candidates, quant_step_um=max(outer_candidate_bin * 0.02, 0.01))
        return candidates, {
            "initial_seed_count": int(len(candidates)),
            "element_seed_count": int(len(element_seed_ids)),
            "relation_seed_count": int(len(relation_seed_specs)),
            "hotspot_seed_count": int(len(hotspot_seed_specs)),
            "relation_seed_ratio": float(relation_seed_ratio),
            "relation_gap_threshold_um": float(relation_gap_threshold_um),
            "shifted_seed_count": 0,
            "candidate_bin_size_um": float(outer_candidate_bin),
            "clip_shifting_enabled": False,
        }

    candidates = []
    shifted_seed_count = 0
    for elem_id in element_seed_ids:
        candidate = _compute_shifted_center_for_seed(
            spatial_index,
            indexed_elements,
            elem_id,
            window_size_um=window_size_um,
            shift_directions=clip_shift_directions,
            neighbor_limit=clip_shift_neighbor_limit,
            boundary_tolerance_um=clip_shift_boundary_tolerance_um,
        )
        if abs(candidate["center_shift"][0]) > 1e-9 or abs(candidate["center_shift"][1]) > 1e-9:
            shifted_seed_count += 1
        candidate["seed_kind"] = "element"
        candidates.append(candidate)

    for spec in relation_seed_specs:
        candidate = _compute_shifted_center_for_seed(
            spatial_index,
            indexed_elements,
            spec["elem_id"],
            window_size_um=window_size_um,
            shift_directions=clip_shift_directions,
            neighbor_limit=clip_shift_neighbor_limit,
            boundary_tolerance_um=clip_shift_boundary_tolerance_um,
            seed_center_override=spec["seed_center"],
            seed_bbox_override=spec.get("seed_bbox"),
        )
        candidate["seed_kind"] = "relation"
        if abs(candidate["center_shift"][0]) > 1e-9 or abs(candidate["center_shift"][1]) > 1e-9:
            shifted_seed_count += 1
        candidates.append(candidate)

    for spec in hotspot_seed_specs:
        candidate = _compute_shifted_center_for_seed(
            spatial_index,
            indexed_elements,
            spec["elem_id"],
            window_size_um=window_size_um,
            shift_directions=clip_shift_directions,
            neighbor_limit=clip_shift_neighbor_limit,
            boundary_tolerance_um=clip_shift_boundary_tolerance_um,
            seed_center_override=spec["seed_center"],
            seed_bbox_override=spec.get("seed_bbox"),
        )
        candidate["seed_kind"] = "hotspot"
        if abs(candidate["center_shift"][0]) > 1e-9 or abs(candidate["center_shift"][1]) > 1e-9:
            shifted_seed_count += 1
        candidates.append(candidate)

    candidates = _dedup_candidate_entries(candidates, quant_step_um=max(outer_candidate_bin * 0.02, 0.01))
    dedup_shifted_seed_count = sum(
        1
        for candidate in candidates
        if abs(candidate["center_shift"][0]) > 1e-9 or abs(candidate["center_shift"][1]) > 1e-9
    )

    return candidates, {
        "initial_seed_count": int(len(candidates)),
        "element_seed_count": int(len(element_seed_ids)),
        "relation_seed_count": int(len(relation_seed_specs)),
        "hotspot_seed_count": int(len(hotspot_seed_specs)),
        "relation_seed_ratio": float(relation_seed_ratio),
        "relation_gap_threshold_um": float(relation_gap_threshold_um),
        "shifted_seed_count": int(dedup_shifted_seed_count),
        "candidate_bin_size_um": float(outer_candidate_bin),
        "clip_shifting_enabled": True,
        "clip_shift_directions": list(_normalize_shift_directions(clip_shift_directions)),
        "clip_shift_neighbor_limit": int(clip_shift_neighbor_limit),
        "clip_shift_boundary_tolerance_um": float(clip_shift_boundary_tolerance_um),
    }


def _polygon_list_area(polygons):
    total_area = 0.0
    for poly in polygons or []:
        if poly is None:
            continue
        try:
            total_area += abs(float(poly.area()))
        except Exception:
            bbox = _safe_bbox_tuple(poly.bounding_box()) if hasattr(poly, "bounding_box") else None
            if bbox is not None:
                total_area += max(0.0, float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))
    return float(total_area)


def _normalize_polygons_to_local_bbox(polygons, bbox):
    min_x, min_y = float(bbox[0]), float(bbox[1])
    normalized = []
    for poly in polygons or []:
        if poly is None or not hasattr(poly, "points"):
            continue
        pts = np.asarray(poly.points, dtype=np.float64)
        if pts.size == 0:
            continue
        shifted = pts - np.array([min_x, min_y], dtype=np.float64)
        normalized.append(
            gdstk.Polygon(
                shifted,
                layer=int(getattr(poly, "layer", 0)),
                datatype=int(getattr(poly, "datatype", 0)),
            )
        )
    return normalized


def _window_xor_ratio(record_a, record_b):
    window_area = max(
        float(record_a.get("window_area", 0.0)),
        float(record_b.get("window_area", 0.0)),
        1e-12,
    )
    polygons_a = record_a.get("normalized_polygons", [])
    polygons_b = record_b.get("normalized_polygons", [])
    try:
        xor_polygons = gdstk.boolean(polygons_a, polygons_b, "xor")
        xor_area = _polygon_list_area(xor_polygons)
    except Exception:
        try:
            inter_polygons = gdstk.boolean(polygons_a, polygons_b, "and")
            inter_area = _polygon_list_area(inter_polygons)
        except Exception:
            inter_area = 0.0
        area_a = _polygon_list_area(polygons_a)
        area_b = _polygon_list_area(polygons_b)
        xor_area = max(0.0, area_a + area_b - 2.0 * inter_area)
    return float(np.clip(xor_area / window_area, 0.0, 1.0))


def _build_window_record(sample, outer_polygons, invariants, signature, pattern_hash, invariant_key, **extra_meta):
    outer_bbox = tuple(float(v) for v in sample.outer_bbox)
    window_area = max(
        1e-12,
        float((outer_bbox[2] - outer_bbox[0]) * (outer_bbox[3] - outer_bbox[1])),
    )
    record = {
        "sample": sample,
        "outer_polygons": list(outer_polygons),
        "invariants": np.asarray(invariants, dtype=np.float64),
        "signature": np.asarray(signature, dtype=np.float32),
        "pattern_hash": str(pattern_hash),
        "invariant_key": tuple(int(v) for v in invariant_key),
        "stable_bucket_key": tuple(
            int(v) for v in extra_meta.pop(
                "stable_bucket_key",
                _stable_invariant_bucket(invariants, relative_tolerance=0.08),
            )
        ),
        "normalized_polygons": _normalize_polygons_to_local_bbox(outer_polygons, outer_bbox),
        "window_area": float(window_area),
        "seed_kind_votes": _normalize_seed_kind_votes(
            extra_meta.pop("seed_kind_votes", _seed_kind_vote_counter(sample.seed_kind))
        ),
    }
    record.update(extra_meta)
    return record


def _sync_record_seed_kind(record):
    sample = record["sample"]
    sample.seed_kind = _resolve_seed_kind(
        record.get("seed_kind_votes"),
        fallback=getattr(sample, "seed_kind", DEFAULT_SEED_KIND),
    )
    return sample.seed_kind


def _build_shift_candidate_record(origin_idx, origin_record, center, spatial_index, indexed_elements,
                                  window_size_um, context_width_um, max_elements_per_window,
                                  quant_step_um, signature_bins):
    center = (float(center[0]), float(center[1]))
    inner_bbox = _make_centered_bbox(center, window_size_um, window_size_um)
    outer_side = float(window_size_um) + 2.0 * float(context_width_um)
    outer_bbox = _make_centered_bbox(center, outer_side, outer_side)
    outer_polygons = _approx_clip_indexed_elements_to_bbox(
        spatial_index,
        indexed_elements,
        outer_bbox,
        center_xy=center,
        max_elements=max_elements_per_window,
    )
    if not outer_polygons:
        return None

    inner_polygons = _approx_clip_polygons_with_bbox(outer_polygons, inner_bbox, "and")
    if not inner_polygons:
        return None

    invariants = _compute_window_invariants(outer_polygons)
    invariant_key = _quantize_window_invariants(invariants, quant_step_um)
    pattern_hash, _ = _enhanced_window_hash(outer_polygons, quant_step_um=quant_step_um)
    signature = _window_signature(outer_polygons, outer_bbox, n_bins=max(4, int(signature_bins)))
    seed_center = origin_record["sample"].seed_center or origin_record["sample"].center

    return {
        "origin_idx": int(origin_idx),
        "center": center,
        "seed_center": (float(seed_center[0]), float(seed_center[1])),
        "center_shift": (
            float(center[0] - seed_center[0]),
            float(center[1] - seed_center[1]),
        ),
        "inner_bbox": tuple(float(v) for v in inner_bbox),
        "outer_bbox": tuple(float(v) for v in outer_bbox),
        "outer_polygons": list(outer_polygons),
        "invariants": np.asarray(invariants, dtype=np.float64),
        "signature": np.asarray(signature, dtype=np.float32),
        "pattern_hash": str(pattern_hash),
        "invariant_key": tuple(int(v) for v in invariant_key),
        "stable_bucket_key": tuple(
            int(v) for v in _stable_invariant_bucket(invariants, relative_tolerance=0.08)
        ),
        "normalized_polygons": _normalize_polygons_to_local_bbox(outer_polygons, outer_bbox),
        "window_area": max(
            1e-12,
            float((outer_bbox[2] - outer_bbox[0]) * (outer_bbox[3] - outer_bbox[1])),
        ),
        "seed_kind_votes": _normalize_seed_kind_votes(origin_record.get("seed_kind_votes")),
    }


def _generate_pattern_shift_candidates_for_record(origin_idx, origin_record, spatial_index, indexed_elements,
                                                  window_size_um, context_width_um, max_elements_per_window,
                                                  quant_step_um, signature_bins,
                                                  shift_directions=("left", "right", "up", "down"),
                                                  neighbor_limit=128, boundary_tolerance_um=0.02,
                                                  max_events_per_direction=3):
    base_sample = origin_record["sample"]
    base_center = (float(base_sample.center[0]), float(base_sample.center[1]))
    base_candidate = {
        "origin_idx": int(origin_idx),
        "center": base_center,
        "seed_center": tuple(float(v) for v in (base_sample.seed_center or base_center)),
        "seed_kind": _normalize_seed_kind(base_sample.seed_kind),
        "center_shift": tuple(float(v) for v in (base_sample.center_shift or (0.0, 0.0))),
        "inner_bbox": tuple(float(v) for v in base_sample.inner_bbox),
        "outer_bbox": tuple(float(v) for v in base_sample.outer_bbox),
        "outer_polygons": list(origin_record["outer_polygons"]),
        "invariants": np.asarray(origin_record["invariants"], dtype=np.float64),
        "signature": np.asarray(origin_record["signature"], dtype=np.float32),
        "pattern_hash": str(origin_record["pattern_hash"]),
        "invariant_key": tuple(int(v) for v in origin_record["invariant_key"]),
        "stable_bucket_key": tuple(int(v) for v in origin_record["stable_bucket_key"]),
        "normalized_polygons": list(origin_record["normalized_polygons"]),
        "window_area": float(origin_record["window_area"]),
        "generated_by_shift": False,
        "shift_distance_um": 0.0,
    }

    raw_ids = list(base_sample.raw_instance_ids)
    if not raw_ids:
        return [base_candidate]

    seed_elem_id = int(raw_ids[0])
    if seed_elem_id < 0 or seed_elem_id >= len(indexed_elements):
        return [base_candidate]

    seed_center = base_sample.seed_center or base_center
    seed_bbox = indexed_elements[seed_elem_id]["bbox"]
    anchor_bbox = _build_shift_anchor_bbox(seed_bbox, seed_center, window_size_um)
    half_span = float(window_size_um) / 2.0
    outer_side = float(window_size_um) + 2.0 * float(context_width_um)
    scan_side = max(outer_side, float(window_size_um) * 2.0, 1e-6)
    scan_bbox = _make_centered_bbox(base_center, scan_side, scan_side)
    neighbor_ids = _select_relevant_element_ids(
        spatial_index,
        indexed_elements,
        scan_bbox,
        center_xy=base_center,
        max_elements=max(8, int(neighbor_limit)),
    )

    x_edges = []
    y_edges = []
    for nid in neighbor_ids:
        bbox = indexed_elements[nid]["bbox"]
        x_edges.extend([float(bbox[0]), float(bbox[2])])
        y_edges.extend([float(bbox[1]), float(bbox[3])])

    x_interval = _axis_shift_interval(anchor_bbox, base_center, half_span, "x", shift_directions)
    y_interval = _axis_shift_interval(anchor_bbox, base_center, half_span, "y", shift_directions)
    x_shifts = _collect_axis_shift_values(base_center[0], half_span, x_edges, x_interval, max_count=12)
    y_shifts = _collect_axis_shift_values(base_center[1], half_span, y_edges, y_interval, max_count=12)

    candidates = [base_candidate]
    seen_hashes = {base_candidate["pattern_hash"]}
    seen_centers = {(round(base_center[0], 6), round(base_center[1], 6))}

    def _directional_values(values, positive):
        filtered = [float(v) for v in values if (v > 1e-9 if positive else v < -1e-9)]
        filtered.sort(key=lambda v: (abs(v), v))
        return filtered[:max(1, int(max_events_per_direction))]

    directional_shifts = {
        "left": [(sx, 0.0) for sx in _directional_values(x_shifts, positive=False)],
        "right": [(sx, 0.0) for sx in _directional_values(x_shifts, positive=True)],
        "down": [(0.0, sy) for sy in _directional_values(y_shifts, positive=False)],
        "up": [(0.0, sy) for sy in _directional_values(y_shifts, positive=True)],
    }

    for direction in _normalize_shift_directions(shift_directions):
        for dx, dy in directional_shifts.get(direction, []):
            new_center = (float(base_center[0] + dx), float(base_center[1] + dy))
            center_key = (round(new_center[0], 6), round(new_center[1], 6))
            if center_key in seen_centers:
                continue
            candidate = _build_shift_candidate_record(
                origin_idx,
                origin_record,
                new_center,
                spatial_index,
                indexed_elements,
                window_size_um=window_size_um,
                context_width_um=context_width_um,
                max_elements_per_window=max_elements_per_window,
                quant_step_um=quant_step_um,
                signature_bins=signature_bins,
            )
            if candidate is None:
                continue
            if candidate["pattern_hash"] in seen_hashes:
                continue
            seen_hashes.add(candidate["pattern_hash"])
            seen_centers.add(center_key)
            candidate["generated_by_shift"] = True
            candidate["shift_distance_um"] = float(math.hypot(dx, dy))
            candidates.append(candidate)

    return candidates


def _candidate_matches_window_record(candidate, target_record, signature_floor=0.84,
                                     area_match_ratio=0.92, invariant_dist_limit=0.12):
    if candidate["pattern_hash"] == target_record["pattern_hash"]:
        return True

    inv_dist = _invariant_relative_distance(candidate["invariants"], target_record["invariants"])
    if inv_dist > float(invariant_dist_limit):
        return False

    sig_sim = _cosine_similarity_1d(candidate["signature"], target_record["signature"])
    if sig_sim < float(signature_floor):
        return False

    xor_ratio = _window_xor_ratio(candidate, target_record)
    return bool(xor_ratio <= max(0.0, 1.0 - float(area_match_ratio)))


def _greedy_select_cover_candidates(candidates, cluster_weights):
    uncovered = set(range(len(cluster_weights)))
    selected = []
    uncovered_version = 0

    def _candidate_score(candidate, current_uncovered):
        coverage = candidate.get("_coverage_set")
        if coverage is None:
            coverage = {int(v) for v in candidate.get("coverage", set())}
            candidate["_coverage_set"] = coverage
        active_coverage = coverage & current_uncovered
        if not active_coverage:
            return None, None
        return (
            len(active_coverage),
            int(sum(int(cluster_weights[k]) for k in active_coverage)),
            0 if candidate.get("generated_by_shift", False) else 1,
            -float(candidate.get("shift_distance_um", 0.0)),
        ), active_coverage

    def _push_candidate(heap, idx, version_snapshot, current_uncovered):
        score, _ = _candidate_score(candidates[idx], current_uncovered)
        if score is None:
            return
        heapq.heappush(
            heap,
            (-score[0], -score[1], -score[2], -score[3], int(version_snapshot), int(idx)),
        )

    heap = []
    for idx, candidate in enumerate(candidates):
        _push_candidate(heap, idx, uncovered_version, uncovered)

    while uncovered and heap:
        _, _, _, _, version_snapshot, idx = heapq.heappop(heap)
        if int(version_snapshot) != int(uncovered_version):
            _push_candidate(heap, idx, uncovered_version, uncovered)
            continue

        candidate = candidates[idx]
        _, active_coverage = _candidate_score(candidate, uncovered)
        if active_coverage is None:
            continue

        selected.append(candidate)
        uncovered -= active_coverage
        uncovered_version += 1

    return selected


def _compress_window_records_with_shift_cover(window_records, spatial_index, indexed_elements,
                                              window_size_um, context_width_um, max_elements_per_window,
                                              quant_step_um, signature_bins, similarity_threshold=0.96,
                                              shift_directions=("left", "right", "up", "down"),
                                              neighbor_limit=128, boundary_tolerance_um=0.02):
    initial_count = int(len(window_records))
    if initial_count <= 1:
        return window_records

    signature_floor = float(np.clip(float(similarity_threshold) - 0.10, 0.78, 0.92))
    area_match_ratio = float(np.clip(float(similarity_threshold) - 0.04, 0.88, 0.96))
    bucket_rel_tol = 0.08
    bucket_neighbor_dims = 3
    pattern_shift_bucket_index = {}
    for target_idx, target_record in enumerate(window_records):
        bucket_key = tuple(
            int(v) for v in target_record.get(
                "stable_bucket_key",
                _stable_invariant_bucket(target_record["invariants"], relative_tolerance=bucket_rel_tol),
            )
        )
        pattern_shift_bucket_index.setdefault(bucket_key, []).append((target_idx, target_record))

    all_candidates = []
    for origin_idx, record in enumerate(window_records):
        all_candidates.extend(
            _generate_pattern_shift_candidates_for_record(
                origin_idx,
                record,
                spatial_index,
                indexed_elements,
                window_size_um=window_size_um,
                context_width_um=context_width_um,
                max_elements_per_window=max_elements_per_window,
                quant_step_um=quant_step_um,
                signature_bins=signature_bins,
                shift_directions=shift_directions,
                neighbor_limit=neighbor_limit,
                boundary_tolerance_um=boundary_tolerance_um,
            )
        )

    for idx, candidate in enumerate(all_candidates):
        coverage = {int(candidate["origin_idx"])}
        candidate_bucket_key = tuple(
            int(v) for v in candidate.get(
                "stable_bucket_key",
                _stable_invariant_bucket(candidate["invariants"], relative_tolerance=bucket_rel_tol),
            )
        )
        seen_targets = set()
        for neighbor_key in _neighboring_stable_bucket_keys(
                candidate_bucket_key,
                leading_dims=bucket_neighbor_dims,
        ):
            for target_idx, target_record in pattern_shift_bucket_index.get(neighbor_key, ()):
                if target_idx == int(candidate["origin_idx"]) or target_idx in seen_targets:
                    continue
                seen_targets.add(target_idx)
                if _candidate_matches_window_record(
                        candidate,
                        target_record,
                        signature_floor=signature_floor,
                        area_match_ratio=area_match_ratio,
                ):
                    coverage.add(int(target_idx))
        candidate["coverage"] = coverage
        if (idx + 1) % 200 == 0 or idx + 1 == len(all_candidates):
            print(f" pattern shifting 匹配进度: {idx + 1}/{len(all_candidates)}")

    cluster_weights = [int(record["sample"].duplicate_count) for record in window_records]
    selected_candidates = _greedy_select_cover_candidates(all_candidates, cluster_weights)

    merged_records = []
    assigned = set()
    for candidate in selected_candidates:
        covered = sorted(set(candidate.get("coverage", set())) - assigned)
        if not covered:
            continue
        assigned.update(covered)

        origin_record = window_records[int(candidate["origin_idx"])]
        raw_instance_ids = []
        raw_instance_centers = []
        duplicate_count = 0
        covered_window_ids = []
        merged_seed_kind_votes = Counter()
        for cluster_idx in covered:
            member_record = window_records[cluster_idx]
            member_sample = member_record["sample"]
            duplicate_count += int(member_sample.duplicate_count)
            raw_instance_ids.extend(int(v) for v in member_sample.raw_instance_ids)
            raw_instance_centers.extend((float(x), float(y)) for x, y in member_sample.raw_instance_centers)
            covered_window_ids.append(str(member_sample.sample_id))
            merged_seed_kind_votes.update(_normalize_seed_kind_votes(member_record.get("seed_kind_votes")))

        merged_seed_kind = _resolve_seed_kind(dict(merged_seed_kind_votes), fallback=origin_record["sample"].seed_kind)

        merged_sample = LayoutWindowSample(
            sample_id=f"pattern_shift_{len(merged_records):06d}",
            source_name=str(origin_record["sample"].source_name),
            center=(float(candidate["center"][0]), float(candidate["center"][1])),
            seed_center=(float(candidate["seed_center"][0]), float(candidate["seed_center"][1])),
            center_shift=(float(candidate["center_shift"][0]), float(candidate["center_shift"][1])),
            inner_bbox=tuple(float(v) for v in candidate["inner_bbox"]),
            outer_bbox=tuple(float(v) for v in candidate["outer_bbox"]),
            pattern_hash=str(candidate["pattern_hash"]),
            invariant_key=tuple(int(v) for v in candidate["invariant_key"]),
            duplicate_count=int(duplicate_count),
            raw_instance_ids=raw_instance_ids,
            raw_instance_centers=raw_instance_centers,
            seed_kind=merged_seed_kind,
        )
        merged_records.append(
            _build_window_record(
                merged_sample,
                candidate["outer_polygons"],
                candidate["invariants"],
                candidate["signature"],
                candidate["pattern_hash"],
                candidate["invariant_key"],
                generated_by_pattern_shift=bool(candidate.get("generated_by_shift", False)),
                shift_distance_um=float(candidate.get("shift_distance_um", 0.0)),
                merged_window_count=int(len(covered)),
                covered_window_ids=covered_window_ids,
                origin_window_id=str(origin_record["sample"].sample_id),
                seed_kind_votes=merged_seed_kind_votes,
            )
        )

    if len(assigned) < initial_count:
        for cluster_idx, record in enumerate(window_records):
            if cluster_idx not in assigned:
                fallback_record = dict(record)
                fallback_record.setdefault("generated_by_pattern_shift", False)
                fallback_record.setdefault("shift_distance_um", 0.0)
                fallback_record.setdefault("merged_window_count", 1)
                fallback_record.setdefault("covered_window_ids", [str(record["sample"].sample_id)])
                fallback_record.setdefault("origin_window_id", str(record["sample"].sample_id))
                merged_records.append(fallback_record)

    return merged_records


def _build_candidate_window_record(candidate, spatial_index, indexed_elements, *,
                                   source_name, source_window_id, window_size_um, context_width_um,
                                   max_elements_per_window, quant_step_um,
                                   signature_bins, stable_bucket_rel_tol=0.08):
    elem_id = int(candidate["elem_id"])
    center = candidate["center"]
    seed_center = candidate.get("seed_center", center)
    center_shift = candidate.get("center_shift", (0.0, 0.0))
    seed_kind = _normalize_seed_kind(candidate.get("seed_kind", DEFAULT_SEED_KIND))
    outer_side = float(window_size_um) + 2.0 * float(context_width_um)
    inner_bbox = _make_centered_bbox(center, window_size_um, window_size_um)
    outer_bbox = _make_centered_bbox(center, outer_side, outer_side)

    outer_polygons = _approx_clip_indexed_elements_to_bbox(
        spatial_index,
        indexed_elements,
        outer_bbox,
        center_xy=center,
        max_elements=max_elements_per_window,
    )
    if not outer_polygons:
        return None

    inner_polygons = _approx_clip_polygons_with_bbox(outer_polygons, inner_bbox, "and")
    if not inner_polygons:
        return None

    invariants = _compute_window_invariants(outer_polygons)
    invariant_key = _quantize_window_invariants(invariants, quant_step_um)
    dedup_bucket_key = _stable_invariant_bucket(invariants, relative_tolerance=stable_bucket_rel_tol)
    pattern_hash, _ = _enhanced_window_hash(outer_polygons, quant_step_um=quant_step_um)
    signature = _window_signature(outer_polygons, outer_bbox, n_bins=max(4, int(signature_bins)))

    sample = LayoutWindowSample(
        sample_id=str(source_window_id),
        source_name=str(source_name),
        center=(float(center[0]), float(center[1])),
        seed_center=(float(seed_center[0]), float(seed_center[1])),
        center_shift=(float(center_shift[0]), float(center_shift[1])),
        inner_bbox=tuple(float(v) for v in inner_bbox),
        outer_bbox=tuple(float(v) for v in outer_bbox),
        pattern_hash=pattern_hash,
        invariant_key=tuple(int(v) for v in invariant_key),
        duplicate_count=1,
        raw_instance_ids=[int(elem_id)],
        raw_instance_centers=[(float(center[0]), float(center[1]))],
        seed_kind=seed_kind,
    )
    return _build_window_record(
        sample,
        outer_polygons,
        invariants,
        signature,
        pattern_hash,
        invariant_key,
        stable_bucket_key=dedup_bucket_key,
        seed_kind_votes=_seed_kind_vote_counter(seed_kind),
    )


def _build_initial_window_records(candidate_entries, spatial_index, indexed_elements, *,
                                  source_name, window_size_um, context_width_um,
                                  progress_every, enable_coarse_prefilter,
                                  coarse_dedup_quant_um, max_elements_per_window,
                                  quant_step_um, signature_bins):
    raw_records = []
    coarse_skipped = 0
    coarse_seen = set()
    total_candidates = len(candidate_entries)
    outer_side = float(window_size_um) + 2.0 * float(context_width_um)

    for processed, candidate in enumerate(candidate_entries, 1):
        if (
                processed == 1
                or processed == total_candidates
                or (progress_every > 0 and processed % progress_every == 0)
        ):
            progress_percent = (processed / max(1, total_candidates)) * 100.0
            print(
                f"候选中心处理进度: {processed}/{total_candidates} ({progress_percent:.2f}%)",
                end="\r",
            )

        center = candidate["center"]
        outer_bbox = _make_centered_bbox(center, outer_side, outer_side)
        if enable_coarse_prefilter:
            coarse_sig = _coarse_window_descriptor(
                center,
                outer_bbox,
                spatial_index,
                indexed_elements,
                quant_step_um=float(coarse_dedup_quant_um),
            )
            if coarse_sig in coarse_seen:
                coarse_skipped += 1
                continue
            coarse_seen.add(coarse_sig)

        record = _build_candidate_window_record(
            candidate,
            spatial_index,
            indexed_elements,
            source_name=source_name,
            source_window_id=f"window_candidate_{processed - 1:06d}",
            window_size_um=window_size_um,
            context_width_um=context_width_um,
            max_elements_per_window=max_elements_per_window,
            quant_step_um=quant_step_um,
            signature_bins=signature_bins,
        )
        if record is not None:
            raw_records.append(record)

    return raw_records, {"coarse_prefilter_skipped": int(coarse_skipped)}


def _deduplicate_window_records(window_records, *, enable_geometry_dedup, similarity_threshold):
    if not enable_geometry_dedup:
        return list(window_records), {
            "exact_hash_merged": 0,
            "similar_window_merged": 0,
            "dedup_bucket_count": 0,
        }

    dedup_registry = {}
    deduped_records = []
    exact_merged = 0
    similar_merged = 0

    for record in window_records:
        bucket_key = record["stable_bucket_key"]
        bucket = dedup_registry.setdefault(bucket_key, [])
        matched = None
        matched_reason = None
        for existing in bucket:
            if existing["hash"] == record["pattern_hash"]:
                matched = existing
                matched_reason = "exact"
                break
            inv_dist = _invariant_relative_distance(record["invariants"], existing["invariants"])
            sig_sim = _cosine_similarity_1d(record["signature"], existing["signature"])
            if inv_dist <= 0.06 and sig_sim >= float(similarity_threshold):
                matched = existing
                matched_reason = "similar"
                break

        if matched is not None:
            sample = matched["record"]["sample"]
            incoming_sample = record["sample"]
            sample.duplicate_count += int(incoming_sample.duplicate_count)
            sample.raw_instance_ids.extend(int(v) for v in incoming_sample.raw_instance_ids)
            sample.raw_instance_centers.extend((float(x), float(y)) for x, y in incoming_sample.raw_instance_centers)
            merged_votes = _normalize_seed_kind_votes(matched["record"].get("seed_kind_votes"))
            merged_votes.update(_normalize_seed_kind_votes(record.get("seed_kind_votes")))
            matched["record"]["seed_kind_votes"] = merged_votes
            _sync_record_seed_kind(matched["record"])
            if matched_reason == "exact":
                exact_merged += 1
            else:
                similar_merged += 1
            continue

        _sync_record_seed_kind(record)
        deduped_records.append(record)
        bucket.append({
            "hash": record["pattern_hash"],
            "invariants": record["invariants"],
            "signature": record["signature"],
            "record": record,
        })

    return deduped_records, {
        "exact_hash_merged": int(exact_merged),
        "similar_window_merged": int(similar_merged),
        "dedup_bucket_count": int(len(dedup_registry)),
    }


def _compress_window_records(window_records, spatial_index, indexed_elements, *,
                             window_size_um, context_width_um, max_elements_per_window,
                             quant_step_um, signature_bins, similarity_threshold,
                             clip_shift_directions, clip_shift_neighbor_limit,
                             clip_shift_boundary_tolerance_um):
    if len(window_records) <= 1:
        return window_records
    print("\n开始执行固定 shift-cover 窗口压缩...")
    return _compress_window_records_with_shift_cover(
        window_records,
        spatial_index,
        indexed_elements,
        window_size_um=window_size_um,
        context_width_um=context_width_um,
        max_elements_per_window=max_elements_per_window,
        quant_step_um=quant_step_um,
        signature_bins=signature_bins,
        similarity_threshold=similarity_threshold,
        shift_directions=clip_shift_directions,
        neighbor_limit=clip_shift_neighbor_limit,
        boundary_tolerance_um=clip_shift_boundary_tolerance_um,
    )


def _materialize_window_samples(window_records):
    sample_libs = []
    sample_infos = []

    for sample_idx, record in enumerate(window_records):
        sample = record["sample"]
        sample.sample_id = f"window_{sample_idx:06d}"
        _sync_record_seed_kind(record)
        sample_lib = gdstk.Library()
        sample_cell = gdstk.Cell(sample.sample_id)
        sample_cell.add(*record["outer_polygons"])
        sample_lib.add(sample_cell)

        sample_info = sample.to_dict()
        compression_trace = {}
        if "generated_by_pattern_shift" in record:
            compression_trace["generated_by_pattern_shift"] = bool(record.get("generated_by_pattern_shift", False))
        if "shift_distance_um" in record:
            compression_trace["shift_distance_um"] = float(record.get("shift_distance_um", 0.0))
        if "merged_window_count" in record:
            compression_trace["merged_window_count"] = int(record.get("merged_window_count", 1))
        if "covered_window_ids" in record:
            compression_trace["covered_window_ids"] = [str(v) for v in record.get("covered_window_ids", [])]
        if "origin_window_id" in record:
            compression_trace["origin_window_id"] = str(record.get("origin_window_id"))
        if compression_trace:
            sample_info["compression_trace"] = dict(compression_trace)

            # Legacy mirror: keep top-level fields for one compatibility cycle so
            # existing review / post-processing scripts keep working unchanged.
            sample_info.update(compression_trace)

        sample_libs.append(sample_lib)
        sample_infos.append(sample_info)

    return sample_libs, sample_infos


def _build_window_generation_metadata(indexed_elements, center_meta, *,
                                      total_candidates, unique_window_count,
                                      exact_hash_merged, similar_window_merged,
                                      dedup_bucket_count, candidate_bin_size_um,
                                      window_size_um, context_width_um,
                                      max_elements_per_window, hash_precision_nm,
                                      enable_clip_shifting, shifted_seed_count):
    return {
        "original_element_count": int(len(indexed_elements)),
        "raw_center_count": int(total_candidates),
        "unique_window_count": int(unique_window_count),
        "compression_ratio": float(unique_window_count) / float(max(1, total_candidates)),
        "candidate_bin_size_um": float(center_meta.get(
            "candidate_bin_size_um",
            candidate_bin_size_um if candidate_bin_size_um is not None else max((window_size_um + 2.0 * context_width_um) * 2.0, 10.0, 1e-6),
        )),
        "max_elements_per_window": int(max_elements_per_window),
        "window_size_um": float(window_size_um),
        "context_width_um": float(context_width_um),
        "hash_precision_nm": float(hash_precision_nm),
        "clip_shifting_enabled": bool(center_meta.get("clip_shifting_enabled", enable_clip_shifting)),
        "shifted_seed_count": int(shifted_seed_count),
        "element_seed_count": int(center_meta.get("element_seed_count", total_candidates)),
        "relation_seed_count": int(center_meta.get("relation_seed_count", 0)),
        "hotspot_seed_count": int(center_meta.get("hotspot_seed_count", 0)),
        "dedup_bucket_count": int(dedup_bucket_count),
        "exact_hash_merged": int(exact_hash_merged),
        "similar_window_merged": int(similar_window_merged),
    }

def generate_layout_window_samples(
        lib,
        window_size_um=1.35,
        context_width_um=0.675,
        progress_every=200,
        enable_geometry_dedup=True,
        hash_precision_nm=5.0,
        similarity_threshold=0.96,
        signature_bins=12,
        coarse_dedup_quant_um=0.02,
        enable_coarse_prefilter=True,
        candidate_bin_size_um=None,
        relation_seed_ratio=0.2,
        relation_gap_threshold_um=0.08,
        max_elements_per_window=256,
        enable_clip_shifting=True,
        clip_shift_directions=("left", "right", "up", "down"),
        clip_shift_neighbor_limit=128,
        clip_shift_boundary_tolerance_um=0.02,
        return_metadata=False,
        source_name="layout"):
    """
    基于元素中心生成候选窗口，并使用增强顶点哈希+几何不变量去重。

    每个样本保留外扩窗口(中心矩形 + 一圈上下文)的几何；特征提取时再拆分内圈/外圈。

    去重策略说明：
    1. hash_precision_nm 仅用于“增强顶点哈希”的精确量化。
    2. 相似窗口候选分桶使用独立的稳定粗桶，避免 1~2nm 级裁剪抖动导致桶过碎。
    3. 真正判定是否合并时，仍然同时检查增强哈希 / 几何不变量距离 / 签名相似度。

    clip shifting 合入方式：
    1. 先按空间分桶挑选候选 seed。
    2. 再对每个 seed 做局部边界对齐式微调，得到最终中心。
    3. 只有当 shift 确实改善了窗口边界与局部几何边界的贴合时，才接受平移。

    性能说明：
    1. 候选中心不会直接取所有几何元素，而是先按空间分桶挑代表中心。
    2. 对每个代表中心，按 clip shifting 思路在允许范围内做边界对齐式平移，选出更优中心。
    3. 在精确窗口去重前，先使用邻域相对 bbox 的轻量描述符做预去重。
    4. 每个窗口只保留最多 max_elements_per_window 个最相关局部几何。

    这三步都是为了让大版图样本在工程上可跑通；如果希望更接近原始全量几何，
    可以放宽 candidate_bin_size_um / max_elements_per_window，代价是耗时显著增加。
    """
    spatial_index, indexed_elements, layout_bbox = _build_layout_spatial_index(lib)
    if spatial_index is None or not indexed_elements:
        print("警告: 版图中没有找到有效的polygon/path，无法生成中心窗口样本")
        return ([], [], {}) if return_metadata else ([], [])

    quant_step_um = max(1e-6, float(hash_precision_nm) * 1e-3)
    candidate_entries, center_meta = _select_candidate_centers(
        indexed_elements,
        spatial_index,
        window_size_um=window_size_um,
        candidate_bin_size_um=candidate_bin_size_um,
        relation_seed_ratio=relation_seed_ratio,
        relation_gap_threshold_um=relation_gap_threshold_um,
        enable_clip_shifting=enable_clip_shifting,
        clip_shift_directions=clip_shift_directions,
        clip_shift_neighbor_limit=clip_shift_neighbor_limit,
        clip_shift_boundary_tolerance_um=clip_shift_boundary_tolerance_um,
    )
    total_candidates = len(candidate_entries)
    processed = 0
    print(
        f"开始基于几何中心生成窗口样本，原始元素数: {len(indexed_elements)}, "
        f"候选中心数: {total_candidates}"
    )
    raw_records, build_stats = _build_initial_window_records(
        candidate_entries,
        spatial_index,
        indexed_elements,
        source_name=source_name,
        window_size_um=window_size_um,
        context_width_um=context_width_um,
        progress_every=progress_every,
        enable_coarse_prefilter=enable_coarse_prefilter,
        coarse_dedup_quant_um=coarse_dedup_quant_um,
        max_elements_per_window=max_elements_per_window,
        quant_step_um=quant_step_um,
        signature_bins=signature_bins,
    )
    window_records, dedup_stats = _deduplicate_window_records(
        raw_records,
        enable_geometry_dedup=enable_geometry_dedup,
        similarity_threshold=similarity_threshold,
    )
    window_records = _compress_window_records(
        window_records,
        spatial_index,
        indexed_elements,
        window_size_um=window_size_um,
        context_width_um=context_width_um,
        max_elements_per_window=max_elements_per_window,
        quant_step_um=quant_step_um,
        signature_bins=signature_bins,
        similarity_threshold=similarity_threshold,
        clip_shift_directions=clip_shift_directions,
        clip_shift_neighbor_limit=clip_shift_neighbor_limit,
        clip_shift_boundary_tolerance_um=clip_shift_boundary_tolerance_um,
    )
    sample_libs, sample_infos = _materialize_window_samples(window_records)

    metadata = _build_window_generation_metadata(
        indexed_elements,
        center_meta,
        total_candidates=total_candidates,
        unique_window_count=len(window_records),
        exact_hash_merged=dedup_stats["exact_hash_merged"],
        similar_window_merged=dedup_stats["similar_window_merged"],
        dedup_bucket_count=dedup_stats["dedup_bucket_count"],
        candidate_bin_size_um=candidate_bin_size_um,
        window_size_um=window_size_um,
        context_width_um=context_width_um,
        max_elements_per_window=max_elements_per_window,
        hash_precision_nm=hash_precision_nm,
        enable_clip_shifting=enable_clip_shifting,
        shifted_seed_count=int(center_meta.get("shifted_seed_count", 0)),
    )
    metadata["coarse_prefilter_skipped"] = int(build_stats["coarse_prefilter_skipped"])
    metadata["layout_bbox"] = list(layout_bbox) if layout_bbox is not None else None

    print(f"\n共处理 {total_candidates} 个候选中心，去重后保留 {len(sample_infos)} 个窗口样本")
    if return_metadata:
        return sample_libs, sample_infos, metadata
    return sample_libs, sample_infos


def check_if_large_layout(filepath, size_threshold=10):
    """
    检查版图是否为大版图
    :param filepath: 文件路径
    :param size_threshold: 尺寸阈值（微米），超过此值认为是大版图
    :return: (is_large, width, height)
    """
    try:
        lib = _read_oas_only_library(filepath)

        # 获取所有单元格的边界框
        all_bboxes = []
        for cell in lib.cells:
            bbox = cell.bounding_box()
            if bbox is not None:
                all_bboxes.append(bbox)

        if not all_bboxes:
            return False, 0, 0

        # 计算整体边界框
        min_x = min(bbox[0][0] for bbox in all_bboxes)
        min_y = min(bbox[0][1] for bbox in all_bboxes)
        max_x = max(bbox[1][0] for bbox in all_bboxes)
        max_y = max(bbox[1][1] for bbox in all_bboxes)

        width = max_x - min_x
        height = max_y - min_y
        is_large = width > size_threshold or height > size_threshold
        return is_large, width, height
    except Exception as e:
        print(f"检查版图尺寸时出错 {filepath}: {e}")
        return False, 0, 0


class LayerOperationProcessor:
    """处理层操作的类"""

    def __init__(self):
        # 预定义的操作规则
        self.operation_rules = {
            # 示例：相减操作 - layer1 - layer2 -> new_layer
            '2413/0': {
                'operation': 'subtract',
                'target_layer': '2410/0',
                'result_layer': '2413_subtracted/0'  # 示例结果层
            },
            # 可以在这里添加更多操作规则
            # 'layer1': {
            #     'operation': 'union|intersect|subtract',
            #     'target_layer': 'layer2',
            #     'result_layer': 'new_layer'
            # }
        }

    def register_operation_rule(self, source_layer, operation, target_layer, result_layer):
        """注册新的操作规则"""
        self.operation_rules[tuple(source_layer)] = {
            'operation': operation,
            'target_layer': tuple(target_layer),
            'result_layer': tuple(result_layer)
        }

    def apply_layer_operations(self, lib):
        """对库中的所有单元格应用层操作"""
        for cell in lib.cells:
            self._apply_operations_to_cell(cell)
        return lib

    def _apply_operations_to_cell(self, cell):
        """对单个单元格应用层操作"""
        # 获取所有层的数据
        # gdstk没有直接按spec获取的函数，需要遍历
        all_geometries = {}
        all_geometries['polygons'] = cell.get_polygons(depth=None)
        all_geometries['paths'] = cell.get_paths(depth=None)

        # 按层和数据类型组织
        geometries_by_spec = {}
        for geom_type, geoms in all_geometries.items():
            for geom in geoms:
                key = (geom.layer, geom.datatype)
                if key not in geometries_by_spec:
                    geometries_by_spec[key] = {'polygons': [], 'paths': []}
                geometries_by_spec[key][geom_type].append(geom)

        # 执行操作
        for source_key, rule in self.operation_rules.items():
            target_key = rule['target_layer']
            result_key = rule['result_layer']

            # 检查源层和目标层是否存在
            if source_key in geometries_by_spec and target_key in geometries_by_spec:
                source_geoms = geometries_by_spec[source_key]
                target_geoms = geometries_by_spec[target_key]

                # 目前只处理多边形
                if 'polygons' in source_geoms and 'polygons' in target_geoms:
                    source_polys_list = source_geoms['polygons']
                    target_polys_list = target_geoms['polygons']

                    if rule['operation'] == 'subtract':
                        # 执行相减操作
                        source_set = []
                        for p in source_polys_list:
                            source_set.extend(p.polygons if hasattr(p, 'polygons') else [p])
                        target_set = []
                        for p in target_polys_list:
                            target_set.extend(p.polygons if hasattr(p, 'polygons') else [p])

                        if source_set and target_set:
                            result_polyset = gdstk.boolean(source_set, target_set, 'not')

                            if result_polyset is not None:
                                # 创建新多边形对象
                                for poly in result_polyset:
                                    result_poly_obj = gdstk.Polygon(poly)
                                    result_poly_obj.layer = result_key[0]
                                    result_poly_obj.datatype = result_key[1]

                                    # 添加结果层
                                    cell.add(result_poly_obj)

                    # union 和 intersect 逻辑类似，此处省略具体实现
                    # elif rule['operation'] == 'union':
                    #     ...
                    # elif rule['operation'] == 'intersect':
                    #     ...

        return cell


class LayoutFeatureExtractor:
    """从 OASIS 文件中提取特征的类"""

    def __init__(self):
        # 优化后的特征名称，去除了冗余特征，增加了关键特征
        # 全局特征（10个）
        self.base_feature_names = [
            'num_layers', 'num_polygons', 'num_paths', 'num_labels', 'total_area', 'total_perimeter',
            'bbox_width', 'bbox_height', 'aspect_ratio', 'density'
        ]
        # 局部特征（19个）
        # self.enhanced_feature_names = [
        #     'local_density_mean', 'local_density_std', 'local_density_max_min_diff',
        #     'aspect_ratio_mean', 'aspect_ratio_std', 'compactness_mean', 'line_width_mean',
        #     'rotation_angle_std', 'connected_components', 'max_component_ratio',
        #     'nearest_neighbor_mean', 'nearest_neighbor_std', 'rectangle_ratio',
        #     'convex_hull_ratio', 'edge_alignment', 'hole_ratio', 'fractal_dimension',
        #     'spatial_spectrum_mean', 'spatial_spectrum_std'
        # ]
        self.enhanced_feature_names = [
            # 空间分布特征（16个）
            *[f'grid_density_{i}' for i in range(16)],  # 4x4网格密度

            # 形状复杂度特征（6个）
            'avg_perimeter_area_ratio',  # 平均周长面积比
            'area_std',  # 面积标准差
            'area_q25',  # 面积25分数
            'area_q50',  # 面积中位数
            'area_q75',  # 面积75分数
            'avg_compactness',  # 平均紧密度

            # 层间关系特征（5个）
            'layer_overlap_score',  # 层间重叠度
            'layer_alignment_score',  # 层间对齐度
            'main_layer_area_ratio',  # 主要层面积比例
            'layer_distance_mean',  # 层间距离均值
            'layer_distance_std',  # 层间距离标准差

            # 局部图案特征（5个）
            'repetition_score',  # 重复结构得分
            'symmetry_score',  # 对称性得分
            'regularity_score',  # 规则性得分
            'local_density_variation',  # 局部密度变化
            'edge_density_ratio',  # 边缘密度比

            # Radon结构特征（8个）
            *[f'radon_proj_std_{i}' for i in range(8)],
        ]
        # 完整特征列表
        self.feature_names = self.base_feature_names + self.enhanced_feature_names

        # 初始化层操作处理器
        self.layer_processor = LayerOperationProcessor()
        # 注册默认操作：2413/0 - 2410/0 -> 2413_subtracted/0
        self.layer_processor.register_operation_rule(
            '2413/0', 'subtract', '2410/0', '2413_subtracted/0'
        )

    def _load_cells(self, filepath: str, apply_layer_operations: bool = True) -> Dict[str, Any]:
        try:
            lib = _read_oas_only_library(filepath)
        except Exception as e:
            print(f"读取文件失败 {filepath}: {e}")
            raise e

        if apply_layer_operations:
            lib = self.layer_processor.apply_layer_operations(lib)
        return {cell.name: cell for cell in lib.cells}

    def _extract_base_features(self, cells):
        """提取基础特征"""
        num_layers = set()
        num_polygons = 0
        num_paths = 0
        num_labels = 0
        total_area = 0.0
        total_perimeter = 0.0
        point_blocks = []  # 用于计算全局bbox

        polygon_areas = []
        polygon_perimeters = []
        layer_areas = {}

        # 用于计算全局bbox
        for cell_name, cell in cells.items():
            # 统计多边形
            for polygon in cell.polygons:
                num_polygons += 1
                layer = polygon.layer if hasattr(polygon, 'layer') else 0
                num_layers.add(layer)
                points = np.asarray(polygon.points, dtype=np.float64)

                area = polygon.area()
                total_area += area
                polygon_areas.append(area)

                # 记录层面积
                if layer not in layer_areas:
                    layer_areas[layer] = 0.0
                layer_areas[layer] += area

                perimeter = float(polygon_perimeter(points))
                total_perimeter += perimeter
                polygon_perimeters.append(perimeter)

                # 收集点用于边界框计算
                if len(points) > 0:
                    point_blocks.append(points)

            # 统计路径
            for path in cell.paths:
                num_paths += 1
                num_layers.add(path.layer if hasattr(path, 'layer') else 0)

            # 统计标签
            for label in cell.labels:
                num_labels += 1
                num_layers.add(label.layer if hasattr(label, 'layer') else 0)

        # 计算全局边界框
        if point_blocks:
            points_array = np.vstack(point_blocks)
            min_coord = np.min(points_array, axis=0)
            max_coord = np.max(points_array, axis=0)
            bbox_width = max_coord[0] - min_coord[0]
            bbox_height = max_coord[1] - min_coord[1]
        else:
            min_coord, max_coord = np.array([0.0, 0.0]), np.array([0.0, 0.0])
            bbox_width = 0.0
            bbox_height = 0.0

        # 计算全局纵横比
        if bbox_height > 0:
            aspect_ratio = bbox_width / bbox_height
        else:
            aspect_ratio = 0.0

        # 计算全局密度（面积/边界框面积）
        bbox_area = bbox_width * bbox_height
        if bbox_area > 0:
            density = total_area / bbox_area
        else:
            density = 0.0

        all_points_array = points_array if point_blocks else np.array([])

        base_features = {
            'num_layers': len(num_layers),
            'num_polygons': num_polygons,
            'num_paths': num_paths,
            'num_labels': num_labels,
            'total_area': total_area,
            'total_perimeter': total_perimeter,
            'bbox_width': bbox_width,
            'bbox_height': bbox_height,
            'aspect_ratio': aspect_ratio,
            'density': density,
            'polygon_areas': np.array(polygon_areas) if polygon_areas else np.array([]),
            'polygon_perimeters': np.array(polygon_perimeters) if polygon_perimeters else np.array([]),
            'layer_areas': layer_areas,
            'all_points': all_points_array,
            'min_coord': min_coord,
            'max_coord': max_coord,
        }

        return base_features

    def _extract_spatial_features(self, base_features):
        """提取空间分布特征"""
        all_points = base_features['all_points']
        min_coord = base_features['min_coord']
        max_coord = base_features['max_coord']

        if len(all_points) == 0 or not np.all(max_coord > min_coord):
            return np.zeros(16, dtype=np.float32)

        # 使用二维直方图统计点密度，避免边界遗漏和循环开销
        hist, _, _ = np.histogram2d(
            all_points[:, 0],
            all_points[:, 1],
            bins=(4, 4),
            range=[[min_coord[0], max_coord[0]], [min_coord[1], max_coord[1]]]
        )

        hist_sum = np.sum(hist)
        if hist_sum > 0:
            hist = hist / hist_sum

        return hist.flatten().astype(np.float32)

    def _extract_shape_features(self, base_features):
        """提取形状复杂度特征"""
        areas = base_features['polygon_areas']
        perimeters = base_features['polygon_perimeters']

        if len(areas) == 0:
            return np.zeros(6)

        # 1. 平均周长面积比
        valid_mask = (areas > 0) & (perimeters > 0)
        if np.any(valid_mask):
            perimeter_area_ratios = perimeters[valid_mask] / areas[valid_mask]
            avg_perimeter_area_ratio = np.mean(perimeter_area_ratios)
        else:
            avg_perimeter_area_ratio = 0.0

        # 2. 面积标准差
        area_std = np.std(areas) if len(areas) > 1 else 0.0

        # 3. 面积分位数
        area_q25 = np.percentile(areas, 25) if len(areas) > 0 else 0.0
        area_q50 = np.percentile(areas, 50) if len(areas) > 0 else 0.0
        area_q75 = np.percentile(areas, 75) if len(areas) > 0 else 0.0

        # 4. 平均紧密度（4pi*面积/周长^2）
        if np.any(valid_mask):
            compactness = 4 * np.pi * areas[valid_mask] / (perimeters[valid_mask] ** 2)
            avg_compactness = np.mean(compactness)
        else:
            avg_compactness = 0.0

        shape_features = np.array([
            avg_perimeter_area_ratio,
            area_std,
            area_q25,
            area_q50,
            area_q75,
            avg_compactness
        ], dtype=np.float32)

        return shape_features

    def _extract_layer_features(self, base_features):
        """提取层间关系特征"""
        layer_areas = base_features['layer_areas']
        all_points = base_features['all_points']
        min_coord = base_features['min_coord']
        max_coord = base_features['max_coord']

        if not layer_areas:
            return np.zeros(5, dtype=np.float32)

        layers = list(layer_areas.keys())
        areas = np.array([layer_areas[l] for l in layers], dtype=np.float32)

        total_area = float(np.sum(areas))
        if total_area > 0:
            normalized_areas = areas / total_area
            entropy = -np.sum(normalized_areas * np.log(normalized_areas + 1e-10))
            max_entropy = np.log(len(layers) + 1e-10)
            layer_overlap_score = entropy / max_entropy if max_entropy > 0 else 0.0
            main_layer_area_ratio = float(np.max(areas) / total_area)
        else:
            layer_overlap_score = 0.0
            main_layer_area_ratio = 0.0

        # 用点云的空间离散程度刻画跨层对齐：越集中越对齐
        if len(all_points) > 2 and np.all(max_coord > min_coord):
            normalized_points = (all_points - min_coord) / (max_coord - min_coord + 1e-10)
            spread = float(np.mean(np.std(normalized_points, axis=0)))
            layer_alignment_score = float(np.clip(1.0 - spread, 0.0, 1.0))
        else:
            layer_alignment_score = 0.5

        # 根据层面积分布估计层间差异
        if len(areas) > 1 and total_area > 0:
            sorted_ratios = np.sort(areas / total_area)
            deltas = np.diff(sorted_ratios)
            layer_distance_mean = float(np.mean(deltas)) if len(deltas) > 0 else 0.0
            layer_distance_std = float(np.std(deltas)) if len(deltas) > 0 else 0.0
        else:
            layer_distance_mean = 0.0
            layer_distance_std = 0.0

        layer_features = np.array([
            layer_overlap_score,
            layer_alignment_score,
            main_layer_area_ratio,
            layer_distance_mean,
            layer_distance_std
        ], dtype=np.float32)

        return layer_features

    def _extract_pattern_features(self, base_features):
        """提取局部图案特征"""
        all_points = base_features['all_points']
        num_polygons = base_features['num_polygons']
        total_area = base_features['total_area']
        bbox_width = base_features['bbox_width']
        bbox_height = base_features['bbox_height']
        min_coord = base_features['min_coord']
        max_coord = base_features['max_coord']

        if len(all_points) == 0 or not np.all(max_coord > min_coord):
            return np.array([0.0, 0.5, 0.5, 0.0, 0.3], dtype=np.float32)

        # 8x8占据网格用于图案统计
        occ, _, _ = np.histogram2d(
            all_points[:, 0],
            all_points[:, 1],
            bins=(8, 8),
            range=[[min_coord[0], max_coord[0]], [min_coord[1], max_coord[1]]]
        )
        occ_sum = np.sum(occ)
        if occ_sum > 0:
            occ = occ / occ_sum

        hist_counts, _ = np.histogram(occ.flatten(), bins=6, range=(0.0, float(np.max(occ) + 1e-10)))
        repetition_score = float(np.max(hist_counts) / max(1, np.sum(hist_counts)))

        h_sym = 1.0 - float(np.mean(np.abs(occ - np.flipud(occ))))
        v_sym = 1.0 - float(np.mean(np.abs(occ - np.fliplr(occ))))
        symmetry_score = float(np.clip((h_sym + v_sym) / 2.0, 0.0, 1.0))

        gx = np.diff(occ, axis=0)
        gy = np.diff(occ, axis=1)
        grad_mean = float((np.mean(np.abs(gx)) + np.mean(np.abs(gy))) / 2.0)
        regularity_score = float(np.clip(1.0 - grad_mean * 8.0, 0.0, 1.0))

        local_density_variation = float(np.std(occ))

        edge_mask = np.zeros_like(occ, dtype=bool)
        edge_mask[0, :] = True
        edge_mask[-1, :] = True
        edge_mask[:, 0] = True
        edge_mask[:, -1] = True
        edge_density = float(np.sum(occ[edge_mask]))
        center_density = float(np.sum(occ[~edge_mask]))
        edge_density_ratio = edge_density / (center_density + 1e-10)

        bbox_area = bbox_width * bbox_height
        geometric_density = total_area / bbox_area if bbox_area > 0 else 0.0
        repetition_score = float(np.clip(0.7 * repetition_score + 0.3 * min(num_polygons / 200.0, 1.0), 0.0, 1.0))
        local_density_variation = float(np.clip(local_density_variation + 0.1 * geometric_density, 0.0, 1.0))

        pattern_features = np.array([
            repetition_score,
            symmetry_score,
            regularity_score,
            local_density_variation,
            edge_density_ratio
        ], dtype=np.float32)

        return pattern_features

    def _extract_radon_features(self, base_features, n_angles: int = 8):
        """提取基于占据栅格投影的结构特征。"""
        all_points = base_features['all_points']
        min_coord = base_features['min_coord']
        max_coord = base_features['max_coord']

        if len(all_points) == 0 or not np.all(max_coord > min_coord):
            return np.zeros(n_angles, dtype=np.float32)

        occ, _, _ = np.histogram2d(
            all_points[:, 0],
            all_points[:, 1],
            bins=(8, 8),
            range=[[min_coord[0], max_coord[0]], [min_coord[1], max_coord[1]]]
        )
        occ = (occ > 0).astype(np.float32)
        return extract_radon_features(occ, n_angles=n_angles)

    def _build_feature_blocks(self, cells) -> Dict[str, np.ndarray]:
        base_features = self._extract_base_features(cells)

        base_feature_vector = np.array([
            base_features['num_layers'],
            base_features['num_polygons'],
            base_features['num_paths'],
            base_features['num_labels'],
            base_features['total_area'],
            base_features['total_perimeter'],
            base_features['bbox_width'],
            base_features['bbox_height'],
            base_features['aspect_ratio'],
            base_features['density']
        ], dtype=np.float32)

        spatial_features = self._extract_spatial_features(base_features)
        shape_features = self._extract_shape_features(base_features)
        layer_features = self._extract_layer_features(base_features)
        pattern_features = self._extract_pattern_features(base_features)
        radon_features = self._extract_radon_features(base_features)

        return {
            "base": np.asarray(base_feature_vector, dtype=np.float32),
            "spatial": np.asarray(spatial_features, dtype=np.float32),
            "shape": np.asarray(shape_features, dtype=np.float32),
            "layer": np.asarray(layer_features, dtype=np.float32),
            "pattern": np.asarray(pattern_features, dtype=np.float32),
            "radon": np.asarray(radon_features, dtype=np.float32),
        }

    def _combine_feature_blocks(self, feature_blocks: Dict[str, np.ndarray], block_weights=None) -> np.ndarray:
        weights = _normalize_feature_block_weights(block_weights, default_weights=DEFAULT_INNER_BLOCK_WEIGHTS)
        vectors = []
        for name in FEATURE_BLOCK_NAMES:
            block = np.asarray(feature_blocks.get(name, np.zeros(0, dtype=np.float32)), dtype=np.float32)
            vectors.append(float(weights.get(name, 1.0)) * block)
        return np.concatenate(vectors).astype(np.float32)

    def _build_feature_vector(self, cells, block_weights=None) -> np.ndarray:
        return self._combine_feature_blocks(
            self._build_feature_blocks(cells),
            block_weights=block_weights,
        )

    def extract_from_cells(self, cells, block_weights=None) -> np.ndarray:
        return self._build_feature_vector(cells, block_weights=block_weights)

    def _clip_cells_by_rect(self, cells, bbox, operation):
        clipped_cells = {}

        def _proxy_polygons_for_geometry(geometry):
            if hasattr(geometry, "points"):
                return [geometry]
            geom_bbox = _geometry_bbox(geometry)
            if geom_bbox is None:
                return []
            return [_rect_polygon_from_bbox(
                geom_bbox,
                layer=getattr(geometry, "layer", 0),
                datatype=getattr(geometry, "datatype", 0),
            )]

        for cell_name, cell in cells.items():
            clipped_cell = gdstk.Cell(f"{cell_name}_{operation}")

            for geometry in itertools.chain(cell.polygons, cell.paths):
                clipped = _approx_clip_polygons_with_bbox(_proxy_polygons_for_geometry(geometry), bbox, operation)
                if clipped:
                    clipped_cell.add(*clipped)

            if len(clipped_cell.polygons) > 0 or len(clipped_cell.paths) > 0 or len(clipped_cell.labels) > 0:
                clipped_cells[cell_name] = clipped_cell

        return clipped_cells

    def extract_from_window_sample(self,
                                   filepath: str,
                                   sample_info: Dict[str, Any],
                                   apply_layer_operations: bool = True,
                                   inner_weight: float = 1.0,
                                   outer_weight: float = 0.35,
                                   inner_block_weights: Optional[Dict[str, float]] = None,
                                   outer_block_weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """针对中心矩形窗口样本提取特征：内圈/外圈分开计算，再按权重融合"""
        cells = self._load_cells(filepath, apply_layer_operations=apply_layer_operations)
        inner_bbox = tuple(float(v) for v in sample_info.get("inner_bbox", []))
        if len(inner_bbox) != 4:
            return self.extract_from_cells(cells)

        inner_cells = self._clip_cells_by_rect(cells, inner_bbox, "and")
        outer_ring_cells = self._clip_cells_by_rect(cells, inner_bbox, "not")

        inner_blocks = self._build_feature_blocks(inner_cells)
        outer_blocks = self._build_feature_blocks(outer_ring_cells)
        inner_block_weights = _normalize_feature_block_weights(
            inner_block_weights,
            default_weights=DEFAULT_INNER_BLOCK_WEIGHTS,
        )
        outer_block_weights = _normalize_feature_block_weights(
            outer_block_weights,
            default_weights=DEFAULT_OUTER_BLOCK_WEIGHTS,
        )

        fused_blocks = {}
        for name in FEATURE_BLOCK_NAMES:
            inner_block = np.asarray(inner_blocks.get(name, np.zeros(0, dtype=np.float32)), dtype=np.float32)
            outer_block = np.asarray(outer_blocks.get(name, np.zeros(0, dtype=np.float32)), dtype=np.float32)
            fused_blocks[name] = (
                float(inner_weight) * float(inner_block_weights.get(name, 1.0)) * inner_block
                + float(outer_weight) * float(outer_block_weights.get(name, 1.0)) * outer_block
            )
        return self._combine_feature_blocks(fused_blocks, block_weights=DEFAULT_INNER_BLOCK_WEIGHTS)

    def extract_from_layout(self, filepath: str, apply_layer_operations: bool = True) -> np.ndarray:
        """从 OASIS 文件中提取特征"""
        cells = self._load_cells(filepath, apply_layer_operations=apply_layer_operations)
        return self.extract_from_cells(cells)


class SimilarityCalculator:
    """计算中心窗口样本相似度的类。"""

    def __init__(self, method='euclidean'):
        self.method = method

    def _compute_sub_similarity(self, features_i: np.ndarray, features_j: np.ndarray) -> np.ndarray:
        if self.method == 'cosine':
            sub_sim = cosine_similarity(features_i, features_j)
        elif self.method == 'euclidean':
            dists = pairwise_distances(features_i, features_j, metric='euclidean')
            sub_sim = 1.0 / (1.0 + dists)
        else:
            raise ValueError(f"不支持的相似度计算方法：{self.method}")

        return np.asarray(sub_sim, dtype=np.float32)

    def compute_similarity_submatrix_from_features(self, features: np.ndarray, indices: List[int]) -> np.ndarray:
        """根据特征和索引计算子相似度矩阵并返回numpy数组"""
        if features is None or len(indices) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        idx = np.array(sorted(set(indices)), dtype=np.int64)
        sub_features = np.asarray(features[idx], dtype=np.float32)
        sim = self._compute_sub_similarity(sub_features, sub_features)
        np.fill_diagonal(sim, 1.0)
        return np.asarray(sim, dtype=np.float32)


class HDBSCANClustering:
    """使用全样本HDBSCAN算法进行聚类。"""

    def __init__(self, min_cluster_size=8, min_samples=4,
                 cluster_selection_epsilon=0.0, cluster_selection_method='eom',
                 metric='euclidean',
                 enable_singleton_postmerge=True,
                 singleton_merge_threshold=0.93):
        self.min_cluster_size = max(2, int(min_cluster_size))
        self.min_samples = None if min_samples is None else max(1, int(min_samples))
        self.cluster_selection_epsilon = max(0.0, float(cluster_selection_epsilon))
        self.cluster_selection_method = str(cluster_selection_method)
        self.metric = str(metric)
        self.enable_singleton_postmerge = bool(enable_singleton_postmerge)
        self.singleton_merge_threshold = float(singleton_merge_threshold)
        self.outlier_score_merge_limit = float(DEFAULT_POSTMERGE_OUTLIER_THRESHOLD)
        self.last_postmerge_stats = {}
        self.last_soft_membership = {}

    def _prepare_hdbscan_input(self, features: np.ndarray) -> Tuple[np.ndarray, str, str]:
        """将用户度量映射到 HDBSCAN 可高效处理的输入形式。"""
        metric = str(self.metric).lower().strip()
        X = np.asarray(features, dtype=np.float32)

        # hdbscan 的 tree 路线不直接支持 cosine。对 L2 归一化后的向量使用欧氏距离，
        # 与 cosine 距离单调等价，同时避免构建 O(n^2) 的 precomputed 距离矩阵。
        if metric == 'cosine':
            X = normalize(X, norm='l2', copy=True).astype(np.float32, copy=False)
            return X, 'euclidean', 'cosine(via_l2_euclidean)'

        return X, metric, metric

    def _cluster_medoid_index(self, features: np.ndarray, cluster: List[int], metric_mode: str) -> int:
        if len(cluster) <= 1:
            return int(cluster[0])
        idx = np.asarray(cluster, dtype=np.int64)
        subset = np.asarray(features[idx], dtype=np.float32)
        if metric_mode == 'euclidean':
            dists = pairwise_distances(subset, metric='euclidean')
            return int(idx[int(np.argmin(np.mean(dists, axis=1)))])
        sims = cosine_similarity(subset)
        return int(idx[int(np.argmax(np.mean(sims, axis=1)))])

    def _build_soft_membership_summary(self,
                                       raw_labels: np.ndarray,
                                       raw_cluster_labels: List[int],
                                       membership_matrix: Optional[np.ndarray],
                                       final_clusters: List[List[int]]) -> Dict[str, Any]:
        if membership_matrix is None:
            return {}
        matrix = np.asarray(membership_matrix, dtype=np.float32)
        if matrix.ndim != 2 or matrix.size == 0:
            return {}

        final_cluster_sets = [set(int(v) for v in cluster) for cluster in final_clusters]
        raw_to_final = {}
        for raw_label in raw_cluster_labels:
            members = set(int(v) for v in np.where(raw_labels == int(raw_label))[0].tolist())
            final_cluster_id = None
            if members:
                for cluster_id, cluster_members in enumerate(final_cluster_sets):
                    if members.issubset(cluster_members):
                        final_cluster_id = int(cluster_id)
                        break
            raw_to_final[int(raw_label)] = final_cluster_id

        max_membership_probability = []
        for row in matrix:
            row = np.asarray(row, dtype=np.float32).reshape(-1)
            if row.size == 0:
                max_membership_probability.append(0.0)
                continue

            max_membership_probability.append(float(np.max(row)))

        return {
            "enabled": True,
            "pre_postmerge_cluster_labels": [int(v) for v in raw_cluster_labels],
            "pre_to_final_cluster_map": {str(k): v for k, v in raw_to_final.items()},
            "max_membership_probability": max_membership_probability,
        }

    def _postmerge_noise_singletons(self, features: np.ndarray, clusters: List[List[int]],
                                    noise_indices: np.ndarray, metric_mode: str,
                                    outlier_scores: Optional[np.ndarray] = None) -> Tuple[List[int], List[List[int]], Dict[str, Any]]:
        noise_list = [int(v) for v in np.asarray(noise_indices, dtype=np.int64).tolist()]
        copied_clusters = [list(cluster) for cluster in clusters]
        stats = {
            "enabled": bool(self.enable_singleton_postmerge),
            "noise_count_before": int(len(noise_list)),
            "eligible_noise_count": 0,
            "merged_noise_count": 0,
            "hard_noise_count": 0,
            "noise_count_after": int(len(noise_list)),
            "outlier_score_threshold": float(self.outlier_score_merge_limit),
        }
        if (not self.enable_singleton_postmerge) or len(noise_list) == 0:
            return noise_list, copied_clusters, stats

        score_array = np.asarray(outlier_scores, dtype=np.float32) if outlier_scores is not None else None
        eligible_noise = []
        hard_noise = []
        for idx in noise_list:
            score = None
            if score_array is not None and 0 <= int(idx) < len(score_array):
                raw_score = float(score_array[int(idx)])
                if np.isfinite(raw_score):
                    score = raw_score
            if score is not None and score >= self.outlier_score_merge_limit:
                hard_noise.append(int(idx))
            else:
                eligible_noise.append(int(idx))

        target_clusters = [list(cluster) for cluster in copied_clusters if len(cluster) >= 3]
        passthrough_clusters = [list(cluster) for cluster in copied_clusters if len(cluster) < 3]
        stats["eligible_noise_count"] = int(len(eligible_noise))
        stats["hard_noise_count"] = int(len(hard_noise))
        if not target_clusters:
            remaining_noise = list(eligible_noise) + list(hard_noise)
            stats["noise_count_after"] = int(len(remaining_noise))
            return remaining_noise, copied_clusters, stats

        medoid_indices = []
        spread_values = []
        for cluster in target_clusters:
            medoid_idx = self._cluster_medoid_index(features, cluster, metric_mode)
            medoid_indices.append(medoid_idx)
            cluster_feat = np.asarray(features[np.asarray(cluster, dtype=np.int64)], dtype=np.float32)
            medoid_feat = np.asarray(features[medoid_idx], dtype=np.float32)
            if metric_mode == 'euclidean':
                dists = np.linalg.norm(cluster_feat - medoid_feat, axis=1)
                valid = dists[dists > 1e-9]
            else:
                sims = cosine_similarity(cluster_feat, medoid_feat.reshape(1, -1)).reshape(-1)
                valid = sims[sims < 0.999999]
            if valid.size > 0:
                spread_values.append(valid.astype(np.float32))

        if metric_mode == 'euclidean':
            if not spread_values:
                remaining_noise = list(eligible_noise) + list(hard_noise)
                stats["noise_count_after"] = int(len(remaining_noise))
                return remaining_noise, copied_clusters, stats
            merge_threshold = float(np.percentile(np.concatenate(spread_values), 50.0))
        else:
            merge_threshold = float(self.singleton_merge_threshold)

        remaining_noise = list(hard_noise)
        merged_count = 0
        for idx in np.asarray(eligible_noise, dtype=np.int64):
            point = np.asarray(features[int(idx)], dtype=np.float32)
            best_cluster = None
            if metric_mode == 'euclidean':
                best_value = float('inf')
                for cluster, medoid_idx in zip(target_clusters, medoid_indices):
                    dist = float(np.linalg.norm(point - np.asarray(features[medoid_idx], dtype=np.float32)))
                    if dist < best_value:
                        best_value = dist
                        best_cluster = cluster
                if best_cluster is not None and best_value <= max(merge_threshold, 1e-6):
                    best_cluster.append(int(idx))
                    merged_count += 1
                else:
                    remaining_noise.append(int(idx))
            else:
                best_value = -1.0
                for cluster, medoid_idx in zip(target_clusters, medoid_indices):
                    sim = float(cosine_similarity(point.reshape(1, -1), np.asarray(features[medoid_idx], dtype=np.float32).reshape(1, -1))[0, 0])
                    if sim > best_value:
                        best_value = sim
                        best_cluster = cluster
                if best_cluster is not None and best_value >= merge_threshold:
                    best_cluster.append(int(idx))
                    merged_count += 1
                else:
                    remaining_noise.append(int(idx))

        stats["merged_noise_count"] = int(merged_count)
        stats["noise_count_after"] = int(len(remaining_noise))
        merged_clusters = target_clusters + passthrough_clusters
        if merged_count > 0 or stats["hard_noise_count"] > 0:
            print(
                "singleton post-merge:"
                f" 合并 {merged_count} 个噪声点,"
                f" 保留硬噪声 {stats['hard_noise_count']} 个"
            )
        return remaining_noise, merged_clusters, stats

    def cluster_from_features(self, features: np.ndarray) -> List[List[int]]:
        n = int(len(features))
        self.last_postmerge_stats = {
            "enabled": bool(self.enable_singleton_postmerge),
            "noise_count_before": 0,
            "eligible_noise_count": 0,
            "merged_noise_count": 0,
            "hard_noise_count": 0,
            "noise_count_after": 0,
            "outlier_score_threshold": float(self.outlier_score_merge_limit),
        }
        self.last_soft_membership = {}
        if n <= 0:
            return []
        if n == 1:
            return [[0]]

        effective_min_cluster_size = min(self.min_cluster_size, n)
        effective_min_samples = self.min_samples
        if effective_min_samples is not None:
            effective_min_samples = min(max(1, int(effective_min_samples)), n)
        prepared_features, runtime_metric, metric_label = self._prepare_hdbscan_input(features)

        print(
            "使用hdbscan包进行全样本聚类..."
            f" min_cluster_size={effective_min_cluster_size},"
            f" min_samples={effective_min_samples},"
            f" metric={metric_label},"
            f" selection={self.cluster_selection_method}"
        )
        core_dist_n_jobs = max(1, min(4, (os.cpu_count() or 1) // 2))
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=effective_min_cluster_size,
            min_samples=effective_min_samples,
            metric=runtime_metric,
            cluster_selection_method=self.cluster_selection_method,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            prediction_data=True,
            core_dist_n_jobs=core_dist_n_jobs,
        )
        labels = clusterer.fit_predict(prepared_features)
        soft_membership_matrix = None
        if hasattr(hdbscan, "all_points_membership_vectors"):
            try:
                soft_membership_matrix = hdbscan.all_points_membership_vectors(clusterer)
            except Exception:
                soft_membership_matrix = None

        unique_labels = sorted(set(int(v) for v in labels if int(v) >= 0))
        clusters = [np.where(labels == cid)[0].tolist() for cid in unique_labels]

        noise_indices = np.where(labels < 0)[0]
        metric_mode = 'cosine' if str(metric_label).startswith('cosine') else str(runtime_metric)
        noise_indices, clusters, postmerge_stats = self._postmerge_noise_singletons(
            prepared_features,
            clusters,
            noise_indices,
            metric_mode=metric_mode,
            outlier_scores=getattr(clusterer, "outlier_scores_", None),
        )
        postmerge_stats["core_dist_n_jobs"] = int(core_dist_n_jobs)
        self.last_postmerge_stats = postmerge_stats
        for idx in noise_indices:
            clusters.append([int(idx)])

        clusters.sort(key=lambda c: (-len(c), c[0]))
        self.last_soft_membership = self._build_soft_membership_summary(
            raw_labels=np.asarray(labels, dtype=np.int64),
            raw_cluster_labels=[int(v) for v in unique_labels],
            membership_matrix=soft_membership_matrix,
            final_clusters=clusters,
        )
        return clusters

class RepresentativeSelector:
    """选择每个聚类代表性中心窗口的类。"""

    def select_representatives(self, clusters: List[List[int]],
                               similarity_provider: Optional[Any] = None) -> List[int]:
        """使用 medoid 选择每个聚类的代表性样本。"""
        return [
            self._select_by_medoid(cluster, similarity_provider)
            for cluster in clusters
            if cluster
        ]

    def _select_by_medoid(self, cluster: List[int],
                          similarity_provider: Optional[Any]) -> int:
        cluster_sim = similarity_provider(cluster) if similarity_provider is not None else None
        if cluster_sim is None or cluster_sim.size == 0:
            return cluster[0]

        best_medoid = -1
        best_avg_similarity = -1.0
        for local_i, global_i in enumerate(cluster):
            row = np.asarray(cluster_sim[local_i], dtype=np.float32)
            if len(cluster) > 1:
                avg_sim = float((np.sum(row) - row[local_i]) / (len(cluster) - 1))
            else:
                avg_sim = 1.0
            if avg_sim > best_avg_similarity:
                best_avg_similarity = avg_sim
                best_medoid = global_i
        return best_medoid


def _extract_feature_worker(task):
    """特征提取worker函数"""
    if not isinstance(task, (tuple, list)) or len(task) != 4:
        raise ValueError(
            "_extract_feature_worker expects a 4-item task: "
            "(idx, filepath, sample_info, feature_config)"
        )
    idx, filepath, sample_info, feature_config = task

    try:
        extractor = LayoutFeatureExtractor()
        feat = _extract_feature_with_extractor(
            extractor,
            filepath,
            sample_info=sample_info,
            feature_config=feature_config,
        )
        return idx, filepath, np.asarray(feat, dtype=np.float32), None
    except Exception as e:
        return idx, filepath, None, str(e)


def _extract_feature_with_extractor(extractor, filepath, sample_info=None, feature_config=None):
    """
    使用给定 extractor 提取单个文件的特征。

    这里默认不再重复执行 layer operations，因为 load_files 阶段已经根据配置
    对输入文件完成了预处理；特征提取阶段只负责读取结果并生成向量。
    """
    feature_config = feature_config or {}
    if sample_info is not None:
        return extractor.extract_from_window_sample(
            filepath,
            sample_info=sample_info,
            apply_layer_operations=False,
            inner_weight=float(feature_config.get("inner_feature_weight", 1.0)),
            outer_weight=float(feature_config.get("outer_feature_weight", 0.35)),
            inner_block_weights=feature_config.get("inner_block_weights"),
            outer_block_weights=feature_config.get("outer_block_weights"),
        )
    return extractor.extract_from_layout(filepath, apply_layer_operations=False)


class LayoutClusteringPipeline:
    """
    Layout中心窗口聚类分析管道。

    负责把“窗口生成 -> 特征提取 -> 聚类 -> 代表样本选择”
    串成完整流程，并在大版图场景下保留窗口级元数据与去重统计。
    """

    def _cfg(self, key: str, default: Any) -> Any:
        return self.config.get(key, default)

    def _cfg_bool(self, key: str, default: bool) -> bool:
        return bool(self._cfg(key, default))

    def _cfg_int(self, key: str, default: int, minimum: Optional[int] = None) -> int:
        value = int(self._cfg(key, default))
        if minimum is not None:
            value = max(int(minimum), value)
        return value

    def _cfg_float(self, key: str, default: float, minimum: Optional[float] = None) -> float:
        value = float(self._cfg(key, default))
        if minimum is not None:
            value = max(float(minimum), value)
        return value

    def _init_components(self) -> None:
        self.feature_extractor = LayoutFeatureExtractor()
        self.similarity_calculator = SimilarityCalculator(
            method=self._cfg('similarity_method', 'euclidean'),
        )
        self.clustering_algorithm = HDBSCANClustering(
            min_cluster_size=self._cfg_int('min_cluster_size', 8, minimum=2),
            min_samples=self._cfg('min_samples', 4),
            cluster_selection_epsilon=self._cfg_float('cluster_selection_epsilon', 0.0, minimum=0.0),
            cluster_selection_method=self._cfg('cluster_selection_method', 'eom'),
            metric=self._cfg('similarity_method', 'euclidean'),
            enable_singleton_postmerge=self._cfg_bool('enable_singleton_postmerge', True),
            singleton_merge_threshold=self._cfg_float('singleton_merge_threshold', 0.93, minimum=0.0),
        )
        self.representative_selector = RepresentativeSelector()

    def _init_pipeline_settings(self) -> None:
        self.apply_layer_operations = self._cfg_bool('apply_layer_operations', True)
        self.clip_size_um = self._cfg_float('clip_size_um', 1.35)
        self.context_width_um = self._cfg_float('context_width_um', 0.675)
        self.inner_feature_weight = self._cfg_float('inner_feature_weight', 1.0)
        self.outer_feature_weight = self._cfg_float('outer_feature_weight', 0.35)
        self.inner_block_weights = _normalize_feature_block_weights(
            self._cfg('inner_block_weights', None),
            default_weights=DEFAULT_INNER_BLOCK_WEIGHTS,
        )
        self.outer_block_weights = _normalize_feature_block_weights(
            self._cfg('outer_block_weights', None),
            default_weights=DEFAULT_OUTER_BLOCK_WEIGHTS,
        )
        self.sample_similarity_threshold = self._cfg_float('sample_similarity_threshold', 0.96)
        self.window_signature_bins = self._cfg_int('window_signature_bins', 20, minimum=1)
        self.enable_coarse_prefilter = self._cfg_bool('enable_coarse_prefilter', True)
        self.coarse_prefilter_quant_um = self._cfg_float('coarse_prefilter_quant_um', 0.02, minimum=0.0)
        self.candidate_bin_size_um = self._cfg('candidate_bin_size_um', None)
        self.relation_seed_ratio = self._cfg_float('relation_seed_ratio', 0.2, minimum=0.0)
        self.relation_gap_threshold_um = self._cfg_float('relation_gap_threshold_um', 0.08, minimum=0.0)
        self.max_elements_per_window = self._cfg_int('max_elements_per_window', 256, minimum=1)
        self.enable_clip_shifting = self._cfg_bool('enable_clip_shifting', True)
        self.clip_shift_directions = self._cfg('clip_shift_directions', "left,right,up,down")
        self.clip_shift_neighbor_limit = self._cfg_int('clip_shift_neighbor_limit', 128, minimum=1)
        self.clip_shift_boundary_tolerance_um = self._cfg_float('clip_shift_boundary_tolerance_um', 0.02, minimum=0.0)
        self.split_progress_every = self._cfg_int('split_progress_every', 200, minimum=1)
        self.feature_progress_every = self._cfg_int('feature_progress_every', 50, minimum=1)
        self.feature_workers = self._cfg_int('feature_workers', 2, minimum=1)
        self.feature_parallel_threshold = self._cfg_int('feature_parallel_threshold', 64, minimum=1)
        self.enable_geometry_dedup = self._cfg_bool('enable_geometry_dedup', True)
        self.hash_precision_nm = self._cfg_float('hash_precision_nm', 5.0, minimum=0.0)

    def _init_runtime_state(self) -> None:
        self.filepaths = []
        self.features = None
        self.precluster_features = None
        self.clusters = []
        self.representatives = []
        self.pattern_weights = []
        self.pattern_dedup_info = {}
        self.active_feature_names = list(self.feature_extractor.feature_names)
        self.feature_selector = None
        self.feature_scaler = None
        self.feature_space_info = {}
        self.sample_infos = []
        self.cluster_review_info = {}
        self._owned_temp_dir = None

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._init_components()
        self._init_pipeline_settings()
        self._init_runtime_state()

    def _create_run_temp_dir(self) -> Path:
        if self._owned_temp_dir is not None:
            shutil.rmtree(self._owned_temp_dir, ignore_errors=True)
        preferred_root = Path(__file__).resolve().parent / "_temp_runs"
        preferred_root.mkdir(parents=True, exist_ok=True)
        self._owned_temp_dir = preferred_root / f"layout_clustering_hdbscan_v5_simp_{uuid.uuid4().hex[:8]}"
        self._owned_temp_dir.mkdir(parents=True, exist_ok=False)
        return self._owned_temp_dir

    def _read_layout_library(self, filepath: str):
        return _read_oas_only_library(filepath)

    def _append_direct_file(self, filepath: str, temp_dir: Path,
                            processed_files: List[str], processed_infos: List[Optional[Dict[str, Any]]],
                            weights: List[int]) -> None:
        _ensure_oas_input_path(filepath)
        if self.apply_layer_operations:
            print(f" 对文件 {os.path.basename(filepath)} 应用层操作...")
            lib = self._read_layout_library(filepath)
            lib = self.feature_extractor.layer_processor.apply_layer_operations(lib)
            temp_dir.mkdir(parents=True, exist_ok=True)
            processed_path = _make_ascii_temp_output_path(
                temp_dir,
                prefix="processed",
                source_path=filepath,
                suffix=".oas",
            )
            _write_oas_library(lib, str(processed_path))
            processed_files.append(str(processed_path))
        else:
            processed_files.append(filepath)
        processed_infos.append(None)
        weights.append(1)

    def _append_window_files(self, filepath: str, temp_dir: Path,
                             processed_files: List[str], processed_infos: List[Optional[Dict[str, Any]]],
                             weights: List[int]) -> Dict[str, Any]:
        lib = self._read_layout_library(filepath)
        if self.apply_layer_operations:
            print(f" 对文件 {os.path.basename(filepath)} 应用层操作...")
            lib = self.feature_extractor.layer_processor.apply_layer_operations(lib)

        sample_libs, sample_infos, split_meta = generate_layout_window_samples(
            lib,
            window_size_um=self.clip_size_um,
            context_width_um=self.context_width_um,
            progress_every=self.split_progress_every,
            enable_geometry_dedup=self.enable_geometry_dedup,
            hash_precision_nm=self.hash_precision_nm,
            similarity_threshold=self.sample_similarity_threshold,
            signature_bins=self.window_signature_bins,
            coarse_dedup_quant_um=self.coarse_prefilter_quant_um,
            enable_coarse_prefilter=self.enable_coarse_prefilter,
            candidate_bin_size_um=self.candidate_bin_size_um,
            relation_seed_ratio=self.relation_seed_ratio,
            relation_gap_threshold_um=self.relation_gap_threshold_um,
            max_elements_per_window=self.max_elements_per_window,
            enable_clip_shifting=self.enable_clip_shifting,
            clip_shift_directions=self.clip_shift_directions,
            clip_shift_neighbor_limit=self.clip_shift_neighbor_limit,
            clip_shift_boundary_tolerance_um=self.clip_shift_boundary_tolerance_um,
            return_metadata=True,
            source_name=os.path.basename(filepath),
        )

        for j, (sample_lib, sample_info) in enumerate(zip(sample_libs, sample_infos)):
            temp_dir.mkdir(parents=True, exist_ok=True)
            clip_path = _make_ascii_temp_output_path(
                temp_dir,
                prefix="window",
                source_path=filepath,
                suffix=".oas",
                index=j,
            )
            _write_oas_library(sample_lib, str(clip_path))
            processed_files.append(str(clip_path))
            processed_infos.append(sample_info)
            weights.append(int(sample_info.get("duplicate_count", 1)))
        print(f" 生成 {len(sample_infos)} 个去重窗口样本")
        return split_meta

    def _record_split_metadata(self, split_meta: Dict[str, Any],
                               dedup_meta_template: Dict[str, Any],
                               dedup_stats: Dict[str, int]) -> None:
        for key in (
                'candidate_bin_size_um',
                'max_elements_per_window',
                'window_size_um',
                'context_width_um',
                'hash_precision_nm',
                'clip_shifting_enabled',
        ):
            if key in split_meta and key not in dedup_meta_template:
                dedup_meta_template[key] = split_meta[key]
        for key in (
                'raw_center_count',
                'unique_window_count',
                'exact_hash_merged',
                'similar_window_merged',
                'original_element_count',
                'shifted_seed_count',
                'element_seed_count',
                'relation_seed_count',
                'hotspot_seed_count',
                'dedup_bucket_count',
        ):
            dedup_stats[key] += int(split_meta.get(key, 0))

    def _load_single_layout_file(self, filepath: str, temp_dir: Path, split_large_layout: bool,
                                 dedup_meta_template: Dict[str, Any], dedup_stats: Dict[str, int],
                                 processed_files: List[str], processed_infos: List[Optional[Dict[str, Any]]],
                                 weights: List[int]) -> None:
        if not split_large_layout:
            self._append_direct_file(filepath, temp_dir, processed_files, processed_infos, weights)
            return

        try:
            is_large, width, height = check_if_large_layout(filepath)
            if is_large:
                print(f"检测到大版图 ({width:.2f} x {height:.2f} um)，开始基于中心窗口采样...")
                split_meta = self._append_window_files(
                    filepath,
                    temp_dir,
                    processed_files,
                    processed_infos,
                    weights,
                )
                self._record_split_metadata(split_meta, dedup_meta_template, dedup_stats)
                return

            print(f"文件不是大版图 ({width:.2f} x {height:.2f} um)，按原始文件处理")
        except Exception as e:
            print(f"中心窗口采样失败: {e}")

        self._append_direct_file(filepath, temp_dir, processed_files, processed_infos, weights)

    def _build_pattern_dedup_info(self, dedup_meta_template: Dict[str, Any],
                                  dedup_stats: Dict[str, int]) -> Dict[str, Any]:
        return {
            'raw_center_count': int(dedup_stats['raw_center_count']),
            'unique_window_count': int(dedup_stats['unique_window_count']),
            'compression_ratio': float(dedup_stats['unique_window_count']) / float(max(1, dedup_stats['raw_center_count'])),
            'exact_hash_merged': int(dedup_stats['exact_hash_merged']),
            'similar_window_merged': int(dedup_stats['similar_window_merged']),
            'candidate_bin_size_um': dedup_meta_template.get(
                'candidate_bin_size_um',
                None if self.candidate_bin_size_um is None else float(self.candidate_bin_size_um)
            ),
            'max_elements_per_window': int(dedup_meta_template.get('max_elements_per_window', self.max_elements_per_window)),
            'window_size_um': float(dedup_meta_template.get('window_size_um', self.clip_size_um)),
            'context_width_um': float(dedup_meta_template.get('context_width_um', self.context_width_um)),
            'original_element_count': int(dedup_stats.get('original_element_count', 0)),
            'hash_precision_nm': float(dedup_meta_template.get('hash_precision_nm', self.hash_precision_nm)),
            'clip_shifting_enabled': bool(dedup_meta_template.get('clip_shifting_enabled', self.enable_clip_shifting)),
            'shifted_seed_count': int(dedup_stats.get('shifted_seed_count', 0)),
            'element_seed_count': int(dedup_stats.get('element_seed_count', dedup_stats['raw_center_count'])),
            'relation_seed_count': int(dedup_stats.get('relation_seed_count', 0)),
            'hotspot_seed_count': int(dedup_stats.get('hotspot_seed_count', 0)),
            'dedup_bucket_count': int(dedup_stats.get('dedup_bucket_count', 0)),
        }

    def _features_for_indexing(self) -> np.ndarray:
        if self.features is None:
            return np.empty((0, 0), dtype=np.float32)
        if self.features.ndim == 1:
            return self.features.reshape(1, -1)
        return self.features

    def _sample_shift_distance(self, sample_info: Optional[Dict[str, Any]]) -> float:
        if not sample_info:
            return 0.0
        shift = sample_info.get("center_shift")
        if not shift or len(shift) != 2:
            return 0.0
        return float(math.hypot(float(shift[0]), float(shift[1])))

    def _select_interpret_similarity_indices(self, cluster_indices: List[int]) -> Tuple[List[int], bool]:
        indices = sorted({int(v) for v in cluster_indices})
        if len(indices) <= MAX_INTERPRET_SIMILARITY_SAMPLES:
            return indices, True
        rng = np.random.default_rng(42)
        sampled = rng.choice(
            np.asarray(indices, dtype=np.int64),
            MAX_INTERPRET_SIMILARITY_SAMPLES,
            replace=False,
        )
        return sorted(int(v) for v in sampled.tolist()), False

    def _build_feature_space_info(self, log_transform_enabled: bool,
                                  standardize_enabled: bool) -> Dict[str, str]:
        precluster_steps = []
        if log_transform_enabled:
            precluster_steps.append("log1p")
        precluster_steps.append("variance_threshold(1e-4)")
        cluster_steps = list(precluster_steps)
        if standardize_enabled:
            cluster_steps.append("robust_scaler")
        return {
            "cluster": " -> ".join(cluster_steps),
            "pre_cluster": " -> ".join(precluster_steps),
            "output_features": "cluster_space",
        }

    def _mean_similarity_from_submatrix(self, similarity_submatrix: np.ndarray) -> float:
        sim = np.asarray(similarity_submatrix, dtype=np.float32)
        if sim.ndim != 2 or sim.size == 0:
            return 0.0
        n = int(sim.shape[0])
        if n <= 1:
            return 1.0
        total = float(np.sum(sim) - np.trace(sim))
        return float(total / float(max(1, n * (n - 1))))

    def _compute_intra_cluster_avg_similarity(self, cluster_indices: List[int]) -> Tuple[float, int, bool]:
        if not cluster_indices:
            return 0.0, 0, True
        similarity_indices, exact = self._select_interpret_similarity_indices(cluster_indices)
        similarity_submatrix = self.similarity_calculator.compute_similarity_submatrix_from_features(
            self.features,
            similarity_indices,
        )
        return (
            self._mean_similarity_from_submatrix(similarity_submatrix),
            int(len(similarity_indices)),
            bool(exact),
        )

    def _build_window_interpretation_summary(self) -> Dict[str, Any]:
        sample_infos = [info for info in self.sample_infos if info is not None]
        if not sample_infos:
            return {}

        duplicate_counts = np.asarray(
            [int(info.get("duplicate_count", 1)) for info in sample_infos],
            dtype=np.int32,
        )
        shift_distances = np.asarray(
            [self._sample_shift_distance(info) for info in sample_infos],
            dtype=np.float32,
        )
        shifted_mask = shift_distances > 1e-9

        return {
            "window_count": int(len(sample_infos)),
            "duplicate_weight_sum": int(np.sum(duplicate_counts)),
            "avg_duplicate_count": float(np.mean(duplicate_counts)),
            "max_duplicate_count": int(np.max(duplicate_counts)),
            "shifted_window_count": int(np.sum(shifted_mask)),
            "shifted_window_ratio": float(np.mean(shifted_mask.astype(np.float32))),
            "avg_shift_distance_um": float(np.mean(shift_distances)),
            "max_shift_distance_um": float(np.max(shift_distances)),
        }

    def _build_cluster_interpretation_summary(self, cluster_indices: List[int],
                                              representative_index: int,
                                              representative_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        intra_cluster_avg_similarity, similarity_sample_count, similarity_exact = (
            self._compute_intra_cluster_avg_similarity(cluster_indices)
        )
        cluster_sample_infos = [
            self.sample_infos[idx] if idx < len(self.sample_infos) else None
            for idx in cluster_indices
        ]
        valid_infos = [info for info in cluster_sample_infos if info is not None]
        if not valid_infos:
            return {
                "duplicate_weight_sum": int(len(cluster_indices)),
                "avg_duplicate_count": 1.0,
                "shifted_sample_count": 0,
                "shifted_sample_ratio": 0.0,
                "source_names": [],
                "representative_duplicate_count": int(1),
                "representative_shift_distance_um": 0.0,
                "intra_cluster_avg_similarity": float(intra_cluster_avg_similarity),
                "intra_cluster_similarity_sample_count": int(similarity_sample_count),
                "intra_cluster_similarity_exact": bool(similarity_exact),
            }

        duplicate_counts = np.asarray(
            [int(info.get("duplicate_count", 1)) for info in valid_infos],
            dtype=np.int32,
        )
        shift_distances = np.asarray(
            [self._sample_shift_distance(info) for info in valid_infos],
            dtype=np.float32,
        )
        source_names = sorted({str(info.get("source_name", "")) for info in valid_infos if info.get("source_name")})

        return {
            "duplicate_weight_sum": int(np.sum(duplicate_counts)),
            "avg_duplicate_count": float(np.mean(duplicate_counts)),
            "shifted_sample_count": int(np.sum(shift_distances > 1e-9)),
            "shifted_sample_ratio": float(np.mean((shift_distances > 1e-9).astype(np.float32))),
            "source_names": source_names,
            "representative_duplicate_count": int(
                (representative_metadata or {}).get("duplicate_count", 1)
            ),
            "representative_shift_distance_um": float(
                self._sample_shift_distance(representative_metadata)
            ),
            "intra_cluster_avg_similarity": float(intra_cluster_avg_similarity),
            "intra_cluster_similarity_sample_count": int(similarity_sample_count),
            "intra_cluster_similarity_exact": bool(similarity_exact),
        }

    def _build_result_summary(self, cluster_details: List[Dict[str, Any]]) -> Dict[str, Any]:
        cluster_sizes = [int(c["size"]) for c in cluster_details]
        if cluster_sizes:
            largest_cluster = max(cluster_sizes)
            smallest_cluster = min(cluster_sizes)
            mean_cluster = float(np.mean(cluster_sizes))
        else:
            largest_cluster = 0
            smallest_cluster = 0
            mean_cluster = 0.0

        summary = {
            "input_file_count": int(len(self.filepaths)),
            "cluster_count": int(len(cluster_details)),
            "sample_count": int(sum(cluster_sizes)),
            "largest_cluster_size": int(largest_cluster),
            "smallest_cluster_size": int(smallest_cluster),
            "mean_cluster_size": float(mean_cluster),
        }

        if self.pattern_dedup_info:
            raw_centers = int(self.pattern_dedup_info.get("raw_center_count", 0))
            unique_windows = int(self.pattern_dedup_info.get("unique_window_count", 0))
            shifted_seeds = int(self.pattern_dedup_info.get("shifted_seed_count", 0))
            summary["window_generation"] = {
                "raw_center_count": raw_centers,
                "unique_window_count": unique_windows,
                "dedup_reduction_ratio": float(1.0 - (unique_windows / max(1, raw_centers))),
                "shifted_seed_count": shifted_seeds,
                "shifted_seed_ratio": float(shifted_seeds / max(1, int(self.pattern_dedup_info.get("initial_seed_count", raw_centers)))),
            }

        postmerge_stats = getattr(self.clustering_algorithm, "last_postmerge_stats", None)
        if postmerge_stats:
            summary["singleton_postmerge"] = {
                "noise_count_before": int(postmerge_stats.get("noise_count_before", 0)),
                "eligible_noise_count": int(postmerge_stats.get("eligible_noise_count", 0)),
                "merged_noise_count": int(postmerge_stats.get("merged_noise_count", 0)),
                "hard_noise_count": int(postmerge_stats.get("hard_noise_count", 0)),
                "noise_count_after": int(postmerge_stats.get("noise_count_after", 0)),
            }

        soft_membership = getattr(self.clustering_algorithm, "last_soft_membership", None)
        if soft_membership and soft_membership.get("enabled"):
            probs = np.asarray(soft_membership.get("max_membership_probability", []), dtype=np.float32)
            if probs.size > 0:
                summary["soft_membership"] = {
                    "mean_max_membership_probability": float(np.mean(probs)),
                    "median_max_membership_probability": float(np.median(probs)),
                    "low_confidence_count": int(np.sum(probs < 0.5)),
                }

        window_summary = self._build_window_interpretation_summary()
        if window_summary:
            summary["window_samples"] = window_summary
        return summary

    def load_files(self, input_dir: str, pattern: str = "*.oas", split_large_layout: bool = True) -> List[str]:
        """加载 OASIS 文件"""
        input_path = Path(input_dir)
        temp_dir = self._create_run_temp_dir()
        self.filepaths = []
        self.sample_infos = []
        self.pattern_weights = []
        self.pattern_dedup_info = {}

        dedup_stats = {
            'raw_center_count': 0,
            'unique_window_count': 0,
            'exact_hash_merged': 0,
            'similar_window_merged': 0,
            'original_element_count': 0,
            'shifted_seed_count': 0,
            'element_seed_count': 0,
            'relation_seed_count': 0,
            'hotspot_seed_count': 0,
            'dedup_bucket_count': 0,
        }
        dedup_meta_template = {}

        if input_path.is_file():
            filepath = str(input_path)
            _ensure_oas_input_path(filepath)
            self._load_single_layout_file(
                filepath,
                temp_dir,
                split_large_layout,
                dedup_meta_template,
                dedup_stats,
                self.filepaths,
                self.sample_infos,
                self.pattern_weights,
            )
        else:
            oasis_files = list(input_path.glob("*.oas"))
            all_files = list(oasis_files)

            if pattern != "*.oas":
                pattern_files = [
                    f for f in input_path.glob(pattern)
                    if f.is_file() and f.suffix.lower() == ".oas"
                ]
                all_files = sorted(set(all_files + pattern_files))

            if split_large_layout:
                print("检查目录中的文件是否为大版图...")
                for i, file_path in enumerate(all_files):
                    filepath = str(file_path)
                    print(f"检查文件 {i + 1}/{len(all_files)}: {os.path.basename(filepath)}")
                    self._load_single_layout_file(
                        filepath,
                        temp_dir,
                        True,
                        dedup_meta_template,
                        dedup_stats,
                        self.filepaths,
                        self.sample_infos,
                        self.pattern_weights,
                    )
            else:
                for filepath in [str(f) for f in all_files]:
                    self._append_direct_file(filepath, temp_dir, self.filepaths, self.sample_infos, self.pattern_weights)
            print(f"找到 {len(self.filepaths)} 个文件")

        if len(self.sample_infos) != len(self.filepaths):
            self.sample_infos = [None] * len(self.filepaths)
        if len(self.pattern_weights) != len(self.filepaths):
            self.pattern_weights = [1] * len(self.filepaths)

        if dedup_stats['raw_center_count'] > 0:
            self.pattern_dedup_info = self._build_pattern_dedup_info(dedup_meta_template, dedup_stats)

        return self.filepaths

    def _build_feature_tasks(self, feature_config: Dict[str, Any]) -> List[Tuple[int, str, Optional[Dict[str, Any]], Dict[str, Any]]]:
        return [
            (
                i,
                fp,
                self.sample_infos[i] if i < len(self.sample_infos) else None,
                feature_config,
            )
            for i, fp in enumerate(self.filepaths)
        ]

    def _iter_feature_results(self, tasks):
        total_files = len(tasks)
        use_parallel = (
            self.feature_workers > 1
            and total_files >= self.feature_parallel_threshold
        )
        if use_parallel:
            print(f"并行处理: workers={self.feature_workers}, files={total_files}")
            completed = 0
            old_silent_flag = os.environ.get("LC_SILENT_CHILD")
            os.environ["LC_SILENT_CHILD"] = "1"
            try:
                try:
                    with ProcessPoolExecutor(max_workers=self.feature_workers) as executor:
                        futures = [executor.submit(_extract_feature_worker, task) for task in tasks]
                        for future in as_completed(futures):
                            idx, filepath, feat, err = future.result()
                            completed += 1
                            if (
                                completed == 1
                                or completed == total_files
                                or (completed % self.feature_progress_every == 0)
                            ):
                                print(f"处理 {completed}/{total_files}: {os.path.basename(filepath)}")
                            yield idx, filepath, feat, err
                    return
                except PermissionError as e:
                    print(f"并行特征提取不可用，回退到串行模式: {e}")
            finally:
                if old_silent_flag is None:
                    os.environ.pop("LC_SILENT_CHILD", None)
                else:
                    os.environ["LC_SILENT_CHILD"] = old_silent_flag

        for i, (idx, filepath, sample_info, feature_config) in enumerate(tasks):
            if (
                i == 0
                or i == total_files - 1
                or ((i + 1) % self.feature_progress_every == 0)
            ):
                print(f"处理 {i + 1}/{total_files}: {os.path.basename(filepath)}")
            try:
                feat = _extract_feature_with_extractor(
                    self.feature_extractor,
                    filepath,
                    sample_info=sample_info,
                    feature_config=feature_config,
                )
                yield idx, filepath, np.asarray(feat, dtype=np.float32), None
            except Exception as e:
                yield idx, filepath, None, str(e)

    def extract_features(self) -> np.ndarray:
        """从所有文件中提取特征并合并"""
        if not self.filepaths:
            raise ValueError("没有加载任何文件")
        self.active_feature_names = list(self.feature_extractor.feature_names)
        self.feature_selector = None
        self.feature_scaler = None
        self.precluster_features = None

        features_list = []
        valid_files = []
        valid_sample_infos = []
        total_files = len(self.filepaths)
        feature_config = {
            "inner_feature_weight": self.inner_feature_weight,
            "outer_feature_weight": self.outer_feature_weight,
            "inner_block_weights": dict(self.inner_block_weights),
            "outer_block_weights": dict(self.outer_block_weights),
        }
        tasks = self._build_feature_tasks(feature_config)
        collected = []
        for idx, filepath, feat, err in self._iter_feature_results(tasks):
            if err is not None or feat is None:
                print(f"文件 {filepath} 提取失败: {err}")
                continue
            sample_info = self.sample_infos[idx] if idx < len(self.sample_infos) else None
            collected.append((idx, filepath, feat, sample_info))

        collected.sort(key=lambda x: x[0])
        for _, filepath, feat, sample_info in collected:
            valid_files.append(filepath)
            valid_sample_infos.append(sample_info)
            features_list.append(feat)

        # 更新文件路径列表（仅包含成功处理的文件）
        self.filepaths = valid_files
        self.sample_infos = valid_sample_infos

        # 检查是否成功提取到特征
        if len(features_list) == 0:
            raise ValueError("没有成功提取任何特征向量")

        base_features = np.asarray(features_list, dtype=np.float32)
        log_transform_enabled = bool(self.config.get('log_transform', True))
        standardize_enabled = bool(self.config.get('standardize', True))
        self.feature_space_info = self._build_feature_space_info(
            log_transform_enabled,
            standardize_enabled,
        )

        # 对长尾分布特征做对数压缩，减少面积/周长等量纲主导
        if log_transform_enabled:
            base_features = np.log1p(np.maximum(base_features, 0.0))

        if base_features.ndim == 1:
            if len(base_features) > 0:
                base_features = base_features.reshape(1, -1)
            else:
                print("警告：没有有效的特征数据")
        elif base_features.ndim != 2:
            print(f"特征维度异常: {base_features.shape}")

        transformed = np.asarray(base_features, dtype=np.float32)
        selector = VarianceThreshold(threshold=1e-4)
        selected = transformed
        selected_names = list(self.feature_extractor.feature_names)
        try:
            candidate = selector.fit_transform(transformed)
            if candidate.shape[1] > 0:
                mask = selector.get_support()
                selected = candidate
                selected_names = [
                    name for name, keep in zip(self.feature_extractor.feature_names, mask) if keep
                ]
                if candidate.shape[1] != transformed.shape[1]:
                    self.feature_selector = selector
                    print(f"低方差过滤后保留 {candidate.shape[1]}/{transformed.shape[1]} 个特征")
        except ValueError:
            selected = transformed
            selected_names = list(self.feature_extractor.feature_names)

        # precluster_features 保持在 log1p + VarianceThreshold 空间，用于解释；
        # self.features 是最终聚类空间（额外经过 RobustScaler，如果启用）。
        self.precluster_features = np.asarray(selected, dtype=np.float32)
        self.active_feature_names = list(selected_names)

        if standardize_enabled and len(self.precluster_features) > 0 and self.precluster_features.ndim == 2:
            self.feature_scaler = RobustScaler(quantile_range=(25.0, 75.0))
            self.features = self.feature_scaler.fit_transform(self.precluster_features)
        else:
            self.features = np.asarray(self.precluster_features, dtype=np.float32)

        self.features = np.asarray(self.features, dtype=np.float32)

        print(f"提取的特征矩阵: {self.features.shape}")
        return self.features

    def perform_clustering(self) -> List[List[int]]:
        """执行聚类分析"""
        n_samples = len(self.features) if self.features is not None else 0
        print(f"聚类样本数: {n_samples}")
        self.clusters = self.clustering_algorithm.cluster_from_features(self.features)

        print(f"发现 {len(self.clusters)} 个聚类")
        for i, cluster in enumerate(self.clusters):
            print(f" 聚类 {i}: {len(cluster)} 个样本")
        return self.clusters

    def select_representatives(self) -> List[int]:
        """选择每个聚类的代表性样本"""
        if not self.clusters:
            self.perform_clustering()

        print("选择代表性样本...")
        self.representatives = self.representative_selector.select_representatives(
            self.clusters,
            similarity_provider=lambda cluster: self.similarity_calculator.compute_similarity_submatrix_from_features(
                self.features, cluster
            ),
        )
        for i, (cluster, rep) in enumerate(zip(self.clusters, self.representatives)):
            if 0 <= rep < len(self.filepaths):
                print(f" 聚类 {i}: 代表样本 {rep} ({os.path.basename(self.filepaths[rep])})")
            else:
                print(f" 聚类 {i}: 代表样本 {rep}")
        return self.representatives

    def run_pipeline(self, input_path: str, split_large_layout: bool = False) -> Dict[str, Any]:
        """运行完整的聚类分析管道"""
        # 1. 加载文件
        self.load_files(input_path, split_large_layout=split_large_layout)
        if not self.filepaths:
            return {"error": "没有找到可处理的文件"}

        # 2. 提取特征
        self.extract_features()

        # 3. 执行聚类
        self.perform_clustering()

        # 4. 选择代表性样本
        self.select_representatives()

        # 5. 返回结果
        result = self.get_results()
        return result

    def get_results(self) -> Dict[str, Any]:
        """获取聚类结果"""
        if not self.clusters or not self.representatives:
            return {}

        # 构建详细结果
        cluster_details = []
        features_for_indexing = self._features_for_indexing()
        for i, (cluster, rep_idx) in enumerate(zip(self.clusters, self.representatives)):
            # 确保cluster中的索引在有效范围内
            valid_cluster_indices = [idx for idx in cluster if 0 <= idx < len(self.features)]
            if not valid_cluster_indices:
                print(f"警告: 聚类 {i} 中的所有索引都无效，跳过此聚类")
                continue

            cluster_files = [self.filepaths[idx] for idx in valid_cluster_indices]
            representative_file = self.filepaths[rep_idx]
            cluster_metadata = [
                self.sample_infos[idx] if idx < len(self.sample_infos) else None
                for idx in valid_cluster_indices
            ]
            representative_metadata = self.sample_infos[rep_idx] if rep_idx < len(self.sample_infos) else None

            cluster_features = features_for_indexing[valid_cluster_indices]
            cluster_mean = np.mean(cluster_features, axis=0).tolist()
            cluster_std = np.std(cluster_features, axis=0).tolist()

            cluster_details.append({
                "cluster_id": i,
                "size": len(valid_cluster_indices),
                "sample_indices": valid_cluster_indices,
                "sample_files": cluster_files,
                "sample_metadata": cluster_metadata,
                "representative_index": rep_idx,
                "representative_file": representative_file,
                "representative_metadata": representative_metadata,
                "interpretation": self._build_cluster_interpretation_summary(
                    valid_cluster_indices,
                    rep_idx,
                    representative_metadata,
                ),
                "cluster_mean_features": cluster_mean,
                "cluster_std_features": cluster_std
            })

        # # 总体统计
        # total_samples = sum(len(c) for c in self.clusters)
        # cluster_sizes = [len(c) for c in self.clusters]

        result = {
            "total_files": len(self.filepaths),
            "total_clusters": len(cluster_details),  # 修正：使用实际有效的聚类数
            "total_samples": sum(len(c["sample_indices"]) for c in cluster_details),  # 修正：使用实际有效的样本数
            "cluster_sizes": [c["size"] for c in cluster_details],
            "result_summary": self._build_result_summary(cluster_details),
            "singleton_postmerge": getattr(self.clustering_algorithm, "last_postmerge_stats", {}),
            "soft_membership": getattr(self.clustering_algorithm, "last_soft_membership", {}),
            "clusters": cluster_details,
            "file_list": self.filepaths,
            "file_metadata": self.sample_infos,
            "feature_names": list(self.active_feature_names),
            "feature_space": dict(self.feature_space_info),
            "cluster_review": dict(self.cluster_review_info),
            "config": self.config,
            "pattern_dedup": self.pattern_dedup_info
        }
        return result

    def save_results(self, output_path: str, format: str = "json"):
        """保存聚类结果"""
        result = self.get_results()
        if format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"结果已保存到: {output_path}")
        elif format == "txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("半导体Layout中心窗口聚类分析结果\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"总文件数: {result['total_files']}\n")
                f.write(f"总聚类数: {result['total_clusters']}\n")
                f.write(f"总样本数: {result['total_samples']}\n\n")
                f.write("聚类大小分布:\n")
                for i, size in enumerate(result['cluster_sizes']):
                    f.write(f" 聚类 {i}: {size} 个样本\n")
                f.write("\n")
                f.write("详细聚类信息:\n")
                for cluster in result['clusters']:
                    f.write(f"\n聚类 {cluster['cluster_id']} (大小: {cluster['size']}):\n")
                    f.write(f" 代表文件: {os.path.basename(cluster['representative_file'])}\n")
                    f.write(f" 代表索引: {cluster['representative_index']}\n")
                    f.write(f" 样本文件:\n")
                    for file in cluster['sample_files']:
                        f.write(f" - {os.path.basename(file)}\n")
            print(f"结果已保存到: {output_path}")
        else:
            print(f"不支持的格式: {format}")

    def export_cluster_review(self, output_dir: str) -> Dict[str, Any]:
        """按聚类结果导出review目录，便于人工检查窗口样本。"""
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

            rep_path = str(cluster.get("representative_file", ""))
            representative_files.append(rep_path)
            for member_idx, src in enumerate(cluster.get("sample_files", [])):
                src_path = Path(src)
                if not src_path.exists():
                    missing_files.append(str(src_path))
                    continue
                prefix = "REP__" if str(src_path) == rep_path else "sample__"
                dest_name = f"{prefix}{member_idx:04d}__{src_path.name}"
                shutil.copy2(src_path, cluster_dir / dest_name)
                exported_file_count += 1

        rep_txt = review_root / "representative_files.txt"
        with open(rep_txt, 'w', encoding='utf-8') as f:
            for filepath in representative_files:
                f.write(f"{filepath}\n")

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

        print(f"聚类review目录已导出到: {review_root}")
        return dict(self.cluster_review_info)

    def cleanup(self):
        """保留统一接口，HDBSCAN版本当前无临时相似度文件需要清理。"""
        if self._owned_temp_dir is not None:
            temp_root = Path(self._owned_temp_dir).parent
            shutil.rmtree(self._owned_temp_dir, ignore_errors=True)
            self._owned_temp_dir = None
            try:
                temp_root.rmdir()
            except OSError:
                pass


def main():
    """主函数：命令行接口。"""
    parser = argparse.ArgumentParser(
        description="半导体Layout中心窗口HDBSCAN聚类分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:

1) 基本用法
python layout_clustering_hdbscan.py ./input_data --output results.json

2) 分析单个 OASIS 文件
python layout_clustering_hdbscan.py ./design.oas --output results.json

3) 大版图中心窗口采样 + 聚类
python layout_clustering_hdbscan.py ./large_design.oas --split-layout --clip-size 1.35 --context-width 0.675 --output results.json

4) 调整聚类粒度
python layout_clustering_hdbscan.py ./large_design.oas --split-layout --min-cluster-size 8 --min-samples 4 --output results.json

5) 调整窗口去重与候选中心分桶
python layout_clustering_hdbscan.py ./large_design.oas --split-layout --sample-similarity-threshold 0.97 --candidate-bin-size-um 6.0 --hash-precision-nm 5.0 --output results.json

6) 并行特征提取与窗口局部几何上限
python layout_clustering_hdbscan.py ./input_data --feature-workers 4 --max-elements-per-window 512 --output results.json


注意:
- HDBSCAN版本直接对全样本特征做聚类，不构建完整NxN相似度矩阵
- 对于大版图，优先调整 --split-layout / --clip-size / --context-width
- 默认启用了高性能近似几何去重；如果想保留更多局部几何，可减小 --candidate-bin-size-um 或增大 --max-elements-per-window
- 普通帮助仅展示常用参数；旧版高级参数仍可解析，用于复现实验
        """
    )
    parser.add_argument("input", help="输入文件或目录路径")
    parser.add_argument("--output", "-o", default="clustering_results.json",
                        help="输出文件路径 (默认: clustering_results.json)")
    parser.add_argument("--format", "-f", default="json", choices=["json", "txt"], help="输出格式 (默认: json)")
    parser.add_argument("--export-cluster-review-dir", default="./output_clusters",
                        help="按聚类结果导出review目录，将窗口样本复制到各聚类文件夹并标记 representative")

    # 配置参数
    parser.add_argument("--similarity", default="euclidean", choices=["cosine", "euclidean"],
                        help=argparse.SUPPRESS)
    parser.add_argument("--min-cluster-size", type=int, default=8, help="HDBSCAN最小簇大小 (默认: 8)")
    parser.add_argument("--min-samples", type=int, default=4, help="HDBSCAN min_samples (默认: 4)")
    parser.add_argument("--cluster-selection-epsilon", type=float, default=0.0,
                        help=argparse.SUPPRESS)
    parser.add_argument("--cluster-selection-method", default="eom", choices=["eom", "leaf"],
                        help=argparse.SUPPRESS)
    parser.add_argument("--disable-singleton-postmerge", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--singleton-merge-threshold", type=float, default=0.93,
                        help=argparse.SUPPRESS)

    # 中心窗口采样选项
    parser.add_argument("--split-layout", action="store_true", help="对大版图启用中心窗口采样")
    parser.add_argument("--clip-size", type=float, default=1.35, help="中心矩形的边长 (微米, 默认: 1.35)")
    parser.add_argument("--context-width", type=float, default=0.675,
                        help="中心矩形外围上下文宽度 (微米, 默认: 0.675)")
    parser.add_argument("--inner-feature-weight", type=float, default=1.0,
                        help=argparse.SUPPRESS)
    parser.add_argument("--outer-feature-weight", type=float, default=0.35,
                        help=argparse.SUPPRESS)
    parser.add_argument("--inner-block-weights", default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--outer-block-weights", default="base=0.1,spatial=0.2,shape=0.15,layer=0.35,pattern=0.3,radon=0.25",
                        help=argparse.SUPPRESS)
    parser.add_argument("--sample-similarity-threshold", type=float, default=0.96,
                        help="几何去重时判定高度相似窗口的签名相似度阈值 (默认: 0.96)")
    parser.add_argument("--window-signature-bins", type=int, default=20,
                        help=argparse.SUPPRESS)
    parser.add_argument("--disable-coarse-prefilter", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--coarse-prefilter-quant-um", type=float, default=0.02,
                        help=argparse.SUPPRESS)
    parser.add_argument("--candidate-bin-size-um", type=float, default=None,
                        help="候选中心点空间分桶尺寸；默认自动取 max(10um, 外扩窗口边长的2倍)")
    parser.add_argument("--relation-seed-ratio", type=float, default=0.2,
                        help=argparse.SUPPRESS)
    parser.add_argument("--relation-gap-threshold-um", type=float, default=0.08,
                        help=argparse.SUPPRESS)
    parser.add_argument("--max-elements-per-window", type=int, default=256,
                        help="每个窗口保留的最大局部几何元素数 (默认: 256)")
    parser.add_argument("--disable-clip-shifting", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--clip-shift-directions", default="left,right,up,down",
                        help=argparse.SUPPRESS)
    parser.add_argument("--clip-shift-neighbor-limit", type=int, default=128,
                        help=argparse.SUPPRESS)
    parser.add_argument("--clip-shift-boundary-tol-um", type=float, default=0.02,
                        help=argparse.SUPPRESS)

    # 新增参数
    # 层操作选项
    parser.add_argument("--apply-layer-ops", action="store_true",
                        help="应用层操作（如2413/0 - 2410/0 -> 2413_subtracted/0）")
    parser.add_argument("--register-op", nargs=4, metavar=('SOURCE_LAYER', 'TARGET_LAYER', 'OPERATION', 'RESULT_LAYER'),
                        help="注册新的层操作规则: SOURCE_LAYER TARGET_LAYER OPERATION RESULT_LAYER")

    # 其他选项
    parser.add_argument("--pattern", default="*.oas", help="文件匹配模式 (默认: *.oas，仅扫描 OASIS 文件)")
    parser.add_argument("--no-standardize", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-log-transform", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument("--feature-workers", type=int, default=2,
                        help="特征提取的并行工作进程数 (默认: 2)")
    parser.add_argument("--feature-parallel-threshold", type=int, default=64,
                        help=argparse.SUPPRESS)
    parser.add_argument("--disable-geometry-dedup", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--hash-precision-nm", type=float, default=5.0,
                        help="增强顶点哈希量化精度（纳米，仅影响精确哈希，默认: 5.0）")
    args = parser.parse_args()

    # 创建配置字典
    config = {
        "similarity_method": args.similarity,
        "standardize": not args.no_standardize,
        "apply_layer_operations": args.apply_layer_ops,
        "log_transform": not args.no_log_transform,
        "clip_size_um": args.clip_size,
        "context_width_um": args.context_width,
        "inner_feature_weight": args.inner_feature_weight,
        "outer_feature_weight": args.outer_feature_weight,
        "inner_block_weights": args.inner_block_weights,
        "outer_block_weights": args.outer_block_weights,
        "sample_similarity_threshold": args.sample_similarity_threshold,
        "window_signature_bins": args.window_signature_bins,
        "enable_coarse_prefilter": (not args.disable_coarse_prefilter),
        "coarse_prefilter_quant_um": args.coarse_prefilter_quant_um,
        "candidate_bin_size_um": args.candidate_bin_size_um,
        "relation_seed_ratio": args.relation_seed_ratio,
        "relation_gap_threshold_um": args.relation_gap_threshold_um,
        "max_elements_per_window": args.max_elements_per_window,
        "enable_clip_shifting": (not args.disable_clip_shifting),
        "clip_shift_directions": args.clip_shift_directions,
        "clip_shift_neighbor_limit": args.clip_shift_neighbor_limit,
        "clip_shift_boundary_tolerance_um": args.clip_shift_boundary_tol_um,
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
        "cluster_selection_epsilon": args.cluster_selection_epsilon,
        "cluster_selection_method": args.cluster_selection_method,
        "enable_singleton_postmerge": (not args.disable_singleton_postmerge),
        "singleton_merge_threshold": args.singleton_merge_threshold,
        "feature_workers": args.feature_workers,
        "feature_parallel_threshold": args.feature_parallel_threshold,
        "enable_geometry_dedup": (not args.disable_geometry_dedup),
        "hash_precision_nm": args.hash_precision_nm,
    }

    # 检查依赖
    print("=" * 60)
    print("半导体Layout中心窗口聚类分析工具")
    print("=" * 60)
    print()

    pipeline = None
    try:
        # 创建并运行聚类管道
        pipeline = LayoutClusteringPipeline(config)

        # 如果提供了自定义层操作，注册到处理器
        if args.register_op:
            source_layer, target_layer, operation, result_layer = args.register_op
            # 解析字符串为元组
            source_key = tuple(map(int, source_layer.split('/')))
            target_key = tuple(map(int, target_layer.split('/')))
            result_key = tuple(map(int, result_layer.split('/')))
            pipeline.feature_extractor.layer_processor.register_operation_rule(
                source_key, operation, target_key, result_key
            )
            print(f"已注册层操作: {source_layer} {operation} {target_layer} -> {result_layer}")

        # 运行完整管道
        result = pipeline.run_pipeline(args.input, split_large_layout=args.split_layout)

        if "error" in result:
            print(f"错误: {result['error']}")
            return 1

        # 保存结果
        if args.export_cluster_review_dir:
            pipeline.export_cluster_review(args.export_cluster_review_dir)
        pipeline.save_results(args.output, args.format)

        print("\n" + "=" * 60)
        print("聚类分析完成!")
        print(f" 输入: {args.input}")
        print(f" 发现: {len(pipeline.clusters)} 个聚类")
        print(f" 输出: {args.output}")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if pipeline is not None:
            pipeline.cleanup()


if __name__ == "__main__":
    sys.exit(main())

