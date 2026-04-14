#!/usr/bin/env python3
"""
Liu 2025 marker-driven layout pattern clustering.
输入：
- OASIS 版图文件或目录
主线：
1. 从 design layer 与 marker layer 生成 marker-centered pattern items。
2. 基于多阶段 pruning、dual-backend alignment 和 sparse similarity graph 构建 coarse graph。
3. 使用 surprisal lazy greedy SCP 和有限轮次 closed-loop refinement 形成最终聚类。
说明：
- 只保留论文主线，不再包含旧框架的候选中心、特征导出、review/export 后处理链。
- 默认使用 bbox proxy 裁剪 design 几何来近似生成局部 pattern。
"""
import itertools
import os
import math
import heapq
import hashlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import numpy as np
import json
from dataclasses import dataclass, field, replace
from rtree import index
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import argparse
import gdstk
from skimage.draw import polygon as sk_polygon


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


def _ensure_oas_input_path(filepath: str) -> None:
    if os.path.splitext(filepath)[1].lower() != ".oas":
        raise ValueError(f"当前版本仅支持 OASIS (.oas) 输入: {filepath}")


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
    """单个 marker-centered pattern window 的基础元数据。"""
    sample_id: str
    source_name: str
    center: Tuple[float, float]
    outer_bbox: Tuple[float, float, float, float]
    duplicate_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_name": self.source_name,
            "center": [float(self.center[0]), float(self.center[1])],
            "outer_bbox": [float(v) for v in self.outer_bbox],
            "duplicate_count": int(self.duplicate_count),
        }


@dataclass
class PatternItem:
    """论文主线中的统一 pattern item。"""
    item_id: int
    source_path: str
    source_name: str
    sample_info: Dict[str, Any]
    outer_bbox: Tuple[float, float, float, float]
    outer_polygons: List[Any]
    graph_search_key: Tuple[int, ...]
    graph_invariants: np.ndarray
    graph_signature_grid: np.ndarray
    graph_signature_proj_x: np.ndarray
    graph_signature_proj_y: np.ndarray
    duplicate_count: int = 1
    marker_bbox: Optional[Tuple[float, float, float, float]] = None
    marker_source: str = "none"
    alignment_x_intervals: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float64))
    alignment_y_intervals: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float64))
    topology_features: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    raster_bitmap_cache: Dict[int, np.ndarray] = field(default_factory=dict)
    fft_frequency_cache: Dict[int, np.ndarray] = field(default_factory=dict)


@dataclass
class SimilarityEdge:
    source_idx: int
    target_idx: int
    shift: Tuple[float, float]
    shift_norm_um: float
    relaxed_xor_ratio: float
    signature_similarity: float
    alignment_backend: str = "fast_minmax"

    def reversed(self) -> "SimilarityEdge":
        return SimilarityEdge(
            source_idx=int(self.target_idx),
            target_idx=int(self.source_idx),
            shift=(-float(self.shift[0]), -float(self.shift[1])),
            shift_norm_um=float(self.shift_norm_um),
            relaxed_xor_ratio=float(self.relaxed_xor_ratio),
            signature_similarity=float(self.signature_similarity),
            alignment_backend=str(self.alignment_backend),
        )


@dataclass
class RoundConfig:
    round_index: int
    graph_signature_floor: float
    graph_area_match_ratio: float
    strict_signature_floor: float
    strict_area_match_ratio: float
    graph_invariant_score_limit: float
    alignment_backend: str
    graph_constraint_mode: str
    strict_constraint_mode: str
    strict_edge_threshold_um: float
    graph_topology_threshold: float
    strict_topology_threshold: float
    graph_search_neighbor_radius: int
    graph_max_shift_ratio: float
    graph_shift_norm_ratio: float
    graph_overlap_ratio: float
    graph_marker_overlap_ratio: float
    strict_max_shift_ratio: float
    strict_shift_norm_ratio: float
    strict_overlap_ratio: float
    strict_marker_overlap_ratio: float
    fft_grid_size: int


@dataclass
class AlignmentResult:
    member_idx: int
    rep_idx: int
    accepted: bool
    shift: Tuple[float, float]
    shift_norm_um: float
    shifted_xor_ratio: float
    shifted_signature_similarity: float
    alignment_backend: str
    constraint_mode: str
    max_edge_displacement_um: float
    mean_edge_displacement_um: float


@dataclass
class CoarseCluster:
    rep_idx: int
    member_indices: List[int]
    score: float


@dataclass
class FinalCluster:
    rep_id: int
    member_indices: List[int]
    alignment_results: List[AlignmentResult] = field(default_factory=list)


def _safe_bbox_tuple(bbox):
    if bbox is None:
        return None
    try:
        if len(bbox) == 4 and not isinstance(bbox[0], (tuple, list, np.ndarray)):
            coords = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        else:
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


def _bbox_size(bbox):
    return (
        max(0.0, float(bbox[2] - bbox[0])),
        max(0.0, float(bbox[3] - bbox[1])),
    )


def _point_in_bbox(point_xy, bbox):
    if bbox is None:
        return True
    px, py = float(point_xy[0]), float(point_xy[1])
    return (
        float(bbox[0]) - 1e-9 <= px <= float(bbox[2]) + 1e-9
        and float(bbox[1]) - 1e-9 <= py <= float(bbox[3]) + 1e-9
    )


def _expand_bbox(bbox, margin_um):
    if bbox is None:
        return None
    margin = max(0.0, float(margin_um))
    return (
        float(bbox[0]) - margin,
        float(bbox[1]) - margin,
        float(bbox[2]) + margin,
        float(bbox[3]) + margin,
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


def _collect_expanded_layer_geometries(lib, layer_specs: List[Tuple[int, int]]):
    """按 top cell 展开 hierarchy/repetition 后收集目标 layer 几何。"""
    requested_layers = {
        (int(layer), int(datatype)): []
        for layer, datatype in layer_specs
        if layer is not None and datatype is not None
    }
    top_cells = list(lib.top_level()) or list(lib.cells)
    top_cell_names = [str(getattr(cell, "name", "")) for cell in top_cells]
    for top_cell in top_cells:
        polygons = top_cell.get_polygons(apply_repetitions=True, include_paths=True, depth=None)
        for polygon in polygons:
            layer_key = _element_layer_datatype(polygon)
            if layer_key not in requested_layers:
                continue
            bbox = _geometry_bbox(polygon)
            if bbox is None:
                continue
            requested_layers[layer_key].append({
                "element": polygon,
                "type": "polygon",
                "bbox": tuple(float(v) for v in bbox),
                "cell_name": str(getattr(top_cell, "name", "")),
            })
    return requested_layers, top_cell_names


def _build_layout_spatial_index(expanded_geometries: List[Dict[str, Any]]):
    """为展开后的 design 几何建立 R 树索引。"""
    spatial_index = index.Index()
    indexed_elements = []
    for geometry in expanded_geometries:
        bbox = _safe_bbox_tuple(geometry.get("bbox"))
        if bbox is None:
            continue
        spatial_index.insert(len(indexed_elements), bbox)
        indexed_elements.append({
            "element": geometry["element"],
            "type": str(geometry.get("type", "polygon")),
            "bbox": bbox,
            "cell_name": str(geometry.get("cell_name", "")),
        })
    if not indexed_elements:
        return None, [], None
    return spatial_index, indexed_elements, _safe_bbox_tuple(spatial_index.bounds)


def _parse_layer_datatype_spec(spec: Optional[str]) -> Optional[Tuple[int, int]]:
    if spec is None:
        return None
    text = str(spec).strip()
    if not text:
        return None
    if "/" not in text:
        raise ValueError(f"层规格必须为 LAYER/DATATYPE 格式: {spec}")
    layer_text, datatype_text = text.split("/", 1)
    return int(layer_text), int(datatype_text)


def _collect_layer_bboxes(expanded_geometries: List[Dict[str, Any]]) -> List[Tuple[float, float, float, float]]:
    return [
        tuple(float(v) for v in geometry["bbox"])
        for geometry in expanded_geometries
        if _safe_bbox_tuple(geometry.get("bbox")) is not None
    ]


def _bboxes_touch_or_overlap(a, b, tolerance: float = 1e-9) -> bool:
    tol = max(0.0, float(tolerance))
    return not (
        float(a[2]) < float(b[0]) - tol
        or float(b[2]) < float(a[0]) - tol
        or float(a[3]) < float(b[1]) - tol
        or float(b[3]) < float(a[1]) - tol
    )


def _merge_touching_marker_bboxes(bboxes: List[Tuple[float, float, float, float]], *,
                                  tolerance: float = 1e-9) -> List[Tuple[float, float, float, float]]:
    if not bboxes:
        return []
    parents = list(range(len(bboxes)))

    def find(x: int) -> int:
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    def union(a_idx: int, b_idx: int) -> None:
        ra = find(a_idx)
        rb = find(b_idx)
        if ra != rb:
            parents[rb] = ra
    bbox_index = index.Index()
    expanded_bboxes = []
    tol = max(0.0, float(tolerance))
    for bbox_id, bbox in enumerate(bboxes):
        expanded = (
            float(bbox[0]) - tol,
            float(bbox[1]) - tol,
            float(bbox[2]) + tol,
            float(bbox[3]) + tol,
        )
        expanded_bboxes.append(expanded)
        bbox_index.insert(int(bbox_id), expanded)
    for bbox_id, bbox in enumerate(bboxes):
        for neighbor_id in bbox_index.intersection(expanded_bboxes[bbox_id]):
            neighbor_id = int(neighbor_id)
            if neighbor_id <= bbox_id:
                continue
            if _bboxes_touch_or_overlap(bbox, bboxes[neighbor_id], tolerance=tol):
                union(int(bbox_id), neighbor_id)
    groups: Dict[int, List[Tuple[float, float, float, float]]] = {}
    for bbox_id, bbox in enumerate(bboxes):
        groups.setdefault(find(int(bbox_id)), []).append(tuple(float(v) for v in bbox))
    merged = []
    for group in groups.values():
        merged.append((
            min(float(b[0]) for b in group),
            min(float(b[1]) for b in group),
            max(float(b[2]) for b in group),
            max(float(b[3]) for b in group),
        ))
    merged.sort(key=lambda bbox: (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])))
    return merged


def _marker_source_for_layer(layer_spec: Optional[Tuple[int, int]]) -> str:
    if layer_spec is None:
        return "none"
    return f"layer:{int(layer_spec[0])}/{int(layer_spec[1])}"


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


def _bbox_area(bbox):
    if bbox is None:
        return 0.0
    width, height = _bbox_size(bbox)
    return float(width * height)


def _shift_bbox(bbox, dx, dy):
    return (
        float(bbox[0]) + float(dx),
        float(bbox[1]) + float(dy),
        float(bbox[2]) + float(dx),
        float(bbox[3]) + float(dy),
    )


def _distribution_mean_p95(values):
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.percentile(arr, 95))


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


def _compute_polygon_adjacency_graph(polygons, distance_threshold_um=0.01):
    """计算多边形邻接图
    Args:
        polygons: gdstk.Polygon对象列表
        distance_threshold_um: 邻接距离阈值（微米）
    Returns:
        adjacency_list: 邻接表，adjacency_list[i] = [j1, j2, ...]
        edge_features: 边特征字典，键为(i,j)元组
    """
    if not polygons:
        return [], {}
    n = len(polygons)
    adjacency_list = [[] for _ in range(n)]
    edge_features = {}
    # 提取多边形边界框
    bboxes = []
    for poly in polygons:
        if poly is None or len(poly.points) < 3:
            bboxes.append((0.0, 0.0, 0.0, 0.0))
        else:
            points = np.asarray(poly.points, dtype=np.float64)
            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)
            bboxes.append((x_min, y_min, x_max, y_max))
    # 检测邻接关系
    for i in range(n):
        x1_min, y1_min, x1_max, y1_max = bboxes[i]
        if x1_max - x1_min == 0 and y1_max - y1_min == 0:
            continue  # 空多边形
        for j in range(i + 1, n):
            x2_min, y2_min, x2_max, y2_max = bboxes[j]
            if x2_max - x2_min == 0 and y2_max - y2_min == 0:
                continue  # 空多边形
            # 计算边界框距离
            dx = max(x1_min - x2_max, x2_min - x1_max, 0)
            dy = max(y1_min - y2_max, y2_min - y1_max, 0)
            distance = math.sqrt(dx*dx + dy*dy)
            if distance <= distance_threshold_um:
                adjacency_list[i].append(j)
                adjacency_list[j].append(i)
                # 计算边特征
                edge_key = (i, j) if i < j else (j, i)
                edge_features[edge_key] = {
                    "distance": distance,
                    "bbox_overlap_x": max(0, min(x1_max, x2_max) - max(x1_min, x2_min)),
                    "bbox_overlap_y": max(0, min(y1_max, y2_max) - max(y1_min, y2_min)),
                }
    return adjacency_list, edge_features


def _compute_topology_features(polygons, distance_threshold_um=0.01):
    """计算拓扑特征
    Args:
        polygons: gdstk.Polygon对象列表
        distance_threshold_um: 邻接距离阈值
    Returns:
        topology_features: 拓扑特征向量
    """
    if not polygons:
        return np.zeros(8, dtype=np.float32)
    adjacency_list, edge_features = _compute_polygon_adjacency_graph(
        polygons, distance_threshold_um
    )
    n = len(polygons)
    features = []
    # 1. 连通分量数量
    visited = [False] * n
    component_sizes = []
    for i in range(n):
        if not visited[i] and adjacency_list[i]:
            stack = [i]
            component_size = 0
            while stack:
                node = stack.pop()
                if not visited[node]:
                    visited[node] = True
                    component_size += 1
                    for neighbor in adjacency_list[node]:
                        if not visited[neighbor]:
                            stack.append(neighbor)
            if component_size > 0:
                component_sizes.append(component_size)
    features.append(len(component_sizes))  # 连通分量数
    if component_sizes:
        features.append(max(component_sizes))  # 最大连通分量大小
        features.append(np.mean(component_sizes))  # 平均连通分量大小
        features.append(np.std(component_sizes))  # 连通分量大小标准差
    else:
        features.extend([0.0, 0.0, 0.0])
    # 2. 度分布统计
    degrees = [len(adj) for adj in adjacency_list]
    if degrees:
        features.append(max(degrees))  # 最大度
        features.append(np.mean(degrees))  # 平均度
        features.append(np.std(degrees))  # 度标准差
    else:
        features.extend([0.0, 0.0, 0.0])
    # 3. 边特征统计
    edge_distances = [feat["distance"] for feat in edge_features.values()]
    if edge_distances:
        features.append(np.mean(edge_distances))  # 平均边距离
    else:
        features.append(0.0)
    return np.array(features, dtype=np.float32)


def _topology_distance(features_a, features_b):
    """计算两个拓扑特征向量之间的欧氏距离
    Args:
        features_a: 拓扑特征向量A
        features_b: 拓扑特征向量B
    Returns:
        distance: 欧氏距离
    """
    if features_a is None or features_b is None or len(features_a) == 0 or len(features_b) == 0:
        return float('inf')
    # 确保特征向量长度相同
    min_len = min(len(features_a), len(features_b))
    dist = np.linalg.norm(features_a[:min_len] - features_b[:min_len])
    return float(dist)


def _window_geometry_stats(polygons, outer_bbox):
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
    outer_bbox = tuple(float(v) for v in outer_bbox)
    outer_w, outer_h = _bbox_size(outer_bbox)
    outer_window_area = max(float(outer_w * outer_h), 1e-12)
    outer_window_span = max(float(outer_w), float(outer_h), 1e-12)
    if not point_sets:
        return {
            "polygon_count": 0.0,
            "total_area": 0.0,
            "total_perimeter": 0.0,
            "vertex_count": 0.0,
            "bbox_long": 0.0,
            "bbox_short": 0.0,
            "radius_mean": 0.0,
            "radius_std": 0.0,
            "density": 0.0,
            "extent_density": 0.0,
            "fill_ratio": 0.0,
            "aspect_ratio": 0.0,
            "bbox_long_norm": 0.0,
            "bbox_short_norm": 0.0,
            "radius_mean_norm": 0.0,
            "radius_std_norm": 0.0,
            "log_component_count": 0.0,
            "outer_window_area": float(outer_window_area),
            "outer_window_span": float(outer_window_span),
        }
    all_points = np.vstack(point_sets)
    min_coord = np.min(all_points, axis=0)
    max_coord = np.max(all_points, axis=0)
    width = float(max_coord[0] - min_coord[0])
    height = float(max_coord[1] - min_coord[1])
    bbox_long = max(width, height)
    bbox_short = min(width, height)
    centroid = np.mean(all_points, axis=0)
    radii = np.linalg.norm(all_points - centroid, axis=1)
    radius_mean = float(np.mean(radii)) if len(radii) > 0 else 0.0
    radius_std = float(np.std(radii)) if len(radii) > 0 else 0.0
    density = float(total_area) / max(width * height, 1e-12)
    extent_density = float(total_area) / max(bbox_long * bbox_short, 1e-12)
    fill_ratio = float(total_area) / float(outer_window_area)
    aspect_ratio = float(bbox_short) / max(float(bbox_long), 1e-12)
    return {
        "polygon_count": float(len(point_sets)),
        "total_area": float(total_area),
        "total_perimeter": float(total_perimeter),
        "vertex_count": float(vertex_count),
        "bbox_long": float(bbox_long),
        "bbox_short": float(bbox_short),
        "radius_mean": float(radius_mean),
        "radius_std": float(radius_std),
        "density": float(density),
        "extent_density": float(extent_density),
        "fill_ratio": float(fill_ratio),
        "aspect_ratio": float(aspect_ratio),
        "bbox_long_norm": float(bbox_long) / float(outer_window_span),
        "bbox_short_norm": float(bbox_short) / float(outer_window_span),
        "radius_mean_norm": float(radius_mean) / float(outer_window_span),
        "radius_std_norm": float(radius_std) / float(outer_window_span),
        "log_component_count": float(np.log1p(len(point_sets))),
        "outer_window_area": float(outer_window_area),
        "outer_window_span": float(outer_window_span),
    }


def _compute_graph_invariants(polygons, outer_bbox):
    stats = _window_geometry_stats(polygons, outer_bbox)
    return np.asarray([
        float(stats["log_component_count"]),
        float(stats["fill_ratio"]),
        float(stats["bbox_long_norm"]),
        float(stats["bbox_short_norm"]),
        float(stats["aspect_ratio"]),
        float(stats["extent_density"]),
        float(stats["radius_mean_norm"]),
        float(stats["radius_std_norm"]),
    ], dtype=np.float64)


def _coarse_signature_search_key(graph_signature_grid, graph_signature_proj_x, graph_signature_proj_y):
    grid = np.asarray(graph_signature_grid, dtype=np.float32).reshape(10, 10)
    pooled = grid.reshape(5, 2, 5, 2).sum(axis=(1, 3))
    flat = pooled.reshape(-1)
    if flat.size == 0:
        return (0, 0, 0, 0, 0, 0)
    top_order = np.argsort(-flat, kind="stable")[:4]
    top_key = tuple(sorted(int(v) for v in top_order.tolist()))
    px = np.asarray(graph_signature_proj_x, dtype=np.float32).ravel()
    py = np.asarray(graph_signature_proj_y, dtype=np.float32).ravel()
    x_coords = np.arange(px.size, dtype=np.float32)
    y_coords = np.arange(py.size, dtype=np.float32)
    px_sum = max(float(np.sum(px)), 1e-9)
    py_sum = max(float(np.sum(py)), 1e-9)
    centroid_x = int(round(float(np.dot(x_coords, px) / px_sum) / 2.0))
    centroid_y = int(round(float(np.dot(y_coords, py) / py_sum) / 2.0))
    centroid_x = max(0, min(4, centroid_x))
    centroid_y = max(0, min(4, centroid_y))
    return top_key + (int(centroid_x), int(centroid_y))


def _neighboring_graph_search_keys(search_key, radius=1):
    base_key = tuple(int(v) for v in search_key)
    if len(base_key) < 2:
        return [base_key]
    coarse_prefix = base_key[:-2]
    base_x = int(base_key[-2])
    base_y = int(base_key[-1])
    neighbor_radius = max(0, int(radius))
    keys = []
    seen = set()
    for dx, dy in itertools.product(range(-neighbor_radius, neighbor_radius + 1), repeat=2):
        candidate = coarse_prefix + (base_x + int(dx), base_y + int(dy))
        if candidate not in seen:
            seen.add(candidate)
            keys.append(candidate)
    return keys


def _rasterize_polygons_bitmap(polygons, bbox, grid_size):
    min_x, min_y, max_x, max_y = (float(v) for v in bbox)
    grid_n = max(1, int(grid_size))
    bitmap = np.zeros((grid_n, grid_n), dtype=np.float32)
    if max_x <= min_x or max_y <= min_y:
        return bitmap
    span_x = max(max_x - min_x, 1e-12)
    span_y = max(max_y - min_y, 1e-12)
    for poly in polygons or []:
        if poly is None or not hasattr(poly, "points"):
            continue
        pts = np.asarray(poly.points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
            continue
        cols = (pts[:, 0] - min_x) / span_x * float(grid_n - 1)
        rows = (pts[:, 1] - min_y) / span_y * float(grid_n - 1)
        rr, cc = sk_polygon(rows, cols, shape=bitmap.shape)
        if rr.size == 0 or cc.size == 0:
            continue
        bitmap[rr, cc] = 1.0
    return bitmap


def _pool_bitmap(bitmap: np.ndarray, pooled_bins: int) -> np.ndarray:
    src = np.asarray(bitmap, dtype=np.float32)
    bins = max(1, int(pooled_bins))
    h, w = src.shape
    pooled = np.zeros((bins, bins), dtype=np.float32)
    row_edges = np.linspace(0, h, bins + 1, dtype=np.int32)
    col_edges = np.linspace(0, w, bins + 1, dtype=np.int32)
    for ix in range(bins):
        r0 = int(row_edges[ix])
        r1 = int(row_edges[ix + 1])
        if r1 <= r0:
            r1 = min(h, r0 + 1)
        for iy in range(bins):
            c0 = int(col_edges[iy])
            c1 = int(col_edges[iy + 1])
            if c1 <= c0:
                c1 = min(w, c0 + 1)
            cell = src[r0:r1, c0:c1]
            pooled[ix, iy] = float(np.mean(cell)) if cell.size else 0.0
    return pooled


def _window_occupancy_signature(polygons, bbox, n_bins=10, raster_grid_size=40):
    bitmap = _rasterize_polygons_bitmap(polygons, bbox, int(raster_grid_size))
    pooled = _pool_bitmap(bitmap, int(n_bins))
    total = float(np.sum(pooled))
    if total > 0.0:
        pooled = pooled / total
    proj_x = np.sum(pooled, axis=1, dtype=np.float32)
    proj_y = np.sum(pooled, axis=0, dtype=np.float32)
    return (
        pooled.astype(np.float32).reshape(-1),
        np.asarray(proj_x, dtype=np.float32),
        np.asarray(proj_y, dtype=np.float32),
    )


def _pattern_item_raster_bitmap(item: PatternItem, grid_size=64):
    key = int(grid_size)
    bitmap = item.raster_bitmap_cache.get(key)
    if bitmap is not None and getattr(bitmap, "shape", None) == (key, key):
        return bitmap
    bitmap = _rasterize_polygons_bitmap(item.outer_polygons, item.outer_bbox, key)
    item.raster_bitmap_cache[key] = bitmap
    return bitmap


def _pattern_item_fft_frequency(item: PatternItem, grid_size=64):
    key = int(grid_size)
    frequency = item.fft_frequency_cache.get(key)
    if frequency is not None and getattr(frequency, "shape", None) == (key, key):
        return frequency
    bitmap = _pattern_item_raster_bitmap(item, grid_size=key)
    centered = bitmap.astype(np.float32) - float(np.mean(bitmap))
    frequency = np.fft.fft2(centered)
    item.fft_frequency_cache[key] = frequency
    return frequency


def _phase_correlation_shift_from_frequency(freq_a, freq_b):
    fa = np.asarray(freq_a)
    fb = np.asarray(freq_b)
    if fa.size == 0 or fb.size == 0 or fa.shape != fb.shape:
        return None
    if float(np.sum(np.abs(fa))) <= 1e-9 or float(np.sum(np.abs(fb))) <= 1e-9:
        return None
    cross_power = fa * np.conj(fb)
    magnitude = np.abs(cross_power)
    valid = magnitude > 1e-12
    if not np.any(valid):
        return None
    cross_power = np.where(valid, cross_power / np.maximum(magnitude, 1e-12), 0.0)
    corr = np.fft.ifft2(cross_power).real
    peak_index = np.unravel_index(int(np.argmax(corr)), corr.shape)
    peak_value = float(corr[peak_index])
    if not np.isfinite(peak_value) or peak_value <= 0.0:
        return None
    shift_y, shift_x = (int(peak_index[0]), int(peak_index[1]))
    if shift_y > corr.shape[0] // 2:
        shift_y -= corr.shape[0]
    if shift_x > corr.shape[1] // 2:
        shift_x -= corr.shape[1]
    return float(shift_x), float(shift_y), float(peak_value)


def _refine_shift_with_fft_pcm(source_item: PatternItem, target_item: PatternItem, initial_shift, *,
                               grid_size=64):
    freq_a = _pattern_item_fft_frequency(source_item, grid_size=int(grid_size))
    freq_b = _pattern_item_fft_frequency(target_item, grid_size=int(grid_size))
    pcm_result = _phase_correlation_shift_from_frequency(freq_a, freq_b)
    if pcm_result is None:
        return None
    cell_dx, cell_dy, peak_value = pcm_result
    source_w, source_h = _bbox_size(source_item.outer_bbox)
    target_w, target_h = _bbox_size(target_item.outer_bbox)
    span_x = max((float(source_w) + float(target_w)) * 0.5, 1e-6)
    span_y = max((float(source_h) + float(target_h)) * 0.5, 1e-6)
    delta_shift = (
        float(cell_dx) * span_x / float(grid_size),
        float(cell_dy) * span_y / float(grid_size),
    )
    outer_window_span = max(source_w, source_h, target_w, target_h, 1e-6)
    if max(abs(delta_shift[0]), abs(delta_shift[1])) > outer_window_span * 0.20:
        return None
    refined_shift = (
        float(initial_shift[0]) + float(delta_shift[0]),
        float(initial_shift[1]) + float(delta_shift[1]),
    )
    return {
        "shift": refined_shift,
        "delta_shift": delta_shift,
        "peak_value": float(peak_value),
    }


def _graph_invariant_distance(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    floors = np.asarray([0.25, 0.02, 0.02, 0.02, 0.03, 0.03, 0.02, 0.02], dtype=np.float64)
    weights = np.asarray([0.08, 0.24, 0.10, 0.08, 0.18, 0.14, 0.10, 0.08], dtype=np.float64)
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), floors)
    errs = np.minimum(np.abs(a - b) / denom, 1.0)
    critical_cap_exceeded = bool(errs[1] > 0.45 or errs[4] > 0.45 or errs[5] > 0.45)
    score = float(np.dot(errs, weights))
    return score, errs, critical_cap_exceeded


def _graph_signature_similarity(source_item: PatternItem, target_item: PatternItem) -> float:
    grid_sim = _cosine_similarity_1d(source_item.graph_signature_grid, target_item.graph_signature_grid)
    proj_x_sim = _cosine_similarity_1d(source_item.graph_signature_proj_x, target_item.graph_signature_proj_x)
    proj_y_sim = _cosine_similarity_1d(source_item.graph_signature_proj_y, target_item.graph_signature_proj_y)
    return float(0.6 * grid_sim + 0.2 * proj_x_sim + 0.2 * proj_y_sim)


def _cosine_similarity_1d(vec_a, vec_b):
    a = np.asarray(vec_a, dtype=np.float64).ravel()
    b = np.asarray(vec_b, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.dot(a, b) / denom)


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


def _extract_axis_projection_intervals(polygons):
    x_intervals = []
    y_intervals = []
    for poly in polygons or []:
        if poly is None or not hasattr(poly, "bounding_box"):
            continue
        bbox = _safe_bbox_tuple(poly.bounding_box())
        if bbox is None:
            continue
        x_intervals.append((float(bbox[0]), float(bbox[2])))
        y_intervals.append((float(bbox[1]), float(bbox[3])))
    return (
        np.asarray(x_intervals, dtype=np.float64),
        np.asarray(y_intervals, dtype=np.float64),
    )


def _build_window_record(sample, outer_polygons, **extra_meta):
    alignment_x_intervals, alignment_y_intervals = _extract_axis_projection_intervals(outer_polygons)
    local_outer_bbox = extra_meta.pop("local_outer_bbox", None)
    marker_bbox = extra_meta.pop("marker_bbox", None)
    absolute_marker_bbox = extra_meta.pop("absolute_marker_bbox", None)
    graph_invariants = np.asarray(extra_meta.pop("graph_invariants", []), dtype=np.float64)
    topology_features = np.asarray(extra_meta.pop("topology_features", []), dtype=np.float64)
    graph_signature_grid = np.asarray(extra_meta.pop("graph_signature_grid", []), dtype=np.float32)
    graph_signature_proj_x = np.asarray(extra_meta.pop("graph_signature_proj_x", []), dtype=np.float32)
    graph_signature_proj_y = np.asarray(extra_meta.pop("graph_signature_proj_y", []), dtype=np.float32)
    record = {
        "sample": sample,
        "outer_polygons": list(outer_polygons),
        "local_outer_bbox": tuple(float(v) for v in local_outer_bbox) if local_outer_bbox is not None else tuple(float(v) for v in sample.outer_bbox),
        "graph_invariants": graph_invariants,
        "topology_features": topology_features,
        "graph_signature_grid": graph_signature_grid,
        "graph_signature_proj_x": graph_signature_proj_x,
        "graph_signature_proj_y": graph_signature_proj_y,
        "graph_search_key": tuple(
            int(v) for v in extra_meta.pop(
                "graph_search_key",
                (),
            )
        ),
        "alignment_x_intervals": np.asarray(alignment_x_intervals, dtype=np.float64),
        "alignment_y_intervals": np.asarray(alignment_y_intervals, dtype=np.float64),
    }
    if marker_bbox is not None:
        record["marker_bbox"] = tuple(float(v) for v in marker_bbox)
    if absolute_marker_bbox is not None:
        record["absolute_marker_bbox"] = tuple(float(v) for v in absolute_marker_bbox)
    record.update(extra_meta)
    return record


def _pattern_item_center(item: PatternItem) -> Tuple[float, float]:
    return _bbox_center(item.outer_bbox)


def _pattern_item_absolute_center(item: PatternItem) -> Tuple[float, float]:
    center = item.sample_info.get("center")
    if isinstance(center, (list, tuple)) and len(center) == 2:
        return float(center[0]), float(center[1])
    return _bbox_center(item.outer_bbox)


def _pattern_item_window_area(item: PatternItem) -> float:
    bbox = tuple(float(v) for v in item.outer_bbox)
    return max(1e-12, float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))


def _quantize_exact_float(value: float, grid: float = 1e-4) -> int:
    return int(round(float(value) / float(grid)))


def _quantize_exact_bbox(bbox, grid: float = 1e-4) -> Tuple[int, int, int, int]:
    return tuple(_quantize_exact_float(float(v), grid=grid) for v in bbox)


def _exact_item_group_key(item: PatternItem) -> str:
    """稳定 exact key：局部 bbox proxy 列表 + marker bbox + graph signature。"""
    polygon_bboxes = []
    for poly in item.outer_polygons or []:
        bbox = _geometry_bbox(poly)
        if bbox is None:
            continue
        polygon_bboxes.append((
            int(getattr(poly, "layer", 0)),
            int(getattr(poly, "datatype", 0)),
            _quantize_exact_bbox(bbox),
        ))
    polygon_bboxes.sort()
    marker_bbox = item.marker_bbox if item.marker_bbox is not None else (0.0, 0.0, 0.0, 0.0)
    payload = {
        "marker_bbox": _quantize_exact_bbox(marker_bbox),
        "outer_bbox": _quantize_exact_bbox(item.outer_bbox),
        "polygons": polygon_bboxes,
        "graph_signature_grid": np.round(np.asarray(item.graph_signature_grid, dtype=np.float32), 6).tolist(),
        "graph_signature_proj_x": np.round(np.asarray(item.graph_signature_proj_x, dtype=np.float32), 6).tolist(),
        "graph_signature_proj_y": np.round(np.asarray(item.graph_signature_proj_y, dtype=np.float32), 6).tolist(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def _solve_axis_minmax_shift(source_intervals, target_intervals):
    if (
        source_intervals is None
        or target_intervals is None
        or len(source_intervals) == 0
        or len(target_intervals) == 0
    ):
        return 0.0
    best_quality = None
    best_shift = 0.0
    for src_min, src_max in source_intervals:
        for tgt_min, tgt_max in target_intervals:
            dmin = float(tgt_min) - float(src_max)
            dmax = float(tgt_max) - float(src_min)
            quality = float(dmax - dmin)
            shift = float((dmin + dmax) * 0.5)
            if (
                best_quality is None
                or quality < best_quality - 1e-12
                or (abs(quality - best_quality) <= 1e-12 and abs(shift) < abs(best_shift) - 1e-12)
            ):
                best_quality = quality
                best_shift = shift
    return float(best_shift)


def _solve_geometric_minmax_shift(source_item: PatternItem, target_item: PatternItem):
    source_x = source_item.alignment_x_intervals
    source_y = source_item.alignment_y_intervals
    target_x = target_item.alignment_x_intervals
    target_y = target_item.alignment_y_intervals
    dx = _solve_axis_minmax_shift(source_x, target_x)
    dy = _solve_axis_minmax_shift(source_y, target_y)
    return float(dx), float(dy)


def _compute_edge_displacement_metrics(source_item: PatternItem, target_item: PatternItem, shift):

    def _axis_edges(intervals: np.ndarray, delta: float = 0.0) -> np.ndarray:
        if intervals is None or len(intervals) == 0:
            return np.empty(0, dtype=np.float64)
        shifted = np.asarray(intervals, dtype=np.float64) + float(delta)
        return np.sort(shifted.reshape(-1))

    def _axis_residuals(source_edges: np.ndarray, target_edges: np.ndarray, penalty: float) -> np.ndarray:
        pair_count = min(len(source_edges), len(target_edges))
        residuals = []
        if pair_count > 0:
            residuals.extend(np.abs(source_edges[:pair_count] - target_edges[:pair_count]).tolist())
        extra = abs(len(source_edges) - len(target_edges))
        if extra > 0:
            residuals.extend([float(penalty)] * int(extra))
        return np.asarray(residuals, dtype=np.float64)
    source_span = max(_bbox_size(tuple(float(v) for v in source_item.outer_bbox)))
    target_span = max(_bbox_size(tuple(float(v) for v in target_item.outer_bbox)))
    penalty = max(1e-6, float(max(source_span, target_span)))
    residuals = np.concatenate([
        _axis_residuals(_axis_edges(source_item.alignment_x_intervals, float(shift[0])),
                        _axis_edges(target_item.alignment_x_intervals),
                        penalty),
        _axis_residuals(_axis_edges(source_item.alignment_y_intervals, float(shift[1])),
                        _axis_edges(target_item.alignment_y_intervals),
                        penalty),
    ])
    if residuals.size == 0:
        return 0.0, 0.0
    return float(np.max(residuals)), float(np.mean(residuals))


def _translate_polygons(polygons, shift):
    dx, dy = float(shift[0]), float(shift[1])
    translated = []
    offset = np.array([dx, dy], dtype=np.float64)
    for poly in polygons or []:
        if poly is None or not hasattr(poly, "points"):
            continue
        pts = np.asarray(poly.points, dtype=np.float64)
        if pts.size == 0:
            continue
        translated.append(
            gdstk.Polygon(
                pts + offset,
                layer=int(getattr(poly, "layer", 0)),
                datatype=int(getattr(poly, "datatype", 0)),
            )
        )
    return translated


def _compute_aligned_pair_metrics(source_item: PatternItem, target_item: PatternItem, shift):
    shifted_polygons = _translate_polygons(source_item.outer_polygons, shift)
    target_outer_bbox = tuple(float(v) for v in target_item.outer_bbox)
    shifted_polygons = _approx_clip_polygons_with_bbox(
        shifted_polygons,
        target_outer_bbox,
        "and",
    )
    if not shifted_polygons:
        return {
            "signature_similarity": 0.0,
            "xor_ratio": 1.0,
            "shift_norm_um": float(math.hypot(float(shift[0]), float(shift[1]))),
        }
    signature_similarity = _graph_signature_similarity(source_item, target_item)
    xor_polygons = gdstk.boolean(
        shifted_polygons,
        target_item.outer_polygons,
        "xor",
        precision=1e-6,
    )
    xor_ratio = _polygon_list_area(xor_polygons) / max(
        1e-12,
        float(max(_pattern_item_window_area(source_item), _pattern_item_window_area(target_item))),
    )
    return {
        "signature_similarity": float(signature_similarity),
        "xor_ratio": float(xor_ratio),
        "shift_norm_um": float(math.hypot(float(shift[0]), float(shift[1]))),
    }


def _evaluate_alignment_locality_constraints(source_item: PatternItem, target_item: PatternItem, shift, *,
                                             max_shift_ratio=0.75,
                                             shift_norm_ratio=0.60,
                                             min_shifted_bbox_overlap_ratio=0.10,
                                             min_source_marker_overlap_ratio=0.0):
    source_center = _pattern_item_center(source_item)
    target_center = _pattern_item_center(target_item)
    source_bbox = tuple(float(v) for v in source_item.outer_bbox)
    target_bbox = tuple(float(v) for v in target_item.outer_bbox)
    source_w, source_h = _bbox_size(source_bbox)
    target_w, target_h = _bbox_size(target_bbox)
    outer_window_size = max(source_w, source_h, target_w, target_h, 1e-6)
    raw_center_distance_um = float(math.hypot(source_center[0] - target_center[0], source_center[1] - target_center[1]))
    shift_dx = float(shift[0])
    shift_dy = float(shift[1])
    shift_norm = math.hypot(shift_dx, shift_dy)
    shifted_center = (float(source_center[0] + shift_dx), float(source_center[1] + shift_dy))
    shifted_center_distance_um = float(math.hypot(shifted_center[0] - target_center[0], shifted_center[1] - target_center[1]))
    max_shift_cap = outer_window_size * float(max_shift_ratio)
    max_shift_cap_exceeded = max(abs(shift_dx), abs(shift_dy)) > max_shift_cap
    shift_norm_exceeded = shift_norm > outer_window_size * float(shift_norm_ratio)
    if max_shift_cap_exceeded:
        return {
            "ok": False,
            "reject_reason": "rejected_by_max_shift_cap",
            "raw_center_distance_um": raw_center_distance_um,
            "shifted_center_distance_um": shifted_center_distance_um,
            "shifted_bbox_overlap_ratio": 0.0,
            "target_marker_miss": False,
            "source_marker_miss": False,
            "max_shift_cap_exceeded": True,
            "shift_norm_exceeded": bool(shift_norm_exceeded),
        }
    if shift_norm_exceeded:
        return {
            "ok": False,
            "reject_reason": "rejected_by_shift_norm",
            "raw_center_distance_um": raw_center_distance_um,
            "shifted_center_distance_um": shifted_center_distance_um,
            "shifted_bbox_overlap_ratio": 0.0,
            "target_marker_miss": False,
            "source_marker_miss": False,
            "max_shift_cap_exceeded": bool(max_shift_cap_exceeded),
            "shift_norm_exceeded": True,
        }
    source_marker_bbox = source_item.marker_bbox
    source_marker_miss = False
    source_marker_overlap_ratio = 1.0
    if source_marker_bbox is not None:
        expanded_source_marker_bbox = _expand_bbox(
            source_marker_bbox,
            max(0.02, outer_window_size * 0.05),
        )
        shifted_source_marker_bbox = _shift_bbox(source_marker_bbox, shift_dx, shift_dy)
        source_marker_overlap_bbox = _bbox_intersection(shifted_source_marker_bbox, expanded_source_marker_bbox)
        source_marker_overlap_area = _bbox_area(source_marker_overlap_bbox)
        source_marker_area = max(1e-12, _bbox_area(source_marker_bbox))
        source_marker_overlap_ratio = float(source_marker_overlap_area / source_marker_area)
        source_marker_miss = source_marker_overlap_ratio < float(min_source_marker_overlap_ratio)
        if source_marker_miss:
            return {
                "ok": False,
                "reject_reason": "rejected_by_source_marker",
                "raw_center_distance_um": raw_center_distance_um,
                "shifted_center_distance_um": shifted_center_distance_um,
                "shifted_bbox_overlap_ratio": 0.0,
                "source_marker_overlap_ratio": float(source_marker_overlap_ratio),
                "target_marker_miss": False,
                "source_marker_miss": True,
                "max_shift_cap_exceeded": bool(max_shift_cap_exceeded),
                "shift_norm_exceeded": bool(shift_norm_exceeded),
            }
    target_marker_bbox = target_item.marker_bbox
    target_marker_miss = False
    if target_marker_bbox is not None:
        target_marker_miss = not _point_in_bbox(
            shifted_center,
            _expand_bbox(target_marker_bbox, max(0.02, outer_window_size * 0.05)),
        )
    shifted_source_bbox = _shift_bbox(source_bbox, shift_dx, shift_dy)
    overlap_bbox = _bbox_intersection(shifted_source_bbox, target_bbox)
    overlap_area = _bbox_area(overlap_bbox)
    overlap_denom = max(1e-12, min(_bbox_area(source_bbox), _bbox_area(target_bbox)))
    shifted_bbox_overlap_ratio = float(overlap_area / overlap_denom)
    if shifted_bbox_overlap_ratio < float(min_shifted_bbox_overlap_ratio):
        return {
            "ok": False,
            "reject_reason": "rejected_by_shifted_bbox_overlap",
            "raw_center_distance_um": raw_center_distance_um,
            "shifted_center_distance_um": shifted_center_distance_um,
            "shifted_bbox_overlap_ratio": shifted_bbox_overlap_ratio,
            "source_marker_overlap_ratio": float(source_marker_overlap_ratio),
            "target_marker_miss": bool(target_marker_miss),
            "source_marker_miss": bool(source_marker_miss),
            "max_shift_cap_exceeded": bool(max_shift_cap_exceeded),
            "shift_norm_exceeded": bool(shift_norm_exceeded),
        }
    return {
        "ok": True,
        "reject_reason": None,
        "raw_center_distance_um": raw_center_distance_um,
        "shifted_center_distance_um": shifted_center_distance_um,
        "shifted_bbox_overlap_ratio": shifted_bbox_overlap_ratio,
        "source_marker_overlap_ratio": float(source_marker_overlap_ratio),
        "target_marker_miss": bool(target_marker_miss),
        "source_marker_miss": bool(source_marker_miss),
        "max_shift_cap_exceeded": bool(max_shift_cap_exceeded),
        "shift_norm_exceeded": bool(shift_norm_exceeded),
    }


def _new_rejection_counters(*keys):
    return {str(key): 0 for key in keys}


def _new_fft_pcm_counters(*keys):
    return {str(key): 0 for key in keys}


def _should_try_fft_pcm_for_graph(locality_eval, metrics, area_match_ratio):
    area_limit = max(0.0, 1.0 - float(area_match_ratio))
    if not bool(locality_eval.get("ok", False)):
        reject_reason = str(locality_eval.get("reject_reason"))
        return reject_reason in {
            "rejected_by_max_shift_cap",
            "rejected_by_shift_norm",
            "rejected_by_source_marker",
            "rejected_by_shifted_bbox_overlap",
        }
    if metrics is None:
        return False
    xor_ratio = float(metrics.get("xor_ratio", 1.0))
    return area_limit < xor_ratio <= area_limit + 0.08


def _should_try_fft_pcm_for_strict(locality_eval, metrics, area_match_ratio, edge_exceeded):
    area_limit = max(0.0, 1.0 - float(area_match_ratio))
    if not bool(locality_eval.get("ok", False)):
        reject_reason = str(locality_eval.get("reject_reason"))
        return reject_reason in {
            "rejected_by_max_shift_cap",
            "rejected_by_shift_norm",
            "rejected_by_source_marker",
            "rejected_by_shifted_bbox_overlap",
        }
    if edge_exceeded:
        return True
    if metrics is None:
        return False
    xor_ratio = float(metrics.get("xor_ratio", 1.0))
    return area_limit < xor_ratio <= area_limit + 0.05


def _build_graph_track_edge_fast_minmax_area_cosine(source_idx, target_idx, source_item, target_item, *,
                                                    signature_floor, area_match_ratio,
                                                    graph_invariant_score_limit,
                                                    round_config: RoundConfig):
    track_meta = {
        "fft_pcm_attempt_count": 0,
        "fft_pcm_accept_count": 0,
        "fft_pcm_improved_locality_count": 0,
        "fft_pcm_improved_area_count": 0,
        "fft_pcm_shift_delta_sum_um": 0.0,
        "target_marker_miss": False,
        "source_marker_miss": False,
        "source_marker_overlap_ratio": 1.0,
        "raw_center_distance_um": 0.0,
        "shifted_center_distance_um": 0.0,
        "shifted_bbox_overlap_ratio": 0.0,
    }
    invariant_score, _, critical_cap_exceeded = _graph_invariant_distance(
        source_item.graph_invariants,
        target_item.graph_invariants,
    )
    if critical_cap_exceeded or invariant_score > float(graph_invariant_score_limit):
        return None, "rejected_by_invariant", track_meta
    topology_dist = _topology_distance(
        source_item.topology_features,
        target_item.topology_features,
    )
    if topology_dist > float(round_config.graph_topology_threshold):
        return None, "rejected_by_topology", track_meta
    signature_similarity = _graph_signature_similarity(source_item, target_item)
    if signature_similarity < float(signature_floor):
        return None, "rejected_by_signature", track_meta
    shift = _solve_geometric_minmax_shift(source_item, target_item)
    locality_eval = _evaluate_alignment_locality_constraints(
        source_item,
        target_item,
        shift,
        max_shift_ratio=float(round_config.graph_max_shift_ratio),
        shift_norm_ratio=float(round_config.graph_shift_norm_ratio),
        min_shifted_bbox_overlap_ratio=float(round_config.graph_overlap_ratio),
        min_source_marker_overlap_ratio=float(round_config.graph_marker_overlap_ratio),
    )
    track_meta.update({
        "target_marker_miss": bool(locality_eval.get("target_marker_miss", False)),
        "source_marker_miss": bool(locality_eval.get("source_marker_miss", False)),
        "source_marker_overlap_ratio": float(locality_eval.get("source_marker_overlap_ratio", track_meta["source_marker_overlap_ratio"])),
        "raw_center_distance_um": float(locality_eval.get("raw_center_distance_um", 0.0)),
        "shifted_center_distance_um": float(locality_eval.get("shifted_center_distance_um", 0.0)),
        "shifted_bbox_overlap_ratio": float(locality_eval.get("shifted_bbox_overlap_ratio", 0.0)),
    })
    metrics = None
    area_limit = max(0.0, 1.0 - float(area_match_ratio))
    if bool(locality_eval.get("ok", False)):
        metrics = _compute_aligned_pair_metrics(source_item, target_item, shift)
        if metrics["xor_ratio"] <= area_limit:
            return SimilarityEdge(
                source_idx=int(source_idx),
                target_idx=int(target_idx),
                shift=(float(shift[0]), float(shift[1])),
                shift_norm_um=float(metrics["shift_norm_um"]),
                relaxed_xor_ratio=float(metrics["xor_ratio"]),
                signature_similarity=float(metrics["signature_similarity"]),
                alignment_backend="fast_minmax",
            ), None, track_meta
    if _should_try_fft_pcm_for_graph(locality_eval, metrics, area_match_ratio):
        track_meta["fft_pcm_attempt_count"] += 1
        refined = _refine_shift_with_fft_pcm(
            source_item,
            target_item,
            shift,
            grid_size=int(round_config.fft_grid_size),
        )
        if refined is not None:
            refined_shift = tuple(float(v) for v in refined["shift"])
            refined_locality_eval = _evaluate_alignment_locality_constraints(
                source_item,
                target_item,
                refined_shift,
                max_shift_ratio=float(round_config.graph_max_shift_ratio),
                shift_norm_ratio=float(round_config.graph_shift_norm_ratio),
                min_shifted_bbox_overlap_ratio=float(round_config.graph_overlap_ratio),
                min_source_marker_overlap_ratio=float(round_config.graph_marker_overlap_ratio),
            )
            refined_metrics = None
            if bool(refined_locality_eval.get("ok", False)):
                refined_metrics = _compute_aligned_pair_metrics(source_item, target_item, refined_shift)
            locality_improved = (not bool(locality_eval.get("ok", False))) and bool(refined_locality_eval.get("ok", False))
            area_improved = (
                metrics is not None
                and refined_metrics is not None
                and float(refined_metrics["xor_ratio"]) + 1e-9 < float(metrics["xor_ratio"])
            )
            if bool(refined_locality_eval.get("ok", False)) and refined_metrics is not None and refined_metrics["xor_ratio"] <= area_limit and (locality_improved or area_improved or metrics is None):
                track_meta["fft_pcm_accept_count"] += 1
                track_meta["fft_pcm_shift_delta_sum_um"] += float(math.hypot(*refined["delta_shift"]))
                if locality_improved:
                    track_meta["fft_pcm_improved_locality_count"] += 1
                if area_improved:
                    track_meta["fft_pcm_improved_area_count"] += 1
                track_meta.update({
                    "target_marker_miss": bool(refined_locality_eval.get("target_marker_miss", False)),
                    "source_marker_miss": bool(refined_locality_eval.get("source_marker_miss", False)),
                    "source_marker_overlap_ratio": float(refined_locality_eval.get("source_marker_overlap_ratio", track_meta["source_marker_overlap_ratio"])),
                    "raw_center_distance_um": float(refined_locality_eval.get("raw_center_distance_um", 0.0)),
                    "shifted_center_distance_um": float(refined_locality_eval.get("shifted_center_distance_um", 0.0)),
                    "shifted_bbox_overlap_ratio": float(refined_locality_eval.get("shifted_bbox_overlap_ratio", 0.0)),
                })
                return SimilarityEdge(
                    source_idx=int(source_idx),
                    target_idx=int(target_idx),
                    shift=(float(refined_shift[0]), float(refined_shift[1])),
                    shift_norm_um=float(refined_metrics["shift_norm_um"]),
                    relaxed_xor_ratio=float(refined_metrics["xor_ratio"]),
                    signature_similarity=float(refined_metrics["signature_similarity"]),
                    alignment_backend="fast_minmax+fft_pcm",
                ), None, track_meta
            if not bool(refined_locality_eval.get("ok", False)):
                locality_eval = refined_locality_eval
            elif refined_metrics is not None:
                metrics = refined_metrics
                locality_eval = refined_locality_eval
            track_meta.update({
                "target_marker_miss": bool(locality_eval.get("target_marker_miss", False)),
                "source_marker_miss": bool(locality_eval.get("source_marker_miss", False)),
                "source_marker_overlap_ratio": float(locality_eval.get("source_marker_overlap_ratio", track_meta["source_marker_overlap_ratio"])),
                "raw_center_distance_um": float(locality_eval.get("raw_center_distance_um", 0.0)),
                "shifted_center_distance_um": float(locality_eval.get("shifted_center_distance_um", 0.0)),
                "shifted_bbox_overlap_ratio": float(locality_eval.get("shifted_bbox_overlap_ratio", 0.0)),
            })
    if not bool(locality_eval.get("ok", False)):
        return None, str(locality_eval.get("reject_reason", "rejected_by_locality")), track_meta
    if metrics is None:
        metrics = _compute_aligned_pair_metrics(source_item, target_item, shift)
    if metrics["xor_ratio"] > area_limit:
        return None, "rejected_by_area", track_meta
    return SimilarityEdge(
        source_idx=int(source_idx),
        target_idx=int(target_idx),
        shift=(float(shift[0]), float(shift[1])),
        shift_norm_um=float(metrics["shift_norm_um"]),
        relaxed_xor_ratio=float(metrics["xor_ratio"]),
        signature_similarity=float(metrics["signature_similarity"]),
        alignment_backend="fast_minmax",
    ), None, track_meta


def _build_strict_track_alignment_fast_minmax_hybrid_edge_area(source_item: PatternItem,
                                                               target_item: PatternItem,
                                                               cached_edge, *,
                                                               signature_floor,
                                                               area_match_ratio,
                                                               round_config: RoundConfig):
    track_meta = {
        "fft_pcm_attempt_count": 0,
        "fft_pcm_accept_count": 0,
        "fft_pcm_improved_edge_count": 0,
        "fft_pcm_improved_area_count": 0,
        "fft_pcm_shift_delta_sum_um": 0.0,
        "target_marker_miss": False,
        "source_marker_miss": False,
        "source_marker_overlap_ratio": 1.0,
        "raw_center_distance_um": 0.0,
        "shifted_center_distance_um": 0.0,
        "shifted_bbox_overlap_ratio": 0.0,
    }
    if cached_edge is None:
        return None, "rejected_by_locality", track_meta
    topology_dist = _topology_distance(
        source_item.topology_features,
        target_item.topology_features,
    )
    if topology_dist > float(round_config.strict_topology_threshold):
        return None, "rejected_by_topology", track_meta
    shift = tuple(float(v) for v in cached_edge.shift)
    locality_eval = _evaluate_alignment_locality_constraints(
        source_item,
        target_item,
        shift,
        max_shift_ratio=float(round_config.strict_max_shift_ratio),
        shift_norm_ratio=float(round_config.strict_shift_norm_ratio),
        min_shifted_bbox_overlap_ratio=float(round_config.strict_overlap_ratio),
        min_source_marker_overlap_ratio=float(round_config.strict_marker_overlap_ratio),
    )
    track_meta.update({
        "target_marker_miss": bool(locality_eval.get("target_marker_miss", False)),
        "source_marker_miss": bool(locality_eval.get("source_marker_miss", False)),
        "source_marker_overlap_ratio": float(locality_eval.get("source_marker_overlap_ratio", track_meta["source_marker_overlap_ratio"])),
        "raw_center_distance_um": float(locality_eval.get("raw_center_distance_um", 0.0)),
        "shifted_center_distance_um": float(locality_eval.get("shifted_center_distance_um", 0.0)),
        "shifted_bbox_overlap_ratio": float(locality_eval.get("shifted_bbox_overlap_ratio", 0.0)),
    })
    metrics = None
    max_edge_displacement_um = None
    mean_edge_displacement_um = None
    edge_exceeded = False
    area_limit = max(0.0, 1.0 - float(area_match_ratio))
    if bool(locality_eval.get("ok", False)):
        metrics = _compute_aligned_pair_metrics(source_item, target_item, shift)
        if metrics["signature_similarity"] < float(signature_floor):
            reject_reason = "rejected_by_signature"
        elif metrics["xor_ratio"] > area_limit:
            reject_reason = "rejected_by_area"
        else:
            max_edge_displacement_um, mean_edge_displacement_um = _compute_edge_displacement_metrics(
                source_item,
                target_item,
                shift,
            )
            edge_exceeded = max_edge_displacement_um > float(round_config.strict_edge_threshold_um)
            reject_reason = "rejected_by_edge" if edge_exceeded else None
            if reject_reason is None:
                return AlignmentResult(
                    member_idx=int(source_item.item_id),
                    rep_idx=int(target_item.item_id),
                    accepted=True,
                    shift=(float(shift[0]), float(shift[1])),
                    shift_norm_um=float(metrics["shift_norm_um"]),
                    shifted_xor_ratio=float(metrics["xor_ratio"]),
                    shifted_signature_similarity=float(metrics["signature_similarity"]),
                    alignment_backend=str(cached_edge.alignment_backend or round_config.alignment_backend),
                    constraint_mode=str(round_config.strict_constraint_mode),
                    max_edge_displacement_um=float(max_edge_displacement_um),
                    mean_edge_displacement_um=float(mean_edge_displacement_um),
                ), None, track_meta
    else:
        reject_reason = str(locality_eval.get("reject_reason", "rejected_by_locality"))
    if _should_try_fft_pcm_for_strict(locality_eval, metrics, area_match_ratio, edge_exceeded):
        track_meta["fft_pcm_attempt_count"] += 1
        refined = _refine_shift_with_fft_pcm(
            source_item,
            target_item,
            shift,
            grid_size=int(round_config.fft_grid_size),
        )
        if refined is not None:
            refined_shift = tuple(float(v) for v in refined["shift"])
            refined_locality_eval = _evaluate_alignment_locality_constraints(
                source_item,
                target_item,
                refined_shift,
                max_shift_ratio=float(round_config.strict_max_shift_ratio),
                shift_norm_ratio=float(round_config.strict_shift_norm_ratio),
                min_shifted_bbox_overlap_ratio=float(round_config.strict_overlap_ratio),
                min_source_marker_overlap_ratio=float(round_config.strict_marker_overlap_ratio),
            )
            refined_metrics = None
            refined_max_edge = None
            refined_mean_edge = None
            refined_reject_reason = None
            if bool(refined_locality_eval.get("ok", False)):
                refined_metrics = _compute_aligned_pair_metrics(source_item, target_item, refined_shift)
                if refined_metrics["signature_similarity"] < float(signature_floor):
                    refined_reject_reason = "rejected_by_signature"
                elif refined_metrics["xor_ratio"] > area_limit:
                    refined_reject_reason = "rejected_by_area"
                else:
                    refined_max_edge, refined_mean_edge = _compute_edge_displacement_metrics(
                        source_item,
                        target_item,
                        refined_shift,
                    )
                    if refined_max_edge > float(round_config.strict_edge_threshold_um):
                        refined_reject_reason = "rejected_by_edge"
            else:
                refined_reject_reason = str(refined_locality_eval.get("reject_reason", "rejected_by_locality"))
            edge_improved = (
                edge_exceeded
                and refined_max_edge is not None
                and max_edge_displacement_um is not None
                and float(refined_max_edge) + 1e-9 < float(max_edge_displacement_um)
            )
            area_improved = (
                metrics is not None
                and refined_metrics is not None
                and float(refined_metrics["xor_ratio"]) + 1e-9 < float(metrics["xor_ratio"])
            )
            if refined_reject_reason is None and (
                reject_reason in {"rejected_by_max_shift_cap", "rejected_by_shift_norm", "rejected_by_source_marker"}
                or reject_reason == "rejected_by_edge"
                or reject_reason == "rejected_by_area"
                or edge_improved
                or area_improved
            ):
                track_meta["fft_pcm_accept_count"] += 1
                track_meta["fft_pcm_shift_delta_sum_um"] += float(math.hypot(*refined["delta_shift"]))
                if edge_improved:
                    track_meta["fft_pcm_improved_edge_count"] += 1
                if area_improved:
                    track_meta["fft_pcm_improved_area_count"] += 1
                track_meta.update({
                    "target_marker_miss": bool(refined_locality_eval.get("target_marker_miss", False)),
                    "source_marker_miss": bool(refined_locality_eval.get("source_marker_miss", False)),
                    "source_marker_overlap_ratio": float(refined_locality_eval.get("source_marker_overlap_ratio", track_meta["source_marker_overlap_ratio"])),
                    "raw_center_distance_um": float(refined_locality_eval.get("raw_center_distance_um", 0.0)),
                    "shifted_center_distance_um": float(refined_locality_eval.get("shifted_center_distance_um", 0.0)),
                    "shifted_bbox_overlap_ratio": float(refined_locality_eval.get("shifted_bbox_overlap_ratio", 0.0)),
                })
                return AlignmentResult(
                    member_idx=int(source_item.item_id),
                    rep_idx=int(target_item.item_id),
                    accepted=True,
                    shift=(float(refined_shift[0]), float(refined_shift[1])),
                    shift_norm_um=float(refined_metrics["shift_norm_um"]),
                    shifted_xor_ratio=float(refined_metrics["xor_ratio"]),
                    shifted_signature_similarity=float(refined_metrics["signature_similarity"]),
                    alignment_backend="fast_minmax+fft_pcm",
                    constraint_mode=str(round_config.strict_constraint_mode),
                    max_edge_displacement_um=float(refined_max_edge),
                    mean_edge_displacement_um=float(refined_mean_edge),
                ), None, track_meta
            reject_reason = str(refined_reject_reason or reject_reason)
            locality_eval = refined_locality_eval
            track_meta.update({
                "target_marker_miss": bool(locality_eval.get("target_marker_miss", False)),
                "source_marker_miss": bool(locality_eval.get("source_marker_miss", False)),
                "source_marker_overlap_ratio": float(locality_eval.get("source_marker_overlap_ratio", track_meta["source_marker_overlap_ratio"])),
                "raw_center_distance_um": float(locality_eval.get("raw_center_distance_um", 0.0)),
                "shifted_center_distance_um": float(locality_eval.get("shifted_center_distance_um", 0.0)),
                "shifted_bbox_overlap_ratio": float(locality_eval.get("shifted_bbox_overlap_ratio", 0.0)),
            })
    return None, str(reject_reason or "rejected_by_locality"), track_meta


def _dispatch_graph_track_edge(source_idx, target_idx, source_item, target_item, *,
                               round_config: RoundConfig):
    backend = str(round_config.alignment_backend)
    constraint_mode = str(round_config.graph_constraint_mode)
    if backend == "fast_minmax+fft_pcm" and constraint_mode == "area_cosine":
        return _build_graph_track_edge_fast_minmax_area_cosine(
            source_idx,
            target_idx,
            source_item,
            target_item,
            signature_floor=round_config.graph_signature_floor,
            area_match_ratio=round_config.graph_area_match_ratio,
            graph_invariant_score_limit=round_config.graph_invariant_score_limit,
            round_config=round_config,
        )
    raise ValueError(f"Unsupported graph track: backend={backend}, constraint_mode={constraint_mode}")


def _dispatch_strict_track_alignment(source_item: PatternItem, target_item: PatternItem, cached_edge, *,
                                     round_config: RoundConfig):
    backend = str(round_config.alignment_backend)
    constraint_mode = str(round_config.strict_constraint_mode)
    if backend == "fast_minmax+fft_pcm" and constraint_mode == "hybrid_edge_area":
        return _build_strict_track_alignment_fast_minmax_hybrid_edge_area(
            source_item,
            target_item,
            cached_edge,
            signature_floor=round_config.strict_signature_floor,
            area_match_ratio=round_config.strict_area_match_ratio,
            round_config=round_config,
        )
    raise ValueError(f"Unsupported strict track: backend={backend}, constraint_mode={constraint_mode}")


def _iter_blocks(values, block_size):
    size = max(1, int(block_size))
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _evaluate_graph_source_block(pattern_items, source_indices, bucket_index, round_config: RoundConfig):
    candidate_pair_count = 0
    block_results = []
    neighbor_radius = max(0, int(round_config.graph_search_neighbor_radius))
    for source_idx in source_indices:
        source_item = pattern_items[source_idx]
        source_bucket = tuple(int(v) for v in source_item.graph_search_key)
        seen_targets = set()
        for neighbor_key in _neighboring_graph_search_keys(source_bucket, radius=neighbor_radius):
            for target_idx in bucket_index.get(neighbor_key, ()):
                if target_idx <= source_idx or target_idx in seen_targets:
                    continue
                seen_targets.add(target_idx)
                candidate_pair_count += 1
                target_item = pattern_items[target_idx]
                edge, reject_reason, track_meta = _dispatch_graph_track_edge(
                    source_idx,
                    target_idx,
                    source_item,
                    target_item,
                    round_config=round_config,
                )
                block_results.append((source_idx, target_idx, edge, reject_reason, track_meta))
    return candidate_pair_count, block_results


def _evaluate_strict_member_block(pattern_items, rep_idx, member_indices, adjacency, round_config: RoundConfig):
    block_results = []
    rep_item = pattern_items[rep_idx]
    for member_idx in member_indices:
        strict_result, reject_reason, track_meta = _dispatch_strict_track_alignment(
            pattern_items[member_idx],
            rep_item,
            adjacency.get(member_idx, {}).get(rep_idx),
            round_config=round_config,
        )
        block_results.append((member_idx, strict_result, reject_reason, track_meta))
    return block_results


def build_sparse_similarity_graph(pattern_items: List[PatternItem], *,
                                  round_config: RoundConfig,
                                  workers: int = 1,
                                  block_size: int = 64,
                                  parallel_min_items: int = 256):
    adjacency: Dict[int, Dict[int, SimilarityEdge]] = {idx: {} for idx in range(len(pattern_items))}
    bucket_index = {}
    edge_count = 0
    candidate_pair_count = 0
    rejection_counts = _new_rejection_counters(
        "rejected_by_invariant",
        "rejected_by_topology",
        "rejected_by_signature",
        "rejected_by_locality",
        "rejected_by_max_shift_cap",
        "rejected_by_shift_norm",
        "rejected_by_source_marker",
        "rejected_by_shifted_bbox_overlap",
        "rejected_by_area",
    )
    fft_pcm_counts = _new_fft_pcm_counters(
        "fft_pcm_attempt_count",
        "fft_pcm_accept_count",
        "fft_pcm_improved_locality_count",
        "fft_pcm_improved_area_count",
    )
    fft_pcm_shift_delta_sum_um = 0.0
    target_marker_miss_count = 0
    max_shift_cap_exceeded_count = 0
    shift_norm_exceeded_count = 0
    raw_center_distance_values = []
    shifted_bbox_overlap_values = []
    source_marker_overlap_values = []
    raw_center_distance_over_1x_window_count = 0
    for idx, item in enumerate(pattern_items):
        bucket_index.setdefault(tuple(int(v) for v in item.graph_search_key), []).append(int(idx))

    source_blocks = list(_iter_blocks(list(range(len(pattern_items))), int(block_size)))
    if int(workers) > 1 and len(pattern_items) >= int(parallel_min_items):
        with ThreadPoolExecutor(max_workers=int(workers)) as executor:
            block_iter = executor.map(
                _evaluate_graph_source_block,
                itertools.repeat(pattern_items),
                source_blocks,
                itertools.repeat(bucket_index),
                itertools.repeat(round_config),
            )
            block_outputs = list(block_iter)
    else:
        block_outputs = [
            _evaluate_graph_source_block(pattern_items, source_block, bucket_index, round_config)
            for source_block in source_blocks
        ]

    for block_candidate_pair_count, block_results in block_outputs:
        candidate_pair_count += int(block_candidate_pair_count)
        for source_idx, target_idx, edge, reject_reason, track_meta in block_results:
            target_marker_miss_count += int(bool(track_meta.get("target_marker_miss", False)))
            max_shift_cap_exceeded_count += int(bool(track_meta.get("max_shift_cap_exceeded", False)))
            shift_norm_exceeded_count += int(bool(track_meta.get("shift_norm_exceeded", False)))
            raw_center_distance_um = float(track_meta.get("raw_center_distance_um", 0.0))
            shifted_bbox_overlap_ratio = float(track_meta.get("shifted_bbox_overlap_ratio", 0.0))
            source_marker_overlap_ratio = float(track_meta.get("source_marker_overlap_ratio", 1.0))
            raw_center_distance_values.append(raw_center_distance_um)
            shifted_bbox_overlap_values.append(shifted_bbox_overlap_ratio)
            source_marker_overlap_values.append(source_marker_overlap_ratio)
            outer_window_span = max(
                max(_bbox_size(pattern_items[source_idx].outer_bbox)),
                max(_bbox_size(pattern_items[target_idx].outer_bbox)),
                1e-6,
            )
            raw_center_distance_over_1x_window_count += int(raw_center_distance_um > outer_window_span)
            for key in fft_pcm_counts:
                fft_pcm_counts[key] += int(track_meta.get(key, 0))
            fft_pcm_shift_delta_sum_um += float(track_meta.get("fft_pcm_shift_delta_sum_um", 0.0))
            if edge is None:
                if reject_reason in rejection_counts:
                    rejection_counts[reject_reason] += 1
                    if reject_reason in {
                        "rejected_by_max_shift_cap",
                        "rejected_by_shift_norm",
                        "rejected_by_source_marker",
                        "rejected_by_shifted_bbox_overlap",
                    }:
                        rejection_counts["rejected_by_locality"] += 1
                continue
            adjacency[source_idx][target_idx] = edge
            adjacency[target_idx][source_idx] = edge.reversed()
            edge_count += 1
    degrees = [len(neighbors) for neighbors in adjacency.values()]
    raw_center_distance_mean_um, raw_center_distance_p95_um = _distribution_mean_p95(raw_center_distance_values)
    return adjacency, {
        "active_item_count": int(len(pattern_items)),
        "candidate_pair_count": int(candidate_pair_count),
        "graph_edge_count": int(edge_count),
        "max_degree": int(max(degrees) if degrees else 0),
        "mean_degree": float(np.mean(degrees)) if degrees else 0.0,
        "graph_invariant_score_limit": float(round_config.graph_invariant_score_limit),
        "graph_signature_type": "occupancy_grid_v1",
        "graph_invariant_type": "weighted_norm_v1",
        "graph_search_key_type": "occupancy_top4set_centroid_v2",
        "graph_search_neighbor_mode": f"centroid_{2 * int(round_config.graph_search_neighbor_radius) + 1}x{2 * int(round_config.graph_search_neighbor_radius) + 1}",
        "parallel_workers": int(max(1, workers)),
        "parallel_block_size": int(max(1, block_size)),
        "alignment_backend": str(round_config.alignment_backend),
        "graph_constraint_mode": str(round_config.graph_constraint_mode),
        "strict_constraint_mode": str(round_config.strict_constraint_mode),
        "strict_edge_threshold_um": float(round_config.strict_edge_threshold_um),
        "graph_topology_threshold": float(round_config.graph_topology_threshold),
        "graph_max_shift_ratio": float(round_config.graph_max_shift_ratio),
        "graph_shift_norm_ratio": float(round_config.graph_shift_norm_ratio),
        "graph_overlap_ratio": float(round_config.graph_overlap_ratio),
        "graph_marker_overlap_ratio": float(round_config.graph_marker_overlap_ratio),
        "graph_track_impl": f"{round_config.alignment_backend}.{round_config.graph_constraint_mode}",
        "fft_pcm_avg_shift_delta_um": (
            float(fft_pcm_shift_delta_sum_um) / max(1, int(fft_pcm_counts["fft_pcm_accept_count"]))
            if int(fft_pcm_counts["fft_pcm_accept_count"]) > 0 else 0.0
        ),
        "target_marker_miss_count": int(target_marker_miss_count),
        "max_shift_cap_exceeded_count": int(max_shift_cap_exceeded_count),
        "shift_norm_exceeded_count": int(shift_norm_exceeded_count),
        "raw_center_distance_mean_um": float(raw_center_distance_mean_um),
        "raw_center_distance_p95_um": float(raw_center_distance_p95_um),
        "raw_center_distance_over_1x_window_count": int(raw_center_distance_over_1x_window_count),
        "shifted_bbox_overlap_mean": (
            float(np.mean(shifted_bbox_overlap_values)) if shifted_bbox_overlap_values else 0.0
        ),
        "source_marker_overlap_mean": (
            float(np.mean(source_marker_overlap_values)) if source_marker_overlap_values else 0.0
        ),
        **fft_pcm_counts,
        **rejection_counts,
    }


def solve_surprisal_lazy_cover(adjacency, cluster_weights):
    node_count = int(len(cluster_weights))
    uncovered = set(range(node_count))
    uncovered_version = 0
    selected = []
    selected_scores = {}
    coverage_sets = {
        idx: {int(idx)} | {int(v) for v in adjacency.get(idx, {}).keys()}
        for idx in range(node_count)
    }
    surprisal = {
        idx: 1.0 / float(1 + len(adjacency.get(idx, {})))
        for idx in range(node_count)
    }

    def _candidate_score(idx, current_uncovered):
        active_coverage = coverage_sets[idx] & current_uncovered
        if not active_coverage:
            return None, None
        active_neighbors = active_coverage - {int(idx)}
        score = float(surprisal[idx]) + float(sum(surprisal[k] for k in active_neighbors))
        covered_weight = int(sum(int(cluster_weights[k]) for k in active_coverage))
        return (
            float(score),
            int(covered_weight),
            int(len(active_coverage)),
            int(cluster_weights[idx]),
            -int(idx),
        ), active_coverage

    def _push_candidate(heap, idx, version_snapshot, current_uncovered):
        score, _ = _candidate_score(idx, current_uncovered)
        if score is None:
            return
        heapq.heappush(
            heap,
            (-score[0], -score[1], -score[2], -score[3], -score[4], int(version_snapshot), int(idx)),
        )
    heap = []
    for idx in range(node_count):
        _push_candidate(heap, idx, uncovered_version, uncovered)
    while uncovered and heap:
        _, _, _, _, _, version_snapshot, idx = heapq.heappop(heap)
        if int(version_snapshot) != int(uncovered_version):
            _push_candidate(heap, idx, uncovered_version, uncovered)
            continue
        score, active_coverage = _candidate_score(idx, uncovered)
        if active_coverage is None:
            continue
        selected.append(int(idx))
        selected_scores[int(idx)] = float(score[0]) if score is not None else 0.0
        uncovered -= active_coverage
        uncovered_version += 1
    return [
        CoarseCluster(
            rep_idx=int(idx),
            member_indices=sorted(int(v) for v in coverage_sets[idx]),
            score=float(selected_scores.get(int(idx), 0.0)),
        )
        for idx in selected
    ]


def refine_clusters_by_alignment(pattern_items, adjacency, coarse_clusters, *,
                                 round_config: RoundConfig,
                                 workers: int = 1,
                                 block_size: int = 128,
                                 parallel_min_items: int = 256):
    final_clusters = []
    orphan_indices = set(range(len(pattern_items)))
    assigned = set()
    verified_merge_count = 0
    shift_values = []
    edge_max_values = []
    rejection_counts = _new_rejection_counters(
        "rejected_by_locality",
        "rejected_by_max_shift_cap",
        "rejected_by_shift_norm",
        "rejected_by_source_marker",
        "rejected_by_shifted_bbox_overlap",
        "rejected_by_topology",
        "rejected_by_signature",
        "rejected_by_area",
        "rejected_by_edge",
    )
    fft_pcm_counts = _new_fft_pcm_counters(
        "fft_pcm_attempt_count",
        "fft_pcm_accept_count",
        "fft_pcm_improved_edge_count",
        "fft_pcm_improved_area_count",
    )
    fft_pcm_shift_delta_sum_um = 0.0
    target_marker_miss_count = 0
    max_shift_cap_exceeded_count = 0
    shift_norm_exceeded_count = 0
    raw_center_distance_values = []
    shifted_bbox_overlap_values = []
    source_marker_overlap_values = []
    raw_center_distance_over_1x_window_count = 0
    for coarse_cluster in coarse_clusters:
        rep_idx = int(coarse_cluster.rep_idx)
        if rep_idx in assigned:
            continue
        accepted_member_indices = [rep_idx]
        alignment_results = [
            AlignmentResult(
                member_idx=int(rep_idx),
                rep_idx=int(rep_idx),
                accepted=True,
                shift=(0.0, 0.0),
                shift_norm_um=0.0,
                shifted_xor_ratio=0.0,
                shifted_signature_similarity=1.0,
                alignment_backend=str(round_config.alignment_backend),
                constraint_mode=str(round_config.strict_constraint_mode),
                max_edge_displacement_um=0.0,
                mean_edge_displacement_um=0.0,
            )
        ]
        assigned.add(rep_idx)
        orphan_indices.discard(rep_idx)
        member_indices = [
            int(v) for v in sorted(int(v) for v in coarse_cluster.member_indices if int(v) != rep_idx)
            if int(v) not in assigned and int(v) != rep_idx
        ]
        member_blocks = list(_iter_blocks(member_indices, int(block_size)))
        if int(workers) > 1 and len(member_indices) >= int(parallel_min_items):
            with ThreadPoolExecutor(max_workers=int(workers)) as executor:
                block_outputs = list(
                    executor.map(
                        _evaluate_strict_member_block,
                        itertools.repeat(pattern_items),
                        itertools.repeat(rep_idx),
                        member_blocks,
                        itertools.repeat(adjacency),
                        itertools.repeat(round_config),
                    )
                )
        else:
            block_outputs = [
                _evaluate_strict_member_block(pattern_items, rep_idx, member_block, adjacency, round_config)
                for member_block in member_blocks
            ]
        for block_results in block_outputs:
            for member_idx, strict_result, reject_reason, track_meta in block_results:
                target_marker_miss_count += int(bool(track_meta.get("target_marker_miss", False)))
                max_shift_cap_exceeded_count += int(bool(track_meta.get("max_shift_cap_exceeded", False)))
                shift_norm_exceeded_count += int(bool(track_meta.get("shift_norm_exceeded", False)))
                raw_center_distance_um = float(track_meta.get("raw_center_distance_um", 0.0))
                shifted_bbox_overlap_ratio = float(track_meta.get("shifted_bbox_overlap_ratio", 0.0))
                source_marker_overlap_ratio = float(track_meta.get("source_marker_overlap_ratio", 1.0))
                raw_center_distance_values.append(raw_center_distance_um)
                shifted_bbox_overlap_values.append(shifted_bbox_overlap_ratio)
                source_marker_overlap_values.append(source_marker_overlap_ratio)
                outer_window_span = max(
                    max(_bbox_size(pattern_items[member_idx].outer_bbox)),
                    max(_bbox_size(pattern_items[rep_idx].outer_bbox)),
                    1e-6,
                )
                raw_center_distance_over_1x_window_count += int(raw_center_distance_um > outer_window_span)
                for key in fft_pcm_counts:
                    fft_pcm_counts[key] += int(track_meta.get(key, 0))
                fft_pcm_shift_delta_sum_um += float(track_meta.get("fft_pcm_shift_delta_sum_um", 0.0))
                if strict_result is None:
                    if reject_reason in rejection_counts:
                        rejection_counts[reject_reason] += 1
                        if reject_reason in {
                            "rejected_by_max_shift_cap",
                            "rejected_by_shift_norm",
                            "rejected_by_source_marker",
                            "rejected_by_shifted_bbox_overlap",
                        }:
                            rejection_counts["rejected_by_locality"] += 1
                    continue
                accepted_member_indices.append(member_idx)
                alignment_results.append(strict_result)
                assigned.add(member_idx)
                orphan_indices.discard(member_idx)
                verified_merge_count += 1
                shift_values.append(float(strict_result.shift_norm_um))
                edge_max_values.append(float(strict_result.max_edge_displacement_um))
        final_clusters.append(
            FinalCluster(
                rep_id=int(pattern_items[rep_idx].item_id),
                member_indices=[int(pattern_items[idx].item_id) for idx in accepted_member_indices],
                alignment_results=list(alignment_results),
            )
        )
    orphan_ids = sorted(int(pattern_items[idx].item_id) for idx in orphan_indices)
    raw_center_distance_mean_um, raw_center_distance_p95_um = _distribution_mean_p95(raw_center_distance_values)
    return final_clusters, orphan_ids, {
        "coarse_cluster_count": int(len(coarse_clusters)),
        "verified_merge_count": int(verified_merge_count),
        "orphan_count": int(len(orphan_ids)),
        "avg_shift_um": float(np.mean(shift_values)) if shift_values else 0.0,
        "max_shift_um": float(np.max(shift_values)) if shift_values else 0.0,
        "avg_edge_displacement_um": float(np.mean(edge_max_values)) if edge_max_values else 0.0,
        "max_edge_displacement_um": float(np.max(edge_max_values)) if edge_max_values else 0.0,
        "parallel_workers": int(max(1, workers)),
        "parallel_block_size": int(max(1, block_size)),
        "strict_topology_threshold": float(round_config.strict_topology_threshold),
        "strict_marker_overlap_ratio": float(round_config.strict_marker_overlap_ratio),
        "strict_track_impl": f"{round_config.alignment_backend}.{round_config.strict_constraint_mode}",
        "fft_pcm_avg_shift_delta_um": (
            float(fft_pcm_shift_delta_sum_um) / max(1, int(fft_pcm_counts["fft_pcm_accept_count"]))
            if int(fft_pcm_counts["fft_pcm_accept_count"]) > 0 else 0.0
        ),
        "shift_values": list(shift_values),
        "target_marker_miss_count": int(target_marker_miss_count),
        "max_shift_cap_exceeded_count": int(max_shift_cap_exceeded_count),
        "shift_norm_exceeded_count": int(shift_norm_exceeded_count),
        "raw_center_distance_mean_um": float(raw_center_distance_mean_um),
        "raw_center_distance_p95_um": float(raw_center_distance_p95_um),
        "raw_center_distance_over_1x_window_count": int(raw_center_distance_over_1x_window_count),
        "shifted_bbox_overlap_mean": (
            float(np.mean(shifted_bbox_overlap_values)) if shifted_bbox_overlap_values else 0.0
        ),
        "source_marker_overlap_mean": (
            float(np.mean(source_marker_overlap_values)) if source_marker_overlap_values else 0.0
        ),
        **fft_pcm_counts,
        **rejection_counts,
    }


class SCPClosedLoopClustering:
    """单一 closed-loop / SCP 聚类器。"""

    def __init__(self, similarity_threshold: float = 0.96, max_iterations: int = 3, workers: Optional[int] = None):
        self.similarity_threshold = float(similarity_threshold)
        self.max_iterations = max(1, int(max_iterations))
        self.workers = max(1, int(workers or os.cpu_count() or 1))
        self.graph_parallel_block_size = 64
        self.refine_parallel_block_size = 128
        self.parallel_min_block_items = 256
        self.last_debug = {
            "alignment_round_debug": [],
            "sparse_graph": {"rounds": []},
            "alignment_refinement": {"rounds": []},
            "closed_loop": {"rounds": [], "executed_rounds": 0, "final_orphan_count": 0},
            "alignment_graph_edge_count": 0,
            "alignment_rounds": 0,
            "alignment_verified_merge_count": 0,
            "alignment_orphan_count": 0,
            "alignment_avg_shift_um": 0.0,
        }

    def _round_configs(self) -> List[RoundConfig]:
        configs = []
        for round_index in range(1, self.max_iterations + 1):
            signature_floor = min(0.76, 0.72 + 0.02 * float(round_index - 1))
            invariant_limits = {1: 0.24, 2: 0.22}
            graph_invariant_score_limit = float(invariant_limits.get(int(round_index), 0.20))
            tighten = float(round_index - 1) * 0.015
            graph_area_match_ratio = float(np.clip(self.similarity_threshold - 0.15 + tighten, 0.77, 0.95))
            if int(round_index) == 1:
                graph_search_neighbor_radius = 2
                graph_max_shift_ratio = 1.15
                graph_shift_norm_ratio = 0.95
                graph_topology_threshold = 8.0
                graph_overlap_ratio = 0.10
            elif int(round_index) == 2:
                graph_search_neighbor_radius = 1
                graph_max_shift_ratio = 1.00
                graph_shift_norm_ratio = 0.85
                graph_topology_threshold = 6.5
                graph_overlap_ratio = 0.10
            else:
                graph_search_neighbor_radius = 1
                graph_max_shift_ratio = 0.90
                graph_shift_norm_ratio = 0.80
                graph_topology_threshold = 5.0
                graph_overlap_ratio = 0.10
            configs.append(
                RoundConfig(
                    round_index=int(round_index),
                    graph_signature_floor=float(signature_floor),
                    graph_area_match_ratio=float(graph_area_match_ratio),
                    strict_signature_floor=float(np.clip(self.similarity_threshold - 0.12 + tighten, 0.76, 0.92)),
                    strict_area_match_ratio=float(np.clip(self.similarity_threshold - 0.08 + tighten, 0.84, 0.97)),
                    graph_invariant_score_limit=graph_invariant_score_limit,
                    alignment_backend="fast_minmax+fft_pcm",
                    graph_constraint_mode="area_cosine",
                    strict_constraint_mode="hybrid_edge_area",
                    strict_edge_threshold_um=0.0,
                    graph_topology_threshold=float(graph_topology_threshold),
                    strict_topology_threshold=3.0,
                    graph_search_neighbor_radius=int(graph_search_neighbor_radius),
                    graph_max_shift_ratio=float(graph_max_shift_ratio),
                    graph_shift_norm_ratio=float(graph_shift_norm_ratio),
                    graph_overlap_ratio=float(graph_overlap_ratio),
                    graph_marker_overlap_ratio=0.0,
                    strict_max_shift_ratio=0.75,
                    strict_shift_norm_ratio=0.60,
                    strict_overlap_ratio=0.20,
                    strict_marker_overlap_ratio=0.10,
                    fft_grid_size=64,
                )
            )
        return configs

    def cluster(self, pattern_items: List[PatternItem]) -> List[FinalCluster]:
        item_count = len(pattern_items)
        self.last_debug = {
            "alignment_round_debug": [],
            "sparse_graph": {"rounds": []},
            "alignment_refinement": {"rounds": []},
            "closed_loop": {"rounds": [], "executed_rounds": 0, "final_orphan_count": item_count},
            "alignment_graph_edge_count": 0,
            "alignment_rounds": 0,
            "alignment_verified_merge_count": 0,
            "alignment_orphan_count": item_count,
            "alignment_avg_shift_um": 0.0,
        }
        if item_count <= 0:
            return []
        if item_count == 1:
            item = pattern_items[0]
            self.last_debug["alignment_orphan_count"] = 1
            self.last_debug["closed_loop"]["final_orphan_count"] = 1
            return [FinalCluster(rep_id=int(item.item_id), member_indices=[int(item.item_id)])]
        final_clusters: List[FinalCluster] = []
        working_ids = [int(item.item_id) for item in pattern_items]
        graph_edge_count = 0
        verified_merge_count = 0
        all_shift_values: List[float] = []
        round_debug = []
        sparse_graph_rounds = []
        refinement_rounds = []
        closed_loop_rounds = []
        executed_rounds = 0
        item_map = {int(item.item_id): item for item in pattern_items}
        for round_config in self._round_configs():
            if not working_ids:
                break
            executed_rounds = int(round_config.round_index)
            round_items = [item_map[item_id] for item_id in working_ids]
            round_outer_window_span = max(
                (max(_bbox_size(item.outer_bbox)) for item in round_items),
                default=0.0,
            )
            edge_floor = 0.03 if int(round_config.round_index) == 1 else 0.02
            edge_ratio = 0.08 if int(round_config.round_index) == 1 else 0.06
            effective_round_config = replace(
                round_config,
                strict_edge_threshold_um=float(max(edge_floor, round_outer_window_span * edge_ratio)),
            )
            marker_sources = sorted({str(item.marker_source or "none") for item in round_items})
            round_marker_source = marker_sources[0] if len(marker_sources) == 1 else "mixed"
            adjacency, graph_debug = build_sparse_similarity_graph(
                round_items,
                round_config=effective_round_config,
                workers=self.workers,
                block_size=self.graph_parallel_block_size,
                parallel_min_items=self.parallel_min_block_items,
            )
            graph_edge_count += int(graph_debug["graph_edge_count"])
            cluster_weights = [int(item.duplicate_count) for item in round_items]
            coarse_clusters = solve_surprisal_lazy_cover(adjacency, cluster_weights)
            round_final_clusters, orphan_ids, round_stats = refine_clusters_by_alignment(
                round_items,
                adjacency,
                coarse_clusters,
                round_config=effective_round_config,
                workers=self.workers,
                block_size=self.refine_parallel_block_size,
                parallel_min_items=self.parallel_min_block_items,
            )
            final_clusters.extend(round_final_clusters)
            verified_merge_count += int(round_stats["verified_merge_count"])
            all_shift_values.extend(list(round_stats.get("shift_values", [])))
            sparse_graph_round = {
                "round": int(round_config.round_index),
                "marker_source": str(round_marker_source),
                **graph_debug,
            }
            refinement_round = {
                "round": int(round_config.round_index),
                "marker_source": str(round_marker_source),
                "coarse_cluster_count": int(len(coarse_clusters)),
                "final_cluster_count": int(len(round_final_clusters)),
                "alignment_backend": str(effective_round_config.alignment_backend),
                "graph_constraint_mode": str(effective_round_config.graph_constraint_mode),
                "strict_constraint_mode": str(effective_round_config.strict_constraint_mode),
                "strict_edge_threshold_um": float(effective_round_config.strict_edge_threshold_um),
                **{key: value for key, value in round_stats.items() if key != "shift_values"},
            }
            closed_loop_round = {
                "round": int(round_config.round_index),
                "marker_source": str(round_marker_source),
                "active_item_count": int(graph_debug["active_item_count"]),
                "candidate_pair_count": int(graph_debug["candidate_pair_count"]),
                "graph_edge_count": int(graph_debug["graph_edge_count"]),
                "coarse_cluster_count": int(len(coarse_clusters)),
                "verified_merge_count": int(round_stats["verified_merge_count"]),
                "orphan_count": int(len(orphan_ids)),
                "avg_shift_um": float(round_stats["avg_shift_um"]),
                "max_shift_um": float(round_stats["max_shift_um"]),
                "avg_edge_displacement_um": float(round_stats.get("avg_edge_displacement_um", 0.0)),
                "max_edge_displacement_um": float(round_stats.get("max_edge_displacement_um", 0.0)),
                "alignment_backend": str(effective_round_config.alignment_backend),
                "graph_constraint_mode": str(effective_round_config.graph_constraint_mode),
                "strict_constraint_mode": str(effective_round_config.strict_constraint_mode),
                "strict_edge_threshold_um": float(effective_round_config.strict_edge_threshold_um),
                "graph_track_impl": str(graph_debug.get("graph_track_impl", "")),
                "strict_track_impl": str(round_stats.get("strict_track_impl", "")),
            }
            sparse_graph_rounds.append(sparse_graph_round)
            refinement_rounds.append(refinement_round)
            closed_loop_rounds.append(closed_loop_round)
            round_debug.append(dict(closed_loop_round))
            if not orphan_ids:
                working_ids = []
                break
            working_ids = orphan_ids
        for orphan_id in working_ids:
            final_clusters.append(FinalCluster(rep_id=int(orphan_id), member_indices=[int(orphan_id)]))
        final_clusters.sort(key=lambda cluster: (-len(cluster.member_indices), int(cluster.rep_id)))
        final_orphan_count = int(len(working_ids))
        graph_fft_attempts = int(sum(int(r.get("fft_pcm_attempt_count", 0)) for r in sparse_graph_rounds))
        graph_fft_accepts = int(sum(int(r.get("fft_pcm_accept_count", 0)) for r in sparse_graph_rounds))
        strict_fft_attempts = int(sum(int(r.get("fft_pcm_attempt_count", 0)) for r in refinement_rounds))
        strict_fft_accepts = int(sum(int(r.get("fft_pcm_accept_count", 0)) for r in refinement_rounds))
        fft_shift_weighted_sum = float(sum(
            float(r.get("fft_pcm_avg_shift_delta_um", 0.0)) * float(r.get("fft_pcm_accept_count", 0))
            for r in sparse_graph_rounds + refinement_rounds
        ))
        fft_total_accepts = int(graph_fft_accepts + strict_fft_accepts)
        self.last_debug = {
            "alignment_round_debug": round_debug,
            "sparse_graph": {
                "rounds": sparse_graph_rounds,
                "total_edge_count": int(graph_edge_count),
            },
            "alignment_refinement": {
                "rounds": refinement_rounds,
                "verified_merge_count": int(verified_merge_count),
                "final_orphan_count": int(final_orphan_count),
            },
            "closed_loop": {
                "rounds": closed_loop_rounds,
                "executed_rounds": int(executed_rounds),
                "final_orphan_count": int(final_orphan_count),
                "fft_pcm_total_attempts": int(graph_fft_attempts + strict_fft_attempts),
                "fft_pcm_total_accepts": int(fft_total_accepts),
                "fft_pcm_avg_shift_delta_um": (
                    float(fft_shift_weighted_sum) / max(1, fft_total_accepts)
                    if fft_total_accepts > 0 else 0.0
                ),
            },
            "alignment_graph_edge_count": int(graph_edge_count),
            "alignment_rounds": int(executed_rounds),
            "alignment_verified_merge_count": int(verified_merge_count),
            "alignment_orphan_count": int(final_orphan_count),
            "alignment_avg_shift_um": float(np.mean(all_shift_values)) if all_shift_values else 0.0,
        }
        return final_clusters


class LayoutClusteringPipeline:
    """严格对齐 Liu 2025 论文主线的 marker-driven closed-loop pipeline。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = dict(config or {})
        self.input_pattern = str(self.config.get("pattern", "*.oas") or "*.oas")
        self.design_layer_spec = _parse_layer_datatype_spec(self.config.get("design_layer", "1/0"))
        self.marker_layer_spec = _parse_layer_datatype_spec(self.config.get("marker_layer", "2/0"))
        self.pattern_radius_um = max(1e-6, float(self.config.get("pattern_radius_um", 1.35)))
        self.progress_every = max(1, int(self.config.get("progress_every", 200)))
        self.workers = max(1, int(self.config.get("workers", os.cpu_count() or 1)))
        self.clusterer = SCPClosedLoopClustering(
            similarity_threshold=float(self.config.get("similarity_threshold", 0.96)),
            max_iterations=max(1, int(self.config.get("max_iterations", 3))),
            workers=self.workers,
        )
        if self.design_layer_spec is None:
            raise ValueError("design_layer is required, e.g. 1/0")
        if self.marker_layer_spec is None:
            raise ValueError("marker_layer is required, e.g. 2/0")
        self.pattern_items: List[PatternItem] = []
        self.graph_items: List[PatternItem] = []
        self.item_group_members: Dict[int, List[int]] = {}
        self.item_to_group: Dict[int, int] = {}
        self.exact_group_info: Dict[str, Any] = {}
        self.final_clusters: List[FinalCluster] = []
        self.pattern_generation_info: Dict[str, Any] = {}
        self.closed_loop_info: Dict[str, Any] = {}
        self.source_files: List[str] = []

    def _collect_input_files(self, input_path: str) -> List[str]:
        path = _resolve_fs_path(input_path)
        if path.is_file():
            _ensure_oas_input_path(str(path))
            return [str(path)]
        if not path.exists():
            raise FileNotFoundError(f"输入路径不存在: {path}")
        files = sorted(str(p) for p in path.rglob(self.input_pattern) if p.is_file())
        files = [fp for fp in files if Path(fp).suffix.lower() == ".oas"]
        return files

    def _record_to_sample_info(self, record: Dict[str, Any], source_path: str, marker_index: int) -> Dict[str, Any]:
        sample_info = record["sample"].to_dict()
        sample_info["source_path"] = str(source_path)
        sample_info["marker_index"] = int(marker_index)
        marker_bbox = record.get("absolute_marker_bbox", record.get("marker_bbox"))
        if marker_bbox is not None:
            sample_info["marker_bbox"] = [float(v) for v in marker_bbox]
        sample_info["marker_source"] = str(record.get("marker_source", "none"))
        return sample_info

    def _pattern_item_from_record(self, record: Dict[str, Any], source_path: str,
                                  source_name: str, marker_index: int, item_id: int) -> PatternItem:
        marker_bbox = record.get("marker_bbox")
        sample_info = self._record_to_sample_info(record, source_path, marker_index)
        return PatternItem(
            item_id=int(item_id),
            source_path=str(source_path),
            source_name=str(source_name),
            sample_info=sample_info,
            outer_bbox=tuple(float(v) for v in record.get("local_outer_bbox", record["sample"].outer_bbox)),
            outer_polygons=list(record.get("outer_polygons", [])),
            graph_search_key=tuple(int(v) for v in record.get("graph_search_key", ())),
            graph_invariants=np.asarray(record.get("graph_invariants", []), dtype=np.float64),
            graph_signature_grid=np.asarray(record.get("graph_signature_grid", []), dtype=np.float32),
            graph_signature_proj_x=np.asarray(record.get("graph_signature_proj_x", []), dtype=np.float32),
            graph_signature_proj_y=np.asarray(record.get("graph_signature_proj_y", []), dtype=np.float32),
            topology_features=np.asarray(record.get("topology_features", []), dtype=np.float64),
            duplicate_count=int(record["sample"].duplicate_count),
            marker_bbox=tuple(float(v) for v in marker_bbox) if marker_bbox is not None else None,
            marker_source=str(record.get("marker_source", "none")),
            alignment_x_intervals=np.asarray(record.get("alignment_x_intervals", []), dtype=np.float64),
            alignment_y_intervals=np.asarray(record.get("alignment_y_intervals", []), dtype=np.float64),
        )

    def _build_exact_group_items(self) -> List[PatternItem]:
        groups: Dict[str, List[PatternItem]] = {}
        for item in self.pattern_items:
            groups.setdefault(_exact_item_group_key(item), []).append(item)
        grouped_items: List[PatternItem] = []
        self.item_group_members = {}
        self.item_to_group = {}
        sorted_groups = sorted(
            groups.values(),
            key=lambda members: min(int(item.item_id) for item in members),
        )
        for group_id, members in enumerate(sorted_groups):
            members = sorted(members, key=lambda item: int(item.item_id))
            representative = members[0]
            member_ids = [int(item.item_id) for item in members]
            sample_info = dict(representative.sample_info)
            sample_info["exact_group_id"] = int(group_id)
            sample_info["exact_group_size"] = int(len(member_ids))
            grouped_item = replace(
                representative,
                item_id=int(group_id),
                sample_info=sample_info,
                duplicate_count=int(len(member_ids)),
                raster_bitmap_cache={},
                fft_frequency_cache={},
            )
            grouped_items.append(grouped_item)
            self.item_group_members[int(group_id)] = member_ids
            for member_id in member_ids:
                self.item_to_group[int(member_id)] = int(group_id)
        total_item_count = int(len(self.pattern_items))
        active_group_count = int(len(grouped_items))
        self.graph_items = grouped_items
        self.exact_group_info = {
            "enabled": True,
            "total_item_count": total_item_count,
            "active_group_count": active_group_count,
            "exact_group_compression_ratio": (
                float(active_group_count) / max(1, total_item_count)
            ),
            "exact_group_reduction_factor": (
                float(total_item_count) / max(1, active_group_count)
            ),
        }
        return list(grouped_items)

    def _expand_group_clusters(self, group_clusters: List[FinalCluster]) -> List[FinalCluster]:
        expanded_clusters: List[FinalCluster] = []
        for group_cluster in group_clusters:
            member_ids: List[int] = []
            for group_id in group_cluster.member_indices:
                member_ids.extend(self.item_group_members.get(int(group_id), [int(group_id)]))
            representative_members = self.item_group_members.get(int(group_cluster.rep_id), [int(group_cluster.rep_id)])
            rep_id = int(representative_members[0])
            alignment_results = []
            for result in getattr(group_cluster, "alignment_results", []) or []:
                member_original = self.item_group_members.get(int(result.member_idx), [int(result.member_idx)])[0]
                rep_original = self.item_group_members.get(int(result.rep_idx), [int(result.rep_idx)])[0]
                alignment_results.append(
                    replace(result, member_idx=int(member_original), rep_idx=int(rep_original))
                )
            expanded_clusters.append(
                FinalCluster(
                    rep_id=rep_id,
                    member_indices=sorted(int(v) for v in member_ids),
                    alignment_results=alignment_results,
                )
            )
        expanded_clusters.sort(key=lambda cluster: (-len(cluster.member_indices), int(cluster.rep_id)))
        return expanded_clusters

    def _build_marker_window_records(self, lib, *, source_name: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        expanded_layers, top_cell_names = _collect_expanded_layer_geometries(
            lib,
            [self.design_layer_spec, self.marker_layer_spec],
        )
        design_geometries = expanded_layers.get(tuple(self.design_layer_spec), [])
        marker_geometries = expanded_layers.get(tuple(self.marker_layer_spec), [])
        spatial_index, indexed_elements, layout_bbox = _build_layout_spatial_index(design_geometries)
        if spatial_index is None or not indexed_elements:
            raise ValueError(f"文件 {source_name} 的 design layer {self.design_layer_spec[0]}/{self.design_layer_spec[1]} 中没有有效几何")
        raw_marker_bboxes = _collect_layer_bboxes(marker_geometries)
        if not raw_marker_bboxes:
            raise ValueError(
                f"文件 {source_name} 中未找到 marker layer {self.marker_layer_spec[0]}/{self.marker_layer_spec[1]} 的任何几何"
            )
        marker_bboxes = _merge_touching_marker_bboxes(raw_marker_bboxes)
        records = []
        marker_source = _marker_source_for_layer(self.marker_layer_spec)
        window_side_um = float(self.pattern_radius_um * 2.0)
        total_markers = len(marker_bboxes)
        for marker_index, marker_bbox in enumerate(marker_bboxes):
            if total_markers >= self.progress_every and (
                marker_index == 0
                or (marker_index + 1) % self.progress_every == 0
                or (marker_index + 1) == total_markers
            ):
                print(f"marker 处理进度: {marker_index + 1}/{total_markers} ({(marker_index + 1) / total_markers:.2%})")
            center = _bbox_center(marker_bbox)
            absolute_outer_bbox = _make_centered_bbox(center, window_side_um, window_side_um)
            local_outer_bbox = _make_centered_bbox((0.0, 0.0), window_side_um, window_side_um)
            absolute_outer_polygons = _approx_clip_indexed_elements_to_bbox(
                spatial_index,
                indexed_elements,
                absolute_outer_bbox,
                center_xy=center,
                max_elements=None,
            )
            outer_polygons = _translate_polygons(absolute_outer_polygons, (-float(center[0]), -float(center[1])))
            local_marker_bbox = _shift_bbox(marker_bbox, -float(center[0]), -float(center[1]))
            graph_invariants = _compute_graph_invariants(outer_polygons, local_outer_bbox)
            topology_features = _compute_topology_features(outer_polygons)
            graph_signature_grid, graph_signature_proj_x, graph_signature_proj_y = _window_occupancy_signature(
                outer_polygons,
                local_outer_bbox,
                n_bins=10,
                raster_grid_size=40,
            )
            graph_search_key = _coarse_signature_search_key(
                graph_signature_grid,
                graph_signature_proj_x,
                graph_signature_proj_y,
            )
            sample = LayoutWindowSample(
                sample_id=f"marker_{int(marker_index):06d}",
                source_name=str(source_name),
                center=(float(center[0]), float(center[1])),
                outer_bbox=tuple(float(v) for v in absolute_outer_bbox),
                duplicate_count=1,
            )
            records.append(
                _build_window_record(
                    sample,
                    outer_polygons,
                    local_outer_bbox=tuple(float(v) for v in local_outer_bbox),
                    graph_search_key=graph_search_key,
                    graph_invariants=graph_invariants,
                    topology_features=topology_features,
                    graph_signature_grid=graph_signature_grid,
                    graph_signature_proj_x=graph_signature_proj_x,
                    graph_signature_proj_y=graph_signature_proj_y,
                    marker_bbox=tuple(float(v) for v in local_marker_bbox),
                    absolute_marker_bbox=tuple(float(v) for v in marker_bbox),
                    marker_source=marker_source,
                )
            )
        metadata = {
            "source_name": str(source_name),
            "design_layer": f"{int(self.design_layer_spec[0])}/{int(self.design_layer_spec[1])}",
            "marker_layer": f"{int(self.marker_layer_spec[0])}/{int(self.marker_layer_spec[1])}",
            "hierarchy_expanded": True,
            "top_cell_names": list(top_cell_names),
            "expanded_design_geometry_count": int(len(design_geometries)),
            "expanded_marker_geometry_count": int(len(marker_geometries)),
            "layout_bbox": [float(v) for v in layout_bbox] if layout_bbox is not None else None,
            "raw_marker_geometry_count": int(len(raw_marker_bboxes)),
            "merged_marker_count": int(len(marker_bboxes)),
            "generated_item_count": int(len(records)),
            "pattern_radius_um": float(self.pattern_radius_um),
            "window_size_um": float(window_side_um),
            "marker_source": str(marker_source),
        }
        return records, metadata

    def load_files(self, input_path: str) -> List[PatternItem]:
        files = self._collect_input_files(input_path)
        if not files:
            raise FileNotFoundError(f"未找到任何 OAS 文件: {input_path}")
        self.source_files = list(files)
        self.pattern_items = []
        self.graph_items = []
        self.item_group_members = {}
        self.item_to_group = {}
        self.exact_group_info = {}
        per_file = []
        next_item_id = 0
        print(f"准备处理 {len(files)} 个 OAS 文件")
        for file_index, filepath in enumerate(files, start=1):
            source_name = os.path.basename(filepath)
            print(f"处理文件 {file_index}/{len(files)}: {source_name}")
            lib = _read_oas_only_library(filepath)
            records, metadata = self._build_marker_window_records(lib, source_name=source_name)
            per_file.append(dict(metadata))
            for marker_index, record in enumerate(records):
                self.pattern_items.append(
                    self._pattern_item_from_record(
                        record,
                        filepath,
                        source_name,
                        marker_index,
                        next_item_id,
                    )
                )
                next_item_id += 1
        total_items = len(self.pattern_items)
        self.pattern_generation_info = {
            "input_file_count": int(len(files)),
            "total_item_count": int(total_items),
            "design_layer": f"{int(self.design_layer_spec[0])}/{int(self.design_layer_spec[1])}",
            "marker_layer": f"{int(self.marker_layer_spec[0])}/{int(self.marker_layer_spec[1])}",
            "pattern_radius_um": float(self.pattern_radius_um),
            "window_size_um": float(self.pattern_radius_um * 2.0),
            "sources": per_file,
        }
        return list(self.pattern_items)

    def perform_clustering(self) -> List[FinalCluster]:
        if not self.pattern_items:
            self.final_clusters = []
            self.graph_items = []
            self.exact_group_info = {
                "enabled": True,
                "total_item_count": 0,
                "active_group_count": 0,
                "exact_group_compression_ratio": 0.0,
                "exact_group_reduction_factor": 0.0,
            }
            self.closed_loop_info = dict(self.clusterer.last_debug)
            return []
        graph_items = self._build_exact_group_items()
        print(
            f"SCP 聚类样本数: {len(self.pattern_items)}，"
            f"exact grouping 后 graph 节点数: {len(graph_items)}"
        )
        group_clusters = self.clusterer.cluster(graph_items)
        self.final_clusters = self._expand_group_clusters(group_clusters)
        self.closed_loop_info = dict(self.clusterer.last_debug)
        print(f"发现 {len(self.final_clusters)} 个 SCP 聚类")
        for idx, cluster in enumerate(self.final_clusters[:20]):
            print(f" 聚类 {idx}: {len(cluster.member_indices)} 个样本")
        if len(self.final_clusters) > 20:
            print(f" ... 其余 {len(self.final_clusters) - 20} 个聚类省略")
        return list(self.final_clusters)

    def _cluster_entry(self, cluster_id: int, cluster: FinalCluster, item_map: Dict[int, PatternItem]) -> Dict[str, Any]:
        member_ids = [int(v) for v in cluster.member_indices]
        rep_item = item_map[int(cluster.rep_id)]
        members = []
        for member_id in member_ids:
            item = item_map[int(member_id)]
            members.append({
                "item_id": int(item.item_id),
                "source_name": str(item.source_name),
                "center": [float(v) for v in _pattern_item_absolute_center(item)],
                "marker_bbox": item.sample_info.get("marker_bbox"),
            })
        alignment_results = []
        for result in getattr(cluster, "alignment_results", []) or []:
            alignment_results.append({
                "member_item_id": int(result.member_idx),
                "representative_item_id": int(result.rep_idx),
                "accepted": bool(result.accepted),
                "shift": [float(result.shift[0]), float(result.shift[1])],
                "shift_norm_um": float(result.shift_norm_um),
                "shifted_xor_ratio": float(result.shifted_xor_ratio),
                "shifted_signature_similarity": float(result.shifted_signature_similarity),
                "alignment_backend": str(result.alignment_backend),
                "constraint_mode": str(result.constraint_mode),
                "max_edge_displacement_um": float(result.max_edge_displacement_um),
                "mean_edge_displacement_um": float(result.mean_edge_displacement_um),
            })
        return {
            "cluster_id": int(cluster_id),
            "size": int(len(member_ids)),
            "representative_item_id": int(cluster.rep_id),
            "representative_source_name": str(rep_item.source_name),
            "representative_center": [float(v) for v in _pattern_item_absolute_center(rep_item)],
            "representative_marker_bbox": rep_item.sample_info.get("marker_bbox"),
            "member_item_ids": member_ids,
            "members": members,
            "alignment_results": alignment_results,
        }

    def get_results(self) -> Dict[str, Any]:
        item_map = {int(item.item_id): item for item in self.pattern_items}
        clusters = [
            self._cluster_entry(cluster_id, cluster, item_map)
            for cluster_id, cluster in enumerate(self.final_clusters)
        ]
        singleton_clusters = sum(1 for cluster in clusters if int(cluster["size"]) == 1)
        return {
            "summary": {
                "input_file_count": int(len(self.source_files)),
                "total_item_count": int(len(self.pattern_items)),
                "active_group_count": int(self.exact_group_info.get("active_group_count", len(self.graph_items))),
                "exact_group_compression_ratio": float(
                    self.exact_group_info.get("exact_group_compression_ratio", 1.0)
                ),
                "cluster_count": int(len(clusters)),
                "singleton_cluster_count": int(singleton_clusters),
                "workers": int(self.workers),
                "design_layer": f"{int(self.design_layer_spec[0])}/{int(self.design_layer_spec[1])}",
                "marker_layer": f"{int(self.marker_layer_spec[0])}/{int(self.marker_layer_spec[1])}",
                "pattern_radius_um": float(self.pattern_radius_um),
            },
            "pattern_generation": {
                **dict(self.pattern_generation_info),
                "exact_grouping": dict(self.exact_group_info),
            },
            "sparse_graph": dict(self.closed_loop_info.get("sparse_graph", {})),
            "alignment_refinement": dict(self.closed_loop_info.get("alignment_refinement", {})),
            "closed_loop": dict(self.closed_loop_info.get("closed_loop", {})),
            "clusters": clusters,
        }

    def save_results(self, result: Dict[str, Any], output_path: str, output_format: str = "json") -> None:
        out_path = _resolve_fs_path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fmt = str(output_format).strip().lower()
        if fmt == "txt":
            summary = result.get("summary", {})
            clusters = result.get("clusters", [])
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("Closed-Loop SCP Clustering Summary\n")
                f.write("=" * 40 + "\n")
                f.write(f"Input files: {summary.get('input_file_count', 0)}\n")
                f.write(f"Total items: {summary.get('total_item_count', 0)}\n")
                f.write(f"Active graph groups: {summary.get('active_group_count', 0)}\n")
                f.write(f"Exact group compression ratio: {summary.get('exact_group_compression_ratio', 0.0)}\n")
                f.write(f"Cluster count: {summary.get('cluster_count', 0)}\n")
                f.write(f"Singleton clusters: {summary.get('singleton_cluster_count', 0)}\n")
                f.write(f"Design layer: {summary.get('design_layer', '')}\n")
                f.write(f"Marker layer: {summary.get('marker_layer', '')}\n")
                f.write(f"Pattern radius (um): {summary.get('pattern_radius_um', 0.0)}\n\n")
                for cluster in clusters:
                    f.write(
                        f"Cluster {cluster['cluster_id']}: size={cluster['size']}, rep={cluster['representative_item_id']} ({cluster['representative_source_name']})\n"
                    )
            print(f"结果已保存到: {out_path}")
            return
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {out_path}")

    def run_pipeline(self, input_path: str) -> Dict[str, Any]:
        self.load_files(input_path)
        self.perform_clustering()
        return self.get_results()


def main():
    parser = argparse.ArgumentParser(
        description="Liu 2025 marker-driven closed-loop SCP layout clustering",
    )
    parser.add_argument("input", help="输入的 OAS 文件或目录")
    parser.add_argument("--output", "-o", default="clustering_results.json", help="输出文件路径")
    parser.add_argument("--format", "-f", choices=["json", "txt"], default="json", help="输出格式")
    parser.add_argument("--pattern", default="*.oas", help="目录输入时的文件匹配模式 (默认: *.oas)")
    parser.add_argument("--design-layer", default="1/0", help="design layer，格式 LAYER/DATATYPE (默认: 1/0)")
    parser.add_argument("--marker-layer", default="2/0", help="marker layer，格式 LAYER/DATATYPE (默认: 2/0)")
    parser.add_argument("--pattern-radius", type=float, default=1.35, help="pattern 半径，单位微米 (默认: 1.35)")
    parser.add_argument("--similarity-threshold", type=float, default=0.96, help="closed-loop 相似度阈值 (默认: 0.96)")
    parser.add_argument("--max-iterations", type=int, default=3, help="closed-loop 最大迭代轮数 (默认: 3)")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1), help="graph/refinement 并行 worker 数 (默认: CPU 核心数)")
    args = parser.parse_args()
    config = {
        "pattern": args.pattern,
        "design_layer": args.design_layer,
        "marker_layer": args.marker_layer,
        "pattern_radius_um": args.pattern_radius,
        "similarity_threshold": args.similarity_threshold,
        "max_iterations": args.max_iterations,
        "workers": args.workers,
    }
    pipeline = LayoutClusteringPipeline(config)
    result = pipeline.run_pipeline(args.input)
    pipeline.save_results(result, args.output, args.format)
    summary = result.get("summary", {})
    print("\n" + "=" * 60)
    print("Closed-loop SCP clustering completed")
    print(f" Input: {args.input}")
    print(f" Items: {summary.get('total_item_count', 0)}")
    print(f" Clusters: {summary.get('cluster_count', 0)}")
    print(f" Output: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
