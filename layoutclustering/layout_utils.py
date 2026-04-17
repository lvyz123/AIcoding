"""OAS I/O and marker raster utilities for optimized behavior clustering.

本模块是 `layout_clustering_optimized.py` 的底层工具集合，统一承载 OAS 读写、
中文/非 ASCII 路径桥接、marker-centered clip 构建、bitmap rasterization、
exact hash 去重和 sample/representative OAS 物化。

它不是聚类算法模块，不负责 representative selection、behavior verification 或输出
schema；这些仍由主脚本完成。本模块只保留当前 optimized 主线实际调用的底层能力。
"""

from __future__ import annotations

import hashlib
import math
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gdstk
import numpy as np


DEFAULT_PIXEL_SIZE_NM = 10
DEFAULT_OUTPUT_LAYER = 1
DEFAULT_OUTPUT_DATATYPE = 0


@dataclass
class MarkerRecord:
    """单个 marker 对应的 clip/bitmap 记录，供 optimized 主脚本聚类和导出使用。"""

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
    """clip hash + expanded hash 完全一致的 marker records 集合。"""

    exact_cluster_id: int
    representative: MarkerRecord
    members: List[MarkerRecord]

    @property
    def weight(self) -> int:
        """返回 exact duplicate 数量，optimized 主脚本用它作为 coverage 权重。"""
        return len(self.members)


@dataclass
class LayoutIndex:
    """单个 OAS 文件的轻量空间索引，保存 pattern bbox 数组和 marker polygons。"""

    indexed_elements: List[Dict[str, Any]]
    bbox_x0: np.ndarray
    bbox_y0: np.ndarray
    bbox_x1: np.ndarray
    bbox_y1: np.ndarray
    marker_polygons: List[gdstk.Polygon]


def _make_centered_bbox(center_xy, width_um, height_um):
    """创建以给定点为中心的边界框。"""
    cx, cy = float(center_xy[0]), float(center_xy[1])
    half_w = float(width_um) / 2.0
    half_h = float(height_um) / 2.0
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def _bbox_center(bbox):
    """计算边界框中心；非法 bbox 返回原点。"""
    if bbox is None or len(bbox) < 4:
        return (0.0, 0.0)
    x0, y0, x1, y1 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _safe_bbox_tuple(bbox):
    """把 gdstk bbox 或四元序列安全转换为 `(x0, y0, x1, y1)`。"""
    if bbox is None:
        return None
    try:
        if hasattr(bbox, "__len__") and len(bbox) == 2:
            lower = bbox[0]
            upper = bbox[1]
            if hasattr(lower, "__len__") and hasattr(upper, "__len__") and len(lower) >= 2 and len(upper) >= 2:
                return (float(lower[0]), float(lower[1]), float(upper[0]), float(upper[1]))
        if hasattr(bbox, "__len__") and len(bbox) >= 4:
            return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except (TypeError, ValueError):
        pass
    return None


def _is_ascii_path(filepath: str) -> bool:
    """检查路径是否可直接交给 gdstk 的 C 层文件接口。"""
    try:
        str(filepath).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _oas_bridge_path(prefix: str) -> Path:
    """为 gdstk 创建 ASCII-only 临时 OAS 路径，用于桥接中文目录读写。"""
    bridge_dir = Path(__file__).resolve().parent / "_oas_path_bridge"
    bridge_dir.mkdir(parents=True, exist_ok=True)
    return bridge_dir / f"{prefix}_{uuid.uuid4().hex}.oas"


def _read_oas_only_library(filepath: str):
    """读取 OASIS 文件，返回 gdstk.Library 对象。"""
    path = Path(str(filepath))
    try:
        if _is_ascii_path(str(path)):
            lib = gdstk.read_oas(str(path))
        else:
            temp_path = _oas_bridge_path("read")
            try:
                shutil.copy2(path, temp_path)
                lib = gdstk.read_oas(str(temp_path))
            finally:
                try:
                    temp_path.unlink()
                except OSError:
                    pass
    except Exception as exc:
        raise IOError(f"读取 OASIS 文件失败 {filepath}: {exc}") from exc
    if lib is None:
        raise IOError(f"无法解析 OASIS 文件 {filepath} (可能是空的或格式不正确)")
    return lib


def _write_oas_library(lib: gdstk.Library, filepath: str) -> None:
    """将 gdstk.Library 写入 OASIS 文件。"""
    path = Path(str(filepath))
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if _is_ascii_path(str(path)):
            lib.write_oas(str(path))
        else:
            temp_path = _oas_bridge_path("write")
            try:
                lib.write_oas(str(temp_path))
                shutil.copy2(temp_path, path)
            finally:
                try:
                    temp_path.unlink()
                except OSError:
                    pass
    except Exception as exc:
        raise IOError(f"写入 OASIS 文件失败 {filepath}: {exc}") from exc


def _element_layer_datatype(element) -> Tuple[int, int]:
    """获取 gdstk 元素的 layer/datatype。"""
    layer = getattr(element, "layer", 0)
    datatype = getattr(element, "datatype", 0)
    return int(layer), int(datatype)


def _polygon_vertices_array(polygon):
    """将 polygon 顶点转换为 float64 numpy 数组；无效 polygon 返回 None。"""
    if polygon is None or not hasattr(polygon, "points"):
        return None
    points = polygon.points
    if points is None or len(points) == 0:
        return None
    return np.asarray(points, dtype=np.float64)


def _make_output_polygons(
    bitmap: np.ndarray,
    bbox: Tuple[float, float, float, float],
    pixel_size_um: float,
    *,
    layer: int = DEFAULT_OUTPUT_LAYER,
    datatype: int = DEFAULT_OUTPUT_DATATYPE,
) -> List[gdstk.Polygon]:
    """把 bool bitmap 按行程压缩为矩形 polygons，用于写出 sample/representative OAS。"""
    height, width = bitmap.shape
    if height == 0 or width == 0:
        return []

    x0, y0, _, _ = bbox
    polygons: List[gdstk.Polygon] = []
    mask = np.asarray(bitmap, dtype=bool)
    for row in range(height):
        col = 0
        while col < width:
            if not mask[row, col]:
                col += 1
                continue
            run_start = col
            while col < width and mask[row, col]:
                col += 1
            run_end = col
            rect_x0 = float(x0) + run_start * float(pixel_size_um)
            rect_x1 = float(x0) + run_end * float(pixel_size_um)
            rect_y0 = float(y0) + row * float(pixel_size_um)
            rect_y1 = float(y0) + (row + 1) * float(pixel_size_um)
            polygons.append(
                gdstk.rectangle(
                    (rect_x0, rect_y0),
                    (rect_x1, rect_y1),
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
    """把单个 clip bitmap 写成 OAS 文件，返回输出路径字符串。"""
    lib = gdstk.Library()
    cell = gdstk.Cell(str(sample_id))
    polygons = _make_output_polygons(bitmap, bbox, pixel_size_um)
    if polygons:
        cell.add(*polygons)
    lib.add(cell)
    _write_oas_library(lib, str(output_path))
    return str(output_path)


def _bitmap_transforms(bitmap: np.ndarray) -> Tuple[np.ndarray, ...]:
    """生成 bitmap 的 8 种旋转/翻转等价形态，用于 canonical exact hash。"""
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
    """对 bitmap 做 8 向对称归一化，生成稳定 payload。"""
    if bitmap.size == 0 or not np.any(bitmap):
        return b"empty"

    payloads: List[bytes] = []
    for transformed in _bitmap_transforms(bitmap):
        contig = np.ascontiguousarray(transformed.astype(np.uint8, copy=False))
        packed = np.packbits(contig.reshape(-1))
        payloads.append(f"{contig.shape[0]}x{contig.shape[1]}:".encode("ascii") + packed.tobytes())
    return min(payloads)


def _canonical_bitmap_hash(bitmap: np.ndarray) -> Tuple[str, bytes]:
    """返回 bitmap canonical payload 的 SHA-256 hash 和 payload 本体。"""
    payload = _canonical_bitmap_payload(bitmap)
    return hashlib.sha256(payload).hexdigest(), payload


def _parse_layer_spec(layer_spec: str) -> Tuple[int, int]:
    """解析 `layer/datatype` 字符串，返回整数 layer/datatype 元组。"""
    try:
        layer_str, datatype_str = str(layer_spec).split("/", 1)
        return int(layer_str.strip()), int(datatype_str.strip())
    except Exception as exc:
        raise ValueError(f"Invalid marker layer '{layer_spec}', expected '<layer>/<datatype>'") from exc


def _make_sample_filename(prefix: str, source_name: str, index_value: int) -> str:
    """生成 ASCII-safe 的 sample/representative OAS 文件名。"""
    stem = "".join(ch if ch.isascii() and (ch.isalnum() or ch in "-_") else "_" for ch in str(source_name))
    stem = stem.strip("._-") or "layout"
    return f"{prefix}_{stem}_{int(index_value):06d}.oas"


def _window_pixels(window_um: float, pixel_size_um: float) -> int:
    """把物理窗口尺寸转换成 raster bitmap 像素数，至少返回 1。"""
    return max(1, int(math.ceil(float(window_um) / max(float(pixel_size_um), 1e-12) - 1e-12)))


def _raster_window_spec(
    marker_bbox: Tuple[float, float, float, float],
    marker_center: Tuple[float, float],
    clip_size_um: float,
    pixel_size_um: float,
) -> Dict[str, Any]:
    """根据 marker 位置生成 clip / expanded window 的物理坐标和像素坐标。"""
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


def _query_candidate_ids(layout_index: LayoutIndex, bbox: Tuple[float, float, float, float]) -> List[int]:
    """查询 bbox 相交的 pattern 图元编号，供 marker window rasterization 使用。"""
    bbox_x0, bbox_y0, bbox_x1, bbox_y1 = bbox
    mask = (
        (layout_index.bbox_x1 > float(bbox_x0))
        & (layout_index.bbox_x0 < float(bbox_x1))
        & (layout_index.bbox_y1 > float(bbox_y0))
        & (layout_index.bbox_y0 < float(bbox_y1))
    )
    return [int(value) for value in np.flatnonzero(mask).tolist()]


def _polygon_strip_spans(
    points: np.ndarray,
    bbox: Tuple[float, float, float, float],
) -> List[Tuple[float, float, float, float]]:
    """把 polygon 按水平 strip 分解为 spans，便于直接 rasterize 到 bitmap。"""
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
    """把候选 pattern polygons rasterize 到指定 bbox/shape 的 bool bitmap 中。"""
    height, width = int(shape[0]), int(shape[1])
    bitmap = np.zeros((height, width), dtype=bool)
    if height <= 0 or width <= 0:
        return bitmap

    bbox_x0, bbox_y0 = float(bbox[0]), float(bbox[1])
    for elem_id in candidate_ids:
        polygon = indexed_elements[int(elem_id)]["element"]
        points = _polygon_vertices_array(polygon)
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
    """按量化 bbox 从 expanded bitmap 中切出 clip bitmap。"""
    x0, y0, x1, y1 = bbox_q
    return np.ascontiguousarray(bitmap[y0:y1, x0:x1], dtype=bool)


class MarkerRasterBuilder:
    """OAS -> marker bitmap records 的构建器，不承担聚类或 representative selection。"""

    def __init__(self, *, config: Dict[str, Any], temp_dir: Path, layer_processor: Optional[Any] = None):
        """初始化 marker layer、clip size、像素尺寸和可选 layer operation processor。"""
        self.config = dict(config)
        self.temp_dir = Path(temp_dir)
        self.layer_processor = layer_processor
        self.clip_size_um = float(self.config.get("clip_size_um", 1.35))
        self.hotspot_layer = self.config.get("hotspot_layer")
        self.apply_layer_operations = bool(self.config.get("apply_layer_operations", False))
        self.pixel_size_nm = int(self.config.get("pixel_size_nm", DEFAULT_PIXEL_SIZE_NM))
        self.pixel_size_um = float(self.pixel_size_nm) / 1000.0

        if not self.hotspot_layer:
            raise ValueError("optimized marker raster builder requires a marker/hotspot layer")
        if self.pixel_size_um <= 0.0:
            raise ValueError("pixel_size_nm must be positive")
        self.hotspot_layer_tuple = _parse_layer_spec(str(self.hotspot_layer))

    def _discover_input_files(self, input_path: str) -> List[Path]:
        """发现输入路径中的 OAS 文件；文件路径直接返回，目录只收集一层 `*.oas`。"""
        path = Path(input_path)
        if path.is_file():
            return [path]
        return sorted([item for item in path.glob("*.oas") if item.is_file()])

    def _prepare_layout(self, filepath: Path) -> LayoutIndex:
        """读取并 flatten 布局，把 pattern 与 marker 图形拆分出来，并建立 bbox 索引。"""
        lib = _read_oas_only_library(str(filepath))
        if self.apply_layer_operations and self.layer_processor is not None:
            lib = self.layer_processor.apply_layer_operations(lib)

        top_cells = list(lib.top_level()) or list(lib.cells)
        pattern_polygons: List[gdstk.Polygon] = []
        marker_polygons: List[gdstk.Polygon] = []
        for top_cell in top_cells:
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
        """围绕单个 marker 构建 bitmap record，并生成 exact clustering 所需 hash。"""
        marker_bbox = _safe_bbox_tuple(marker_poly.bounding_box())
        if marker_bbox is None:
            return None

        marker_center = _bbox_center(marker_bbox)
        raster_spec = _raster_window_spec(marker_bbox, marker_center, self.clip_size_um, self.pixel_size_um)
        expanded_bbox = raster_spec["expanded_bbox"]
        candidate_ids = _query_candidate_ids(layout_index, expanded_bbox)
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

    def _group_exact_clusters(self, marker_records: Sequence[MarkerRecord]) -> List[ExactCluster]:
        """用 clip hash + expanded hash 合并完全重复窗口，并回填 exact_cluster_id。"""
        buckets: Dict[Tuple[str, str], List[MarkerRecord]] = {}
        for record in marker_records:
            buckets.setdefault((record.clip_hash, record.expanded_hash), []).append(record)

        exact_clusters: List[ExactCluster] = []
        for cluster_id, members in enumerate(sorted(buckets.values(), key=lambda items: (items[0].source_name, items[0].marker_id))):
            members_sorted = sorted(members, key=lambda item: (item.source_name, item.marker_id))
            for member in members_sorted:
                member.exact_cluster_id = int(cluster_id)
            exact_clusters.append(
                ExactCluster(
                    exact_cluster_id=int(cluster_id),
                    representative=members_sorted[0],
                    members=members_sorted,
                )
            )
        return exact_clusters
