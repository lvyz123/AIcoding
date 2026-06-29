#!/usr/bin/env python3
"""hotspotdetection 的轻量版图工具层。

本文件位于 recipe selector 的最底层，只承担四类职责：
1. 读取 OAS/OASIS，并按 marker layer 分离 hotspot marker 与普通 pattern 图元。
2. 围绕 marker 或任意中心点生成固定窗口 bitmap，供 MP/AF/AP/care-area 复用。
3. 提供 bbox 空间查询、bitmap fingerprint 和 review OAS 写出。
4. 保持实现确定、轻量，不在这里引入 detector、训练逻辑或复杂数据库。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import gdstk
import numpy as np


DEFAULT_PIXEL_SIZE_NM = 10


@dataclass
class LayoutIndex:
    """保存已展开 layout 的轻量空间索引。"""

    indexed_elements: List[Dict[str, Any]]
    bbox_x0: np.ndarray
    bbox_y0: np.ndarray
    bbox_x1: np.ndarray
    bbox_y1: np.ndarray
    marker_polygons: List[gdstk.Polygon]


@dataclass
class MarkerRecord:
    """记录一个 hotspot marker 对应的裁剪窗口和 provenance。"""

    marker_id: str
    source_path: str
    marker_index: int
    marker_center: Tuple[float, float]
    clip_bbox: Tuple[float, float, float, float]
    clip_bitmap: np.ndarray

    def to_metadata(self) -> Dict[str, Any]:
        """转换为 backend / recipe selector 使用的 JSON 友好 metadata。"""
        return {
            "marker_id": str(self.marker_id),
            "source_path": str(self.source_path),
            "marker_index": int(self.marker_index),
            "marker_center": [float(self.marker_center[0]), float(self.marker_center[1])],
            "clip_bbox": [float(value) for value in self.clip_bbox],
        }


def _parse_layer_spec(value: str | Sequence[int] | Tuple[int, int]) -> Tuple[int, int]:
    """解析 layer/datatype 字符串或二元组。"""
    if isinstance(value, str):
        parts = re.split(r"[/,:]", value.strip())
        if len(parts) != 2:
            raise ValueError(f"Layer spec must be layer/datatype: {value}")
        return int(parts[0]), int(parts[1])
    if len(value) != 2:
        raise ValueError(f"Layer spec must have two integers: {value}")
    return int(value[0]), int(value[1])


def _element_layer_datatype(element: Any) -> Tuple[int, int]:
    """读取 gdstk 图元的 layer/datatype。"""
    return int(getattr(element, "layer", 0)), int(getattr(element, "datatype", 0))


def _safe_bbox_tuple(element: Any) -> Tuple[float, float, float, float] | None:
    """读取图元 bbox；无效 bbox 返回 None。"""
    bbox = element.bounding_box()
    if bbox is None:
        return None
    (x0, y0), (x1, y1) = bbox
    if not all(math.isfinite(float(value)) for value in (x0, y0, x1, y1)):
        return None
    if float(x1) <= float(x0) or float(y1) <= float(y0):
        return None
    return float(x0), float(y0), float(x1), float(y1)


def _bbox_center(bbox: Sequence[float]) -> Tuple[float, float]:
    """计算 bbox 中心点。"""
    return 0.5 * (float(bbox[0]) + float(bbox[2])), 0.5 * (float(bbox[1]) + float(bbox[3]))


def _query_candidate_ids(layout_index: LayoutIndex, bbox: Sequence[float]) -> np.ndarray:
    """返回与查询 bbox 相交的 pattern 图元索引。"""
    if len(layout_index.indexed_elements) == 0:
        return np.zeros((0,), dtype=np.int64)
    x0, y0, x1, y1 = (float(value) for value in bbox)
    mask = (
        (layout_index.bbox_x1 >= x0)
        & (layout_index.bbox_x0 <= x1)
        & (layout_index.bbox_y1 >= y0)
        & (layout_index.bbox_y0 <= y1)
    )
    return np.nonzero(mask)[0].astype(np.int64)


def _raster_window_spec(center_xy: Tuple[float, float], window_size_um: float, pixel_size_um: float) -> Dict[str, Any]:
    """生成中心窗口的 bbox、shape 和像素尺寸。"""
    size = float(window_size_um)
    pixel = max(float(pixel_size_um), 1e-6)
    width = max(1, int(math.ceil(size / pixel)))
    half = 0.5 * size
    cx, cy = float(center_xy[0]), float(center_xy[1])
    return {
        "center": (cx, cy),
        "clip_bbox": (cx - half, cy - half, cx + half, cy + half),
        "shape": (width, width),
        "pixel_size_um": pixel,
    }


def _fill_bitmap_from_elements(bitmap: np.ndarray, clip_bbox: Sequence[float], elements: Sequence[Mapping[str, Any]]) -> None:
    """用图元 bbox 近似填充窗口 bitmap。"""
    x0, y0, x1, y1 = (float(value) for value in clip_bbox)
    height, width = bitmap.shape
    sx = float(width) / max(1e-12, x1 - x0)
    sy = float(height) / max(1e-12, y1 - y0)
    for item in elements:
        bx0, by0, bx1, by1 = (float(value) for value in item["bbox"])
        ix0 = max(0, min(width, int(math.floor((bx0 - x0) * sx))))
        ix1 = max(0, min(width, int(math.ceil((bx1 - x0) * sx))))
        iy0 = max(0, min(height, int(math.floor((by0 - y0) * sy))))
        iy1 = max(0, min(height, int(math.ceil((by1 - y0) * sy))))
        if ix1 > ix0 and iy1 > iy0:
            bitmap[iy0:iy1, ix0:ix1] = True


def rasterize_centered_window(
    layout_index: LayoutIndex,
    center_xy: Tuple[float, float],
    window_size_um: float,
    pixel_size_um: float,
) -> Dict[str, Any]:
    """按任意中心点 rasterize 一个局部窗口，供 MP/AF/AP/care-area 复用。"""
    spec = _raster_window_spec(center_xy, window_size_um, pixel_size_um)
    clip_bbox = spec["clip_bbox"]
    candidate_ids = _query_candidate_ids(layout_index, clip_bbox)
    elements = [layout_index.indexed_elements[int(index)] for index in candidate_ids]
    bitmap = np.zeros(spec["shape"], dtype=bool)
    _fill_bitmap_from_elements(bitmap, clip_bbox, elements)
    return {
        "center": (float(center_xy[0]), float(center_xy[1])),
        "clip_bbox": [float(value) for value in clip_bbox],
        "clip_bitmap": np.ascontiguousarray(bitmap, dtype=bool),
        "pixel_size_um": float(pixel_size_um),
        "element_count": int(len(elements)),
    }


def bitmap_fingerprint(bitmap: np.ndarray, grid_size: int = 8) -> np.ndarray:
    """把 bitmap 压缩为 L2-normalized grid fingerprint。"""
    arr = np.asarray(bitmap, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((grid_size * grid_size + 4,), dtype=np.float32)
    h, w = arr.shape[:2]
    ys = np.linspace(0, h, grid_size + 1, dtype=int)
    xs = np.linspace(0, w, grid_size + 1, dtype=int)
    values: List[float] = []
    for iy in range(grid_size):
        for ix in range(grid_size):
            block = arr[ys[iy] : ys[iy + 1], xs[ix] : xs[ix + 1]]
            values.append(float(np.mean(block)) if block.size else 0.0)
    density = float(np.mean(arr))
    edge_x = float(np.mean(np.abs(arr[:, 1:] - arr[:, :-1]))) if w > 1 else 0.0
    edge_y = float(np.mean(np.abs(arr[1:, :] - arr[:-1, :]))) if h > 1 else 0.0
    aspect = float(w) / float(max(1, h))
    vec = np.asarray(values + [density, edge_x, edge_y, min(4.0, aspect) / 4.0], dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-12 else vec


def _write_oas_library(lib: gdstk.Library, filepath: str) -> None:
    """写出 OAS；调用方负责保证路径语义。"""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    lib.write_oas(str(path))


def _materialize_clip_bitmap(
    bitmap: np.ndarray,
    clip_bbox: Sequence[float],
    sample_id: str | None = None,
    output_path: str | Path | None = None,
    pixel_size_um: float | None = None,
    *,
    layer: int = 1,
    datatype: int = 0,
) -> str:
    """把 bitmap 物化为 review OAS；兼容 selector 的旧位置参数调用。"""
    del pixel_size_um
    if output_path is None:
        raise ValueError("output_path is required")
    arr = np.asarray(bitmap, dtype=bool)
    x0, y0, x1, y1 = (float(value) for value in clip_bbox)
    height, width = arr.shape if arr.ndim == 2 else (0, 0)
    lib = gdstk.Library()
    cell_name = re.sub(r"[^A-Za-z0-9_]+", "_", str(sample_id or "CLIP"))[:60] or "CLIP"
    cell = gdstk.Cell(cell_name)
    if height > 0 and width > 0:
        px = (x1 - x0) / float(width)
        py = (y1 - y0) / float(height)
        for iy, ix in np.argwhere(arr):
            cell.add(
                gdstk.rectangle(
                    (x0 + ix * px, y0 + iy * py),
                    (x0 + (ix + 1) * px, y0 + (iy + 1) * py),
                    layer=int(layer),
                    datatype=int(datatype),
                )
            )
    lib.add(cell)
    _write_oas_library(lib, str(output_path))
    return str(output_path)


class MarkerRasterBuilder:
    """recipe backend / preprocess 共用的 marker raster 构建器。"""

    def __init__(
        self,
        config: Mapping[str, Any],
        temp_dir: str | Path,
        *,
        layer_processor: Any | None = None,
        recursive_input: bool = False,
    ) -> None:
        self.config = dict(config)
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.marker_layer = _parse_layer_spec(str(self.config.get("hotspot_layer", self.config.get("marker_layer", "999/0"))))
        self.clip_size_um = float(self.config.get("clip_size_um", self.config.get("clip_size", 1.35)))
        self.pixel_size_um = float(self.config.get("pixel_size_nm", DEFAULT_PIXEL_SIZE_NM)) / 1000.0
        self.layer_processor = layer_processor
        self.recursive_input = bool(self.config.get("recursive_input", recursive_input))

    def _discover_input_files(self, input_path: str | Path) -> List[Path]:
        """发现输入 OAS/OASIS 文件。"""
        path = Path(input_path)
        if path.is_file():
            return [path]
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Input path does not exist: {path}")
        iterator = path.rglob("*") if self.recursive_input else path.glob("*")
        return sorted([item for item in iterator if item.is_file() and item.suffix.lower() in {".oas", ".oasis"}])

    def _prepare_layout(self, filepath: str | Path) -> LayoutIndex:
        """读取 OAS，并分离 marker polygons 与普通 pattern elements。"""
        lib = gdstk.read_oas(str(filepath))
        if self.layer_processor is not None and bool(self.config.get("apply_layer_operations", False)):
            lib = self.layer_processor.apply_layer_operations(lib)
        marker_polygons: List[gdstk.Polygon] = []
        indexed: List[Dict[str, Any]] = []
        for cell in lib.top_level():
            try:
                cell.flatten()
            except Exception:
                pass
            for poly in cell.polygons:
                bbox = _safe_bbox_tuple(poly)
                if bbox is None:
                    continue
                layer_spec = _element_layer_datatype(poly)
                if layer_spec == self.marker_layer:
                    marker_polygons.append(poly)
                    continue
                if self.layer_processor is not None and not self.layer_processor.should_keep_pattern_layer(layer_spec):
                    continue
                indexed.append({"bbox": bbox, "element": poly, "layer": layer_spec[0], "datatype": layer_spec[1]})
        return LayoutIndex(
            indexed_elements=indexed,
            bbox_x0=np.asarray([item["bbox"][0] for item in indexed], dtype=np.float64),
            bbox_y0=np.asarray([item["bbox"][1] for item in indexed], dtype=np.float64),
            bbox_x1=np.asarray([item["bbox"][2] for item in indexed], dtype=np.float64),
            bbox_y1=np.asarray([item["bbox"][3] for item in indexed], dtype=np.float64),
            marker_polygons=marker_polygons,
        )

    def _build_marker_record(
        self,
        filepath: str | Path,
        marker_index: int,
        marker_poly: gdstk.Polygon,
        layout_index: LayoutIndex,
    ) -> MarkerRecord | None:
        """围绕单个 marker 生成稳定 marker record。"""
        bbox = _safe_bbox_tuple(marker_poly)
        if bbox is None:
            return None
        center = _bbox_center(bbox)
        window = rasterize_centered_window(layout_index, center, self.clip_size_um, self.pixel_size_um)
        source = Path(filepath)
        marker_id = f"{source.stem}__marker_{int(marker_index):06d}"
        return MarkerRecord(
            marker_id=marker_id,
            source_path=str(source),
            marker_index=int(marker_index),
            marker_center=(float(center[0]), float(center[1])),
            clip_bbox=tuple(float(value) for value in window["clip_bbox"]),
            clip_bitmap=np.asarray(window["clip_bitmap"], dtype=bool),
        )

    def build_records(self, input_path: str | Path) -> List[MarkerRecord]:
        """读取输入 OAS/OASIS，并为 marker layer 上的每个 marker 生成裁剪记录。"""
        records: List[MarkerRecord] = []
        for filepath in self._discover_input_files(input_path):
            layout_index = self._prepare_layout(filepath)
            for marker_index, marker_poly in enumerate(layout_index.marker_polygons):
                record = self._build_marker_record(filepath, marker_index, marker_poly, layout_index)
                if record is not None:
                    records.append(record)
        return records
