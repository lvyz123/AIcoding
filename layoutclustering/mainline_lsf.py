#!/usr/bin/env python3
"""LSF 版本的独立 raster-first 聚类核心。

中文整体算法流程与原理介绍：
1. 本文件承载 optimized_v2_lsf 的核心数据结构、raster、exact hash、candidate 生成、coverage、
   set cover 和 final verification 逻辑。代码不 import optimized_v1 或旧 mainline，但主算法语义对齐
   optimized_v1：geometry-driven seed、exact hash、systematic shift candidate、
   candidate bundle coverage、greedy set cover、final verification。
2. prepare 阶段读取 OAS，应用可选 layer operation，按重复阵列、长条图形和残余局部图形生成代表 seed。
   run-shard 只把 seed raster 成 marker records，输出 JSON metadata 与 NPZ bitmap，方便 LSF job 分发。
3. prepare-coverage 汇总 marker records，生成全局 exact clusters，并把所有 shift candidates 按 shape 和
   严格 bitmap key 合成 candidate bundle bucket。这个阶段只写轻量索引和代表 bitmap，不提前构建
   full GraphDescriptor 或 ECC geometry cache。
4. run-coverage-shard 处理一段 source exact clusters，读取对应 shape 的全局 candidate bundle bucket，
   执行 v1 风格 coverage：exact hash direct、cheap shortlist、lazy full GraphDescriptor prefilter、
   packed/dilated/donut 懒 geometry cache，并把 coverage set 压成 CSR offsets/values。
5. merge-coverage 汇总 CSR coverage，用 lazy greedy set cover 选择代表 candidate，再只为 selected
   candidates 懒加载 bitmap，并为 target exact cluster 临时生成 shift-witness candidates 做最终验证，
   避免在汇总阶段长期持有全部 candidate Python 对象。
6. 代码保持 Python 3.6 兼容：不使用 dataclasses、from __future__ annotations、现代 union type、
   内置泛型类型标注或 scipy.optimize.milp。
"""

import csv
import hashlib
import heapq
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import gdstk
import numpy as np
from scipy import ndimage

try:
    import hnswlib
except ImportError:
    hnswlib = None

from layout_utils_lsf import (
    _bbox_center,
    _bbox_intersection,
    _element_layer_datatype,
    _make_centered_bbox,
    _polygon_vertices_array,
    _read_oas_only_library,
    _safe_bbox_tuple,
    _write_oas_library,
)
from layer_operations_lsf import LayerOperationProcessor


PIPELINE_MODE = "optimized_v2_lsf"
SEED_MODE = "geometry_driven_shift_lsf"
GRID_STEP_RATIO = 0.5
GRID_BUCKET_QUANT_UM = 0.08
PATTERN_COVERAGE_COORD_QUANT_UM = 1e-6
PATTERN_TYPE_COVERAGE_THRESHOLD = 0.95
GRID_MAX_DESCRIPTOR_NEIGHBORS = 256
DEFAULT_PIXEL_SIZE_NM = 10
SEED_TYPE_ARRAY = "array_representative"
SEED_TYPE_ARRAY_SPACE = "array_spacing"
SEED_TYPE_LONG = "long_shape_path"
SEED_TYPE_RESIDUAL = "residual_local_grid"
ECC_DONUT_OVERLAP_RATIO = 0.20
ECC_RESIDUAL_RATIO = 1e-3
ECC_ALLOW_DONUT_AUTO_PASS = False
BIT_COUNT_TABLE = np.array([bin(value).count("1") for value in range(256)], dtype=np.uint8)
COVERAGE_CHUNK_BYTE_BUDGET = 8 * 1024 * 1024
COVERAGE_SHORTLIST_MAX_TARGETS = 64
COVERAGE_EXACT_SHORTLIST_MAX_GROUPS = 512
GRAPH_INVARIANT_LIMIT = 0.22
GRAPH_TOPOLOGY_THRESHOLD = 6.5
GRAPH_SIGNATURE_THRESHOLD = 0.74
STRICT_INVARIANT_LIMIT = 0.20
STRICT_TOPOLOGY_THRESHOLD = 3.0
STRICT_SIGNATURE_THRESHOLD = 0.84
CHEAP_FILL_ABS_LIMIT = 0.12
CHEAP_AREA_DENSITY_ABS_LIMIT = 0.18
COVERAGE_FULL_PREFILTER_MIN_PROBE_PAIRS = 512
COVERAGE_FULL_PREFILTER_MIN_REJECT_RATE = 0.02
CANDIDATE_BUNDLE_SPLIT_MIN_GROUPS = 64
CANDIDATE_BUNDLE_FILL_BIN_WIDTH = 0.04
DIAGONAL_SHIFT_AXIS_MAX_COUNT = 3
DIAGONAL_SHIFT_MAX_COUNT = 2
QUALITY_PER_CLUSTER_INTRA_PAIR_LIMIT = 200
QUALITY_LOW_VISUAL_PURITY_THRESHOLD = 0.80
LOW_SHIFT_MAX_UM = 0.20
LOW_PAIRWISE_REVIEW_THRESHOLD = 0.50
OVERMERGE_EXACT_COUNT_TRIGGER = 500
OVERMERGE_WEIGHT_RATIO_TRIGGER = 0.02
OVERMERGE_LARGE_SHIFT_RATIO = 0.20
OVERMERGE_LARGE_SHIFT_EXACT_TRIGGER = 50
REVIEW_MERGE_CANDIDATE_TOP_N = 5000
REVIEW_MERGE_CANDIDATE_TIER_QUOTAS = {"high": 3000, "medium": 1000, "low": 1000}
REVIEW_MERGE_HIGH_SHIFT_DISTANCE_UM = 0.10
REVIEW_MERGE_MEDIUM_SHIFT_DISTANCE_UM = 0.20
SAFE_RECALL_MERGE_TOP_PAIR_N = 200
SAFE_RECALL_MERGE_MAX_UNION_EXACT_COUNT = 2000
GREEDY_PURITY_GATE_EXACT_TRIGGER = 500
GREEDY_PURITY_GATE_WEIGHT_RATIO = 0.005
GREEDY_PURITY_GATE_LARGE_SHIFT_RATIO = 0.15
GREEDY_PURITY_GATE_LARGE_SHIFT_EXACT_TRIGGER = 20
GREEDY_PURITY_GATE_STAGE1_SAMPLE_PAIRS = 32
GREEDY_PURITY_GATE_STAGE1_PASS_FAIL_RATE = 0.20
GREEDY_PURITY_GATE_STAGE1_REJECT_FAIL_RATE = 0.80
GREEDY_PURITY_GATE_STAGE2_SAMPLE_PAIRS = 100
GREEDY_PURITY_GATE_STAGE2_REJECT_FAIL_RATE = 0.50
LONG_SHAPE_CROSS_SEED_SIGNATURE_THRESHOLD = 0.90
GEOMETRY_REJECT_REASON_ORDER = (
    "geometry_shape_mismatch",
    "geometry_empty_mismatch",
    "geometry_exact_mismatch",
    "geometry_acc_xor",
    "geometry_residual_candidate",
    "geometry_residual_target",
    "geometry_donut_overlap",
)
_POOL_EDGE_CACHE = {}
_COVERAGE_STRUCTURE_CACHE = {}


class GraphDescriptor(object):
    """v1 风格 full prefilter 使用的 bitmap 图形描述符。"""

    __slots__ = ("invariants", "topology", "signature_grid", "signature_proj_x", "signature_proj_y")

    def __init__(self, invariants, topology, signature_grid, signature_proj_x, signature_proj_y):
        self.invariants = np.asarray(invariants, dtype=np.float64)
        self.topology = np.asarray(topology, dtype=np.float64)
        self.signature_grid = np.asarray(signature_grid, dtype=np.float32)
        self.signature_proj_x = np.asarray(signature_proj_x, dtype=np.float32)
        self.signature_proj_y = np.asarray(signature_proj_y, dtype=np.float32)


class CheapDescriptor(object):
    """v1 coverage shortlist 使用的低成本 bitmap 描述符。"""

    __slots__ = ("invariants", "signature_grid", "signature_proj_x", "signature_proj_y", "area_px")

    def __init__(self, invariants, signature_grid, signature_proj_x, signature_proj_y, area_px):
        self.invariants = np.asarray(invariants, dtype=np.float32)
        self.signature_grid = np.asarray(signature_grid, dtype=np.float32)
        self.signature_proj_x = np.asarray(signature_proj_x, dtype=np.float32)
        self.signature_proj_y = np.asarray(signature_proj_y, dtype=np.float32)
        self.area_px = int(area_px)


class SimpleSpatialIndex(object):
    """用简单 bbox 数组提供 intersection 查询，避免 LSF 环境额外依赖 rtree。"""

    __slots__ = (
        "bboxes",
        "bbox_x0",
        "bbox_y0",
        "bbox_x1",
        "bbox_y1",
        "layout_x0",
        "layout_y0",
        "layout_x1",
        "layout_y1",
        "bin_size_x",
        "bin_size_y",
        "bin_count_x",
        "bin_count_y",
        "bins",
    )

    def __init__(self, bboxes):
        self.bboxes = list(bboxes)
        self.bbox_x0 = np.asarray([bbox[0] for bbox in self.bboxes], dtype=np.float64)
        self.bbox_y0 = np.asarray([bbox[1] for bbox in self.bboxes], dtype=np.float64)
        self.bbox_x1 = np.asarray([bbox[2] for bbox in self.bboxes], dtype=np.float64)
        self.bbox_y1 = np.asarray([bbox[3] for bbox in self.bboxes], dtype=np.float64)
        if self.bbox_x0.size == 0:
            self.layout_x0 = self.layout_y0 = self.layout_x1 = self.layout_y1 = 0.0
            self.bin_size_x = self.bin_size_y = 1.0
            self.bin_count_x = self.bin_count_y = 1
            self.bins = {}
            return
        self.layout_x0 = float(np.min(self.bbox_x0))
        self.layout_y0 = float(np.min(self.bbox_y0))
        self.layout_x1 = float(np.max(self.bbox_x1))
        self.layout_y1 = float(np.max(self.bbox_y1))
        width = max(0.0, self.layout_x1 - self.layout_x0)
        height = max(0.0, self.layout_y1 - self.layout_y0)
        axis_count = int(min(192, max(16, math.ceil(math.sqrt(float(len(self.bboxes))) * 1.5))))
        self.bin_count_x = axis_count if width > 0.0 else 1
        self.bin_count_y = axis_count if height > 0.0 else 1
        self.bin_size_x = width / float(self.bin_count_x) if width > 0.0 else 1.0
        self.bin_size_y = height / float(self.bin_count_y) if height > 0.0 else 1.0
        self.bins = {}
        for index, bbox in enumerate(self.bboxes):
            ix0, ix1 = self._bin_range(float(bbox[0]), float(bbox[2]), self.layout_x0, self.bin_size_x, self.bin_count_x)
            iy0, iy1 = self._bin_range(float(bbox[1]), float(bbox[3]), self.layout_y0, self.bin_size_y, self.bin_count_y)
            for ix in range(ix0, ix1 + 1):
                for iy in range(iy0, iy1 + 1):
                    key = int(iy) * int(self.bin_count_x) + int(ix)
                    bucket = self.bins.get(key)
                    if bucket is None:
                        self.bins[key] = [int(index)]
                    else:
                        bucket.append(int(index))

    @staticmethod
    def _bin_range(start, end, origin, bin_size, bin_count):
        """把 bbox 坐标范围映射到空间索引 bin 范围。"""

        if bin_count <= 1:
            return 0, 0
        first = int(math.floor((float(start) - float(origin)) / float(bin_size)))
        last = int(math.floor((float(end) - float(origin)) / float(bin_size)))
        first = max(0, min(int(bin_count) - 1, first))
        last = max(0, min(int(bin_count) - 1, last))
        if last < first:
            return last, first
        return first, last

    def intersection(self, bbox):
        """用固定网格 bin 缩小候选，再做精确 bbox 相交过滤。"""

        x0, y0, x1, y1 = [float(v) for v in bbox]
        if self.bbox_x0.size == 0:
            return np.asarray([], dtype=np.int64)
        if x1 <= self.layout_x0 or x0 >= self.layout_x1 or y1 <= self.layout_y0 or y0 >= self.layout_y1:
            return np.asarray([], dtype=np.int64)
        ix0, ix1 = self._bin_range(x0, x1, self.layout_x0, self.bin_size_x, self.bin_count_x)
        iy0, iy1 = self._bin_range(y0, y1, self.layout_y0, self.bin_size_y, self.bin_count_y)
        candidates = set()
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                bucket = self.bins.get(int(iy) * int(self.bin_count_x) + int(ix))
                if bucket:
                    candidates.update(bucket)
        if not candidates:
            return np.asarray([], dtype=np.int64)
        candidate_ids = np.asarray(sorted(candidates), dtype=np.int64)
        mask = (
            (self.bbox_x1[candidate_ids] > x0)
            & (self.bbox_x0[candidate_ids] < x1)
            & (self.bbox_y1[candidate_ids] > y0)
            & (self.bbox_y0[candidate_ids] < y1)
        )
        return candidate_ids[mask].astype(np.int64, copy=False)

    def stats(self):
        """返回空间索引 bin 负载摘要，供 LSF shard 诊断使用。"""

        loads = [int(len(values)) for values in self.bins.values()]
        if loads:
            max_load = int(max(loads))
            avg_load = float(sum(loads)) / float(len(loads))
        else:
            max_load = 0
            avg_load = 0.0
        return {
            "spatial_index_bin_count": int(len(self.bins)),
            "spatial_index_axis_bin_count_x": int(self.bin_count_x),
            "spatial_index_axis_bin_count_y": int(self.bin_count_y),
            "max_bin_load": int(max_load),
            "avg_bin_load": float(avg_load),
        }


class LayoutIndex(object):
    """版图查询缓存，保存 polygon、bbox 数组和有效层摘要。"""

    __slots__ = (
        "indexed_elements",
        "bbox_x0",
        "bbox_y0",
        "bbox_x1",
        "bbox_y1",
        "spatial_index",
        "effective_pattern_layers",
        "excluded_helper_layers",
    )

    def __init__(self, indexed_elements, effective_pattern_layers, excluded_helper_layers):
        self.indexed_elements = list(indexed_elements)
        bboxes = [item["bbox"] for item in self.indexed_elements]
        self.bbox_x0 = np.asarray([bbox[0] for bbox in bboxes], dtype=np.float64)
        self.bbox_y0 = np.asarray([bbox[1] for bbox in bboxes], dtype=np.float64)
        self.bbox_x1 = np.asarray([bbox[2] for bbox in bboxes], dtype=np.float64)
        self.bbox_y1 = np.asarray([bbox[3] for bbox in bboxes], dtype=np.float64)
        self.spatial_index = SimpleSpatialIndex(bboxes)
        self.effective_pattern_layers = list(effective_pattern_layers)
        self.excluded_helper_layers = list(excluded_helper_layers)


def filter_layout_index_by_bbox(layout_index, bbox):
    """按 bbox 过滤 LayoutIndex 中的图元，供 LSF shard 使用 halo 裁剪工作集。"""

    filtered = []
    for item in layout_index.indexed_elements:
        if _bbox_intersection(item["bbox"], bbox) is not None:
            filtered.append(item)
    return LayoutIndex(filtered, layout_index.effective_pattern_layers, layout_index.excluded_helper_layers)


def spatial_index_stats(layout_index):
    """返回 LayoutIndex 空间索引的轻量诊断统计。"""

    if layout_index is None or getattr(layout_index, "spatial_index", None) is None:
        return {
            "spatial_index_bin_count": 0,
            "spatial_index_axis_bin_count_x": 0,
            "spatial_index_axis_bin_count_y": 0,
            "max_bin_load": 0,
            "avg_bin_load": 0.0,
        }
    return layout_index.spatial_index.stats()


def write_layout_index_oas(layout_index, filepath, cell_name="TILE"):
    """把 LayoutIndex 中的 polygon 写成独立 OAS tile。"""

    lib = gdstk.Library()
    cell = gdstk.Cell(str(cell_name)[:64] or "TILE")
    element_count = 0
    for item in layout_index.indexed_elements:
        element = item.get("element")
        points = _polygon_vertices_array(element)
        if points is None:
            continue
        layer = int(item.get("layer", getattr(element, "layer", 0)))
        datatype = int(item.get("datatype", getattr(element, "datatype", 0)))
        cell.add(gdstk.Polygon(np.asarray(points, dtype=np.float64).copy(), layer=layer, datatype=datatype))
        element_count += 1
    lib.add(cell)
    _write_oas_library(lib, filepath)
    return {
        "tile_element_count": int(element_count),
        "tile_oas_bytes": int(os.path.getsize(str(filepath))) if os.path.exists(str(filepath)) else 0,
    }


class GridSeedCandidate(object):
    """geometry-driven seed 记录。"""

    __slots__ = ("center", "seed_bbox", "grid_ix", "grid_iy", "bucket_weight", "seed_type")

    def __init__(self, center, seed_bbox, grid_ix, grid_iy, bucket_weight=1, seed_type=SEED_TYPE_RESIDUAL):
        self.center = (float(center[0]), float(center[1]))
        self.seed_bbox = tuple(float(v) for v in seed_bbox)
        self.grid_ix = int(grid_ix)
        self.grid_iy = int(grid_iy)
        self.bucket_weight = int(bucket_weight)
        self.seed_type = str(seed_type)

    def to_json(self):
        """把 seed 转成 JSON 可写格式。"""

        return {
            "center": list(self.center),
            "seed_bbox": list(self.seed_bbox),
            "grid_ix": int(self.grid_ix),
            "grid_iy": int(self.grid_iy),
            "bucket_weight": int(self.bucket_weight),
            "seed_type": str(self.seed_type),
        }

    @classmethod
    def from_json(cls, payload):
        """从 JSON payload 还原 seed。"""

        return cls(
            payload["center"],
            payload["seed_bbox"],
            int(payload["grid_ix"]),
            int(payload["grid_iy"]),
            int(payload["bucket_weight"]),
            str(payload["seed_type"]),
        )


class MarkerRecord(object):
    """单个 synthetic seed 栅格化后的 marker 记录。"""

    __slots__ = (
        "marker_id",
        "source_path",
        "source_name",
        "marker_bbox",
        "marker_center",
        "clip_bbox",
        "expanded_bbox",
        "clip_bbox_q",
        "expanded_bbox_q",
        "marker_bbox_q",
        "shift_limits_px",
        "clip_bitmap",
        "expanded_bitmap",
        "clip_hash",
        "expanded_hash",
        "clip_area",
        "seed_weight",
        "exact_cluster_id",
        "metadata",
    )

    def __init__(self, **kwargs):
        for name in self.__slots__:
            setattr(self, name, kwargs.get(name))
        if self.seed_weight is None:
            self.seed_weight = 1
        if self.exact_cluster_id is None:
            self.exact_cluster_id = -1
        if self.metadata is None:
            self.metadata = {}


class ExactCluster(object):
    """全局 exact hash 聚合后的簇。"""

    __slots__ = ("exact_cluster_id", "exact_key", "representative", "members", "member_count", "weight_sum")

    def __init__(self, exact_cluster_id, exact_key, representative, members, member_count=None, weight_sum=None):
        self.exact_cluster_id = int(exact_cluster_id)
        self.exact_key = str(exact_key)
        self.representative = representative
        self.members = list(members)
        self.member_count = int(member_count) if member_count is not None else int(len(self.members))
        if weight_sum is None:
            total = 0
            for member in self.members:
                total += max(1, int(member.seed_weight))
            self.weight_sum = int(total)
        else:
            self.weight_sum = int(weight_sum)

    @property
    def weight(self):
        """返回簇内 seed 权重和。"""

        return int(self.weight_sum)


class CandidateClip(object):
    """由 exact cluster representative 派生出的 candidate clip。"""

    __slots__ = (
        "candidate_id",
        "origin_exact_cluster_id",
        "origin_exact_key",
        "center",
        "clip_bbox",
        "clip_bbox_q",
        "clip_bitmap",
        "clip_hash",
        "shift_direction",
        "shift_distance_um",
        "coverage",
        "source_marker_id",
        "origin_seed_type",
    )

    def __init__(self, **kwargs):
        for name in self.__slots__:
            setattr(self, name, kwargs.get(name))
        if self.coverage is None:
            self.coverage = set()


def _pack_bitmap(bitmap):
    """把 bool bitmap 压缩成 packbits。"""

    return np.packbits(np.ascontiguousarray(bitmap, dtype=np.uint8).reshape(-1))


def bitmap_shape_key(shape):
    """把 bitmap shape 转成稳定的 JSON key。"""

    return "%dx%d" % (int(shape[0]), int(shape[1]))


def _bitmap_transforms(bitmap):
    """返回 bitmap 的 8 种旋转/镜像等价形态。"""

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


def _canonical_bitmap_payload(bitmap):
    """对 bitmap 做方向归一化，生成 exact hash payload。"""

    if bitmap.size == 0 or not np.any(bitmap):
        return b"empty"
    payloads = []
    for transformed in _bitmap_transforms(bitmap):
        contig = np.ascontiguousarray(transformed.astype(np.uint8, copy=False))
        packed = np.packbits(contig.reshape(-1))
        payloads.append(("%dx%d:" % (contig.shape[0], contig.shape[1])).encode("ascii") + packed.tobytes())
    return min(payloads)


def _canonical_bitmap_hash(bitmap):
    """返回 canonical bitmap hash 和 payload。"""

    payload = _canonical_bitmap_payload(bitmap)
    return hashlib.sha256(payload).hexdigest(), payload


def _bitmap_exact_key(bitmap):
    """返回严格方向相关 bitmap key。"""

    packed = _pack_bitmap(bitmap)
    return int(bitmap.shape[0]), int(bitmap.shape[1]), packed.tobytes()


def _bitcount_sum_rows(byte_matrix):
    """对二维 uint8 矩阵逐行做 popcount 求和。"""

    if byte_matrix.size == 0:
        return np.zeros((byte_matrix.shape[0],), dtype=np.int64)
    return BIT_COUNT_TABLE[byte_matrix].sum(axis=1, dtype=np.int64)


def _chunk_indices_by_row_width(indices, row_width_bytes, byte_budget=COVERAGE_CHUNK_BYTE_BUDGET):
    """按 packed-bitmap 行宽切分目标索引，避免一次构造过大的矩阵。"""

    if indices.size == 0:
        return
    safe_row_width = max(int(row_width_bytes), 1)
    chunk_size = max(1, int(byte_budget) // safe_row_width)
    for start in range(0, int(indices.size), chunk_size):
        yield indices[start : start + chunk_size]


def _pool_bitmap(bitmap, bins=10):
    """把 bitmap 池化成固定大小密度网格，供 signature 和 shortlist 使用。"""

    src_bool = np.asarray(bitmap, dtype=bool)
    pooled = np.zeros((int(bins), int(bins)), dtype=np.float32)
    if src_bool.ndim != 2 or src_bool.size == 0:
        return pooled

    h, w = int(src_bool.shape[0]), int(src_bool.shape[1])
    if not np.any(src_bool):
        return pooled
    if h < bins or w < bins:
        active = np.argwhere(src_bool)
        if active.size == 0:
            return pooled
        row_edges = np.linspace(0, h, int(bins) + 1, dtype=np.int32)
        col_edges = np.linspace(0, w, int(bins) + 1, dtype=np.int32)
        row_bins = np.searchsorted(row_edges[1:], active[:, 0], side="right")
        col_bins = np.searchsorted(col_edges[1:], active[:, 1], side="right")
        counts = np.zeros((int(bins), int(bins)), dtype=np.float32)
        np.add.at(counts, (row_bins, col_bins), 1.0)
    else:
        cache_key = (h, w, int(bins))
        cached_edges = _POOL_EDGE_CACHE.get(cache_key)
        if cached_edges is None:
            cached_edges = (
                np.linspace(0, h, int(bins) + 1, dtype=np.int32),
                np.linspace(0, w, int(bins) + 1, dtype=np.int32),
            )
            _POOL_EDGE_CACHE[cache_key] = cached_edges
        row_edges, col_edges = cached_edges
        src_float = src_bool.astype(np.float32, copy=False)
        counts = np.add.reduceat(src_float, row_edges[:-1], axis=0)
        counts = np.add.reduceat(counts, col_edges[:-1], axis=1)[: int(bins), : int(bins)].astype(np.float32, copy=False)

    row_sizes = np.maximum(np.diff(row_edges).astype(np.float32, copy=False), 1.0)
    col_sizes = np.maximum(np.diff(col_edges).astype(np.float32, copy=False), 1.0)
    pooled = counts / (row_sizes[:, None] * col_sizes[None, :])
    total = float(np.sum(pooled))
    if total > 0.0:
        pooled /= total
    return pooled


def _graph_bitmap_descriptor(bitmap):
    """从 bitmap 提取 v1 full prefilter 的 invariant、topology 和 signature。"""

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
        invariants,
        topology,
        pooled.reshape(-1).astype(np.float32),
        np.sum(pooled, axis=1, dtype=np.float32),
        np.sum(pooled, axis=0, dtype=np.float32),
    )


def _cheap_bitmap_descriptor(bitmap):
    """提取不依赖连通域标记的 v1 cheap descriptor。"""

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
        invariants,
        pooled.reshape(-1).astype(np.float32),
        np.sum(pooled, axis=1, dtype=np.float32),
        np.sum(pooled, axis=0, dtype=np.float32),
        int(active_count),
    )


def _signature_embedding(desc):
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


def _cheap_descriptor_arrays_for_bitmaps(bitmaps):
    """批量提取同 shape bitmap 的 cheap descriptor 数组，避免 prepare-coverage 逐 group 循环。"""

    masks = np.asarray(bitmaps, dtype=bool)
    if masks.ndim != 3:
        masks = masks.reshape((0, 0, 0))
    n, h, w = masks.shape
    total_px = max(int(h) * int(w), 1)
    active_count = np.count_nonzero(masks.reshape((int(n), -1)), axis=1).astype(np.int64)
    fill_ratio = active_count.astype(np.float32) / float(total_px)
    invariants = np.zeros((int(n), 8), dtype=np.float32)
    invariants[:, 1] = fill_ratio
    active_mask = active_count > 0
    if np.any(active_mask):
        row_any = np.any(masks, axis=2)
        col_any = np.any(masks, axis=1)
        first_row = np.argmax(row_any, axis=1)
        last_row = int(h) - 1 - np.argmax(row_any[:, ::-1], axis=1)
        first_col = np.argmax(col_any, axis=1)
        last_col = int(w) - 1 - np.argmax(col_any[:, ::-1], axis=1)
        bbox_h = (last_row - first_row + 1).astype(np.float32)
        bbox_w = (last_col - first_col + 1).astype(np.float32)
        bbox_long = np.maximum(bbox_w, bbox_h)
        bbox_short = np.minimum(bbox_w, bbox_h)
        span = max(float(h), float(w), 1.0)
        extent_area = np.maximum(bbox_w * bbox_h, 1.0)
        active_float = active_count.astype(np.float32)
        invariants[active_mask, 2] = bbox_long[active_mask] / span
        invariants[active_mask, 3] = bbox_short[active_mask] / span
        invariants[active_mask, 4] = bbox_short[active_mask] / np.maximum(bbox_long[active_mask], 1.0)
        invariants[active_mask, 5] = active_float[active_mask] / extent_area[active_mask]

    bins = 10
    cached_edges = _POOL_EDGE_CACHE.get((int(h), int(w), int(bins)))
    if cached_edges is None:
        cached_edges = (
            np.linspace(0, int(h), int(bins) + 1, dtype=np.int32),
            np.linspace(0, int(w), int(bins) + 1, dtype=np.int32),
        )
        _POOL_EDGE_CACHE[(int(h), int(w), int(bins))] = cached_edges
    row_edges, col_edges = cached_edges
    pooled = np.zeros((int(n), int(bins), int(bins)), dtype=np.float32)
    masks_float = masks.astype(np.float32, copy=False)
    row_sizes = np.maximum(np.diff(row_edges).astype(np.float32, copy=False), 1.0)
    col_sizes = np.maximum(np.diff(col_edges).astype(np.float32, copy=False), 1.0)
    for row_idx in range(int(bins)):
        r0 = int(row_edges[row_idx])
        r1 = int(row_edges[row_idx + 1])
        for col_idx in range(int(bins)):
            c0 = int(col_edges[col_idx])
            c1 = int(col_edges[col_idx + 1])
            pooled[:, row_idx, col_idx] = np.sum(masks_float[:, r0:r1, c0:c1], axis=(1, 2)) / (
                row_sizes[row_idx] * col_sizes[col_idx]
            )
    pooled_sum = np.sum(pooled.reshape((int(n), -1)), axis=1)
    nonzero = pooled_sum > 0.0
    if np.any(nonzero):
        pooled[nonzero] = pooled[nonzero] / pooled_sum[nonzero, None, None]
    signature_grid = pooled.reshape((int(n), int(bins) * int(bins))).astype(np.float32, copy=False)
    signature_proj_x = np.sum(pooled, axis=1, dtype=np.float32)
    signature_proj_y = np.sum(pooled, axis=2, dtype=np.float32)
    signature_vectors = np.concatenate(
        [signature_grid, 0.5 * signature_proj_x, 0.5 * signature_proj_y],
        axis=1,
    ).astype(np.float32, copy=False)
    norms = np.linalg.norm(signature_vectors, axis=1, keepdims=True)
    signature_vectors = signature_vectors / np.maximum(norms, 1e-6)
    peak_grid = np.where(np.any(signature_grid > 0.0, axis=1), np.argmax(signature_grid, axis=1), -1)
    peak_x = np.where(np.any(signature_proj_x > 0.0, axis=1), np.argmax(signature_proj_x, axis=1), -1)
    peak_y = np.where(np.any(signature_proj_y > 0.0, axis=1), np.argmax(signature_proj_y, axis=1), -1)
    subgroup_keys = np.stack(
        [
            np.rint(invariants[:, 1] * 40.0),
            np.rint(np.maximum(invariants[:, 2], invariants[:, 3]) * 10.0),
            np.rint(np.minimum(invariants[:, 2], invariants[:, 3]) * 10.0),
            peak_grid,
            np.where((peak_x >= 0) & (peak_y >= 0), peak_x * 31 + peak_y, -1),
        ],
        axis=1,
    ).astype(np.int32)
    return {
        "area_px": active_count,
        "cheap_invariants": invariants,
        "cheap_signature_grid": signature_grid,
        "cheap_signature_proj_x": signature_proj_x,
        "cheap_signature_proj_y": signature_proj_y,
        "cheap_signature_vectors": signature_vectors,
        "cheap_subgroup_keys": subgroup_keys,
    }


def _normalized_matrix(rows):
    """把一组向量堆叠成按行 L2 归一化的矩阵。"""

    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.ndim != 2:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms <= 1e-12, 1.0, norms)
    return matrix / norms


def _graph_invariant_distance(desc_a, desc_b):
    """计算 full graph invariant 加权距离，并标记关键维度是否越界。"""

    source_inv = np.asarray(desc_a.invariants, dtype=np.float64)
    target_inv = np.asarray(desc_b.invariants, dtype=np.float64)
    floors = np.asarray([0.25, 0.02, 0.02, 0.02, 0.03, 0.03, 0.02, 0.02], dtype=np.float64)
    weights = np.asarray([0.08, 0.24, 0.10, 0.08, 0.18, 0.14, 0.10, 0.08], dtype=np.float64)
    denom = np.maximum(np.maximum(np.abs(source_inv), np.abs(target_inv)), floors)
    errs = np.minimum(np.abs(target_inv - source_inv) / denom, 1.0)
    critical = bool((errs[1] > 0.45) or (errs[4] > 0.45) or (errs[5] > 0.45))
    return float(errs @ weights), critical


def _graph_signature_similarity(desc_a, desc_b):
    """计算 full graph signature 的 v1 风格相似度。"""

    source_grid = _normalized_matrix([desc_a.signature_grid])[0]
    source_proj_x = _normalized_matrix([desc_a.signature_proj_x])[0]
    source_proj_y = _normalized_matrix([desc_a.signature_proj_y])[0]
    target_grid = _normalized_matrix([desc_b.signature_grid])[0]
    target_proj_x = _normalized_matrix([desc_b.signature_proj_x])[0]
    target_proj_y = _normalized_matrix([desc_b.signature_proj_y])[0]
    return float(0.6 * (target_grid @ source_grid) + 0.2 * (target_proj_x @ source_proj_x) + 0.2 * (target_proj_y @ source_proj_y))


def _strict_graph_gate_reason(candidate_desc, witness_desc):
    """执行 final verification 专用 strict graph gate，并返回细分原因。"""

    invariant_score, critical = _graph_invariant_distance(candidate_desc, witness_desc)
    if critical or invariant_score > float(STRICT_INVARIANT_LIMIT):
        return False, "graph_invariant"
    topology_dist = float(np.linalg.norm(np.asarray(candidate_desc.topology, dtype=np.float64) - np.asarray(witness_desc.topology, dtype=np.float64)))
    if topology_dist > float(STRICT_TOPOLOGY_THRESHOLD):
        return False, "graph_topology"
    if _graph_signature_similarity(candidate_desc, witness_desc) < float(STRICT_SIGNATURE_THRESHOLD):
        return False, "graph_signature"
    return True, "verified_shift_witness"


def _coverage_cheap_subgroup_key(desc):
    """根据 cheap descriptor 生成 v1 coverage shortlist 子分组 key。"""

    invariants = np.asarray(desc.invariants, dtype=np.float32)
    fill_ratio = float(invariants[1]) if invariants.size > 1 else 0.0
    bbox_long = float(invariants[2]) if invariants.size > 2 else 0.0
    bbox_short = float(invariants[3]) if invariants.size > 3 else 0.0
    signature_grid = np.asarray(desc.signature_grid, dtype=np.float32)
    peak_grid = int(np.argmax(signature_grid)) if signature_grid.size and np.any(signature_grid > 0.0) else -1
    proj_x = np.asarray(desc.signature_proj_x, dtype=np.float32)
    proj_y = np.asarray(desc.signature_proj_y, dtype=np.float32)
    peak_x = int(np.argmax(proj_x)) if np.any(proj_x > 0.0) else -1
    peak_y = int(np.argmax(proj_y)) if np.any(proj_y > 0.0) else -1
    return (
        int(round(fill_ratio * 40.0)),
        int(round(max(bbox_long, bbox_short) * 10.0)),
        int(round(min(bbox_long, bbox_short) * 10.0)),
        peak_grid,
        (peak_x * 31 + peak_y) if peak_x >= 0 and peak_y >= 0 else -1,
    )


def _candidate_bundle_fill_bin(area_px, clip_pixels):
    """按 fill-ratio 生成 candidate bundle 子桶编号，子桶只用于减少加载面。"""

    if int(clip_pixels) <= 0:
        return 0
    fill_ratio = float(area_px) / float(max(int(clip_pixels), 1))
    fill_ratio = min(max(fill_ratio, 0.0), 1.0)
    return int(math.floor(fill_ratio / float(CANDIDATE_BUNDLE_FILL_BIN_WIDTH) + 1e-12))


def _candidate_bundle_fill_bin_for_bitmap(bitmap):
    """根据 bitmap 面积生成 candidate bundle 子桶编号。"""

    mask = np.asarray(bitmap, dtype=bool)
    return _candidate_bundle_fill_bin(int(np.count_nonzero(mask)), int(mask.size))


def coverage_fill_bin_for_bitmap(bitmap):
    """返回 coverage source/target 分桶共用的 fill-bin 编号。"""

    return _candidate_bundle_fill_bin_for_bitmap(bitmap)


def _candidate_bundle_fill_neighbor_bins(fill_bin):
    """返回保守 fill 邻域，范围宽于 cheap fill gate，避免 bucket 过滤改变召回。"""

    radius = int(math.ceil(float(CHEAP_FILL_ABS_LIMIT) / float(CANDIDATE_BUNDLE_FILL_BIN_WIDTH))) + 1
    max_bin = int(math.ceil(1.0 / float(CANDIDATE_BUNDLE_FILL_BIN_WIDTH)))
    start = max(0, int(fill_bin) - radius)
    end = min(max_bin, int(fill_bin) + radius)
    return range(start, end + 1)


def _exact_cosine_topk_labels(vectors, k):
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


def _coverage_structure(tol_px):
    """返回 coverage ECC 使用的方形结构元，并按容差像素缓存。"""

    tol = int(tol_px)
    cached = _COVERAGE_STRUCTURE_CACHE.get(tol)
    if cached is None:
        cached = np.ones((2 * tol + 1, 2 * tol + 1), dtype=bool)
        _COVERAGE_STRUCTURE_CACHE[tol] = cached
    return cached


def _exact_key_for_record(record):
    """返回 marker 的全局 exact key。"""

    return "%s:%s" % (str(record.clip_hash), str(record.expanded_hash))


def _window_pixels(window_um, pixel_size_um):
    """把物理窗口尺寸转换成像素数。"""

    return max(1, int(math.ceil(float(window_um) / max(float(pixel_size_um), 1e-12) - 1e-12)))


def _raster_window_spec(marker_bbox, marker_center, clip_size_um, pixel_size_um):
    """根据 marker bbox 生成 clip/expanded 的物理和像素窗口。"""

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
    return {
        "clip_bbox": clip_bbox,
        "expanded_bbox": expanded_bbox,
        "clip_bbox_q": (
            int(left_extra_px),
            int(bottom_extra_px),
            int(left_extra_px + clip_width_px),
            int(bottom_extra_px + clip_height_px),
        ),
        "expanded_bbox_q": (0, 0, int(width_px), int(height_px)),
        "marker_bbox_q": (
            max(0, int(math.floor((marker_bbox[0] - expanded_bbox[0]) / pixel_size_um + 1e-9))),
            max(0, int(math.floor((marker_bbox[1] - expanded_bbox[1]) / pixel_size_um + 1e-9))),
            min(width_px, int(math.ceil((marker_bbox[2] - expanded_bbox[0]) / pixel_size_um - 1e-9))),
            min(height_px, int(math.ceil((marker_bbox[3] - expanded_bbox[1]) / pixel_size_um - 1e-9))),
        ),
        "shape": (int(height_px), int(width_px)),
        "shift_limits_px": {
            "x": (-int(left_extra_px), int(right_extra_px)),
            "y": (-int(bottom_extra_px), int(top_extra_px)),
        },
    }


def _query_candidate_ids(layout_index, bbox):
    """查询与 bbox 相交的 element id。"""

    return [int(idx) for idx in layout_index.spatial_index.intersection(tuple(float(v) for v in bbox))]


def marker_query_candidate_stats(records):
    """汇总 seed raster 时每个 expanded bbox 查询到的候选图元数量。"""

    counts = []
    for record in records:
        metadata = getattr(record, "metadata", {}) or {}
        counts.append(int(metadata.get("query_candidate_count", 0)))
    if not counts:
        return {
            "query_candidate_count_p50": 0.0,
            "query_candidate_count_p95": 0.0,
            "query_candidate_count_p99": 0.0,
            "query_candidate_count_max": 0,
        }
    values = np.asarray(counts, dtype=np.float64)
    return {
        "query_candidate_count_p50": float(np.percentile(values, 50)),
        "query_candidate_count_p95": float(np.percentile(values, 95)),
        "query_candidate_count_p99": float(np.percentile(values, 99)),
        "query_candidate_count_max": int(np.max(values)),
    }


def _polygon_strip_spans(points, bbox):
    """把 polygon 分解成水平 strip span，用于快速 raster。"""

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or len(pts) < 3:
        return []
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) < 3:
        return []
    y_levels = sorted(set(float(value) for value in pts[:, 1].tolist()))
    spans = []
    if len(y_levels) < 2:
        return spans
    for y0, y1 in zip(y_levels[:-1], y_levels[1:]):
        strip_y0 = max(float(y0), float(bbox[1]))
        strip_y1 = min(float(y1), float(bbox[3]))
        if strip_y0 >= strip_y1:
            continue
        y_mid = 0.5 * (strip_y0 + strip_y1)
        xs = []
        for start, end in zip(pts, np.roll(pts, -1, axis=0)):
            x0 = float(start[0])
            y0_edge = float(start[1])
            x1 = float(end[0])
            y1_edge = float(end[1])
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


def _fill_bitmap_from_elements(indexed_elements, candidate_ids, bbox, shape, pixel_size_um):
    """把 expanded window 内的几何元素栅格化成 bool bitmap。"""

    height = int(shape[0])
    width = int(shape[1])
    bitmap = np.zeros((height, width), dtype=bool)
    if height <= 0 or width <= 0:
        return bitmap
    bbox_x0 = float(bbox[0])
    bbox_y0 = float(bbox[1])
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


def _slice_bitmap(bitmap, bbox_q):
    """按像素 bbox 切出连续 bool bitmap。"""

    x0, y0, x1, y1 = [int(v) for v in bbox_q]
    return np.ascontiguousarray(bitmap[y0:y1, x0:x1], dtype=bool)


def _collect_boundary_positions(axis_mask):
    """从一维 occupancy mask 中提取边界位置。"""

    padded = np.concatenate((np.array([False], dtype=bool), axis_mask.astype(bool), np.array([False], dtype=bool)))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return sorted(set(int(value) for value in starts.tolist() + ends.tolist()))


def _score_shift_px(shift_px, base_low, base_high, boundary_positions, tolerance_px):
    """给某个 shift 像素值打分。"""

    left_boundary = base_low + int(shift_px)
    right_boundary = base_high + int(shift_px)
    if not boundary_positions:
        return 0, 0, -abs(int(shift_px))
    best_gap = min(min(abs(int(edge) - left_boundary), abs(int(edge) - right_boundary)) for edge in boundary_positions)
    touch_count = sum(
        1
        for edge in boundary_positions
        if min(abs(int(edge) - left_boundary), abs(int(edge) - right_boundary)) <= tolerance_px
    )
    return touch_count, -int(best_gap), -abs(int(shift_px))


def _collect_shift_values_px(boundary_positions, base_low, base_high, shift_interval, tolerance_px, max_count):
    """根据边界贴合程度挑选 systematic shift 候选。"""

    low, high = [int(v) for v in shift_interval]
    values = set([0, int(low), int(high)])
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


def prepare_layout(filepath, layer_processor=None, apply_layer_operations=False):
    """读取 OAS 并构建 LayoutIndex。"""

    lib = _read_oas_only_library(str(filepath))
    if apply_layer_operations and layer_processor is not None:
        lib = layer_processor.apply_layer_operations(lib)
    top_cells = list(lib.top_level()) or list(lib.cells)
    pattern_polygons = []
    seen_pattern_layers = set()
    for top_cell in top_cells:
        polygons = top_cell.get_polygons(apply_repetitions=True, include_paths=True, depth=None)
        for poly in polygons:
            layer, datatype = _element_layer_datatype(poly)
            layer_spec = (int(layer), int(datatype))
            seen_pattern_layers.add(layer_spec)
            if apply_layer_operations and layer_processor is not None:
                if not layer_processor.should_keep_pattern_layer(layer_spec):
                    continue
            pattern_polygons.append(poly)
    if apply_layer_operations and layer_processor is not None:
        effective_pattern_layers = layer_processor.effective_pattern_layers(seen_pattern_layers)
        effective_pattern_set = set(effective_pattern_layers)
        excluded_helper_layers = sorted(spec for spec in seen_pattern_layers if spec not in effective_pattern_set)
    else:
        effective_pattern_layers = sorted(seen_pattern_layers)
        excluded_helper_layers = []

    indexed_elements = []
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
    return LayoutIndex(indexed_elements, effective_pattern_layers, excluded_helper_layers)


def _layout_bbox(layout_index):
    """返回所有 pattern element 的总 bbox。"""

    if len(layout_index.indexed_elements) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        float(np.min(layout_index.bbox_x0)),
        float(np.min(layout_index.bbox_y0)),
        float(np.max(layout_index.bbox_x1)),
        float(np.max(layout_index.bbox_y1)),
    )


def _grid_anchor_index(layout_origin, coord, grid_step_um):
    """返回某坐标所在 anchor grid index。"""

    step = max(float(grid_step_um), 1e-9)
    return int(math.floor((float(coord) - float(layout_origin)) / step + 1e-9))


def _quantized_value(value, quant_step_um=GRID_BUCKET_QUANT_UM):
    """把物理坐标量化成稳定整数，供 seed 分组使用。"""

    return int(round(float(value) / max(float(quant_step_um), 1e-9)))


def _seed_bbox_for_center(center_xy, grid_step_um):
    """围绕 seed 中心生成局部 marker bbox，不再使用全域 grid cell。"""

    return _make_centered_bbox((float(center_xy[0]), float(center_xy[1])), float(grid_step_um), float(grid_step_um))


def _make_geometry_seed(layout_bbox, center_xy, grid_step_um, seed_type, bucket_weight=1):
    """用统一规则构造 geometry-driven seed。"""

    center = (float(center_xy[0]), float(center_xy[1]))
    return GridSeedCandidate(
        center,
        _seed_bbox_for_center(center, grid_step_um),
        _grid_anchor_index(layout_bbox[0], center[0], grid_step_um),
        _grid_anchor_index(layout_bbox[1], center[1], grid_step_um),
        int(bucket_weight),
        str(seed_type),
    )


def _element_size_key(item):
    """按 layer/datatype 和 bbox 尺寸量化分组，用于重复阵列识别。"""

    bbox = tuple(float(v) for v in item["bbox"])
    return (
        int(item["layer"]),
        int(item["datatype"]),
        _quantized_value(bbox[2] - bbox[0]),
        _quantized_value(bbox[3] - bbox[1]),
    )


def _is_long_shape_bbox(bbox, clip_size_um):
    """识别长条图形，避免把长条所在大 bbox 扩成二维 seed 网格。"""

    width = max(0.0, float(bbox[2]) - float(bbox[0]))
    height = max(0.0, float(bbox[3]) - float(bbox[1]))
    if width <= 0.0 or height <= 0.0:
        return False
    long_edge = max(width, height)
    short_edge = min(width, height)
    aspect = long_edge / max(short_edge, 1e-9)
    return bool(long_edge >= 4.0 * float(clip_size_um) and aspect >= 4.0)


def _axis_seed_positions(start, end, grid_step_um):
    """沿一维区间生成端点、中点和固定步长采样点。"""

    start = float(start)
    end = float(end)
    step = max(float(grid_step_um), 1e-9)
    if end <= start:
        return [0.5 * (start + end)]
    length = end - start
    margin = min(0.5 * step, 0.5 * length)
    first = start + margin
    last = end - margin
    values = [first, last, 0.5 * (start + end)]
    value = first
    while value <= last + 1e-9:
        values.append(value)
        value += step
    unique = {}
    for value in values:
        clipped = min(max(float(value), start), end)
        unique[_quantized_value(clipped, 1e-6)] = clipped
    return [unique[key] for key in sorted(unique)]


def _detect_long_shape_ids(layout_index, clip_size_um, excluded_ids=None):
    """返回长条图形的 element id 集合。"""

    excluded = set(int(value) for value in (excluded_ids or set()))
    long_ids = set()
    for idx, item in enumerate(layout_index.indexed_elements):
        if int(idx) in excluded:
            continue
        if _is_long_shape_bbox(item["bbox"], clip_size_um):
            long_ids.add(int(idx))
    return long_ids


def _array_edge_class(index, max_index):
    """把阵列坐标映射成左/内/右或下/内/上的三类边界状态。"""

    index = int(index)
    max_index = int(max_index)
    if index <= 0:
        return 0
    if index >= max_index:
        return 2
    return 1


def _array_occupancy_signature(occupied, col, row):
    """生成某个阵列局部 3x3 occupancy signature。"""

    signature = 0
    bit = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if (int(col) + dx, int(row) + dy) in occupied:
                signature |= 1 << bit
            bit += 1
    return int(signature)


def _record_array_spacing_representative(representatives, rep_key, center, weight=1):
    """把一个 spacing 位置合并进代表表。"""

    current = representatives.get(rep_key)
    if current is None:
        representatives[rep_key] = {"center": center, "weight": int(weight)}
    else:
        current["weight"] += int(weight)


def _array_cell_class_count(max_index, edge_class):
    """返回某个阵列 cell 边界类别包含的坐标数量。"""

    max_index = int(max_index)
    edge_class = int(edge_class)
    if max_index < 0:
        return 0
    if edge_class in (0, 2):
        return 1
    return max(0, int(max_index) - 1)


def _array_gap_class_count(max_index, edge_class):
    """返回某个相邻 cell gap 边界类别包含的间距数量。"""

    max_index = int(max_index)
    edge_class = int(edge_class)
    if max_index <= 0:
        return 0
    if edge_class in (0, 2):
        return 1
    return max(0, int(max_index) - 2)


def _array_cell_sample_index(max_index, edge_class):
    """为某个 cell 边界类别挑一个代表坐标。"""

    max_index = int(max_index)
    edge_class = int(edge_class)
    if edge_class == 0:
        return 0
    if edge_class == 2:
        return max(0, max_index)
    return max(0, min(max_index, max_index // 2))


def _array_gap_sample_index(max_index, edge_class):
    """为某个相邻 cell gap 边界类别挑一个代表 gap 起点。"""

    max_index = int(max_index)
    edge_class = int(edge_class)
    if edge_class == 0:
        return 0
    if edge_class == 2:
        return max(0, max_index - 1)
    return max(1, min(max_index - 2, max_index // 2))


def _build_dense_array_spacing_representatives(group_key, occupied, x_centers, y_centers, max_col, max_row):
    """对满阵列用解析计数生成 spacing 代表，避免遍历海量相邻对。"""

    representatives = {}
    raw_count = 0
    for edge_x in (0, 1, 2):
        gap_count_x = _array_gap_class_count(max_col, edge_x)
        if gap_count_x <= 0:
            continue
        gap_col = _array_gap_sample_index(max_col, edge_x)
        for edge_y in (0, 1, 2):
            row_count = _array_cell_class_count(max_row, edge_y)
            if row_count <= 0:
                continue
            row = _array_cell_sample_index(max_row, edge_y)
            weight = int(gap_count_x) * int(row_count)
            raw_count += weight
            center = (0.5 * (float(x_centers[gap_col]) + float(x_centers[gap_col + 1])), float(y_centers[row]))
            signature = _array_occupancy_signature(occupied, gap_col, row)
            rep_key = (group_key, "x_spacing", int(edge_x), int(edge_y), int(signature))
            _record_array_spacing_representative(representatives, rep_key, center, weight)

    for edge_x in (0, 1, 2):
        col_count = _array_cell_class_count(max_col, edge_x)
        if col_count <= 0:
            continue
        col = _array_cell_sample_index(max_col, edge_x)
        for edge_y in (0, 1, 2):
            gap_count_y = _array_gap_class_count(max_row, edge_y)
            if gap_count_y <= 0:
                continue
            gap_row = _array_gap_sample_index(max_row, edge_y)
            weight = int(col_count) * int(gap_count_y)
            raw_count += weight
            center = (float(x_centers[col]), 0.5 * (float(y_centers[gap_row]) + float(y_centers[gap_row + 1])))
            signature = _array_occupancy_signature(occupied, col, gap_row)
            rep_key = (group_key, "y_spacing", int(edge_x), int(edge_y), int(signature))
            _record_array_spacing_representative(representatives, rep_key, center, weight)

    for edge_x in (0, 1, 2):
        gap_count_x = _array_gap_class_count(max_col, edge_x)
        if gap_count_x <= 0:
            continue
        gap_col = _array_gap_sample_index(max_col, edge_x)
        for edge_y in (0, 1, 2):
            gap_count_y = _array_gap_class_count(max_row, edge_y)
            if gap_count_y <= 0:
                continue
            gap_row = _array_gap_sample_index(max_row, edge_y)
            weight = int(gap_count_x) * int(gap_count_y)
            raw_count += weight
            center = (
                0.5 * (float(x_centers[gap_col]) + float(x_centers[gap_col + 1])),
                0.5 * (float(y_centers[gap_row]) + float(y_centers[gap_row + 1])),
            )
            signature = _array_occupancy_signature(occupied, gap_col, gap_row)
            rep_key = (group_key, "corner_spacing", int(edge_x), int(edge_y), int(signature))
            _record_array_spacing_representative(representatives, rep_key, center, weight)
    return representatives, int(raw_count)


def _build_array_spacing_representatives(group_key, occupied, x_centers, y_centers, max_col, max_row):
    """为规则阵列生成 x/y/corner 三类图元间距代表。"""

    representatives = {}
    raw_count = 0
    for col, row in sorted(occupied):
        col = int(col)
        row = int(row)
        if (col + 1, row) in occupied:
            raw_count += 1
            center = (0.5 * (float(x_centers[col]) + float(x_centers[col + 1])), float(y_centers[row]))
            edge_x = 0 if col == 0 else (2 if col + 1 >= int(max_col) else 1)
            edge_y = _array_edge_class(row, max_row)
            signature = _array_occupancy_signature(occupied, col, row)
            rep_key = (group_key, "x_spacing", int(edge_x), int(edge_y), int(signature))
            _record_array_spacing_representative(representatives, rep_key, center)
        if (col, row + 1) in occupied:
            raw_count += 1
            center = (float(x_centers[col]), 0.5 * (float(y_centers[row]) + float(y_centers[row + 1])))
            edge_x = _array_edge_class(col, max_col)
            edge_y = 0 if row == 0 else (2 if row + 1 >= int(max_row) else 1)
            signature = _array_occupancy_signature(occupied, col, row)
            rep_key = (group_key, "y_spacing", int(edge_x), int(edge_y), int(signature))
            _record_array_spacing_representative(representatives, rep_key, center)
        if (col + 1, row) in occupied and (col, row + 1) in occupied and (col + 1, row + 1) in occupied:
            raw_count += 1
            center = (
                0.5 * (float(x_centers[col]) + float(x_centers[col + 1])),
                0.5 * (float(y_centers[row]) + float(y_centers[row + 1])),
            )
            edge_x = 0 if col == 0 else (2 if col + 1 >= int(max_col) else 1)
            edge_y = 0 if row == 0 else (2 if row + 1 >= int(max_row) else 1)
            signature = _array_occupancy_signature(occupied, col, row)
            rep_key = (group_key, "corner_spacing", int(edge_x), int(edge_y), int(signature))
            _record_array_spacing_representative(representatives, rep_key, center)
    return representatives, int(raw_count)


def _build_array_representative_seeds(layout_index, layout_bbox, grid_step_um):
    """识别规则二维阵列，并为边界/内部/邻域类型生成代表 seed。"""

    grouped = {}
    for idx, item in enumerate(layout_index.indexed_elements):
        bbox = tuple(float(v) for v in item["bbox"])
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        grouped.setdefault(_element_size_key(item), []).append((int(idx), item, bbox))

    seeds = []
    spacing_seeds = []
    classified_ids = set()
    array_group_count = 0
    audit_groups = []
    for group_key, entries in grouped.items():
        if len(entries) < 16:
            continue
        x_center_by_key = {}
        y_center_by_key = {}
        for _, _, bbox in entries:
            x_center = 0.5 * (bbox[0] + bbox[2])
            y_center = 0.5 * (bbox[1] + bbox[3])
            x_center_by_key.setdefault(_quantized_value(x_center), float(x_center))
            y_center_by_key.setdefault(_quantized_value(y_center), float(y_center))
        x_keys = sorted(x_center_by_key.keys())
        y_keys = sorted(y_center_by_key.keys())
        if len(x_keys) < 3 or len(y_keys) < 3:
            continue
        x_index = dict((key, idx) for idx, key in enumerate(x_keys))
        y_index = dict((key, idx) for idx, key in enumerate(y_keys))
        x_centers = [float(x_center_by_key[key]) for key in x_keys]
        y_centers = [float(y_center_by_key[key]) for key in y_keys]
        occupied = set()
        item_positions = []
        for item_id, item, bbox in entries:
            pos = (
                int(x_index[_quantized_value(0.5 * (bbox[0] + bbox[2]))]),
                int(y_index[_quantized_value(0.5 * (bbox[1] + bbox[3]))]),
            )
            occupied.add(pos)
            item_positions.append((item_id, item, bbox, pos))
        occupancy_ratio = float(len(occupied)) / float(max(1, len(x_keys) * len(y_keys)))
        if occupancy_ratio < 0.15:
            continue

        array_group_count += 1
        representatives = {}
        max_col = int(len(x_keys) - 1)
        max_row = int(len(y_keys) - 1)
        for item_id, item, bbox, pos in item_positions:
            col, row = int(pos[0]), int(pos[1])
            edge_x = _array_edge_class(col, max_col)
            edge_y = _array_edge_class(row, max_row)
            signature = _array_occupancy_signature(occupied, col, row)
            rep_key = (group_key, int(edge_x), int(edge_y), int(signature))
            current = representatives.get(rep_key)
            if current is None:
                representatives[rep_key] = {"bbox": bbox, "weight": 1}
            else:
                current["weight"] += 1
            classified_ids.add(int(item_id))

        for rep in representatives.values():
            center = _bbox_center(rep["bbox"])
            seeds.append(_make_geometry_seed(layout_bbox, center, grid_step_um, SEED_TYPE_ARRAY, int(rep["weight"])))
        if occupancy_ratio >= 0.98 and len(occupied) >= 1024:
            spacing_representatives, spacing_raw_count = _build_dense_array_spacing_representatives(
                group_key,
                occupied,
                x_centers,
                y_centers,
                max_col,
                max_row,
            )
            spacing_generation_mode = "dense_analytic"
        else:
            spacing_representatives, spacing_raw_count = _build_array_spacing_representatives(
                group_key,
                occupied,
                x_centers,
                y_centers,
                max_col,
                max_row,
            )
            spacing_generation_mode = "exact_sparse"
        for rep in spacing_representatives.values():
            spacing_seeds.append(_make_geometry_seed(layout_bbox, rep["center"], grid_step_um, SEED_TYPE_ARRAY_SPACE, int(rep["weight"])))
        audit_groups.append(
            {
                "array_group_id": int(array_group_count - 1),
                "layer": int(group_key[0]),
                "datatype": int(group_key[1]),
                "bbox_size_key": [int(group_key[2]), int(group_key[3])],
                "element_count": int(len(entries)),
                "occupied_count": int(len(occupied)),
                "x_grid_count": int(len(x_keys)),
                "y_grid_count": int(len(y_keys)),
                "occupancy_ratio": float(occupancy_ratio),
                "center_representative_count": int(len(representatives)),
                "center_weight_total": int(sum(int(rep["weight"]) for rep in representatives.values())),
                "spacing_generation_mode": str(spacing_generation_mode),
                "spacing_raw_count": int(spacing_raw_count),
                "spacing_representative_count": int(len(spacing_representatives)),
                "spacing_weight_total": int(sum(int(rep["weight"]) for rep in spacing_representatives.values())),
            }
        )
    return seeds, spacing_seeds, classified_ids, int(array_group_count), audit_groups


def _build_long_shape_path_seeds(layout_index, layout_bbox, grid_step_um, long_ids):
    """为长条图形生成一维路径 seed。"""

    seeds = []
    for item_id in sorted(int(value) for value in long_ids):
        item = layout_index.indexed_elements[int(item_id)]
        bbox = tuple(float(v) for v in item["bbox"])
        width = max(0.0, bbox[2] - bbox[0])
        height = max(0.0, bbox[3] - bbox[1])
        horizontal = bool(width >= height)
        axis_values = _axis_seed_positions(bbox[0], bbox[2], grid_step_um) if horizontal else _axis_seed_positions(bbox[1], bbox[3], grid_step_um)
        for other_id in layout_index.spatial_index.intersection(bbox):
            if int(other_id) == int(item_id):
                continue
            other_bbox = tuple(float(v) for v in layout_index.indexed_elements[int(other_id)]["bbox"])
            overlap = _bbox_intersection(bbox, other_bbox)
            if overlap is None:
                continue
            axis_values.append(0.5 * (overlap[0] + overlap[2]) if horizontal else 0.5 * (overlap[1] + overlap[3]))
        unique_values = {}
        for axis_value in axis_values:
            unique_values[_quantized_value(axis_value, 1e-6)] = float(axis_value)
        for value in [unique_values[key] for key in sorted(unique_values)]:
            center = (float(value), 0.5 * (bbox[1] + bbox[3])) if horizontal else (0.5 * (bbox[0] + bbox[2]), float(value))
            seeds.append(_make_geometry_seed(layout_bbox, center, grid_step_um, SEED_TYPE_LONG, 1))
    return seeds


def _build_residual_local_grid_seeds(layout_index, layout_bbox, grid_step_um, classified_ids):
    """只在残余图形自身 bbox 内生成局部 grid seed。"""

    seeds = []
    residual_count = 0
    for idx, item in enumerate(layout_index.indexed_elements):
        if int(idx) in classified_ids:
            continue
        bbox = tuple(float(v) for v in item["bbox"])
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        residual_count += 1
        for x_value in _axis_seed_positions(bbox[0], bbox[2], grid_step_um):
            for y_value in _axis_seed_positions(bbox[1], bbox[3], grid_step_um):
                seeds.append(_make_geometry_seed(layout_bbox, (float(x_value), float(y_value)), grid_step_um, SEED_TYPE_RESIDUAL, 1))
    return seeds, int(residual_count)


def _dedupe_geometry_seeds(seeds):
    """按全局 grid anchor 去重，并让 spacing seed 与普通 seed 各保留一个槽位。"""

    priority = {SEED_TYPE_ARRAY: 3, SEED_TYPE_LONG: 2, SEED_TYPE_RESIDUAL: 1}
    buckets = {}
    for seed in seeds:
        slot = "spacing" if str(seed.seed_type) == SEED_TYPE_ARRAY_SPACE else "normal"
        key = (int(seed.grid_ix), int(seed.grid_iy), slot)
        current = buckets.get(key)
        if current is None:
            buckets[key] = seed
            continue
        total_weight = int(current.bucket_weight) + int(seed.bucket_weight)
        if int(priority.get(str(seed.seed_type), 0)) > int(priority.get(str(current.seed_type), 0)):
            seed.bucket_weight = int(total_weight)
            buckets[key] = seed
        else:
            current.bucket_weight = int(total_weight)
    return sorted(buckets.values(), key=lambda seed: (int(seed.grid_ix), int(seed.grid_iy), str(seed.seed_type)))


def _seed_type_counts(seeds):
    """统计各类 seed 的数量。"""

    return dict(Counter(str(seed.seed_type) for seed in seeds))


def _clip_window_key(bbox):
    """把 clip window bbox 量化成稳定去重键。"""

    return (
        _quantized_value(float(bbox[0]), PATTERN_COVERAGE_COORD_QUANT_UM),
        _quantized_value(float(bbox[1]), PATTERN_COVERAGE_COORD_QUANT_UM),
        _quantized_value(float(bbox[2]), PATTERN_COVERAGE_COORD_QUANT_UM),
        _quantized_value(float(bbox[3]), PATTERN_COVERAGE_COORD_QUANT_UM),
    )


def _unique_clip_window_bboxes(seeds, clip_size_um):
    """按 seed center 生成去重后的 clip window bbox 列表。"""

    unique_windows = {}
    for seed in seeds:
        bbox = _make_centered_bbox(
            (float(seed.center[0]), float(seed.center[1])),
            float(clip_size_um),
            float(clip_size_um),
        )
        bbox = tuple(float(value) for value in bbox)
        unique_windows.setdefault(_clip_window_key(bbox), bbox)
    return list(unique_windows.values())


def _bbox_line_overlaps(left, right):
    """判断 bbox 是否在允许零宽线段的意义下相交。"""

    return bool(
        min(float(left[2]), float(right[2])) >= max(float(left[0]), float(right[0])) - 1e-12
        and min(float(left[3]), float(right[3])) >= max(float(left[1]), float(right[1])) - 1e-12
    )


def _bbox_contains(outer, inner):
    """判断 outer bbox 是否完整包含 inner bbox。"""

    return bool(
        float(outer[0]) <= float(inner[0])
        and float(outer[1]) <= float(inner[1])
        and float(outer[2]) >= float(inner[2])
        and float(outer[3]) >= float(inner[3])
    )


def _clip_bbox_intersection(left, right):
    """返回两个 bbox 的正面积交集。"""

    x0 = max(float(left[0]), float(right[0]))
    y0 = max(float(left[1]), float(right[1]))
    x1 = min(float(left[2]), float(right[2]))
    y1 = min(float(left[3]), float(right[3]))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _merge_intervals(intervals):
    """合并一维区间并返回总覆盖长度。"""

    if not intervals:
        return 0.0
    sorted_intervals = sorted((float(start), float(end)) for start, end in intervals)
    merged_length = 0.0
    current_start, current_end = sorted_intervals[0]
    for start, end in sorted_intervals[1:]:
        if start <= current_end + 1e-12:
            current_end = max(current_end, end)
            continue
        merged_length += max(0.0, current_end - current_start)
        current_start, current_end = start, end
    merged_length += max(0.0, current_end - current_start)
    return float(merged_length)


def _segment_rect_interval(start_xy, end_xy, rect):
    """计算线段落在轴对齐矩形内的参数区间。"""

    x0 = float(start_xy[0])
    y0 = float(start_xy[1])
    dx = float(end_xy[0]) - x0
    dy = float(end_xy[1]) - y0
    t0 = 0.0
    t1 = 1.0
    checks = (
        (-dx, x0 - float(rect[0])),
        (dx, float(rect[2]) - x0),
        (-dy, y0 - float(rect[1])),
        (dy, float(rect[3]) - y0),
    )
    for p_value, q_value in checks:
        if abs(p_value) <= 1e-15:
            if q_value < -1e-12:
                return None
            continue
        ratio = q_value / p_value
        if p_value < 0.0:
            if ratio > t1 + 1e-12:
                return None
            t0 = max(t0, ratio)
        else:
            if ratio < t0 - 1e-12:
                return None
            t1 = min(t1, ratio)
    if t1 <= t0:
        return None
    return (max(0.0, t0), min(1.0, t1))


def _covered_polygon_edge_length(polygon, clip_windows):
    """按真实 polygon 边段统计被 clip window 覆盖的边长。"""

    if not clip_windows:
        return 0.0
    points = np.asarray(polygon.points, dtype=np.float64)
    if points.shape[0] < 2:
        return 0.0
    covered_length = 0.0
    point_count = int(points.shape[0])
    for point_idx in range(point_count):
        start_xy = points[point_idx]
        end_xy = points[(point_idx + 1) % point_count]
        dx = float(end_xy[0]) - float(start_xy[0])
        dy = float(end_xy[1]) - float(start_xy[1])
        segment_length = math.hypot(dx, dy)
        if segment_length <= 0.0:
            continue
        segment_bbox = (
            min(float(start_xy[0]), float(end_xy[0])),
            min(float(start_xy[1]), float(end_xy[1])),
            max(float(start_xy[0]), float(end_xy[0])),
            max(float(start_xy[1]), float(end_xy[1])),
        )
        intervals = []
        for window_bbox in clip_windows:
            if not _bbox_line_overlaps(segment_bbox, window_bbox):
                continue
            interval = _segment_rect_interval(start_xy, end_xy, window_bbox)
            if interval is not None:
                intervals.append(interval)
        covered_length += segment_length * _merge_intervals(intervals)
    return float(covered_length)


def _rectangle_union_area(rectangles):
    """用扫描线计算一组轴对齐矩形的 union 面积。"""

    events = []
    for rect in rectangles:
        x0, y0, x1, y1 = (float(value) for value in rect)
        if x1 <= x0 or y1 <= y0:
            continue
        events.append((x0, 1, y0, y1))
        events.append((x1, -1, y0, y1))
    if not events:
        return 0.0
    events.sort(key=lambda item: item[0])
    active = Counter()
    area = 0.0
    previous_x = float(events[0][0])
    for x_value, delta, y0, y1 in events:
        x_value = float(x_value)
        if x_value > previous_x and active:
            active_intervals = [interval for interval, count in active.items() if int(count) > 0]
            area += (x_value - previous_x) * _merge_intervals(active_intervals)
        interval_key = (float(y0), float(y1))
        active[interval_key] += int(delta)
        if active[interval_key] <= 0:
            del active[interval_key]
        previous_x = x_value
    return float(area)


def _is_bbox_filling_polygon(bbox, polygon_area):
    """判断 polygon 面积是否基本填满自身 bbox。"""

    bbox_area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
    if bbox_area <= 0.0:
        return False
    return bool(abs(float(polygon_area) - bbox_area) <= 1e-9 * max(float(polygon_area), bbox_area, 1.0))


def _covered_polygon_area(polygon, bbox, polygon_area, clip_windows):
    """按 polygon 面积统计 clip window union 的真实覆盖面积。"""

    if polygon_area <= 0.0 or not clip_windows:
        return 0.0
    if any(_bbox_contains(window_bbox, bbox) for window_bbox in clip_windows):
        return float(polygon_area)
    clipped_rectangles = []
    for window_bbox in clip_windows:
        clipped = _clip_bbox_intersection(window_bbox, bbox)
        if clipped is not None:
            clipped_rectangles.append(clipped)
    if not clipped_rectangles:
        return 0.0
    if _is_bbox_filling_polygon(bbox, polygon_area):
        return float(min(float(polygon_area), _rectangle_union_area(clipped_rectangles)))
    clip_polygons = [gdstk.rectangle((rect[0], rect[1]), (rect[2], rect[3])) for rect in clipped_rectangles]
    covered_polygons = gdstk.boolean(polygon, clip_polygons, "and", precision=PATTERN_COVERAGE_COORD_QUANT_UM)
    covered_area = float(sum(float(covered.area()) for covered in covered_polygons))
    return float(min(float(polygon_area), covered_area))


def _build_clip_window_index(clip_window_bboxes):
    """为 clip window bbox 构建轻量空间索引。"""

    return SimpleSpatialIndex([tuple(float(value) for value in bbox) for bbox in clip_window_bboxes])


def _seed_coverage_audit(layout_index, seeds, clip_size_um, grid_step_um):
    """按真实 target polygon 的边长、面积和轻量 type 统计 coverage。"""

    del grid_step_um
    clip_window_bboxes = _unique_clip_window_bboxes(seeds, clip_size_um)
    clip_window_index = _build_clip_window_index(clip_window_bboxes) if clip_window_bboxes else None
    total_edge_length = 0.0
    covered_edge_length = 0.0
    total_polygon_area = 0.0
    covered_polygon_area = 0.0
    type_edge_weights = Counter()
    covered_type_keys = set()
    covered_polygon_count = 0

    for item in layout_index.indexed_elements:
        polygon = item["element"]
        bbox = tuple(float(value) for value in item["bbox"])
        polygon_area = float(polygon.area())
        edge_length = float(polygon.perimeter())
        if polygon_area <= 0.0 and edge_length <= 0.0:
            continue
        type_key = _element_size_key(item)
        type_edge_weights[type_key] += float(edge_length)
        total_edge_length += float(edge_length)
        total_polygon_area += max(0.0, float(polygon_area))
        if clip_window_index is None:
            intersecting_windows = []
        else:
            query_bbox = (bbox[0] - 1e-12, bbox[1] - 1e-12, bbox[2] + 1e-12, bbox[3] + 1e-12)
            intersecting_window_ids = list(clip_window_index.intersection(query_bbox))
            intersecting_windows = [clip_window_bboxes[int(window_id)] for window_id in intersecting_window_ids]
        edge_covered = min(edge_length, _covered_polygon_edge_length(polygon, intersecting_windows))
        area_covered = min(polygon_area, _covered_polygon_area(polygon, bbox, polygon_area, intersecting_windows))
        covered_edge_length += max(0.0, float(edge_covered))
        covered_polygon_area += max(0.0, float(area_covered))
        edge_ratio = _safe_ratio(edge_covered, edge_length)
        area_ratio = _safe_ratio(area_covered, polygon_area)
        if edge_ratio >= PATTERN_TYPE_COVERAGE_THRESHOLD or area_ratio >= PATTERN_TYPE_COVERAGE_THRESHOLD:
            covered_type_keys.add(type_key)
            covered_polygon_count += 1

    type_weight_total = float(sum(float(value) for value in type_edge_weights.values()))
    type_weight_covered = float(sum(float(type_edge_weights[key]) for key in covered_type_keys))
    return {
        "target_edge_length_total": float(total_edge_length),
        "target_edge_length_covered": float(covered_edge_length),
        "target_edge_length_coverage_ratio": float(_clamp01(_safe_ratio(covered_edge_length, total_edge_length))),
        "target_polygon_area_total": float(total_polygon_area),
        "target_polygon_area_covered": float(covered_polygon_area),
        "target_polygon_area_coverage_ratio": float(_clamp01(_safe_ratio(covered_polygon_area, total_polygon_area))),
        "target_pattern_type_weight_total": float(type_weight_total),
        "target_pattern_type_weight_covered": float(type_weight_covered),
        "weighted_pattern_type_coverage_ratio": float(_clamp01(_safe_ratio(type_weight_covered, type_weight_total))),
        "target_pattern_type_count": int(len(type_edge_weights)),
        "covered_pattern_type_count": int(len(covered_type_keys)),
        "target_polygon_count": int(len(layout_index.indexed_elements)),
        "covered_polygon_count": int(covered_polygon_count),
        "clip_window_count": int(len(clip_window_bboxes)),
    }


def _empty_seed_stats(ratio, grid_step_um):
    """返回空版图下仍保持字段完整的 seed 统计。"""

    return {
        "seed_strategy": "geometry_driven",
        "grid_step_ratio": float(ratio),
        "grid_step_um": float(grid_step_um),
        "grid_seed_count": 0,
        "initial_seed_count": 0,
        "bucketed_seed_count": 0,
        "seed_bucket_merged_count": 0,
        "array_seed_count": 0,
        "array_spacing_seed_count": 0,
        "long_shape_seed_count": 0,
        "residual_seed_count": 0,
        "array_group_count": 0,
        "array_spacing_group_count": 0,
        "long_shape_count": 0,
        "residual_element_count": 0,
        "array_spacing_weight_total": 0,
        "seed_weight_total": 0,
        "seed_type_counts": {},
        "seed_audit": {
            "seed_strategy": "geometry_driven",
            "grid_step_ratio": float(ratio),
            "grid_step_um": float(grid_step_um),
            "array_group_count": 0,
            "array_spacing_group_count": 0,
            "array_groups": [],
        },
        "seed_coverage_audit": {
            "target_edge_length_total": 0.0,
            "target_edge_length_covered": 0.0,
            "target_edge_length_coverage_ratio": 0.0,
            "target_polygon_area_total": 0.0,
            "target_polygon_area_covered": 0.0,
            "target_polygon_area_coverage_ratio": 0.0,
            "target_pattern_type_weight_total": 0.0,
            "target_pattern_type_weight_covered": 0.0,
            "weighted_pattern_type_coverage_ratio": 0.0,
            "target_pattern_type_count": 0,
            "covered_pattern_type_count": 0,
            "target_polygon_count": 0,
            "covered_polygon_count": 0,
            "clip_window_count": 0,
        },
    }


def build_geometry_driven_seed_candidates(layout_index, clip_size_um):
    """按图形/重复阵列直接生成代表 seed，不再扫描全域 eligible grid。"""

    ratio = float(GRID_STEP_RATIO)
    grid_step_um = float(clip_size_um) * float(ratio)
    if len(layout_index.indexed_elements) == 0:
        return [], _empty_seed_stats(ratio, grid_step_um)
    layout_bbox = _layout_bbox(layout_index)
    array_seeds, array_spacing_seeds, array_ids, array_group_count, array_audit_groups = _build_array_representative_seeds(layout_index, layout_bbox, grid_step_um)
    long_ids = _detect_long_shape_ids(layout_index, clip_size_um, array_ids)
    long_seeds = _build_long_shape_path_seeds(layout_index, layout_bbox, grid_step_um, long_ids)
    classified_ids = set(array_ids)
    classified_ids.update(int(value) for value in long_ids)
    residual_seeds, residual_count = _build_residual_local_grid_seeds(layout_index, layout_bbox, grid_step_um, classified_ids)
    raw_seeds = list(array_seeds) + list(array_spacing_seeds) + list(long_seeds) + list(residual_seeds)
    seeds = _dedupe_geometry_seeds(raw_seeds)
    seed_type_counts = _seed_type_counts(seeds)
    array_spacing_group_count = sum(1 for group in array_audit_groups if int(group.get("spacing_representative_count", 0)) > 0)
    array_spacing_weight_total = int(sum(max(1, int(seed.bucket_weight)) for seed in array_spacing_seeds))
    seed_coverage_audit = _seed_coverage_audit(layout_index, seeds, clip_size_um, grid_step_um)
    seed_audit = {
        "seed_strategy": "geometry_driven",
        "grid_step_ratio": float(ratio),
        "grid_step_um": float(grid_step_um),
        "array_group_count": int(array_group_count),
        "array_spacing_group_count": int(array_spacing_group_count),
        "array_spacing_seed_count": int(len(array_spacing_seeds)),
        "array_spacing_weight_total": int(array_spacing_weight_total),
        "array_groups": array_audit_groups,
    }
    return seeds, {
        "seed_strategy": "geometry_driven",
        "grid_step_ratio": float(ratio),
        "grid_step_um": float(grid_step_um),
        "grid_seed_count": int(len(raw_seeds)),
        "initial_seed_count": int(len(raw_seeds)),
        "bucketed_seed_count": int(len(seeds)),
        "seed_bucket_merged_count": int(max(0, len(raw_seeds) - len(seeds))),
        "array_seed_count": int(len(array_seeds)),
        "array_spacing_seed_count": int(len(array_spacing_seeds)),
        "long_shape_seed_count": int(len(long_seeds)),
        "residual_seed_count": int(len(residual_seeds)),
        "array_group_count": int(array_group_count),
        "array_spacing_group_count": int(array_spacing_group_count),
        "long_shape_count": int(len(long_ids)),
        "residual_element_count": int(residual_count),
        "array_spacing_weight_total": int(array_spacing_weight_total),
        "seed_weight_total": int(sum(max(1, int(seed.bucket_weight)) for seed in seeds)),
        "seed_type_counts": seed_type_counts,
        "seed_audit": seed_audit,
        "seed_coverage_audit": seed_coverage_audit,
    }


def marker_record_from_seed(filepath, marker_index, seed, layout_index, config):
    """围绕单个 grid seed 构建 MarkerRecord。"""

    pixel_size_um = float(int(config.get("pixel_size_nm", DEFAULT_PIXEL_SIZE_NM))) / 1000.0
    clip_size_um = float(config.get("clip_size_um", 1.35))
    marker_bbox = tuple(float(v) for v in seed.seed_bbox)
    marker_center = _bbox_center(marker_bbox)
    raster_spec = _raster_window_spec(marker_bbox, marker_center, clip_size_um, pixel_size_um)
    expanded_bbox = raster_spec["expanded_bbox"]
    candidate_ids = _query_candidate_ids(layout_index, expanded_bbox)
    expanded_bitmap = _fill_bitmap_from_elements(
        layout_index.indexed_elements,
        candidate_ids,
        expanded_bbox,
        raster_spec["shape"],
        pixel_size_um,
    )
    clip_bitmap = _slice_bitmap(expanded_bitmap, raster_spec["clip_bbox_q"])
    clip_hash, _ = _canonical_bitmap_hash(clip_bitmap)
    expanded_hash, _ = _canonical_bitmap_hash(expanded_bitmap)
    source_path = str(filepath)
    source_name = Path(str(filepath)).name
    record = MarkerRecord(
        marker_id="%s__seed_%06d" % (Path(str(filepath)).stem, int(marker_index)),
        source_path=source_path,
        source_name=source_name,
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
        clip_area=float(np.count_nonzero(clip_bitmap)) * (pixel_size_um ** 2),
        seed_weight=int(seed.bucket_weight),
        exact_cluster_id=-1,
        metadata={
            "seed_bbox": list(seed.seed_bbox),
            "grid_ix": int(seed.grid_ix),
            "grid_iy": int(seed.grid_iy),
            "bucket_weight": int(seed.bucket_weight),
            "seed_type": str(seed.seed_type),
            "query_candidate_count": int(len(candidate_ids)),
        },
    )
    return record


def group_exact_clusters(marker_records):
    """按 clip_hash + expanded_hash 做全局 exact clustering。"""

    buckets = {}
    for record in marker_records:
        key = _exact_key_for_record(record)
        buckets.setdefault(key, []).append(record)
    exact_clusters = []
    sorted_items = sorted(buckets.items(), key=lambda item: (item[1][0].source_name, item[1][0].marker_id))
    for cluster_id, item in enumerate(sorted_items):
        exact_key, members = item
        for member in members:
            member.exact_cluster_id = int(cluster_id)
        exact_clusters.append(ExactCluster(int(cluster_id), exact_key, members[0], list(members)))
    return exact_clusters


def build_candidate_clip(cluster, clip_bbox, clip_bbox_q, bitmap, shift_direction, shift_distance_um, candidate_index):
    """把一个 bitmap 切片封装成 CandidateClip。"""

    clip_hash, _ = _canonical_bitmap_hash(bitmap)
    coverage = set([int(cluster.exact_cluster_id)]) if str(shift_direction) == "base" else set()
    return CandidateClip(
        candidate_id="cand_%06d_%03d" % (int(cluster.exact_cluster_id), int(candidate_index)),
        origin_exact_cluster_id=int(cluster.exact_cluster_id),
        origin_exact_key=str(cluster.exact_key),
        center=_bbox_center(clip_bbox),
        clip_bbox=tuple(float(v) for v in clip_bbox),
        clip_bbox_q=tuple(int(v) for v in clip_bbox_q),
        clip_bitmap=np.ascontiguousarray(bitmap, dtype=bool),
        clip_hash=str(clip_hash),
        shift_direction=str(shift_direction),
        shift_distance_um=float(shift_distance_um),
        coverage=coverage,
        source_marker_id=str(cluster.representative.marker_id),
        origin_seed_type=_exact_cluster_seed_type(cluster),
    )


def _diagonal_shift_direction(shift_x_px, shift_y_px):
    """把二维位移方向映射成稳定的 diagonal shift 名称。"""

    if int(shift_x_px) >= 0 and int(shift_y_px) >= 0:
        return "diag_ne"
    if int(shift_x_px) < 0 and int(shift_y_px) >= 0:
        return "diag_nw"
    if int(shift_x_px) >= 0 and int(shift_y_px) < 0:
        return "diag_se"
    return "diag_sw"


def _shift_candidate_cost(candidate):
    """hash 去重时优先保留 base，其次轴向 shift，最后才是 diagonal shift。"""

    direction = str(candidate.shift_direction)
    if direction == "base":
        shift_kind = 0
    elif direction.startswith("diag_"):
        shift_kind = 2
    else:
        shift_kind = 1
    return shift_kind, abs(float(candidate.shift_distance_um)), str(candidate.candidate_id)


def _limited_nonzero_shifts(shift_values, max_count):
    """从 systematic shift 列表中取少量非零位移，用于构造 diagonal 组合。"""

    values = []
    seen = set()
    for value in shift_values:
        value = int(value)
        if value == 0 or value in seen:
            continue
        seen.add(value)
        values.append(value)
        if len(values) >= int(max_count):
            break
    return values


def _rank_diagonal_shift_pairs(x_shifts, y_shifts, max_count):
    """按距离由近到远挑选有限个 diagonal 位移组合。"""

    ranked = []
    for shift_x_px in x_shifts:
        for shift_y_px in y_shifts:
            distance_sq = int(shift_x_px) * int(shift_x_px) + int(shift_y_px) * int(shift_y_px)
            manhattan = abs(int(shift_x_px)) + abs(int(shift_y_px))
            ranked.append((distance_sq, manhattan, int(shift_x_px), int(shift_y_px)))
    ranked.sort()
    return [(item[2], item[3]) for item in ranked[: max(0, int(max_count))]]


def generate_candidates_for_cluster(cluster, config):
    """为 exact cluster 生成 base、轴向 systematic shift 与少量 diagonal shift candidates。"""

    rep = cluster.representative
    pixel_size_um = float(int(config.get("pixel_size_nm", DEFAULT_PIXEL_SIZE_NM))) / 1000.0
    edge_tolerance_um = float(config.get("edge_tolerance_um", 0.02))
    candidates = [
        build_candidate_clip(cluster, rep.clip_bbox, rep.clip_bbox_q, rep.clip_bitmap, "base", 0.0, 0)
    ]
    base_x0, base_y0, base_x1, base_y1 = [int(v) for v in rep.clip_bbox_q]
    tolerance_px = max(0, int(math.ceil(edge_tolerance_um / max(pixel_size_um, 1e-12) - 1e-12)))
    max_shift_count = int(config.get("max_shift_count", 12))

    occupied_cols = np.any(rep.expanded_bitmap, axis=0)
    x_boundaries = _collect_boundary_positions(occupied_cols)
    x_interval = list(rep.shift_limits_px["x"])
    x_shift_values = _collect_shift_values_px(x_boundaries, base_x0, base_x1, x_interval, tolerance_px, max_shift_count)
    for shift_px in x_shift_values:
        if shift_px == 0:
            continue
        clip_bbox_q = (base_x0 + shift_px, base_y0, base_x1 + shift_px, base_y1)
        bitmap = _slice_bitmap(rep.expanded_bitmap, clip_bbox_q)
        shift_um = float(shift_px) * pixel_size_um
        clip_bbox = (rep.clip_bbox[0] + shift_um, rep.clip_bbox[1], rep.clip_bbox[2] + shift_um, rep.clip_bbox[3])
        candidates.append(
            build_candidate_clip(
                cluster,
                clip_bbox,
                clip_bbox_q,
                bitmap,
                "right" if shift_px > 0 else "left",
                abs(shift_um),
                len(candidates),
            )
        )

    occupied_rows = np.any(rep.expanded_bitmap, axis=1)
    y_boundaries = _collect_boundary_positions(occupied_rows)
    y_interval = list(rep.shift_limits_px["y"])
    y_shift_values = _collect_shift_values_px(y_boundaries, base_y0, base_y1, y_interval, tolerance_px, max_shift_count)
    for shift_px in y_shift_values:
        if shift_px == 0:
            continue
        clip_bbox_q = (base_x0, base_y0 + shift_px, base_x1, base_y1 + shift_px)
        bitmap = _slice_bitmap(rep.expanded_bitmap, clip_bbox_q)
        shift_um = float(shift_px) * pixel_size_um
        clip_bbox = (rep.clip_bbox[0], rep.clip_bbox[1] + shift_um, rep.clip_bbox[2], rep.clip_bbox[3] + shift_um)
        candidates.append(
            build_candidate_clip(
                cluster,
                clip_bbox,
                clip_bbox_q,
                bitmap,
                "up" if shift_px > 0 else "down",
                abs(shift_um),
                len(candidates),
            )
        )

    x_diagonal_shifts = _limited_nonzero_shifts(x_shift_values, DIAGONAL_SHIFT_AXIS_MAX_COUNT)
    y_diagonal_shifts = _limited_nonzero_shifts(y_shift_values, DIAGONAL_SHIFT_AXIS_MAX_COUNT)
    for shift_x_px, shift_y_px in _rank_diagonal_shift_pairs(x_diagonal_shifts, y_diagonal_shifts, DIAGONAL_SHIFT_MAX_COUNT):
        clip_bbox_q = (base_x0 + shift_x_px, base_y0 + shift_y_px, base_x1 + shift_x_px, base_y1 + shift_y_px)
        bitmap = _slice_bitmap(rep.expanded_bitmap, clip_bbox_q)
        shift_x_um = float(shift_x_px) * pixel_size_um
        shift_y_um = float(shift_y_px) * pixel_size_um
        clip_bbox = (
            rep.clip_bbox[0] + shift_x_um,
            rep.clip_bbox[1] + shift_y_um,
            rep.clip_bbox[2] + shift_x_um,
            rep.clip_bbox[3] + shift_y_um,
        )
        candidates.append(
            build_candidate_clip(
                cluster,
                clip_bbox,
                clip_bbox_q,
                bitmap,
                _diagonal_shift_direction(shift_x_px, shift_y_px),
                math.sqrt(float(shift_x_px * shift_x_px + shift_y_px * shift_y_px)) * pixel_size_um,
                len(candidates),
            )
        )

    deduped = {}
    for candidate in candidates:
        current = deduped.get(candidate.clip_hash)
        if current is None:
            deduped[candidate.clip_hash] = candidate
            continue
        if _shift_candidate_cost(candidate) < _shift_candidate_cost(current):
            deduped[candidate.clip_hash] = candidate
    return list(sorted(deduped.values(), key=lambda item: item.candidate_id))


def candidate_shift_summary(candidates):
    """汇总 candidate shift 方向统计，供结果 JSON 和 LSF manifest 诊断使用。"""

    def read_field(candidate, name, default):
        """兼容完整 CandidateClip 对象和 coverage shard 的轻量 metadata dict。"""

        if isinstance(candidate, dict):
            return candidate.get(name, default)
        return getattr(candidate, name, default)

    direction_counts = Counter(str(read_field(candidate, "shift_direction", "unknown")) for candidate in candidates)
    max_shift_distance_um = 0.0
    for candidate in candidates:
        max_shift_distance_um = max(max_shift_distance_um, abs(float(read_field(candidate, "shift_distance_um", 0.0))))
    diagonal_count = 0
    for direction, count in direction_counts.items():
        if str(direction).startswith("diag_"):
            diagonal_count += int(count)
    return {
        "candidate_direction_counts": dict(direction_counts),
        "diagonal_candidate_count": int(diagonal_count),
        "max_shift_distance_um": float(max_shift_distance_um),
    }


def _safe_ratio(numerator, denominator):
    """在分母可能为零时安全计算比例。"""

    denominator_value = float(denominator)
    if abs(denominator_value) <= 1e-12:
        return 0.0
    return float(numerator) / denominator_value


def _histogram_percentile(histogram, percentile):
    """从离散直方图中计算百分位，避免展开成超长列表。"""

    if not histogram:
        return 0.0
    ordered = sorted((int(value), int(count)) for value, count in histogram.items() if int(count) > 0)
    if not ordered:
        return 0.0
    total = int(sum(count for _, count in ordered))
    if total <= 0:
        return 0.0
    rank = int(math.ceil(float(percentile) * float(total) / 100.0)) - 1
    rank = max(0, min(rank, total - 1))
    seen = 0
    for value, count in ordered:
        seen += int(count)
        if seen > rank:
            return float(value)
    return float(ordered[-1][0])


def _numeric_percentile(values, percentile):
    """返回浮点序列的百分位数；空序列时返回 0。"""

    if not values:
        return 0.0
    return float(np.percentile(np.asarray(list(values), dtype=np.float32), float(percentile)))


def _clamp01(value):
    """把评分约束到 0 到 1 区间。"""

    return float(min(1.0, max(0.0, float(value))))


def _marker_seed_type(record):
    """读取 marker record 上游 seed type，缺失时回退为 unknown。"""

    metadata = dict(getattr(record, "metadata", {}) or {})
    seed_type = metadata.get("seed_type")
    if seed_type in (None, ""):
        auto_seed = dict(metadata.get("auto_seed", {}) or {})
        seed_type = auto_seed.get("seed_type")
    if seed_type in (None, ""):
        return "unknown"
    return str(seed_type)


def _exact_cluster_seed_type(exact_cluster):
    """返回 exact cluster representative 的 seed type。"""

    return _marker_seed_type(exact_cluster.representative)


def _candidate_origin_seed_type(candidate, exact_by_id):
    """返回 candidate 来源 representative 的 seed type。"""

    seed_type = getattr(candidate, "origin_seed_type", None)
    if seed_type not in (None, "", "unknown"):
        return str(seed_type)
    source_cluster = exact_by_id.get(int(candidate.origin_exact_cluster_id))
    if source_cluster is None:
        return "unknown"
    return _exact_cluster_seed_type(source_cluster)


def _worst_case_proxy(bitmap):
    """用边缘密度和填充稀疏度估计 OPC 建模中的 worst-case 倾向。"""

    mask = np.asarray(bitmap, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        return 0.0
    edge_count = int(np.count_nonzero(mask[:, 1:] != mask[:, :-1])) + int(np.count_nonzero(mask[1:, :] != mask[:-1, :]))
    active_count = max(int(np.count_nonzero(mask)), 1)
    fill_ratio = float(active_count) / float(max(mask.size, 1))
    edge_density = float(edge_count) / float(active_count)
    return float(edge_density * (1.0 - min(fill_ratio, 1.0)))


def _bitmap_risk_score(bitmap):
    """基于 bitmap 估计 weak-point 风险，数值越高越值得优先 review。"""

    if bitmap is None:
        return 0.0
    worst_score = float(_worst_case_proxy(bitmap))
    distance_score = float(distance_worst_case_proxy(bitmap))
    return _clamp01(0.55 * min(worst_score, 1.0) + 0.45 * min(distance_score, 1.0))


def _candidate_risk_score(candidate):
    """估计 candidate 的 weak-point 风险。"""

    return float(_bitmap_risk_score(candidate.clip_bitmap))


def _candidate_opc_center_base_score(candidate):
    """估计与 coverage 无关的 OPC center 几何基础分。"""

    bitmap = np.asarray(candidate.clip_bitmap, dtype=bool)
    if bitmap.ndim != 2 or bitmap.size == 0 or not np.any(bitmap):
        return 0.0
    height, width = int(bitmap.shape[0]), int(bitmap.shape[1])
    y_indices, x_indices = np.nonzero(bitmap)
    active_count = max(int(y_indices.size), 1)
    min_margin_px = min(
        int(np.min(x_indices)),
        int(np.min(y_indices)),
        int(width - 1 - np.max(x_indices)),
        int(height - 1 - np.max(y_indices)),
    )
    target_margin_px = max(1.0, 0.10 * float(min(width, height)))
    margin_score = _clamp01(float(min_margin_px) / target_margin_px)
    border_active = (
        int(np.count_nonzero(bitmap[0, :]))
        + int(np.count_nonzero(bitmap[-1, :]))
        + int(np.count_nonzero(bitmap[:, 0]))
        + int(np.count_nonzero(bitmap[:, -1]))
    )
    completeness_score = _clamp01(1.0 - float(border_active) / float(max(active_count, 1)))
    context_density_score = _clamp01(_safe_ratio(active_count, max(int(bitmap.size), 1)))
    clip_width_um = max(float(candidate.clip_bbox[2]) - float(candidate.clip_bbox[0]), 1e-12)
    clip_height_um = max(float(candidate.clip_bbox[3]) - float(candidate.clip_bbox[1]), 1e-12)
    shift_ref_um = max(0.5 * min(clip_width_um, clip_height_um), 1e-12)
    shift_score = _clamp01(1.0 - abs(float(candidate.shift_distance_um)) / shift_ref_um)
    return _clamp01(
        0.30 * completeness_score
        + 0.25 * margin_score
        + 0.15 * context_density_score
        + 0.10 * shift_score
    )


def _candidate_opc_center_score(candidate, coverage_weight=0, weight_denominator=1):
    """综合几何基础分与 repeat support，估计 candidate center 的 OPC 代表性。"""

    repeat_support_score = _clamp01(_safe_ratio(int(coverage_weight), max(int(weight_denominator), 1)))
    return _clamp01(_candidate_opc_center_base_score(candidate) + 0.20 * repeat_support_score)


def _candidate_representative_score(candidate, coverage_weight, weight_denominator):
    """综合 OPC center、coverage 支持和风险，生成代表点评分。"""

    coverage_score = _clamp01(_safe_ratio(int(coverage_weight), max(int(weight_denominator), 1)))
    return _clamp01(
        0.45 * _candidate_opc_center_score(candidate, coverage_weight=int(coverage_weight), weight_denominator=int(weight_denominator))
        + 0.35 * coverage_score
        + 0.20 * _candidate_risk_score(candidate)
    )


def _candidate_greedy_tiebreak(candidate):
    """生成 greedy set cover 在评分之后使用的稳定 tie-break。"""

    return (
        0 if str(candidate.shift_direction) == "base" else 1,
        abs(float(candidate.shift_distance_um)),
        int(candidate.origin_exact_cluster_id),
        str(candidate.candidate_id),
    )


def _sample_exact_id_pairs(exact_ids, limit):
    """从 exact id 列表中稳定抽取少量 pair，避免质量检查退化成全量二次复杂度。"""

    ids = sorted(int(value) for value in exact_ids)
    if len(ids) < 2 or int(limit) <= 0:
        return []
    total_pairs = len(ids) * (len(ids) - 1) // 2
    if total_pairs <= int(limit):
        return [(ids[left], ids[right]) for left in range(len(ids) - 1) for right in range(left + 1, len(ids))]
    pairs = []
    seen = set()
    stride_a = max(1, len(ids) // 17)
    stride_b = max(1, len(ids) // 29)
    attempt = 0
    while len(pairs) < int(limit) and attempt < int(limit) * 20:
        left = (attempt * stride_a + attempt // 3) % len(ids)
        gap = 1 + ((attempt * stride_b + len(ids) // 3) % (len(ids) - 1))
        right = (left + gap) % len(ids)
        if left != right:
            pair = (ids[min(left, right)], ids[max(left, right)])
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
        attempt += 1
    return pairs


def _empty_greedy_purity_gate_stats():
    """返回 greedy purity gate 的 stage1/stage2 调试计数器。"""

    return {
        "greedy_purity_gate_candidate_count": 0,
        "greedy_purity_gate_sampled_pair_count": 0,
        "greedy_purity_gate_fail_pair_count": 0,
        "greedy_purity_gate_reject_count": 0,
        "greedy_purity_gate_stage1_pass_count": 0,
        "greedy_purity_gate_stage1_reject_count": 0,
        "greedy_purity_gate_stage2_count": 0,
        "greedy_purity_gate_stage2_reject_count": 0,
        "greedy_purity_gate_seconds": 0.0,
    }


def _ensure_greedy_purity_gate_stats(config):
    """把 greedy purity gate 计数器挂到当前 config，供 result summary 汇总。"""

    if not isinstance(config, dict):
        return _empty_greedy_purity_gate_stats()
    stats = config.get("_greedy_purity_gate_stats")
    if not isinstance(stats, dict):
        stats = _empty_greedy_purity_gate_stats()
        config["_greedy_purity_gate_stats"] = stats
    return stats


def _record_gate_stat(stats, key, amount=1):
    """递增 greedy purity gate 计数器。"""

    stats[str(key)] = stats.get(str(key), 0) + amount


def _exact_pair_matches_by_base(left_exact_id, right_exact_id, exact_by_id, base_candidate_by_exact, config, descriptor_cache):
    """用两个 exact cluster 的 base candidate 做双向 strict witness 验证。"""

    left_candidate = base_candidate_by_exact.get(int(left_exact_id))
    right_candidate = base_candidate_by_exact.get(int(right_exact_id))
    left_cluster = exact_by_id.get(int(left_exact_id))
    right_cluster = exact_by_id.get(int(right_exact_id))
    if left_candidate is None or right_candidate is None or left_cluster is None or right_cluster is None:
        return None
    left_matches, _, _ = _candidate_matches_exact(left_candidate, right_cluster, config, descriptor_cache)
    if left_matches:
        return True
    right_matches, _, _ = _candidate_matches_exact(right_candidate, left_cluster, config, descriptor_cache)
    return bool(right_matches)


def _sample_gate_pairs(sampled_pairs, exact_by_id, base_candidate_by_exact, config, descriptor_cache):
    """统计一批 gate 抽样 pair 的 strict verification 失败数。"""

    checked = 0
    failed = 0
    for left_exact_id, right_exact_id in sampled_pairs:
        matched = _exact_pair_matches_by_base(
            int(left_exact_id),
            int(right_exact_id),
            exact_by_id,
            base_candidate_by_exact,
            config,
            descriptor_cache,
        )
        if matched is None:
            continue
        checked += 1
        if not matched:
            failed += 1
    return int(checked), int(failed)


def _candidate_shift_distance(candidate_or_metadata):
    """兼容 CandidateClip 和 metadata dict 读取 shift 距离。"""

    if isinstance(candidate_or_metadata, dict):
        return float(candidate_or_metadata.get("shift_distance_um", 0.0))
    return float(getattr(candidate_or_metadata, "shift_distance_um", 0.0))


def _greedy_purity_gate_passes(candidate_or_metadata, covered_ids, weights, total_weight, exact_by_id, base_candidate_by_exact, config, stats, descriptor_cache):
    """执行 v1 stage1/stage2 greedy purity gate，避免高风险候选过度合并。"""

    gate_started = time.perf_counter()
    exact_count = int(len(covered_ids))
    try:
        if exact_count < 2:
            return True
        covered_weight = int(sum(weights.get(int(exact_id), 0) for exact_id in covered_ids))
        weight_ratio = _safe_ratio(covered_weight, total_weight)
        shift_ratio = _safe_ratio(
            abs(_candidate_shift_distance(candidate_or_metadata)),
            max(float(config.get("clip_size_um", 1.35)), 1e-12),
        )
        needs_gate = (
            exact_count >= GREEDY_PURITY_GATE_EXACT_TRIGGER
            or weight_ratio >= GREEDY_PURITY_GATE_WEIGHT_RATIO
            or (
                shift_ratio > GREEDY_PURITY_GATE_LARGE_SHIFT_RATIO
                and exact_count >= GREEDY_PURITY_GATE_LARGE_SHIFT_EXACT_TRIGGER
            )
        )
        if not needs_gate:
            return True
        stage1_pairs = _sample_exact_id_pairs(covered_ids, GREEDY_PURITY_GATE_STAGE1_SAMPLE_PAIRS)
        if not stage1_pairs:
            return True
        checked_count, fail_count = _sample_gate_pairs(
            stage1_pairs,
            exact_by_id,
            base_candidate_by_exact,
            config,
            descriptor_cache,
        )
        if checked_count <= 0:
            return True
        _record_gate_stat(stats, "greedy_purity_gate_candidate_count", 1)
        fail_rate = _safe_ratio(fail_count, checked_count)
        if fail_rate >= GREEDY_PURITY_GATE_STAGE1_REJECT_FAIL_RATE:
            _record_gate_stat(stats, "greedy_purity_gate_stage1_reject_count", 1)
            _record_gate_stat(stats, "greedy_purity_gate_reject_count", 1)
            _record_gate_stat(stats, "greedy_purity_gate_sampled_pair_count", int(checked_count))
            _record_gate_stat(stats, "greedy_purity_gate_fail_pair_count", int(fail_count))
            return False
        if fail_rate <= GREEDY_PURITY_GATE_STAGE1_PASS_FAIL_RATE:
            _record_gate_stat(stats, "greedy_purity_gate_stage1_pass_count", 1)
            _record_gate_stat(stats, "greedy_purity_gate_sampled_pair_count", int(checked_count))
            _record_gate_stat(stats, "greedy_purity_gate_fail_pair_count", int(fail_count))
            return True

        _record_gate_stat(stats, "greedy_purity_gate_stage2_count", 1)
        stage1_seen = set((int(left), int(right)) for left, right in stage1_pairs)
        stage2_pairs = [
            (int(left), int(right))
            for left, right in _sample_exact_id_pairs(covered_ids, GREEDY_PURITY_GATE_STAGE2_SAMPLE_PAIRS)
            if (int(left), int(right)) not in stage1_seen
        ]
        stage2_checked, stage2_failed = _sample_gate_pairs(
            stage2_pairs,
            exact_by_id,
            base_candidate_by_exact,
            config,
            descriptor_cache,
        )
        checked_count += int(stage2_checked)
        fail_count += int(stage2_failed)
        _record_gate_stat(stats, "greedy_purity_gate_sampled_pair_count", int(checked_count))
        _record_gate_stat(stats, "greedy_purity_gate_fail_pair_count", int(fail_count))
        fail_rate = _safe_ratio(fail_count, checked_count)
        if fail_rate >= GREEDY_PURITY_GATE_STAGE2_REJECT_FAIL_RATE:
            _record_gate_stat(stats, "greedy_purity_gate_stage2_reject_count", 1)
            _record_gate_stat(stats, "greedy_purity_gate_reject_count", 1)
            return False
        return True
    finally:
        stats["greedy_purity_gate_seconds"] = float(stats.get("greedy_purity_gate_seconds", 0.0)) + (
            time.perf_counter() - gate_started
        )


def _overmerge_score_and_reason(
    cluster_weight,
    total_weight,
    exact_cluster_count,
    representative_seed_type,
    shift_distance_um,
    clip_size_um,
    representative_visual_pass_ratio,
    pairwise_geometry_purity,
    pairwise_geometry_sample_status,
):
    """根据规模、shift、seed 类型与 purity 采样结果计算 over-merge 风险。"""

    exact_count = int(exact_cluster_count)
    weight_ratio = _safe_ratio(int(cluster_weight), max(int(total_weight), 1))
    shift_ratio = _safe_ratio(abs(float(shift_distance_um)), max(float(clip_size_um), 1e-12))
    sampled = str(pairwise_geometry_sample_status) == "sampled" and pairwise_geometry_purity is not None
    pairwise_badness = 1.0 - float(pairwise_geometry_purity) if sampled else 0.0
    high_visual_purity = float(representative_visual_pass_ratio) >= float(QUALITY_LOW_VISUAL_PURITY_THRESHOLD)
    score = (
        0.35 * min(_safe_ratio(exact_count, OVERMERGE_EXACT_COUNT_TRIGGER), 1.0)
        + 0.25 * min(_safe_ratio(weight_ratio, OVERMERGE_WEIGHT_RATIO_TRIGGER), 1.0)
        + 0.25 * _clamp01(pairwise_badness)
        + 0.10 * (1.0 if str(representative_seed_type) == SEED_TYPE_LONG else 0.0)
        + 0.05 * min(_safe_ratio(shift_ratio, OVERMERGE_LARGE_SHIFT_RATIO), 1.0)
    )
    reasons = []
    if exact_count >= OVERMERGE_EXACT_COUNT_TRIGGER and not high_visual_purity:
        reasons.append("large_exact_count")
    if weight_ratio >= OVERMERGE_WEIGHT_RATIO_TRIGGER and not high_visual_purity:
        reasons.append("large_weight_ratio")
    if sampled and float(pairwise_geometry_purity) <= float(LOW_PAIRWISE_REVIEW_THRESHOLD):
        reasons.append("low_pairwise_geometry_purity")
    if str(representative_seed_type) == SEED_TYPE_LONG and not high_visual_purity:
        reasons.append("long_shape_path")
    if shift_ratio > OVERMERGE_LARGE_SHIFT_RATIO and not high_visual_purity:
        reasons.append("large_shift")
    return float(_clamp01(score)), "|".join(reasons) if reasons else "ok"


def _review_merge_confidence(candidate_seed_type, target_seed_type, shift_distance_um):
    """按 seed type 与 shift 距离给 review merge 边做固定分层。"""

    candidate_type = str(candidate_seed_type or "unknown")
    target_type = str(target_seed_type or "unknown")
    shift_distance = float(abs(float(shift_distance_um)))
    low_reasons = []
    if candidate_type == SEED_TYPE_LONG:
        low_reasons.append("long_shape_candidate")
    if candidate_type != target_type:
        low_reasons.append("seed_type_mismatch")
    if shift_distance > float(REVIEW_MERGE_MEDIUM_SHIFT_DISTANCE_UM):
        low_reasons.append("large_shift")
    if low_reasons:
        return "low", "|".join(low_reasons)
    if shift_distance <= float(REVIEW_MERGE_HIGH_SHIFT_DISTANCE_UM):
        return "high", "same_seed_small_shift"
    return "medium", "same_seed_medium_shift"


def _review_merge_confidence_rank(confidence_tier):
    """返回 review merge 分层排序权重，high 最靠前。"""

    return {"high": 0, "medium": 1, "low": 2}.get(str(confidence_tier), 3)


def _sorted_review_merge_candidate_rows(rows):
    """按分层、权重、singleton 优先级和 shift 距离排序 review merge 候选。"""

    return sorted(
        (dict(row) for row in rows),
        key=lambda item: (
            _review_merge_confidence_rank(str(item.get("confidence_tier", ""))),
            -int(item.get("edge_weight", 0) or 0),
            -int(bool(item.get("target_is_singleton", False))),
            float(item.get("shift_distance_um", 0.0) or 0.0),
            int(item.get("source_exact_cluster_id", 0) or 0),
            int(item.get("target_exact_cluster_id", 0) or 0),
            str(item.get("candidate_id", "")),
        ),
    )


def _tier_balanced_review_merge_candidate_rows(rows):
    """按 high/medium/low 固定配额保留 review merge 候选，避免 high tier 独占内部证据。"""

    sorted_rows = _sorted_review_merge_candidate_rows(rows)
    total_limit = max(0, int(REVIEW_MERGE_CANDIDATE_TOP_N))
    selected_rows = []
    for tier in ("high", "medium", "low"):
        quota = max(0, int(dict(REVIEW_MERGE_CANDIDATE_TIER_QUOTAS).get(tier, 0)))
        if quota <= 0:
            continue
        tier_rows = [dict(row) for row in sorted_rows if str(row.get("confidence_tier", "")) == tier]
        selected_rows.extend(tier_rows[:quota])
    return selected_rows[:total_limit]


def _append_bounded_review_merge_row(rows, row):
    """流式保留 safe merge 需要的 top N review 候选，避免大版图无界攒 dict。"""

    rows.append(row)
    prune_threshold = max(int(REVIEW_MERGE_CANDIDATE_TOP_N) * 2, int(REVIEW_MERGE_CANDIDATE_TOP_N) + 1)
    if len(rows) > prune_threshold:
        rows[:] = _tier_balanced_review_merge_candidate_rows(rows)


def _maybe_float(value):
    """把可选数值转成 float，空值保持为 None。"""

    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_low_shift_low_pairwise_review(shift_distance_um, pairwise_geometry_purity, pairwise_geometry_sample_status, sampled_pair_count=0):
    """判断 cluster 是否属于低 shift + 低 pairwise 的真实 over-merge review 上界。"""

    purity = _maybe_float(pairwise_geometry_purity)
    shift_distance = _maybe_float(shift_distance_um)
    if purity is None or shift_distance is None:
        return False
    sampled = str(pairwise_geometry_sample_status) == "sampled" or int(sampled_pair_count or 0) > 0
    return (
        sampled
        and abs(float(shift_distance)) <= float(LOW_SHIFT_MAX_UM)
        and float(purity) <= float(LOW_PAIRWISE_REVIEW_THRESHOLD)
    )


def _cluster_pair_key(row):
    """把 review merge 行归一成无向 final cluster pair。"""

    left = int(row.get("source_cluster_id", 0) or 0)
    right = int(row.get("target_cluster_id", 0) or 0)
    return (min(left, right), max(left, right))


def _endpoint_quality_by_cluster_id(cluster_payloads, cluster_quality_by_index):
    """把 cluster payload 和质量指标转成 final cluster id 索引。"""

    quality_by_id = {}
    for payload in cluster_payloads:
        cluster_index = int(payload.get("cluster_index", 0))
        cluster_id = int(cluster_index) + 1
        metrics = dict(cluster_quality_by_index.get(cluster_index, {}) or {})
        quality_by_id[cluster_id] = {
            "cluster_weight": int(payload.get("cluster_weight", 0) or 0),
            "exact_cluster_count": int(len(payload.get("exact_ids", []) or [])),
            "shift_distance_um": float(payload.get("shift_distance_um", 0.0) or 0.0),
            "pairwise_geometry_purity": metrics.get("pairwise_geometry_purity"),
            "pairwise_geometry_sample_status": str(metrics.get("pairwise_geometry_sample_status", "unknown")),
            "pairwise_geometry_sampled_pair_count": int(metrics.get("pairwise_geometry_sampled_pair_count", 0) or 0),
            "overmerge_reason": str(metrics.get("overmerge_reason", "ok") or "ok"),
        }
    return quality_by_id


def _pairwise_endpoint_is_low_quality(endpoint_quality):
    """判断 pair 端点是否触及低 pairwise 质量。"""

    purity = _maybe_float(endpoint_quality.get("pairwise_geometry_purity"))
    if purity is None:
        return False
    sampled = (
        str(endpoint_quality.get("pairwise_geometry_sample_status", "unknown")) == "sampled"
        or int(endpoint_quality.get("pairwise_geometry_sampled_pair_count", 0) or 0) > 0
    )
    return bool(sampled and float(purity) <= float(LOW_PAIRWISE_REVIEW_THRESHOLD))


def _review_merge_pair_bucket(pair_row, source_quality, target_quality):
    """按端点质量和候选置信度给 cluster pair 分桶，方便 safe recall merge 只吃低风险边。"""

    if int(pair_row.get("high_conf_row_count", 0) or 0) <= 0:
        return "low_confidence_candidate"
    if (
        abs(float(source_quality.get("shift_distance_um", 0.0) or 0.0)) > float(LOW_SHIFT_MAX_UM)
        or abs(float(target_quality.get("shift_distance_um", 0.0) or 0.0)) > float(LOW_SHIFT_MAX_UM)
    ):
        return "high_shift_touching_candidate"
    if _pairwise_endpoint_is_low_quality(source_quality) or _pairwise_endpoint_is_low_quality(target_quality):
        return "overmerge_touching_candidate"
    return "safe_recall_candidate"


def _sorted_review_merge_cluster_pair_rows(rows):
    """按 pair 权重和稳定 id 排序 pair 级 review merge 证据。"""

    return sorted(
        (dict(row) for row in rows),
        key=lambda item: (
            -int(item.get("pair_edge_weight_sum", 0) or 0),
            -int(item.get("high_conf_edge_weight_sum", 0) or 0),
            -int(item.get("row_count", 0) or 0),
            int(item.get("source_cluster_id", 0) or 0),
            int(item.get("target_cluster_id", 0) or 0),
        ),
    )


def _review_merge_cluster_pair_rows(review_merge_rows, endpoint_quality_by_id):
    """把 bounded review merge 行聚合成 safe merge 使用的 cluster-pair 级证据。"""

    pair_state = {}
    for row in review_merge_rows:
        source_cluster_id, target_cluster_id = _cluster_pair_key(row)
        if source_cluster_id <= 0 or target_cluster_id <= 0 or source_cluster_id == target_cluster_id:
            continue
        state = pair_state.setdefault(
            (source_cluster_id, target_cluster_id),
            {
                "source_cluster_id": int(source_cluster_id),
                "target_cluster_id": int(target_cluster_id),
                "pair_edge_weight_sum": 0,
                "row_count": 0,
                "candidate_ids": set(),
                "singleton_target_row_count": 0,
                "tier_row_counts": Counter(),
                "tier_edge_weights": Counter(),
            },
        )
        edge_weight = max(int(row.get("edge_weight", 0) or 0), 0)
        tier = str(row.get("confidence_tier", "") or "")
        state["pair_edge_weight_sum"] = int(state["pair_edge_weight_sum"]) + int(edge_weight)
        state["row_count"] = int(state["row_count"]) + 1
        state["candidate_ids"].add(str(row.get("candidate_id", "")))
        if bool(row.get("target_is_singleton", False)):
            state["singleton_target_row_count"] = int(state["singleton_target_row_count"]) + 1
        state["tier_row_counts"][tier] += 1
        state["tier_edge_weights"][tier] += int(edge_weight)

    pair_rows = []
    for (source_cluster_id, target_cluster_id), state in pair_state.items():
        source_quality = dict(endpoint_quality_by_id.get(int(source_cluster_id), {}) or {})
        target_quality = dict(endpoint_quality_by_id.get(int(target_cluster_id), {}) or {})
        row = {
            "source_cluster_id": int(source_cluster_id),
            "target_cluster_id": int(target_cluster_id),
            "pair_edge_weight_sum": int(state["pair_edge_weight_sum"]),
            "row_count": int(state["row_count"]),
            "unique_candidate_count": int(len(state["candidate_ids"])),
            "singleton_target_row_count": int(state["singleton_target_row_count"]),
            "high_conf_row_count": int(state["tier_row_counts"].get("high", 0)),
            "medium_conf_row_count": int(state["tier_row_counts"].get("medium", 0)),
            "low_conf_row_count": int(state["tier_row_counts"].get("low", 0)),
            "high_conf_edge_weight_sum": int(state["tier_edge_weights"].get("high", 0)),
            "medium_conf_edge_weight_sum": int(state["tier_edge_weights"].get("medium", 0)),
            "low_conf_edge_weight_sum": int(state["tier_edge_weights"].get("low", 0)),
            "source_cluster_weight": int(source_quality.get("cluster_weight", 0) or 0),
            "target_cluster_weight": int(target_quality.get("cluster_weight", 0) or 0),
            "source_exact_cluster_count": int(source_quality.get("exact_cluster_count", 0) or 0),
            "target_exact_cluster_count": int(target_quality.get("exact_cluster_count", 0) or 0),
            "source_shift_distance_um": float(source_quality.get("shift_distance_um", 0.0) or 0.0),
            "target_shift_distance_um": float(target_quality.get("shift_distance_um", 0.0) or 0.0),
            "source_pairwise_geometry_purity": source_quality.get("pairwise_geometry_purity"),
            "target_pairwise_geometry_purity": target_quality.get("pairwise_geometry_purity"),
            "source_overmerge_reason": str(source_quality.get("overmerge_reason", "ok") or "ok"),
            "target_overmerge_reason": str(target_quality.get("overmerge_reason", "ok") or "ok"),
        }
        row["pair_review_bucket"] = _review_merge_pair_bucket(row, source_quality, target_quality)
        pair_rows.append(row)
    return _sorted_review_merge_cluster_pair_rows(pair_rows)


CSV_OUTPUT_COLUMNS = [
    "groupID",
    "cluster_id",
    "center_x_um",
    "center_y_um",
    "clip_size",
    "group_weight",
    "risk_score",
    "risk_rank",
]

CLUSTER_REPRESENTATIVE_CSV_COLUMNS = [
    "cluster_id",
    "center_x_um",
    "center_y_um",
    "clip_size",
    "cluster_size",
    "cluster_weight",
    "exact_cluster_count",
    "representative_seed_type",
    "shift_direction",
    "shift_distance_um",
    "representative_score",
    "opc_center_score",
    "risk_score",
    "risk_rank",
]

CLUSTER_REPRESENTATIVE_QUALITY_COLUMNS = [
    "representative_visual_pass_ratio",
    "representative_visual_fail_count",
    "representative_visual_checked_count",
    "representative_visual_sample_status",
    "pairwise_geometry_purity",
    "pairwise_geometry_fail_count",
    "pairwise_geometry_sampled_pair_count",
    "pairwise_geometry_sample_status",
    "overmerge_score",
    "overmerge_reason",
]


def _csv_scalar(value):
    """把 CSV 单元格中的空值和 numpy 标量归一成 csv 可写值。"""

    if value is None:
        return ""
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _final_cluster_id_by_exact_id(result):
    """建立 exact cluster 到最终 1-based cluster_id 的映射。"""

    mapping = {}
    for cluster in result.get("clusters", []):
        final_cluster_id = int(cluster.get("cluster_id", 0)) + 1
        for exact_id in cluster.get("exact_cluster_ids", []):
            mapping[int(exact_id)] = int(final_cluster_id)
    return mapping


def _cluster_representative_csv_columns(include_quality_metrics):
    """返回当前 representative CSV 应使用的列顺序。"""

    columns = list(CLUSTER_REPRESENTATIVE_CSV_COLUMNS)
    if include_quality_metrics:
        columns.extend(CLUSTER_REPRESENTATIVE_QUALITY_COLUMNS)
    return columns


def _csv_exact_cluster_review_row(group_id, cluster_id, exact_cluster, clip_size_um, group_weight, risk_score, risk_rank):
    """把单个 marker-defined exact cluster 转成主 review CSV 行。"""

    center = list(getattr(exact_cluster.representative, "marker_center", ()) or [])
    row = {
        "groupID": int(group_id),
        "cluster_id": int(cluster_id),
        "center_x_um": _csv_scalar(center[0] if len(center) > 0 else ""),
        "center_y_um": _csv_scalar(center[1] if len(center) > 1 else ""),
        "clip_size": float(clip_size_um),
        "group_weight": int(group_weight),
        "risk_score": float(risk_score),
        "risk_rank": int(risk_rank),
    }
    return dict((column, _csv_scalar(row.get(column))) for column in CSV_OUTPUT_COLUMNS)


def _csv_cluster_representative_row(row, columns=None):
    """把 cluster representative 结果行规整成固定列顺序。"""

    ordered_columns = list(columns) if columns is not None else list(CLUSTER_REPRESENTATIVE_CSV_COLUMNS)
    return dict((column, _csv_scalar(row.get(column))) for column in ordered_columns)


def _exact_cluster_weight(exact_cluster):
    """返回 exact cluster 的权重，优先使用成员 seed_weight 累计。"""

    return int(max(1, int(getattr(exact_cluster, "weight", 1))))


def _assign_risk_ranks(rows, id_column):
    """按 risk_score 降序为输出行写入稳定的 risk_rank。"""

    ranked = sorted(rows, key=lambda row: (-float(row.get("risk_score", 0.0)), int(row.get(id_column, 0))))
    for rank, row in enumerate(ranked, start=1):
        row["risk_rank"] = int(rank)


def _bitmap_ecc_match_reason(bitmap_a, bitmap_b, edge_tolerance_um, pixel_size_um):
    """执行 ECC 几何匹配，并返回细分拒绝原因。"""

    if bitmap_a.shape != bitmap_b.shape:
        return False, "geometry_shape_mismatch"
    if not bitmap_a.any() and not bitmap_b.any():
        return True, "geometry_verified"
    if not bitmap_a.any() or not bitmap_b.any():
        return False, "geometry_empty_mismatch"
    tol_px = max(0, int(math.ceil(float(edge_tolerance_um) / max(float(pixel_size_um), 1e-12) - 1e-12)))
    if tol_px <= 0:
        if bool(np.array_equal(bitmap_a, bitmap_b)):
            return True, "geometry_verified"
        return False, "geometry_exact_mismatch"
    structure = np.ones((2 * tol_px + 1, 2 * tol_px + 1), dtype=bool)
    dilated_a = ndimage.binary_dilation(bitmap_a, structure=structure)
    dilated_b = ndimage.binary_dilation(bitmap_b, structure=structure)
    area_a = max(float(np.count_nonzero(bitmap_a)), 1.0)
    area_b = max(float(np.count_nonzero(bitmap_b)), 1.0)
    residual_a = np.count_nonzero(bitmap_a & ~dilated_b) / area_a
    residual_b = np.count_nonzero(bitmap_b & ~dilated_a) / area_b
    if residual_a > ECC_RESIDUAL_RATIO:
        return False, "geometry_residual_candidate"
    if residual_b > ECC_RESIDUAL_RATIO:
        return False, "geometry_residual_target"
    eroded_a = ndimage.binary_erosion(bitmap_a, structure=structure, border_value=0)
    eroded_b = ndimage.binary_erosion(bitmap_b, structure=structure, border_value=0)
    donut_a = dilated_a & ~eroded_a
    donut_b = dilated_b & ~eroded_b
    donut_area_a = int(np.count_nonzero(donut_a))
    donut_area_b = int(np.count_nonzero(donut_b))
    if donut_area_a == 0 or donut_area_b == 0:
        if not bool(ECC_ALLOW_DONUT_AUTO_PASS):
            return True, "geometry_donut_degenerate"
        return True, "geometry_verified"
    overlap = int(np.count_nonzero(donut_a & donut_b))
    denom = max(min(donut_area_a, donut_area_b), 1)
    if float(overlap) / float(denom) >= ECC_DONUT_OVERLAP_RATIO:
        return True, "geometry_verified"
    return False, "geometry_donut_overlap"


def _geometry_match_reason(candidate, witness, config):
    """执行 final verification 几何判定，并返回细分原因。"""

    candidate_bitmap = np.asarray(candidate.clip_bitmap, dtype=bool)
    witness_bitmap = np.asarray(witness.clip_bitmap, dtype=bool)
    if candidate_bitmap.shape != witness_bitmap.shape:
        return False, "geometry_shape_mismatch"
    if str(config.get("geometry_match_mode", "ecc")).lower() == "acc":
        xor_ratio = float(np.count_nonzero(candidate_bitmap ^ witness_bitmap)) / float(max(candidate_bitmap.size, 1))
        if xor_ratio <= max(0.0, 1.0 - float(config.get("area_match_ratio", 0.96))) + 1e-12:
            return True, "geometry_verified"
        return False, "geometry_acc_xor"
    pixel_size_um = float(int(config.get("pixel_size_nm", DEFAULT_PIXEL_SIZE_NM))) / 1000.0
    return _bitmap_ecc_match_reason(
        candidate_bitmap,
        witness_bitmap,
        float(config.get("edge_tolerance_um", 0.02)),
        pixel_size_um,
    )


def _descriptor_for_candidate(candidate, descriptor_cache):
    """读取或计算 selected candidate 的 full graph descriptor。"""

    key = str(getattr(candidate, "candidate_id", id(candidate)))
    descriptor = descriptor_cache.get(key)
    if descriptor is None:
        descriptor = _graph_bitmap_descriptor(candidate.clip_bitmap)
        descriptor_cache[key] = descriptor
    return descriptor


def _candidate_matches_witness(candidate, witness, config, descriptor_cache):
    """判断 selected candidate 是否能通过单个 target witness 的严格验证。"""

    if str(candidate.clip_hash) == str(witness.clip_hash):
        return True, "exact_hash"

    geometry_ok, geometry_reason = _geometry_match_reason(candidate, witness, config)
    if not geometry_ok:
        return False, geometry_reason

    candidate_desc = _descriptor_for_candidate(candidate, descriptor_cache)
    witness_desc = _graph_bitmap_descriptor(witness.clip_bitmap)
    return _strict_graph_gate_reason(candidate_desc, witness_desc)


def _dominant_reject_reason(reasons):
    """从多个 witness 失败原因中挑出最有诊断价值的最终拒绝原因。"""

    reason_counts = Counter(str(reason) for reason in reasons if reason)
    if not reason_counts:
        return "unknown"
    graph_reasons = [reason for reason in reason_counts if str(reason).startswith("graph_")]
    if graph_reasons:
        return max(sorted(graph_reasons), key=lambda reason: reason_counts[reason])
    for reason in GEOMETRY_REJECT_REASON_ORDER:
        if reason in reason_counts:
            return reason
    return max(sorted(reason_counts), key=lambda reason: reason_counts[reason])


def _target_witness_candidates(exact_cluster, config):
    """为 target exact cluster 临时生成 witness candidates。"""

    return generate_candidates_for_cluster(exact_cluster, config)


def _candidate_matches_exact(candidate, exact_cluster, config, descriptor_cache):
    """使用统一 shift-witness 语义判断 candidate 是否覆盖某个 exact cluster。"""

    reject_reasons = []
    for witness in _target_witness_candidates(exact_cluster, config):
        matched, reason = _candidate_matches_witness(candidate, witness, config, descriptor_cache)
        if matched:
            return True, reason, str(witness.shift_direction)
        reject_reasons.append(str(reason))
    return False, _dominant_reject_reason(reject_reasons), "none"


def distance_worst_case_proxy(bitmap):
    """用距离变换估计窄线宽、窄间距带来的代表样本风险。"""

    if bitmap is None:
        return 0.0
    mask = np.asarray(bitmap, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        return 0.0

    width_dist = ndimage.distance_transform_edt(mask)
    width_values = width_dist[mask]
    width_risk = float(np.mean(1.0 / (width_values + 1.0))) if width_values.size else 0.0

    space_values = np.empty(0, dtype=np.float64)
    if np.any(~mask):
        space_dist = ndimage.distance_transform_edt(~mask)
        space_values = space_dist[~mask]
    space_risk = float(np.mean(1.0 / (space_values + 1.0))) if space_values.size else 0.0

    edge_count = int(np.count_nonzero(mask[:, 1:] != mask[:, :-1])) + int(np.count_nonzero(mask[1:, :] != mask[:-1, :]))
    active_count = max(int(np.count_nonzero(mask)), 1)
    edge_risk = min(float(edge_count) / float(active_count * 4), 1.0)
    return float(0.45 * width_risk + 0.35 * space_risk + 0.20 * edge_risk)


def _empty_coverage_detail_seconds():
    """返回 coverage 内部分阶段耗时，字段对齐 optimized_v1。"""

    return {
        "light_bundle_build": 0.0,
        "shortlist_index": 0.0,
        "prefilter": 0.0,
        "full_descriptor_cache": 0.0,
        "full_prefilter": 0.0,
        "geometry_cache": 0.0,
        "geometry_match": 0.0,
    }


def _empty_coverage_debug_stats():
    """返回 v1 风格 coverage 诊断计数器。"""

    return {
        "bundle_count": 0,
        "max_bundle_group_count": 0,
        "candidate_group_count": 0,
        "geometry_pair_count": 0,
        "geometry_pass": 0,
        "geometry_reject": 0,
        "exact_hash_pass": 0,
        "exact_hash_pairs": 0,
        "cheap_reject": 0,
        "cheap_fill_reject": 0,
        "cheap_area_density_reject": 0,
        "full_prefilter_reject": 0,
        "invariant_reject": 0,
        "topology_reject": 0,
        "signature_reject": 0,
        "full_descriptor_cache_group_count": 0,
        "geometry_cache_group_count": 0,
        "geometry_dilated_cache_group_count": 0,
        "geometry_donut_cache_group_count": 0,
        "shortlist_subgroup_count": 0,
        "shortlist_max_subgroup_size": 0,
        "shortlist_exact_subgroup_count": 0,
        "shortlist_hnsw_subgroup_count": 0,
        "full_prefilter_probe_pair_count": 0,
        "full_prefilter_probe_reject_count": 0,
        "full_prefilter_disabled_bundle_count": 0,
        "precomputed_cheap_descriptor_count": 0,
        "precomputed_packed_bitmap_count": 0,
        "shortlist_precomputed_bundle_count": 0,
        "xor_reject": 0,
        "skipped_self_or_existing": 0,
        "donut_auto_pass_pair_count": 0,
        "donut_auto_pass_source_shift_counts": {},
        "donut_auto_pass_source_seed_type_counts": {},
        "donut_degenerate_strict_graph_pass_count": 0,
        "donut_degenerate_strict_graph_reject_count": 0,
        "long_shape_cross_seed_guard_pair_count": 0,
        "long_shape_cross_seed_guard_pass_count": 0,
        "long_shape_cross_seed_guard_reject_count": 0,
        "greedy_purity_gate_candidate_count": 0,
        "greedy_purity_gate_sampled_pair_count": 0,
        "greedy_purity_gate_fail_pair_count": 0,
        "greedy_purity_gate_reject_count": 0,
        "greedy_purity_gate_stage1_pass_count": 0,
        "greedy_purity_gate_stage1_reject_count": 0,
        "greedy_purity_gate_stage2_count": 0,
        "greedy_purity_gate_stage2_reject_count": 0,
        "greedy_purity_gate_seconds": 0.0,
    }


def _coverage_group_from_candidates(candidates):
    """把严格 bitmap 相同的 candidates 合成一个 coverage group。"""

    grouped = {}
    for candidate in candidates:
        strict_key = _bitmap_exact_key(candidate.clip_bitmap)
        bucket = grouped.get(strict_key)
        if bucket is None:
            bucket = {
                "representative": candidate,
                "candidates": [],
                "origin_ids": set(),
                "strict_key": strict_key,
            }
            grouped[strict_key] = bucket
        bucket["candidates"].append(candidate)
        bucket["origin_ids"].add(int(candidate.origin_exact_cluster_id))
    return list(grouped.values())


def create_candidate_bundle_accumulator():
    """创建 candidate bundle 增量累加器，避免 prepare-coverage 常驻全量 candidate 对象。"""

    return {"shape_groups": {}, "candidate_count": 0}


def add_candidates_to_candidate_bundle_accumulator(accumulator, candidates):
    """把一批 candidates 增量合并进 bundle 累加器。"""

    shape_groups = accumulator.setdefault("shape_groups", {})
    for candidate in candidates:
        shape = tuple(int(value) for value in candidate.clip_bitmap.shape)
        strict_key = _bitmap_exact_key(candidate.clip_bitmap)
        groups = shape_groups.get(shape)
        if groups is None:
            groups = {}
            shape_groups[shape] = groups
        bucket = groups.get(strict_key)
        if bucket is None:
            bucket = {
                "representative": candidate,
                "origin_ids": set(),
                "strict_key": strict_key,
                "candidate_ids": [],
                "candidate_count": 0,
            }
            groups[strict_key] = bucket
        bucket["origin_ids"].add(int(candidate.origin_exact_cluster_id))
        bucket["candidate_ids"].append(str(candidate.candidate_id))
        bucket["candidate_count"] = int(bucket.get("candidate_count", 0)) + 1
        accumulator["candidate_count"] = int(accumulator.get("candidate_count", 0)) + 1
    return accumulator


def candidate_bundles_from_accumulator(accumulator):
    """把增量累加器转成统一的 bundle writer 输入结构。"""

    bundles = {}
    for shape in sorted(accumulator.get("shape_groups", {})):
        groups = accumulator["shape_groups"][shape]
        bundle = {"shape": tuple(int(value) for value in shape), "candidate_groups": list(groups.values())}
        bundles[tuple(int(value) for value in shape)] = _finalize_candidate_bundle(bundle)
    return bundles


def _finalize_candidate_bundle(bundle):
    """补齐 bundle 的数组索引和缓存字段。"""

    areas = []
    hashes = []
    strict_key_to_index = {}
    hash_to_indices = {}
    for idx, group in enumerate(bundle["candidate_groups"]):
        representative = group["representative"]
        strict_key = group.get("strict_key")
        if strict_key is None:
            strict_key = _bitmap_exact_key(representative.clip_bitmap)
            group["strict_key"] = strict_key
        areas.append(int(np.count_nonzero(representative.clip_bitmap)))
        hashes.append(str(representative.clip_hash))
        strict_key_to_index[strict_key] = int(idx)
        hash_to_indices.setdefault(str(representative.clip_hash), []).append(int(idx))
    bundle["areas"] = np.asarray(areas, dtype=np.int64)
    bundle["hashes"] = list(hashes)
    bundle["hashes_np"] = np.asarray(hashes)
    bundle["hash_to_indices"] = hash_to_indices
    bundle["strict_key_to_index"] = strict_key_to_index
    bundle["representatives"] = [group["representative"] for group in bundle["candidate_groups"]]
    bundle["origin_ids"] = [tuple(sorted(int(value) for value in group["origin_ids"])) for group in bundle["candidate_groups"]]
    bundle["clip_pixels"] = int(bundle["shape"][0]) * int(bundle["shape"][1])
    bundle["geometry_cache_by_idx"] = {}
    bundle["full_descriptor_cache_by_idx"] = {}
    bundle["cheap_descriptor_cache_by_idx"] = {}
    bundle["full_prefilter_disabled"] = False
    bundle["full_prefilter_probe_pairs"] = 0
    bundle["full_prefilter_probe_rejects"] = 0
    bundle["full_prefilter_probe_done"] = False
    return bundle


def build_candidate_match_bundles(candidates):
    """按 shape 和严格 bitmap key 构建 v1 风格轻量 candidate bundle。"""

    bundles = {}
    groups = _coverage_group_from_candidates(candidates)
    for group in groups:
        shape = tuple(int(value) for value in group["representative"].clip_bitmap.shape)
        bundle = bundles.get(shape)
        if bundle is None:
            bundle = {"shape": shape, "candidate_groups": []}
            bundles[shape] = bundle
        bundle["candidate_groups"].append(group)
    for bundle in bundles.values():
        _finalize_candidate_bundle(bundle)
    return bundles


def _candidate_bundle_payload(bundle, shape_key, output_json, output_npz, extra_payload):
    """生成 candidate bundle bucket 的 JSON payload。"""

    payload = dict(extra_payload or {})
    payload["shape_key"] = str(shape_key)
    payload["group_count"] = int(len(bundle["candidate_groups"]))
    payload["output_json"] = str(output_json)
    payload["output_npz"] = str(output_npz)
    groups = []
    for idx, group in enumerate(bundle["candidate_groups"]):
        metadata = candidate_metadata(group["representative"], include_coverage=False)
        metadata["group_index"] = int(idx)
        metadata["origin_ids"] = [int(value) for value in sorted(group["origin_ids"])]
        candidate_ids = group.get("candidate_ids")
        if candidate_ids is None:
            candidate_ids = [str(candidate.candidate_id) for candidate in group.get("candidates", [])]
        metadata["candidate_ids"] = list(candidate_ids)
        candidate_count = group.get("candidate_count")
        if candidate_count is None:
            candidate_count = int(len(group.get("candidates", [])))
        metadata["candidate_count"] = int(candidate_count)
        groups.append(metadata)
    payload["groups"] = groups
    return payload


def _write_candidate_bundle_bucket(root, bundle, shape_key, safe_key, extra_payload):
    """写出一个 candidate bundle bucket，并返回 manifest 索引项。"""

    output_json = root / ("candidate_bundle_%s.json" % str(safe_key))
    output_npz = root / ("candidate_bundle_%s.npz" % str(safe_key))
    bitmaps = np.asarray([group["representative"].clip_bitmap for group in bundle["candidate_groups"]], dtype=bool)
    flat_bitmaps = bitmaps.reshape((int(bitmaps.shape[0]), -1))
    packed_bitmaps = np.packbits(flat_bitmaps, axis=1)
    cheap_arrays = _cheap_descriptor_arrays_for_bitmaps(bitmaps)
    np.savez_compressed(
        str(output_npz),
        group_bitmaps=bitmaps,
        packed_bitmaps=packed_bitmaps,
        area_px=cheap_arrays["area_px"],
        cheap_invariants=cheap_arrays["cheap_invariants"],
        cheap_signature_grid=cheap_arrays["cheap_signature_grid"],
        cheap_signature_proj_x=cheap_arrays["cheap_signature_proj_x"],
        cheap_signature_proj_y=cheap_arrays["cheap_signature_proj_y"],
        cheap_signature_vectors=cheap_arrays["cheap_signature_vectors"],
        cheap_subgroup_keys=cheap_arrays["cheap_subgroup_keys"],
    )
    payload = _candidate_bundle_payload(bundle, shape_key, output_json, output_npz, extra_payload)
    payload["precomputed_fields"] = [
        "packed_bitmaps",
        "area_px",
        "cheap_invariants",
        "cheap_signature_grid",
        "cheap_signature_proj_x",
        "cheap_signature_proj_y",
        "cheap_signature_vectors",
        "cheap_subgroup_keys",
    ]
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=json_default)
    return {
        "shape_key": str(shape_key),
        "group_count": int(len(bundle["candidate_groups"])),
        "output_json": str(output_json),
        "output_npz": str(output_npz),
    }


def _split_candidate_bundle_by_fill_bin(bundle):
    """把较大的 shape bundle 按 fill-ratio 子桶拆分，降低 coverage shard 加载面。"""

    sub_bundles = {}
    for group in bundle["candidate_groups"]:
        fill_bin = _candidate_bundle_fill_bin_for_bitmap(group["representative"].clip_bitmap)
        sub_bundle = sub_bundles.get(int(fill_bin))
        if sub_bundle is None:
            sub_bundle = {"shape": bundle["shape"], "candidate_groups": []}
            sub_bundles[int(fill_bin)] = sub_bundle
        sub_bundle["candidate_groups"].append(group)
    for sub_bundle in sub_bundles.values():
        _finalize_candidate_bundle(sub_bundle)
    return sub_bundles


def save_candidate_bundle_index(candidates, bundle_root, extra_payload=None):
    """把全局 candidate bundle 按 shape 写成 JSON + NPZ，供 coverage shard 只读加载。"""

    root = Path(str(bundle_root))
    if not root.exists():
        root.mkdir(parents=True)
    bundles = build_candidate_match_bundles(candidates)
    return _save_candidate_bundle_index_from_bundles(bundles, root, extra_payload, int(len(candidates)))


def _save_candidate_bundle_index_from_bundles(bundles, root, extra_payload, candidate_count):
    """把 bundle 字典写成 candidate bundle 索引。"""

    shape_buckets = {}
    total_groups = 0
    total_file_buckets = 0
    file_bucket_group_sizes = []
    for shape in sorted(bundles):
        bundle = bundles[shape]
        shape_key = bitmap_shape_key(shape)
        safe_key = str(shape_key).replace("x", "_")
        group_count = int(len(bundle["candidate_groups"]))
        total_groups += group_count
        if group_count >= int(CANDIDATE_BUNDLE_SPLIT_MIN_GROUPS):
            sub_bundles = _split_candidate_bundle_by_fill_bin(bundle)
            sub_items = {}
            for fill_bin in sorted(sub_bundles):
                sub_bundle = sub_bundles[int(fill_bin)]
                sub_safe_key = "%s_f%03d" % (safe_key, int(fill_bin))
                sub_extra = dict(extra_payload or {})
                sub_extra["bucket_mode"] = "fill_bin"
                sub_extra["fill_bin"] = int(fill_bin)
                item = _write_candidate_bundle_bucket(root, sub_bundle, shape_key, sub_safe_key, sub_extra)
                item["bucket_mode"] = "fill_bin"
                item["fill_bin"] = int(fill_bin)
                sub_items[str(fill_bin)] = item
                total_file_buckets += 1
                file_bucket_group_sizes.append(int(item["group_count"]))
            shape_buckets[str(shape_key)] = {
                "shape_key": str(shape_key),
                "group_count": int(group_count),
                "bucket_mode": "fill_bin",
                "bucket_count": int(len(sub_items)),
                "buckets": sub_items,
            }
        else:
            item = _write_candidate_bundle_bucket(root, bundle, shape_key, safe_key, extra_payload)
            item["bucket_mode"] = "shape"
            item["bucket_count"] = 1
            shape_buckets[str(shape_key)] = item
            total_file_buckets += 1
            file_bucket_group_sizes.append(int(item["group_count"]))
    group_sizes = [len(bundle["candidate_groups"]) for bundle in bundles.values()]
    return {
        "bucket_count": int(total_file_buckets),
        "shape_bucket_count": int(len(shape_buckets)),
        "candidate_count": int(candidate_count),
        "candidate_group_count": int(total_groups),
        "max_bundle_group_count": int(max(group_sizes, default=0)),
        "max_file_bucket_group_count": int(max(file_bucket_group_sizes, default=0)),
        "bucket_split_mode": "shape_fill_bin",
        "fill_bin_width": float(CANDIDATE_BUNDLE_FILL_BIN_WIDTH),
        "fill_bin_neighbor_radius": int(math.ceil(float(CHEAP_FILL_ABS_LIMIT) / float(CANDIDATE_BUNDLE_FILL_BIN_WIDTH))) + 1,
        "shape_buckets": shape_buckets,
    }


def save_candidate_bundle_index_from_accumulator(accumulator, bundle_root, extra_payload=None):
    """把 chunked candidate 累加器写成 bundle 索引，避免全量 candidate 常驻内存。"""

    root = Path(str(bundle_root))
    if not root.exists():
        root.mkdir(parents=True)
    bundles = candidate_bundles_from_accumulator(accumulator)
    return _save_candidate_bundle_index_from_bundles(
        bundles,
        root,
        extra_payload,
        int(accumulator.get("candidate_count", 0)),
    )


def load_candidate_bundle_bucket(json_path, npz_path):
    """读取单个 shape 的 candidate bundle bucket。"""

    with Path(str(json_path)).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    arrays = np.load(str(npz_path), allow_pickle=False)
    try:
        bitmaps = arrays["group_bitmaps"]
        groups = []
        for idx, metadata in enumerate(payload.get("groups", [])):
            candidate = candidate_from_metadata(metadata, bitmaps[int(idx)])
            origin_ids = set(int(value) for value in metadata.get("origin_ids", []))
            if not origin_ids:
                origin_ids.add(int(candidate.origin_exact_cluster_id))
            groups.append(
                {
                    "representative": candidate,
                    "candidates": [],
                    "origin_ids": origin_ids,
                    "strict_key": _bitmap_exact_key(candidate.clip_bitmap),
                    "candidate_ids": [str(value) for value in metadata.get("candidate_ids", [])],
                    "candidate_count": int(metadata.get("candidate_count", len(metadata.get("candidate_ids", [])))),
                }
            )
        if len(groups):
            shape = tuple(int(value) for value in groups[0]["representative"].clip_bitmap.shape)
        else:
            shape = tuple(int(value) for value in str(payload.get("shape_key", "0x0")).split("x"))
        bundle = _finalize_candidate_bundle({"shape": shape, "candidate_groups": groups})
        for key in (
            "packed_bitmaps",
            "area_px",
            "cheap_invariants",
            "cheap_signature_grid",
            "cheap_signature_proj_x",
            "cheap_signature_proj_y",
            "cheap_signature_vectors",
            "cheap_subgroup_keys",
        ):
            if key in arrays:
                bundle["precomputed_" + key] = np.asarray(arrays[key])
    finally:
        arrays.close()
    return bundle, payload


def _candidate_bundle_bucket_items_for_shape(shape_item, fill_bins=None):
    """从 shape 索引项中取出需要加载的 candidate bundle 文件项。"""

    if "buckets" not in shape_item:
        return [shape_item]
    sub_items = dict(shape_item.get("buckets", {}))
    if fill_bins is None:
        return [sub_items[key] for key in sorted(sub_items, key=lambda value: int(value))]
    items = []
    seen = set()
    for fill_bin in sorted(int(value) for value in fill_bins):
        item = sub_items.get(str(fill_bin))
        if item is None:
            continue
        key = str(item.get("output_json", "")) + "|" + str(item.get("output_npz", ""))
        if key in seen:
            continue
        seen.add(key)
        items.append(item)
    return items


def _merge_candidate_bundle_parts(shape_key, parts):
    """把同一 shape 的若干子桶合并成 coverage 可直接使用的 bundle。"""

    groups = []
    shape = None
    for part in parts:
        if shape is None:
            shape = tuple(int(value) for value in part["shape"])
        groups.extend(part["candidate_groups"])
    if shape is None:
        shape = tuple(int(value) for value in str(shape_key).split("x"))
    bundle = _finalize_candidate_bundle({"shape": shape, "candidate_groups": groups})
    for key in (
        "precomputed_packed_bitmaps",
        "precomputed_area_px",
        "precomputed_cheap_invariants",
        "precomputed_cheap_signature_grid",
        "precomputed_cheap_signature_proj_x",
        "precomputed_cheap_signature_proj_y",
        "precomputed_cheap_signature_vectors",
        "precomputed_cheap_subgroup_keys",
    ):
        arrays = [part[key] for part in parts if key in part]
        if len(arrays) == len(parts) and arrays:
            bundle[key] = np.concatenate(arrays, axis=0)
    return bundle


def load_candidate_bundle_buckets_for_shapes(bundle_index, shape_keys):
    """按 shape_key 加载 coverage shard 实际需要的 target candidate bundles。"""

    buckets = {}
    shape_bucket_map = dict(bundle_index.get("shape_buckets", {}))
    for shape_key in sorted(set(str(value) for value in shape_keys)):
        item = shape_bucket_map.get(str(shape_key))
        if item is None:
            continue
        parts = []
        for bucket_item in _candidate_bundle_bucket_items_for_shape(item):
            bundle, _ = load_candidate_bundle_bucket(bucket_item["output_json"], bucket_item["output_npz"])
            parts.append(bundle)
        buckets[str(shape_key)] = _merge_candidate_bundle_parts(shape_key, parts)
    return buckets


def load_candidate_bundle_buckets_for_candidates(bundle_index, candidates):
    """按 source candidates 的 shape/fill 邻域懒加载 candidate bundle 子桶。"""

    shape_bucket_map = dict(bundle_index.get("shape_buckets", {}))
    shape_to_bins = {}
    for candidate in candidates:
        shape_key = bitmap_shape_key(candidate.clip_bitmap.shape)
        fill_bin = _candidate_bundle_fill_bin_for_bitmap(candidate.clip_bitmap)
        bins = shape_to_bins.setdefault(str(shape_key), set())
        bins.update(int(value) for value in _candidate_bundle_fill_neighbor_bins(fill_bin))

    buckets = {}
    loaded_bucket_keys = []
    loaded_group_count = 0
    for shape_key in sorted(shape_to_bins):
        item = shape_bucket_map.get(str(shape_key))
        if item is None:
            continue
        parts = []
        for bucket_item in _candidate_bundle_bucket_items_for_shape(item, shape_to_bins[str(shape_key)]):
            bundle, _ = load_candidate_bundle_bucket(bucket_item["output_json"], bucket_item["output_npz"])
            parts.append(bundle)
            loaded_group_count += int(len(bundle["candidate_groups"]))
            loaded_bucket_keys.append("%s:%s" % (str(shape_key), str(bucket_item.get("fill_bin", "shape"))))
        if parts:
            buckets[str(shape_key)] = _merge_candidate_bundle_parts(shape_key, parts)
    stats = {
        "shape_count_requested": int(len(shape_to_bins)),
        "shape_count_loaded": int(len(buckets)),
        "bucket_count_loaded": int(len(loaded_bucket_keys)),
        "candidate_group_count_loaded": int(loaded_group_count),
        "shape_keys_loaded": sorted(buckets.keys()),
        "bucket_keys_loaded": loaded_bucket_keys[:128],
    }
    return buckets, stats


def _bundle_cheap_descriptor(bundle, group_idx):
    """按需获取 target bundle group 的 cheap descriptor。"""

    cache = bundle.setdefault("cheap_descriptor_cache_by_idx", {})
    idx = int(group_idx)
    descriptor = cache.get(idx)
    if descriptor is None:
        if "precomputed_cheap_invariants" in bundle:
            descriptor = CheapDescriptor(
                bundle["precomputed_cheap_invariants"][idx],
                bundle["precomputed_cheap_signature_grid"][idx],
                bundle["precomputed_cheap_signature_proj_x"][idx],
                bundle["precomputed_cheap_signature_proj_y"][idx],
                int(bundle["precomputed_area_px"][idx]) if "precomputed_area_px" in bundle else int(bundle["areas"][idx]),
            )
        else:
            descriptor = _cheap_bitmap_descriptor(bundle["representatives"][idx].clip_bitmap)
        cache[idx] = descriptor
    return descriptor


def _bundle_full_descriptor(bundle, group_idx, detail_seconds, debug_stats):
    """按需获取 target bundle group 的 full graph descriptor。"""

    cache = bundle.setdefault("full_descriptor_cache_by_idx", {})
    idx = int(group_idx)
    descriptor = cache.get(idx)
    if descriptor is None:
        started = time.perf_counter()
        descriptor = _graph_bitmap_descriptor(bundle["representatives"][idx].clip_bitmap)
        cache[idx] = descriptor
        detail_seconds["full_descriptor_cache"] += time.perf_counter() - started
        debug_stats["full_descriptor_cache_group_count"] += 1
    return descriptor


def _group_full_descriptor(group):
    """按需获取 source group 的 full graph descriptor。"""

    descriptor = group.get("full_descriptor")
    if descriptor is None:
        descriptor = _graph_bitmap_descriptor(group["representative"].clip_bitmap)
        group["full_descriptor"] = descriptor
    return descriptor


def _init_coverage_geometry_cache(owner):
    """初始化 coverage 几何缓存，只生成 packed bitmap 和面积。"""

    bitmap = np.ascontiguousarray(owner.clip_bitmap.astype(bool, copy=False))
    return {
        "bitmap": bitmap,
        "packed": np.packbits(bitmap.reshape(-1)),
        "area_px": int(np.count_nonzero(bitmap)),
    }


def _init_bundle_geometry_cache(bundle, group_idx, debug_stats):
    """初始化 target bundle 几何缓存，优先复用 NPZ 中的 packed bitmap。"""

    idx = int(group_idx)
    bitmap = np.ascontiguousarray(bundle["representatives"][idx].clip_bitmap.astype(bool, copy=False))
    packed_bitmaps = bundle.get("precomputed_packed_bitmaps")
    area_px = bundle.get("precomputed_area_px")
    if packed_bitmaps is not None and area_px is not None:
        debug_stats["precomputed_packed_bitmap_count"] += 1
        return {
            "bitmap": bitmap,
            "packed": np.asarray(packed_bitmaps[idx], dtype=np.uint8),
            "area_px": int(area_px[idx]),
        }
    return _init_coverage_geometry_cache(bundle["representatives"][idx])


def _extend_coverage_dilated_cache(cache, tol_px):
    """把 coverage 几何缓存扩展到膨胀层，不计算 donut。"""

    if int(tol_px) <= 0 or "packed_dilated" in cache:
        return
    structure = _coverage_structure(int(tol_px))
    dilated = ndimage.binary_dilation(cache["bitmap"], structure=structure)
    cache["dilated"] = np.ascontiguousarray(dilated, dtype=bool)
    cache["packed_dilated"] = np.packbits(cache["dilated"].reshape(-1))
    cache["dilated_area_px"] = int(np.count_nonzero(cache["dilated"]))


def _extend_coverage_donut_cache(cache, tol_px):
    """把 coverage 几何缓存扩展到 donut 层，供最终 ECC overlap 使用。"""

    if int(tol_px) <= 0 or "packed_donut" in cache:
        return
    _extend_coverage_dilated_cache(cache, int(tol_px))
    structure = _coverage_structure(int(tol_px))
    eroded = ndimage.binary_erosion(cache["bitmap"], structure=structure, border_value=0)
    donut = cache["dilated"] & ~eroded
    cache["packed_donut"] = np.packbits(np.ascontiguousarray(donut, dtype=bool).reshape(-1))
    cache["donut_area_px"] = int(np.count_nonzero(donut))


def _bundle_geometry_level_for_key(key):
    """根据缓存字段名推导所需的 coverage 几何缓存层级。"""

    if key in ("packed_donut", "donut_area_px"):
        return "donut"
    if key in ("packed_dilated", "dilated_area_px"):
        return "dilated"
    return "packed"


def _bundle_geometry_cache(bundle, group_idx, tol_px, detail_seconds, debug_stats, level="packed"):
    """按需获取 target bundle group 的分层几何缓存。"""

    cache_by_idx = bundle.setdefault("geometry_cache_by_idx", {})
    idx = int(group_idx)
    cached = cache_by_idx.get(idx)
    if cached is None:
        started = time.perf_counter()
        cached = _init_bundle_geometry_cache(bundle, idx, debug_stats)
        cache_by_idx[idx] = cached
        detail_seconds["geometry_cache"] += time.perf_counter() - started
        debug_stats["geometry_cache_group_count"] += 1
    if level in ("dilated", "donut") and int(tol_px) > 0 and "packed_dilated" not in cached:
        started = time.perf_counter()
        _extend_coverage_dilated_cache(cached, int(tol_px))
        detail_seconds["geometry_cache"] += time.perf_counter() - started
        debug_stats["geometry_dilated_cache_group_count"] += 1
    if level == "donut" and int(tol_px) > 0 and "packed_donut" not in cached:
        started = time.perf_counter()
        _extend_coverage_donut_cache(cached, int(tol_px))
        detail_seconds["geometry_cache"] += time.perf_counter() - started
        debug_stats["geometry_donut_cache_group_count"] += 1
    return cached


def _source_geometry_cache(group, tol_px, detail_seconds, debug_stats, level="packed"):
    """按需获取 source group 的分层几何缓存。"""

    cached = group.get("geometry_cache")
    if cached is None:
        started = time.perf_counter()
        cached = _init_coverage_geometry_cache(group["representative"])
        group["geometry_cache"] = cached
        detail_seconds["geometry_cache"] += time.perf_counter() - started
        debug_stats["geometry_cache_group_count"] += 1
    if level in ("dilated", "donut") and int(tol_px) > 0 and "packed_dilated" not in cached:
        started = time.perf_counter()
        _extend_coverage_dilated_cache(cached, int(tol_px))
        detail_seconds["geometry_cache"] += time.perf_counter() - started
        debug_stats["geometry_dilated_cache_group_count"] += 1
    if level == "donut" and int(tol_px) > 0 and "packed_donut" not in cached:
        started = time.perf_counter()
        _extend_coverage_donut_cache(cached, int(tol_px))
        detail_seconds["geometry_cache"] += time.perf_counter() - started
        debug_stats["geometry_donut_cache_group_count"] += 1
    return cached


def _bundle_geometry_matrix(bundle, indices, tol_px, key, detail_seconds, debug_stats):
    """按需把若干 target group 的几何缓存字段堆叠成矩阵。"""

    level = _bundle_geometry_level_for_key(str(key))
    rows = [
        np.asarray(_bundle_geometry_cache(bundle, int(group_idx), int(tol_px), detail_seconds, debug_stats, level=level)[key], dtype=np.uint8)
        for group_idx in np.asarray(indices, dtype=np.int64).tolist()
    ]
    if not rows:
        return np.empty((0, 0), dtype=np.uint8)
    return np.stack(rows, axis=0)


def _bundle_geometry_values(bundle, indices, tol_px, key, detail_seconds, debug_stats):
    """按需读取若干 target group 的几何缓存标量字段。"""

    level = _bundle_geometry_level_for_key(str(key))
    values = [
        int(_bundle_geometry_cache(bundle, int(group_idx), int(tol_px), detail_seconds, debug_stats, level=level)[key])
        for group_idx in np.asarray(indices, dtype=np.int64).tolist()
    ]
    return np.asarray(values, dtype=np.int64)


def _build_bundle_shortlist_index(bundle, debug_stats):
    """为一个 shape bundle 预构建 cheap 分组 shortlist。"""

    group_count = int(len(bundle["candidate_groups"]))
    precomputed = (
        "precomputed_cheap_invariants" in bundle
        and "precomputed_cheap_signature_grid" in bundle
        and "precomputed_cheap_signature_proj_x" in bundle
        and "precomputed_cheap_signature_proj_y" in bundle
        and "precomputed_cheap_signature_vectors" in bundle
        and "precomputed_cheap_subgroup_keys" in bundle
    )
    if precomputed:
        debug_stats["shortlist_precomputed_bundle_count"] += 1
        debug_stats["precomputed_cheap_descriptor_count"] += group_count
        cheap_invariant_matrix = np.asarray(bundle["precomputed_cheap_invariants"], dtype=np.float32)
        signature_vectors = np.asarray(bundle["precomputed_cheap_signature_vectors"], dtype=np.float32)
        subgroup_keys = [tuple(int(value) for value in row) for row in np.asarray(bundle["precomputed_cheap_subgroup_keys"], dtype=np.int32)]
        cheap_signature_grid_matrix = _normalized_matrix(bundle["precomputed_cheap_signature_grid"])
        cheap_signature_proj_x_matrix = _normalized_matrix(bundle["precomputed_cheap_signature_proj_x"])
        cheap_signature_proj_y_matrix = _normalized_matrix(bundle["precomputed_cheap_signature_proj_y"])
    else:
        descriptors = [_bundle_cheap_descriptor(bundle, idx) for idx in range(group_count)]
        signature_vectors = np.asarray([_signature_embedding(desc) for desc in descriptors], dtype=np.float32)
        subgroup_keys = [_coverage_cheap_subgroup_key(desc) for desc in descriptors]
        cheap_invariant_matrix = np.asarray([desc.invariants for desc in descriptors], dtype=np.float32)
        cheap_signature_grid_matrix = _normalized_matrix([desc.signature_grid for desc in descriptors])
        cheap_signature_proj_x_matrix = _normalized_matrix([desc.signature_proj_x for desc in descriptors])
        cheap_signature_proj_y_matrix = _normalized_matrix([desc.signature_proj_y for desc in descriptors])
    group_count = int(signature_vectors.shape[0])
    neighbor_labels = np.full(
        (group_count, min(int(COVERAGE_SHORTLIST_MAX_TARGETS) + 1, max(group_count, 1))),
        -1,
        dtype=np.int64,
    )
    subgroups = {}
    for idx, subgroup_key in enumerate(subgroup_keys):
        subgroups.setdefault(subgroup_key, []).append(int(idx))
    debug_stats["shortlist_subgroup_count"] += int(len(subgroups))
    debug_stats["shortlist_max_subgroup_size"] = max(
        int(debug_stats.get("shortlist_max_subgroup_size", 0)),
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
        if int(group_indices.size) <= int(COVERAGE_EXACT_SHORTLIST_MAX_GROUPS) or hnswlib is None:
            debug_stats["shortlist_exact_subgroup_count"] += 1
            local_labels = _exact_cosine_topk_labels(group_vectors, k)
        else:
            debug_stats["shortlist_hnsw_subgroup_count"] += 1
            index = hnswlib.Index(space="cosine", dim=int(group_vectors.shape[1]))
            index.init_index(max_elements=int(group_indices.size), ef_construction=max(64, k * 2), M=12)
            index.add_items(group_vectors, np.arange(int(group_indices.size), dtype=np.int64))
            index.set_ef(max(64, k * 2))
            local_labels, _ = index.knn_query(group_vectors, k=k)
        mapped_labels = group_indices[np.asarray(local_labels, dtype=np.int64)]
        for row_idx, global_idx in enumerate(group_indices.tolist()):
            neighbor_labels[int(global_idx), :k] = mapped_labels[row_idx, :k]

    reverse_neighbors = [set() for _ in range(group_count)]
    for row_idx in range(group_count):
        for target_idx in neighbor_labels[row_idx].tolist():
            if target_idx >= 0 and int(target_idx) != int(row_idx):
                reverse_neighbors[int(target_idx)].add(int(row_idx))
    return {
        "neighbor_labels": np.asarray(neighbor_labels, dtype=np.int64),
        "reverse_neighbors": [np.asarray(sorted(values), dtype=np.int64) for values in reverse_neighbors],
        "cheap_invariant_matrix": cheap_invariant_matrix,
        "cheap_signature_grid_matrix": cheap_signature_grid_matrix,
        "cheap_signature_proj_x_matrix": cheap_signature_proj_x_matrix,
        "cheap_signature_proj_y_matrix": cheap_signature_proj_y_matrix,
    }


def _source_target_index(source_group, target_bundle):
    """返回 source group 在全局 target bundle 中的 group index。"""

    return target_bundle.get("strict_key_to_index", {}).get(source_group.get("strict_key"))


def _shortlist_target_indices(source_group, target_bundle, shortlist_index):
    """返回 direct + reverse 对称化后的 target group shortlist。"""

    source_idx = _source_target_index(source_group, target_bundle)
    if source_idx is None:
        raise RuntimeError("source candidate group is missing from global candidate bundle")
    labels = np.asarray(shortlist_index["neighbor_labels"][int(source_idx)], dtype=np.int64)
    direct = labels[(labels >= 0) & (labels != int(source_idx))]
    reverse = np.asarray(shortlist_index["reverse_neighbors"][int(source_idx)], dtype=np.int64)
    if direct.size and reverse.size:
        merged = np.unique(np.concatenate([direct, reverse]))
    elif direct.size:
        merged = np.unique(direct)
    else:
        merged = np.unique(reverse)
    return merged[merged != int(source_idx)]


def _batch_prefilter(source_group, target_bundle, shortlist_index, target_indices, detail_seconds, debug_stats):
    """对一个 source group 的 shortlist targets 执行 cheap 与懒加载 full prefilter。"""

    if target_indices.size == 0:
        return target_indices

    cheap_inv_mat = np.asarray(shortlist_index["cheap_invariant_matrix"], dtype=np.float32)
    source_idx = _source_target_index(source_group, target_bundle)
    if source_idx is None:
        raise RuntimeError("source candidate group is missing from global candidate bundle")
    source_cheap = cheap_inv_mat[int(source_idx)]
    target_cheap = cheap_inv_mat[target_indices]
    cheap_floors = np.asarray([0.02, 0.03, 0.03], dtype=np.float32)
    cheap_source = source_cheap[[1, 4, 5]]
    cheap_target = target_cheap[:, [1, 4, 5]]
    cheap_denom = np.maximum(np.maximum(np.abs(cheap_source)[None, :], np.abs(cheap_target)), cheap_floors[None, :])
    cheap_errs = np.abs(cheap_target - cheap_source[None, :]) / cheap_denom
    cheap_ratio_ok = np.all(cheap_errs <= 0.45, axis=1)
    fill_ok = np.abs(target_cheap[:, 1] - source_cheap[1]) <= CHEAP_FILL_ABS_LIMIT
    density_ok = np.abs(target_cheap[:, 5] - source_cheap[5]) <= CHEAP_AREA_DENSITY_ABS_LIMIT
    cheap_ok = cheap_ratio_ok & fill_ok & density_ok
    debug_stats["cheap_fill_reject"] += int(np.count_nonzero(cheap_ratio_ok & ~fill_ok))
    debug_stats["cheap_area_density_reject"] += int(np.count_nonzero(cheap_ratio_ok & fill_ok & ~density_ok))
    debug_stats["cheap_reject"] += int(np.count_nonzero(~cheap_ok))
    target_indices = target_indices[cheap_ok]
    if target_indices.size == 0:
        return target_indices

    if bool(target_bundle.get("full_prefilter_disabled", False)):
        return target_indices

    full_started = time.perf_counter()
    full_input_count = int(target_indices.size)
    source_desc = _group_full_descriptor(source_group)
    target_descs = [_bundle_full_descriptor(target_bundle, int(idx), detail_seconds, debug_stats) for idx in target_indices.tolist()]

    source_inv = np.asarray(source_desc.invariants, dtype=np.float64)
    target_inv = np.asarray([desc.invariants for desc in target_descs], dtype=np.float64)
    inv_floors = np.asarray([0.25, 0.02, 0.02, 0.02, 0.03, 0.03, 0.02, 0.02], dtype=np.float64)
    inv_weights = np.asarray([0.08, 0.24, 0.10, 0.08, 0.18, 0.14, 0.10, 0.08], dtype=np.float64)
    inv_denom = np.maximum(np.maximum(np.abs(source_inv)[None, :], np.abs(target_inv)), inv_floors[None, :])
    inv_errs = np.minimum(np.abs(target_inv - source_inv[None, :]) / inv_denom, 1.0)
    critical = (inv_errs[:, 1] > 0.45) | (inv_errs[:, 4] > 0.45) | (inv_errs[:, 5] > 0.45)
    invariant_ok = (~critical) & ((inv_errs @ inv_weights) <= GRAPH_INVARIANT_LIMIT)
    debug_stats["invariant_reject"] += int(np.count_nonzero(~invariant_ok))
    if not np.all(invariant_ok):
        keep = invariant_ok.tolist()
        target_indices = target_indices[invariant_ok]
        target_descs = [desc for desc, ok in zip(target_descs, keep) if ok]

    if target_indices.size:
        source_topology = np.asarray(source_desc.topology, dtype=np.float64)
        target_topology = np.asarray([desc.topology for desc in target_descs], dtype=np.float64)
        topology_dist = np.linalg.norm(target_topology - source_topology[None, :], axis=1)
        topology_ok = topology_dist <= GRAPH_TOPOLOGY_THRESHOLD
        debug_stats["topology_reject"] += int(np.count_nonzero(~topology_ok))
        if not np.all(topology_ok):
            keep = topology_ok.tolist()
            target_indices = target_indices[topology_ok]
            target_descs = [desc for desc, ok in zip(target_descs, keep) if ok]

    if target_indices.size:
        source_grid = _normalized_matrix([source_desc.signature_grid])[0]
        source_proj_x = _normalized_matrix([source_desc.signature_proj_x])[0]
        source_proj_y = _normalized_matrix([source_desc.signature_proj_y])[0]
        target_grid = _normalized_matrix([desc.signature_grid for desc in target_descs])
        target_proj_x = _normalized_matrix([desc.signature_proj_x for desc in target_descs])
        target_proj_y = _normalized_matrix([desc.signature_proj_y for desc in target_descs])
        signature_sim = 0.6 * (target_grid @ source_grid) + 0.2 * (target_proj_x @ source_proj_x) + 0.2 * (target_proj_y @ source_proj_y)
        signature_ok = signature_sim >= GRAPH_SIGNATURE_THRESHOLD
        debug_stats["signature_reject"] += int(np.count_nonzero(~signature_ok))
        target_indices = target_indices[signature_ok]

    full_reject_count = full_input_count - int(target_indices.size)
    debug_stats["full_prefilter_reject"] += int(full_reject_count)
    if not bool(target_bundle.get("full_prefilter_probe_done", False)):
        probe_pairs = int(target_bundle.get("full_prefilter_probe_pairs", 0)) + full_input_count
        probe_rejects = int(target_bundle.get("full_prefilter_probe_rejects", 0)) + full_reject_count
        target_bundle["full_prefilter_probe_pairs"] = probe_pairs
        target_bundle["full_prefilter_probe_rejects"] = probe_rejects
        debug_stats["full_prefilter_probe_pair_count"] += full_input_count
        debug_stats["full_prefilter_probe_reject_count"] += full_reject_count
        if probe_pairs >= int(COVERAGE_FULL_PREFILTER_MIN_PROBE_PAIRS):
            target_bundle["full_prefilter_probe_done"] = True
            reject_rate = float(probe_rejects) / float(max(probe_pairs, 1))
            if reject_rate < float(COVERAGE_FULL_PREFILTER_MIN_REJECT_RATE):
                target_bundle["full_prefilter_disabled"] = True
                debug_stats["full_prefilter_disabled_bundle_count"] += 1
    detail_seconds["full_prefilter"] += time.perf_counter() - full_started
    return target_indices


def _source_group_seed_type(group):
    """返回 source coverage group 代表候选的 seed type。"""

    candidate = group.get("representative")
    seed_type = getattr(candidate, "origin_seed_type", "unknown")
    if seed_type in (None, ""):
        return "unknown"
    return str(seed_type)


def _bundle_group_seed_type(bundle, group_idx):
    """返回 target bundle group 代表候选的 seed type。"""

    candidate = bundle["representatives"][int(group_idx)]
    seed_type = getattr(candidate, "origin_seed_type", "unknown")
    if seed_type in (None, ""):
        return "unknown"
    return str(seed_type)


def _strict_graph_filter_target_indices(source_group, target_bundle, target_indices, detail_seconds, debug_stats, pass_stat_key, reject_stat_key):
    """用 strict graph gate 过滤一批 target group，并记录通过/拒绝计数。"""

    source_desc = _group_full_descriptor(source_group)
    kept = []
    reject_count = 0
    for target_idx in np.asarray(target_indices, dtype=np.int64).tolist():
        target_desc = _bundle_full_descriptor(target_bundle, int(target_idx), detail_seconds, debug_stats)
        graph_ok, _ = _strict_graph_gate_reason(source_desc, target_desc)
        if graph_ok:
            kept.append(int(target_idx))
        else:
            reject_count += 1
    pass_count = int(len(kept))
    debug_stats[pass_stat_key] = int(debug_stats.get(pass_stat_key, 0)) + int(pass_count)
    debug_stats[reject_stat_key] = int(debug_stats.get(reject_stat_key, 0)) + int(reject_count)
    return np.asarray(kept, dtype=np.int64)


def _apply_long_shape_cross_seed_guard(source_group, target_bundle, matched_indices, detail_seconds, debug_stats):
    """限制 long_shape_path candidate 跨 seed type 泛化，避免长线模板桥接其它上下文。"""

    matched_indices = np.asarray(matched_indices, dtype=np.int64)
    if matched_indices.size == 0 or _source_group_seed_type(source_group) != SEED_TYPE_LONG:
        return matched_indices

    source_desc = _group_full_descriptor(source_group)
    kept = []
    guard_pair_count = 0
    reject_count = 0
    for target_idx in matched_indices.tolist():
        target_seed_type = _bundle_group_seed_type(target_bundle, int(target_idx))
        if target_seed_type in (SEED_TYPE_LONG, "unknown"):
            kept.append(int(target_idx))
            continue
        guard_pair_count += 1
        target_desc = _bundle_full_descriptor(target_bundle, int(target_idx), detail_seconds, debug_stats)
        graph_ok, _ = _strict_graph_gate_reason(source_desc, target_desc)
        signature_ok = _graph_signature_similarity(source_desc, target_desc) >= float(LONG_SHAPE_CROSS_SEED_SIGNATURE_THRESHOLD)
        if graph_ok and signature_ok:
            kept.append(int(target_idx))
        else:
            reject_count += 1

    pass_count = int(guard_pair_count) - int(reject_count)
    debug_stats["long_shape_cross_seed_guard_pair_count"] = int(debug_stats.get("long_shape_cross_seed_guard_pair_count", 0)) + int(guard_pair_count)
    debug_stats["long_shape_cross_seed_guard_pass_count"] = int(debug_stats.get("long_shape_cross_seed_guard_pass_count", 0)) + int(pass_count)
    debug_stats["long_shape_cross_seed_guard_reject_count"] = int(debug_stats.get("long_shape_cross_seed_guard_reject_count", 0)) + int(reject_count)
    return np.asarray(kept, dtype=np.int64)


def _matched_target_indices(source_group, target_bundle, target_indices, config, tol_px, detail_seconds, debug_stats):
    """用 v1 packed/dilated/donut geometry cache 判断 source group 能匹配哪些 target groups。"""

    if target_indices.size == 0:
        return np.asarray([], dtype=np.int64)
    source_cache = _source_geometry_cache(source_group, int(tol_px), detail_seconds, debug_stats)
    source_packed = np.asarray(source_cache["packed"], dtype=np.uint8)
    packed_row_bytes = max(int(source_packed.size), 1)
    matched_chunks = []
    matching_mode = str(config.get("geometry_match_mode", "ecc")).lower()
    ratio_limit = max(0.0, 1.0 - float(config.get("area_match_ratio", 0.96)))
    if matching_mode == "acc":
        clip_pixels = max(float(target_bundle["clip_pixels"]), 1.0)
        for target_chunk in _chunk_indices_by_row_width(target_indices, packed_row_bytes):
            target_packed = _bundle_geometry_matrix(target_bundle, target_chunk, tol_px, "packed", detail_seconds, debug_stats)
            geometry_started = time.perf_counter()
            xor_rows = _bitcount_sum_rows(np.bitwise_xor(target_packed, source_packed[None, :]))
            matched_chunks.append(target_chunk[(xor_rows / clip_pixels) <= ratio_limit])
            detail_seconds["geometry_match"] += time.perf_counter() - geometry_started
    elif int(tol_px) <= 0:
        for target_chunk in _chunk_indices_by_row_width(target_indices, packed_row_bytes):
            target_packed = _bundle_geometry_matrix(target_bundle, target_chunk, tol_px, "packed", detail_seconds, debug_stats)
            geometry_started = time.perf_counter()
            exact_equal = np.all(target_packed == source_packed[None, :], axis=1)
            matched_chunks.append(target_chunk[exact_equal])
            detail_seconds["geometry_match"] += time.perf_counter() - geometry_started
    else:
        source_area = float(source_cache["area_px"])
        source_area_limit = ECC_RESIDUAL_RATIO * max(source_area, 1.0)
        source_dilated_cache = _source_geometry_cache(source_group, int(tol_px), detail_seconds, debug_stats, level="dilated")
        source_dilated_area = int(source_dilated_cache["dilated_area_px"])
        source_packed_dilated = np.asarray(source_dilated_cache["packed_dilated"], dtype=np.uint8)
        for target_chunk in _chunk_indices_by_row_width(target_indices, packed_row_bytes * 4):
            geometry_started = time.perf_counter()
            target_areas = target_bundle["areas"][target_chunk].astype(np.float64)
            target_area_limits = ECC_RESIDUAL_RATIO * np.maximum(target_areas, 1.0)
            area_candidate_indices = target_chunk[target_areas <= float(source_dilated_area) + target_area_limits]
            detail_seconds["geometry_match"] += time.perf_counter() - geometry_started
            if not area_candidate_indices.size:
                continue
            target_dilated_areas = _bundle_geometry_values(target_bundle, area_candidate_indices, tol_px, "dilated_area_px", detail_seconds, debug_stats)
            geometry_started = time.perf_counter()
            overlap_indices = area_candidate_indices[source_area <= target_dilated_areas.astype(np.float64) + source_area_limit]
            detail_seconds["geometry_match"] += time.perf_counter() - geometry_started
            if overlap_indices.size:
                target_packed_dilated = _bundle_geometry_matrix(target_bundle, overlap_indices, tol_px, "packed_dilated", detail_seconds, debug_stats)
                geometry_started = time.perf_counter()
                residual_source_counts = _bitcount_sum_rows(np.bitwise_and(source_packed[None, :], np.bitwise_not(target_packed_dilated)))
                overlap_indices = overlap_indices[residual_source_counts <= source_area_limit]
                detail_seconds["geometry_match"] += time.perf_counter() - geometry_started
            if overlap_indices.size:
                overlap_target_limits = ECC_RESIDUAL_RATIO * np.maximum(target_bundle["areas"][overlap_indices].astype(np.float64), 1.0)
                target_packed = _bundle_geometry_matrix(target_bundle, overlap_indices, tol_px, "packed", detail_seconds, debug_stats)
                geometry_started = time.perf_counter()
                residual_target_counts = _bitcount_sum_rows(np.bitwise_and(target_packed, np.bitwise_not(source_packed_dilated[None, :])))
                overlap_indices = overlap_indices[residual_target_counts <= overlap_target_limits]
                detail_seconds["geometry_match"] += time.perf_counter() - geometry_started
            if not overlap_indices.size:
                continue
            source_donut_cache = _source_geometry_cache(source_group, int(tol_px), detail_seconds, debug_stats, level="donut")
            source_donut_area = int(source_donut_cache["donut_area_px"])
            source_packed_donut = np.asarray(source_donut_cache["packed_donut"], dtype=np.uint8)
            target_donut_areas = _bundle_geometry_values(target_bundle, overlap_indices, tol_px, "donut_area_px", detail_seconds, debug_stats)
            geometry_started = time.perf_counter()
            auto_true = (source_donut_area == 0) | (target_donut_areas == 0)
            if np.any(auto_true):
                auto_count = int(np.count_nonzero(auto_true))
                debug_stats["donut_auto_pass_pair_count"] = int(debug_stats.get("donut_auto_pass_pair_count", 0)) + int(auto_count)
                source_candidate = source_group["representative"]
                _increment_named_counter(
                    debug_stats["donut_auto_pass_source_shift_counts"],
                    getattr(source_candidate, "shift_direction", "unknown"),
                    auto_count,
                )
                _increment_named_counter(
                    debug_stats["donut_auto_pass_source_seed_type_counts"],
                    _source_group_seed_type(source_group),
                    auto_count,
                )
                auto_indices = overlap_indices[auto_true]
                if bool(ECC_ALLOW_DONUT_AUTO_PASS):
                    matched_chunks.append(auto_indices)
                else:
                    strict_auto_indices = _strict_graph_filter_target_indices(
                        source_group,
                        target_bundle,
                        auto_indices,
                        detail_seconds,
                        debug_stats,
                        "donut_degenerate_strict_graph_pass_count",
                        "donut_degenerate_strict_graph_reject_count",
                    )
                    if strict_auto_indices.size:
                        matched_chunks.append(strict_auto_indices)
            overlap_indices = overlap_indices[~auto_true]
            target_donut_areas = target_donut_areas[~auto_true]
            detail_seconds["geometry_match"] += time.perf_counter() - geometry_started
            if overlap_indices.size:
                target_packed_donut = _bundle_geometry_matrix(target_bundle, overlap_indices, tol_px, "packed_donut", detail_seconds, debug_stats)
                geometry_started = time.perf_counter()
                overlap_counts = _bitcount_sum_rows(np.bitwise_and(target_packed_donut, source_packed_donut[None, :]))
                overlap_denominator = np.maximum(np.minimum(target_donut_areas, source_donut_area).astype(np.float64), 1.0)
                overlap_ok = (overlap_counts / overlap_denominator) >= ECC_DONUT_OVERLAP_RATIO
                matched_chunks.append(overlap_indices[overlap_ok])
                detail_seconds["geometry_match"] += time.perf_counter() - geometry_started

    non_empty_chunks = [chunk for chunk in matched_chunks if chunk.size]
    if not non_empty_chunks:
        return np.asarray([], dtype=np.int64)
    matched_indices = np.concatenate(non_empty_chunks, axis=0)
    return _apply_long_shape_cross_seed_guard(source_group, target_bundle, matched_indices, detail_seconds, debug_stats)


def evaluate_candidate_coverage_against_bundles(candidates, target_bundles_by_shape, config):
    """按 v1 主算法对 source candidates 和全局 target candidate bundle 构建 coverage。"""

    pixel_size_um = float(int(config.get("pixel_size_nm", DEFAULT_PIXEL_SIZE_NM))) / 1000.0
    tol_px = max(0, int(math.ceil(float(config.get("edge_tolerance_um", 0.02)) / max(pixel_size_um, 1e-12) - 1e-12)))
    detail_seconds = _empty_coverage_detail_seconds()
    debug_stats = _empty_coverage_debug_stats()
    for candidate in candidates:
        candidate.coverage = set(candidate.coverage)
    started = time.perf_counter()
    source_bundles = build_candidate_match_bundles(candidates)
    detail_seconds["light_bundle_build"] += time.perf_counter() - started
    bundle_sizes = [len(bundle["candidate_groups"]) for bundle in source_bundles.values()]
    debug_stats["bundle_count"] = int(len(source_bundles))
    debug_stats["max_bundle_group_count"] = int(max(bundle_sizes, default=0))
    debug_stats["candidate_group_count"] = int(sum(bundle_sizes))

    shortlist_cache = {}
    for shape, source_bundle in source_bundles.items():
        shape_key = bitmap_shape_key(shape)
        target_bundle = target_bundles_by_shape.get(str(shape_key))
        if target_bundle is None:
            continue
        if str(shape_key) not in shortlist_cache:
            shortlist_started = time.perf_counter()
            shortlist_cache[str(shape_key)] = _build_bundle_shortlist_index(target_bundle, debug_stats)
            detail_seconds["shortlist_index"] += time.perf_counter() - shortlist_started
        shortlist_index = shortlist_cache[str(shape_key)]
        coverage_by_group = []
        for grouped_candidates in source_bundle["candidate_groups"]:
            group_coverage = set()
            for candidate in grouped_candidates["candidates"]:
                group_coverage.update(int(value) for value in candidate.coverage)
            coverage_by_group.append(group_coverage)

        for source_idx, source_group in enumerate(source_bundle["candidate_groups"]):
            source_hash = str(source_group["representative"].clip_hash)
            same_hash_indices = target_bundle["hash_to_indices"].get(source_hash, [])
            for target_idx in same_hash_indices:
                before = int(len(coverage_by_group[source_idx]))
                coverage_by_group[source_idx].update(target_bundle["origin_ids"][int(target_idx)])
                added = int(len(coverage_by_group[source_idx])) - before
                debug_stats["exact_hash_pass"] += max(0, added)
                debug_stats["exact_hash_pairs"] += 1
            target_indices = _shortlist_target_indices(source_group, target_bundle, shortlist_index)
            if target_indices.size == 0:
                continue
            target_indices = target_indices[target_bundle["hashes_np"][target_indices] != source_hash]
            if target_indices.size == 0:
                continue
            prefilter_started = time.perf_counter()
            target_indices = _batch_prefilter(source_group, target_bundle, shortlist_index, target_indices, detail_seconds, debug_stats)
            detail_seconds["prefilter"] += time.perf_counter() - prefilter_started
            if target_indices.size == 0:
                continue
            debug_stats["geometry_pair_count"] += int(target_indices.size)
            matched_indices = _matched_target_indices(source_group, target_bundle, target_indices, config, tol_px, detail_seconds, debug_stats)
            matched_set = set(int(value) for value in matched_indices.tolist())
            debug_stats["geometry_pass"] += int(len(matched_set))
            debug_stats["geometry_reject"] += int(target_indices.size) - int(len(matched_set))
            for target_idx in matched_set:
                coverage_by_group[source_idx].update(target_bundle["origin_ids"][int(target_idx)])

        for group_idx, group in enumerate(source_bundle["candidate_groups"]):
            for candidate in group["candidates"]:
                candidate.coverage = set(coverage_by_group[int(group_idx)])

    debug_stats["coverage_detail_seconds"] = dict((key, float(value)) for key, value in detail_seconds.items())
    return debug_stats


def evaluate_candidate_coverage(candidates, exact_clusters, config):
    """集中式路径：用所有 candidates 构建全局 bundle，再按 v1 主算法计算 coverage。"""

    del exact_clusters
    target_bundles = build_candidate_match_bundles(candidates)
    return evaluate_candidate_coverage_against_bundles(candidates, dict((bitmap_shape_key(shape), bundle) for shape, bundle in target_bundles.items()), config)


def generate_candidates_for_cluster_range(exact_clusters, config, start, end):
    """为一段 exact cluster 生成 source candidates。"""

    candidates = []
    for cluster in exact_clusters[int(start) : int(end)]:
        candidates.extend(generate_candidates_for_cluster(cluster, config))
    return candidates


def candidate_metadata(candidate, include_coverage=True):
    """把 CandidateClip 的非 bitmap 字段转成 JSON metadata。"""

    bitmap_shape = []
    if candidate.clip_bitmap is not None:
        bitmap_shape = [int(candidate.clip_bitmap.shape[0]), int(candidate.clip_bitmap.shape[1])]
    payload = {
        "candidate_id": str(candidate.candidate_id),
        "origin_exact_cluster_id": int(candidate.origin_exact_cluster_id),
        "origin_exact_key": str(candidate.origin_exact_key),
        "center": list(candidate.center),
        "clip_bbox": list(candidate.clip_bbox),
        "clip_bbox_q": list(candidate.clip_bbox_q),
        "bitmap_shape": bitmap_shape,
        "clip_hash": str(candidate.clip_hash),
        "shift_direction": str(candidate.shift_direction),
        "shift_distance_um": float(candidate.shift_distance_um),
        "source_marker_id": str(candidate.source_marker_id),
        "origin_seed_type": str(getattr(candidate, "origin_seed_type", "unknown") or "unknown"),
        "risk_score": float(_candidate_risk_score(candidate)),
        "opc_center_base_score": float(_candidate_opc_center_base_score(candidate)),
    }
    if include_coverage:
        payload["coverage"] = [int(value) for value in sorted(candidate.coverage)]
    return payload


def candidate_from_metadata(metadata, bitmap):
    """从 metadata 和 bitmap 还原 CandidateClip。"""

    clip_bitmap = None
    if bitmap is not None:
        clip_bitmap = np.ascontiguousarray(bitmap, dtype=bool)
    return CandidateClip(
        candidate_id=str(metadata["candidate_id"]),
        origin_exact_cluster_id=int(metadata["origin_exact_cluster_id"]),
        origin_exact_key=str(metadata.get("origin_exact_key", "")),
        center=tuple(float(v) for v in metadata["center"]),
        clip_bbox=tuple(float(v) for v in metadata["clip_bbox"]),
        clip_bbox_q=tuple(int(v) for v in metadata["clip_bbox_q"]),
        clip_bitmap=clip_bitmap,
        clip_hash=str(metadata["clip_hash"]),
        shift_direction=str(metadata["shift_direction"]),
        shift_distance_um=float(metadata["shift_distance_um"]),
        coverage=set(int(value) for value in metadata.get("coverage", [])),
        source_marker_id=str(metadata["source_marker_id"]),
        origin_seed_type=str(metadata.get("origin_seed_type", "unknown") or "unknown"),
    )


def candidate_from_metadata_light(metadata):
    """只从 metadata 还原 CandidateClip，不加载 bitmap。"""

    return candidate_from_metadata(metadata, None)


def _coverage_arrays_for_candidates(candidates):
    """把候选 coverage set 压成 offsets/values 数组。"""

    offsets = [0]
    values = []
    for candidate in candidates:
        for exact_id in sorted(candidate.coverage):
            values.append(int(exact_id))
        offsets.append(int(len(values)))
    return np.asarray(offsets, dtype=np.int64), np.asarray(values, dtype=np.int64)


def _coverage_from_arrays(offsets, values, idx):
    """从 offsets/values 数组还原单个候选的 coverage set。"""

    start = int(offsets[int(idx)])
    end = int(offsets[int(idx) + 1])
    return set(int(value) for value in values[start:end])


def save_coverage_shard(candidates, json_path, npz_path, extra_payload):
    """保存 coverage shard 的 source candidates 与 coverage 结果。"""

    if candidates:
        bitmaps = np.asarray([candidate.clip_bitmap for candidate in candidates], dtype=bool)
    else:
        bitmaps = np.zeros((0, 0, 0), dtype=bool)
    coverage_offsets, coverage_values = _coverage_arrays_for_candidates(candidates)
    np.savez_compressed(
        str(npz_path),
        candidate_bitmaps=bitmaps,
        coverage_offsets=coverage_offsets,
        coverage_values=coverage_values,
    )
    payload = dict(extra_payload)
    payload["coverage_storage"] = "npz_offsets_v1"
    payload["coverage_value_count"] = int(len(coverage_values))
    metadata = []
    for idx, candidate in enumerate(candidates):
        item = candidate_metadata(candidate, include_coverage=False)
        item["bitmap_index"] = int(idx)
        item["coverage_index"] = int(idx)
        metadata.append(item)
    payload["candidates"] = metadata
    with Path(str(json_path)).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=json_default)


def load_coverage_shard_csr_metadata(json_path, npz_path):
    """读取 coverage shard 候选 metadata 和 CSR coverage 数组，不创建 CandidateClip。"""

    with Path(str(json_path)).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    metadata_items = list(payload.get("candidates", []))
    arrays = np.load(str(npz_path), allow_pickle=False)
    try:
        if "coverage_offsets" not in arrays.files or "coverage_values" not in arrays.files:
            raise RuntimeError("coverage shard npz missing CSR coverage arrays")
        offsets = np.asarray(arrays["coverage_offsets"], dtype=np.int64)
        values = np.asarray(arrays["coverage_values"], dtype=np.int64)
        return metadata_items, offsets, values, payload
    finally:
        arrays.close()


def load_coverage_candidate_bitmaps(npz_path, bitmap_indexes):
    """从 coverage shard npz 中按下标读取候选 bitmap。"""

    arrays = np.load(str(npz_path), allow_pickle=False)
    try:
        bitmaps = arrays["candidate_bitmaps"]
        loaded = {}
        for idx in sorted(set(int(value) for value in bitmap_indexes)):
            loaded[int(idx)] = np.ascontiguousarray(bitmaps[int(idx)], dtype=bool)
    finally:
        arrays.close()
    return loaded


def _csr_candidate_covered_set(offsets, values, candidate_index, uncovered):
    """返回单个 CSR candidate 当前能覆盖的 uncovered exact ids。"""

    start = int(offsets[int(candidate_index)])
    end = int(offsets[int(candidate_index) + 1])
    covered = set()
    for exact_id in values[start:end]:
        exact_id = int(exact_id)
        if exact_id in uncovered:
            covered.add(exact_id)
    return covered


def _metadata_representative_score(item, coverage_weight, total_weight):
    """基于轻量 metadata 估计 distributed greedy 的代表点评分。"""

    coverage_score = _clamp01(_safe_ratio(int(coverage_weight), max(int(total_weight), 1)))
    opc_center_score = _clamp01(float(item.get("opc_center_base_score", 0.0)) + 0.20 * coverage_score)
    return _clamp01(0.45 * opc_center_score + 0.35 * coverage_score + 0.20 * float(item.get("risk_score", 0.0)))


def _csr_candidate_priority(metadata, offsets, values, candidate_index, uncovered, weights, exact_hash_by_id, total_weight):
    """计算 CSR candidate 在当前 uncovered 集合下的 greedy priority。"""

    start = int(offsets[int(candidate_index)])
    end = int(offsets[int(candidate_index) + 1])
    covered_weight = 0
    count = 0
    seen = set()
    confidence_weighted_sum = 0.0
    confidence_weight = 0
    item = metadata[int(candidate_index)]
    candidate_hash = str(item.get("clip_hash", ""))
    origin_exact_id = int(item.get("origin_exact_cluster_id", 0))
    for exact_id in values[start:end]:
        exact_id = int(exact_id)
        if exact_id in uncovered and exact_id not in seen:
            seen.add(exact_id)
            count += 1
            exact_weight = int(weights.get(exact_id, 1))
            covered_weight += exact_weight
            if int(exact_id) == int(origin_exact_id):
                confidence = 1.0
            else:
                confidence = 1.0 if candidate_hash == str(exact_hash_by_id.get(int(exact_id), "")) else 0.8
            confidence_weighted_sum += float(confidence) * float(exact_weight)
            confidence_weight += int(exact_weight)
    representative_score = _metadata_representative_score(item, covered_weight, total_weight)
    confidence_proxy = float(_safe_ratio(confidence_weighted_sum, max(confidence_weight, 1)))
    return (
        -int(covered_weight),
        -int(count),
        -float(representative_score),
        -float(confidence_proxy),
        0 if str(item.get("shift_direction", "")) == "base" else 1,
        abs(float(item.get("shift_distance_um", 0.0))),
        int(item.get("origin_exact_cluster_id", 0)),
        str(item.get("candidate_id", "")),
    )


def greedy_cover_csr(candidate_metadata, coverage_offsets, coverage_values, exact_clusters, config=None):
    """基于 CSR coverage 数组执行 greedy set cover，避免为所有候选创建 coverage set。"""

    uncovered = set(int(cluster.exact_cluster_id) for cluster in exact_clusters)
    weights = dict((int(cluster.exact_cluster_id), int(cluster.weight)) for cluster in exact_clusters)
    exact_by_id = dict((int(cluster.exact_cluster_id), cluster) for cluster in exact_clusters)
    exact_hash_by_id = dict((int(cluster.exact_cluster_id), str(cluster.representative.clip_hash)) for cluster in exact_clusters)
    total_weight = max(int(sum(weights.values())), 1)
    base_by_exact = {}
    base_candidate_by_exact = dict((int(cluster.exact_cluster_id), _base_candidate_for_exact_cluster(cluster)) for cluster in exact_clusters)
    gate_config = config if isinstance(config, dict) else dict(config or {})
    gate_stats = _ensure_greedy_purity_gate_stats(gate_config)
    descriptor_cache = {}
    selected = []
    selected_indexes = set()
    rejected_indexes = set()
    heap = []
    for idx, metadata in enumerate(candidate_metadata):
        if str(metadata.get("shift_direction", "")) == "base":
            base_by_exact[int(metadata.get("origin_exact_cluster_id", -1))] = int(idx)
        heapq.heappush(
            heap,
            (
                _csr_candidate_priority(
                    candidate_metadata,
                    coverage_offsets,
                    coverage_values,
                    int(idx),
                    uncovered,
                    weights,
                    exact_hash_by_id,
                    total_weight,
                ),
                int(idx),
            ),
        )

    def _purity_gate_passes(metadata, covered_ids):
        return _greedy_purity_gate_passes(
            metadata,
            tuple(sorted(int(value) for value in covered_ids)),
            weights,
            total_weight,
            exact_by_id,
            base_candidate_by_exact,
            gate_config,
            gate_stats,
            descriptor_cache,
        )

    while uncovered:
        best = None
        covered_now = set()
        while heap:
            saved_priority, candidate_index = heapq.heappop(heap)
            if int(candidate_index) in selected_indexes or int(candidate_index) in rejected_indexes:
                continue
            current_priority = _csr_candidate_priority(
                candidate_metadata,
                coverage_offsets,
                coverage_values,
                int(candidate_index),
                uncovered,
                weights,
                exact_hash_by_id,
                total_weight,
            )
            if current_priority != saved_priority:
                heapq.heappush(heap, (current_priority, int(candidate_index)))
                continue
            current_covered = _csr_candidate_covered_set(coverage_offsets, coverage_values, int(candidate_index), uncovered)
            if current_covered:
                if not _purity_gate_passes(candidate_metadata[int(candidate_index)], current_covered):
                    rejected_indexes.add(int(candidate_index))
                    continue
                best = int(candidate_index)
                covered_now = current_covered
                break
        if best is None:
            missing = min(uncovered)
            best = int(base_by_exact[missing])
            covered_now = _csr_candidate_covered_set(coverage_offsets, coverage_values, best, uncovered)
            if not covered_now:
                covered_now = set([missing])
        selected.append(int(best))
        selected_indexes.add(int(best))
        uncovered -= covered_now
    gate_config["_greedy_purity_gate_rejected_ids"] = set(
        str(candidate_metadata[int(index)].get("candidate_id", "")) for index in rejected_indexes
    )
    return selected


def selected_candidates_from_csr(candidate_metadata, coverage_offsets, coverage_values, selected_indexes):
    """只把 selected CSR candidates 还原成 CandidateClip。"""

    selected = []
    for candidate_index in selected_indexes:
        metadata = candidate_metadata[int(candidate_index)]
        candidate = candidate_from_metadata_light(metadata)
        candidate.coverage = _coverage_from_arrays(coverage_offsets, coverage_values, int(candidate_index))
        selected.append(candidate)
    return selected


def greedy_cover(candidates, exact_clusters, config=None):
    """按 coverage 权重执行 greedy set cover。"""

    uncovered = set(int(cluster.exact_cluster_id) for cluster in exact_clusters)
    weights = dict((int(cluster.exact_cluster_id), int(cluster.weight)) for cluster in exact_clusters)
    exact_by_id = dict((int(cluster.exact_cluster_id), cluster) for cluster in exact_clusters)
    total_weight = max(int(sum(weights.values())), 1)
    gate_config = config if isinstance(config, dict) else dict(config or {})
    gate_stats = _ensure_greedy_purity_gate_stats(gate_config)
    descriptor_cache = {}
    base_by_exact = {}
    candidate_by_id = {}
    selected = []
    selected_ids = set()
    rejected_ids = set()
    heap = []
    for candidate in candidates:
        candidate_by_id[candidate.candidate_id] = candidate
        if candidate.shift_direction == "base":
            base_by_exact[int(candidate.origin_exact_cluster_id)] = candidate

    def _purity_gate_passes(candidate, covered_ids):
        return _greedy_purity_gate_passes(
            candidate,
            tuple(sorted(int(value) for value in covered_ids)),
            weights,
            total_weight,
            exact_by_id,
            base_by_exact,
            gate_config,
            gate_stats,
            descriptor_cache,
        )

    def priority(candidate):
        covered_now = set(candidate.coverage) & uncovered
        covered_weight = int(sum(weights.get(cid, 1) for cid in covered_now))
        representative_score = float(_candidate_representative_score(candidate, covered_weight, total_weight))
        confidence_weighted_sum = 0.0
        confidence_weight = 0
        for exact_id in covered_now:
            exact_cluster = exact_by_id.get(int(exact_id))
            if exact_cluster is None:
                continue
            exact_weight = int(weights.get(int(exact_id), 0))
            if exact_weight <= 0:
                continue
            if int(exact_id) == int(candidate.origin_exact_cluster_id):
                confidence = 1.0
            else:
                confidence = 1.0 if str(candidate.clip_hash) == str(exact_cluster.representative.clip_hash) else 0.8
            confidence_weighted_sum += float(confidence) * float(exact_weight)
            confidence_weight += exact_weight
        confidence_proxy = float(_safe_ratio(confidence_weighted_sum, max(confidence_weight, 1)))
        return (
            -covered_weight,
            -len(covered_now),
            -float(representative_score),
            -float(confidence_proxy),
            *_candidate_greedy_tiebreak(candidate),
        )

    for candidate in candidates:
        heapq.heappush(heap, (priority(candidate), candidate.candidate_id))
    while uncovered:
        best = None
        covered_now = set()
        while heap:
            saved_priority, candidate_id = heapq.heappop(heap)
            if candidate_id in selected_ids or candidate_id in rejected_ids:
                continue
            candidate = candidate_by_id[candidate_id]
            current_priority = priority(candidate)
            if current_priority != saved_priority:
                heapq.heappush(heap, (current_priority, candidate_id))
                continue
            current_covered = set(candidate.coverage) & uncovered
            if current_covered:
                if not _purity_gate_passes(candidate, current_covered):
                    rejected_ids.add(candidate_id)
                    continue
                best = candidate
                covered_now = current_covered
                break
        if best is None:
            missing = min(uncovered)
            best = base_by_exact[missing]
            covered_now = set(best.coverage) & uncovered
            if not covered_now:
                covered_now = set([missing])
        selected.append(best)
        selected_ids.add(best.candidate_id)
        uncovered -= covered_now
    gate_config["_greedy_purity_gate_rejected_ids"] = set(str(candidate_id) for candidate_id in rejected_ids)
    return selected


def _empty_verification_stats():
    """返回 final verification 统计字段。"""

    return {
        "verified_pass": 0,
        "verified_reject": 0,
        "singleton_created": 0,
        "witness_attempted": 0,
        "witness_verified_pass": 0,
    }


def _empty_final_verification_breakdown():
    """返回 final verification 通过与拒绝原因统计。"""

    return {
        "pass_reason_counts": {},
        "reject_reason_counts": {},
        "witness_shift_direction_counts": {},
    }


def _increment_named_counter(counter, name, amount=1):
    """对字符串键的计数器做自增。"""

    key = str(name) if name not in (None, "") else "unknown"
    counter[key] = int(counter.get(key, 0)) + int(amount)


def _assign_exact_clusters(selected_candidates, exact_clusters):
    """把 exact clusters 分配给最合适的 selected candidate，再进入最终验证。"""

    assignments = dict((str(candidate.candidate_id), []) for candidate in selected_candidates)
    exact_by_id = dict((int(cluster.exact_cluster_id), cluster) for cluster in exact_clusters)
    best_candidate_by_exact = {}
    best_key_by_exact = {}
    for candidate in selected_candidates:
        for exact_id in candidate.coverage:
            exact_cluster = exact_by_id.get(int(exact_id))
            if exact_cluster is None:
                continue
            key = (
                0 if str(candidate.clip_hash) == str(exact_cluster.representative.clip_hash) else 1,
                0 if int(candidate.origin_exact_cluster_id) == int(exact_cluster.exact_cluster_id) else 1,
                0 if str(candidate.shift_direction) == "base" else 1,
                abs(float(candidate.shift_distance_um)),
                str(candidate.candidate_id),
            )
            previous_key = best_key_by_exact.get(int(exact_id))
            if previous_key is None or key < previous_key:
                best_key_by_exact[int(exact_id)] = key
                best_candidate_by_exact[int(exact_id)] = candidate

    for exact_cluster in exact_clusters:
        exact_id = int(exact_cluster.exact_cluster_id)
        candidate = best_candidate_by_exact.get(exact_id)
        if candidate is None:
            raise RuntimeError("No selected candidate covers exact cluster %d" % int(exact_id))
        assignments[str(candidate.candidate_id)].append(exact_cluster)
    return assignments


def build_compact_result(marker_records, exact_clusters, candidates, selected_candidates, coverage_stats, config, runtime_summary):
    """构建最终 compact 结果字典。"""

    marker_count = len(marker_records) if marker_records is not None else sum(int(cluster.member_count) for cluster in exact_clusters)
    cluster_sizes = []
    clusters = []
    assigned = set()
    verification_stats = _empty_verification_stats()
    verification_breakdown = _empty_final_verification_breakdown()
    descriptor_cache = {}
    assignments = _assign_exact_clusters(selected_candidates, exact_clusters)
    for candidate in selected_candidates:
        assigned_exact_clusters = sorted(assignments.get(str(candidate.candidate_id), []), key=lambda item: int(item.exact_cluster_id))
        accepted_ids = []
        for exact_cluster in assigned_exact_clusters:
            exact_id = int(exact_cluster.exact_cluster_id)
            verification_stats["witness_attempted"] += 1
            matched, reason, witness_direction = _candidate_matches_exact(candidate, exact_cluster, config, descriptor_cache)
            if matched:
                accepted_ids.append(int(exact_id))
                assigned.add(int(exact_id))
                verification_stats["verified_pass"] += 1
                verification_stats["witness_verified_pass"] += 1
                _increment_named_counter(verification_breakdown["pass_reason_counts"], reason)
                _increment_named_counter(verification_breakdown["witness_shift_direction_counts"], witness_direction)
            else:
                verification_stats["verified_reject"] += 1
                verification_stats["singleton_created"] += 1
                _increment_named_counter(verification_breakdown["reject_reason_counts"], reason)
        if not accepted_ids:
            continue
        accepted_by_id = dict((int(cluster.exact_cluster_id), cluster) for cluster in assigned_exact_clusters)
        size = sum(int(accepted_by_id[exact_id].member_count) for exact_id in accepted_ids)
        distance_score = distance_worst_case_proxy(candidate.clip_bitmap)
        cluster_sizes.append(int(size))
        clusters.append(
            {
                "cluster_id": int(len(clusters)),
                "size": int(size),
                "selected_candidate_id": str(candidate.candidate_id),
                "selected_shift_direction": str(candidate.shift_direction),
                "selected_shift_distance_um": float(candidate.shift_distance_um),
                "distance_worst_case_score": float(distance_score),
                "exact_cluster_ids": accepted_ids,
                "representative_marker_id": str(candidate.source_marker_id),
            }
        )
    for exact_cluster in exact_clusters:
        if int(exact_cluster.exact_cluster_id) in assigned:
            continue
        cluster_sizes.append(int(exact_cluster.member_count))
        clusters.append(
            {
                "cluster_id": int(len(clusters)),
                "size": int(exact_cluster.member_count),
                "selected_candidate_id": "singleton_%06d" % int(exact_cluster.exact_cluster_id),
                "selected_shift_direction": "base",
                "selected_shift_distance_um": 0.0,
                "distance_worst_case_score": 0.0,
                "exact_cluster_ids": [int(exact_cluster.exact_cluster_id)],
                "representative_marker_id": str(exact_cluster.representative.marker_id),
            }
        )
    coverage_detail_seconds = dict(coverage_stats.get("coverage_detail_seconds", {}))
    coverage_debug_stats = dict((key, value) for key, value in coverage_stats.items() if key != "coverage_detail_seconds")
    greedy_gate_stats = dict(config.get("_greedy_purity_gate_stats", {}) if isinstance(config, dict) else {})
    for key, value in greedy_gate_stats.items():
        coverage_debug_stats[str(key)] = value
    shift_summary = candidate_shift_summary(candidates)
    selected_shift_summary = candidate_shift_summary(selected_candidates)
    candidate_group_count = int(coverage_debug_stats.get("candidate_group_count", 0))
    return {
        "pipeline_mode": PIPELINE_MODE,
        "seed_mode": SEED_MODE,
        "seed_strategy": "geometry_driven",
        "grid_step_ratio": float(GRID_STEP_RATIO),
        "geometry_match_mode": str(config.get("geometry_match_mode", "ecc")),
        "pixel_size_nm": int(config.get("pixel_size_nm", DEFAULT_PIXEL_SIZE_NM)),
        "clip_size_um": float(config.get("clip_size_um", 1.35)),
        "marker_count": int(marker_count),
        "exact_cluster_count": int(len(exact_clusters)),
        "candidate_count": int(len(candidates)),
        "candidate_group_count": int(candidate_group_count),
        "selected_candidate_count": int(len(selected_candidates)),
        "total_clusters": int(len(clusters)),
        "total_samples": int(marker_count),
        "cluster_sizes": cluster_sizes,
        "coverage_detail_seconds": coverage_detail_seconds,
        "coverage_debug_stats": coverage_debug_stats,
        "final_verification_stats": dict(verification_stats),
        "final_verification_breakdown": dict(verification_breakdown),
        "candidate_direction_counts": dict(shift_summary["candidate_direction_counts"]),
        "diagonal_candidate_count": int(shift_summary["diagonal_candidate_count"]),
        "max_shift_distance_um": float(shift_summary["max_shift_distance_um"]),
        "selected_candidate_direction_counts": dict(selected_shift_summary["candidate_direction_counts"]),
        "selected_diagonal_candidate_count": int(selected_shift_summary["diagonal_candidate_count"]),
        "clusters": clusters,
        "result_summary": {
            "pipeline_mode": PIPELINE_MODE,
            "seed_mode": SEED_MODE,
            "seed_strategy": "geometry_driven",
            "grid_step_ratio": float(GRID_STEP_RATIO),
            "candidate_direction_counts": dict(shift_summary["candidate_direction_counts"]),
            "diagonal_candidate_count": int(shift_summary["diagonal_candidate_count"]),
            "max_shift_distance_um": float(shift_summary["max_shift_distance_um"]),
            "coverage_detail_seconds": coverage_detail_seconds,
            "coverage_debug_stats": coverage_debug_stats,
            "final_verification_stats": dict(verification_stats),
            "final_verification_breakdown": dict(verification_breakdown),
            "candidate_group_count": int(candidate_group_count),
            "timing_seconds": dict(runtime_summary),
        },
    }


def _base_candidate_for_exact_cluster(exact_cluster):
    """把 exact cluster representative 映射成 marker-defined base candidate。"""

    representative = exact_cluster.representative
    return CandidateClip(
        candidate_id="exact_%06d_base" % int(exact_cluster.exact_cluster_id),
        origin_exact_cluster_id=int(exact_cluster.exact_cluster_id),
        origin_exact_key=str(exact_cluster.exact_key),
        center=tuple(getattr(representative, "marker_center", ()) or _bbox_center(representative.clip_bbox)),
        clip_bbox=tuple(representative.clip_bbox),
        clip_bbox_q=tuple(int(value) for value in representative.clip_bbox_q),
        clip_bitmap=np.ascontiguousarray(representative.clip_bitmap, dtype=bool),
        clip_hash=str(representative.clip_hash),
        shift_direction="base",
        shift_distance_um=0.0,
        coverage=set([int(exact_cluster.exact_cluster_id)]),
        source_marker_id=str(representative.marker_id),
        origin_seed_type=_exact_cluster_seed_type(exact_cluster),
    )


def _candidate_matches_exact_loose(candidate, exact_cluster, config):
    """执行宽松版 exact 匹配：exact hash 或 geometry 通过即视为 recall 候选。"""

    for witness in _target_witness_candidates(exact_cluster, config):
        if str(candidate.clip_hash) == str(witness.clip_hash):
            return True
        geometry_ok, _ = _geometry_match_reason(candidate, witness, config)
        if geometry_ok:
            return True
    return False


def _candidate_field(candidate_or_metadata, name, default=None):
    """兼容 CandidateClip 和 metadata dict 读取候选字段。"""

    if isinstance(candidate_or_metadata, dict):
        return candidate_or_metadata.get(name, default)
    return getattr(candidate_or_metadata, name, default)


def _candidate_evidence_rows(candidates, coverage_offsets=None, coverage_values=None):
    """把集中式 CandidateClip 或分布式 metadata+CSR 统一成 coverage evidence 行。"""

    rows = []
    offsets = coverage_offsets
    values = coverage_values
    if offsets is not None:
        offsets = np.asarray(offsets, dtype=np.int64)
    if values is not None:
        values = np.asarray(values, dtype=np.int64)
    for idx, item in enumerate(candidates or []):
        if offsets is not None and values is not None and int(idx) + 1 < len(offsets):
            coverage = _coverage_from_arrays(offsets, values, int(idx))
        elif isinstance(item, dict):
            coverage = set(int(value) for value in item.get("coverage", []))
        else:
            coverage = set(int(value) for value in getattr(item, "coverage", set()))
        rows.append(
            {
                "candidate_id": str(_candidate_field(item, "candidate_id", "")),
                "origin_exact_cluster_id": int(_candidate_field(item, "origin_exact_cluster_id", -1)),
                "origin_seed_type": str(_candidate_field(item, "origin_seed_type", "unknown") or "unknown"),
                "shift_direction": str(_candidate_field(item, "shift_direction", "unknown")),
                "shift_distance_um": float(_candidate_field(item, "shift_distance_um", 0.0) or 0.0),
                "coverage": set(int(value) for value in coverage),
                "item": item,
            }
        )
    return rows


def _candidate_clip_from_evidence(evidence, candidate_bitmap_locations=None):
    """从 evidence 行恢复 CandidateClip；分布式路径按需加载单个 bitmap。"""

    item = evidence.get("item")
    coverage = set(int(value) for value in evidence.get("coverage", set()))
    if isinstance(item, CandidateClip):
        item.coverage = set(coverage)
        return item
    if not isinstance(item, dict):
        return None
    metadata = dict(item)
    bitmap = None
    location = dict((candidate_bitmap_locations or {}).get(str(metadata.get("candidate_id", "")), {}) or {})
    if location:
        loaded = load_coverage_candidate_bitmaps(location.get("npz_path"), [int(location.get("bitmap_index", 0))])
        bitmap = loaded.get(int(location.get("bitmap_index", 0)))
    candidate = candidate_from_metadata(metadata, bitmap)
    candidate.coverage = set(coverage)
    return candidate


def _cluster_units_from_result(result, exact_clusters, selected_candidates):
    """把 build_compact_result 的 cluster dict 还原成后续质量评估使用的 cluster units。"""

    exact_by_id = dict((int(cluster.exact_cluster_id), cluster) for cluster in exact_clusters)
    selected_by_id = dict((str(candidate.candidate_id), candidate) for candidate in selected_candidates)
    units = []
    for cluster in sorted(result.get("clusters", []), key=lambda item: int(item.get("cluster_id", 0))):
        exact_ids = [int(exact_id) for exact_id in cluster.get("exact_cluster_ids", [])]
        assigned = [exact_by_id[int(exact_id)] for exact_id in exact_ids if int(exact_id) in exact_by_id]
        if not assigned:
            continue
        candidate = selected_by_id.get(str(cluster.get("selected_candidate_id", "")))
        if candidate is None:
            candidate = _base_candidate_for_exact_cluster(assigned[0])
        units.append((candidate, assigned))
    return units


def _materialize_result_from_cluster_units(result, cluster_units, exact_clusters, config):
    """把 safe recall merge 后的 cluster units 写回 result 主结构。"""

    exact_by_id = dict((int(cluster.exact_cluster_id), cluster) for cluster in exact_clusters)
    total_exact_weight = max(int(sum(_exact_cluster_weight(cluster) for cluster in exact_clusters)), 1)
    clusters = []
    cluster_sizes = []
    for cluster_index, (candidate, assigned_exact_clusters) in enumerate(cluster_units):
        exact_ids = [int(cluster.exact_cluster_id) for cluster in assigned_exact_clusters]
        cluster_size = int(sum(int(cluster.member_count) for cluster in assigned_exact_clusters))
        cluster_weight = int(sum(_exact_cluster_weight(cluster) for cluster in assigned_exact_clusters))
        risk_score = float(_candidate_risk_score(candidate)) if getattr(candidate, "clip_bitmap", None) is not None else 0.0
        representative_score = float(_candidate_representative_score(candidate, cluster_weight, total_exact_weight))
        opc_center_score = float(_candidate_opc_center_score(candidate, cluster_weight, total_exact_weight))
        center = list(getattr(candidate, "center", ()) or [])
        cluster_sizes.append(int(cluster_size))
        clusters.append(
            {
                "cluster_id": int(cluster_index),
                "size": int(cluster_size),
                "selected_candidate_id": str(candidate.candidate_id),
                "selected_shift_direction": str(candidate.shift_direction),
                "selected_shift_distance_um": float(candidate.shift_distance_um),
                "distance_worst_case_score": float(distance_worst_case_proxy(getattr(candidate, "clip_bitmap", None))),
                "exact_cluster_ids": exact_ids,
                "representative_marker_id": str(getattr(candidate, "source_marker_id", "")),
                "selected_center": list(center),
                "representative_seed_type": str(_candidate_origin_seed_type(candidate, exact_by_id)),
                "representative_score": float(representative_score),
                "opc_center_score": float(opc_center_score),
                "risk_score": float(risk_score),
            }
        )
    result["clusters"] = clusters
    result["cluster_sizes"] = cluster_sizes
    result["total_clusters"] = int(len(clusters))
    result.setdefault("result_summary", {})["total_clusters"] = int(len(clusters))


def _exact_pair_matches_loose_by_base(left_exact_id, right_exact_id, exact_by_id, base_candidate_by_exact, config):
    """用 exact cluster 的 base candidate 双向执行宽松 geometry pairwise 判断。"""

    left_candidate = base_candidate_by_exact.get(int(left_exact_id))
    right_candidate = base_candidate_by_exact.get(int(right_exact_id))
    left_cluster = exact_by_id.get(int(left_exact_id))
    right_cluster = exact_by_id.get(int(right_exact_id))
    if left_candidate is None or right_candidate is None or left_cluster is None or right_cluster is None:
        return None
    if _candidate_matches_exact_loose(left_candidate, right_cluster, config):
        return True
    return bool(_candidate_matches_exact_loose(right_candidate, left_cluster, config))


def _build_coverage_graph_fragmentation_metrics_exact_review(candidate_evidence_rows, selected_candidates, cluster_payloads, exact_by_id, rejected_candidate_ids):
    """复用 coverage 边分层审计 fragmentation，不把 gate reject 和 review 边混成单个 recall。"""

    exact_to_final_cluster = {}
    final_cluster_exact_counts = {}
    accepted_exact_ids_by_candidate = {}
    for payload in cluster_payloads:
        exact_ids = [int(exact_id) for exact_id in payload.get("exact_ids", [])]
        cluster_index = int(payload.get("cluster_index", 0))
        final_cluster_exact_counts[cluster_index] = int(len(exact_ids))
        for exact_id in exact_ids:
            exact_to_final_cluster[int(exact_id)] = int(cluster_index)
        candidate = payload.get("candidate")
        if candidate is not None:
            accepted_exact_ids_by_candidate[str(candidate.candidate_id)] = set(exact_ids)

    exact_weights = dict((int(exact_id), int(_exact_cluster_weight(cluster))) for exact_id, cluster in exact_by_id.items())
    selected_ids = set(str(candidate.candidate_id) for candidate in selected_candidates)
    rejected_ids = set(str(candidate_id) for candidate_id in rejected_candidate_ids or set())
    raw_seen_edges = set()
    raw_edge_count = 0
    raw_cross_edge_count = 0
    raw_edge_weight = 0
    raw_cross_edge_weight = 0
    gate_rejected_edge_count = 0
    gate_rejected_edge_weight = 0
    review_merge_candidate_edge_count = 0
    review_merge_candidate_edge_weight = 0
    review_merge_candidate_rows = []
    review_merge_weight_by_tier = Counter()
    review_merge_edge_count_by_tier = Counter()
    high_conf_singleton_mergeable_edge_weight = 0
    singleton_trusted_mergeable_edge_count = 0
    singleton_trusted_mergeable_edge_weight = 0

    for evidence in candidate_evidence_rows:
        candidate_id = str(evidence.get("candidate_id", ""))
        source_id = int(evidence.get("origin_exact_cluster_id", -1))
        source_final_cluster = exact_to_final_cluster.get(source_id)
        if source_final_cluster is None:
            continue
        candidate_seed_type = str(evidence.get("origin_seed_type", "unknown"))
        for target_id_raw in sorted(evidence.get("coverage", set())):
            target_id = int(target_id_raw)
            if target_id == source_id or target_id not in exact_to_final_cluster:
                continue
            edge_key = (source_id, target_id)
            if edge_key in raw_seen_edges:
                continue
            raw_seen_edges.add(edge_key)
            edge_weight = max(int(exact_weights.get(target_id, 0)), 0)
            raw_edge_count += 1
            raw_edge_weight += int(edge_weight)
            target_final_cluster = int(exact_to_final_cluster[target_id])
            is_cross_cluster = target_final_cluster != int(source_final_cluster)
            if is_cross_cluster:
                raw_cross_edge_count += 1
                raw_cross_edge_weight += int(edge_weight)
            if candidate_id in rejected_ids:
                gate_rejected_edge_count += 1
                gate_rejected_edge_weight += int(edge_weight)
            elif candidate_id not in selected_ids and is_cross_cluster:
                review_merge_candidate_edge_count += 1
                review_merge_candidate_edge_weight += int(edge_weight)
                target_seed_type = _exact_cluster_seed_type(exact_by_id[target_id])
                confidence_tier, confidence_reason = _review_merge_confidence(
                    candidate_seed_type,
                    target_seed_type,
                    float(evidence.get("shift_distance_um", 0.0)),
                )
                target_is_singleton = int(final_cluster_exact_counts.get(target_final_cluster, 0)) == 1
                review_merge_weight_by_tier[str(confidence_tier)] += int(edge_weight)
                review_merge_edge_count_by_tier[str(confidence_tier)] += 1
                if target_is_singleton:
                    singleton_trusted_mergeable_edge_count += 1
                    singleton_trusted_mergeable_edge_weight += int(edge_weight)
                    if str(confidence_tier) == "high":
                        high_conf_singleton_mergeable_edge_weight += int(edge_weight)
                _append_bounded_review_merge_row(
                    review_merge_candidate_rows,
                    {
                        "source_exact_cluster_id": int(source_id),
                        "target_exact_cluster_id": int(target_id),
                        "source_cluster_id": int(source_final_cluster) + 1,
                        "target_cluster_id": int(target_final_cluster) + 1,
                        "candidate_id": str(candidate_id),
                        "edge_weight": int(edge_weight),
                        "candidate_seed_type": str(candidate_seed_type),
                        "target_seed_type": str(target_seed_type),
                        "shift_direction": str(evidence.get("shift_direction", "unknown")),
                        "shift_distance_um": float(evidence.get("shift_distance_um", 0.0)),
                        "confidence_tier": str(confidence_tier),
                        "confidence_reason": str(confidence_reason),
                        "target_is_singleton": bool(target_is_singleton),
                        "source_cluster_exact_count": int(final_cluster_exact_counts.get(source_final_cluster, 0)),
                        "target_cluster_exact_count": int(final_cluster_exact_counts.get(target_final_cluster, 0)),
                    },
                )

    trusted_seen_edges = set()
    trusted_edge_count = 0
    trusted_cross_edge_count = 0
    trusted_edge_weight = 0
    trusted_cross_edge_weight = 0
    for candidate in selected_candidates:
        candidate_id = str(candidate.candidate_id)
        source_id = int(candidate.origin_exact_cluster_id)
        source_final_cluster = exact_to_final_cluster.get(source_id)
        if source_final_cluster is None:
            continue
        coverage_ids = set(int(value) for value in getattr(candidate, "coverage", set()))
        accepted_ids = accepted_exact_ids_by_candidate.get(candidate_id, set())
        for target_id in sorted(accepted_ids):
            if target_id == source_id or target_id not in coverage_ids or target_id not in exact_to_final_cluster:
                continue
            edge_key = (source_id, target_id)
            if edge_key in trusted_seen_edges:
                continue
            trusted_seen_edges.add(edge_key)
            edge_weight = max(int(exact_weights.get(target_id, 0)), 0)
            trusted_edge_count += 1
            trusted_edge_weight += int(edge_weight)
            target_final_cluster = int(exact_to_final_cluster[target_id])
            if target_final_cluster != int(source_final_cluster):
                trusted_cross_edge_count += 1
                trusted_cross_edge_weight += int(edge_weight)
                if int(final_cluster_exact_counts.get(target_final_cluster, 0)) == 1:
                    singleton_trusted_mergeable_edge_count += 1
                    singleton_trusted_mergeable_edge_weight += int(edge_weight)

    raw_cross_ratio = float(_safe_ratio(raw_cross_edge_weight, raw_edge_weight))
    trusted_cross_ratio = float(_safe_ratio(trusted_cross_edge_weight, trusted_edge_weight))
    trusted_or_review_weight = int(trusted_edge_weight) + int(review_merge_candidate_edge_weight)
    bounded_review_merge_rows = _tier_balanced_review_merge_candidate_rows(review_merge_candidate_rows)
    return {
        "raw_coverage_graph_edge_count": int(raw_edge_count),
        "raw_coverage_graph_cross_cluster_edge_count": int(raw_cross_edge_count),
        "raw_coverage_graph_edge_weight": int(raw_edge_weight),
        "raw_coverage_graph_cross_cluster_edge_weight": int(raw_cross_edge_weight),
        "raw_coverage_graph_recall": float(_clamp01(1.0 - raw_cross_ratio)),
        "raw_coverage_graph_cross_cluster_edge_weight_ratio": float(raw_cross_ratio),
        "trusted_fragmentation_edge_count": int(trusted_edge_count),
        "trusted_fragmentation_cross_edge_count": int(trusted_cross_edge_count),
        "trusted_fragmentation_edge_weight": int(trusted_edge_weight),
        "trusted_fragmentation_cross_edge_weight": int(trusted_cross_edge_weight),
        "trusted_fragmentation_recall": float(_clamp01(1.0 - trusted_cross_ratio)),
        "gate_rejected_edge_count": int(gate_rejected_edge_count),
        "gate_rejected_edge_weight": int(gate_rejected_edge_weight),
        "gate_rejected_edge_weight_ratio": float(_safe_ratio(gate_rejected_edge_weight, raw_edge_weight)),
        "review_merge_candidate_edge_count": int(review_merge_candidate_edge_count),
        "review_merge_candidate_edge_weight": int(review_merge_candidate_edge_weight),
        "review_merge_candidate_weight_ratio": float(_safe_ratio(review_merge_candidate_edge_weight, raw_edge_weight)),
        "high_conf_review_merge_edge_count": int(review_merge_edge_count_by_tier.get("high", 0)),
        "medium_conf_review_merge_edge_count": int(review_merge_edge_count_by_tier.get("medium", 0)),
        "low_conf_review_merge_edge_count": int(review_merge_edge_count_by_tier.get("low", 0)),
        "high_conf_review_merge_weight_ratio": float(_safe_ratio(review_merge_weight_by_tier.get("high", 0), raw_edge_weight)),
        "medium_conf_review_merge_weight_ratio": float(_safe_ratio(review_merge_weight_by_tier.get("medium", 0), raw_edge_weight)),
        "low_conf_review_merge_weight_ratio": float(_safe_ratio(review_merge_weight_by_tier.get("low", 0), raw_edge_weight)),
        "high_conf_singleton_mergeable_weight_ratio": float(_safe_ratio(high_conf_singleton_mergeable_edge_weight, raw_edge_weight)),
        "singleton_trusted_mergeable_edge_count": int(singleton_trusted_mergeable_edge_count),
        "singleton_trusted_mergeable_edge_weight": int(singleton_trusted_mergeable_edge_weight),
        "singleton_trusted_mergeable_weight_ratio": float(_safe_ratio(singleton_trusted_mergeable_edge_weight, trusted_or_review_weight)),
        "review_merge_candidate_rows": bounded_review_merge_rows,
    }


def _build_quality_metrics_exact_review(cluster_units, exact_clusters, candidate_evidence_rows, selected_candidates, config, final_verification_stats, coverage_debug_stats, rejected_candidate_ids):
    """按问题域构建 representative、pairwise geometry 与 fragmentation 质量指标。"""

    exact_by_id = dict((int(cluster.exact_cluster_id), cluster) for cluster in exact_clusters)
    base_candidate_by_id = dict((int(cluster.exact_cluster_id), _base_candidate_for_exact_cluster(cluster)) for cluster in exact_clusters)
    total_weight = int(sum(_exact_cluster_weight(cluster) for cluster in exact_clusters))
    cluster_payloads = []
    for cluster_index, (candidate, assigned_exact_clusters) in enumerate(cluster_units):
        exact_ids = [int(cluster.exact_cluster_id) for cluster in assigned_exact_clusters]
        sample_count = int(sum(int(exact_by_id[exact_id].member_count) for exact_id in exact_ids if exact_id in exact_by_id))
        cluster_weight = int(sum(_exact_cluster_weight(exact_by_id[exact_id]) for exact_id in exact_ids if exact_id in exact_by_id))
        cluster_payloads.append(
            {
                "cluster_index": int(cluster_index),
                "exact_ids": exact_ids,
                "sample_count": int(sample_count),
                "cluster_weight": int(cluster_weight),
                "representative_seed_type": str(_candidate_origin_seed_type(candidate, exact_by_id)),
                "shift_distance_um": float(getattr(candidate, "shift_distance_um", 0.0)),
                "candidate": candidate,
                "assigned_exact_clusters": list(assigned_exact_clusters),
            }
        )

    singleton_clusters = [payload for payload in cluster_payloads if int(payload["sample_count"]) <= 1]
    merged_repeat_weight = int(sum(int(payload["cluster_weight"]) for payload in cluster_payloads if int(payload["sample_count"]) > 1))
    visual_pair_count = 0
    visual_fail_count = 0
    visual_sampled_cluster_count = 0
    unknown_cluster_count = 0
    no_pair_cluster_count = 0
    weighted_pairwise_geometry_purity_sum = 0.0
    weighted_pairwise_geometry_purity_weight = 0
    low_pairwise_geometry_weight = 0
    low_shift_low_pairwise_review_weight = 0
    low_shift_low_pairwise_review_cluster_count = 0
    representative_visual_checked_count = 0
    representative_visual_pass_count = 0
    representative_visual_fail_count = 0
    representative_visual_checked_weight = 0
    representative_visual_pass_weight = 0
    representative_visual_fail_weight = 0
    weighted_visual_purity_sum = 0.0
    weighted_visual_purity_weight = 0
    low_visual_purity_weight = 0
    representative_quality_by_index = {}
    cluster_quality_by_index = {}

    for payload in cluster_payloads:
        candidate = payload["candidate"]
        local_rep_checked = 0
        local_rep_pass = 0
        local_rep_fail = 0
        local_rep_checked_weight = 0
        local_rep_pass_weight = 0
        local_rep_fail_weight = 0
        for exact_cluster in payload["assigned_exact_clusters"]:
            exact_weight = int(_exact_cluster_weight(exact_cluster))
            matched = _candidate_matches_exact_loose(candidate, exact_cluster, config)
            local_rep_checked += 1
            local_rep_checked_weight += int(exact_weight)
            representative_visual_checked_count += 1
            representative_visual_checked_weight += int(exact_weight)
            if matched:
                local_rep_pass += 1
                local_rep_pass_weight += int(exact_weight)
                representative_visual_pass_count += 1
                representative_visual_pass_weight += int(exact_weight)
            else:
                local_rep_fail += 1
                local_rep_fail_weight += int(exact_weight)
                representative_visual_fail_count += 1
                representative_visual_fail_weight += int(exact_weight)
        local_rep_weighted_pass_ratio = float(_safe_ratio(local_rep_pass_weight, local_rep_checked_weight))
        weighted_visual_purity_sum += local_rep_weighted_pass_ratio * float(payload["cluster_weight"])
        weighted_visual_purity_weight += int(payload["cluster_weight"])
        if local_rep_weighted_pass_ratio < float(QUALITY_LOW_VISUAL_PURITY_THRESHOLD):
            low_visual_purity_weight += int(payload["cluster_weight"])
        representative_quality_by_index[int(payload["cluster_index"])] = {
            "representative_visual_pass_ratio": float(_safe_ratio(local_rep_pass, local_rep_checked)),
            "representative_visual_fail_count": int(local_rep_fail),
            "representative_visual_checked_count": int(local_rep_checked),
            "representative_visual_sample_status": "sampled" if local_rep_checked > 0 else "no_member",
            "representative_visual_weighted_pass_ratio": local_rep_weighted_pass_ratio,
            "representative_visual_reject_weight_ratio": float(_safe_ratio(local_rep_fail_weight, local_rep_checked_weight)),
        }

        exact_ids = sorted(int(exact_id) for exact_id in payload["exact_ids"])
        local_visual_pair_count = 0
        local_visual_fail_count = 0
        if len(exact_ids) < 2:
            cluster_quality = dict(representative_quality_by_index[int(payload["cluster_index"])])
            cluster_quality.update(
                {
                    "pairwise_geometry_purity": None,
                    "pairwise_geometry_fail_count": 0,
                    "pairwise_geometry_sampled_pair_count": 0,
                    "pairwise_geometry_sample_status": "no_pair",
                }
            )
            cluster_quality_by_index[int(payload["cluster_index"])] = cluster_quality
            no_pair_cluster_count += 1
            continue
        for source_exact_id, target_exact_id in _sample_exact_id_pairs(exact_ids, QUALITY_PER_CLUSTER_INTRA_PAIR_LIMIT):
            matched = _exact_pair_matches_loose_by_base(source_exact_id, target_exact_id, exact_by_id, base_candidate_by_id, config)
            if matched is None:
                continue
            visual_pair_count += 1
            local_visual_pair_count += 1
            if not matched:
                visual_fail_count += 1
                local_visual_fail_count += 1

        visual_purity = None
        visual_status = "unknown"
        if local_visual_pair_count > 0:
            visual_purity = float(_clamp01(1.0 - _safe_ratio(local_visual_fail_count, local_visual_pair_count)))
            visual_status = "sampled"
            visual_sampled_cluster_count += 1
            weighted_pairwise_geometry_purity_sum += float(visual_purity) * float(payload["cluster_weight"])
            weighted_pairwise_geometry_purity_weight += int(payload["cluster_weight"])
            if float(visual_purity) < float(QUALITY_LOW_VISUAL_PURITY_THRESHOLD):
                low_pairwise_geometry_weight += int(payload["cluster_weight"])
            if _is_low_shift_low_pairwise_review(
                payload["shift_distance_um"],
                visual_purity,
                visual_status,
                sampled_pair_count=local_visual_pair_count,
            ):
                low_shift_low_pairwise_review_cluster_count += 1
                low_shift_low_pairwise_review_weight += int(payload["cluster_weight"])
        if visual_status == "unknown":
            unknown_cluster_count += 1
        cluster_quality = dict(representative_quality_by_index[int(payload["cluster_index"])])
        cluster_quality.update(
            {
                "pairwise_geometry_purity": visual_purity,
                "pairwise_geometry_fail_count": int(local_visual_fail_count),
                "pairwise_geometry_sampled_pair_count": int(local_visual_pair_count),
                "pairwise_geometry_sample_status": visual_status,
            }
        )
        cluster_quality_by_index[int(payload["cluster_index"])] = cluster_quality

    coverage_graph_metrics = _build_coverage_graph_fragmentation_metrics_exact_review(
        candidate_evidence_rows,
        selected_candidates,
        cluster_payloads,
        exact_by_id,
        rejected_candidate_ids,
    )
    visual_fail_rate = float(_safe_ratio(visual_fail_count, visual_pair_count))
    for payload in cluster_payloads:
        cluster_index = int(payload["cluster_index"])
        metrics = cluster_quality_by_index[int(cluster_index)]
        score, reason = _overmerge_score_and_reason(
            int(payload["cluster_weight"]),
            max(int(total_weight), 1),
            int(len(payload["exact_ids"])),
            str(payload["representative_seed_type"]),
            float(payload["shift_distance_um"]),
            float(config.get("clip_size_um", 1.35)),
            float(metrics.get("representative_visual_pass_ratio", 0.0)),
            metrics.get("pairwise_geometry_purity"),
            str(metrics.get("pairwise_geometry_sample_status", "unknown")),
        )
        metrics["overmerge_score"] = float(score)
        metrics["overmerge_reason"] = str(reason)
    review_merge_cluster_pair_rows = _review_merge_cluster_pair_rows(
        coverage_graph_metrics.get("review_merge_candidate_rows", []),
        _endpoint_quality_by_cluster_id(cluster_payloads, cluster_quality_by_index),
    )
    representative_visual_purity = float(_safe_ratio(representative_visual_pass_count, representative_visual_checked_count))
    weighted_representative_visual_purity = float(_safe_ratio(weighted_visual_purity_sum, weighted_visual_purity_weight))
    low_representative_quality_weight_ratio = float(_safe_ratio(low_visual_purity_weight, total_weight))
    return {
        "singleton_ratio": float(_safe_ratio(len(singleton_clusters), len(cluster_payloads))),
        "singleton_weight_ratio": float(_safe_ratio(sum(int(payload["cluster_weight"]) for payload in singleton_clusters), total_weight)),
        "verified_pass_ratio": float(_safe_ratio(int(final_verification_stats.get("verified_pass", 0)), max(len(exact_clusters), 1))),
        "merged_repeat_weight_ratio": float(_safe_ratio(merged_repeat_weight, total_weight)),
        "representative_visual_pass_ratio": float(_safe_ratio(representative_visual_pass_count, representative_visual_checked_count)),
        "representative_visual_weighted_pass_ratio": float(_safe_ratio(representative_visual_pass_weight, representative_visual_checked_weight)),
        "representative_visual_reject_weight_ratio": float(_safe_ratio(representative_visual_fail_weight, representative_visual_checked_weight)),
        "representative_visual_checked_count": int(representative_visual_checked_count),
        "representative_visual_fail_count": int(representative_visual_fail_count),
        "pairwise_geometry_fail_rate": float(visual_fail_rate),
        "pairwise_geometry_purity": float(_clamp01(1.0 - visual_fail_rate)),
        "weighted_pairwise_geometry_purity": float(_safe_ratio(weighted_pairwise_geometry_purity_sum, weighted_pairwise_geometry_purity_weight)),
        "low_pairwise_geometry_weight_ratio": float(_safe_ratio(low_pairwise_geometry_weight, total_weight)),
        "representative_visual_purity": representative_visual_purity,
        "weighted_representative_visual_purity": weighted_representative_visual_purity,
        "low_representative_quality_weight_ratio": low_representative_quality_weight_ratio,
        "visual_purity_score": representative_visual_purity,
        "weighted_visual_purity": weighted_representative_visual_purity,
        "low_visual_purity_weight_ratio": low_representative_quality_weight_ratio,
        "pairwise_geometry_sampled_cluster_count": int(visual_sampled_cluster_count),
        "pairwise_geometry_unknown_cluster_count": int(unknown_cluster_count),
        "pairwise_geometry_no_pair_cluster_count": int(no_pair_cluster_count),
        "low_shift_low_pairwise_review_weight_ratio": float(_safe_ratio(low_shift_low_pairwise_review_weight, total_weight)),
        "low_shift_low_pairwise_review_cluster_count": int(low_shift_low_pairwise_review_cluster_count),
        "pairwise_review_sampled_weight_ratio": float(_safe_ratio(weighted_pairwise_geometry_purity_weight, total_weight)),
        "donut_auto_pass_pair_count": int(coverage_debug_stats.get("donut_auto_pass_pair_count", 0)),
        "pairwise_geometry_sampled_pair_count": int(visual_pair_count),
        **dict(coverage_graph_metrics),
        "review_merge_cluster_pair_rows": review_merge_cluster_pair_rows,
        "cluster_quality_by_index": dict(
            (
                int(cluster_index),
                {
                    "representative_visual_pass_ratio": float(metrics.get("representative_visual_pass_ratio", 0.0)),
                    "representative_visual_fail_count": int(metrics.get("representative_visual_fail_count", 0)),
                    "representative_visual_checked_count": int(metrics.get("representative_visual_checked_count", 0)),
                    "representative_visual_sample_status": str(metrics.get("representative_visual_sample_status", "unknown")),
                    "pairwise_geometry_purity": metrics.get("pairwise_geometry_purity"),
                    "pairwise_geometry_fail_count": int(metrics.get("pairwise_geometry_fail_count", 0)),
                    "pairwise_geometry_sampled_pair_count": int(metrics.get("pairwise_geometry_sampled_pair_count", 0)),
                    "pairwise_geometry_sample_status": str(metrics.get("pairwise_geometry_sample_status", "unknown")),
                    "overmerge_score": float(metrics.get("overmerge_score", 0.0)),
                    "overmerge_reason": str(metrics.get("overmerge_reason", "ok")),
                },
            )
            for cluster_index, metrics in sorted(cluster_quality_by_index.items())
        ),
    }


def _safe_recall_merge_exact_review(cluster_units, quality_metrics, candidate_evidence_by_id, candidate_bitmap_locations, config):
    """对 safe recall pair 做保守合并，必须通过 candidate union coverage 与 strict verification。"""

    original_units = [(candidate, list(exact_clusters)) for candidate, exact_clusters in cluster_units]
    metrics = {
        "safe_recall_merge_enabled": True,
        "safe_recall_merge_candidate_pair_count": 0,
        "safe_recall_merge_attempted_pair_count": 0,
        "safe_recall_merge_merged_pair_count": 0,
        "safe_recall_merge_cluster_reduction": 0,
        "safe_recall_merge_checked_exact_count": 0,
        "safe_recall_merge_reject_reason_counts": {},
    }
    review_rows = [dict(row) for row in list(quality_metrics.get("review_merge_candidate_rows", []) or [])]
    pair_rows = [dict(row) for row in list(quality_metrics.get("review_merge_cluster_pair_rows", []) or [])]
    if not review_rows or not pair_rows:
        return original_units, metrics

    unit_by_cluster_id = dict(
        (int(cluster_index) + 1, (candidate, list(assigned_exact_clusters)))
        for cluster_index, (candidate, assigned_exact_clusters) in enumerate(original_units)
    )
    selected_candidate_ids = set(str(candidate.candidate_id) for candidate, _ in original_units)
    safe_pair_rows = {}
    skip_reasons = Counter()
    for row in pair_rows:
        pair_key = _cluster_pair_key(row)
        if str(row.get("pair_review_bucket", "")) != "safe_recall_candidate":
            continue
        safe_pair_rows[pair_key] = row
    metrics["safe_recall_merge_candidate_pair_count"] = int(len(safe_pair_rows))
    if not safe_pair_rows:
        return original_units, metrics

    high_rows_by_pair = defaultdict(list)
    for row in review_rows:
        if str(row.get("confidence_tier", "")) != "high":
            continue
        pair_key = _cluster_pair_key(row)
        if pair_key in safe_pair_rows:
            high_rows_by_pair[pair_key].append(row)

    exact_weight_by_id = {}
    for _, assigned_exact_clusters in original_units:
        for exact_cluster in assigned_exact_clusters:
            exact_weight_by_id[int(exact_cluster.exact_cluster_id)] = int(_exact_cluster_weight(exact_cluster))

    merge_candidates = []
    for pair_key, pair_row in safe_pair_rows.items():
        rows_for_pair = list(high_rows_by_pair.get(pair_key, []))
        if not rows_for_pair:
            skip_reasons["no_high_conf_evidence"] += 1
            continue
        unique_target_ids = set(int(row.get("target_exact_cluster_id", -1)) for row in rows_for_pair)
        unique_target_weight = int(sum(exact_weight_by_id.get(exact_id, 0) for exact_id in unique_target_ids))
        high_edge_weight = int(sum(max(int(row.get("edge_weight", 0) or 0), 0) for row in rows_for_pair))
        merge_candidates.append((-unique_target_weight, -high_edge_weight, pair_key, pair_row, rows_for_pair))
    merge_candidates.sort(key=lambda item: (item[0], item[1], item[2][0], item[2][1]))

    consumed_cluster_ids = set()
    used_merge_candidate_ids = set()
    merged_units_by_cluster_id = {}
    candidate_cache = {}
    descriptor_cache = {}

    def _record_skip(reason):
        skip_reasons[str(reason)] += 1

    def _load_candidate(candidate_id):
        if str(candidate_id) in candidate_cache:
            return candidate_cache[str(candidate_id)]
        evidence = candidate_evidence_by_id.get(str(candidate_id))
        if evidence is None:
            candidate_cache[str(candidate_id)] = None
            return None
        candidate = _candidate_clip_from_evidence(evidence, candidate_bitmap_locations)
        candidate_cache[str(candidate_id)] = candidate
        return candidate

    for _, _, pair_key, pair_row, rows_for_pair in merge_candidates[:SAFE_RECALL_MERGE_TOP_PAIR_N]:
        source_cluster_id, target_cluster_id = pair_key
        if source_cluster_id in consumed_cluster_ids or target_cluster_id in consumed_cluster_ids:
            _record_skip("cluster_already_merged")
            continue
        if (
            str(pair_row.get("source_overmerge_reason", "ok") or "ok") != "ok"
            or str(pair_row.get("target_overmerge_reason", "ok") or "ok") != "ok"
        ):
            _record_skip("endpoint_overmerge_reason")
            continue
        source_unit = unit_by_cluster_id.get(source_cluster_id)
        target_unit = unit_by_cluster_id.get(target_cluster_id)
        if source_unit is None or target_unit is None:
            _record_skip("missing_endpoint_cluster")
            continue
        union_exact_clusters = list(source_unit[1]) + list(target_unit[1])
        if len(union_exact_clusters) > int(SAFE_RECALL_MERGE_MAX_UNION_EXACT_COUNT):
            _record_skip("union_exact_count_limit")
            continue

        metrics["safe_recall_merge_attempted_pair_count"] = int(metrics["safe_recall_merge_attempted_pair_count"]) + 1
        metrics["safe_recall_merge_checked_exact_count"] = int(metrics["safe_recall_merge_checked_exact_count"]) + int(len(union_exact_clusters))
        union_exact_ids = set(int(exact_cluster.exact_cluster_id) for exact_cluster in union_exact_clusters)
        candidate_weight_by_id = Counter()
        for row in rows_for_pair:
            candidate_weight_by_id[str(row.get("candidate_id", ""))] += max(int(row.get("edge_weight", 0) or 0), 0)
        accepted_candidate = None
        candidate_failure_reason = "missing_candidate_evidence"
        for candidate_id, _ in sorted(candidate_weight_by_id.items(), key=lambda item: (-int(item[1]), str(item[0]))):
            if candidate_id in used_merge_candidate_ids:
                candidate_failure_reason = "candidate_already_used"
                continue
            candidate = _load_candidate(candidate_id)
            if candidate is None:
                candidate_failure_reason = "missing_candidate_evidence"
                continue
            if getattr(candidate, "clip_bitmap", None) is None:
                candidate_failure_reason = "missing_candidate_bitmap"
                continue
            coverage_ids = set(int(exact_id) for exact_id in getattr(candidate, "coverage", set()))
            if not union_exact_ids.issubset(coverage_ids):
                candidate_failure_reason = "candidate_missing_union_coverage"
                continue
            all_matched = True
            for exact_cluster in union_exact_clusters:
                matched, _, _ = _candidate_matches_exact(candidate, exact_cluster, config, descriptor_cache)
                if not matched:
                    all_matched = False
                    candidate_failure_reason = "strict_verification_reject"
                    break
            if all_matched:
                accepted_candidate = candidate
                break
        if accepted_candidate is None:
            _record_skip(candidate_failure_reason)
            continue
        merged_cluster_id = min(source_cluster_id, target_cluster_id)
        merged_exact_clusters = sorted(union_exact_clusters, key=lambda item: int(item.exact_cluster_id))
        merged_units_by_cluster_id[merged_cluster_id] = (accepted_candidate, merged_exact_clusters)
        consumed_cluster_ids.update([source_cluster_id, target_cluster_id])
        used_merge_candidate_ids.add(str(accepted_candidate.candidate_id))
        selected_candidate_ids.add(str(accepted_candidate.candidate_id))
        metrics["safe_recall_merge_merged_pair_count"] = int(metrics["safe_recall_merge_merged_pair_count"]) + 1

    if not merged_units_by_cluster_id:
        metrics["safe_recall_merge_reject_reason_counts"] = dict((str(key), int(value)) for key, value in sorted(skip_reasons.items()))
        return original_units, metrics

    merged_units = []
    for cluster_id, unit in unit_by_cluster_id.items():
        if cluster_id in merged_units_by_cluster_id:
            merged_units.append(merged_units_by_cluster_id[cluster_id])
        if cluster_id in consumed_cluster_ids:
            continue
        merged_units.append(unit)
    metrics["safe_recall_merge_cluster_reduction"] = int(len(original_units) - len(merged_units))
    metrics["safe_recall_merge_reject_reason_counts"] = dict((str(key), int(value)) for key, value in sorted(skip_reasons.items()))
    return merged_units, metrics


def _quality_metrics_summary(quality_metrics):
    """生成 result_summary 中使用的精简质量指标，避免输出内部 review 行。"""

    hidden = set(["cluster_quality_by_index", "review_merge_candidate_rows", "review_merge_cluster_pair_rows"])
    return dict((str(key), value) for key, value in dict(quality_metrics).items() if str(key) not in hidden)


def build_compact_result_exact_review(
    marker_records,
    exact_clusters,
    candidates,
    selected_candidates,
    coverage_stats,
    config,
    runtime_summary,
    coverage_offsets=None,
    coverage_values=None,
    candidate_bitmap_locations=None,
):
    """基于既有 compact result 补齐 exact-review 输出所需的质量指标与摘要字段。"""

    result = build_compact_result(marker_records, exact_clusters, candidates, selected_candidates, coverage_stats, config, runtime_summary)
    quality_enabled = bool(config.get("compute_quality_metrics", False))
    result["quality_metrics_enabled"] = bool(quality_enabled)
    result["result_summary"]["quality_metrics_enabled"] = bool(quality_enabled)
    if quality_enabled:
        candidate_evidence_rows = _candidate_evidence_rows(candidates, coverage_offsets, coverage_values)
        candidate_evidence_by_id = dict((str(row.get("candidate_id", "")), row) for row in candidate_evidence_rows)
        rejected_candidate_ids = config.get("_greedy_purity_gate_rejected_ids", set()) if isinstance(config, dict) else set()
        cluster_units = _cluster_units_from_result(result, exact_clusters, selected_candidates)
        quality_metrics = _build_quality_metrics_exact_review(
            cluster_units,
            exact_clusters,
            candidate_evidence_rows,
            [candidate for candidate, _ in cluster_units],
            config,
            dict(result.get("final_verification_stats", {}) or {}),
            dict(result.get("coverage_debug_stats", {}) or {}),
            rejected_candidate_ids,
        )
        cluster_units, safe_recall_merge_metrics = _safe_recall_merge_exact_review(
            cluster_units,
            quality_metrics,
            candidate_evidence_by_id,
            candidate_bitmap_locations or {},
            config,
        )
        if int(safe_recall_merge_metrics.get("safe_recall_merge_cluster_reduction", 0) or 0) > 0:
            _materialize_result_from_cluster_units(result, cluster_units, exact_clusters, config)
            quality_metrics = _build_quality_metrics_exact_review(
                cluster_units,
                exact_clusters,
                candidate_evidence_rows,
                [candidate for candidate, _ in cluster_units],
                config,
                dict(result.get("final_verification_stats", {}) or {}),
                dict(result.get("coverage_debug_stats", {}) or {}),
                rejected_candidate_ids,
            )
        quality_metrics.update(safe_recall_merge_metrics)
        cluster_quality_by_index = dict(quality_metrics.pop("cluster_quality_by_index", {}))
        result["quality_metrics"] = dict(quality_metrics)
        result["cluster_quality_by_index"] = dict(cluster_quality_by_index)
        result["result_summary"]["quality_metrics"] = _quality_metrics_summary(quality_metrics)
    return result


def write_result_csv(
    output_path,
    result,
    exact_clusters,
    selected_candidates,
    config,
    marker_metadata_by_exact_id=None,
    candidates=None,
    candidate_bundle_index=None,
):
    """保存主 Exact Cluster Review CSV，并自动派生 Cluster Representative CSV。"""

    del marker_metadata_by_exact_id, candidate_bundle_index
    output = Path(str(output_path))
    if output.suffix.lower() != ".csv":
        raise ValueError("optimized_v2_lsf 主结果只支持 .csv 文件，请将 --output 改为 .csv 后缀: %s" % str(output_path))
    if not output.parent.exists():
        output.parent.mkdir(parents=True)

    clip_size_um = float(config.get("clip_size_um", result.get("clip_size_um", 1.35)))
    exact_by_id = dict((int(cluster.exact_cluster_id), cluster) for cluster in exact_clusters)
    selected_by_id = dict((str(candidate.candidate_id), candidate) for candidate in selected_candidates)
    candidate_lookup = dict(selected_by_id)
    for item in candidates or []:
        candidate_id = str(_candidate_field(item, "candidate_id", ""))
        if candidate_id and candidate_id not in candidate_lookup:
            candidate_lookup[candidate_id] = item
    exact_to_final_cluster = _final_cluster_id_by_exact_id(result)
    quality_metrics_enabled = bool(result.get("quality_metrics_enabled", False))
    cluster_quality_by_index = dict(result.get("cluster_quality_by_index", {}) or {})
    cluster_csv_columns = _cluster_representative_csv_columns(quality_metrics_enabled)

    review_rows = []
    for exact_cluster in sorted(exact_clusters, key=lambda item: int(item.exact_cluster_id)):
        exact_id = int(exact_cluster.exact_cluster_id)
        final_cluster_id = exact_to_final_cluster.get(exact_id)
        if final_cluster_id is None:
            raise RuntimeError("exact cluster %d 缺少 final cluster 映射" % int(exact_id))
        review_rows.append(
            _csv_exact_cluster_review_row(
                int(exact_id) + 1,
                int(final_cluster_id),
                exact_cluster,
                float(clip_size_um),
                int(_exact_cluster_weight(exact_cluster)),
                float(_bitmap_risk_score(getattr(exact_cluster.representative, "clip_bitmap", None))),
                0,
            )
        )
    _assign_risk_ranks(review_rows, "groupID")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in sorted(review_rows, key=lambda item: int(item["groupID"])):
            writer.writerow(dict((column, _csv_scalar(row.get(column))) for column in CSV_OUTPUT_COLUMNS))
    row_count = int(len(review_rows))
    if int(row_count) != int(len(exact_clusters)):
        raise RuntimeError("CSV row count mismatch: wrote %d, expected %d exact clusters" % (int(row_count), int(len(exact_clusters))))

    cluster_output = output.with_name("%s_cluster_representatives.csv" % output.stem)
    total_exact_weight = max(int(sum(_exact_cluster_weight(cluster) for cluster in exact_clusters)), 1)
    cluster_rows = []
    for cluster in sorted(result.get("clusters", []), key=lambda item: int(item.get("cluster_id", 0))):
        internal_cluster_id = int(cluster.get("cluster_id", 0))
        final_cluster_id = int(internal_cluster_id) + 1
        exact_ids = [int(exact_id) for exact_id in cluster.get("exact_cluster_ids", [])]
        if not exact_ids:
            continue
        selected_candidate = candidate_lookup.get(str(cluster.get("selected_candidate_id", "")))
        if isinstance(selected_candidate, CandidateClip):
            center = list(getattr(selected_candidate, "center", ()) or [])
            cluster_weight = int(sum(_exact_cluster_weight(exact_by_id[exact_id]) for exact_id in exact_ids if exact_id in exact_by_id))
            representative_seed_type = _candidate_origin_seed_type(selected_candidate, exact_by_id)
            representative_score = float(_candidate_representative_score(selected_candidate, cluster_weight, total_exact_weight))
            opc_center_score = float(_candidate_opc_center_score(selected_candidate, cluster_weight, total_exact_weight))
            risk_score = float(_candidate_risk_score(selected_candidate))
        elif isinstance(selected_candidate, dict):
            center = list(selected_candidate.get("center", []) or [])
            cluster_weight = int(sum(_exact_cluster_weight(exact_by_id[exact_id]) for exact_id in exact_ids if exact_id in exact_by_id))
            representative_seed_type = str(selected_candidate.get("origin_seed_type", "unknown") or "unknown")
            representative_score = float(_metadata_representative_score(selected_candidate, cluster_weight, total_exact_weight))
            opc_center_score = float(
                _clamp01(
                    float(selected_candidate.get("opc_center_base_score", 0.0))
                    + 0.20 * _safe_ratio(cluster_weight, total_exact_weight)
                )
            )
            risk_score = float(selected_candidate.get("risk_score", 0.0))
        elif cluster.get("selected_center") is not None:
            center = list(cluster.get("selected_center", []) or [])
            cluster_weight = int(sum(_exact_cluster_weight(exact_by_id[exact_id]) for exact_id in exact_ids if exact_id in exact_by_id))
            representative_seed_type = str(cluster.get("representative_seed_type", "unknown") or "unknown")
            representative_score = float(cluster.get("representative_score", 0.0))
            opc_center_score = float(cluster.get("opc_center_score", 0.0))
            risk_score = float(cluster.get("risk_score", 0.0))
        else:
            exact_cluster = exact_by_id.get(int(exact_ids[0]))
            if exact_cluster is None:
                raise RuntimeError("final cluster %d 缺少 representative exact cluster" % int(final_cluster_id))
            center = list(getattr(exact_cluster.representative, "marker_center", ()) or [])
            cluster_weight = int(sum(_exact_cluster_weight(exact_by_id[exact_id]) for exact_id in exact_ids if exact_id in exact_by_id))
            representative_seed_type = _exact_cluster_seed_type(exact_cluster)
            risk_score = float(_bitmap_risk_score(getattr(exact_cluster.representative, "clip_bitmap", None)))
            opc_center_score = float(_clamp01(0.65 * (1.0 - min(risk_score, 1.0)) + 0.35 * _safe_ratio(cluster_weight, total_exact_weight)))
            representative_score = float(_clamp01(0.45 * opc_center_score + 0.35 * _safe_ratio(cluster_weight, total_exact_weight) + 0.20 * risk_score))
        row = {
            "cluster_id": int(final_cluster_id),
            "center_x_um": _csv_scalar(center[0] if len(center) > 0 else ""),
            "center_y_um": _csv_scalar(center[1] if len(center) > 1 else ""),
            "clip_size": float(clip_size_um),
            "cluster_size": int(cluster.get("size", 0)),
            "cluster_weight": int(cluster_weight),
            "exact_cluster_count": int(len(exact_ids)),
            "representative_seed_type": str(representative_seed_type),
            "shift_direction": str(cluster.get("selected_shift_direction", "base")),
            "shift_distance_um": float(cluster.get("selected_shift_distance_um", 0.0)),
            "representative_score": float(representative_score),
            "opc_center_score": float(opc_center_score),
            "risk_score": float(risk_score),
            "risk_rank": 0,
        }
        quality_payload = dict(
            cluster_quality_by_index.get(str(internal_cluster_id), {})
            or cluster_quality_by_index.get(int(internal_cluster_id), {})
            or {}
        )
        if quality_metrics_enabled:
            row["representative_visual_pass_ratio"] = float(quality_payload.get("representative_visual_pass_ratio", 0.0))
            row["representative_visual_fail_count"] = int(quality_payload.get("representative_visual_fail_count", 0))
            row["representative_visual_checked_count"] = int(quality_payload.get("representative_visual_checked_count", 0))
            row["representative_visual_sample_status"] = str(
                quality_payload.get("representative_visual_sample_status", "unknown")
            )
            pairwise_geometry_purity = quality_payload.get("pairwise_geometry_purity")
            row["pairwise_geometry_purity"] = (
                None if pairwise_geometry_purity is None else float(pairwise_geometry_purity)
            )
            row["pairwise_geometry_fail_count"] = int(quality_payload.get("pairwise_geometry_fail_count", 0))
            row["pairwise_geometry_sampled_pair_count"] = int(
                quality_payload.get("pairwise_geometry_sampled_pair_count", 0)
            )
            row["pairwise_geometry_sample_status"] = str(quality_payload.get("pairwise_geometry_sample_status", "unknown"))
            row["overmerge_score"] = float(quality_payload.get("overmerge_score", 0.0))
            row["overmerge_reason"] = str(quality_payload.get("overmerge_reason", "ok"))
        cluster_rows.append(row)
    _assign_risk_ranks(cluster_rows, "cluster_id")
    with cluster_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cluster_csv_columns, lineterminator="\n")
        writer.writeheader()
        for row in sorted(cluster_rows, key=lambda item: int(item["cluster_id"])):
            writer.writerow(_csv_cluster_representative_row(row, cluster_csv_columns))

    result["result_csv_path"] = str(output)
    result["result_csv_row_count"] = int(row_count)
    result["result_csv_columns"] = list(CSV_OUTPUT_COLUMNS)
    result["cluster_representative_csv_path"] = str(cluster_output)
    result["cluster_representative_csv_row_count"] = int(len(cluster_rows))
    result["cluster_representative_csv_columns"] = list(cluster_csv_columns)
    summary = result.setdefault("result_summary", {})
    summary["result_csv_path"] = str(output)
    summary["result_csv_row_count"] = int(row_count)
    summary["result_csv_columns"] = list(CSV_OUTPUT_COLUMNS)
    summary["cluster_representative_csv_path"] = str(cluster_output)
    summary["cluster_representative_csv_row_count"] = int(len(cluster_rows))
    summary["cluster_representative_csv_columns"] = list(cluster_csv_columns)
    return {
        "path": str(output),
        "row_count": int(row_count),
        "columns": list(CSV_OUTPUT_COLUMNS),
        "cluster_representative_path": str(cluster_output),
        "cluster_representative_row_count": int(len(cluster_rows)),
        "cluster_representative_columns": list(cluster_csv_columns),
    }


def record_metadata(record):
    """把 MarkerRecord 的非 bitmap 字段转成 JSON metadata。"""

    return {
        "marker_id": str(record.marker_id),
        "source_path": str(record.source_path),
        "source_name": str(record.source_name),
        "marker_bbox": list(record.marker_bbox),
        "marker_center": list(record.marker_center),
        "clip_bbox": list(record.clip_bbox),
        "expanded_bbox": list(record.expanded_bbox),
        "clip_bbox_q": list(record.clip_bbox_q),
        "expanded_bbox_q": list(record.expanded_bbox_q),
        "marker_bbox_q": list(record.marker_bbox_q),
        "shift_limits_px": {
            "x": list(record.shift_limits_px["x"]),
            "y": list(record.shift_limits_px["y"]),
        },
        "clip_hash": str(record.clip_hash),
        "expanded_hash": str(record.expanded_hash),
        "clip_area": float(record.clip_area),
        "seed_weight": int(record.seed_weight),
        "exact_cluster_id": int(record.exact_cluster_id),
        "metadata": dict(record.metadata),
    }


def record_from_metadata(metadata, clip_bitmap, expanded_bitmap):
    """从 metadata 和 bitmap 还原 MarkerRecord。"""

    shift_limits = metadata["shift_limits_px"]
    return MarkerRecord(
        marker_id=metadata["marker_id"],
        source_path=metadata["source_path"],
        source_name=metadata["source_name"],
        marker_bbox=tuple(metadata["marker_bbox"]),
        marker_center=tuple(metadata["marker_center"]),
        clip_bbox=tuple(metadata["clip_bbox"]),
        expanded_bbox=tuple(metadata["expanded_bbox"]),
        clip_bbox_q=tuple(int(v) for v in metadata["clip_bbox_q"]),
        expanded_bbox_q=tuple(int(v) for v in metadata["expanded_bbox_q"]),
        marker_bbox_q=tuple(int(v) for v in metadata["marker_bbox_q"]),
        shift_limits_px={
            "x": tuple(int(v) for v in shift_limits["x"]),
            "y": tuple(int(v) for v in shift_limits["y"]),
        },
        clip_bitmap=np.ascontiguousarray(clip_bitmap, dtype=bool),
        expanded_bitmap=np.ascontiguousarray(expanded_bitmap, dtype=bool),
        clip_hash=metadata["clip_hash"],
        expanded_hash=metadata["expanded_hash"],
        clip_area=float(metadata["clip_area"]),
        seed_weight=int(metadata.get("seed_weight", 1)),
        exact_cluster_id=int(metadata.get("exact_cluster_id", -1)),
        metadata=dict(metadata.get("metadata", {})),
    )


def save_shard_records(records, json_path, npz_path, extra_payload):
    """把 shard marker records 保存为 JSON metadata + npz bitmap。"""

    clip_bitmaps = np.asarray([record.clip_bitmap for record in records], dtype=bool)
    expanded_bitmaps = np.asarray([record.expanded_bitmap for record in records], dtype=bool)
    np.savez_compressed(str(npz_path), clip_bitmaps=clip_bitmaps, expanded_bitmaps=expanded_bitmaps)
    payload = dict(extra_payload)
    payload["records"] = [record_metadata(record) for record in records]
    with Path(str(json_path)).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_shard_records(json_path, npz_path):
    """从 shard JSON + npz 还原 MarkerRecord 列表。"""

    with Path(str(json_path)).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    arrays = np.load(str(npz_path), allow_pickle=False)
    try:
        clip_bitmaps = arrays["clip_bitmaps"]
        expanded_bitmaps = arrays["expanded_bitmaps"]
        records = []
        for idx, metadata in enumerate(payload.get("records", [])):
            records.append(record_from_metadata(metadata, clip_bitmaps[int(idx)], expanded_bitmaps[int(idx)]))
    finally:
        arrays.close()
    return records, payload


def save_exact_index(exact_clusters, json_path, npz_path, extra_payload):
    """保存全局 exact index，只保留 representative bitmap 和簇规模信息。"""

    representatives = [cluster.representative for cluster in exact_clusters]
    same_clip_shape = len(set(tuple(record.clip_bitmap.shape) for record in representatives)) <= 1
    same_expanded_shape = len(set(tuple(record.expanded_bitmap.shape) for record in representatives)) <= 1
    if representatives and same_clip_shape and same_expanded_shape:
        clip_bitmaps = np.asarray([record.clip_bitmap for record in representatives], dtype=bool)
        expanded_bitmaps = np.asarray([record.expanded_bitmap for record in representatives], dtype=bool)
        np.savez_compressed(str(npz_path), clip_bitmaps=clip_bitmaps, expanded_bitmaps=expanded_bitmaps)
        bitmap_storage = "stacked"
    elif representatives:
        arrays = {}
        for idx, record in enumerate(representatives):
            arrays["clip_%06d" % int(idx)] = np.ascontiguousarray(record.clip_bitmap, dtype=bool)
            arrays["expanded_%06d" % int(idx)] = np.ascontiguousarray(record.expanded_bitmap, dtype=bool)
        np.savez_compressed(str(npz_path), **arrays)
        bitmap_storage = "per_record"
    else:
        clip_bitmaps = np.zeros((0, 0, 0), dtype=bool)
        expanded_bitmaps = np.zeros((0, 0, 0), dtype=bool)
        np.savez_compressed(str(npz_path), clip_bitmaps=clip_bitmaps, expanded_bitmaps=expanded_bitmaps)
        bitmap_storage = "stacked"
    payload = dict(extra_payload)
    payload["bitmap_storage"] = bitmap_storage
    clusters = []
    for cluster in exact_clusters:
        clusters.append(
            {
                "exact_cluster_id": int(cluster.exact_cluster_id),
                "exact_key": str(cluster.exact_key),
                "member_count": int(cluster.member_count),
                "weight_sum": int(cluster.weight),
                "representative": record_metadata(cluster.representative),
            }
        )
    payload["clusters"] = clusters
    with Path(str(json_path)).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=json_default)


def load_exact_index(json_path, npz_path):
    """读取全局 exact index，并还原可用于 coverage 的 ExactCluster 列表。"""

    with Path(str(json_path)).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    arrays = np.load(str(npz_path), allow_pickle=False)
    try:
        has_stacked = "clip_bitmaps" in arrays.files
        if has_stacked:
            clip_bitmaps = arrays["clip_bitmaps"]
            expanded_bitmaps = arrays["expanded_bitmaps"]
        exact_clusters = []
        for idx, cluster_payload in enumerate(payload.get("clusters", [])):
            if has_stacked:
                clip_bitmap = clip_bitmaps[int(idx)]
                expanded_bitmap = expanded_bitmaps[int(idx)]
            else:
                clip_bitmap = arrays["clip_%06d" % int(idx)]
                expanded_bitmap = arrays["expanded_%06d" % int(idx)]
            representative = record_from_metadata(
                cluster_payload["representative"],
                clip_bitmap,
                expanded_bitmap,
            )
            representative.exact_cluster_id = int(cluster_payload["exact_cluster_id"])
            exact_clusters.append(
                ExactCluster(
                    int(cluster_payload["exact_cluster_id"]),
                    str(cluster_payload["exact_key"]),
                    representative,
                    [],
                    int(cluster_payload.get("member_count", 1)),
                    int(cluster_payload.get("weight_sum", 1)),
                )
            )
    finally:
        arrays.close()
    return exact_clusters, payload


def json_default(value):
    """把 numpy / Path 等对象转成 JSON 可序列化值。"""

    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Object of type %s is not JSON serializable" % type(value).__name__)
