#!/usr/bin/env python3
"""
Optimized geometry-driven layout clustering v1.

中文整体说明：
1. 这个版本在前半段改成 geometry-driven seed 主线，不再使用 uniform grid 全域扫窗，
   而是按阵列代表点、阵列 spacing、长条路径和 residual 局部 grid 来生成 seed。
   这样做的目的，是在保留 Recall 的同时，让采样更贴近规则阵列、长线和间距弱点的真实结构。
2. grid 步长仍固定为 clip size 的 50%，但这里只作为局部 marker bbox 尺寸和 anchor 量化基准，
   不再表示“全版图每个 grid cell 都要生成一个 seed”。
3. geometry-driven seed 会先做 anchor 级别去重；其中 spacing seed 拥有独立槽位，允许和普通 seed
   在同一 anchor 下各保留一个代表，同时累计 bucket weight 供后续 exact hash / set cover 使用。
4. candidate 生成仍以 base + 轴向 systematic shift 为主，但会补少量 diagonal shift，
   修补需要同时 x/y 对齐的覆盖缺口。
5. 后半段流程尽量复用已经验证过的稳定主线：exact hash 合并完全重复窗口，
   graph descriptor prefilter 剪掉明显不可能匹配的 candidate 对，再用 ACC / ECC
   做最终几何 gate；coverage 边构建、lazy-heap greedy set cover 和 final verification
   都沿用当前优化版的实现。
6. final verification 仍然坚持“失败就拆回 singleton”的解释性策略，不做跨 cluster
   自动修补。换句话说，v1 的思路是：前半段用更结构化的 seed 尽量抓住真实弱点，
   后半段再严格验证，保证最终 cluster 的 representative-member 关系仍然容易人工理解和 review。

设计原则：
- 主线唯一：layer-op 过滤 -> geometry-driven seed -> shift-cover -> verified clustering。
- v1 不再保留 pair/drc seed strategy 及其 CLI、日志、结果字段，避免旧路线残留干扰。
- 后半段尽量不动，只升级 seed/candidate 前端，便于隔离变量、观察 geometry-driven 主线的真实效果。
"""

from __future__ import annotations

import argparse
from collections import Counter
import ctypes
import gc
import hashlib
import heapq
import json
import math
import shutil
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

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
    _bitcount_sum_rows,
    _chunk_indices_by_row_width,
    _collect_boundary_positions,
    _collect_shift_values_px,
    _make_centered_bbox,
    _make_sample_filename,
    _materialize_clip_bitmap,
    _raster_window_spec,
    _slice_bitmap,
)


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
EXPORT_DISTANCE_SCORE_TOPK = 8
DIAGONAL_SHIFT_AXIS_MAX_COUNT = 3
DIAGONAL_SHIFT_MAX_COUNT = 2

DUMMY_MARKER_LAYER = "65535/65535"
PIPELINE_MODE = "optimized_v1"
SEED_MODE = "geometry_driven_shift"
GRID_STEP_RATIO = 0.5
GRID_BUCKET_QUANT_UM = 0.08
GRID_MAX_DESCRIPTOR_NEIGHBORS = 256
COVERAGE_SHORTLIST_MAX_TARGETS = 64
COVERAGE_EXACT_SHORTLIST_MAX_GROUPS = 512
MEGA_BUNDLE_PAIR_TRACKER_DISABLE_THRESHOLD = 200_000
COVERAGE_BUCKETED_GROUP_THRESHOLD = 200_000
COVERAGE_FILL_BIN_WIDTH = 0.04
STRICT_BITMAP_DIGEST_SIZE = 16
PRE_RASTER_FINGERPRINT_QUANT_PX = 2
_POOL_EDGE_CACHE: Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]] = {}
_COVERAGE_STRUCTURE_CACHE: Dict[int, np.ndarray] = {}
PACKED_EXPANDED_BITMAP_KEY = "optimized_expanded_bitmap_packed"
PACKED_EXPANDED_SHAPE_KEY = "optimized_expanded_bitmap_shape"
PACKED_CLIP_BITMAP_KEY = "optimized_clip_bitmap_packed"
PACKED_CLIP_SHAPE_KEY = "optimized_clip_bitmap_shape"
PACKED_CANDIDATE_CLIP_BITMAP_KEY = "optimized_candidate_clip_bitmap_packed"
PACKED_CANDIDATE_CLIP_SHAPE_KEY = "optimized_candidate_clip_bitmap_shape"
EXPORT_CHEAP_FEATURE_KEY = "optimized_export_cheap_feature"
EXPORT_WORST_SCORE_KEY = "optimized_export_worst_score"
EXPORT_DISTANCE_SCORE_KEY = "optimized_export_distance_score"
SEED_TYPE_ARRAY = "array_representative"
SEED_TYPE_ARRAY_SPACE = "array_spacing"
SEED_TYPE_LONG = "long_shape_path"
SEED_TYPE_RESIDUAL = "residual_local_grid"


@dataclass(frozen=True, slots=True)
class GridSeedCandidate:
    """geometry-driven 阶段的 seed 记录。"""

    center: Tuple[float, float]
    seed_bbox: Tuple[float, float, float, float]
    grid_ix: int
    grid_iy: int
    bucket_weight: int = 1
    seed_type: str = SEED_TYPE_RESIDUAL


@dataclass(frozen=True, slots=True)
class GraphDescriptor:
    """clip bitmap 提取出的轻量图形描述符，用于 prefilter。"""

    invariants: np.ndarray
    topology: np.ndarray
    signature_grid: np.ndarray
    signature_proj_x: np.ndarray
    signature_proj_y: np.ndarray


@dataclass(frozen=True, slots=True)
class CheapDescriptor:
    """coverage shortlist 使用的低成本 bitmap 描述符。"""

    invariants: np.ndarray
    signature_grid: np.ndarray
    signature_proj_x: np.ndarray
    signature_proj_y: np.ndarray
    area_px: int


@dataclass(frozen=True, slots=True)
class RasterPayload:
    """采集阶段复用栅格结果的轻量载荷，避免缓存完整 MarkerRecord。"""

    clip_bitmap: np.ndarray
    expanded_bitmap: np.ndarray
    clip_hash: str
    expanded_hash: str
    clip_area: float
    expanded_bitmap_packed: np.ndarray | None = None
    expanded_bitmap_shape: Tuple[int, int] | None = None
    graph_descriptor: GraphDescriptor | None = None
    cheap_descriptor: CheapDescriptor | None = None


@dataclass(slots=True)
class CoverageCandidateGroup:
    """coverage 阶段的紧凑候选组，只保留一个长期常驻 candidate。"""

    best_candidate: CandidateClip
    packed_clip_bitmap: np.ndarray
    clip_bitmap_shape: Tuple[int, int]
    area_px: int
    clip_hash: str
    origin_ids: np.ndarray
    logical_candidate_count: int
    direction_counts: Dict[str, int]
    coverage: Sequence[int]
    materialized_candidates: Tuple[CandidateClip, ...] = ()


def _empty_prefilter_stats() -> Dict[str, int]:
    """返回 prefilter / geometry gate 阶段的统计计数器。"""

    return {
        "exact_hash_pass": 0,
        "cheap_reject": 0,
        "cheap_fill_reject": 0,
        "cheap_area_density_reject": 0,
        "full_prefilter_reject": 0,
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
        "shortlist_payload_build": 0.0,
        "shortlist_payload_release": 0.0,
        "prefilter": 0.0,
        "full_descriptor_cache": 0.0,
        "full_prefilter": 0.0,
        "geometry_cache": 0.0,
        "geometry_cache_release": 0.0,
        "geometry_match": 0.0,
        "bucket_index_build": 0.0,
        "bucket_window_index": 0.0,
        "bucket_window_release": 0.0,
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
        "geometry_cache_live_peak_count": 0,
        "geometry_cache_release_count": 0,
        "geometry_cache_live_after_bundle_count": 0,
        "full_descriptor_cache_group_count": 0,
        "full_prefilter_probe_pair_count": 0,
        "full_prefilter_probe_reject_count": 0,
        "full_prefilter_disabled_bundle_count": 0,
        "shortlist_subgroup_count": 0,
        "shortlist_exact_subgroup_count": 0,
        "shortlist_hnsw_subgroup_count": 0,
        "shortlist_max_subgroup_size": 0,
        "shortlist_payload_peak_count": 0,
        "shortlist_payload_release_count": 0,
        "lazy_signature_embedding_group_count": 0,
        "signature_embedding_live_peak_count": 0,
        "pair_tracker_mode": "unset",
        "pair_tracker_disabled_bundle_count": 0,
        "pair_tracker_row_count": 0,
        "bucketed_coverage_bundle_count": 0,
        "coverage_fill_bin_count": 0,
        "max_fill_bin_group_count": 0,
        "max_bucket_window_group_count": 0,
        "bucketed_source_group_count": 0,
        "bucketed_target_group_count": 0,
        "window_bitmap_live_peak_count": 0,
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


def _empty_memory_debug() -> Dict[str, Any]:
    """返回运行期内存诊断字段。"""

    return {
        "rss_collect_markers_mb": None,
        "rss_exact_cluster_mb": None,
        "rss_candidate_generation_mb": None,
        "rss_coverage_eval_mb": None,
        "rss_set_cover_mb": None,
        "rss_result_build_mb": None,
        "rss_peak_estimate_mb": None,
        "released_marker_expanded_count": 0,
        "released_marker_clip_count": 0,
        "released_candidate_clip_count": 0,
        "released_cache_owner_count": 0,
        "pre_raster_payload_cache_count": 0,
        "exact_bitmap_payload_cache_count": 0,
        "packed_marker_expanded_count": 0,
        "unpacked_marker_expanded_count": 0,
        "packed_marker_clip_count": 0,
        "candidate_bitmap_pool_unique_count": 0,
        "candidate_bitmap_pool_hit_count": 0,
        "released_candidate_list_ref_count": 0,
        "strict_digest_key_count": 0,
        "strict_digest_collision_count": 0,
        "strict_key_bytes_avoided_estimate_mb": 0.0,
        "early_duplicate_shift_candidate_count": 0,
        "candidate_object_avoided_count": 0,
        "signature_embedding_bytes_avoided_estimate_mb": 0.0,
        "online_exact_group_count": 0,
        "light_member_record_count": 0,
        "released_marker_clip_early_count": 0,
        "released_marker_expanded_early_count": 0,
        "packed_candidate_group_bitmap_count": 0,
        "unpacked_candidate_group_bitmap_count": 0,
        "candidate_group_bitmap_bytes_avoided_estimate_mb": 0.0,
    }


def _process_rss_mb() -> float | None:
    """读取当前进程 RSS，优先走 Windows API。"""

    try:
        class _ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        process = kernel32.GetCurrentProcess()
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        ok = psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
        if ok:
            return round(float(counters.WorkingSetSize) / (1024.0 * 1024.0), 3)
    except Exception:
        pass
    try:
        import resource  # type: ignore

        return round(float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0, 3)
    except Exception:
        return None


def _coverage_to_array(values: Sequence[int] | Iterable[int]) -> np.ndarray:
    """把 coverage id 集合规范化成有序去重的 int32 数组。"""

    array = np.asarray(list(int(value) for value in values), dtype=np.int32)
    if array.size == 0:
        return np.asarray([], dtype=np.int32)
    return np.unique(array)


def _coverage_union(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """合并两个有序 coverage 数组。"""

    if left.size == 0:
        return np.asarray(right, dtype=np.int32)
    if right.size == 0:
        return np.asarray(left, dtype=np.int32)
    return np.union1d(left, right).astype(np.int32, copy=False)


def _coverage_overlap(values: Sequence[int], uncovered: set[int]) -> Tuple[int, ...]:
    """计算某个 candidate 当前仍能覆盖的 exact cluster。"""

    if not values or not uncovered:
        return ()
    return tuple(int(value) for value in values if int(value) in uncovered)


def _clear_match_cache_keys(owner: Any, keys: Sequence[str]) -> bool:
    """清理对象 match_cache 中指定键，返回是否真的删掉了内容。"""

    match_cache = getattr(owner, "match_cache", None)
    if not isinstance(match_cache, dict):
        return False
    removed = False
    for key in keys:
        if key in match_cache:
            del match_cache[key]
            removed = True
    return removed


def _pack_bitmap_payload(bitmap: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """把二维 bool bitmap 压缩为 packed bitset 与 shape。"""

    mask = np.ascontiguousarray(np.asarray(bitmap, dtype=bool))
    shape = (int(mask.shape[0]), int(mask.shape[1]))
    packed = np.packbits(mask.reshape(-1))
    return packed, shape


def _unpack_bitmap_payload(packed: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """把 packed bitset 还原为二维 bool bitmap。"""

    height, width = int(shape[0]), int(shape[1])
    bits = np.unpackbits(np.asarray(packed, dtype=np.uint8), count=max(height * width, 0))
    return np.ascontiguousarray(bits.reshape((height, width)).astype(bool, copy=False))


def _strict_bitmap_digest(packed: np.ndarray, shape: Tuple[int, int]) -> bytes:
    """为 strict bitmap key 生成短 digest，shape 仍保留在外层 key。"""

    height, width = int(shape[0]), int(shape[1])
    digest = hashlib.blake2b(digest_size=int(STRICT_BITMAP_DIGEST_SIZE))
    digest.update(height.to_bytes(4, "little", signed=False))
    digest.update(width.to_bytes(4, "little", signed=False))
    digest.update(np.asarray(packed, dtype=np.uint8).tobytes())
    return digest.digest()


def _strict_bitmap_digest_key(bitmap: np.ndarray) -> Tuple[Tuple[int, int, bytes], np.ndarray, np.ndarray, int]:
    """返回 optimized v1 的 strict digest key、规范 bitmap、packed payload 和 packed 字节长度。"""

    mask = np.ascontiguousarray(np.asarray(bitmap, dtype=bool))
    shape = (int(mask.shape[0]), int(mask.shape[1]))
    packed = np.packbits(mask.reshape(-1))
    key = (shape[0], shape[1], _strict_bitmap_digest(packed, shape))
    return key, mask, np.asarray(packed, dtype=np.uint8), int(packed.size)


def _same_bitmap(left: np.ndarray, right: np.ndarray) -> bool:
    """比较两个 bitmap 是否逐像素完全一致。"""

    left_mask = np.asarray(left, dtype=bool)
    right_mask = np.asarray(right, dtype=bool)
    return bool(left_mask.shape == right_mask.shape and np.array_equal(left_mask, right_mask))


def _attach_packed_expanded_bitmap(record: MarkerRecord, packed: np.ndarray, shape: Tuple[int, int]) -> None:
    """把压缩后的 expanded bitmap 挂到 marker cache。"""

    record.match_cache[PACKED_EXPANDED_BITMAP_KEY] = np.asarray(packed, dtype=np.uint8)
    record.match_cache[PACKED_EXPANDED_SHAPE_KEY] = (int(shape[0]), int(shape[1]))


def _pack_marker_expanded_bitmap(record: MarkerRecord) -> bool:
    """把 marker 的 expanded bitmap 压缩进 match_cache，并释放二维 bool 副本。"""

    bitmap = getattr(record, "expanded_bitmap", None)
    if bitmap is None:
        return False
    packed, shape = _pack_bitmap_payload(bitmap)
    _attach_packed_expanded_bitmap(record, packed, shape)
    record.expanded_bitmap = None
    return True


def _expanded_bitmap_for_marker(record: MarkerRecord) -> np.ndarray:
    """按需取回 marker 的 expanded bitmap，优先使用压缩缓存还原。"""

    bitmap = getattr(record, "expanded_bitmap", None)
    if bitmap is not None:
        return np.ascontiguousarray(np.asarray(bitmap, dtype=bool))
    packed = record.match_cache.get(PACKED_EXPANDED_BITMAP_KEY)
    shape = record.match_cache.get(PACKED_EXPANDED_SHAPE_KEY)
    assert packed is not None and shape is not None, "生成 candidate 时 expanded bitmap packed cache 不应为空"
    return _unpack_bitmap_payload(np.asarray(packed, dtype=np.uint8), tuple(int(value) for value in shape))


def _pack_marker_clip_bitmap(record: MarkerRecord) -> bool:
    """把 marker clip bitmap 压缩到 cache，供后续代表样本评分按需还原。"""

    bitmap = getattr(record, "clip_bitmap", None)
    if bitmap is None or PACKED_CLIP_BITMAP_KEY in record.match_cache:
        return False
    packed, shape = _pack_bitmap_payload(bitmap)
    record.match_cache[PACKED_CLIP_BITMAP_KEY] = packed
    record.match_cache[PACKED_CLIP_SHAPE_KEY] = shape
    return True


def _attach_packed_candidate_clip_bitmap(candidate: CandidateClip, packed: np.ndarray, shape: Tuple[int, int]) -> None:
    """把候选 clip bitmap 的 packed 形态挂到 candidate cache。"""

    candidate.match_cache[PACKED_CANDIDATE_CLIP_BITMAP_KEY] = np.asarray(packed, dtype=np.uint8)
    candidate.match_cache[PACKED_CANDIDATE_CLIP_SHAPE_KEY] = (int(shape[0]), int(shape[1]))


def _candidate_clip_bitmap(candidate: CandidateClip) -> np.ndarray:
    """按需取回 candidate 的 clip bitmap，优先使用 at-rest packed payload。"""

    bitmap = getattr(candidate, "clip_bitmap", None)
    if bitmap is not None:
        return np.ascontiguousarray(np.asarray(bitmap, dtype=bool))
    packed = candidate.match_cache.get(PACKED_CANDIDATE_CLIP_BITMAP_KEY)
    shape = candidate.match_cache.get(PACKED_CANDIDATE_CLIP_SHAPE_KEY)
    assert packed is not None and shape is not None, "candidate bitmap 恢复时 packed payload 不应为空"
    return _unpack_bitmap_payload(np.asarray(packed, dtype=np.uint8), tuple(int(value) for value in shape))


def _clip_bitmap_for_export(record: MarkerRecord) -> np.ndarray:
    """为导出代表评分按需获取 clip bitmap。"""

    bitmap = getattr(record, "clip_bitmap", None)
    if bitmap is not None:
        return np.ascontiguousarray(np.asarray(bitmap, dtype=bool))
    packed = record.match_cache.get(PACKED_CLIP_BITMAP_KEY)
    shape = record.match_cache.get(PACKED_CLIP_SHAPE_KEY)
    assert packed is not None and shape is not None, "导出代表评分时 clip bitmap packed cache 不应为空"
    return _unpack_bitmap_payload(np.asarray(packed, dtype=np.uint8), tuple(int(value) for value in shape))


def _release_marker_expanded_bitmap(record: MarkerRecord) -> bool:
    """释放 marker 的 expanded bitmap。"""

    released = False
    if getattr(record, "expanded_bitmap", None) is not None:
        record.expanded_bitmap = None
        released = True
    for key in (PACKED_EXPANDED_BITMAP_KEY, PACKED_EXPANDED_SHAPE_KEY):
        if key in record.match_cache:
            del record.match_cache[key]
            released = True
    return released


def _release_marker_clip_payload(record: MarkerRecord, keep_clip_bitmap: bool) -> bool:
    """释放 marker 上结果阶段不再需要的大对象。"""

    released = _release_marker_expanded_bitmap(record)
    if not keep_clip_bitmap and getattr(record, "clip_bitmap", None) is not None:
        record.clip_bitmap = None
        released = True
    released = _clear_match_cache_keys(
        record,
        (
            "optimized_graph_descriptor",
            "optimized_cheap_descriptor",
        ),
    ) or released
    return released


def _release_candidate_geometry_payload(candidate: CandidateClip, keep_clip_bitmap: bool) -> bool:
    """释放 candidate 上 coverage / verification 阶段的几何缓存。"""

    released = _clear_match_cache_keys(
        candidate,
        tuple(
            [str(key) for key in list(candidate.match_cache.keys()) if str(key).startswith("optimized_ecc_tol_")]
            + [
                "optimized_graph_descriptor",
                "optimized_cheap_descriptor",
            ]
        ),
    )
    if not keep_clip_bitmap and getattr(candidate, "clip_bitmap", None) is not None:
        candidate.clip_bitmap = None
        released = True
    if not keep_clip_bitmap:
        released = _clear_match_cache_keys(
            candidate,
            (
                PACKED_CANDIDATE_CLIP_BITMAP_KEY,
                PACKED_CANDIDATE_CLIP_SHAPE_KEY,
            ),
        ) or released
    return released


def _raster_payload_from_record(record: MarkerRecord) -> RasterPayload:
    """从 marker record 提取可复用的最小栅格载荷。"""

    packed_expanded = record.match_cache.get(PACKED_EXPANDED_BITMAP_KEY)
    packed_shape = record.match_cache.get(PACKED_EXPANDED_SHAPE_KEY)
    return RasterPayload(
        clip_bitmap=record.clip_bitmap,
        expanded_bitmap=record.expanded_bitmap,
        clip_hash=str(record.clip_hash),
        expanded_hash=str(record.expanded_hash),
        clip_area=float(record.clip_area),
        expanded_bitmap_packed=np.asarray(packed_expanded, dtype=np.uint8) if packed_expanded is not None else None,
        expanded_bitmap_shape=tuple(int(value) for value in packed_shape) if packed_shape is not None else None,
        graph_descriptor=record.match_cache.get("optimized_graph_descriptor"),
        cheap_descriptor=record.match_cache.get("optimized_cheap_descriptor"),
    )


def _should_stream_result(sample_count: int, cluster_count: int, materialize_outputs: bool) -> bool:
    """判断当前结果是否应走流式 JSON 写出路径。"""

    if materialize_outputs:
        return False
    return int(sample_count) >= 1000 or int(cluster_count) >= 1000


def _pair_tracker(group_count: int, *, force_source_unique: bool = False) -> Dict[str, Any]:
    """创建 bundle 内部 pair 去重跟踪器。"""

    group_count = int(group_count)
    if force_source_unique or group_count > int(MEGA_BUNDLE_PAIR_TRACKER_DISABLE_THRESHOLD):
        return {
            "mode": "source_unique",
            "group_count": group_count,
            "source_idx": None,
            "seen_targets": set(),
        }
    if group_count <= 8192:
        pair_count = (group_count * max(group_count - 1, 0)) // 2
        return {
            "mode": "bitset",
            "group_count": group_count,
            "bits": np.zeros((pair_count + 7) // 8, dtype=np.uint8),
        }
    return {
        "mode": "rows",
        "group_count": group_count,
        "rows": {},
    }


def _pair_tracker_test_and_set(tracker: Dict[str, Any], source_idx: int, target_idx: int) -> bool:
    """测试并设置某个无向 pair 是否已经比较过。"""

    left = min(int(source_idx), int(target_idx))
    right = max(int(source_idx), int(target_idx))
    if left == right:
        return True
    mode = str(tracker["mode"])
    if mode == "source_unique":
        if tracker.get("source_idx") != int(source_idx):
            tracker["source_idx"] = int(source_idx)
            tracker["seen_targets"] = set()
        seen_targets = tracker["seen_targets"]
        target = int(target_idx)
        if target in seen_targets:
            return True
        seen_targets.add(target)
        return False
    if mode == "bitset":
        group_count = int(tracker["group_count"])
        offset = left * group_count - (left * (left + 1)) // 2 + (right - left - 1)
        byte_index = int(offset // 8)
        bit_index = int(offset % 8)
        mask = np.uint8(1 << bit_index)
        current = tracker["bits"][byte_index]
        if int(current & mask) != 0:
            return True
        tracker["bits"][byte_index] = np.uint8(current | mask)
        return False
    rows = tracker["rows"]
    row = rows.get(left)
    if row is None:
        row = np.zeros(int(tracker["group_count"]) - left - 1, dtype=bool)
        rows[left] = row
    local_idx = right - left - 1
    if bool(row[local_idx]):
        return True
    row[local_idx] = True
    return False


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


def _quantized_value(value: float, quant_step_um: float = GRID_BUCKET_QUANT_UM) -> int:
    """把物理坐标量化成稳定整数，供 geometry seed 分类使用。"""

    return int(round(float(value) / max(float(quant_step_um), 1e-9)))


def _seed_bbox_for_center(center_xy: Tuple[float, float], grid_step_um: float) -> Tuple[float, float, float, float]:
    """围绕 seed center 生成局部 marker bbox。"""

    return _make_centered_bbox((float(center_xy[0]), float(center_xy[1])), float(grid_step_um), float(grid_step_um))


def _make_geometry_seed(
    layout_bbox: Tuple[float, float, float, float],
    center_xy: Tuple[float, float],
    grid_step_um: float,
    seed_type: str,
    bucket_weight: int = 1,
) -> GridSeedCandidate:
    """按统一规则构造 geometry-driven seed。"""

    center = (float(center_xy[0]), float(center_xy[1]))
    return GridSeedCandidate(
        center=center,
        seed_bbox=_seed_bbox_for_center(center, grid_step_um),
        grid_ix=_grid_anchor_index(layout_bbox[0], center[0], grid_step_um),
        grid_iy=_grid_anchor_index(layout_bbox[1], center[1], grid_step_um),
        bucket_weight=int(bucket_weight),
        seed_type=str(seed_type),
    )


def _element_size_key(item: Dict[str, Any]) -> Tuple[int, int, int, int]:
    """按 layer/datatype 和 bbox 尺寸量化分组，用于重复阵列识别。"""

    bbox = tuple(float(v) for v in item["bbox"])
    return (
        int(item["layer"]),
        int(item["datatype"]),
        _quantized_value(bbox[2] - bbox[0]),
        _quantized_value(bbox[3] - bbox[1]),
    )


def _bbox_intersection(
    bbox_a: Tuple[float, float, float, float],
    bbox_b: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float] | None:
    """返回两个 bbox 的相交区域；不相交时返回 None。"""

    x0 = max(float(bbox_a[0]), float(bbox_b[0]))
    y0 = max(float(bbox_a[1]), float(bbox_b[1]))
    x1 = min(float(bbox_a[2]), float(bbox_b[2]))
    y1 = min(float(bbox_a[3]), float(bbox_b[3]))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _is_long_shape_bbox(bbox: Tuple[float, float, float, float], clip_size_um: float) -> bool:
    """识别长条图形，避免大 bbox 被扩成二维 seed 网格。"""

    width = max(0.0, float(bbox[2]) - float(bbox[0]))
    height = max(0.0, float(bbox[3]) - float(bbox[1]))
    if width <= 0.0 or height <= 0.0:
        return False
    long_edge = max(width, height)
    short_edge = min(width, height)
    aspect = long_edge / max(short_edge, 1e-9)
    return bool(long_edge >= 4.0 * float(clip_size_um) and aspect >= 4.0)


def _axis_seed_positions(start: float, end: float, grid_step_um: float) -> List[float]:
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
    unique: Dict[int, float] = {}
    for value in values:
        clipped = min(max(float(value), start), end)
        unique[_quantized_value(clipped, 1e-6)] = clipped
    return [unique[key] for key in sorted(unique)]


def _detect_long_shape_ids(
    layout_index: LayoutIndex,
    clip_size_um: float,
    excluded_ids: set[int] | None = None,
) -> set[int]:
    """返回长条图形的 element id 集合。"""

    excluded = set(int(value) for value in (excluded_ids or set()))
    long_ids: set[int] = set()
    for idx, item in enumerate(layout_index.indexed_elements):
        if int(idx) in excluded:
            continue
        if _is_long_shape_bbox(tuple(float(v) for v in item["bbox"]), clip_size_um):
            long_ids.add(int(idx))
    return long_ids


def _array_edge_class(index: int, max_index: int) -> int:
    """把阵列坐标映射成左/内/右或下/内/上的三类边界状态。"""

    index = int(index)
    max_index = int(max_index)
    if index <= 0:
        return 0
    if index >= max_index:
        return 2
    return 1


def _array_occupancy_signature(occupied: set[Tuple[int, int]], col: int, row: int) -> int:
    """生成某个阵列局部 3x3 occupancy signature。"""

    signature = 0
    bit = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if (int(col) + dx, int(row) + dy) in occupied:
                signature |= 1 << bit
            bit += 1
    return int(signature)


def _record_array_spacing_representative(
    representatives: Dict[Tuple[Any, ...], Dict[str, Any]],
    rep_key: Tuple[Any, ...],
    center: Tuple[float, float],
    weight: int = 1,
) -> None:
    """把一个 spacing 位置合并进代表表。"""

    current = representatives.get(rep_key)
    if current is None:
        representatives[rep_key] = {"center": center, "weight": int(weight)}
    else:
        current["weight"] += int(weight)


def _array_cell_class_count(max_index: int, edge_class: int) -> int:
    """返回某个阵列 cell 边界类别包含的坐标数量。"""

    max_index = int(max_index)
    edge_class = int(edge_class)
    if max_index < 0:
        return 0
    if edge_class in (0, 2):
        return 1
    return max(0, int(max_index) - 1)


def _array_gap_class_count(max_index: int, edge_class: int) -> int:
    """返回某个相邻 cell gap 边界类别包含的间距数量。"""

    max_index = int(max_index)
    edge_class = int(edge_class)
    if max_index <= 0:
        return 0
    if edge_class in (0, 2):
        return 1
    return max(0, int(max_index) - 2)


def _array_cell_sample_index(max_index: int, edge_class: int) -> int:
    """为某个 cell 边界类别挑一个代表坐标。"""

    max_index = int(max_index)
    edge_class = int(edge_class)
    if edge_class == 0:
        return 0
    if edge_class == 2:
        return max(0, max_index)
    return max(0, min(max_index, max_index // 2))


def _array_gap_sample_index(max_index: int, edge_class: int) -> int:
    """为某个相邻 cell gap 边界类别挑一个代表 gap 起点。"""

    max_index = int(max_index)
    edge_class = int(edge_class)
    if edge_class == 0:
        return 0
    if edge_class == 2:
        return max(0, max_index - 1)
    return max(1, min(max_index - 2, max_index // 2))


def _build_dense_array_spacing_representatives(
    group_key: Tuple[int, int, int, int],
    occupied: set[Tuple[int, int]],
    x_centers: Sequence[float],
    y_centers: Sequence[float],
    max_col: int,
    max_row: int,
) -> Tuple[Dict[Tuple[Any, ...], Dict[str, Any]], int]:
    """对满阵列用解析计数生成 spacing 代表，避免遍历海量相邻对。"""

    representatives: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
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


def _build_array_spacing_representatives(
    group_key: Tuple[int, int, int, int],
    occupied: set[Tuple[int, int]],
    x_centers: Sequence[float],
    y_centers: Sequence[float],
    max_col: int,
    max_row: int,
) -> Tuple[Dict[Tuple[Any, ...], Dict[str, Any]], int]:
    """为规则阵列生成 x/y/corner 三类图元间距代表。"""

    representatives: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
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


def _build_array_representative_seeds(
    layout_index: LayoutIndex,
    layout_bbox: Tuple[float, float, float, float],
    grid_step_um: float,
) -> Tuple[List[GridSeedCandidate], List[GridSeedCandidate], set[int], int, List[Dict[str, Any]]]:
    """识别规则二维阵列，并为边界/内部/邻域类型生成代表 seed。"""

    grouped: Dict[Tuple[int, int, int, int], List[Tuple[int, Dict[str, Any], Tuple[float, float, float, float]]]] = {}
    for idx, item in enumerate(layout_index.indexed_elements):
        bbox = tuple(float(v) for v in item["bbox"])
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        grouped.setdefault(_element_size_key(item), []).append((int(idx), item, bbox))

    seeds: List[GridSeedCandidate] = []
    spacing_seeds: List[GridSeedCandidate] = []
    classified_ids: set[int] = set()
    array_group_count = 0
    audit_groups: List[Dict[str, Any]] = []
    for group_key, entries in grouped.items():
        if len(entries) < 16:
            continue
        x_center_by_key: Dict[int, float] = {}
        y_center_by_key: Dict[int, float] = {}
        for _, _, bbox in entries:
            x_center = 0.5 * (bbox[0] + bbox[2])
            y_center = 0.5 * (bbox[1] + bbox[3])
            x_center_by_key.setdefault(_quantized_value(x_center), float(x_center))
            y_center_by_key.setdefault(_quantized_value(y_center), float(y_center))
        x_keys = sorted(x_center_by_key.keys())
        y_keys = sorted(y_center_by_key.keys())
        if len(x_keys) < 3 or len(y_keys) < 3:
            continue

        x_index = {key: idx for idx, key in enumerate(x_keys)}
        y_index = {key: idx for idx, key in enumerate(y_keys)}
        x_centers = [float(x_center_by_key[key]) for key in x_keys]
        y_centers = [float(y_center_by_key[key]) for key in y_keys]
        occupied: set[Tuple[int, int]] = set()
        item_positions: List[Tuple[int, Dict[str, Any], Tuple[float, float, float, float], Tuple[int, int]]] = []
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
        representatives: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        max_col = int(len(x_keys) - 1)
        max_row = int(len(y_keys) - 1)
        for item_id, _, bbox, pos in item_positions:
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
                group_key, occupied, x_centers, y_centers, max_col, max_row
            )
            spacing_generation_mode = "dense_analytic"
        else:
            spacing_representatives, spacing_raw_count = _build_array_spacing_representatives(
                group_key, occupied, x_centers, y_centers, max_col, max_row
            )
            spacing_generation_mode = "exact_sparse"

        for rep in spacing_representatives.values():
            spacing_seeds.append(
                _make_geometry_seed(layout_bbox, rep["center"], grid_step_um, SEED_TYPE_ARRAY_SPACE, int(rep["weight"]))
            )

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


def _build_long_shape_path_seeds(
    layout_index: LayoutIndex,
    layout_bbox: Tuple[float, float, float, float],
    grid_step_um: float,
    long_ids: set[int],
) -> List[GridSeedCandidate]:
    """为长条图形生成一维路径 seed。"""

    seeds: List[GridSeedCandidate] = []
    for item_id in sorted(int(value) for value in long_ids):
        item = layout_index.indexed_elements[int(item_id)]
        bbox = tuple(float(v) for v in item["bbox"])
        width = max(0.0, bbox[2] - bbox[0])
        height = max(0.0, bbox[3] - bbox[1])
        horizontal = bool(width >= height)
        axis_values = (
            _axis_seed_positions(bbox[0], bbox[2], grid_step_um)
            if horizontal
            else _axis_seed_positions(bbox[1], bbox[3], grid_step_um)
        )
        for other_id in layout_index.spatial_index.intersection(bbox):
            if int(other_id) == int(item_id):
                continue
            other_bbox = tuple(float(v) for v in layout_index.indexed_elements[int(other_id)]["bbox"])
            overlap = _bbox_intersection(bbox, other_bbox)
            if overlap is None:
                continue
            axis_values.append(0.5 * (overlap[0] + overlap[2]) if horizontal else 0.5 * (overlap[1] + overlap[3]))
        unique_values: Dict[int, float] = {}
        for axis_value in axis_values:
            unique_values[_quantized_value(axis_value, 1e-6)] = float(axis_value)
        for value in [unique_values[key] for key in sorted(unique_values)]:
            center = (float(value), 0.5 * (bbox[1] + bbox[3])) if horizontal else (0.5 * (bbox[0] + bbox[2]), float(value))
            seeds.append(_make_geometry_seed(layout_bbox, center, grid_step_um, SEED_TYPE_LONG, 1))
    return seeds


def _build_residual_local_grid_seeds(
    layout_index: LayoutIndex,
    layout_bbox: Tuple[float, float, float, float],
    grid_step_um: float,
    classified_ids: set[int],
) -> Tuple[List[GridSeedCandidate], int]:
    """只在残余图形自身 bbox 内生成局部 grid seed。"""

    seeds: List[GridSeedCandidate] = []
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


def _dedupe_geometry_seeds(seeds: Sequence[GridSeedCandidate]) -> List[GridSeedCandidate]:
    """按全局 grid anchor 去重，并让 spacing seed 与普通 seed 各保留一个槽位。"""

    priority = {SEED_TYPE_ARRAY: 3, SEED_TYPE_LONG: 2, SEED_TYPE_RESIDUAL: 1}
    buckets: Dict[Tuple[int, int, str], GridSeedCandidate] = {}
    for seed in seeds:
        slot = "spacing" if str(seed.seed_type) == SEED_TYPE_ARRAY_SPACE else "normal"
        key = (int(seed.grid_ix), int(seed.grid_iy), slot)
        current = buckets.get(key)
        if current is None:
            buckets[key] = seed
            continue
        total_weight = int(current.bucket_weight) + int(seed.bucket_weight)
        if int(priority.get(str(seed.seed_type), 0)) > int(priority.get(str(current.seed_type), 0)):
            buckets[key] = replace(seed, bucket_weight=int(total_weight))
        else:
            buckets[key] = replace(current, bucket_weight=int(total_weight))
    return sorted(buckets.values(), key=lambda seed: (int(seed.grid_ix), int(seed.grid_iy), str(seed.seed_type)))


def _seed_type_counts(seeds: Sequence[GridSeedCandidate]) -> Dict[str, int]:
    """统计各类 seed 的数量。"""

    return {str(key): int(value) for key, value in Counter(str(seed.seed_type) for seed in seeds).items()}


def _empty_seed_stats(grid_step_um: float) -> Dict[str, Any]:
    """返回空版图下仍保持字段完整的 seed 统计。"""

    return {
        "seed_strategy": "geometry_driven",
        "grid_step_ratio": float(GRID_STEP_RATIO),
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
            "grid_step_ratio": float(GRID_STEP_RATIO),
            "grid_step_um": float(grid_step_um),
            "array_group_count": 0,
            "array_spacing_group_count": 0,
            "array_spacing_seed_count": 0,
            "array_spacing_weight_total": 0,
            "array_groups": [],
        },
    }


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
) -> Tuple[List[GridSeedCandidate], Dict[str, Any]]:
    """按图形结构直接生成 geometry-driven seeds。"""

    grid_step_um = float(clip_size_um) * float(GRID_STEP_RATIO)
    if len(layout_index.indexed_elements) == 0:
        return [], _empty_seed_stats(grid_step_um)

    layout_bbox = _layout_bbox(layout_index)
    array_seeds, array_spacing_seeds, array_ids, array_group_count, array_audit_groups = _build_array_representative_seeds(
        layout_index,
        layout_bbox,
        grid_step_um,
    )
    long_ids = _detect_long_shape_ids(layout_index, clip_size_um, array_ids)
    long_seeds = _build_long_shape_path_seeds(layout_index, layout_bbox, grid_step_um, long_ids)
    classified_ids = set(array_ids)
    classified_ids.update(int(value) for value in long_ids)
    residual_seeds, residual_count = _build_residual_local_grid_seeds(layout_index, layout_bbox, grid_step_um, classified_ids)

    raw_seeds = list(array_seeds) + list(array_spacing_seeds) + list(long_seeds) + list(residual_seeds)
    seeds = _dedupe_geometry_seeds(raw_seeds)
    seed_type_counts = _seed_type_counts(seeds)
    array_spacing_group_count = sum(
        1 for group in array_audit_groups if int(group.get("spacing_representative_count", 0)) > 0
    )
    array_spacing_weight_total = int(sum(max(1, int(seed.bucket_weight)) for seed in array_spacing_seeds))
    seed_audit = {
        "seed_strategy": "geometry_driven",
        "grid_step_ratio": float(GRID_STEP_RATIO),
        "grid_step_um": float(grid_step_um),
        "array_group_count": int(array_group_count),
        "array_spacing_group_count": int(array_spacing_group_count),
        "array_spacing_seed_count": int(len(array_spacing_seeds)),
        "array_spacing_weight_total": int(array_spacing_weight_total),
        "array_groups": array_audit_groups,
    }
    return seeds, {
        "seed_strategy": "geometry_driven",
        "grid_step_ratio": float(GRID_STEP_RATIO),
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
    print("seed 策略: geometry_driven")
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
            f"  raw/dedup/merged geometry seed 数: {result.get('grid_seed_count')} / "
            f"{result.get('bucketed_seed_count')} / {result.get('seed_bucket_merged_count')}\n"
        )
        handle.write(
            f"  array/spacing/long/residual seed 数: {result.get('array_seed_count', 0)} / "
            f"{result.get('array_spacing_seed_count', 0)} / {result.get('long_shape_seed_count', 0)} / "
            f"{result.get('residual_seed_count', 0)}\n"
        )
        handle.write(
            f"  array group / spacing group / long shape / residual element 数: "
            f"{result.get('array_group_count', 0)} / {result.get('array_spacing_group_count', 0)} / "
            f"{result.get('long_shape_count', 0)} / {result.get('residual_element_count', 0)}\n"
        )
        handle.write(f"  seed type counts: {result.get('seed_type_counts', {})}\n")
        handle.write(
            f"  pre-raster cache hit/miss: {result.get('pre_raster_cache_hit', 0)} / "
            f"{result.get('pre_raster_cache_miss', 0)}\n"
        )
        handle.write(
            f"  exact bitmap cache hit/miss: {result.get('exact_bitmap_cache_hit', 0)} / "
            f"{result.get('exact_bitmap_cache_miss', 0)}\n"
        )
        handle.write(
            f"  raster payload cache 数: {result.get('pre_raster_payload_cache_count', 0)} / "
            f"{result.get('exact_bitmap_payload_cache_count', 0)}\n"
        )
        handle.write(f"  exact cluster 数: {result.get('exact_cluster_count')}\n")
        handle.write(f"  candidate 数: {result.get('candidate_count')}\n")
        handle.write(f"  candidate group 数: {result.get('candidate_group_count', 0)}\n")
        handle.write(f"  避免常驻 candidate 对象数: {result.get('candidate_object_avoided_count', 0)}\n")
        handle.write(f"  candidate 方向分布: {result.get('candidate_direction_counts', {})}\n")
        handle.write(f"  diagonal candidate 数: {result.get('diagonal_candidate_count', 0)}\n")
        handle.write(f"  selected candidate 数: {result.get('selected_candidate_count')}\n")
        handle.write(f"  final cluster 数: {result.get('total_clusters')}\n")
        handle.write(f"  selected candidate 方向分布: {result.get('selected_candidate_direction_counts')}\n")
        handle.write(f"  selected diagonal candidate 数: {result.get('selected_diagonal_candidate_count', 0)}\n")
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
        handle.write("\nseed audit:\n")
        handle.write(json.dumps(result.get("seed_audit", {}), ensure_ascii=False, indent=2, default=_json_default))
        handle.write("\nresult detail seconds:\n")
        handle.write(json.dumps(result.get("result_detail_seconds", {}), ensure_ascii=False, indent=2))
        handle.write("\nfinal verification stats:\n")
        handle.write(json.dumps(result.get("final_verification_stats", {}), ensure_ascii=False, indent=2))
        handle.write("\nfinal verification detail seconds:\n")
        handle.write(json.dumps(result.get("final_verification_detail_seconds", {}), ensure_ascii=False, indent=2))
        handle.write("\nmemory debug:\n")
        handle.write(json.dumps(result.get("memory_debug", {}), ensure_ascii=False, indent=2, default=_json_default))
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
            if "__stream_state" in result:
                _stream_save_results(result, handle)
            elif sample_count >= 1000 or cluster_count >= 1000:
                json.dump(result, handle, ensure_ascii=False, separators=(",", ":"), default=_json_default)
            else:
                json.dump(result, handle, indent=2, ensure_ascii=False, default=_json_default)
    print(f"结果已保存到: {output}")


def _stream_save_results(result: Dict[str, Any], handle: Any) -> None:
    """把大样本结果按流式数组方式写出，避免构造巨大的中间 JSON 对象。"""

    stream_state = dict(result.get("__stream_state", {}) or {})
    ordered_records = list(stream_state.get("ordered_records", []) or [])
    cluster_units = list(stream_state.get("cluster_units", []) or [])
    sample_index_map = dict(stream_state.get("sample_index_map", {}) or {})
    metadata_builder = stream_state.get("metadata_builder")
    cluster_builder = stream_state.get("cluster_builder")
    if metadata_builder is None or cluster_builder is None:
        compact_result = {key: value for key, value in result.items() if key != "__stream_state"}
        json.dump(compact_result, handle, ensure_ascii=False, separators=(",", ":"), default=_json_default)
        return

    def _dump_value(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=_json_default)

    def _write_field(name: str, value: Any, first_field: bool) -> bool:
        if not first_field:
            handle.write(",")
        handle.write(_dump_value(str(name)))
        handle.write(":")
        handle.write(_dump_value(value))
        return False

    handle.write("{")
    first = True
    for key, value in result.items():
        if key in {"__stream_state", "clusters", "file_metadata"}:
            continue
        first = _write_field(str(key), value, first)

    if not first:
        handle.write(",")
    handle.write(_dump_value("file_metadata"))
    handle.write(":[")
    for idx, record in enumerate(ordered_records):
        if idx:
            handle.write(",")
        handle.write(_dump_value(metadata_builder(record)))
    handle.write("]")

    handle.write(",")
    handle.write(_dump_value("clusters"))
    handle.write(":[")
    written_cluster = 0
    for cluster_index, (candidate, assigned_exact_clusters) in enumerate(cluster_units):
        if written_cluster:
            handle.write(",")
        handle.write(_dump_value(cluster_builder(cluster_index, candidate, assigned_exact_clusters, sample_index_map)))
        written_cluster += 1
    handle.write("]}")
    for record in ordered_records:
        _release_marker_clip_payload(record, keep_clip_bitmap=False)
    gc.collect()


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
        if isinstance(owner, CandidateClip):
            bitmap = _candidate_clip_bitmap(owner)
        else:
            bitmap = getattr(owner, "clip_bitmap", None)
            assert bitmap is not None, "构建图描述符时 clip bitmap 不应为空"
        descriptor = _bitmap_descriptor(bitmap)
        owner.match_cache["optimized_graph_descriptor"] = descriptor
    return descriptor


def _cheap_descriptor(owner: Any) -> CheapDescriptor:
    """从缓存中获取 cheap 描述符；若不存在则现场计算。"""

    descriptor = owner.match_cache.get("optimized_cheap_descriptor")
    if descriptor is None:
        if isinstance(owner, CandidateClip):
            bitmap = _candidate_clip_bitmap(owner)
        else:
            bitmap = getattr(owner, "clip_bitmap", None)
            assert bitmap is not None, "构建 cheap 描述符时 clip bitmap 不应为空"
        descriptor = _cheap_bitmap_descriptor(bitmap)
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


def _distance_worst_case_proxy(bitmap: np.ndarray) -> float:
    """用距离变换补充估计窄线宽、窄间距带来的代表样本风险。"""

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


def _ensure_export_rerank_cache(record: MarkerRecord, *, include_distance: bool = False) -> None:
    """为可能提前释放 bitmap 的样本缓存导出代表重排所需分数。"""

    bitmap = getattr(record, "clip_bitmap", None)
    if bitmap is None:
        return
    if EXPORT_CHEAP_FEATURE_KEY not in record.match_cache:
        record.match_cache[EXPORT_CHEAP_FEATURE_KEY] = _cheap_feature_vector(record)
    if EXPORT_WORST_SCORE_KEY not in record.match_cache:
        record.match_cache[EXPORT_WORST_SCORE_KEY] = float(_worst_case_proxy(bitmap))
    if include_distance and EXPORT_DISTANCE_SCORE_KEY not in record.match_cache:
        record.match_cache[EXPORT_DISTANCE_SCORE_KEY] = float(_distance_worst_case_proxy(bitmap))


def _export_cheap_feature(record: MarkerRecord) -> np.ndarray:
    """读取导出代表重排特征；缺失时按需从 bitmap 或 packed bitmap 计算。"""

    cached = record.match_cache.get(EXPORT_CHEAP_FEATURE_KEY)
    if cached is not None:
        return np.asarray(cached, dtype=np.float32)
    bitmap = _clip_bitmap_for_export(record)
    desc = _cheap_bitmap_descriptor(bitmap)
    feature = np.concatenate(
        [
            np.asarray(desc.invariants, dtype=np.float32),
            np.asarray(desc.signature_grid, dtype=np.float32),
            0.5 * np.asarray(desc.signature_proj_x, dtype=np.float32),
            0.5 * np.asarray(desc.signature_proj_y, dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    record.match_cache[EXPORT_CHEAP_FEATURE_KEY] = feature
    return feature


def _export_worst_score(record: MarkerRecord) -> float:
    """读取导出代表 worst-case 分数；缺失时按需从 bitmap 或 packed bitmap 计算。"""

    cached = record.match_cache.get(EXPORT_WORST_SCORE_KEY)
    if cached is not None:
        return float(cached)
    score = float(_worst_case_proxy(_clip_bitmap_for_export(record)))
    record.match_cache[EXPORT_WORST_SCORE_KEY] = score
    return score


def _export_distance_score(record: MarkerRecord) -> float:
    """读取导出代表 distance-transform 分数；缺失时按需从 packed bitmap 计算。"""

    cached = record.match_cache.get(EXPORT_DISTANCE_SCORE_KEY)
    if cached is not None:
        return float(cached)
    score = float(_distance_worst_case_proxy(_clip_bitmap_for_export(record)))
    record.match_cache[EXPORT_DISTANCE_SCORE_KEY] = score
    return score


def _rerank_export_representative(cluster_members: Sequence[MarkerRecord]) -> Tuple[MarkerRecord, Dict[str, float]]:
    """在 cluster 内选择更适合导出的 representative sample。"""

    members = list(cluster_members)
    if not members:
        raise ValueError("cluster_members must not be empty")
    if len(members) == 1:
        return members[0], {
            "score": 1.0,
            "medoid_score": 1.0,
            "worst_case_score": float(_export_worst_score(members[0])),
            "distance_worst_case_score": 0.0,
            "weight_score": float(math.log1p(max(1, int(members[0].seed_weight)))),
        }

    features = np.asarray([_export_cheap_feature(member) for member in members], dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms <= 1e-12, 1.0, norms)
    normalized = features / norms
    centroid = np.mean(normalized, axis=0)
    centroid_norm = float(np.linalg.norm(centroid))
    if centroid_norm > 1e-12:
        centroid = centroid / centroid_norm
    medoid_scores = normalized @ centroid
    medoid_scores = (medoid_scores + 1.0) * 0.5

    worst_scores = np.asarray([_export_worst_score(member) for member in members], dtype=np.float32)
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

    base_scores = 0.45 * medoid_scores + 0.35 * worst_norm + 0.20 * weight_norm
    distance_scores = np.zeros(len(members), dtype=np.float32)
    distance_norm = np.zeros(len(members), dtype=np.float32)
    topk = min(int(EXPORT_DISTANCE_SCORE_TOPK), len(members))
    if topk > 0:
        candidate_indices = np.argsort(base_scores)[-topk:]
        candidate_distance_scores = np.asarray(
            [_export_distance_score(members[int(idx)]) for idx in candidate_indices],
            dtype=np.float32,
        )
        distance_scores[candidate_indices] = candidate_distance_scores
        distance_max = float(np.max(candidate_distance_scores)) if candidate_distance_scores.size else 0.0
        if distance_max > 1e-12:
            distance_norm[candidate_indices] = candidate_distance_scores / distance_max

    risk_norm = 0.65 * worst_norm + 0.35 * distance_norm
    scores = base_scores.copy()
    if topk > 0:
        scores[candidate_indices] = (
            0.45 * medoid_scores[candidate_indices]
            + 0.35 * risk_norm[candidate_indices]
            + 0.20 * weight_norm[candidate_indices]
        )
    best_idx = int(np.argmax(scores))
    return members[best_idx], {
        "score": float(scores[best_idx]),
        "medoid_score": float(medoid_scores[best_idx]),
        "worst_case_score": float(worst_scores[best_idx]),
        "distance_worst_case_score": float(distance_scores[best_idx]),
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


def _diagonal_shift_direction(shift_x_px: int, shift_y_px: int) -> str:
    """把二维位移映射成稳定的 diagonal shift 名称。"""

    if int(shift_x_px) >= 0 and int(shift_y_px) >= 0:
        return "diag_ne"
    if int(shift_x_px) < 0 and int(shift_y_px) >= 0:
        return "diag_nw"
    if int(shift_x_px) >= 0 and int(shift_y_px) < 0:
        return "diag_se"
    return "diag_sw"


def _shift_candidate_cost(candidate: CandidateClip) -> Tuple[int, float, str]:
    """hash 去重时优先保留 base，其次轴向 shift，最后才是 diagonal shift。"""

    direction = str(candidate.shift_direction)
    if direction == "base":
        shift_kind = 0
    elif direction.startswith("diag_"):
        shift_kind = 2
    else:
        shift_kind = 1
    return shift_kind, abs(float(candidate.shift_distance_um)), str(candidate.candidate_id)


def _limited_nonzero_shifts(shift_values: Sequence[int], max_count: int) -> List[int]:
    """从单轴 systematic shift 列表中取少量非零位移，用于构造 diagonal 组合。"""

    values: List[int] = []
    seen: set[int] = set()
    for value in shift_values:
        value = int(value)
        if value == 0 or value in seen:
            continue
        seen.add(value)
        values.append(value)
        if len(values) >= int(max_count):
            break
    return values


def _rank_diagonal_shift_pairs(
    x_shifts: Sequence[int],
    y_shifts: Sequence[int],
    max_count: int,
) -> List[Tuple[int, int]]:
    """按距离由近到远挑选有限个 diagonal 位移组合。"""

    ranked: List[Tuple[int, int, int, int]] = []
    for shift_x_px in x_shifts:
        for shift_y_px in y_shifts:
            distance_sq = int(shift_x_px) * int(shift_x_px) + int(shift_y_px) * int(shift_y_px)
            manhattan = abs(int(shift_x_px)) + abs(int(shift_y_px))
            ranked.append((distance_sq, manhattan, int(shift_x_px), int(shift_y_px)))
    ranked.sort()
    return [(item[2], item[3]) for item in ranked[: max(0, int(max_count))]]


def _candidate_shift_summary(candidates: Sequence[CandidateClip]) -> Dict[str, Any]:
    """汇总 candidate shift 方向统计，供结果 JSON 和日志诊断使用。"""

    direction_counts = Counter(str(candidate.shift_direction) for candidate in candidates)
    max_shift_distance_um = max((abs(float(candidate.shift_distance_um)) for candidate in candidates), default=0.0)
    diagonal_count = sum(
        int(count) for direction, count in direction_counts.items() if str(direction).startswith("diag_")
    )
    return {
        "candidate_direction_counts": {str(direction): int(count) for direction, count in direction_counts.items()},
        "diagonal_candidate_count": int(diagonal_count),
        "max_shift_distance_um": float(max_shift_distance_um),
    }


def _candidate_shift_summary_from_counts(
    direction_counts: Counter[str],
    max_shift_distance_um: float,
) -> Dict[str, Any]:
    """把累计方向计数压成与旧版一致的 candidate 统计结构。"""

    diagonal_count = sum(
        int(count) for direction, count in direction_counts.items() if str(direction).startswith("diag_")
    )
    return {
        "candidate_direction_counts": {str(direction): int(count) for direction, count in direction_counts.items()},
        "diagonal_candidate_count": int(diagonal_count),
        "max_shift_distance_um": float(max_shift_distance_um),
    }


def _candidate_greedy_tiebreak(candidate: CandidateClip) -> Tuple[int, float, int, str]:
    """在 coverage 完全相同的前提下，复用 greedy set cover 的候选优先级。"""

    return (
        -1 if str(candidate.shift_direction) == "base" else 0,
        abs(float(candidate.shift_distance_um)),
        int(candidate.origin_exact_cluster_id),
        str(candidate.candidate_id),
    )


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


def _init_coverage_geometry_cache(bitmap: np.ndarray) -> Dict[str, Any]:
    """初始化 coverage 几何缓存，只生成 packed bitmap 和面积。"""

    bitmap = np.ascontiguousarray(np.asarray(bitmap, dtype=bool))
    return {
        "packed": np.packbits(bitmap.reshape(-1)),
        "area_px": int(np.count_nonzero(bitmap)),
        "shape": tuple(int(value) for value in bitmap.shape),
    }


def _extend_coverage_dilated_cache(cache: Dict[str, Any], tol_px: int, bitmap: np.ndarray) -> None:
    """把 coverage 几何缓存扩展到膨胀层，不计算 donut。"""

    if int(tol_px) <= 0 or "packed_dilated" in cache:
        return
    bitmap = np.ascontiguousarray(np.asarray(bitmap, dtype=bool))
    structure = _coverage_structure(int(tol_px))
    dilated = np.ascontiguousarray(ndimage.binary_dilation(bitmap, structure=structure), dtype=bool)
    cache["packed_dilated"] = np.packbits(dilated.reshape(-1))
    cache["dilated_area_px"] = int(np.count_nonzero(dilated))


def _extend_coverage_donut_cache(cache: Dict[str, Any], tol_px: int, bitmap: np.ndarray) -> None:
    """把 coverage 几何缓存扩展到 donut 层，供最终 ECC overlap 使用。"""

    if int(tol_px) <= 0 or "packed_donut" in cache:
        return
    _extend_coverage_dilated_cache(cache, int(tol_px), bitmap)
    bitmap = np.ascontiguousarray(np.asarray(bitmap, dtype=bool))
    structure = _coverage_structure(int(tol_px))
    dilated = np.ascontiguousarray(ndimage.binary_dilation(bitmap, structure=structure), dtype=bool)
    eroded = ndimage.binary_erosion(bitmap, structure=structure, border_value=0)
    donut = dilated & ~eroded
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

    if isinstance(owner, CandidateClip):
        bitmap = _candidate_clip_bitmap(owner)
    else:
        bitmap = getattr(owner, "clip_bitmap", None)
        assert bitmap is not None, "ECC 缓存构建时 clip_bitmap 不应为空"
        bitmap = np.ascontiguousarray(np.asarray(bitmap, dtype=bool))
    cache = {
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

    candidate_bitmap = _candidate_clip_bitmap(candidate)
    target_bitmap = getattr(target, "clip_bitmap", None)
    assert target_bitmap is not None, "ECC 匹配时 target.clip_bitmap 不应为空"
    candidate_bitmap = np.ascontiguousarray(np.asarray(candidate_bitmap, dtype=bool))
    target_bitmap = np.ascontiguousarray(np.asarray(target_bitmap, dtype=bool))

    if candidate_bitmap.shape != target_bitmap.shape:
        return False
    if not candidate_bitmap.any() and not target_bitmap.any():
        return True
    if not candidate_bitmap.any() or not target_bitmap.any():
        return False

    tol_px = max(0, int(math.ceil(float(edge_tolerance_um) / max(float(pixel_size_um), 1e-12) - 1e-12)))
    if tol_px <= 0:
        return bool(np.array_equal(candidate_bitmap, target_bitmap))

    cand = _ecc_cache(candidate, tol_px)
    tgt = _ecc_cache(target, tol_px)
    cand_area = max(float(cand["area"]), 1.0)
    tgt_area = max(float(tgt["area"]), 1.0)
    residual_cand = np.count_nonzero(candidate_bitmap & ~tgt["dilated"]) / cand_area
    residual_tgt = np.count_nonzero(target_bitmap & ~cand["dilated"]) / tgt_area
    if residual_cand > ECC_RESIDUAL_RATIO or residual_tgt > ECC_RESIDUAL_RATIO:
        return False
    if int(cand["donut_area"]) == 0 or int(tgt["donut_area"]) == 0:
        return True
    overlap = int(np.count_nonzero(cand["donut"] & tgt["donut"]))
    denom = max(min(int(cand["donut_area"]), int(tgt["donut_area"])), 1)
    return float(overlap / denom) >= ECC_DONUT_OVERLAP_RATIO


class OptimizedMainlineRunner(MainlineRunner):
    """geometry-driven optimized v1 主运行器，负责串起完整聚类流程。"""

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
        self.memory_debug = _empty_memory_debug()
        self._base_candidate_by_exact_id: Dict[int, CandidateClip] = {}
        self._seed_stats_by_source: Dict[str, Dict[str, Any]] = {}
        self._candidate_bitmap_pool: Dict[Tuple[int, int, bytes], List[np.ndarray]] = {}
        self.materialize_outputs = bool(config.get("materialize_outputs", False))

    def _log(self, message: str) -> None:
        """统一中文过程日志输出入口。"""

        print(message)

    def _mark_memory(self, key: str) -> None:
        """记录某个阶段结束后的 RSS，并同步更新峰值估计。"""

        value = _process_rss_mb()
        self.memory_debug[str(key)] = value
        peak = self.memory_debug.get("rss_peak_estimate_mb")
        if value is not None:
            if peak is None or float(value) > float(peak):
                self.memory_debug["rss_peak_estimate_mb"] = float(value)
            self._log(f"内存 RSS[{key}] = {value:.3f} MB")
        else:
            self._log(f"内存 RSS[{key}] = unavailable")

    def _release_marker_records_after_exact_cluster(
        self,
        marker_records: Sequence[MarkerRecord],
        exact_clusters: Sequence[ExactCluster],
    ) -> None:
        """在 exact cluster 完成后释放非 representative 的大 bitmap。"""

        representative_ids = {id(cluster.representative) for cluster in exact_clusters}
        for record in marker_records:
            if id(record) in representative_ids:
                continue
            if not self.materialize_outputs:
                _ensure_export_rerank_cache(record, include_distance=False)
                if _pack_marker_clip_bitmap(record):
                    self.memory_debug["packed_marker_clip_count"] += 1
                if _release_marker_clip_payload(record, keep_clip_bitmap=False):
                    self.memory_debug["released_marker_clip_count"] += 1
                continue
            if _release_marker_expanded_bitmap(record):
                self.memory_debug["released_marker_expanded_count"] += 1
        gc.collect()

    def _lighten_online_exact_member_record(self, record: MarkerRecord) -> None:
        """在 online exact grouping 时尽早释放非代表成员的大 bitmap。"""

        if self.materialize_outputs:
            if _release_marker_expanded_bitmap(record):
                self.memory_debug["released_marker_expanded_early_count"] += 1
            return

        self.memory_debug["light_member_record_count"] += 1
        _ensure_export_rerank_cache(record, include_distance=True)
        had_expanded = bool(
            getattr(record, "expanded_bitmap", None) is not None
            or PACKED_EXPANDED_BITMAP_KEY in record.match_cache
        )
        had_clip = bool(getattr(record, "clip_bitmap", None) is not None)
        _release_marker_clip_payload(record, keep_clip_bitmap=False)
        if had_expanded:
            self.memory_debug["released_marker_expanded_early_count"] += 1
        if had_clip and getattr(record, "clip_bitmap", None) is None:
            self.memory_debug["released_marker_clip_early_count"] += 1

    def _register_online_exact_record(
        self,
        record: MarkerRecord,
        marker_records: List[MarkerRecord],
        exact_clusters: List[ExactCluster],
        exact_index_by_key: Dict[Tuple[str, str], int],
    ) -> None:
        """把新生成的 marker 直接并入 online exact cluster 累加器。"""

        marker_records.append(record)
        exact_key = (str(record.clip_hash), str(record.expanded_hash))
        existing_cluster_id = exact_index_by_key.get(exact_key)
        if existing_cluster_id is None:
            cluster_id = int(len(exact_clusters))
            record.exact_cluster_id = cluster_id
            exact_clusters.append(
                ExactCluster(
                    exact_cluster_id=cluster_id,
                    representative=record,
                    members=[record],
                )
            )
            exact_index_by_key[exact_key] = cluster_id
            return

        cluster = exact_clusters[int(existing_cluster_id)]
        record.exact_cluster_id = int(cluster.exact_cluster_id)
        cluster.members.append(record)
        self._lighten_online_exact_member_record(record)

    def _release_representative_expanded_bitmaps(self, exact_clusters: Sequence[ExactCluster]) -> None:
        """在 candidate 生成完成后释放 representative 的 expanded bitmap。"""

        for cluster in exact_clusters:
            if _release_marker_expanded_bitmap(cluster.representative):
                self.memory_debug["released_marker_expanded_count"] += 1
        gc.collect()

    def _release_unselected_candidates(self, candidates: Sequence[CandidateClip], selected_ids: set[str]) -> None:
        """在 set cover 后释放未入选 candidate 的 bitmap 和缓存。"""

        for candidate in candidates:
            keep_clip = str(candidate.candidate_id) in selected_ids or str(candidate.shift_direction) == "base"
            if _release_candidate_geometry_payload(candidate, keep_clip_bitmap=keep_clip):
                self.memory_debug["released_cache_owner_count"] += 1
            if not keep_clip and getattr(candidate, "clip_bitmap", None) is None:
                self.memory_debug["released_candidate_clip_count"] += 1
        gc.collect()

    def _release_unselected_candidate_groups(
        self,
        candidate_groups: Sequence[CoverageCandidateGroup],
        selected_ids: set[str],
    ) -> None:
        """在 set cover 后释放未入选 candidate group 的长期常驻 bitmap 与缓存。"""

        for candidate_group in candidate_groups:
            candidate = candidate_group.best_candidate
            keep_clip = str(candidate.candidate_id) in selected_ids or str(candidate.shift_direction) == "base"
            if _release_candidate_geometry_payload(candidate, keep_clip_bitmap=keep_clip):
                self.memory_debug["released_cache_owner_count"] += 1
            if not keep_clip and getattr(candidate, "clip_bitmap", None) is None:
                self.memory_debug["released_candidate_clip_count"] += 1
        gc.collect()

    def _release_marker_records_before_metadata(self, marker_records: Sequence[MarkerRecord]) -> None:
        """在不物化 sample 时，verification 完成后释放 marker bitmap。"""

        if self.materialize_outputs:
            return
        for record in marker_records:
            if _release_marker_clip_payload(record, keep_clip_bitmap=False):
                self.memory_debug["released_marker_clip_count"] += 1
        gc.collect()

    def _intern_candidate_bitmap(self, bitmap: np.ndarray, clip_hash: str) -> np.ndarray:
        """按逐像素 exact key 共享 candidate bitmap，减少重复 ndarray 常驻。"""

        del clip_hash
        key, mask, _, packed_size = _strict_bitmap_digest_key(bitmap)
        cached_list = self._candidate_bitmap_pool.get(key)
        if cached_list is not None:
            for cached in cached_list:
                if _same_bitmap(cached, mask):
                    self.memory_debug["candidate_bitmap_pool_hit_count"] += 1
                    return cached
            cached_list.append(mask)
            self.memory_debug["strict_digest_collision_count"] += 1
        else:
            self._candidate_bitmap_pool[key] = [mask]
            self.memory_debug["strict_digest_key_count"] += 1
        self.memory_debug["candidate_bitmap_pool_unique_count"] += 1
        self.memory_debug["strict_key_bytes_avoided_estimate_mb"] += float(packed_size) / (1024.0 * 1024.0)
        return mask

    def _park_candidate_group_bitmap(
        self,
        candidate: CandidateClip,
        packed: np.ndarray,
        shape: Tuple[int, int],
    ) -> None:
        """把 candidate clip bitmap 切到 packed-at-rest 形态，降低长期常驻内存。"""

        if PACKED_CANDIDATE_CLIP_BITMAP_KEY not in candidate.match_cache:
            _attach_packed_candidate_clip_bitmap(candidate, packed, shape)
            self.memory_debug["packed_candidate_group_bitmap_count"] += 1
            self.memory_debug["candidate_group_bitmap_bytes_avoided_estimate_mb"] += (
                float(max(int(shape[0]) * int(shape[1]) - int(np.asarray(packed, dtype=np.uint8).size), 0))
                / (1024.0 * 1024.0)
            )
        candidate.clip_bitmap = None

    def _merge_coverage_candidate(
        self,
        group_buckets: Dict[Tuple[int, int, bytes], List[CoverageCandidateGroup]],
        ordered_groups: List[CoverageCandidateGroup],
        candidate: CandidateClip,
        *,
        retain_materialized_candidates: bool,
    ) -> None:
        """把一个 candidate 合并进全局 strict bitmap group，避免全量对象长期常驻。"""

        bitmap = getattr(candidate, "clip_bitmap", None)
        assert bitmap is not None, "coverage candidate group 合并时 candidate.clip_bitmap 不应为空"
        key, mask, packed, packed_size = _strict_bitmap_digest_key(bitmap)
        bucket_list = group_buckets.get(key)
        if bucket_list is None:
            bucket_list = []
            group_buckets[key] = bucket_list
        matched_group = None
        for candidate_group in bucket_list:
            if (
                tuple(int(value) for value in candidate_group.clip_bitmap_shape) == tuple(int(value) for value in mask.shape)
                and np.array_equal(
                    np.asarray(candidate_group.packed_clip_bitmap, dtype=np.uint8),
                    np.asarray(packed, dtype=np.uint8),
                )
            ):
                matched_group = candidate_group
                break
        if matched_group is None:
            if bucket_list:
                self.memory_debug["strict_digest_collision_count"] += 1
            packed_shape = (int(mask.shape[0]), int(mask.shape[1]))
            candidate_group = CoverageCandidateGroup(
                best_candidate=candidate,
                packed_clip_bitmap=np.asarray(packed, dtype=np.uint8),
                clip_bitmap_shape=packed_shape,
                area_px=int(np.count_nonzero(mask)),
                clip_hash=str(candidate.clip_hash),
                origin_ids=_coverage_to_array([int(candidate.origin_exact_cluster_id)]),
                logical_candidate_count=1,
                direction_counts={str(candidate.shift_direction): 1},
                coverage=_coverage_to_array(candidate.coverage),
                materialized_candidates=(candidate,) if retain_materialized_candidates else (),
            )
            if not retain_materialized_candidates:
                self._park_candidate_group_bitmap(candidate, candidate_group.packed_clip_bitmap, packed_shape)
            bucket_list.append(candidate_group)
            ordered_groups.append(candidate_group)
            self.memory_debug["strict_digest_key_count"] = max(
                int(self.memory_debug.get("strict_digest_key_count", 0)),
                int(len(group_buckets)),
            )
            self.memory_debug["strict_key_bytes_avoided_estimate_mb"] = max(
                float(self.memory_debug.get("strict_key_bytes_avoided_estimate_mb", 0.0)),
                float(packed_size) / (1024.0 * 1024.0),
            )
            return

        matched_group.logical_candidate_count += 1
        matched_group.origin_ids = _coverage_union(
            np.asarray(matched_group.origin_ids, dtype=np.int32),
            _coverage_to_array([int(candidate.origin_exact_cluster_id)]),
        )
        matched_group.coverage = _coverage_union(
            _coverage_to_array(matched_group.coverage),
            _coverage_to_array(candidate.coverage),
        )
        direction = str(candidate.shift_direction)
        matched_group.direction_counts[direction] = int(matched_group.direction_counts.get(direction, 0)) + 1
        if retain_materialized_candidates:
            matched_group.materialized_candidates = tuple(matched_group.materialized_candidates) + (candidate,)
        if _candidate_greedy_tiebreak(candidate) < _candidate_greedy_tiebreak(matched_group.best_candidate):
            previous_best = matched_group.best_candidate
            matched_group.best_candidate = candidate
            if not retain_materialized_candidates:
                self._park_candidate_group_bitmap(candidate, matched_group.packed_clip_bitmap, matched_group.clip_bitmap_shape)
            if (
                not retain_materialized_candidates
                and (
                str(previous_best.shift_direction) != "base"
                and previous_best is not self._base_candidate_by_exact_id.get(int(previous_best.origin_exact_cluster_id))
                )
            ):
                if _release_candidate_geometry_payload(previous_best, keep_clip_bitmap=False):
                    self.memory_debug["released_cache_owner_count"] += 1
                if getattr(previous_best, "clip_bitmap", None) is None:
                    self.memory_debug["released_candidate_clip_count"] += 1
            elif not retain_materialized_candidates:
                self._park_candidate_group_bitmap(
                    previous_best,
                    matched_group.packed_clip_bitmap,
                    matched_group.clip_bitmap_shape,
                )
        elif (
            not retain_materialized_candidates
            and (
            str(candidate.shift_direction) != "base"
            and candidate is not self._base_candidate_by_exact_id.get(int(candidate.origin_exact_cluster_id))
            )
        ):
            if _release_candidate_geometry_payload(candidate, keep_clip_bitmap=False):
                self.memory_debug["released_cache_owner_count"] += 1
            if getattr(candidate, "clip_bitmap", None) is None:
                self.memory_debug["released_candidate_clip_count"] += 1
        elif not retain_materialized_candidates:
            self._park_candidate_group_bitmap(
                candidate,
                matched_group.packed_clip_bitmap,
                matched_group.clip_bitmap_shape,
            )

    def _build_global_coverage_candidate_groups(
        self,
        exact_clusters: Sequence[ExactCluster],
    ) -> Tuple[List[CoverageCandidateGroup], int, Dict[str, Any]]:
        """逐个 exact cluster 生成 candidate，并立即压成全局紧凑 coverage group。"""

        group_buckets: Dict[Tuple[int, int, bytes], List[CoverageCandidateGroup]] = {}
        ordered_groups: List[CoverageCandidateGroup] = []
        direction_counts: Counter[str] = Counter()
        max_shift_distance_um = 0.0
        candidate_count = 0
        for exact_cluster in exact_clusters:
            cluster_candidates = self._generate_candidates_for_cluster(exact_cluster)
            candidate_count += int(len(cluster_candidates))
            for candidate in cluster_candidates:
                direction_counts[str(candidate.shift_direction)] += 1
                max_shift_distance_um = max(max_shift_distance_um, abs(float(candidate.shift_distance_um)))
                self._merge_coverage_candidate(
                    group_buckets,
                    ordered_groups,
                    candidate,
                    retain_materialized_candidates=False,
                )
            cluster_candidates.clear()
        self.memory_debug["candidate_object_avoided_count"] = max(0, int(candidate_count - len(ordered_groups)))
        candidate_shift_summary = _candidate_shift_summary_from_counts(direction_counts, max_shift_distance_um)
        return ordered_groups, int(candidate_count), candidate_shift_summary

    def _release_candidate_bitmap_pool(self) -> None:
        """清空 candidate bitmap intern pool 的字典引用。"""

        if self._candidate_bitmap_pool:
            self._candidate_bitmap_pool.clear()
            gc.collect()

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
        """根据 geometry-driven seed bbox 计算父类 marker raster 所需窗口参数。"""

        marker_bbox = tuple(float(value) for value in candidate.seed_bbox)
        marker_center = _bbox_center(marker_bbox)
        return _raster_window_spec(marker_bbox, marker_center, self.clip_size_um, self.pixel_size_um)

    def _clone_cached_record(
        self,
        cached: MarkerRecord | RasterPayload,
        filepath: Path,
        marker_index: int,
        candidate: GridSeedCandidate,
    ) -> MarkerRecord:
        """复用已栅格化轻量载荷，并替换当前 seed 的身份字段。"""

        payload = _raster_payload_from_record(cached) if isinstance(cached, MarkerRecord) else cached
        marker_bbox = tuple(float(value) for value in candidate.seed_bbox)
        marker_center = _bbox_center(marker_bbox)
        raster_spec = self._seed_raster_spec(candidate)
        match_cache: Dict[str, Any] = {}
        if payload.graph_descriptor is not None:
            match_cache["optimized_graph_descriptor"] = payload.graph_descriptor
        if payload.cheap_descriptor is not None:
            match_cache["optimized_cheap_descriptor"] = payload.cheap_descriptor
        record = MarkerRecord(
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
            clip_bitmap=payload.clip_bitmap,
            expanded_bitmap=payload.expanded_bitmap,
            clip_hash=payload.clip_hash,
            expanded_hash=payload.expanded_hash,
            clip_area=float(payload.clip_area),
            seed_weight=int(candidate.bucket_weight),
            exact_cluster_id=-1,
            match_cache=match_cache,
        )
        if payload.expanded_bitmap_packed is not None and payload.expanded_bitmap_shape is not None:
            _attach_packed_expanded_bitmap(record, payload.expanded_bitmap_packed, payload.expanded_bitmap_shape)
        return record

    def _apply_seed_metadata(self, record: MarkerRecord, filepath: Path, marker_index: int, candidate: GridSeedCandidate) -> None:
        """把当前 geometry-driven seed 的样本身份和 metadata 写入 record。"""

        record.marker_id = f"{filepath.stem}__seed_{int(marker_index):06d}"
        record.seed_weight = int(candidate.bucket_weight)
        record.exact_cluster_id = -1
        record.match_cache["auto_seed"] = {
            "seed_bbox": list(candidate.seed_bbox),
            "grid_ix": int(candidate.grid_ix),
            "grid_iy": int(candidate.grid_iy),
            "grid_cell_bbox": list(candidate.seed_bbox),
            "bucket_weight": int(candidate.bucket_weight),
            "seed_type": str(candidate.seed_type),
        }

    def _collect_marker_records_for_file(
        self,
        filepath: Path,
        online_exact_state: Dict[str, Any] | None = None,
    ) -> List[MarkerRecord]:
        """对单个 OAS 文件执行 geometry-driven seed 生成，并可选地直接接入 online exact grouping。"""

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
            f"文件 {filepath.name}: raw geometry seed {seed_stats['grid_seed_count']}, "
            f"去重后 seed {seed_stats['bucketed_seed_count']}, pattern 元素 {len(layout_index.indexed_elements)}"
        )

        records: List[MarkerRecord] = []
        online_mode = online_exact_state is not None
        online_marker_records = online_exact_state["marker_records"] if online_mode else []
        online_exact_clusters = online_exact_state["exact_clusters"] if online_mode else []
        online_exact_index = online_exact_state["exact_index_by_key"] if online_mode else {}
        pre_raster_cache: Dict[str, RasterPayload] = {}
        exact_bitmap_cache: Dict[Tuple[str, str], RasterPayload] = {}
        cache_stats = {
            "pre_raster_cache_hit": 0,
            "pre_raster_cache_miss": 0,
            "exact_bitmap_cache_hit": 0,
            "exact_bitmap_cache_miss": 0,
            "pre_raster_payload_cache_count": 0,
            "exact_bitmap_payload_cache_count": 0,
        }
        generated_count = 0
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
                if getattr(record, "expanded_bitmap", None) is not None and _pack_marker_expanded_bitmap(record):
                    self.memory_debug["packed_marker_expanded_count"] += 1
                generated_count += 1
                if online_mode:
                    self._register_online_exact_record(record, online_marker_records, online_exact_clusters, online_exact_index)
                else:
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
            if _pack_marker_expanded_bitmap(record):
                self.memory_debug["packed_marker_expanded_count"] += 1
            exact_key = (str(record.clip_hash), str(record.expanded_hash))
            cached = exact_bitmap_cache.get(exact_key)
            if cached is None:
                cache_stats["exact_bitmap_cache_miss"] += 1
                exact_bitmap_cache[exact_key] = _raster_payload_from_record(record)
            else:
                cache_stats["exact_bitmap_cache_hit"] += 1
                record.clip_bitmap = cached.clip_bitmap
                record.expanded_bitmap = cached.expanded_bitmap
                record.clip_hash = cached.clip_hash
                record.expanded_hash = cached.expanded_hash
                record.clip_area = float(cached.clip_area)
                if cached.expanded_bitmap_packed is not None and cached.expanded_bitmap_shape is not None:
                    _attach_packed_expanded_bitmap(record, cached.expanded_bitmap_packed, cached.expanded_bitmap_shape)
                if cached.graph_descriptor is not None:
                    record.match_cache["optimized_graph_descriptor"] = cached.graph_descriptor
                if cached.cheap_descriptor is not None:
                    record.match_cache["optimized_cheap_descriptor"] = cached.cheap_descriptor
            pre_raster_cache[pre_key] = _raster_payload_from_record(record)
            generated_count += 1
            if online_mode:
                self._register_online_exact_record(record, online_marker_records, online_exact_clusters, online_exact_index)
            else:
                records.append(record)

        cache_stats["pre_raster_payload_cache_count"] = int(len(pre_raster_cache))
        cache_stats["exact_bitmap_payload_cache_count"] = int(len(exact_bitmap_cache))
        self.memory_debug["pre_raster_payload_cache_count"] += int(len(pre_raster_cache))
        self.memory_debug["exact_bitmap_payload_cache_count"] += int(len(exact_bitmap_cache))
        self._seed_stats_by_source[str(filepath)].update(cache_stats)
        self._log(f"文件 {filepath.name}: 生成 geometry-driven seed 窗口 {generated_count} 个")
        self._log(
            f"文件 {filepath.name}: pre-raster cache hit/miss "
            f"{cache_stats['pre_raster_cache_hit']}/{cache_stats['pre_raster_cache_miss']}, "
            f"exact bitmap cache hit/miss "
            f"{cache_stats['exact_bitmap_cache_hit']}/{cache_stats['exact_bitmap_cache_miss']}"
        )
        if online_mode:
            return []
        return records

    def run(self, input_path: str) -> Dict[str, Any]:
        """执行 geometry-driven optimized v1 主流程并返回最终结果字典。"""

        started_at = time.perf_counter()
        self.memory_debug = _empty_memory_debug()
        input_files = self._discover_input_files(input_path)
        if not input_files:
            raise ValueError("No .oas files found")

        self._seed_stats_by_source = {}
        self._log(f"发现输入 OAS 文件数: {len(input_files)}")
        self._log("开始收集 geometry-driven seed 窗口...")
        marker_started = time.perf_counter()
        marker_records: List[MarkerRecord] = []
        exact_clusters: List[ExactCluster] = []
        online_exact_state = {
            "marker_records": marker_records,
            "exact_clusters": exact_clusters,
            "exact_index_by_key": {},
        }
        for filepath in input_files:
            if self.apply_layer_operations:
                self._log(f" 对文件 {filepath.name} 应用层操作...")
            self._collect_marker_records_for_file(filepath, online_exact_state=online_exact_state)
        marker_elapsed = time.perf_counter() - marker_started

        if not marker_records:
            raise ValueError("No geometry-driven seeds produced from layout geometry")
        total_grid_seeds = sum(int(stats.get("grid_seed_count", 0)) for stats in self._seed_stats_by_source.values())
        total_bucketed = sum(int(stats.get("bucketed_seed_count", 0)) for stats in self._seed_stats_by_source.values())
        self._log(
            f"geometry-driven seed 收集完成: raw seed {total_grid_seeds}, "
            f"去重后 seed {total_bucketed}, 样本 {len(marker_records)}"
        )
        self._mark_memory("rss_collect_markers_mb")

        self._log("开始 exact hash 聚合...")
        dedup_started = time.perf_counter()
        self.memory_debug["online_exact_group_count"] = int(len(exact_clusters))
        dedup_elapsed = time.perf_counter() - dedup_started
        self._log(f"exact hash 聚合完成: {len(marker_records)} -> {len(exact_clusters)}")
        self._mark_memory("rss_exact_cluster_mb")

        self._log("开始生成 systematic shift candidates...")
        candidate_started = time.perf_counter()
        candidate_groups, candidate_count, candidate_shift_summary = self._build_global_coverage_candidate_groups(
            exact_clusters
        )
        candidate_elapsed = time.perf_counter() - candidate_started
        self._release_representative_expanded_bitmaps(exact_clusters)
        self._log(
            f"candidate 生成完成: {candidate_count} 个, "
            f"group={len(candidate_groups)}, "
            f"方向分布={candidate_shift_summary['candidate_direction_counts']}, "
            f"diagonal={candidate_shift_summary['diagonal_candidate_count']}"
        )
        self._mark_memory("rss_candidate_generation_mb")

        self._log("开始构建 verified coverage edges...")
        coverage_started = time.perf_counter()
        self.prefilter_stats = _empty_prefilter_stats()
        self._evaluate_candidate_coverage(candidate_groups, exact_clusters)
        coverage_elapsed = time.perf_counter() - coverage_started
        self._log(f"coverage 构建完成: {self.prefilter_stats}")
        self._mark_memory("rss_coverage_eval_mb")

        self._log("开始 greedy set cover...")
        cover_started = time.perf_counter()
        selected_candidates = self._greedy_cover(candidate_groups, exact_clusters)
        cover_elapsed = time.perf_counter() - cover_started
        self._release_unselected_candidate_groups(
            candidate_groups,
            {str(candidate.candidate_id) for candidate in selected_candidates},
        )
        self.memory_debug["released_candidate_list_ref_count"] += int(len(candidate_groups))
        candidate_groups.clear()
        gc.collect()
        selected_direction_counts = dict(Counter(str(candidate.shift_direction) for candidate in selected_candidates))
        self._log(
            f"set cover 完成: selected candidates={len(selected_candidates)}, "
            f"方向分布={selected_direction_counts}"
        )
        self._mark_memory("rss_set_cover_mb")

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
            candidate_count=candidate_count,
            candidate_group_count=int(self.coverage_debug_stats.get("candidate_group_count", 0)),
            candidate_shift_summary=candidate_shift_summary,
        )
        self._mark_memory("rss_result_build_mb")
        result["memory_debug"] = dict(self.memory_debug)
        if isinstance(result.get("result_summary"), dict):
            result["result_summary"]["memory_debug"] = dict(self.memory_debug)
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
        candidate.coverage = (int(cluster.exact_cluster_id),) if str(shift_direction) == "base" else ()
        candidate.clip_bitmap = self._intern_candidate_bitmap(candidate.clip_bitmap, candidate.clip_hash)
        return candidate

    def _generate_candidates_for_cluster(self, cluster: ExactCluster) -> List[CandidateClip]:
        """为一个 exact cluster 生成 base、轴向 systematic shift 与少量 diagonal candidates。"""

        rep = cluster.representative
        expanded_bitmap = _expanded_bitmap_for_marker(rep)
        self.memory_debug["unpacked_marker_expanded_count"] += 1
        proposals: List[Dict[str, Any]] = []
        proposal_slots: Dict[Tuple[int, int, bytes], List[int]] = {}

        def _proposal_cost(direction: str, shift_distance_um: float, proposal_index: int) -> Tuple[int, float, int]:
            """按 base、轴向、diagonal 的稳定优先级给 proposal 排序。"""

            if str(direction) == "base":
                shift_kind = 0
            elif str(direction).startswith("diag_"):
                shift_kind = 2
            else:
                shift_kind = 1
            return shift_kind, abs(float(shift_distance_um)), int(proposal_index)

        def _add_proposal(
            clip_bbox: Tuple[float, float, float, float],
            clip_bbox_q: Tuple[int, int, int, int],
            bitmap: np.ndarray,
            shift_direction: str,
            shift_distance_um: float,
        ) -> None:
            """在同一个 exact cluster 内先做 strict duplicate 去重，再创建 candidate。"""

            key, mask, _, _ = _strict_bitmap_digest_key(bitmap)
            proposal_index = int(len(proposals))
            cost = _proposal_cost(str(shift_direction), float(shift_distance_um), proposal_index)
            slots = proposal_slots.get(key)
            if slots is None:
                slots = []
                proposal_slots[key] = slots
            for existing_idx in slots:
                existing = proposals[int(existing_idx)]
                if _same_bitmap(existing["bitmap"], mask):
                    self.memory_debug["early_duplicate_shift_candidate_count"] += 1
                    if cost < existing["cost"]:
                        existing.update(
                            {
                                "clip_bbox": clip_bbox,
                                "clip_bbox_q": clip_bbox_q,
                                "bitmap": mask,
                                "shift_direction": str(shift_direction),
                                "shift_distance_um": float(shift_distance_um),
                                "cost": cost,
                            }
                        )
                    return
            if slots:
                self.memory_debug["strict_digest_collision_count"] += 1
            slots.append(proposal_index)
            proposals.append(
                {
                    "clip_bbox": clip_bbox,
                    "clip_bbox_q": clip_bbox_q,
                    "bitmap": mask,
                    "shift_direction": str(shift_direction),
                    "shift_distance_um": float(shift_distance_um),
                    "cost": cost,
                }
            )

        base_bitmap = getattr(rep, "clip_bitmap", None)
        assert base_bitmap is not None, "生成 base candidate 时 representative.clip_bitmap 不应为空"
        _add_proposal(rep.clip_bbox, rep.clip_bbox_q, base_bitmap, "base", 0.0)

        base_x0, base_y0, base_x1, base_y1 = rep.clip_bbox_q
        tolerance_px = max(0, int(math.ceil(self.shift_boundary_tolerance_um / max(self.pixel_size_um, 1e-12) - 1e-12)))
        max_shift_count = 8 if self.geometry_mode == "fast" else 12

        occupied_cols = np.any(expanded_bitmap, axis=0)
        x_boundaries = _collect_boundary_positions(occupied_cols)
        x_interval = list(rep.shift_limits_px["x"])
        x_shift_values = _collect_shift_values_px(
            x_boundaries,
            base_x0,
            base_x1,
            (int(x_interval[0]), int(x_interval[1])),
            tolerance_px,
            max_count=max_shift_count,
        )
        for shift_px in x_shift_values:
            if shift_px == 0:
                continue
            clip_bbox_q = (base_x0 + int(shift_px), base_y0, base_x1 + int(shift_px), base_y1)
            bitmap = _slice_bitmap(expanded_bitmap, clip_bbox_q)
            shift_um = float(shift_px) * self.pixel_size_um
            clip_bbox = (
                rep.clip_bbox[0] + shift_um,
                rep.clip_bbox[1],
                rep.clip_bbox[2] + shift_um,
                rep.clip_bbox[3],
            )
            _add_proposal(
                clip_bbox,
                clip_bbox_q,
                bitmap,
                "right" if shift_px > 0 else "left",
                abs(shift_um),
            )

        occupied_rows = np.any(expanded_bitmap, axis=1)
        y_boundaries = _collect_boundary_positions(occupied_rows)
        y_interval = list(rep.shift_limits_px["y"])
        y_shift_values = _collect_shift_values_px(
            y_boundaries,
            base_y0,
            base_y1,
            (int(y_interval[0]), int(y_interval[1])),
            tolerance_px,
            max_count=max_shift_count,
        )
        for shift_px in y_shift_values:
            if shift_px == 0:
                continue
            clip_bbox_q = (base_x0, base_y0 + int(shift_px), base_x1, base_y1 + int(shift_px))
            bitmap = _slice_bitmap(expanded_bitmap, clip_bbox_q)
            shift_um = float(shift_px) * self.pixel_size_um
            clip_bbox = (
                rep.clip_bbox[0],
                rep.clip_bbox[1] + shift_um,
                rep.clip_bbox[2],
                rep.clip_bbox[3] + shift_um,
            )
            _add_proposal(
                clip_bbox,
                clip_bbox_q,
                bitmap,
                "up" if shift_px > 0 else "down",
                abs(shift_um),
            )

        x_diagonal_shifts = _limited_nonzero_shifts(x_shift_values, DIAGONAL_SHIFT_AXIS_MAX_COUNT)
        y_diagonal_shifts = _limited_nonzero_shifts(y_shift_values, DIAGONAL_SHIFT_AXIS_MAX_COUNT)
        for shift_x_px, shift_y_px in _rank_diagonal_shift_pairs(
            x_diagonal_shifts,
            y_diagonal_shifts,
            DIAGONAL_SHIFT_MAX_COUNT,
        ):
            clip_bbox_q = (
                base_x0 + int(shift_x_px),
                base_y0 + int(shift_y_px),
                base_x1 + int(shift_x_px),
                base_y1 + int(shift_y_px),
            )
            bitmap = _slice_bitmap(expanded_bitmap, clip_bbox_q)
            shift_x_um = float(shift_x_px) * self.pixel_size_um
            shift_y_um = float(shift_y_px) * self.pixel_size_um
            clip_bbox = (
                rep.clip_bbox[0] + shift_x_um,
                rep.clip_bbox[1] + shift_y_um,
                rep.clip_bbox[2] + shift_x_um,
                rep.clip_bbox[3] + shift_y_um,
            )
            _add_proposal(
                clip_bbox,
                clip_bbox_q,
                bitmap,
                _diagonal_shift_direction(int(shift_x_px), int(shift_y_px)),
                math.sqrt(float(shift_x_px * shift_x_px + shift_y_px * shift_y_px)) * self.pixel_size_um,
            )

        candidates = [
            self._build_candidate_clip(
                cluster,
                tuple(proposal["clip_bbox"]),
                tuple(int(value) for value in proposal["clip_bbox_q"]),
                np.asarray(proposal["bitmap"], dtype=bool),
                str(proposal["shift_direction"]),
                float(proposal["shift_distance_um"]),
                int(candidate_index),
            )
            for candidate_index, proposal in enumerate(proposals)
        ]
        deduped: Dict[str, CandidateClip] = {}
        for candidate in candidates:
            current = deduped.get(candidate.clip_hash)
            if current is None or _shift_candidate_cost(candidate) < _shift_candidate_cost(current):
                deduped[candidate.clip_hash] = candidate
        candidates = list(sorted(deduped.values(), key=lambda item: item.candidate_id))
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

    def _coerce_coverage_candidate_groups(
        self,
        candidates: Sequence[CandidateClip] | Sequence[CoverageCandidateGroup],
    ) -> List[CoverageCandidateGroup]:
        """把输入统一转成 coverage candidate group，兼容旧测试直接传 candidate 列表。"""

        if len(candidates) == 0:
            return []
        first = candidates[0]
        if isinstance(first, CoverageCandidateGroup):
            return list(candidates)
        group_buckets: Dict[Tuple[int, int, bytes], List[CoverageCandidateGroup]] = {}
        ordered_groups: List[CoverageCandidateGroup] = []
        for candidate in candidates:
            self._merge_coverage_candidate(
                group_buckets,
                ordered_groups,
                candidate,
                retain_materialized_candidates=True,
            )
        return ordered_groups

    def _build_candidate_match_bundles(
        self,
        candidates: Sequence[CandidateClip] | Sequence[CoverageCandidateGroup],
        tol_px: int,
    ) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """按 bitmap shape 构建轻量 coverage bundle，不预先生成 ECC cache。"""

        del tol_px
        candidate_groups = self._coerce_coverage_candidate_groups(candidates)
        bundles: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for candidate_group in candidate_groups:
            representative_candidate = candidate_group.best_candidate
            shape = tuple(int(value) for value in candidate_group.clip_bitmap_shape)
            bundle = bundles.setdefault(
                shape,
                {
                    "areas": [],
                    "hashes": [],
                    "origin_ids": [],
                    "candidate_groups": [],
                    "representatives": [],
                    "clip_pixels": int(shape[0]) * int(shape[1]),
                    "bitmap_cache_by_idx": {},
                    "geometry_cache_by_idx": {},
                    "full_descriptor_cache_by_idx": {},
                    "full_prefilter_disabled": False,
                    "full_prefilter_probe_pairs": 0,
                    "full_prefilter_probe_rejects": 0,
                    "full_prefilter_probe_done": False,
                },
            )
            bundle["areas"].append(int(candidate_group.area_px))
            bundle["hashes"].append(str(candidate_group.clip_hash))
            bundle["origin_ids"].append(_coverage_to_array(candidate_group.origin_ids))
            bundle["candidate_groups"].append(candidate_group)
            bundle["representatives"].append(representative_candidate)

        for bundle in bundles.values():
            bundle["areas"] = np.asarray(bundle["areas"], dtype=np.int64)
            bundle["hashes_np"] = np.asarray(bundle["hashes"])
            hash_to_indices: Dict[str, List[int]] = {}
            for idx, clip_hash in enumerate(bundle["hashes"]):
                hash_to_indices.setdefault(clip_hash, []).append(idx)
            bundle["hash_to_indices"] = hash_to_indices
        return bundles

    def _initial_bundle_coverage(self, bundle: Dict[str, Any]) -> List[np.ndarray]:
        """构建 bundle group 的初始 coverage 数组。"""

        coverage_by_group: List[np.ndarray] = []
        for candidate_group in bundle["candidate_groups"]:
            coverage_by_group.append(_coverage_to_array(candidate_group.coverage))
        return coverage_by_group

    def _apply_same_hash_coverage(self, bundle: Dict[str, Any], coverage_by_group: List[np.ndarray]) -> None:
        """在完整 shape bundle 内传播 exact hash 直通 coverage。"""

        for same_hash_indices in bundle["hash_to_indices"].values():
            if len(same_hash_indices) <= 1:
                continue
            hash_origin_ids = np.asarray([], dtype=np.int32)
            for idx in same_hash_indices:
                hash_origin_ids = _coverage_union(hash_origin_ids, bundle["origin_ids"][int(idx)])
            for idx in same_hash_indices:
                coverage_by_group[int(idx)] = _coverage_union(coverage_by_group[int(idx)], hash_origin_ids)
            for left in range(len(same_hash_indices) - 1):
                source_idx = int(same_hash_indices[left])
                source_candidate = bundle["representatives"][source_idx]
                for right in range(left + 1, len(same_hash_indices)):
                    target_idx = int(same_hash_indices[right])
                    target_candidate = bundle["representatives"][target_idx]
                    self._record_exact_hash_match(source_candidate, target_candidate)

    def _coverage_fill_bin(self, area_px: int, clip_pixels: int) -> int:
        """按 fill ratio 计算 coverage 低内存桶编号。"""

        fill_ratio = float(area_px) / max(float(clip_pixels), 1.0)
        return int(math.floor(max(fill_ratio, 0.0) / max(float(COVERAGE_FILL_BIN_WIDTH), 1e-12)))

    def _build_bundle_fill_bins(self, bundle: Dict[str, Any]) -> Dict[int, np.ndarray]:
        """把一个超大 shape bundle 切成 fill-bin source 桶。"""

        started = time.perf_counter()
        clip_pixels = int(bundle["clip_pixels"])
        temp_bins: Dict[int, List[int]] = {}
        for group_idx, area_px in enumerate(np.asarray(bundle["areas"], dtype=np.int64).tolist()):
            fill_bin = self._coverage_fill_bin(int(area_px), clip_pixels)
            temp_bins.setdefault(fill_bin, []).append(int(group_idx))
        fill_bins = {
            int(fill_bin): np.asarray(indices, dtype=np.int32)
            for fill_bin, indices in sorted(temp_bins.items())
        }
        self.coverage_detail_seconds["bucket_index_build"] += time.perf_counter() - started
        self.coverage_debug_stats["coverage_fill_bin_count"] += int(len(fill_bins))
        self.coverage_debug_stats["max_fill_bin_group_count"] = max(
            int(self.coverage_debug_stats.get("max_fill_bin_group_count", 0)),
            max((int(indices.size) for indices in fill_bins.values()), default=0),
        )
        return fill_bins

    def _bundle_window(self, bundle: Dict[str, Any], global_indices: np.ndarray) -> Dict[str, Any]:
        """为 bucketed coverage 构造只含局部 group 的临时 bundle。"""

        indices = np.asarray(global_indices, dtype=np.int32)
        hashes = [bundle["hashes"][int(idx)] for idx in indices.tolist()]
        return {
            "areas": np.asarray(bundle["areas"], dtype=np.int64)[indices],
            "hashes": hashes,
            "hashes_np": np.asarray(hashes),
            "origin_ids": [bundle["origin_ids"][int(idx)] for idx in indices.tolist()],
            "candidate_groups": [bundle["candidate_groups"][int(idx)] for idx in indices.tolist()],
            "representatives": [bundle["representatives"][int(idx)] for idx in indices.tolist()],
            "clip_pixels": int(bundle["clip_pixels"]),
            "bitmap_cache_by_idx": {},
            "geometry_cache_by_idx": {},
            "full_descriptor_cache_by_idx": {},
            "full_prefilter_disabled": False,
            "full_prefilter_probe_pairs": 0,
            "full_prefilter_probe_rejects": 0,
            "full_prefilter_probe_done": False,
            "global_indices": indices,
        }

    def _bundle_group_bitmap(self, bundle: Dict[str, Any], group_idx: int, *, persist: bool) -> np.ndarray:
        """按需恢复 bundle group 的 bitmap；可选挂入当前 window 局部缓存。"""

        idx = int(group_idx)
        if persist:
            cache_by_idx = bundle.setdefault("bitmap_cache_by_idx", {})
            cached = cache_by_idx.get(idx)
            if cached is not None:
                return np.ascontiguousarray(np.asarray(cached, dtype=bool))
        candidate_group = bundle["candidate_groups"][idx]
        representative = bundle["representatives"][idx]
        bitmap = getattr(representative, "clip_bitmap", None)
        if bitmap is None:
            bitmap = _unpack_bitmap_payload(
                np.asarray(candidate_group.packed_clip_bitmap, dtype=np.uint8),
                tuple(int(value) for value in candidate_group.clip_bitmap_shape),
            )
            self.memory_debug["unpacked_candidate_group_bitmap_count"] += 1
        bitmap = np.ascontiguousarray(np.asarray(bitmap, dtype=bool))
        if persist:
            cache_by_idx[idx] = bitmap
            self.coverage_debug_stats["window_bitmap_live_peak_count"] = max(
                int(self.coverage_debug_stats.get("window_bitmap_live_peak_count", 0)),
                int(len(cache_by_idx)),
            )
        return bitmap

    def _release_bundle_bitmap_cache(self, bundle: Dict[str, Any]) -> None:
        """释放当前 coverage window 的局部 unpack bitmap 缓存。"""

        cache_by_idx = bundle.get("bitmap_cache_by_idx")
        if isinstance(cache_by_idx, dict) and cache_by_idx:
            cache_by_idx.clear()

    def _release_shortlist_payloads(self, shortlist_index: Dict[str, Any]) -> None:
        """释放 window 内仍未自动释放的 shortlist payload。"""

        payloads = shortlist_index.get("subgroup_payloads")
        if not isinstance(payloads, dict) or not payloads:
            return
        started = time.perf_counter()
        released = int(len(payloads))
        live_count = int(
            sum(int(payload.get("embedding_group_count", 0)) for payload in payloads.values() if isinstance(payload, dict))
        )
        payloads.clear()
        shortlist_index["live_signature_embedding_groups"] = max(
            0,
            int(shortlist_index.get("live_signature_embedding_groups", 0)) - live_count,
        )
        self.coverage_detail_seconds["shortlist_payload_release"] += time.perf_counter() - started
        self.coverage_debug_stats["shortlist_payload_release_count"] += released

    def _record_pair_tracker_mode(self, compared_pairs: Dict[str, Any]) -> None:
        """把当前 coverage window 的 pair tracker 模式并入统计。"""

        tracker_mode = str(compared_pairs.get("mode", "unset"))
        previous_mode = str(self.coverage_debug_stats.get("pair_tracker_mode", "unset"))
        self.coverage_debug_stats["pair_tracker_mode"] = (
            tracker_mode if previous_mode in {"unset", tracker_mode} else "mixed"
        )
        if tracker_mode == "source_unique":
            self.coverage_debug_stats["pair_tracker_disabled_bundle_count"] += 1

    def _process_coverage_window(
        self,
        bundle: Dict[str, Any],
        coverage_by_group: List[np.ndarray],
        source_local_indices: np.ndarray | None,
        tol_px: int,
        ratio_limit: float,
        *,
        force_source_unique: bool,
        detail_key: str,
    ) -> None:
        """处理一个完整 bundle 或 fill-bin window 的 shortlist/prefilter/geometry。"""

        shortlist_started = time.perf_counter()
        shortlist_index = self._build_bundle_shortlist_index(bundle)
        self.coverage_detail_seconds[detail_key] += time.perf_counter() - shortlist_started
        group_count = len(bundle["candidate_groups"])
        if source_local_indices is None:
            source_indices = np.arange(max(0, group_count - 1), dtype=np.int32)
        else:
            source_indices = np.asarray(source_local_indices, dtype=np.int32)
        local_to_global = np.asarray(
            bundle.get("global_indices", np.arange(group_count, dtype=np.int32)),
            dtype=np.int32,
        )

        compared_pairs = _pair_tracker(group_count, force_source_unique=force_source_unique)
        self._record_pair_tracker_mode(compared_pairs)
        try:
            for source_idx in source_indices.tolist():
                source_idx = int(source_idx)
                source_candidate = bundle["representatives"][source_idx]
                target_indices = self._shortlist_target_indices(bundle, shortlist_index, source_idx)
                if target_indices.size == 0:
                    continue

                target_indices = target_indices[bundle["hashes_np"][target_indices] != bundle["hashes_np"][source_idx]]
                if target_indices.size == 0:
                    continue

                fresh_targets: List[int] = []
                for target_idx in target_indices.tolist():
                    if _pair_tracker_test_and_set(compared_pairs, int(source_idx), int(target_idx)):
                        continue
                    fresh_targets.append(int(target_idx))
                if not fresh_targets:
                    continue
                target_indices = np.asarray(fresh_targets, dtype=np.int32)

                prefilter_started = time.perf_counter()
                target_indices = self._batch_prefilter(bundle, shortlist_index, source_idx, target_indices)
                self.coverage_detail_seconds["prefilter"] += time.perf_counter() - prefilter_started
                self._release_bundle_full_descriptor_cache(bundle)
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
                        try:
                            target_packed = self._bundle_geometry_matrix(bundle, target_chunk, tol_px, "packed")
                            geometry_started = time.perf_counter()
                            xor_rows = _bitcount_sum_rows(
                                np.bitwise_xor(target_packed, source_packed[None, :])
                            )
                            matched_chunks.append(target_chunk[(xor_rows / clip_pixels) <= ratio_limit])
                            self.coverage_detail_seconds["geometry_match"] += time.perf_counter() - geometry_started
                        finally:
                            self._release_bundle_geometry_cache(bundle, target_chunk)
                else:
                    if tol_px <= 0:
                        for target_chunk in _chunk_indices_by_row_width(target_indices, packed_row_bytes):
                            try:
                                target_packed = self._bundle_geometry_matrix(bundle, target_chunk, tol_px, "packed")
                                geometry_started = time.perf_counter()
                                exact_equal = np.all(
                                    target_packed == source_packed[None, :],
                                    axis=1,
                                )
                                matched_chunks.append(target_chunk[exact_equal])
                                self.coverage_detail_seconds["geometry_match"] += time.perf_counter() - geometry_started
                            finally:
                                self._release_bundle_geometry_cache(bundle, target_chunk)
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
                            try:
                                matched_chunk = self._ecc_positive_tolerance_chunk_matches(
                                    bundle,
                                    int(source_idx),
                                    target_chunk,
                                    tol_px,
                                    source_area,
                                    source_area_limit,
                                    source_dilated_area,
                                    source_packed,
                                    source_packed_dilated,
                                )
                                if matched_chunk.size:
                                    matched_chunks.append(matched_chunk)
                            except Exception:
                                self._release_bundle_geometry_cache(bundle, [source_idx])
                                raise
                            finally:
                                self._release_bundle_geometry_cache(bundle, target_chunk)

                self._release_bundle_geometry_cache(bundle, [source_idx])
                non_empty_chunks = [chunk for chunk in matched_chunks if chunk.size]
                matched_indices = (
                    np.concatenate(non_empty_chunks, axis=0).astype(np.int32, copy=False)
                    if non_empty_chunks
                    else np.asarray([], dtype=np.int32)
                )

                matched_set = {int(target_idx) for target_idx in matched_indices.tolist()}
                for target_idx in target_indices.tolist():
                    self._record_geometry_result(
                        source_candidate,
                        bundle["representatives"][int(target_idx)],
                        int(target_idx) in matched_set,
                    )
                source_global_idx = int(local_to_global[source_idx])
                for target_idx in matched_indices:
                    target_idx = int(target_idx)
                    target_global_idx = int(local_to_global[target_idx])
                    coverage_by_group[source_global_idx] = _coverage_union(
                        coverage_by_group[source_global_idx],
                        bundle["origin_ids"][target_idx],
                    )
                    coverage_by_group[target_global_idx] = _coverage_union(
                        coverage_by_group[target_global_idx],
                        bundle["origin_ids"][source_idx],
                    )
        finally:
            release_started = time.perf_counter()
            self._release_shortlist_payloads(shortlist_index)
            self._release_bundle_full_descriptor_cache(bundle)
            self._release_bundle_geometry_cache(bundle)
            self._release_bundle_bitmap_cache(bundle)
            if str(compared_pairs.get("mode")) == "rows":
                self.coverage_debug_stats["pair_tracker_row_count"] += int(len(compared_pairs.get("rows", {})))
            self.coverage_debug_stats["geometry_cache_live_after_bundle_count"] = max(
                int(self.coverage_debug_stats.get("geometry_cache_live_after_bundle_count", 0)),
                int(len(bundle.get("geometry_cache_by_idx", {}))),
            )
            if detail_key == "bucket_window_index":
                self.coverage_detail_seconds["bucket_window_release"] += time.perf_counter() - release_started

    def _evaluate_bucketed_bundle_coverage(
        self,
        bundle: Dict[str, Any],
        coverage_by_group: List[np.ndarray],
        tol_px: int,
        ratio_limit: float,
    ) -> None:
        """对超大 shape bundle 按 fill-bin window 顺序构建 coverage。"""

        fill_bins = self._build_bundle_fill_bins(bundle)
        if not fill_bins:
            return
        self.coverage_debug_stats["bucketed_coverage_bundle_count"] += 1
        radius = int(math.ceil(float(CHEAP_FILL_ABS_LIMIT) / max(float(COVERAGE_FILL_BIN_WIDTH), 1e-12))) + 1
        for fill_bin, source_global_indices in fill_bins.items():
            neighbor_arrays = [
                fill_bins[int(candidate_bin)]
                for candidate_bin in range(int(fill_bin) - radius, int(fill_bin) + radius + 1)
                if int(candidate_bin) in fill_bins
            ]
            if neighbor_arrays:
                window_global_indices = np.unique(np.concatenate(neighbor_arrays, axis=0)).astype(np.int32, copy=False)
            else:
                window_global_indices = np.asarray(source_global_indices, dtype=np.int32)
            self.coverage_debug_stats["bucketed_source_group_count"] += int(source_global_indices.size)
            self.coverage_debug_stats["bucketed_target_group_count"] += int(window_global_indices.size)
            self.coverage_debug_stats["max_bucket_window_group_count"] = max(
                int(self.coverage_debug_stats.get("max_bucket_window_group_count", 0)),
                int(window_global_indices.size),
            )
            window_bundle = self._bundle_window(bundle, window_global_indices)
            source_local_indices = np.searchsorted(window_global_indices, np.asarray(source_global_indices, dtype=np.int32))
            self._process_coverage_window(
                window_bundle,
                coverage_by_group,
                source_local_indices.astype(np.int32, copy=False),
                tol_px,
                ratio_limit,
                force_source_unique=True,
                detail_key="bucket_window_index",
            )
            del window_bundle

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
            cached = _init_coverage_geometry_cache(self._bundle_group_bitmap(bundle, idx, persist=True))
            cache_by_idx[idx] = cached
            self.coverage_detail_seconds["geometry_cache"] += time.perf_counter() - started
            self.coverage_debug_stats["geometry_cache_group_count"] += 1
            self._note_bundle_geometry_cache_live(bundle)

        if level in {"dilated", "donut"} and int(tol_px) > 0 and "packed_dilated" not in cached:
            started = time.perf_counter()
            _extend_coverage_dilated_cache(cached, int(tol_px), self._bundle_group_bitmap(bundle, idx, persist=True))
            self.coverage_detail_seconds["geometry_cache"] += time.perf_counter() - started
            self.coverage_debug_stats["geometry_dilated_cache_group_count"] += 1

        if level == "donut" and int(tol_px) > 0 and "packed_donut" not in cached:
            started = time.perf_counter()
            _extend_coverage_donut_cache(cached, int(tol_px), self._bundle_group_bitmap(bundle, idx, persist=True))
            self.coverage_detail_seconds["geometry_cache"] += time.perf_counter() - started
            self.coverage_debug_stats["geometry_donut_cache_group_count"] += 1
        return cached

    def _note_bundle_geometry_cache_live(self, bundle: Dict[str, Any]) -> None:
        """记录当前 bundle 几何缓存的 live 峰值。"""

        live_count = int(len(bundle.get("geometry_cache_by_idx", {})))
        self.coverage_debug_stats["geometry_cache_live_peak_count"] = max(
            int(self.coverage_debug_stats.get("geometry_cache_live_peak_count", 0)),
            live_count,
        )

    def _release_bundle_geometry_cache(self, bundle: Dict[str, Any], indices: Iterable[int] | None = None) -> int:
        """释放 bundle 内指定 group 的 coverage 几何缓存。"""

        cache_by_idx = bundle.get("geometry_cache_by_idx")
        if not isinstance(cache_by_idx, dict) or not cache_by_idx:
            return 0
        started = time.perf_counter()
        if indices is None:
            released = int(len(cache_by_idx))
            cache_by_idx.clear()
        else:
            released = 0
            for idx in {int(value) for value in indices}:
                if idx in cache_by_idx:
                    del cache_by_idx[idx]
                    released += 1
        if released:
            self.coverage_detail_seconds["geometry_cache_release"] += time.perf_counter() - started
            self.coverage_debug_stats["geometry_cache_release_count"] += int(released)
        return released

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

    def _bundle_full_descriptor(self, bundle: Dict[str, Any], group_idx: int) -> GraphDescriptor:
        """按需获取 coverage full prefilter 使用的图描述符。"""

        cache_by_idx = bundle.setdefault("full_descriptor_cache_by_idx", {})
        idx = int(group_idx)
        cached = cache_by_idx.get(idx)
        if cached is None:
            started = time.perf_counter()
            bitmap = self._bundle_group_bitmap(bundle, idx, persist=True)
            cached = _bitmap_descriptor(bitmap)
            cache_by_idx[idx] = cached
            self.coverage_detail_seconds["full_descriptor_cache"] += time.perf_counter() - started
            self.coverage_debug_stats["full_descriptor_cache_group_count"] += 1
        return cached

    def _release_bundle_full_descriptor_cache(self, bundle: Dict[str, Any]) -> None:
        """释放当前 bundle 的 full descriptor cache，避免大 bundle 长期常驻。"""

        cache_by_idx = bundle.get("full_descriptor_cache_by_idx")
        if isinstance(cache_by_idx, dict) and cache_by_idx:
            cache_by_idx.clear()

    def _build_bundle_shortlist_index(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        """为一个 bitmap-shape bundle 预构建 subgroup 级懒加载 shortlist 索引。"""

        group_count = int(len(bundle["representatives"]))
        cheap_invariants = np.empty((group_count, 8), dtype=np.float32)
        subgroup_members: Dict[Tuple[int, int, int, int, int], np.ndarray] = {}
        subgroup_remaining: Dict[Tuple[int, int, int, int, int], int] = {}
        source_subgroup_ids = np.empty(group_count, dtype=np.int32)
        subgroup_key_to_id: Dict[Tuple[int, int, int, int, int], int] = {}
        subgroup_keys: List[Tuple[int, int, int, int, int]] = []
        temp_subgroups: Dict[Tuple[int, int, int, int, int], List[int]] = {}
        for idx, representative in enumerate(bundle["representatives"]):
            del representative
            bitmap = self._bundle_group_bitmap(bundle, int(idx), persist=False)
            desc = _cheap_bitmap_descriptor(bitmap)
            cheap_invariants[int(idx)] = np.asarray(desc.invariants, dtype=np.float32)
            subgroup_key = _coverage_cheap_subgroup_key(desc)
            subgroup_id = subgroup_key_to_id.get(subgroup_key)
            if subgroup_id is None:
                subgroup_id = int(len(subgroup_keys))
                subgroup_key_to_id[subgroup_key] = subgroup_id
                subgroup_keys.append(subgroup_key)
            source_subgroup_ids[int(idx)] = int(subgroup_id)
            temp_subgroups.setdefault(subgroup_key, []).append(int(idx))

        self.coverage_debug_stats["shortlist_subgroup_count"] += int(len(temp_subgroups))
        self.coverage_debug_stats["shortlist_max_subgroup_size"] = max(
            int(self.coverage_debug_stats.get("shortlist_max_subgroup_size", 0)),
            max((len(indices) for indices in temp_subgroups.values()), default=0),
        )
        self.memory_debug["signature_embedding_bytes_avoided_estimate_mb"] = max(
            float(self.memory_debug.get("signature_embedding_bytes_avoided_estimate_mb", 0.0)),
            float(group_count * 120 * np.dtype(np.float32).itemsize) / (1024.0 * 1024.0),
        )
        for subgroup_key, indices in temp_subgroups.items():
            group_indices = np.asarray(indices, dtype=np.int32)
            subgroup_members[subgroup_key] = group_indices
            subgroup_remaining[subgroup_key] = int(group_indices.size)

        return {
            "cheap_invariants": cheap_invariants,
            "bundle": bundle,
            "source_subgroup_ids": source_subgroup_ids,
            "subgroup_keys": subgroup_keys,
            "subgroup_members": subgroup_members,
            "subgroup_remaining": subgroup_remaining,
            "subgroup_payloads": {},
            "live_signature_embedding_groups": 0,
        }

    def _ensure_shortlist_payload(
        self,
        shortlist_index: Dict[str, Any],
        subgroup_key: Tuple[int, int, int, int, int],
    ) -> Dict[str, Any]:
        """按需构建某个 subgroup 的 shortlist payload。"""

        payloads = shortlist_index["subgroup_payloads"]
        payload = payloads.get(subgroup_key)
        if payload is not None:
            return payload

        started = time.perf_counter()
        group_indices = np.asarray(shortlist_index["subgroup_members"][subgroup_key], dtype=np.int32)
        k = min(int(COVERAGE_SHORTLIST_MAX_TARGETS) + 1, int(group_indices.size))
        if group_indices.size <= 1:
            mapped_labels = group_indices.reshape(-1, 1).astype(np.int32, copy=False)
        elif group_indices.size <= k:
            mapped_labels = np.tile(group_indices[None, :], (int(group_indices.size), 1)).astype(np.int32, copy=False)
        else:
            group_vectors = np.empty((int(group_indices.size), 120), dtype=np.float32)
            for local_idx, group_idx in enumerate(group_indices.tolist()):
                bitmap = self._bundle_group_bitmap(shortlist_index["bundle"], int(group_idx), persist=False)
                group_vectors[int(local_idx)] = _signature_embedding(_cheap_bitmap_descriptor(bitmap))
            if int(group_indices.size) <= int(COVERAGE_EXACT_SHORTLIST_MAX_GROUPS):
                self.coverage_debug_stats["shortlist_exact_subgroup_count"] += 1
                local_labels = _exact_cosine_topk_labels(group_vectors, k).astype(np.int32, copy=False)
            else:
                self.coverage_debug_stats["shortlist_hnsw_subgroup_count"] += 1
                index = hnswlib.Index(space="cosine", dim=int(group_vectors.shape[1]))
                index.init_index(max_elements=int(group_indices.size), ef_construction=max(64, k * 2), M=12)
                index.add_items(group_vectors, np.arange(int(group_indices.size), dtype=np.int32))
                index.set_ef(max(64, k * 2))
                local_labels, _ = index.knn_query(group_vectors, k=k)
                local_labels = np.asarray(local_labels, dtype=np.int32)
            mapped_labels = group_indices[local_labels].astype(np.int32, copy=False)

        payload = {
            "group_indices": group_indices,
            "mapped_labels": mapped_labels,
            "embedding_group_count": int(group_indices.size),
        }
        payloads[subgroup_key] = payload
        shortlist_index["live_signature_embedding_groups"] = int(shortlist_index.get("live_signature_embedding_groups", 0)) + int(
            group_indices.size
        )
        self.coverage_detail_seconds["shortlist_payload_build"] += time.perf_counter() - started
        self.coverage_debug_stats["shortlist_payload_peak_count"] = max(
            int(self.coverage_debug_stats.get("shortlist_payload_peak_count", 0)),
            int(len(payloads)),
        )
        self.coverage_debug_stats["lazy_signature_embedding_group_count"] += int(group_indices.size)
        self.coverage_debug_stats["signature_embedding_live_peak_count"] = max(
            int(self.coverage_debug_stats.get("signature_embedding_live_peak_count", 0)),
            int(shortlist_index.get("live_signature_embedding_groups", 0)),
        )
        return payload

    def _release_shortlist_payload(
        self,
        shortlist_index: Dict[str, Any],
        subgroup_key: Tuple[int, int, int, int, int],
    ) -> None:
        """当 subgroup 全部 source 处理完后释放其 shortlist payload。"""

        payloads = shortlist_index["subgroup_payloads"]
        if subgroup_key not in payloads:
            return
        started = time.perf_counter()
        payload = payloads[subgroup_key]
        shortlist_index["live_signature_embedding_groups"] = max(
            0,
            int(shortlist_index.get("live_signature_embedding_groups", 0)) - int(payload.get("embedding_group_count", 0)),
        )
        del payloads[subgroup_key]
        self.coverage_detail_seconds["shortlist_payload_release"] += time.perf_counter() - started
        self.coverage_debug_stats["shortlist_payload_release_count"] += 1

    def _shortlist_target_indices(
        self,
        bundle: Dict[str, Any],
        shortlist_index: Dict[str, Any],
        source_idx: int,
    ) -> np.ndarray:
        """返回某个 source group 的 ANN shortlist 目标组索引。"""

        del bundle
        subgroup_id = int(np.asarray(shortlist_index["source_subgroup_ids"], dtype=np.int32)[int(source_idx)])
        subgroup_key = shortlist_index["subgroup_keys"][subgroup_id]
        payload = self._ensure_shortlist_payload(shortlist_index, subgroup_key)
        group_indices = np.asarray(payload["group_indices"], dtype=np.int32)
        local_idx = int(np.searchsorted(group_indices, int(source_idx)))
        assert local_idx < int(group_indices.size) and int(group_indices[local_idx]) == int(source_idx)
        labels = np.asarray(payload["mapped_labels"][local_idx], dtype=np.int32).copy()
        remaining = int(shortlist_index["subgroup_remaining"][subgroup_key]) - 1
        shortlist_index["subgroup_remaining"][subgroup_key] = remaining
        if remaining <= 0:
            self._release_shortlist_payload(shortlist_index, subgroup_key)
        if labels.size == 0:
            return np.asarray([], dtype=np.int32)
        return labels[(labels >= 0) & (labels != int(source_idx))]

    def _batch_prefilter(
        self,
        bundle: Dict[str, Any],
        shortlist_index: Dict[str, Any],
        source_idx: int,
        target_indices: np.ndarray,
    ) -> np.ndarray:
        """对一个 source 的 shortlist targets 执行 cheap 与懒加载 full prefilter。"""

        if target_indices.size == 0:
            return target_indices

        cheap_invariants = np.asarray(shortlist_index["cheap_invariants"], dtype=np.float32)
        source_cheap = cheap_invariants[int(source_idx)]
        target_cheap = cheap_invariants[np.asarray(target_indices, dtype=np.int32)]
        cheap_floors = np.asarray([0.02, 0.03, 0.03], dtype=np.float32)
        cheap_source = source_cheap[[1, 4, 5]]
        cheap_target = target_cheap[:, [1, 4, 5]]
        cheap_denom = np.maximum(np.maximum(np.abs(cheap_source)[None, :], np.abs(cheap_target)), cheap_floors[None, :])
        cheap_errs = np.abs(cheap_target - cheap_source[None, :]) / cheap_denom
        cheap_ratio_ok = np.all(cheap_errs <= 0.45, axis=1)
        fill_ok = np.abs(target_cheap[:, 1] - source_cheap[1]) <= CHEAP_FILL_ABS_LIMIT
        density_ok = np.abs(target_cheap[:, 5] - source_cheap[5]) <= CHEAP_AREA_DENSITY_ABS_LIMIT
        cheap_ok = cheap_ratio_ok & fill_ok & density_ok
        self.prefilter_stats["cheap_fill_reject"] += int(np.count_nonzero(cheap_ratio_ok & ~fill_ok))
        self.prefilter_stats["cheap_area_density_reject"] += int(np.count_nonzero(cheap_ratio_ok & fill_ok & ~density_ok))
        self.prefilter_stats["cheap_reject"] += int(np.count_nonzero(~cheap_ok))
        target_indices = target_indices[cheap_ok]
        if target_indices.size == 0:
            return target_indices

        if bool(bundle.get("full_prefilter_disabled", False)):
            return target_indices

        full_started = time.perf_counter()
        full_input_count = int(target_indices.size)
        source_desc = self._bundle_full_descriptor(bundle, int(source_idx))
        target_descs = [self._bundle_full_descriptor(bundle, int(idx)) for idx in np.asarray(target_indices, dtype=np.int32).tolist()]

        source_inv = np.asarray(source_desc.invariants, dtype=np.float64)
        target_inv = np.asarray([desc.invariants for desc in target_descs], dtype=np.float64)
        inv_floors = np.asarray([0.25, 0.02, 0.02, 0.02, 0.03, 0.03, 0.02, 0.02], dtype=np.float64)
        inv_weights = np.asarray([0.08, 0.24, 0.10, 0.08, 0.18, 0.14, 0.10, 0.08], dtype=np.float64)
        inv_denom = np.maximum(np.maximum(np.abs(source_inv)[None, :], np.abs(target_inv)), inv_floors[None, :])
        inv_errs = np.minimum(np.abs(target_inv - source_inv[None, :]) / inv_denom, 1.0)
        critical = (inv_errs[:, 1] > 0.45) | (inv_errs[:, 4] > 0.45) | (inv_errs[:, 5] > 0.45)
        invariant_ok = (~critical) & ((inv_errs @ inv_weights) <= GRAPH_INVARIANT_LIMIT)
        self.prefilter_stats["invariant_reject"] += int(np.count_nonzero(~invariant_ok))
        if not np.all(invariant_ok):
            keep = invariant_ok.tolist()
            target_indices = target_indices[invariant_ok]
            target_descs = [desc for desc, ok in zip(target_descs, keep) if ok]

        if target_indices.size:
            source_topology = np.asarray(source_desc.topology, dtype=np.float64)
            target_topology = np.asarray([desc.topology for desc in target_descs], dtype=np.float64)
            topology_dist = np.linalg.norm(target_topology - source_topology[None, :], axis=1)
            topology_ok = topology_dist <= GRAPH_TOPOLOGY_THRESHOLD
            self.prefilter_stats["topology_reject"] += int(np.count_nonzero(~topology_ok))
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
            signature_sim = (
                0.6 * (target_grid @ source_grid)
                + 0.2 * (target_proj_x @ source_proj_x)
                + 0.2 * (target_proj_y @ source_proj_y)
            )
            signature_ok = signature_sim >= GRAPH_SIGNATURE_THRESHOLD
            self.prefilter_stats["signature_reject"] += int(np.count_nonzero(~signature_ok))
            target_indices = target_indices[signature_ok]

        full_reject_count = full_input_count - int(target_indices.size)
        self.prefilter_stats["full_prefilter_reject"] += full_reject_count
        if not bool(bundle.get("full_prefilter_probe_done", False)):
            probe_pairs = int(bundle.get("full_prefilter_probe_pairs", 0)) + full_input_count
            probe_rejects = int(bundle.get("full_prefilter_probe_rejects", 0)) + full_reject_count
            bundle["full_prefilter_probe_pairs"] = probe_pairs
            bundle["full_prefilter_probe_rejects"] = probe_rejects
            self.coverage_debug_stats["full_prefilter_probe_pair_count"] += full_input_count
            self.coverage_debug_stats["full_prefilter_probe_reject_count"] += full_reject_count
            if probe_pairs >= int(COVERAGE_FULL_PREFILTER_MIN_PROBE_PAIRS):
                bundle["full_prefilter_probe_done"] = True
                reject_rate = float(probe_rejects) / float(max(probe_pairs, 1))
                if reject_rate < float(COVERAGE_FULL_PREFILTER_MIN_REJECT_RATE):
                    bundle["full_prefilter_disabled"] = True
                    self.coverage_debug_stats["full_prefilter_disabled_bundle_count"] += 1
        self.coverage_detail_seconds["full_prefilter"] += time.perf_counter() - full_started
        return target_indices

    def _ecc_positive_tolerance_chunk_matches(
        self,
        bundle: Dict[str, Any],
        source_idx: int,
        target_chunk: np.ndarray,
        tol_px: int,
        source_area: float,
        source_area_limit: float,
        source_dilated_area: int,
        source_packed: np.ndarray,
        source_packed_dilated: np.ndarray,
    ) -> np.ndarray:
        """处理一个 ECC 正容差 target chunk，并把释放职责交给调用方 finally。"""

        geometry_started = time.perf_counter()
        target_areas = bundle["areas"][target_chunk].astype(np.float64)
        target_area_limits = ECC_RESIDUAL_RATIO * np.maximum(target_areas, 1.0)
        area_candidate_indices = target_chunk[
            target_areas <= float(source_dilated_area) + target_area_limits
        ]
        self.coverage_detail_seconds["geometry_match"] += time.perf_counter() - geometry_started
        if not area_candidate_indices.size:
            return np.asarray([], dtype=np.int32)

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
            return np.asarray([], dtype=np.int32)

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
        matched_chunks: List[np.ndarray] = []
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

        non_empty = [chunk for chunk in matched_chunks if chunk.size]
        if not non_empty:
            return np.asarray([], dtype=np.int32)
        return np.concatenate(non_empty, axis=0).astype(np.int32, copy=False)

    def _evaluate_candidate_coverage(
        self,
        candidates: Sequence[CandidateClip] | Sequence[CoverageCandidateGroup],
        exact_clusters: Sequence[ExactCluster],
    ) -> None:
        """用 grid-aware shortlist + ACC/ECC 构建 candidate 覆盖关系。"""

        del exact_clusters
        tol_px = max(0, int(math.ceil(float(self.edge_tolerance_um) / max(float(self.pixel_size_um), 1e-12) - 1e-12)))
        self.coverage_detail_seconds = _empty_coverage_detail_seconds()
        self.coverage_debug_stats = _empty_coverage_debug_stats()

        light_bundle_started = time.perf_counter()
        bundles = self._build_candidate_match_bundles(candidates, tol_px)
        self._release_candidate_bitmap_pool()
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

        ratio_limit = max(0.0, 1.0 - float(self.area_match_ratio))
        for bundle in bundles.values():
            group_count = len(bundle["candidate_groups"])
            coverage_by_group = self._initial_bundle_coverage(bundle)
            self._apply_same_hash_coverage(bundle, coverage_by_group)
            if group_count > int(COVERAGE_BUCKETED_GROUP_THRESHOLD):
                self._evaluate_bucketed_bundle_coverage(bundle, coverage_by_group, tol_px, ratio_limit)
            else:
                self._process_coverage_window(
                    bundle,
                    coverage_by_group,
                    None,
                    tol_px,
                    ratio_limit,
                    force_source_unique=False,
                    detail_key="shortlist_index",
                )
            for group_idx, candidate_group in enumerate(bundle["candidate_groups"]):
                group_coverage_tuple = tuple(int(value) for value in coverage_by_group[group_idx].tolist())
                candidate_group.coverage = group_coverage_tuple
                candidate_group.best_candidate.coverage = group_coverage_tuple
                for candidate in candidate_group.materialized_candidates:
                    candidate.coverage = group_coverage_tuple

        self._log(
            "coverage 几何统计: "
            f"pair={self.coverage_debug_stats['geometry_pair_count']}, "
            f"full desc={self.coverage_debug_stats['full_descriptor_cache_group_count']}, "
            f"cache group={self.coverage_debug_stats['geometry_cache_group_count']}, "
            f"dilated={self.coverage_debug_stats['geometry_dilated_cache_group_count']}, "
            f"donut={self.coverage_debug_stats['geometry_donut_cache_group_count']}, "
            f"live peak={self.coverage_debug_stats['geometry_cache_live_peak_count']}, "
            f"release={self.coverage_debug_stats['geometry_cache_release_count']}, "
            f"after bundle={self.coverage_debug_stats['geometry_cache_live_after_bundle_count']}, "
            f"bucketed={self.coverage_debug_stats['bucketed_coverage_bundle_count']}, "
            f"fill bins={self.coverage_debug_stats['coverage_fill_bin_count']}, "
            f"max fill bin={self.coverage_debug_stats['max_fill_bin_group_count']}, "
            f"max bucket window={self.coverage_debug_stats['max_bucket_window_group_count']}, "
            f"detail={{{', '.join(f'{k}: {round(v, 3)}' for k, v in self.coverage_detail_seconds.items())}}}"
        )

    def _greedy_cover(
        self,
        candidates: Sequence[CandidateClip] | Sequence[CoverageCandidateGroup],
        exact_clusters: Sequence[ExactCluster],
    ) -> List[CandidateClip]:
        """按 coverage、cluster 数与 shift 代价执行 lazy-heap greedy set cover。"""

        candidate_groups = self._coerce_coverage_candidate_groups(candidates)
        uncovered = {int(cluster.exact_cluster_id) for cluster in exact_clusters}
        weights = {int(cluster.exact_cluster_id): int(cluster.weight) for cluster in exact_clusters}
        selected: List[CandidateClip] = []
        selected_ids: set[str] = set()
        group_by_id = {group.best_candidate.candidate_id: group for group in candidate_groups}
        heap: List[Tuple[Tuple[Any, ...], str]] = []

        def _priority(candidate_group: CoverageCandidateGroup) -> Tuple[Any, ...]:
            candidate = candidate_group.best_candidate
            covered_now = _coverage_overlap(candidate_group.coverage, uncovered)
            return (
                -sum(weights[cid] for cid in covered_now),
                -len(covered_now),
                -1 if candidate.shift_direction == "base" else 0,
                abs(candidate.shift_distance_um),
                int(candidate.origin_exact_cluster_id),
                candidate.candidate_id,
            )

        for candidate_group in candidate_groups:
            candidate = candidate_group.best_candidate
            heapq.heappush(heap, (_priority(candidate_group), candidate.candidate_id))

        while uncovered:
            best: CandidateClip | None = None
            covered_now: Tuple[int, ...] = ()
            while heap:
                saved_priority, candidate_id = heapq.heappop(heap)
                if candidate_id in selected_ids:
                    continue
                candidate_group = group_by_id[candidate_id]
                candidate = candidate_group.best_candidate
                current_priority = _priority(candidate_group)
                if current_priority != saved_priority:
                    heapq.heappush(heap, (current_priority, candidate_id))
                    continue
                current_covered = _coverage_overlap(candidate_group.coverage, uncovered)
                if current_covered:
                    best = candidate
                    covered_now = current_covered
                    break
            if best is None:
                missing = min(uncovered)
                best = self._base_candidate_by_exact_id[missing]
                covered_now = _coverage_overlap(best.coverage, uncovered)
                if not covered_now:
                    covered_now = (missing,)
            selected.append(best)
            selected_ids.add(best.candidate_id)
            uncovered.difference_update(int(value) for value in covered_now)
        return selected

    def _geometry_passes(self, candidate: CandidateClip, target: MarkerRecord) -> bool:
        """根据当前模式执行 ACC 或 ECC 最终几何判定。"""

        candidate_bitmap = _candidate_clip_bitmap(candidate)
        target_bitmap = getattr(target, "clip_bitmap", None)
        assert target_bitmap is not None, "最终几何匹配时 target.clip_bitmap 不应为空"
        if candidate_bitmap.shape != target_bitmap.shape:
            return False
        if self.matching_mode == "acc":
            xor_ratio = float(np.count_nonzero(candidate_bitmap ^ target_bitmap)) / float(
                max(candidate_bitmap.size, 1)
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
        selected_info = dict(record.match_cache.get("selected_candidate_info", {}) or {})
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
            "selected_candidate_id": selected_info.get("selected_candidate_id"),
            "selected_shift_direction": selected_info.get("selected_shift_direction"),
            "selected_shift_distance_um": selected_info.get("selected_shift_distance_um"),
            "seed_weight": int(record.seed_weight),
            "seed_bbox": auto_seed.get("seed_bbox"),
            "grid_ix": auto_seed.get("grid_ix"),
            "grid_iy": auto_seed.get("grid_iy"),
            "grid_cell_bbox": auto_seed.get("grid_cell_bbox"),
            "seed_type": auto_seed.get("seed_type"),
        }

    def _annotate_cluster_member_selection(
        self,
        cluster_units: Sequence[Tuple[CandidateClip, List[ExactCluster]]],
    ) -> None:
        """把最终 cluster 选择信息写回 member 的 match_cache，便于流式输出复用。"""

        for candidate, assigned_exact_clusters in cluster_units:
            payload = {
                "selected_candidate_id": str(candidate.candidate_id),
                "selected_shift_direction": str(candidate.shift_direction),
                "selected_shift_distance_um": float(candidate.shift_distance_um),
            }
            for exact_cluster in assigned_exact_clusters:
                for member in exact_cluster.members:
                    member.match_cache["selected_candidate_info"] = dict(payload)

    def _build_stream_cluster_output(
        self,
        cluster_index: int,
        candidate: CandidateClip,
        assigned_exact_clusters: Sequence[ExactCluster],
        sample_index_map: Dict[str, int],
    ) -> Dict[str, Any]:
        """在流式 JSON 写出时按需构建单个 cluster 的输出字典。"""

        cluster_members = list(
            sorted(
                (member for exact_cluster in assigned_exact_clusters for member in exact_cluster.members),
                key=lambda item: (item.source_name, item.marker_id),
            )
        )
        export_member, export_scores = _rerank_export_representative(cluster_members)
        sample_indices = [sample_index_map[member.marker_id] for member in cluster_members]
        sample_metadata = [self._sample_metadata(member) for member in cluster_members]
        exact_cluster_ids = [int(exact_cluster.exact_cluster_id) for exact_cluster in assigned_exact_clusters]
        return {
            "cluster_id": int(cluster_index),
            "pipeline_mode": PIPELINE_MODE,
            "size": int(len(cluster_members)),
            "sample_indices": sample_indices,
            "sample_files": [],
            "sample_metadata": sample_metadata,
            "representative_file": None,
            "cover_representative_file": None,
            "export_representative_file": None,
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
                "sample_index": int(sample_index_map[export_member.marker_id]),
                "source_name": str(export_member.source_name),
                "seed_weight": int(export_member.seed_weight),
                "score": float(export_scores["score"]),
                "medoid_score": float(export_scores["medoid_score"]),
                "worst_case_score": float(export_scores["worst_case_score"]),
                "distance_worst_case_score": float(export_scores["distance_worst_case_score"]),
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

    def _build_results(
        self,
        marker_records: Sequence[MarkerRecord],
        exact_clusters: Sequence[ExactCluster],
        selected_candidates: Sequence[CandidateClip],
        solver_used: str,
        runtime_summary: Dict[str, float],
        candidate_count: int,
        candidate_group_count: int,
        candidate_shift_summary: Dict[str, Any],
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

        verification_started = time.perf_counter()
        cluster_units = self._verified_cluster_units(selected_candidates, exact_clusters)
        self.result_detail_seconds["final_verification"] += time.perf_counter() - verification_started
        self._annotate_cluster_member_selection(cluster_units)
        stream_result = _should_stream_result(len(ordered_records), len(cluster_units), materialize_outputs)

        sample_started = time.perf_counter()
        for sample_index, record in enumerate(ordered_records):
            if materialize_outputs:
                sample_path = sample_dir / _make_sample_filename("sample", record.source_name, sample_index)
                clip_bitmap = getattr(record, "clip_bitmap", None)
                assert clip_bitmap is not None, "物化 sample 时 record.clip_bitmap 不应为空"
                materialize_started = time.perf_counter()
                sample_file = _materialize_clip_bitmap(
                    clip_bitmap,
                    record.clip_bbox,
                    record.marker_id,
                    sample_path,
                    self.pixel_size_um,
                )
                self.result_detail_seconds["sample_materialize"] += time.perf_counter() - materialize_started
                sample_file_map[record.marker_id] = sample_file
                file_list.append(sample_file)
            sample_index_map[record.marker_id] = int(sample_index)
            metadata = self._sample_metadata(record)
            if not stream_result:
                file_metadata.append(metadata)
            if materialize_outputs:
                _ensure_export_rerank_cache(record, include_distance=False)
                if _pack_marker_clip_bitmap(record):
                    self.memory_debug["packed_marker_clip_count"] += 1
            if materialize_outputs and _release_marker_clip_payload(record, keep_clip_bitmap=False):
                self.memory_debug["released_marker_clip_count"] += 1
        self.result_detail_seconds["sample_metadata"] += time.perf_counter() - sample_started
        clusters_output: List[Dict[str, Any]] = []
        cluster_sizes: List[int] = []
        final_cluster_direction_counter: Counter[str] = Counter()
        max_shift = 0.0

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
            cluster_sizes.append(int(len(cluster_members)))
            final_cluster_direction_counter[str(candidate.shift_direction)] += 1
            max_shift = max(max_shift, float(abs(candidate.shift_distance_um)))

            export_member, export_scores = _rerank_export_representative(cluster_members)
            representative_file = None
            if materialize_outputs:
                rep_path = representative_dir / _make_sample_filename("rep", cluster_members[0].source_name, cluster_index)
                candidate_bitmap = _candidate_clip_bitmap(candidate)
                rep_materialize_started = time.perf_counter()
                representative_file = _materialize_clip_bitmap(
                    candidate_bitmap,
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
            sample_metadata = (
                [self._sample_metadata(member) for member in cluster_members]
                if stream_result
                else [dict(file_metadata[sample_index_map[member.marker_id]]) for member in cluster_members]
            )

            exact_cluster_ids = [int(exact_cluster.exact_cluster_id) for exact_cluster in assigned_exact_clusters]
            if not stream_result:
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
                            "distance_worst_case_score": float(export_scores["distance_worst_case_score"]),
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
            if _release_candidate_geometry_payload(candidate, keep_clip_bitmap=False):
                self.memory_debug["released_cache_owner_count"] += 1
                if getattr(candidate, "clip_bitmap", None) is None:
                    self.memory_debug["released_candidate_clip_count"] += 1
        self.result_detail_seconds["cluster_output"] += time.perf_counter() - cluster_output_started
        if not stream_result:
            self._release_marker_records_before_metadata(ordered_records)

        layer_ops = self._layer_operations()
        layer_summary = self._effective_layer_summary()
        apply_layer_operations = bool(self.apply_layer_operations)
        seed_strategy = "geometry_driven"
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
        pre_raster_payload_cache_count = sum(
            int(stats.get("pre_raster_payload_cache_count", 0)) for stats in self._seed_stats_by_source.values()
        )
        exact_bitmap_payload_cache_count = sum(
            int(stats.get("exact_bitmap_payload_cache_count", 0)) for stats in self._seed_stats_by_source.values()
        )
        array_seed_count = sum(int(stats.get("array_seed_count", 0)) for stats in self._seed_stats_by_source.values())
        array_spacing_seed_count = sum(int(stats.get("array_spacing_seed_count", 0)) for stats in self._seed_stats_by_source.values())
        long_shape_seed_count = sum(int(stats.get("long_shape_seed_count", 0)) for stats in self._seed_stats_by_source.values())
        residual_seed_count = sum(int(stats.get("residual_seed_count", 0)) for stats in self._seed_stats_by_source.values())
        array_group_count = sum(int(stats.get("array_group_count", 0)) for stats in self._seed_stats_by_source.values())
        array_spacing_group_count = sum(int(stats.get("array_spacing_group_count", 0)) for stats in self._seed_stats_by_source.values())
        long_shape_count = sum(int(stats.get("long_shape_count", 0)) for stats in self._seed_stats_by_source.values())
        residual_element_count = sum(int(stats.get("residual_element_count", 0)) for stats in self._seed_stats_by_source.values())
        array_spacing_weight_total = sum(int(stats.get("array_spacing_weight_total", 0)) for stats in self._seed_stats_by_source.values())
        seed_weight_total = sum(int(stats.get("seed_weight_total", 0)) for stats in self._seed_stats_by_source.values())
        seed_type_counter: Counter[str] = Counter()
        aggregated_array_groups: List[Dict[str, Any]] = []
        for source_path, stats in self._seed_stats_by_source.items():
            for seed_type, count in dict(stats.get("seed_type_counts", {})).items():
                seed_type_counter[str(seed_type)] += int(count)
            source_audit = dict(stats.get("seed_audit", {}) or {})
            for group in list(source_audit.get("array_groups", []) or []):
                group_payload = dict(group)
                group_payload["source_path"] = str(source_path)
                aggregated_array_groups.append(group_payload)
        seed_type_counts = {str(key): int(value) for key, value in sorted(seed_type_counter.items())}
        seed_audit = {
            "seed_strategy": seed_strategy,
            "grid_step_ratio": float(grid_step_ratio),
            "grid_step_um": float(grid_step_um),
            "array_group_count": int(array_group_count),
            "array_spacing_group_count": int(array_spacing_group_count),
            "array_spacing_seed_count": int(array_spacing_seed_count),
            "array_spacing_weight_total": int(array_spacing_weight_total),
            "array_groups": aggregated_array_groups,
        }
        selected_candidate_direction_counts = {
            str(direction): int(count)
            for direction, count in sorted(Counter(str(candidate.shift_direction) for candidate in selected_candidates).items())
        }
        selected_diagonal_candidate_count = int(
            sum(count for direction, count in selected_candidate_direction_counts.items() if str(direction).startswith("diag_"))
        )
        final_cluster_direction_counts = {
            str(direction): int(count)
            for direction, count in sorted(final_cluster_direction_counter.items())
        }
        total_clusters = int(len(cluster_units) if stream_result else len(clusters_output))
        total_samples = int(len(ordered_records) if stream_result else len(file_metadata))
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
            "pre_raster_payload_cache_count": int(pre_raster_payload_cache_count),
            "exact_bitmap_payload_cache_count": int(exact_bitmap_payload_cache_count),
            "array_seed_count": int(array_seed_count),
            "array_spacing_seed_count": int(array_spacing_seed_count),
            "long_shape_seed_count": int(long_shape_seed_count),
            "residual_seed_count": int(residual_seed_count),
            "array_group_count": int(array_group_count),
            "array_spacing_group_count": int(array_spacing_group_count),
            "long_shape_count": int(long_shape_count),
            "residual_element_count": int(residual_element_count),
            "array_spacing_weight_total": int(array_spacing_weight_total),
            "seed_weight_total": int(seed_weight_total),
            "seed_type_counts": seed_type_counts,
            "seed_audit": seed_audit,
            "exact_cluster_count": int(len(exact_clusters)),
            "candidate_count": int(candidate_count),
            "candidate_group_count": int(candidate_group_count),
            "candidate_object_avoided_count": int(self.memory_debug.get("candidate_object_avoided_count", 0)),
            "candidate_direction_counts": dict(candidate_shift_summary["candidate_direction_counts"]),
            "diagonal_candidate_count": int(candidate_shift_summary["diagonal_candidate_count"]),
            "selected_candidate_count": int(len(selected_candidates)),
            "selected_candidate_direction_counts": selected_candidate_direction_counts,
            "selected_diagonal_candidate_count": int(selected_diagonal_candidate_count),
            "total_clusters": int(total_clusters),
            "total_samples": int(total_samples),
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
            "memory_debug": dict(self.memory_debug),
            "clusters": clusters_output if not stream_result else [],
            "file_list": file_list,
            "file_metadata": file_metadata if not stream_result else [],
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
                "pre_raster_payload_cache_count": int(pre_raster_payload_cache_count),
                "exact_bitmap_payload_cache_count": int(exact_bitmap_payload_cache_count),
                "array_seed_count": int(array_seed_count),
                "array_spacing_seed_count": int(array_spacing_seed_count),
                "long_shape_seed_count": int(long_shape_seed_count),
                "residual_seed_count": int(residual_seed_count),
                "array_group_count": int(array_group_count),
                "array_spacing_group_count": int(array_spacing_group_count),
                "long_shape_count": int(long_shape_count),
                "residual_element_count": int(residual_element_count),
                "array_spacing_weight_total": int(array_spacing_weight_total),
                "seed_weight_total": int(seed_weight_total),
                "seed_type_counts": seed_type_counts,
                "seed_audit": seed_audit,
                "exact_cluster_count": int(len(exact_clusters)),
                "candidate_count": int(candidate_count),
                "candidate_group_count": int(candidate_group_count),
                "candidate_object_avoided_count": int(self.memory_debug.get("candidate_object_avoided_count", 0)),
                "candidate_direction_counts": dict(candidate_shift_summary["candidate_direction_counts"]),
                "diagonal_candidate_count": int(candidate_shift_summary["diagonal_candidate_count"]),
                "selected_candidate_count": int(len(selected_candidates)),
                "selected_candidate_direction_counts": selected_candidate_direction_counts,
                "selected_diagonal_candidate_count": int(selected_diagonal_candidate_count),
                "total_clusters": int(total_clusters),
                "total_samples": int(total_samples),
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
                "memory_debug": dict(self.memory_debug),
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
                "diagonal_shift_axis_max_count": int(DIAGONAL_SHIFT_AXIS_MAX_COUNT),
                "diagonal_shift_max_count": int(DIAGONAL_SHIFT_MAX_COUNT),
                "graph_invariant_limit": GRAPH_INVARIANT_LIMIT,
                "graph_topology_threshold": GRAPH_TOPOLOGY_THRESHOLD,
                "graph_signature_threshold": GRAPH_SIGNATURE_THRESHOLD,
                "strict_invariant_limit": STRICT_INVARIANT_LIMIT,
                "strict_topology_threshold": STRICT_TOPOLOGY_THRESHOLD,
                "strict_signature_threshold": STRICT_SIGNATURE_THRESHOLD,
            },
            "cluster_review": {},
        }
        if stream_result:
            result["__stream_state"] = {
                "ordered_records": ordered_records,
                "sample_index_map": sample_index_map,
                "cluster_units": cluster_units,
                "metadata_builder": self._sample_metadata,
                "cluster_builder": self._build_stream_cluster_output,
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
    """构建当前 optimized v1 geometry-driven 版本的命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="Optimized geometry-driven layout clustering v1")
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
    - 这个版本只使用 geometry-driven seed 主线，seed 前端会按阵列、spacing、长条和 residual 图形采样，
      grid 步长仍固定为 clip size 的 50%，不再保留 pair/drc 分支。
    - seed 会先做 geometry dedupe，再进入 exact hash / graph prefilter / ACC-ECC / shift-cover 主线；
      因此 `marker_count` 在输出里表示最终保留下来的 synthetic seed 样本数。
    - candidate 生成会在轴向 shift 之外补少量 diagonal shift，用于覆盖需要同时 x/y 移动的对齐情况。
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
            f"raw/dedup/merged geometry seed 数: {result.get('grid_seed_count', 0)} / "
            f"{result.get('bucketed_seed_count', 0)} / {result.get('seed_bucket_merged_count', 0)}"
        )
        print(f"seed type counts: {result.get('seed_type_counts', {})}")
        print(f"candidate group 数: {result.get('candidate_group_count', 0)}")
        print(f"避免常驻 candidate 对象数: {result.get('candidate_object_avoided_count', 0)}")
        print(f"candidate 方向分布: {result.get('candidate_direction_counts', {})}")
        print(f"diagonal candidate 数: {result.get('diagonal_candidate_count', 0)}")
        print(f"selected candidate 方向分布: {result.get('selected_candidate_direction_counts', {})}")
        print(f"selected diagonal candidate 数: {result.get('selected_diagonal_candidate_count', 0)}")
        print(f"final cluster 方向分布: {result.get('final_cluster_direction_counts', {})}")
        print(f"final verification: {result.get('final_verification_stats', {})}")
        print(f"memory debug: {result.get('memory_debug', {})}")
        return 0
    except Exception as exc:
        print(f"运行失败: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

