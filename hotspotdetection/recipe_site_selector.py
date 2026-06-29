


#!/usr/bin/env python3
"""CD-SEM hotspot recipe site prototype。

本脚本是当前 hotspot detection 新任务的无训练全流程 prototype，目标不是量产级
detector，而是先把「hotspot marker -> marker 邻域 MP discovery -> AF/AP ->
recipe 输出」闭环跑通。第一版明确要求输入已经包含 hotspot marker layer 和
behavior manifest，其中 `aerial_npz` 必填，`pv/epe/nils/resist` 可选；暂不支持
layout-only fallback、监督训练 detector 或 full-chip blind scan。

整体算法流程:
1. 调用 `hotspot_recipe_notrain_backend.py` 生成 handcrafted FV、ANN coverage
   selection 和 behavior verification，得到 selected representatives。
2. 对每个 selected representative，在 source marker 周边搜索 critical geometry
   anchors，并选出更适合量测的 MP candidate。
3. 按 MP hotspot score、behavior risk、pattern rarity、cluster coverage 计算
   recipe 优先级。
4. 在每个 selected MP 周围滑窗搜索 autofocus candidate；候选需要与 MP 图形相似、
   有足够边缘/角点可对焦，并且不与 MP core 重叠。
5. 在每个 selected MP 周围滑窗搜索 addressing candidate；候选需要在局部搜索区域
   中唯一、非周期重复，并且包含足够可匹配结构。
6. 输出 `recipe_sites.csv`、`recipe_sites.json` 和 `recipe_review/site_XXXX/`，
   对失败 site 保留明确 reject reason。

注意: cluster/member/representative 在本脚本中只是 provenance；对外主输出使用
MP/AF/AP、recipe site 和 reject reason。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from care_area_generator import CareAreaExpansionResult, CareAreaFamily, CareAreaInstance, RejectedCareAreaSeed, build_care_area_groups
import hotspot_recipe_notrain_backend as mp_backend
from metrology_context import build_metrology_context_audit, compute_metrology_context, context_to_summary
from mp_candidate_generator import MPCandidate, MPDiscoveryResult, _candidate_type, discover_mp_candidates
from pattern_memory import append_pattern_memory_export, build_memory_prior_audit, export_pattern_memory
from ring_context import compute_ring_context
from review_evidence_audit import (
    build_review_evidence_audit,
    compute_evidence_contradiction_audit,
    compute_expected_feasibility_audit,
    compute_graph_context,
    compute_pattern_taxonomy_audit,
    enrich_graph_pool_context,
    update_evidence_with_site_outcome,
)
from subset_objective_selection import (
    build_subset_objective_audit,
    candidate_sort_key,
    compute_marginal_gain,
    prepare_candidates,
    update_selected_counts,
)
from layout_utils import (
    DEFAULT_PIXEL_SIZE_NM,
    MarkerRasterBuilder,
    _materialize_clip_bitmap,
    bitmap_fingerprint,
    rasterize_centered_window,
)
from matchability_audit import compute_af_matchability, compute_ap_matchability


PIPELINE_MODE = "hotspot_recipe_selector_v0"
CSV_FIELDS = (
    "site_id",
    "recipe_status",
    "reject_reason",
    "source_marker_id",
    "care_area_family_id",
    "care_area_instance_id",
    "care_area_type",
    "care_area_match_score",
    "care_area_homogeneity_score",
    "care_area_instance_count",
    "care_area_seed_marker_id",
    "care_area_instance_bbox",
    "metrology_priority_score",
    "metrology_priority_class",
    "site_reliability_risk",
    "recipe_waste_penalty",
    "metrology_context_group_id",
    "selection_profile_id",
    "hotspot_cluster_id",
    "mp_candidate_id",
    "mp_candidate_rank",
    "mp_selection_gain",
    "mp_x_um",
    "mp_y_um",
    "mp_source_marker_x_um",
    "mp_source_marker_y_um",
    "mp_source_marker_distance_um",
    "mp_candidate_type",
    "mp_hotspot_score",
    "mp_verified",
    "mp_reject_reason",
    "mp_discovery_components_json",
    "mp_clip_bbox",
    "mp_priority_score",
    "mp_risk_components_json",
    "af_x_um",
    "af_y_um",
    "af_score",
    "af_distance_um",
    "af_similarity",
    "af_reject_reason",
    "af_acceptance_checks_json",
    "ap_x_um",
    "ap_y_um",
    "ap_score",
    "ap_uniqueness_score",
    "ap_peak_count",
    "ap_peak_margin_proxy",
    "ap_peak_ratio",
    "ap_distance_um",
    "ap_reject_reason",
    "ap_acceptance_checks_json",
    "ap_global_duplicate",
    "ap_global_duplicate_with",
    "ap_global_similarity",
    "mp_template_size_um",
    "af_template_size_um",
    "ap_template_size_um",
    "mp_oas",
    "af_oas",
    "ap_oas",
)


@dataclass
class WindowCandidate:
    """保存局部滑窗候选点的 bitmap、坐标、距离和评分字段。"""

    x: float
    y: float
    distance_um: float
    window: Dict[str, Any]
    score: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    acceptance_checks: Dict[str, bool] = field(default_factory=dict)
    peak_count: int = 0
    accepted: bool = False
    reject_reason: str = ""


@dataclass
class MPPoolCandidateInfo:
    """保存进入全局 MP pool 的单个候选点、cluster provenance 和选择状态。"""

    cluster_id: int
    representative_marker_id: str
    marker_ids: List[str]
    representative_metadata: Dict[str, Any]
    cluster: Dict[str, Any]
    mp_candidate_id: str
    mp_candidate_rank: int
    mp_window: Dict[str, Any]
    source_marker_center: Tuple[float, float]
    mp_candidate_type: str
    mp_hotspot_score: float
    mp_verified: bool
    mp_reject_reason: str
    mp_verification_components: Dict[str, float]
    mp_discovery_components: Dict[str, float]
    mp_discovery: MPDiscoveryResult
    raw_components: Dict[str, float]
    care_area_family_id: str = ""
    care_area_instance_id: str = ""
    care_area_type: str = ""
    care_area_match_score: float = 0.0
    care_area_homogeneity_score: float = 0.0
    care_area_instance_count: int = 0
    care_area_seed_marker_id: str = ""
    care_area_instance_bbox: List[float] = field(default_factory=list)
    metrology_priority_score: float = 0.0
    metrology_priority_class: str = "low"
    site_reliability_risk: float = 0.0
    recipe_waste_penalty: float = 0.0
    metrology_context_group_id: str = ""
    selection_profile_id: str = ""
    metrology_context_components: Dict[str, float] = field(default_factory=dict)
    score_components: Dict[str, float] = field(default_factory=dict)
    mp_priority_score: float = 0.0
    mp_selection_gain: float = 0.0
    pool_status: str = "candidate"
    pool_reject_reason: str = ""
    ring_context: Dict[str, Any] = field(default_factory=dict)
    memory_prior_audit: Dict[str, Any] = field(default_factory=dict)
    graph_context_audit: Dict[str, Any] = field(default_factory=dict)
    evidence_contradiction_audit: Dict[str, Any] = field(default_factory=dict)
    pattern_taxonomy_audit: Dict[str, Any] = field(default_factory=dict)
    expected_feasibility_audit: Dict[str, Any] = field(default_factory=dict)
    subset_objective_components: Dict[str, Any] = field(default_factory=dict)
    subset_objective_target_bins: Dict[str, str] = field(default_factory=dict)
    subset_objective_targets: Dict[str, Any] = field(default_factory=dict)
    subset_objective_status: str = "candidate"


class RecipeWindowCache:
    """按 source_path 缓存 layout index，并提供任意中心窗口 rasterize 能力。"""

    def __init__(
        self,
        *,
        marker_layer: str,
        clip_size_um: float,
        output_dir: Path,
        apply_layer_operations: bool,
        layer_processor: Any | None,
        recursive_input: bool,
    ):
        """初始化底层 MarkerRasterBuilder；这里只复用其 OAS flatten 和 layer ops 能力。"""
        self.builder = MarkerRasterBuilder(
            config={
                "hotspot_layer": str(marker_layer),
                "clip_size_um": float(clip_size_um),
                "pixel_size_nm": DEFAULT_PIXEL_SIZE_NM,
                "apply_layer_operations": bool(apply_layer_operations),
                "recursive_input": bool(recursive_input),
            },
            temp_dir=output_dir / "_recipe_window_cache",
            layer_processor=layer_processor if apply_layer_operations else None,
        )
        self.layout_by_path: Dict[str, Any] = {}

    @property
    def pixel_size_um(self) -> float:
        """返回当前 raster pixel size，单位 um。"""
        return float(self.builder.pixel_size_um)

    def layout_index(self, source_path: str | Path) -> Any:
        """读取并缓存单个 OAS 的 layout index。"""
        key = str(Path(source_path).resolve())
        if key not in self.layout_by_path:
            self.layout_by_path[key] = self.builder._prepare_layout(Path(source_path))
        return self.layout_by_path[key]

    def window(self, source_path: str | Path, center_xy: Tuple[float, float], clip_size_um: float) -> Dict[str, Any]:
        """围绕指定中心点生成窗口 bitmap。"""
        return rasterize_centered_window(
            self.layout_index(source_path),
            center_xy,
            float(clip_size_um),
            self.pixel_size_um,
        )


def _json_default(value: Any) -> Any:
    """把 numpy/path 等对象转换成 JSON 可序列化类型。"""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _clip01(value: float) -> float:
    """把数值限制在 0-1 区间。"""
    if not math.isfinite(float(value)):
        return 0.0
    return float(min(1.0, max(0.0, float(value))))


def _norm01(values: Sequence[float], *, equal_value: float = 1.0) -> List[float]:
    """对一组非负分数做 min-max 归一化；全相等时返回指定默认值。"""
    if not values:
        return []
    arr = np.asarray([float(value) for value in values], dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return [0.0 for _ in values]
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi <= lo + 1e-12:
        return [_clip01(equal_value) for _ in values]
    return [_clip01((float(value) - lo) / (hi - lo)) for value in arr.tolist()]


def _bitmap_density(bitmap: np.ndarray) -> float:
    """计算 bitmap 中 pattern pixel 的占比。"""
    arr = np.asarray(bitmap, dtype=bool)
    if arr.size == 0:
        return 0.0
    return float(np.count_nonzero(arr)) / float(arr.size)


def _edge_density_score(bitmap: np.ndarray) -> float:
    """用水平/垂直跳变密度估计局部边缘丰富度。"""
    arr = np.asarray(bitmap, dtype=bool)
    if arr.size == 0:
        return 0.0
    horizontal = np.count_nonzero(arr[:, 1:] != arr[:, :-1]) if arr.shape[1] > 1 else 0
    vertical = np.count_nonzero(arr[1:, :] != arr[:-1, :]) if arr.shape[0] > 1 else 0
    raw = float(horizontal + vertical) / float(max(1, arr.size))
    return _clip01(raw * 10.0)


def _corner_density_score(bitmap: np.ndarray) -> float:
    """用 2x2 patch 中的非单调变化估计 corner/jog 丰富度。"""
    arr = np.asarray(bitmap, dtype=bool)
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        return 0.0
    tl = arr[:-1, :-1]
    tr = arr[:-1, 1:]
    bl = arr[1:, :-1]
    br = arr[1:, 1:]
    counts = tl.astype(np.uint8) + tr.astype(np.uint8) + bl.astype(np.uint8) + br.astype(np.uint8)
    corner_like = np.count_nonzero((counts == 1) | (counts == 3))
    raw = float(corner_like) / float(max(1, counts.size))
    return _clip01(raw * 80.0)


def _entropy_score(bitmap: np.ndarray) -> float:
    """计算二值图案的简化熵分数，避免选择过空或过满的 AP。"""
    p = _clip01(_bitmap_density(bitmap))
    if p <= 1e-9 or p >= 1.0 - 1e-9:
        return 0.0
    entropy = -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))
    return _clip01(entropy)


def _layout_complexity_score(bitmap: np.ndarray) -> float:
    """组合密度、边缘和角点，得到 MP 排序用的 layout complexity。"""
    density = _bitmap_density(bitmap)
    density_balance = _clip01(1.0 - abs(density - 0.35) / 0.35)
    return _clip01(0.45 * _edge_density_score(bitmap) + 0.35 * _corner_density_score(bitmap) + 0.20 * density_balance)


def _focus_quality_score(bitmap: np.ndarray) -> float:
    """组合边缘和角点，估计 AF candidate 的可对焦质量。"""
    return _clip01(0.65 * _edge_density_score(bitmap) + 0.35 * _corner_density_score(bitmap))


def _hotspot_core_risk_proxy(bitmap: np.ndarray, *, similarity_to_mp: float) -> float:
    """估计 AF candidate 是否过于像潜在 defect core，作为弱惩罚和 review 信号。"""
    complexity = _layout_complexity_score(bitmap)
    return _clip01(float(similarity_to_mp) * float(complexity))


def _distance_score(distance_um: float, radius_um: float) -> float:
    """距离越近分数越高，但调用方已经排除了与 MP core 重叠的候选。"""
    return _clip01(1.0 - float(distance_um) / max(float(radius_um), 1e-9))


def _overlap_slices(shape: Tuple[int, int], dy: int, dx: int) -> Tuple[Tuple[slice, slice], Tuple[slice, slice]]:
    """为两个同尺寸 bitmap 的相对平移生成重叠切片。"""
    height, width = int(shape[0]), int(shape[1])
    ay0 = max(0, dy)
    ay1 = min(height, height + dy)
    by0 = max(0, -dy)
    by1 = min(height, height - dy)
    ax0 = max(0, dx)
    ax1 = min(width, width + dx)
    bx0 = max(0, -dx)
    bx1 = min(width, width - dx)
    return (slice(ay0, ay1), slice(ax0, ax1)), (slice(by0, by1), slice(bx0, bx1))


def _shifted_iou(bitmap_a: np.ndarray, bitmap_b: np.ndarray, *, max_shift_px: int = 2) -> float:
    """计算带小范围平移容差的二值 IoU，相似 AF 搜索使用。"""
    a = np.asarray(bitmap_a, dtype=bool)
    b = np.asarray(bitmap_b, dtype=bool)
    if a.shape != b.shape or a.size == 0:
        return 0.0
    best = 0.0
    for dy in range(-int(max_shift_px), int(max_shift_px) + 1):
        for dx in range(-int(max_shift_px), int(max_shift_px) + 1):
            a_slice, b_slice = _overlap_slices(a.shape, dy, dx)
            aa = a[a_slice]
            bb = b[b_slice]
            if aa.size == 0:
                continue
            union = np.count_nonzero(aa | bb)
            if union == 0:
                continue
            score = float(np.count_nonzero(aa & bb)) / float(union)
            best = max(best, score)
    return _clip01(best)


def _iter_search_centers(
    center_xy: Tuple[float, float],
    *,
    radius_um: float,
    step_um: float,
    min_distance_um: float,
    max_distance_um: float | None = None,
) -> Iterable[Tuple[float, float, float]]:
    """按固定步长生成圆形搜索区域内的候选中心。"""
    cx, cy = float(center_xy[0]), float(center_xy[1])
    radius = float(radius_um)
    step = max(float(step_um), 1e-9)
    extent = int(math.floor(radius / step + 1e-9))
    candidates: List[Tuple[float, float, float]] = []
    for iy in range(-extent, extent + 1):
        for ix in range(-extent, extent + 1):
            dx = float(ix) * step
            dy = float(iy) * step
            distance = math.hypot(dx, dy)
            if distance > radius + 1e-9 or distance < float(min_distance_um) - 1e-9:
                continue
            candidates.append((cx + dx, cy + dy, distance))
    candidates.sort(key=lambda item: (item[2], item[1], item[0]))
    return candidates


def _build_search_candidates(
    window_cache: RecipeWindowCache,
    *,
    source_path: str,
    center_xy: Tuple[float, float],
    clip_size_um: float,
    radius_um: float,
    step_um: float,
    min_distance_um: float,
    max_distance_um: float | None = None,
) -> List[WindowCandidate]:
    """为单个 MP 构建 AF/AP 共用的局部滑窗候选。"""
    candidates: List[WindowCandidate] = []
    effective_radius = min(float(radius_um), float(max_distance_um)) if max_distance_um is not None and float(max_distance_um) > 0.0 else float(radius_um)
    for x, y, distance in _iter_search_centers(
        center_xy,
        radius_um=effective_radius,
        step_um=step_um,
        min_distance_um=min_distance_um,
    ):
        window = window_cache.window(source_path, (x, y), clip_size_um)
        if not np.any(window["clip_bitmap"]):
            continue
        candidates.append(WindowCandidate(float(x), float(y), float(distance), window))
    return candidates


def _select_af_candidate(
    candidates: Sequence[WindowCandidate],
    *,
    mp_bitmap: np.ndarray,
    radius_um: float,
) -> WindowCandidate | None:
    """按固定公式选择 AF candidate，低于阈值时返回 best rejected candidate。"""
    best: WindowCandidate | None = None
    for candidate in candidates:
        similarity = _shifted_iou(mp_bitmap, candidate.window["clip_bitmap"], max_shift_px=2)
        focus_quality = _focus_quality_score(candidate.window["clip_bitmap"])
        distance_score = _distance_score(candidate.distance_um, radius_um)
        hotspot_core_risk = _hotspot_core_risk_proxy(candidate.window["clip_bitmap"], similarity_to_mp=similarity)
        matchability = compute_af_matchability(candidate.window["clip_bitmap"], focus_quality=focus_quality)
        score = _clip01(0.55 * similarity + 0.30 * focus_quality + 0.15 * distance_score - 0.05 * hotspot_core_risk)
        candidate.score = score
        candidate.components = {
            "layout_similarity_to_mp": float(similarity),
            "focus_quality": float(focus_quality),
            "distance_score": float(distance_score),
            "hotspot_core_risk": float(hotspot_core_risk),
            **matchability,
        }
        candidate.acceptance_checks = {
            "similarity_ok": bool(similarity >= 0.35),
            "focus_quality_ok": bool(focus_quality >= 0.20),
            "score_ok": bool(score >= 0.55),
            "too_hotspot_like_warning": bool(hotspot_core_risk >= 0.85),
            "hotspot_core_safe": bool(hotspot_core_risk < 0.85),
        }
        candidate.accepted = bool(score >= 0.55 and hotspot_core_risk < 0.85)
        if candidate.accepted:
            candidate.reject_reason = ""
        elif not candidate.acceptance_checks["hotspot_core_safe"]:
            candidate.reject_reason = "too_hotspot_like"
        elif not candidate.acceptance_checks["similarity_ok"]:
            candidate.reject_reason = "low_similarity"
        elif not candidate.acceptance_checks["focus_quality_ok"]:
            candidate.reject_reason = "low_focus_quality"
        else:
            candidate.reject_reason = "low_af_score"
        if best is None or (candidate.accepted and not best.accepted) or (candidate.accepted == best.accepted and candidate.score > best.score):
            best = candidate
    return best


def _annotate_ap_uniqueness(candidates: Sequence[WindowCandidate], *, ignore_radius_um: float) -> None:
    """为 AP candidates 计算近似 nearest-neighbor margin 和 peak count。"""
    if not candidates:
        return
    features = np.vstack([bitmap_fingerprint(candidate.window["clip_bitmap"]) for candidate in candidates]).astype(np.float32)
    centers = np.asarray([[candidate.x, candidate.y] for candidate in candidates], dtype=np.float32)
    chunk = 512
    for start in range(0, len(candidates), chunk):
        stop = min(len(candidates), start + chunk)
        sims = features[start:stop] @ features.T
        distances = np.linalg.norm(centers[start:stop, None, :] - centers[None, :, :], axis=2)
        sims[distances < float(ignore_radius_um)] = -1.0
        top2 = np.max(sims, axis=1) if sims.size else np.full((stop - start,), -1.0, dtype=np.float32)
        peak_counts = np.count_nonzero(sims >= 0.92, axis=1) + 1
        for offset, candidate in enumerate(candidates[start:stop]):
            nearest_similarity = max(0.0, float(top2[offset]))
            peak_margin = _clip01(1.0 - nearest_similarity)
            candidate.components["nearest_similarity"] = nearest_similarity
            candidate.components["uniqueness_score"] = _clip01(1.0 - nearest_similarity)
            candidate.components["template_main_peak_score"] = 1.0
            candidate.components["template_second_peak_score"] = nearest_similarity
            candidate.components["template_peak_margin"] = peak_margin
            candidate.components["template_peak_ratio"] = float(min(100.0, 1.0 / max(nearest_similarity, 1e-6)))
            candidate.peak_count = int(peak_counts[offset])


def _select_ap_candidate(
    candidates: Sequence[WindowCandidate],
    *,
    radius_um: float,
    ignore_radius_um: float,
) -> WindowCandidate | None:
    """按固定公式选择 AP candidate，并执行唯一性阈值和多峰拒绝。"""
    _annotate_ap_uniqueness(candidates, ignore_radius_um=ignore_radius_um)
    best: WindowCandidate | None = None
    for candidate in candidates:
        uniqueness = float(candidate.components.get("uniqueness_score", 0.0))
        peak_margin = float(candidate.components.get("template_peak_margin", uniqueness))
        density = _bitmap_density(candidate.window["clip_bitmap"])
        entropy = _entropy_score(candidate.window["clip_bitmap"])
        edge_density = _edge_density_score(candidate.window["clip_bitmap"])
        corner_density = _corner_density_score(candidate.window["clip_bitmap"])
        distance_score = _distance_score(candidate.distance_um, radius_um)
        matchability = compute_ap_matchability(
            candidate.window["clip_bitmap"],
            descriptor_margin=peak_margin,
            nearest_similarity=float(candidate.components.get("nearest_similarity", 0.0)),
            peak_count=int(candidate.peak_count),
        )
        score = _clip01(0.50 * uniqueness + 0.20 * entropy + 0.20 * corner_density + 0.10 * distance_score)
        candidate.score = score
        candidate.components.update(
            {
                "pattern_density": float(density),
                "entropy_score": float(entropy),
                "edge_density_score": float(edge_density),
                "corner_density_score": float(corner_density),
                "distance_score": float(distance_score),
                **matchability,
            }
        )
        rich_enough = bool(0.05 <= density <= 0.90 and entropy >= 0.20 and max(edge_density, corner_density) >= 0.05)
        candidate.acceptance_checks = {
            "peak_count_ok": bool(candidate.peak_count <= 3),
            "uniqueness_ok": bool(uniqueness >= 0.08 and peak_margin >= 0.08),
            "density_ok": bool(0.05 <= density <= 0.90),
            "entropy_ok": bool(entropy >= 0.20),
            "edge_corner_ok": bool(max(edge_density, corner_density) >= 0.05),
        }
        candidate.accepted = bool(candidate.peak_count <= 3 and uniqueness >= 0.08 and peak_margin >= 0.08 and rich_enough)
        if candidate.accepted:
            candidate.reject_reason = ""
        elif not candidate.acceptance_checks["peak_count_ok"]:
            candidate.reject_reason = "too_many_peaks"
        elif not candidate.acceptance_checks["uniqueness_ok"]:
            candidate.reject_reason = "low_uniqueness"
        elif not candidate.acceptance_checks["density_ok"]:
            candidate.reject_reason = "density_out_of_range"
        elif not candidate.acceptance_checks["entropy_ok"]:
            candidate.reject_reason = "low_entropy"
        else:
            candidate.reject_reason = "low_edge_corner"
        if best is None or (candidate.accepted and not best.accepted) or (candidate.accepted == best.accepted and candidate.score > best.score):
            best = candidate
    return best


def _copy_backend_args(args: argparse.Namespace, output_dir: Path) -> argparse.Namespace:
    """构造 no-train backend 所需参数，避免 recipe CLI 与旧入口耦合。"""
    return argparse.Namespace(
        input_path=str(args.input_path),
        marker_layer=str(args.marker_layer),
        clip_size=float(args.clip_size),
        behavior_manifest=str(args.behavior_manifest),
        ann_top_k=64,
        coverage_target=float(args.mp_coverage_target),
        facility_min_gain=1e-4,
        behavior_verification_threshold=0.08,
        local_residual_threshold=0.12,
        similarity_tau=None,
        similarity_tau_min=0.10,
        verification_shift_px=1,
        risk_weight_scale=1.0,
        recursive_input=bool(args.recursive_input),
        high_risk_quantile=0.90,
        apply_layer_ops=bool(args.apply_layer_ops or args.register_op),
        register_op=args.register_op,
    )


def _metadata_by_marker(backend_result: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """从 no-train 输出中建立 marker_id 到 metadata 的索引。"""
    result: Dict[str, Dict[str, Any]] = {}
    for item in backend_result.get("file_metadata", []) or []:
        marker_id = str(item.get("marker_id", ""))
        if marker_id:
            result[marker_id] = dict(item)
    return result


def _lightweight_discovery_from_instance(
    family: CareAreaFamily,
    instance: CareAreaInstance,
    *,
    behavior_risk: float,
    behavior_risk_enabled: bool,
) -> MPDiscoveryResult:
    """把已匹配通过的 expanded care-area instance 转成一个轻量 rank-0 MP 候选。"""
    candidate_type = _candidate_type(instance.raw_sources) if instance.raw_sources else str(family.seed_candidate.candidate_type)
    components = dict(family.seed_candidate.components)
    components.update(
        {
            "care_area_lightweight_instance": 1.0,
            "care_area_match_score": float(instance.match_score),
            "care_area_homogeneity_score": float(instance.homogeneity_score),
            "known_marker_similarity": float(instance.bitmap_similarity),
            "fragment_signature_similarity": float(instance.fragment_signature_similarity),
            "behavior_risk": float(behavior_risk),
            "behavior_weight_redistributed": 0.0 if behavior_risk_enabled else 1.0,
        }
    )
    score = _clip01(0.70 * float(family.seed_candidate.score) + 0.20 * float(instance.match_score) + 0.10 * float(instance.homogeneity_score))
    center = (float(instance.center[0]), float(instance.center[1]))
    seed_center = (float(family.seed_center[0]), float(family.seed_center[1]))
    density = _bitmap_density(instance.window["clip_bitmap"])
    has_bitmap = bool(np.any(instance.window["clip_bitmap"]))
    instance_ok = bool(has_bitmap and 0.03 <= density <= 0.92)
    if instance_ok:
        verification_reason = ""
    elif not has_bitmap:
        verification_reason = "empty_bitmap"
    elif density < 0.03:
        verification_reason = "sparse_bitmap"
    else:
        verification_reason = "uniform_bitmap"
    verification_components = dict(family.seed_candidate.verification_components)
    verification_components["density"] = float(density)
    verification_components["lightweight_instance_density_check"] = 1.0
    components["lightweight_instance_density"] = float(density)
    candidate = MPCandidate(
        x=center[0],
        y=center[1],
        distance_um=math.hypot(center[0] - seed_center[0], center[1] - seed_center[1]),
        candidate_type=candidate_type,
        sources=list(instance.raw_sources or family.seed_candidate.sources),
        window=instance.window,
        score=score,
        components=components,
        accepted=instance_ok,
        reject_reason=verification_reason,
        verified=bool(family.seed_candidate.verified and instance_ok),
        verification_reason=verification_reason,
        verification_components=verification_components,
        proposal_metrics={"care_area_instance_rank0": 1.0},
    )
    return MPDiscoveryResult(
        selected_candidate=candidate,
        top_candidates=[candidate],
        raw_candidate_count=1,
        rasterized_candidate_count=1,
        empty_rejected_count=0 if has_bitmap else 1,
        nms_rejected_count=0,
        verification_rejected_count=0 if candidate.verified else 1,
        mp_discovery_reason="care_area_instance_rank0",
        behavior_risk_enabled=bool(behavior_risk_enabled),
        rule_coverage_audit={
            "semantic_marker_covered": bool(candidate.verified),
            "care_area_lightweight_instance": True,
            "candidate_type_distribution": {candidate_type: 1},
        },
    )


def _family_representativeness_for_pool(coverage_weight: int) -> float:
    """用 family/cluster 覆盖规模估计该 MP 候选的代表性。"""
    return _clip01(math.log1p(float(max(1, int(coverage_weight)))) / math.log1p(20.0))


def _candidate_pattern_rarity(mp_candidate: MPCandidate, coverage_weight: int) -> float:
    """优先使用 MP candidate 自身 rarity，缺省时退化为 coverage 反比。"""
    fallback = 1.0 / float(max(1, int(coverage_weight)))
    return _clip01(float(mp_candidate.components.get("local_rarity", mp_candidate.components.get("pattern_rarity", fallback))))


def _candidate_localization_confidence(mp_candidate: MPCandidate) -> float:
    """从 candidate voting / support 信息估计 MP 定位置信度。"""
    components = mp_candidate.components
    if "proposal_voting" in components:
        return _clip01(float(components.get("proposal_voting", 0.0)))
    if "voting_confidence" in components:
        return _clip01(float(components.get("voting_confidence", 0.0)))
    if "supporting_anchor_count" in components:
        return _clip01(float(components.get("supporting_anchor_count", 0.0)) / 3.0)
    return 1.0 if bool(mp_candidate.verified) else 0.0


def _mp_candidate_metrology_summary(
    *,
    family: CareAreaFamily,
    instance: CareAreaInstance,
    mp_candidate: MPCandidate,
    effective_behavior_risk: float,
    coverage_weight: int,
) -> Tuple[Dict[str, Any], float, float, float]:
    """基于单个 MP candidate 的实际窗口和几何组件计算量测 context。"""
    pattern_rarity = _candidate_pattern_rarity(mp_candidate, coverage_weight)
    localization_confidence = _candidate_localization_confidence(mp_candidate)
    representativeness = _family_representativeness_for_pool(coverage_weight)
    context = compute_metrology_context(
        care_area_type=str(family.care_area_type),
        bitmap=mp_candidate.window["clip_bitmap"],
        components=mp_candidate.components,
        inherited_behavior_risk=float(effective_behavior_risk),
        family_representativeness=float(representativeness),
        pattern_rarity=float(pattern_rarity),
        mp_localization_confidence=float(localization_confidence),
        family_homogeneity=float(instance.homogeneity_score),
        signature_quality=float(instance.signature_quality),
        mp_verified=bool(mp_candidate.verified),
    )
    return context_to_summary(context), float(pattern_rarity), float(localization_confidence), float(representativeness)


def _lightweight_instance_metrology_summary(
    *,
    instance: CareAreaInstance,
    coverage_weight: int,
) -> Tuple[Dict[str, Any], float, float, float]:
    """复用 expanded lightweight instance 已有 context，避免对同一窗口重复计算。"""
    summary = {
        "metrology_priority_score": float(instance.metrology_priority_score),
        "metrology_priority_class": str(instance.metrology_priority_class),
        "site_reliability_risk": float(instance.site_reliability_risk),
        "recipe_waste_penalty": float(instance.recipe_waste_penalty),
        "metrology_context_group_id": str(instance.metrology_context_group_id),
        "selection_profile_id": str(instance.selection_profile_id),
        "metrology_context_components": dict(instance.metrology_context_components),
    }
    components = dict(instance.metrology_context_components)
    pattern_rarity = _clip01(float(components.get("pattern_rarity", 0.0)))
    localization_confidence = _clip01(float(components.get("mp_localization_confidence", 0.0)))
    representativeness = _clip01(float(components.get("family_representativeness", _family_representativeness_for_pool(coverage_weight))))
    return summary, float(pattern_rarity), float(localization_confidence), float(representativeness)


def _apply_metrology_summary_to_pool_info(info: MPPoolCandidateInfo, summary: Mapping[str, Any]) -> None:
    """把 candidate-level metrology context 写回 MP pool info 和 raw components。"""
    info.metrology_priority_score = float(summary["metrology_priority_score"])
    info.metrology_priority_class = str(summary["metrology_priority_class"])
    info.site_reliability_risk = float(summary["site_reliability_risk"])
    info.recipe_waste_penalty = float(summary["recipe_waste_penalty"])
    info.metrology_context_group_id = str(summary["metrology_context_group_id"])
    info.selection_profile_id = str(summary["selection_profile_id"])
    info.metrology_context_components = dict(summary["metrology_context_components"])
    info.raw_components["metrology_priority_score"] = float(info.metrology_priority_score)
    info.raw_components["recipe_waste_penalty"] = float(info.recipe_waste_penalty)
    info.raw_components["site_reliability_risk"] = float(info.site_reliability_risk)
    info.score_components["site_reliability_risk"] = float(info.site_reliability_risk)
    info.score_components["recipe_waste_penalty"] = float(info.recipe_waste_penalty)
    info.score_components["metrology_priority_raw"] = float(info.metrology_priority_score)
    info.score_components["metrology_priority_class"] = str(info.metrology_priority_class)
    info.score_components["metrology_context_group_id"] = str(info.metrology_context_group_id)
    info.score_components["selection_profile_id"] = str(info.selection_profile_id)


def _build_mp_candidate_pool(
    backend_result: Mapping[str, Any],
    *,
    window_cache: RecipeWindowCache,
    clip_size_um: float,
    mp_search_radius_um: float,
    candidate_step_um: float,
    mp_candidates_per_marker: int,
    max_care_area_instances_per_family: int,
    min_feature_um: float | None = None,
    mp_template_size_um: float | None = None,
) -> Tuple[List[MPPoolCandidateInfo], CareAreaExpansionResult]:
    """把 no-train clusters 先展开成 care-area instances，再生成全局 MP candidate pool。"""
    metadata_by_marker = _metadata_by_marker(backend_result)
    mp_window_size = float(mp_template_size_um) if mp_template_size_um is not None and float(mp_template_size_um) > 0.0 else float(clip_size_um)
    care_area_result = build_care_area_groups(
        backend_result,
        metadata_by_marker=metadata_by_marker,
        layout_index_for_source=window_cache.layout_index,
        window_for_source=window_cache.window,
        pixel_size_um=window_cache.pixel_size_um,
        window_size_um=float(mp_window_size),
        mp_search_radius_um=float(mp_search_radius_um),
        step_um=float(candidate_step_um),
        mp_candidates_per_marker=max(1, int(mp_candidates_per_marker)),
        max_instances_per_family=max(1, int(max_care_area_instances_per_family)),
        min_feature_um=min_feature_um,
    )
    all_risk_zero = all(float(family.behavior_risk) <= 1e-12 for family in care_area_result.families)
    infos: List[MPPoolCandidateInfo] = []
    for family in care_area_result.families:
        representative_marker_id = str(family.seed_marker_id)
        representative_metadata = dict(family.representative_metadata)
        source_path = str(family.source_path)
        center_values = representative_metadata.get("marker_center", [0.0, 0.0])
        source_marker_center = (float(center_values[0]), float(center_values[1]))
        coverage_weight = max(1, int(family.cluster_size), int(len(family.instances)))
        for instance in family.instances:
            instance_center = (float(instance.center[0]), float(instance.center[1]))
            risk_attenuation = 1.0 if bool(instance.is_seed_instance) else _clip01(float(instance.match_score) * float(instance.homogeneity_score))
            effective_behavior_risk = float(family.behavior_risk) * float(risk_attenuation)
            if bool(instance.is_seed_instance):
                marker_window = window_cache.window(source_path, instance_center, mp_window_size)
                discovery = discover_mp_candidates(
                    layout_index=window_cache.layout_index(source_path),
                    marker_center=instance_center,
                    marker_window=marker_window,
                    window_size_um=float(mp_window_size),
                    pixel_size_um=window_cache.pixel_size_um,
                    search_radius_um=float(mp_search_radius_um),
                    step_um=float(candidate_step_um),
                    behavior_risk=float(effective_behavior_risk),
                    behavior_risk_enabled=not all_risk_zero,
                    min_feature_um=min_feature_um,
                    top_k=max(1, int(mp_candidates_per_marker)),
                )
            else:
                discovery = _lightweight_discovery_from_instance(
                    family,
                    instance,
                    behavior_risk=float(effective_behavior_risk),
                    behavior_risk_enabled=not all_risk_zero,
                )
            for rank, mp_candidate in enumerate(discovery.top_candidates):
                candidate_id = f"{family.family_id}__inst_{int(instance.instance_rank):04d}__mpcand_{int(rank):03d}"
                if bool(instance.is_seed_instance):
                    metrology_summary, pattern_rarity, localization_confidence, family_representativeness = _mp_candidate_metrology_summary(
                        family=family,
                        instance=instance,
                        mp_candidate=mp_candidate,
                        effective_behavior_risk=float(effective_behavior_risk),
                        coverage_weight=int(coverage_weight),
                    )
                else:
                    metrology_summary, pattern_rarity, localization_confidence, family_representativeness = _lightweight_instance_metrology_summary(
                        instance=instance,
                        coverage_weight=int(coverage_weight),
                    )
                infos.append(
                    MPPoolCandidateInfo(
                        cluster_id=int(family.cluster_id),
                        representative_marker_id=representative_marker_id,
                        marker_ids=list(family.marker_ids),
                        representative_metadata=representative_metadata,
                        cluster=dict(family.cluster),
                        mp_candidate_id=candidate_id,
                        mp_candidate_rank=int(rank),
                        mp_window=mp_candidate.window,
                        source_marker_center=source_marker_center,
                        mp_candidate_type=mp_candidate.candidate_type,
                        mp_hotspot_score=float(mp_candidate.score),
                        mp_verified=bool(mp_candidate.verified),
                        mp_reject_reason=str(mp_candidate.verification_reason),
                        mp_verification_components=dict(mp_candidate.verification_components),
                        mp_discovery_components=dict(mp_candidate.components),
                        mp_discovery=discovery,
                        raw_components={
                            "mp_hotspot_score": float(mp_candidate.score),
                            "behavior_risk": float(effective_behavior_risk),
                            "seed_behavior_risk": float(family.behavior_risk),
                            "effective_behavior_risk": float(effective_behavior_risk),
                            "risk_attenuation_factor": float(risk_attenuation),
                            "pattern_rarity": float(pattern_rarity),
                            "mp_localization_confidence": float(localization_confidence),
                            "family_representativeness": float(family_representativeness),
                            "care_area_signature_quality": float(instance.signature_quality),
                            "cluster_coverage": float(coverage_weight),
                            "care_area_match_score": float(instance.match_score),
                            "care_area_homogeneity_score": float(instance.homogeneity_score),
                            "metrology_priority_score": float(metrology_summary["metrology_priority_score"]),
                            "recipe_waste_penalty": float(metrology_summary["recipe_waste_penalty"]),
                            "site_reliability_risk": float(metrology_summary["site_reliability_risk"]),
                        },
                        care_area_family_id=str(family.family_id),
                        care_area_instance_id=str(instance.instance_id),
                        care_area_type=str(family.care_area_type),
                        care_area_match_score=float(instance.match_score),
                        care_area_homogeneity_score=float(instance.homogeneity_score),
                        care_area_instance_count=int(len(family.instances)),
                        care_area_seed_marker_id=str(family.seed_marker_id),
                        care_area_instance_bbox=[float(value) for value in instance.bbox],
                        metrology_priority_score=float(metrology_summary["metrology_priority_score"]),
                        metrology_priority_class=str(metrology_summary["metrology_priority_class"]),
                        site_reliability_risk=float(metrology_summary["site_reliability_risk"]),
                        recipe_waste_penalty=float(metrology_summary["recipe_waste_penalty"]),
                        metrology_context_group_id=str(metrology_summary["metrology_context_group_id"]),
                        selection_profile_id=str(metrology_summary["selection_profile_id"]),
                        metrology_context_components=dict(metrology_summary["metrology_context_components"]),
                    )
                )
    return infos, care_area_result


def _score_mp_candidates(infos: Sequence[MPPoolCandidateInfo], *, all_risk_zero: bool | None = None) -> None:
    """按固定 MP 优先级公式给 clusters 打分。"""
    if not infos:
        return
    valid_indices = [index for index, info in enumerate(infos) if bool(info.mp_verified) and info.pool_status == "candidate"]
    global_rarity_by_index = [0.0 for _ in infos]
    if len(valid_indices) == 1:
        global_rarity_by_index[valid_indices[0]] = 1.0
    elif len(valid_indices) > 1:
        features = np.vstack([bitmap_fingerprint(infos[index].mp_window["clip_bitmap"]) for index in valid_indices]).astype(np.float32)
        sims = features @ features.T
        np.fill_diagonal(sims, -1.0)
        nearest = np.max(sims, axis=1)
        for offset, index in enumerate(valid_indices):
            global_rarity_by_index[index] = _clip01(1.0 - max(0.0, float(nearest[offset])))
    cluster_rarity = [0.0 for _ in infos]
    if valid_indices:
        valid_cluster_sizes = [float(infos[index].raw_components.get("cluster_coverage", 1.0)) for index in valid_indices]
        valid_cluster_rarity = _norm01([-value for value in valid_cluster_sizes], equal_value=1.0)
        for offset, index in enumerate(valid_indices):
            cluster_rarity[index] = float(valid_cluster_rarity[offset])
    for index, info in enumerate(infos):
        local_rarity = float(info.raw_components.get("pattern_rarity", 0.0))
        global_rarity = float(global_rarity_by_index[index])
        pattern_novelty = _clip01(0.45 * local_rarity + 0.45 * global_rarity + 0.10 * float(cluster_rarity[index]))
        info.raw_components["global_candidate_rarity"] = float(global_rarity)
        info.raw_components["cluster_rarity"] = float(cluster_rarity[index])
        info.raw_components["pattern_novelty"] = float(pattern_novelty)
        info.raw_components["low_recipe_waste_confidence"] = _clip01(1.0 - float(info.raw_components.get("recipe_waste_penalty", 0.0)))
    keys = ("mp_hotspot_score", "behavior_risk", "pattern_novelty", "cluster_coverage", "mp_localization_confidence", "low_recipe_waste_confidence")
    normalized: Dict[str, List[float]] = {key: [0.0 for _ in infos] for key in keys}
    for key in keys:
        valid_values = [
            math.log1p(float(infos[index].raw_components.get(key, 0.0))) if key == "cluster_coverage" else float(infos[index].raw_components.get(key, 0.0))
            for index in valid_indices
        ]
        valid_normalized = _norm01(valid_values, equal_value=1.0) if valid_values else []
        for offset, index in enumerate(valid_indices):
            normalized[key][index] = float(valid_normalized[offset])
    risk_indices = valid_indices if valid_indices else list(range(len(infos)))
    if all_risk_zero is None:
        all_risk_zero = all(float(infos[index].raw_components.get("behavior_risk", 0.0)) <= 1e-12 for index in risk_indices)
    all_risk_zero = bool(all_risk_zero)
    if all_risk_zero:
        weights = {
            "mp_hotspot_score": 0.30 / 0.80,
            "behavior_risk": 0.0,
            "pattern_novelty": 0.15 / 0.80,
            "cluster_coverage": 0.15 / 0.80,
            "mp_localization_confidence": 0.10 / 0.80,
            "low_recipe_waste_confidence": 0.10 / 0.80,
        }
    else:
        weights = {
            "mp_hotspot_score": 0.30,
            "behavior_risk": 0.20,
            "pattern_novelty": 0.15,
            "cluster_coverage": 0.15,
            "mp_localization_confidence": 0.10,
            "low_recipe_waste_confidence": 0.10,
        }
    for index, info in enumerate(infos):
        info.score_components = {key: float(normalized[key][index]) for key in keys}
        info.score_components["local_rarity"] = float(info.raw_components.get("pattern_rarity", 0.0))
        info.score_components["global_candidate_rarity"] = float(info.raw_components.get("global_candidate_rarity", 0.0))
        info.score_components["cluster_rarity"] = float(info.raw_components.get("cluster_rarity", 0.0))
        info.score_components["care_area_match_score"] = float(info.raw_components.get("care_area_match_score", 0.0))
        info.score_components["care_area_homogeneity_score"] = float(info.raw_components.get("care_area_homogeneity_score", 0.0))
        info.score_components["site_reliability_risk"] = float(info.raw_components.get("site_reliability_risk", 0.0))
        info.score_components["recipe_waste_penalty"] = float(info.raw_components.get("recipe_waste_penalty", 0.0))
        info.score_components["metrology_priority_raw"] = float(info.raw_components.get("metrology_priority_score", 0.0))
        info.score_components["metrology_priority_class"] = str(info.metrology_priority_class)
        info.score_components["metrology_context_group_id"] = str(info.metrology_context_group_id)
        info.score_components["selection_profile_id"] = str(info.selection_profile_id)
        info.score_components["seed_behavior_risk"] = float(info.raw_components.get("seed_behavior_risk", info.raw_components.get("behavior_risk", 0.0)))
        info.score_components["effective_behavior_risk"] = float(info.raw_components.get("effective_behavior_risk", info.raw_components.get("behavior_risk", 0.0)))
        info.score_components["risk_attenuation_factor"] = float(info.raw_components.get("risk_attenuation_factor", 1.0))
        info.score_components["risk_weight_redistributed"] = 1.0 if all_risk_zero else 0.0
        info.mp_priority_score = float(
            sum(float(weights[key]) * float(info.score_components[key]) for key in keys)
        )
        waste_penalty = float(info.raw_components.get("recipe_waste_penalty", 0.0))
        metrology_priority = float(info.raw_components.get("metrology_priority_score", 0.0))
        waste_excess = _clip01((waste_penalty - 0.50) / 0.50)
        demotion_factor = 1.0 - 0.25 * waste_excess
        if metrology_priority > 0.70:
            demotion_factor += 0.10 * waste_excess * _clip01((metrology_priority - 0.70) / 0.30)
        demotion_factor = _clip01(demotion_factor)
        soft_demoted = bool(demotion_factor < 1.0 - 1e-12)
        info.mp_priority_score *= float(demotion_factor)
        info.score_components["recipe_waste_soft_demoted"] = 1.0 if soft_demoted else 0.0
        info.score_components["recipe_waste_demotion_factor"] = float(demotion_factor)


def _mp_center(info: MPPoolCandidateInfo) -> Tuple[float, float]:
    """返回 MP pool candidate 的中心坐标。"""
    center = info.mp_window.get("center", [0.0, 0.0])
    return (float(center[0]), float(center[1]))


def _pool_duplicate_similarity(
    left: MPPoolCandidateInfo,
    right: MPPoolCandidateInfo,
    *,
    left_fingerprint: np.ndarray | None = None,
    right_fingerprint: np.ndarray | None = None,
) -> float:
    """计算两个全局 MP pool candidates 的近似模板相似度。"""
    left_bitmap = left.mp_window["clip_bitmap"]
    right_bitmap = right.mp_window["clip_bitmap"]
    shifted = _shifted_iou(left_bitmap, right_bitmap, max_shift_px=2)
    left_fp = np.asarray(left_fingerprint, dtype=np.float32) if left_fingerprint is not None else bitmap_fingerprint(left_bitmap)
    right_fp = np.asarray(right_fingerprint, dtype=np.float32) if right_fingerprint is not None else bitmap_fingerprint(right_bitmap)
    coarse = float(np.dot(left_fp, right_fp))
    if shifted >= 0.88 or (shifted >= 0.70 and coarse >= 0.90):
        return _clip01(max(shifted, coarse))
    return _clip01(min(shifted, coarse))


def _pre_dedup_mp_candidate_pool(infos: Sequence[MPPoolCandidateInfo], *, duplicate_radius_um: float) -> int:
    """在全局 rarity 打分前去掉明显重复 MP，避免重复候选污染新颖度。"""
    kept: List[MPPoolCandidateInfo] = []
    rejected_count = 0
    ordered = sorted(
        infos,
        key=lambda item: (-float(item.mp_hotspot_score), -float(item.care_area_match_score), int(item.mp_candidate_rank), str(item.mp_candidate_id)),
    )
    fingerprint_cache = {
        str(info.mp_candidate_id): bitmap_fingerprint(info.mp_window["clip_bitmap"])
        for info in ordered
        if bool(info.mp_verified) and info.pool_status == "candidate"
    }
    for info in ordered:
        if not bool(info.mp_verified) or info.pool_status != "candidate":
            continue
        center = _mp_center(info)
        is_duplicate = False
        for existing in kept:
            other_center = _mp_center(existing)
            if math.hypot(center[0] - other_center[0], center[1] - other_center[1]) > float(duplicate_radius_um):
                continue
            if _pool_duplicate_similarity(
                existing,
                info,
                left_fingerprint=fingerprint_cache.get(str(existing.mp_candidate_id)),
                right_fingerprint=fingerprint_cache.get(str(info.mp_candidate_id)),
            ) >= 0.88:
                is_duplicate = True
                break
        if is_duplicate:
            info.pool_status = "rejected"
            info.pool_reject_reason = "mp_pool_preduplicate"
            rejected_count += 1
        else:
            kept.append(info)
    return int(rejected_count)


def _suppress_pool_duplicates(
    infos: Sequence[MPPoolCandidateInfo],
    *,
    selected: MPPoolCandidateInfo,
    duplicate_radius_um: float,
    fingerprint_cache: Mapping[str, np.ndarray],
) -> None:
    """在全局 MP pool 中抑制与已选 candidate 空间近邻且模板相似的候选。"""
    selected_center = _mp_center(selected)
    for info in infos:
        if info.pool_status != "candidate":
            continue
        if math.hypot(_mp_center(info)[0] - selected_center[0], _mp_center(info)[1] - selected_center[1]) > float(duplicate_radius_um):
            continue
        similarity = _pool_duplicate_similarity(
            selected,
            info,
            left_fingerprint=fingerprint_cache.get(str(selected.mp_candidate_id)),
            right_fingerprint=fingerprint_cache.get(str(info.mp_candidate_id)),
        )
        if similarity >= 0.92:
            info.pool_status = "rejected"
            info.pool_reject_reason = "mp_pool_duplicate"


def _objective_candidate_input(info: MPPoolCandidateInfo) -> Dict[str, Any]:
    """把 MP pool dataclass 压成 subset objective selector 需要的轻量输入。"""
    return {
        "mp_candidate_id": str(info.mp_candidate_id),
        "mp_candidate_rank": int(info.mp_candidate_rank),
        "pool_status": str(info.pool_status),
        "pool_reject_reason": str(info.pool_reject_reason),
        "mp_verified": bool(info.mp_verified),
        "source_marker_id": str(info.representative_marker_id),
        "hotspot_cluster_id": int(info.cluster_id),
        "care_area_family_id": str(info.care_area_family_id or info.cluster_id),
        "care_area_instance_id": str(info.care_area_instance_id),
        "care_area_type": str(info.care_area_type or info.mp_candidate_type),
        "mp_candidate_type": str(info.mp_candidate_type),
        "metrology_context_group_id": str(info.metrology_context_group_id or f"{info.care_area_type or info.mp_candidate_type}__unknown"),
        "metrology_priority_score": float(info.metrology_priority_score),
        "recipe_waste_penalty": float(info.recipe_waste_penalty),
        "mp_hotspot_score": float(info.mp_hotspot_score),
        "mp_priority_score": float(info.mp_priority_score),
        "score_components": dict(info.score_components),
        "raw_components": dict(info.raw_components),
        "evidence_contradiction_audit": dict(info.evidence_contradiction_audit),
        "pattern_taxonomy_audit": dict(info.pattern_taxonomy_audit),
        "expected_feasibility_audit": dict(info.expected_feasibility_audit),
    }


def _sync_subset_objective_to_info(info: MPPoolCandidateInfo, candidate: Mapping[str, Any]) -> None:
    """把 objective selector 结果写回 MP pool candidate，供 review 输出复用。"""
    info.subset_objective_components = dict(candidate.get("subset_objective_components", {}) or {})
    info.subset_objective_target_bins = {
        str(key): str(value)
        for key, value in dict(candidate.get("subset_objective_target_bins", {}) or {}).items()
    }
    info.subset_objective_targets = dict(candidate.get("subset_objective_targets", {}) or {})
    info.subset_objective_status = str(candidate.get("subset_objective_status", info.pool_status))
    info.mp_selection_gain = float(candidate.get("subset_objective_marginal_gain", info.mp_selection_gain))
    info.score_components["subset_objective_components"] = dict(info.subset_objective_components)
    info.score_components["subset_objective_target_bins"] = dict(info.subset_objective_target_bins)
    info.score_components["subset_objective_status"] = str(info.subset_objective_status)
    info.score_components["subset_objective_marginal_gain"] = float(info.mp_selection_gain)


def _select_mp_candidate_pool(
    infos: Sequence[MPPoolCandidateInfo],
    *,
    max_sites: int,
    duplicate_radius_um: float,
) -> List[MPPoolCandidateInfo]:
    """按 Casati-style 多目标边际收益选择全局 MP subset。"""
    selected: List[MPPoolCandidateInfo] = []
    for info in infos:
        if info.pool_status == "candidate" and not bool(info.mp_verified):
            info.pool_status = "rejected"
            info.pool_reject_reason = "mp_pool_invalid"
            info.mp_selection_gain = 0.0
    fingerprint_cache = {
        str(info.mp_candidate_id): bitmap_fingerprint(info.mp_window["clip_bitmap"])
        for info in infos
    }
    candidates_by_id: Dict[str, Dict[str, Any]] = {
        str(info.mp_candidate_id): _objective_candidate_input(info)
        for info in infos
    }
    objective_meta = prepare_candidates(list(candidates_by_id.values()), max_sites=max(0, int(max_sites)))
    target_distribution = dict(objective_meta.get("target_distribution", {}) or {})
    for candidate in candidates_by_id.values():
        candidate["subset_objective_targets"] = target_distribution
    selected_bin_counts: Dict[str, int] = {}

    def objective_spatial_novelty(info: MPPoolCandidateInfo) -> float:
        """根据已选 MP 最近距离估计空间多样性收益。"""
        if not selected:
            return 1.0
        center = _mp_center(info)
        min_distance = min(math.hypot(center[0] - _mp_center(item)[0], center[1] - _mp_center(item)[1]) for item in selected)
        return _clip01(min_distance / max(float(duplicate_radius_um) * 3.0, 1e-9))

    while len(selected) < max(0, int(max_sites)):
        best: MPPoolCandidateInfo | None = None
        best_key: Tuple[float, float, float, int, str] | None = None
        for info in infos:
            if info.pool_status != "candidate":
                continue
            candidate = candidates_by_id[str(info.mp_candidate_id)]
            gain_components = compute_marginal_gain(
                candidate,
                selected_counts=selected_bin_counts,
                targets=target_distribution,
                spatial_diversity_gain=objective_spatial_novelty(info),
            )
            candidate["subset_objective_components"].update(gain_components)
            candidate["subset_objective_marginal_gain"] = float(gain_components["subset_objective_marginal_gain"])
            candidate["subset_objective_status"] = "candidate"
            _sync_subset_objective_to_info(info, candidate)
            key = candidate_sort_key(candidate, float(gain_components["subset_objective_marginal_gain"]))
            if best_key is None or key > best_key:
                best = info
                best_key = key
        if best is None:
            break
        best.pool_status = "selected"
        best.pool_reject_reason = ""
        best_candidate = candidates_by_id[str(best.mp_candidate_id)]
        best_candidate["pool_status"] = "selected"
        best_candidate["pool_reject_reason"] = ""
        best_candidate["subset_objective_status"] = "selected"
        _sync_subset_objective_to_info(best, best_candidate)
        selected.append(best)
        update_selected_counts(selected_bin_counts, best_candidate)
        _suppress_pool_duplicates(
            infos,
            selected=best,
            duplicate_radius_um=float(duplicate_radius_um),
            fingerprint_cache=fingerprint_cache,
        )
        for info in infos:
            candidate = candidates_by_id[str(info.mp_candidate_id)]
            candidate["pool_status"] = str(info.pool_status)
            candidate["pool_reject_reason"] = str(info.pool_reject_reason)
            if info.pool_reject_reason == "mp_pool_duplicate":
                candidate["subset_objective_status"] = "mp_pool_duplicate"
                _sync_subset_objective_to_info(info, candidate)

    spatial_cache = {
        str(info.mp_candidate_id): objective_spatial_novelty(info)
        for info in infos
        if info.pool_status == "candidate"
    }
    for info in infos:
        if info.pool_status == "candidate":
            candidate = candidates_by_id[str(info.mp_candidate_id)]
            gain_components = compute_marginal_gain(
                candidate,
                selected_counts=selected_bin_counts,
                targets=target_distribution,
                spatial_diversity_gain=float(spatial_cache[str(info.mp_candidate_id)]),
            )
            candidate["subset_objective_components"].update(gain_components)
            candidate["subset_objective_marginal_gain"] = float(gain_components["subset_objective_marginal_gain"])
            candidate["subset_objective_status"] = "mp_pool_over_budget"
            info.pool_status = "rejected"
            info.pool_reject_reason = "mp_pool_over_budget"
            candidate["pool_status"] = "rejected"
            candidate["pool_reject_reason"] = "mp_pool_over_budget"
            _sync_subset_objective_to_info(info, candidate)
        elif info.pool_status == "rejected" and not info.subset_objective_components:
            candidate = candidates_by_id[str(info.mp_candidate_id)]
            candidate["subset_objective_status"] = str(info.pool_reject_reason or "rejected")
            _sync_subset_objective_to_info(info, candidate)
    return selected

    """按全局 budget-aware gain 从 MP candidate pool 中贪心选择 recipe MPs。"""
    for info in infos:
        if info.pool_status == "candidate" and not bool(info.mp_verified):
            info.pool_status = "rejected"
            info.pool_reject_reason = "mp_pool_invalid"
            info.mp_selection_gain = 0.0
    selected: List[MPPoolCandidateInfo] = []
    selected_families: set[str] = set()
    selected_care_types: set[str] = set()
    fingerprint_cache = {
        str(info.mp_candidate_id): bitmap_fingerprint(info.mp_window["clip_bitmap"])
        for info in infos
    }

    def spatial_novelty(info: MPPoolCandidateInfo) -> float:
        """根据已选 MP 的空间距离估计新候选的空间新颖度。"""
        if not selected:
            return 1.0
        center = _mp_center(info)
        min_distance = min(math.hypot(center[0] - _mp_center(item)[0], center[1] - _mp_center(item)[1]) for item in selected)
        return _clip01(min_distance / max(float(duplicate_radius_um) * 3.0, 1e-9))

    while len(selected) < max(0, int(max_sites)):
        best: MPPoolCandidateInfo | None = None
        best_key: Tuple[float, float, int, int, str] | None = None
        for info in infos:
            if info.pool_status != "candidate":
                continue
            family_id = str(info.care_area_family_id or info.cluster_id)
            care_type = str(info.care_area_type or info.mp_candidate_type)
            family_novelty = 1.0 if family_id not in selected_families else 0.35
            type_novelty = 1.0 if care_type not in selected_care_types else 0.5
            spatial_value = spatial_cache[str(info.mp_candidate_id)]
            gain = float(0.65 * info.mp_priority_score + 0.15 * family_novelty + 0.10 * type_novelty + 0.10 * spatial_novelty(info))
            info.mp_selection_gain = gain
            key = (gain, float(info.mp_priority_score), -int(info.mp_candidate_rank), -int(info.cluster_id), str(info.mp_candidate_id))
            if best_key is None or key > best_key:
                best = info
                best_key = key
        if best is None:
            break
        best.pool_status = "selected"
        best.pool_reject_reason = ""
        selected.append(best)
        selected_families.add(str(best.care_area_family_id or best.cluster_id))
        selected_care_types.add(str(best.care_area_type or best.mp_candidate_type))
        _suppress_pool_duplicates(
            infos,
            selected=best,
            duplicate_radius_um=float(duplicate_radius_um),
            fingerprint_cache=fingerprint_cache,
        )

    for info in infos:
        if info.pool_status == "candidate":
            family_id = str(info.care_area_family_id or info.cluster_id)
            care_type = str(info.care_area_type or info.mp_candidate_type)
            family_novelty = 1.0 if family_id not in selected_families else 0.35
            type_novelty = 1.0 if care_type not in selected_care_types else 0.5
            info.mp_selection_gain = float(
                0.65 * info.mp_priority_score
                + 0.15 * family_novelty
                + 0.10 * type_novelty
                + 0.10 * float(spatial_value)
            )
            info.pool_status = "rejected"
            info.pool_reject_reason = "mp_pool_over_budget"
    return selected


def _pool_candidate_summary(info: MPPoolCandidateInfo) -> Dict[str, Any]:
    """生成 mp_candidate_pool.json 使用的候选摘要。"""
    center = _mp_center(info)
    if not info.ring_context:
        info.ring_context = _ring_context_for_window(info.mp_window)
    return {
        "mp_candidate_id": str(info.mp_candidate_id),
        "mp_candidate_rank": int(info.mp_candidate_rank),
        "pool_status": str(info.pool_status),
        "pool_reject_reason": str(info.pool_reject_reason),
        "mp_selection_gain": float(info.mp_selection_gain),
        "subset_objective_components": dict(info.subset_objective_components),
        "subset_objective_target_bins": dict(info.subset_objective_target_bins),
        "subset_objective_status": str(info.subset_objective_status),
        "subset_objective_marginal_gain": float(info.mp_selection_gain),
        "source_marker_id": str(info.representative_marker_id),
        "care_area_family_id": str(info.care_area_family_id),
        "care_area_instance_id": str(info.care_area_instance_id),
        "care_area_type": str(info.care_area_type),
        "care_area_match_score": float(info.care_area_match_score),
        "care_area_homogeneity_score": float(info.care_area_homogeneity_score),
        "care_area_instance_count": int(info.care_area_instance_count),
        "care_area_seed_marker_id": str(info.care_area_seed_marker_id),
        "care_area_instance_bbox": [float(value) for value in info.care_area_instance_bbox],
        "metrology_priority_score": float(info.metrology_priority_score),
        "metrology_priority_class": str(info.metrology_priority_class),
        "site_reliability_risk": float(info.site_reliability_risk),
        "recipe_waste_penalty": float(info.recipe_waste_penalty),
        "metrology_context_group_id": str(info.metrology_context_group_id),
        "selection_profile_id": str(info.selection_profile_id),
        "metrology_context_components": dict(info.metrology_context_components),
        "hotspot_cluster_id": int(info.cluster_id),
        "member_marker_ids": list(info.marker_ids),
        "mp_x_um": float(center[0]),
        "mp_y_um": float(center[1]),
        "mp_candidate_type": str(info.mp_candidate_type),
        "mp_hotspot_score": float(info.mp_hotspot_score),
        "mp_verified": bool(info.mp_verified),
        "mp_reject_reason": str(info.mp_reject_reason),
        "mp_verification_components": dict(info.mp_verification_components),
        "mp_priority_score": float(info.mp_priority_score),
        "mp_risk_components": dict(info.score_components),
        "mp_discovery_components": dict(info.mp_discovery_components),
        "mp_rule_coverage_audit": dict(getattr(info.mp_discovery, "rule_coverage_audit", {}) or {}),
        "clip_bbox": [float(value) for value in info.mp_window.get("clip_bbox", [])],
        "bitmap_fingerprint": bitmap_fingerprint(info.mp_window["clip_bitmap"]).tolist(),
        "ring_context": dict(info.ring_context),
        "memory_prior_audit": dict(info.memory_prior_audit),
        "graph_context_audit": dict(info.graph_context_audit),
        "evidence_contradiction_audit": dict(info.evidence_contradiction_audit),
        "pattern_taxonomy_audit": dict(info.pattern_taxonomy_audit),
        "expected_feasibility_audit": dict(info.expected_feasibility_audit),
        **dict(info.memory_prior_audit),
    }


def _row_base(
    *,
    site_id: str,
    recipe_status: str,
    reject_reason: str,
    source_marker_id: str,
    hotspot_cluster_id: int,
    mp_x_um: float,
    mp_y_um: float,
    mp_clip_bbox: Sequence[float],
    mp_priority_score: float,
    mp_risk_components: Mapping[str, Any],
    care_area_family_id: str = "",
    care_area_instance_id: str = "",
    care_area_type: str = "",
    care_area_match_score: Any = "",
    care_area_homogeneity_score: Any = "",
    care_area_instance_count: Any = "",
    care_area_seed_marker_id: str = "",
    care_area_instance_bbox: Sequence[float] | None = None,
    metrology_priority_score: Any = "",
    metrology_priority_class: str = "",
    site_reliability_risk: Any = "",
    recipe_waste_penalty: Any = "",
    metrology_context_group_id: str = "",
    selection_profile_id: str = "",
    mp_candidate_id: str = "",
    mp_candidate_rank: Any = "",
    mp_selection_gain: Any = "",
    mp_source_marker_x_um: Any = "",
    mp_source_marker_y_um: Any = "",
    mp_source_marker_distance_um: Any = "",
    mp_candidate_type: str = "",
    mp_hotspot_score: Any = "",
    mp_verified: Any = "",
    mp_reject_reason: str = "",
    mp_discovery_components: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """构建 CSV/JSON 共用的基础 row。"""
    return {
        "site_id": site_id,
        "recipe_status": recipe_status,
        "reject_reason": reject_reason,
        "source_marker_id": source_marker_id,
        "care_area_family_id": str(care_area_family_id),
        "care_area_instance_id": str(care_area_instance_id),
        "care_area_type": str(care_area_type),
        "care_area_match_score": care_area_match_score,
        "care_area_homogeneity_score": care_area_homogeneity_score,
        "care_area_instance_count": care_area_instance_count,
        "care_area_seed_marker_id": str(care_area_seed_marker_id),
        "care_area_instance_bbox": json.dumps([float(value) for value in (care_area_instance_bbox or [])], ensure_ascii=False),
        "metrology_priority_score": metrology_priority_score,
        "metrology_priority_class": str(metrology_priority_class),
        "site_reliability_risk": site_reliability_risk,
        "recipe_waste_penalty": recipe_waste_penalty,
        "metrology_context_group_id": str(metrology_context_group_id),
        "selection_profile_id": str(selection_profile_id),
        "hotspot_cluster_id": int(hotspot_cluster_id),
        "mp_candidate_id": str(mp_candidate_id),
        "mp_candidate_rank": mp_candidate_rank,
        "mp_selection_gain": mp_selection_gain,
        "mp_x_um": float(mp_x_um),
        "mp_y_um": float(mp_y_um),
        "mp_source_marker_x_um": mp_source_marker_x_um,
        "mp_source_marker_y_um": mp_source_marker_y_um,
        "mp_source_marker_distance_um": mp_source_marker_distance_um,
        "mp_candidate_type": str(mp_candidate_type),
        "mp_hotspot_score": mp_hotspot_score,
        "mp_verified": mp_verified,
        "mp_reject_reason": str(mp_reject_reason),
        "mp_discovery_components_json": json.dumps(dict(mp_discovery_components or {}), ensure_ascii=False, default=_json_default),
        "mp_clip_bbox": json.dumps([float(value) for value in mp_clip_bbox], ensure_ascii=False),
        "mp_priority_score": float(mp_priority_score),
        "mp_risk_components_json": json.dumps(dict(mp_risk_components), ensure_ascii=False, default=_json_default),
        "af_x_um": "",
        "af_y_um": "",
        "af_score": "",
        "af_distance_um": "",
        "af_similarity": "",
        "af_reject_reason": "",
        "af_acceptance_checks_json": "",
        "ap_x_um": "",
        "ap_y_um": "",
        "ap_score": "",
        "ap_uniqueness_score": "",
        "ap_peak_count": "",
        "ap_peak_margin_proxy": "",
        "ap_peak_ratio": "",
        "ap_distance_um": "",
        "ap_reject_reason": "",
        "ap_acceptance_checks_json": "",
        "ap_global_duplicate": "",
        "ap_global_duplicate_with": "",
        "ap_global_similarity": "",
        "mp_template_size_um": "",
        "af_template_size_um": "",
        "ap_template_size_um": "",
        "mp_oas": "",
        "af_oas": "",
        "ap_oas": "",
    }


def _pool_candidate_compact_summary(info: MPPoolCandidateInfo) -> Dict[str, Any]:
    """生成 recipe_sites.json 内嵌使用的轻量候选摘要，避免重复写入大 review 字段。"""
    center = _mp_center(info)
    return {
        "mp_candidate_id": str(info.mp_candidate_id),
        "mp_candidate_rank": int(info.mp_candidate_rank),
        "pool_status": str(info.pool_status),
        "pool_reject_reason": str(info.pool_reject_reason),
        "source_marker_id": str(info.representative_marker_id),
        "hotspot_cluster_id": int(info.cluster_id),
        "care_area_family_id": str(info.care_area_family_id),
        "care_area_instance_id": str(info.care_area_instance_id),
        "care_area_type": str(info.care_area_type),
        "care_area_seed_marker_id": str(info.care_area_seed_marker_id),
        "care_area_match_score": float(info.care_area_match_score),
        "care_area_homogeneity_score": float(info.care_area_homogeneity_score),
        "metrology_context_group_id": str(info.metrology_context_group_id),
        "mp_x_um": float(center[0]),
        "mp_y_um": float(center[1]),
        "mp_candidate_type": str(info.mp_candidate_type),
        "mp_hotspot_score": float(info.mp_hotspot_score),
        "mp_priority_score": float(info.mp_priority_score),
        "mp_selection_gain": float(info.mp_selection_gain),
        "subset_objective_marginal_gain": float(info.mp_selection_gain),
        "subset_objective_target_bins": dict(info.subset_objective_target_bins),
        "subset_objective_status": str(info.subset_objective_status),
        "mp_verified": bool(info.mp_verified),
        "mp_reject_reason": str(info.mp_reject_reason),
    }


def _candidate_status_counts(infos: Sequence[MPPoolCandidateInfo]) -> Dict[str, int]:
    """统计同源 marker 下 MP candidate 的 pool 状态分布。"""
    counts: Dict[str, int] = {}
    for info in infos:
        status = str(info.pool_status or "unknown")
        counts[status] = int(counts.get(status, 0)) + 1
    return dict(sorted(counts.items()))


def _sorted_source_marker_candidates(infos: Sequence[MPPoolCandidateInfo]) -> List[MPPoolCandidateInfo]:
    """按 selected、边际收益和优先级排序同源 marker 候选，供 compact review 使用。"""
    return sorted(
        infos,
        key=lambda item: (
            0 if item.pool_status == "selected" else 1,
            -float(item.mp_selection_gain),
            -float(item.mp_priority_score),
            int(item.mp_candidate_rank),
            str(item.mp_candidate_id),
        ),
    )


def _build_source_marker_candidate_index(candidates_by_marker: Mapping[str, Sequence[MPPoolCandidateInfo]]) -> Dict[str, Any]:
    """按 source marker 汇总候选索引，替代在每个 site_summary 中重复嵌入大列表。"""
    markers: Dict[str, Any] = {}
    for marker_id, infos in sorted(candidates_by_marker.items()):
        ordered = _sorted_source_marker_candidates(list(infos))
        markers[str(marker_id)] = {
            "source_marker_id": str(marker_id),
            "total_count": int(len(ordered)),
            "status_counts": _candidate_status_counts(ordered),
            "top_candidates": [_pool_candidate_compact_summary(info) for info in ordered[:10]],
        }
    return {
        "source_marker_count": int(len(markers)),
        "markers": markers,
    }


def _attach_memory_prior_audit(infos: Sequence[MPPoolCandidateInfo], *, store_dir: Path) -> Dict[str, Any]:
    """读取历史 pattern memory store，并把只读 prior audit 挂到当前 MP pool。"""
    pool_without_prior = []
    for info in infos:
        current_prior = dict(info.memory_prior_audit)
        info.memory_prior_audit = {}
        pool_without_prior.append(_pool_candidate_summary(info))
        info.memory_prior_audit = current_prior
    audit = build_memory_prior_audit(pool_without_prior, store_dir=store_dir)
    by_candidate = dict(audit.get("by_candidate_id", {}) or {})
    for info in infos:
        info.memory_prior_audit = dict(by_candidate.get(str(info.mp_candidate_id), {}) or {})
    return audit


def _candidate_review_audit_input(info: MPPoolCandidateInfo) -> Dict[str, Any]:
    """整理低优先级 review audit 所需的 MP candidate 标量输入。"""
    values: Dict[str, Any] = dict(info.raw_components)
    values.update(
        {
            "mp_candidate_id": str(info.mp_candidate_id),
            "mp_candidate_type": str(info.mp_candidate_type),
            "mp_hotspot_score": float(info.mp_hotspot_score),
            "mp_verified": bool(info.mp_verified),
            "metrology_priority_score": float(info.metrology_priority_score),
            "recipe_waste_penalty": float(info.recipe_waste_penalty),
            "site_reliability_risk": float(info.site_reliability_risk),
            "care_area_match_score": float(info.care_area_match_score),
            "care_area_homogeneity_score": float(info.care_area_homogeneity_score),
            "pattern_novelty": float(info.raw_components.get("pattern_novelty", info.raw_components.get("pattern_rarity", 0.0))),
            "mp_discovery_components": dict(info.mp_discovery_components),
        }
    )
    return values


def _attach_review_evidence_audits(infos: Sequence[MPPoolCandidateInfo]) -> Dict[str, Any]:
    """给 MP pool 补充 graph/evidence/taxonomy/feasibility 审查字段，不改变任何排序分数。"""
    graph_contexts: List[Dict[str, Any]] = []
    family_ids: List[str] = []
    for info in infos:
        if not info.ring_context:
            info.ring_context = _ring_context_for_window(info.mp_window)
        info.graph_context_audit = compute_graph_context(info.mp_window.get("clip_bitmap", np.zeros((0, 0), dtype=bool)))
        graph_contexts.append(info.graph_context_audit)
        family_ids.append(str(info.care_area_family_id or info.cluster_id))
    graph_summary = enrich_graph_pool_context(graph_contexts, family_ids)
    for info in infos:
        audit_input = _candidate_review_audit_input(info)
        evidence = compute_evidence_contradiction_audit(
            audit_input,
            graph_context=info.graph_context_audit,
            ring_context=info.ring_context,
            memory_prior=info.memory_prior_audit,
        )
        info.evidence_contradiction_audit = evidence
        info.pattern_taxonomy_audit = compute_pattern_taxonomy_audit(
            audit_input,
            graph_context=info.graph_context_audit,
            evidence_audit=evidence,
            memory_prior=info.memory_prior_audit,
        )
        info.expected_feasibility_audit = compute_expected_feasibility_audit(
            audit_input,
            graph_context=info.graph_context_audit,
            memory_prior=info.memory_prior_audit,
        )
    return {"graph_context_summary": graph_summary}


def _window_pixel_size_um(window: Mapping[str, Any]) -> float:
    """根据窗口 bbox 和 bitmap 尺寸估计 pixel size，供 ring-context audit 使用。"""
    bitmap = np.asarray(window.get("clip_bitmap", []), dtype=bool)
    bbox = window.get("clip_bbox", []) or []
    if bitmap.ndim != 2 or bitmap.size == 0 or len(bbox) < 4:
        return float(DEFAULT_PIXEL_SIZE_NM) / 1000.0
    width_um = abs(float(bbox[2]) - float(bbox[0]))
    height_um = abs(float(bbox[3]) - float(bbox[1]))
    pixel_x = width_um / float(max(1, bitmap.shape[1]))
    pixel_y = height_um / float(max(1, bitmap.shape[0]))
    pixel = max(float(DEFAULT_PIXEL_SIZE_NM) / 1000.0, min(float(pixel_x), float(pixel_y)))
    return float(pixel)


def _ring_context_for_window(window: Mapping[str, Any]) -> Dict[str, Any]:
    """对已有 review window 计算同心环上下文审查字段。"""
    return compute_ring_context(
        window.get("clip_bitmap", np.zeros((0, 0), dtype=bool)),
        pixel_size_um=_window_pixel_size_um(window),
    )


def _materialize_candidate(
    candidate: WindowCandidate | Dict[str, Any],
    *,
    output_path: Path,
    sample_id: str,
    pixel_size_um: float,
) -> str:
    """把 MP/AF/AP candidate 写成 OAS review 文件。"""
    window = candidate.window if isinstance(candidate, WindowCandidate) else candidate
    return _materialize_clip_bitmap(
        np.asarray(window["clip_bitmap"], dtype=bool),
        tuple(float(value) for value in window["clip_bbox"]),
        sample_id,
        output_path,
        pixel_size_um,
    )


def _site_details_from_candidate(candidate: WindowCandidate | None) -> Dict[str, Any] | None:
    """把候选对象转换成 JSON 友好的详情结构。"""
    if candidate is None:
        return None
    return {
        "x_um": float(candidate.x),
        "y_um": float(candidate.y),
        "distance_um": float(candidate.distance_um),
        "score": float(candidate.score),
        "components": dict(candidate.components),
        "acceptance_checks": dict(candidate.acceptance_checks),
        "reject_reason": str(candidate.reject_reason),
        "peak_count": int(candidate.peak_count),
        "accepted": bool(candidate.accepted),
        "fingerprint": bitmap_fingerprint(candidate.window["clip_bitmap"]).tolist(),
        "ring_context": _ring_context_for_window(candidate.window),
        "clip_bbox": [float(value) for value in candidate.window["clip_bbox"]],
    }


def _is_lightweight_expanded_mp(info: MPPoolCandidateInfo) -> bool:
    """判断 selected MP 是否来自 expanded care-area instance 的轻量 rank-0 候选。"""
    audit = getattr(info.mp_discovery, "rule_coverage_audit", {}) or {}
    return bool(audit.get("care_area_lightweight_instance"))


def _refine_selected_expanded_mp(
    info: MPPoolCandidateInfo,
    *,
    window_cache: RecipeWindowCache,
    args: argparse.Namespace,
    source_path: str,
    mp_template_size_um: float,
) -> bool:
    """只对最终入选的 expanded lightweight MP 重跑一次完整 discovery，提升 MP core 定位精度。"""
    if not info.mp_verified or not _is_lightweight_expanded_mp(info):
        return False
    pre_center = _mp_center(info)
    marker_window = window_cache.window(source_path, pre_center, float(mp_template_size_um))
    behavior_risk = float(info.raw_components.get("effective_behavior_risk", info.raw_components.get("behavior_risk", 0.0)))
    discovery = discover_mp_candidates(
        layout_index=window_cache.layout_index(source_path),
        marker_center=pre_center,
        marker_window=marker_window,
        window_size_um=float(mp_template_size_um),
        pixel_size_um=window_cache.pixel_size_um,
        search_radius_um=float(args.mp_search_radius_um),
        step_um=float(args.candidate_step_um),
        behavior_risk=behavior_risk,
        behavior_risk_enabled=bool(getattr(info.mp_discovery, "behavior_risk_enabled", True)),
        min_feature_um=getattr(args, "min_feature_um", None),
        top_k=1,
    )
    candidate = discovery.selected_candidate
    info.raw_components["selected_expanded_mp_refine_attempted"] = 1.0
    info.raw_components["pre_refine_mp_x_um"] = float(pre_center[0])
    info.raw_components["pre_refine_mp_y_um"] = float(pre_center[1])
    info.raw_components["pre_refine_mp_priority_score"] = float(info.mp_priority_score)
    info.score_components["pre_refine_mp_priority_score"] = float(info.mp_priority_score)
    if not candidate.verified:
        info.raw_components["pre_refine_pool_status"] = str(info.pool_status)
        info.raw_components["selected_expanded_mp_refined"] = 0.0
        info.raw_components["selected_expanded_mp_refine_failed"] = 1.0
        info.pool_status = "rejected"
        info.pool_reject_reason = "post_selection_refine_failed"
        info.mp_verified = False
        info.mp_reject_reason = "post_selection_refine_failed"
        info.mp_verification_components = dict(candidate.verification_components)
        info.mp_verification_components["post_selection_refine_failed"] = 1.0
        info.site_reliability_risk = 1.0
        info.recipe_waste_penalty = 1.0
        info.raw_components["site_reliability_risk"] = 1.0
        info.raw_components["recipe_waste_penalty"] = 1.0
        info.raw_components["low_recipe_waste_confidence"] = 0.0
        info.score_components["site_reliability_risk"] = 1.0
        info.score_components["recipe_waste_penalty"] = 1.0
        info.score_components["low_recipe_waste_confidence"] = 0.0
        info.metrology_context_components["post_selection_refine_failed"] = 1.0
        info.mp_discovery_components["post_selection_refine_failed"] = 1.0
        info.mp_discovery_components["post_selection_refine_reject_reason"] = str(candidate.verification_reason or candidate.reject_reason)
        return False

    refined_center = (float(candidate.x), float(candidate.y))
    pattern_rarity = _candidate_pattern_rarity(candidate, int(info.raw_components.get("cluster_coverage", 1.0)))
    localization_confidence = _candidate_localization_confidence(candidate)
    family_representativeness = _clip01(
        float(info.raw_components.get("family_representativeness", _family_representativeness_for_pool(int(info.raw_components.get("cluster_coverage", 1.0)))))
    )
    context = compute_metrology_context(
        care_area_type=str(info.care_area_type),
        bitmap=candidate.window["clip_bitmap"],
        components=candidate.components,
        inherited_behavior_risk=float(behavior_risk),
        family_representativeness=float(family_representativeness),
        pattern_rarity=float(pattern_rarity),
        mp_localization_confidence=float(localization_confidence),
        family_homogeneity=float(info.care_area_homogeneity_score),
        signature_quality=float(info.raw_components.get("care_area_signature_quality", 1.0)),
        mp_verified=bool(candidate.verified),
    )
    info.mp_window = candidate.window
    info.mp_candidate_type = str(candidate.candidate_type)
    info.mp_hotspot_score = float(candidate.score)
    info.mp_verified = bool(candidate.verified)
    info.mp_reject_reason = str(candidate.verification_reason)
    info.mp_verification_components = dict(candidate.verification_components)
    info.mp_discovery = discovery
    info.mp_discovery_components = dict(candidate.components)
    info.mp_discovery_components.update(
        {
            "post_selection_refine": 1.0,
            "pre_refine_mp_x_um": float(pre_center[0]),
            "pre_refine_mp_y_um": float(pre_center[1]),
            "refine_shift_um": float(math.hypot(refined_center[0] - pre_center[0], refined_center[1] - pre_center[1])),
        }
    )
    info.raw_components["mp_hotspot_score"] = float(candidate.score)
    info.raw_components["pattern_rarity"] = float(pattern_rarity)
    info.raw_components["mp_localization_confidence"] = float(localization_confidence)
    info.raw_components["family_representativeness"] = float(family_representativeness)
    info.raw_components["selected_expanded_mp_refined"] = 1.0
    info.raw_components["selected_expanded_mp_refine_failed"] = 0.0
    info.raw_components["post_refine_mp_x_um"] = float(refined_center[0])
    info.raw_components["post_refine_mp_y_um"] = float(refined_center[1])
    info.raw_components["refine_shift_um"] = float(info.mp_discovery_components["refine_shift_um"])
    _apply_metrology_summary_to_pool_info(info, context_to_summary(context))
    info.raw_components["post_refine_priority_stale"] = 1.0
    info.score_components["post_refine_priority_stale"] = 1.0
    return True


def _construct_selected_site(
    info: MPPoolCandidateInfo,
    *,
    site_index: int,
    output_dir: Path,
    review_dir: Path,
    window_cache: RecipeWindowCache,
    args: argparse.Namespace,
    source_marker_candidates: Sequence[MPPoolCandidateInfo],
) -> Dict[str, Any]:
    """围绕一个 selected MP 构造 AF/AP，并返回完整 recipe site row。"""
    site_id = f"site_{int(site_index):04d}"
    source_path = str(info.representative_metadata.get("source_path", ""))
    mp_template_size = float(getattr(args, "mp_template_size_um", 0.0) or args.clip_size)
    af_template_size = float(getattr(args, "af_template_size_um", 0.0) or args.clip_size)
    ap_template_size = float(getattr(args, "ap_template_size_um", 0.0) or args.clip_size)
    _refine_selected_expanded_mp(
        info,
        window_cache=window_cache,
        args=args,
        source_path=source_path,
        mp_template_size_um=float(mp_template_size),
    )
    mp_center = tuple(float(value) for value in info.mp_window["center"][:2])
    info.ring_context = _ring_context_for_window(info.mp_window)
    source_marker_center = tuple(float(value) for value in info.source_marker_center[:2])
    source_marker_distance = math.hypot(mp_center[0] - source_marker_center[0], mp_center[1] - source_marker_center[1])
    sem_shift_limit = getattr(args, "sem_image_shift_limit_um", None)
    af_max_distance = float(sem_shift_limit) if sem_shift_limit is not None and float(sem_shift_limit) > 0.0 else None
    site_dir = review_dir / site_id
    site_dir.mkdir(parents=True, exist_ok=True)
    mp_oas = _materialize_candidate(
        info.mp_window,
        output_path=site_dir / "mp.oas",
        sample_id=f"{site_id}_mp",
        pixel_size_um=window_cache.pixel_size_um,
    )

    af_candidates: List[WindowCandidate] = []
    ap_candidates: List[WindowCandidate] = []
    af: WindowCandidate | None = None
    ap: WindowCandidate | None = None
    if info.mp_verified:
        af_reference_window = window_cache.window(source_path, mp_center, af_template_size)
        af_candidates = _build_search_candidates(
            window_cache,
            source_path=source_path,
            center_xy=mp_center,
            clip_size_um=af_template_size,
            radius_um=float(args.af_search_radius_um),
            step_um=float(args.candidate_step_um),
            min_distance_um=float(args.min_site_distance_um),
            max_distance_um=af_max_distance,
        )
        af = _select_af_candidate(af_candidates, mp_bitmap=af_reference_window["clip_bitmap"], radius_um=float(args.af_search_radius_um))

        ap_candidates = _build_search_candidates(
            window_cache,
            source_path=source_path,
            center_xy=mp_center,
            clip_size_um=ap_template_size,
            radius_um=float(args.ap_search_radius_um),
            step_um=float(args.candidate_step_um),
            min_distance_um=float(args.min_site_distance_um),
        )
        ignore_radius = max(float(args.candidate_step_um) * 2.0, float(ap_template_size) * 0.5)
        ap = _select_ap_candidate(ap_candidates, radius_um=float(args.ap_search_radius_um), ignore_radius_um=ignore_radius)

    reject_reasons: List[str] = []
    if not info.mp_verified:
        if info.mp_reject_reason == "post_selection_refine_failed" or bool(info.mp_discovery_components.get("post_selection_refine_failed", 0.0)):
            reject_reasons.append("post_selection_refine_failed")
        else:
            reject_reasons.append("no_valid_mp")
    else:
        if af is None or not af.accepted:
            reject_reasons.append("no_safe_af")
        if ap is None or not ap.accepted:
            reject_reasons.append("no_unique_ap")
    recipe_status = "selected" if not reject_reasons else "rejected"
    reject_reason = ";".join(reject_reasons)

    row = _row_base(
        site_id=site_id,
        recipe_status=recipe_status,
        reject_reason=reject_reason,
        source_marker_id=info.representative_marker_id,
        care_area_family_id=info.care_area_family_id,
        care_area_instance_id=info.care_area_instance_id,
        care_area_type=info.care_area_type,
        care_area_match_score=float(info.care_area_match_score),
        care_area_homogeneity_score=float(info.care_area_homogeneity_score),
        care_area_instance_count=int(info.care_area_instance_count),
        care_area_seed_marker_id=info.care_area_seed_marker_id,
        care_area_instance_bbox=info.care_area_instance_bbox,
        metrology_priority_score=float(info.metrology_priority_score),
        metrology_priority_class=str(info.metrology_priority_class),
        site_reliability_risk=float(info.site_reliability_risk),
        recipe_waste_penalty=float(info.recipe_waste_penalty),
        metrology_context_group_id=str(info.metrology_context_group_id),
        selection_profile_id=str(info.selection_profile_id),
        hotspot_cluster_id=info.cluster_id,
        mp_x_um=mp_center[0],
        mp_y_um=mp_center[1],
        mp_clip_bbox=info.mp_window["clip_bbox"],
        mp_priority_score=info.mp_priority_score,
        mp_risk_components=info.score_components,
        mp_candidate_id=info.mp_candidate_id,
        mp_candidate_rank=int(info.mp_candidate_rank),
        mp_selection_gain=float(info.mp_selection_gain),
        mp_source_marker_x_um=source_marker_center[0],
        mp_source_marker_y_um=source_marker_center[1],
        mp_source_marker_distance_um=float(source_marker_distance),
        mp_candidate_type=info.mp_candidate_type,
        mp_hotspot_score=float(info.mp_hotspot_score),
        mp_verified=bool(info.mp_verified),
        mp_reject_reason=str(info.mp_reject_reason),
        mp_discovery_components=info.mp_discovery_components,
    )
    row["mp_template_size_um"] = float(mp_template_size)
    row["af_template_size_um"] = float(af_template_size)
    row["ap_template_size_um"] = float(ap_template_size)
    row["mp_oas"] = mp_oas
    if info.mp_verified and af is None:
        row["af_reject_reason"] = "no_candidate"
        row["af_acceptance_checks_json"] = json.dumps({"candidate_found": False}, ensure_ascii=False)
    if info.mp_verified and ap is None:
        row["ap_reject_reason"] = "no_candidate"
        row["ap_acceptance_checks_json"] = json.dumps({"candidate_found": False}, ensure_ascii=False)

    if af is not None:
        row.update(
            {
                "af_x_um": float(af.x),
                "af_y_um": float(af.y),
                "af_score": float(af.score),
                "af_distance_um": float(af.distance_um),
                "af_similarity": float(af.components.get("layout_similarity_to_mp", 0.0)),
                "af_reject_reason": str(af.reject_reason),
                "af_acceptance_checks_json": json.dumps(dict(af.acceptance_checks), ensure_ascii=False, default=_json_default),
            }
        )
        if af.accepted:
            row["af_oas"] = _materialize_candidate(
                af,
                output_path=site_dir / "af.oas",
                sample_id=f"{site_id}_af",
                pixel_size_um=window_cache.pixel_size_um,
            )

    if ap is not None:
        row.update(
            {
                "ap_x_um": float(ap.x),
                "ap_y_um": float(ap.y),
                "ap_score": float(ap.score),
                "ap_uniqueness_score": float(ap.components.get("uniqueness_score", 0.0)),
                "ap_peak_count": int(ap.peak_count),
                "ap_peak_margin_proxy": float(ap.components.get("template_peak_margin", 0.0)),
                "ap_peak_ratio": float(ap.components.get("template_peak_ratio", 0.0)),
                "ap_distance_um": float(ap.distance_um),
                "ap_reject_reason": str(ap.reject_reason),
                "ap_acceptance_checks_json": json.dumps(dict(ap.acceptance_checks), ensure_ascii=False, default=_json_default),
            }
        )
        if ap.accepted:
            row["ap_oas"] = _materialize_candidate(
                ap,
                output_path=site_dir / "ap.oas",
                sample_id=f"{site_id}_ap",
                pixel_size_um=window_cache.pixel_size_um,
            )

    source_marker_ordered_candidates = _sorted_source_marker_candidates(source_marker_candidates)
    details = {
        "site": dict(row),
        "mp_candidate": {
            "marker_id": info.representative_marker_id,
            "cluster_id": int(info.cluster_id),
            "mp_candidate_id": str(info.mp_candidate_id),
            "mp_candidate_rank": int(info.mp_candidate_rank),
            "mp_selection_gain": float(info.mp_selection_gain),
            "subset_objective_components": dict(info.subset_objective_components),
            "subset_objective_target_bins": dict(info.subset_objective_target_bins),
            "subset_objective_status": str(info.subset_objective_status),
            "subset_objective_marginal_gain": float(info.mp_selection_gain),
            "pool_status": str(info.pool_status),
            "pool_reject_reason": str(info.pool_reject_reason),
            "care_area_family_id": str(info.care_area_family_id),
            "care_area_instance_id": str(info.care_area_instance_id),
            "care_area_type": str(info.care_area_type),
            "care_area_match_score": float(info.care_area_match_score),
            "care_area_homogeneity_score": float(info.care_area_homogeneity_score),
            "care_area_instance_count": int(info.care_area_instance_count),
            "care_area_seed_marker_id": str(info.care_area_seed_marker_id),
            "care_area_instance_bbox": [float(value) for value in info.care_area_instance_bbox],
            "metrology_priority_score": float(info.metrology_priority_score),
            "metrology_priority_class": str(info.metrology_priority_class),
            "site_reliability_risk": float(info.site_reliability_risk),
            "recipe_waste_penalty": float(info.recipe_waste_penalty),
            "metrology_context_group_id": str(info.metrology_context_group_id),
            "selection_profile_id": str(info.selection_profile_id),
            "metrology_context_components": dict(info.metrology_context_components),
            "member_marker_ids": list(info.marker_ids),
            "source_marker_center": [float(source_marker_center[0]), float(source_marker_center[1])],
            "mp_center": [float(mp_center[0]), float(mp_center[1])],
            "mp_source_marker_distance_um": float(source_marker_distance),
            "mp_candidate_type": str(info.mp_candidate_type),
            "mp_hotspot_score": float(info.mp_hotspot_score),
            "mp_verified": bool(info.mp_verified),
            "mp_reject_reason": str(info.mp_reject_reason),
            "mp_verification_components": dict(info.mp_verification_components),
            "raw_components": dict(info.raw_components),
            "score_components": dict(info.score_components),
            "discovery_components": dict(info.mp_discovery_components),
            "discovery": info.mp_discovery.to_summary(),
            "ring_context": dict(info.ring_context),
            "memory_prior_audit": dict(info.memory_prior_audit),
            "graph_context_audit": dict(info.graph_context_audit),
            "evidence_contradiction_audit": dict(info.evidence_contradiction_audit),
            "pattern_taxonomy_audit": dict(info.pattern_taxonomy_audit),
            "expected_feasibility_audit": dict(info.expected_feasibility_audit),
            "source_marker_candidate_total_count": int(len(source_marker_ordered_candidates)),
            "source_marker_candidate_status_counts": _candidate_status_counts(source_marker_ordered_candidates),
            "source_marker_top_candidates": [
                _pool_candidate_compact_summary(candidate) for candidate in source_marker_ordered_candidates[:10]
            ],
            "clip_bbox": [float(value) for value in info.mp_window["clip_bbox"]],
        },
        "af_candidate": _site_details_from_candidate(af),
        "ap_candidate": _site_details_from_candidate(ap),
        "af_candidate_count": int(len(af_candidates)),
        "ap_candidate_count": int(len(ap_candidates)),
    }
    row["_details"] = details
    return row


def _rejected_member_rows(
    infos: Sequence[MPPoolCandidateInfo],
    *,
    selected_infos: Sequence[MPPoolCandidateInfo],
    constructed_infos: Sequence[MPPoolCandidateInfo] | None = None,
    metadata_by_marker: Mapping[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """为未作为 recipe MP 的 markers 生成 rejected rows。"""
    effective_selected = [info for info in selected_infos if info.pool_status == "selected" and bool(info.mp_verified)]
    selected_cluster_ids = {int(info.cluster_id) for info in effective_selected}
    selected_reps = {str(info.representative_marker_id) for info in (constructed_infos or selected_infos)}
    processed_clusters: set[int] = set()
    rows: List[Dict[str, Any]] = []
    for info in infos:
        if int(info.cluster_id) in processed_clusters:
            continue
        processed_clusters.add(int(info.cluster_id))
        reason = "covered_by_representative" if int(info.cluster_id) in selected_cluster_ids else "over_budget"
        for marker_id in info.marker_ids:
            if marker_id in selected_reps:
                continue
            metadata = metadata_by_marker.get(marker_id, info.representative_metadata)
            center = metadata.get("marker_center", info.representative_metadata.get("marker_center", [0.0, 0.0]))
            clip_bbox = metadata.get("clip_bbox", info.representative_metadata.get("clip_bbox", [0.0, 0.0, 0.0, 0.0]))
            rows.append(
                _row_base(
                    site_id=f"rejected_{len(rows):04d}",
                    recipe_status="rejected",
                    reject_reason=reason,
                    source_marker_id=str(marker_id),
                    hotspot_cluster_id=int(info.cluster_id),
                    care_area_family_id=info.care_area_family_id,
                    care_area_instance_id=info.care_area_instance_id,
                    care_area_type=info.care_area_type,
                    care_area_match_score=float(info.care_area_match_score),
                    care_area_homogeneity_score=float(info.care_area_homogeneity_score),
                    care_area_instance_count=int(info.care_area_instance_count),
                    care_area_seed_marker_id=info.care_area_seed_marker_id,
                    care_area_instance_bbox=info.care_area_instance_bbox,
                    metrology_priority_score=float(info.metrology_priority_score),
                    metrology_priority_class=str(info.metrology_priority_class),
                    site_reliability_risk=float(info.site_reliability_risk),
                    recipe_waste_penalty=float(info.recipe_waste_penalty),
                    metrology_context_group_id=str(info.metrology_context_group_id),
                    selection_profile_id=str(info.selection_profile_id),
                    mp_x_um=float(center[0]),
                    mp_y_um=float(center[1]),
                    mp_clip_bbox=clip_bbox,
                    mp_priority_score=float(info.mp_priority_score),
                    mp_risk_components=info.score_components,
                    mp_candidate_id="",
                    mp_candidate_rank="",
                    mp_selection_gain="",
                    mp_source_marker_x_um=float(center[0]),
                    mp_source_marker_y_um=float(center[1]),
                    mp_source_marker_distance_um=0.0,
                    mp_candidate_type=reason,
                    mp_hotspot_score="",
                    mp_verified="",
                    mp_reject_reason="",
                    mp_discovery_components={},
                )
            )
    return rows


def _effective_selected_infos_from_rows(
    selected_infos: Sequence[MPPoolCandidateInfo],
    rows: Sequence[Mapping[str, Any]],
) -> List[MPPoolCandidateInfo]:
    """根据最终 recipe row 状态筛出真正形成有效 recipe site 的 MP 候选。"""
    selected_ids = {
        str(row.get("mp_candidate_id", ""))
        for row in rows
        if row.get("recipe_status") == "selected" and row.get("mp_candidate_id")
    }
    return [
        info
        for info in selected_infos
        if str(info.mp_candidate_id) in selected_ids and info.pool_status == "selected" and bool(info.mp_verified)
    ]


def _rejected_care_area_seed_rows(rejected_seeds: Sequence[RejectedCareAreaSeed]) -> List[Dict[str, Any]]:
    """为没有形成 care-area family 的 seed markers 生成 rejected provenance rows。"""
    rows: List[Dict[str, Any]] = []
    for seed in rejected_seeds:
        metadata = dict(seed.representative_metadata)
        center = metadata.get("marker_center", [0.0, 0.0])
        clip_bbox = metadata.get("clip_bbox", [0.0, 0.0, 0.0, 0.0])
        rows.append(
            _row_base(
                site_id=f"rejected_care_area_{len(rows):04d}",
                recipe_status="rejected",
                reject_reason=str(seed.reason),
                source_marker_id=str(seed.marker_id),
                hotspot_cluster_id=int(seed.cluster_id),
                mp_x_um=float(center[0]),
                mp_y_um=float(center[1]),
                mp_clip_bbox=clip_bbox,
                mp_priority_score=0.0,
                mp_risk_components={},
                mp_candidate_type=str(seed.reason),
                mp_source_marker_x_um=float(center[0]),
                mp_source_marker_y_um=float(center[1]),
                mp_source_marker_distance_um=0.0,
                care_area_seed_marker_id=str(seed.marker_id),
            )
        )
    return rows


def _append_reject_reason(row: Dict[str, Any], reason: str) -> None:
    """给 recipe row 追加去重后的 reject reason。"""
    reasons = [item for item in str(row.get("reject_reason", "")).split(";") if item]
    if reason not in reasons:
        reasons.append(reason)
    row["reject_reason"] = ";".join(reasons)


def _apply_global_ap_uniqueness(rows: Sequence[Dict[str, Any]], *, duplicate_threshold: float = 0.92) -> int:
    """在 recipe 层面对已选 AP 做全局相似性检查，避免多个 site 使用高度重复的 AP。"""
    selected: List[Tuple[int, np.ndarray, float]] = []
    for index, row in enumerate(rows):
        if row.get("recipe_status") != "selected":
            continue
        details = row.get("_details", {})
        ap_details = details.get("ap_candidate") if isinstance(details, Mapping) else None
        if not ap_details or not ap_details.get("accepted"):
            continue
        fingerprint = np.asarray(ap_details.get("fingerprint", []), dtype=np.float32)
        if fingerprint.size == 0:
            continue
        selected.append((index, fingerprint, float(row.get("ap_score", 0.0) or 0.0)))

    rejected: set[int] = set()
    for left_pos in range(len(selected)):
        left_index, left_fp, left_score = selected[left_pos]
        if left_index in rejected:
            continue
        for right_index, right_fp, right_score in selected[left_pos + 1:]:
            if right_index in rejected:
                continue
            similarity = _clip01(float(np.dot(left_fp, right_fp)))
            if similarity < float(duplicate_threshold):
                continue
            loser_index, winner_index = (right_index, left_index) if left_score >= right_score else (left_index, right_index)
            rows[loser_index]["recipe_status"] = "rejected"
            rows[loser_index]["ap_global_duplicate"] = True
            rows[loser_index]["ap_global_duplicate_with"] = str(rows[winner_index].get("site_id", ""))
            rows[loser_index]["ap_global_similarity"] = float(similarity)
            _append_reject_reason(rows[loser_index], "ap_global_duplicate")
            details = rows[loser_index].get("_details", {})
            if isinstance(details, dict):
                details["site"] = dict(_public_row(rows[loser_index]))
                if isinstance(details.get("ap_candidate"), dict):
                    details["ap_candidate"]["global_duplicate"] = True
                    details["ap_candidate"]["global_duplicate_with"] = str(rows[winner_index].get("site_id", ""))
                    details["ap_candidate"]["global_similarity"] = float(similarity)
            rejected.add(loser_index)
    return len(rejected)


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """写出 recipe site CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_FIELDS})


def _public_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """移除内部详情字段，生成 JSON/CSV 共用的公开 row。"""
    return {key: value for key, value in row.items() if not str(key).startswith("_")}


def _write_site_summaries(rows: Sequence[Dict[str, Any]], review_dir: Path) -> None:
    """在所有全局检查完成后写出 per-site summary，避免 review 文件状态滞后。"""
    for row in rows:
        details = row.get("_details")
        if not isinstance(details, dict):
            continue
        site_id = str(row.get("site_id", ""))
        if not site_id:
            continue
        details["site"] = dict(_public_row(row))
        site_dir = review_dir / site_id
        site_dir.mkdir(parents=True, exist_ok=True)
        with (site_dir / "site_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(details, handle, indent=2, ensure_ascii=False, default=_json_default)


def run_recipe_selector(args: argparse.Namespace) -> Dict[str, Any]:
    """执行 recipe prototype 全流程并写出 CSV/JSON/review。"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    review_dir = output_dir / "recipe_review"
    if review_dir.exists():
        shutil.rmtree(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)

    backend_args = _copy_backend_args(args, output_dir)
    backend_result = mp_backend.run_notrain_mp_selection(backend_args)

    apply_layer_operations = bool(args.apply_layer_ops or args.register_op)
    layer_processor = mp_backend._make_layer_processor(args.register_op or [])
    window_cache = RecipeWindowCache(
        marker_layer=str(args.marker_layer),
        clip_size_um=float(args.clip_size),
        output_dir=output_dir,
        apply_layer_operations=apply_layer_operations,
        layer_processor=layer_processor,
        recursive_input=bool(args.recursive_input),
    )
    infos, care_area_result = _build_mp_candidate_pool(
        backend_result,
        window_cache=window_cache,
        clip_size_um=float(args.clip_size),
        mp_search_radius_um=float(args.mp_search_radius_um),
        candidate_step_um=float(args.candidate_step_um),
        mp_candidates_per_marker=int(args.mp_candidates_per_marker),
        max_care_area_instances_per_family=int(args.max_care_area_instances_per_family),
        min_feature_um=getattr(args, "min_feature_um", None),
        mp_template_size_um=getattr(args, "mp_template_size_um", None),
    )
    mp_template_size = float(getattr(args, "mp_template_size_um", None) or args.clip_size)
    duplicate_radius = max(float(args.candidate_step_um), 0.25 * mp_template_size)
    mp_pool_preduplicate_count = _pre_dedup_mp_candidate_pool(infos, duplicate_radius_um=duplicate_radius)
    all_family_risk_zero = all(float(family.behavior_risk) <= 1e-12 for family in care_area_result.families)
    _score_mp_candidates(infos, all_risk_zero=all_family_risk_zero)
    ranked_infos = sorted(infos, key=lambda item: (-item.mp_priority_score, item.cluster_id, item.representative_marker_id, item.mp_candidate_rank))
    pattern_memory_store_dir = Path(__file__).resolve().parent / "pattern_memory_store"
    memory_prior_audit = _attach_memory_prior_audit(ranked_infos, store_dir=pattern_memory_store_dir)
    memory_prior_summary = dict(memory_prior_audit.get("summary", {}) or {})
    review_evidence_pre_summary = _attach_review_evidence_audits(ranked_infos)
    selected_infos = _select_mp_candidate_pool(infos, max_sites=int(args.max_sites), duplicate_radius_um=duplicate_radius)
    ranked_infos = sorted(
        infos,
        key=lambda item: (
            0 if item.pool_status == "selected" else 1,
            -float(item.mp_selection_gain),
            -float(item.mp_priority_score),
            item.cluster_id,
            str(item.representative_marker_id),
            int(item.mp_candidate_rank),
        ),
    )
    metadata_by_marker = _metadata_by_marker(backend_result)
    candidates_by_marker: Dict[str, List[MPPoolCandidateInfo]] = {}
    for info in ranked_infos:
        candidates_by_marker.setdefault(str(info.representative_marker_id), []).append(info)
    source_marker_candidate_index = _build_source_marker_candidate_index(candidates_by_marker)

    rows: List[Dict[str, Any]] = []
    for site_index, info in enumerate(selected_infos):
        row = _construct_selected_site(
            info,
            site_index=site_index,
            output_dir=output_dir,
            review_dir=review_dir,
            window_cache=window_cache,
            args=args,
            source_marker_candidates=candidates_by_marker.get(str(info.representative_marker_id), []),
        )
        rows.append(row)
    ap_global_duplicate_count = _apply_global_ap_uniqueness(rows)
    for row in rows:
        details = row.get("_details")
        if isinstance(details, dict):
            update_evidence_with_site_outcome(details)
    effective_selected_infos = _effective_selected_infos_from_rows(selected_infos, rows)
    rows.extend(_rejected_care_area_seed_rows(care_area_result.rejected_seeds))
    rows.extend(
        _rejected_member_rows(
            ranked_infos,
            selected_infos=effective_selected_infos,
            constructed_infos=selected_infos,
            metadata_by_marker=metadata_by_marker,
        )
    )
    _write_site_summaries(rows, review_dir)
    site_details = [dict(row.get("_details", {})) for row in rows if row.get("_details")]

    public_rows = [_public_row(row) for row in rows]
    selected_count = sum(1 for row in public_rows if row.get("recipe_status") == "selected")
    rejected_count = sum(1 for row in public_rows if row.get("recipe_status") == "rejected")
    mp_candidate_pool_full_summary = [_pool_candidate_summary(info) for info in ranked_infos]
    mp_candidate_pool_compact_summary = [_pool_candidate_compact_summary(info) for info in ranked_infos]
    care_area_summary = care_area_result.to_summary()
    metrology_context_audit = build_metrology_context_audit(
        care_area_groups=care_area_summary,
        mp_candidate_pool=mp_candidate_pool_full_summary,
        rows=public_rows,
    )
    review_evidence_audit = build_review_evidence_audit(
        mp_candidate_pool=mp_candidate_pool_full_summary,
        site_details=site_details,
    )
    subset_target_distribution = next((info.subset_objective_targets for info in ranked_infos if info.subset_objective_targets), {})
    subset_objective_audit = build_subset_objective_audit(
        mp_candidate_pool=mp_candidate_pool_full_summary,
        site_details=site_details,
        target_distribution=subset_target_distribution,
    )
    pattern_memory_summary = export_pattern_memory(
        mp_candidate_pool=mp_candidate_pool_full_summary,
        rows=public_rows,
        output_dir=review_dir / "pattern_memory_export",
    )
    if bool(getattr(args, "skip_pattern_memory_store_append", False)):
        pattern_memory_store_summary = {
            "skipped": True,
            "store_dir": str(pattern_memory_store_dir),
            "records_jsonl": "",
            "vectors_npz": "",
            "memory_audit_json": "",
            "ring_outcome_audit_json": "",
            "record_count": 0,
            "added_record_count": 0,
            "duplicate_skipped_count": 0,
            "vector_shape": [0, 0],
            "estimated_disk_bytes": 0,
        }
    else:
        pattern_memory_store_summary = append_pattern_memory_export(
            export_dir=Path(pattern_memory_summary["pattern_memory_export_dir"]),
            store_dir=pattern_memory_store_dir,
        )
    metrology_summary = dict(metrology_context_audit.get("summary", {}) or {})
    review_evidence_summary = dict(review_evidence_audit.get("summary", {}) or {})
    subset_objective_summary = dict(subset_objective_audit.get("summary", {}) or {})
    result = {
        "pipeline_mode": PIPELINE_MODE,
        "config": {
            "input_path": str(args.input_path),
            "marker_layer": str(args.marker_layer),
            "behavior_manifest": str(args.behavior_manifest),
            "clip_size": float(args.clip_size),
            "mp_template_size_um": float(getattr(args, "mp_template_size_um", None) or args.clip_size),
            "af_template_size_um": float(getattr(args, "af_template_size_um", None) or args.clip_size),
            "ap_template_size_um": float(getattr(args, "ap_template_size_um", None) or args.clip_size),
            "min_feature_um": getattr(args, "min_feature_um", None),
            "sem_image_shift_limit_um": getattr(args, "sem_image_shift_limit_um", None),
            "max_sites": int(args.max_sites),
            "mp_coverage_target": float(args.mp_coverage_target),
            "mp_search_radius_um": float(args.mp_search_radius_um),
            "mp_candidates_per_marker": int(args.mp_candidates_per_marker),
            "max_care_area_instances_per_family": int(args.max_care_area_instances_per_family),
            "af_search_radius_um": float(args.af_search_radius_um),
            "ap_search_radius_um": float(args.ap_search_radius_um),
            "candidate_step_um": float(args.candidate_step_um),
            "min_site_distance_um": float(args.min_site_distance_um),
            "recursive_input": bool(args.recursive_input),
            "apply_layer_operations": apply_layer_operations,
            "skip_pattern_memory_store_append": bool(getattr(args, "skip_pattern_memory_store_append", False)),
        },
        "summary": {
            "backend_cluster_count": int(len(backend_result.get("clusters", []) or [])),
            "care_area_family_count": int(len(care_area_result.families)),
            "care_area_instance_count": int(sum(len(family.instances) for family in care_area_result.families)),
            "rejected_care_area_seed_count": int(len(care_area_result.rejected_seeds)),
            "mp_candidate_pool_count": int(len(infos)),
            "selected_mp_candidate_count": int(len(selected_infos)),
            "effective_selected_mp_candidate_count": int(len(effective_selected_infos)),
            "selected_recipe_site_count": int(selected_count),
            "rejected_row_count": int(rejected_count),
            "eligible_mp_candidate_count": int(sum(1 for info in infos if info.mp_verified)),
            "invalid_mp_candidate_count": int(sum(1 for info in infos if not info.mp_verified)),
            "mp_pool_preduplicate_count": int(mp_pool_preduplicate_count),
            "selected_expanded_mp_refine_attempted_count": int(sum(1 for info in selected_infos if float(info.raw_components.get("selected_expanded_mp_refine_attempted", 0.0)) > 0.0)),
            "selected_expanded_mp_refined_count": int(sum(1 for info in selected_infos if float(info.raw_components.get("selected_expanded_mp_refined", 0.0)) > 0.0)),
            "selected_expanded_mp_refine_failed_count": int(sum(1 for info in selected_infos if float(info.raw_components.get("selected_expanded_mp_refine_failed", 0.0)) > 0.0)),
            "ap_global_duplicate_count": int(ap_global_duplicate_count),
            "metrology_context_group_count": int(metrology_summary.get("metrology_context_group_count", 0)),
            "selected_metrology_context_group_count": int(metrology_summary.get("selected_metrology_context_group_count", 0)),
            "selected_by_metrology_priority_class": dict(metrology_summary.get("selected_by_metrology_priority_class", {}) or {}),
            "selected_by_metrology_context_group": dict(metrology_summary.get("selected_by_metrology_context_group", {}) or {}),
            "pattern_memory_record_count": int(pattern_memory_summary.get("pattern_memory_record_count", 0)),
            "pattern_memory_vector_shape": list(pattern_memory_summary.get("pattern_memory_vector_shape", [0, 0])),
            "pattern_memory_estimated_disk_bytes": int(pattern_memory_summary.get("pattern_memory_estimated_disk_bytes", 0)),
            "pattern_memory_store_record_count": int(pattern_memory_store_summary.get("record_count", 0)),
            "pattern_memory_store_added_record_count": int(pattern_memory_store_summary.get("added_record_count", 0)),
            "pattern_memory_store_duplicate_skipped_count": int(pattern_memory_store_summary.get("duplicate_skipped_count", 0)),
            "pattern_memory_store_vector_shape": list(pattern_memory_store_summary.get("vector_shape", [0, 0])),
            "pattern_memory_store_estimated_disk_bytes": int(pattern_memory_store_summary.get("estimated_disk_bytes", 0)),
            "pattern_memory_store_append_skipped": bool(pattern_memory_store_summary.get("skipped", False)),
            "memory_prior_candidate_count": int(memory_prior_summary.get("candidate_count", 0)),
            "memory_prior_store_record_count": int(memory_prior_summary.get("store_record_count", 0)),
            "memory_prior_candidates_with_neighbors": int(memory_prior_summary.get("candidates_with_neighbors", 0)),
            "memory_prior_avg_neighbor_count": float(memory_prior_summary.get("avg_neighbor_count", 0.0)),
            "memory_prior_avg_confidence": float(memory_prior_summary.get("avg_prior_confidence", 0.0)),
            "review_graph_avg_mp_graph_rarity": float(review_evidence_summary.get("avg_mp_graph_rarity", 0.0)),
            "review_avg_expected_recipe_feasibility_proxy": float(review_evidence_summary.get("avg_expected_recipe_feasibility_proxy", 0.0)),
            "review_taxonomy_by_class": dict(review_evidence_summary.get("taxonomy_by_class", {}) or {}),
            "review_static_contradiction_by_tag": dict(review_evidence_summary.get("static_contradiction_by_tag", {}) or {}),
            "review_outcome_contradiction_by_tag": dict(review_evidence_summary.get("outcome_contradiction_by_tag", {}) or {}),
            "subset_objective_score": float(subset_objective_summary.get("subset_objective_score", 0.0)),
            "subset_objective_gap_count": int(subset_objective_summary.get("subset_objective_gap_count", 0)),
            "subset_objective_high_value_missed_count": int(subset_objective_summary.get("subset_objective_high_value_missed_count", 0)),
            "subset_objective_high_risk_non_executable_count": int(subset_objective_summary.get("subset_objective_high_risk_non_executable_count", 0)),
            "selected_subset_objective_by_category": dict(subset_objective_summary.get("selected_subset_objective_by_category", {}) or {}),
            "total_output_rows": int(len(public_rows)),
            "review_dir": str(review_dir),
        },
        "backend_summary": backend_result.get("result_summary", {}),
        "care_area_groups": care_area_summary,
        "mp_candidate_pool": mp_candidate_pool_compact_summary,
        "metrology_context_audit": metrology_context_audit,
        "memory_prior_audit": memory_prior_audit,
        "review_evidence_pre_summary": review_evidence_pre_summary,
        "review_evidence_audit": review_evidence_audit,
        "subset_objective_audit": subset_objective_audit,
        "pattern_memory_export": pattern_memory_summary,
        "pattern_memory_store": pattern_memory_store_summary,
        "source_marker_candidate_index": {
            "source_marker_count": int(source_marker_candidate_index.get("source_marker_count", 0)),
        },
        "sites": public_rows,
        "site_details": site_details,
    }

    csv_path = output_dir / "recipe_sites.csv"
    json_path = output_dir / "recipe_sites.json"
    backend_path = output_dir / "_notrain_backend.json"
    pool_path = review_dir / "mp_candidate_pool.json"
    care_area_path = review_dir / "care_area_groups.json"
    source_marker_index_path = review_dir / "source_marker_candidate_index.json"
    metrology_audit_path = review_dir / "metrology_context_audit.json"
    review_evidence_audit_path = review_dir / "review_evidence_audit.json"
    subset_objective_audit_path = review_dir / "subset_objective_audit.json"
    pattern_memory_dir = review_dir / "pattern_memory_export"
    result["outputs"] = {
        "recipe_sites_csv": str(csv_path),
        "recipe_sites_json": str(json_path),
        "recipe_review_dir": str(review_dir),
        "mp_candidate_pool_json": str(pool_path),
        "care_area_groups_json": str(care_area_path),
        "source_marker_candidate_index_json": str(source_marker_index_path),
        "metrology_context_audit_json": str(metrology_audit_path),
        "review_evidence_audit_json": str(review_evidence_audit_path),
        "subset_objective_audit_json": str(subset_objective_audit_path),
        "pattern_memory_export_dir": str(pattern_memory_dir),
        "pattern_memory_records_jsonl": str(pattern_memory_dir / "records.jsonl"),
        "pattern_memory_vectors_npz": str(pattern_memory_dir / "vectors.npz"),
        "pattern_memory_store_dir": str(pattern_memory_store_summary.get("store_dir", "")),
        "pattern_memory_store_manifest": str(Path(pattern_memory_store_summary.get("store_dir", "")) / "manifest.json"),
        "pattern_memory_store_records_jsonl": str(pattern_memory_store_summary.get("records_jsonl", "")),
        "pattern_memory_store_vectors_npz": str(pattern_memory_store_summary.get("vectors_npz", "")),
        "pattern_memory_store_memory_audit_json": str(pattern_memory_store_summary.get("memory_audit_json", "")),
        "pattern_memory_store_ring_outcome_audit_json": str(pattern_memory_store_summary.get("ring_outcome_audit_json", "")),
        "notrain_backend_json": str(backend_path),
    }
    _write_csv(public_rows, csv_path)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, default=_json_default)
    with backend_path.open("w", encoding="utf-8") as handle:
        json.dump(backend_result, handle, indent=2, ensure_ascii=False, default=_json_default)
    with pool_path.open("w", encoding="utf-8") as handle:
        json.dump(mp_candidate_pool_full_summary, handle, indent=2, ensure_ascii=False, default=_json_default)
    with care_area_path.open("w", encoding="utf-8") as handle:
        json.dump(result["care_area_groups"], handle, indent=2, ensure_ascii=False, default=_json_default)
    with source_marker_index_path.open("w", encoding="utf-8") as handle:
        json.dump(source_marker_candidate_index, handle, indent=2, ensure_ascii=False, default=_json_default)
    with metrology_audit_path.open("w", encoding="utf-8") as handle:
        json.dump(result["metrology_context_audit"], handle, indent=2, ensure_ascii=False, default=_json_default)
    with review_evidence_audit_path.open("w", encoding="utf-8") as handle:
        json.dump(result["review_evidence_audit"], handle, indent=2, ensure_ascii=False, default=_json_default)
    with subset_objective_audit_path.open("w", encoding="utf-8") as handle:
        json.dump(result["subset_objective_audit"], handle, indent=2, ensure_ascii=False, default=_json_default)
    return result


def _build_parser() -> argparse.ArgumentParser:
    """构建 recipe prototype CLI，并说明第一版输入边界。"""
    epilog = """
示例:

python recipe_site_selector.py input.oas --marker-layer 999/0 --behavior-manifest behavior_inputs --output-dir recipe_out --max-sites 100

注意:
- 第一版必须提供 behavior manifest，且每条样本必须有 aerial_npz。
- MP pool 来自 seeded care-area expansion；不是任意窗口 full-chip blind scan。
- `source_marker_id` 表示 seed marker provenance，不代表 MP 必定位于 marker 邻域内。
- 输出以 recipe site 为主，cluster/member/representative 只作为 provenance。
- `--register-op` 会自动启用 layer operations；层格式固定为 layer/datatype，例如 1/0。
"""
    parser = argparse.ArgumentParser(
        description="CD-SEM hotspot recipe site prototype",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_path", help="输入 OASIS 文件或目录")
    parser.add_argument("--marker-layer", required=True, help="hotspot marker 层，格式 layer/datatype，例如 999/0")
    parser.add_argument("--behavior-manifest", required=True, help="behavior JSONL manifest 路径，或 preprocess 输出目录")
    parser.add_argument("--output-dir", required=True, help="recipe_sites.csv/json 和 review 目录输出位置")
    parser.add_argument("--clip-size", type=float, default=1.35, help="MP/AF/AP clip 边长，单位 um")
    parser.add_argument("--mp-template-size-um", type=float, default=None, help="MP core template 边长；不提供时使用 --clip-size")
    parser.add_argument("--af-template-size-um", type=float, default=None, help="AF template 边长；不提供时使用 --clip-size")
    parser.add_argument("--ap-template-size-um", type=float, default=None, help="AP template 边长；不提供时使用 --clip-size")
    parser.add_argument("--max-sites", type=int, default=100, help="最多进入 AF/AP 构造的 selected MP 数量")
    parser.add_argument("--mp-coverage-target", type=float, default=0.985, help="no-train backend 的 coverage selection 目标")
    parser.add_argument("--mp-search-radius-um", type=float, default=0.8, help="提取 seed family 和 care-area instance 内 MP candidate 的搜索半径，单位 um")
    parser.add_argument("--mp-candidates-per-marker", type=int, default=5, help="每个 representative marker 进入全局池的 MP candidate 数量")
    parser.add_argument("--max-care-area-instances-per-family", type=int, default=80, help="每个 seeded care-area family 最多展开的同类实例数")
    parser.add_argument("--min-feature-um", type=float, default=None, help="可选工艺最小特征尺寸，用于约束 critical spacing gap")
    parser.add_argument("--af-search-radius-um", type=float, default=3.0, help="AF candidate 搜索半径，单位 um")
    parser.add_argument("--sem-image-shift-limit-um", type=float, default=None, help="可选 SEM image-shift 可达半径；AF 候选超出该距离会被硬过滤")
    parser.add_argument("--ap-search-radius-um", type=float, default=5.0, help="AP candidate 搜索半径，单位 um")
    parser.add_argument("--candidate-step-um", type=float, default=0.2, help="MP/AF/AP candidate 滑窗步长，单位 um")
    parser.add_argument("--min-site-distance-um", type=float, default=0.5, help="AF/AP 与 MP core 的最小距离，单位 um")
    parser.add_argument("--recursive-input", action="store_true", help="输入为目录时递归搜索 .oas 文件")
    parser.add_argument("--apply-layer-ops", action="store_true", help="运行前应用注册的 boolean layer operations")
    parser.add_argument("--skip-pattern-memory-store-append", action="store_true", help="只导出本次 run 的 pattern memory，不追加到持久 pattern_memory_store")
    parser.add_argument(
        "--register-op",
        action="append",
        nargs=4,
        metavar=("SOURCE_LAYER", "TARGET_LAYER", "OPERATION", "RESULT_LAYER"),
        help="注册层操作规则，例如 --register-op 1/0 2/0 subtract 10/0",
    )
    return parser


def main() -> int:
    """命令行入口：执行 prototype 并打印关键输出路径。"""
    parser = _build_parser()
    args = parser.parse_args()
    try:
        result = run_recipe_selector(args)
        outputs = result.get("outputs", {})
        summary = result.get("summary", {})
        print("CD-SEM recipe selector 完成")
        print(f"selected recipe sites: {summary.get('selected_recipe_site_count', 0)}")
        print(f"rejected rows: {summary.get('rejected_row_count', 0)}")
        print(f"recipe_sites.csv: {outputs.get('recipe_sites_csv')}")
        print(f"recipe_sites.json: {outputs.get('recipe_sites_json')}")
        print(f"recipe_review: {outputs.get('recipe_review_dir')}")
        return 0
    except Exception as exc:
        print(f"运行失败: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
