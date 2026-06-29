#!/usr/bin/env python3
"""Hu 2020 风格的 seeded care-area expansion 模块。

本模块位于 `recipe_site_selector.py` 和 `mp_candidate_generator.py` 之间，负责把
representative hotspot marker 周边的高风险几何模式抽象成 care-area family，再在同一
OAS 内搜索 look-alike instances。它不是完整的 Anchor Printed Pattern Database，也不做
监督 ML risk model；当前只落地 Hu 2020 中最适合 prototype 的核心思想：

1. 从已知 hotspot marker 周边提取 seed weak-pattern family。
2. 用 fragment / bbox anchor 生成 DDD-lite 风格的全 OAS candidate anchor table。
3. 按 bitmap similarity、fragment signature similarity 和 anchor type match 搜索同类实例。
4. 输出 homogeneous care-area groups，供后续 MP discovery、budget selection 和 AF/AP 使用。

适用边界：
- 只做 seeded full-OAS expansion，不做 full-chip blind window scan。
- family 风险分数来自 seed marker；下游 MP pool 会按 match/homogeneity 做 risk 衰减。
- 不重新切 candidate-level behavior image。
- 每个 family 的实例数用固定 cap 控制，避免第一版 review 输出失控。
- look-alike 搜索在 rasterize 前会用 type/signature/source strength 做 cheap pre-score，并保留
  少量 tile-balanced 探索样本；audit 会按 top/tile/fallback 来源统计命中率。它不是
  seed-distance hard prune，因此远处同类结构仍可入选。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Sequence, Set, Tuple

import numpy as np

from layout_utils import LayoutIndex, _query_candidate_ids, bitmap_fingerprint, rasterize_centered_window
from metrology_context import MetrologyContext, compute_metrology_context, context_to_summary
from mp_candidate_generator import (
    MPDiscoveryResult,
    MPCandidate,
    _add_density_transition_anchors,
    _add_fragment_anchors,
    _add_geometry_anchors,
    _add_pair_anchors,
    _bbox_center,
    _candidate_type,
    _distance,
    _shifted_iou,
    discover_mp_candidates,
)


CARE_AREA_MATCH_THRESHOLD = 0.78
MAX_ANCHOR_ELEMENTS = 2000


@dataclass
class CareAreaInstance:
    """单个 care-area family 在版图中的一个同类实例。"""

    instance_id: str
    family_id: str
    instance_rank: int
    source_path: str
    center: Tuple[float, float]
    bbox: Tuple[float, float, float, float]
    window: Dict[str, Any]
    care_area_type: str
    match_score: float
    bitmap_similarity: float
    bitmap_shifted_iou: float
    bitmap_fingerprint_similarity: float
    fragment_signature_similarity: float
    anchor_type_match: float
    homogeneity_score: float
    raw_sources: List[str]
    signature: List[float]
    signature_quality: float = 1.0
    fingerprint: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.float32))
    is_seed_instance: bool = False
    selection_source: str = ""
    metrology_priority_score: float = 0.0
    metrology_priority_class: str = "low"
    site_reliability_risk: float = 0.0
    recipe_waste_penalty: float = 0.0
    metrology_context_group_id: str = ""
    selection_profile_id: str = ""
    metrology_context_components: Dict[str, float] = field(default_factory=dict)

    def to_summary(self) -> Dict[str, Any]:
        """返回不含 bitmap 大数组的 JSON review 摘要。"""
        return {
            "care_area_instance_id": str(self.instance_id),
            "care_area_family_id": str(self.family_id),
            "instance_rank": int(self.instance_rank),
            "source_path": str(self.source_path),
            "center_um": [float(self.center[0]), float(self.center[1])],
            "bbox": [float(value) for value in self.bbox],
            "care_area_type": str(self.care_area_type),
            "care_area_match_score": float(self.match_score),
            "bitmap_similarity": float(self.bitmap_similarity),
            "bitmap_shifted_iou": float(self.bitmap_shifted_iou),
            "bitmap_fingerprint_similarity": float(self.bitmap_fingerprint_similarity),
            "fragment_signature_similarity": float(self.fragment_signature_similarity),
            "anchor_type_match": float(self.anchor_type_match),
            "care_area_homogeneity_score": float(self.homogeneity_score),
            "raw_sources": list(self.raw_sources),
            "signature": [float(value) for value in self.signature],
            "signature_quality": float(self.signature_quality),
            "is_seed_instance": bool(self.is_seed_instance),
            "selection_source": str(self.selection_source),
            "metrology_priority_score": float(self.metrology_priority_score),
            "metrology_priority_class": str(self.metrology_priority_class),
            "site_reliability_risk": float(self.site_reliability_risk),
            "recipe_waste_penalty": float(self.recipe_waste_penalty),
            "metrology_context_group_id": str(self.metrology_context_group_id),
            "selection_profile_id": str(self.selection_profile_id),
            "metrology_context_components": dict(self.metrology_context_components),
        }


@dataclass
class CareAreaFamily:
    """由一个 representative marker 提炼出的 high-risk look-alike pattern family。"""

    family_id: str
    family_rank: int
    seed_marker_id: str
    cluster_id: int
    source_path: str
    marker_ids: List[str]
    representative_metadata: Dict[str, Any]
    cluster: Dict[str, Any]
    seed_center: Tuple[float, float]
    seed_candidate: MPCandidate
    seed_discovery: MPDiscoveryResult
    care_area_type: str
    behavior_risk: float
    cluster_size: int
    fingerprint: np.ndarray
    signature: List[float]
    signature_gap_norm_um: float
    instances: List[CareAreaInstance] = field(default_factory=list)
    seed_behavior_risk: float = 0.0
    homogeneity_score: float = 0.0
    candidate_anchor_count: int = 0
    rejected_instance_count: int = 0
    expanded_instance_count: int = 0
    is_singleton_family: bool = True
    expansion_confidence: float = 0.0
    merged_seed_family_ids: List[str] = field(default_factory=list)
    merged_cluster_ids: List[int] = field(default_factory=list)
    merged_behavior_risk_values: List[float] = field(default_factory=list)
    anchor_table_audit: Dict[str, Any] = field(default_factory=dict)
    instance_reject_reasons: Dict[str, int] = field(default_factory=dict)
    metrology_priority_score: float = 0.0
    metrology_priority_class: str = "low"
    site_reliability_risk: float = 0.0
    recipe_waste_penalty: float = 0.0
    metrology_context_group_id: str = ""
    selection_profile_id: str = ""
    metrology_context_components: Dict[str, float] = field(default_factory=dict)

    def to_summary(self) -> Dict[str, Any]:
        """返回 care_area_groups.json 使用的 family 摘要。"""
        return {
            "care_area_family_id": str(self.family_id),
            "family_rank": int(self.family_rank),
            "care_area_type": str(self.care_area_type),
            "seed_marker_id": str(self.seed_marker_id),
            "hotspot_cluster_id": int(self.cluster_id),
            "member_marker_ids": list(self.marker_ids),
            "source_path": str(self.source_path),
            "seed_center_um": [float(self.seed_center[0]), float(self.seed_center[1])],
            "behavior_risk": float(self.behavior_risk),
            "seed_behavior_risk": float(self.seed_behavior_risk),
            "merged_behavior_risk_values": [float(value) for value in self.merged_behavior_risk_values],
            "merged_cluster_ids": [int(value) for value in self.merged_cluster_ids],
            "cluster_size": int(self.cluster_size),
            "signature": [float(value) for value in self.signature],
            "signature_gap_norm_um": float(self.signature_gap_norm_um),
            "care_area_homogeneity_score": float(self.homogeneity_score),
            "care_area_instance_count": int(len(self.instances)),
            "candidate_anchor_count": int(self.candidate_anchor_count),
            "rejected_instance_count": int(self.rejected_instance_count),
            "expanded_instance_count": int(self.expanded_instance_count),
            "is_singleton_family": bool(self.is_singleton_family),
            "care_area_expansion_confidence": float(self.expansion_confidence),
            "merged_seed_family_ids": list(self.merged_seed_family_ids),
            "anchor_table_audit": dict(self.anchor_table_audit),
            "instance_reject_reasons": dict(self.instance_reject_reasons),
            "metrology_priority_score": float(self.metrology_priority_score),
            "metrology_priority_class": str(self.metrology_priority_class),
            "site_reliability_risk": float(self.site_reliability_risk),
            "recipe_waste_penalty": float(self.recipe_waste_penalty),
            "metrology_context_group_id": str(self.metrology_context_group_id),
            "selection_profile_id": str(self.selection_profile_id),
            "metrology_context_components": dict(self.metrology_context_components),
            "seed_candidate": self.seed_candidate.to_summary(),
            "seed_discovery": self.seed_discovery.to_summary(),
            "instances": [instance.to_summary() for instance in self.instances],
        }


@dataclass
class RejectedCareAreaSeed:
    """没有形成 care-area family 的 representative marker provenance。"""

    marker_id: str
    cluster_id: int
    marker_ids: List[str]
    representative_metadata: Dict[str, Any]
    cluster: Dict[str, Any]
    reason: str
    seed_discovery: MPDiscoveryResult

    def to_summary(self) -> Dict[str, Any]:
        """返回被拒绝 seed 的轻量审查信息。"""
        center = self.representative_metadata.get("marker_center", [0.0, 0.0])
        return {
            "marker_id": str(self.marker_id),
            "hotspot_cluster_id": int(self.cluster_id),
            "member_marker_ids": list(self.marker_ids),
            "source_path": str(self.representative_metadata.get("source_path", "")),
            "marker_center_um": [float(center[0]), float(center[1])],
            "reject_reason": str(self.reason),
            "seed_discovery": self.seed_discovery.to_summary(),
        }


@dataclass
class CareAreaExpansionResult:
    """care-area expansion 的完整结果，供 recipe 主流程使用。"""

    families: List[CareAreaFamily]
    rejected_seeds: List[RejectedCareAreaSeed]
    anchor_table_audits_by_source: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_summary(self) -> Dict[str, Any]:
        """返回 recipe_sites.json 使用的结构化摘要。"""
        return {
            "care_area_family_count": int(len(self.families)),
            "care_area_instance_count": int(sum(len(family.instances) for family in self.families)),
            "rejected_seed_count": int(len(self.rejected_seeds)),
            "anchor_table_audits_by_source": dict(self.anchor_table_audits_by_source),
            "families": [family.to_summary() for family in self.families],
            "rejected_seeds": [seed.to_summary() for seed in self.rejected_seeds],
        }


def _clip01(value: float) -> float:
    """把数值限制在 0-1 区间。"""
    if not math.isfinite(float(value)):
        return 0.0
    return float(min(1.0, max(0.0, float(value))))


def _apply_metrology_context(target: Any, context: MetrologyContext) -> None:
    """把 CD-SEM 量测 context 字段写入 family 或 instance 对象。"""
    summary = context_to_summary(context)
    target.metrology_priority_score = float(summary["metrology_priority_score"])
    target.metrology_priority_class = str(summary["metrology_priority_class"])
    target.site_reliability_risk = float(summary["site_reliability_risk"])
    target.recipe_waste_penalty = float(summary["recipe_waste_penalty"])
    target.metrology_context_group_id = str(summary["metrology_context_group_id"])
    target.selection_profile_id = str(summary["selection_profile_id"])
    target.metrology_context_components = dict(summary["metrology_context_components"])


def _family_representativeness(instance_count: int, cluster_size: int) -> float:
    """估计一个 family 的量测代表性，实例越多、cluster 越大越值得抽样。"""
    coverage = max(1, int(instance_count), int(cluster_size))
    return _clip01(math.log1p(float(coverage)) / math.log1p(20.0))


def _signature_components(signature: Sequence[float]) -> Dict[str, float]:
    """把固定顺序 signature 转回 metrology scoring 可读的几何 component。"""
    values = [float(value) for value in signature]
    padded = values + [0.0 for _ in range(max(0, 10 - len(values)))]
    return {
        "fragment_context_score": _clip01(padded[0]),
        "fragment_corner_score": _clip01(padded[1]),
        "fragment_line_end_score": _clip01(padded[2]),
        "fragment_facing_pair_score": _clip01(padded[3]),
        "layout_complexity": _clip01(padded[6]),
        "proposal_voting": _clip01(padded[7]),
        "density_transition_score": _clip01(padded[8]),
        "density_local": _clip01(padded[9]),
    }


def _refresh_metrology_contexts(family: CareAreaFamily) -> None:
    """在 family homogeneity 确定后刷新 family 和所有 instances 的量测 context。"""
    representativeness = _family_representativeness(len(family.instances), family.cluster_size)
    seed_pattern_rarity = float(family.seed_candidate.components.get("local_rarity", family.seed_candidate.components.get("pattern_rarity", 0.0)))
    seed_context = compute_metrology_context(
        care_area_type=family.care_area_type,
        bitmap=family.seed_candidate.window["clip_bitmap"],
        components=family.seed_candidate.components,
        inherited_behavior_risk=float(family.behavior_risk),
        family_representativeness=float(representativeness),
        pattern_rarity=float(seed_pattern_rarity),
        mp_localization_confidence=float(family.seed_candidate.components.get("proposal_voting", 1.0)),
        family_homogeneity=float(family.homogeneity_score),
        signature_quality=_signature_quality(family.signature),
        mp_verified=bool(family.seed_candidate.verified),
    )
    _apply_metrology_context(family, seed_context)

    for instance in family.instances:
        if bool(instance.is_seed_instance):
            context = seed_context
        else:
            components = _signature_components(instance.signature)
            effective_risk = float(family.behavior_risk) * _clip01(float(instance.match_score) * float(instance.homogeneity_score))
            context = compute_metrology_context(
                care_area_type=instance.care_area_type,
                bitmap=instance.window["clip_bitmap"],
                components=components,
                inherited_behavior_risk=float(effective_risk),
                family_representativeness=float(representativeness),
                pattern_rarity=_clip01(float(seed_pattern_rarity) * float(instance.match_score)),
                mp_localization_confidence=_clip01(0.50 * float(instance.match_score) + 0.50 * float(instance.anchor_type_match)),
                family_homogeneity=float(instance.homogeneity_score),
                signature_quality=float(instance.signature_quality),
                mp_verified=bool(instance.match_score >= CARE_AREA_MATCH_THRESHOLD),
            )
        _apply_metrology_context(instance, context)


def _sanitize_id(value: str) -> str:
    """生成适合写入文件名和 JSON key 的稳定短标识。"""
    chars = [char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value)]
    return "".join(chars).strip("_") or "unknown"


def care_area_type_from_candidate_type(candidate_type: str) -> str:
    """把 MP anchor 类型映射到 Hu-style care area type。"""
    candidate_type = str(candidate_type)
    if candidate_type in {"fragment_facing_pair_anchor", "critical_spacing_anchor"}:
        return "spacing"
    if candidate_type in {"fragment_line_end_anchor", "line_end_anchor"}:
        return "line_end"
    if candidate_type in {"fragment_corner_anchor", "corner_or_jog_anchor"}:
        return "corner_jog"
    if candidate_type == "density_transition_anchor":
        return "density_transition"
    return ""


def _gap_norm_um(min_feature_um: float | None) -> float:
    """根据工艺最小特征尺寸确定 gap signature 的归一化尺度。"""
    if min_feature_um is not None and float(min_feature_um) > 0.0:
        return max(0.05, min(0.60, 3.0 * float(min_feature_um)))
    return 0.30


def _signature_from_components(components: Mapping[str, Any], *, gap_norm_um: float = 0.30) -> List[float]:
    """从 MP discovery components 或 raw anchor metrics 中提取可比较的 fragment signature。"""
    internal_gap = float(components.get("internal_facing_distance_um", 0.0) or 0.0)
    external_gap = float(components.get("external_facing_distance_um", 0.0) or 0.0)
    gap_norm = max(float(gap_norm_um), 1e-9)
    return [
        _clip01(float(components.get("fragment_context_score", 0.0) or 0.0)),
        _clip01(float(components.get("fragment_corner_score", 0.0) or 0.0)),
        _clip01(float(components.get("fragment_line_end_score", 0.0) or 0.0)),
        _clip01(float(components.get("fragment_facing_pair_score", 0.0) or 0.0)),
        _clip01(internal_gap / gap_norm),
        _clip01(external_gap / gap_norm),
        _clip01(float(components.get("layout_complexity", 0.0) or 0.0)),
        _clip01(float(components.get("proposal_voting", components.get("voting_confidence", 0.0)) or 0.0)),
        _clip01(float(components.get("density_transition_score", 0.0) or 0.0)),
        _clip01(float(components.get("density_local", 0.0) or 0.0)),
    ]


def _signature_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """计算两个固定长度 signature 的轻量相似度。"""
    if not left or not right:
        return 0.0
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    size = min(left_arr.size, right_arr.size)
    if size == 0:
        return 0.0
    return _clip01(1.0 - float(np.mean(np.abs(left_arr[:size] - right_arr[:size]))))


def _signature_quality(signature: Sequence[float]) -> float:
    """估计 signature 中真正携带几何信息的比例，用于稀疏 signature 的保守兜底。"""
    if not signature:
        return 0.0
    values = np.asarray(signature, dtype=np.float64)
    if values.size == 0:
        return 0.0
    informative = np.count_nonzero(np.abs(values) > 1e-6)
    return _clip01(float(informative) / float(values.size))


def _bitmap_similarity_parts(left_bitmap: np.ndarray, right_bitmap: np.ndarray, right_fingerprint: np.ndarray, left_fingerprint: np.ndarray) -> Dict[str, float]:
    """返回 care-area bitmap 匹配的细分分数，避免单一粗特征直接放行。"""
    shifted = _shifted_iou(left_bitmap, right_bitmap, max_shift_px=2)
    coarse = _clip01(float(np.dot(left_fingerprint, right_fingerprint)))
    return {
        "bitmap_similarity": _clip01(0.60 * shifted + 0.40 * coarse),
        "bitmap_shifted_iou": float(shifted),
        "bitmap_fingerprint_similarity": float(coarse),
    }


def _bitmap_similarity(left_bitmap: np.ndarray, right_bitmap: np.ndarray, right_fingerprint: np.ndarray, left_fingerprint: np.ndarray) -> float:
    """返回 weighted bitmap similarity，供 NMS 等只需要单值的路径使用。"""
    return float(_bitmap_similarity_parts(left_bitmap, right_bitmap, right_fingerprint, left_fingerprint)["bitmap_similarity"])


def _anchor_type_match(seed_type: str, candidate_type: str) -> float:
    """在同一 care-area type 内区分 fragment 精确匹配和 bbox/proxy 近似匹配。"""
    seed = str(seed_type)
    candidate = str(candidate_type)
    if seed == candidate:
        return 1.0
    if care_area_type_from_candidate_type(seed) != care_area_type_from_candidate_type(candidate):
        return 0.0
    if seed.startswith("fragment_") or candidate.startswith("fragment_"):
        return 0.85
    return 0.70


def _local_elements(layout_index: LayoutIndex, center: Tuple[float, float], radius_um: float) -> List[Mapping[str, Any]]:
    """查询某个几何中心附近的 pattern 元素，用于局部 anchor proposal。"""
    cx, cy = float(center[0]), float(center[1])
    bbox = (cx - float(radius_um), cy - float(radius_um), cx + float(radius_um), cy + float(radius_um))
    ids = _query_candidate_ids(layout_index, bbox)
    return [layout_index.indexed_elements[int(index)] for index in ids]


def _bbox_union(items: Sequence[Mapping[str, Any]]) -> Tuple[float, float, float, float]:
    """计算一组 indexed element 的 bbox 并集；空输入返回零面积 bbox。"""
    if not items:
        return (0.0, 0.0, 0.0, 0.0)
    x0 = min(float(item["bbox"][0]) for item in items)
    y0 = min(float(item["bbox"][1]) for item in items)
    x1 = max(float(item["bbox"][2]) for item in items)
    y1 = max(float(item["bbox"][3]) for item in items)
    return (x0, y0, x1, y1)


def _layout_bbox(layout_index: LayoutIndex) -> Tuple[float, float, float, float]:
    """返回 layout index 覆盖的整体 bbox。"""
    if not layout_index.indexed_elements:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        float(np.min(layout_index.bbox_x0)),
        float(np.min(layout_index.bbox_y0)),
        float(np.max(layout_index.bbox_x1)),
        float(np.max(layout_index.bbox_y1)),
    )


def _bbox_area(bbox: Sequence[float]) -> float:
    """计算 bbox 面积，非法或退化 bbox 按 0 处理。"""
    if len(bbox) < 4:
        return 0.0
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def _tile_key(center: Tuple[float, float], layout_bbox: Sequence[float], tile_count: int) -> Tuple[int, int]:
    """把元素中心映射到固定空间 tile，供全 OAS 均匀采样使用。"""
    x0, y0, x1, y1 = (float(value) for value in layout_bbox)
    width = max(float(x1) - float(x0), 1e-9)
    height = max(float(y1) - float(y0), 1e-9)
    tx = min(int(tile_count) - 1, max(0, int((float(center[0]) - x0) / width * int(tile_count))))
    ty = min(int(tile_count) - 1, max(0, int((float(center[1]) - y0) / height * int(tile_count))))
    return (tx, ty)


def _anchor_interest_score(item: Mapping[str, Any]) -> float:
    """估计单个图元作为 care-area anchor 起点的几何价值。"""
    x0, y0, x1, y1 = (float(value) for value in item.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    if width <= 1e-9 or height <= 1e-9:
        return 0.0
    min_dim = min(width, height)
    max_dim = max(width, height)
    aspect_score = _clip01((max_dim / max(min_dim, 1e-9)) / 8.0)
    narrow_score = _clip01(0.08 / max(min_dim, 1e-9))
    compact_score = _clip01(0.02 / max(width * height, 1e-9))
    return _clip01(0.45 * aspect_score + 0.40 * narrow_score + 0.15 * compact_score)


def _seed_distance_lookup(
    items: Sequence[Mapping[str, Any]],
    seed_centers: Sequence[Tuple[float, float]],
    *,
    chunk_size: int = 4096,
) -> Dict[int, float]:
    """批量计算图元到最近 seed 的距离，避免排序时反复遍历 seed 列表。"""
    if not seed_centers:
        return {}
    seed_arr = np.asarray(seed_centers, dtype=np.float64)
    result: Dict[int, float] = {}
    for start in range(0, len(items), int(chunk_size)):
        chunk = list(items[start : start + int(chunk_size)])
        centers = np.asarray([_bbox_center(item["bbox"]) for item in chunk], dtype=np.float64)
        diff = centers[:, None, :] - seed_arr[None, :, :]
        distances = np.min(np.hypot(diff[:, :, 0], diff[:, :, 1]), axis=1)
        for item, distance in zip(chunk, distances.tolist()):
            result[id(item)] = float(distance)
    return result


def _select_anchor_elements(
    layout_index: LayoutIndex,
    *,
    seed_centers: Sequence[Tuple[float, float]] | None = None,
) -> Tuple[List[Mapping[str, Any]], Dict[str, Any]]:
    """用 deterministic tile-balanced sampling 选择参与 anchor table 的图元。"""
    total = int(len(layout_index.indexed_elements))
    layout_box = _layout_bbox(layout_index)
    seed_centers = list(seed_centers or [])
    if total <= MAX_ANCHOR_ELEMENTS:
        elements = list(layout_index.indexed_elements)
        processed_box = _bbox_union(elements)
        tile_count = 1
        tile_ratio = 1.0 if total > 0 else 0.0
    else:
        tile_count = max(4, min(32, int(math.ceil(math.sqrt(MAX_ANCHOR_ELEMENTS / 2.0)))))
        buckets: Dict[Tuple[int, int], List[Mapping[str, Any]]] = {}
        for item in layout_index.indexed_elements:
            key = _tile_key(_bbox_center(item["bbox"]), layout_box, tile_count)
            buckets.setdefault(key, []).append(item)
        seed_distance_by_id = _seed_distance_lookup(layout_index.indexed_elements, seed_centers)
        for bucket in buckets.values():
            bucket.sort(
                key=lambda item: (
                    -_anchor_interest_score(item),
                    seed_distance_by_id.get(id(item), 0.0),
                    _bbox_center(item["bbox"])[1],
                    _bbox_center(item["bbox"])[0],
                )
            )
        elements = []
        positions = {key: 0 for key in buckets}
        keys = sorted(buckets)
        while len(elements) < MAX_ANCHOR_ELEMENTS:
            added = False
            for key in keys:
                index = positions[key]
                bucket = buckets[key]
                if index >= len(bucket):
                    continue
                elements.append(bucket[index])
                positions[key] = index + 1
                added = True
                if len(elements) >= MAX_ANCHOR_ELEMENTS:
                    break
            if not added:
                break
        processed_box = _bbox_union(elements)
        processed_tiles = {_tile_key(_bbox_center(item["bbox"]), layout_box, tile_count) for item in elements}
        all_tiles = {_tile_key(_bbox_center(item["bbox"]), layout_box, tile_count) for item in layout_index.indexed_elements}
        tile_ratio = float(len(processed_tiles)) / float(max(1, len(all_tiles)))
    area_ratio = _bbox_area(processed_box) / max(_bbox_area(layout_box), 1e-9)
    audit = {
        "layout_bbox": [float(value) for value in layout_box],
        "processed_bbox": [float(value) for value in processed_box],
        "processed_area_ratio": float(_clip01(area_ratio)),
        "tile_count_per_axis": int(tile_count),
        "tile_coverage_ratio": float(_clip01(tile_ratio)),
        "tile_interest_sorted": bool(total > MAX_ANCHOR_ELEMENTS),
        "seed_proximity_weighted": bool(seed_centers),
        "seed_center_count": int(len(seed_centers)),
        "seed_distance_precomputed": bool(seed_centers and total > MAX_ANCHOR_ELEMENTS),
    }
    return elements, audit


def _build_seeded_anchor_table(
    *,
    layout_index: LayoutIndex,
    target_types: Set[str],
    local_radius_um: float,
    step_um: float,
    pixel_size_um: float,
    min_feature_um: float | None,
    seed_centers: Sequence[Tuple[float, float]] | None = None,
) -> Tuple[Dict[Tuple[int, int], Dict[str, Any]], Dict[str, Any]]:
    """围绕全 OAS pattern elements 生成与 seed family 类型相关的 anchor table 和审查摘要。"""
    raw: Dict[Tuple[int, int], Dict[str, Any]] = {}
    total_elements = int(len(layout_index.indexed_elements))
    elements, coverage_audit = _select_anchor_elements(layout_index, seed_centers=seed_centers)
    for item in elements:
        center = _bbox_center(item["bbox"])
        local = _local_elements(layout_index, center, float(local_radius_um))
        if target_types & {"spacing", "line_end", "corner_jog"}:
            _add_fragment_anchors(
                raw,
                elements=local,
                center_xy=center,
                radius_um=float(local_radius_um),
                step_um=float(step_um),
                min_feature_um=min_feature_um,
            )
            _add_geometry_anchors(raw, elements=local, center_xy=center, radius_um=float(local_radius_um), step_um=float(step_um))
            _add_pair_anchors(
                raw,
                elements=local,
                center_xy=center,
                radius_um=float(local_radius_um),
                step_um=float(step_um),
                min_feature_um=min_feature_um,
            )
        if "density_transition" in target_types:
            _add_density_transition_anchors(
                raw,
                layout_index=layout_index,
                center_xy=center,
                radius_um=float(local_radius_um),
                step_um=float(step_um),
                pixel_size_um=float(pixel_size_um),
            )
    source_counts: Dict[str, int] = {}
    for item in raw.values():
        for source in item.get("sources", []) or []:
            source_name = str(source)
            source_counts[source_name] = int(source_counts.get(source_name, 0)) + 1
    audit = {
        "total_layout_element_count": total_elements,
        "anchor_table_element_limit": int(MAX_ANCHOR_ELEMENTS),
        "processed_element_count": int(len(elements)),
        "anchor_table_cap_hit": bool(total_elements > MAX_ANCHOR_ELEMENTS),
        "target_care_area_types": sorted(str(value) for value in target_types),
        "candidate_anchor_count": int(len(raw)),
        "candidate_anchor_count_by_source": dict(sorted(source_counts.items())),
    }
    audit.update(coverage_audit)
    return raw, audit


def _make_seed_instance(family: CareAreaFamily) -> CareAreaInstance:
    """把 seed MP candidate 包装成 family 的第一个 care-area instance。"""
    bbox = tuple(float(value) for value in family.seed_candidate.window.get("clip_bbox", [0.0, 0.0, 0.0, 0.0]))
    return CareAreaInstance(
        instance_id=f"{family.family_id}__inst_0000",
        family_id=family.family_id,
        instance_rank=0,
        source_path=family.source_path,
        center=(float(family.seed_candidate.x), float(family.seed_candidate.y)),
        bbox=bbox,
        window=family.seed_candidate.window,
        care_area_type=family.care_area_type,
        match_score=1.0,
        bitmap_similarity=1.0,
        bitmap_shifted_iou=1.0,
        bitmap_fingerprint_similarity=1.0,
        fragment_signature_similarity=1.0,
        anchor_type_match=1.0,
        homogeneity_score=1.0,
        raw_sources=list(family.seed_candidate.sources),
        signature=list(family.signature),
        signature_quality=_signature_quality(family.signature),
        fingerprint=np.asarray(family.fingerprint, dtype=np.float32),
        is_seed_instance=True,
        selection_source="seed",
    )


def _instance_from_anchor(
    family: CareAreaFamily,
    *,
    layout_index: LayoutIndex,
    raw_anchor: Mapping[str, Any],
    window_size_um: float,
    pixel_size_um: float,
) -> Tuple[CareAreaInstance | None, str]:
    """把 raw anchor 转成 care-area instance candidate，并返回拒绝原因。"""
    sources = sorted(str(source) for source in raw_anchor.get("sources", []) or [])
    candidate_type = _candidate_type(sources)
    care_area_type = care_area_type_from_candidate_type(candidate_type)
    if care_area_type != family.care_area_type:
        return None, "type_mismatch"
    center = (float(raw_anchor["x"]), float(raw_anchor["y"]))
    signature = _signature_from_components(raw_anchor.get("metrics", {}) or {}, gap_norm_um=float(family.signature_gap_norm_um))
    signature_score = _signature_similarity(family.signature, signature)
    signature_quality = min(_signature_quality(family.signature), _signature_quality(signature))
    anchor_type_match = _anchor_type_match(family.seed_candidate.candidate_type, candidate_type)
    # signature 信息足够时先做轻量门控，避免明显不匹配的 anchor 继续触发 rasterize。
    if family.care_area_type == "density_transition":
        if signature_score < 0.35:
            return None, "signature_gate_reject"
    else:
        if signature_quality >= 0.50 and signature_score < 0.55:
            return None, "signature_gate_reject"
        if signature_quality < 0.50 and anchor_type_match < 0.85:
            return None, "signature_sparse_gate_reject"
    window = rasterize_centered_window(layout_index, center, float(window_size_um), float(pixel_size_um))
    bitmap = np.asarray(window["clip_bitmap"], dtype=bool)
    if bitmap.size == 0 or not np.any(bitmap):
        return None, "empty_bitmap"
    fingerprint = bitmap_fingerprint(bitmap)
    bitmap_parts = _bitmap_similarity_parts(family.seed_candidate.window["clip_bitmap"], bitmap, fingerprint, family.fingerprint)
    bitmap_score = float(bitmap_parts["bitmap_similarity"])
    if family.care_area_type == "density_transition":
        match_score = _clip01(0.50 * bitmap_score + 0.35 * signature_score + 0.15 * anchor_type_match)
        if bitmap_score < 0.45:
            return None, "bitmap_gate_reject"
        if signature_score < 0.35:
            return None, "signature_gate_reject"
    else:
        if signature_quality >= 0.50:
            match_score = _clip01(0.50 * bitmap_score + 0.35 * signature_score + 0.15 * anchor_type_match)
        else:
            match_score = _clip01(0.75 * bitmap_score + 0.25 * anchor_type_match)
        if bitmap_score < 0.55:
            return None, "bitmap_gate_reject"
        if signature_quality >= 0.50 and signature_score < 0.55:
            return None, "signature_gate_reject"
        if signature_quality < 0.50 and (bitmap_score < 0.68 or anchor_type_match < 0.85):
            return None, "signature_sparse_gate_reject"
    if match_score < CARE_AREA_MATCH_THRESHOLD:
        return None, "match_score_reject"
    bbox = tuple(float(value) for value in window.get("clip_bbox", [0.0, 0.0, 0.0, 0.0]))
    return CareAreaInstance(
        instance_id="",
        family_id=family.family_id,
        instance_rank=-1,
        source_path=family.source_path,
        center=center,
        bbox=bbox,
        window=window,
        care_area_type=care_area_type,
        match_score=float(match_score),
        bitmap_similarity=float(bitmap_score),
        bitmap_shifted_iou=float(bitmap_parts["bitmap_shifted_iou"]),
        bitmap_fingerprint_similarity=float(bitmap_parts["bitmap_fingerprint_similarity"]),
        fragment_signature_similarity=float(signature_score),
        anchor_type_match=float(anchor_type_match),
        homogeneity_score=0.0,
        raw_sources=sources,
        signature=signature,
        signature_quality=float(signature_quality),
        fingerprint=np.asarray(fingerprint, dtype=np.float32),
    ), ""


def _anchor_source_strength(sources: Sequence[str]) -> float:
    """按 anchor 来源估计无需 rasterize 的几何可信度。"""
    weights = {
        "fragment_facing_pair_anchor": 1.0,
        "critical_spacing_anchor": 1.0,
        "fragment_line_end_anchor": 0.9,
        "line_end_anchor": 0.85,
        "fragment_corner_anchor": 0.85,
        "corner_or_jog_anchor": 0.8,
        "density_transition_anchor": 0.75,
    }
    return _clip01(max((float(weights.get(str(source), 0.5)) for source in sources), default=0.5))


def _prescore_typed_anchors(
    family: CareAreaFamily,
    raw_anchors: Mapping[Any, Mapping[str, Any]],
) -> List[Tuple[float, Mapping[str, Any]]]:
    """批量计算同 care-area type raw anchors 的 cheap pre-score。"""
    anchors: List[Mapping[str, Any]] = []
    type_scores: List[float] = []
    source_scores: List[float] = []
    signatures: List[List[float]] = []
    for raw_anchor in raw_anchors.values():
        sources = sorted(str(source) for source in raw_anchor.get("sources", []) or [])
        candidate_type = _candidate_type(sources)
        if care_area_type_from_candidate_type(candidate_type) != family.care_area_type:
            continue
        anchors.append(raw_anchor)
        type_scores.append(float(_anchor_type_match(family.seed_candidate.candidate_type, candidate_type)))
        source_scores.append(float(_anchor_source_strength(sources)))
        signatures.append(_signature_from_components(raw_anchor.get("metrics", {}) or {}, gap_norm_um=float(family.signature_gap_norm_um)))
    if not anchors:
        return []
    signature_matrix = np.asarray(signatures, dtype=np.float64)
    family_signature = np.asarray(family.signature, dtype=np.float64)
    if signature_matrix.ndim != 2 or family_signature.size == 0:
        signature_scores = np.zeros((len(anchors),), dtype=np.float64)
    else:
        width = min(int(signature_matrix.shape[1]), int(family_signature.size))
        if width <= 0:
            signature_scores = np.zeros((len(anchors),), dtype=np.float64)
        else:
            signature_scores = 1.0 - np.mean(np.abs(signature_matrix[:, :width] - family_signature[None, :width]), axis=1)
    scores = (
        0.45 * np.asarray(type_scores, dtype=np.float64)
        + 0.35 * np.asarray(signature_scores, dtype=np.float64)
        + 0.20 * np.asarray(source_scores, dtype=np.float64)
    )
    return [(_clip01(float(score)), anchor) for score, anchor in zip(scores.tolist(), anchors)]


def _anchor_stable_key(anchor: Mapping[str, Any]) -> Tuple[float, float, str]:
    """用坐标和候选类型生成稳定 key，避免依赖对象 id。"""
    sources = sorted(str(source) for source in anchor.get("sources", []) or [])
    return (round(float(anchor.get("x", 0.0)), 6), round(float(anchor.get("y", 0.0)), 6), _candidate_type(sources))


def _select_prescored_anchors(
    typed_anchors: Sequence[Tuple[float, Mapping[str, Any]]],
    *,
    instantiate_cap: int,
    layout_bbox: Sequence[float],
) -> Tuple[List[Tuple[float, Mapping[str, Any]]], Dict[str, Any], Dict[Tuple[float, float, str], str]]:
    """从 prescore 排序结果中保留主 top-K，同时抽一点 tile-balanced 探索样本。"""
    cap = max(0, int(instantiate_cap))
    if cap <= 0 or not typed_anchors:
        return [], {"pre_score_top_anchor_count": 0, "pre_score_tile_anchor_count": 0, "pre_score_fallback_anchor_count": 0, "pre_score_stratified": False}, {}
    ordered = list(typed_anchors)
    if len(ordered) <= cap:
        source_by_key = {_anchor_stable_key(anchor): "top" for _, anchor in ordered}
        return ordered, {
            "pre_score_top_anchor_count": int(len(ordered)),
            "pre_score_tile_anchor_count": 0,
            "pre_score_fallback_anchor_count": 0,
            "pre_score_stratified": False,
        }, source_by_key
    top_quota = max(1, min(cap, int(round(0.80 * float(cap)))))
    selected = list(ordered[:top_quota])
    selected_keys = {_anchor_stable_key(anchor) for _, anchor in selected}
    source_by_key = {key: "top" for key in selected_keys}
    remaining = [(score, anchor) for score, anchor in ordered[top_quota:] if _anchor_stable_key(anchor) not in selected_keys]
    tile_quota = max(0, cap - len(selected))
    tile_count = max(2, min(16, int(math.ceil(math.sqrt(max(1, tile_quota))))))
    buckets: Dict[Tuple[int, int], List[Tuple[float, Mapping[str, Any]]]] = {}
    for score, anchor in remaining:
        center = (float(anchor.get("x", 0.0)), float(anchor.get("y", 0.0)))
        buckets.setdefault(_tile_key(center, layout_bbox, tile_count), []).append((score, anchor))
    for bucket in buckets.values():
        bucket.sort(key=lambda item: (-float(item[0]), float(item[1].get("y", 0.0)), float(item[1].get("x", 0.0))))
    selected_tile_counts: Dict[Tuple[int, int], int] = {}
    for _, anchor in selected:
        center = (float(anchor.get("x", 0.0)), float(anchor.get("y", 0.0)))
        key = _tile_key(center, layout_bbox, tile_count)
        selected_tile_counts[key] = int(selected_tile_counts.get(key, 0)) + 1
    keys = sorted(buckets, key=lambda key: (int(selected_tile_counts.get(key, 0)), key[0], key[1]))
    positions = {key: 0 for key in keys}
    tile_added = 0
    while len(selected) < cap and tile_added < tile_quota:
        added = False
        for key in keys:
            position = positions[key]
            bucket = buckets[key]
            if position >= len(bucket):
                continue
            selected.append(bucket[position])
            anchor_key = _anchor_stable_key(bucket[position][1])
            selected_keys.add(anchor_key)
            source_by_key[anchor_key] = "tile"
            positions[key] = position + 1
            tile_added += 1
            added = True
            if len(selected) >= cap or tile_added >= tile_quota:
                break
        if not added:
            break
    if len(selected) < cap:
        fallback_added = 0
        for item in remaining:
            anchor_key = _anchor_stable_key(item[1])
            if anchor_key in selected_keys:
                continue
            selected.append(item)
            selected_keys.add(anchor_key)
            source_by_key[anchor_key] = "fallback"
            fallback_added += 1
            if len(selected) >= cap:
                break
    else:
        fallback_added = 0
    return selected, {
        "pre_score_top_anchor_count": int(top_quota),
        "pre_score_tile_anchor_count": int(tile_added),
        "pre_score_fallback_anchor_count": int(fallback_added),
        "pre_score_stratified": True,
    }, source_by_key


def _instance_fingerprint(instance: CareAreaInstance) -> np.ndarray:
    """读取 instance 缓存 fingerprint；旧对象缺失时再做一次轻量计算。"""
    fingerprint = np.asarray(instance.fingerprint, dtype=np.float32)
    if fingerprint.size:
        return fingerprint
    return bitmap_fingerprint(instance.window["clip_bitmap"])


def _is_duplicate_instance(left: CareAreaInstance, right: CareAreaInstance, *, radius_um: float) -> bool:
    """判断两个 care-area instances 是否是同一局部图形的重复提案。"""
    if _distance(left.center, right.center) > float(radius_um):
        return False
    similarity = _bitmap_similarity(
        left.window["clip_bitmap"],
        right.window["clip_bitmap"],
        _instance_fingerprint(right),
        _instance_fingerprint(left),
    )
    return bool(similarity >= 0.88)


def _finalize_family_instances(
    family: CareAreaFamily,
    candidates: Sequence[CareAreaInstance],
    *,
    max_instances: int,
    duplicate_radius_um: float,
    pre_reject_reasons: Mapping[str, int] | None = None,
) -> None:
    """执行 instance NMS、homogeneity 计算和稳定编号。"""
    kept: List[CareAreaInstance] = [_make_seed_instance(family)]
    nms_reject = 0
    cap_reject = 0
    ordered = sorted(candidates, key=lambda item: (-item.match_score, item.center[1], item.center[0]))
    processed = 0
    max_keep = max(1, int(max_instances))
    for candidate in ordered:
        if len(kept) >= max_keep:
            cap_reject = len(ordered) - processed
            break
        processed += 1
        if any(_is_duplicate_instance(candidate, existing, radius_um=float(duplicate_radius_um)) for existing in kept):
            nms_reject += 1
            continue
        kept.append(candidate)
    scores = [float(instance.match_score) for instance in kept]
    if scores:
        percentile = float(np.percentile(np.asarray(scores, dtype=np.float64), 10))
        family.homogeneity_score = _clip01(0.65 * float(np.mean(scores)) + 0.35 * percentile)
    else:
        family.homogeneity_score = 0.0
    for rank, instance in enumerate(kept):
        instance.instance_rank = int(rank)
        instance.instance_id = f"{family.family_id}__inst_{rank:04d}"
        instance.homogeneity_score = float(family.homogeneity_score)
    family.instances = kept
    family.expanded_instance_count = max(0, len(kept) - 1)
    family.is_singleton_family = bool(family.expanded_instance_count == 0)
    family.expansion_confidence = 0.0 if family.is_singleton_family else float(family.homogeneity_score)
    reject_reasons = {str(key): int(value) for key, value in (pre_reject_reasons or {}).items() if int(value) > 0}
    if nms_reject:
        reject_reasons["nms_reject"] = int(nms_reject)
    if cap_reject:
        reject_reasons["cap_reject"] = int(cap_reject)
    family.instance_reject_reasons = dict(sorted(reject_reasons.items()))
    family.rejected_instance_count = int(sum(family.instance_reject_reasons.values()))
    _refresh_metrology_contexts(family)


def _cluster_payloads(backend_result: Mapping[str, Any], metadata_by_marker: Mapping[str, Dict[str, Any]]) -> List[Tuple[Mapping[str, Any], str, Dict[str, Any], float]]:
    """从 backend result 中提取 representative marker payload。"""
    payloads: List[Tuple[Mapping[str, Any], str, Dict[str, Any], float]] = []
    for cluster in backend_result.get("clusters", []) or []:
        marker_id = str(cluster.get("marker_id", ""))
        metadata = metadata_by_marker.get(marker_id)
        if not metadata:
            raise ValueError(f"Missing representative metadata for marker {marker_id}")
        representative_payload = dict(cluster.get("representative_metadata", {}) or {})
        behavior_risk = float(representative_payload.get("normalized_risk_score", 0.0) or 0.0)
        payloads.append((cluster, marker_id, metadata, behavior_risk))
    return payloads


def _families_are_duplicates(left: CareAreaFamily, right: CareAreaFamily, *, radius_um: float) -> bool:
    """判断两个 seed family 是否只是同一局部 care-area 的重复提案。"""
    if int(left.cluster_id) != int(right.cluster_id):
        return False
    if left.source_path != right.source_path or left.care_area_type != right.care_area_type:
        return False
    if _distance(left.seed_center, right.seed_center) > float(radius_um):
        return False
    bitmap_score = _bitmap_similarity(
        left.seed_candidate.window["clip_bitmap"],
        right.seed_candidate.window["clip_bitmap"],
        np.asarray(right.fingerprint, dtype=np.float32),
        np.asarray(left.fingerprint, dtype=np.float32),
    )
    signature_score = _signature_similarity(left.signature, right.signature)
    return bool(bitmap_score >= 0.88 and signature_score >= 0.90)


def _merge_duplicate_families(families: Sequence[CareAreaFamily], *, radius_um: float) -> List[CareAreaFamily]:
    """在全图 expansion 前合并重复 seed family，避免同一弱点家族重复扩展。"""
    ordered = sorted(families, key=lambda item: (-float(item.seed_candidate.score), item.cluster_id, item.family_rank, item.family_id))
    kept: List[CareAreaFamily] = []
    for family in ordered:
        duplicate_of: CareAreaFamily | None = None
        for existing in kept:
            if _families_are_duplicates(existing, family, radius_um=radius_um):
                duplicate_of = existing
                break
        if duplicate_of is None:
            kept.append(family)
            continue
        duplicate_of.merged_seed_family_ids.append(str(family.family_id))
        for marker_id in family.marker_ids:
            if marker_id not in duplicate_of.marker_ids:
                duplicate_of.marker_ids.append(marker_id)
        if int(family.cluster_id) not in duplicate_of.merged_cluster_ids:
            duplicate_of.merged_cluster_ids.append(int(family.cluster_id))
        duplicate_of.merged_behavior_risk_values.append(float(family.behavior_risk))
        duplicate_of.behavior_risk = max(float(duplicate_of.behavior_risk), float(family.behavior_risk))
        duplicate_of.cluster_size = max(int(duplicate_of.cluster_size), int(family.cluster_size))
    kept.sort(key=lambda item: (item.cluster_id, item.family_rank, item.family_id))
    for rank, family in enumerate(kept):
        family.family_rank = int(rank)
    return kept


def build_care_area_groups(
    backend_result: Mapping[str, Any],
    *,
    metadata_by_marker: Mapping[str, Dict[str, Any]],
    layout_index_for_source: Callable[[str], LayoutIndex],
    window_for_source: Callable[[str, Tuple[float, float], float], Mapping[str, Any]],
    pixel_size_um: float,
    window_size_um: float,
    mp_search_radius_um: float,
    step_um: float,
    mp_candidates_per_marker: int,
    max_instances_per_family: int,
    min_feature_um: float | None = None,
) -> CareAreaExpansionResult:
    """从 representative markers 生成 care-area families，并在同一 OAS 内展开实例。"""
    payloads = _cluster_payloads(backend_result, metadata_by_marker)
    all_risk_zero = all(payload[3] <= 1e-12 for payload in payloads)
    gap_norm = _gap_norm_um(min_feature_um)
    families: List[CareAreaFamily] = []
    rejected: List[RejectedCareAreaSeed] = []

    for cluster, marker_id, metadata, behavior_risk in payloads:
        center_values = metadata.get("marker_center")
        if not isinstance(center_values, Sequence) or len(center_values) < 2:
            raise ValueError(f"Missing marker_center for marker {marker_id}")
        source_path = str(metadata.get("source_path", ""))
        marker_center = (float(center_values[0]), float(center_values[1]))
        layout_index = layout_index_for_source(source_path)
        marker_window = window_for_source(source_path, marker_center, float(window_size_um))
        discovery = discover_mp_candidates(
            layout_index=layout_index,
            marker_center=marker_center,
            marker_window=marker_window,
            window_size_um=float(window_size_um),
            pixel_size_um=float(pixel_size_um),
            search_radius_um=float(mp_search_radius_um),
            step_um=float(step_um),
            behavior_risk=float(behavior_risk),
            behavior_risk_enabled=not all_risk_zero,
            min_feature_um=min_feature_um,
            top_k=max(1, int(mp_candidates_per_marker)),
        )
        seed_candidates = [
            candidate
            for candidate in discovery.top_candidates
            if candidate.verified and care_area_type_from_candidate_type(candidate.candidate_type)
        ]
        cluster_id = int(cluster.get("cluster_id", len(families)))
        marker_ids = [str(value) for value in cluster.get("marker_ids", []) or [marker_id]]
        if not seed_candidates:
            rejected.append(
                RejectedCareAreaSeed(
                    marker_id=marker_id,
                    cluster_id=cluster_id,
                    marker_ids=marker_ids,
                    representative_metadata=dict(metadata),
                    cluster=dict(cluster),
                    reason="no_care_area_family",
                    seed_discovery=discovery,
                )
            )
            continue
        for family_rank, seed in enumerate(seed_candidates):
            family_id = f"cafam_{cluster_id:04d}_{_sanitize_id(marker_id)}_{family_rank:03d}"
            family = CareAreaFamily(
                family_id=family_id,
                family_rank=int(family_rank),
                seed_marker_id=marker_id,
                cluster_id=cluster_id,
                source_path=source_path,
                marker_ids=marker_ids,
                representative_metadata=dict(metadata),
                cluster=dict(cluster),
                seed_center=(float(seed.x), float(seed.y)),
                seed_candidate=seed,
                seed_discovery=discovery,
                care_area_type=care_area_type_from_candidate_type(seed.candidate_type),
                behavior_risk=float(behavior_risk),
                cluster_size=max(1, int(cluster.get("size", 1))),
                fingerprint=bitmap_fingerprint(seed.window["clip_bitmap"]),
                signature=_signature_from_components(seed.components, gap_norm_um=gap_norm),
                signature_gap_norm_um=float(gap_norm),
                seed_behavior_risk=float(behavior_risk),
                merged_behavior_risk_values=[float(behavior_risk)],
                merged_cluster_ids=[cluster_id],
            )
            families.append(family)

    families = _merge_duplicate_families(
        families,
        radius_um=max(float(step_um), 0.25 * float(window_size_um)),
    )
    target_types = {family.care_area_type for family in families}
    seed_centers_by_source: Dict[str, List[Tuple[float, float]]] = {}
    for family in families:
        seed_centers_by_source.setdefault(str(family.source_path), []).append(
            (float(family.seed_center[0]), float(family.seed_center[1]))
        )
    anchor_table_by_source: Dict[str, Dict[Tuple[int, int], Dict[str, Any]]] = {}
    anchor_audit_by_source: Dict[str, Dict[str, Any]] = {}
    for family in families:
        if family.source_path not in anchor_table_by_source:
            anchor_table, anchor_audit = _build_seeded_anchor_table(
                layout_index=layout_index_for_source(family.source_path),
                target_types=target_types,
                local_radius_um=max(float(mp_search_radius_um), float(window_size_um)),
                step_um=float(step_um),
                pixel_size_um=float(pixel_size_um),
                min_feature_um=min_feature_um,
                seed_centers=seed_centers_by_source.get(str(family.source_path), []),
            )
            anchor_table_by_source[family.source_path] = anchor_table
            anchor_audit_by_source[family.source_path] = anchor_audit
        raw_anchors = anchor_table_by_source[family.source_path]
        family.anchor_table_audit = dict(anchor_audit_by_source.get(family.source_path, {}))
        family.candidate_anchor_count = int(len(raw_anchors))
        candidates: List[CareAreaInstance] = []
        layout_index = layout_index_for_source(family.source_path)
        pre_reject_reasons: Dict[str, int] = {}
        typed_anchors = _prescore_typed_anchors(family, raw_anchors)
        typed_anchors.sort(key=lambda item: (-float(item[0]), float(item[1].get("y", 0.0)), float(item[1].get("x", 0.0))))
        instantiate_cap = max(128, 6 * int(max_instances_per_family))
        selected_anchors, pre_score_selection_audit, pre_score_source_by_key = _select_prescored_anchors(
            typed_anchors,
            instantiate_cap=int(instantiate_cap),
            layout_bbox=family.anchor_table_audit.get("layout_bbox", _layout_bbox(layout_index)),
        )
        family.anchor_table_audit.update(
            {
                "pre_score_candidate_count": int(len(typed_anchors)),
                "selected_anchor_count": int(len(selected_anchors)),
                "pre_score_limited": bool(len(typed_anchors) > len(selected_anchors)),
                "pre_score_limit": int(instantiate_cap),
                **pre_score_selection_audit,
            }
        )
        instantiated_count = 0
        pre_score_early_stop = False
        pre_nms_match_limit = max(16, 2 * int(max_instances_per_family))
        instantiated_by_source: Dict[str, int] = {}
        match_by_source: Dict[str, int] = {}
        reject_by_source: Dict[str, Dict[str, int]] = {}
        for _, raw_anchor in selected_anchors:
            if len(candidates) >= pre_nms_match_limit:
                pre_score_early_stop = True
                break
            selection_source = str(pre_score_source_by_key.get(_anchor_stable_key(raw_anchor), "unknown"))
            instantiated_by_source[selection_source] = int(instantiated_by_source.get(selection_source, 0)) + 1
            instantiated_count += 1
            instance, reject_reason = _instance_from_anchor(
                family,
                layout_index=layout_index,
                raw_anchor=raw_anchor,
                window_size_um=float(window_size_um),
                pixel_size_um=float(pixel_size_um),
            )
            if instance is None:
                reason = str(reject_reason or "instance_reject")
                pre_reject_reasons[reason] = int(pre_reject_reasons.get(reason, 0)) + 1
                source_reasons = reject_by_source.setdefault(selection_source, {})
                source_reasons[reason] = int(source_reasons.get(reason, 0)) + 1
            else:
                instance.selection_source = selection_source
                match_by_source[selection_source] = int(match_by_source.get(selection_source, 0)) + 1
                candidates.append(instance)
        match_rate_by_source = {
            source: float(match_by_source.get(source, 0)) / float(max(1, count))
            for source, count in instantiated_by_source.items()
        }
        family.anchor_table_audit.update(
            {
                "instantiated_anchor_count": int(instantiated_count),
                "pre_score_match_count": int(len(candidates)),
                "pre_score_match_rate": float(len(candidates)) / float(max(1, instantiated_count)),
                "pre_score_instantiated_count_by_source": dict(sorted(instantiated_by_source.items())),
                "pre_score_match_count_by_source": dict(sorted(match_by_source.items())),
                "pre_score_match_rate_by_source": dict(sorted(match_rate_by_source.items())),
                "pre_score_reject_reasons_by_source": {
                    source: dict(sorted(reasons.items())) for source, reasons in sorted(reject_by_source.items())
                },
                "pre_score_reject_reasons": dict(sorted(pre_reject_reasons.items())),
                "pre_score_early_stop": bool(pre_score_early_stop),
                "pre_score_pre_nms_match_limit": int(pre_nms_match_limit),
            }
        )
        _finalize_family_instances(
            family,
            candidates,
            max_instances=int(max_instances_per_family),
            duplicate_radius_um=max(float(step_um), 0.25 * float(window_size_um)),
            pre_reject_reasons=pre_reject_reasons,
        )
        final_instances = [instance for instance in family.instances if not instance.is_seed_instance]
        final_by_source: Dict[str, int] = {}
        for instance in final_instances:
            source = str(instance.selection_source or "unknown")
            final_by_source[source] = int(final_by_source.get(source, 0)) + 1
        final_rate_by_source = {
            source: float(final_by_source.get(source, 0)) / float(max(1, match_by_source.get(source, 0)))
            for source in set(final_by_source) | set(match_by_source)
        }
        family.anchor_table_audit["pre_score_final_instance_count"] = int(len(final_instances))
        family.anchor_table_audit["pre_score_final_instance_count_by_source"] = dict(sorted(final_by_source.items()))
        family.anchor_table_audit["pre_score_final_instance_rate_by_source"] = dict(sorted(final_rate_by_source.items()))
        family.anchor_table_audit["pre_score_early_stop_final_shortfall"] = bool(
            pre_score_early_stop and int(family.anchor_table_audit["pre_score_final_instance_count"]) < int(math.ceil(0.70 * float(max_instances_per_family)))
        )
        family.anchor_table_audit["pre_score_final_instance_ratio"] = float(family.anchor_table_audit["pre_score_final_instance_count"]) / float(max(1, int(max_instances_per_family)))
    return CareAreaExpansionResult(families=families, rejected_seeds=rejected, anchor_table_audits_by_source=anchor_audit_by_source)

