#!/usr/bin/env python3
"""CD-SEM 量测语境下的 NanoPoint-inspired context scoring。

本模块只借鉴 NanoPoint 的 design-aware grouping / sampling 思想，不引入光学检测的
noise floor、inspection sensitivity 或 run-hotter threshold 语义。这里的分数只回答
CD-SEM recipe selector 的问题：某个 care-area / MP 是否值得占用 SEM 量测 slot，以及
它是否存在代表性差、AF/AP 可能失败或低可执行性的浪费风险。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


@dataclass
class MetrologyContext:
    """保存单个 care-area 或 MP candidate 的量测价值与 recipe 浪费风险。"""

    metrology_priority_score: float
    metrology_priority_class: str
    site_reliability_risk: float
    recipe_waste_penalty: float
    metrology_context_group_id: str
    selection_profile_id: str
    components: Dict[str, float] = field(default_factory=dict)


def _clip01(value: float) -> float:
    """把数值限制在 0-1 区间。"""
    if not math.isfinite(float(value)):
        return 0.0
    return float(min(1.0, max(0.0, float(value))))


def _bitmap_density(bitmap: Any) -> float:
    """计算 bitmap 中 pattern pixel 的占比。"""
    arr = np.asarray(bitmap, dtype=bool)
    if arr.size == 0:
        return 0.0
    return float(np.count_nonzero(arr)) / float(arr.size)


def _edge_density_score(bitmap: Any) -> float:
    """用水平/垂直跳变密度估计局部可对焦结构丰富度。"""
    arr = np.asarray(bitmap, dtype=bool)
    if arr.size == 0:
        return 0.0
    horizontal = np.count_nonzero(arr[:, 1:] != arr[:, :-1]) if arr.shape[1] > 1 else 0
    vertical = np.count_nonzero(arr[1:, :] != arr[:-1, :]) if arr.shape[0] > 1 else 0
    return _clip01(float(horizontal + vertical) / float(max(1, arr.size)) * 10.0)


def _corner_density_score(bitmap: Any) -> float:
    """用 2x2 patch 的非单调变化估计角点/拐折结构丰富度。"""
    arr = np.asarray(bitmap, dtype=bool)
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        return 0.0
    tl = arr[:-1, :-1]
    tr = arr[:-1, 1:]
    bl = arr[1:, :-1]
    br = arr[1:, 1:]
    counts = tl.astype(np.uint8) + tr.astype(np.uint8) + bl.astype(np.uint8) + br.astype(np.uint8)
    return _clip01(float(np.count_nonzero((counts == 1) | (counts == 3))) / float(max(1, counts.size)) * 80.0)


def _layout_complexity_score(bitmap: Any) -> float:
    """估计局部图形是否具备足够 SEM 对焦/匹配结构。"""
    density = _bitmap_density(bitmap)
    density_balance = _clip01(1.0 - abs(density - 0.35) / 0.35)
    return _clip01(0.45 * _edge_density_score(bitmap) + 0.35 * _corner_density_score(bitmap) + 0.20 * density_balance)


def _density_extreme_risk(bitmap: Any) -> float:
    """估计过空或过满 bitmap 对 recipe slot 的浪费风险。"""
    density = _bitmap_density(bitmap)
    if density < 0.03 or density >= 0.92:
        return 1.0
    sparse_risk = _clip01((0.08 - density) / 0.08)
    uniform_risk = _clip01((density - 0.85) / 0.07)
    return max(float(sparse_risk), float(uniform_risk))


def _care_area_type_prior(care_area_type: str) -> float:
    """给不同 weak-pattern 类型一个固定的工艺风险先验。"""
    mapping = {
        "spacing": 1.0,
        "line_end": 0.85,
        "corner_jog": 0.75,
        "density_transition": 0.65,
    }
    return float(mapping.get(str(care_area_type), 0.50))


def _hotspot_geometry_risk(care_area_type: str, components: Mapping[str, Any]) -> float:
    """从 MP/care-area 几何信号中提取 hotspot-like 风险。"""
    actual_signal = max(
        float(components.get("core_defect_proxy_score", 0.0) or 0.0),
        float(components.get("critical_geometry_score", 0.0) or 0.0),
        float(components.get("geometry_core_score", 0.0) or 0.0),
        float(components.get("fragment_facing_pair_score", 0.0) or 0.0),
        float(components.get("fragment_line_end_score", 0.0) or 0.0),
        float(components.get("fragment_corner_score", 0.0) or 0.0),
        float(components.get("density_transition_score", 0.0) or 0.0),
    )
    return _clip01(0.30 * _care_area_type_prior(care_area_type) + 0.70 * _clip01(actual_signal))


def _priority_class(score: float) -> str:
    """把量测优先级分成 high / mid / low 三档。"""
    value = float(score)
    if value >= 0.66:
        return "high"
    if value >= 0.33:
        return "mid"
    return "low"


def compute_metrology_context(
    *,
    care_area_type: str,
    bitmap: Any,
    components: Mapping[str, Any] | None = None,
    inherited_behavior_risk: float = 0.0,
    family_representativeness: float = 1.0,
    pattern_rarity: float = 0.0,
    mp_localization_confidence: float = 1.0,
    family_homogeneity: float = 1.0,
    signature_quality: float = 1.0,
    mp_verified: bool = True,
) -> MetrologyContext:
    """计算 CD-SEM 语境下的量测优先级和 recipe slot 浪费风险。"""
    comps = dict(components or {})
    hotspot_geometry_risk = _hotspot_geometry_risk(care_area_type, comps)
    inherited_behavior_risk = _clip01(float(inherited_behavior_risk))
    family_representativeness = _clip01(float(family_representativeness))
    pattern_rarity = _clip01(float(pattern_rarity))
    mp_localization_confidence = _clip01(float(mp_localization_confidence))
    family_homogeneity = _clip01(float(family_homogeneity))
    signature_quality = _clip01(float(signature_quality))
    focus_structure_score = _layout_complexity_score(bitmap)
    weak_mp_verification = 0.0 if bool(mp_verified) else 1.0
    sparse_or_uniform_bitmap_risk = _density_extreme_risk(bitmap)
    high_repetition_or_ap_ambiguity_proxy = _clip01(1.0 - pattern_rarity)
    low_family_homogeneity = _clip01(1.0 - family_homogeneity)
    signature_sparse_penalty = _clip01(1.0 - signature_quality)

    metrology_priority_score = _clip01(
        0.30 * hotspot_geometry_risk
        + 0.20 * inherited_behavior_risk
        + 0.20 * family_representativeness
        + 0.15 * pattern_rarity
        + 0.15 * mp_localization_confidence
    )
    site_reliability_risk = _clip01(
        0.25 * low_family_homogeneity
        + 0.20 * weak_mp_verification
        + 0.20 * high_repetition_or_ap_ambiguity_proxy
        + 0.15 * _clip01(1.0 - focus_structure_score)
        + 0.10 * sparse_or_uniform_bitmap_risk
        + 0.10 * signature_sparse_penalty
    )
    recipe_waste_penalty = float(site_reliability_risk)
    priority_class = _priority_class(metrology_priority_score)
    group_id = f"{care_area_type or 'unknown'}__{priority_class}"

    context_components = {
        "hotspot_geometry_risk": float(hotspot_geometry_risk),
        "inherited_behavior_risk": float(inherited_behavior_risk),
        "family_representativeness": float(family_representativeness),
        "pattern_rarity": float(pattern_rarity),
        "mp_localization_confidence": float(mp_localization_confidence),
        "low_family_homogeneity": float(low_family_homogeneity),
        "weak_mp_verification": float(weak_mp_verification),
        "high_repetition_or_ap_ambiguity_proxy": float(high_repetition_or_ap_ambiguity_proxy),
        "low_focus_structure_proxy": float(_clip01(1.0 - focus_structure_score)),
        "sparse_or_uniform_bitmap_risk": float(sparse_or_uniform_bitmap_risk),
        "signature_sparse_penalty": float(signature_sparse_penalty),
        "bitmap_density": float(_bitmap_density(bitmap)),
        "focus_structure_score": float(focus_structure_score),
    }

    return MetrologyContext(
        metrology_priority_score=float(metrology_priority_score),
        metrology_priority_class=str(priority_class),
        site_reliability_risk=float(site_reliability_risk),
        recipe_waste_penalty=float(recipe_waste_penalty),
        metrology_context_group_id=str(group_id),
        selection_profile_id=f"metrology_profile__{care_area_type or 'unknown'}__{priority_class}",
        components=context_components,
    )


def context_to_summary(context: MetrologyContext) -> Dict[str, Any]:
    """把 context 对象转换为可写入 JSON/CSV 的扁平字段。"""
    return {
        "metrology_priority_score": float(context.metrology_priority_score),
        "metrology_priority_class": str(context.metrology_priority_class),
        "site_reliability_risk": float(context.site_reliability_risk),
        "recipe_waste_penalty": float(context.recipe_waste_penalty),
        "metrology_context_group_id": str(context.metrology_context_group_id),
        "selection_profile_id": str(context.selection_profile_id),
        "metrology_context_components": dict(context.components),
    }


def _mean(values: Sequence[float]) -> float:
    """计算均值；空输入返回 0。"""
    if not values:
        return 0.0
    return float(sum(float(value) for value in values)) / float(len(values))


def _top_reject_reasons(rows: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    """统计一组 recipe rows 的主要拒绝原因。"""
    counts: Dict[str, int] = {}
    for row in rows:
        reason = str(row.get("reject_reason", "") or row.get("pool_reject_reason", ""))
        if not reason:
            continue
        for part in reason.split(";"):
            key = part.strip()
            if key:
                counts[key] = int(counts.get(key, 0)) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:8])


def _aggregate_group(
    *,
    families: Sequence[Mapping[str, Any]],
    instances: Sequence[Mapping[str, Any]],
    pool_items: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """汇总一个 metrology context group 的候选、入选和失败统计。"""
    selected_rows = [row for row in rows if str(row.get("recipe_status", "")) == "selected"]
    rejected_rows = [row for row in rows if str(row.get("recipe_status", "")) == "rejected"]
    verified_count = sum(1 for item in pool_items if bool(item.get("mp_verified", False)))
    mp_count = len(pool_items)
    return {
        "care_area_family_count": int(len(families)),
        "care_area_instance_count": int(len(instances)),
        "mp_candidate_count": int(mp_count),
        "selected_site_count": int(len(selected_rows)),
        "rejected_site_count": int(len(rejected_rows)),
        "mp_verified_rate": float(verified_count) / float(mp_count) if mp_count else 0.0,
        "af_fail_count": int(sum(1 for row in rows if "no_safe_af" in str(row.get("reject_reason", "")))),
        "ap_fail_count": int(sum(1 for row in rows if "no_unique_ap" in str(row.get("reject_reason", "")))),
        "ap_global_duplicate_count": int(sum(1 for row in rows if bool(row.get("ap_global_duplicate", False)) or "ap_global_duplicate" in str(row.get("reject_reason", "")))),
        "avg_metrology_priority_score": _mean([float(item.get("metrology_priority_score", 0.0) or 0.0) for item in pool_items]),
        "avg_site_reliability_risk": _mean([float(item.get("site_reliability_risk", 0.0) or 0.0) for item in pool_items]),
        "avg_recipe_waste_penalty": _mean([float(item.get("recipe_waste_penalty", 0.0) or 0.0) for item in pool_items]),
        "top_reject_reasons": _top_reject_reasons(list(rows) + list(pool_items)),
    }


def build_metrology_context_audit(
    *,
    care_area_groups: Mapping[str, Any],
    mp_candidate_pool: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """按量测优先级、care-area 类型和 context group 汇总 recipe 审查信息。"""
    families = list(care_area_groups.get("families", []) or [])
    instances: list[Mapping[str, Any]] = []
    for family in families:
        for instance in family.get("instances", []) or []:
            instances.append(instance)

    def key_value(item: Mapping[str, Any], key: str) -> str:
        value = str(item.get(key, "") or "")
        return value or "unknown"

    def bucket(field: str) -> Dict[str, Any]:
        keys = sorted(
            {
                key_value(item, field)
                for item in list(families) + list(instances) + list(mp_candidate_pool) + list(rows)
                if key_value(item, field) != "unknown"
            }
        )
        result: Dict[str, Any] = {}
        for key in keys:
            result[key] = _aggregate_group(
                families=[item for item in families if key_value(item, field) == key],
                instances=[item for item in instances if key_value(item, field) == key],
                pool_items=[item for item in mp_candidate_pool if key_value(item, field) == key],
                rows=[item for item in rows if key_value(item, field) == key],
            )
        return result

    selected_rows = [row for row in rows if str(row.get("recipe_status", "")) == "selected"]
    selected_groups = sorted({key_value(row, "metrology_context_group_id") for row in selected_rows if key_value(row, "metrology_context_group_id") != "unknown"})
    selected_by_class: Dict[str, int] = {}
    selected_by_group: Dict[str, int] = {}
    for row in selected_rows:
        priority_class = key_value(row, "metrology_priority_class")
        context_group = key_value(row, "metrology_context_group_id")
        selected_by_class[priority_class] = int(selected_by_class.get(priority_class, 0)) + 1
        selected_by_group[context_group] = int(selected_by_group.get(context_group, 0)) + 1

    return {
        "by_metrology_priority_class": bucket("metrology_priority_class"),
        "by_care_area_type": bucket("care_area_type"),
        "by_metrology_context_group": bucket("metrology_context_group_id"),
        "summary": {
            "metrology_context_group_count": int(len({key_value(item, "metrology_context_group_id") for item in mp_candidate_pool if key_value(item, "metrology_context_group_id") != "unknown"})),
            "selected_metrology_context_group_count": int(len(selected_groups)),
            "selected_by_metrology_priority_class": dict(sorted(selected_by_class.items())),
            "selected_by_metrology_context_group": dict(sorted(selected_by_group.items())),
        },
    }
