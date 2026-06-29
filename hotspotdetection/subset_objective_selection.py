#!/usr/bin/env python3
"""Casati-style objective subset selection 主线模块。

本模块位于 `recipe_site_selector.py` 的全局 MP pool selection 层，只负责一件事：
把已经生成并补齐 review evidence 的 MP candidates，按多目标边际收益选择一个有限
recipe subset。它不生成 MP，不构造 AF/AP，也不改变任何 hard gate。

算法流程：
1. 从候选已有字段派生 objective risk、recipe feasibility 和 candidate value。
2. 按 candidate pool 的 value 分布自动推导 care-area / family / context / taxonomy /
   risk / feasibility target bins。
3. 用贪心 marginal gain 选择 subset，并对已覆盖 bin 使用递减收益。
4. 输出候选级 objective components、target bins、selection trace 和 coverage gap audit。

当前版本是确定性、无训练实现；所有权重固定在代码中，不提供额外 CLI knob。
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

import numpy as np


SCHEMA_VERSION = "subset_objective_selection_v1"
TARGET_CATEGORIES = (
    "care_area_type",
    "care_area_family_id",
    "metrology_context_group_id",
    "pattern_taxonomy_class",
    "risk_bin",
    "feasibility_bin",
)


def clip01(value: float) -> float:
    """把数值限制在 0-1 区间；非有限值按 0 处理。"""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return float(min(1.0, max(0.0, numeric)))


def safe_float(mapping: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    """从 mapping 中读取 float，缺失或非法时返回默认值。"""
    try:
        return float(mapping.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def score_bin(value: float) -> str:
    """把连续分数压成固定 low/mid/high bin。"""
    numeric = clip01(value)
    if numeric < 0.33:
        return "low"
    if numeric < 0.66:
        return "mid"
    return "high"


def compute_objective_components(candidate: Mapping[str, Any]) -> Dict[str, float]:
    """从现有候选字段派生 Casati-style selection 所需的候选级价值信号。"""
    evidence = candidate.get("evidence_contradiction_audit", {}) or {}
    feasibility = candidate.get("expected_feasibility_audit", {}) or {}
    taxonomy = candidate.get("pattern_taxonomy_audit", {}) or {}
    score_components = candidate.get("score_components", candidate.get("mp_risk_components", {}) or {}) or {}
    raw_components = candidate.get("raw_components", {}) or {}

    mp_hotspot = clip01(safe_float(candidate, "mp_hotspot_score", safe_float(score_components, "mp_hotspot_score", 0.0)))
    behavior = clip01(
        safe_float(
            score_components,
            "effective_behavior_risk",
            safe_float(raw_components, "effective_behavior_risk", safe_float(raw_components, "behavior_risk", safe_float(score_components, "behavior_risk", 0.0))),
        )
    )
    defect_evidence = clip01(safe_float(evidence, "defect_evidence_proxy_score", 0.0))
    metrology_priority = clip01(safe_float(candidate, "metrology_priority_score", safe_float(score_components, "metrology_priority_raw", 0.0)))
    pattern_novelty = clip01(safe_float(score_components, "pattern_novelty", safe_float(raw_components, "pattern_novelty", safe_float(raw_components, "pattern_rarity", 0.0))))
    expected_feasibility = clip01(safe_float(feasibility, "expected_recipe_feasibility_proxy", 0.5))
    recipe_waste = clip01(safe_float(candidate, "recipe_waste_penalty", safe_float(score_components, "recipe_waste_penalty", 0.0)))
    priority_anchor = clip01(safe_float(candidate, "mp_priority_score", 0.0))
    htc_like = clip01(safe_float(taxonomy, "htc_like_score", 0.0))

    objective_risk = clip01(
        0.35 * mp_hotspot
        + 0.20 * behavior
        + 0.20 * defect_evidence
        + 0.15 * metrology_priority
        + 0.10 * pattern_novelty
    )
    objective_feasibility = clip01(0.70 * expected_feasibility + 0.30 * (1.0 - recipe_waste))
    objective_value = clip01(objective_risk * (0.50 + 0.50 * objective_feasibility))

    return {
        "objective_risk_score": float(objective_risk),
        "objective_feasibility_score": float(objective_feasibility),
        "objective_candidate_value": float(objective_value),
        "mp_hotspot_score": float(mp_hotspot),
        "effective_behavior_risk": float(behavior),
        "defect_evidence_proxy_score": float(defect_evidence),
        "metrology_priority_score": float(metrology_priority),
        "pattern_novelty": float(pattern_novelty),
        "expected_recipe_feasibility_proxy": float(expected_feasibility),
        "recipe_waste_penalty": float(recipe_waste),
        "priority_anchor_gain": float(priority_anchor),
        "htc_like_score": float(htc_like),
    }


def compute_target_bins(candidate: Mapping[str, Any], components: Mapping[str, float]) -> Dict[str, str]:
    """生成候选参与 coverage balance 的目标 bin。"""
    taxonomy = candidate.get("pattern_taxonomy_audit", {}) or {}
    care_type = str(candidate.get("care_area_type") or candidate.get("mp_candidate_type") or "unknown")
    family_id = str(candidate.get("care_area_family_id") or candidate.get("hotspot_cluster_id") or candidate.get("source_marker_id") or "unknown")
    context_group = str(candidate.get("metrology_context_group_id") or f"{care_type}__unknown")
    taxonomy_class = str(taxonomy.get("pattern_taxonomy_class", "ambiguous") or "ambiguous")
    return {
        "care_area_type": care_type,
        "care_area_family_id": family_id,
        "metrology_context_group_id": context_group,
        "pattern_taxonomy_class": taxonomy_class,
        "risk_bin": score_bin(float(components.get("objective_risk_score", 0.0))),
        "feasibility_bin": score_bin(float(components.get("objective_feasibility_score", 0.0))),
    }


def _bin_key(category: str, value: str) -> str:
    """生成 audit 中稳定可读的 bin key。"""
    return f"{category}:{value}"


def build_target_distribution(candidates: Sequence[Mapping[str, Any]], max_sites: int) -> Dict[str, Dict[str, Dict[str, float]]]:
    """按 pool 内 value 权重推导每类 target bin 的目标数量。"""
    targets: Dict[str, Dict[str, Dict[str, float]]] = {category: {} for category in TARGET_CATEGORIES}
    eligible_candidates = [
        candidate
        for candidate in candidates
        if str(candidate.get("pool_status", "candidate")) == "candidate" and bool(candidate.get("mp_verified", True))
    ]
    if not eligible_candidates or int(max_sites) <= 0:
        return targets
    for category in TARGET_CATEGORIES:
        weights: Dict[str, float] = defaultdict(float)
        for candidate in eligible_candidates:
            bins = candidate.get("subset_objective_target_bins", {}) or {}
            components = candidate.get("subset_objective_components", {}) or {}
            value = float(components.get("objective_candidate_value", 0.0))
            bin_value = str(bins.get(category, "unknown"))
            if category == "pattern_taxonomy_class" and bin_value == "htc_like":
                continue
            weights[bin_value] += max(0.0, value)
        total = float(sum(weights.values()))
        if total <= 0.0:
            active_bins = sorted(weights)
            share = 1.0 / float(max(1, len(active_bins)))
            for bin_value in active_bins:
                targets[category][bin_value] = {
                    "weighted_pool_share": float(share),
                    "pool_value": 0.0,
                    "target_count": int(max(1, math.ceil(float(max_sites) * share))),
                }
            continue
        for bin_value, value in sorted(weights.items()):
            share = float(value) / total
            targets[category][bin_value] = {
                "weighted_pool_share": float(share),
                "pool_value": float(value),
                "target_count": int(max(1, math.ceil(float(max_sites) * share))),
            }
    return targets


def _target_count(targets: Mapping[str, Mapping[str, Mapping[str, float]]], category: str, bin_value: str) -> int:
    """读取某个 bin 的目标数量，缺失时按 1 处理。"""
    try:
        return int(targets.get(category, {}).get(str(bin_value), {}).get("target_count", 1))
    except (TypeError, ValueError):
        return 1


def _diminishing_multiplier(selected_counts: Mapping[str, int], targets: Mapping[str, Mapping[str, Mapping[str, float]]], category: str, bin_value: str) -> float:
    """按已选数量和目标数量计算递减收益倍率。"""
    key = _bin_key(category, str(bin_value))
    count = int(selected_counts.get(key, 0))
    multiplier = 1.0 / math.sqrt(1.0 + float(count))
    if count >= _target_count(targets, category, str(bin_value)):
        multiplier *= 0.30
    return float(multiplier)


def compute_marginal_gain(
    candidate: Mapping[str, Any],
    *,
    selected_counts: Mapping[str, int],
    targets: Mapping[str, Mapping[str, Mapping[str, float]]],
    spatial_diversity_gain: float,
) -> Dict[str, float]:
    """计算候选在当前 selected subset 状态下的边际收益拆解。"""
    components = candidate.get("subset_objective_components", {}) or {}
    bins = candidate.get("subset_objective_target_bins", {}) or {}
    value = float(components.get("objective_candidate_value", 0.0))
    risk = float(components.get("objective_risk_score", 0.0))
    feasibility = float(components.get("objective_feasibility_score", 0.0))
    priority_anchor = float(components.get("priority_anchor_gain", 0.0))
    recipe_waste = float(components.get("recipe_waste_penalty", 0.0))
    taxonomy_class = str(bins.get("pattern_taxonomy_class", "ambiguous"))
    htc_score = float(components.get("htc_like_score", 0.0))

    risk_gain = risk * _diminishing_multiplier(selected_counts, targets, "risk_bin", str(bins.get("risk_bin", "unknown")))
    family_gain = value * _diminishing_multiplier(selected_counts, targets, "care_area_family_id", str(bins.get("care_area_family_id", "unknown")))
    context_type_gain = value * _diminishing_multiplier(selected_counts, targets, "care_area_type", str(bins.get("care_area_type", "unknown")))
    context_group_gain = value * _diminishing_multiplier(selected_counts, targets, "metrology_context_group_id", str(bins.get("metrology_context_group_id", "unknown")))
    context_gain = 0.50 * context_type_gain + 0.50 * context_group_gain
    if taxonomy_class == "htc_like":
        taxonomy_gain = 0.0
    else:
        taxonomy_gain = value * _diminishing_multiplier(selected_counts, targets, "pattern_taxonomy_class", taxonomy_class)
    feasibility_gain = feasibility * _diminishing_multiplier(selected_counts, targets, "feasibility_bin", str(bins.get("feasibility_bin", "unknown")))
    spatial_gain = clip01(float(spatial_diversity_gain))
    htc_penalty = recipe_waste * max(htc_score, 1.0 if taxonomy_class == "htc_like" else 0.0)
    marginal = (
        0.24 * risk_gain
        + 0.16 * family_gain
        + 0.14 * context_gain
        + 0.12 * taxonomy_gain
        + 0.14 * feasibility_gain
        + 0.10 * spatial_gain
        + 0.10 * priority_anchor
        - 0.08 * htc_penalty
    )
    return {
        "risk_coverage_gain": float(risk_gain),
        "family_coverage_gain": float(family_gain),
        "context_coverage_gain": float(context_gain),
        "care_area_type_coverage_gain": float(context_type_gain),
        "metrology_context_coverage_gain": float(context_group_gain),
        "taxonomy_balance_gain": float(taxonomy_gain),
        "recipe_feasibility_gain": float(feasibility_gain),
        "spatial_diversity_gain": float(spatial_gain),
        "priority_anchor_gain": float(priority_anchor),
        "htc_waste_penalty": float(htc_penalty),
        "subset_objective_marginal_gain": float(marginal),
    }


def prepare_candidates(candidates: Sequence[MutableMapping[str, Any]], max_sites: int) -> Dict[str, Any]:
    """补齐 objective components / bins，并返回 pool-derived target distribution。"""
    for candidate in candidates:
        components = compute_objective_components(candidate)
        bins = compute_target_bins(candidate, components)
        candidate["subset_objective_components"] = components
        candidate["subset_objective_target_bins"] = bins
        candidate.setdefault("subset_objective_status", str(candidate.get("pool_status", "candidate")))
        candidate.setdefault("subset_objective_marginal_gain", 0.0)
    targets = build_target_distribution(candidates, max_sites=max_sites)
    return {"schema_version": SCHEMA_VERSION, "target_distribution": targets}


def update_selected_counts(
    selected_counts: MutableMapping[str, int],
    candidate: Mapping[str, Any],
) -> None:
    """把一个已选候选贡献的 target bins 写入 selected count。"""
    bins = candidate.get("subset_objective_target_bins", {}) or {}
    for category in TARGET_CATEGORIES:
        bin_value = str(bins.get(category, "unknown"))
        if category == "pattern_taxonomy_class" and bin_value == "htc_like":
            continue
        key = _bin_key(category, bin_value)
        selected_counts[key] = int(selected_counts.get(key, 0)) + 1


def candidate_sort_key(candidate: Mapping[str, Any], marginal_gain: float) -> tuple[float, float, float, int, str]:
    """提供确定性贪心 tie-breaker。"""
    components = candidate.get("subset_objective_components", {}) or {}
    return (
        float(marginal_gain),
        float(components.get("objective_candidate_value", 0.0)),
        float(candidate.get("mp_priority_score", 0.0)),
        -int(candidate.get("mp_candidate_rank", 0) or 0),
        str(candidate.get("mp_candidate_id", "")),
    )


def _distribution(candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, int]]:
    """统计候选或已选集合在各 objective category 上的分布。"""
    result: Dict[str, Dict[str, int]] = {category: {} for category in TARGET_CATEGORIES}
    for candidate in candidates:
        bins = candidate.get("subset_objective_target_bins", {}) or {}
        for category in TARGET_CATEGORIES:
            value = str(bins.get(category, "unknown"))
            counter = result[category]
            counter[value] = int(counter.get(value, 0)) + 1
    return result


def _coverage_gaps(
    targets: Mapping[str, Mapping[str, Mapping[str, float]]],
    selected_distribution: Mapping[str, Mapping[str, int]],
) -> list[Dict[str, Any]]:
    """列出预算内未满足的目标 bin。"""
    gaps: list[Dict[str, Any]] = []
    for category, bins in targets.items():
        for bin_value, target_info in bins.items():
            selected = int(selected_distribution.get(category, {}).get(bin_value, 0))
            target = int(target_info.get("target_count", 0))
            gap = max(0, target - selected)
            if gap > 0:
                gaps.append(
                    {
                        "category": str(category),
                        "bin": str(bin_value),
                        "target_count": int(target),
                        "selected_count": int(selected),
                        "gap": int(gap),
                        "weighted_pool_share": float(target_info.get("weighted_pool_share", 0.0)),
                    }
                )
    gaps.sort(key=lambda item: (-int(item["gap"]), -float(item["weighted_pool_share"]), str(item["category"]), str(item["bin"])))
    return gaps


def _mean(values: Iterable[float]) -> float:
    """计算均值；空集合返回 0。"""
    values_list = [float(value) for value in values]
    return float(np.mean(values_list)) if values_list else 0.0


def build_subset_objective_audit(
    *,
    mp_candidate_pool: Sequence[Mapping[str, Any]],
    site_details: Sequence[Mapping[str, Any]],
    target_distribution: Mapping[str, Mapping[str, Mapping[str, float]]] | None = None,
    max_high_value: int = 50,
) -> Dict[str, Any]:
    """汇总 subset objective 的目标分布、coverage gap 和最终不可执行高风险候选。"""
    selected_candidates = [candidate for candidate in mp_candidate_pool if str(candidate.get("pool_status", "")) == "selected"]
    pool_distribution = _distribution(mp_candidate_pool)
    selected_distribution = _distribution(selected_candidates)
    targets = {
        str(category): dict((target_distribution or {}).get(str(category), {}))
        for category in TARGET_CATEGORIES
    }
    if not any(targets.values()):
        targets = build_target_distribution(mp_candidate_pool, max_sites=max(1, len(selected_candidates)))
    gaps = _coverage_gaps(targets, selected_distribution)
    selected_ids = {str(candidate.get("mp_candidate_id", "")) for candidate in selected_candidates}
    value_by_candidate = {
        str(candidate.get("mp_candidate_id", "")): float((candidate.get("subset_objective_components", {}) or {}).get("objective_candidate_value", 0.0))
        for candidate in mp_candidate_pool
    }
    selected_values = [value_by_candidate.get(str(candidate.get("mp_candidate_id", "")), 0.0) for candidate in selected_candidates]
    high_value_missed: list[Dict[str, Any]] = []
    for candidate in mp_candidate_pool:
        candidate_id = str(candidate.get("mp_candidate_id", ""))
        if candidate_id in selected_ids:
            continue
        components = candidate.get("subset_objective_components", {}) or {}
        value = float(components.get("objective_candidate_value", 0.0))
        if value < 0.50:
            continue
        high_value_missed.append(
            {
                "mp_candidate_id": candidate_id,
                "pool_status": str(candidate.get("pool_status", "")),
                "pool_reject_reason": str(candidate.get("pool_reject_reason", "")),
                "objective_candidate_value": float(value),
                "objective_risk_score": float(components.get("objective_risk_score", 0.0)),
                "objective_feasibility_score": float(components.get("objective_feasibility_score", 0.0)),
                "target_bins": dict(candidate.get("subset_objective_target_bins", {}) or {}),
            }
        )
    high_value_missed.sort(key=lambda item: (-float(item["objective_candidate_value"]), str(item["mp_candidate_id"])))

    high_risk_non_exec: list[Dict[str, Any]] = []
    for details in site_details:
        site = details.get("site", {}) if isinstance(details, Mapping) else {}
        mp_candidate = details.get("mp_candidate", {}) if isinstance(details, Mapping) else {}
        if str(site.get("recipe_status", "")) != "rejected":
            continue
        components = mp_candidate.get("subset_objective_components", {}) or {}
        risk = float(components.get("objective_risk_score", 0.0))
        if risk < 0.66:
            continue
        high_risk_non_exec.append(
            {
                "site_id": str(site.get("site_id", "")),
                "mp_candidate_id": str(mp_candidate.get("mp_candidate_id", "")),
                "reject_reason": str(site.get("reject_reason", "")),
                "objective_risk_score": float(risk),
                "objective_feasibility_score": float(components.get("objective_feasibility_score", 0.0)),
                "subset_objective_marginal_gain": float(mp_candidate.get("subset_objective_marginal_gain", mp_candidate.get("mp_selection_gain", 0.0))),
                "target_bins": dict(mp_candidate.get("subset_objective_target_bins", {}) or {}),
            }
        )
    high_risk_non_exec.sort(key=lambda item: (-float(item["objective_risk_score"]), str(item["mp_candidate_id"])))

    selected_by_category: Dict[str, Dict[str, int]] = {
        category: dict(selected_distribution.get(category, {}))
        for category in TARGET_CATEGORIES
    }
    trace = [
        {
            "selection_order": int(index),
            "mp_candidate_id": str(candidate.get("mp_candidate_id", "")),
            "subset_objective_marginal_gain": float(candidate.get("subset_objective_marginal_gain", candidate.get("mp_selection_gain", 0.0))),
            "objective_candidate_value": float((candidate.get("subset_objective_components", {}) or {}).get("objective_candidate_value", 0.0)),
            "target_bins": dict(candidate.get("subset_objective_target_bins", {}) or {}),
        }
        for index, candidate in enumerate(sorted(selected_candidates, key=lambda item: (-float(item.get("subset_objective_marginal_gain", item.get("mp_selection_gain", 0.0))), str(item.get("mp_candidate_id", "")))))
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "summary": {
            "candidate_count": int(len(mp_candidate_pool)),
            "selected_candidate_count": int(len(selected_candidates)),
            "subset_objective_score": float(sum(selected_values)),
            "subset_objective_gap_count": int(len(gaps)),
            "subset_objective_high_value_missed_count": int(len(high_value_missed)),
            "subset_objective_high_risk_non_executable_count": int(len(high_risk_non_exec)),
            "selected_subset_objective_by_category": selected_by_category,
            "avg_selected_objective_candidate_value": _mean(selected_values),
            "avg_pool_objective_candidate_value": _mean(value_by_candidate.values()),
        },
        "target_distribution": targets,
        "pool_distribution": pool_distribution,
        "selected_distribution": selected_distribution,
        "objective_breakdown": {
            "avg_objective_risk_score": _mean((candidate.get("subset_objective_components", {}) or {}).get("objective_risk_score", 0.0) for candidate in mp_candidate_pool),
            "avg_objective_feasibility_score": _mean((candidate.get("subset_objective_components", {}) or {}).get("objective_feasibility_score", 0.0) for candidate in mp_candidate_pool),
            "avg_objective_candidate_value": _mean(value_by_candidate.values()),
        },
        "coverage_gaps": gaps,
        "selected_marginal_gain_trace": trace,
        "high_value_missed_candidates": high_value_missed[: int(max_high_value)],
        "high_risk_non_executable_candidates": high_risk_non_exec[: int(max_high_value)],
    }
