#!/usr/bin/env python3
"""CD-SEM recipe selector 的低优先级审查信号层。

本模块位于主选择流程之后、review 输出之前，只补充 audit 信息，不改变 MP
priority、global selection、AF/AP hard gate 或 pattern memory prior。当前实现有四类信号：

1. graph-lite context：从已经切出的 bitmap 中提取连通块、近邻关系和环境复杂度。
2. evidence / contradiction：把几何、behavior、care-area、voting、ring、memory 证据拆开，
   并标记高价值但证据弱、强证据但执行失败等矛盾。
3. TNSB / HTC taxonomy：只做 review 标签，帮助人工区分新颖 weak pattern 与高重复低价值候选。
4. expected feasibility proxy：用当前审查信号估计 AF/AP/recipe 可执行性，但不参与排序。

这些字段的作用是让后续优化有可检查的证据基础，而不是提前把低置信度信号写进主算法。
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Mapping, MutableMapping, Sequence

import numpy as np


REVIEW_AUDIT_SCHEMA_VERSION = "recipe_review_evidence_audit_v1"


def _clip01(value: float) -> float:
    """把数值限制在 0-1 区间。"""
    if not math.isfinite(float(value)):
        return 0.0
    return float(min(1.0, max(0.0, float(value))))


def _safe_float(mapping: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    """从 mapping 中读取 float，缺失或非法时返回默认值。"""
    try:
        return float(mapping.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _edge_density(bitmap: np.ndarray) -> float:
    """计算 bitmap 内部 0/1 跳变比例，作为轻量边缘密度。"""
    arr = np.asarray(bitmap, dtype=bool)
    if arr.ndim != 2 or arr.size == 0:
        return 0.0
    vertical = np.count_nonzero(arr[1:, :] != arr[:-1, :])
    horizontal = np.count_nonzero(arr[:, 1:] != arr[:, :-1])
    denom = max(1, arr.shape[0] * max(0, arr.shape[1] - 1) + arr.shape[1] * max(0, arr.shape[0] - 1))
    return _clip01(float(vertical + horizontal) / float(denom))


def _connected_components(bitmap: np.ndarray) -> list[Dict[str, Any]]:
    """用 4-neighbor 连通块近似 layout graph node。"""
    arr = np.asarray(bitmap, dtype=bool)
    if arr.ndim != 2 or arr.size == 0:
        return []
    visited = np.zeros(arr.shape, dtype=bool)
    components: list[Dict[str, Any]] = []
    height, width = arr.shape
    for y in range(height):
        for x in range(width):
            if not bool(arr[y, x]) or bool(visited[y, x]):
                continue
            stack = [(y, x)]
            visited[y, x] = True
            pixels: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                pixels.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and bool(arr[ny, nx]) and not bool(visited[ny, nx]):
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            ys = np.asarray([item[0] for item in pixels], dtype=np.float32)
            xs = np.asarray([item[1] for item in pixels], dtype=np.float32)
            y0, y1 = int(np.min(ys)), int(np.max(ys))
            x0, x1 = int(np.min(xs)), int(np.max(xs))
            bbox_h = max(1, y1 - y0 + 1)
            bbox_w = max(1, x1 - x0 + 1)
            components.append(
                {
                    "area_px": int(len(pixels)),
                    "center_y": float(np.mean(ys)),
                    "center_x": float(np.mean(xs)),
                    "bbox": [int(y0), int(x0), int(y1), int(x1)],
                    "aspect_ratio": float(max(bbox_h, bbox_w) / max(1, min(bbox_h, bbox_w))),
                }
            )
    return components


def _component_relation_stats(components: Sequence[Mapping[str, Any]], shape: tuple[int, int]) -> Dict[str, float]:
    """用连通块中心距离和方向近似 graph edge 分布。"""
    node_count = len(components)
    if node_count < 2:
        return {
            "graph_edge_count": 0.0,
            "graph_edge_density": 0.0,
            "graph_relation_balance": 0.0,
            "graph_horizontal_relation_ratio": 0.0,
            "graph_vertical_relation_ratio": 0.0,
            "graph_diagonal_relation_ratio": 0.0,
        }
    height, width = int(shape[0]), int(shape[1])
    distance_gate = max(2.0, 0.35 * float(max(height, width)))
    horizontal = 0
    vertical = 0
    diagonal = 0
    edge_count = 0
    for left_index in range(node_count):
        left = components[left_index]
        lx = float(left["center_x"])
        ly = float(left["center_y"])
        for right in components[left_index + 1 :]:
            dx = float(right["center_x"]) - lx
            dy = float(right["center_y"]) - ly
            distance = math.hypot(dx, dy)
            if distance > distance_gate:
                continue
            edge_count += 1
            if abs(dx) >= abs(dy) * 1.7:
                horizontal += 1
            elif abs(dy) >= abs(dx) * 1.7:
                vertical += 1
            else:
                diagonal += 1
    possible = max(1, node_count * (node_count - 1) // 2)
    if edge_count <= 0:
        balance = 0.0
    else:
        ratios = np.asarray([horizontal, vertical, diagonal], dtype=np.float32) / float(edge_count)
        balance = _clip01(1.0 - float(np.max(ratios) - np.min(ratios)))
    return {
        "graph_edge_count": float(edge_count),
        "graph_edge_density": _clip01(float(edge_count) / float(possible)),
        "graph_relation_balance": float(balance),
        "graph_horizontal_relation_ratio": float(horizontal) / float(edge_count) if edge_count else 0.0,
        "graph_vertical_relation_ratio": float(vertical) / float(edge_count) if edge_count else 0.0,
        "graph_diagonal_relation_ratio": float(diagonal) / float(edge_count) if edge_count else 0.0,
    }


def compute_graph_context(clip_bitmap: Any) -> Dict[str, Any]:
    """从已切出的 MP bitmap 中计算 graph-lite 审查字段。"""
    arr = np.asarray(clip_bitmap, dtype=bool)
    if arr.ndim != 2 or arr.size == 0:
        return {
            "graph_node_count": 0,
            "graph_edge_count": 0,
            "graph_edge_density": 0.0,
            "graph_env_complexity": 0.0,
            "graph_feature_vector": [0.0] * 8,
        }
    components = _connected_components(arr)
    node_count = len(components)
    density = float(np.count_nonzero(arr)) / float(arr.size)
    edge_density = _edge_density(arr)
    areas = np.asarray([float(item["area_px"]) for item in components], dtype=np.float32)
    aspects = np.asarray([float(item["aspect_ratio"]) for item in components], dtype=np.float32)
    area_cv = float(np.std(areas) / max(float(np.mean(areas)), 1e-9)) if areas.size else 0.0
    aspect_mean = float(np.mean(np.minimum(aspects, 8.0)) / 8.0) if aspects.size else 0.0
    relation = _component_relation_stats(components, arr.shape)
    node_score = _clip01(float(node_count) / 8.0)
    density_balance = _clip01(1.0 - abs(float(density) - 0.35) / 0.35)
    graph_env_complexity = _clip01(
        0.25 * node_score
        + 0.25 * float(relation["graph_edge_density"])
        + 0.25 * edge_density
        + 0.15 * density_balance
        + 0.10 * _clip01(area_cv)
    )
    feature_vector = [
        node_score,
        float(relation["graph_edge_density"]),
        float(relation["graph_relation_balance"]),
        edge_density,
        _clip01(density),
        density_balance,
        _clip01(area_cv),
        _clip01(aspect_mean),
    ]
    return {
        "graph_node_count": int(node_count),
        "graph_edge_count": int(relation["graph_edge_count"]),
        "graph_edge_density": float(relation["graph_edge_density"]),
        "graph_relation_balance": float(relation["graph_relation_balance"]),
        "graph_horizontal_relation_ratio": float(relation["graph_horizontal_relation_ratio"]),
        "graph_vertical_relation_ratio": float(relation["graph_vertical_relation_ratio"]),
        "graph_diagonal_relation_ratio": float(relation["graph_diagonal_relation_ratio"]),
        "graph_bitmap_density": float(_clip01(density)),
        "graph_bitmap_edge_density": float(edge_density),
        "graph_component_area_cv": float(_clip01(area_cv)),
        "graph_component_aspect_score": float(_clip01(aspect_mean)),
        "graph_env_complexity": float(graph_env_complexity),
        "graph_nearest_similarity": 0.0,
        "mp_graph_rarity": 1.0,
        "care_area_graph_similarity": 0.0,
        "care_area_graph_support_count": 0,
        "graph_feature_vector": [float(value) for value in feature_vector],
    }


def graph_context_vector(graph_context: Mapping[str, Any]) -> np.ndarray:
    """把 graph-lite 审查字段压成短向量，用于 pool 内相似度和 rarity。"""
    return np.asarray(graph_context.get("graph_feature_vector", []) or [], dtype=np.float32)


def _normalized_matrix(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """按行 L2 归一化；空向量返回空矩阵。"""
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)
    vector_dim = max([int(vector.size) for vector in vectors] + [1])
    matrix = np.zeros((len(vectors), vector_dim), dtype=np.float32)
    for index, vector in enumerate(vectors):
        arr = np.asarray(vector, dtype=np.float32)
        if arr.size:
            matrix[index, : min(vector_dim, arr.size)] = arr[:vector_dim]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, np.maximum(norms, 1e-12), out=np.zeros_like(matrix), where=norms > 0)


def enrich_graph_pool_context(graph_contexts: Sequence[MutableMapping[str, Any]], family_ids: Sequence[str]) -> Dict[str, Any]:
    """在全局 MP pool 内补充 graph nearest similarity、rarity 和同 family 支持度。"""
    vectors = [graph_context_vector(context) for context in graph_contexts]
    matrix = _normalized_matrix(vectors)
    if matrix.size == 0:
        return {"candidate_count": int(len(graph_contexts)), "avg_mp_graph_rarity": 0.0}
    sims = matrix @ matrix.T
    for index, context in enumerate(graph_contexts):
        if len(graph_contexts) <= 1:
            nearest = 0.0
        else:
            row = sims[index].copy()
            row[index] = -1.0
            nearest = max(0.0, float(np.max(row)))
        same_family = [
            float(sims[index, other])
            for other in range(len(graph_contexts))
            if other != index and str(family_ids[other]) == str(family_ids[index])
        ]
        context["graph_nearest_similarity"] = float(_clip01(nearest))
        context["mp_graph_rarity"] = float(_clip01(1.0 - nearest))
        context["care_area_graph_similarity"] = float(_clip01(max(same_family))) if same_family else 0.0
        context["care_area_graph_support_count"] = int(len(same_family))
    rarities = [float(context.get("mp_graph_rarity", 0.0)) for context in graph_contexts]
    return {
        "candidate_count": int(len(graph_contexts)),
        "avg_mp_graph_rarity": float(np.mean(rarities)) if rarities else 0.0,
        "candidate_with_family_graph_support_count": int(sum(1 for context in graph_contexts if int(context.get("care_area_graph_support_count", 0)) > 0)),
    }


def _ring_evidence(ring_context: Mapping[str, Any]) -> float:
    """把 ring-context proxy 压成一个审查证据分数。"""
    proxy = [float(value) for value in ring_context.get("ring_proxy_score", []) or []]
    selected = float(ring_context.get("ring_selected_proxy_score", 0.0) or 0.0)
    if not proxy:
        return _clip01(selected)
    return _clip01(0.65 * float(np.mean(proxy)) + 0.35 * min(1.0, selected / 3.0))


def _memory_success_signal(memory_prior: Mapping[str, Any]) -> float:
    """从只读 memory prior 中取一个中性默认的历史可执行性信号。"""
    confidence = _safe_float(memory_prior, "memory_prior_confidence", 0.0)
    success = _safe_float(memory_prior, "memory_recipe_success_prior", 0.5)
    waste = _safe_float(memory_prior, "memory_waste_prior", 0.5)
    return _clip01((1.0 - confidence) * 0.5 + confidence * (0.65 * success + 0.35 * (1.0 - waste)))


def compute_evidence_contradiction_audit(
    candidate: Mapping[str, Any],
    *,
    graph_context: Mapping[str, Any],
    ring_context: Mapping[str, Any],
    memory_prior: Mapping[str, Any],
) -> Dict[str, Any]:
    """拆分 MP 候选的多源证据，并标记不需要真实 label 的静态矛盾。"""
    discovery = candidate.get("mp_discovery_components", {}) or {}
    geometry = _clip01(_safe_float(candidate, "mp_hotspot_score", 0.0))
    behavior = _clip01(_safe_float(candidate, "behavior_risk", _safe_float(candidate, "effective_behavior_risk", 0.0)))
    care_area = _clip01(0.50 * _safe_float(candidate, "care_area_match_score", 0.0) + 0.50 * _safe_float(candidate, "care_area_homogeneity_score", 0.0))
    voting = _clip01(
        max(
            _safe_float(candidate, "mp_localization_confidence", 0.0),
            _safe_float(discovery, "proposal_voting", 0.0),
            _safe_float(discovery, "voting_confidence", 0.0),
            _safe_float(discovery, "supporting_anchor_count", 0.0) / 3.0,
        )
    )
    graph = _clip01(_safe_float(graph_context, "graph_env_complexity", 0.0))
    ring = _ring_evidence(ring_context)
    memory = _memory_success_signal(memory_prior)
    evidence = _clip01(
        0.24 * geometry
        + 0.16 * behavior
        + 0.18 * care_area
        + 0.14 * voting
        + 0.10 * graph
        + 0.10 * ring
        + 0.08 * memory
    )
    tags: list[str] = []
    if _safe_float(candidate, "metrology_priority_score", 0.0) >= 0.66 and evidence < 0.40:
        tags.append("high_priority_low_evidence")
    if evidence >= 0.66 and _safe_float(candidate, "recipe_waste_penalty", 0.0) >= 0.66:
        tags.append("high_evidence_high_waste")
    if evidence >= 0.66 and _safe_float(memory_prior, "memory_prior_confidence", 0.0) >= 0.30 and _safe_float(memory_prior, "memory_waste_prior", 0.5) >= 0.60:
        tags.append("memory_conflicts_with_current_evidence")
    return {
        "geometry_evidence": float(geometry),
        "behavior_evidence": float(behavior),
        "care_area_evidence": float(care_area),
        "voting_evidence": float(voting),
        "graph_evidence": float(graph),
        "ring_evidence": float(ring),
        "memory_evidence": float(memory),
        "defect_evidence_proxy_score": float(evidence),
        "static_contradiction_tags": tags,
        "outcome_contradiction_tags": [],
    }


def compute_pattern_taxonomy_audit(
    candidate: Mapping[str, Any],
    *,
    graph_context: Mapping[str, Any],
    evidence_audit: Mapping[str, Any],
    memory_prior: Mapping[str, Any],
) -> Dict[str, Any]:
    """生成 TNSB/HTC 风格的审查标签，不把标签接入排序。"""
    memory_conf = _safe_float(memory_prior, "memory_prior_confidence", 0.0)
    memory_nearest = _safe_float(memory_prior, "memory_nearest_similarity", 0.0)
    memory_newness = _clip01(1.0 - memory_conf * memory_nearest)
    graph_rarity = _safe_float(graph_context, "mp_graph_rarity", 1.0)
    pattern_novelty = _safe_float(candidate, "pattern_novelty", _safe_float(candidate, "pattern_rarity", 0.0))
    family_support = _clip01(
        0.45 * _safe_float(candidate, "care_area_match_score", 0.0)
        + 0.35 * _safe_float(candidate, "care_area_homogeneity_score", 0.0)
        + 0.20 * _safe_float(graph_context, "care_area_graph_similarity", 0.0)
    )
    evidence = _safe_float(evidence_audit, "defect_evidence_proxy_score", 0.0)
    waste = _safe_float(candidate, "recipe_waste_penalty", 0.0)
    repetition = _clip01(
        0.45 * _safe_float(memory_prior, "memory_nearest_similarity", 0.0)
        + 0.35 * _safe_float(graph_context, "graph_nearest_similarity", 0.0)
        + 0.20 * (1.0 - _safe_float(candidate, "pattern_novelty", pattern_novelty))
    )
    tnsb_like = _clip01(0.30 * evidence + 0.25 * pattern_novelty + 0.20 * graph_rarity + 0.15 * memory_newness + 0.10 * (1.0 - family_support))
    htc_like = _clip01(0.30 * repetition + 0.25 * family_support + 0.20 * waste + 0.15 * (1.0 - evidence) + 0.10 * _safe_float(memory_prior, "memory_waste_prior", 0.5))
    known_like = _clip01(0.45 * family_support + 0.35 * (memory_conf * memory_nearest) + 0.20 * (1.0 - graph_rarity))
    if tnsb_like >= 0.62 and tnsb_like >= htc_like + 0.08:
        taxonomy_class = "tnsb_like"
    elif htc_like >= 0.62 and htc_like >= tnsb_like + 0.08:
        taxonomy_class = "htc_like"
    elif known_like >= 0.62:
        taxonomy_class = "known_like"
    else:
        taxonomy_class = "ambiguous"
    return {
        "tnsb_like_score": float(tnsb_like),
        "htc_like_score": float(htc_like),
        "known_like_score": float(known_like),
        "pattern_taxonomy_class": taxonomy_class,
        "taxonomy_components": {
            "memory_newness": float(memory_newness),
            "graph_rarity": float(graph_rarity),
            "pattern_novelty": float(pattern_novelty),
            "family_support": float(family_support),
            "repetition_proxy": float(repetition),
            "evidence_proxy": float(evidence),
            "waste_proxy": float(waste),
        },
    }


def compute_expected_feasibility_audit(
    candidate: Mapping[str, Any],
    *,
    graph_context: Mapping[str, Any],
    memory_prior: Mapping[str, Any],
) -> Dict[str, Any]:
    """估计 recipe 可执行性，只用于 audit，不参与 backfill 或排序。"""
    confidence = _safe_float(memory_prior, "memory_prior_confidence", 0.0)
    memory_af = (1.0 - confidence) * 0.5 + confidence * _safe_float(memory_prior, "memory_af_success_prior", 0.5)
    memory_ap = (1.0 - confidence) * 0.5 + confidence * _safe_float(memory_prior, "memory_ap_success_prior", 0.5)
    memory_dup_safe = (1.0 - confidence) * 0.5 + confidence * (1.0 - _safe_float(memory_prior, "memory_ap_duplicate_prior", 0.5))
    waste_safe = 1.0 - _safe_float(candidate, "recipe_waste_penalty", 0.0)
    localization = _safe_float(candidate, "mp_localization_confidence", 0.0)
    graph_complexity = _safe_float(graph_context, "graph_env_complexity", 0.0)
    graph_unique = _safe_float(graph_context, "mp_graph_rarity", 0.0)
    mp_valid = 1.0 if bool(candidate.get("mp_verified", True)) else 0.0
    af_proxy = _clip01(0.45 * memory_af + 0.25 * waste_safe + 0.20 * graph_complexity + 0.10 * localization)
    ap_proxy = _clip01(0.40 * memory_ap + 0.25 * memory_dup_safe + 0.20 * graph_unique + 0.15 * waste_safe)
    recipe_proxy = _clip01(0.35 * af_proxy + 0.35 * ap_proxy + 0.20 * mp_valid + 0.10 * waste_safe)
    return {
        "expected_af_pass_proxy": float(af_proxy),
        "expected_ap_pass_proxy": float(ap_proxy),
        "expected_recipe_feasibility_proxy": float(recipe_proxy),
        "feasibility_prior_confidence": float(confidence),
    }


def update_evidence_with_site_outcome(details: MutableMapping[str, Any]) -> None:
    """在 AF/AP 构造完成后，把 outcome contradiction 写回 site_summary 详情。"""
    site = details.get("site", {}) or {}
    mp_candidate = details.get("mp_candidate", {}) or {}
    audit = mp_candidate.get("evidence_contradiction_audit", {}) or {}
    evidence = _safe_float(audit, "defect_evidence_proxy_score", 0.0)
    tags = list(audit.get("outcome_contradiction_tags", []) or [])
    reject_reason = str(site.get("reject_reason", ""))
    if evidence >= 0.66 and site.get("recipe_status") == "rejected":
        tags.append("rejected_high_evidence")
    if evidence >= 0.66 and ("no_safe_af" in reject_reason or "no_unique_ap" in reject_reason):
        tags.append("high_evidence_no_afap")
    if evidence >= 0.66 and "post_selection_refine_failed" in reject_reason:
        tags.append("high_evidence_refine_failed")
    ap_candidate = details.get("ap_candidate") or {}
    if isinstance(ap_candidate, Mapping):
        ap_matchability = _safe_float(ap_candidate.get("components", {}) or {}, "layout_matchability_score", 0.0)
        if ap_matchability >= 0.65 and "no_unique_ap" in reject_reason:
            tags.append("high_matchability_no_unique_ap")
    deduped = []
    for tag in tags:
        if tag not in deduped:
            deduped.append(str(tag))
    audit["outcome_contradiction_tags"] = deduped
    mp_candidate["evidence_contradiction_audit"] = audit
    details["mp_candidate"] = mp_candidate


def build_review_evidence_audit(
    mp_candidate_pool: Sequence[Mapping[str, Any]],
    site_details: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """汇总 graph/evidence/taxonomy/feasibility 审查字段，写入 recipe_sites.json。"""
    taxonomy_counter: Counter[str] = Counter()
    static_tags: Counter[str] = Counter()
    outcome_tags: Counter[str] = Counter()
    feasibility_values: list[float] = []
    graph_rarity_values: list[float] = []
    for candidate in mp_candidate_pool:
        taxonomy = candidate.get("pattern_taxonomy_audit", {}) or {}
        taxonomy_counter[str(taxonomy.get("pattern_taxonomy_class", "unknown"))] += 1
        evidence = candidate.get("evidence_contradiction_audit", {}) or {}
        static_tags.update(str(tag) for tag in evidence.get("static_contradiction_tags", []) or [])
        feasibility = candidate.get("expected_feasibility_audit", {}) or {}
        feasibility_values.append(_safe_float(feasibility, "expected_recipe_feasibility_proxy", 0.0))
        graph = candidate.get("graph_context_audit", {}) or {}
        graph_rarity_values.append(_safe_float(graph, "mp_graph_rarity", 0.0))
    for details in site_details:
        mp_candidate = details.get("mp_candidate", {}) if isinstance(details, Mapping) else {}
        evidence = mp_candidate.get("evidence_contradiction_audit", {}) if isinstance(mp_candidate, Mapping) else {}
        outcome_tags.update(str(tag) for tag in evidence.get("outcome_contradiction_tags", []) or [])
    return {
        "schema_version": REVIEW_AUDIT_SCHEMA_VERSION,
        "summary": {
            "candidate_count": int(len(mp_candidate_pool)),
            "site_detail_count": int(len(site_details)),
            "taxonomy_by_class": dict(taxonomy_counter),
            "static_contradiction_by_tag": dict(static_tags),
            "outcome_contradiction_by_tag": dict(outcome_tags),
            "avg_expected_recipe_feasibility_proxy": float(np.mean(feasibility_values)) if feasibility_values else 0.0,
            "avg_mp_graph_rarity": float(np.mean(graph_rarity_values)) if graph_rarity_values else 0.0,
        },
    }
