#!/usr/bin/env python3
"""CD-SEM recipe selector 的同心环上下文审查特征。

本模块是 Zhang 2016 MCMI 思路的轻量审查版，位于 MP / care-area bitmap 已经切出
之后，只读取现有 `clip_bitmap`，不重新 rasterize、不改变 MP/AF/AP 打分。当前版本
用于回答一个 review 问题：MP 周围不同半径上的 layout context 是否足够结构化、是否
存在明显远近场差异，以及哪些半径值得后续进入真正的 outcome-driven 特征选择。

整体流程:
1. 围绕 bitmap 中心，在固定半径上采样二值 occupancy。
2. 为每个半径计算 density、edge crossing、asymmetry 和离散 pattern code。
3. 用无监督 proxy score + 最小半径间距 DP 选出少量非冗余半径，供 audit 使用。

注意: 本模块不使用 hotspot/non-hotspot label，也不把 ring feature 接入 selection score。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence

import numpy as np


DEFAULT_RING_RADII_UM = (0.10, 0.20, 0.35, 0.50, 0.80, 1.20)
DEFAULT_RING_SAMPLE_COUNT = 32


def _clip01(value: float) -> float:
    """把数值限制在 0-1 区间。"""
    if not math.isfinite(float(value)):
        return 0.0
    return float(min(1.0, max(0.0, float(value))))


def _sample_ring_bits(bitmap: np.ndarray, radius_px: float, sample_count: int) -> list[int]:
    """沿指定半径采样固定数量的二值 occupancy。"""
    arr = np.asarray(bitmap, dtype=bool)
    if arr.size == 0 or radius_px <= 0.0:
        return [0 for _ in range(int(sample_count))]
    height, width = arr.shape
    cy = (height - 1) * 0.5
    cx = (width - 1) * 0.5
    bits: list[int] = []
    for index in range(int(sample_count)):
        theta = 2.0 * math.pi * float(index) / float(sample_count)
        y = int(round(cy + float(radius_px) * math.sin(theta)))
        x = int(round(cx + float(radius_px) * math.cos(theta)))
        if 0 <= y < height and 0 <= x < width:
            bits.append(1 if bool(arr[y, x]) else 0)
        else:
            bits.append(0)
    return bits


def _ring_mask(bitmap: np.ndarray, radius_px: float) -> np.ndarray:
    """生成接近指定半径的一像素宽 ring mask。"""
    arr = np.asarray(bitmap, dtype=bool)
    if arr.ndim != 2 or arr.size == 0 or radius_px <= 0.0:
        return np.zeros_like(arr, dtype=bool)
    height, width = arr.shape
    yy, xx = np.indices(arr.shape)
    cy = (height - 1) * 0.5
    cx = (width - 1) * 0.5
    distance = np.sqrt((yy.astype(np.float32) - cy) ** 2 + (xx.astype(np.float32) - cx) ** 2)
    half_width = max(0.75, min(2.0, float(radius_px) * 0.08))
    return np.abs(distance - float(radius_px)) <= float(half_width)


def _pattern_code(bits: Sequence[int]) -> int:
    """把环上采样 bit 压成一个稳定整数 code。"""
    code = 0
    for index, bit in enumerate(bits[:63]):
        if int(bit):
            code |= 1 << int(index)
    return int(code)


def _edge_crossing_score(bits: Sequence[int]) -> float:
    """统计环上相邻采样点的 0/1 跳变比例。"""
    if not bits:
        return 0.0
    transitions = 0
    count = len(bits)
    for index in range(count):
        transitions += 1 if int(bits[index]) != int(bits[(index + 1) % count]) else 0
    return _clip01(float(transitions) / float(max(1, count)))


def _asymmetry_score(bitmap: np.ndarray, mask: np.ndarray) -> float:
    """估计指定 ring 在上下/左右方向上的占用不对称性。"""
    arr = np.asarray(bitmap, dtype=bool)
    if arr.size == 0 or not np.any(mask):
        return 0.0
    height, width = arr.shape
    yy, xx = np.indices(arr.shape)
    cy = (height - 1) * 0.5
    cx = (width - 1) * 0.5

    def density(region: np.ndarray) -> float:
        region_mask = np.logical_and(mask, region)
        if not np.any(region_mask):
            return 0.0
        return float(np.count_nonzero(np.logical_and(arr, region_mask))) / float(np.count_nonzero(region_mask))

    left = density(xx < cx)
    right = density(xx >= cx)
    down = density(yy < cy)
    up = density(yy >= cy)
    return _clip01(0.5 * abs(left - right) + 0.5 * abs(up - down))


def compute_ring_context(
    clip_bitmap: Any,
    *,
    pixel_size_um: float,
    radii_um: Sequence[float] | None = None,
    sample_count: int = DEFAULT_RING_SAMPLE_COUNT,
) -> Dict[str, Any]:
    """从已有 clip bitmap 计算同心环上下文特征。"""
    arr = np.asarray(clip_bitmap, dtype=bool)
    radii = [float(value) for value in (radii_um or DEFAULT_RING_RADII_UM)]
    pixel = float(pixel_size_um) if float(pixel_size_um) > 0.0 else 1.0
    density_profile: list[float] = []
    edge_profile: list[float] = []
    asymmetry_profile: list[float] = []
    pattern_codes: list[int] = []
    proxy_scores: list[float] = []

    for radius_um in radii:
        radius_px = float(radius_um) / pixel
        mask = _ring_mask(arr, radius_px)
        if np.any(mask):
            density = float(np.count_nonzero(np.logical_and(arr, mask))) / float(np.count_nonzero(mask))
        else:
            density = 0.0
        bits = _sample_ring_bits(arr, radius_px, int(sample_count))
        edge = _edge_crossing_score(bits)
        asymmetry = _asymmetry_score(arr, mask)
        density_balance = _clip01(1.0 - abs(float(density) - 0.35) / 0.35)
        proxy = _clip01(0.45 * edge + 0.35 * density_balance + 0.20 * asymmetry)
        density_profile.append(float(_clip01(density)))
        edge_profile.append(float(edge))
        asymmetry_profile.append(float(asymmetry))
        pattern_codes.append(_pattern_code(bits))
        proxy_scores.append(float(proxy))

    selected = select_nonredundant_radii(radii, proxy_scores, max_count=3, min_spacing_um=0.20)
    return {
        "ring_radii_um": [float(value) for value in radii],
        "ring_density_profile": density_profile,
        "ring_edge_crossing_profile": edge_profile,
        "ring_asymmetry_profile": asymmetry_profile,
        "ring_pattern_code": pattern_codes,
        "ring_proxy_score": proxy_scores,
        "ring_selected_radii_um": [float(value) for value in selected["selected_radii_um"]],
        "ring_selected_indices": [int(value) for value in selected["selected_indices"]],
        "ring_selected_proxy_score": float(selected["selected_proxy_score"]),
    }


def select_nonredundant_radii(
    radii_um: Sequence[float],
    scores: Sequence[float],
    *,
    max_count: int,
    min_spacing_um: float,
) -> Dict[str, Any]:
    """用带最小半径间距约束的 DP 选择少量非冗余半径。"""
    items = sorted(
        [(float(radius), _clip01(float(score)), int(index)) for index, (radius, score) in enumerate(zip(radii_um, scores))],
        key=lambda item: item[0],
    )
    if not items or int(max_count) <= 0:
        return {"selected_indices": [], "selected_radii_um": [], "selected_proxy_score": 0.0}

    count = len(items)
    limit = min(int(max_count), count)
    prev_allowed: list[int] = []
    for i, (radius, _, _) in enumerate(items):
        prev = -1
        for j in range(i - 1, -1, -1):
            if radius - items[j][0] >= float(min_spacing_um) - 1e-12:
                prev = j
                break
        prev_allowed.append(prev)

    dp = [[0.0 for _ in range(limit + 1)] for _ in range(count + 1)]
    take = [[False for _ in range(limit + 1)] for _ in range(count + 1)]
    for i in range(1, count + 1):
        radius, score, _ = items[i - 1]
        for k in range(1, limit + 1):
            skip_score = dp[i - 1][k]
            prev_index = prev_allowed[i - 1]
            take_score = float(score) + dp[prev_index + 1][k - 1]
            if take_score > skip_score + 1e-12:
                dp[i][k] = take_score
                take[i][k] = True
            else:
                dp[i][k] = skip_score

    best_k = max(range(limit + 1), key=lambda k: dp[count][k])
    selected_items: list[tuple[float, float, int]] = []
    i = count
    k = best_k
    while i > 0 and k > 0:
        if take[i][k]:
            selected_items.append(items[i - 1])
            i = prev_allowed[i - 1] + 1
            k -= 1
        else:
            i -= 1
    selected_items.reverse()
    return {
        "selected_indices": [int(item[2]) for item in selected_items],
        "selected_radii_um": [float(item[0]) for item in selected_items],
        "selected_proxy_score": float(sum(float(item[1]) for item in selected_items)),
    }


def ring_context_vector(context: Mapping[str, Any]) -> np.ndarray:
    """把 ring-context audit 字段压成紧凑 float 向量，供 pattern memory 保存。"""
    values: list[float] = []
    for key in ("ring_density_profile", "ring_edge_crossing_profile", "ring_asymmetry_profile", "ring_proxy_score"):
        values.extend(float(value) for value in context.get(key, []) or [])
    return np.asarray(values, dtype=np.float32)
