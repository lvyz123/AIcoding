"""AP/AF layout-side matchability 审查工具。

本模块位于 CD-SEM recipe selector 的执行可行性审查层，只读取已经 rasterize
出的 bitmap，不重新切窗，也不接入真实 SEM、SIFT/FLANN、GAN 或 optical flow。

算法原则:
1. 用 2x2 局部占用变化近似 keypoint / corner / line-end 支撑度。
2. 用边缘方向分布和 keypoint 空间分散度估计模板是否容易稳定匹配。
3. 用轻量自相关 proxy 估计周期重复风险。
4. 输出只作为 review/audit 字段，不改变 AP/AF 原有 score 和 hard gate。
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np


def _clip01(value: float) -> float:
    """把审查分数裁剪到 0-1。"""
    return float(max(0.0, min(1.0, float(value))))


def _as_bitmap(bitmap: np.ndarray) -> np.ndarray:
    """把输入安全转换成二维 bool bitmap。"""
    arr = np.asarray(bitmap, dtype=bool)
    if arr.ndim != 2:
        return np.zeros((0, 0), dtype=bool)
    return arr


def _safe_iou(left: np.ndarray, right: np.ndarray) -> float:
    """计算两个 bool patch 的 IoU；空交并时返回 0。"""
    inter = int(np.logical_and(left, right).sum())
    union = int(np.logical_or(left, right).sum())
    if union <= 0:
        return 0.0
    return float(inter / union)


def _keypoint_coordinates(bitmap: np.ndarray) -> np.ndarray:
    """用 2x2 patch 的单侧占用变化提取轻量 keypoint 坐标。"""
    b = _as_bitmap(bitmap)
    if b.shape[0] < 2 or b.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    patch_sum = (
        b[:-1, :-1].astype(np.int16)
        + b[1:, :-1].astype(np.int16)
        + b[:-1, 1:].astype(np.int16)
        + b[1:, 1:].astype(np.int16)
    )
    # 1/3 表示 corner、line-end 或 jog；2 且对角占用表示斜向/交错变化。
    diag_change = np.logical_xor(b[:-1, :-1], b[1:, 1:]) & np.logical_xor(b[1:, :-1], b[:-1, 1:])
    keypoint_map = (patch_sum == 1) | (patch_sum == 3) | ((patch_sum == 2) & diag_change)
    ys, xs = np.nonzero(keypoint_map)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.column_stack([ys.astype(np.float32) + 0.5, xs.astype(np.float32) + 0.5])


def _keypoint_spread(coords: np.ndarray, shape: tuple[int, int]) -> float:
    """估计 keypoint 在窗口中的二维分散程度。"""
    if coords.shape[0] < 2 or shape[0] <= 1 or shape[1] <= 1:
        return 0.0
    y_span = float(coords[:, 0].max() - coords[:, 0].min()) / float(max(1, shape[0] - 1))
    x_span = float(coords[:, 1].max() - coords[:, 1].min()) / float(max(1, shape[1] - 1))
    return _clip01(math.sqrt(max(0.0, x_span * y_span)))


def _orientation_entropy(bitmap: np.ndarray) -> float:
    """用水平、垂直和两类对角边缘响应估计方向熵。"""
    b = _as_bitmap(bitmap)
    if b.shape[0] < 2 or b.shape[1] < 2:
        return 0.0
    counts = np.asarray(
        [
            int(np.count_nonzero(b[:, 1:] != b[:, :-1])),
            int(np.count_nonzero(b[1:, :] != b[:-1, :])),
            int(np.count_nonzero(b[1:, 1:] != b[:-1, :-1])),
            int(np.count_nonzero(b[1:, :-1] != b[:-1, 1:])),
        ],
        dtype=np.float64,
    )
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    probs = counts[counts > 0] / total
    entropy = -float(np.sum(probs * np.log(probs)))
    return _clip01(entropy / math.log(4.0))


def _self_periodicity_penalty(bitmap: np.ndarray) -> float:
    """用少量平移自相关估计周期重复风险。"""
    b = _as_bitmap(bitmap)
    if b.shape[0] < 4 or b.shape[1] < 4 or not np.any(b):
        return 0.0
    shifts: set[tuple[int, int]] = set()
    for frac in (0.20, 0.25, 0.33, 0.50):
        dy = max(1, int(round(b.shape[0] * frac)))
        dx = max(1, int(round(b.shape[1] * frac)))
        shifts.update({(dy, 0), (-dy, 0), (0, dx), (0, -dx), (dy, dx), (-dy, dx)})
    best = 0.0
    for dy, dx in shifts:
        y0_a = max(0, dy)
        y1_a = min(b.shape[0], b.shape[0] + dy)
        x0_a = max(0, dx)
        x1_a = min(b.shape[1], b.shape[1] + dx)
        y0_b = max(0, -dy)
        y1_b = min(b.shape[0], b.shape[0] - dy)
        x0_b = max(0, -dx)
        x1_b = min(b.shape[1], b.shape[1] - dx)
        if y1_a <= y0_a or x1_a <= x0_a:
            continue
        best = max(best, _safe_iou(b[y0_a:y1_a, x0_a:x1_a], b[y0_b:y1_b, x0_b:x1_b]))
    return _clip01(best)


def compute_ap_matchability(
    bitmap: np.ndarray,
    *,
    descriptor_margin: float,
    nearest_similarity: float,
    peak_count: int,
) -> Dict[str, float]:
    """计算 AP 模板在 layout 侧的可匹配性审查指标。"""
    b = _as_bitmap(bitmap)
    coords = _keypoint_coordinates(b)
    keypoint_count = int(coords.shape[0])
    keypoint_density = _clip01(1.0 - math.exp(-float(keypoint_count) / 6.0))
    spread = _keypoint_spread(coords, b.shape)
    orientation = _orientation_entropy(b)
    self_periodicity = _self_periodicity_penalty(b)
    nearest_penalty = _clip01(float(nearest_similarity))
    peak_penalty = _clip01((float(max(1, int(peak_count))) - 1.0) / 3.0)
    blended_periodicity = _clip01(0.50 * self_periodicity + 0.30 * nearest_penalty + 0.20 * peak_penalty)
    periodicity = max(blended_periodicity, 0.75 * self_periodicity, 0.75 * nearest_penalty, 0.75 * peak_penalty)
    margin = _clip01(float(descriptor_margin))
    score = _clip01(
        0.25 * keypoint_density
        + 0.20 * spread
        + 0.20 * orientation
        + 0.20 * margin
        + 0.15 * (1.0 - periodicity)
    )
    return {
        "keypoint_count": float(keypoint_count),
        "keypoint_density_score": float(keypoint_density),
        "keypoint_spread": float(spread),
        "orientation_entropy": float(orientation),
        "descriptor_margin": float(margin),
        "periodicity_penalty": float(periodicity),
        "layout_matchability_score": float(score),
    }


def compute_af_matchability(
    bitmap: np.ndarray,
    *,
    focus_quality: float,
) -> Dict[str, float]:
    """计算 AF 窗口在 layout 侧的可对焦/可匹配性审查指标。"""
    b = _as_bitmap(bitmap)
    coords = _keypoint_coordinates(b)
    keypoint_count = int(coords.shape[0])
    keypoint_density = _clip01(1.0 - math.exp(-float(keypoint_count) / 6.0))
    spread = _keypoint_spread(coords, b.shape)
    orientation = _orientation_entropy(b)
    periodicity = _self_periodicity_penalty(b)
    score = _clip01(
        0.30 * _clip01(float(focus_quality))
        + 0.25 * keypoint_density
        + 0.20 * spread
        + 0.15 * orientation
        + 0.10 * (1.0 - periodicity)
    )
    return {
        "keypoint_count": float(keypoint_count),
        "keypoint_density_score": float(keypoint_density),
        "keypoint_spread": float(spread),
        "orientation_entropy": float(orientation),
        "descriptor_margin": 0.5,
        "periodicity_penalty": float(periodicity),
        "layout_matchability_score": float(score),
    }
