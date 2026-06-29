#!/usr/bin/env python3
"""Care-area / marker 局部 MP candidate discovery 模块。

本模块覆盖主流程中的 MP discovery 步骤，由 `recipe_site_selector.py` 调用，不是独立
recipe 主入口。它既用于 representative marker 周边的 seed care-area family 提取，也用于
expanded care-area instance 内定位真实 MP。当前实现只用 layout geometry 和上游传入的
effective behavior risk，不做 full-chip blind scan、监督训练，也不重新裁剪 candidate 级
behavior image。

整体算法流程：
1. 以上游给定中心点为中心，在固定半径内生成 center baseline、local grid 和
   critical geometry anchors。
2. critical geometry anchors 只使用版图几何：corner/jog、line-end、narrow space、
   bridge/pinch proxy 和 density-transition proxy。
3. 对每个候选中心调用 `rasterize_centered_window` 切出 MP bitmap。
4. 用 critical geometry、effective behavior risk、与中心窗口的相似度、
   layout complexity、局部 rarity 和 proposal voting 计算 `mp_hotspot_score`。
5. 对局部相邻且图形高度相似的候选做轻量 NMS，并输出每个局部中心的 top-K MP
   candidates 供全局 budget selection 使用。

适用边界：
- 只在代表 marker 或 accepted care-area instance 的局部邻域内搜索，不做任意窗口盲扫。
- 不新增监督训练，不调用 lithography simulator。
- candidate 级 behavior map 暂不重切，只使用上游传入的 risk 并做距离衰减。
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

import numpy as np

from layout_utils import LayoutIndex, _query_candidate_ids, bitmap_fingerprint, rasterize_centered_window


SOURCE_PRIORITY = (
    "fragment_facing_pair_anchor",
    "critical_spacing_anchor",
    "fragment_line_end_anchor",
    "line_end_anchor",
    "fragment_corner_anchor",
    "corner_or_jog_anchor",
    "density_transition_anchor",
    "marker_center_baseline",
    "local_grid_probe",
)
SEMANTIC_SOURCES = {
    "fragment_facing_pair_anchor",
    "critical_spacing_anchor",
    "fragment_line_end_anchor",
    "line_end_anchor",
    "fragment_corner_anchor",
    "corner_or_jog_anchor",
    "density_transition_anchor",
}
FRAGMENT_SOURCES = {
    "fragment_facing_pair_anchor",
    "fragment_line_end_anchor",
    "fragment_corner_anchor",
}
MAX_GEOMETRY_ELEMENTS = 96
MAX_RAW_CANDIDATES = 512


@dataclass(frozen=True)
class EdgeFragment:
    """版图局部边片段；用于 Ding 2011 风格的 MP 几何上下文提案。"""

    center: Tuple[float, float]
    orientation: str
    length: float
    endpoints: Tuple[Tuple[float, float], Tuple[float, float]]
    parent_bbox: Tuple[float, float, float, float]
    side: str
    is_line_end_proxy: bool
    element_index: int


@dataclass
class MPCandidate:
    """保存一个 MP 候选点、局部窗口、评分组件和筛选状态。"""

    x: float
    y: float
    distance_um: float
    candidate_type: str
    sources: List[str]
    window: Dict[str, Any]
    score: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    accepted: bool = True
    reject_reason: str = ""
    verified: bool = False
    verification_reason: str = ""
    verification_components: Dict[str, float] = field(default_factory=dict)
    proposal_metrics: Dict[str, float] = field(default_factory=dict)

    def to_summary(self) -> Dict[str, Any]:
        """返回可写入 JSON 的轻量候选摘要，不包含 bitmap 大数组。"""
        return {
            "x_um": float(self.x),
            "y_um": float(self.y),
            "distance_um": float(self.distance_um),
            "candidate_type": str(self.candidate_type),
            "sources": list(self.sources),
            "score": float(self.score),
            "components": dict(self.components),
            "mp_verified": bool(self.verified),
            "mp_reject_reason": str(self.verification_reason),
            "verification_components": dict(self.verification_components),
            "proposal_metrics": dict(self.proposal_metrics),
            "accepted": bool(self.accepted),
            "reject_reason": str(self.reject_reason),
            "clip_bbox": [float(value) for value in self.window.get("clip_bbox", [])],
        }


@dataclass
class MPDiscoveryResult:
    """保存单个 marker 邻域的 MP discovery 结果和审查信息。"""

    selected_candidate: MPCandidate
    top_candidates: List[MPCandidate]
    raw_candidate_count: int
    rasterized_candidate_count: int
    empty_rejected_count: int
    nms_rejected_count: int
    verification_rejected_count: int
    mp_discovery_reason: str
    behavior_risk_enabled: bool
    rule_coverage_audit: Dict[str, Any] = field(default_factory=dict)

    def to_summary(self) -> Dict[str, Any]:
        """返回 site_summary.json 使用的 MP discovery 审查摘要。"""
        return {
            "selected_candidate": self.selected_candidate.to_summary(),
            "top_candidates": [candidate.to_summary() for candidate in self.top_candidates],
            "raw_candidate_count": int(self.raw_candidate_count),
            "rasterized_candidate_count": int(self.rasterized_candidate_count),
            "empty_rejected_count": int(self.empty_rejected_count),
            "nms_rejected_count": int(self.nms_rejected_count),
            "verification_rejected_count": int(self.verification_rejected_count),
            "mp_discovery_reason": str(self.mp_discovery_reason),
            "behavior_risk_enabled": bool(self.behavior_risk_enabled),
            "rule_coverage_audit": dict(self.rule_coverage_audit),
        }


def _clip01(value: float) -> float:
    """把数值限制在 0 到 1 之间，非有限值按 0 处理。"""
    if not math.isfinite(float(value)):
        return 0.0
    return float(min(1.0, max(0.0, float(value))))


def _bitmap_density(bitmap: np.ndarray) -> float:
    """计算 bitmap 中 pattern pixel 的占比。"""
    arr = np.asarray(bitmap, dtype=bool)
    if arr.size == 0:
        return 0.0
    return float(np.count_nonzero(arr)) / float(arr.size)


def _edge_density_score(bitmap: np.ndarray) -> float:
    """用水平和垂直跳变估计局部边缘丰富度。"""
    arr = np.asarray(bitmap, dtype=bool)
    if arr.size == 0:
        return 0.0
    horizontal = np.count_nonzero(arr[:, 1:] != arr[:, :-1]) if arr.shape[1] > 1 else 0
    vertical = np.count_nonzero(arr[1:, :] != arr[:-1, :]) if arr.shape[0] > 1 else 0
    raw = float(horizontal + vertical) / float(max(1, arr.size))
    return _clip01(raw * 10.0)


def _corner_density_score(bitmap: np.ndarray) -> float:
    """用 2x2 patch 的非单调变化估计 corner/jog 丰富度。"""
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


def _layout_complexity_score(bitmap: np.ndarray) -> float:
    """组合密度、边缘和角点，估计候选窗口的版图复杂度。"""
    density = _bitmap_density(bitmap)
    density_balance = _clip01(1.0 - abs(density - 0.35) / 0.35)
    return _clip01(0.45 * _edge_density_score(bitmap) + 0.35 * _corner_density_score(bitmap) + 0.20 * density_balance)


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
    """计算带小范围平移容差的二值 IoU。"""
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
            best = max(best, float(np.count_nonzero(aa & bb)) / float(union))
    return _clip01(best)


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """计算两个物理坐标点之间的欧氏距离。"""
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _inside_radius(x: float, y: float, center_xy: Tuple[float, float], radius_um: float) -> bool:
    """判断候选点是否落在 marker 邻域搜索半径内。"""
    return _distance((x, y), center_xy) <= float(radius_um) + 1e-9


def _candidate_type(sources: Sequence[str]) -> str:
    """按固定优先级把多个 proposal source 合并成人类可读候选类型。"""
    alias = {
        "bridge_pinch": "critical_spacing_anchor",
        "narrow_space": "critical_spacing_anchor",
        "edge_pair": "critical_spacing_anchor",
        "fragment_facing_pair": "fragment_facing_pair_anchor",
        "fragment_line_end": "fragment_line_end_anchor",
        "fragment_corner": "fragment_corner_anchor",
        "line_end": "line_end_anchor",
        "corner_jog": "corner_or_jog_anchor",
        "density_transition": "density_transition_anchor",
        "local_grid": "local_grid_probe",
        "marker_center": "marker_center_baseline",
    }
    source_set = {alias.get(str(source), str(source)) for source in sources}
    for source in SOURCE_PRIORITY:
        if source in source_set:
            return source
    return "local_grid_probe"


def _add_raw_candidate(
    raw: Dict[Tuple[int, int], Dict[str, Any]],
    *,
    x: float,
    y: float,
    source: str,
    center_xy: Tuple[float, float],
    radius_um: float,
    step_um: float,
    metrics: Mapping[str, float] | None = None,
) -> None:
    """添加一个 raw candidate，并把落在同一量化位置的 proposal source 合并。"""
    if not _inside_radius(x, y, center_xy, radius_um):
        return
    quant = max(float(step_um) * 0.25, 1e-4)
    key = (int(round(float(x) / quant)), int(round(float(y) / quant)))
    item = raw.setdefault(key, {"x": float(x), "y": float(y), "sources": set(), "metrics": {}, "source_points": []})
    item["sources"].add(str(source))
    item["source_points"].append({"source": str(source), "x": float(x), "y": float(y)})
    for metric_name, metric_value in (metrics or {}).items():
        try:
            value = float(metric_value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        name = str(metric_name)
        if name.endswith("_distance_um") and name in item["metrics"]:
            item["metrics"][name] = min(float(item["metrics"][name]), value)
        elif name.endswith("_distance_um"):
            item["metrics"][name] = value
        else:
            item["metrics"][name] = max(float(item["metrics"].get(name, 0.0)), value)


def _iter_grid_centers(center_xy: Tuple[float, float], radius_um: float, step_um: float) -> List[Tuple[float, float]]:
    """生成 marker 邻域内的局部 grid candidate。"""
    cx, cy = float(center_xy[0]), float(center_xy[1])
    step = max(float(step_um), 1e-9)
    extent = int(math.floor(float(radius_um) / step + 1e-9))
    centers: List[Tuple[float, float]] = []
    for iy in range(-extent, extent + 1):
        for ix in range(-extent, extent + 1):
            x = cx + float(ix) * step
            y = cy + float(iy) * step
            if _inside_radius(x, y, center_xy, radius_um):
                centers.append((float(x), float(y)))
    return centers


def _query_local_elements(layout_index: LayoutIndex, center_xy: Tuple[float, float], radius_um: float) -> List[Mapping[str, Any]]:
    """查询 marker 邻域内的 pattern bbox，并限制最大元素数以避免第一版过重。"""
    cx, cy = float(center_xy[0]), float(center_xy[1])
    bbox = (cx - float(radius_um), cy - float(radius_um), cx + float(radius_um), cy + float(radius_um))
    ids = _query_candidate_ids(layout_index, bbox)
    elements = [layout_index.indexed_elements[int(index)] for index in ids]
    elements.sort(key=lambda item: (_distance(_bbox_center(item["bbox"]), center_xy), item["bbox"]))
    return elements[:MAX_GEOMETRY_ELEMENTS]


def _bbox_center(bbox: Sequence[float]) -> Tuple[float, float]:
    """计算 bbox 中心。"""
    return ((float(bbox[0]) + float(bbox[2])) * 0.5, (float(bbox[1]) + float(bbox[3])) * 0.5)


def _polygon_points(item: Mapping[str, Any]) -> np.ndarray | None:
    """从 indexed element 中读取 polygon 顶点；不可用时返回 None。"""
    element = item.get("element")
    if element is None or not hasattr(element, "points"):
        return None
    points = getattr(element, "points", None)
    if points is None or len(points) < 3:
        return None
    return np.asarray(points, dtype=np.float64)


def _fragment_side(orientation: str, center: Tuple[float, float], bbox: Sequence[float]) -> str:
    """用 fragment 所在 bbox 边界给出粗略 side 标签。"""
    x, y = float(center[0]), float(center[1])
    x0, y0, x1, y1 = (float(value) for value in bbox)
    tolerance = 1e-6
    if orientation == "horizontal":
        if abs(y - y0) <= tolerance:
            return "bottom"
        if abs(y - y1) <= tolerance:
            return "top"
    if orientation == "vertical":
        if abs(x - x0) <= tolerance:
            return "left"
        if abs(x - x1) <= tolerance:
            return "right"
    return "inner"


def _is_line_end_fragment(orientation: str, length: float, bbox: Sequence[float]) -> bool:
    """判断 fragment 是否像长条图形的短边 line-end。"""
    x0, y0, x1, y1 = (float(value) for value in bbox)
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    if width <= 1e-9 or height <= 1e-9:
        return False
    if width >= 2.0 * height and orientation == "vertical":
        return float(length) <= 1.25 * height
    if height >= 2.0 * width and orientation == "horizontal":
        return float(length) <= 1.25 * width
    return False


def _extract_edge_fragments(elements: Sequence[Mapping[str, Any]]) -> List[EdgeFragment]:
    """把局部 polygon 拆成 Manhattan edge fragments；无顶点时退化为 bbox 四边。"""
    fragments: List[EdgeFragment] = []
    for element_index, item in enumerate(elements):
        bbox = tuple(float(value) for value in item["bbox"])
        points = _polygon_points(item)
        if points is None:
            x0, y0, x1, y1 = bbox
            points = np.asarray([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], dtype=np.float64)
        if len(points) < 2:
            continue
        closed_points = np.vstack([points, points[0]])
        for first, second in zip(closed_points[:-1], closed_points[1:]):
            x0, y0 = float(first[0]), float(first[1])
            x1, y1 = float(second[0]), float(second[1])
            dx = x1 - x0
            dy = y1 - y0
            if abs(dx) <= 1e-9 and abs(dy) <= 1e-9:
                continue
            if abs(dy) <= 1e-9:
                orientation = "horizontal"
                length = abs(dx)
            elif abs(dx) <= 1e-9:
                orientation = "vertical"
                length = abs(dy)
            else:
                continue
            center = ((x0 + x1) * 0.5, (y0 + y1) * 0.5)
            fragments.append(
                EdgeFragment(
                    center=center,
                    orientation=orientation,
                    length=float(length),
                    endpoints=((x0, y0), (x1, y1)),
                    parent_bbox=bbox,
                    side=_fragment_side(orientation, center, bbox),
                    is_line_end_proxy=_is_line_end_fragment(orientation, float(length), bbox),
                    element_index=int(element_index),
                )
            )
    return fragments


def _add_fragment_corner_anchors(
    raw: Dict[Tuple[int, int], Dict[str, Any]],
    *,
    elements: Sequence[Mapping[str, Any]],
    center_xy: Tuple[float, float],
    radius_um: float,
    step_um: float,
) -> None:
    """从 polygon 顶点生成 fragment corner anchors。"""
    for item in elements:
        points = _polygon_points(item)
        if points is None:
            x0, y0, x1, y1 = (float(value) for value in item["bbox"])
            points = np.asarray([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], dtype=np.float64)
        if len(points) < 3:
            continue
        for point in points:
            _add_raw_candidate(
                raw,
                x=float(point[0]),
                y=float(point[1]),
                source="fragment_corner_anchor",
                center_xy=center_xy,
                radius_um=radius_um,
                step_um=step_um,
                metrics={
                    "fragment_corner_score": 1.0,
                    "fragment_context_score": 0.65,
                    "fragment_anchor_count": 1.0,
                },
            )


def _add_fragment_line_end_anchors(
    raw: Dict[Tuple[int, int], Dict[str, Any]],
    *,
    fragments: Sequence[EdgeFragment],
    center_xy: Tuple[float, float],
    radius_um: float,
    step_um: float,
) -> None:
    """从长条图形短边中心生成 fragment line-end anchors。"""
    for fragment in fragments:
        if not fragment.is_line_end_proxy:
            continue
        _add_raw_candidate(
            raw,
            x=float(fragment.center[0]),
            y=float(fragment.center[1]),
            source="fragment_line_end_anchor",
            center_xy=center_xy,
            radius_um=radius_um,
            step_um=step_um,
            metrics={
                "fragment_line_end_score": 1.0,
                "fragment_context_score": 0.75,
                "fragment_anchor_count": 1.0,
            },
        )


def _add_fragment_facing_pair_anchors(
    raw: Dict[Tuple[int, int], Dict[str, Any]],
    *,
    fragments: Sequence[EdgeFragment],
    center_xy: Tuple[float, float],
    radius_um: float,
    step_um: float,
    min_feature_um: float | None = None,
) -> None:
    """从相对 edge fragments 生成 narrow-space/facing-pair anchors。"""
    if min_feature_um is not None and float(min_feature_um) > 0.0:
        max_gap = max(0.02, min(0.50, float(min_feature_um) * 1.5))
    else:
        max_gap = max(0.08, min(0.30, float(step_um) * 1.5))
    ordered_fragments = sorted(
        fragments,
        key=lambda fragment: (_distance(fragment.center, center_xy), -float(fragment.length), fragment.orientation, fragment.side),
    )[:MAX_GEOMETRY_ELEMENTS]
    for first, second in itertools.combinations(ordered_fragments, 2):
        if first.element_index == second.element_index:
            continue
        if first.orientation != second.orientation:
            continue
        if first.orientation == "vertical":
            ax = float(first.center[0])
            bx = float(second.center[0])
            if abs(ax - bx) > float(max_gap) + 1e-9:
                continue
            if ax < bx and not (first.side == "right" and second.side == "left"):
                continue
            if bx < ax and not (first.side == "left" and second.side == "right"):
                continue
            gap = abs(ax - bx)
            if gap <= 1e-9 or gap > max_gap:
                continue
            a_y0, a_y1 = sorted((first.endpoints[0][1], first.endpoints[1][1]))
            b_y0, b_y1 = sorted((second.endpoints[0][1], second.endpoints[1][1]))
            overlap = _overlap_length(a_y0, a_y1, b_y0, b_y1)
            if overlap < max(0.02, 0.35 * min(first.length, second.length)):
                continue
            x_mid = (ax + bx) * 0.5
            y_mid = (max(a_y0, b_y0) + min(a_y1, b_y1)) * 0.5
        else:
            ay = float(first.center[1])
            by = float(second.center[1])
            if abs(ay - by) > float(max_gap) + 1e-9:
                continue
            if ay < by and not (first.side == "top" and second.side == "bottom"):
                continue
            if by < ay and not (first.side == "bottom" and second.side == "top"):
                continue
            gap = abs(ay - by)
            if gap <= 1e-9 or gap > max_gap:
                continue
            a_x0, a_x1 = sorted((first.endpoints[0][0], first.endpoints[1][0]))
            b_x0, b_x1 = sorted((second.endpoints[0][0], second.endpoints[1][0]))
            overlap = _overlap_length(a_x0, a_x1, b_x0, b_x1)
            if overlap < max(0.02, 0.35 * min(first.length, second.length)):
                continue
            x_mid = (max(a_x0, b_x0) + min(a_x1, b_x1)) * 0.5
            y_mid = (ay + by) * 0.5
        facing_score = _clip01(1.0 - float(gap) / max(float(max_gap), 1e-9))
        context_score = _clip01(float(overlap) / max(float(overlap) + float(gap), 1e-9))
        _add_raw_candidate(
            raw,
            x=x_mid,
            y=y_mid,
            source="fragment_facing_pair_anchor",
            center_xy=center_xy,
            radius_um=radius_um,
            step_um=step_um,
            metrics={
                "fragment_context_score": context_score,
                "fragment_anchor_count": 2.0,
                "fragment_facing_pair_score": facing_score,
                "internal_facing_distance_um": float(gap),
                "external_facing_distance_um": float(gap),
            },
        )


def _add_fragment_anchors(
    raw: Dict[Tuple[int, int], Dict[str, Any]],
    *,
    elements: Sequence[Mapping[str, Any]],
    center_xy: Tuple[float, float],
    radius_um: float,
    step_um: float,
    min_feature_um: float | None = None,
) -> None:
    """汇总 fragment corner、line-end 和 facing-pair 三类 MP anchors。"""
    fragments = _extract_edge_fragments(elements)
    if not fragments:
        return
    _add_fragment_corner_anchors(raw, elements=elements, center_xy=center_xy, radius_um=radius_um, step_um=step_um)
    _add_fragment_line_end_anchors(raw, fragments=fragments, center_xy=center_xy, radius_um=radius_um, step_um=step_um)
    _add_fragment_facing_pair_anchors(
        raw,
        fragments=fragments,
        center_xy=center_xy,
        radius_um=radius_um,
        step_um=step_um,
        min_feature_um=min_feature_um,
    )


def _raw_source_rank(sources: Sequence[str]) -> int:
    """按 proposal source 的几何价值给 raw candidate 排序。"""
    source_set = set(sources)
    ranks = {source: index for index, source in enumerate(SOURCE_PRIORITY)}
    return min((ranks.get(source, len(SOURCE_PRIORITY)) for source in source_set), default=len(SOURCE_PRIORITY))


def _add_geometry_anchors(
    raw: Dict[Tuple[int, int], Dict[str, Any]],
    *,
    elements: Sequence[Mapping[str, Any]],
    center_xy: Tuple[float, float],
    radius_um: float,
    step_um: float,
) -> None:
    """从 bbox 几何中生成 corner 和 line-end anchors。"""
    for item in elements:
        x0, y0, x1, y1 = (float(value) for value in item["bbox"])
        width = max(0.0, x1 - x0)
        height = max(0.0, y1 - y0)
        if width <= 1e-9 or height <= 1e-9:
            continue
        cx, cy = _bbox_center((x0, y0, x1, y1))

        aspect = max(width, height) / max(min(width, height), 1e-9)
        if aspect >= 2.0 and height >= width:
            _add_raw_candidate(raw, x=cx, y=y0, source="line_end_anchor", center_xy=center_xy, radius_um=radius_um, step_um=step_um)
            _add_raw_candidate(raw, x=cx, y=y1, source="line_end_anchor", center_xy=center_xy, radius_um=radius_um, step_um=step_um)
        elif aspect >= 2.0:
            _add_raw_candidate(raw, x=x0, y=cy, source="line_end_anchor", center_xy=center_xy, radius_um=radius_um, step_um=step_um)
            _add_raw_candidate(raw, x=x1, y=cy, source="line_end_anchor", center_xy=center_xy, radius_um=radius_um, step_um=step_um)

        for x, y in ((x0, y0), (x0, y1), (x1, y0), (x1, y1)):
            _add_raw_candidate(raw, x=x, y=y, source="corner_or_jog_anchor", center_xy=center_xy, radius_um=radius_um, step_um=step_um)


def _window_density(bitmap: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> float:
    """计算 bitmap 局部窗口密度，空窗口按 0 处理。"""
    patch = bitmap[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    if patch.size == 0:
        return 0.0
    return float(np.count_nonzero(patch)) / float(patch.size)


def _add_density_transition_anchors(
    raw: Dict[Tuple[int, int], Dict[str, Any]],
    *,
    layout_index: LayoutIndex,
    center_xy: Tuple[float, float],
    radius_um: float,
    step_um: float,
    pixel_size_um: float,
) -> None:
    """用 marker 邻域 bitmap 的局部密度梯度生成真正的 density-transition anchors。"""
    window = rasterize_centered_window(layout_index, center_xy, 2.0 * float(radius_um), float(pixel_size_um))
    bitmap = np.asarray(window["clip_bitmap"], dtype=bool)
    if bitmap.size == 0 or not np.any(bitmap):
        return
    x_min, y_min, x_max, y_max = (float(value) for value in window["clip_bbox"])
    px = max(float(pixel_size_um), 1e-9)
    half_px = max(2, int(round(max(float(step_um), px) / px * 0.5)))
    candidates: List[Tuple[float, float, float, Dict[str, float]]] = []
    for x, y in _iter_grid_centers(center_xy, radius_um, step_um):
        ix = int(round((float(x) - x_min) / px))
        iy = int(round((float(y) - y_min) / px))
        if iy - half_px < 0 or iy + half_px >= bitmap.shape[0] or ix - half_px < 0 or ix + half_px >= bitmap.shape[1]:
            continue
        left = _window_density(bitmap, iy - half_px, iy + half_px + 1, ix - 2 * half_px, ix)
        right = _window_density(bitmap, iy - half_px, iy + half_px + 1, ix + 1, ix + 2 * half_px + 1)
        down = _window_density(bitmap, iy - 2 * half_px, iy, ix - half_px, ix + half_px + 1)
        up = _window_density(bitmap, iy + 1, iy + 2 * half_px + 1, ix - half_px, ix + half_px + 1)
        local = _window_density(bitmap, iy - half_px, iy + half_px + 1, ix - half_px, ix + half_px + 1)
        if local <= 0.02 or local >= 0.98:
            continue
        gradient = max(abs(left - right), abs(down - up))
        if gradient >= 0.35:
            candidates.append(
                (
                    float(gradient),
                    float(x),
                    float(y),
                    {
                        "density_transition_score": float(gradient),
                        "density_local": float(local),
                        "density_left": float(left),
                        "density_right": float(right),
                        "density_up": float(up),
                        "density_down": float(down),
                    },
                )
            )
    candidates.sort(key=lambda item: (-item[0], _distance((item[1], item[2]), center_xy), item[2], item[1]))
    for _, x, y, metrics in candidates[:24]:
        _add_raw_candidate(
            raw,
            x=x,
            y=y,
            source="density_transition_anchor",
            center_xy=center_xy,
            radius_um=radius_um,
            step_um=step_um,
            metrics=metrics,
        )


def _overlap_length(a0: float, a1: float, b0: float, b1: float) -> float:
    """计算两个一维区间的重叠长度。"""
    return max(0.0, min(float(a1), float(b1)) - max(float(a0), float(b0)))


def _add_pair_anchors(
    raw: Dict[Tuple[int, int], Dict[str, Any]],
    *,
    elements: Sequence[Mapping[str, Any]],
    center_xy: Tuple[float, float],
    radius_um: float,
    step_um: float,
    min_feature_um: float | None = None,
) -> None:
    """从相邻 bbox 对中生成 narrow-space、edge-pair 和 bridge/pinch proxy anchors。"""
    if min_feature_um is not None and float(min_feature_um) > 0.0:
        max_gap = max(0.02, min(0.50, float(min_feature_um) * 1.5))
    else:
        max_gap = max(0.08, min(0.30, float(step_um) * 1.5))
    for first, second in itertools.combinations(elements[:MAX_GEOMETRY_ELEMENTS], 2):
        ax0, ay0, ax1, ay1 = (float(value) for value in first["bbox"])
        bx0, by0, bx1, by1 = (float(value) for value in second["bbox"])
        a_width, a_height = ax1 - ax0, ay1 - ay0
        b_width, b_height = bx1 - bx0, by1 - by0
        if min(a_width, a_height, b_width, b_height) <= 1e-9:
            continue
        sep_x = max(0.0, max(ax0, bx0) - min(ax1, bx1))
        sep_y = max(0.0, max(ay0, by0) - min(ay1, by1))
        if sep_x > max_gap and sep_y > max_gap:
            continue

        y_overlap = _overlap_length(ay0, ay1, by0, by1)
        min_height = min(a_height, b_height)
        if y_overlap >= 0.35 * min_height:
            gap = bx0 - ax1 if ax1 <= bx0 else ax0 - bx1 if bx1 <= ax0 else -1.0
            if 0.0 <= gap <= max_gap:
                x_mid = (ax1 + bx0) * 0.5 if ax1 <= bx0 else (bx1 + ax0) * 0.5
                y_mid = (max(ay0, by0) + min(ay1, by1)) * 0.5
                source = "critical_spacing_anchor"
                _add_raw_candidate(raw, x=x_mid, y=y_mid, source=source, center_xy=center_xy, radius_um=radius_um, step_um=step_um)

        x_overlap = _overlap_length(ax0, ax1, bx0, bx1)
        min_width = min(a_width, b_width)
        if x_overlap >= 0.35 * min_width:
            gap = by0 - ay1 if ay1 <= by0 else ay0 - by1 if by1 <= ay0 else -1.0
            if 0.0 <= gap <= max_gap:
                x_mid = (max(ax0, bx0) + min(ax1, bx1)) * 0.5
                y_mid = (ay1 + by0) * 0.5 if ay1 <= by0 else (by1 + ay0) * 0.5
                source = "critical_spacing_anchor"
                _add_raw_candidate(raw, x=x_mid, y=y_mid, source=source, center_xy=center_xy, radius_um=radius_um, step_um=step_um)


def _build_raw_candidates(
    layout_index: LayoutIndex,
    *,
    marker_center: Tuple[float, float],
    search_radius_um: float,
    step_um: float,
    pixel_size_um: float,
    min_feature_um: float | None = None,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """汇总 marker center、grid 和 geometry anchors，生成 raw candidate map。"""
    raw: Dict[Tuple[int, int], Dict[str, Any]] = {}
    _add_raw_candidate(
        raw,
        x=float(marker_center[0]),
        y=float(marker_center[1]),
        source="marker_center_baseline",
        center_xy=marker_center,
        radius_um=search_radius_um,
        step_um=step_um,
    )
    for x, y in _iter_grid_centers(marker_center, search_radius_um, step_um):
        _add_raw_candidate(raw, x=x, y=y, source="local_grid_probe", center_xy=marker_center, radius_um=search_radius_um, step_um=step_um)
    elements = _query_local_elements(layout_index, marker_center, search_radius_um)
    _add_fragment_anchors(
        raw,
        elements=elements,
        center_xy=marker_center,
        radius_um=search_radius_um,
        step_um=step_um,
        min_feature_um=min_feature_um,
    )
    _add_geometry_anchors(raw, elements=elements, center_xy=marker_center, radius_um=search_radius_um, step_um=step_um)
    _add_pair_anchors(
        raw,
        elements=elements,
        center_xy=marker_center,
        radius_um=search_radius_um,
        step_um=step_um,
        min_feature_um=min_feature_um,
    )
    _add_density_transition_anchors(
        raw,
        layout_index=layout_index,
        center_xy=marker_center,
        radius_um=search_radius_um,
        step_um=step_um,
        pixel_size_um=float(pixel_size_um),
    )
    if len(raw) <= MAX_RAW_CANDIDATES:
        return raw
    ordered = sorted(
        raw.items(),
        key=lambda item: (_raw_source_rank(item[1]["sources"]), _distance((item[1]["x"], item[1]["y"]), marker_center), item[0]),
    )
    return dict(ordered[:MAX_RAW_CANDIDATES])


def _source_score(sources: Set[str]) -> float:
    """根据 proposal source 估计 critical geometry 的先验强度。"""
    if "fragment_facing_pair_anchor" in sources:
        return 1.0
    if "critical_spacing_anchor" in sources:
        return 0.92
    if "fragment_line_end_anchor" in sources:
        return 0.86
    if "line_end_anchor" in sources:
        return 0.80
    if "fragment_corner_anchor" in sources:
        return 0.76
    if "corner_or_jog_anchor" in sources:
        return 0.70
    if "density_transition_anchor" in sources:
        return 0.55
    if "marker_center_baseline" in sources:
        return 0.25
    return 0.15


def _fragment_context_proxy_score(proposal_metrics: Mapping[str, float]) -> float:
    """把 fragment corner、line-end 和 facing-pair 上下文压缩成 0-1 分数。"""
    corner = _clip01(float(proposal_metrics.get("fragment_corner_score", 0.0)))
    line_end = _clip01(float(proposal_metrics.get("fragment_line_end_score", 0.0)))
    facing_pair = _clip01(float(proposal_metrics.get("fragment_facing_pair_score", 0.0)))
    context = _clip01(float(proposal_metrics.get("fragment_context_score", 0.0)))
    return _clip01(0.35 * facing_pair + 0.25 * line_end + 0.20 * corner + 0.20 * context)


def _defect_core_proxy_score(bitmap: np.ndarray, sources: Set[str], proposal_metrics: Mapping[str, float] | None = None) -> float:
    """显式估计候选是否像 narrow-space、line-end、corner/jog 等 defect core。"""
    density = _bitmap_density(bitmap)
    density_balance = _clip01(1.0 - abs(density - 0.35) / 0.35)
    bitmap_score = _clip01(0.45 * _edge_density_score(bitmap) + 0.35 * _corner_density_score(bitmap) + 0.20 * density_balance)
    fragment_score = _fragment_context_proxy_score(proposal_metrics or {})
    source_score = max(_source_score(sources), fragment_score)
    if fragment_score <= 1e-12:
        return _clip01(0.55 * source_score + 0.45 * bitmap_score)
    return _clip01(0.45 * source_score + 0.35 * bitmap_score + 0.20 * fragment_score)
def _context_support_score(bitmap: np.ndarray, sources: Set[str]) -> float:
    """估计 MP 周边是否有足够 OPE/context 信息，但不把复杂度直接当作 defect core。"""
    source_bonus = 0.12 if set(sources) & SEMANTIC_SOURCES else 0.0
    return _clip01(_layout_complexity_score(bitmap) + source_bonus)


def _source_family(source: str) -> str:
    """把具体 anchor source 归并成稳定的几何家族，供局部 voting 使用。"""
    if source in {"fragment_facing_pair_anchor", "critical_spacing_anchor"}:
        return "spacing"
    if source in {"fragment_line_end_anchor", "line_end_anchor"}:
        return "line_end"
    if source in {"fragment_corner_anchor", "corner_or_jog_anchor"}:
        return "corner"
    if source == "density_transition_anchor":
        return "density_transition"
    if source == "marker_center_baseline":
        return "marker_center"
    return "local_grid"


def _annotate_spatial_voting(candidates: Sequence[MPCandidate], *, support_radius_um: float) -> None:
    """根据邻近 anchor 的空间聚集程度写入 proposal voting 元数据。"""
    radius = max(float(support_radius_um), 1e-9)
    for candidate in candidates:
        semantic_anchor_count = 0
        families: Set[str] = set()
        for other in candidates:
            if _distance((candidate.x, candidate.y), (other.x, other.y)) > radius:
                continue
            semantic_sources = set(other.sources) & SEMANTIC_SOURCES
            semantic_anchor_count += len(semantic_sources)
            families.update(_source_family(source) for source in semantic_sources)
        if semantic_anchor_count <= 0:
            spatial_vote = 0.15 if "marker_center_baseline" in set(candidate.sources) else 0.10
        else:
            spatial_vote = _clip01(0.30 + 0.12 * min(4, semantic_anchor_count) + 0.11 * min(3, len(families)))
        candidate.proposal_metrics["supporting_anchor_count"] = float(semantic_anchor_count)
        candidate.proposal_metrics["supporting_anchor_family_count"] = float(len(families))
        candidate.proposal_metrics["spatial_proposal_voting"] = float(spatial_vote)


def _voting_confidence(sources: Set[str], proposal_metrics: Mapping[str, float] | None = None) -> float:
    """根据多个 proposal source 的重合程度估计局部投票置信度。"""
    if proposal_metrics and "spatial_proposal_voting" in proposal_metrics:
        return _clip01(float(proposal_metrics.get("spatial_proposal_voting", 0.0)))
    semantic_count = len(set(sources) & SEMANTIC_SOURCES)
    if semantic_count <= 0:
        return 0.15 if "marker_center_baseline" in sources else 0.10
    if semantic_count == 1:
        return 0.55
    if semantic_count == 2:
        return 0.80
    return 1.0


def _rarity_scores(candidates: Sequence[MPCandidate]) -> List[float]:
    """用局部 nearest-neighbor 相似度反推 pattern rarity。"""
    if not candidates:
        return []
    if len(candidates) == 1:
        return [1.0]
    features = np.vstack([bitmap_fingerprint(candidate.window["clip_bitmap"]) for candidate in candidates]).astype(np.float32)
    centers = np.asarray([[candidate.x, candidate.y] for candidate in candidates], dtype=np.float32)
    sims = features @ features.T
    distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    sims[distances < 1e-9] = -1.0
    nearest = np.max(sims, axis=1)
    return [_clip01(1.0 - max(0.0, float(value))) for value in nearest.tolist()]


def _score_candidates(
    candidates: Sequence[MPCandidate],
    *,
    marker_bitmap: np.ndarray,
    marker_center: Tuple[float, float],
    search_radius_um: float,
    behavior_risk: float,
    behavior_risk_enabled: bool,
) -> None:
    """按固定公式给所有 MP candidates 计算 `mp_hotspot_score`。"""
    rarities = _rarity_scores(candidates)
    if behavior_risk_enabled:
        weights = {
            "core_defect_proxy_score": 0.40,
            "context_support_score": 0.15,
            "marker_behavior_support": 0.20,
            "marker_similarity": 0.05,
            "local_rarity": 0.10,
            "proposal_voting": 0.10,
        }
    else:
        weights = {
            "core_defect_proxy_score": 0.40 / 0.80,
            "context_support_score": 0.15 / 0.80,
            "marker_behavior_support": 0.0,
            "marker_similarity": 0.05 / 0.80,
            "local_rarity": 0.10 / 0.80,
            "proposal_voting": 0.10 / 0.80,
        }
    for index, candidate in enumerate(candidates):
        sources = set(candidate.sources)
        distance_decay = _clip01(math.exp(-2.0 * candidate.distance_um / max(float(search_radius_um), 1e-9)))
        inherited_risk = _clip01(float(behavior_risk) * distance_decay) if behavior_risk_enabled else 0.0
        core_defect_proxy_score = _defect_core_proxy_score(candidate.window["clip_bitmap"], sources, candidate.proposal_metrics)
        context_support_score = _context_support_score(candidate.window["clip_bitmap"], sources)
        marker_similarity = _shifted_iou(marker_bitmap, candidate.window["clip_bitmap"], max_shift_px=2)
        local_rarity = float(rarities[index]) if index < len(rarities) else 0.0
        proposal_voting = _voting_confidence(sources, candidate.proposal_metrics)
        components = {
            "core_defect_proxy_score": core_defect_proxy_score,
            "context_support_score": context_support_score,
            "marker_behavior_support": inherited_risk,
            "marker_similarity": marker_similarity,
            "local_rarity": local_rarity,
            "proposal_voting": proposal_voting,
            "fragment_context_score": _clip01(float(candidate.proposal_metrics.get("fragment_context_score", 0.0))),
            "fragment_corner_score": _clip01(float(candidate.proposal_metrics.get("fragment_corner_score", 0.0))),
            "fragment_line_end_score": _clip01(float(candidate.proposal_metrics.get("fragment_line_end_score", 0.0))),
            "fragment_facing_pair_score": _clip01(float(candidate.proposal_metrics.get("fragment_facing_pair_score", 0.0))),
            "fragment_anchor_count": float(candidate.proposal_metrics.get("fragment_anchor_count", 0.0)),
            "supporting_anchor_count": float(candidate.proposal_metrics.get("supporting_anchor_count", 0.0)),
            "supporting_anchor_family_count": float(candidate.proposal_metrics.get("supporting_anchor_family_count", 0.0)),
            "internal_facing_distance_um": float(candidate.proposal_metrics.get("internal_facing_distance_um", 0.0)),
            "external_facing_distance_um": float(candidate.proposal_metrics.get("external_facing_distance_um", 0.0)),
            "density_transition_score": float(candidate.proposal_metrics.get("density_transition_score", 0.0)),
            "density_local": float(candidate.proposal_metrics.get("density_local", 0.0)),
            "layout_complexity": _layout_complexity_score(candidate.window["clip_bitmap"]),
            "geometry_core_score": core_defect_proxy_score,
            "critical_geometry_score": core_defect_proxy_score,
            "inherited_behavior_risk": inherited_risk,
            "known_marker_similarity": marker_similarity,
            "pattern_rarity": local_rarity,
            "voting_confidence": proposal_voting,
            "behavior_weight_redistributed": 0.0 if behavior_risk_enabled else 1.0,
        }
        candidate.components = components
        candidate.score = _clip01(sum(float(weights[key]) * float(components[key]) for key in weights))


def _apply_nms(candidates: Sequence[MPCandidate], *, nms_radius_um: float) -> int:
    """对相邻且高度相似的候选做轻量 NMS，返回被抑制的候选数量。"""
    kept: List[MPCandidate] = []
    rejected = 0
    ordered = sorted(candidates, key=lambda item: (-item.score, item.distance_um, item.y, item.x))
    for candidate in ordered:
        suppressed = False
        for kept_candidate in kept:
            if _distance((candidate.x, candidate.y), (kept_candidate.x, kept_candidate.y)) > float(nms_radius_um):
                continue
            similarity = _shifted_iou(candidate.window["clip_bitmap"], kept_candidate.window["clip_bitmap"], max_shift_px=2)
            fingerprint_similarity = float(
                np.dot(bitmap_fingerprint(candidate.window["clip_bitmap"]), bitmap_fingerprint(kept_candidate.window["clip_bitmap"]))
            )
            if max(similarity, fingerprint_similarity) >= 0.88:
                candidate.accepted = False
                candidate.reject_reason = "nms_suppressed"
                rejected += 1
                suppressed = True
                break
        if not suppressed:
            candidate.accepted = True
            candidate.reject_reason = ""
            kept.append(candidate)
    return rejected


def _verify_mp_candidate(candidate: MPCandidate, *, search_radius_um: float) -> None:
    """对单个 MP candidate 做轻量 sanity check，避免空白或均匀区域进入 AF/AP 构造。"""
    bitmap = np.asarray(candidate.window["clip_bitmap"], dtype=bool)
    density = _bitmap_density(bitmap)
    edge_score = _edge_density_score(bitmap)
    corner_score = _corner_density_score(bitmap)
    geometry_score = float(candidate.components.get("geometry_core_score", candidate.components.get("critical_geometry_score", 0.0)))
    has_anchor = _has_geometry_anchor(candidate)
    candidate.verification_components = {
        "density": float(density),
        "edge_score": float(edge_score),
        "corner_score": float(corner_score),
        "geometry_core_score": float(geometry_score),
        "has_geometry_anchor": 1.0 if has_anchor else 0.0,
    }
    candidate.verified = True
    candidate.verification_reason = ""
    if bitmap.size == 0 or np.count_nonzero(bitmap) == 0:
        candidate.verified = False
        candidate.verification_reason = "empty_bitmap"
    elif density < 0.03:
        candidate.verified = False
        candidate.verification_reason = "sparse_bitmap"
    elif density >= 0.92:
        candidate.verified = False
        candidate.verification_reason = "uniform_bitmap"
    elif candidate.distance_um > float(search_radius_um) + 1e-9:
        candidate.verified = False
        candidate.verification_reason = "outside_marker_neighborhood"
    elif not has_anchor and geometry_score < 0.22:
        candidate.verified = False
        candidate.verification_reason = "weak_geometry_signal"
    elif has_anchor and max(edge_score, corner_score, geometry_score) < 0.18:
        candidate.verified = False
        candidate.verification_reason = "weak_geometry_signal"
    if not candidate.verified:
        candidate.accepted = False
        candidate.reject_reason = candidate.verification_reason


def _verify_candidates(candidates: Sequence[MPCandidate], *, search_radius_um: float) -> int:
    """批量执行 MP verification，并返回被验证拒绝的候选数量。"""
    rejected = 0
    for candidate in candidates:
        if not candidate.accepted:
            continue
        _verify_mp_candidate(candidate, search_radius_um=float(search_radius_um))
        if not candidate.verified:
            rejected += 1
    return rejected


def _has_geometry_anchor(candidate: MPCandidate) -> bool:
    """判断候选是否来自真实几何 anchor，而不只是 marker center 或 local grid。"""
    return bool(set(candidate.sources) & SEMANTIC_SOURCES)


def _pick_selected_candidate(candidates: Sequence[MPCandidate], marker_center: Tuple[float, float]) -> Tuple[MPCandidate, str]:
    """选择单个 best MP；没有有效几何 anchor 时固定回退到 marker center。"""
    accepted = [candidate for candidate in candidates if candidate.accepted]
    geometry_candidates = [candidate for candidate in accepted if _has_geometry_anchor(candidate)]
    if geometry_candidates:
        selected = sorted(geometry_candidates, key=lambda item: (-item.score, item.distance_um, item.y, item.x))[0]
        return selected, "geometry_anchor"
    baseline_candidates = [candidate for candidate in candidates if "marker_center_baseline" in set(candidate.sources)]
    selected = sorted(baseline_candidates, key=lambda item: (_distance((item.x, item.y), marker_center), -item.score))[0]
    return selected, "fallback_marker_center"


def _rule_coverage_audit(candidates: Sequence[MPCandidate], selected: MPCandidate, reason: str) -> Dict[str, Any]:
    """生成 Gai-style 规则覆盖审查摘要；只用于 review，不参与打分。"""
    candidate_type_counts: Dict[str, int] = {}
    reject_reason_counts: Dict[str, int] = {}
    semantic_verified = 0
    fragment_verified = 0
    for candidate in candidates:
        candidate_type_counts[candidate.candidate_type] = candidate_type_counts.get(candidate.candidate_type, 0) + 1
        if candidate.verified and set(candidate.sources) & SEMANTIC_SOURCES:
            semantic_verified += 1
        if candidate.verified and set(candidate.sources) & FRAGMENT_SOURCES:
            fragment_verified += 1
        reason_key = candidate.reject_reason or candidate.verification_reason
        if reason_key:
            reject_reason_counts[reason_key] = reject_reason_counts.get(reason_key, 0) + 1
    return {
        "semantic_marker_covered": bool(semantic_verified > 0),
        "fragment_marker_covered": bool(fragment_verified > 0),
        "fallback_marker_center": bool(str(reason) == "fallback_marker_center"),
        "selected_candidate_type": str(selected.candidate_type),
        "verified_semantic_candidate_count": int(semantic_verified),
        "verified_fragment_candidate_count": int(fragment_verified),
        "candidate_type_counts": dict(sorted(candidate_type_counts.items())),
        "reject_reason_counts": dict(sorted(reject_reason_counts.items())),
    }


def discover_mp_candidates(
    *,
    layout_index: LayoutIndex,
    marker_center: Tuple[float, float],
    marker_window: Mapping[str, Any],
    window_size_um: float,
    pixel_size_um: float,
    search_radius_um: float,
    step_um: float,
    behavior_risk: float,
    behavior_risk_enabled: bool,
    min_feature_um: float | None = None,
    top_k: int = 5,
) -> MPDiscoveryResult:
    """在单个 hotspot marker 邻域内发现 top-K 个可进入全局池的 MP candidates。"""
    center = (float(marker_center[0]), float(marker_center[1]))
    radius = max(float(search_radius_um), 1e-9)
    step = max(float(step_um), 1e-9)
    keep_count = max(1, int(top_k))
    raw = _build_raw_candidates(
        layout_index,
        marker_center=center,
        search_radius_um=radius,
        step_um=step,
        pixel_size_um=float(pixel_size_um),
        min_feature_um=min_feature_um,
    )

    candidates: List[MPCandidate] = []
    empty_rejected_count = 0
    for item in raw.values():
        sources = sorted(str(source) for source in item["sources"])
        x = float(item["x"])
        y = float(item["y"])
        window = rasterize_centered_window(layout_index, (x, y), float(window_size_um), float(pixel_size_um))
        is_baseline = "marker_center_baseline" in set(sources)
        if not np.any(window["clip_bitmap"]) and not is_baseline:
            empty_rejected_count += 1
            continue
        candidates.append(
            MPCandidate(
                x=x,
                y=y,
                distance_um=_distance((x, y), center),
                candidate_type=_candidate_type(sources),
                sources=sources,
                window=window,
                proposal_metrics=dict(item.get("metrics", {}) or {}),
            )
        )

    if not candidates:
        window = rasterize_centered_window(layout_index, center, float(window_size_um), float(pixel_size_um))
        candidates.append(
            MPCandidate(
                x=center[0],
                y=center[1],
                distance_um=0.0,
                candidate_type="marker_center_baseline",
                sources=["marker_center_baseline"],
                window=window,
            )
        )

    _annotate_spatial_voting(candidates, support_radius_um=max(step, 0.25 * float(window_size_um)))
    _score_candidates(
        candidates,
        marker_bitmap=np.asarray(marker_window["clip_bitmap"], dtype=bool),
        marker_center=center,
        search_radius_um=radius,
        behavior_risk=float(behavior_risk),
        behavior_risk_enabled=bool(behavior_risk_enabled),
    )
    nms_radius = max(float(step), 0.25 * float(window_size_um))
    nms_rejected_count = _apply_nms(candidates, nms_radius_um=nms_radius)
    verification_rejected_count = _verify_candidates(candidates, search_radius_um=radius)
    selected, reason = _pick_selected_candidate(candidates, center)
    had_verification = bool(selected.verification_reason or selected.verification_components)
    if not selected.verified:
        _verify_mp_candidate(selected, search_radius_um=radius)
        if not selected.verified and not had_verification:
            verification_rejected_count += 1
    if selected.verified:
        selected.accepted = True
        selected.reject_reason = ""
    accepted_candidates = [candidate for candidate in candidates if candidate.accepted]
    if selected not in accepted_candidates:
        accepted_candidates.append(selected)
    ordered_candidates = sorted(accepted_candidates, key=lambda item: (-item.score, item.distance_um, item.y, item.x))
    if selected not in ordered_candidates[:keep_count]:
        ordered_candidates = [selected] + [candidate for candidate in ordered_candidates if candidate is not selected]
    top_candidates = ordered_candidates[:keep_count]
    return MPDiscoveryResult(
        selected_candidate=selected,
        top_candidates=top_candidates,
        raw_candidate_count=int(len(raw)),
        rasterized_candidate_count=int(len(candidates)),
        empty_rejected_count=int(empty_rejected_count),
        nms_rejected_count=int(nms_rejected_count),
        verification_rejected_count=int(verification_rejected_count),
        mp_discovery_reason=reason,
        behavior_risk_enabled=bool(behavior_risk_enabled),
        rule_coverage_audit=_rule_coverage_audit(candidates, selected, reason),
    )


