#!/usr/bin/env python3
"""No-training 手工特征向量提取器。

本脚本服务于 `layout_clustering_optimized_notrain.py`，目标是在 AutoEncoder
无法训练或暂时不可用时，仍然为 optimized behavior clustering 提供稳定、可解释、
无需训练的 feature vector。输出契约与 AutoEncoder encode 完全一致：一个包含
`sample_ids` 和 `features` 的 `features.npz`，以及一份描述特征块、维度和权重的
metadata JSON。

整体流程:
1. 读取 behavior manifest，要求 `aerial_npz` 必填，`pv/epe/nils/resist` 若出现则
   必须对所有样本全局可用。
2. 读取 OAS/OASIS，并按 marker layer 构建 marker-centered clip bitmap；若启用
   layer operations，则先执行 boolean 层操作。
3. 对 aerial image 提取 DCT 低频、FFT radial/angular spectrum、HOG 和 gradient
   统计，作为 behavior-first 主特征块。
4. 对可选 PV/EPE/NILS/resist 图像提取统计特征，用于补充工艺风险和行为差异。
5. 从 layout clip bitmap 中提取密度、连通分量、Radon、边缘方向、critical
   width/space proxy 等几何特征。
6. 基于 clip 内 polygon bbox proxy 构建轻量图，并用 Weisfeiler-Lehman hashing
   生成拓扑签名，帮助区分频谱相近但拓扑不同的 layout。
7. 每个 block 单独归一化后按固定权重拼接，再做整体 L2 normalize，写出
   `features.npz`。

注意: 这些 FV 只负责 ANN retrieval 和 representative coverage ordering；最终
cluster 成员是否成立仍由主脚本中的 behavior SSIM verification 决定。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
from scipy.fft import dctn
from skimage.feature import hog
from skimage.transform import radon, resize

from layer_operations import LayerOperationProcessor
from layout_utils import (
    DEFAULT_PIXEL_SIZE_NM,
    LayoutIndex,
    MarkerRasterBuilder,
    MarkerRecord,
    _query_candidate_ids,
)


IMAGE_KEY = "image"
TARGET_IMAGE_SHAPE = (64, 64)
DCT_SIZE = 16
FFT_RADIAL_BINS = 32
FFT_ANGULAR_BINS = 32
WL_DIM = 128
WL_ITERATIONS = 2
WL_MAX_NODES = 128
OPTIONAL_BEHAVIOR_CHANNELS = ("pv", "epe", "nils", "resist")


@dataclass
class BehaviorRow:
    """保存一条 manifest 样本的标识和各行为图像路径。"""

    sample_id: str
    marker_id: str
    paths: Dict[str, str]


@dataclass
class FeatureBlock:
    """表示拼接前的一个特征块，包括原始向量、权重和元数据。"""

    name: str
    values: np.ndarray
    weight: float
    metadata: Dict[str, Any]


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


def _read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """读取 JSONL 文件，并确保每个非空行都是 JSON object。"""
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Manifest row must be an object at {path}:{line_no}")
            records.append(item)
    if not records:
        raise ValueError("Behavior manifest is empty")
    return records


def _resolve_path(path_text: str, base_dir: Path) -> str:
    """把 manifest 中的相对路径解析为相对于 manifest 文件目录的绝对路径。"""
    path = Path(path_text)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _load_npz_image(path: str | Path) -> np.ndarray:
    """从单个 NPZ 中读取键名为 `image` 的二维 float32 图像。"""
    with np.load(str(path), allow_pickle=False) as data:
        if IMAGE_KEY not in data:
            raise ValueError(f"NPZ {path} must contain key '{IMAGE_KEY}'")
        image = np.asarray(data[IMAGE_KEY], dtype=np.float32)
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 2:
        raise ValueError(f"Image in {path} must be 2-D")
    return np.ascontiguousarray(image, dtype=np.float32)


def _load_manifest(path: str | Path) -> Tuple[Dict[str, BehaviorRow], List[str]]:
    """读取 behavior manifest，并校验可选 channel 必须全局全有或全无。"""
    manifest_path = Path(path)
    base_dir = manifest_path.resolve().parent
    records = _read_jsonl(manifest_path)
    optional_channels: List[str] = []
    for channel in OPTIONAL_BEHAVIOR_CHANNELS:
        key = f"{channel}_npz"
        present = [record for record in records if record.get(key)]
        if present and len(present) != len(records):
            raise ValueError(f"Optional channel {channel} must be present for all rows or none")
        if present:
            optional_channels.append(channel)

    rows: Dict[str, BehaviorRow] = {}
    for record in records:
        for key in ("sample_id", "marker_id", "aerial_npz"):
            if key not in record:
                raise ValueError(f"Behavior manifest row missing required field: {key}")
        sample_id = str(record["sample_id"])
        marker_id = str(record["marker_id"])
        if marker_id in rows:
            raise ValueError(f"Duplicate marker_id in behavior manifest: {marker_id}")
        paths = {"aerial": _resolve_path(str(record["aerial_npz"]), base_dir)}
        for channel in optional_channels:
            paths[channel] = _resolve_path(str(record[f"{channel}_npz"]), base_dir)
        rows[marker_id] = BehaviorRow(sample_id=sample_id, marker_id=marker_id, paths=paths)
    return rows, optional_channels


def _finite_vector(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """把输入压平成有限 float32 向量，并用 0 替换 NaN/Inf。"""
    array = np.asarray(values, dtype=np.float32).ravel()
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _l2_normalize(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """对向量做 L2 归一化；全零向量保持不变。"""
    vector = _finite_vector(values)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector
    return (vector / norm).astype(np.float32)


def _signed_log1p(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """对带符号高动态范围数值做 `sign(x) * log1p(abs(x))` 压缩。"""
    vector = _finite_vector(values)
    return (np.sign(vector) * np.log1p(np.abs(vector))).astype(np.float32)


def _robust_scale_image(image: np.ndarray) -> np.ndarray:
    """按 1/99 分位裁剪图像，并缩放到稳定的 0..1 范围。"""
    arr = np.asarray(image, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        return np.zeros((1, 1), dtype=np.float32)
    lo, hi = np.percentile(arr, [1.0, 99.0])
    if float(hi - lo) <= 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    arr = np.clip(arr, lo, hi)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _resize_image(image: np.ndarray, shape: Tuple[int, int] = TARGET_IMAGE_SHAPE) -> np.ndarray:
    """把输入图像缩放到手工特征使用的固定尺寸。"""
    scaled = _robust_scale_image(image)
    if tuple(scaled.shape) == tuple(shape):
        return np.ascontiguousarray(scaled, dtype=np.float32)
    return resize(
        scaled,
        shape,
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)


def _binned_means(values: np.ndarray, bin_ids: np.ndarray, n_bins: int) -> np.ndarray:
    """根据整数 bin id 计算每个 bin 内数值均值。"""
    out = np.zeros(int(n_bins), dtype=np.float32)
    for idx in range(int(n_bins)):
        mask = bin_ids == idx
        if np.any(mask):
            out[idx] = float(np.mean(values[mask]))
    return out


def _spectrum_profiles(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """从 log FFT spectrum 中提取 radial 和 angular 能量分布。"""
    spectrum = np.fft.fftshift(np.abs(np.fft.fft2(image)))
    spectrum = np.log1p(spectrum).astype(np.float32)
    h, w = spectrum.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    cy = (h - 1.0) / 2.0
    cx = (w - 1.0) / 2.0
    dy = yy - cy
    dx = xx - cx
    radius = np.sqrt(dx * dx + dy * dy)
    max_radius = max(float(np.max(radius)), 1e-6)
    radial_ids = np.minimum((radius / max_radius * FFT_RADIAL_BINS).astype(np.int32), FFT_RADIAL_BINS - 1)
    angles = (np.arctan2(dy, dx) + math.pi) / (2.0 * math.pi)
    angular_ids = np.minimum((angles * FFT_ANGULAR_BINS).astype(np.int32), FFT_ANGULAR_BINS - 1)
    return (
        _binned_means(spectrum, radial_ids, FFT_RADIAL_BINS),
        _binned_means(spectrum, angular_ids, FFT_ANGULAR_BINS),
    )


def _image_gradient_stats(image: np.ndarray) -> np.ndarray:
    """计算图像梯度幅值和方向的紧凑统计特征。"""
    gy, gx = np.gradient(image.astype(np.float32))
    mag = np.sqrt(gx * gx + gy * gy)
    if mag.size == 0:
        return np.zeros(8, dtype=np.float32)
    return np.asarray(
        [
            float(np.mean(mag)),
            float(np.std(mag)),
            float(np.max(mag)),
            float(np.percentile(mag, 50.0)),
            float(np.percentile(mag, 90.0)),
            float(np.mean(np.abs(gx))),
            float(np.mean(np.abs(gy))),
            float(np.std(np.arctan2(gy, gx))),
        ],
        dtype=np.float32,
    )


def _aerial_feature_block(image: np.ndarray) -> np.ndarray:
    """从 aerial image 中提取 DCT、FFT、HOG 和 gradient 主特征块。"""
    resized = _resize_image(image)
    coeff = dctn(resized, norm="ortho")[:DCT_SIZE, :DCT_SIZE]
    dct_features = _signed_log1p(coeff.ravel())
    radial, angular = _spectrum_profiles(resized)
    hog_features = hog(
        resized,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)
    gradient_stats = _image_gradient_stats(resized)
    return np.concatenate([dct_features, radial, angular, hog_features, gradient_stats]).astype(np.float32)


def _connected_stats(mask: np.ndarray) -> Tuple[float, float]:
    """统计二值 mask 的连通分量数量和最大分量占比。"""
    labels, count = ndimage.label(np.asarray(mask, dtype=bool))
    if int(count) <= 0:
        return 0.0, 0.0
    sizes = np.bincount(labels.ravel())[1:]
    largest = float(np.max(sizes)) if sizes.size else 0.0
    return float(count), largest / max(1.0, float(np.count_nonzero(mask)))


def _behavior_stats_for_image(image: np.ndarray) -> np.ndarray:
    """从可选仿真图像中提取幅值、分位数、梯度和高响应连通域统计。"""
    arr = np.asarray(image, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    abs_arr = np.abs(arr)
    if arr.size == 0:
        return np.zeros(18, dtype=np.float32)
    threshold = float(np.percentile(abs_arr, 90.0))
    mask = abs_arr >= threshold if threshold > 0.0 else abs_arr > 0.0
    comp_count, largest_ratio = _connected_stats(mask)
    gy, gx = np.gradient(arr)
    grad = np.sqrt(gx * gx + gy * gy)
    return np.asarray(
        [
            float(np.mean(arr)),
            float(np.std(arr)),
            float(np.min(arr)),
            float(np.max(arr)),
            float(np.percentile(arr, 10.0)),
            float(np.percentile(arr, 50.0)),
            float(np.percentile(arr, 90.0)),
            float(np.percentile(arr, 99.0)),
            float(np.mean(abs_arr)),
            float(np.max(abs_arr)),
            float(np.mean(arr > 0.0)),
            float(np.mean(mask)),
            float(np.mean(grad)),
            float(np.std(grad)),
            float(np.max(grad)),
            float(np.percentile(grad, 90.0)),
            comp_count,
            largest_ratio,
        ],
        dtype=np.float32,
    )


def _optional_behavior_block(row: BehaviorRow, channels: Sequence[str]) -> np.ndarray:
    """按 manifest 可用 channel 拼接 PV/EPE/NILS/resist 统计特征块。"""
    vectors = []
    for channel in channels:
        vectors.append(_behavior_stats_for_image(_load_npz_image(row.paths[channel])))
    if not vectors:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(vectors).astype(np.float32)


def _downsample_bitmap(bitmap: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """把 bool bitmap 下采样成指定尺寸的 float 图像。"""
    arr = np.asarray(bitmap, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(shape, dtype=np.float32)
    return resize(arr, shape, order=1, mode="reflect", anti_aliasing=True, preserve_range=True).astype(np.float32)


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    """从二值版图 mask 中提取简单边界像素 mask。"""
    bool_mask = np.asarray(mask, dtype=bool)
    if not np.any(bool_mask):
        return np.zeros_like(bool_mask, dtype=bool)
    eroded = ndimage.binary_erosion(bool_mask)
    return bool_mask & ~eroded


def _local_boundary_counts(boundary: np.ndarray) -> Tuple[float, float, float]:
    """基于边界像素邻域估计 line-end、corner 和 jog proxy 比例。"""
    b = np.asarray(boundary, dtype=bool)
    if not np.any(b):
        return 0.0, 0.0, 0.0
    padded = np.pad(b.astype(np.uint8), 1, mode="constant")
    endpoints = 0
    corners = 0
    jogs = 0
    ys, xs = np.nonzero(b)
    for y, x in zip(ys, xs):
        y0 = int(y) + 1
        x0 = int(x) + 1
        window = padded[y0 - 1 : y0 + 2, x0 - 1 : x0 + 2].copy()
        window[1, 1] = 0
        neighbor_count = int(np.sum(window))
        if neighbor_count <= 1:
            endpoints += 1
        elif neighbor_count == 2:
            horizontal = bool(window[1, 0] and window[1, 2])
            vertical = bool(window[0, 1] and window[2, 1])
            if not (horizontal or vertical):
                corners += 1
        elif neighbor_count >= 3:
            jogs += 1
    scale = max(1.0, float(np.count_nonzero(b)))
    return float(endpoints) / scale, float(corners) / scale, float(jogs) / scale


def _edge_orientation_histogram(mask: np.ndarray, bins: int = 8) -> np.ndarray:
    """计算由边界梯度幅值加权的边方向直方图。"""
    image = np.asarray(mask, dtype=np.float32)
    if image.size == 0 or not np.any(image):
        return np.zeros(bins, dtype=np.float32)
    gy, gx = np.gradient(image)
    mag = np.sqrt(gx * gx + gy * gy)
    angles = (np.arctan2(gy, gx) + math.pi) / (2.0 * math.pi)
    ids = np.minimum((angles * bins).astype(np.int32), bins - 1)
    hist = np.zeros(bins, dtype=np.float32)
    for idx in range(bins):
        hist[idx] = float(np.sum(mag[ids == idx]))
    total = float(np.sum(hist))
    if total > 1e-12:
        hist /= total
    return hist


def _layout_geometry_block(record: MarkerRecord, pixel_size_um: float) -> np.ndarray:
    """从 marker clip bitmap 中提取几何、Radon 和 critical feature proxy 特征。"""
    mask = np.asarray(record.clip_bitmap, dtype=bool)
    h, w = mask.shape
    if h == 0 or w == 0:
        return np.zeros(114, dtype=np.float32)

    fill = float(np.mean(mask))
    density_grid = _downsample_bitmap(mask, (8, 8)).ravel()
    boundary = _boundary_mask(mask)
    edge_density = float(np.mean(boundary))

    yy, xx = np.indices(mask.shape)
    center_mask = (xx >= 0.25 * w) & (xx < 0.75 * w) & (yy >= 0.25 * h) & (yy < 0.75 * h)
    center_fill = float(np.mean(mask[center_mask])) if np.any(center_mask) else 0.0
    ring_fill = float(np.mean(mask[~center_mask])) if np.any(~center_mask) else 0.0
    center_ring_ratio = center_fill / max(ring_fill, 1e-6)

    comp_count, largest_ratio = _connected_stats(mask)
    mask_float = mask.astype(np.float32)
    h_sym = 1.0 - float(np.mean(np.abs(mask_float - np.flipud(mask_float))))
    v_sym = 1.0 - float(np.mean(np.abs(mask_float - np.fliplr(mask_float))))
    symmetry = float(np.clip((h_sym + v_sym) / 2.0, 0.0, 1.0))
    small = _downsample_bitmap(mask, (32, 32))
    gx = np.diff(small, axis=1)
    gy = np.diff(small, axis=0)
    regularity = float(np.clip(1.0 - 4.0 * ((np.mean(np.abs(gx)) + np.mean(np.abs(gy))) / 2.0), 0.0, 1.0))

    theta = np.linspace(0.0, 180.0, 8, endpoint=False)
    radon_img = radon(small, theta=theta, circle=False)
    radon_stats = np.concatenate([np.mean(radon_img, axis=0), np.std(radon_img, axis=0)]).astype(np.float32)

    fg_dist = ndimage.distance_transform_edt(mask) * float(pixel_size_um)
    bg_dist = ndimage.distance_transform_edt(~mask) * float(pixel_size_um)
    fg_values = fg_dist[mask]
    bg_values = bg_dist[~mask]
    fg_quantiles = np.percentile(fg_values, [10, 25, 50, 75, 90]) if fg_values.size else np.zeros(5)
    bg_quantiles = np.percentile(bg_values, [10, 25, 50, 75, 90]) if bg_values.size else np.zeros(5)
    endpoints, corners, jogs = _local_boundary_counts(boundary)
    orientation_hist = _edge_orientation_histogram(mask, bins=8)

    topology_flags = np.asarray(
        [
            1.0 if fill >= 0.35 else 0.0,
            1.0 if fill <= 0.08 else 0.0,
            1.0 if edge_density >= 0.20 else 0.0,
            1.0 if comp_count >= 4.0 else 0.0,
        ],
        dtype=np.float32,
    )
    scalar = np.asarray(
        [
            fill,
            edge_density,
            center_fill,
            ring_fill,
            center_ring_ratio,
            comp_count,
            largest_ratio,
            symmetry,
            regularity,
            endpoints,
            corners,
            jogs,
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            scalar,
            density_grid.astype(np.float32),
            radon_stats,
            np.asarray(fg_quantiles, dtype=np.float32),
            np.asarray(bg_quantiles, dtype=np.float32),
            orientation_hist,
            topology_flags,
        ]
    ).astype(np.float32)


def _stable_hash(text: str) -> int:
    """为 WL label 生成跨进程稳定的整数 hash。"""
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _bin_value(value: float, thresholds: Sequence[float]) -> int:
    """把连续标量映射到由阈值定义的小整数 bin。"""
    for idx, threshold in enumerate(thresholds):
        if float(value) <= float(threshold):
            return int(idx)
    return int(len(thresholds))


def _clip_bbox(bbox: Sequence[float], clip_bbox: Sequence[float]) -> Optional[Tuple[float, float, float, float]]:
    """计算单个 polygon bbox 与 marker clip bbox 的交集。"""
    x0 = max(float(bbox[0]), float(clip_bbox[0]))
    y0 = max(float(bbox[1]), float(clip_bbox[1]))
    x1 = min(float(bbox[2]), float(clip_bbox[2]))
    y1 = min(float(bbox[3]), float(clip_bbox[3]))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _wl_nodes(record: MarkerRecord, layout_index: LayoutIndex) -> List[Dict[str, Any]]:
    """为单个 marker clip 构建确定性的 polygon bbox proxy 图节点。"""
    candidate_ids = _query_candidate_ids(layout_index, record.clip_bbox)
    nodes: List[Dict[str, Any]] = []
    clip = record.clip_bbox
    clip_w = max(float(clip[2] - clip[0]), 1e-9)
    clip_h = max(float(clip[3] - clip[1]), 1e-9)
    clip_area = clip_w * clip_h
    marker_cx, marker_cy = record.marker_center
    for idx in candidate_ids:
        item = layout_index.indexed_elements[int(idx)]
        clipped = _clip_bbox(item["bbox"], clip)
        if clipped is None:
            continue
        x0, y0, x1, y1 = clipped
        width = x1 - x0
        height = y1 - y0
        area = width * height
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        center_dist = math.hypot(cx - marker_cx, cy - marker_cy)
        gx = min(3, max(0, int((cx - clip[0]) / clip_w * 4.0)))
        gy = min(3, max(0, int((cy - clip[1]) / clip_h * 4.0)))
        boundary_touch = int(x0 <= clip[0] or y0 <= clip[1] or x1 >= clip[2] or y1 >= clip[3])
        area_bin = _bin_value(area / max(clip_area, 1e-12), [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2])
        aspect = width / max(height, 1e-9)
        aspect_bin = _bin_value(abs(math.log(max(aspect, 1e-9))), [0.15, 0.4, 0.8, 1.2, 2.0])
        width_bin = _bin_value(width / clip_w, [0.02, 0.05, 0.10, 0.20, 0.40])
        height_bin = _bin_value(height / clip_h, [0.02, 0.05, 0.10, 0.20, 0.40])
        label = (
            f"L{int(item['layer'])}/{int(item['datatype'])}|A{area_bin}|W{width_bin}|H{height_bin}|"
            f"R{aspect_bin}|G{gx}_{gy}|B{boundary_touch}"
        )
        nodes.append(
            {
                "bbox": clipped,
                "center": (cx, cy),
                "layer": int(item["layer"]),
                "datatype": int(item["datatype"]),
                "label": label,
                "center_dist": center_dist,
            }
        )
    nodes.sort(key=lambda node: (float(node["center_dist"]), node["label"], node["bbox"]))
    return nodes[:WL_MAX_NODES]


def _bbox_gap_and_relation(a: Sequence[float], b: Sequence[float]) -> Tuple[float, str]:
    """计算两个 bbox 的间距和粗空间关系标签。"""
    xgap = max(0.0, max(float(a[0]), float(b[0])) - min(float(a[2]), float(b[2])))
    ygap = max(0.0, max(float(a[1]), float(b[1])) - min(float(a[3]), float(b[3])))
    dist = math.hypot(xgap, ygap)
    if xgap <= 1e-9 and ygap <= 1e-9:
        relation = "overlap"
    elif dist <= 1e-9:
        relation = "touch"
    elif xgap <= 1e-9 or ygap <= 1e-9:
        relation = "project"
    else:
        relation = "diag"
    return dist, relation


def _edge_label(a: Mapping[str, Any], b: Mapping[str, Any], clip_diag: float) -> Tuple[float, str]:
    """根据两个 bbox proxy 节点生成 WL 图边标签。"""
    dist, relation = _bbox_gap_and_relation(a["bbox"], b["bbox"])
    ax, ay = a["center"]
    bx, by = b["center"]
    angle = (math.atan2(by - ay, bx - ax) + math.pi) / (2.0 * math.pi)
    direction_bin = int(min(7, max(0, math.floor(angle * 8.0))))
    dist_bin = _bin_value(dist / max(clip_diag, 1e-9), [0.005, 0.02, 0.05, 0.10, 0.20])
    layer_rel = "same" if (a["layer"], a["datatype"]) == (b["layer"], b["datatype"]) else "cross"
    return dist, f"{layer_rel}|D{direction_bin}|R{dist_bin}|{relation}"


def _wl_adjacency(nodes: Sequence[Mapping[str, Any]], record: MarkerRecord) -> List[List[Tuple[int, str]]]:
    """为 WL relabeling 构建稀疏且确定性的邻接表。"""
    n = len(nodes)
    adjacency: List[List[Tuple[int, str]]] = [[] for _ in range(n)]
    if n <= 1:
        return adjacency
    clip = record.clip_bbox
    clip_diag = math.hypot(float(clip[2] - clip[0]), float(clip[3] - clip[1]))
    cutoff = max(0.02, 0.12 * clip_diag)
    pair_rows: List[List[Tuple[float, int, str]]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist, label = _edge_label(nodes[i], nodes[j], clip_diag)
            pair_rows[i].append((dist, j, label))
            pair_rows[j].append((dist, i, label))
            if dist <= cutoff or "overlap" in label or "touch" in label:
                adjacency[i].append((j, label))
                adjacency[j].append((i, label))
    for i in range(n):
        if adjacency[i]:
            continue
        for _, j, label in sorted(pair_rows[i], key=lambda item: (item[0], item[1]))[:4]:
            adjacency[i].append((j, label))
    return adjacency


def _layout_wl_block(record: MarkerRecord, layout_index: LayoutIndex) -> np.ndarray:
    """计算 marker clip 的 hashed Weisfeiler-Lehman 拓扑签名。"""
    nodes = _wl_nodes(record, layout_index)
    if not nodes:
        return np.zeros(WL_DIM, dtype=np.float32)
    adjacency = _wl_adjacency(nodes, record)
    labels = [str(node["label"]) for node in nodes]
    counts = np.zeros(WL_DIM, dtype=np.float32)
    for iteration in range(WL_ITERATIONS + 1):
        for label in labels:
            counts[_stable_hash(f"{iteration}|{label}") % WL_DIM] += 1.0
        if iteration == WL_ITERATIONS:
            break
        next_labels = []
        for idx, label in enumerate(labels):
            neighbor_tokens = sorted(f"{edge}|{labels[nbr]}" for nbr, edge in adjacency[idx])
            token = f"{label}{{{';'.join(neighbor_tokens)}}}"
            next_labels.append(hashlib.sha1(token.encode("utf-8")).hexdigest()[:20])
        labels = next_labels
    return _l2_normalize(np.sqrt(counts))


def _block_weights(has_behavior_stats: bool) -> Dict[str, float]:
    """返回固定 block 权重；可选 behavior block 缺失时把权重并入 aerial block。"""
    if has_behavior_stats:
        return {
            "aerial_image": 0.55,
            "optional_behavior_stats": 0.20,
            "layout_geometry": 0.10,
            "layout_wl_graph": 0.15,
        }
    return {
        "aerial_image": 0.75,
        "layout_geometry": 0.10,
        "layout_wl_graph": 0.15,
    }


def _fuse_blocks(blocks: Sequence[FeatureBlock]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """对各特征块分别归一化、加权拼接，并对最终 FV 做整体 L2 归一化。"""
    weighted_vectors = []
    block_metadata = []
    for block in blocks:
        normalized = _l2_normalize(block.values)
        weighted = float(block.weight) * normalized
        weighted_vectors.append(weighted.astype(np.float32))
        block_metadata.append(
            {
                "name": block.name,
                "dim": int(normalized.size),
                "weight": float(block.weight),
                "raw_norm": float(np.linalg.norm(_finite_vector(block.values))),
                "metadata": dict(block.metadata),
            }
        )
    fused = np.concatenate(weighted_vectors).astype(np.float32) if weighted_vectors else np.zeros(0, dtype=np.float32)
    return _l2_normalize(fused), block_metadata


class HandcraftedFeatureBuilder(MarkerRasterBuilder):
    """复用 MarkerRasterBuilder 收集 marker records，并服务手工特征提取。"""

    def collect_records(self, input_path: str) -> List[Tuple[Path, MarkerRecord, LayoutIndex]]:
        """收集 marker records，并保留构建它们时使用的 LayoutIndex。"""
        records: List[Tuple[Path, MarkerRecord, LayoutIndex]] = []
        input_files = self._discover_input_files(input_path)
        if not input_files:
            raise ValueError("No .oas files found")
        for filepath in input_files:
            layout_index = self._prepare_layout(filepath)
            for marker_index, marker_poly in enumerate(layout_index.marker_polygons):
                record = self._build_marker_record(filepath, marker_index, marker_poly, layout_index)
                if record is not None:
                    records.append((filepath, record, layout_index))
        if not records:
            raise ValueError("No hotspot markers found on the configured marker layer")
        return records


def _feature_for_record(
    row: BehaviorRow,
    record: MarkerRecord,
    layout_index: LayoutIndex,
    *,
    optional_channels: Sequence[str],
    pixel_size_um: float,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """为单个 marker record 构建融合后的手工 FV 和 block metadata。"""
    weights = _block_weights(bool(optional_channels))
    blocks: List[FeatureBlock] = [
        FeatureBlock(
            "aerial_image",
            _aerial_feature_block(_load_npz_image(row.paths["aerial"])),
            weights["aerial_image"],
            {"target_shape": list(TARGET_IMAGE_SHAPE), "dct_size": DCT_SIZE},
        )
    ]
    if optional_channels:
        blocks.append(
            FeatureBlock(
                "optional_behavior_stats",
                _optional_behavior_block(row, optional_channels),
                weights["optional_behavior_stats"],
                {"channels": list(optional_channels)},
            )
        )
    blocks.append(
        FeatureBlock(
            "layout_geometry",
            _layout_geometry_block(record, pixel_size_um),
            weights["layout_geometry"],
            {"source": "clip_bitmap", "pixel_size_um": float(pixel_size_um)},
        )
    )
    blocks.append(
        FeatureBlock(
            "layout_wl_graph",
            _layout_wl_block(record, layout_index),
            weights["layout_wl_graph"],
            {"wl_iterations": WL_ITERATIONS, "wl_dim": WL_DIM, "max_nodes": WL_MAX_NODES},
        )
    )
    return _fuse_blocks(blocks)


def _make_layer_processor(register_ops: Sequence[Sequence[str]] | None) -> LayerOperationProcessor:
    """根据 CLI 注册规则构建 layer boolean 操作处理器。"""
    processor = LayerOperationProcessor()
    for source_layer, target_layer, operation, result_layer in register_ops or []:
        processor.register_operation_rule(source_layer, operation, target_layer, result_layer)
    return processor


def encode_handcrafted_features(
    *,
    input_path: str,
    marker_layer: str,
    behavior_manifest: str,
    features_out: str | Path,
    metadata_out: str | Path | None = None,
    clip_size_um: float = 1.35,
    pixel_size_nm: int = DEFAULT_PIXEL_SIZE_NM,
    apply_layer_operations: bool = False,
    layer_processor: LayerOperationProcessor | None = None,
) -> Dict[str, Any]:
    """把所有与 manifest 对齐的 marker 样本编码为手工 FV，并写出 NPZ/metadata。"""
    manifest_rows, optional_channels = _load_manifest(behavior_manifest)
    output = Path(features_out)
    builder = HandcraftedFeatureBuilder(
        config={
            "hotspot_layer": str(marker_layer),
            "clip_size_um": float(clip_size_um),
            "pixel_size_nm": int(pixel_size_nm),
            "apply_layer_operations": bool(apply_layer_operations),
        },
        temp_dir=output.parent,
        layer_processor=layer_processor if apply_layer_operations else None,
    )
    records = builder.collect_records(str(input_path))
    features: List[np.ndarray] = []
    sample_ids: List[str] = []
    first_block_metadata: List[Dict[str, Any]] = []
    matched_markers = set()
    for _, record, layout_index in records:
        row = manifest_rows.get(str(record.marker_id))
        if row is None:
            continue
        feature, block_metadata = _feature_for_record(
            row,
            record,
            layout_index,
            optional_channels=optional_channels,
            pixel_size_um=builder.pixel_size_um,
        )
        sample_ids.append(row.sample_id)
        features.append(feature.astype(np.float32))
        matched_markers.add(row.marker_id)
        if not first_block_metadata:
            first_block_metadata = block_metadata

    missing = sorted(set(manifest_rows) - matched_markers)
    if missing:
        raise ValueError(f"Behavior manifest markers not found in OAS marker layer: {missing[:5]}")
    if not features:
        raise ValueError("No handcrafted features were generated")
    matrix = np.vstack(features).astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, sample_ids=np.asarray(sample_ids, dtype=str), features=matrix)

    metadata = {
        "feature_source": "handcraft",
        "feature_count": int(matrix.shape[0]),
        "feature_dim": int(matrix.shape[1]),
        "sample_id_count": int(len(sample_ids)),
        "optional_behavior_channels": list(optional_channels),
        "block_metadata": first_block_metadata,
        "features_out": str(output),
        "marker_layer": str(marker_layer),
        "clip_size_um": float(clip_size_um),
        "pixel_size_nm": int(pixel_size_nm),
    }
    metadata_path = Path(metadata_out) if metadata_out else output.with_suffix(".metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False, default=_json_default)
    metadata["metadata_out"] = str(metadata_path)
    return metadata


def _build_parser() -> argparse.ArgumentParser:
    """构建手工特征提取 CLI，并在帮助信息中给出用法和注意点。"""
    epilog = """
示例:

1) 为 no-train 主脚本预先生成 handcrafted FV
python feature_extractor_handcraft.py encode ./input.oas --marker-layer 999/0 --behavior-manifest behavior.jsonl --features-out features_handcraft.npz

2) 同时导出 metadata JSON
python feature_extractor_handcraft.py encode ./input.oas --marker-layer 999/0 --behavior-manifest behavior.jsonl --features-out features_handcraft.npz --metadata-out features_handcraft.metadata.json

3) 启用 layer boolean 操作后再提取特征
python feature_extractor_handcraft.py encode ./input.oas --marker-layer 999/0 --behavior-manifest behavior.jsonl --features-out features_handcraft.npz --register-op 1/0 2/0 subtract 10/0

注意:
- manifest 必须提供 aerial_npz；pv/epe/nils/resist 若出现，必须所有样本都提供。
- 输出 features.npz 与 AutoEncoder encode 输出兼容，包含 sample_ids 和 features。
- 手工 FV 只用于 ANN/coverage ordering；最终 cluster 成员仍由主脚本 behavior verification 判断。
- WL graph signature 基于 clip 内 polygon bbox proxy，不替代精确几何比较。
"""
    parser = argparse.ArgumentParser(
        description="no-training 手工 FV 提取器",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    encode = subparsers.add_parser(
        "encode",
        help="把 OAS marker windows 编码为 handcrafted FV",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    encode.add_argument("input_path", help="输入 OAS 文件或目录")
    encode.add_argument("--marker-layer", required=True, help="marker 层，例如 999/0")
    encode.add_argument("--behavior-manifest", required=True, help="behavior JSONL manifest 路径")
    encode.add_argument("--features-out", required=True, help="输出 features.npz 路径")
    encode.add_argument("--metadata-out", default="", help="可选 metadata JSON 输出路径")
    encode.add_argument("--clip-size", type=float, default=1.35, help="marker-centered clip 边长，单位 um")
    encode.add_argument("--pixel-size-nm", type=int, default=DEFAULT_PIXEL_SIZE_NM, help="raster pixel size，单位 nm")
    encode.add_argument("--apply-layer-ops", action="store_true", help="提取前应用注册的 layer operations")
    encode.add_argument(
        "--register-op",
        action="append",
        nargs=4,
        metavar=("SOURCE_LAYER", "TARGET_LAYER", "OPERATION", "RESULT_LAYER"),
        help="注册层操作规则，例如 --register-op 1/0 2/0 subtract 10/0",
    )
    return parser


def main() -> int:
    """命令行入口：解析参数、执行 encode、打印输出路径和特征维度。"""
    parser = _build_parser()
    args = parser.parse_args()
    if args.command != "encode":
        parser.error("Unsupported command")
    register_ops = args.register_op or []
    apply_layer_operations = bool(args.apply_layer_ops or register_ops)
    try:
        layer_processor = _make_layer_processor(register_ops)
        metadata = encode_handcrafted_features(
            input_path=str(args.input_path),
            marker_layer=str(args.marker_layer),
            behavior_manifest=str(args.behavior_manifest),
            features_out=str(args.features_out),
            metadata_out=str(args.metadata_out) if args.metadata_out else None,
            clip_size_um=float(args.clip_size),
            pixel_size_nm=int(args.pixel_size_nm),
            apply_layer_operations=apply_layer_operations,
            layer_processor=layer_processor,
        )
        print(f"handcrafted features saved to: {args.features_out}")
        print(f"feature metadata saved to: {metadata.get('metadata_out')}")
        print(f"feature dim: {metadata.get('feature_dim')}, samples: {metadata.get('feature_count')}")
        return 0
    except Exception as exc:
        print(f"feature extraction failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
