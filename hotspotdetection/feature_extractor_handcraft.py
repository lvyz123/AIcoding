#!/usr/bin/env python3
"""无训练 recipe backend 使用的 handcrafted feature 提取。

本文件不是独立入口，而是 `hotspot_recipe_notrain_backend.py` 的底层特征模块。
当前版本只服务 CD-SEM/aerial_npz 单图像输入，不再保留 EPE/PV/NILS/resist 等旧分支。

算法流程：
1. 从 marker layout bitmap 和 CD-SEM/aerial 图像中读取固定窗口数组。
2. 分别计算密度、边缘、投影、分块直方图和简单梯度统计。
3. 拼接为 L2 归一化 feature vector，供无训练 backend 做相似性和代表选择。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from layout_utils import bitmap_fingerprint


LEGACY_BEHAVIOR_KEYS = {
    "epe_npz",
    "pv_npz",
    "nils_npz",
    "resist_npz",
    "epe_path",
    "pv_path",
    "nils_path",
    "resist_path",
}


def _as_float_image(array: Any) -> np.ndarray:
    """把输入数组转换为 0-1 范围内的二维 float 图像。"""
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 3:
        arr = np.mean(arr, axis=2)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {arr.shape}")
    finite = np.isfinite(arr)
    if not np.all(finite):
        arr = np.where(finite, arr, 0.0)
    lo = float(np.min(arr)) if arr.size else 0.0
    hi = float(np.max(arr)) if arr.size else 0.0
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return np.ascontiguousarray(arr, dtype=np.float32)


def load_aerial_npz(path: str | Path) -> np.ndarray:
    """读取 manifest 中的 `aerial_npz`，默认使用 `image` key。"""
    npz_path = Path(path)
    if not npz_path.exists():
        raise FileNotFoundError(f"aerial_npz does not exist: {npz_path}")
    data = np.load(str(npz_path))
    if isinstance(data, np.lib.npyio.NpzFile):
        try:
            key = "image" if "image" in data.files else data.files[0]
            return _as_float_image(data[key])
        finally:
            data.close()
    return _as_float_image(data)


def validate_behavior_row(row: Mapping[str, Any]) -> None:
    """校验 behavior manifest 行，只允许当前主线需要的 CD-SEM/aerial_npz 输入。"""
    legacy = sorted(key for key in LEGACY_BEHAVIOR_KEYS if key in row and row.get(key) not in (None, ""))
    if legacy:
        raise ValueError(f"Legacy behavior channels are no longer supported: {', '.join(legacy)}")
    if not row.get("aerial_npz"):
        raise ValueError("behavior manifest row must contain aerial_npz")


def _grid_means(arr: np.ndarray, grid_size: int) -> np.ndarray:
    """计算固定网格均值，用于保留粗粒度空间分布。"""
    h, w = arr.shape
    ys = np.linspace(0, h, grid_size + 1, dtype=int)
    xs = np.linspace(0, w, grid_size + 1, dtype=int)
    values = []
    for iy in range(grid_size):
        for ix in range(grid_size):
            block = arr[ys[iy] : ys[iy + 1], xs[ix] : xs[ix + 1]]
            values.append(float(np.mean(block)) if block.size else 0.0)
    return np.asarray(values, dtype=np.float32)


def _profile_features(arr: np.ndarray, bins: int = 16) -> np.ndarray:
    """提取行列投影 profile 的低维摘要。"""
    if arr.size == 0:
        return np.zeros((bins * 2,), dtype=np.float32)
    row_profile = np.mean(arr, axis=1)
    col_profile = np.mean(arr, axis=0)
    row_idx = np.linspace(0, len(row_profile), bins + 1, dtype=int)
    col_idx = np.linspace(0, len(col_profile), bins + 1, dtype=int)
    row_values = [float(np.mean(row_profile[row_idx[i] : row_idx[i + 1]])) if row_idx[i + 1] > row_idx[i] else 0.0 for i in range(bins)]
    col_values = [float(np.mean(col_profile[col_idx[i] : col_idx[i + 1]])) if col_idx[i + 1] > col_idx[i] else 0.0 for i in range(bins)]
    return np.asarray(row_values + col_values, dtype=np.float32)


def _image_stats(arr: np.ndarray) -> np.ndarray:
    """提取图像强度、梯度和分位数统计。"""
    if arr.size == 0:
        return np.zeros((14,), dtype=np.float32)
    gx = np.abs(arr[:, 1:] - arr[:, :-1]) if arr.shape[1] > 1 else np.zeros((arr.shape[0], 0), dtype=np.float32)
    gy = np.abs(arr[1:, :] - arr[:-1, :]) if arr.shape[0] > 1 else np.zeros((0, arr.shape[1]), dtype=np.float32)
    grad = np.concatenate([gx.ravel(), gy.ravel()]) if gx.size or gy.size else np.zeros((1,), dtype=np.float32)
    qs = np.percentile(arr, [5, 25, 50, 75, 95]).astype(np.float32)
    stats = [
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.min(arr)),
        float(np.max(arr)),
        float(np.mean(grad)),
        float(np.std(grad)),
        float(np.mean(arr > 0.25)),
        float(np.mean(arr > 0.50)),
        float(np.mean(arr > 0.75)),
    ]
    return np.asarray(stats + [float(value) for value in qs], dtype=np.float32)


def extract_handcrafted_feature(layout_bitmap: np.ndarray, aerial_image: np.ndarray) -> np.ndarray:
    """为单个 marker 样本生成 L2 归一化 handcrafted feature。"""
    layout = np.asarray(layout_bitmap, dtype=bool)
    aerial = _as_float_image(aerial_image)
    layout_float = layout.astype(np.float32)
    parts = [
        bitmap_fingerprint(layout),
        _grid_means(aerial, 8),
        _profile_features(layout_float, 16),
        _profile_features(aerial, 16),
        _image_stats(layout_float),
        _image_stats(aerial),
    ]
    vector = np.concatenate([np.asarray(part, dtype=np.float32).ravel() for part in parts]).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector


def encode_handcrafted_features(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """批量编码 marker 样本，返回 feature matrix 和轻量 metadata。"""
    features = []
    metadata = []
    for sample in samples:
        row = dict(sample.get("behavior", {}) or {})
        validate_behavior_row(row)
        aerial = load_aerial_npz(row["aerial_npz"])
        feature = extract_handcrafted_feature(sample["clip_bitmap"], aerial)
        features.append(feature)
        meta = dict(sample.get("metadata", {}) or {})
        meta["aerial_npz"] = str(row["aerial_npz"])
        meta["risk_score"] = float(row.get("risk_score", 0.0) or 0.0)
        metadata.append(meta)
    matrix = np.vstack(features).astype(np.float32) if features else np.zeros((0, 0), dtype=np.float32)
    return {"features": matrix, "metadata": metadata}
