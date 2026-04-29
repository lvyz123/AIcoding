#!/usr/bin/env python3
"""LSF 版本使用的独立版图工具函数。

中文说明：
1. 本文件只放 v2_lsf 运行所需的最小 OAS、bbox、polygon 工具。
2. 不 import 旧版 layout_utils.py，保证 LSF 版本可以独立部署。
3. 代码保持 Python 3.6 兼容，不使用现代类型语法。
"""

import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional, Tuple

import gdstk
import numpy as np


def _ascii_safe_token(value, fallback="layout"):
    """生成适合作为临时文件名的 ASCII token。"""

    if not value:
        return str(fallback)
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", str(value).strip())
    return (cleaned or str(fallback))[:64]


def _is_ascii_path(filepath):
    """判断路径是否能直接交给 gdstk 的 C 层接口。"""

    try:
        str(filepath).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _temporary_ascii_path(prefix, suffix):
    """为非 ASCII 路径桥接生成临时 ASCII 文件路径。"""

    root = Path(tempfile.gettempdir()) / "layout_lsf_ascii_bridge"
    if not root.exists():
        root.mkdir(parents=True)
    filename = "%s_%s%s" % (_ascii_safe_token(prefix, "oas"), uuid.uuid4().hex[:8], suffix)
    return root / filename


def _read_oas_only_library(filepath):
    """读取 OAS/OASIS 文件，并在必要时通过 ASCII 临时路径桥接。"""

    src = Path(str(filepath)).resolve()
    if _is_ascii_path(str(src)):
        return gdstk.read_oas(str(src))
    bridge = _temporary_ascii_path(src.stem, src.suffix or ".oas")
    shutil.copy2(str(src), str(bridge))
    try:
        return gdstk.read_oas(str(bridge))
    finally:
        try:
            os.remove(str(bridge))
        except OSError:
            pass


def _write_oas_library(lib, filepath):
    """写出 OAS/OASIS 文件，并在必要时通过 ASCII 临时路径桥接。"""

    dst = Path(str(filepath)).resolve()
    parent = dst.parent
    if not parent.exists():
        parent.mkdir(parents=True)
    if _is_ascii_path(str(dst)):
        lib.write_oas(str(dst))
        return
    bridge = _temporary_ascii_path(dst.stem, dst.suffix or ".oas")
    try:
        lib.write_oas(str(bridge))
        shutil.copy2(str(bridge), str(dst))
    finally:
        try:
            os.remove(str(bridge))
        except OSError:
            pass


def _resolve_fs_path(path_value):
    """把输入值解析成绝对文件系统路径。"""

    return Path(str(path_value)).resolve()


def _make_centered_bbox(center_xy, width_um, height_um):
    """按中心点和宽高生成 bbox。"""

    cx = float(center_xy[0])
    cy = float(center_xy[1])
    half_w = float(width_um) * 0.5
    half_h = float(height_um) * 0.5
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def _bbox_center(bbox):
    """返回 bbox 中心点。"""

    if bbox is None or len(bbox) < 4:
        return (0.0, 0.0)
    x0 = float(bbox[0])
    y0 = float(bbox[1])
    x1 = float(bbox[2])
    y1 = float(bbox[3])
    return ((x0 + x1) * 0.5, (y0 + y1) * 0.5)


def _bbox_intersection(bbox_a, bbox_b):
    """计算两个 bbox 的交集；无交集时返回 None。"""

    if bbox_a is None or bbox_b is None:
        return None
    ax0, ay0, ax1, ay1 = [float(v) for v in bbox_a]
    bx0, by0, bx1, by1 = [float(v) for v in bbox_b]
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix0 < ix1 and iy0 < iy1:
        return (ix0, iy0, ix1, iy1)
    return None


def _safe_bbox_tuple(bbox):
    """把 gdstk bbox 或普通序列安全转换成四元组。"""

    if bbox is None:
        return None
    try:
        if hasattr(bbox, "__len__") and len(bbox) == 2:
            lower = bbox[0]
            upper = bbox[1]
            if hasattr(lower, "__len__") and hasattr(upper, "__len__"):
                if len(lower) >= 2 and len(upper) >= 2:
                    return (float(lower[0]), float(lower[1]), float(upper[0]), float(upper[1]))
        if hasattr(bbox, "__len__") and len(bbox) >= 4:
            return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except (TypeError, ValueError):
        return None
    return None


def _element_layer_datatype(element):
    """返回 gdstk element 的 layer/datatype。"""

    return int(getattr(element, "layer", 0)), int(getattr(element, "datatype", 0))


def _polygon_vertices_array(polygon):
    """返回 polygon 顶点数组；非法 polygon 返回 None。"""

    if polygon is None or not hasattr(polygon, "points"):
        return None
    points = np.asarray(polygon.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
        return None
    return points[:, :2]
