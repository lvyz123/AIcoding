"""
Layout聚类工具函数模块
包含几何处理、文件操作、数学计算等通用工具函数
"""

import numpy as np
import math
import shutil
import uuid
from pathlib import Path
from typing import List, Tuple, Set, Dict, Any, Optional
import gdstk

# 从主文件导入必要函数（避免循环导入）
# 这些函数将在重构过程中逐步移动


def _ascii_safe_token(value: str, fallback: str = "layout") -> str:
    """生成ASCII安全的令牌字符串"""
    import re
    if not value:
        return str(fallback)
    cleaned = re.sub(r'[^a-zA-Z0-9_-]', '_', str(value).strip())
    if not cleaned:
        return str(fallback)
    return cleaned[:64]


def _make_ascii_temp_output_path(temp_dir, *, prefix: str, source_path: str) -> str:
    """生成ASCII安全的临时输出路径"""
    import uuid
    from pathlib import Path
    safe_prefix = _ascii_safe_token(prefix, "temp")
    safe_source = _ascii_safe_token(Path(source_path).name, "source")
    uid = uuid.uuid4().hex[:8]
    filename = f"{safe_prefix}_{safe_source}_{uid}.oas"
    return str(temp_dir / filename)


def _resolve_fs_path(path_value: Any) -> Path:
    """解析文件系统路径"""
    from pathlib import Path
    return Path(str(path_value)).resolve()


def _pushd(directory: Path):
    """临时切换目录的上下文管理器"""
    import os
    from contextlib import contextmanager

    @contextmanager
    def pushd_context():
        original = os.getcwd()
        try:
            os.chdir(str(directory))
            yield
        finally:
            os.chdir(original)

    return pushd_context()


def _make_centered_bbox(center_xy, width_um, height_um):
    """创建以给定点为中心的边界框"""
    cx, cy = float(center_xy[0]), float(center_xy[1])
    half_w = float(width_um) / 2.0
    half_h = float(height_um) / 2.0
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def _bbox_center(bbox):
    """计算边界框中心"""
    if bbox is None or len(bbox) < 4:
        return (0.0, 0.0)
    x0, y0, x1, y1 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _bbox_intersection(bbox_a, bbox_b):
    """计算两个边界框的交集"""
    if bbox_a is None or bbox_b is None:
        return None
    ax0, ay0, ax1, ay1 = float(bbox_a[0]), float(bbox_a[1]), float(bbox_a[2]), float(bbox_a[3])
    bx0, by0, bx1, by1 = float(bbox_b[0]), float(bbox_b[1]), float(bbox_b[2]), float(bbox_b[3])
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix0 < ix1 and iy0 < iy1:
        return (ix0, iy0, ix1, iy1)
    return None


def _safe_bbox_tuple(bbox):
    """安全转换边界框为元组"""
    if bbox is None:
        return None
    try:
        if hasattr(bbox, '__len__') and len(bbox) == 2:
            lower = bbox[0]
            upper = bbox[1]
            if hasattr(lower, '__len__') and hasattr(upper, '__len__') and len(lower) >= 2 and len(upper) >= 2:
                return (float(lower[0]), float(lower[1]), float(upper[0]), float(upper[1]))
        if hasattr(bbox, '__len__') and len(bbox) >= 4:
            return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except (TypeError, ValueError):
        pass
    return None


def _is_ascii_path(filepath: str) -> bool:
    """检查路径是否可直接交给 gdstk 的 C 层文件接口。"""
    try:
        str(filepath).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _oas_bridge_path(prefix: str) -> Path:
    """为 gdstk 创建 ASCII-only 临时 OAS 路径，用于桥接中文目录读写。"""
    bridge_dir = Path(__file__).resolve().parent / "_oas_path_bridge"
    bridge_dir.mkdir(parents=True, exist_ok=True)
    return bridge_dir / f"{prefix}_{uuid.uuid4().hex}.oas"


# 更多工具函数可以在此添加


def _read_oas_only_library(filepath: str):
    """读取 OASIS 文件，返回 gdstk.Library 对象"""
    path = Path(str(filepath))
    try:
        if _is_ascii_path(str(path)):
            lib = gdstk.read_oas(str(path))
        else:
            temp_path = _oas_bridge_path("read")
            try:
                shutil.copy2(path, temp_path)
                lib = gdstk.read_oas(str(temp_path))
            finally:
                try:
                    temp_path.unlink()
                except OSError:
                    pass
    except Exception as e:
        raise IOError(f"读取 OASIS 文件失败 {filepath}: {e}") from e
    if lib is None:
        raise IOError(f"无法解析 OASIS 文件 {filepath} (可能是空的或格式不正确)")
    return lib


def _write_oas_library(lib: gdstk.Library, filepath: str) -> None:
    """将 gdstk.Library 写入 OASIS 文件"""
    path = Path(str(filepath))
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if _is_ascii_path(str(path)):
            lib.write_oas(str(path))
        else:
            temp_path = _oas_bridge_path("write")
            try:
                lib.write_oas(str(temp_path))
                shutil.copy2(temp_path, path)
            finally:
                try:
                    temp_path.unlink()
                except OSError:
                    pass
    except Exception as e:
        raise IOError(f"写入 OASIS 文件失败 {filepath}: {e}")


def _element_layer_datatype(element):
    """获取元素的层和数据类型"""
    layer = getattr(element, "layer", 0)
    datatype = getattr(element, "datatype", 0)
    return (layer, datatype)


def _polygon_vertices_array(polygon):
    """将多边形顶点转换为 numpy 数组"""
    if polygon is None:
        return None
    if hasattr(polygon, "points"):
        points = polygon.points
    else:
        return None
    if points is None or len(points) == 0:
        return None
    return np.asarray(points, dtype=np.float64)


def polygon_perimeter(polygon):
    """计算多边形周长"""
    points = np.asarray(polygon.points, dtype=np.float64)
    if len(points) < 2:
        return 0.0
    if np.allclose(points[0], points[-1]):
        points = points[:-1]  # 闭合多边形，移除重复的最后一个点
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0, append=points[0:1])  # 首尾相连
    return float(np.sum(np.sqrt(np.sum(diffs * diffs, axis=1))))
