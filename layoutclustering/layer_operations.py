#!/usr/bin/env python3
"""Layer boolean operation helpers used before marker clipping.

本模块负责在 optimized 主流程读取 OAS 后、收集 marker 前，对用户注册的
层操作规则执行简单 deterministic boolean operation。它的设计很薄：只维护规则列表，
不参与聚类、不改变输出格式，也不引入旧算法分支。

规则语义固定为:

source_layer operation target_layer -> result_layer

支持的 operation:

- subtract: source NOT target
- union: source OR target
- intersect: source AND target

所有 layer spec 都使用 `layer/datatype` 或 `(layer, datatype)` 形式。执行时会遍历
top cells 中展开后的 polygon，把结果 polygon 克隆到 result layer；原始层图形不删除。
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import gdstk
import numpy as np


LayerSpec = Tuple[int, int]


def _normalize_layer_spec(spec) -> LayerSpec:
    """把字符串或序列形式的 layer spec 规范化成 `(layer, datatype)` 元组。"""
    if isinstance(spec, str):
        layer_str, datatype_str = spec.split("/", 1)
        return int(layer_str.strip()), int(datatype_str.strip())
    if isinstance(spec, Sequence) and len(spec) >= 2:
        return int(spec[0]), int(spec[1])
    raise ValueError(f"Invalid layer spec: {spec}")


def _clone_polygon(poly: gdstk.Polygon, *, layer: int, datatype: int) -> gdstk.Polygon:
    """复制 polygon 顶点，并写入新的 layer/datatype，避免修改原图形对象。"""
    points = np.array(poly.points, dtype=np.float64, copy=True)
    return gdstk.Polygon(points, layer=int(layer), datatype=int(datatype))


class LayerOperationProcessor:
    """保存并执行用户注册的 boolean layer operation 规则。"""

    _OPERATION_MAP = {
        "subtract": "not",
        "union": "or",
        "intersect": "and",
    }

    def __init__(self):
        """初始化空规则列表。"""
        self.operation_rules: List[dict] = []

    def register_operation_rule(self, source_layer, operation, target_layer, result_layer) -> None:
        """注册一条 layer boolean 规则，并校验操作类型和 layer spec。"""
        op = str(operation).strip().lower()
        if op not in self._OPERATION_MAP:
            raise ValueError(f"Unsupported layer operation: {operation}")
        self.operation_rules.append(
            {
                "source_layer": _normalize_layer_spec(source_layer),
                "target_layer": _normalize_layer_spec(target_layer),
                "result_layer": _normalize_layer_spec(result_layer),
                "operation": op,
            }
        )

    def _collect_polygons(self, cell: gdstk.Cell, layer_spec: LayerSpec) -> List[gdstk.Polygon]:
        """从指定 cell 中收集目标 layer/datatype 的展开 polygon 副本。"""
        polygons = []
        for poly in cell.get_polygons(apply_repetitions=True, include_paths=True, depth=None):
            if (int(poly.layer), int(poly.datatype)) != layer_spec:
                continue
            polygons.append(_clone_polygon(poly, layer=int(poly.layer), datatype=int(poly.datatype)))
        return polygons

    def apply_layer_operations(self, lib: gdstk.Library) -> gdstk.Library:
        """对 library 的 top cells 应用所有规则，并把结果 polygon 加回 result layer。"""
        if not self.operation_rules:
            return lib
        for cell in list(lib.top_level()) or list(lib.cells):
            for rule in self.operation_rules:
                source_polygons = self._collect_polygons(cell, rule["source_layer"])
                target_polygons = self._collect_polygons(cell, rule["target_layer"])
                if not source_polygons or not target_polygons:
                    continue
                result = gdstk.boolean(
                    source_polygons,
                    target_polygons,
                    self._OPERATION_MAP[rule["operation"]],
                ) or []
                if not result:
                    continue
                cell.add(
                    *[
                        _clone_polygon(
                            poly,
                            layer=rule["result_layer"][0],
                            datatype=rule["result_layer"][1],
                        )
                        for poly in result
                    ]
                )
        return lib
