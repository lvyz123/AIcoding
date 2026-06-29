#!/usr/bin/env python3
"""当前 optimized 主线使用的最小层布尔操作工具。

中文说明：
1. 这里的 layer operation 只负责把 `source/target` 两层做布尔运算，生成新的 `result layer`。
2. 对当前主线来说，`source_layer` / `target_layer` 是辅助层（helper layer），它们的职责只是参与布尔运算。
3. 真正进入后续聚类的 pattern geometry 需要由主线再做一次过滤：
   - helper-only layer 不参与聚类
   - result layer 参与聚类
   - 未出现在任何规则中的其它层保持原样参与
4. 这样可以避免把 source / target / result 三层同时送入聚类，造成元素数膨胀和重复语义。
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Set, Tuple

import gdstk
import numpy as np


LayerSpec = Tuple[int, int]


def _normalize_layer_spec(spec) -> LayerSpec:
    if isinstance(spec, str):
        layer_str, datatype_str = spec.split("/", 1)
        return int(layer_str.strip()), int(datatype_str.strip())
    if isinstance(spec, Sequence) and len(spec) >= 2:
        return int(spec[0]), int(spec[1])
    raise ValueError(f"Invalid layer spec: {spec}")


def _clone_polygon(poly: gdstk.Polygon, *, layer: int, datatype: int) -> gdstk.Polygon:
    points = np.array(poly.points, dtype=np.float64, copy=True)
    return gdstk.Polygon(points, layer=int(layer), datatype=int(datatype))


class LayerOperationProcessor:
    """管理 layer operation 规则，并负责执行布尔运算。"""

    _OPERATION_MAP = {
        "subtract": "not",
        "union": "or",
        "intersect": "and",
    }

    def __init__(self):
        self.operation_rules: List[dict] = []

    def register_operation_rule(self, source_layer, operation, target_layer, result_layer) -> None:
        """注册一条 `source operation target -> result` 的布尔规则。"""

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
        """从单个 cell 中收集指定层上的 polygon，并做一份独立拷贝。"""

        polygons = []
        for poly in cell.get_polygons(apply_repetitions=True, include_paths=True, depth=None):
            if (int(poly.layer), int(poly.datatype)) != layer_spec:
                continue
            polygons.append(_clone_polygon(poly, layer=int(poly.layer), datatype=int(poly.datatype)))
        return polygons

    def helper_layer_specs(self) -> Set[LayerSpec]:
        """返回所有只作为 source/target 出现的 helper layer 候选集合。"""

        layers: Set[LayerSpec] = set()
        for rule in self.operation_rules:
            layers.add(_normalize_layer_spec(rule["source_layer"]))
            layers.add(_normalize_layer_spec(rule["target_layer"]))
        return layers

    def result_layer_specs(self) -> Set[LayerSpec]:
        """返回所有由布尔运算生成的 result layer 集合。"""

        return {_normalize_layer_spec(rule["result_layer"]) for rule in self.operation_rules}

    def should_keep_pattern_layer(self, layer_spec: LayerSpec) -> bool:
        """判断某个层在 layer-op 之后是否仍应参与聚类。

        规则：
        - helper-only layer 不参与聚类
        - result layer 始终保留
        - 未出现在任何规则中的层保持原样保留
        """

        normalized = _normalize_layer_spec(layer_spec)
        result_layers = self.result_layer_specs()
        helper_layers = self.helper_layer_specs()
        if normalized in result_layers:
            return True
        if normalized in helper_layers:
            return False
        return True

    def effective_pattern_layers(self, seen_layers: Iterable[LayerSpec]) -> List[LayerSpec]:
        """对已出现的层集合应用 helper/result 语义过滤，返回最终聚类层。"""

        retained = {_normalize_layer_spec(spec) for spec in seen_layers if self.should_keep_pattern_layer(spec)}
        return sorted(retained)

    def apply_layer_operations(self, lib: gdstk.Library) -> gdstk.Library:
        """对库中的顶层 cell 执行已注册的布尔规则，并把结果写到 result layer。"""

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
