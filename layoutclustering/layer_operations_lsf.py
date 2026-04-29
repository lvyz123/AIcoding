#!/usr/bin/env python3
"""LSF 版本使用的独立 layer operation 工具。

中文说明：
1. 本文件复制并精简旧版 layer operation 语义，但不 import 旧脚本。
2. 支持 subtract / union / intersect 三类布尔运算。
3. 代码保持 Python 3.6 兼容，供 layout_clustering_optimized_v2_lsf.py 独立部署。
"""

from typing import Iterable, List, Sequence, Set, Tuple

import gdstk
import numpy as np


LayerSpec = Tuple[int, int]


def _normalize_layer_spec(spec):
    """把 1/0 或二元序列规范化成 layer/datatype。"""

    if isinstance(spec, str):
        layer_str, datatype_str = spec.split("/", 1)
        return int(layer_str.strip()), int(datatype_str.strip())
    if isinstance(spec, Sequence) and len(spec) >= 2:
        return int(spec[0]), int(spec[1])
    raise ValueError("Invalid layer spec: %s" % (spec,))


def _clone_polygon(poly, layer, datatype):
    """复制 polygon 并写入目标层号。"""

    points = np.array(poly.points, dtype=np.float64, copy=True)
    return gdstk.Polygon(points, layer=int(layer), datatype=int(datatype))


class LayerOperationProcessor(object):
    """管理 v2_lsf 的 layer operation 规则。"""

    _OPERATION_MAP = {
        "subtract": "not",
        "union": "or",
        "intersect": "and",
    }

    def __init__(self):
        self.operation_rules = []

    def register_operation_rule(self, source_layer, operation, target_layer, result_layer):
        """注册一条 source operation target -> result 的布尔规则。"""

        op = str(operation).strip().lower()
        if op not in self._OPERATION_MAP:
            raise ValueError("Unsupported layer operation: %s" % operation)
        self.operation_rules.append(
            {
                "source_layer": _normalize_layer_spec(source_layer),
                "target_layer": _normalize_layer_spec(target_layer),
                "result_layer": _normalize_layer_spec(result_layer),
                "operation": op,
            }
        )

    def _collect_polygons(self, cell, layer_spec):
        """从 cell 中收集某层 polygon，并复制成独立对象。"""

        polygons = []
        for poly in cell.get_polygons(apply_repetitions=True, include_paths=True, depth=None):
            if (int(poly.layer), int(poly.datatype)) != layer_spec:
                continue
            polygons.append(_clone_polygon(poly, int(poly.layer), int(poly.datatype)))
        return polygons

    def helper_layer_specs(self):
        """返回只作为 source/target 出现的 helper layer 集合。"""

        layers = set()
        for rule in self.operation_rules:
            layers.add(_normalize_layer_spec(rule["source_layer"]))
            layers.add(_normalize_layer_spec(rule["target_layer"]))
        return layers

    def result_layer_specs(self):
        """返回所有 result layer 集合。"""

        return set(_normalize_layer_spec(rule["result_layer"]) for rule in self.operation_rules)

    def should_keep_pattern_layer(self, layer_spec):
        """判断 layer-op 后某层是否应参与聚类。"""

        normalized = _normalize_layer_spec(layer_spec)
        result_layers = self.result_layer_specs()
        helper_layers = self.helper_layer_specs()
        if normalized in result_layers:
            return True
        if normalized in helper_layers:
            return False
        return True

    def effective_pattern_layers(self, seen_layers):
        """根据 helper/result 语义返回真正参与聚类的层。"""

        retained = set()
        for spec in seen_layers:
            normalized = _normalize_layer_spec(spec)
            if self.should_keep_pattern_layer(normalized):
                retained.add(normalized)
        return sorted(retained)

    def apply_layer_operations(self, lib):
        """对库中 top cells 执行已注册的布尔运算。"""

        if not self.operation_rules:
            return lib
        cells = list(lib.top_level()) or list(lib.cells)
        for cell in cells:
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
                for poly in result:
                    cell.add(
                        _clone_polygon(
                            poly,
                            int(rule["result_layer"][0]),
                            int(rule["result_layer"][1]),
                        )
                    )
        return lib
