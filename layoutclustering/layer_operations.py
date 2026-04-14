#!/usr/bin/env python3
"""Minimal layer-operation helpers used by the current mainline."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

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
    """Apply boolean layer operations and emit result polygons on target layers."""

    _OPERATION_MAP = {
        "subtract": "not",
        "union": "or",
        "intersect": "and",
    }

    def __init__(self):
        self.operation_rules: List[dict] = []

    def register_operation_rule(self, source_layer, operation, target_layer, result_layer) -> None:
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
        polygons = []
        for poly in cell.get_polygons(apply_repetitions=True, include_paths=True, depth=None):
            if (int(poly.layer), int(poly.datatype)) != layer_spec:
                continue
            polygons.append(_clone_polygon(poly, layer=int(poly.layer), datatype=int(poly.datatype)))
        return polygons

    def apply_layer_operations(self, lib: gdstk.Library) -> gdstk.Library:
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
