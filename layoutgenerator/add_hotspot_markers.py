#!/usr/bin/env python3
"""Add hotspot marker rectangles on a dedicated layer to an OASIS layout."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _import_gdstk() -> Any:
    try:
        import gdstk  # type: ignore

        return gdstk
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency 'gdstk'. Install with: pip install gdstk"
        ) from exc


@dataclass(slots=True)
class MarkerConfig:
    input_path: Path
    output_path: Path
    report_path: Path
    marker_layer: int
    marker_datatype: int
    marker_size_um: float
    markers_per_layer: int
    bin_size_um: float
    polygon_radius_um: float
    path_radius_um: float
    min_marker_spacing_um: float


@dataclass(slots=True)
class PolygonRecord:
    layer: int
    x: float
    y: float
    width: float
    height: float
    vertex_count: int


@dataclass(slots=True)
class PathRecord:
    layer: int
    x: float
    y: float


@dataclass(slots=True)
class MarkerRecord:
    cell_name: str
    source_layer: int
    marker_layer: int
    marker_datatype: int
    hotspot_style: str
    center_x_um: float
    center_y_um: float
    marker_size_um: float
    score: float
    vertex_count: int
    aspect_ratio: float
    area_um2: float
    edge_distance_um: float
    nearest_polygon_um: float | None
    nearby_polygon_count: int
    nearest_path_um: float | None


@dataclass(slots=True)
class CandidateRecord:
    cell_name: str
    source_layer: int
    center_x_um: float
    center_y_um: float
    score: float
    vertex_count: int
    aspect_ratio: float
    area_um2: float
    edge_distance_um: float
    nearest_polygon_um: float | None
    nearby_polygon_count: int
    nearest_path_um: float | None


def parse_args() -> MarkerConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect polygons in an OASIS layout and add small hotspot marker "
            "rectangles on a dedicated layer."
        )
    )
    parser.add_argument("input", help="Input OASIS layout path")
    parser.add_argument(
        "--output",
        default=None,
        help="Output OASIS layout path. Defaults to overwrite the input file safely.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="JSON report path for selected hotspot markers",
    )
    parser.add_argument("--marker-layer", type=int, default=999)
    parser.add_argument("--marker-datatype", type=int, default=0)
    parser.add_argument("--marker-size", type=float, default=0.1, help="Marker width/height in um")
    parser.add_argument(
        "--markers-per-layer",
        type=int,
        default=5,
        help="How many hotspot markers to place for each cell and source layer",
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=2.0,
        help="Spatial bin size in um used for local density checks",
    )
    parser.add_argument(
        "--polygon-radius",
        type=float,
        default=1.6,
        help="Radius in um used to score polygon crowding",
    )
    parser.add_argument(
        "--path-radius",
        type=float,
        default=1.2,
        help="Radius in um used to score polygon proximity to paths",
    )
    parser.add_argument(
        "--min-marker-spacing",
        type=float,
        default=3.0,
        help="Minimum spacing in um between selected markers inside the same cell/layer",
    )

    args = parser.parse_args()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        parser.error(f"Input file not found: {input_path}")
    if input_path.suffix.lower() != ".oas":
        parser.error("Only OASIS (.oas) files are supported")

    output_path = Path(args.output).resolve() if args.output else input_path
    report_path = (
        Path(args.report).resolve()
        if args.report
        else output_path.with_name(f"{output_path.stem}_hotspot_markers.json")
    )

    cfg = MarkerConfig(
        input_path=input_path,
        output_path=output_path,
        report_path=report_path,
        marker_layer=args.marker_layer,
        marker_datatype=args.marker_datatype,
        marker_size_um=float(args.marker_size),
        markers_per_layer=int(args.markers_per_layer),
        bin_size_um=float(args.bin_size),
        polygon_radius_um=float(args.polygon_radius),
        path_radius_um=float(args.path_radius),
        min_marker_spacing_um=float(args.min_marker_spacing),
    )
    _validate_config(cfg, parser)
    return cfg


def _validate_config(config: MarkerConfig, parser: argparse.ArgumentParser) -> None:
    if config.marker_layer < 0 or config.marker_datatype < 0:
        parser.error("marker layer/datatype must be >= 0")
    if config.marker_size_um <= 0:
        parser.error("marker size must be > 0")
    if config.markers_per_layer <= 0:
        parser.error("markers-per-layer must be > 0")
    if config.bin_size_um <= 0:
        parser.error("bin-size must be > 0")
    if config.polygon_radius_um <= 0 or config.path_radius_um <= 0:
        parser.error("polygon-radius and path-radius must be > 0")
    if config.min_marker_spacing_um < 0:
        parser.error("min-marker-spacing must be >= 0")


def _bin_key(x: float, y: float, bin_size: float) -> tuple[int, int]:
    return int(math.floor(x / bin_size)), int(math.floor(y / bin_size))


def _polygon_record(poly: Any) -> PolygonRecord:
    points = poly.points
    xs = points[:, 0]
    ys = points[:, 1]
    x0 = float(xs.min())
    x1 = float(xs.max())
    y0 = float(ys.min())
    y1 = float(ys.max())
    return PolygonRecord(
        layer=int(poly.layer),
        x=(x0 + x1) * 0.5,
        y=(y0 + y1) * 0.5,
        width=x1 - x0,
        height=y1 - y0,
        vertex_count=len(points),
    )


def _path_records(path: Any) -> list[PathRecord]:
    records: list[PathRecord] = []
    spines = path.path_spines()
    layers = tuple(int(layer) for layer in path.layers) if path.layers else (0,)

    for idx, spine in enumerate(spines):
        layer = layers[min(idx, len(layers) - 1)]
        start = spine[0]
        end = spine[-1]
        records.append(
            PathRecord(
                layer=layer,
                x=float((start[0] + end[0]) * 0.5),
                y=float((start[1] + end[1]) * 0.5),
            )
        )
    return records


def _build_spatial_bins(
    records: list[PolygonRecord] | list[PathRecord],
    bin_size_um: float,
) -> dict[tuple[int, int], list[Any]]:
    bins: dict[tuple[int, int], list[Any]] = defaultdict(list)
    for record in records:
        bins[_bin_key(record.x, record.y, bin_size_um)].append(record)
    return bins


def _score_polygon(
    record: PolygonRecord,
    polygon_bins: dict[tuple[int, int], list[PolygonRecord]],
    path_bins: dict[tuple[int, int], list[PathRecord]],
    config: MarkerConfig,
) -> tuple[float, float | None, int, float | None]:
    bx, by = _bin_key(record.x, record.y, config.bin_size_um)
    nearest_polygon = math.inf
    nearby_polygon_count = 0
    nearest_path = math.inf

    polygon_steps = max(1, int(math.ceil(config.polygon_radius_um / config.bin_size_um)))
    path_steps = max(1, int(math.ceil(config.path_radius_um / config.bin_size_um)))

    for ix in range(bx - polygon_steps, bx + polygon_steps + 1):
        for iy in range(by - polygon_steps, by + polygon_steps + 1):
            for other in polygon_bins.get((ix, iy), []):
                if other is record:
                    continue
                distance = math.hypot(record.x - other.x, record.y - other.y)
                if distance < config.polygon_radius_um:
                    nearby_polygon_count += 1
                    if distance < nearest_polygon:
                        nearest_polygon = distance

    for ix in range(bx - path_steps, bx + path_steps + 1):
        for iy in range(by - path_steps, by + path_steps + 1):
            for other in path_bins.get((ix, iy), []):
                distance = math.hypot(record.x - other.x, record.y - other.y)
                if distance < config.path_radius_um and distance < nearest_path:
                    nearest_path = distance

    complexity_score = max(0.0, (record.vertex_count - 4) * 1.4)
    aspect_ratio = max(record.width, record.height) / max(min(record.width, record.height), 1e-6)
    size_score = max(record.width, record.height)
    polygon_proximity_score = (
        max(0.0, config.polygon_radius_um - nearest_polygon) * 4.0
        if nearest_polygon < math.inf
        else 0.0
    )
    polygon_density_score = min(nearby_polygon_count, 6) * 0.35
    path_proximity_score = (
        max(0.0, config.path_radius_um - nearest_path) * 3.0
        if nearest_path < math.inf
        else 0.0
    )
    elongation_score = max(0.0, aspect_ratio - 1.15) * 0.45
    score = (
        complexity_score
        + polygon_proximity_score
        + polygon_density_score
        + path_proximity_score
        + elongation_score
        + size_score * 0.15
    )

    return (
        score,
        (None if nearest_polygon == math.inf else nearest_polygon),
        nearby_polygon_count,
        (None if nearest_path == math.inf else nearest_path),
    )


def _style_sort_key(style_name: str, candidate: CandidateRecord) -> tuple[Any, ...]:
    nearest_polygon = (
        candidate.nearest_polygon_um
        if candidate.nearest_polygon_um is not None
        else float("inf")
    )
    nearest_path = (
        candidate.nearest_path_um if candidate.nearest_path_um is not None else float("inf")
    )

    if style_name == "dense_complex":
        return (
            -candidate.score,
            -candidate.vertex_count,
            nearest_polygon,
            candidate.edge_distance_um,
        )
    if style_name == "crowded_array":
        return (
            -candidate.nearby_polygon_count,
            nearest_polygon,
            -candidate.score,
            candidate.edge_distance_um,
        )
    if style_name == "complex_polygon":
        return (
            -candidate.vertex_count,
            -candidate.aspect_ratio,
            -candidate.area_um2,
            -candidate.score,
        )
    if style_name == "edge_stress":
        return (
            candidate.edge_distance_um,
            -candidate.score,
            -candidate.vertex_count,
            nearest_polygon,
        )
    if style_name == "path_transition":
        return (
            nearest_path,
            -candidate.score,
            -candidate.nearby_polygon_count,
            candidate.edge_distance_um,
        )

    return (
        -candidate.score,
        -candidate.nearby_polygon_count,
        -candidate.area_um2,
        candidate.edge_distance_um,
    )


def _build_style_buckets(
    candidates: list[CandidateRecord],
) -> list[tuple[str, list[CandidateRecord]]]:
    max_score = max((candidate.score for candidate in candidates), default=0.0)
    dense_complex = [
        candidate
        for candidate in candidates
        if candidate.score >= max_score * 0.75 or candidate.vertex_count >= 6
    ]
    crowded_array = [
        candidate for candidate in candidates if candidate.nearby_polygon_count >= 4
    ]
    complex_polygon = [
        candidate
        for candidate in candidates
        if candidate.vertex_count >= 6 or candidate.aspect_ratio >= 1.3
    ]
    edge_stress = [
        candidate for candidate in candidates if candidate.edge_distance_um <= 30.0
    ]
    path_transition = [candidate for candidate in candidates if candidate.nearest_path_um is not None]
    broad_relaxed = [
        candidate
        for candidate in candidates
        if candidate.score >= max_score * 0.45 or candidate.nearby_polygon_count >= 2
    ]

    style_buckets = [
        ("dense_complex", dense_complex),
        ("crowded_array", crowded_array),
        ("complex_polygon", complex_polygon),
        ("edge_stress", edge_stress),
        ("path_transition", path_transition),
        ("broad_relaxed", broad_relaxed if broad_relaxed else candidates),
    ]

    return [
        (style_name, sorted(bucket, key=lambda item, name=style_name: _style_sort_key(name, item)))
        for style_name, bucket in style_buckets
        if bucket
    ]


def _select_markers_for_layer(
    cell_name: str,
    source_layer: int,
    polygons: list[PolygonRecord],
    paths: list[PathRecord],
    config: MarkerConfig,
    cell_bbox: tuple[tuple[float, float], tuple[float, float]],
) -> list[MarkerRecord]:
    polygon_bins = _build_spatial_bins(polygons, config.bin_size_um)
    path_bins = _build_spatial_bins(paths, config.bin_size_um)
    x0, y0 = cell_bbox[0]
    x1, y1 = cell_bbox[1]
    candidates: list[CandidateRecord] = []

    for polygon in polygons:
        score, nearest_polygon, nearby_polygon_count, nearest_path = _score_polygon(
            polygon,
            polygon_bins,
            path_bins,
            config,
        )
        aspect_ratio = max(polygon.width, polygon.height) / max(min(polygon.width, polygon.height), 1e-6)
        area_um2 = polygon.width * polygon.height
        edge_distance_um = min(
            polygon.x - x0,
            x1 - polygon.x,
            polygon.y - y0,
            y1 - polygon.y,
        )
        candidates.append(
            CandidateRecord(
                cell_name=cell_name,
                source_layer=source_layer,
                center_x_um=polygon.x,
                center_y_um=polygon.y,
                score=score,
                vertex_count=polygon.vertex_count,
                aspect_ratio=aspect_ratio,
                area_um2=area_um2,
                edge_distance_um=edge_distance_um,
                nearest_polygon_um=nearest_polygon,
                nearby_polygon_count=nearby_polygon_count,
                nearest_path_um=nearest_path,
            )
        )

    style_buckets = _build_style_buckets(candidates)
    selected: list[MarkerRecord] = []
    selected_positions: list[tuple[float, float]] = []
    seen_positions: set[tuple[float, float]] = set()
    min_spacing_sq = config.min_marker_spacing_um * config.min_marker_spacing_um
    bucket_positions = {style_name: 0 for style_name, _ in style_buckets}

    while len(selected) < config.markers_per_layer:
        picked_in_round = False
        for style_name, bucket in style_buckets:
            position = bucket_positions[style_name]
            while position < len(bucket):
                candidate = bucket[position]
                position += 1
                key = (round(candidate.center_x_um, 6), round(candidate.center_y_um, 6))
                if key in seen_positions:
                    continue
                if any(
                    (candidate.center_x_um - x) ** 2 + (candidate.center_y_um - y) ** 2
                    < min_spacing_sq
                    for x, y in selected_positions
                ):
                    continue

                selected.append(
                    MarkerRecord(
                        cell_name=candidate.cell_name,
                        source_layer=candidate.source_layer,
                        marker_layer=config.marker_layer,
                        marker_datatype=config.marker_datatype,
                        hotspot_style=style_name,
                        center_x_um=candidate.center_x_um,
                        center_y_um=candidate.center_y_um,
                        marker_size_um=config.marker_size_um,
                        score=candidate.score,
                        vertex_count=candidate.vertex_count,
                        aspect_ratio=candidate.aspect_ratio,
                        area_um2=candidate.area_um2,
                        edge_distance_um=candidate.edge_distance_um,
                        nearest_polygon_um=candidate.nearest_polygon_um,
                        nearby_polygon_count=candidate.nearby_polygon_count,
                        nearest_path_um=candidate.nearest_path_um,
                    )
                )
                selected_positions.append((candidate.center_x_um, candidate.center_y_um))
                seen_positions.add(key)
                picked_in_round = True
                break
            bucket_positions[style_name] = position
            if len(selected) >= config.markers_per_layer:
                break

        if not picked_in_round:
            break

    return selected


def _add_marker_rectangles(gdstk: Any, cell: Any, markers: list[MarkerRecord], config: MarkerConfig) -> None:
    half = config.marker_size_um * 0.5
    rectangles = [
        gdstk.rectangle(
            (marker.center_x_um - half, marker.center_y_um - half),
            (marker.center_x_um + half, marker.center_y_um + half),
            layer=config.marker_layer,
            datatype=config.marker_datatype,
        )
        for marker in markers
    ]
    if rectangles:
        cell.add(*rectangles)


def _write_library(lib: Any, output_path: Path, overwrite_in_place: bool) -> None:
    if overwrite_in_place:
        with tempfile.NamedTemporaryFile(
            prefix=f"{output_path.stem}_",
            suffix=output_path.suffix,
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
        try:
            lib.write_oas(str(temp_path))
            temp_path.replace(output_path)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
    else:
        lib.write_oas(str(output_path))


def run(config: MarkerConfig) -> dict[str, Any]:
    gdstk = _import_gdstk()
    lib = gdstk.read_oas(str(config.input_path))

    # Make reruns idempotent by clearing the marker layer before analysis.
    for cell in lib.cells:
        cell.filter(
            [(config.marker_layer, config.marker_datatype)],
            remove=True,
            polygons=True,
            paths=True,
            labels=False,
        )

    report: dict[str, Any] = {
        "input_file": str(config.input_path),
        "output_file": str(config.output_path),
        "marker_layer": config.marker_layer,
        "marker_datatype": config.marker_datatype,
        "marker_size_um": config.marker_size_um,
        "markers_per_layer": config.markers_per_layer,
        "style_counts": {},
        "cells": [],
    }

    total_markers = 0
    global_style_counts: dict[str, int] = defaultdict(int)
    for cell in lib.cells:
        cell_bbox = cell.bounding_box()
        if cell_bbox is None:
            continue
        polygons_by_layer: dict[int, list[PolygonRecord]] = defaultdict(list)
        paths_by_layer: dict[int, list[PathRecord]] = defaultdict(list)

        for poly in cell.polygons:
            polygons_by_layer[int(poly.layer)].append(_polygon_record(poly))

        for path in cell.paths:
            for record in _path_records(path):
                paths_by_layer[record.layer].append(record)

        cell_markers: list[MarkerRecord] = []
        cell_entry: dict[str, Any] = {"name": cell.name, "layers": []}

        for layer in sorted(polygons_by_layer):
            if layer == config.marker_layer:
                continue

            selected = _select_markers_for_layer(
                cell_name=cell.name,
                source_layer=layer,
                polygons=polygons_by_layer[layer],
                paths=paths_by_layer.get(layer, []),
                config=config,
                cell_bbox=cell_bbox,
            )
            layer_style_counts: dict[str, int] = defaultdict(int)
            for marker in selected:
                layer_style_counts[marker.hotspot_style] += 1
                global_style_counts[marker.hotspot_style] += 1
            cell_markers.extend(selected)
            total_markers += len(selected)
            cell_entry["layers"].append(
                {
                    "source_layer": layer,
                    "source_polygon_count": len(polygons_by_layer[layer]),
                    "style_counts": dict(sorted(layer_style_counts.items())),
                    "selected_markers": [asdict(marker) for marker in selected],
                }
            )

        _add_marker_rectangles(gdstk, cell, cell_markers, config)
        report["cells"].append(cell_entry)

    payload = report | {
        "total_markers": total_markers,
        "style_counts": dict(sorted(global_style_counts.items())),
    }
    config.report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_library(lib, config.output_path, overwrite_in_place=(config.output_path == config.input_path))
    return payload


def main() -> int:
    config = parse_args()
    try:
        report = run(config)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return 2

    print(f"Updated layout: {config.output_path}")
    print(f"Hotspot markers added: {report['total_markers']}")
    print(f"Report: {config.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
