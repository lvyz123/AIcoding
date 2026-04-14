#!/usr/bin/env python3
"""Simple, parameterized GDSII/OASIS layout generator for chip layout simulation."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DENSITY_PRESETS: dict[str, tuple[int, int]] = {
    "low": (4, 10),
    "medium": (8, 20),
    "high": (40, 80),
}


@dataclass
class GeneratorConfig:
    width_um: float
    height_um: float
    layers: int
    cells: int
    count: int
    start_index: int
    output_format: str
    output_dir: str
    base_name: str
    density: str
    min_shapes_per_layer: int
    max_shapes_per_layer: int
    target_primitives: int | None
    path_ratio: float
    margin_um: float
    seed: int | None


def _import_gdstk() -> Any:
    try:
        import gdstk  # type: ignore

        return gdstk
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency 'gdstk'. Install with: pip install gdstk"
        ) from exc


def _load_json_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}

    config_path = Path(path)
    if not config_path.exists():
        raise ValueError(f"Config file not found: {config_path}")

    try:
        data = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in config file: {config_path}") from exc

    if not isinstance(data, dict):
        raise ValueError("Config JSON must be an object")

    return data


def _build_config(data: dict[str, Any]) -> GeneratorConfig:
    density = str(data.get("density", "medium")).lower()
    if density not in DENSITY_PRESETS:
        raise ValueError("density must be one of: low, medium, high")

    default_min, default_max = DENSITY_PRESETS[density]

    cfg = GeneratorConfig(
        width_um=float(data.get("width", 200.0)),
        height_um=float(data.get("height", 200.0)),
        layers=int(data.get("layers", 4)),
        cells=int(data.get("cells", 1)),
        count=int(data.get("count", 1)),
        start_index=int(data.get("start_index", 0)),
        output_format=str(data.get("format", "gds")),
        output_dir=str(data.get("output_dir", "out")),
        base_name=str(data.get("base_name", "layout")),
        density=density,
        min_shapes_per_layer=int(data.get("min_shapes", default_min)),
        max_shapes_per_layer=int(data.get("max_shapes", default_max)),
        target_primitives=(
            None
            if data.get("target_primitives") is None
            else int(data.get("target_primitives"))
        ),
        path_ratio=float(data.get("path_ratio", 0.2)),
        margin_um=float(data.get("margin", 5.0)),
        seed=(None if data.get("seed") is None else int(data.get("seed"))),
    )
    _validate_config(cfg)
    return cfg


def _validate_config(config: GeneratorConfig) -> None:
    if config.width_um <= 0 or config.height_um <= 0:
        raise ValueError("width and height must be > 0")
    if config.layers <= 0:
        raise ValueError("layers must be > 0")
    if config.cells <= 0:
        raise ValueError("cells must be > 0")
    if config.count <= 0:
        raise ValueError("count must be > 0")
    if config.start_index < 0:
        raise ValueError("start_index must be >= 0")
    if config.min_shapes_per_layer <= 0 or config.max_shapes_per_layer <= 0:
        raise ValueError("min_shapes and max_shapes must be > 0")
    if config.min_shapes_per_layer > config.max_shapes_per_layer:
        raise ValueError("min_shapes cannot be greater than max_shapes")
    if config.target_primitives is not None and config.target_primitives <= 0:
        raise ValueError("target_primitives must be > 0")
    if not 0.0 <= config.path_ratio <= 1.0:
        raise ValueError("path_ratio must be between 0 and 1")
    if config.margin_um < 0:
        raise ValueError("margin must be >= 0")
    if config.margin_um * 2 >= min(config.width_um, config.height_um):
        raise ValueError("margin is too large for the given width/height")
    if config.output_format not in {"gds", "oas"}:
        raise ValueError("format must be one of: gds, oas")


def parse_args() -> GeneratorConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Generate simulated chip layout files in GDSII or OASIS format "
            "based on simple geometric rules."
        )
    )
    parser.add_argument("--config", default=None, help="JSON config file path")
    parser.add_argument("--width", type=float, default=None, help="Layout width in um")
    parser.add_argument("--height", type=float, default=None, help="Layout height in um")
    parser.add_argument("--layers", type=int, default=None, help="Number of layers")
    parser.add_argument("--cells", type=int, default=None, help="Total number of cells in the layout")
    parser.add_argument("--count", type=int, default=None, help="Number of layouts to generate")
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Starting numeric suffix for output files",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=("gds", "oas"),
        default=None,
        help="Output file format: gds or oas",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output folder for generated layout files",
    )
    parser.add_argument(
        "--base-name",
        default=None,
        help="Base file name prefix",
    )
    parser.add_argument(
        "--density",
        choices=("low", "medium", "high"),
        default=None,
        help="Pattern density preset (maps to default min/max shapes per layer)",
    )
    parser.add_argument(
        "--min-shapes",
        type=int,
        default=None,
        help="Minimum shape count per layer",
    )
    parser.add_argument(
        "--max-shapes",
        type=int,
        default=None,
        help="Maximum shape count per layer",
    )
    parser.add_argument(
        "--target-primitives",
        type=int,
        default=None,
        help="Exact total number of polygon+path primitives across all cells/layers",
    )
    parser.add_argument(
        "--path-ratio",
        type=float,
        default=None,
        help="Fraction of primitives generated as path objects (0 to 1)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=None,
        help="Keep-out margin from layout edge in um",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic generation",
    )

    args = parser.parse_args()

    try:
        file_config = _load_json_config(args.config)
        cli_overrides = {
            "width": args.width,
            "height": args.height,
            "layers": args.layers,
            "cells": args.cells,
            "count": args.count,
            "start_index": args.start_index,
            "format": args.output_format,
            "output_dir": args.output_dir,
            "base_name": args.base_name,
            "density": args.density,
            "min_shapes": args.min_shapes,
            "max_shapes": args.max_shapes,
            "target_primitives": args.target_primitives,
            "path_ratio": args.path_ratio,
            "margin": args.margin,
            "seed": args.seed,
        }
        cli_overrides = {k: v for k, v in cli_overrides.items() if v is not None}

        merged = {**file_config, **cli_overrides}
        return _build_config(merged)
    except ValueError as exc:
        parser.error(str(exc))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def _distribute_exact(total: int, buckets: int) -> list[int]:
    if buckets <= 0:
        raise ValueError("buckets must be > 0")

    base, remainder = divmod(total, buckets)
    return [base + (1 if i < remainder else 0) for i in range(buckets)]


def _split_shape_types(total: int, path_ratio: float) -> tuple[int, int]:
    if total <= 0:
        return 0, 0

    path_count = int(round(total * path_ratio))
    path_count = max(0, min(total, path_count))
    if 0.0 < path_ratio < 1.0 and total > 1:
        path_count = max(1, min(total - 1, path_count))

    polygon_count = total - path_count
    return polygon_count, path_count


def _make_polygon_points(cx: float, cy: float, span_x: float, span_y: float, variant: int) -> list[tuple[float, float]]:
    if variant == 0:
        return [
            (cx - span_x, cy - span_y),
            (cx + span_x, cy - span_y),
            (cx + span_x, cy + span_y),
            (cx - span_x, cy + span_y),
        ]

    if variant == 1:
        return [
            (cx, cy - span_y),
            (cx + span_x, cy),
            (cx, cy + span_y),
            (cx - span_x, cy),
        ]

    if variant == 2:
        return [
            (cx - span_x, cy - span_y * 0.8),
            (cx + span_x * 0.6, cy - span_y),
            (cx + span_x, cy + span_y * 0.4),
            (cx - span_x * 0.5, cy + span_y),
        ]

    return [
        (cx - span_x * 0.9, cy),
        (cx - span_x * 0.35, cy - span_y),
        (cx + span_x * 0.35, cy - span_y),
        (cx + span_x * 0.9, cy),
        (cx + span_x * 0.35, cy + span_y),
        (cx - span_x * 0.35, cy + span_y),
    ]


def _resolve_primitive_plan(
    config: GeneratorConfig,
    rng: random.Random,
) -> list[list[int]]:
    if config.target_primitives is not None:
        cell_totals = _distribute_exact(config.target_primitives, config.cells)
        return [_distribute_exact(total, config.layers) for total in cell_totals]

    return [
        [
            rng.randint(config.min_shapes_per_layer, config.max_shapes_per_layer)
            for _ in range(config.layers)
        ]
        for _ in range(config.cells)
    ]


def _add_layer_primitives(
    gdstk: Any,
    cell: Any,
    layer_id: int,
    width: float,
    height: float,
    margin: float,
    primitive_count: int,
    path_ratio: float,
    cell_index: int,
    total_cells: int,
) -> dict[str, int]:
    if primitive_count <= 0:
        return {"primitive_count": 0, "polygon_count": 0, "path_count": 0}

    x0 = margin
    y0 = margin
    x1 = width - margin
    y1 = height - margin
    usable_w = x1 - x0
    usable_h = y1 - y0
    cols = max(1, math.ceil(math.sqrt(primitive_count * usable_w / usable_h)))
    rows = max(1, math.ceil(primitive_count / cols))
    pitch_x = usable_w / cols
    pitch_y = usable_h / rows
    base_pitch = min(pitch_x, pitch_y)
    polygon_count, path_count = _split_shape_types(primitive_count, path_ratio)
    batch: list[Any] = []

    polygon_span_x = max(0.04, min(pitch_x * 0.36, 0.72))
    polygon_span_y = max(0.04, min(pitch_y * 0.36, 0.72))
    path_half_x = max(0.06, min(pitch_x * 0.42, 0.95))
    path_half_y = max(0.06, min(pitch_y * 0.42, 0.95))
    path_width_base = max(0.03, min(base_pitch * 0.18, 0.18))

    for idx in range(primitive_count):
        row, col = divmod(idx, cols)
        cx = x0 + (col + 0.5) * pitch_x
        cy = y0 + (row + 0.5) * pitch_y

        jitter_x = ((((idx * 17 + cell_index * 11 + layer_id * 5) % 19) / 18.0) - 0.5) * pitch_x * 0.24
        jitter_y = ((((idx * 23 + cell_index * 7 + layer_id * 3) % 17) / 16.0) - 0.5) * pitch_y * 0.24
        cx = _clamp(cx + jitter_x, x0 + pitch_x * 0.2, x1 - pitch_x * 0.2)
        cy = _clamp(cy + jitter_y, y0 + pitch_y * 0.2, y1 - pitch_y * 0.2)

        if idx < polygon_count:
            variant = (idx + cell_index + layer_id) % 4
            span_x = min(polygon_span_x * (0.72 + 0.08 * ((idx + layer_id) % 4)), pitch_x * 0.46)
            span_y = min(polygon_span_y * (0.72 + 0.08 * ((idx + cell_index) % 4)), pitch_y * 0.46)
            points = _make_polygon_points(cx, cy, span_x, span_y, variant)
            batch.append(gdstk.Polygon(points, layer=layer_id, datatype=0))
        else:
            path_idx = idx - polygon_count
            orientation = (path_idx + cell_index + layer_id) % 3
            width_scale = 0.78 + 0.06 * (path_idx % 4)
            path_width = min(path_width_base * width_scale, base_pitch * 0.28)

            if orientation == 0:
                half_len_x = min(path_half_x * (0.82 + 0.05 * (path_idx % 5)), pitch_x * 0.48)
                start = (_clamp(cx - half_len_x, x0, x1), cy)
                end = (_clamp(cx + half_len_x, x0, x1), cy)
            elif orientation == 1:
                half_len_y = min(path_half_y * (0.82 + 0.05 * (path_idx % 5)), pitch_y * 0.48)
                start = (cx, _clamp(cy - half_len_y, y0, y1))
                end = (cx, _clamp(cy + half_len_y, y0, y1))
            else:
                half_len_x = min(path_half_x * (0.74 + 0.05 * (path_idx % 5)), pitch_x * 0.4)
                half_len_y = min(path_half_y * (0.74 + 0.05 * (path_idx % 5)), pitch_y * 0.4)
                start = (_clamp(cx - half_len_x, x0, x1), _clamp(cy - half_len_y, y0, y1))
                end = (_clamp(cx + half_len_x, x0, x1), _clamp(cy + half_len_y, y0, y1))

            batch.append(
                gdstk.FlexPath(
                    [start, end],
                    path_width,
                    simple_path=True,
                    layer=layer_id,
                    datatype=1,
                )
            )

        if len(batch) >= 4096:
            cell.add(*batch)
            batch.clear()

    if batch:
        cell.add(*batch)

    return {
        "primitive_count": primitive_count,
        "polygon_count": polygon_count,
        "path_count": path_count,
    }


def generate_layouts(config: GeneratorConfig) -> list[dict[str, Any]]:
    gdstk = _import_gdstk()
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(config.seed)
    summaries: list[dict[str, Any]] = []

    for index in range(config.count):
        lib = gdstk.Library(unit=1e-6, precision=1e-9)
        file_index = config.start_index + index
        top = lib.new_cell(f"TOP_{file_index:03d}")
        cells = [top]

        for child_index in range(1, config.cells):
            child = lib.new_cell(f"CELL_{file_index:03d}_{child_index:02d}")
            top.add(gdstk.Reference(child, (0, 0)))
            cells.append(child)

        cell_layer_plan = _resolve_primitive_plan(config, rng)
        layer_primitive_counts = {str(layer_id): 0 for layer_id in range(1, config.layers + 1)}
        cell_summaries: list[dict[str, Any]] = []
        total_primitives = 0
        total_polygons = 0
        total_paths = 0

        for cell_index, cell in enumerate(cells):
            per_layer_counts = cell_layer_plan[cell_index]
            cell_layer_summary: dict[str, int] = {}
            cell_primitive_count = 0
            cell_polygon_count = 0
            cell_path_count = 0

            for layer_offset, primitive_count in enumerate(per_layer_counts):
                layer_id = layer_offset + 1
                stats = _add_layer_primitives(
                    gdstk=gdstk,
                    cell=cell,
                    layer_id=layer_id,
                    width=config.width_um,
                    height=config.height_um,
                    margin=config.margin_um,
                    primitive_count=primitive_count,
                    path_ratio=config.path_ratio,
                    cell_index=cell_index,
                    total_cells=config.cells,
                )
                cell_layer_summary[str(layer_id)] = stats["primitive_count"]
                layer_primitive_counts[str(layer_id)] += stats["primitive_count"]
                cell_primitive_count += stats["primitive_count"]
                cell_polygon_count += stats["polygon_count"]
                cell_path_count += stats["path_count"]

            total_primitives += cell_primitive_count
            total_polygons += cell_polygon_count
            total_paths += cell_path_count
            cell_summaries.append(
                {
                    "name": cell.name,
                    "primitive_count": cell_primitive_count,
                    "polygon_count": cell_polygon_count,
                    "path_count": cell_path_count,
                    "layer_primitive_counts": cell_layer_summary,
                }
            )

        file_name = f"{config.base_name}_{file_index:03d}.{config.output_format}"
        file_path = out_dir / file_name

        if config.output_format == "gds":
            lib.write_gds(str(file_path))
        else:
            lib.write_oas(str(file_path))

        summaries.append(
            {
                "index": file_index,
                "file": str(file_path),
                "layers": config.layers,
                "cells": config.cells,
                "total_primitives": total_primitives,
                "polygon_count": total_polygons,
                "path_count": total_paths,
                "layer_primitive_counts": layer_primitive_counts,
                "cell_summaries": cell_summaries,
                "width_um": config.width_um,
                "height_um": config.height_um,
            }
        )

    return summaries


def write_report(config: GeneratorConfig, summaries: list[dict[str, Any]]) -> Path:
    report_path = Path(config.output_dir) / f"{config.base_name}_report.json"
    payload = {
        "config": asdict(config),
        "generated_files": len(summaries),
        "layouts": summaries,
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path


def main() -> int:
    config = parse_args()
    try:
        summaries = generate_layouts(config)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    report = write_report(config, summaries)
    print(f"Generated {len(summaries)} layout file(s) in '{config.output_dir}'.")
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
