#!/usr/bin/env python3
"""Generate a DRAM-like 4-layer sample_layout_001 with realistic polygon scale."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DramSample001Config:
    width_um: float
    height_um: float
    tile_size_um: float
    rows: int
    cols: int
    layers: tuple[int, int, int, int]
    output_path: Path
    report_path: Path


def _import_gdstk() -> Any:
    try:
        import gdstk  # type: ignore

        return gdstk
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency 'gdstk'. Install with: pip install gdstk"
        ) from exc


def parse_args() -> DramSample001Config:
    parser = argparse.ArgumentParser(
        description="Generate a DRAM-like multilayer sample_layout_001 in OASIS format."
    )
    parser.add_argument("--width", type=float, default=550.0, help="Layout width in um")
    parser.add_argument("--height", type=float, default=550.0, help="Layout height in um")
    parser.add_argument("--tile-size", type=float, default=5.5, help="Subarray tile size in um")
    parser.add_argument(
        "--output",
        default=r"C:\Users\81932\Documents\AIcoding\layoutgenerator\out_oas\sample_layout_001.oas",
        help="Output OASIS path",
    )
    parser.add_argument(
        "--report",
        default=r"C:\Users\81932\Documents\AIcoding\layoutgenerator\out_oas\sample_layout_001_report.json",
        help="Output JSON report path",
    )
    args = parser.parse_args()

    cols = int(round(args.width / args.tile_size))
    rows = int(round(args.height / args.tile_size))
    cfg = DramSample001Config(
        width_um=float(args.width),
        height_um=float(args.height),
        tile_size_um=float(args.tile_size),
        rows=rows,
        cols=cols,
        layers=(1, 2, 3, 4),
        output_path=Path(args.output).resolve(),
        report_path=Path(args.report).resolve(),
    )
    _validate_config(cfg, parser)
    return cfg


def _validate_config(config: DramSample001Config, parser: argparse.ArgumentParser) -> None:
    if config.width_um <= 0 or config.height_um <= 0 or config.tile_size_um <= 0:
        parser.error("width, height, and tile-size must be > 0")
    if config.rows <= 0 or config.cols <= 0:
        parser.error("rows and cols must be > 0")
    if not math.isclose(config.cols * config.tile_size_um, config.width_um, rel_tol=0, abs_tol=1e-6):
        parser.error("width must be an integer multiple of tile-size")
    if not math.isclose(config.rows * config.tile_size_um, config.height_um, rel_tol=0, abs_tol=1e-6):
        parser.error("height must be an integer multiple of tile-size")


def _add_rect(
    gdstk: Any,
    cell: Any,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    layer: int,
) -> None:
    cell.add(gdstk.rectangle((min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1)), layer=layer, datatype=0))


def _build_tile(gdstk: Any, lib: Any, config: DramSample001Config) -> tuple[Any, dict[str, int]]:
    cell = lib.new_cell("DRAM_ARRAY_TILE_001")
    tile = config.tile_size_um
    layer1, layer2, layer3, layer4 = config.layers
    counts = {str(layer): 0 for layer in config.layers}

    # Layer 1: vertical bitline-like stripes with slight end trimming.
    line_width = 0.11
    x_pitch = 0.38
    x_positions = [0.28 + i * x_pitch for i in range(14)]
    for index, x in enumerate(x_positions):
        bottom = 0.12 if index % 4 == 0 else 0.0
        top = tile - (0.12 if index % 5 == 0 else 0.0)
        _add_rect(gdstk, cell, x - line_width / 2.0, bottom, x + line_width / 2.0, top, layer1)
        counts[str(layer1)] += 1

    # Layer 2: horizontal wordline-like stripes.
    line_height = 0.11
    y_pitch = 0.38
    y_positions = [0.28 + i * y_pitch for i in range(14)]
    for index, y in enumerate(y_positions):
        left = 0.12 if index % 3 == 0 else 0.0
        right = tile - (0.12 if index % 5 == 2 else 0.0)
        _add_rect(gdstk, cell, left, y - line_height / 2.0, right, y + line_height / 2.0, layer2)
        counts[str(layer2)] += 1

    # Layer 3: dense contact/cell landing islands at intersections.
    contact_w = 0.095
    contact_h = 0.085
    contact_x = [0.33 + i * 0.40 for i in range(12)]
    contact_y = [0.33 + i * 0.40 for i in range(12)]
    for y in contact_y:
        for x in contact_x:
            _add_rect(gdstk, cell, x - contact_w / 2.0, y - contact_h / 2.0, x + contact_w / 2.0, y + contact_h / 2.0, layer3)
            counts[str(layer3)] += 1

    # Layer 4: local bridge/jog/endcap features around key intersections.
    bridge_rows = [0.72, 1.52, 2.32, 3.12]
    bridge_cols = [0.55, 1.65, 2.75, 3.85, 4.75]
    for y in bridge_rows:
        for x in bridge_cols:
            _add_rect(gdstk, cell, x, y - 0.045, x + 0.22, y + 0.045, layer4)
            counts[str(layer4)] += 1

    line_end_cols = [0.55, 1.31, 2.07, 2.83, 3.59, 4.35]
    for x in line_end_cols:
        _add_rect(gdstk, cell, x - 0.08, 0.10, x + 0.08, 0.25, layer4)
        _add_rect(gdstk, cell, x - 0.08, tile - 0.25, x + 0.08, tile - 0.10, layer4)
        counts[str(layer4)] += 2

    support_spines = [0.82, 1.62, 2.42, 3.22, 4.02, 4.82, 5.12, 5.32]
    for x in support_spines:
        _add_rect(gdstk, cell, x - 0.03, 2.10, x + 0.03, 2.90, layer4)
        counts[str(layer4)] += 1

    node_pads = [(0.95, 4.55), (1.95, 4.15), (2.95, 4.55), (3.95, 4.15), (4.95, 4.55), (2.75, 1.00)]
    for x, y in node_pads:
        _add_rect(gdstk, cell, x - 0.07, y - 0.07, x + 0.07, y + 0.07, layer4)
        counts[str(layer4)] += 1

    return cell, counts


def _add_top_extras(gdstk: Any, top: Any, config: DramSample001Config) -> dict[str, int]:
    layer4 = config.layers[3]
    counts = {str(layer4): 0}
    width = config.width_um
    half_h = config.height_um / 2.0

    def add_linear_band(count: int, y_center: float, x_margin: float, rect_w: float, rect_h: float) -> None:
        span = width - 2.0 * x_margin
        pitch = span / count
        for idx in range(count):
            cx = x_margin + (idx + 0.5) * pitch
            _add_rect(gdstk, top, cx - rect_w / 2.0, y_center - rect_h / 2.0, cx + rect_w / 2.0, y_center + rect_h / 2.0, layer4)
            counts[str(layer4)] += 1

    # Central sense-amplifier corridor edges.
    add_linear_band(2048, half_h - 0.22, 0.22, 0.08, 0.06)
    add_linear_band(2048, half_h + 0.22, 0.22, 0.08, 0.06)

    # Two subarray block-boundary bands.
    add_linear_band(2018, config.height_um * 0.25, 0.30, 0.08, 0.06)
    add_linear_band(2018, config.height_um * 0.75, 0.30, 0.08, 0.06)

    # Exact boundary frame so the readback bbox stays 550 x 550 um.
    _add_rect(gdstk, top, 0.0, 0.0, width, 0.02, layer4)
    _add_rect(gdstk, top, 0.0, config.height_um - 0.02, width, config.height_um, layer4)
    _add_rect(gdstk, top, 0.0, 0.0, 0.02, config.height_um, layer4)
    _add_rect(gdstk, top, width - 0.02, 0.0, width, config.height_um, layer4)
    counts[str(layer4)] += 4

    return counts


def generate_layout(config: DramSample001Config) -> dict[str, Any]:
    gdstk = _import_gdstk()
    config.output_path.parent.mkdir(parents=True, exist_ok=True)

    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    tile_cell, tile_counts = _build_tile(gdstk, lib, config)
    top = lib.new_cell("TOP_001")

    ref = gdstk.Reference(tile_cell, (0, 0))
    ref.repetition = gdstk.Repetition(columns=config.cols, rows=config.rows, spacing=(config.tile_size_um, config.tile_size_um))
    top.add(ref)

    extra_counts = _add_top_extras(gdstk, top, config)

    lib.write_oas(str(config.output_path))

    readback = gdstk.read_oas(str(config.output_path))
    readback_top = next(cell for cell in readback.cells if cell.name == "TOP_001")
    bbox = readback_top.bounding_box()
    polygon_layer_pairs = sorted(
        {
            (int(poly.layer), int(poly.datatype))
            for cell in readback.cells
            for poly in cell.polygons
        }
    )
    per_layer_counts = {
        str(layer): len(
            readback_top.get_polygons(
                apply_repetitions=True,
                include_paths=True,
                depth=None,
                layer=layer,
                datatype=0,
            )
        )
        for layer in config.layers
    }
    total_polygons = sum(per_layer_counts.values())

    report = {
        "config": {
            **asdict(config),
            "output_path": str(config.output_path),
            "report_path": str(config.report_path),
        },
        "tile_polygon_counts": tile_counts,
        "top_extra_polygon_counts": extra_counts,
        "verification": {
            "polygon_layer_pairs": polygon_layer_pairs,
            "top_bbox_um": (
                None
                if bbox is None
                else {
                    "x0": float(bbox[0][0]),
                    "y0": float(bbox[0][1]),
                    "x1": float(bbox[1][0]),
                    "y1": float(bbox[1][1]),
                    "width": float(bbox[1][0] - bbox[0][0]),
                    "height": float(bbox[1][1] - bbox[0][1]),
                }
            ),
            "per_layer_polygons": per_layer_counts,
            "total_polygons": total_polygons,
        },
    }
    config.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    config = parse_args()
    try:
        report = generate_layout(config)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    print(f"Generated layout: {config.output_path}")
    print(f"Report: {config.report_path}")
    print(
        "Verification: "
        f"total polygons={report['verification']['total_polygons']}, "
        f"per-layer={report['verification']['per_layer_polygons']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
