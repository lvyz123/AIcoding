#!/usr/bin/env python3
"""Generate a DRAM-like M1-only OASIS layout with a dedicated marker layer."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DramM1Config:
    width_um: float
    height_um: float
    tile_size_um: float
    rows: int
    cols: int
    m1_layer: int
    m1_datatype: int
    marker_layer: int
    marker_datatype: int
    marker_size_um: float
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


def parse_args() -> DramM1Config:
    parser = argparse.ArgumentParser(
        description="Generate a DRAM-style M1 layout with a marker layer in OASIS format."
    )
    parser.add_argument(
        "--width",
        type=float,
        default=2000.0,
        help="Layout width in um",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=2000.0,
        help="Layout height in um",
    )
    parser.add_argument(
        "--tile-size",
        type=float,
        default=125.0,
        help="Tile size in um",
    )
    parser.add_argument(
        "--m1-layer",
        type=int,
        default=1,
        help="M1 layer number",
    )
    parser.add_argument(
        "--marker-layer",
        type=int,
        default=999,
        help="Marker layer number",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=0.1,
        help="Marker width/height in um",
    )
    parser.add_argument(
        "--output",
        default=r"C:\Users\81932\Documents\AIcoding\layoutgenerator\out_oas\sample_layout_002.oas",
        help="Output OASIS file path",
    )
    parser.add_argument(
        "--report",
        default=r"C:\Users\81932\Documents\AIcoding\layoutgenerator\out_oas\sample_layout_002_report.json",
        help="Output JSON report path",
    )
    args = parser.parse_args()

    cols = int(round(args.width / args.tile_size))
    rows = int(round(args.height / args.tile_size))
    cfg = DramM1Config(
        width_um=float(args.width),
        height_um=float(args.height),
        tile_size_um=float(args.tile_size),
        rows=rows,
        cols=cols,
        m1_layer=int(args.m1_layer),
        m1_datatype=0,
        marker_layer=int(args.marker_layer),
        marker_datatype=0,
        marker_size_um=float(args.marker_size),
        output_path=Path(args.output).resolve(),
        report_path=Path(args.report).resolve(),
    )
    _validate_config(cfg, parser)
    return cfg


def _validate_config(config: DramM1Config, parser: argparse.ArgumentParser) -> None:
    if config.width_um <= 0 or config.height_um <= 0 or config.tile_size_um <= 0:
        parser.error("width, height, and tile-size must be > 0")
    if config.marker_size_um <= 0:
        parser.error("marker-size must be > 0")
    if config.m1_layer < 0 or config.marker_layer < 0:
        parser.error("layer numbers must be >= 0")
    if config.rows <= 0 or config.cols <= 0:
        parser.error("rows and cols must be > 0")
    if not math.isclose(config.cols * config.tile_size_um, config.width_um, rel_tol=0, abs_tol=1e-6):
        parser.error("width must be an integer multiple of tile-size")
    if not math.isclose(config.rows * config.tile_size_um, config.height_um, rel_tol=0, abs_tol=1e-6):
        parser.error("height must be an integer multiple of tile-size")
    if config.m1_layer == config.marker_layer:
        parser.error("m1-layer and marker-layer must be different")


def _add_rect(
    gdstk: Any,
    cell: Any,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    layer: int,
    datatype: int,
) -> None:
    cell.add(gdstk.rectangle((min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1)), layer=layer, datatype=datatype))


def _dedupe_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for x, y in points:
        key = (round(x, 6), round(y, 6))
        if key not in seen:
            seen.add(key)
            unique.append((x, y))
    return unique


def _take_evenly(points: list[tuple[float, float]], count: int) -> list[tuple[float, float]]:
    unique = _dedupe_points(points)
    if count <= 0 or not unique:
        return []
    if len(unique) <= count:
        return unique[:]

    step = len(unique) / count
    selected: list[tuple[float, float]] = []
    used_indices: set[int] = set()
    for idx in range(count):
        pick = min(len(unique) - 1, int((idx + 0.5) * step))
        while pick in used_indices and pick + 1 < len(unique):
            pick += 1
        while pick in used_indices and pick - 1 >= 0:
            pick -= 1
        if pick in used_indices:
            continue
        used_indices.add(pick)
        selected.append(unique[pick])
    return selected


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def _variant_name(variant: int) -> str:
    return f"DRAM_M1_TILE_{variant:02d}"


def _build_tile(gdstk: Any, lib: Any, config: DramM1Config, variant: int) -> tuple[Any, dict[str, Any]]:
    cell = lib.new_cell(_variant_name(variant))
    tile = config.tile_size_um
    margin = 4.0
    line_width = 0.22
    track_pitch = 0.82
    available = tile - 2.0 * margin - line_width
    track_count = int(math.floor(available / track_pitch)) + 1
    xs = [margin + line_width / 2.0 + i * track_pitch for i in range(track_count)]

    pad_ys = [
        23.0 + 0.6 * (variant % 2),
        61.5 - 0.4 * (variant // 2),
        101.0 + 0.5 * ((variant + 1) % 2),
    ]
    strap_ys = [
        34.0 + 0.5 * variant,
        74.0 - 0.3 * (variant % 2),
        111.0 + 0.4 * (variant // 2),
    ]
    dense_ys = [
        46.0 + 0.8 * variant,
        88.5 - 0.5 * variant,
        116.0 - 0.6 * (variant % 2),
    ]

    m1_polygons = 0
    line_end_candidates: list[tuple[float, float]] = []
    pad_candidates: list[tuple[float, float]] = []
    strap_candidates: list[tuple[float, float]] = []
    bridge_candidates: list[tuple[float, float]] = []
    dense_candidates: list[tuple[float, float]] = []

    for track_index, x in enumerate(xs):
        bottom_trim = 0.0
        top_trim = 0.0

        if (track_index + variant) % 18 in {2, 11}:
            bottom_trim = 5.4 + 0.2 * (variant % 2)
            line_end_candidates.append((x, bottom_trim + 0.45))
        if (track_index + variant) % 18 in {6, 15}:
            top_trim = 5.4 + 0.25 * (variant // 2)
            line_end_candidates.append((x, tile - top_trim - 0.45))

        _add_rect(
            gdstk,
            cell,
            x - line_width / 2.0,
            bottom_trim,
            x + line_width / 2.0,
            tile - top_trim,
            config.m1_layer,
            config.m1_datatype,
        )
        m1_polygons += 1

        if track_index % 6 in {1, 4}:
            for band_index, center_y in enumerate(pad_ys):
                pad_width = 0.38 + 0.04 * ((track_index + band_index + variant) % 3)
                pad_height = 0.72 + 0.06 * ((band_index + variant) % 2)
                _add_rect(
                    gdstk,
                    cell,
                    x - pad_width / 2.0,
                    center_y - pad_height / 2.0,
                    x + pad_width / 2.0,
                    center_y + pad_height / 2.0,
                    config.m1_layer,
                    config.m1_datatype,
                )
                m1_polygons += 1

                if track_index % 24 in {(4 + variant) % 24, (16 + variant) % 24}:
                    x_offset = 0.14 if band_index % 2 == 0 else -0.14
                    pad_candidates.append((x + x_offset, center_y))

    for group_index, start_track in enumerate(range(0, track_count - 7, 12)):
        left_index = start_track + 1
        span_tracks = 4 + ((group_index + variant) % 3)
        right_index = min(track_count - 1, left_index + span_tracks)
        left_x = xs[left_index] - 0.18
        right_x = xs[right_index] + 0.18

        for band_index, center_y in enumerate(strap_ys):
            if (group_index + band_index + variant) % 2 != 0:
                continue

            strap_height = 0.16 + 0.02 * ((band_index + variant) % 2)
            dogbone_width = 0.28
            dogbone_height = 0.30
            _add_rect(
                gdstk,
                cell,
                left_x,
                center_y - strap_height / 2.0,
                right_x,
                center_y + strap_height / 2.0,
                config.m1_layer,
                config.m1_datatype,
            )
            _add_rect(
                gdstk,
                cell,
                left_x - dogbone_width / 2.0,
                center_y - dogbone_height / 2.0,
                left_x + dogbone_width / 2.0,
                center_y + dogbone_height / 2.0,
                config.m1_layer,
                config.m1_datatype,
            )
            _add_rect(
                gdstk,
                cell,
                right_x - dogbone_width / 2.0,
                center_y - dogbone_height / 2.0,
                right_x + dogbone_width / 2.0,
                center_y + dogbone_height / 2.0,
                config.m1_layer,
                config.m1_datatype,
            )
            m1_polygons += 3
            strap_candidates.extend([(left_x, center_y), (right_x, center_y)])
            bridge_candidates.extend(
                [
                    (left_x + 0.18, center_y + strap_height),
                    (left_x + 0.18, center_y - strap_height),
                    (right_x - 0.18, center_y + strap_height),
                    (right_x - 0.18, center_y - strap_height),
                ]
            )

    for group_index, start_track in enumerate(range(5 + variant, track_count - 4, 18)):
        center_x = 0.5 * (xs[start_track] + xs[min(start_track + 3, track_count - 1)])
        center_y = dense_ys[group_index % len(dense_ys)]
        dense_candidates.append((center_x, center_y))
        if group_index % 2 == 0:
            dense_candidates.append((center_x + 0.38, center_y - 12.0))
        else:
            dense_candidates.append((center_x - 0.38, center_y + 9.0))

    marker_targets = {
        "line_end": 20,
        "bridge_edge": 12,
        "strap_edge": 8,
        "pad_transition": 4,
        "dense_cluster": 4,
    }
    marker_sources = {
        "line_end": line_end_candidates,
        "bridge_edge": bridge_candidates,
        "pad_transition": pad_candidates,
        "strap_edge": strap_candidates,
        "dense_cluster": dense_candidates,
    }

    selected_markers: list[tuple[str, tuple[float, float]]] = []
    used_marker_keys: set[tuple[float, float]] = set()
    for label, count in marker_targets.items():
        for point in _take_evenly(marker_sources[label], count):
            key = (round(point[0], 6), round(point[1], 6))
            if key in used_marker_keys:
                continue
            used_marker_keys.add(key)
            selected_markers.append((label, point))

    marker_goal = sum(marker_targets.values())
    if len(selected_markers) < marker_goal:
        fallback_pool = []
        for source_points in marker_sources.values():
            fallback_pool.extend(source_points)
        for point in _take_evenly(fallback_pool, marker_goal * 2):
            key = (round(point[0], 6), round(point[1], 6))
            if key in used_marker_keys:
                continue
            used_marker_keys.add(key)
            selected_markers.append(("fallback", point))
            if len(selected_markers) >= marker_goal:
                break

    half = config.marker_size_um / 2.0
    marker_breakdown: dict[str, int] = {}
    for label, (x, y) in selected_markers[:marker_goal]:
        x = _clip(x, half, tile - half)
        y = _clip(y, half, tile - half)
        marker_breakdown[label] = marker_breakdown.get(label, 0) + 1
        _add_rect(
            gdstk,
            cell,
            x - half,
            y - half,
            x + half,
            y + half,
            config.marker_layer,
            config.marker_datatype,
        )

    return cell, {
        "track_count": track_count,
        "m1_polygons": m1_polygons,
        "marker_polygons": min(marker_goal, len(selected_markers)),
        "marker_breakdown": marker_breakdown,
    }


def _add_periphery(gdstk: Any, top: Any, config: DramM1Config) -> int:
    m1 = config.m1_layer
    dt = config.m1_datatype
    width = config.width_um
    height = config.height_um
    count = 0

    frame = 2.2
    _add_rect(gdstk, top, 0.0, 0.0, width, frame, m1, dt)
    _add_rect(gdstk, top, 0.0, height - frame, width, height, m1, dt)
    _add_rect(gdstk, top, 0.0, 0.0, frame, height, m1, dt)
    _add_rect(gdstk, top, width - frame, 0.0, width, height, m1, dt)
    count += 4

    _add_rect(gdstk, top, 70.0, height / 2.0 - 3.0, width - 70.0, height / 2.0 + 3.0, m1, dt)
    _add_rect(gdstk, top, 18.0, 18.0, 30.0, height - 18.0, m1, dt)
    _add_rect(gdstk, top, width - 30.0, 18.0, width - 18.0, height - 18.0, m1, dt)
    count += 3
    return count


def _corridor_points(
    config: DramM1Config,
    y_offsets: list[float],
    x_pitch_um: float,
    x_margin_um: float = 72.0,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    x = x_margin_um
    while x <= config.width_um - x_margin_um:
        for offset in y_offsets:
            points.append((x, config.height_um / 2.0 + offset))
        x += x_pitch_um
    return points


def _add_markers_from_points(
    gdstk: Any,
    cell: Any,
    config: DramM1Config,
    label: str,
    points: list[tuple[float, float]],
    count: int,
) -> dict[str, Any]:
    half = config.marker_size_um / 2.0
    selected = _take_evenly(points, count)
    for x, y in selected:
        x = _clip(x, half, config.width_um - half)
        y = _clip(y, half, config.height_um - half)
        _add_rect(
            gdstk,
            cell,
            x - half,
            y - half,
            x + half,
            y + half,
            config.marker_layer,
            config.marker_datatype,
        )
    return {"style": label, "count": len(selected)}


def _add_sense_amp_corridor_markers(gdstk: Any, top: Any, config: DramM1Config) -> dict[str, Any]:
    """Place DRAM-specific markers around the central sense-amplifier corridor."""
    targets = {
        "sense_amp_corridor_edge": 2048,
        "sense_amp_track_crossing": 1024,
        "sense_amp_strap_neck": 512,
        "sense_amp_boundary_feed": 512,
    }
    sources = {
        "sense_amp_corridor_edge": _corridor_points(config, [-3.18, 3.18], 0.82),
        "sense_amp_track_crossing": _corridor_points(config, [-1.8, -0.72, 0.72, 1.8], 1.64),
        "sense_amp_strap_neck": _corridor_points(config, [-2.55, 2.55], 3.28),
        "sense_amp_boundary_feed": _corridor_points(config, [-3.85, 3.85], 4.10),
    }

    breakdown: dict[str, int] = {}
    for label, target_count in targets.items():
        stats = _add_markers_from_points(
            gdstk,
            top,
            config,
            label,
            sources[label],
            target_count,
        )
        breakdown[label] = stats["count"]

    return {"marker_polygons": sum(breakdown.values()), "marker_breakdown": breakdown}


def generate_layout(config: DramM1Config) -> dict[str, Any]:
    gdstk = _import_gdstk()
    config.output_path.parent.mkdir(parents=True, exist_ok=True)

    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    top = lib.new_cell("TOP_002")
    variant_cells: list[Any] = []
    variant_stats: dict[str, dict[str, Any]] = {}

    for variant in range(4):
        cell, stats = _build_tile(gdstk, lib, config, variant)
        variant_cells.append(cell)
        variant_stats[cell.name] = stats

    variant_usage = {cell.name: 0 for cell in variant_cells}
    for row in range(config.rows):
        for col in range(config.cols):
            variant = ((row & 1) << 1) | (col & 1)
            cell = variant_cells[variant]
            top.add(gdstk.Reference(cell, (col * config.tile_size_um, row * config.tile_size_um)))
            variant_usage[cell.name] += 1

    top_level_m1_polygons = _add_periphery(gdstk, top, config)
    corridor_marker_stats = _add_sense_amp_corridor_markers(gdstk, top, config)

    lib.write_oas(str(config.output_path))

    readback = gdstk.read_oas(str(config.output_path))
    readback_top = next(cell for cell in readback.cells if cell.name == "TOP_002")
    bbox = readback_top.bounding_box()
    m1_count = len(
        readback_top.get_polygons(
            apply_repetitions=True,
            include_paths=True,
            depth=None,
            layer=config.m1_layer,
            datatype=config.m1_datatype,
        )
    )
    marker_count = len(
        readback_top.get_polygons(
            apply_repetitions=True,
            include_paths=True,
            depth=None,
            layer=config.marker_layer,
            datatype=config.marker_datatype,
        )
    )
    layer_pairs = sorted(
        {
            (int(poly.layer), int(poly.datatype))
            for cell in readback.cells
            for poly in cell.polygons
        }
    )

    report = {
        "config": {
            **asdict(config),
            "output_path": str(config.output_path),
            "report_path": str(config.report_path),
        },
        "tile_variants": variant_stats,
        "variant_usage": variant_usage,
        "top_level_m1_polygons": top_level_m1_polygons,
        "top_level_marker_polygons": corridor_marker_stats["marker_polygons"],
        "top_level_marker_breakdown": corridor_marker_stats["marker_breakdown"],
        "verification": {
            "polygon_layer_pairs": layer_pairs,
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
            "instanced_m1_polygons": m1_count,
            "instanced_marker_polygons": marker_count,
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
        f"M1 polygons={report['verification']['instanced_m1_polygons']}, "
        f"marker polygons={report['verification']['instanced_marker_polygons']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
