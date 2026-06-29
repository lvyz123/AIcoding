#!/usr/bin/env python3
"""生成仿 DRAM 产品 M0 层的 1mm x 1mm OASIS 版图。

版本：simulated_m0_v1

整体算法流程与原理：
1. 使用 um 作为版图坐标单位，默认输出 1000um x 1000um，对应约 1mm x 1mm。
2. 将版图划分为上下两个存储阵列区，中间保留 sense-amplifier/local-interconnect 走廊。
3. 阵列区用层级化 tile 重复表达 M0 局部互连：密集纵向短线、landing pad、短桥接和端部修整。
4. 顶层补充子阵列边界、中心感放走廊、左右 row/column decoder 连接和外围边界框。
5. 只写入 M0 一个版图层，默认 layer/datatype 为 0/0；JSON 报告用于检查尺寸、层号和多边形数量。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(slots=True)
class DramM0Config:
    """保存 M0 版图生成参数，所有尺寸单位均为 um。"""

    width_um: float
    height_um: float
    tile_size_um: float
    array_margin_um: float
    sense_corridor_um: float
    half_array_rows: int
    array_cols: int
    m0_layer: int
    m0_datatype: int
    output_path: Path
    report_path: Path


def _import_gdstk() -> Any:
    """导入 gdstk，并在缺少依赖时给出可执行的安装提示。"""

    try:
        import gdstk  # type: ignore

        return gdstk
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("缺少依赖 gdstk，请先执行：pip install gdstk") from exc


def parse_args() -> DramM0Config:
    """解析命令行参数，并补齐阵列行列数等派生配置。"""

    parser = argparse.ArgumentParser(
        description="生成仿 DRAM 产品 M0 层的 1mm x 1mm OASIS 版图。"
    )
    parser.add_argument("--width", type=float, default=1000.0, help="版图宽度，单位 um")
    parser.add_argument("--height", type=float, default=1000.0, help="版图高度，单位 um")
    parser.add_argument("--tile-size", type=float, default=10.0, help="阵列 tile 尺寸，单位 um")
    parser.add_argument("--array-margin", type=float, default=80.0, help="阵列到版图边界的留白，单位 um")
    parser.add_argument("--sense-corridor", type=float, default=20.0, help="中心感放走廊高度，单位 um")
    parser.add_argument("--m0-layer", type=int, default=0, help="M0 层号")
    parser.add_argument("--m0-datatype", type=int, default=0, help="M0 datatype")
    parser.add_argument(
        "--output",
        default=str(SCRIPT_DIR / "out_oas" / "simulated_m0.oas"),
        help="输出 OASIS 文件路径",
    )
    parser.add_argument(
        "--report",
        default=str(SCRIPT_DIR / "out_oas" / "simulated_m0_report.json"),
        help="输出 JSON 检查报告路径",
    )
    args = parser.parse_args()

    array_width = args.width - 2.0 * args.array_margin
    half_array_height = (args.height - 2.0 * args.array_margin - args.sense_corridor) / 2.0
    cfg = DramM0Config(
        width_um=float(args.width),
        height_um=float(args.height),
        tile_size_um=float(args.tile_size),
        array_margin_um=float(args.array_margin),
        sense_corridor_um=float(args.sense_corridor),
        half_array_rows=int(round(half_array_height / args.tile_size)),
        array_cols=int(round(array_width / args.tile_size)),
        m0_layer=int(args.m0_layer),
        m0_datatype=int(args.m0_datatype),
        output_path=Path(args.output).resolve(),
        report_path=Path(args.report).resolve(),
    )
    _validate_config(cfg, parser)
    return cfg


def _validate_config(config: DramM0Config, parser: argparse.ArgumentParser) -> None:
    """校验默认和用户输入尺寸，保证阵列可被 tile 整齐铺满。"""

    if config.width_um <= 0 or config.height_um <= 0:
        parser.error("width 和 height 必须大于 0")
    if config.tile_size_um <= 0:
        parser.error("tile-size 必须大于 0")
    if config.array_margin_um < 0 or config.sense_corridor_um <= 0:
        parser.error("array-margin 必须不小于 0，sense-corridor 必须大于 0")
    if config.m0_layer < 0 or config.m0_datatype < 0:
        parser.error("M0 layer/datatype 必须不小于 0")

    array_width = config.width_um - 2.0 * config.array_margin_um
    half_array_height = (config.height_um - 2.0 * config.array_margin_um - config.sense_corridor_um) / 2.0
    if array_width <= 0 or half_array_height <= 0:
        parser.error("阵列区尺寸必须为正，请减小 array-margin 或 sense-corridor")
    if config.array_cols <= 0 or config.half_array_rows <= 0:
        parser.error("阵列行列数必须为正")
    if not math.isclose(config.array_cols * config.tile_size_um, array_width, rel_tol=0, abs_tol=1e-6):
        parser.error("width - 2 * array-margin 必须是 tile-size 的整数倍")
    if not math.isclose(config.half_array_rows * config.tile_size_um, half_array_height, rel_tol=0, abs_tol=1e-6):
        parser.error("(height - 2 * array-margin - sense-corridor) / 2 必须是 tile-size 的整数倍")


def _add_rect(
    gdstk: Any,
    cell: Any,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    config: DramM0Config,
) -> None:
    """向 cell 中加入一个 M0 矩形，并自动修正坐标顺序。"""

    cell.add(
        gdstk.rectangle(
            (min(x0, x1), min(y0, y1)),
            (max(x0, x1), max(y0, y1)),
            layer=config.m0_layer,
            datatype=config.m0_datatype,
        )
    )


def _tile_name(variant: int) -> str:
    """根据奇偶变体编号返回稳定的 tile cell 名称。"""

    return f"DRAM_M0_TILE_{variant:02d}"


def _build_m0_tile(gdstk: Any, lib: Any, config: DramM0Config, variant: int) -> tuple[Any, dict[str, int]]:
    """构造一个 M0 阵列 tile，表达局部互连线、landing pad 与短桥接。"""

    cell = lib.new_cell(_tile_name(variant))
    tile = config.tile_size_um
    counts = {"track": 0, "landing_pad": 0, "bridge": 0, "line_end": 0}

    track_width = 0.07
    track_pitch = 0.34
    x = 0.22 + 0.02 * (variant & 1)
    track_index = 0
    track_centers: list[float] = []
    while x <= tile - 0.22:
        bottom_trim = 0.12 if (track_index + variant) % 9 == 0 else 0.0
        top_trim = 0.14 if (track_index + 2 * variant) % 11 == 3 else 0.0
        _add_rect(
            gdstk,
            cell,
            x - track_width / 2.0,
            bottom_trim,
            x + track_width / 2.0,
            tile - top_trim,
            config,
        )
        counts["track"] += 1
        track_centers.append(x)

        if bottom_trim > 0:
            _add_rect(gdstk, cell, x - 0.09, 0.04, x + 0.09, 0.22, config)
            counts["line_end"] += 1
        if top_trim > 0:
            _add_rect(gdstk, cell, x - 0.09, tile - 0.24, x + 0.09, tile - 0.05, config)
            counts["line_end"] += 1

        if (track_index + variant) % 4 == 1:
            for pad_y in (1.55, 4.95 + 0.05 * (variant >> 1), 8.35):
                _add_rect(gdstk, cell, x - 0.095, pad_y - 0.13, x + 0.095, pad_y + 0.13, config)
                counts["landing_pad"] += 1

        x += track_pitch
        track_index += 1

    for group_index in range(0, len(track_centers) - 4, 6):
        left = track_centers[group_index + 1] - 0.04
        right = track_centers[group_index + 4] + 0.04
        y = 2.35 + 2.05 * ((group_index // 6 + variant) % 3)
        _add_rect(gdstk, cell, left, y - 0.045, right, y + 0.045, config)
        _add_rect(gdstk, cell, left - 0.06, y - 0.09, left + 0.08, y + 0.09, config)
        _add_rect(gdstk, cell, right - 0.08, y - 0.09, right + 0.06, y + 0.09, config)
        counts["bridge"] += 3

    return cell, counts


def _add_variant_repetition(
    gdstk: Any,
    top: Any,
    cell: Any,
    origin_x: float,
    origin_y: float,
    columns: int,
    rows: int,
    config: DramM0Config,
) -> int:
    """用二维 repetition 放置同一种奇偶 tile，并返回实际实例数量。"""

    if columns <= 0 or rows <= 0:
        return 0
    ref = gdstk.Reference(cell, (origin_x, origin_y))
    ref.repetition = gdstk.Repetition(
        columns=columns,
        rows=rows,
        spacing=(2.0 * config.tile_size_um, 2.0 * config.tile_size_um),
    )
    top.add(ref)
    return columns * rows


def _add_array_tiles(gdstk: Any, top: Any, tile_cells: list[Any], config: DramM0Config) -> dict[str, int]:
    """在上下两个阵列区铺放四种奇偶 tile，减少文件体积并保留阵列规律。"""

    usage = {cell.name: 0 for cell in tile_cells}
    lower_origin_y = config.array_margin_um
    upper_origin_y = config.array_margin_um + config.half_array_rows * config.tile_size_um + config.sense_corridor_um

    for base_y in (lower_origin_y, upper_origin_y):
        for row_parity in range(2):
            for col_parity in range(2):
                variant = (row_parity << 1) | col_parity
                columns = (config.array_cols - col_parity + 1) // 2
                rows = (config.half_array_rows - row_parity + 1) // 2
                count = _add_variant_repetition(
                    gdstk,
                    top,
                    tile_cells[variant],
                    config.array_margin_um + col_parity * config.tile_size_um,
                    base_y + row_parity * config.tile_size_um,
                    columns,
                    rows,
                    config,
                )
                usage[tile_cells[variant].name] += count

    return usage


def _add_sense_corridor(gdstk: Any, top: Any, config: DramM0Config) -> dict[str, int]:
    """添加中心感放走廊的 M0 边界轨、短连接和密集 landing pad。"""

    counts = {"corridor_rail": 0, "feed_pad": 0, "local_bridge": 0}
    x0 = config.array_margin_um
    x1 = config.width_um - config.array_margin_um
    y0 = config.array_margin_um + config.half_array_rows * config.tile_size_um
    y1 = y0 + config.sense_corridor_um

    for y in (y0 + 0.35, y0 + 2.1, y1 - 2.1, y1 - 0.35):
        _add_rect(gdstk, top, x0, y - 0.08, x1, y + 0.08, config)
        counts["corridor_rail"] += 1

    x = x0 + 0.34
    while x < x1:
        _add_rect(gdstk, top, x - 0.055, y0 - 0.50, x + 0.055, y0 + 0.50, config)
        _add_rect(gdstk, top, x - 0.055, y1 - 0.50, x + 0.055, y1 + 0.50, config)
        counts["feed_pad"] += 2
        x += 0.68

    bridge_pitch = 6.8
    x = x0 + 1.7
    while x < x1 - 1.7:
        _add_rect(gdstk, top, x - 1.05, y0 + 8.9, x + 1.05, y0 + 9.08, config)
        _add_rect(gdstk, top, x - 0.10, y0 + 7.8, x + 0.10, y0 + 10.2, config)
        counts["local_bridge"] += 2
        x += bridge_pitch

    return counts


def _add_subarray_boundaries(gdstk: Any, top: Any, config: DramM0Config) -> dict[str, int]:
    """添加子阵列边界处的稀疏 M0 拼接轨和边界 tap。"""

    counts = {"vertical_boundary": 0, "horizontal_boundary": 0, "boundary_tap": 0}
    x0 = config.array_margin_um
    x1 = config.width_um - config.array_margin_um
    lower_y0 = config.array_margin_um
    lower_y1 = lower_y0 + config.half_array_rows * config.tile_size_um
    upper_y0 = lower_y1 + config.sense_corridor_um
    upper_y1 = upper_y0 + config.half_array_rows * config.tile_size_um
    block_pitch = 10.0 * config.tile_size_um

    x = x0 + block_pitch
    while x < x1 - 1e-9:
        for y_start, y_stop in ((lower_y0, lower_y1), (upper_y0, upper_y1)):
            _add_rect(gdstk, top, x - 0.16, y_start, x + 0.16, y_stop, config)
            counts["vertical_boundary"] += 1
            y = y_start + 2.0
            while y < y_stop:
                _add_rect(gdstk, top, x - 0.34, y - 0.11, x + 0.34, y + 0.11, config)
                counts["boundary_tap"] += 1
                y += 20.0
        x += block_pitch

    for y_start, y_stop in ((lower_y0, lower_y1), (upper_y0, upper_y1)):
        y = y_start + block_pitch
        while y < y_stop - 1e-9:
            _add_rect(gdstk, top, x0, y - 0.14, x1, y + 0.14, config)
            counts["horizontal_boundary"] += 1
            y += block_pitch

    return counts


def _add_periphery(gdstk: Any, top: Any, config: DramM0Config) -> dict[str, int]:
    """添加外围 row/column decoder 局部互连、边界框和角落连接结构。"""

    counts = {"edge_frame": 0, "decoder_rail": 0, "corner_mesh": 0}
    w = config.width_um
    h = config.height_um
    m = config.array_margin_um

    frame = 0.20
    _add_rect(gdstk, top, 0.0, 0.0, w, frame, config)
    _add_rect(gdstk, top, 0.0, h - frame, w, h, config)
    _add_rect(gdstk, top, 0.0, 0.0, frame, h, config)
    _add_rect(gdstk, top, w - frame, 0.0, w, h, config)
    counts["edge_frame"] += 4

    for x in (22.0, 36.0, 50.0, w - 50.0, w - 36.0, w - 22.0):
        _add_rect(gdstk, top, x - 0.22, 24.0, x + 0.22, h - 24.0, config)
        counts["decoder_rail"] += 1

    y = 34.0
    while y <= h - 34.0:
        _add_rect(gdstk, top, 16.0, y - 0.14, m - 8.0, y + 0.14, config)
        _add_rect(gdstk, top, w - m + 8.0, y - 0.14, w - 16.0, y + 0.14, config)
        counts["decoder_rail"] += 2
        y += 18.0

    x = m
    while x <= w - m:
        _add_rect(gdstk, top, x - 0.13, 18.0, x + 0.13, m - 14.0, config)
        _add_rect(gdstk, top, x - 0.13, h - m + 14.0, x + 0.13, h - 18.0, config)
        counts["decoder_rail"] += 2
        x += 12.0

    for cx in (36.0, w - 36.0):
        for cy in (36.0, h - 36.0):
            for offset in (-8.0, 0.0, 8.0):
                _add_rect(gdstk, top, cx - 10.0, cy + offset - 0.12, cx + 10.0, cy + offset + 0.12, config)
                _add_rect(gdstk, top, cx + offset - 0.12, cy - 10.0, cx + offset + 0.12, cy + 10.0, config)
                counts["corner_mesh"] += 2

    return counts


def _readback_report(gdstk: Any, config: DramM0Config) -> dict[str, Any]:
    """重新读取 OAS 文件，检查顶层 bbox、M0 层号和展开后的多边形数量。"""

    readback = gdstk.read_oas(str(config.output_path))
    top = next(cell for cell in readback.cells if cell.name == "SIMULATED_DRAM_M0")
    bbox = top.bounding_box()
    layer_pairs = sorted({(int(poly.layer), int(poly.datatype)) for cell in readback.cells for poly in cell.polygons})
    m0_polygons = len(
        top.get_polygons(
            apply_repetitions=True,
            include_paths=True,
            depth=None,
            layer=config.m0_layer,
            datatype=config.m0_datatype,
        )
    )
    return {
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
        "instanced_m0_polygons": m0_polygons,
    }


def generate_layout(config: DramM0Config) -> dict[str, Any]:
    """生成 M0 OAS 文件，写出报告，并返回报告内容。"""

    gdstk = _import_gdstk()
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.parent.mkdir(parents=True, exist_ok=True)

    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    top = lib.new_cell("SIMULATED_DRAM_M0")
    tile_cells: list[Any] = []
    tile_stats: dict[str, dict[str, int]] = {}

    for variant in range(4):
        tile_cell, stats = _build_m0_tile(gdstk, lib, config, variant)
        tile_cells.append(tile_cell)
        tile_stats[tile_cell.name] = stats

    variant_usage = _add_array_tiles(gdstk, top, tile_cells, config)
    top_counts = {
        "sense_corridor": _add_sense_corridor(gdstk, top, config),
        "subarray_boundaries": _add_subarray_boundaries(gdstk, top, config),
        "periphery": _add_periphery(gdstk, top, config),
    }

    lib.write_oas(str(config.output_path))
    verification = _readback_report(gdstk, config)

    report = {
        "version": "simulated_m0_v1",
        "config": {
            **asdict(config),
            "output_path": str(config.output_path),
            "report_path": str(config.report_path),
        },
        "array": {
            "array_cols": config.array_cols,
            "half_array_rows": config.half_array_rows,
            "total_array_tile_instances": sum(variant_usage.values()),
            "lower_array_origin_um": {"x": config.array_margin_um, "y": config.array_margin_um},
            "upper_array_origin_um": {
                "x": config.array_margin_um,
                "y": config.array_margin_um + config.half_array_rows * config.tile_size_um + config.sense_corridor_um,
            },
        },
        "tile_polygon_counts": tile_stats,
        "variant_usage": variant_usage,
        "top_feature_counts": top_counts,
        "verification": verification,
    }
    config.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    """命令行入口。

    使用说明：
    - 默认直接生成 layoutgenerator/out_oas/simulated_m0.oas。
    - 默认 M0 层号为 0/0；如果下游工具需要其他层号，可用 --m0-layer 和 --m0-datatype 覆盖。
    - 该版图用于仿真和算法测试，不代表任何真实 DRAM 产品或 PDK 的 DRC/LVS 合法结果。
    """

    config = parse_args()
    try:
        report = generate_layout(config)
    except RuntimeError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2

    bbox = report["verification"]["top_bbox_um"]
    print(f"已生成版图：{config.output_path}")
    print(f"检查报告：{config.report_path}")
    print(
        "检查结果："
        f"bbox={bbox['width']:.3f}um x {bbox['height']:.3f}um，"
        f"M0 多边形={report['verification']['instanced_m0_polygons']}，"
        f"层号={report['verification']['polygon_layer_pairs']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
