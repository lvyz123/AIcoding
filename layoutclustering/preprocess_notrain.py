#!/usr/bin/env python3
"""Prepare behavior manifest and aerial NPZ files for the no-training pipeline.

本脚本把已经按 marker clip 预裁剪好的 aerial 图像转换成
`layout_clustering_optimized_notrain.py` 可直接使用的数据目录。它只负责格式转换
和 marker 对齐，不做大图裁剪、坐标配准或光刻仿真。

整体流程:
1. 读取带 marker layer 的 OAS/OASIS，复用 `MarkerRasterBuilder` 生成稳定 marker id。
2. 扫描 aerial 图像目录，按文件名匹配完整 marker id、`marker_000000` 或裸编号。
3. 缺失 aerial 的 marker 直接跳过；重复匹配时按路径字符串排序取第一张。
4. 读取 png/jpg/tiff/bmp/npy/npz/dm3/dm4，统一转成二维 float32 image。
5. 写出 `aerial_npz/<marker_id>.npz`、`behavior.jsonl` 和 `preprocess_summary.json`。

生成的 output directory 可以直接传给 no-train 主脚本的 `--behavior-manifest` 参数。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
import tifffile

from layout_utils import DEFAULT_PIXEL_SIZE_NM, MarkerRasterBuilder, MarkerRecord


SUPPORTED_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy", ".npz", ".dm3", ".dm4")
IMAGE_KEY = "image"


def _json_default(value: Any) -> Any:
    """把 numpy/path 对象转换成 JSON 可序列化类型。"""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _discover_aerial_files(aerial_dir: str | Path, *, recursive: bool) -> List[Path]:
    """扫描 aerial 目录，返回所有支持格式的文件路径。"""
    root = Path(aerial_dir)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"aerial_dir is not a directory: {root}")
    iterator = root.rglob("*") if recursive else root.glob("*")
    files = [path for path in iterator if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES]
    return sorted(files, key=lambda item: str(item).lower())


def _collect_marker_records(input_path: str, marker_layer: str, clip_size_um: float, pixel_size_nm: int) -> List[MarkerRecord]:
    """读取 layout 并按 marker layer 生成 MarkerRecord 列表。"""
    builder = MarkerRasterBuilder(
        config={
            "hotspot_layer": str(marker_layer),
            "clip_size_um": float(clip_size_um),
            "pixel_size_nm": int(pixel_size_nm),
            "apply_layer_operations": False,
        },
        temp_dir=Path("_preprocess_notrain_tmp"),
    )
    records: List[MarkerRecord] = []
    for filepath in builder._discover_input_files(str(input_path)):
        layout_index = builder._prepare_layout(filepath)
        for marker_index, marker_poly in enumerate(layout_index.marker_polygons):
            record = builder._build_marker_record(filepath, marker_index, marker_poly, layout_index)
            if record is not None:
                records.append(record)
    if not records:
        raise ValueError("No markers found on the configured marker layer")
    return records


def _marker_index_token(marker_id: str) -> str:
    """从 marker id 中提取六位编号，例如 `000123`。"""
    match = re.search(r"marker_(\d+)$", str(marker_id))
    if not match:
        raise ValueError(f"Cannot extract marker index from marker_id: {marker_id}")
    return match.group(1)


def _stem_contains_token(stem: str, token: str) -> bool:
    """判断文件 stem 是否包含带弱边界的 token，避免 `000001` 匹配到 `0000012`。"""
    pattern = rf"(?<![A-Za-z0-9]){re.escape(token.lower())}(?![A-Za-z0-9])"
    return re.search(pattern, stem.lower()) is not None


def _match_priority(record: MarkerRecord, image_path: Path) -> int | None:
    """返回 aerial 文件匹配某个 marker 的优先级；None 表示不匹配。"""
    stem = image_path.stem.lower()
    marker_id = str(record.marker_id).lower()
    index_token = _marker_index_token(record.marker_id)
    if _stem_contains_token(stem, marker_id):
        return 0
    if _stem_contains_token(stem, f"marker_{index_token}"):
        return 1
    if _stem_contains_token(stem, index_token):
        return 2
    return None


def _match_aerial_files(records: Sequence[MarkerRecord], aerial_files: Sequence[Path]) -> Tuple[Dict[str, Path], List[str], List[Dict[str, Any]]]:
    """按文件名匹配 marker 到 aerial；缺失跳过，重复取排序后的第一张。"""
    chosen: Dict[str, Path] = {}
    missing: List[str] = []
    duplicates: List[Dict[str, Any]] = []
    for record in records:
        candidates: List[Tuple[int, str, Path]] = []
        for image_path in aerial_files:
            priority = _match_priority(record, image_path)
            if priority is not None:
                candidates.append((priority, str(image_path).lower(), image_path))
        if not candidates:
            missing.append(str(record.marker_id))
            continue
        candidates.sort(key=lambda item: (item[0], item[1]))
        chosen[str(record.marker_id)] = candidates[0][2]
        if len(candidates) > 1:
            duplicates.append(
                {
                    "marker_id": str(record.marker_id),
                    "chosen": str(candidates[0][2]),
                    "candidate_count": int(len(candidates)),
                    "candidates_preview": [str(item[2]) for item in candidates[:5]],
                }
            )
    return chosen, missing, duplicates


def _read_npz_image(path: Path) -> np.ndarray:
    """读取 NPZ 图像，要求包含 key=`image`。"""
    with np.load(str(path), allow_pickle=False) as data:
        if IMAGE_KEY not in data:
            raise ValueError(f"NPZ {path} must contain key '{IMAGE_KEY}'")
        return np.asarray(data[IMAGE_KEY])


def _read_dm_image(path: Path) -> np.ndarray:
    """读取 DM3/DM4 图像；ncempy 是本路径的必需依赖。"""
    from ncempy.io import dm  # type: ignore

    if hasattr(dm, "dmReader"):
        payload = dm.dmReader(str(path))
        if isinstance(payload, dict):
            for key in ("data", "imageData", "array"):
                if key in payload:
                    return np.asarray(payload[key])
    if hasattr(dm, "fileDM"):
        reader = dm.fileDM(str(path))
        try:
            reader.parseHeader()
            dataset = reader.getDataset(0)
            if isinstance(dataset, dict):
                for key in ("data", "imageData", "array"):
                    if key in dataset:
                        return np.asarray(dataset[key])
            return np.asarray(dataset)
        finally:
            close = getattr(reader, "close", None)
            if callable(close):
                close()
    raise RuntimeError(f"Could not read DM image data from {path}")


def _read_image(path: Path) -> np.ndarray:
    """按文件后缀读取 aerial 图像或数组。"""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(str(path), allow_pickle=False))
    if suffix == ".npz":
        return _read_npz_image(path)
    if suffix in (".tif", ".tiff"):
        return np.asarray(tifffile.imread(str(path)))
    if suffix in (".dm3", ".dm4"):
        return _read_dm_image(path)
    with Image.open(path) as image:
        return np.asarray(image)


def _to_grayscale_2d(image: np.ndarray, *, path: Path) -> np.ndarray:
    """把输入图像规范成二维数组，RGB/RGBA 转 luminance，多页图取第一页。"""
    array = np.asarray(image)
    array = np.squeeze(array)
    if array.ndim == 3 and array.shape[-1] in (3, 4):
        rgb = array[..., :3].astype(np.float32)
        return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    if array.ndim == 3:
        array = np.asarray(array[0])
        if array.ndim == 3 and array.shape[-1] in (3, 4):
            rgb = array[..., :3].astype(np.float32)
            return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Image {path} must be convertible to a 2-D aerial image; got shape {array.shape}")
    return array.astype(np.float32, copy=False)


def _normalize_image(image: np.ndarray, *, normalize: bool, path: Path) -> np.ndarray:
    """检查 finite 并按需 min-max normalize 到 [0, 1]。"""
    array = np.asarray(image, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Image {path} contains NaN or Inf")
    if not normalize:
        return np.ascontiguousarray(array, dtype=np.float32)
    min_value = float(np.min(array))
    max_value = float(np.max(array))
    if max_value <= min_value + 1e-12:
        return np.zeros_like(array, dtype=np.float32)
    return np.ascontiguousarray((array - min_value) / (max_value - min_value), dtype=np.float32)


def _load_aerial_image(path: Path, *, normalize: bool) -> np.ndarray:
    """读取并标准化单张 aerial 图像。"""
    raw = _read_image(path)
    gray = _to_grayscale_2d(raw, path=path)
    return _normalize_image(gray, normalize=normalize, path=path)


def _image_shape_summary(shapes: Sequence[Tuple[int, int]]) -> Dict[str, Any]:
    """统计输出 aerial 图像 shape 分布。"""
    counts: Dict[str, int] = {}
    for shape in shapes:
        key = f"{int(shape[0])}x{int(shape[1])}"
        counts[key] = counts.get(key, 0) + 1
    return {
        "unique_shape_count": int(len(counts)),
        "shape_counts": dict(sorted(counts.items())),
    }


def preprocess_notrain(
    *,
    input_path: str,
    marker_layer: str,
    aerial_dir: str,
    output_dir: str,
    clip_size_um: float = 1.35,
    pixel_size_nm: int = DEFAULT_PIXEL_SIZE_NM,
    normalize: bool = True,
    recursive: bool = True,
) -> Dict[str, Any]:
    """执行完整预处理，返回 summary 字典。"""
    output_root = Path(output_dir)
    aerial_npz_dir = output_root / "aerial_npz"
    output_root.mkdir(parents=True, exist_ok=True)
    aerial_npz_dir.mkdir(parents=True, exist_ok=True)

    records = _collect_marker_records(input_path, marker_layer, clip_size_um, pixel_size_nm)
    aerial_files = _discover_aerial_files(aerial_dir, recursive=recursive)
    matched, missing_markers, duplicate_markers = _match_aerial_files(records, aerial_files)

    manifest_path = output_root / "behavior.jsonl"
    shapes: List[Tuple[int, int]] = []
    written = 0
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            image_path = matched.get(str(record.marker_id))
            if image_path is None:
                continue
            image = _load_aerial_image(image_path, normalize=normalize)
            shapes.append((int(image.shape[0]), int(image.shape[1])))
            npz_path = aerial_npz_dir / f"{record.marker_id}.npz"
            np.savez_compressed(npz_path, image=image)
            row = {
                "sample_id": str(record.marker_id),
                "source_path": str(record.source_path),
                "marker_id": str(record.marker_id),
                "clip_bbox": [float(value) for value in record.clip_bbox],
                "aerial_npz": f"aerial_npz/{npz_path.name}",
                "risk_score": 0.0,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    summary = {
        "input_path": str(input_path),
        "marker_layer": str(marker_layer),
        "marker_count": int(len(records)),
        "matched_marker_count": int(len(matched)),
        "skipped_missing_aerial_count": int(len(missing_markers)),
        "duplicate_aerial_marker_count": int(len(duplicate_markers)),
        "written_npz_count": int(written),
        "behavior_manifest": str(manifest_path),
        "aerial_npz_dir": str(aerial_npz_dir),
        "missing_marker_preview": missing_markers[:20],
        "duplicate_marker_preview": duplicate_markers[:20],
        "image_shape_summary": _image_shape_summary(shapes),
        "normalization": "minmax_0_1" if normalize else "none",
        "recursive": bool(recursive),
        "supported_suffixes": list(SUPPORTED_SUFFIXES),
    }
    summary_path = output_root / "preprocess_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=_json_default)
    summary["preprocess_summary"] = str(summary_path)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器，并给出 no-train 直连用法。"""
    epilog = """
示例:

1) 生成 behavior.jsonl 和 aerial NPZ
python preprocess_notrain.py input.oas --marker-layer 999/0 --aerial-dir aerial_images --output-dir notrain_inputs

2) 直接把输出目录传给 no-train 主脚本
python layout_clustering_optimized_notrain.py input.oas --marker-layer 999/0 --behavior-manifest notrain_inputs --output results_notrain.json

注意:
- aerial 图片必须已经按 marker clip 预裁剪好；本脚本不做大图裁剪或坐标配准。
- 缺失 aerial 的 marker 会被跳过。
- 重复匹配同一 marker 时按路径字符串升序取第一张。
- DM3/DM4 需要 ncempy。
"""
    parser = argparse.ArgumentParser(
        description="为 no-training layout clustering 生成 behavior.jsonl 和 aerial_npz",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_path", help="输入 OAS/OASIS 文件或目录")
    parser.add_argument("--marker-layer", required=True, help="marker 层，格式 layer/datatype，例如 999/0")
    parser.add_argument("--aerial-dir", required=True, help="已按 marker 裁剪好的 aerial 图像目录")
    parser.add_argument("--output-dir", required=True, help="输出目录，可直接传给 no-train --behavior-manifest")
    parser.add_argument("--clip-size", type=float, default=1.35, help="marker-centered clip 边长，单位 um")
    parser.add_argument("--pixel-size-nm", type=int, default=DEFAULT_PIXEL_SIZE_NM, help="marker raster pixel size，单位 nm")
    parser.add_argument("--no-normalize", action="store_true", help="关闭默认 min-max normalize")
    parser.add_argument("--no-recursive", action="store_true", help="只扫描 aerial-dir 顶层，不递归子目录")
    return parser


def main() -> int:
    """命令行入口：运行预处理并打印中文摘要。"""
    parser = _build_parser()
    args = parser.parse_args()
    try:
        summary = preprocess_notrain(
            input_path=str(args.input_path),
            marker_layer=str(args.marker_layer),
            aerial_dir=str(args.aerial_dir),
            output_dir=str(args.output_dir),
            clip_size_um=float(args.clip_size),
            pixel_size_nm=int(args.pixel_size_nm),
            normalize=not bool(args.no_normalize),
            recursive=not bool(args.no_recursive),
        )
        print("预处理完成")
        print(f"marker 总数: {summary['marker_count']}")
        print(f"匹配 aerial 的 marker 数: {summary['matched_marker_count']}")
        print(f"跳过缺失 aerial 的 marker 数: {summary['skipped_missing_aerial_count']}")
        print(f"重复匹配 marker 数: {summary['duplicate_aerial_marker_count']}")
        print(f"behavior manifest: {summary['behavior_manifest']}")
        print(f"aerial npz dir: {summary['aerial_npz_dir']}")
        return 0
    except Exception as exc:
        print(f"预处理失败: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
