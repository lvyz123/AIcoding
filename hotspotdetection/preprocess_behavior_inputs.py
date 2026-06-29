#!/usr/bin/env python3
"""把预裁剪 CD-SEM 图像整理成 recipe selector 的 behavior manifest。

本脚本是输入准备工具，不参与 MP/AF/AP 评分。它的职责是：
1. 从 OAS/OASIS marker layer 读取 marker 顺序和 marker_id。
2. 从图像目录读取与 marker 对应的 CD-SEM/aerial 图像。
3. 写出 `behavior.jsonl` 和 `aerial_npz/*.npz`，供 no-train backend 使用。

当前主线只支持 `aerial_npz` 一个图像字段，不再保留 EPE/PV/NILS/resist 接口。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from layout_utils import DEFAULT_PIXEL_SIZE_NM, MarkerRasterBuilder


IMAGE_SUFFIXES = {".npz", ".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _as_float_image(array: Any) -> np.ndarray:
    """把输入图像转换为 0-1 范围内的二维 float32 数组。"""
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 3:
        arr = np.mean(arr, axis=2)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {arr.shape}")
    lo = float(np.min(arr)) if arr.size else 0.0
    hi = float(np.max(arr)) if arr.size else 0.0
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return np.ascontiguousarray(arr, dtype=np.float32)


def _load_image(path: Path) -> np.ndarray:
    """读取 npz/npy 或常见位图格式。"""
    suffix = path.suffix.lower()
    if suffix == ".npz":
        data = np.load(str(path))
        try:
            key = "image" if "image" in data.files else data.files[0]
            return _as_float_image(data[key])
        finally:
            data.close()
    if suffix == ".npy":
        return _as_float_image(np.load(str(path)))
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Reading non-NPY images requires Pillow") from exc
    with Image.open(path) as image:
        return _as_float_image(np.asarray(image.convert("L"), dtype=np.float32))


def _discover_images(image_dir: str | Path) -> List[Path]:
    """按文件名稳定排序发现图像文件。"""
    root = Path(image_dir)
    if root.is_file():
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"image directory does not exist: {root}")
    return sorted(item for item in root.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)


def _image_index(paths: Sequence[Path]) -> Dict[str, Path]:
    """用文件 stem 建立索引，重复文件保留第一张。"""
    result: Dict[str, Path] = {}
    for path in paths:
        result.setdefault(path.stem, path)
    return result


def _choose_image(marker_id: str, marker_index: int, paths: Sequence[Path], by_stem: Dict[str, Path]) -> Path | None:
    """优先按 marker_id/stem 匹配，失败后按 marker 顺序匹配。"""
    if marker_id in by_stem:
        return by_stem[marker_id]
    short_marker = re.sub(r"__marker_\d{6}$", "", marker_id)
    for stem, path in by_stem.items():
        if marker_id in stem or stem in marker_id or short_marker in stem:
            return path
    if marker_index < len(paths):
        return paths[marker_index]
    return None


def preprocess_behavior_inputs(
    input_path: str | Path,
    *,
    marker_layer: str,
    image_dir: str | Path,
    output_dir: str | Path,
    clip_size_um: float = 1.35,
    default_risk_score: float = 0.0,
    recursive_input: bool = False,
) -> Dict[str, Any]:
    """执行预处理，并返回输出路径与匹配统计。"""
    output = Path(output_dir)
    aerial_dir = output / "aerial_npz"
    aerial_dir.mkdir(parents=True, exist_ok=True)
    builder = MarkerRasterBuilder(
        config={
            "hotspot_layer": str(marker_layer),
            "clip_size_um": float(clip_size_um),
            "pixel_size_nm": DEFAULT_PIXEL_SIZE_NM,
            "recursive_input": bool(recursive_input),
        },
        temp_dir=output / "_preprocess_marker_cache",
    )
    records = builder.build_records(input_path)
    images = _discover_images(image_dir)
    images_by_stem = _image_index(images)
    manifest_path = output / "behavior.jsonl"
    rows: List[Dict[str, Any]] = []
    skipped = 0
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            image_path = _choose_image(record.marker_id, record.marker_index, images, images_by_stem)
            if image_path is None:
                skipped += 1
                continue
            image = _load_image(image_path)
            aerial_path = aerial_dir / f"{record.marker_id}.npz"
            np.savez_compressed(aerial_path, image=image.astype(np.float32))
            row = {
                "sample_id": record.marker_id,
                "source_path": record.source_path,
                "marker_id": record.marker_id,
                "marker_index": int(record.marker_index),
                "clip_bbox": [float(value) for value in record.clip_bbox],
                "aerial_npz": str(aerial_path),
                "risk_score": float(default_risk_score),
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "behavior_manifest": str(manifest_path),
        "aerial_npz_dir": str(aerial_dir),
        "marker_count": int(len(records)),
        "matched_count": int(len(rows)),
        "skipped_count": int(skipped),
    }


def _build_parser() -> argparse.ArgumentParser:
    """构建预处理 CLI。"""
    parser = argparse.ArgumentParser(description="生成 CD-SEM recipe selector behavior manifest")
    parser.add_argument("input_path")
    parser.add_argument("--marker-layer", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--clip-size", type=float, default=1.35)
    parser.add_argument("--default-risk-score", type=float, default=0.0)
    parser.add_argument("--recursive-input", action="store_true")
    return parser


def main() -> int:
    """命令行入口，输出 manifest 生成摘要。"""
    args = _build_parser().parse_args()
    result = preprocess_behavior_inputs(
        args.input_path,
        marker_layer=args.marker_layer,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        clip_size_um=float(args.clip_size),
        default_risk_score=float(args.default_risk_score),
        recursive_input=bool(args.recursive_input),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
