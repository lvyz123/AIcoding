#!/usr/bin/env python3
"""CD-SEM recipe selector 的无训练 representative marker 后端。

本文件是 `recipe_site_selector.py` 的第一步后端，不是最终 recipe 主入口。它负责：
1. 从 OAS/OASIS marker layer 读取已有 hotspot markers。
2. 读取 behavior manifest，并校验每个有效样本只使用 `aerial_npz`。
3. 生成 handcrafted feature vector 与 marker metadata。
4. 用确定性 representative selection 输出 clusters，供后续 care-area expansion 使用。

当前恢复版刻意保持简单：每个 marker 先作为一个独立 cluster。后续全局预算选择、care-area
扩展、MP/AF/AP 构造都在 `recipe_site_selector.py` 中完成。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from feature_extractor_handcraft import encode_handcrafted_features, validate_behavior_row
from layer_operations import LayerOperationProcessor
from layout_utils import DEFAULT_PIXEL_SIZE_NM, MarkerRasterBuilder


def _resolve_manifest_path(path: str | Path) -> Path:
    """解析 behavior manifest；允许传入 jsonl 文件或包含 behavior.jsonl 的目录。"""
    manifest = Path(path)
    if manifest.is_dir():
        manifest = manifest / "behavior.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"behavior manifest does not exist: {manifest}")
    return manifest


def _read_behavior_manifest(path: str | Path) -> List[Dict[str, Any]]:
    """读取 behavior.jsonl，并拒绝旧的 EPE/PV/NILS/resist 分支。"""
    manifest = _resolve_manifest_path(path)
    rows: List[Dict[str, Any]] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            validate_behavior_row(row)
            aerial_path = Path(str(row["aerial_npz"]))
            if not aerial_path.is_absolute():
                aerial_path = manifest.parent / aerial_path
            row["aerial_npz"] = str(aerial_path)
            row["_manifest_line"] = int(line_no)
            rows.append(row)
    return rows


def _behavior_by_marker(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """按 marker_id 建立 behavior 行索引；重复 marker 保留第一条。"""
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        marker_id = str(row.get("marker_id") or row.get("sample_id") or "")
        if marker_id and marker_id not in result:
            result[marker_id] = dict(row)
    return result


def _make_layer_processor(register_ops: Iterable[Sequence[str]] | None) -> LayerOperationProcessor:
    """根据 CLI 的 `--register-op SOURCE TARGET OP RESULT` 构建 layer operation 处理器。"""
    processor = LayerOperationProcessor()
    for item in register_ops or []:
        if len(item) != 4:
            raise ValueError(f"register-op must have four fields, got: {item}")
        source_layer, target_layer, operation, result_layer = item
        processor.register_operation_rule(source_layer, operation, target_layer, result_layer)
    return processor


def _normalize_risks(values: Sequence[float]) -> List[float]:
    """把 manifest risk_score 压到 0-1；全 0 时保持全 0，交给下游重分配权重。"""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or float(np.max(arr)) <= 1e-12:
        return [0.0 for _ in values]
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        return [float(np.clip(value, 0.0, 1.0)) for value in values]
    return [float(np.clip((float(value) - lo) / (hi - lo), 0.0, 1.0)) for value in values]


def _build_samples(args: argparse.Namespace) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """构建 marker 样本和已匹配的 metadata。"""
    layer_processor = _make_layer_processor(getattr(args, "register_op", None) or [])
    builder = MarkerRasterBuilder(
        config={
            "hotspot_layer": str(args.marker_layer),
            "clip_size_um": float(args.clip_size),
            "pixel_size_nm": DEFAULT_PIXEL_SIZE_NM,
            "apply_layer_operations": bool(getattr(args, "apply_layer_ops", False)),
            "recursive_input": bool(getattr(args, "recursive_input", False)),
        },
        temp_dir=Path(getattr(args, "output_dir", Path.cwd())) / "_notrain_marker_cache",
        layer_processor=layer_processor if bool(getattr(args, "apply_layer_ops", False)) else None,
    )
    behavior_rows = _read_behavior_manifest(args.behavior_manifest)
    behavior_by_marker = _behavior_by_marker(behavior_rows)
    samples: List[Dict[str, Any]] = []
    metadata: List[Dict[str, Any]] = []
    for record in builder.build_records(args.input_path):
        behavior = behavior_by_marker.get(record.marker_id)
        if behavior is None:
            continue
        meta = record.to_metadata()
        meta.update(
            {
                "sample_id": str(behavior.get("sample_id", record.marker_id)),
                "aerial_npz": str(behavior["aerial_npz"]),
                "risk_score": float(behavior.get("risk_score", 0.0) or 0.0),
            }
        )
        samples.append({"metadata": meta, "behavior": behavior, "clip_bitmap": record.clip_bitmap})
        metadata.append(meta)
    if not samples:
        raise ValueError("No marker records matched behavior manifest rows")
    return samples, metadata


def run_notrain_mp_selection(args: argparse.Namespace) -> Dict[str, Any]:
    """执行无训练 representative marker selection，并返回 recipe selector 所需 schema。"""
    samples, metadata = _build_samples(args)
    encoded = encode_handcrafted_features(samples)
    features = np.asarray(encoded["features"], dtype=np.float32)
    risks = [float(item.get("risk_score", 0.0) or 0.0) for item in metadata]
    normalized_risks = _normalize_risks(risks)
    clusters: List[Dict[str, Any]] = []
    for index, item in enumerate(metadata):
        marker_id = str(item["marker_id"])
        item["normalized_risk_score"] = float(normalized_risks[index])
        item["feature_l2_norm"] = float(np.linalg.norm(features[index])) if features.size else 0.0
        cluster = {
            "cluster_id": int(index),
            "marker_id": marker_id,
            "marker_ids": [marker_id],
            "size": 1,
            "cluster_coverage": 1,
            "representative_metadata": dict(item),
            "normalized_risk_score": float(normalized_risks[index]),
        }
        clusters.append(cluster)
    return {
        "pipeline_mode": "hotspot_recipe_notrain_backend_v1",
        "clusters": clusters,
        "file_metadata": metadata,
        "feature_shape": list(features.shape),
        "result_summary": {
            "input_marker_count": int(len(metadata)),
            "selected_cluster_count": int(len(clusters)),
            "feature_dimension": int(features.shape[1]) if features.ndim == 2 else 0,
            "all_risk_scores_are_zero": bool(all(value <= 1e-12 for value in risks)),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    """构建 backend 调试 CLI。"""
    parser = argparse.ArgumentParser(description="无训练 representative marker 后端")
    parser.add_argument("input_path")
    parser.add_argument("--marker-layer", required=True)
    parser.add_argument("--behavior-manifest", required=True)
    parser.add_argument("--clip-size", type=float, default=1.35)
    parser.add_argument("--output-dir", default="_notrain_backend_out")
    parser.add_argument("--recursive-input", action="store_true")
    parser.add_argument("--apply-layer-ops", action="store_true")
    parser.add_argument("--register-op", action="append", nargs=4, metavar=("SOURCE_LAYER", "TARGET_LAYER", "OPERATION", "RESULT_LAYER"))
    return parser


def main() -> int:
    """命令行调试入口，只打印 backend summary。"""
    args = _build_parser().parse_args()
    result = run_notrain_mp_selection(args)
    print(json.dumps(result["result_summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
