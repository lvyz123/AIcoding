#!/usr/bin/env python3
"""No-training optimized behavior clustering entrypoint.

本脚本是完全独立的无训练光刻行为 coverage 聚类入口，用于 AutoEncoder 无法训练、
训练结果暂不可用，或希望先运行 deterministic baseline 的场景。它不再 import 或调用
`layout_clustering_optimized.py`，而是把 behavior clustering 主流程在本文件内独立维护；
共享的只有 OAS/marker raster 底层工具 `layout_utils.py`、layer boolean 工具
`layer_operations.py`，以及 handcrafted FV 生成器 `feature_extractor_handcraft.py`。

整体流程:
1. 读取 OAS/OASIS 输入和真实 marker layer；后续版本默认有 marker，不走 auto-marker。
2. 如启用 `--apply-layer-ops` 或 `--register-op`，先执行 layer boolean 操作。
3. 自动在 `_temp_runs/.../handcraft_features.npz` 生成 handcrafted FV：
   aerial DCT/FFT/HOG/gradient + 可选 PV/EPE/NILS/resist stats + layout geometry
   critical proxy + polygon WL graph signature。
4. 在本脚本内部执行 exact hash 去重、ANN top-K graph、weighted facility location、
   weighted k-center 补洞。
5. 对 representative-member 关系执行 behavior final verification。aerial SSIM 一定参与；
   EPE/PV/NILS 若在 manifest 中全局可用，则按固定权重加入 weighted score。
6. verification 失败的 exact cluster 不跨 cluster reassign，直接形成 singleton/base
   representative，保证输出结果可解释。
7. 输出 JSON/TXT summary，并可选导出 review 目录和 aerial/resist/PV diff。

注意: no-train FV 只负责候选召回和 coverage ordering；最终 cluster 质量仍由 behavior image
verification 兜底。AE 版和 no-train 版现在互不依赖，便于两条路线独立演进。
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import hnswlib
import numpy as np
from skimage.metrics import structural_similarity

from feature_extractor_handcraft import encode_handcrafted_features
from layer_operations import LayerOperationProcessor
from layout_utils import (
    DEFAULT_PIXEL_SIZE_NM,
    ExactCluster,
    MarkerRasterBuilder,
    MarkerRecord,
    _make_sample_filename,
    _materialize_clip_bitmap,
)


PIPELINE_MODE = "optimized_notrain"
IMAGE_KEY = "image"
BEHAVIOR_CHANNELS = ("aerial", "epe", "pv", "nils")
DIFF_CHANNELS = ("aerial", "resist", "pv")
DEFAULT_BEHAVIOR_WEIGHTS = {
    "aerial": 0.60,
    "epe": 0.15,
    "pv": 0.15,
    "nils": 0.10,
}


@dataclass
class BehaviorSample:
    """一条来自 behavior manifest 的样本记录，保存 marker、图像路径和风险分数。"""

    sample_id: str
    marker_id: str
    source_path: str
    clip_bbox: Tuple[float, float, float, float]
    paths: Dict[str, str]
    risk_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    images: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class BehaviorUnit:
    """一个参与 representative selection 的最小单元：exact cluster + 行为样本 + FV + 权重。"""

    exact_cluster: ExactCluster
    behavior: BehaviorSample
    feature: np.ndarray
    weight: float

    @property
    def exact_cluster_id(self) -> int:
        """返回该行为单元对应的 exact cluster id，便于输出和索引。"""
        return int(self.exact_cluster.exact_cluster_id)

    @property
    def marker_id(self) -> str:
        """返回 exact cluster representative 的 marker id，作为人类可读标识。"""
        return str(self.exact_cluster.representative.marker_id)


@dataclass
class VerificationResult:
    """记录 representative 与 member 的 behavior verification 结果。"""

    passed: bool
    weighted_distance: float
    channel_distances: Dict[str, float]


def _json_default(value: Any) -> Any:
    """把 numpy/path 等对象转换为 JSON 可序列化类型。"""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _layer_spec_text(layer_spec: Any) -> str:
    """把 layer spec 规范化为 `layer/datatype` 文本，便于 metadata 和日志显示。"""
    if isinstance(layer_spec, str):
        return str(layer_spec).strip()
    if isinstance(layer_spec, Sequence) and len(layer_spec) >= 2:
        return f"{int(layer_spec[0])}/{int(layer_spec[1])}"
    return str(layer_spec)


def _layer_operation_payload(layer_processor: Any) -> List[Dict[str, str]]:
    """从 LayerOperationProcessor 中提取可写入结果 JSON 的规则列表。"""
    rules = list(getattr(layer_processor, "operation_rules", []) or [])
    return [
        {
            "source_layer": _layer_spec_text(rule.get("source_layer")),
            "target_layer": _layer_spec_text(rule.get("target_layer")),
            "operation": str(rule.get("operation", "")),
            "result_layer": _layer_spec_text(rule.get("result_layer")),
        }
        for rule in rules
    ]


def _make_layer_processor(register_ops: Sequence[Sequence[str]] | None) -> LayerOperationProcessor:
    """根据 CLI 传入的 `--register-op` 规则构建 layer boolean 操作处理器。"""
    processor = LayerOperationProcessor()
    for source_layer, target_layer, operation, result_layer in register_ops or []:
        processor.register_operation_rule(source_layer, operation, target_layer, result_layer)
    return processor


def _review_dir_from_args(args: argparse.Namespace) -> str | None:
    """解析 review 目录参数；同时支持新版 `--review-dir` 和旧兼容别名。"""
    review_dir = getattr(args, "review_dir", None)
    legacy_dir = getattr(args, "export_cluster_review_dir", None)
    if review_dir and legacy_dir:
        print(f"同时指定 --review-dir 和 --export-cluster-review-dir，使用 --review-dir: {review_dir}")
        return str(review_dir)
    return str(review_dir or legacy_dir) if (review_dir or legacy_dir) else None



def _read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """读取 JSONL 文件，并保证每个非空行都是 JSON object。"""
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Manifest row must be an object at {path}:{line_no}")
            records.append(item)
    return records


def _resolve_path(path_text: str, base_dir: Path) -> str:
    """把 manifest 中的相对路径解析为相对于 manifest 文件所在目录的绝对路径。"""
    path = Path(path_text)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _resolve_behavior_manifest_path(path: str | Path) -> Path:
    """解析 behavior manifest 输入；若传入目录则自动使用其中的 behavior.jsonl。"""
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "behavior.jsonl"
    return manifest_path


def _load_behavior_manifest(path: str | Path) -> Dict[str, BehaviorSample]:
    """读取 behavior manifest，构建 sample_id 到 BehaviorSample 的映射。"""
    manifest_path = _resolve_behavior_manifest_path(path)
    base_dir = manifest_path.resolve().parent
    samples: Dict[str, BehaviorSample] = {}
    for item in _read_jsonl(manifest_path):
        required = ("sample_id", "source_path", "marker_id", "clip_bbox", "aerial_npz")
        missing = [key for key in required if key not in item]
        if missing:
            raise ValueError(f"Behavior manifest row missing required fields: {missing}")
        sample_id = str(item["sample_id"])
        marker_id = str(item["marker_id"])
        if sample_id in samples:
            raise ValueError(f"Duplicate behavior sample_id: {sample_id}")
        clip_bbox_values = item["clip_bbox"]
        if not isinstance(clip_bbox_values, Sequence) or len(clip_bbox_values) != 4:
            raise ValueError(f"clip_bbox must contain four values for sample {sample_id}")
        paths: Dict[str, str] = {}
        for channel in ("layout", "aerial", "resist", "epe", "pv", "nils"):
            key = f"{channel}_npz"
            if item.get(key):
                paths[channel] = _resolve_path(str(item[key]), base_dir)
        samples[sample_id] = BehaviorSample(
            sample_id=sample_id,
            marker_id=marker_id,
            source_path=str(item["source_path"]),
            clip_bbox=tuple(float(v) for v in clip_bbox_values),
            paths=paths,
            risk_score=float(item.get("risk_score", 0.0) or 0.0),
            metadata=dict(item),
        )
    if not samples:
        raise ValueError("Behavior manifest is empty")
    _validate_behavior_channels(samples)
    return samples


def _validate_behavior_channels(samples: Mapping[str, BehaviorSample]) -> None:
    """校验 behavior channel 契约：aerial 必填，可选 channel 必须全局一致。"""
    for sample in samples.values():
        if "aerial" not in sample.paths:
            raise ValueError(f"Sample {sample.sample_id} is missing aerial_npz")
    for channel in ("epe", "pv", "nils"):
        present = [sample.sample_id for sample in samples.values() if channel in sample.paths]
        if present and len(present) != len(samples):
            missing = [sample.sample_id for sample in samples.values() if channel not in sample.paths]
            raise ValueError(f"Optional channel {channel} must be provided for all samples or none; missing preview: {missing[:5]}")


def _load_feature_npz(path: str | Path) -> Dict[str, np.ndarray]:
    """读取 AutoEncoder encode 输出的 features.npz，并按 sample_id 建索引。"""
    with np.load(str(path), allow_pickle=False) as data:
        if "sample_ids" not in data or "features" not in data:
            raise ValueError("Feature NPZ must contain sample_ids and features arrays")
        sample_ids = [str(value) for value in data["sample_ids"].astype(str).tolist()]
        features = np.asarray(data["features"], dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("features array must be 2-D")
    if len(sample_ids) != int(features.shape[0]):
        raise ValueError("sample_ids length must match features rows")
    mapping: Dict[str, np.ndarray] = {}
    for idx, sample_id in enumerate(sample_ids):
        if sample_id in mapping:
            raise ValueError(f"Duplicate feature sample_id: {sample_id}")
        mapping[sample_id] = np.ascontiguousarray(features[idx], dtype=np.float32)
    return mapping


def _load_npz_image(path: str | Path) -> np.ndarray:
    """读取 behavior NPZ 中键名为 `image` 的二维 float32 图像。"""
    with np.load(str(path), allow_pickle=False) as data:
        if IMAGE_KEY not in data:
            raise ValueError(f"NPZ {path} must contain key '{IMAGE_KEY}'")
        image = np.asarray(data[IMAGE_KEY], dtype=np.float32)
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 2:
        raise ValueError(f"Image in {path} must be a 2-D float32 array")
    return np.ascontiguousarray(image, dtype=np.float32)


def _behavior_image(sample: BehaviorSample, channel: str) -> np.ndarray:
    """按需加载并缓存某个样本的 behavior 图像，避免重复读 NPZ。"""
    if channel not in sample.paths:
        raise ValueError(f"Sample {sample.sample_id} has no {channel} path")
    if channel not in sample.images:
        sample.images[channel] = _load_npz_image(sample.paths[channel])
    return sample.images[channel]


def _ssim_distance(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """计算两张 behavior 图像的 SSIM distance，即 `1 - SSIM`。"""
    a = np.asarray(image_a, dtype=np.float32)
    b = np.asarray(image_b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"SSIM image shape mismatch: {a.shape} vs {b.shape}")
    if min(a.shape) < 3:
        denom = max(float(max(np.max(a), np.max(b)) - min(np.min(a), np.min(b))), 1.0)
        return float(np.mean(np.abs(a - b)) / denom)
    data_range = float(max(np.max(a), np.max(b)) - min(np.min(a), np.min(b)))
    if data_range <= 1e-12:
        return 0.0 if np.allclose(a, b) else 1.0
    win_size = min(7, int(min(a.shape)))
    if win_size % 2 == 0:
        win_size -= 1
    score = structural_similarity(a, b, data_range=data_range, win_size=max(win_size, 3))
    return float(max(0.0, min(2.0, 1.0 - score)))


def _available_verification_channels(samples: Mapping[str, BehaviorSample]) -> List[str]:
    """根据 manifest 判断 final verification 中实际启用哪些 behavior channel。"""
    channels = ["aerial"]
    for channel in ("epe", "pv", "nils"):
        if all(channel in sample.paths for sample in samples.values()):
            channels.append(channel)
    return channels


def _normalized_behavior_weights(channels: Sequence[str]) -> Dict[str, float]:
    """按当前可用 channel 重新归一化默认 behavior verification 权重。"""
    raw = {channel: float(DEFAULT_BEHAVIOR_WEIGHTS[channel]) for channel in channels}
    total = max(float(sum(raw.values())), 1e-12)
    return {channel: value / total for channel, value in raw.items()}


def _behavior_verification(
    rep: BehaviorSample,
    target: BehaviorSample,
    *,
    channels: Sequence[str],
    threshold: float,
) -> VerificationResult:
    """对 representative 和 target 执行多 channel behavior verification。"""
    weights = _normalized_behavior_weights(channels)
    distances: Dict[str, float] = {}
    weighted = 0.0
    for channel in channels:
        dist = _ssim_distance(_behavior_image(rep, channel), _behavior_image(target, channel))
        distances[channel] = float(dist)
        weighted += weights[channel] * float(dist)
    return VerificationResult(
        passed=bool(weighted <= float(threshold) + 1e-12),
        weighted_distance=float(weighted),
        channel_distances=distances,
    )


def _feature_matrix(units: Sequence[BehaviorUnit]) -> np.ndarray:
    """把所有 BehaviorUnit 的 FV 堆叠成 L2-normalized 特征矩阵。"""
    if not units:
        return np.empty((0, 0), dtype=np.float32)
    matrix = np.vstack([np.asarray(unit.feature, dtype=np.float32).reshape(1, -1) for unit in units])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)
    return np.ascontiguousarray(matrix, dtype=np.float32)


def _ann_topk_graph(features: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """使用 hnswlib 建立 L2 ANN top-K 图，返回近邻索引和距离矩阵。"""
    n = int(features.shape[0])
    if n == 0:
        return np.empty((0, 0), dtype=np.int32), np.empty((0, 0), dtype=np.float32)
    k = min(max(int(top_k), 1), n)
    index = hnswlib.Index(space="l2", dim=int(features.shape[1]))
    index.init_index(max_elements=n, ef_construction=100, M=16)
    index.add_items(features, np.arange(n))
    index.set_ef(max(50, k * 2))
    labels, distances = index.knn_query(features, k=k)
    return labels.astype(np.int32), np.sqrt(np.maximum(distances, 0.0)).astype(np.float32)


def _similarity_tau(distances: np.ndarray) -> float:
    """从 ANN 距离分布估计 RBF 相似度转换中的 tau 尺度。"""
    valid = np.asarray(distances, dtype=np.float32).ravel()
    valid = valid[valid > 1e-9]
    if valid.size == 0:
        return 1.0
    return float(max(np.median(valid), 1e-6))


def _sparse_similarity(
    neighbors: np.ndarray,
    distances: np.ndarray,
    tau: float,
) -> Tuple[List[Dict[int, float]], List[List[Tuple[int, float]]]]:
    """把 top-K 距离转换成稀疏相似图，同时构建 facility selection 所需的反向索引。"""
    n = int(neighbors.shape[0])
    rows: List[Dict[int, float]] = [dict() for _ in range(n)]
    reverse: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    denom = max(float(tau) ** 2, 1e-12)
    for row in range(n):
        rows[row][row] = 1.0
        for col, dist in zip(neighbors[row].tolist(), distances[row].tolist()):
            sim = float(math.exp(-(float(dist) ** 2) / denom))
            rows[row][int(col)] = max(rows[row].get(int(col), 0.0), sim)
    for row, values in enumerate(rows):
        for col, sim in list(values.items()):
            rows[col][row] = max(rows[col].get(row, 0.0), sim)
    for row, values in enumerate(rows):
        for col, sim in values.items():
            reverse[col].append((row, float(sim)))
    return rows, reverse


def _weighted_facility_location(
    weights: np.ndarray,
    reverse_similarity: Sequence[Sequence[Tuple[int, float]]],
    *,
    coverage_target: float,
    min_gain: float,
) -> Tuple[List[int], float, List[float]]:
    """贪心求解 weighted facility location，选择能带来最大 coverage gain 的 reps。"""
    total_weight = max(float(np.sum(weights)), 1e-12)
    best_sim = np.zeros_like(weights, dtype=np.float64)
    selected: List[int] = []
    selected_set = set()
    gains: List[float] = []

    while True:
        current_score = float(np.dot(weights, best_sim) / total_weight)
        if current_score >= float(coverage_target):
            break
        best_idx = -1
        best_gain = 0.0
        for candidate_idx, affected in enumerate(reverse_similarity):
            if candidate_idx in selected_set:
                continue
            gain = 0.0
            for row, sim in affected:
                if sim > best_sim[row]:
                    gain += float(weights[row]) * (float(sim) - float(best_sim[row]))
            if gain > best_gain + 1e-12:
                best_gain = float(gain)
                best_idx = int(candidate_idx)
        if best_idx < 0 or (best_gain / total_weight) < float(min_gain):
            break
        selected.append(best_idx)
        selected_set.add(best_idx)
        gains.append(best_gain / total_weight)
        for row, sim in reverse_similarity[best_idx]:
            if sim > best_sim[row]:
                best_sim[row] = float(sim)
    if not selected and len(weights):
        selected.append(int(np.argmax(weights)))
    final_score = float(np.dot(weights, best_sim) / total_weight)
    return selected, final_score, gains


def _nearest_selected(features: np.ndarray, selected: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """计算每个样本距离最近的已选 representative 以及对应 FV 距离。"""
    if not selected:
        n = int(features.shape[0])
        return np.full(n, -1, dtype=np.int32), np.full(n, np.inf, dtype=np.float32)
    selected_features = features[np.asarray(selected, dtype=np.int32)]
    dists = np.linalg.norm(features[:, None, :] - selected_features[None, :, :], axis=2)
    nearest_pos = np.argmin(dists, axis=1)
    nearest_ids = np.asarray([selected[int(pos)] for pos in nearest_pos], dtype=np.int32)
    nearest_dist = dists[np.arange(int(features.shape[0])), nearest_pos].astype(np.float32)
    return nearest_ids, nearest_dist


def _weighted_k_center_fill(
    features: np.ndarray,
    units: Sequence[BehaviorUnit],
    selected: Sequence[int],
    *,
    tau: float,
    high_risk_quantile: float,
) -> List[int]:
    """用 weighted k-center 思路补充 farthest / high-risk 未覆盖样本。"""
    selected_set = set(int(value) for value in selected)
    nearest_ids, nearest_dist = _nearest_selected(features, selected)
    risks = np.asarray([unit.behavior.risk_score for unit in units], dtype=np.float32)
    risk_cutoff = float(np.quantile(risks, float(high_risk_quantile))) if risks.size else math.inf
    additions: List[int] = []
    for idx, dist in sorted(enumerate(nearest_dist.tolist()), key=lambda item: -item[1]):
        if idx in selected_set:
            continue
        is_far = float(dist) > float(tau)
        is_high_risk_hole = risks[idx] >= risk_cutoff and float(dist) > 0.5 * float(tau)
        if is_far or is_high_risk_hole or int(nearest_ids[idx]) < 0:
            additions.append(int(idx))
            selected_set.add(int(idx))
    return additions


class NotrainOptimizedRunner(MarkerRasterBuilder):
    """行为覆盖聚类主运行器，复用 MarkerRasterBuilder 的 marker clip 构建能力。"""

    def __init__(self, *, config: Dict[str, Any], temp_dir: Path, layer_processor: Any | None = None):
        """初始化 runner，读取 behavior manifest/FV，并把参数转换为 marker raster 配置。"""
        clean_config = {
            "apply_layer_operations": bool(config.get("apply_layer_operations", False)),
            "clip_size_um": float(config.get("clip_size_um", 1.35)),
            "hotspot_layer": str(config["marker_layer"]),
            "pixel_size_nm": int(config.get("pixel_size_nm", DEFAULT_PIXEL_SIZE_NM)),
        }
        super().__init__(config=clean_config, temp_dir=temp_dir, layer_processor=layer_processor)
        self.pipeline_mode = PIPELINE_MODE
        self.feature_source = "handcraft"
        self.feature_metadata = dict(config.get("feature_metadata", {}) or {})
        self.behavior_manifest_path = str(config["behavior_manifest"])
        self.feature_npz_path = str(config["feature_npz"])
        self.ann_top_k = int(config.get("ann_top_k", 64))
        self.coverage_target = float(config.get("coverage_target", 0.985))
        self.facility_min_gain = float(config.get("facility_min_gain", 1e-4))
        self.behavior_verification_threshold = float(config.get("behavior_verification_threshold", 0.08))
        self.high_risk_quantile = float(config.get("high_risk_quantile", 0.90))
        self.export_diff_channels = tuple(config.get("export_diff_channels", ()))

        self.behavior_samples = _load_behavior_manifest(self.behavior_manifest_path)
        self.behavior_by_marker_id: Dict[str, BehaviorSample] = {}
        for sample in self.behavior_samples.values():
            if sample.marker_id in self.behavior_by_marker_id:
                raise ValueError(f"Duplicate behavior marker_id: {sample.marker_id}")
            self.behavior_by_marker_id[str(sample.marker_id)] = sample
        self.feature_by_sample_id = _load_feature_npz(self.feature_npz_path)
        missing_features = [
            sample.sample_id
            for sample in self.behavior_samples.values()
            if sample.sample_id not in self.feature_by_sample_id and sample.marker_id not in self.feature_by_sample_id
        ]
        if missing_features:
            raise ValueError(f"Feature NPZ missing sample_ids from behavior manifest: {missing_features[:5]}")
        self.verification_channels = _available_verification_channels(self.behavior_samples)
        self.behavior_stats: Dict[str, Any] = {}
        self.input_marker_count = 0
        self.skipped_missing_behavior_count = 0
        self.behavior_verification_stats = {
            "verified_pass": 0,
            "verified_reject": 0,
            "singleton_created": 0,
        }

    def _log(self, message: str) -> None:
        """统一的中文过程日志出口，当前直接打印到 stdout。"""
        print(message)

    def _layer_operations(self) -> List[Dict[str, str]]:
        """返回当前启用的 layer operation 规则，用于输出 metadata。"""
        return _layer_operation_payload(self.layer_processor)

    def _collect_marker_records_for_file(self, filepath: Path) -> List[MarkerRecord]:
        """读取单个 OAS 文件，应用 layer ops 后按 marker layer 构建 marker records。"""
        layout_index = self._prepare_layout(filepath)
        self._log(f"文件 {filepath.name}: marker 数 {len(layout_index.marker_polygons)}, pattern 元素数 {len(layout_index.indexed_elements)}")
        records: List[MarkerRecord] = []
        for marker_index, marker_poly in enumerate(layout_index.marker_polygons):
            record = self._build_marker_record(filepath, marker_index, marker_poly, layout_index)
            if record is not None:
                records.append(record)
        self._log(f"文件 {filepath.name}: 生成窗口样本 {len(records)} 个")
        return records

    def _behavior_for_record(self, record: MarkerRecord) -> BehaviorSample:
        """根据 marker record 找到对应的 behavior manifest 样本。"""
        if str(record.marker_id) in self.behavior_samples:
            return self.behavior_samples[str(record.marker_id)]
        if str(record.marker_id) in self.behavior_by_marker_id:
            return self.behavior_by_marker_id[str(record.marker_id)]
        raise ValueError(f"No behavior manifest entry for marker_id={record.marker_id}")

    def _feature_for_sample(self, sample: BehaviorSample) -> np.ndarray:
        """根据 behavior sample_id 或 marker_id 找到对应 handcrafted FV。"""
        for sample_id in (sample.sample_id, sample.marker_id):
            if sample_id in self.feature_by_sample_id:
                return np.asarray(self.feature_by_sample_id[sample_id], dtype=np.float32)
        raise ValueError(f"No feature vector for sample_id={sample.sample_id} marker_id={sample.marker_id}")

    def _make_behavior_units(self, exact_clusters: Sequence[ExactCluster]) -> List[BehaviorUnit]:
        """把 exact clusters 转换为 behavior selection 使用的加权单元。"""
        units: List[BehaviorUnit] = []
        for cluster in exact_clusters:
            behavior = self._behavior_for_record(cluster.representative)
            feature = self._feature_for_sample(behavior)
            weight = float(cluster.weight) * (1.0 + max(0.0, float(behavior.risk_score)))
            units.append(BehaviorUnit(cluster, behavior, np.ascontiguousarray(feature, dtype=np.float32), float(weight)))
        return units

    def _select_representatives(self, units: Sequence[BehaviorUnit]) -> Tuple[List[int], np.ndarray]:
        """基于 FV ANN 图执行 facility location 和 k-center 补洞，选出 representative 下标。"""
        # FV 相似图只负责提出覆盖关系；最终成员仍必须通过 behavior verification。
        features = _feature_matrix(units)
        neighbors, distances = _ann_topk_graph(features, self.ann_top_k)
        tau = _similarity_tau(distances)
        _, reverse = _sparse_similarity(neighbors, distances, tau)
        weights = np.asarray([unit.weight for unit in units], dtype=np.float64)
        facility_selected, coverage_score, gains = _weighted_facility_location(
            weights,
            reverse,
            coverage_target=self.coverage_target,
            min_gain=self.facility_min_gain,
        )
        kcenter_added = _weighted_k_center_fill(
            features,
            units,
            facility_selected,
            tau=tau,
            high_risk_quantile=self.high_risk_quantile,
        )
        selected = list(dict.fromkeys([*facility_selected, *kcenter_added]))
        self.behavior_stats = {
            "fv_dim": int(features.shape[1]) if features.size else 0,
            "ann_top_k": int(min(self.ann_top_k, len(units))),
            "ann_edge_count": int(neighbors.size),
            "similarity_tau": float(tau),
            "coverage_score": float(coverage_score),
            "facility_selected_count": int(len(facility_selected)),
            "kcenter_added_count": int(len(kcenter_added)),
            "facility_gain_history": [float(value) for value in gains[:25]],
        }
        return selected, features

    def _assign_units(self, selected: Sequence[int], features: np.ndarray) -> Dict[int, List[int]]:
        """把每个 behavior unit 分配给 FV 空间中最近的已选 representative。"""
        nearest, _ = _nearest_selected(features, selected)
        assignments = {int(rep_idx): [] for rep_idx in selected}
        for unit_idx, rep_idx in enumerate(nearest.tolist()):
            assignments.setdefault(int(rep_idx), []).append(int(unit_idx))
        return assignments

    def _verified_cluster_units(
        self,
        selected: Sequence[int],
        units: Sequence[BehaviorUnit],
        features: np.ndarray,
    ) -> List[Tuple[int, List[int], Dict[int, VerificationResult]]]:
        """对 representative-member 关系做最终 behavior verification，并创建 singleton。"""
        assignments = self._assign_units(selected, features)
        self.behavior_verification_stats = {
            "verified_pass": 0,
            "verified_reject": 0,
            "singleton_created": 0,
        }
        cluster_units: List[Tuple[int, List[int], Dict[int, VerificationResult]]] = []
        singleton_units: List[Tuple[int, List[int], Dict[int, VerificationResult]]] = []

        for rep_idx in selected:
            accepted: List[int] = []
            details: Dict[int, VerificationResult] = {}
            rep_sample = units[int(rep_idx)].behavior
            for unit_idx in assignments.get(int(rep_idx), []):
                target_sample = units[int(unit_idx)].behavior
                result = _behavior_verification(
                    rep_sample,
                    target_sample,
                    channels=self.verification_channels,
                    threshold=self.behavior_verification_threshold,
                )
                details[int(unit_idx)] = result
                if result.passed:
                    accepted.append(int(unit_idx))
                    self.behavior_verification_stats["verified_pass"] += 1
                else:
                    self.behavior_verification_stats["verified_reject"] += 1
                    self.behavior_verification_stats["singleton_created"] += 1
                    singleton_result = VerificationResult(True, 0.0, {channel: 0.0 for channel in self.verification_channels})
                    singleton_units.append((int(unit_idx), [int(unit_idx)], {int(unit_idx): singleton_result}))
            if accepted:
                cluster_units.append((int(rep_idx), accepted, details))
        return singleton_units + cluster_units

    def run(self, input_path: str) -> Dict[str, Any]:
        """执行完整 no-training behavior clustering 流程并返回结果字典。"""
        started_at = time.perf_counter()
        input_files = self._discover_input_files(input_path)
        if not input_files:
            raise ValueError("No .oas files found")

        marker_records: List[MarkerRecord] = []
        self._log(f"发现输入 OAS 文件数: {len(input_files)}")
        self._log(f"开始收集 marker 窗口，marker 层: {self.hotspot_layer}")
        marker_started = time.perf_counter()
        for filepath in input_files:
            if self.apply_layer_operations:
                self._log(f" 对文件 {filepath.name} 应用层操作...")
            marker_records.extend(self._collect_marker_records_for_file(filepath))
        marker_elapsed = time.perf_counter() - marker_started
        if not marker_records:
            raise ValueError("No hotspot markers found on the configured marker layer")
        self.input_marker_count = int(len(marker_records))
        marker_records = [record for record in marker_records if str(record.marker_id) in self.behavior_by_marker_id]
        self.skipped_missing_behavior_count = int(self.input_marker_count - len(marker_records))
        if self.skipped_missing_behavior_count:
            self._log(f"跳过缺少 behavior/aerial 的 marker: {self.skipped_missing_behavior_count}")
        if not marker_records:
            raise ValueError("No markers with behavior/aerial entries remain after filtering")

        self._log("开始 exact hash 去重和权重累计...")
        exact_started = time.perf_counter()
        exact_clusters = self._group_exact_clusters(marker_records)
        behavior_units = self._make_behavior_units(exact_clusters)
        exact_elapsed = time.perf_counter() - exact_started
        self._log(f"exact hash 去重完成: {len(marker_records)} -> {len(exact_clusters)}")

        self._log("开始 ANN top-K graph + weighted facility location + k-center 补洞...")
        select_started = time.perf_counter()
        selected, features = self._select_representatives(behavior_units)
        select_elapsed = time.perf_counter() - select_started
        self._log(
            f"representative 选择完成: facility={self.behavior_stats['facility_selected_count']}, "
            f"k-center={self.behavior_stats['kcenter_added_count']}, total={len(selected)}"
        )

        self._log("开始 behavior final verification...")
        verify_started = time.perf_counter()
        cluster_units = self._verified_cluster_units(selected, behavior_units, features)
        verify_elapsed = time.perf_counter() - verify_started
        self._log(f"behavior verification 完成: {self.behavior_verification_stats}")

        return self._build_results(
            marker_records,
            exact_clusters,
            behavior_units,
            cluster_units,
            runtime_summary={
                "collect_markers": round(marker_elapsed, 6),
                "exact_cluster": round(exact_elapsed, 6),
                "behavior_selection": round(select_elapsed, 6),
                "behavior_verification": round(verify_elapsed, 6),
                "total": round(time.perf_counter() - started_at, 6),
            },
        )

    def _sample_metadata(self, record: MarkerRecord, behavior: BehaviorSample | None = None) -> Dict[str, Any]:
        """生成单个输出 sample 的 metadata，供 JSON 和 review 目录使用。"""
        metadata = {
            "pipeline_mode": self.pipeline_mode,
            "marker_id": str(record.marker_id),
            "exact_cluster_id": int(record.exact_cluster_id),
            "source_path": str(record.source_path),
            "source_name": str(record.source_name),
            "marker_bbox": list(record.marker_bbox),
            "marker_center": list(record.marker_center),
            "clip_bbox": list(record.clip_bbox),
            "selected_representative_marker_id": None,
            "behavior_weighted_distance": None,
            "behavior_channel_distances": {},
        }
        if behavior is not None:
            metadata.update(
                {
                    "behavior_sample_id": behavior.sample_id,
                    "behavior_paths": dict(behavior.paths),
                    "risk_score": float(behavior.risk_score),
                    "feature_source": self.feature_source,
                }
            )
        return metadata

    def _build_results(
        self,
        marker_records: Sequence[MarkerRecord],
        exact_clusters: Sequence[ExactCluster],
        behavior_units: Sequence[BehaviorUnit],
        cluster_units: Sequence[Tuple[int, List[int], Dict[int, VerificationResult]]],
        *,
        runtime_summary: Dict[str, float],
    ) -> Dict[str, Any]:
        """物化 sample/representative clip，并组装最终 JSON 结果结构。"""
        del exact_clusters
        sample_dir = self.temp_dir / "samples"
        representative_dir = self.temp_dir / "representatives"
        sample_dir.mkdir(parents=True, exist_ok=True)
        representative_dir.mkdir(parents=True, exist_ok=True)

        behavior_by_rep_marker = {unit.marker_id: unit.behavior for unit in behavior_units}
        ordered_records = list(sorted(marker_records, key=lambda item: (item.source_name, item.marker_id)))
        sample_file_map: Dict[str, str] = {}
        sample_index_map: Dict[str, int] = {}
        file_list: List[str] = []
        file_metadata: List[Dict[str, Any]] = []
        record_to_behavior: Dict[str, BehaviorSample] = {}
        for unit in behavior_units:
            for member in unit.exact_cluster.members:
                record_to_behavior[str(member.marker_id)] = unit.behavior

        for sample_index, record in enumerate(ordered_records):
            sample_path = sample_dir / _make_sample_filename("sample", record.source_name, sample_index)
            sample_file = _materialize_clip_bitmap(record.clip_bitmap, record.clip_bbox, record.marker_id, sample_path, self.pixel_size_um)
            sample_file_map[record.marker_id] = sample_file
            sample_index_map[record.marker_id] = int(sample_index)
            file_list.append(sample_file)
            file_metadata.append(self._sample_metadata(record, record_to_behavior.get(str(record.marker_id))))

        clusters_output: List[Dict[str, Any]] = []
        for cluster_index, (rep_idx, assigned_unit_indices, verification_details) in enumerate(cluster_units):
            rep_unit = behavior_units[int(rep_idx)]
            assigned_units = [behavior_units[int(unit_idx)] for unit_idx in assigned_unit_indices]
            cluster_members = list(
                sorted(
                    (member for unit in assigned_units for member in unit.exact_cluster.members),
                    key=lambda item: (item.source_name, item.marker_id),
                )
            )
            if not cluster_members:
                continue

            rep_record = rep_unit.exact_cluster.representative
            rep_path = representative_dir / _make_sample_filename("rep", rep_record.source_name, cluster_index)
            representative_file = _materialize_clip_bitmap(
                rep_record.clip_bitmap,
                rep_record.clip_bbox,
                rep_record.marker_id,
                rep_path,
                self.pixel_size_um,
            )
            sample_indices = [sample_index_map[member.marker_id] for member in cluster_members]
            sample_files = [sample_file_map[member.marker_id] for member in cluster_members]
            sample_metadata = []
            member_behavior_paths = {}
            detail_by_exact = {
                behavior_units[int(unit_idx)].exact_cluster_id: detail
                for unit_idx, detail in verification_details.items()
            }
            for unit in assigned_units:
                detail = detail_by_exact.get(
                    unit.exact_cluster_id,
                    VerificationResult(True, 0.0, {channel: 0.0 for channel in self.verification_channels}),
                )
                for member in unit.exact_cluster.members:
                    sample_index = sample_index_map[member.marker_id]
                    metadata = dict(file_metadata[sample_index])
                    metadata.update(
                        {
                            "selected_representative_marker_id": rep_unit.marker_id,
                            "behavior_weighted_distance": float(detail.weighted_distance),
                            "behavior_channel_distances": dict(detail.channel_distances),
                        }
                    )
                    file_metadata[sample_index] = dict(metadata)
                    sample_metadata.append(metadata)
                    member_behavior_paths[str(member.marker_id)] = dict(unit.behavior.paths)

            exact_cluster_ids = [int(unit.exact_cluster_id) for unit in assigned_units]
            clusters_output.append(
                {
                    "cluster_id": int(cluster_index),
                    "pipeline_mode": self.pipeline_mode,
                    "size": int(len(cluster_members)),
                    "sample_indices": sample_indices,
                    "sample_files": sample_files,
                    "sample_metadata": sample_metadata,
                    "representative_file": representative_file,
                    "representative_metadata": {
                        "pipeline_mode": self.pipeline_mode,
                        "marker_id": rep_unit.marker_id,
                        "exact_cluster_id": int(rep_unit.exact_cluster_id),
                        "behavior_sample_id": rep_unit.behavior.sample_id,
                        "behavior_paths": dict(rep_unit.behavior.paths),
                        "risk_score": float(rep_unit.behavior.risk_score),
                        "verification_channels": list(self.verification_channels),
                        "feature_source": self.feature_source,
                    },
                    "marker_id": rep_unit.marker_id,
                    "exact_cluster_id": int(rep_unit.exact_cluster_id),
                    "marker_ids": [str(member.marker_id) for member in cluster_members],
                    "exact_cluster_ids": exact_cluster_ids,
                    "member_behavior_paths": member_behavior_paths,
                }
            )

        cluster_sizes = [int(cluster["size"]) for cluster in clusters_output]
        layer_ops = self._layer_operations()
        config = {
            "pipeline_mode": self.pipeline_mode,
            "marker_layer": str(self.hotspot_layer),
            "clip_size": float(self.clip_size_um),
            "pixel_size_nm": int(self.pixel_size_nm),
            "behavior_manifest": str(self.behavior_manifest_path),
            "feature_npz": str(self.feature_npz_path),
            "feature_source": str(self.feature_source),
            "feature_metadata": dict(self.feature_metadata),
            "input_marker_count": int(self.input_marker_count or len(marker_records)),
            "skipped_missing_behavior_count": int(self.skipped_missing_behavior_count),
            "ann_top_k": int(self.ann_top_k),
            "coverage_target": float(self.coverage_target),
            "facility_min_gain": float(self.facility_min_gain),
            "behavior_verification_threshold": float(self.behavior_verification_threshold),
            "high_risk_quantile": float(self.high_risk_quantile),
            "verification_channels": list(self.verification_channels),
            "apply_layer_operations": bool(self.apply_layer_operations),
            "layer_operation_count": int(len(layer_ops)),
            "layer_operations": layer_ops,
        }
        return {
            "pipeline_mode": self.pipeline_mode,
            "feature_source": str(self.feature_source),
            "feature_metadata": dict(self.feature_metadata),
            "apply_layer_operations": bool(self.apply_layer_operations),
            "layer_operation_count": int(len(layer_ops)),
            "layer_operations": layer_ops,
            "input_marker_count": int(self.input_marker_count or len(marker_records)),
            "skipped_missing_behavior_count": int(self.skipped_missing_behavior_count),
            "marker_count": int(len(marker_records)),
            "exact_cluster_count": int(len(behavior_units)),
            "selected_representative_count": int(len(cluster_units)),
            "selected_candidate_count": int(len(cluster_units)),
            "total_clusters": int(len(clusters_output)),
            "total_samples": int(len(file_list)),
            "total_files": int(len(file_list)),
            "cluster_sizes": cluster_sizes,
            "behavior_stats": dict(self.behavior_stats),
            "behavior_verification_stats": dict(self.behavior_verification_stats),
            "final_verification_stats": dict(self.behavior_verification_stats),
            "clusters": clusters_output,
            "file_list": file_list,
            "file_metadata": file_metadata,
            "result_summary": {
                "pipeline_mode": self.pipeline_mode,
                "marker_count": int(len(marker_records)),
                "exact_cluster_count": int(len(behavior_units)),
                "selected_representative_count": int(len(cluster_units)),
                "total_clusters": int(len(clusters_output)),
                "cluster_sizes": cluster_sizes,
                "behavior_stats": dict(self.behavior_stats),
                "behavior_verification_stats": dict(self.behavior_verification_stats),
                "timing_seconds": dict(runtime_summary),
                "config": dict(config),
            },
            "config": config,
            "cluster_review": {},
        }


def _write_text_summary(result: Dict[str, Any], output_path: Path) -> None:
    """把 JSON 结果压缩成中文 TXT 摘要，方便快速查看关键统计。"""
    cluster_sizes = sorted((int(value) for value in result.get("cluster_sizes", [])), reverse=True)[:10]
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("Lithography Behavior Coverage 聚类结果摘要\n")
        handle.write("=" * 48 + "\n")
        handle.write(f"pipeline mode: {result.get('pipeline_mode')}\n")
        handle.write(f"marker/sample 数: {result.get('marker_count')} / {result.get('total_samples')}\n")
        handle.write(f"exact cluster 数: {result.get('exact_cluster_count')}\n")
        handle.write(f"selected representative 数: {result.get('selected_representative_count')}\n")
        handle.write(f"total cluster 数: {result.get('total_clusters')}\n")
        handle.write(f"top cluster sizes: {cluster_sizes}\n")
        handle.write("\nbehavior stats:\n")
        handle.write(json.dumps(result.get("behavior_stats", {}), ensure_ascii=False, indent=2, default=_json_default))
        handle.write("\nbehavior verification stats:\n")
        handle.write(json.dumps(result.get("behavior_verification_stats", {}), ensure_ascii=False, indent=2, default=_json_default))
        handle.write("\nconfig:\n")
        handle.write(json.dumps(result.get("config", {}), ensure_ascii=False, indent=2, default=_json_default))
        handle.write("\n")


def _save_results(result: Dict[str, Any], output_path: str, output_format: str) -> None:
    """根据 `--format` 保存 JSON 或 TXT 输出文件。"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if str(output_format).lower() == "txt":
        _write_text_summary(result, output)
    else:
        with output.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False, default=_json_default)
    print(f"结果已保存到: {output}")


def _parse_diff_channels(value: str | Sequence[str] | None) -> Tuple[str, ...]:
    """解析 `--export-diff-channels`，只允许 aerial/resist/pv 三类 review diff。"""
    if not value:
        return tuple()
    if isinstance(value, str):
        items = [item.strip().lower() for item in value.split(",") if item.strip()]
    else:
        items = [str(item).strip().lower() for item in value if str(item).strip()]
    invalid = [item for item in items if item not in DIFF_CHANNELS]
    if invalid:
        raise ValueError(f"Unsupported diff channels: {invalid}; supported: {DIFF_CHANNELS}")
    return tuple(dict.fromkeys(items))


def _export_review(result: Dict[str, Any], review_dir: str, *, diff_channels: Sequence[str] = ()) -> Dict[str, Any]:
    """导出人工 review 目录，复制 representative/member clip 并可选生成 diff NPZ。"""
    review_root = Path(review_dir)
    review_root.mkdir(parents=True, exist_ok=True)
    representative_files: List[str] = []
    exported_file_count = 0
    missing_files: List[str] = []
    diff_count = 0

    for cluster in result.get("clusters", []):
        cluster_id = int(cluster["cluster_id"])
        cluster_size = int(cluster["size"])
        cluster_dir = review_root / f"cluster_{cluster_id:04d}_size_{cluster_size:04d}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        representative_path = str(cluster.get("representative_file", ""))
        representative_files.append(representative_path)
        if representative_path:
            rep_src = Path(representative_path)
            if rep_src.exists():
                shutil.copy2(rep_src, cluster_dir / f"REP__selected__{rep_src.name}")
                exported_file_count += 1
            else:
                missing_files.append(str(rep_src))

        for member_idx, src in enumerate(cluster.get("sample_files", [])):
            src_path = Path(src)
            if src_path.exists():
                shutil.copy2(src_path, cluster_dir / f"sample__{member_idx:04d}__{src_path.name}")
                exported_file_count += 1
            else:
                missing_files.append(str(src_path))

        with (cluster_dir / "behavior_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "representative_metadata": cluster.get("representative_metadata", {}),
                    "sample_metadata": cluster.get("sample_metadata", []),
                },
                handle,
                indent=2,
                ensure_ascii=False,
                default=_json_default,
            )

        rep_paths = cluster.get("representative_metadata", {}).get("behavior_paths", {})
        for member_idx, metadata in enumerate(cluster.get("sample_metadata", [])):
            member_paths = metadata.get("behavior_paths", {})
            marker_id = str(metadata.get("marker_id", f"member_{member_idx:04d}"))
            for channel in diff_channels:
                if channel not in rep_paths or channel not in member_paths:
                    continue
                rep_img = _load_npz_image(rep_paths[channel])
                member_img = _load_npz_image(member_paths[channel])
                if rep_img.shape != member_img.shape:
                    continue
                diff_path = cluster_dir / f"diff__{channel}__{member_idx:04d}__{marker_id}.npz"
                np.savez_compressed(diff_path, image=np.abs(rep_img - member_img).astype(np.float32))
                diff_count += 1

    with (review_root / "representative_files.txt").open("w", encoding="utf-8") as handle:
        for filepath in representative_files:
            handle.write(f"{filepath}\n")

    info = {
        "exported": True,
        "review_dir": str(review_root),
        "cluster_count": int(len(result.get("clusters", []))),
        "exported_file_count": int(exported_file_count),
        "representative_file_count": int(len(representative_files)),
        "missing_file_count": int(len(missing_files)),
        "diff_channels": list(diff_channels),
        "diff_file_count": int(diff_count),
    }
    if missing_files:
        info["missing_files_preview"] = missing_files[:10]
    result["cluster_review"] = dict(info)
    return info



def _print_start_banner(args: argparse.Namespace, *, apply_layer_operations: bool, layer_ops: Sequence[Dict[str, str]]) -> None:
    """打印 no-training 入口的中文启动信息，展示输入、特征来源和关键参数。"""
    print("=" * 72)
    print("Lithography Behavior Coverage no-training clustering")
    print("=" * 72)
    print(f"输入路径: {args.input_path}")
    print(f"输出路径: {args.output}")
    print(f"输出格式: {args.format}")
    print(f"marker layer: {args.marker_layer}")
    print(f"clip size: {args.clip_size} um")
    print(f"behavior manifest: {args.behavior_manifest}")
    print(f"feature source: handcrafted DCT/FFT/HOG + behavior stats + layout/WL")
    print(f"ANN top-K: {args.ann_top_k}")
    print(f"coverage target: {args.coverage_target}")
    print(f"behavior verification threshold: {args.behavior_verification_threshold}")
    print(f"diff channels: {args.export_diff_channels or '未启用'}")
    print(f"层操作启用: {'是' if apply_layer_operations else '否'}")
    if layer_ops:
        print(f"层操作规则数: {len(layer_ops)}")
        for idx, rule in enumerate(layer_ops, start=1):
            print(f"  规则 {idx}: {rule['source_layer']} {rule['operation']} {rule['target_layer']} -> {rule['result_layer']}")
    print("=" * 72)


def _build_parser() -> argparse.ArgumentParser:
    """构建 no-training 主脚本 CLI，并在帮助信息中给出用法示例和注意点。"""
    epilog = """
示例:

1) 基本 no-training 行为聚类
python layout_clustering_optimized_notrain.py ./input.oas --marker-layer 999/0 --behavior-manifest behavior.jsonl --output results_notrain.json

2) 导出 review 目录，便于人工检查 representative/member
python layout_clustering_optimized_notrain.py ./input.oas --marker-layer 999/0 --behavior-manifest behavior.jsonl --review-dir review_notrain --output results_notrain.json

3) 导出 aerial/resist/PV diff NPZ
python layout_clustering_optimized_notrain.py ./input.oas --marker-layer 999/0 --behavior-manifest behavior.jsonl --review-dir review_notrain --export-diff-channels aerial,resist,pv

4) 启用 layer boolean 操作，先生成新层再采样和提取手工 FV
python layout_clustering_optimized_notrain.py ./input.oas --marker-layer 999/0 --behavior-manifest behavior.jsonl --register-op 1/0 2/0 subtract 10/0 --output results_notrain.json

5) 调整 coverage 与 verification 严格度
python layout_clustering_optimized_notrain.py ./input.oas --marker-layer 999/0 --behavior-manifest behavior.jsonl --ann-top-k 96 --coverage-target 0.99 --behavior-verification-threshold 0.06

注意:
- 本脚本不需要用户提供外部 feature NPZ，会自动生成 handcrafted FV。
- behavior manifest 必须提供 aerial_npz；pv/epe/nils 若全局可用，会参与 final verification。
- handcrafted FV 包含 aerial 频谱/梯度、optional behavior stats、layout geometry 和 WL topology。
- exact hash 只用于完全重复样本去重和权重累计，不作为最终 cluster 语义。
- final verification 失败的 member 会形成 singleton/base representative，不做跨 cluster 自动拯救。
- `--register-op` 会自动启用 layer operations；层格式固定为 layer/datatype，例如 1/0。
"""
    parser = argparse.ArgumentParser(
        description="marker-driven no-training 光刻行为 coverage 聚类",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_path", help="输入 OASIS 文件或目录")
    parser.add_argument("--output", "-o", default="optimized_notrain_results.json", help="输出文件路径")
    parser.add_argument("--format", "-f", choices=("json", "txt"), default="json", help="输出格式")
    parser.add_argument("--marker-layer", required=True, help="marker 层，格式 layer/datatype，例如 999/0")
    parser.add_argument("--clip-size", type=float, default=1.35, help="marker-centered clip 边长，单位 um")
    parser.add_argument("--behavior-manifest", required=True, help="behavior JSONL manifest 路径，或 preprocess_notrain.py 输出目录")
    parser.add_argument("--ann-top-k", type=int, default=64, help="ANN top-K graph 的近邻数")
    parser.add_argument("--coverage-target", type=float, default=0.985, help="weighted facility location coverage 目标")
    parser.add_argument("--facility-min-gain", type=float, default=1e-4, help="facility selection 最小边际收益")
    parser.add_argument("--behavior-verification-threshold", type=float, default=0.08, help="behavior weighted distance 通过阈值")
    parser.add_argument("--high-risk-quantile", type=float, default=0.90, help="k-center 补洞使用的 high-risk 分位数")
    parser.add_argument("--export-diff-channels", default="", help="可选 diff channel，逗号分隔: aerial,resist,pv")
    parser.add_argument("--review-dir", default=None, help="可选 review 导出目录")
    parser.add_argument("--export-cluster-review-dir", default=None, help="兼容旧版的 review 目录参数别名")
    parser.add_argument("--apply-layer-ops", action="store_true", help="聚类前应用注册的 boolean layer operations")
    parser.add_argument(
        "--register-op",
        action="append",
        nargs=4,
        metavar=("SOURCE_LAYER", "TARGET_LAYER", "OPERATION", "RESULT_LAYER"),
        help="注册层操作规则，例如 --register-op 1/0 2/0 subtract 10/0",
    )
    return parser


def run_notrain(args: argparse.Namespace) -> Dict[str, Any]:
    """执行 no-training 全流程：生成手工 FV，并调用本脚本内置 runner 完成聚类。"""
    register_ops = args.register_op or []
    apply_layer_operations = bool(args.apply_layer_ops or register_ops)
    layer_processor = _make_layer_processor(register_ops)
    layer_ops = _layer_operation_payload(layer_processor)
    diff_channels = _parse_diff_channels(args.export_diff_channels)
    behavior_manifest_path = _resolve_behavior_manifest_path(args.behavior_manifest)
    _print_start_banner(args, apply_layer_operations=apply_layer_operations, layer_ops=layer_ops)

    temp_root = Path(__file__).resolve().parent / "_temp_runs"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"layout_clustering_notrain_{uuid.uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    feature_npz = temp_dir / "handcraft_features.npz"
    feature_metadata_json = temp_dir / "handcraft_features.metadata.json"

    print("开始生成 handcrafted feature vectors...")
    feature_metadata = encode_handcrafted_features(
        input_path=str(args.input_path),
        marker_layer=str(args.marker_layer),
        behavior_manifest=str(behavior_manifest_path),
        features_out=feature_npz,
        metadata_out=feature_metadata_json,
        clip_size_um=float(args.clip_size),
        pixel_size_nm=DEFAULT_PIXEL_SIZE_NM,
        apply_layer_operations=apply_layer_operations,
        layer_processor=layer_processor,
    )
    print(
        "handcrafted FV 生成完成: "
        f"samples={feature_metadata.get('feature_count')}, dim={feature_metadata.get('feature_dim')}"
    )

    config = {
        "feature_metadata": feature_metadata,
        "marker_layer": str(args.marker_layer),
        "clip_size_um": float(args.clip_size),
        "pixel_size_nm": DEFAULT_PIXEL_SIZE_NM,
        "behavior_manifest": str(behavior_manifest_path),
        "feature_npz": str(feature_npz),
        "ann_top_k": int(args.ann_top_k),
        "coverage_target": float(args.coverage_target),
        "facility_min_gain": float(args.facility_min_gain),
        "behavior_verification_threshold": float(args.behavior_verification_threshold),
        "high_risk_quantile": float(args.high_risk_quantile),
        "export_diff_channels": diff_channels,
        "apply_layer_operations": apply_layer_operations,
    }
    runner = NotrainOptimizedRunner(
        config=config,
        temp_dir=temp_dir,
        layer_processor=layer_processor if apply_layer_operations else None,
    )
    result = runner.run(str(args.input_path))
    result["handcraft_feature_npz"] = str(feature_npz)
    result["handcraft_feature_metadata_json"] = str(feature_metadata_json)
    return result


def main() -> int:
    """命令行入口：解析参数、执行 no-training 聚类、保存结果并可选导出 review。"""
    parser = _build_parser()
    args = parser.parse_args()
    try:
        result = run_notrain(args)
        review_dir = _review_dir_from_args(args)
        diff_channels = _parse_diff_channels(args.export_diff_channels)
        if review_dir:
            info = _export_review(result, str(review_dir), diff_channels=diff_channels)
            print(f"cluster review 目录已导出到: {info.get('review_dir', review_dir)}")
        _save_results(result, str(args.output), str(args.format))
        print(f"最终 cluster 数: {result.get('total_clusters', 0)}")
        print(f"最终 marker/sample 数: {result.get('marker_count', 0)} / {result.get('total_samples', 0)}")
        print(f"behavior verification: {result.get('behavior_verification_stats', {})}")
        return 0
    except Exception as exc:
        print(f"运行失败: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
