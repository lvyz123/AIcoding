#!/usr/bin/env python3
"""CD-SEM recipe outcome 的轻量 pattern memory 导出与审查。

本模块实现 Zheng 2025 pattern database 思路的第一步：把单次 recipe run 中已经产生的
MP candidate、care-area provenance、ring-context 和 recipe outcome 压缩写入磁盘。
它不是在线检索数据库，也不把历史 prior 接入当前 selector 排序。当前目标是
低内存、append/export-friendly 的 evidence artifact，并提供只读 nearest-neighbor
audit prior，供后续跨版图复用和人工分析。

整体流程:
1. 从 `mp_candidate_pool.json` 级别的候选摘要中抽取 compact metadata。
2. 将 bitmap fingerprint 与 ring-context 向量写入 `vectors.npz`。
3. 将 provenance / outcome / vector_index 写入 `records.jsonl`。
4. 可选读取持久化 store，为当前候选生成中性默认的历史 outcome prior audit。

注意: records 中不保存完整 bitmap、OAS clip 或大数组；大向量统一放在 NPZ 中。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from ring_context import ring_context_vector, select_nonredundant_radii


STORE_SCHEMA_VERSION = "pattern_memory_store_v1"
MEMORY_PRIOR_SCHEMA_VERSION = "pattern_memory_prior_audit_v1"


def _json_default(value: Any) -> Any:
    """把 numpy/path 等对象转换成 JSON 可序列化类型。"""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _compact_outcome(candidate_id: str, rows_by_candidate_id: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """从 recipe row 中提取与 recipe 可执行性相关的 outcome。"""
    row = rows_by_candidate_id.get(str(candidate_id), {})
    return {
        "recipe_status": str(row.get("recipe_status", "")),
        "reject_reason": str(row.get("reject_reason", "")),
        "refine_failed": "post_selection_refine_failed" in str(row.get("reject_reason", "")),
        "af_pass": bool(row.get("af_oas")),
        "af_reject_reason": str(row.get("af_reject_reason", "")),
        "ap_pass": bool(row.get("ap_oas")) and not bool(row.get("ap_global_duplicate")),
        "ap_reject_reason": str(row.get("ap_reject_reason", "")),
        "ap_global_duplicate": bool(row.get("ap_global_duplicate")),
        "ap_global_duplicate_with": str(row.get("ap_global_duplicate_with", "")),
    }


def _candidate_vector(candidate: Mapping[str, Any]) -> np.ndarray:
    """拼接 bitmap fingerprint 与 ring-context audit 向量。"""
    fingerprint = np.asarray(candidate.get("bitmap_fingerprint", []) or [], dtype=np.float32)
    ring_vector = ring_context_vector(candidate.get("ring_context", {}) or {})
    if fingerprint.size == 0:
        return ring_vector.astype(np.float32, copy=False)
    if ring_vector.size == 0:
        return fingerprint.astype(np.float32, copy=False)
    return np.concatenate([fingerprint, ring_vector]).astype(np.float32, copy=False)


def _neutral_memory_prior() -> Dict[str, Any]:
    """生成 store 为空或无近邻时的中性历史先验。"""
    return {
        "memory_neighbor_count": 0,
        "memory_nearest_similarity": 0.0,
        "memory_avg_similarity": 0.0,
        "memory_recipe_success_prior": 0.5,
        "memory_refine_fail_prior": 0.5,
        "memory_af_success_prior": 0.5,
        "memory_ap_success_prior": 0.5,
        "memory_ap_duplicate_prior": 0.5,
        "memory_waste_prior": 0.5,
        "memory_prior_confidence": 0.0,
    }


def _normalized_matrix(matrix: np.ndarray) -> np.ndarray:
    """返回按行 L2 归一化后的矩阵，零向量保持为零。"""
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return np.divide(arr, np.maximum(norms, 1e-12), out=np.zeros_like(arr, dtype=np.float32), where=norms > 0)


def build_memory_prior_audit(
    mp_candidate_pool: Sequence[Mapping[str, Any]],
    *,
    store_dir: Path,
    top_k: int = 5,
    min_similarity: float = 0.70,
) -> Dict[str, Any]:
    """基于持久化 pattern memory store，为当前候选生成只读 outcome prior audit。"""
    store_dir = Path(store_dir)
    records = _read_records_jsonl(store_dir / "records.jsonl")
    store_vectors, _ = _load_vectors_npz(store_dir / "vectors.npz")
    if not records or store_vectors.size == 0:
        by_candidate = {
            str(candidate.get("mp_candidate_id", "")): _neutral_memory_prior()
            for candidate in mp_candidate_pool
            if str(candidate.get("mp_candidate_id", ""))
        }
        return {
            "schema_version": MEMORY_PRIOR_SCHEMA_VERSION,
            "summary": {
                "candidate_count": int(len(mp_candidate_pool)),
                "store_record_count": int(len(records)),
                "candidates_with_neighbors": 0,
                "avg_neighbor_count": 0.0,
                "avg_prior_confidence": 0.0,
            },
            "by_candidate_id": by_candidate,
        }

    store_items: list[tuple[Mapping[str, Any], np.ndarray]] = []
    for record_index, record in enumerate(records):
        vector_index = int(record.get("vector_index", record_index) or record_index)
        if 0 <= vector_index < int(store_vectors.shape[0]):
            store_items.append((record, np.asarray(store_vectors[vector_index], dtype=np.float32)))
    current_vectors = [_candidate_vector(candidate) for candidate in mp_candidate_pool]
    vector_dim = max(
        [int(store_vectors.shape[1]) if store_vectors.ndim == 2 else 0]
        + [int(vector.size) for vector in current_vectors]
        + [1]
    )
    store_matrix = _pad_matrix([item[1] for item in store_items], vector_dim)
    store_norm = _normalized_matrix(store_matrix)

    by_candidate: Dict[str, Dict[str, Any]] = {}
    neighbor_counts: list[int] = []
    confidences: list[float] = []
    for candidate, vector in zip(mp_candidate_pool, current_vectors):
        candidate_id = str(candidate.get("mp_candidate_id", ""))
        if not candidate_id:
            continue
        prior = _neutral_memory_prior()
        current = _pad_matrix([vector], vector_dim)
        current_norm = _normalized_matrix(current)
        if store_norm.size and current_norm.size and float(np.linalg.norm(current_norm[0])) > 0.0:
            sims = (store_norm @ current_norm[0]).astype(np.float32)
            for index, (record, _) in enumerate(store_items):
                if str(record.get("mp_candidate_id", "")) == candidate_id:
                    sims[index] = -1.0
            valid_indices = np.where(sims >= float(min_similarity))[0]
            if valid_indices.size:
                ordered = sorted(valid_indices.tolist(), key=lambda idx: float(sims[idx]), reverse=True)[: int(top_k)]
                neighbor_records = [store_items[index][0] for index in ordered]
                neighbor_sims = [float(sims[index]) for index in ordered]
                flags = [_outcome_flags(record) for record in neighbor_records]
                total = len(flags)
                selected = sum(int(item["selected"]) for item in flags)
                refine_fail = sum(int(item["refine_fail"]) for item in flags)
                af_pass = sum(int(item["af_pass"]) for item in flags)
                ap_pass = sum(int(item["ap_pass"]) for item in flags)
                ap_duplicate = sum(int(item["ap_duplicate"]) for item in flags)
                waste = sum(int(item["high_waste"]) for item in flags)
                avg_similarity = float(np.mean(neighbor_sims))
                prior = {
                    "memory_neighbor_count": int(total),
                    "memory_nearest_similarity": float(max(neighbor_sims)),
                    "memory_avg_similarity": float(avg_similarity),
                    "memory_recipe_success_prior": _beta_prior(selected, total),
                    "memory_refine_fail_prior": _beta_prior(refine_fail, total),
                    "memory_af_success_prior": _beta_prior(af_pass, total),
                    "memory_ap_success_prior": _beta_prior(ap_pass, total),
                    "memory_ap_duplicate_prior": _beta_prior(ap_duplicate, total),
                    "memory_waste_prior": _beta_prior(waste, total),
                    "memory_prior_confidence": float(min(1.0, (float(total) / float(max(1, int(top_k)))) * avg_similarity)),
                }
        by_candidate[candidate_id] = prior
        neighbor_counts.append(int(prior["memory_neighbor_count"]))
        confidences.append(float(prior["memory_prior_confidence"]))

    return {
        "schema_version": MEMORY_PRIOR_SCHEMA_VERSION,
        "summary": {
            "candidate_count": int(len(mp_candidate_pool)),
            "store_record_count": int(len(records)),
            "candidates_with_neighbors": int(sum(1 for value in neighbor_counts if value > 0)),
            "avg_neighbor_count": float(np.mean(neighbor_counts)) if neighbor_counts else 0.0,
            "avg_prior_confidence": float(np.mean(confidences)) if confidences else 0.0,
        },
        "by_candidate_id": by_candidate,
    }


def _vector_hash(vector: np.ndarray) -> str:
    """计算 compact vector 的稳定哈希，用于 store 级轻量去重。"""
    arr = np.asarray(vector, dtype=np.float32)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def _record_dedup_key(record: Mapping[str, Any], vector_hash: str) -> str:
    """按 schema、candidate provenance 和 vector hash 生成去重 key。"""
    parts = [
        str(record.get("schema_version", "")),
        str(record.get("mp_candidate_id", "")),
        str(record.get("care_area_family_id", "")),
        str(record.get("care_area_instance_id", "")),
        str(record.get("care_area_type", "")),
        str(vector_hash),
    ]
    return "|".join(parts)


def _read_records_jsonl(path: Path) -> list[Dict[str, Any]]:
    """顺序读取 JSONL records；文件不存在时返回空列表。"""
    if not Path(path).exists():
        return []
    records: list[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                records.append(json.loads(text))
    return records


def _write_records_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """顺序写出 JSONL records。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def _load_vectors_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取 compact vectors；文件不存在时返回空矩阵。"""
    if not Path(path).exists():
        return np.zeros((0, 0), dtype=np.float32), np.asarray([], dtype=str)
    with np.load(path, allow_pickle=False) as data:
        vectors = np.asarray(data["vectors"], dtype=np.float32)
        candidate_ids = np.asarray(data["candidate_ids"], dtype=str)
    return vectors, candidate_ids


def _pad_matrix(vectors: Sequence[np.ndarray], vector_dim: int) -> np.ndarray:
    """把不同长度的 compact vector 补齐成统一矩阵。"""
    matrix = np.zeros((len(vectors), int(vector_dim)), dtype=np.float32)
    for index, vector in enumerate(vectors):
        arr = np.asarray(vector, dtype=np.float32)
        if arr.size:
            matrix[index, : min(int(vector_dim), int(arr.size))] = arr[: int(vector_dim)]
    return matrix


def _candidate_record(
    candidate: Mapping[str, Any],
    *,
    vector_index: int,
    vector_dim: int,
    outcome: Mapping[str, Any],
) -> Dict[str, Any]:
    """生成 records.jsonl 中的一条 compact pattern outcome record。"""
    return {
        "schema_version": "pattern_memory_v1",
        "mp_candidate_id": str(candidate.get("mp_candidate_id", "")),
        "source_marker_id": str(candidate.get("source_marker_id", "")),
        "hotspot_cluster_id": int(candidate.get("hotspot_cluster_id", 0) or 0),
        "care_area_family_id": str(candidate.get("care_area_family_id", "")),
        "care_area_instance_id": str(candidate.get("care_area_instance_id", "")),
        "care_area_type": str(candidate.get("care_area_type", "")),
        "care_area_match_score": float(candidate.get("care_area_match_score", 0.0) or 0.0),
        "care_area_homogeneity_score": float(candidate.get("care_area_homogeneity_score", 0.0) or 0.0),
        "metrology_context_group_id": str(candidate.get("metrology_context_group_id", "")),
        "metrology_priority_class": str(candidate.get("metrology_priority_class", "")),
        "metrology_priority_score": float(candidate.get("metrology_priority_score", 0.0) or 0.0),
        "site_reliability_risk": float(candidate.get("site_reliability_risk", 0.0) or 0.0),
        "recipe_waste_penalty": float(candidate.get("recipe_waste_penalty", 0.0) or 0.0),
        "pool_status": str(candidate.get("pool_status", "")),
        "pool_reject_reason": str(candidate.get("pool_reject_reason", "")),
        "mp_candidate_rank": int(candidate.get("mp_candidate_rank", 0) or 0),
        "mp_candidate_type": str(candidate.get("mp_candidate_type", "")),
        "mp_verified": bool(candidate.get("mp_verified", False)),
        "mp_reject_reason": str(candidate.get("mp_reject_reason", "")),
        "mp_hotspot_score": float(candidate.get("mp_hotspot_score", 0.0) or 0.0),
        "mp_priority_score": float(candidate.get("mp_priority_score", 0.0) or 0.0),
        "mp_selection_gain": float(candidate.get("mp_selection_gain", 0.0) or 0.0),
        "mp_center_um": [float(candidate.get("mp_x_um", 0.0) or 0.0), float(candidate.get("mp_y_um", 0.0) or 0.0)],
        "clip_bbox": [float(value) for value in candidate.get("clip_bbox", []) or []],
        "ring_context": dict(candidate.get("ring_context", {}) or {}),
        "outcome": dict(outcome),
        "vector_index": int(vector_index),
        "vector_dim": int(vector_dim),
    }


def export_pattern_memory(
    *,
    mp_candidate_pool: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    """把本次 run 的 MP candidate outcome 导出为轻量 pattern memory artifact。"""
    export_dir = Path(output_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    records_path = export_dir / "records.jsonl"
    vectors_path = export_dir / "vectors.npz"

    rows_by_candidate_id = {
        str(row.get("mp_candidate_id", "")): row
        for row in rows
        if str(row.get("mp_candidate_id", ""))
    }
    vectors = [_candidate_vector(candidate) for candidate in mp_candidate_pool]
    vector_dim = max((int(vector.size) for vector in vectors), default=0)
    matrix = np.zeros((len(vectors), vector_dim), dtype=np.float32)
    for index, vector in enumerate(vectors):
        if vector.size:
            matrix[index, : int(vector.size)] = vector.astype(np.float32, copy=False)

    candidate_ids = np.asarray([str(candidate.get("mp_candidate_id", "")) for candidate in mp_candidate_pool], dtype=str)
    np.savez_compressed(vectors_path, vectors=matrix, candidate_ids=candidate_ids)

    with records_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, candidate in enumerate(mp_candidate_pool):
            candidate_id = str(candidate.get("mp_candidate_id", ""))
            record = _candidate_record(
                candidate,
                vector_index=index,
                vector_dim=vector_dim,
                outcome=_compact_outcome(candidate_id, rows_by_candidate_id),
            )
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")

    total_bytes = int(records_path.stat().st_size + vectors_path.stat().st_size)
    return {
        "pattern_memory_export_dir": str(export_dir),
        "pattern_memory_records_jsonl": str(records_path),
        "pattern_memory_vectors_npz": str(vectors_path),
        "pattern_memory_record_count": int(len(mp_candidate_pool)),
        "pattern_memory_vector_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "pattern_memory_estimated_disk_bytes": int(total_bytes),
    }


def append_pattern_memory_export(
    *,
    export_dir: Path,
    store_dir: Path,
) -> Dict[str, Any]:
    """把单次 run 的 pattern memory export 合并进持久化 store。"""
    export_dir = Path(export_dir)
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    export_records_path = export_dir / "records.jsonl"
    export_vectors_path = export_dir / "vectors.npz"
    store_records_path = store_dir / "records.jsonl"
    store_vectors_path = store_dir / "vectors.npz"
    manifest_path = store_dir / "manifest.json"
    memory_audit_path = store_dir / "memory_audit.json"
    ring_audit_path = store_dir / "ring_outcome_audit.json"

    existing_records = _read_records_jsonl(store_records_path)
    existing_vectors, existing_candidate_ids = _load_vectors_npz(store_vectors_path)
    existing_vector_list = [np.asarray(existing_vectors[index], dtype=np.float32) for index in range(existing_vectors.shape[0])]
    existing_keys = {
        _record_dedup_key(record, str(record.get("vector_hash", "")))
        for record in existing_records
    }

    export_records = _read_records_jsonl(export_records_path)
    export_vectors, _ = _load_vectors_npz(export_vectors_path)
    added_records: list[Dict[str, Any]] = []
    vector_list = list(existing_vector_list)
    candidate_ids = [str(value) for value in existing_candidate_ids.tolist()]
    vector_dim = max(
        [int(existing_vectors.shape[1]) if existing_vectors.ndim == 2 else 0]
        + [int(export_vectors.shape[1]) if export_vectors.ndim == 2 else 0]
    )

    for export_index, record in enumerate(export_records):
        vector = export_vectors[export_index] if export_index < export_vectors.shape[0] else np.zeros((0,), dtype=np.float32)
        vector_hash = _vector_hash(vector)
        key = _record_dedup_key(record, vector_hash)
        if key in existing_keys:
            continue
        new_record = dict(record)
        new_record["store_schema_version"] = STORE_SCHEMA_VERSION
        new_record["export_vector_index"] = int(record.get("vector_index", export_index) or export_index)
        new_record["vector_index"] = int(len(vector_list))
        new_record["vector_hash"] = str(vector_hash)
        existing_keys.add(key)
        added_records.append(new_record)
        vector_list.append(np.asarray(vector, dtype=np.float32))
        candidate_ids.append(str(new_record.get("mp_candidate_id", "")))
        vector_dim = max(int(vector_dim), int(np.asarray(vector).size))

    all_records = list(existing_records) + added_records
    matrix = _pad_matrix(vector_list, vector_dim)
    np.savez_compressed(store_vectors_path, vectors=matrix, candidate_ids=np.asarray(candidate_ids, dtype=str))
    _write_records_jsonl(store_records_path, all_records)

    memory_audit = build_memory_audit(all_records)
    ring_audit = build_ring_outcome_audit(all_records)
    with memory_audit_path.open("w", encoding="utf-8") as handle:
        json.dump(memory_audit, handle, indent=2, ensure_ascii=False, default=_json_default)
    with ring_audit_path.open("w", encoding="utf-8") as handle:
        json.dump(ring_audit, handle, indent=2, ensure_ascii=False, default=_json_default)

    total_bytes = sum(
        int(path.stat().st_size)
        for path in (store_records_path, store_vectors_path, memory_audit_path, ring_audit_path)
        if path.exists()
    )
    manifest = {
        "schema_version": STORE_SCHEMA_VERSION,
        "store_dir": str(store_dir),
        "records_jsonl": str(store_records_path),
        "vectors_npz": str(store_vectors_path),
        "memory_audit_json": str(memory_audit_path),
        "ring_outcome_audit_json": str(ring_audit_path),
        "record_count": int(len(all_records)),
        "added_record_count": int(len(added_records)),
        "duplicate_skipped_count": int(max(0, len(export_records) - len(added_records))),
        "vector_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "estimated_disk_bytes": int(total_bytes),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, default=_json_default)
    return dict(manifest)


def _outcome_flags(record: Mapping[str, Any]) -> Dict[str, bool]:
    """把 record outcome 归一化为 recipe feasibility flags。"""
    outcome = dict(record.get("outcome", {}) or {})
    recipe_status = str(outcome.get("recipe_status", ""))
    reject_reason = str(outcome.get("reject_reason", ""))
    return {
        "selected": recipe_status == "selected",
        "refine_fail": bool(outcome.get("refine_failed", False)) or "post_selection_refine_failed" in reject_reason,
        "af_pass": bool(outcome.get("af_pass", False)),
        "af_fail": "no_safe_af" in reject_reason or bool(outcome.get("af_reject_reason", "")),
        "ap_pass": bool(outcome.get("ap_pass", False)),
        "ap_fail": "no_unique_ap" in reject_reason or bool(outcome.get("ap_reject_reason", "")),
        "ap_duplicate": bool(outcome.get("ap_global_duplicate", False)) or "ap_global_duplicate" in reject_reason,
        "high_waste": recipe_status != "selected",
    }


def _beta_prior(success_count: int, total_count: int, *, alpha: float = 1.0, beta: float = 1.0) -> float:
    """计算 Beta-Binomial 平滑后的成功率先验，仅用于 audit。"""
    return float((float(success_count) + float(alpha)) / (float(total_count) + float(alpha) + float(beta)))


def _group_audit(records: Sequence[Mapping[str, Any]], group_key: str) -> Dict[str, Any]:
    """按指定字段统计 recipe outcome 分布和 audit prior。"""
    groups: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get(group_key, "") or "unknown")].append(record)
    audit: Dict[str, Any] = {}
    for key, items in sorted(groups.items()):
        reason_counts: Counter[str] = Counter()
        selected = af_pass = ap_pass = refine_fail = af_fail = ap_fail = ap_duplicate = 0
        for record in items:
            flags = _outcome_flags(record)
            outcome = dict(record.get("outcome", {}) or {})
            selected += int(flags["selected"])
            af_pass += int(flags["af_pass"])
            ap_pass += int(flags["ap_pass"])
            refine_fail += int(flags["refine_fail"])
            af_fail += int(flags["af_fail"])
            ap_fail += int(flags["ap_fail"])
            ap_duplicate += int(flags["ap_duplicate"])
            reason = str(outcome.get("reject_reason", ""))
            for part in reason.split(";"):
                if part.strip():
                    reason_counts[part.strip()] += 1
        total = len(items)
        waste_count = total - selected
        audit[key] = {
            "record_count": int(total),
            "selected_count": int(selected),
            "refine_fail_count": int(refine_fail),
            "af_fail_count": int(af_fail),
            "ap_fail_count": int(ap_fail),
            "ap_duplicate_count": int(ap_duplicate),
            "reject_reasons": dict(reason_counts),
            "recipe_success_prior": _beta_prior(selected, total),
            "recipe_waste_prior": _beta_prior(waste_count, total),
            "af_success_prior": _beta_prior(af_pass, total),
            "ap_success_prior": _beta_prior(ap_pass, total),
        }
    return audit


def build_memory_audit(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """生成 pattern memory store 的 recipe feasibility 统计审查。"""
    records = list(records)
    return {
        "schema_version": "pattern_memory_audit_v1",
        "summary": {
            "record_count": int(len(records)),
            "selected_count": int(sum(1 for record in records if _outcome_flags(record)["selected"])),
            "rejected_count": int(sum(1 for record in records if not _outcome_flags(record)["selected"])),
        },
        "by_care_area_type": _group_audit(records, "care_area_type"),
        "by_metrology_context_group_id": _group_audit(records, "metrology_context_group_id"),
        "by_mp_candidate_type": _group_audit(records, "mp_candidate_type"),
    }


def _binary_mi(values: Sequence[float], labels: Sequence[bool]) -> float:
    """用二分桶估计单个 ring feature 与二元 outcome 的互信息 proxy。"""
    if len(values) != len(labels) or len(values) < 2:
        return 0.0
    threshold = float(np.median(np.asarray(values, dtype=np.float32)))
    buckets = [1 if float(value) >= threshold else 0 for value in values]
    total = float(len(values))
    mi = 0.0
    for bucket in (0, 1):
        for label in (False, True):
            joint = sum(1 for b, l in zip(buckets, labels) if b == bucket and bool(l) == label)
            if joint <= 0:
                continue
            pxy = float(joint) / total
            px = float(sum(1 for b in buckets if b == bucket)) / total
            py = float(sum(1 for l in labels if bool(l) == label)) / total
            if px > 0.0 and py > 0.0:
                mi += pxy * math.log(pxy / (px * py), 2.0)
    return float(max(0.0, mi))


def _ring_profile_value(record: Mapping[str, Any], profile_key: str, radius_index: int) -> float:
    """读取单条 record 的指定 ring profile 值。"""
    context = dict(record.get("ring_context", {}) or {})
    profile = context.get(profile_key, []) or []
    if radius_index >= len(profile):
        return 0.0
    return float(profile[radius_index])


def build_ring_outcome_audit(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """基于 accumulated memory records 统计 ring radius 与 recipe outcome 的关系。"""
    records = [record for record in records if dict(record.get("ring_context", {}) or {}).get("ring_radii_um")]
    if not records:
        return {
            "schema_version": "ring_outcome_audit_v1",
            "summary": {"record_count": 0, "selected_radii_um": []},
            "radii": [],
        }
    radii = [float(value) for value in dict(records[0].get("ring_context", {}) or {}).get("ring_radii_um", [])]
    proxy_names = ("selected", "af_pass", "ap_pass", "ap_duplicate", "high_waste")
    radius_items: list[Dict[str, Any]] = []
    dp_scores: list[float] = []
    for radius_index, radius in enumerate(radii):
        density = [_ring_profile_value(record, "ring_density_profile", radius_index) for record in records]
        edge = [_ring_profile_value(record, "ring_edge_crossing_profile", radius_index) for record in records]
        asymmetry = [_ring_profile_value(record, "ring_asymmetry_profile", radius_index) for record in records]
        flags = [_outcome_flags(record) for record in records]
        proxy_mi: Dict[str, float] = {}
        for name in proxy_names:
            labels = [bool(flag[name]) for flag in flags]
            proxy_mi[name] = float(max(
                _binary_mi(density, labels),
                _binary_mi(edge, labels),
                _binary_mi(asymmetry, labels),
            ))
        score = float(sum(proxy_mi.values()) / float(len(proxy_mi)))
        dp_scores.append(score)
        radius_items.append(
            {
                "radius_um": float(radius),
                "sample_count": int(len(records)),
                "density_mean": float(np.mean(density)),
                "edge_crossing_mean": float(np.mean(edge)),
                "asymmetry_mean": float(np.mean(asymmetry)),
                "density_low_bin_count": int(sum(1 for value in density if float(value) < 0.33)),
                "density_mid_bin_count": int(sum(1 for value in density if 0.33 <= float(value) < 0.66)),
                "density_high_bin_count": int(sum(1 for value in density if float(value) >= 0.66)),
                "outcome_mi_proxy": proxy_mi,
                "mean_mi_proxy": float(score),
                "confidence": float(min(1.0, len(records) / 20.0)),
            }
        )
    selected = select_nonredundant_radii(radii, dp_scores, max_count=3, min_spacing_um=0.20)
    return {
        "schema_version": "ring_outcome_audit_v1",
        "summary": {
            "record_count": int(len(records)),
            "selected_radii_um": [float(value) for value in selected["selected_radii_um"]],
            "selected_indices": [int(value) for value in selected["selected_indices"]],
            "selected_proxy_score": float(selected["selected_proxy_score"]),
        },
        "radii": radius_items,
    }
