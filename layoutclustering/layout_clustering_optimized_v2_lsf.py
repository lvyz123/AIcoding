#!/usr/bin/env python3
"""面向 LSF 的 optimized v2 聚类入口。

中文整体算法流程与使用说明：
1. 本脚本是 optimized_v1 主算法的 LSF/Python 3.6 独立适配版。代码不 import v1、旧 mainline、
   旧 layout_utils 或旧 layer_operations，但 coverage 主线刻意对齐 v1：candidate bundle、
   cheap shortlist、lazy full GraphDescriptor prefilter、packed/dilated/donut geometry cache、
   greedy set cover 和 final verification。
2. 推荐集群流程为：
   - prepare：读取 OAS，应用可选 layer operation，生成 geometry-driven seeds 和 shard manifest。
   - run-shard：每个 LSF job 只处理一段 seed，输出 marker records 的 JSON/NPZ。
   - prepare-coverage：汇总 marker，生成 exact clusters、全局 candidate bundle buckets 和 coverage source shards。
   - run-coverage-shard：每个 LSF job 读取本 source shard 与需要的 candidate bundle bucket，计算 coverage CSR。
   - merge-coverage：汇总 coverage CSR，执行 greedy set cover，懒加载 selected candidate bitmap 后输出 compact JSON。
3. run-local 用于开发和小 crop 验证；默认走集中式小样本流程，带 --distributed-coverage 时顺序模拟完整 LSF 流程。
4. 代码保持 Python 3.6 兼容：不使用 dataclasses、现代 union type、内置泛型类型标注或 scipy.optimize.milp。

注意点：
- grid_step_ratio 固定保持 v1 主线默认值 0.5，当前版本不开放采样密度实验入口。
- 默认不物化 sample/representative 文件，主输出是 compact JSON。
- prepare-coverage 不 eager 构建 full descriptor 或 ECC geometry cache；这些缓存只在 coverage shard 内按需生成。
- candidate coverage set 使用 NPZ offsets/values 存储，JSON 只保留轻量索引和诊断字段。
- merge-coverage 的 greedy set cover 基于 CSR 数组运行，只把 selected candidates 还原成对象。
- inspect-workdir 只读取 manifest/JSON/zip metadata，用于评估 shard 文件规模和下一轮优化方向。
"""

import argparse
import hashlib
import json
import math
import os
try:
    import resource
except ImportError:
    resource = None
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

from layer_operations_lsf import LayerOperationProcessor
from mainline_lsf import (
    DEFAULT_PIXEL_SIZE_NM,
    GRID_STEP_RATIO,
    PIPELINE_MODE,
    bitmap_shape_key,
    build_compact_result,
    build_uniform_grid_seed_candidates,
    candidate_shift_summary,
    create_candidate_bundle_accumulator,
    coverage_fill_bin_for_bitmap,
    evaluate_candidate_coverage,
    evaluate_candidate_coverage_against_bundles,
    filter_layout_index_by_bbox,
    generate_candidates_for_cluster,
    generate_candidates_for_cluster_range,
    greedy_cover_csr,
    greedy_cover,
    group_exact_clusters,
    json_default,
    load_coverage_candidate_bitmaps,
    load_coverage_shard_csr_metadata,
    load_candidate_bundle_buckets_for_candidates,
    load_candidate_bundle_buckets_for_shapes,
    load_exact_index,
    load_shard_records,
    marker_query_candidate_stats,
    marker_record_from_seed,
    prepare_layout,
    add_candidates_to_candidate_bundle_accumulator,
    save_coverage_shard,
    save_candidate_bundle_index_from_accumulator,
    save_exact_index,
    save_shard_records,
    selected_candidates_from_csr,
    spatial_index_stats,
    write_layout_index_oas,
    GridSeedCandidate,
)


CENTRAL_MERGE_EXACT_CLUSTER_LIMIT = 20000
PREPARE_COVERAGE_CANDIDATE_CHUNK_SIZE = 2000
TARGET_LOAD_WARNING_RATIO = 0.60


def _ensure_dir(path):
    """确保目录存在。"""

    root = Path(str(path))
    if not root.exists():
        root.mkdir(parents=True)
    return root


def _read_json(path):
    """读取 JSON 文件。"""

    with Path(str(path)).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path, payload):
    """写出 JSON 文件。"""

    parent = Path(str(path)).parent
    if not parent.exists():
        parent.mkdir(parents=True)
    with Path(str(path)).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=json_default)


def _write_text(path, text):
    """写出 UTF-8 文本文件。"""

    target = Path(str(path))
    parent = target.parent
    if not parent.exists():
        parent.mkdir(parents=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(str(text))


def _file_size_bytes(path):
    """返回文件大小；文件缺失时返回 0。"""

    target = Path(str(path))
    if not target.exists():
        return 0
    return int(target.stat().st_size)


def _file_stat(path):
    """返回轻量文件状态。"""

    target = Path(str(path))
    return {"path": str(target), "exists": bool(target.exists()), "bytes": _file_size_bytes(target)}


def _max_rss_mb():
    """杩斿洖褰撳墠杩涚▼宄板€间綇鐣欏唴瀛橈紝鏃犳硶鑾峰彇鏃惰繑鍥?None銆?"""

    if resource is None:
        return None
    try:
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, ValueError):
        return None
    if value <= 0.0:
        return None
    if sys.platform == "darwin":
        return round(value / (1024.0 * 1024.0), 3)
    return round(value / 1024.0, 3)


def _input_file_bytes(manifest):
    """杩斿洖鍘熷杈撳叆 OAS 鏂囦欢澶у皬銆?"""

    return int(_file_size_bytes(manifest.get("input_path", "")))


def _merge_shift_summary(total_summary, chunk_summary):
    """鍚堝苟涓€鎵筩andidate shift 璇婃柇缁熻銆?"""

    direction_counts = dict(total_summary.get("candidate_direction_counts", {}))
    for direction, count in chunk_summary.get("candidate_direction_counts", {}).items():
        direction_key = str(direction)
        direction_counts[direction_key] = int(direction_counts.get(direction_key, 0)) + int(count)
    total_summary["candidate_direction_counts"] = direction_counts
    total_summary["diagonal_candidate_count"] = int(total_summary.get("diagonal_candidate_count", 0)) + int(
        chunk_summary.get("diagonal_candidate_count", 0)
    )
    total_summary["max_shift_distance_um"] = max(
        float(total_summary.get("max_shift_distance_um", 0.0)),
        float(chunk_summary.get("max_shift_distance_um", 0.0)),
    )
    return total_summary


def _write_lsf_wrapper_files(work_root, name, commands, job_name):
    """生成可手动提交的 LSF job-array 命令清单和模板。"""

    lsf_root = _ensure_dir(Path(str(work_root)) / "lsf")
    command_file = lsf_root / ("%s.commands" % str(name))
    template_file = lsf_root / ("%s.bsub.template" % str(name))
    command_lines = [str(command) for command in commands]
    _write_text(command_file, "\n".join(command_lines) + ("\n" if command_lines else ""))
    command_count = int(len(command_lines))
    array_end = max(1, command_count)
    template = [
        "#!/bin/sh",
        "#BSUB -J \"%s[1-%d]\"" % (str(job_name), int(array_end)),
        "#BSUB -n 1",
        "#BSUB -M ${MEM_MB:-8000}",
        "#BSUB -R \"rusage[mem=${MEM_MB:-8000}]\"",
        "#BSUB -oo %s.%%J.%%I.out" % str(name),
        "#BSUB -eo %s.%%J.%%I.err" % str(name),
        "COMMAND_FILE=\"%s\"" % str(command_file),
        "CMD=$(sed -n \"${LSB_JOBINDEX}p\" \"$COMMAND_FILE\")",
        "echo \"$CMD\"",
        "eval \"$CMD\"",
        "",
    ]
    _write_text(template_file, "\n".join(template))
    return {
        "mode": "manual_bsub_template_v1",
        "command_file": str(command_file),
        "bsub_template": str(template_file),
        "command_count": int(command_count),
    }


def _write_single_lsf_command(work_root, name, command):
    """生成单条 LSF 后处理命令文件。"""

    lsf_root = _ensure_dir(Path(str(work_root)) / "lsf")
    command_file = lsf_root / ("%s.command" % str(name))
    _write_text(command_file, str(command) + "\n")
    return {"command_file": str(command_file), "command_count": 1}


def _source_fill_bins_for_clusters(clusters):
    """返回 source exact clusters 的 fill-bin 集合。"""

    return sorted(set(int(coverage_fill_bin_for_bitmap(cluster.representative.clip_bitmap)) for cluster in clusters))


def _coverage_source_shard_specs(source_exact_clusters, coverage_shard_size):
    """按 source fill-bin 优先规划 coverage source shards。"""

    specs = []
    total = int(len(source_exact_clusters))
    if total <= 0:
        return specs
    safe_size = max(1, int(coverage_shard_size))
    start = 0
    while start < total:
        fill_bin = int(coverage_fill_bin_for_bitmap(source_exact_clusters[start].representative.clip_bitmap))
        group_end = start + 1
        while group_end < total:
            next_bin = int(coverage_fill_bin_for_bitmap(source_exact_clusters[group_end].representative.clip_bitmap))
            if next_bin != fill_bin:
                break
            group_end += 1
        chunk_start = start
        while chunk_start < group_end:
            chunk_end = min(group_end, int(chunk_start + safe_size))
            clusters = source_exact_clusters[chunk_start:chunk_end]
            fill_bins = _source_fill_bins_for_clusters(clusters)
            specs.append(
                {
                    "start": int(chunk_start),
                    "end": int(chunk_end),
                    "clusters": clusters,
                    "source_fill_bin_values": [int(value) for value in fill_bins],
                    "source_fill_bin_min": int(min(fill_bins)) if fill_bins else 0,
                    "source_fill_bin_max": int(max(fill_bins)) if fill_bins else 0,
                    "source_fill_bin_count": int(len(fill_bins)),
                }
            )
            chunk_start = chunk_end
        start = group_end
    return specs


def _npz_zip_stat(path):
    """只读取 npz zip metadata，不加载 numpy array。"""

    stats = _file_stat(path)
    stats["zip_member_count"] = 0
    stats["zip_compressed_bytes"] = 0
    stats["zip_uncompressed_bytes"] = 0
    stats["zip_members"] = []
    if not stats["exists"]:
        return stats
    try:
        with zipfile.ZipFile(str(path), "r") as archive:
            for info in archive.infolist():
                stats["zip_member_count"] += 1
                stats["zip_compressed_bytes"] += int(info.compress_size)
                stats["zip_uncompressed_bytes"] += int(info.file_size)
                stats["zip_members"].append(
                    {
                        "name": str(info.filename),
                        "compressed_bytes": int(info.compress_size),
                        "uncompressed_bytes": int(info.file_size),
                    }
                )
    except (IOError, OSError, zipfile.BadZipFile) as exc:
        stats["zip_error"] = str(exc)
    return stats


def _safe_read_json(path):
    """读取 JSON；缺失或损坏时返回 None 与错误描述。"""

    target = Path(str(path))
    if not target.exists():
        return None, "missing"
    try:
        return _read_json(target), None
    except (IOError, OSError, ValueError) as exc:
        return None, str(exc)


def _add_largest_file(largest, label, path):
    """记录候选大文件，后续输出 top 列表。"""

    stat = _file_stat(path)
    if stat["exists"]:
        largest.append({"label": str(label), "path": stat["path"], "bytes": int(stat["bytes"])})


def _config_payload(args):
    """从 argparse 参数构建 v2_lsf 配置。"""

    config = {}
    config_path = getattr(args, "config", None)
    if config_path:
        config.update(_read_json(config_path))
    config["clip_size_um"] = float(getattr(args, "clip_size", config.get("clip_size_um", 1.35)))
    config["geometry_match_mode"] = str(getattr(args, "geometry_match_mode", config.get("geometry_match_mode", "ecc")))
    config["area_match_ratio"] = float(getattr(args, "area_match_ratio", config.get("area_match_ratio", 0.96)))
    config["edge_tolerance_um"] = float(getattr(args, "edge_tolerance_um", config.get("edge_tolerance_um", 0.02)))
    config["pixel_size_nm"] = int(getattr(args, "pixel_size_nm", config.get("pixel_size_nm", DEFAULT_PIXEL_SIZE_NM)))
    config["grid_step_ratio"] = float(GRID_STEP_RATIO)
    config["apply_layer_operations"] = bool(getattr(args, "apply_layer_ops", config.get("apply_layer_operations", False)))
    return config


def _make_layer_processor(register_ops):
    """根据 CLI 注册参数构建 LSF 独立 layer processor。"""

    processor = LayerOperationProcessor()
    for op in register_ops or []:
        if len(op) != 4:
            raise ValueError("--register-op expects SOURCE_LAYER TARGET_LAYER OPERATION RESULT_LAYER")
        source_layer, target_layer, operation, result_layer = op
        processor.register_operation_rule(source_layer, operation, target_layer, result_layer)
    return processor


def _normalized_register_ops(register_ops):
    """把 layer operation 规则规范化成 manifest 中稳定可比的格式。"""

    normalized = []
    for op in register_ops or []:
        if len(op) != 4:
            raise ValueError("--register-op expects SOURCE_LAYER TARGET_LAYER OPERATION RESULT_LAYER")
        normalized.append([str(op[0]), str(op[1]), str(op[2]).lower(), str(op[3])])
    return normalized


def _seed_bbox_union(seeds):
    """计算一组 seed bbox 的并集。"""

    if not seeds:
        return [0.0, 0.0, 0.0, 0.0]
    x0 = min(float(seed.seed_bbox[0]) for seed in seeds)
    y0 = min(float(seed.seed_bbox[1]) for seed in seeds)
    x1 = max(float(seed.seed_bbox[2]) for seed in seeds)
    y1 = max(float(seed.seed_bbox[3]) for seed in seeds)
    return [x0, y0, x1, y1]


def _expand_bbox(bbox, margin):
    """按 margin 扩展 bbox。"""

    return [
        float(bbox[0]) - float(margin),
        float(bbox[1]) - float(margin),
        float(bbox[2]) + float(margin),
        float(bbox[3]) + float(margin),
    ]


def _config_hash(config):
    """生成配置哈希，方便 shard 校验。"""

    payload = json.dumps(config, sort_keys=True, ensure_ascii=True, default=json_default).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _manifest_config_hash(config, register_ops):
    """生成覆盖 config 与 layer operation 规则的 manifest hash。"""

    return _config_hash({"config": dict(config), "register_ops": _normalized_register_ops(register_ops)})


def _validate_manifest(manifest):
    """校验 manifest 与当前 v2_lsf 的 pipeline/config 是否一致。"""

    if str(manifest.get("pipeline_mode", "")) != PIPELINE_MODE:
        raise RuntimeError("Manifest pipeline_mode mismatch")
    if int(manifest.get("schema_version", 0)) != 1:
        raise RuntimeError("Manifest schema_version mismatch")
    expected_hash = _manifest_config_hash(manifest.get("config", {}), manifest.get("register_ops", []))
    if str(manifest.get("config_hash", "")) != str(expected_hash):
        raise RuntimeError("Manifest config_hash mismatch")


def _layer_specs_to_json(specs):
    """把 layer/datatype 二元组转换成 JSON 友好的列表。"""

    return [[int(layer), int(datatype)] for layer, datatype in specs]


def _write_seed_file(seed_file, seeds):
    """把 seed 列表写成 JSONL。"""

    with Path(str(seed_file)).open("w", encoding="utf-8") as handle:
        for seed in seeds:
            handle.write(json.dumps(seed.to_json(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _read_seed_slice(seed_file, start, end):
    """从 JSONL seed 文件读取指定半开区间。"""

    seeds = []
    with Path(str(seed_file)).open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx < int(start):
                continue
            if idx >= int(end):
                break
            text = line.strip()
            if text:
                seeds.append(GridSeedCandidate.from_json(json.loads(text)))
    return seeds


def _load_all_marker_records(manifest):
    """读取 manifest 下所有 shard 的 marker records。"""

    marker_records = []
    shard_summaries = []
    for shard in manifest["shards"]:
        json_path = shard["output_json"]
        npz_path = shard["output_npz"]
        if not Path(json_path).exists() or not Path(npz_path).exists():
            raise RuntimeError("Missing shard output for shard %s" % shard["shard_id"])
        records, payload = load_shard_records(json_path, npz_path)
        marker_records.extend(records)
        shard_summaries.append(
            {
                "shard_id": int(shard["shard_id"]),
                "marker_count": int(payload.get("marker_count", len(records))),
                "local_candidate_count": int(payload.get("local_candidate_count", 0)),
                "local_exact_cluster_count": int(payload.get("local_exact_cluster_count", 0)),
                "spatial_index_stats": dict(payload.get("spatial_index_stats", {})),
                "query_candidate_count_stats": dict(payload.get("query_candidate_count_stats", {})),
            }
        )
    return marker_records, shard_summaries


def _load_shard_summaries_only(manifest):
    """只读取 shard JSON 摘要，不加载 shard bitmap。"""

    shard_summaries = []
    for shard in manifest["shards"]:
        json_path = shard["output_json"]
        if not Path(json_path).exists():
            raise RuntimeError("Missing shard metadata for shard %s" % shard["shard_id"])
        payload = _read_json(json_path)
        shard_summaries.append(
            {
                "shard_id": int(shard["shard_id"]),
                "marker_count": int(payload.get("marker_count", 0)),
                "local_candidate_count": int(payload.get("local_candidate_count", 0)),
                "local_exact_cluster_count": int(payload.get("local_exact_cluster_count", 0)),
                "spatial_index_stats": dict(payload.get("spatial_index_stats", {})),
                "query_candidate_count_stats": dict(payload.get("query_candidate_count_stats", {})),
            }
        )
    return shard_summaries


def _load_manifest_exact_index(manifest):
    """从 manifest 指向的 exact index 读取全局 exact clusters。"""

    exact_index = manifest.get("exact_index")
    if not exact_index:
        raise RuntimeError("Missing exact_index in manifest; run prepare-coverage first")
    json_path = exact_index["output_json"]
    npz_path = exact_index["output_npz"]
    if not Path(json_path).exists() or not Path(npz_path).exists():
        raise RuntimeError("Missing exact index output; run prepare-coverage first")
    return load_exact_index(json_path, npz_path)


def _target_bucket_map(manifest):
    """读取 manifest 中的 target shape bucket 映射。"""

    buckets = manifest.get("exact_target_buckets", {}).get("shape_buckets", {})
    if not buckets:
        raise RuntimeError("Missing exact_target_buckets in manifest; run prepare-coverage first")
    return buckets


def _load_target_buckets_for_shapes(manifest, shape_keys):
    """按候选 shape 懒加载 target exact bucket。"""

    buckets = _target_bucket_map(manifest)
    exact_clusters = []
    loaded_shapes = []
    for shape_key in sorted(shape_keys):
        bucket = buckets.get(str(shape_key))
        if not bucket:
            continue
        clusters, _ = load_exact_index(bucket["output_json"], bucket["output_npz"])
        exact_clusters.extend(clusters)
        loaded_shapes.append(str(shape_key))
    exact_clusters.sort(key=lambda cluster: int(cluster.exact_cluster_id))
    return exact_clusters, loaded_shapes


def _load_all_target_buckets(manifest):
    """读取所有 target exact buckets，用于 merge-coverage 的 set cover 和 final verification。"""

    buckets = _target_bucket_map(manifest)
    exact_clusters = []
    for shape_key in sorted(buckets):
        bucket = buckets[str(shape_key)]
        clusters, _ = load_exact_index(bucket["output_json"], bucket["output_npz"])
        exact_clusters.extend(clusters)
    exact_clusters.sort(key=lambda cluster: int(cluster.exact_cluster_id))
    return exact_clusters


def _load_selected_candidate_bitmaps(selected_candidates, candidate_bitmap_locations):
    """只为 selected candidates 从 coverage shard npz 回填 bitmap。"""

    selected_by_npz = {}
    for candidate in selected_candidates:
        location = candidate_bitmap_locations.get(str(candidate.candidate_id))
        if not location:
            raise RuntimeError("Missing bitmap location for candidate %s" % candidate.candidate_id)
        selected_by_npz.setdefault(location["npz_path"], []).append((candidate, int(location["bitmap_index"])))

    loaded_count = 0
    shard_load_count = 0
    for npz_path in sorted(selected_by_npz):
        if not Path(npz_path).exists():
            raise RuntimeError("Missing coverage bitmap npz for selected candidate: %s" % npz_path)
        pairs = selected_by_npz[npz_path]
        bitmaps = load_coverage_candidate_bitmaps(npz_path, [idx for _, idx in pairs])
        shard_load_count += 1
        for candidate, bitmap_index in pairs:
            candidate.clip_bitmap = bitmaps[int(bitmap_index)]
            loaded_count += 1
    return {"selected_bitmap_load_count": int(loaded_count), "selected_bitmap_shard_load_count": int(shard_load_count)}


def prepare_stage(input_path, work_dir, config, register_ops, shard_count, shard_size):
    """prepare 阶段：读取版图，生成 seed manifest 与 shard 列表。"""

    started = time.perf_counter()
    work_root = _ensure_dir(work_dir)
    shard_root = _ensure_dir(work_root / "shards")
    tile_root = _ensure_dir(work_root / "tile_oas")
    register_ops = _normalized_register_ops(register_ops)
    apply_layer_ops = bool(config.get("apply_layer_operations", False) or register_ops)
    config = dict(config)
    config["apply_layer_operations"] = bool(apply_layer_ops)
    layer_processor = _make_layer_processor(register_ops)
    layout_index = prepare_layout(input_path, layer_processor, apply_layer_ops)
    grid_step_ratio = float(GRID_STEP_RATIO)
    config["grid_step_ratio"] = grid_step_ratio
    seeds, seed_stats = build_uniform_grid_seed_candidates(
        layout_index,
        float(config.get("clip_size_um", 1.35)),
    )
    seed_audit = dict(seed_stats.pop("seed_audit", {}))
    seed_audit_file = work_root / "seed_audit.json"
    _write_json(seed_audit_file, seed_audit)
    seed_file = work_root / "seeds.jsonl"
    _write_seed_file(seed_file, seeds)

    total = int(len(seeds))
    if int(shard_size) <= 0:
        safe_shards = max(1, int(shard_count))
        shard_size = int(math.ceil(float(max(total, 1)) / float(safe_shards)))
    shard_size = max(1, int(shard_size))
    shard_count_actual = int(math.ceil(float(max(total, 1)) / float(shard_size))) if total else 0
    halo_margin = (
        0.5 * float(config.get("clip_size_um", 1.35))
        + 0.5 * float(config.get("clip_size_um", 1.35)) * float(grid_step_ratio)
        + float(config.get("edge_tolerance_um", 0.02))
    )
    manifest_path = work_root / "manifest.json"
    shards = []
    script_path = Path(__file__).resolve()
    tile_oas_total_bytes = 0
    tile_oas_total_element_count = 0
    for shard_id in range(shard_count_actual):
        start = int(shard_id * shard_size)
        end = min(total, int(start + shard_size))
        shard_seeds = seeds[start:end]
        core_bbox = _seed_bbox_union(shard_seeds)
        halo_bbox = _expand_bbox(core_bbox, halo_margin)
        shard_json = shard_root / ("shard_%04d.json" % shard_id)
        shard_npz = shard_root / ("shard_%04d.npz" % shard_id)
        tile_oas = tile_root / ("shard_%04d.oas" % shard_id)
        tile_index = filter_layout_index_by_bbox(layout_index, halo_bbox)
        tile_stats = write_layout_index_oas(tile_index, tile_oas, "SHARD_%04d" % int(shard_id))
        tile_oas_total_bytes += int(tile_stats.get("tile_oas_bytes", 0))
        tile_oas_total_element_count += int(tile_stats.get("tile_element_count", 0))
        command = "%s %s run-shard --manifest %s --shard-id %d" % (
            sys.executable,
            str(script_path),
            str(manifest_path),
            int(shard_id),
        )
        shards.append(
            {
                "shard_id": int(shard_id),
                "seed_start": int(start),
                "seed_end": int(end),
                "seed_count": int(end - start),
                "core_bbox": core_bbox,
                "halo_bbox": halo_bbox,
                "tile_cache_mode": "per_shard_halo_oas_v1",
                "tile_oas": str(tile_oas),
                "tile_oas_bytes": int(tile_stats.get("tile_oas_bytes", 0)),
                "tile_element_count": int(tile_stats.get("tile_element_count", 0)),
                "output_json": str(shard_json),
                "output_npz": str(shard_npz),
                "command": command,
            }
        )
    manifest = {
        "pipeline_mode": PIPELINE_MODE,
        "schema_version": 1,
        "input_path": str(Path(str(input_path)).resolve()),
        "work_dir": str(work_root.resolve()),
        "seed_file": str(seed_file.resolve()),
        "config": dict(config),
        "register_ops": register_ops,
        "config_hash": _manifest_config_hash(config, register_ops),
        "apply_layer_operations": bool(apply_layer_ops),
        "effective_pattern_layers": _layer_specs_to_json(layout_index.effective_pattern_layers),
        "excluded_helper_layers": _layer_specs_to_json(layout_index.excluded_helper_layers),
        "layout_element_count": int(len(layout_index.indexed_elements)),
        "tile_cache_mode": "per_shard_halo_oas_v1",
        "tile_oas_total_bytes": int(tile_oas_total_bytes),
        "tile_oas_total_element_count": int(tile_oas_total_element_count),
        "spatial_index_stats": spatial_index_stats(layout_index),
        "seed_stats": dict(seed_stats),
        "seed_audit": {
            "output_json": str(seed_audit_file.resolve()),
            "array_group_count": int(seed_audit.get("array_group_count", 0)),
            "array_spacing_group_count": int(seed_audit.get("array_spacing_group_count", 0)),
            "array_spacing_seed_count": int(seed_audit.get("array_spacing_seed_count", 0)),
            "array_spacing_weight_total": int(seed_audit.get("array_spacing_weight_total", 0)),
            "seed_type_counts": dict(seed_stats.get("seed_type_counts", {})),
        },
        "shard_count": int(len(shards)),
        "shards": shards,
        "result_output": str((work_root / "final_result.json").resolve()),
        "input_file_bytes": _file_size_bytes(input_path),
        "max_rss_mb": _max_rss_mb(),
        "timing_seconds": {"prepare": round(time.perf_counter() - started, 6)},
    }
    manifest["lsf_wrapper"] = {
        "run_shards": _write_lsf_wrapper_files(work_root, "run_shards", [shard["command"] for shard in shards], "lc_run_shard")
    }
    _write_json(manifest_path, manifest)
    return manifest


def run_shard_stage(manifest_path, shard_id):
    """run-shard 阶段：处理单个 seed shard，只生成 marker records。"""

    started = time.perf_counter()
    manifest = _read_json(manifest_path)
    _validate_manifest(manifest)
    shard = manifest["shards"][int(shard_id)]
    config = dict(manifest["config"])
    input_path = manifest["input_path"]
    register_ops = _normalized_register_ops(manifest.get("register_ops", []))
    apply_layer_ops = bool(config.get("apply_layer_operations", False) or register_ops)
    layer_processor = _make_layer_processor(register_ops)
    tile_oas = shard.get("tile_oas", "")
    if tile_oas and Path(str(tile_oas)).exists():
        layout_load_path = str(tile_oas)
        layout_load_mode = "tile_oas"
        layout_apply_layer_ops = False
        layout_layer_processor = None
    else:
        layout_load_path = str(input_path)
        layout_load_mode = "full_oas"
        layout_apply_layer_ops = bool(apply_layer_ops)
        layout_layer_processor = layer_processor
    layout_index = prepare_layout(layout_load_path, layout_layer_processor, layout_apply_layer_ops)
    layout_element_count = int(len(layout_index.indexed_elements))
    if layout_load_mode != "tile_oas":
        layout_index = filter_layout_index_by_bbox(layout_index, shard.get("halo_bbox", shard.get("core_bbox", [0, 0, 0, 0])))
    halo_filtered_element_count = int(len(layout_index.indexed_elements))
    shard_spatial_index_stats = spatial_index_stats(layout_index)
    seeds = _read_seed_slice(manifest["seed_file"], int(shard["seed_start"]), int(shard["seed_end"]))
    records = []
    for local_idx, seed in enumerate(seeds):
        global_idx = int(shard["seed_start"]) + int(local_idx)
        records.append(marker_record_from_seed(input_path, global_idx, seed, layout_index, config))
    query_stats = marker_query_candidate_stats(records)

    extra = {
        "pipeline_mode": PIPELINE_MODE,
        "stage": "run-shard",
        "shard_payload_mode": "marker_records_only",
        "manifest_path": str(Path(str(manifest_path)).resolve()),
        "shard_id": int(shard_id),
        "seed_count": int(len(seeds)),
        "marker_count": int(len(records)),
        "apply_layer_operations": bool(apply_layer_ops),
        "layout_load_mode": str(layout_load_mode),
        "layout_load_path": str(layout_load_path),
        "layout_apply_layer_operations": bool(layout_apply_layer_ops),
        "source_input_file_bytes": _file_size_bytes(input_path),
        "tile_cache_mode": str(shard.get("tile_cache_mode", manifest.get("tile_cache_mode", "none"))),
        "tile_oas": str(tile_oas),
        "tile_oas_bytes": int(shard.get("tile_oas_bytes", _file_size_bytes(tile_oas) if tile_oas else 0)),
        "tile_element_count": int(shard.get("tile_element_count", layout_element_count if layout_load_mode == "tile_oas" else 0)),
        "registered_layer_operations": register_ops,
        "effective_pattern_layers": _layer_specs_to_json(layout_index.effective_pattern_layers),
        "excluded_helper_layers": _layer_specs_to_json(layout_index.excluded_helper_layers),
        "layout_element_count": int(layout_element_count),
        "halo_filtered_element_count": int(halo_filtered_element_count),
        "input_file_bytes": _input_file_bytes(manifest),
        "max_rss_mb": _max_rss_mb(),
        "spatial_index_stats": shard_spatial_index_stats,
        "query_candidate_count_stats": query_stats,
        "local_exact_cluster_count": 0,
        "local_candidate_count": 0,
        "local_coverage_debug_stats": {"skipped": True, "reason": "global prepare-coverage recomputes exact/candidates"},
        "candidate_summaries": [],
        "timing_seconds": {"run_shard": round(time.perf_counter() - started, 6)},
    }
    save_shard_records(records, shard["output_json"], shard["output_npz"], extra)
    return extra


def merge_stage(manifest_path, output_path=None):
    """merge 阶段：汇总 shard，执行全局聚类。"""

    started = time.perf_counter()
    manifest = _read_json(manifest_path)
    _validate_manifest(manifest)
    config = dict(manifest["config"])
    marker_records, shard_summaries = _load_all_marker_records(manifest)

    exact_started = time.perf_counter()
    exact_clusters = group_exact_clusters(marker_records)
    exact_seconds = time.perf_counter() - exact_started
    if int(len(exact_clusters)) > int(CENTRAL_MERGE_EXACT_CLUSTER_LIMIT):
        raise RuntimeError(
            "exact_cluster_count=%d 超过集中式 merge 上限 %d，请改走 prepare-coverage -> run-coverage-shard -> merge-coverage 分布式 coverage 流程。"
            % (int(len(exact_clusters)), int(CENTRAL_MERGE_EXACT_CLUSTER_LIMIT))
        )

    candidate_started = time.perf_counter()
    candidates = []
    for cluster in exact_clusters:
        candidates.extend(generate_candidates_for_cluster(cluster, config))
    candidate_seconds = time.perf_counter() - candidate_started

    coverage_started = time.perf_counter()
    coverage_stats = evaluate_candidate_coverage(candidates, exact_clusters, config)
    coverage_seconds = time.perf_counter() - coverage_started

    cover_started = time.perf_counter()
    selected = greedy_cover(candidates, exact_clusters)
    cover_seconds = time.perf_counter() - cover_started

    runtime = {
        "merge_exact_cluster": round(exact_seconds, 6),
        "merge_candidate_generation": round(candidate_seconds, 6),
        "merge_coverage_eval": round(coverage_seconds, 6),
        "merge_set_cover": round(cover_seconds, 6),
    }
    result = build_compact_result(marker_records, exact_clusters, candidates, selected, coverage_stats, config, runtime)
    result["lsf_manifest"] = {
        "manifest_path": str(Path(str(manifest_path)).resolve()),
        "shard_count": int(len(manifest["shards"])),
        "shard_summaries": shard_summaries,
        "seed_stats": dict(manifest.get("seed_stats", {})),
        "seed_audit": dict(manifest.get("seed_audit", {})),
        "spatial_index_stats": dict(manifest.get("spatial_index_stats", {})),
        "lsf_wrapper": dict(manifest.get("lsf_wrapper", {})),
    }
    result["lsf_manifest"]["input_file_bytes"] = _input_file_bytes(manifest)
    result["lsf_manifest"]["max_rss_mb"] = _max_rss_mb()
    result["result_summary"]["timing_seconds"]["merge_total"] = round(time.perf_counter() - started, 6)
    output = output_path or manifest.get("result_output")
    _write_json(output, result)
    return result


def prepare_coverage_stage(manifest_path, coverage_shard_count, coverage_shard_size):
    """prepare-coverage 阶段：基于已完成的 shard 产物规划 coverage source 分片。"""

    started = time.perf_counter()
    manifest = _read_json(manifest_path)
    _validate_manifest(manifest)
    config = dict(manifest["config"])
    detail_seconds = {}
    marker_started = time.perf_counter()
    marker_records, _ = _load_all_marker_records(manifest)
    detail_seconds["prepare_coverage_marker_load"] = time.perf_counter() - marker_started
    index_started = time.perf_counter()
    exact_clusters = group_exact_clusters(marker_records)
    exact_seconds = time.perf_counter() - index_started
    detail_seconds["prepare_coverage_exact_cluster"] = exact_seconds
    sort_started = time.perf_counter()
    total = int(len(exact_clusters))
    source_exact_clusters = sorted(
        exact_clusters,
        key=lambda cluster: (
            int(coverage_fill_bin_for_bitmap(cluster.representative.clip_bitmap)),
            int(cluster.exact_cluster_id),
        ),
    )
    detail_seconds["prepare_coverage_source_sort"] = time.perf_counter() - sort_started
    if int(coverage_shard_size) <= 0:
        safe_shards = max(1, int(coverage_shard_count))
        coverage_shard_size = int(math.ceil(float(max(total, 1)) / float(safe_shards)))
    coverage_shard_size = max(1, int(coverage_shard_size))
    source_shard_specs = _coverage_source_shard_specs(source_exact_clusters, coverage_shard_size)
    source_fill_bin_group_count = int(len(_source_fill_bins_for_clusters(source_exact_clusters)))
    max_source_fill_bin_count = max((int(spec["source_fill_bin_count"]) for spec in source_shard_specs), default=0)
    work_root = _ensure_dir(manifest["work_dir"])
    coverage_root = _ensure_dir(work_root / "coverage_shards")
    source_root = _ensure_dir(work_root / "exact_source_shards")
    target_root = _ensure_dir(work_root / "exact_target_buckets")
    bundle_root = _ensure_dir(work_root / "candidate_bundle_buckets")
    exact_json = work_root / "exact_index.json"
    exact_npz = work_root / "exact_index.npz"
    write_started = time.perf_counter()
    save_exact_index(
        exact_clusters,
        exact_json,
        exact_npz,
        {
            "pipeline_mode": PIPELINE_MODE,
            "stage": "prepare-coverage",
            "marker_count": int(len(marker_records)),
            "exact_cluster_count": int(total),
        },
    )
    write_seconds = time.perf_counter() - write_started
    detail_seconds["prepare_coverage_exact_index_write"] = write_seconds
    target_started = time.perf_counter()
    target_groups = {}
    for cluster in exact_clusters:
        shape_key = bitmap_shape_key(cluster.representative.clip_bitmap.shape)
        target_groups.setdefault(shape_key, []).append(cluster)
    shape_buckets = {}
    for shape_key in sorted(target_groups):
        safe_key = str(shape_key).replace("x", "_")
        bucket_json = target_root / ("target_%s.json" % safe_key)
        bucket_npz = target_root / ("target_%s.npz" % safe_key)
        save_exact_index(
            target_groups[shape_key],
            bucket_json,
            bucket_npz,
            {
                "pipeline_mode": PIPELINE_MODE,
                "stage": "prepare-coverage-target-bucket",
                "shape_key": str(shape_key),
                "exact_cluster_count": int(len(target_groups[shape_key])),
            },
        )
        shape_buckets[str(shape_key)] = {
            "shape_key": str(shape_key),
            "exact_cluster_count": int(len(target_groups[shape_key])),
            "output_json": str(bucket_json),
            "output_npz": str(bucket_npz),
        }
    target_seconds = time.perf_counter() - target_started
    detail_seconds["prepare_coverage_target_bucket_write"] = target_seconds
    candidate_started = time.perf_counter()
    candidate_bundle_accumulator = create_candidate_bundle_accumulator()
    shift_summary = {"candidate_direction_counts": {}, "diagonal_candidate_count": 0, "max_shift_distance_um": 0.0}
    cluster_chunk_size = max(1, int(PREPARE_COVERAGE_CANDIDATE_CHUNK_SIZE))
    for chunk_start in range(0, len(exact_clusters), cluster_chunk_size):
        chunk_end = min(len(exact_clusters), int(chunk_start + cluster_chunk_size))
        chunk_candidates = []
        for cluster in exact_clusters[chunk_start:chunk_end]:
            chunk_candidates.extend(generate_candidates_for_cluster(cluster, config))
        _merge_shift_summary(shift_summary, candidate_shift_summary(chunk_candidates))
        add_candidates_to_candidate_bundle_accumulator(candidate_bundle_accumulator, chunk_candidates)
    candidate_seconds = time.perf_counter() - candidate_started
    detail_seconds["prepare_coverage_candidate_generation"] = candidate_seconds
    bundle_started = time.perf_counter()
    candidate_bundle_index = save_candidate_bundle_index_from_accumulator(
        candidate_bundle_accumulator,
        bundle_root,
        {
            "pipeline_mode": PIPELINE_MODE,
            "stage": "prepare-coverage-candidate-bundle",
        },
    )
    bundle_seconds = time.perf_counter() - bundle_started
    detail_seconds["prepare_coverage_candidate_bundle_write"] = bundle_seconds
    script_path = Path(__file__).resolve()
    coverage_shards = []
    source_started = time.perf_counter()
    for shard_id, spec in enumerate(source_shard_specs):
        start = int(spec["start"])
        end = int(spec["end"])
        output_json = coverage_root / ("coverage_%04d.json" % shard_id)
        output_npz = coverage_root / ("coverage_%04d.npz" % shard_id)
        source_json = source_root / ("source_%04d.json" % shard_id)
        source_npz = source_root / ("source_%04d.npz" % shard_id)
        save_exact_index(
            spec["clusters"],
            source_json,
            source_npz,
            {
                "pipeline_mode": PIPELINE_MODE,
                "stage": "prepare-coverage-source-shard",
                "coverage_shard_id": int(shard_id),
                "exact_start": int(start),
                "exact_end": int(end),
                "exact_cluster_count": int(end - start),
                "source_order": "fill_bin",
                "source_fill_bin_min": int(spec["source_fill_bin_min"]),
                "source_fill_bin_max": int(spec["source_fill_bin_max"]),
                "source_fill_bin_count": int(spec["source_fill_bin_count"]),
                "source_fill_bin_values": [int(value) for value in spec["source_fill_bin_values"]],
            },
        )
        command = "%s %s run-coverage-shard --manifest %s --coverage-shard-id %d" % (
            sys.executable,
            str(script_path),
            str(Path(str(manifest_path)).resolve()),
            int(shard_id),
        )
        coverage_shards.append(
            {
                "coverage_shard_id": int(shard_id),
                "exact_start": int(start),
                "exact_end": int(end),
                "exact_count": int(end - start),
                "source_order": "fill_bin",
                "source_fill_bin_min": int(spec["source_fill_bin_min"]),
                "source_fill_bin_max": int(spec["source_fill_bin_max"]),
                "source_fill_bin_count": int(spec["source_fill_bin_count"]),
                "source_fill_bin_values": [int(value) for value in spec["source_fill_bin_values"]],
                "source_index_json": str(source_json),
                "source_index_npz": str(source_npz),
                "output_json": str(output_json),
                "output_npz": str(output_npz),
                "command": command,
            }
        )
    source_seconds = time.perf_counter() - source_started
    detail_seconds["prepare_coverage_source_shard_write"] = source_seconds
    prepare_total = time.perf_counter() - started
    detail_seconds["prepare_coverage"] = prepare_total
    manifest["coverage_shard_count"] = int(len(coverage_shards))
    manifest["coverage_shards"] = coverage_shards
    manifest["exact_index"] = {
        "output_json": str(exact_json),
        "output_npz": str(exact_npz),
        "marker_count": int(len(marker_records)),
        "exact_cluster_count": int(total),
    }
    manifest["exact_target_buckets"] = {
        "bucket_count": int(len(shape_buckets)),
        "shape_buckets": shape_buckets,
    }
    manifest["candidate_bundle_index"] = candidate_bundle_index
    manifest["coverage_plan"] = {
        "exact_cluster_count": int(total),
        "candidate_count": int(candidate_bundle_index.get("candidate_count", 0)),
        "candidate_direction_counts": dict(shift_summary["candidate_direction_counts"]),
        "diagonal_candidate_count": int(shift_summary["diagonal_candidate_count"]),
        "max_shift_distance_um": float(shift_summary["max_shift_distance_um"]),
        "candidate_group_count": int(candidate_bundle_index.get("candidate_group_count", 0)),
        "max_candidate_bundle_group_count": int(candidate_bundle_index.get("max_bundle_group_count", 0)),
        "max_candidate_file_bucket_group_count": int(candidate_bundle_index.get("max_file_bucket_group_count", 0)),
        "candidate_bundle_bucket_count": int(candidate_bundle_index.get("bucket_count", 0)),
        "candidate_bundle_split_mode": str(candidate_bundle_index.get("bucket_split_mode", "shape")),
        "candidate_chunk_size": int(cluster_chunk_size),
        "coverage_shard_size": int(coverage_shard_size),
        "coverage_source_partition_mode": "fill_bin_grouped",
        "coverage_source_fill_bin_group_count": int(source_fill_bin_group_count),
        "max_source_fill_bin_count_per_shard": int(max_source_fill_bin_count),
        "source_shard_order": "fill_bin",
        "input_file_bytes": _input_file_bytes(manifest),
        "max_rss_mb": _max_rss_mb(),
        "timing_seconds": dict((key, round(float(value), 6)) for key, value in detail_seconds.items()),
    }
    lsf_wrapper = dict(manifest.get("lsf_wrapper", {}))
    lsf_wrapper["run_coverage_shards"] = _write_lsf_wrapper_files(
        work_root,
        "run_coverage_shards",
        [shard["command"] for shard in coverage_shards],
        "lc_run_coverage",
    )
    merge_output = manifest.get("result_output", str(work_root / "final_result.json"))
    merge_command = "%s %s merge-coverage --manifest %s --output %s" % (
        sys.executable,
        str(script_path),
        str(Path(str(manifest_path)).resolve()),
        str(merge_output),
    )
    merge_wrapper = _write_single_lsf_command(work_root, "merge_coverage", merge_command)
    merge_wrapper["command"] = str(merge_command)
    lsf_wrapper["merge_coverage"] = merge_wrapper
    manifest["lsf_wrapper"] = lsf_wrapper
    _write_json(manifest_path, manifest)
    return manifest


def run_coverage_shard_stage(manifest_path, coverage_shard_id):
    """run-coverage-shard 阶段：为一段 source exact cluster 计算候选覆盖。"""

    started = time.perf_counter()
    manifest = _read_json(manifest_path)
    _validate_manifest(manifest)
    if "coverage_shards" not in manifest:
        raise RuntimeError("Missing coverage_shards in manifest; run prepare-coverage first")
    coverage_shard = manifest["coverage_shards"][int(coverage_shard_id)]
    config = dict(manifest["config"])

    source_started = time.perf_counter()
    source_clusters, source_payload = load_exact_index(coverage_shard["source_index_json"], coverage_shard["source_index_npz"])
    source_seconds = time.perf_counter() - source_started

    candidate_started = time.perf_counter()
    candidates = generate_candidates_for_cluster_range(
        source_clusters,
        config,
        0,
        len(source_clusters),
    )
    shift_summary = candidate_shift_summary(candidates)
    candidate_fill_bins = sorted(set(int(coverage_fill_bin_for_bitmap(candidate.clip_bitmap)) for candidate in candidates))
    source_fill_bins = [int(value) for value in coverage_shard.get("source_fill_bin_values", source_payload.get("source_fill_bin_values", []))]
    if not source_fill_bins:
        source_fill_bins = _source_fill_bins_for_clusters(source_clusters)
    candidate_seconds = time.perf_counter() - candidate_started

    target_started = time.perf_counter()
    candidate_shape_keys = set(bitmap_shape_key(candidate.clip_bitmap.shape) for candidate in candidates)
    target_bundles, target_load_stats = load_candidate_bundle_buckets_for_candidates(manifest.get("candidate_bundle_index", {}), candidates)
    loaded_shapes = sorted(target_bundles.keys())
    total_candidate_group_count = int(manifest.get("candidate_bundle_index", {}).get("candidate_group_count", 0))
    target_candidate_group_count_loaded = int(
        target_load_stats.get("candidate_group_count_loaded", sum(len(bundle["candidate_groups"]) for bundle in target_bundles.values()))
    )
    if int(total_candidate_group_count) > 0:
        target_load_ratio = float(target_candidate_group_count_loaded) / float(total_candidate_group_count)
    else:
        target_load_ratio = 0.0
    target_seconds = time.perf_counter() - target_started

    coverage_started = time.perf_counter()
    coverage_stats = evaluate_candidate_coverage_against_bundles(candidates, target_bundles, config)
    coverage_seconds = time.perf_counter() - coverage_started

    extra = {
        "pipeline_mode": PIPELINE_MODE,
        "stage": "run-coverage-shard",
        "manifest_path": str(Path(str(manifest_path)).resolve()),
        "coverage_shard_id": int(coverage_shard_id),
        "exact_start": int(coverage_shard["exact_start"]),
        "exact_end": int(coverage_shard["exact_end"]),
        "source_exact_count": int(coverage_shard["exact_end"]) - int(coverage_shard["exact_start"]),
        "source_index_cluster_count": int(source_payload.get("exact_cluster_count", len(source_clusters))),
        "target_exact_count": int(manifest.get("exact_index", {}).get("exact_cluster_count", 0)),
        "target_shape_count_loaded": int(target_load_stats.get("shape_count_loaded", len(loaded_shapes))),
        "target_bucket_count_loaded": int(target_load_stats.get("bucket_count_loaded", len(loaded_shapes))),
        "target_bucket_shape_keys": loaded_shapes,
        "target_bundle_bucket_keys_loaded_sample": list(target_load_stats.get("bucket_keys_loaded", [])),
        "target_candidate_group_count_loaded": int(target_candidate_group_count_loaded),
        "target_candidate_group_count_total": int(total_candidate_group_count),
        "target_candidate_group_load_ratio": round(float(target_load_ratio), 6),
        "target_load_warning": bool(target_load_ratio > float(TARGET_LOAD_WARNING_RATIO)),
        "candidate_shape_count": int(len(candidate_shape_keys)),
        "candidate_count": int(len(candidates)),
        "candidate_direction_counts": dict(shift_summary["candidate_direction_counts"]),
        "diagonal_candidate_count": int(shift_summary["diagonal_candidate_count"]),
        "max_shift_distance_um": float(shift_summary["max_shift_distance_um"]),
        "source_fill_bin_min": int(min(source_fill_bins)) if source_fill_bins else 0,
        "source_fill_bin_max": int(max(source_fill_bins)) if source_fill_bins else 0,
        "source_fill_bin_count": int(len(source_fill_bins)),
        "source_fill_bin_values": [int(value) for value in source_fill_bins],
        "candidate_fill_bin_min": int(min(candidate_fill_bins)) if candidate_fill_bins else 0,
        "candidate_fill_bin_max": int(max(candidate_fill_bins)) if candidate_fill_bins else 0,
        "candidate_fill_bin_count": int(len(candidate_fill_bins)),
        "candidate_fill_bin_values_sample": [int(value) for value in candidate_fill_bins[:64]],
        "coverage_debug_stats": dict(coverage_stats),
        "input_file_bytes": _input_file_bytes(manifest),
        "max_rss_mb": _max_rss_mb(),
        "timing_seconds": {
            "coverage_source_index_load": round(source_seconds, 6),
            "coverage_candidate_generation": round(candidate_seconds, 6),
            "coverage_target_bucket_load": round(target_seconds, 6),
            "coverage_eval": round(coverage_seconds, 6),
            "run_coverage_shard": round(time.perf_counter() - started, 6),
        },
    }
    save_coverage_shard(candidates, coverage_shard["output_json"], coverage_shard["output_npz"], extra)
    return extra


def merge_coverage_stage(manifest_path, output_path=None):
    """merge-coverage 阶段：汇总 coverage shard 并执行全局 set cover。"""

    started = time.perf_counter()
    manifest = _read_json(manifest_path)
    _validate_manifest(manifest)
    if "coverage_shards" not in manifest:
        raise RuntimeError("Missing coverage_shards in manifest; run prepare-coverage first")
    config = dict(manifest["config"])
    shard_summaries = _load_shard_summaries_only(manifest)

    exact_started = time.perf_counter()
    exact_clusters = _load_all_target_buckets(manifest)
    exact_seconds = time.perf_counter() - exact_started
    exact_payload = dict(manifest.get("exact_index", {}))

    load_started = time.perf_counter()
    candidate_records = []
    candidate_bitmap_locations = {}
    coverage_offsets = [0]
    coverage_value_chunks = []
    coverage_shard_summaries = []
    aggregate_stats = {}
    aggregate_detail_seconds = {}
    for coverage_shard in manifest["coverage_shards"]:
        json_path = coverage_shard["output_json"]
        npz_path = coverage_shard["output_npz"]
        if not Path(json_path).exists() or not Path(npz_path).exists():
            raise RuntimeError("Missing coverage shard output for coverage shard %s" % coverage_shard["coverage_shard_id"])
        shard_metadata, shard_offsets, shard_values, payload = load_coverage_shard_csr_metadata(json_path, npz_path)
        candidate_records.extend(shard_metadata)
        if len(shard_values):
            coverage_value_chunks.append(shard_values)
        for local_idx, metadata in enumerate(shard_metadata):
            coverage_offsets.append(
                int(coverage_offsets[-1])
                + int(shard_offsets[int(local_idx) + 1])
                - int(shard_offsets[int(local_idx)])
            )
            candidate_bitmap_locations[str(metadata["candidate_id"])] = {
                "npz_path": str(npz_path),
                "bitmap_index": int(metadata.get("bitmap_index", local_idx)),
            }
        stats = dict(payload.get("coverage_debug_stats", {}))
        detail_seconds = dict(stats.get("coverage_detail_seconds", {}))
        for key, value in detail_seconds.items():
            aggregate_detail_seconds[str(key)] = float(aggregate_detail_seconds.get(str(key), 0.0)) + float(value)
        for key, value in stats.items():
            if key == "coverage_detail_seconds":
                continue
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                aggregate_stats[str(key)] = aggregate_stats.get(str(key), 0) + value
        coverage_shard_summaries.append(
            {
                "coverage_shard_id": int(coverage_shard["coverage_shard_id"]),
                "source_exact_count": int(payload.get("source_exact_count", coverage_shard.get("exact_count", 0))),
                "candidate_count": int(payload.get("candidate_count", len(shard_metadata))),
                "coverage_storage": str(payload.get("coverage_storage", "json_inline")),
                "coverage_value_count": int(payload.get("coverage_value_count", 0)),
                "target_bucket_count_loaded": int(payload.get("target_bucket_count_loaded", 0)),
                "target_candidate_group_load_ratio": float(payload.get("target_candidate_group_load_ratio", 0.0)),
                "target_load_warning": bool(payload.get("target_load_warning", False)),
                "source_fill_bin_count": int(payload.get("source_fill_bin_count", coverage_shard.get("source_fill_bin_count", 0))),
                "candidate_fill_bin_count": int(payload.get("candidate_fill_bin_count", 0)),
                "geometry_pair_count": int(stats.get("geometry_pair_count", 0)),
                "geometry_pass": int(stats.get("geometry_pass", 0)),
                "exact_hash_pairs": int(stats.get("exact_hash_pairs", 0)),
                "cheap_reject": int(stats.get("cheap_reject", 0)),
                "xor_reject": int(stats.get("xor_reject", 0)),
                "full_prefilter_reject": int(stats.get("full_prefilter_reject", 0)),
                "full_descriptor_cache_group_count": int(stats.get("full_descriptor_cache_group_count", 0)),
            }
        )
    load_seconds = time.perf_counter() - load_started
    if coverage_value_chunks:
        coverage_values = np.concatenate(coverage_value_chunks).astype(np.int64, copy=False)
    else:
        coverage_values = np.zeros((0,), dtype=np.int64)
    coverage_offsets = np.asarray(coverage_offsets, dtype=np.int64)

    cover_started = time.perf_counter()
    selected_indexes = greedy_cover_csr(candidate_records, coverage_offsets, coverage_values, exact_clusters)
    selected = selected_candidates_from_csr(candidate_records, coverage_offsets, coverage_values, selected_indexes)
    cover_seconds = time.perf_counter() - cover_started

    bitmap_started = time.perf_counter()
    selected_bitmap_stats = _load_selected_candidate_bitmaps(selected, candidate_bitmap_locations)
    bitmap_seconds = time.perf_counter() - bitmap_started

    coverage_stats = dict((str(key), int(value) if isinstance(value, int) else value) for key, value in aggregate_stats.items())
    coverage_stats.update(
        {
        "coverage_shard_count": int(len(manifest["coverage_shards"])),
        "source_candidate_count": int(len(candidate_records)),
        "metadata_candidate_count": int(len(candidate_records)),
        "candidate_object_preload_count": 0,
        "coverage_csr_edge_count": int(len(coverage_values)),
        "coverage_csr_offset_count": int(len(coverage_offsets)),
        "candidate_bitmap_preload_count": 0,
        "selected_bitmap_load_count": int(selected_bitmap_stats["selected_bitmap_load_count"]),
        "selected_bitmap_shard_load_count": int(selected_bitmap_stats["selected_bitmap_shard_load_count"]),
        "coverage_detail_seconds": dict((key, round(float(value), 6)) for key, value in aggregate_detail_seconds.items()),
        }
    )
    runtime = {
        "merge_exact_cluster": round(exact_seconds, 6),
        "merge_coverage_metadata_load": round(load_seconds, 6),
        "merge_set_cover": round(cover_seconds, 6),
        "merge_selected_bitmap_load": round(bitmap_seconds, 6),
    }
    result = build_compact_result(None, exact_clusters, candidate_records, selected, coverage_stats, config, runtime)
    result["lsf_manifest"] = {
        "manifest_path": str(Path(str(manifest_path)).resolve()),
        "shard_count": int(len(manifest["shards"])),
        "coverage_shard_count": int(len(manifest["coverage_shards"])),
        "shard_summaries": shard_summaries,
        "coverage_shard_summaries": coverage_shard_summaries,
        "seed_stats": dict(manifest.get("seed_stats", {})),
        "seed_audit": dict(manifest.get("seed_audit", {})),
        "spatial_index_stats": dict(manifest.get("spatial_index_stats", {})),
        "coverage_plan": dict(manifest.get("coverage_plan", {})),
        "exact_index": dict(manifest.get("exact_index", {})),
        "exact_target_buckets": dict(manifest.get("exact_target_buckets", {})),
        "candidate_bundle_index": dict(manifest.get("candidate_bundle_index", {})),
        "lsf_wrapper": dict(manifest.get("lsf_wrapper", {})),
        "input_file_bytes": _input_file_bytes(manifest),
        "max_rss_mb": _max_rss_mb(),
    }
    result["marker_count"] = int(exact_payload.get("marker_count", result["marker_count"]))
    result["total_samples"] = int(exact_payload.get("marker_count", result["total_samples"]))
    result["result_summary"]["timing_seconds"]["merge_coverage_total"] = round(time.perf_counter() - started, 6)
    output = output_path or manifest.get("result_output")
    _write_json(output, result)
    return result


def _candidate_bundle_file_items(candidate_bundle_index):
    """展开 candidate bundle index 中的实际 JSON/NPZ 文件项。"""

    items = []
    for shape_key in sorted(dict(candidate_bundle_index.get("shape_buckets", {}))):
        shape_item = candidate_bundle_index["shape_buckets"][shape_key]
        if "buckets" in shape_item:
            for fill_bin in sorted(dict(shape_item.get("buckets", {})), key=lambda value: int(value)):
                items.append(shape_item["buckets"][fill_bin])
        else:
            items.append(shape_item)
    return items


def inspect_workdir_stage(manifest_path, output_path=None):
    """inspect-workdir 阶段：轻量汇总 LSF workdir 产物规模。"""

    started = time.perf_counter()
    manifest = _read_json(manifest_path)
    largest_files = []
    manifest_stat = _file_stat(manifest_path)
    _add_largest_file(largest_files, "manifest", manifest_path)

    seed_file = manifest.get("seed_file", "")
    seed_stat = _file_stat(seed_file) if seed_file else {"path": "", "exists": False, "bytes": 0}
    if seed_file:
        _add_largest_file(largest_files, "seeds", seed_file)

    shard_summary = {
        "count": int(len(manifest.get("shards", []))),
        "missing_json": 0,
        "missing_npz": 0,
        "missing_tile_oas": 0,
        "json_bytes": 0,
        "npz_bytes": 0,
        "tile_oas_bytes": 0,
        "tile_element_count": 0,
        "npz_zip_uncompressed_bytes": 0,
        "marker_count": 0,
        "local_candidate_count": 0,
        "local_exact_cluster_count": 0,
    }
    for shard in manifest.get("shards", []):
        json_path = shard.get("output_json", "")
        npz_path = shard.get("output_npz", "")
        tile_path = shard.get("tile_oas", "")
        json_stat = _file_stat(json_path)
        npz_stat = _npz_zip_stat(npz_path)
        tile_stat = _file_stat(tile_path) if tile_path else {"exists": False, "bytes": 0}
        _add_largest_file(largest_files, "shard_json_%s" % shard.get("shard_id", ""), json_path)
        _add_largest_file(largest_files, "shard_npz_%s" % shard.get("shard_id", ""), npz_path)
        if tile_path:
            _add_largest_file(largest_files, "tile_oas_%s" % shard.get("shard_id", ""), tile_path)
        shard_summary["json_bytes"] += int(json_stat["bytes"])
        shard_summary["npz_bytes"] += int(npz_stat["bytes"])
        shard_summary["tile_oas_bytes"] += int(tile_stat.get("bytes", 0))
        shard_summary["tile_element_count"] += int(shard.get("tile_element_count", 0))
        shard_summary["npz_zip_uncompressed_bytes"] += int(npz_stat.get("zip_uncompressed_bytes", 0))
        if not json_stat["exists"]:
            shard_summary["missing_json"] += 1
        if not npz_stat["exists"]:
            shard_summary["missing_npz"] += 1
        if tile_path and not tile_stat["exists"]:
            shard_summary["missing_tile_oas"] += 1
        payload, _ = _safe_read_json(json_path)
        if payload:
            shard_summary["marker_count"] += int(payload.get("marker_count", 0))
            shard_summary["local_candidate_count"] += int(payload.get("local_candidate_count", 0))
            shard_summary["local_exact_cluster_count"] += int(payload.get("local_exact_cluster_count", 0))

    exact_index = manifest.get("exact_index", {})
    exact_index_summary = {
        "json": _file_stat(exact_index.get("output_json", "")) if exact_index else {},
        "npz": _npz_zip_stat(exact_index.get("output_npz", "")) if exact_index else {},
        "marker_count": int(exact_index.get("marker_count", 0)) if exact_index else 0,
        "exact_cluster_count": int(exact_index.get("exact_cluster_count", 0)) if exact_index else 0,
    }
    if exact_index:
        _add_largest_file(largest_files, "exact_index_json", exact_index.get("output_json", ""))
        _add_largest_file(largest_files, "exact_index_npz", exact_index.get("output_npz", ""))

    target_bucket_summary = {
        "bucket_count": 0,
        "json_bytes": 0,
        "npz_bytes": 0,
        "npz_zip_uncompressed_bytes": 0,
        "exact_cluster_count": 0,
    }
    shape_buckets = manifest.get("exact_target_buckets", {}).get("shape_buckets", {})
    for shape_key in sorted(shape_buckets):
        bucket = shape_buckets[shape_key]
        json_stat = _file_stat(bucket.get("output_json", ""))
        npz_stat = _npz_zip_stat(bucket.get("output_npz", ""))
        _add_largest_file(largest_files, "target_bucket_json_%s" % shape_key, bucket.get("output_json", ""))
        _add_largest_file(largest_files, "target_bucket_npz_%s" % shape_key, bucket.get("output_npz", ""))
        target_bucket_summary["bucket_count"] += 1
        target_bucket_summary["json_bytes"] += int(json_stat["bytes"])
        target_bucket_summary["npz_bytes"] += int(npz_stat["bytes"])
        target_bucket_summary["npz_zip_uncompressed_bytes"] += int(npz_stat.get("zip_uncompressed_bytes", 0))
        target_bucket_summary["exact_cluster_count"] += int(bucket.get("exact_cluster_count", 0))

    candidate_bundle_index = manifest.get("candidate_bundle_index", {})
    candidate_bundle_summary = {
        "shape_bucket_count": int(candidate_bundle_index.get("shape_bucket_count", len(candidate_bundle_index.get("shape_buckets", {})))),
        "bucket_count": int(candidate_bundle_index.get("bucket_count", 0)),
        "candidate_group_count": int(candidate_bundle_index.get("candidate_group_count", 0)),
        "max_bundle_group_count": int(candidate_bundle_index.get("max_bundle_group_count", 0)),
        "max_file_bucket_group_count": int(candidate_bundle_index.get("max_file_bucket_group_count", 0)),
        "json_bytes": 0,
        "npz_bytes": 0,
        "npz_zip_uncompressed_bytes": 0,
    }
    for item in _candidate_bundle_file_items(candidate_bundle_index):
        json_path = item.get("output_json", "")
        npz_path = item.get("output_npz", "")
        json_stat = _file_stat(json_path)
        npz_stat = _npz_zip_stat(npz_path)
        _add_largest_file(largest_files, "candidate_bundle_json_%s" % item.get("fill_bin", item.get("shape_key", "")), json_path)
        _add_largest_file(largest_files, "candidate_bundle_npz_%s" % item.get("fill_bin", item.get("shape_key", "")), npz_path)
        candidate_bundle_summary["json_bytes"] += int(json_stat["bytes"])
        candidate_bundle_summary["npz_bytes"] += int(npz_stat["bytes"])
        candidate_bundle_summary["npz_zip_uncompressed_bytes"] += int(npz_stat.get("zip_uncompressed_bytes", 0))

    coverage_summary = {
        "count": int(len(manifest.get("coverage_shards", []))),
        "missing_json": 0,
        "missing_npz": 0,
        "json_bytes": 0,
        "npz_bytes": 0,
        "npz_zip_uncompressed_bytes": 0,
        "candidate_count": 0,
        "coverage_value_count": 0,
        "geometry_pair_count": 0,
        "geometry_pass": 0,
        "exact_hash_pairs": 0,
        "cheap_reject": 0,
        "xor_reject": 0,
        "target_bucket_count_loaded": 0,
        "target_candidate_group_count_loaded": 0,
        "target_candidate_group_load_ratio_sum": 0.0,
        "target_candidate_group_load_ratio_max": 0.0,
        "target_load_warning_count": 0,
        "source_fill_bin_count_max": 0,
        "candidate_fill_bin_count_max": 0,
        "coverage_storage_counts": {},
    }
    source_summary = {
        "count": 0,
        "json_bytes": 0,
        "npz_bytes": 0,
        "npz_zip_uncompressed_bytes": 0,
    }
    for coverage_shard in manifest.get("coverage_shards", []):
        source_json = coverage_shard.get("source_index_json", "")
        source_npz = coverage_shard.get("source_index_npz", "")
        if source_json or source_npz:
            source_summary["count"] += 1
            source_json_stat = _file_stat(source_json)
            source_npz_stat = _npz_zip_stat(source_npz)
            source_summary["json_bytes"] += int(source_json_stat["bytes"])
            source_summary["npz_bytes"] += int(source_npz_stat["bytes"])
            source_summary["npz_zip_uncompressed_bytes"] += int(source_npz_stat.get("zip_uncompressed_bytes", 0))
            _add_largest_file(largest_files, "source_index_json_%s" % coverage_shard.get("coverage_shard_id", ""), source_json)
            _add_largest_file(largest_files, "source_index_npz_%s" % coverage_shard.get("coverage_shard_id", ""), source_npz)

        json_path = coverage_shard.get("output_json", "")
        npz_path = coverage_shard.get("output_npz", "")
        json_stat = _file_stat(json_path)
        npz_stat = _npz_zip_stat(npz_path)
        _add_largest_file(largest_files, "coverage_json_%s" % coverage_shard.get("coverage_shard_id", ""), json_path)
        _add_largest_file(largest_files, "coverage_npz_%s" % coverage_shard.get("coverage_shard_id", ""), npz_path)
        coverage_summary["json_bytes"] += int(json_stat["bytes"])
        coverage_summary["npz_bytes"] += int(npz_stat["bytes"])
        coverage_summary["npz_zip_uncompressed_bytes"] += int(npz_stat.get("zip_uncompressed_bytes", 0))
        if not json_stat["exists"]:
            coverage_summary["missing_json"] += 1
        if not npz_stat["exists"]:
            coverage_summary["missing_npz"] += 1
        payload, _ = _safe_read_json(json_path)
        if payload:
            stats = dict(payload.get("coverage_debug_stats", {}))
            storage = str(payload.get("coverage_storage", "unknown"))
            coverage_summary["coverage_storage_counts"][storage] = int(coverage_summary["coverage_storage_counts"].get(storage, 0)) + 1
            coverage_summary["candidate_count"] += int(payload.get("candidate_count", 0))
            coverage_summary["coverage_value_count"] += int(payload.get("coverage_value_count", 0))
            coverage_summary["geometry_pair_count"] += int(stats.get("geometry_pair_count", 0))
            coverage_summary["geometry_pass"] += int(stats.get("geometry_pass", 0))
            coverage_summary["exact_hash_pairs"] += int(stats.get("exact_hash_pairs", 0))
            coverage_summary["cheap_reject"] += int(stats.get("cheap_reject", 0))
            coverage_summary["xor_reject"] += int(stats.get("xor_reject", 0))
            coverage_summary["target_bucket_count_loaded"] += int(payload.get("target_bucket_count_loaded", 0))
            coverage_summary["target_candidate_group_count_loaded"] += int(payload.get("target_candidate_group_count_loaded", 0))
            coverage_summary["target_candidate_group_load_ratio_sum"] += float(payload.get("target_candidate_group_load_ratio", 0.0))
            coverage_summary["target_candidate_group_load_ratio_max"] = max(
                float(coverage_summary["target_candidate_group_load_ratio_max"]),
                float(payload.get("target_candidate_group_load_ratio", 0.0)),
            )
            coverage_summary["target_load_warning_count"] += 1 if payload.get("target_load_warning", False) else 0
            coverage_summary["source_fill_bin_count_max"] = max(
                int(coverage_summary["source_fill_bin_count_max"]),
                int(payload.get("source_fill_bin_count", 0)),
            )
            coverage_summary["candidate_fill_bin_count_max"] = max(
                int(coverage_summary["candidate_fill_bin_count_max"]),
                int(payload.get("candidate_fill_bin_count", 0)),
            )

    if int(coverage_summary["count"]) > 0:
        coverage_summary["target_candidate_group_load_ratio_avg"] = round(
            float(coverage_summary["target_candidate_group_load_ratio_sum"]) / float(coverage_summary["count"]),
            6,
        )
    else:
        coverage_summary["target_candidate_group_load_ratio_avg"] = 0.0
    coverage_summary["target_candidate_group_load_ratio_sum"] = round(
        float(coverage_summary["target_candidate_group_load_ratio_sum"]),
        6,
    )
    coverage_summary["target_candidate_group_load_ratio_max"] = round(
        float(coverage_summary["target_candidate_group_load_ratio_max"]),
        6,
    )

    largest_files.sort(key=lambda item: int(item["bytes"]), reverse=True)
    result = {
        "pipeline_mode": PIPELINE_MODE,
        "stage": "inspect-workdir",
        "manifest_path": str(Path(str(manifest_path)).resolve()),
        "work_dir": str(manifest.get("work_dir", "")),
        "input_file_bytes": _input_file_bytes(manifest),
        "max_rss_mb": _max_rss_mb(),
        "manifest": manifest_stat,
        "seed_file": seed_stat,
        "seed_stats": dict(manifest.get("seed_stats", {})),
        "seed_audit": dict(manifest.get("seed_audit", {})),
        "spatial_index_stats": dict(manifest.get("spatial_index_stats", {})),
        "shards": shard_summary,
        "exact_index": exact_index_summary,
        "exact_source_shards": source_summary,
        "exact_target_buckets": target_bucket_summary,
        "candidate_bundle_buckets": candidate_bundle_summary,
        "coverage_shards": coverage_summary,
        "lsf_wrapper": dict(manifest.get("lsf_wrapper", {})),
        "largest_files": largest_files[:20],
        "inspect_notes": [
            "npz_zip_uncompressed_bytes 来自 zip metadata，不会加载数组。",
            "JSON 摘要会读取对应 JSON 文件；大样本下可优先观察 bytes 与 missing 计数。",
        ],
        "timing_seconds": {"inspect_workdir": round(time.perf_counter() - started, 6)},
    }
    if output_path:
        _write_json(output_path, result)
    return result


def run_local_stage(
    input_path,
    work_dir,
    output_path,
    config,
    register_ops,
    shard_count,
    shard_size,
    distributed_coverage=False,
    coverage_shard_count=1,
    coverage_shard_size=0,
):
    """run-local 阶段：本地顺序模拟 prepare/run-shard/merge 或分布式 coverage 流程。"""

    manifest = prepare_stage(input_path, work_dir, config, register_ops, shard_count, shard_size)
    manifest_path = Path(str(manifest["work_dir"])) / "manifest.json"
    for shard in manifest["shards"]:
        run_shard_stage(manifest_path, int(shard["shard_id"]))
    if distributed_coverage:
        manifest = prepare_coverage_stage(manifest_path, coverage_shard_count, coverage_shard_size)
        for coverage_shard in manifest.get("coverage_shards", []):
            run_coverage_shard_stage(manifest_path, int(coverage_shard["coverage_shard_id"]))
        return merge_coverage_stage(manifest_path, output_path)
    return merge_stage(manifest_path, output_path)


def _add_common_runtime_args(parser):
    """添加 prepare/run-local 共用参数。"""

    parser.add_argument("input_path", help="输入 OAS 文件")
    parser.add_argument("--config", default=None, help="可选 JSON 配置文件路径")
    parser.add_argument("--work-dir", required=True, help="LSF 工作目录")
    parser.add_argument("--clip-size", type=float, default=1.35, help="clip 方形窗口边长，单位 um")
    parser.add_argument("--geometry-match-mode", choices=["acc", "ecc"], default="ecc", help="几何匹配模式")
    parser.add_argument("--area-match-ratio", type=float, default=0.96, help="ACC 面积匹配阈值")
    parser.add_argument("--edge-tolerance-um", type=float, default=0.02, help="ECC 边缘容差，单位 um")
    parser.add_argument("--pixel-size-nm", type=int, default=DEFAULT_PIXEL_SIZE_NM, help="raster 像素尺寸，单位 nm")
    parser.add_argument("--apply-layer-ops", action="store_true", help="启用已注册的 layer operation")
    parser.add_argument(
        "--register-op",
        action="append",
        nargs=4,
        metavar=("SOURCE_LAYER", "TARGET_LAYER", "OPERATION", "RESULT_LAYER"),
        help="注册 layer operation，例如 --register-op 1/0 2/0 subtract 10/0",
    )
    parser.add_argument("--shard-count", type=int, default=1, help="未指定 shard-size 时的目标 shard 数")
    parser.add_argument("--shard-size", type=int, default=2000, help="每个 seed shard 的 seed 数")


def build_parser():
    """构建 v2_lsf CLI 参数解析器。"""

    parser = argparse.ArgumentParser(description="面向 LSF 的 optimized_v2_lsf 版图聚类入口")
    subparsers = parser.add_subparsers(dest="stage")

    prepare_parser = subparsers.add_parser("prepare", help="生成 seed manifest 和 shard 命令")
    _add_common_runtime_args(prepare_parser)

    shard_parser = subparsers.add_parser("run-shard", help="运行一个 seed shard")
    shard_parser.add_argument("--manifest", required=True, help="manifest.json 路径")
    shard_parser.add_argument("--shard-id", type=int, required=True, help="要运行的 shard id")

    merge_parser = subparsers.add_parser("merge", help="集中式合并所有 seed shard 输出")
    merge_parser.add_argument("--manifest", required=True, help="manifest.json 路径")
    merge_parser.add_argument("--output", "-o", default=None, help="最终结果 JSON 路径")

    coverage_prepare_parser = subparsers.add_parser("prepare-coverage", help="生成 coverage shard 计划")
    coverage_prepare_parser.add_argument("--manifest", required=True, help="manifest.json 路径")
    coverage_prepare_parser.add_argument("--coverage-shard-count", type=int, default=1, help="目标 coverage shard 数")
    coverage_prepare_parser.add_argument("--coverage-shard-size", type=int, default=0, help="每个 coverage shard 的 exact cluster 数")

    coverage_shard_parser = subparsers.add_parser("run-coverage-shard", help="运行一个 coverage shard")
    coverage_shard_parser.add_argument("--manifest", required=True, help="manifest.json 路径")
    coverage_shard_parser.add_argument("--coverage-shard-id", type=int, required=True, help="要运行的 coverage shard id")

    coverage_merge_parser = subparsers.add_parser("merge-coverage", help="合并 coverage shard 输出")
    coverage_merge_parser.add_argument("--manifest", required=True, help="manifest.json 路径")
    coverage_merge_parser.add_argument("--output", "-o", default=None, help="最终结果 JSON 路径")

    inspect_parser = subparsers.add_parser("inspect-workdir", help="检查 LSF 工作目录产物规模")
    inspect_parser.add_argument("--manifest", required=True, help="manifest.json 路径")
    inspect_parser.add_argument("--output", "-o", default=None, help="可选检查摘要 JSON 路径")

    local_parser = subparsers.add_parser("run-local", help="本地顺序模拟 LSF 流程")
    _add_common_runtime_args(local_parser)
    local_parser.add_argument("--output", "-o", required=True, help="最终结果 JSON 路径")
    local_parser.add_argument("--distributed-coverage", action="store_true", help="本地顺序模拟 prepare-coverage/coverage-shard/merge-coverage")
    local_parser.add_argument("--coverage-shard-count", type=int, default=1, help="目标 coverage shard 数")
    local_parser.add_argument("--coverage-shard-size", type=int, default=0, help="每个 coverage shard 的 exact cluster 数")
    return parser


def main():
    """命令行主函数。"""

    parser = build_parser()
    args = parser.parse_args()
    if not args.stage:
        parser.print_help()
        return 2
    if args.stage == "prepare":
        config = _config_payload(args)
        manifest = prepare_stage(args.input_path, args.work_dir, config, args.register_op or [], args.shard_count, args.shard_size)
        print(json.dumps({"manifest": str(Path(args.work_dir) / "manifest.json"), "shard_count": manifest["shard_count"]}, ensure_ascii=False))
        return 0
    if args.stage == "run-shard":
        summary = run_shard_stage(args.manifest, args.shard_id)
        print(json.dumps({"shard_id": int(args.shard_id), "marker_count": summary["marker_count"]}, ensure_ascii=False))
        return 0
    if args.stage == "merge":
        result = merge_stage(args.manifest, args.output)
        print(json.dumps({"output": args.output or _read_json(args.manifest).get("result_output"), "total_clusters": result["total_clusters"]}, ensure_ascii=False))
        return 0
    if args.stage == "prepare-coverage":
        manifest = prepare_coverage_stage(args.manifest, args.coverage_shard_count, args.coverage_shard_size)
        print(json.dumps({"manifest": str(Path(args.manifest).resolve()), "coverage_shard_count": manifest["coverage_shard_count"]}, ensure_ascii=False))
        return 0
    if args.stage == "run-coverage-shard":
        summary = run_coverage_shard_stage(args.manifest, args.coverage_shard_id)
        print(json.dumps({"coverage_shard_id": int(args.coverage_shard_id), "candidate_count": summary["candidate_count"]}, ensure_ascii=False))
        return 0
    if args.stage == "merge-coverage":
        result = merge_coverage_stage(args.manifest, args.output)
        print(json.dumps({"output": args.output or _read_json(args.manifest).get("result_output"), "total_clusters": result["total_clusters"]}, ensure_ascii=False))
        return 0
    if args.stage == "inspect-workdir":
        result = inspect_workdir_stage(args.manifest, args.output)
        print(
            json.dumps(
                {
                    "manifest": str(Path(args.manifest).resolve()),
                    "shard_count": result["shards"]["count"],
                    "coverage_shard_count": result["coverage_shards"]["count"],
                    "coverage_candidate_count": result["coverage_shards"]["candidate_count"],
                    "coverage_value_count": result["coverage_shards"]["coverage_value_count"],
                    "output": args.output,
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.stage == "run-local":
        config = _config_payload(args)
        result = run_local_stage(
            args.input_path,
            args.work_dir,
            args.output,
            config,
            args.register_op or [],
            args.shard_count,
            args.shard_size,
            args.distributed_coverage,
            args.coverage_shard_count,
            args.coverage_shard_size,
        )
        print(json.dumps({"output": args.output, "total_clusters": result["total_clusters"]}, ensure_ascii=False))
        return 0
    raise ValueError("Unsupported stage: %s" % args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
