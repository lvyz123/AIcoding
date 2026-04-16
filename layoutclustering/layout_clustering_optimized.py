#!/usr/bin/env python3
"""Optimized marker-driven layout clustering.

Single pipeline:
marker -> exact hash -> topology/signature prefilter -> ACC/ECC edges
-> greedy set cover -> representative-member verification -> export.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy import ndimage

from mainline import (
    DEFAULT_PIXEL_SIZE_NM,
    ECC_DONUT_OVERLAP_RATIO,
    ECC_RESIDUAL_RATIO,
    CandidateClip,
    ExactCluster,
    MainlineRunner,
    MarkerRecord,
    _make_sample_filename,
    _materialize_clip_bitmap,
)


GRAPH_INVARIANT_LIMIT = 0.22
GRAPH_TOPOLOGY_THRESHOLD = 6.5
GRAPH_SIGNATURE_THRESHOLD = 0.74
STRICT_INVARIANT_LIMIT = 0.20
STRICT_TOPOLOGY_THRESHOLD = 3.0
STRICT_SIGNATURE_THRESHOLD = 0.84


@dataclass(frozen=True)
class GraphDescriptor:
    invariants: np.ndarray
    topology: np.ndarray
    signature_grid: np.ndarray
    signature_proj_x: np.ndarray
    signature_proj_y: np.ndarray


def _empty_prefilter_stats() -> Dict[str, int]:
    return {
        "exact_hash_pass": 0,
        "invariant_reject": 0,
        "topology_reject": 0,
        "signature_reject": 0,
        "geometry_reject": 0,
        "geometry_pass": 0,
    }


def _empty_verification_stats() -> Dict[str, int]:
    return {
        "verified_pass": 0,
        "verified_reject": 0,
        "singleton_created": 0,
    }


def _pool_bitmap(bitmap: np.ndarray, bins: int = 10) -> np.ndarray:
    src = np.asarray(bitmap, dtype=np.float32)
    pooled = np.zeros((bins, bins), dtype=np.float32)
    h, w = src.shape
    row_edges = np.linspace(0, h, bins + 1, dtype=np.int32)
    col_edges = np.linspace(0, w, bins + 1, dtype=np.int32)
    for row in range(bins):
        r0, r1 = int(row_edges[row]), int(row_edges[row + 1])
        for col in range(bins):
            c0, c1 = int(col_edges[col]), int(col_edges[col + 1])
            cell = src[r0:max(r1, r0 + 1), c0:max(c1, c0 + 1)]
            pooled[row, col] = float(np.mean(cell)) if cell.size else 0.0
    total = float(np.sum(pooled))
    if total > 0.0:
        pooled /= total
    return pooled


def _bitmap_descriptor(bitmap: np.ndarray) -> GraphDescriptor:
    mask = np.asarray(bitmap, dtype=bool)
    h, w = mask.shape
    total_px = max(int(mask.size), 1)
    active = np.argwhere(mask)
    fill_ratio = float(active.shape[0]) / float(total_px)

    if active.size == 0:
        invariants = np.zeros(8, dtype=np.float64)
        topology = np.zeros(8, dtype=np.float64)
    else:
        rows = active[:, 0].astype(np.float64)
        cols = active[:, 1].astype(np.float64)
        bbox_h = float(np.max(rows) - np.min(rows) + 1.0)
        bbox_w = float(np.max(cols) - np.min(cols) + 1.0)
        bbox_long = max(bbox_w, bbox_h)
        bbox_short = min(bbox_w, bbox_h)
        span = max(float(h), float(w), 1.0)
        extent_area = max(bbox_w * bbox_h, 1.0)
        centroid = np.asarray([float(np.mean(rows)), float(np.mean(cols))])
        radii = np.linalg.norm(active.astype(np.float64) - centroid[None, :], axis=1)

        structure = np.ones((3, 3), dtype=bool)
        labels, component_count = ndimage.label(mask, structure=structure)
        if component_count > 0:
            component_sizes = np.bincount(labels.reshape(-1))[1:].astype(np.float64)
        else:
            component_sizes = np.empty(0, dtype=np.float64)
        logs = np.log1p(component_sizes) if component_sizes.size else np.zeros(1, dtype=np.float64)

        invariants = np.asarray(
            [
                math.log1p(float(component_count)),
                fill_ratio,
                bbox_long / span,
                bbox_short / span,
                bbox_short / max(bbox_long, 1.0),
                float(active.shape[0]) / extent_area,
                float(np.mean(radii)) / span,
                float(np.std(radii)) / span,
            ],
            dtype=np.float64,
        )
        topology = np.asarray(
            [
                math.log1p(float(component_count)),
                float(np.max(logs)),
                float(np.mean(logs)),
                float(np.std(logs)),
                bbox_long / span,
                bbox_short / span,
                fill_ratio,
                float(active.shape[0]) / extent_area,
            ],
            dtype=np.float64,
        )

    pooled = _pool_bitmap(mask, bins=10)
    return GraphDescriptor(
        invariants=invariants,
        topology=topology,
        signature_grid=pooled.reshape(-1).astype(np.float32),
        signature_proj_x=np.sum(pooled, axis=1, dtype=np.float32),
        signature_proj_y=np.sum(pooled, axis=0, dtype=np.float32),
    )


def _descriptor(owner: Any) -> GraphDescriptor:
    descriptor = owner.match_cache.get("optimized_graph_descriptor")
    if descriptor is None:
        descriptor = _bitmap_descriptor(owner.clip_bitmap)
        owner.match_cache["optimized_graph_descriptor"] = descriptor
    return descriptor


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    a = np.asarray(vec_a, dtype=np.float64).ravel()
    b = np.asarray(vec_b, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.dot(a, b) / denom)


def _signature_similarity(desc_a: GraphDescriptor, desc_b: GraphDescriptor) -> float:
    return float(
        0.6 * _cosine_similarity(desc_a.signature_grid, desc_b.signature_grid)
        + 0.2 * _cosine_similarity(desc_a.signature_proj_x, desc_b.signature_proj_x)
        + 0.2 * _cosine_similarity(desc_a.signature_proj_y, desc_b.signature_proj_y)
    )


def _invariant_distance(desc_a: GraphDescriptor, desc_b: GraphDescriptor) -> Tuple[float, bool]:
    a = np.asarray(desc_a.invariants, dtype=np.float64)
    b = np.asarray(desc_b.invariants, dtype=np.float64)
    floors = np.asarray([0.25, 0.02, 0.02, 0.02, 0.03, 0.03, 0.02, 0.02], dtype=np.float64)
    weights = np.asarray([0.08, 0.24, 0.10, 0.08, 0.18, 0.14, 0.10, 0.08], dtype=np.float64)
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), floors)
    errs = np.minimum(np.abs(a - b) / denom, 1.0)
    critical = bool(errs[1] > 0.45 or errs[4] > 0.45 or errs[5] > 0.45)
    return float(np.dot(errs, weights)), critical


def _topology_distance(desc_a: GraphDescriptor, desc_b: GraphDescriptor) -> float:
    a = np.asarray(desc_a.topology, dtype=np.float64)
    b = np.asarray(desc_b.topology, dtype=np.float64)
    return float(np.linalg.norm(a - b))


def _graph_prefilter_passes(candidate: CandidateClip, target: MarkerRecord, *, strict: bool) -> Tuple[bool, str]:
    desc_a = _descriptor(candidate)
    desc_b = _descriptor(target)
    invariant_limit = STRICT_INVARIANT_LIMIT if strict else GRAPH_INVARIANT_LIMIT
    topology_threshold = STRICT_TOPOLOGY_THRESHOLD if strict else GRAPH_TOPOLOGY_THRESHOLD
    signature_threshold = STRICT_SIGNATURE_THRESHOLD if strict else GRAPH_SIGNATURE_THRESHOLD

    invariant_score, critical = _invariant_distance(desc_a, desc_b)
    if critical or invariant_score > invariant_limit:
        return False, "invariant"
    if _topology_distance(desc_a, desc_b) > topology_threshold:
        return False, "topology"
    if _signature_similarity(desc_a, desc_b) < signature_threshold:
        return False, "signature"
    return True, "pass"


def _ecc_cache(owner: Any, tol_px: int) -> Dict[str, Any]:
    key = f"optimized_ecc_tol_{int(tol_px)}"
    cache = owner.match_cache.get(key)
    if cache is not None:
        return cache

    bitmap = np.ascontiguousarray(owner.clip_bitmap.astype(bool, copy=False))
    cache = {
        "bitmap": bitmap,
        "area": int(np.count_nonzero(bitmap)),
    }
    if tol_px > 0:
        structure = np.ones((2 * int(tol_px) + 1, 2 * int(tol_px) + 1), dtype=bool)
        dilated = ndimage.binary_dilation(bitmap, structure=structure)
        eroded = ndimage.binary_erosion(bitmap, structure=structure, border_value=0)
        cache["dilated"] = np.ascontiguousarray(dilated, dtype=bool)
        cache["donut"] = np.ascontiguousarray(dilated & ~eroded, dtype=bool)
        cache["donut_area"] = int(np.count_nonzero(cache["donut"]))
    owner.match_cache[key] = cache
    return cache


def _ecc_match_cached(candidate: CandidateClip, target: MarkerRecord, edge_tolerance_um: float, pixel_size_um: float) -> bool:
    if candidate.clip_bitmap.shape != target.clip_bitmap.shape:
        return False
    if not candidate.clip_bitmap.any() and not target.clip_bitmap.any():
        return True
    if not candidate.clip_bitmap.any() or not target.clip_bitmap.any():
        return False

    tol_px = max(0, int(math.ceil(float(edge_tolerance_um) / max(float(pixel_size_um), 1e-12) - 1e-12)))
    if tol_px <= 0:
        return bool(np.array_equal(candidate.clip_bitmap, target.clip_bitmap))

    cand = _ecc_cache(candidate, tol_px)
    tgt = _ecc_cache(target, tol_px)
    cand_area = max(float(cand["area"]), 1.0)
    tgt_area = max(float(tgt["area"]), 1.0)
    residual_cand = np.count_nonzero(cand["bitmap"] & ~tgt["dilated"]) / cand_area
    residual_tgt = np.count_nonzero(tgt["bitmap"] & ~cand["dilated"]) / tgt_area
    if residual_cand > ECC_RESIDUAL_RATIO or residual_tgt > ECC_RESIDUAL_RATIO:
        return False
    if int(cand["donut_area"]) == 0 or int(tgt["donut_area"]) == 0:
        return True
    overlap = int(np.count_nonzero(cand["donut"] & tgt["donut"]))
    denom = max(min(int(cand["donut_area"]), int(tgt["donut_area"])), 1)
    return float(overlap / denom) >= ECC_DONUT_OVERLAP_RATIO


class OptimizedMainlineRunner(MainlineRunner):
    def __init__(self, *, config: Dict[str, Any], temp_dir: Path):
        clean_config = {
            "clip_size_um": float(config.get("clip_size_um", 1.35)),
            "hotspot_layer": str(config["marker_layer"]),
            "matching_mode": str(config.get("geometry_match_mode", "ecc")),
            "solver": "greedy",
            "geometry_mode": "exact",
            "pixel_size_nm": int(config.get("pixel_size_nm", DEFAULT_PIXEL_SIZE_NM)),
            "area_match_ratio": float(config.get("area_match_ratio", 0.96)),
            "edge_tolerance_um": float(config.get("edge_tolerance_um", 0.02)),
            "clip_shift_directions": "left,right,up,down",
            "clip_shift_boundary_tolerance_um": float(config.get("edge_tolerance_um", 0.02)),
        }
        super().__init__(config=clean_config, temp_dir=temp_dir, layer_processor=None)
        self.prefilter_stats = _empty_prefilter_stats()
        self.final_verification_stats = _empty_verification_stats()
        self._base_candidate_by_exact_id: Dict[int, CandidateClip] = {}

    def _build_marker_record(self, filepath: Path, marker_index: int, marker_poly: Any, layout_index: Any) -> MarkerRecord | None:
        record = super()._build_marker_record(filepath, marker_index, marker_poly, layout_index)
        if record is not None:
            record.match_cache["optimized_graph_descriptor"] = _bitmap_descriptor(record.clip_bitmap)
        return record

    def _build_candidate_clip(
        self,
        cluster: ExactCluster,
        clip_bbox: Tuple[float, float, float, float],
        clip_bbox_q: Tuple[int, int, int, int],
        bitmap: np.ndarray,
        shift_direction: str,
        shift_distance_um: float,
        candidate_index: int,
    ) -> CandidateClip:
        candidate = super()._build_candidate_clip(
            cluster,
            clip_bbox,
            clip_bbox_q,
            bitmap,
            shift_direction,
            shift_distance_um,
            candidate_index,
        )
        candidate.coverage = {int(cluster.exact_cluster_id)} if str(shift_direction) == "base" else set()
        candidate.match_cache["optimized_graph_descriptor"] = _bitmap_descriptor(candidate.clip_bitmap)
        return candidate

    def _generate_candidates_for_cluster(self, cluster: ExactCluster) -> List[CandidateClip]:
        candidates = super()._generate_candidates_for_cluster(cluster)
        base = next((candidate for candidate in candidates if candidate.shift_direction == "base"), None)
        if base is not None:
            self._base_candidate_by_exact_id[int(cluster.exact_cluster_id)] = base
        return candidates

    def _select_solver(self, candidate_count: int) -> str:
        del candidate_count
        return "greedy"

    def _greedy_cover(self, candidates: Sequence[CandidateClip], exact_clusters: Sequence[ExactCluster]) -> List[CandidateClip]:
        uncovered = {int(cluster.exact_cluster_id) for cluster in exact_clusters}
        weights = {int(cluster.exact_cluster_id): int(cluster.weight) for cluster in exact_clusters}
        selected: List[CandidateClip] = []

        while uncovered:
            best = max(
                candidates,
                key=lambda candidate: (
                    sum(weights[cid] for cid in (candidate.coverage & uncovered)),
                    len(candidate.coverage & uncovered),
                    1 if candidate.shift_direction == "base" else 0,
                    -abs(candidate.shift_distance_um),
                    -int(candidate.origin_exact_cluster_id),
                ),
            )
            covered_now = set(best.coverage) & uncovered
            if not covered_now:
                missing = min(uncovered)
                best = self._base_candidate_by_exact_id[missing]
                covered_now = {missing}
            selected.append(best)
            uncovered -= covered_now
        return selected

    def _geometry_passes(self, candidate: CandidateClip, target: MarkerRecord) -> bool:
        if candidate.clip_bitmap.shape != target.clip_bitmap.shape:
            return False
        if self.matching_mode == "acc":
            xor_ratio = float(np.count_nonzero(candidate.clip_bitmap ^ target.clip_bitmap)) / float(
                max(candidate.clip_bitmap.size, 1)
            )
            return bool(xor_ratio <= max(0.0, 1.0 - float(self.area_match_ratio)) + 1e-12)
        return _ecc_match_cached(candidate, target, self.edge_tolerance_um, self.pixel_size_um)

    def _candidate_matches_exact(
        self,
        candidate: CandidateClip,
        exact_cluster: ExactCluster,
        *,
        strict: bool,
        stats: Dict[str, int] | None = None,
    ) -> bool:
        target = exact_cluster.representative
        if candidate.clip_hash == target.clip_hash:
            if stats is not None and not strict:
                stats["exact_hash_pass"] += 1
            return True

        prefilter_ok, reason = _graph_prefilter_passes(candidate, target, strict=strict)
        if not prefilter_ok:
            if stats is not None and not strict:
                stats[f"{reason}_reject"] += 1
            return False
        if not self._geometry_passes(candidate, target):
            if stats is not None and not strict:
                stats["geometry_reject"] += 1
            return False
        if stats is not None and not strict:
            stats["geometry_pass"] += 1
        return True

    def _evaluate_candidate_coverage(
        self,
        candidates: Sequence[CandidateClip],
        exact_clusters: Sequence[ExactCluster],
    ) -> None:
        self.prefilter_stats = _empty_prefilter_stats()
        for candidate in candidates:
            candidate.coverage = {int(candidate.origin_exact_cluster_id)} if candidate.shift_direction == "base" else set()

        for candidate in candidates:
            for exact_cluster in exact_clusters:
                if self._candidate_matches_exact(candidate, exact_cluster, strict=False, stats=self.prefilter_stats):
                    candidate.coverage.add(int(exact_cluster.exact_cluster_id))

    def _verified_cluster_units(
        self,
        selected_candidates: Sequence[CandidateClip],
        exact_clusters: Sequence[ExactCluster],
    ) -> List[Tuple[CandidateClip, List[ExactCluster]]]:
        assignments = self._assign_exact_clusters(selected_candidates, exact_clusters)
        self.final_verification_stats = _empty_verification_stats()
        units: List[Tuple[CandidateClip, List[ExactCluster]]] = []

        for candidate in selected_candidates:
            accepted: List[ExactCluster] = []
            for exact_cluster in assignments.get(candidate.candidate_id, []):
                if self._candidate_matches_exact(candidate, exact_cluster, strict=True):
                    accepted.append(exact_cluster)
                    self.final_verification_stats["verified_pass"] += 1
                else:
                    self.final_verification_stats["verified_reject"] += 1
                    self.final_verification_stats["singleton_created"] += 1
                    base = self._base_candidate_by_exact_id[int(exact_cluster.exact_cluster_id)]
                    units.append((base, [exact_cluster]))
            if accepted:
                units.append((candidate, accepted))
        return units

    def _sample_metadata(self, record: MarkerRecord) -> Dict[str, Any]:
        return {
            "pipeline_mode": "optimized",
            "marker_id": str(record.marker_id),
            "exact_cluster_id": int(record.exact_cluster_id),
            "geometry_match_mode": str(self.matching_mode),
            "source_path": str(record.source_path),
            "source_name": str(record.source_name),
            "marker_bbox": list(record.marker_bbox),
            "marker_center": list(record.marker_center),
            "clip_bbox": list(record.clip_bbox),
            "selected_candidate_id": None,
            "selected_shift_direction": None,
            "selected_shift_distance_um": None,
        }

    def _build_results(
        self,
        marker_records: Sequence[MarkerRecord],
        exact_clusters: Sequence[ExactCluster],
        selected_candidates: Sequence[CandidateClip],
        solver_used: str,
        runtime_summary: Dict[str, float],
        candidate_count: int,
    ) -> Dict[str, Any]:
        del solver_used
        sample_dir = self.temp_dir / "samples"
        representative_dir = self.temp_dir / "representatives"
        sample_dir.mkdir(parents=True, exist_ok=True)
        representative_dir.mkdir(parents=True, exist_ok=True)

        ordered_records = list(sorted(marker_records, key=lambda item: (item.source_name, item.marker_id)))
        sample_file_map: Dict[str, str] = {}
        sample_index_map: Dict[str, int] = {}
        file_list: List[str] = []
        file_metadata: List[Dict[str, Any]] = []

        for sample_index, record in enumerate(ordered_records):
            sample_path = sample_dir / _make_sample_filename("sample", record.source_name, sample_index)
            sample_file = _materialize_clip_bitmap(record.clip_bitmap, record.clip_bbox, record.marker_id, sample_path, self.pixel_size_um)
            sample_file_map[record.marker_id] = sample_file
            sample_index_map[record.marker_id] = int(sample_index)
            file_list.append(sample_file)
            file_metadata.append(self._sample_metadata(record))

        cluster_units = self._verified_cluster_units(selected_candidates, exact_clusters)
        clusters_output: List[Dict[str, Any]] = []

        for cluster_index, (candidate, assigned_exact_clusters) in enumerate(cluster_units):
            cluster_members = list(
                sorted(
                    (member for exact_cluster in assigned_exact_clusters for member in exact_cluster.members),
                    key=lambda item: (item.source_name, item.marker_id),
                )
            )
            if not cluster_members:
                continue

            rep_path = representative_dir / _make_sample_filename("rep", cluster_members[0].source_name, cluster_index)
            representative_file = _materialize_clip_bitmap(
                candidate.clip_bitmap,
                candidate.clip_bbox,
                candidate.candidate_id,
                rep_path,
                self.pixel_size_um,
            )
            sample_indices = [sample_index_map[member.marker_id] for member in cluster_members]
            sample_files = [sample_file_map[member.marker_id] for member in cluster_members]
            sample_metadata = []
            for member in cluster_members:
                sample_index = sample_index_map[member.marker_id]
                metadata = dict(file_metadata[sample_index])
                metadata.update(
                    {
                        "selected_candidate_id": str(candidate.candidate_id),
                        "selected_shift_direction": str(candidate.shift_direction),
                        "selected_shift_distance_um": float(candidate.shift_distance_um),
                    }
                )
                file_metadata[sample_index] = dict(metadata)
                sample_metadata.append(metadata)

            exact_cluster_ids = [int(exact_cluster.exact_cluster_id) for exact_cluster in assigned_exact_clusters]
            clusters_output.append(
                {
                    "cluster_id": int(cluster_index),
                    "pipeline_mode": "optimized",
                    "size": int(len(cluster_members)),
                    "sample_indices": sample_indices,
                    "sample_files": sample_files,
                    "sample_metadata": sample_metadata,
                    "representative_file": representative_file,
                    "representative_metadata": {
                        "pipeline_mode": "optimized",
                        "marker_id": str(candidate.source_marker_id),
                        "exact_cluster_id": int(candidate.origin_exact_cluster_id),
                        "geometry_match_mode": str(self.matching_mode),
                        "selected_candidate_id": str(candidate.candidate_id),
                        "selected_shift_direction": str(candidate.shift_direction),
                        "selected_shift_distance_um": float(candidate.shift_distance_um),
                        "coverage_exact_cluster_ids": exact_cluster_ids,
                    },
                    "marker_id": str(candidate.source_marker_id),
                    "exact_cluster_id": int(candidate.origin_exact_cluster_id),
                    "marker_ids": [str(member.marker_id) for member in cluster_members],
                    "exact_cluster_ids": exact_cluster_ids,
                    "geometry_match_mode": str(self.matching_mode),
                    "selected_candidate_id": str(candidate.candidate_id),
                    "selected_shift_direction": str(candidate.shift_direction),
                    "selected_shift_distance_um": float(candidate.shift_distance_um),
                }
            )

        cluster_sizes = [int(cluster["size"]) for cluster in clusters_output]
        max_shift = max((float(cluster["selected_shift_distance_um"]) for cluster in clusters_output), default=0.0)
        result = {
            "pipeline_mode": "optimized",
            "geometry_match_mode": str(self.matching_mode),
            "pixel_size_nm": int(self.pixel_size_nm),
            "area_match_ratio": float(self.area_match_ratio),
            "edge_tolerance_um": float(self.edge_tolerance_um),
            "marker_count": int(len(marker_records)),
            "exact_cluster_count": int(len(exact_clusters)),
            "candidate_count": int(candidate_count),
            "selected_candidate_count": int(len(selected_candidates)),
            "total_clusters": int(len(clusters_output)),
            "total_samples": int(len(file_list)),
            "total_files": int(len(file_list)),
            "cluster_sizes": cluster_sizes,
            "max_shift_distance_um": float(max_shift),
            "prefilter_stats": dict(self.prefilter_stats),
            "final_verification_stats": dict(self.final_verification_stats),
            "clusters": clusters_output,
            "file_list": file_list,
            "file_metadata": file_metadata,
            "result_summary": {
                "pipeline_mode": "optimized",
                "geometry_match_mode": str(self.matching_mode),
                "pixel_size_nm": int(self.pixel_size_nm),
                "area_match_ratio": float(self.area_match_ratio),
                "edge_tolerance_um": float(self.edge_tolerance_um),
                "marker_count": int(len(marker_records)),
                "exact_cluster_count": int(len(exact_clusters)),
                "candidate_count": int(candidate_count),
                "selected_candidate_count": int(len(selected_candidates)),
                "total_clusters": int(len(clusters_output)),
                "cluster_sizes": cluster_sizes,
                "max_shift_distance_um": float(max_shift),
                "prefilter_stats": dict(self.prefilter_stats),
                "final_verification_stats": dict(self.final_verification_stats),
                "timing_seconds": dict(runtime_summary),
            },
            "config": {
                "marker_layer": str(self.hotspot_layer),
                "clip_size": float(self.clip_size_um),
                "geometry_match_mode": str(self.matching_mode),
                "area_match_ratio": float(self.area_match_ratio),
                "edge_tolerance_um": float(self.edge_tolerance_um),
                "pixel_size_nm": int(self.pixel_size_nm),
                "graph_invariant_limit": GRAPH_INVARIANT_LIMIT,
                "graph_topology_threshold": GRAPH_TOPOLOGY_THRESHOLD,
                "graph_signature_threshold": GRAPH_SIGNATURE_THRESHOLD,
                "strict_invariant_limit": STRICT_INVARIANT_LIMIT,
                "strict_topology_threshold": STRICT_TOPOLOGY_THRESHOLD,
                "strict_signature_threshold": STRICT_SIGNATURE_THRESHOLD,
            },
            "cluster_review": {},
        }
        return result


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _export_review(result: Dict[str, Any], review_dir: str) -> Dict[str, Any]:
    review_root = Path(review_dir)
    review_root.mkdir(parents=True, exist_ok=True)
    representative_files: List[str] = []
    exported_file_count = 0
    missing_files: List[str] = []

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
    }
    if missing_files:
        info["missing_files_preview"] = missing_files[:10]
    result["cluster_review"] = dict(info)
    return info


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimized marker-driven layout clustering")
    parser.add_argument("input_path", help="Input OASIS file or directory")
    parser.add_argument("--output", "-o", default="clustering_optimized_results.json", help="Output JSON path")
    parser.add_argument("--marker-layer", required=True, help="Marker layer in layer/datatype form, e.g. 999/0")
    parser.add_argument("--clip-size", type=float, default=1.35, help="Clip side length in um")
    parser.add_argument("--geometry-match-mode", choices=["acc", "ecc"], default="ecc", help="Final geometry gate")
    parser.add_argument("--area-match-ratio", type=float, default=0.96, help="ACC area match threshold")
    parser.add_argument("--edge-tolerance-um", type=float, default=0.02, help="ECC edge tolerance in um")
    parser.add_argument("--pixel-size-nm", type=int, default=DEFAULT_PIXEL_SIZE_NM, help="Raster pixel size in nm")
    parser.add_argument("--review-dir", default=None, help="Optional review directory")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    temp_root = Path(__file__).resolve().parent / "_temp_runs"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"layout_clustering_optimized_{uuid.uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "marker_layer": str(args.marker_layer),
        "clip_size_um": float(args.clip_size),
        "geometry_match_mode": str(args.geometry_match_mode),
        "area_match_ratio": float(args.area_match_ratio),
        "edge_tolerance_um": float(args.edge_tolerance_um),
        "pixel_size_nm": int(args.pixel_size_nm),
    }
    try:
        runner = OptimizedMainlineRunner(config=config, temp_dir=temp_dir)
        result = runner.run(str(args.input_path))
        if args.review_dir:
            _export_review(result, str(args.review_dir))
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False, default=_json_default)
        print(f"Results written to: {output_path}")
        print(f"Clusters: {result.get('total_clusters', 0)}")
        print(f"Markers: {result.get('marker_count', 0)}")
        print(f"Final verification: {result.get('final_verification_stats', {})}")
        return 0
    except Exception as exc:
        print(f"Run failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
