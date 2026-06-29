#!/usr/bin/env python3
"""CD-SEM/aerial image 轻量仿真器。

本模块用于给 `hotspotdetection` 的 recipe 构建流程准备可跑通的 behavior 输入包。
它不做训练，也不声称替代真实设备或严格光刻仿真；第一版采用物理启发的解析近似，
把 OAS 中指定 pattern layer 的 marker-centered layout clip 转换成 CD-SEM-like 图像。

整体算法流程:
1. 读取 OAS/OASIS 输入，按显式 `--marker-layer` 找到所有 marker。
2. 仅保留显式 `--pattern-layer` 指定的版图图形，marker layer 不参与成像。
3. 围绕每个 marker 生成固定尺寸 layout bitmap。
4. 对 bitmap 执行 CD bias 近似、Gaussian PSF 模糊、signed-distance 边缘带增强、
   行方向扫描非均匀性和轻量 shot/Gaussian noise，得到二维 float32 图像。
5. 对图像按配置做 0..1 归一化，写出 `aerial_npz/`、可选 PNG、`behavior.jsonl`、
   `simulation_summary.json` 和 `simulation_quality_audit.json`。

主函数使用说明:
python sem_simulator.py input.oas --marker-layer 12530/2 --pattern-layer 2530/0 --output-dir sim_behavior

注意点:
- `--marker-layer` 和 `--pattern-layer` 必须显式提供，第一版不自动猜测 layer。
- 输出目录可直接传给 `hotspotdetection/recipe_site_selector.py --behavior-manifest`。
- `aerial_npz` 字段沿用下游硬接口，里面保存的是仿真 CD-SEM-like behavior image。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage


REPO_ROOT = Path(__file__).resolve().parents[1]
HOTSPOT_DIR = REPO_ROOT / "hotspotdetection"
if str(HOTSPOT_DIR) not in sys.path:
    sys.path.insert(0, str(HOTSPOT_DIR))

from layer_operations import LayerOperationProcessor  # noqa: E402
from layout_utils import DEFAULT_PIXEL_SIZE_NM, LayoutIndex, MarkerRasterBuilder, MarkerRecord  # noqa: E402


IMAGE_KEY = "image"
SIMULATION_MODEL = "sem_simulator_v1_1"
MODEL_PROFILES = ("clean", "nominal", "stress")
NORMALIZATION_MODES = ("per-image", "fixed")
RISK_MODES = ("constant", "layout-proxy")
LOW_STRUCTURE_DENSITY_THRESHOLD = 0.002
LOW_STRUCTURE_EDGE_THRESHOLD = 0.006


@dataclass(frozen=True)
class SemModelParams:
    """保存轻量 SEM 成像模型参数。"""

    profile: str = "nominal"
    psf_sigma_nm: float = 18.0
    cd_bias_nm: float = 3.0
    background_level: float = 0.18
    material_gain: float = 0.56
    edge_gain: float = 0.32
    shot_noise_sigma: float = 0.012
    gaussian_noise_sigma: float = 0.010
    scan_drift_sigma: float = 0.018
    scan_ripple_amplitude: float = 0.018
    noise_scale: float = 1.0
    normalization_mode: str = "per-image"


@dataclass(frozen=True)
class SemSimulationConfig:
    """保存一次仿真运行的固定参数。"""

    input_path: str
    marker_layer: str
    pattern_layers: Tuple[Tuple[int, int], ...]
    output_dir: Path
    clip_size_um: float = 1.35
    pixel_size_nm: int = DEFAULT_PIXEL_SIZE_NM
    risk_score: float = 0.0
    risk_mode: str = "constant"
    seed: int = 0
    write_png: bool = True
    apply_layer_operations: bool = False
    register_ops: Tuple[Tuple[str, str, str, str], ...] = ()
    model_params: SemModelParams = SemModelParams()


@dataclass(frozen=True)
class SemImageResult:
    """保存单个 marker 的仿真图像、sidecar 和质量指标。"""

    image: np.ndarray
    raw_image: np.ndarray
    edge_response: np.ndarray
    layout_bitmap: np.ndarray
    pattern_pixel_ratio: float
    edge_density: float
    structure_complexity: float
    simulation_empty: bool
    simulation_low_structure: bool
    normalization_mode: str
    image_mean: float
    image_std: float
    image_hash: str


def _json_default(value: Any) -> Any:
    """把 numpy/path 对象转换为 JSON 可序列化类型。"""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _parse_layer_spec(layer_spec: str) -> Tuple[int, int]:
    """解析 `layer/datatype` 字符串，返回整数 layer/datatype 元组。"""
    try:
        layer_text, datatype_text = str(layer_spec).split("/", 1)
        return int(layer_text.strip()), int(datatype_text.strip())
    except Exception as exc:
        raise ValueError(f"Invalid layer '{layer_spec}', expected '<layer>/<datatype>'") from exc


def _parse_pattern_layers(pattern_layers: Sequence[str]) -> Tuple[Tuple[int, int], ...]:
    """解析并去重 CLI 传入的 pattern layer 列表。"""
    parsed: List[Tuple[int, int]] = []
    seen = set()
    for layer_text in pattern_layers:
        layer = _parse_layer_spec(str(layer_text))
        if layer in seen:
            continue
        parsed.append(layer)
        seen.add(layer)
    if not parsed:
        raise ValueError("At least one --pattern-layer is required")
    return tuple(parsed)


def _layer_text(layer: Tuple[int, int]) -> str:
    """把 layer/datatype 元组转回人类可读字符串。"""
    return f"{int(layer[0])}/{int(layer[1])}"


def _profile_defaults(profile: str) -> SemModelParams:
    """返回指定模型 profile 的默认轻量 SEM 参数。"""
    profile_text = str(profile).strip().lower()
    if profile_text == "clean":
        return SemModelParams(
            profile="clean",
            edge_gain=0.28,
            shot_noise_sigma=0.006,
            gaussian_noise_sigma=0.004,
            scan_drift_sigma=0.006,
            scan_ripple_amplitude=0.006,
            noise_scale=0.5,
        )
    if profile_text == "stress":
        return SemModelParams(
            profile="stress",
            psf_sigma_nm=20.0,
            edge_gain=0.36,
            shot_noise_sigma=0.016,
            gaussian_noise_sigma=0.014,
            scan_drift_sigma=0.030,
            scan_ripple_amplitude=0.030,
            noise_scale=1.6,
        )
    if profile_text != "nominal":
        raise ValueError(f"Unsupported model profile: {profile}")
    return SemModelParams(profile="nominal")


def _build_model_params(
    *,
    model_profile: str,
    psf_sigma_nm: float | None,
    cd_bias_nm: float | None,
    edge_gain: float | None,
    noise_scale: float | None,
    normalization_mode: str,
) -> SemModelParams:
    """合并 profile 默认值和 CLI 显式覆盖值。"""
    normalization = str(normalization_mode).strip().lower()
    if normalization not in NORMALIZATION_MODES:
        raise ValueError(f"Unsupported normalization mode: {normalization_mode}")
    params = _profile_defaults(model_profile)
    if psf_sigma_nm is not None:
        params = replace(params, psf_sigma_nm=float(psf_sigma_nm))
    if cd_bias_nm is not None:
        params = replace(params, cd_bias_nm=float(cd_bias_nm))
    if edge_gain is not None:
        params = replace(params, edge_gain=float(edge_gain))
    if noise_scale is not None:
        params = replace(params, noise_scale=float(noise_scale))
    return replace(params, normalization_mode=normalization)


def _make_layer_processor(register_ops: Sequence[Sequence[str]] | None) -> LayerOperationProcessor:
    """根据 CLI 传入的 layer operation 规则构建处理器。"""
    processor = LayerOperationProcessor()
    for source_layer, target_layer, operation, result_layer in register_ops or []:
        processor.register_operation_rule(source_layer, operation, target_layer, result_layer)
    return processor


def _operation_payload(register_ops: Sequence[Sequence[str]] | None) -> List[Dict[str, str]]:
    """把 layer operation 规则整理为 summary 可写入的结构。"""
    return [
        {
            "source_layer": str(source_layer),
            "target_layer": str(target_layer),
            "operation": str(operation),
            "result_layer": str(result_layer),
        }
        for source_layer, target_layer, operation, result_layer in register_ops or []
    ]


def _filter_layout_index(layout_index: LayoutIndex, pattern_layers: Sequence[Tuple[int, int]]) -> LayoutIndex:
    """只保留指定 pattern layers 的 indexed elements，marker polygons 原样保留。"""
    layer_set = {(int(layer), int(datatype)) for layer, datatype in pattern_layers}
    kept = [
        element
        for element in layout_index.indexed_elements
        if (int(element.get("layer", -1)), int(element.get("datatype", -1))) in layer_set
    ]
    return LayoutIndex(
        indexed_elements=kept,
        bbox_x0=np.asarray([float(item["bbox"][0]) for item in kept], dtype=np.float64),
        bbox_y0=np.asarray([float(item["bbox"][1]) for item in kept], dtype=np.float64),
        bbox_x1=np.asarray([float(item["bbox"][2]) for item in kept], dtype=np.float64),
        bbox_y1=np.asarray([float(item["bbox"][3]) for item in kept], dtype=np.float64),
        marker_polygons=list(layout_index.marker_polygons),
    )


def _collect_marker_records(config: SemSimulationConfig) -> Tuple[List[MarkerRecord], Dict[str, Any]]:
    """读取输入 OAS，并围绕每个 marker 构建已按 pattern layer 过滤的 bitmap record。"""
    layer_processor = _make_layer_processor(config.register_ops)
    builder = MarkerRasterBuilder(
        config={
            "hotspot_layer": str(config.marker_layer),
            "clip_size_um": float(config.clip_size_um),
            "pixel_size_nm": int(config.pixel_size_nm),
            "apply_layer_operations": bool(config.apply_layer_operations),
        },
        temp_dir=config.output_dir / "_sem_raster_cache",
        layer_processor=layer_processor if config.apply_layer_operations else None,
    )
    records: List[MarkerRecord] = []
    source_summaries: List[Dict[str, Any]] = []
    for filepath in builder._discover_input_files(str(config.input_path)):
        raw_layout = builder._prepare_layout(filepath)
        filtered_layout = _filter_layout_index(raw_layout, config.pattern_layers)
        source_written = 0
        for marker_index, marker_poly in enumerate(raw_layout.marker_polygons):
            record = builder._build_marker_record(filepath, marker_index, marker_poly, filtered_layout)
            if record is None:
                continue
            records.append(record)
            source_written += 1
        source_summaries.append(
            {
                "source_path": str(filepath),
                "marker_count": int(len(raw_layout.marker_polygons)),
                "pattern_element_count_before_filter": int(len(raw_layout.indexed_elements)),
                "pattern_element_count_after_filter": int(len(filtered_layout.indexed_elements)),
                "record_count": int(source_written),
            }
        )
    if not records:
        raise ValueError("No markers found on the configured marker layer")
    return records, {"sources": source_summaries}


def _stable_marker_seed(seed: int, marker_id: str) -> int:
    """根据全局 seed 和 marker id 生成可复现的单 marker 随机种子。"""
    payload = f"{int(seed)}|{marker_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32)


def _percentile_normalize(image: np.ndarray, *, allow_contrast_stretch: bool = True) -> np.ndarray:
    """对图像做 percentile clipping，并缩放到 0..1 的 float32。"""
    array = np.asarray(image, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    if not allow_contrast_stretch:
        return np.ascontiguousarray(np.clip(array, 0.0, 1.0), dtype=np.float32)
    lo, hi = np.percentile(array, [0.5, 99.5])
    if float(hi) <= float(lo) + 1e-12:
        return np.ascontiguousarray(np.clip(array, 0.0, 1.0), dtype=np.float32)
    clipped = np.clip(array, float(lo), float(hi))
    normalized = (clipped - float(lo)) / (float(hi) - float(lo))
    return np.ascontiguousarray(normalized, dtype=np.float32)


def _fixed_normalize(image: np.ndarray) -> np.ndarray:
    """按固定强度范围裁剪到 0..1，保留跨 marker 的低信号差异。"""
    array = np.asarray(image, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(np.clip(array, 0.0, 1.0), dtype=np.float32)


def _normalize_image(image: np.ndarray, *, mode: str, low_structure: bool) -> np.ndarray:
    """按指定模式归一化图像；低结构窗口不做强对比拉伸。"""
    if str(mode) == "fixed":
        return _fixed_normalize(image)
    return _percentile_normalize(image, allow_contrast_stretch=not bool(low_structure))


def _scan_nonuniformity(shape: Tuple[int, int], rng: np.random.Generator, params: SemModelParams) -> np.ndarray:
    """生成低频扫描非均匀性场，模拟 SEM 扫描方向上的缓慢漂移。"""
    height, width = int(shape[0]), int(shape[1])
    y_axis = np.linspace(-1.0, 1.0, max(1, height), dtype=np.float32)
    yy = np.repeat(y_axis[:, None], max(1, width), axis=1)
    row_drift = float(rng.normal(0.0, params.scan_drift_sigma)) * yy
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    row_ripple = float(params.scan_ripple_amplitude) * np.sin(2.0 * np.pi * 1.35 * yy + phase)
    field = 1.0 + row_drift + row_ripple
    return np.ascontiguousarray(np.clip(field, 0.88, 1.12), dtype=np.float32)


def _edge_band_response(inside_distance: np.ndarray, outside_distance: np.ndarray, psf_sigma_px: float) -> np.ndarray:
    """基于内外 signed-distance 生成轻量双侧边缘带响应。"""
    sigma = max(0.75, float(psf_sigma_px))
    inside = np.asarray(inside_distance, dtype=np.float32)
    outside = np.asarray(outside_distance, dtype=np.float32)
    distance_to_edge = np.where(inside > 0.0, inside, outside)
    band = np.exp(-np.maximum(distance_to_edge - 1.0, 0.0) / sigma)
    band = ndimage.gaussian_filter(band.astype(np.float32), sigma=max(0.35, sigma * 0.25), mode="nearest")
    max_value = float(np.max(band)) if band.size else 0.0
    if max_value <= 1e-12:
        return np.zeros_like(band, dtype=np.float32)
    return np.ascontiguousarray(band / max_value, dtype=np.float32)


def _structure_metrics(mask: np.ndarray, edge_response: np.ndarray | None = None) -> Dict[str, float | bool]:
    """计算仿真和风险 proxy 共用的轻量结构指标。"""
    bitmap = np.asarray(mask, dtype=bool)
    if bitmap.size == 0:
        return {
            "pattern_pixel_ratio": 0.0,
            "edge_density": 0.0,
            "structure_complexity": 0.0,
            "simulation_empty": True,
            "simulation_low_structure": True,
        }
    density = float(np.count_nonzero(bitmap)) / float(bitmap.size)
    if edge_response is not None and np.asarray(edge_response).size:
        edge_density = float(np.mean(np.asarray(edge_response, dtype=np.float32) > 0.25))
    else:
        eroded = ndimage.binary_erosion(bitmap, structure=np.ones((3, 3), dtype=bool), border_value=0)
        edge_density = float(np.count_nonzero(bitmap ^ eroded)) / float(bitmap.size)
    row_profile = np.mean(bitmap, axis=1) if bitmap.ndim == 2 else np.asarray([], dtype=np.float32)
    col_profile = np.mean(bitmap, axis=0) if bitmap.ndim == 2 else np.asarray([], dtype=np.float32)
    profile_variation = 0.5 * (float(np.std(row_profile)) + float(np.std(col_profile))) if bitmap.size else 0.0
    complexity = float(np.clip(0.65 * edge_density + 0.35 * profile_variation, 0.0, 1.0))
    empty = density <= 0.0
    low_structure = bool(empty or density < LOW_STRUCTURE_DENSITY_THRESHOLD or edge_density < LOW_STRUCTURE_EDGE_THRESHOLD)
    return {
        "pattern_pixel_ratio": float(density),
        "edge_density": float(edge_density),
        "structure_complexity": float(complexity),
        "simulation_empty": bool(empty),
        "simulation_low_structure": bool(low_structure),
    }


def _image_hash(image: np.ndarray) -> str:
    """生成归一化图像的短哈希，用于质量审查中检测重复图像。"""
    array = np.asarray(np.clip(image, 0.0, 1.0) * 255.0 + 0.5, dtype=np.uint8)
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


def simulate_sem_image_detail(
    layout_bitmap: np.ndarray,
    *,
    marker_id: str,
    seed: int,
    pixel_size_nm: int,
    model_params: SemModelParams,
) -> SemImageResult:
    """把单个 layout bitmap 转换成带 sidecar 和质量指标的仿真结果。"""
    mask = np.asarray(layout_bitmap, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"layout_bitmap must be 2-D, got shape {mask.shape}")
    rng = np.random.default_rng(_stable_marker_seed(seed, marker_id))
    if mask.size == 0 or int(np.count_nonzero(mask)) == 0:
        empty_raw = np.full(mask.shape, float(model_params.background_level), dtype=np.float32)
        empty_raw = empty_raw + rng.normal(
            0.0,
            float(model_params.gaussian_noise_sigma) * max(0.0, float(model_params.noise_scale)) * 0.25,
            size=mask.shape,
        )
        empty_image = _fixed_normalize(empty_raw)
        empty_edge = np.zeros(mask.shape, dtype=np.float32)
        return SemImageResult(
            image=np.ascontiguousarray(empty_image, dtype=np.float32),
            raw_image=np.ascontiguousarray(empty_raw, dtype=np.float32),
            edge_response=empty_edge,
            layout_bitmap=np.ascontiguousarray(mask, dtype=bool),
            pattern_pixel_ratio=0.0,
            edge_density=0.0,
            structure_complexity=0.0,
            simulation_empty=True,
            simulation_low_structure=True,
            normalization_mode=str(model_params.normalization_mode),
            image_mean=float(np.mean(empty_image)) if empty_image.size else 0.0,
            image_std=float(np.std(empty_image)) if empty_image.size else 0.0,
            image_hash=_image_hash(empty_image),
        )

    pixel_nm = max(float(pixel_size_nm), 1e-6)
    psf_sigma_px = max(0.65, float(model_params.psf_sigma_nm) / pixel_nm)
    cd_bias_px = float(model_params.cd_bias_nm) / pixel_nm

    inside_distance = ndimage.distance_transform_edt(mask)
    outside_distance = ndimage.distance_transform_edt(~mask)
    signed_distance = inside_distance - outside_distance + cd_bias_px
    transition = 1.0 / (1.0 + np.exp(-signed_distance / max(0.75, psf_sigma_px * 0.55)))
    blurred = ndimage.gaussian_filter(transition.astype(np.float32), sigma=psf_sigma_px * 0.65, mode="nearest")
    edge = _edge_band_response(inside_distance, outside_distance, psf_sigma_px)
    metrics = _structure_metrics(mask, edge)

    image = (
        float(model_params.background_level)
        + float(model_params.material_gain) * blurred
        + float(model_params.edge_gain) * edge
    )
    image = image * _scan_nonuniformity(mask.shape, rng, model_params)
    noise_scale = max(0.0, float(model_params.noise_scale))
    shot_sigma = float(model_params.shot_noise_sigma) * noise_scale * np.sqrt(np.clip(image, 0.0, None))
    image = image + rng.normal(0.0, shot_sigma, size=mask.shape)
    image = image + rng.normal(0.0, float(model_params.gaussian_noise_sigma) * noise_scale, size=mask.shape)

    low_structure = bool(metrics["simulation_low_structure"])
    normalized = _normalize_image(image, mode=model_params.normalization_mode, low_structure=low_structure)
    return SemImageResult(
        image=np.ascontiguousarray(normalized, dtype=np.float32),
        raw_image=np.ascontiguousarray(image, dtype=np.float32),
        edge_response=np.ascontiguousarray(edge, dtype=np.float32),
        layout_bitmap=np.ascontiguousarray(mask, dtype=bool),
        pattern_pixel_ratio=float(metrics["pattern_pixel_ratio"]),
        edge_density=float(metrics["edge_density"]),
        structure_complexity=float(metrics["structure_complexity"]),
        simulation_empty=bool(metrics["simulation_empty"]),
        simulation_low_structure=low_structure,
        normalization_mode=str(model_params.normalization_mode),
        image_mean=float(np.mean(normalized)) if normalized.size else 0.0,
        image_std=float(np.std(normalized)) if normalized.size else 0.0,
        image_hash=_image_hash(normalized),
    )


def simulate_sem_image(layout_bitmap: np.ndarray, *, marker_id: str, seed: int, pixel_size_nm: int) -> np.ndarray:
    """把单个 layout bitmap 转换成仿真 CD-SEM-like 图像，保留旧版调用语义。"""
    result = simulate_sem_image_detail(
        layout_bitmap,
        marker_id=marker_id,
        seed=seed,
        pixel_size_nm=pixel_size_nm,
        model_params=SemModelParams(),
    )
    return result.image


def _write_png(path: Path, image: np.ndarray) -> None:
    """把 0..1 float 图像写成 8-bit 灰度 PNG。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(np.clip(image, 0.0, 1.0) * 255.0 + 0.5, dtype=np.uint8)
    Image.fromarray(array, mode="L").save(path)


def _read_risk_score_csv(path: str | Path | None) -> Dict[str, float]:
    """读取 marker id 到 risk_score 的可选覆盖表。"""
    if path is None:
        return {}
    risk_path = Path(path)
    scores: Dict[str, float] = {}
    with risk_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"marker_id", "risk_score"}
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"risk score CSV missing columns: {sorted(missing)}")
        for row_index, row in enumerate(reader, start=2):
            marker_id = str(row.get("marker_id", "")).strip()
            if not marker_id:
                raise ValueError(f"risk score CSV row {row_index} has empty marker_id")
            if marker_id in scores:
                raise ValueError(f"Duplicate marker_id in risk score CSV: {marker_id}")
            scores[marker_id] = float(row.get("risk_score", 0.0) or 0.0)
    return scores


def _shape_summary(shapes: Sequence[Tuple[int, int]]) -> Dict[str, Any]:
    """统计输出图像 shape 分布。"""
    counts: Dict[str, int] = {}
    for height, width in shapes:
        key = f"{int(height)}x{int(width)}"
        counts[key] = counts.get(key, 0) + 1
    return {"unique_shape_count": int(len(counts)), "shape_counts": dict(sorted(counts.items()))}


def _model_params_payload(params: SemModelParams) -> Dict[str, Any]:
    """把模型参数整理为 summary/audit 使用的 JSON 结构。"""
    return {
        "profile": str(params.profile),
        "psf_sigma_nm": float(params.psf_sigma_nm),
        "cd_bias_nm": float(params.cd_bias_nm),
        "background_level": float(params.background_level),
        "material_gain": float(params.material_gain),
        "edge_gain": float(params.edge_gain),
        "shot_noise_sigma": float(params.shot_noise_sigma),
        "gaussian_noise_sigma": float(params.gaussian_noise_sigma),
        "scan_drift_sigma": float(params.scan_drift_sigma),
        "scan_ripple_amplitude": float(params.scan_ripple_amplitude),
        "noise_scale": float(params.noise_scale),
        "normalization_mode": str(params.normalization_mode),
    }


def _clip01(value: float) -> float:
    """把浮点值裁剪到 0..1。"""
    return float(np.clip(float(value), 0.0, 1.0))


def _layout_proxy_risk(result: SemImageResult) -> Tuple[float, Dict[str, float]]:
    """根据 layout/仿真结构指标生成轻量 pseudo risk。"""
    density_score = _clip01(float(result.pattern_pixel_ratio) / 0.35)
    edge_score = _clip01(float(result.edge_density) / 0.25)
    complexity_score = _clip01(float(result.structure_complexity) / 0.20)
    contrast_score = _clip01(float(result.image_std) / 0.25)
    empty_penalty = 1.0 if result.simulation_empty else 0.0
    risk = (
        0.25 * density_score
        + 0.35 * edge_score
        + 0.25 * complexity_score
        + 0.15 * contrast_score
    ) * (1.0 - empty_penalty)
    components = {
        "density_score": float(density_score),
        "edge_density_score": float(edge_score),
        "structure_complexity_score": float(complexity_score),
        "image_contrast_score": float(contrast_score),
        "empty_penalty": float(empty_penalty),
    }
    return _clip01(risk), components


def _risk_for_result(
    *,
    result: SemImageResult,
    marker_id: str,
    config: SemSimulationConfig,
    risk_overrides: Mapping[str, float],
) -> Tuple[float, str, Dict[str, float], bool]:
    """按 CSV > layout-proxy > constant 的优先级确定 manifest risk_score。"""
    if str(marker_id) in risk_overrides:
        return float(risk_overrides[str(marker_id)]), "csv", {}, True
    if config.risk_mode == "layout-proxy":
        risk_value, components = _layout_proxy_risk(result)
        return float(risk_value), "layout_sem_proxy", components, False
    return float(config.risk_score), "constant", {}, False


def _numeric_distribution(values: Sequence[float]) -> Dict[str, Any]:
    """生成紧凑数值分布摘要。"""
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _build_quality_audit(
    *,
    rows: Sequence[Dict[str, Any]],
    image_hashes: Sequence[str],
    source_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    """构建仿真质量审查摘要，帮助发现无信息量输出包。"""
    marker_count = int(len(rows))
    empty_count = sum(1 for row in rows if bool(row.get("simulation_empty")))
    low_structure_count = sum(1 for row in rows if bool(row.get("simulation_low_structure")))
    risk_values = [float(row.get("risk_score", 0.0) or 0.0) for row in rows]
    std_values = [float(row.get("image_std", 0.0) or 0.0) for row in rows]
    density_values = [float(row.get("pattern_pixel_ratio", 0.0) or 0.0) for row in rows]
    edge_values = [float(row.get("edge_density", 0.0) or 0.0) for row in rows]
    hash_counts: Dict[str, int] = {}
    for image_hash in image_hashes:
        hash_counts[str(image_hash)] = hash_counts.get(str(image_hash), 0) + 1
    duplicate_image_hash_count = sum(1 for count in hash_counts.values() if count > 1)

    warnings: List[str] = []
    if marker_count and empty_count / marker_count > 0.20:
        warnings.append("empty_marker_ratio_above_20_percent")
    if marker_count and low_structure_count / marker_count > 0.40:
        warnings.append("low_structure_ratio_above_40_percent")
    if risk_values and max(risk_values) <= 1e-12:
        warnings.append("all_risk_scores_are_zero")
    if std_values and float(np.median(std_values)) < 0.005:
        warnings.append("image_std_median_too_low")
    if std_values and float(np.median(std_values)) > 0.45:
        warnings.append("image_std_median_too_high")
    for source in source_summary.get("sources", []):
        if int(source.get("pattern_element_count_after_filter", 0)) <= 0:
            warnings.append(f"no_pattern_elements_after_filter:{source.get('source_path', '')}")

    return {
        "marker_count": marker_count,
        "empty_count": int(empty_count),
        "empty_ratio": float(empty_count / marker_count) if marker_count else 0.0,
        "low_structure_count": int(low_structure_count),
        "low_structure_ratio": float(low_structure_count / marker_count) if marker_count else 0.0,
        "duplicate_image_hash_count": int(duplicate_image_hash_count),
        "image_mean_distribution": _numeric_distribution([float(row.get("image_mean", 0.0) or 0.0) for row in rows]),
        "image_std_distribution": _numeric_distribution(std_values),
        "pattern_density_distribution": _numeric_distribution(density_values),
        "edge_density_distribution": _numeric_distribution(edge_values),
        "risk_score_distribution": _numeric_distribution(risk_values),
        "warnings": warnings,
    }


def run_sem_simulation(
    *,
    input_path: str,
    marker_layer: str,
    pattern_layers: Sequence[str],
    output_dir: str,
    clip_size_um: float = 1.35,
    pixel_size_nm: int = DEFAULT_PIXEL_SIZE_NM,
    risk_score: float = 0.0,
    risk_score_csv: str | None = None,
    risk_mode: str = "constant",
    model_profile: str = "nominal",
    psf_sigma_nm: float | None = None,
    cd_bias_nm: float | None = None,
    edge_gain: float | None = None,
    noise_scale: float | None = None,
    normalization_mode: str = "per-image",
    apply_layer_ops: bool = False,
    register_op: Sequence[Sequence[str]] | None = None,
    seed: int = 0,
    write_png: bool = True,
) -> Dict[str, Any]:
    """执行完整 SEM 仿真流程，并返回 summary 字典。"""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    aerial_npz_dir = output_root / "aerial_npz"
    aerial_png_dir = output_root / "aerial_png"
    aerial_npz_dir.mkdir(parents=True, exist_ok=True)
    if write_png:
        aerial_png_dir.mkdir(parents=True, exist_ok=True)

    risk_mode_text = str(risk_mode).strip().lower()
    if risk_mode_text not in RISK_MODES:
        raise ValueError(f"Unsupported risk mode: {risk_mode}")
    model_params = _build_model_params(
        model_profile=str(model_profile),
        psf_sigma_nm=psf_sigma_nm,
        cd_bias_nm=cd_bias_nm,
        edge_gain=edge_gain,
        noise_scale=noise_scale,
        normalization_mode=str(normalization_mode),
    )
    register_ops = tuple(tuple(str(value) for value in op) for op in (register_op or ()))
    config = SemSimulationConfig(
        input_path=str(input_path),
        marker_layer=str(marker_layer),
        pattern_layers=_parse_pattern_layers(pattern_layers),
        output_dir=output_root,
        clip_size_um=float(clip_size_um),
        pixel_size_nm=int(pixel_size_nm),
        risk_score=float(risk_score),
        risk_mode=risk_mode_text,
        seed=int(seed),
        write_png=bool(write_png),
        apply_layer_operations=bool(apply_layer_ops or register_ops),
        register_ops=register_ops,
        model_params=model_params,
    )
    risk_overrides = _read_risk_score_csv(risk_score_csv)
    records, source_summary = _collect_marker_records(config)

    manifest_path = output_root / "behavior.jsonl"
    rows: List[Dict[str, Any]] = []
    shapes: List[Tuple[int, int]] = []
    image_hashes: List[str] = []
    risk_override_hits = 0
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            result = simulate_sem_image_detail(
                record.clip_bitmap,
                marker_id=str(record.marker_id),
                seed=int(config.seed),
                pixel_size_nm=int(config.pixel_size_nm),
                model_params=config.model_params,
            )
            image = result.image
            shapes.append((int(image.shape[0]), int(image.shape[1])))
            image_hashes.append(str(result.image_hash))

            npz_path = aerial_npz_dir / f"{record.marker_id}.npz"
            np.savez_compressed(
                npz_path,
                image=np.ascontiguousarray(image, dtype=np.float32),
                layout_bitmap=np.asarray(result.layout_bitmap, dtype=np.uint8),
                edge_response=np.ascontiguousarray(result.edge_response, dtype=np.float32),
                raw_image=np.ascontiguousarray(result.raw_image, dtype=np.float32),
                normalization_mode=np.asarray(result.normalization_mode),
            )

            png_rel = ""
            if config.write_png:
                png_path = aerial_png_dir / f"{record.marker_id}.png"
                _write_png(png_path, image)
                png_rel = f"aerial_png/{png_path.name}"

            row_risk, risk_source, risk_components, risk_override_hit = _risk_for_result(
                result=result,
                marker_id=str(record.marker_id),
                config=config,
                risk_overrides=risk_overrides,
            )
            if risk_override_hit:
                risk_override_hits += 1

            row = {
                "sample_id": str(record.marker_id),
                "source_path": str(record.source_path),
                "marker_id": str(record.marker_id),
                "marker_center": [float(record.marker_center[0]), float(record.marker_center[1])],
                "clip_bbox": [float(value) for value in record.clip_bbox],
                "aerial_npz": f"aerial_npz/{npz_path.name}",
                "aerial_png": png_rel,
                "risk_score": float(row_risk),
                "risk_score_source": str(risk_source),
                "risk_components": risk_components,
                "pixel_size_nm": int(config.pixel_size_nm),
                "clip_size_um": float(config.clip_size_um),
                "simulation_model": SIMULATION_MODEL,
                "model_profile": str(config.model_params.profile),
                "normalization_mode": str(result.normalization_mode),
                "pattern_pixel_ratio": float(result.pattern_pixel_ratio),
                "edge_density": float(result.edge_density),
                "structure_complexity": float(result.structure_complexity),
                "simulation_empty": bool(result.simulation_empty),
                "simulation_low_structure": bool(result.simulation_low_structure),
                "image_mean": float(result.image_mean),
                "image_std": float(result.image_std),
                "image_hash": str(result.image_hash),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)

    quality_audit = _build_quality_audit(rows=rows, image_hashes=image_hashes, source_summary=source_summary)
    quality_audit_path = output_root / "simulation_quality_audit.json"
    with quality_audit_path.open("w", encoding="utf-8") as handle:
        json.dump(quality_audit, handle, indent=2, ensure_ascii=False, default=_json_default)

    summary = {
        "pipeline_mode": SIMULATION_MODEL,
        "input_path": str(input_path),
        "marker_layer": str(marker_layer),
        "pattern_layers": [_layer_text(layer) for layer in config.pattern_layers],
        "clip_size_um": float(config.clip_size_um),
        "pixel_size_nm": int(config.pixel_size_nm),
        "seed": int(config.seed),
        "write_png": bool(config.write_png),
        "risk_mode": str(config.risk_mode),
        "model_params": _model_params_payload(config.model_params),
        "apply_layer_operations": bool(config.apply_layer_operations),
        "layer_operations": _operation_payload(config.register_ops),
        "marker_count": int(len(records)),
        "written_npz_count": int(len(rows)),
        "written_png_count": int(len(rows) if config.write_png else 0),
        "behavior_manifest": str(manifest_path),
        "aerial_npz_dir": str(aerial_npz_dir),
        "aerial_png_dir": str(aerial_png_dir) if config.write_png else "",
        "simulation_quality_audit": str(quality_audit_path),
        "default_risk_score": float(config.risk_score),
        "risk_score_csv": str(risk_score_csv or ""),
        "risk_score_override_count": int(len(risk_overrides)),
        "risk_score_override_hit_count": int(risk_override_hits),
        "image_shape_summary": _shape_summary(shapes),
        "quality_audit": quality_audit,
        "source_summary": source_summary,
    }
    summary_path = output_root / "simulation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=_json_default)
    summary["simulation_summary"] = str(summary_path)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    epilog = """
示例:
python sem_simulator.py hotspotdetection/clip_for_lyu.oas --marker-layer 12530/2 --pattern-layer 2530/0 --output-dir sem_behavior

输出:
- behavior.jsonl: 直接传给 recipe_site_selector.py --behavior-manifest
- aerial_npz/: 每个 marker 一张 key=image 的二维 float32 NPZ
- aerial_png/: 默认写出的灰度 PNG，供人工检查仿真图像
- simulation_summary.json: 本次仿真摘要
- simulation_quality_audit.json: 空窗口、低结构、风险和图像质量分布审查
"""
    parser = argparse.ArgumentParser(
        description="基于 OAS marker layer 生成仿真 CD-SEM/aerial behavior 输入包",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_path", help="输入 OAS 文件或 OAS 目录")
    parser.add_argument("--marker-layer", required=True, help="marker layer，格式为 layer/datatype，例如 12530/2")
    parser.add_argument(
        "--pattern-layer",
        required=True,
        action="append",
        help="参与成像的 pattern layer，格式为 layer/datatype；多层请重复传入",
    )
    parser.add_argument("--output-dir", required=True, help="输出 behavior 输入包目录")
    parser.add_argument("--clip-size-um", type=float, default=1.35, help="marker-centered clip 尺寸，单位 um")
    parser.add_argument("--pixel-size-nm", type=int, default=DEFAULT_PIXEL_SIZE_NM, help="仿真图像像素尺寸，单位 nm")
    parser.add_argument("--risk-score", type=float, default=0.0, help="默认写入 manifest 的 risk_score")
    parser.add_argument("--risk-score-csv", default=None, help="可选 risk_score 覆盖表，列名为 marker_id,risk_score")
    parser.add_argument(
        "--risk-mode",
        choices=RISK_MODES,
        default="constant",
        help="risk_score 生成模式；constant 使用 --risk-score，layout-proxy 使用版图结构 proxy",
    )
    parser.add_argument(
        "--model-profile",
        choices=MODEL_PROFILES,
        default="nominal",
        help="轻量 SEM 模型 profile；clean 噪声低，stress 用于鲁棒性压力测试",
    )
    parser.add_argument("--psf-sigma-nm", type=float, default=None, help="覆盖 profile 中的 SEM PSF sigma，单位 nm")
    parser.add_argument("--cd-bias-nm", type=float, default=None, help="覆盖 profile 中的 CD bias，单位 nm，可为负值")
    parser.add_argument("--edge-gain", type=float, default=None, help="覆盖 profile 中的边缘增强权重")
    parser.add_argument("--noise-scale", type=float, default=None, help="覆盖 profile 中的整体噪声倍率")
    parser.add_argument(
        "--normalization-mode",
        choices=NORMALIZATION_MODES,
        default="per-image",
        help="图像归一化模式；fixed 保留跨 marker 强度差异",
    )
    parser.add_argument("--apply-layer-ops", action="store_true", help="启用 register-op 指定的 layer boolean 预处理")
    parser.add_argument(
        "--register-op",
        action="append",
        nargs=4,
        metavar=("SOURCE", "TARGET", "OPERATION", "RESULT"),
        help="注册 layer 操作: SOURCE TARGET OPERATION RESULT，operation 支持 subtract/union/intersect",
    )
    parser.add_argument("--seed", type=int, default=0, help="全局随机种子；每个 marker 会派生稳定子种子")
    parser.add_argument("--no-png", action="store_true", help="只写 NPZ 和 manifest，不写 PNG")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口，执行仿真并打印核心输出路径。"""
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary = run_sem_simulation(
        input_path=str(args.input_path),
        marker_layer=str(args.marker_layer),
        pattern_layers=list(args.pattern_layer),
        output_dir=str(args.output_dir),
        clip_size_um=float(args.clip_size_um),
        pixel_size_nm=int(args.pixel_size_nm),
        risk_score=float(args.risk_score),
        risk_score_csv=args.risk_score_csv,
        risk_mode=str(args.risk_mode),
        model_profile=str(args.model_profile),
        psf_sigma_nm=args.psf_sigma_nm,
        cd_bias_nm=args.cd_bias_nm,
        edge_gain=args.edge_gain,
        noise_scale=args.noise_scale,
        normalization_mode=str(args.normalization_mode),
        apply_layer_ops=bool(args.apply_layer_ops),
        register_op=args.register_op,
        seed=int(args.seed),
        write_png=not bool(args.no_png),
    )
    print(f"marker 数: {summary['marker_count']}")
    print(f"写出 NPZ 数: {summary['written_npz_count']}")
    print(f"behavior manifest: {summary['behavior_manifest']}")
    print(f"simulation summary: {summary['simulation_summary']}")
    print(f"quality audit: {summary['simulation_quality_audit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
