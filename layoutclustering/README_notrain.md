# No-Training Optimized Layout Clustering

本版本现在是独立主脚本：`layout_clustering_optimized_notrain.py` 不再 import 或调用
`layout_clustering_optimized.py`。它仅共享 `layout_utils.py`、`layer_operations.py` 和
`feature_extractor_handcraft.py` 这些底层工具，内部维护自己的 ANN、facility location、
k-center、behavior verification、JSON/TXT/review export 流程。这样 AE optimized 主线和
no-training 备用线可以独立演进、独立调参、独立回归。

`layout_clustering_optimized_notrain.py` 是 `layout_clustering_optimized.py` 的无训练备用入口。它用于 AutoEncoder 暂时无法完成训练、训练结果不稳定，或需要先快速建立 deterministic baseline 的场景。

这条路线不改变 optimized 主流程的最终质量保证方式：handcrafted feature vector 只负责 ANN 检索、coverage ordering 和 representative selection；最终 cluster member 是否成立仍由 aerial/EPE/PV/NILS behavior verification 判断。

## 整体流程

```text
OAS + marker layer
-> 可选 layer boolean operations
-> marker-centered clip bitmap
-> handcrafted FV 自动生成
   aerial DCT/FFT/HOG/gradient
   optional PV/EPE/NILS/resist stats
   layout geometry / critical feature proxy
   polygon bbox graph + WL signature
-> exact hash 去重和 duplicate weight 累计
-> ANN top-K graph
-> weighted facility location 选 representative
-> weighted k-center 补 farthest / high-risk holes
-> behavior final verification
-> JSON/TXT + review export
```

## 主入口用法

基本运行：

```bash
python layout_clustering_optimized_notrain.py input.oas \
  --marker-layer 999/0 \
  --behavior-manifest behavior.jsonl \
  --output results_notrain.json
```

导出 review 目录：

```bash
python layout_clustering_optimized_notrain.py input.oas \
  --marker-layer 999/0 \
  --behavior-manifest behavior.jsonl \
  --review-dir review_notrain \
  --output results_notrain.json
```

导出 behavior diff：

```bash
python layout_clustering_optimized_notrain.py input.oas \
  --marker-layer 999/0 \
  --behavior-manifest behavior.jsonl \
  --review-dir review_notrain \
  --export-diff-channels aerial,resist,pv \
  --output results_notrain.json
```

启用 layer boolean 操作：

```bash
python layout_clustering_optimized_notrain.py input.oas \
  --marker-layer 999/0 \
  --behavior-manifest behavior.jsonl \
  --register-op 1/0 2/0 subtract 10/0 \
  --output results_notrain.json
```

`--register-op` 会自动启用 layer operations；显式加 `--apply-layer-ops` 只是让命令更清楚。

## 手工特征脚本

`feature_extractor_handcraft.py` 可以单独使用，输出与 AutoEncoder encode 相同 schema 的 `features.npz`：

```bash
python feature_extractor_handcraft.py encode input.oas \
  --marker-layer 999/0 \
  --behavior-manifest behavior.jsonl \
  --features-out features_handcraft.npz \
  --metadata-out features_handcraft.metadata.json
```

输出 NPZ 包含：

- `sample_ids`: 与 manifest 中 sample 对齐的字符串数组。
- `features`: 已经 block-wise 加权并整体 L2 normalize 的二维 float32 特征矩阵。

metadata JSON 记录：

- `feature_source`
- `feature_count`
- `feature_dim`
- `optional_behavior_channels`
- `block_metadata`
- `marker_layer`
- `clip_size_um`
- `pixel_size_nm`

## 特征块

默认特征融合权重：

- 有 optional behavior stats：`aerial=0.55, behavior=0.20, layout_geometry=0.10, layout_wl_graph=0.15`
- 无 optional behavior stats：`aerial=0.75, layout_geometry=0.10, layout_wl_graph=0.15`

`aerial_image`:

- aerial image resize 到 `64x64`
- DCT 低频 `16x16`
- FFT radial spectrum
- FFT angular spectrum
- HOG
- gradient magnitude/orientation stats

`optional_behavior_stats`:

- 支持 `pv/epe/nils/resist`
- 每个 channel 提取 mean/std/min/max、分位数、abs mean/max、positive ratio、high-response area ratio、gradient stats、connected high-response component stats

`layout_geometry`:

- fill ratio
- 8x8 density grid
- edge density
- center/ring ratio
- connected components
- largest component ratio
- symmetry / regularity
- Radon projection stats
- critical width/space distance-transform quantiles
- line-end / corner / jog proxy
- edge orientation histogram
- dense/sparse/edge-heavy/many-components topology flags

`layout_wl_graph`:

- clip 内 polygon bbox proxy 建图
- node label 包含 layer/datatype、area bin、width/height bin、aspect bin、center grid bin、boundary-touch flag
- edge label 包含 same/cross layer、direction bin、distance bin、overlap/touch/project/diag relation
- WL iterations 固定为 2
- hashed vector dimension 固定为 128

## Manifest 要求

manifest 每行至少包含：

```json
{
  "sample_id": "unit__marker_000000",
  "source_path": "unit.oas",
  "marker_id": "unit__marker_000000",
  "clip_bbox": [0.0, 0.0, 1.0, 1.0],
  "aerial_npz": "unit__marker_000000_aerial.npz"
}
```

可选字段：

- `pv_npz`
- `epe_npz`
- `nils_npz`
- `resist_npz`
- `risk_score`

约束：

- `aerial_npz` 必填。
- `pv/epe/nils/resist` 若任意一个样本提供，则所有样本都必须提供该 channel。
- 每个 NPZ 必须包含二维 float32 兼容数组，键名固定为 `image`。
- manifest 中 `marker_id` 必须能与 OAS marker 顺序生成的 marker id 对齐，例如 `unit__marker_000000`。

## 输出字段

no-training 主脚本会在结果中记录：

- `pipeline_mode: optimized_notrain`
- `feature_source: handcraft`
- `feature_metadata`
- `handcraft_feature_npz`
- `handcraft_feature_metadata_json`

其余 clustering 字段沿用 optimized behavior 主线，包括：

- `marker_count`
- `exact_cluster_count`
- `selected_representative_count`
- `total_clusters`
- `cluster_sizes`
- `behavior_stats`
- `behavior_verification_stats`
- `clusters`
- `file_list`
- `file_metadata`

## 注意点

- no-training 版本仍默认输入带真实 marker layer，不恢复 auto-marker。
- handcrafted FV 不替代 behavior verification；它只影响候选检索和 coverage selection。
- 如果两个样本 FV 很近但 aerial/EPE/PV/NILS verification 不通过，仍会被拆成 singleton/base representative。
- WL graph signature 基于 bbox proxy，作用是补充 topology disambiguation，不做精确几何判定。
- 如已有可靠 AE FV，优先使用 `layout_clustering_optimized.py`；no-training 版本主要用于备用和 baseline 对比。

## 验证命令

```bash
python -m py_compile layout_clustering_optimized.py layout_clustering_optimized_notrain.py feature_extractor_handcraft.py test_handcraft_features.py
python -m unittest test_optimized_clustering -v
python -m unittest test_handcraft_features -v
python layout_clustering_optimized_notrain.py --help
python feature_extractor_handcraft.py encode --help
```
