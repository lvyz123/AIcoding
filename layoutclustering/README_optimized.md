# Optimized Behavior Clustering 使用说明

本目录现在维护两条 marker-driven 光刻行为聚类主线：

- `layout_clustering_optimized.py`：AE/FV 主线。用户先用 AutoEncoder 提取 `features.npz`，主脚本负责 ANN coverage clustering 和 behavior verification。
- `layout_clustering_optimized_notrain.py`：无训练备用主线。主脚本自动调用 `feature_extractor_handcraft.py` 生成 handcrafted FV，不需要外部 `features.npz`。

两条主线的共同目标都是：优先最大化光刻行为 coverage，其次减少 representative / cluster 数，并用 aerial image SSIM 等 behavior verification 保证 cluster member 与 representative 明确一致。

## 两个版本怎么选

| 项目 | AE optimized 版本 | no-train 版本 |
| --- | --- | --- |
| 主脚本 | `layout_clustering_optimized.py` | `layout_clustering_optimized_notrain.py` |
| 特征来源 | AutoEncoder latent FV | 手工特征 FV |
| 是否需要训练 | 需要先训练/加载 AE 模型 | 不需要训练 |
| 是否需要 `--feature-npz` | 需要，必须由 AE encode 生成 | 不需要，主脚本自动生成临时 handcrafted FV |
| 必需 behavior 输入 | `behavior.jsonl` + `aerial_npz` | `behavior.jsonl` + `aerial_npz`，也可直接传 preprocess 输出目录 |
| 推荐场景 | 已有足够 aerial/behavior 数据训练 AE，追求更贴近学习到的光刻行为 embedding | AE 暂不可用、需要 deterministic baseline、快速验证或备用流程 |
| 输出 `pipeline_mode` | `optimized_behavior` | `optimized_notrain` |
| 输出特征 metadata | 只记录外部 `feature_npz` 路径 | 额外记录 `feature_source: handcraft`、`feature_metadata`、`handcraft_feature_npz` |

简单说：**AE 版 = 外部训练并提供 FV；no-train 版 = 不训练，自动提取手工 FV。** 两者后半段的 ANN / facility location / k-center / behavior verification 目标一致，但输入准备方式和特征来源不同。

## 版本 A：AE Optimized 流程

```text
OAS + marker layer
  -> behavior.jsonl + aerial_npz
  -> AutoEncoder encode 生成 features.npz
  -> layout_clustering_optimized.py
  -> exact hash duplicate grouping
  -> ANN top-K graph
  -> weighted facility location 选 reps
  -> weighted k-center 补 farthest / high-risk holes
  -> behavior final verification
  -> JSON/TXT/review export
```

这个版本的决策边界很清楚：主脚本不负责训练，也不负责生成 FV；它只消费 `behavior.jsonl` 和外部 `features.npz`。

## 版本 B：No-Train 流程

```text
OAS + marker layer + aerial image directory
  -> preprocess_notrain.py 生成 behavior.jsonl + aerial_npz
  -> layout_clustering_optimized_notrain.py
  -> 自动生成 handcrafted features.npz
  -> 同样的 ANN / facility / k-center / behavior verification / export
```

这个版本不需要 AE 模型，也不需要外部 `features.npz`。`behavior.jsonl` 仍然必须提供 aerial 数据；如果通过 `preprocess_notrain.py` 生成输入，缺失 aerial 的 marker 会被跳过。

`exact hash` 只用于完全重复 marker window 的去重和 duplicate weight 累计，不作为最终聚类语义。最终 cluster 是否成立由 behavior verification 决定。

## 输入契约

### Layout 与 Marker

输入 OAS/OASIS 必须包含 marker layer。marker id 由底层工具按 OAS 文件名和 marker 顺序生成：

```text
<oas_stem>__marker_000000
<oas_stem>__marker_000001
...
```

例如 `sample_layout_002.oas` 的第一个 marker 是：

```text
sample_layout_002__marker_000000
```

### Behavior Manifest

`behavior.jsonl` 每行对应一个 marker sample，最少包含：

```json
{
  "sample_id": "sample_layout_002__marker_000000",
  "source_path": "sample_layout_002.oas",
  "marker_id": "sample_layout_002__marker_000000",
  "clip_bbox": [0.0, 0.0, 1.35, 1.35],
  "aerial_npz": "aerial_npz/sample_layout_002__marker_000000.npz",
  "risk_score": 0.0
}
```

约束：

- `aerial_npz` 必填。
- 每个 NPZ 必须包含二维 float32 数组，key 固定为 `image`。
- `sample_id` 或 `marker_id` 必须能在 FV 的 `sample_ids` 中找到对应行。
- `epe_npz` / `pv_npz` / `nils_npz` 是数据集级可选项：只要某个 channel 出现，就要求所有样本都提供。
- `resist_npz` 可用于手工特征统计和可选 diff 输出，但当前不参与 final verification weighted score。

## 准备 No-Train 输入

如果你已经有每个 marker 对应的 aerial image，可以先用 `preprocess_notrain.py` 转成 no-train 可直接使用的目录。

```bash
python layoutclustering/preprocess_notrain.py input.oas \
  --marker-layer 999/0 \
  --aerial-dir aerial_images \
  --output-dir notrain_inputs
```

输出目录：

```text
notrain_inputs/
  behavior.jsonl
  preprocess_summary.json
  aerial_npz/
    <marker_id>.npz
```

图片文件名匹配规则：

1. 优先完整 marker id，例如 `sample_layout_002__marker_000123.png`
2. 其次 `marker_000123`
3. 最后裸编号 `000123`

缺失 aerial 的 marker 会被跳过；重复匹配同一 marker 时按路径字符串升序取第一张，并在 `preprocess_summary.json` 中记录。

支持格式：

```text
png, jpg, jpeg, tif, tiff, bmp, npy, npz, dm3, dm4
```

DM3/DM4 读取依赖 `ncempy`。`preprocess_notrain.py` 默认把图像 min-max normalize 到 `[0, 1]`，可用 `--no-normalize` 关闭。

## 运行 No-Train 主线

`layout_clustering_optimized_notrain.py` 不需要 `features.npz`，会自动生成 handcrafted FV。

可以直接把 preprocess 输出目录传给 `--behavior-manifest`：

```bash
python layoutclustering/layout_clustering_optimized_notrain.py input.oas \
  --marker-layer 999/0 \
  --behavior-manifest notrain_inputs \
  --output results_notrain.json
```

也可以传具体 JSONL 文件：

```bash
python layoutclustering/layout_clustering_optimized_notrain.py input.oas \
  --marker-layer 999/0 \
  --behavior-manifest notrain_inputs/behavior.jsonl \
  --output results_notrain.json
```

no-train 输出会额外记录：

```text
pipeline_mode: optimized_notrain
feature_source: handcraft
feature_metadata
handcraft_feature_npz
handcraft_feature_metadata_json
input_marker_count
skipped_missing_behavior_count
```

如果 preprocess 跳过了缺图 marker，no-train 也会只聚类 manifest 中存在 aerial 的 marker 子集。

## 运行 AE Optimized 主线

AE 主线需要先训练/编码 FV。当前脚本名为 `feature_extractor_autoencoder.py`。

训练：

```bash
python layoutclustering/feature_extractor_autoencoder.py train \
  --manifest train.jsonl \
  --model-out ae.pt \
  --latent-dim 128 \
  --epochs 100 \
  --batch-size 128
```

编码：

```bash
python layoutclustering/feature_extractor_autoencoder.py encode \
  --manifest all.jsonl \
  --model ae.pt \
  --features-out features.npz \
  --fv-manifest-out fv_manifest.jsonl
```

`features.npz` 必须包含：

```text
sample_ids
features
```

然后运行主脚本：

```bash
python layoutclustering/layout_clustering_optimized.py input.oas \
  --marker-layer 999/0 \
  --behavior-manifest behavior.jsonl \
  --feature-npz features.npz \
  --output results_optimized.json
```

## Layer Operations

AE optimized 和 no-train 主脚本都支持 layer boolean operations：

```bash
python layoutclustering/layout_clustering_optimized.py input.oas \
  --marker-layer 999/0 \
  --behavior-manifest behavior.jsonl \
  --feature-npz features.npz \
  --register-op 1/0 2/0 subtract 10/0 \
  --register-op 10/0 3/0 intersect 999/0 \
  --output results_layer_ops.json
```

规则含义：

```text
source_layer operation target_layer -> result_layer
```

支持操作：

```text
subtract
union
intersect
```

层格式固定为 `layer/datatype`。只要提供 `--register-op`，脚本会自动启用 layer operations；显式加 `--apply-layer-ops` 只是让命令更清楚。

## 关键参数

两条主线共享这些 clustering 参数：

```text
--ann-top-k 64
```

ANN graph 的每个 sample 近邻数。更大通常提高 coverage 候选质量，但会增加内存和计算量。

```text
--coverage-target 0.985
```

weighted facility location 的 coverage 目标。

```text
--facility-min-gain 1e-4
```

facility selection 的最小边际收益阈值。

```text
--behavior-verification-threshold 0.08
```

behavior weighted distance 的 final verification 阈值。距离越小越严格。

```text
--high-risk-quantile 0.90
```

weighted k-center 补洞时，把 risk score top 10% 作为 high-risk 样本。

## Behavior Verification

final verification 默认一定使用 aerial image SSIM distance：

```text
distance = 1 - SSIM
```

如果 `EPE/PV/NILS` 在 manifest 中全局可用，会加入 weighted score：

```text
aerial: 0.60
EPE:    0.15
PV:     0.15
NILS:   0.10
```

实际可用 channel 会自动重新归一化权重。verification 失败的 member 不做跨 cluster reassign，会直接成为 singleton/base representative，保证结果容易解释和 review。

## Review 与 Diff 输出

导出 review 目录：

```bash
python layoutclustering/layout_clustering_optimized_notrain.py input.oas \
  --marker-layer 999/0 \
  --behavior-manifest notrain_inputs \
  --review-dir review_notrain \
  --output results_notrain.json
```

兼容旧参数：

```text
--export-cluster-review-dir
```

如果两个 review 参数同时出现，优先使用 `--review-dir`。

可选导出 diff NPZ：

```bash
--export-diff-channels aerial,resist,pv
```

每个 cluster 目录会包含：

```text
REP__selected__*.oas
sample__*.oas
behavior_summary.json
diff__*.npz
```

## 输出字段

主要 JSON 字段：

```text
pipeline_mode
apply_layer_operations
layer_operation_count
layer_operations
marker_count
exact_cluster_count
selected_representative_count
selected_candidate_count
total_clusters
total_samples
cluster_sizes
behavior_stats
behavior_verification_stats
final_verification_stats
clusters
file_list
file_metadata
result_summary
config
cluster_review
```

no-train 额外字段：

```text
feature_source
feature_metadata
handcraft_feature_npz
handcraft_feature_metadata_json
input_marker_count
skipped_missing_behavior_count
```

`--format txt` 会写出中文摘要，适合快速查看 cluster 数、top cluster sizes、behavior stats 和 config。

## 当前不包含的旧路线

当前 optimized / no-train 行为版不再保留：

```text
HDBSCAN
ILP solver
FFT/PCM
closed-loop repair
auto-marker
几何 ACC/ECC 作为最终聚类 gate
```

几何信息仍用于 marker clip、exact hash duplicate grouping 和 no-train handcrafted layout/WL 特征；最终 cluster 成员关系由 behavior image verification 兜底。
