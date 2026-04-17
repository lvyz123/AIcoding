# Optimized Behavior Clustering 使用说明

`layout_clustering_optimized.py` 是当前推荐的 marker-driven 光刻行为覆盖聚类入口。它不再把“几何相似”作为最终聚类语义，而是使用 AutoEncoder feature vector 做 coverage representative selection，并用 aerial image SSIM 做 final verification。

主目标顺序固定为：

1. 尽可能增大光刻行为 coverage。
2. 在 coverage 足够高的前提下减少 representative / cluster 数。
3. 保证每个 cluster member 都能与 representative 通过 behavior verification。

## 主流程

```text
marker layer
  -> marker-centered clip
  -> exact hash 去重和 duplicate weight 累计
  -> 载入 AutoEncoder FV
  -> hnswlib ANN top-K graph
  -> weighted facility location 选 reps
  -> weighted k-center 补 farthest / high-risk holes
  -> behavior final verification
  -> JSON/TXT/review export
```

`exact hash` 只用于去重和权重累计，不再定义最终 cluster 语义。最终 cluster 是否成立由 behavior verification 决定。

## 必需输入

### OAS + Marker Layer

输入必须包含 marker layer。每个 marker 会生成一个 centered clip sample。

```bash
python layout_clustering_optimized.py input.oas \
  --marker-layer 999/0 \
  --behavior-manifest behavior.jsonl \
  --feature-npz features.npz \
  --output results_optimized.json
```

### Behavior Manifest

Manifest 使用 JSONL。每行至少包含：

```json
{
  "sample_id": "sample_000001",
  "source_path": "input.oas",
  "marker_id": "input__marker_000001",
  "clip_bbox": [0.0, 0.0, 1.35, 1.35],
  "aerial_npz": "aerial/sample_000001.npz"
}
```

可选字段：

```text
layout_npz
resist_npz
epe_npz
pv_npz
nils_npz
risk_score
```

约束：

- `aerial_npz` 必填。
- 每个 NPZ 必须包含 float32 数组键 `image`。
- `epe/pv/nils` 是数据集级可选项：如果某个 channel 在任意样本中出现，则所有样本都必须提供。
- `sample_id` 或 `marker_id` 必须能在 `features.npz` 中找到对应 FV。

## 生成 AutoEncoder FV

使用独立脚本 `layout_clustering_autoencoder.py`。

训练：

```bash
python layout_clustering_autoencoder.py train \
  --manifest train.jsonl \
  --model-out ae.pt \
  --latent-dim 128 \
  --epochs 100 \
  --batch-size 128
```

编码：

```bash
python layout_clustering_autoencoder.py encode \
  --manifest all.jsonl \
  --model ae.pt \
  --features-out features.npz \
  --fv-manifest-out fv_manifest.jsonl
```

`features.npz` 包含：

```text
sample_ids
features
```

其中 `sample_ids` 必须与主脚本中的 manifest sample 可一一匹配。

## Layer Operations

`--apply-layer-ops` 支持在 clustering 前对 OAS 中不同层做 boolean 操作，并把结果写到新层。常见用途是从多个设计层生成 marker 或 pattern 辅助层。

支持操作：

```text
subtract
union
intersect
```

层格式固定为：

```text
layer/datatype
```

示例：

```bash
python layout_clustering_optimized.py input.oas \
  --marker-layer 999/0 \
  --behavior-manifest behavior.jsonl \
  --feature-npz features.npz \
  --apply-layer-ops \
  --register-op 1/0 2/0 subtract 10/0 \
  --register-op 10/0 3/0 intersect 999/0 \
  --output results_layer_ops.json
```

规则含义：

```text
source_layer operation target_layer -> result_layer
```

只要提供 `--register-op`，脚本会自动启用 layer operations；显式写 `--apply-layer-ops` 只是让命令更清楚。

## 关键参数

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

默认一定使用 aerial image SSIM distance：

```text
distance = 1 - SSIM
```

如果 `EPE/PV/NILS` 全局可用，会加入 weighted score：

```text
aerial: 0.60
EPE:    0.15
PV:     0.15
NILS:   0.10
```

实际可用 channel 会自动重新归一化权重。verification 失败的 member 不做跨 cluster reassign，会直接成为 singleton/base representative，保证结果容易解释。

## Review 输出

导出 review 目录：

```bash
python layout_clustering_optimized.py input.oas \
  --marker-layer 999/0 \
  --behavior-manifest behavior.jsonl \
  --feature-npz features.npz \
  --review-dir review_optimized \
  --output results_optimized.json
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

JSON 输出包含主要字段：

```text
pipeline_mode
apply_layer_operations
layer_operation_count
layer_operations
marker_count
exact_cluster_count
selected_representative_count
total_clusters
total_samples
cluster_sizes
behavior_stats
behavior_verification_stats
clusters
file_list
file_metadata
result_summary
config
cluster_review
```

`--format txt` 会写出中文摘要，适合快速查看：

```bash
python layout_clustering_optimized.py input.oas \
  --marker-layer 999/0 \
  --behavior-manifest behavior.jsonl \
  --feature-npz features.npz \
  --format txt \
  --output summary.txt
```

## 过程显示

脚本默认输出中文阶段日志，包括：

- 启动配置和 layer operation 规则。
- 输入文件数和 marker layer。
- 每个文件的 marker 数、pattern 元素数、窗口样本数。
- exact hash 去重前后数量。
- ANN / facility / k-center representative 选择结果。
- behavior final verification pass/reject/singleton 统计。
- review/export 路径和最终 cluster summary。

## 不包含的旧路线

当前 optimized 行为版不再保留：

```text
HDBSCAN
ILP solver
FFT/PCM
closed-loop repair
auto-marker
几何 ACC/ECC 作为最终聚类 gate
```

几何信息仍可通过 marker clip 和 exact hash 支持去重，但最终聚类质量由 behavior image verification 兜底。
