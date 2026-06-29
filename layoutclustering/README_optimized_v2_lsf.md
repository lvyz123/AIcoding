# Optimized V2 LSF Layout Clustering

`layout_clustering_optimized_v2_lsf.py` 是当前 optimized v1 主算法的 LSF / Python 3.12 独立适配版。

它的目标有两层：

1. 在算法语义上对齐当前 v1：`geometry-driven seed + bounded diagonal shift + shift-witness final verification`。
2. 在运行架构上支持大版图分阶段执行：seed shard、candidate bundle、coverage CSR shard 和最终 merge。

当前版本不 import v1、旧 `mainline`、旧 `layout_utils` 或旧 `layer_operations`。最终用户输出是 CSV-only；LSF 所需的 manifest、JSON、NPZ、tile OAS 只作为中间产物。

## 算法流程

整体流程如下：

1. `prepare` 读取 OAS，应用可选 layer operation，生成 geometry-driven seeds。
2. `prepare` 将 seed 切成 shard，写出 `manifest.json`、tile OAS 和 LSF 命令清单。
3. `run-shard` 处理单个 seed shard，输出 marker records 的 JSON/NPZ。
4. `prepare-coverage` 汇总 marker records，构造 exact clusters、candidate bundle buckets 和 coverage source shards。
5. `run-coverage-shard` 读取 source shard 与 candidate bundle bucket，计算 coverage CSR。
6. `merge-coverage` 汇总 coverage CSR，执行 greedy set cover，并懒加载 selected candidate bitmap。
7. final verification 使用统一 shift-witness 方案：exact hash 直接通过；非 exact hash 必须先过 ACC/ECC geometry，再过内部 strict graph gate。
8. 输出主 Exact Cluster Review CSV 和自动派生的 Cluster Representative CSV。

小样本或 crop 验证可以使用 `run-local` 顺序模拟上述流程；大样本建议走 `prepare-coverage / run-coverage-shard / merge-coverage`。

## 运行方式

本地小样本验证：

```bash
python layout_clustering_optimized_v2_lsf.py run-local ./design.oas --work-dir work_v2_lsf --output clustering_results.csv
```

本地顺序模拟分布式 coverage：

```bash
python layout_clustering_optimized_v2_lsf.py run-local ./design.oas --work-dir work_v2_lsf --output clustering_results.csv --distributed-coverage
```

LSF 分阶段流程：

```bash
python layout_clustering_optimized_v2_lsf.py prepare ./design.oas --work-dir work_v2_lsf
python layout_clustering_optimized_v2_lsf.py run-shard --manifest work_v2_lsf/manifest.json --shard-id 0
python layout_clustering_optimized_v2_lsf.py prepare-coverage --manifest work_v2_lsf/manifest.json
python layout_clustering_optimized_v2_lsf.py run-coverage-shard --manifest work_v2_lsf/manifest.json --coverage-shard-id 0
python layout_clustering_optimized_v2_lsf.py merge-coverage --manifest work_v2_lsf/manifest.json --output clustering_results.csv
```

`prepare` 会在 `work_dir/lsf/` 下生成 `run_shards` 和 `run_coverage_shards` 的命令清单与 bsub 模板；脚本不自动提交作业，便于按集群队列和资源策略手动接入。

## 关键参数

- `input_path`：`prepare` / `run-local` 的输入 OAS 文件。
- `--work-dir`：LSF 工作目录，用于保存 manifest、tile、shard、candidate bundle 和 coverage shard 中间产物。
- `--output`, `-o`：主 Exact Cluster Review CSV 路径；必须使用 `.csv` 后缀。Cluster Representative CSV 自动保存为 `<output_stem>_cluster_representatives.csv`。
- `--clip-size`：clip 边长，单位 `um`，默认 `1.35`。
- `--geometry-match-mode`：最终几何 gate，`acc` 或 `ecc`，默认 `ecc`。
- `--area-match-ratio`：`acc` 模式的面积匹配阈值，默认 `0.96`。
- `--edge-tolerance-um`：`ecc` 模式的边界容差，单位 `um`，默认 `0.02`。
- `--pixel-size-nm`：栅格像素尺寸，默认 `10nm`。
- `--compute-quality-metrics`：可选计算 representative visual、pairwise geometry、fragmentation coverage graph，并执行与 v1 对齐的 singleton absorption 收尾；默认关闭。开启后不写额外诊断 CSV，只在最终 stage JSON 摘要中打印核心指标，并在 Cluster Representative CSV 追加 per-cluster quality 字段。
- `--apply-layer-ops`：启用层操作预处理。
- `--register-op SOURCE_LAYER TARGET_LAYER OPERATION RESULT_LAYER`：注册层操作规则。
- `--distributed-coverage`：仅用于 `run-local`，顺序模拟完整分布式 coverage 流程。
- `--coverage-shard-count` / `--coverage-shard-size`：规划 coverage shard 数量或大小。

当前 v2_lsf 不提供这些旧入口：

- `--format`
- graph / strict threshold 的 CLI 调参入口
- `--coverage-shortlist-max-targets`
- JSON / TXT 最终结果输出
- final verification 的旧 base-only 模式或 fallback 模式

## 输出 CSV

### Exact Cluster Review CSV

`--output/-o` 指定主 CSV。它每行对应一个 marker-defined exact cluster review group，用于 OPC 专业人员按最终 `cluster_id` review purity，并跨 cluster 比较 recall。

列固定为：

```text
groupID,cluster_id,center_x_um,center_y_um,clip_size,group_weight,risk_score,risk_rank
```

字段含义：

- `groupID`：1-based review group ID；当前定义为 `exact_cluster_id + 1`。
- `cluster_id`：final verification 后所属最终 cluster 的 1-based ID。
- `center_x_um`, `center_y_um`：marker-defined layout clip 的中心点，不使用 shift 后 selected candidate center。
- `clip_size`：运行配置中的 clip 边长。
- `group_weight`：该 exact cluster 代表的重复权重。
- `risk_score`：marker-defined base clip 的弱点风险 proxy。
- `risk_rank`：按 `risk_score` 降序生成，`1` 表示最优先 review / weak-point 关注。

`result_csv_row_count` 等于 `exact_cluster_count`，不再等于内部 `candidate_group_count`。

### Cluster Representative CSV

代表 CSV 自动保存为 `<output_stem>_cluster_representatives.csv`，每行对应一个最终 cluster。

默认列为：

```text
cluster_id,center_x_um,center_y_um,clip_size,cluster_size,cluster_weight,exact_cluster_count,representative_seed_type,shift_direction,shift_distance_um,representative_score,opc_center_score,risk_score,risk_rank
```

开启 `--compute-quality-metrics` 后追加：

```text
representative_visual_pass_ratio,representative_visual_fail_count,representative_visual_checked_count,representative_visual_sample_status,pairwise_geometry_purity,pairwise_geometry_fail_count,pairwise_geometry_sampled_pair_count,pairwise_geometry_sample_status,overmerge_score,overmerge_reason
```

默认最终磁盘输出只有这两个 CSV。`manifest.json`、candidate bundle、coverage shard JSON/NPZ 是 LSF 中间产物，不属于最终用户结果。

## 关键诊断

最终类 stage 会打印一行 JSON 摘要，包含：

- `exact_cluster_count`
- `candidate_group_count`
- `selected_candidate_count`
- `final_verification_pass`
- `final_verification_reject`
- `final_verification_singleton`
- `final_verification_reject_reason_counts`
- `total_clusters`
- `seed_coverage_audit`
- `quality_metrics`，仅在开启 `--compute-quality-metrics` 后出现

`quality_metrics` 按问题域拆分输出：`representative_visual_purity` / `weighted_representative_visual_purity` 关注代表点对成员的覆盖质量，`pairwise_geometry_purity` / `weighted_pairwise_geometry_purity` 关注簇内成员两两几何一致性，`raw_coverage_graph_recall` / `trusted_fragmentation_recall` 区分原始 coverage graph 与可信已合并边，`gate_rejected_edge_weight_ratio` / `review_merge_candidate_weight_ratio` 用于判断剩余 recall 信号是被 purity gate 有意拒绝，还是仍有 review merge 候选。

开启 `--compute-quality-metrics` 时，v2_lsf 会执行与当前 v1 收尾 baseline 对齐的 singleton absorption：先使用 review edge 与 strong descriptor/graph agreement 吸收 singleton，再用 strict-only singleton microcluster 做 size-3 clique 与 pair fallback。该机制会真实改写最终 `cluster_units`，但不把 `singleton_absorption_*` / `singleton_microcluster_*` 诊断字段写入 public `quality_metrics` 或终端 final payload。不开启质量指标时不执行这层额外 singleton absorption。

`seed_coverage_audit` 包含：

- `target_edge_length_total`
- `target_edge_length_covered`
- `target_edge_length_coverage_ratio`
- `target_polygon_area_total`
- `target_polygon_area_covered`
- `target_polygon_area_coverage_ratio`
- `target_pattern_type_weight_total`
- `target_pattern_type_weight_covered`
- `weighted_pattern_type_coverage_ratio`
- `target_pattern_type_count`
- `covered_pattern_type_count`
- `target_polygon_count`
- `covered_polygon_count`
- `clip_window_count`

当前 coverage audit 按真实 target polygon 的边长、面积和轻量 pattern type 权重统计，不再输出 occupied-grid、seed-density 或 `clip_window_union_coverage_ratio` 系列旧指标。

## Set-Cover 与代表评分

greedy set cover 的优先级为：

1. `uncovered_exact_count` 最大
2. `uncovered_weight_gain` 最大
3. `representative_score` 最大
4. `coverage_confidence_proxy` 最大
5. base 方向优先
6. `shift_distance_um` 最小
7. `origin_exact_cluster_id`
8. `candidate_id`

`uncovered_exact_count` 是当前主目标，用于优先降低 singleton cluster 数量；`representative_score` 来自 OPC-center score、coverage weight 和 risk score 的组合，用于在覆盖收益相同的候选之间优先选择更适合作为 OPC clip center 的代表点。v2_lsf 当前不继续新增更激进的 residual singleton 压制策略，收尾阶段只对齐 v1 的 bounded singleton absorption 和 strict-only microcluster，最终几何验证仍是合并硬门槛。

## Python 3.12 运行环境与依赖

v2_lsf 只使用 `Python 3.12 supporting packages.txt` 清单内的运行依赖：

- `gdstk`
- `numpy`
- `scipy`
- `scikit-learn`（代码内 import 名为 `sklearn`，用于大 subgroup 的 cosine 近邻 shortlist）

代码已不再维护 Python 3.6 语法兼容约束；可以使用 Python 3.12 下更直接的 `dataclass(slots=True)` 等写法。仍然不依赖 `scipy.optimize.milp`，避免引入额外求解器路径。

推荐回归命令：

```bash
python -m unittest test_optimized_clustering_v2_lsf -v
```
