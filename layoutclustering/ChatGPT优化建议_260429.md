结论：**当前版本已经基本具备 LSF 集群化运行的核心架构，但还没到“生产集群一键稳定跑 full-chip”的程度。**更准确地说，它满足了“算法阶段拆分 + shard 产物落盘 + coverage 分布式计算 + 合并”的主需求；但还缺少 LSF 调度包装、集中式瓶颈削弱、I/O 风险控制和文件部署细节。

## 1. 已经满足的部分

任务需求要求 pattern grouping 能自动找中心点切 clip，并通过精确哈希、预筛选、候选 clip、ACC/ECC 匹配和集合覆盖来减少聚类数量、兼顾 recall/precision。任务描述里明确写了这些目标和流程：自动 clipping、重复 pattern 抓取、精确哈希、invariant/topo/signature 预筛、ACC/ECC 和 set cover。

当前脚本的 LSF 版入口已经把流程拆成了几个适合集群调度的阶段：

```text
prepare
run-shard
prepare-coverage
run-coverage-shard
merge-coverage
inspect-workdir
```

脚本注释里也明确说明：`prepare` 生成 seeds 和 shard manifest，`run-shard` 每个 job 只处理一段 seed，`prepare-coverage` 汇总 marker 并生成 exact clusters / candidate bundle / coverage shards，`run-coverage-shard` 计算 CSR coverage，`merge-coverage` 做 greedy set cover 并懒加载 selected bitmap。

这说明它已经不是单机脚本简单加参数，而是比较合理地拆成了 **seed raster 分布式** 和 **coverage 计算分布式** 两层。这个方向是对的。

## 2. 但当前版本还不能算完全“LSF 生产适配”

最大的问题不是算法，而是工程化边界。

### 问题 A：上传版本的文件名直接运行会失败

你上传的文件名带了 `(3)`，例如：

```text
layout_clustering_optimized_v2_lsf(3).py
mainline_lsf(3).py
layer_operations_lsf(3).py
layout_utils_lsf(3).py
```

但主脚本里 import 的是：

```python
from layer_operations_lsf import LayerOperationProcessor
from mainline_lsf import ...
```

也就是说，在当前上传文件名状态下，直接运行会找不到模块。需要在集群目录里改成标准文件名：

```text
layout_clustering_optimized_v2_lsf.py
mainline_lsf.py
layer_operations_lsf.py
layout_utils_lsf.py
```

否则 LSF 上会直接 `ModuleNotFoundError`。

### 问题 B：没有真正生成 bsub/job-array 脚本

manifest 里会写出每个 shard 的 command，并且 parser 支持 `run-shard`、`run-coverage-shard`、`merge-coverage` 等阶段。 这对 LSF 适配是够用的“底层接口”，但还没有封装：

```bash
bsub -J "pg_shard[1-N]"
bsub -w "done(pg_shard)"
bsub -J "pg_cov[1-M]"
bsub -w "done(pg_cov)"
```

所以目前是 **LSF-ready CLI**，不是完整的 **LSF workflow runner**。

### 问题 C：run-shard 仍会每个 job 重读整个 OAS

`run_shard_stage` 会先 `prepare_layout(input_path, ...)` 读取完整 OAS 并构建完整 layout index，然后才根据 shard 的 `halo_bbox` 做过滤。

这意味着：

```text
CPU/coverage 被分布式了；
但每个 seed shard 的初始 OAS 读取和 LayoutIndex 构建仍是全量的。
```

如果 LSF 节点内存足够，这可以接受；但如果每个 job 只给 4G/8G，或者大量 shard 同时从共享文件系统读同一个大 OAS，会有风险。这个问题对 550um × 550um 的 `tolyu_test2.oas` 可能还能接受，但对更大 full chip 会变成瓶颈。

### 问题 D：prepare-coverage 仍是集中式内存峰值点

`prepare_coverage_stage` 会加载所有 shard records，做全局 exact clustering，然后生成所有 candidates，再写 candidate bundle 和 coverage source shards。

虽然它已经比原始集中式 coverage 好很多，但这里仍然可能成为内存峰值：

```text
所有 marker_records
+ representative clip/expanded bitmap
+ all_candidates
+ candidate bundle 写出过程中的 bitmap/descriptor
```

如果 seed 数量达到几十万级，`prepare-coverage` 仍然可能接近或超过单节点内存限制。

### 问题 E：merge-coverage 仍需一次性加载所有 CSR coverage metadata

`merge_coverage_stage` 不再预加载所有 candidate bitmap，只加载 coverage shard 的 metadata 和 CSR offsets/values，然后执行 `greedy_cover_csr`，最后只懒加载 selected candidate bitmap。 这是明显优化。

但它仍然会把所有 coverage edges 合并到一个 `coverage_values` 数组里。 如果 coverage 边数量特别大，merge 阶段也可能成为第二个集中式瓶颈。

## 3. 当前版本的可行性判断

我会这样定级：

| 维度                          | 评价                                                         |
| --------------------------- | ---------------------------------------------------------- |
| 算法功能对任务需求                   | **基本满足**                                                   |
| LSF 阶段拆分                    | **基本满足**                                                   |
| Python 3.6 / 独立部署           | **基本满足**，但要修正文件名                                           |
| job array / dependency 自动提交 | **未满足**                                                    |
| 大版图单节点内存风险                  | **仍有风险**                                                   |
| shared filesystem I/O 风险    | **仍有风险**                                                   |
| 失败恢复/断点重跑                   | **部分满足**，因为 shard 产物落盘，但缺少自动跳过已完成 shard                    |
| production 可观测性             | **部分满足**，已有 `inspect-workdir`，但还缺 RSS/峰值内存/LSF resource 记录 |

所以答案是：

**可以作为 LSF 适配的第一版运行；如果目标是 7 月算法方案交付，它已经够做集群验证。但如果目标是 full-chip 大规模生产稳定运行，还需要再补一轮工程优化。**

## 4. 推荐的 LSF 运行方式

不要走旧的集中式 `merge`，大版图应走 distributed coverage：

```bash
python layout_clustering_optimized_v2_lsf.py prepare \
  tolyu_test2.oas \
  --work-dir work_pg \
  --clip-size 1.35 \
  --geometry-match-mode ecc \
  --pixel-size-nm 10 \
  --edge-tolerance-um 0.02 \
  --shard-size 1000 \
  --apply-layer-ops \
  --register-op 2413/0 2410/0 subtract 2411/0
```

然后用 manifest 里的 shard 数提交 job array：

```bash
bsub -J "pg_shard[1-N]" -M 12000 -R "rusage[mem=12000]" \
  'python layout_clustering_optimized_v2_lsf.py run-shard \
   --manifest work_pg/manifest.json \
   --shard-id $((LSB_JOBINDEX-1))'
```

shard 完成后：

```bash
python layout_clustering_optimized_v2_lsf.py prepare-coverage \
  --manifest work_pg/manifest.json \
  --coverage-shard-size 200
```

再提交 coverage job array：

```bash
bsub -J "pg_cov[1-M]" -M 12000 -R "rusage[mem=12000]" \
  'python layout_clustering_optimized_v2_lsf.py run-coverage-shard \
   --manifest work_pg/manifest.json \
   --coverage-shard-id $((LSB_JOBINDEX-1))'
```

最后：

```bash
python layout_clustering_optimized_v2_lsf.py merge-coverage \
  --manifest work_pg/manifest.json \
  --output clustering_results.json
```

再检查产物规模：

```bash
python layout_clustering_optimized_v2_lsf.py inspect-workdir \
  --manifest work_pg/manifest.json \
  --output work_pg/inspect.json
```

## 5. 下一步最值得补的优化

优先级我建议这样排：

1. **补 LSF wrapper 脚本**
   自动读取 `manifest.json` 的 `shard_count` / `coverage_shard_count`，生成并提交 `bsub` job array，自动加 dependency。

2. **prepare-coverage 流式化**
   不要一次性 `_load_all_marker_records` 后再 `group_exact_clusters`。可以按 shard 读取 metadata/hash，先做 exact key 聚合，只保留 representative bitmap；候选 bundle 也按 shape/fill-bin 边生成边写。

3. **run-shard 避免每个 job 全量读 OAS**
   更稳的做法是 `prepare` 阶段额外生成 tile OAS / clipped element cache；`run-shard` 只读对应 halo tile，而不是全量 OAS 后过滤。

4. **coverage merge 二级化**
   当 `coverage_value_count` 很大时，先做 per-group partial greedy 或分层 set cover reduce，避免一个 merge job 吃下所有 CSR edge。

5. **加资源诊断字段**
   每个阶段写入 `max_rss_mb`、输入文件大小、npz 解压后大小、candidate group count、coverage edge count。现在已有 `inspect-workdir`，再补内存统计会更适合调 LSF resource request。

总体判断：**这版已经从算法脚本进化成了“可被 LSF 调度的分阶段 pipeline”，方向正确；当前主要短板是调度封装和两个集中式瓶颈，而不是核心 pattern grouping 算法本身。**
