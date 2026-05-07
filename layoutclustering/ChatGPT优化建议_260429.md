整体评价：**这一轮优化是有效的，而且比上一轮更切中 12G laptop 的真实瓶颈。**
上一轮主要解决 “mega bundle 太大、strict key 太重、cache 没释放”；这一轮进一步解决了两个更核心的问题：

1. **不再长期持有 140 万级 CandidateClip 对象**；
2. **不再为整个 coverage window 一次性常驻 full signature embedding 矩阵**。

这两个点都非常关键。我的判断是：**当前版本在 12G laptop 上的可行性明显高于上一版，已经从“高概率 OOM”变成“有较大概率跑完，但仍取决于 marker 阶段和最大 bucket window 的规模”。**

---

## 1. 本轮最重要的优化点

### 1.1 CandidateGroup 先行：这是本轮最有价值的改动

新版引入了 `CoverageCandidateGroup`，coverage 阶段只长期保留一个 `best_candidate`，同时用 `origin_ids / logical_candidate_count / direction_counts` 表示被合并掉的候选。这个设计把 “candidate 对象全集” 改成了 “coverage group 集合”。

更关键的是，主流程现在已经不再构建全量 `all_candidates`。它在 candidate 生成阶段直接调用 `_build_global_coverage_candidate_groups()`，逐个 exact cluster 生成候选、立刻 merge 到全局 group，然后清空 cluster-local candidate 列表。日志也会输出：

```text
candidate 生成完成: <candidate_count> 个, group=<len(candidate_groups)>
```

并记录 `candidate_object_avoided_count`。

这正好解决你前面生产日志里的大问题：

```text
candidate = 1,402,890
coverage group = 914,535
```

旧流程需要长期持有 140 万 candidate object；新流程理论上只长期持有约 91 万 group 的 representative。这个改动对 `rss_candidate_generation_mb` 应该有明显帮助。

**评价：已落地，方向正确，收益大，不牺牲精度/覆盖率。**
因为它只是合并 strict bitmap 完全相同的 candidate；最终 coverage 和 ECC 仍在 group representative 上做。

---

### 1.2 lazy signature embedding：有效降低 coverage shortlist 峰值

上一版 bucketed coverage 解决了 “一个 91 万 group mega bundle” 的问题，但每个 bucket/window 内仍可能一次性构建 signature embedding matrix。现在 `_build_bundle_shortlist_index()` 只预先构建：

* `cheap_invariants`
* `subgroup_members`
* `subgroup_remaining`
* `source_subgroup_ids`

而不是给全部 group 构建 120 维 signature embedding 矩阵。代码还记录 `signature_embedding_bytes_avoided_estimate_mb`，用于估算避免常驻的 embedding 内存。

真正需要 ANN / exact top-k 的时候，`_ensure_shortlist_payload()` 才针对某个 subgroup 构造 `group_vectors`，并在 subgroup 全部 source 处理完后释放 payload，同时更新 `live_signature_embedding_groups` 和 `signature_embedding_live_peak_count`。 

这点非常好。假设一个 bucket window 有 200k group，120 维 float32 embedding 常驻约 96MB；如果 window 更大，例如 500k，就是约 240MB。现在它变成 subgroup 级别临时构建，峰值由 `shortlist_max_subgroup_size` 决定，而不是由整个 window 决定。

**评价：已落地，且对 12G laptop 很关键。**
代价是可能增加一些 CPU 时间，因为 signature embedding 按 subgroup 懒构建，可能会重复计算 cheap descriptor。但对当前目标来说，**用时间换内存是合理的**。

---

### 1.3 之前的 bucketed coverage / strict digest key 仍然保留

当前版本仍保留：

```python
COVERAGE_BUCKETED_GROUP_THRESHOLD = 200_000
COVERAGE_FILL_BIN_WIDTH = 0.04
STRICT_BITMAP_DIGEST_SIZE = 16
```

并保留 bucketed coverage 统计字段：

```text
bucketed_coverage_bundle_count
coverage_fill_bin_count
max_fill_bin_group_count
max_bucket_window_group_count
bucketed_source_group_count
bucketed_target_group_count
```

这些是判断 12G 是否可跑的关键诊断字段。

strict digest key 也仍然保留，并通过 `_same_bitmap()` 处理 hash collision，避免把大 packed bytes 直接作为 dict key。

**评价：上一轮 P0 优化保留完整。**

---

## 2. 当前版本在 12G laptop 上的可行性

我的判断标准不变，但现在成功概率更高了。

### 更可能改善的峰值

这一轮主要会改善两个峰值：

```text
rss_candidate_generation_mb
rss_coverage_eval_mb
```

原因是：

* candidate generation 不再长期持有 `all_candidates`；
* coverage shortlist 不再长期持有全 window signature embedding；
* bucketed coverage + lazy signature 叠加后，coverage 的内存峰值应该明显低于上一版。

### 仍然可能卡住的峰值

最危险的仍然是这几个：

```text
rss_collect_markers_mb
rss_exact_cluster_mb
rss_candidate_generation_mb
max_bucket_window_group_count
shortlist_max_subgroup_size
signature_embedding_live_peak_count
```

尤其是你之前日志里：

```text
rss_collect_markers_mb = 8970 MB
rss_exact_cluster_mb = 8951 MB
```

这个阶段已经非常高。也就是说，即使 candidate/coverage 后面优化得很好，**marker 收集阶段如果仍然接近 9GB，12G laptop 仍然没有太多安全余量**。

我会这样判定：

| 指标                                        | 评价                           |
| ----------------------------------------- | ---------------------------- |
| `rss_collect_markers_mb < 7000`           | 比较安全                         |
| `rss_collect_markers_mb 7000~8500`        | 可跑但要小心                       |
| `rss_collect_markers_mb > 8500`           | 仍有 OOM 风险                    |
| `rss_candidate_generation_mb < 8000`      | 当前版本大概率可继续                   |
| `max_bucket_window_group_count < 200k`    | coverage 较安全                 |
| `max_bucket_window_group_count 200k~500k` | 可能能跑，但慢且有风险                  |
| `shortlist_max_subgroup_size > 100k`      | HNSW / exact top-k 内存和时间风险较高 |

---

## 3. 是否影响精度 / 覆盖率？

这一轮优化基本不应牺牲 precision / recall。

### CandidateGroup 合并

它只合并 strict bitmap 完全一致的 candidate，并保留 origin ids 和 coverage。这个不会降低几何覆盖能力。

### Lazy signature embedding

它不改变 shortlist 的相似度定义，只改变 embedding 构建时机；不会改变理论结果。

### Bucketed coverage

只要 fill-bin 邻域半径仍然覆盖 `CHEAP_FILL_ABS_LIMIT` 的允许范围，它就是计算组织方式，不是新的 hard reject。你现在仍保留 `CHEAP_FILL_ABS_LIMIT = 0.12` 和 `COVERAGE_FILL_BIN_WIDTH = 0.04`，这个组合是合理的。

所以本轮改动整体属于：

```text
内存组织优化 > 算法语义修改
```

风险主要是实现 bug，不是算法召回本身。

---

## 4. 本轮优化后的主要剩余风险

### 风险 1：marker_records 仍是全量常驻

这是现在最值得继续盯的地方。当前主流程仍然会把所有 marker records 放进 `marker_records` 列表，然后 exact cluster，再释放非 representative payload。

对你之前的 `426,838` marker 数量来说，这一步曾经达到接近 9GB。后续 candidate/group 优化救不了这个前置峰值。

如果当前版本 `rss_collect_markers_mb` 仍然高于 8.5GB，下一步重点就不该再动 coverage，而应该做 **marker/exact streaming**。

### 风险 2：candidate group 仍然长期持有 representative bitmap

`CoverageCandidateGroup` 只保留一个长期 candidate，已经比全量 candidate 好很多。但如果 group 数仍然接近 90 万，每个 group 的 representative `clip_bitmap` 仍是常驻对象。

如果 candidate group 数没有明显下降，长期内存仍然不小。下一步可以考虑把 representative bitmap 也 pack 成 bitset，只有 geometry cache 需要时再 unpack。

### 风险 3：lazy embedding 会增加运行时间

现在 `_ensure_shortlist_payload()` 会按 subgroup 临时构建 `group_vectors`，其中每个 group 仍会从 bitmap 重新提 cheap descriptor，再做 `_signature_embedding()`。

这省内存，但可能明显变慢。
如果运行时间变得很长，可以考虑一个折中：

* 不保存完整 120 维 float32 embedding；
* 保存压缩版 cheap signature，例如 `float16` 或 `uint8` quantized pooled signature；
* subgroup 内再转 float32 做 HNSW / cosine。

这样能减少重复 descriptor 计算，同时不回到大矩阵常驻。

### 风险 4：当前 v1 仍不是 Python 3.6 兼容

如果这轮优化仍要求 Python 3.6 兼容，需要提醒一下：当前 `layout_clustering_optimized_v1.py` 使用了：

* `from __future__ import annotations`
* `@dataclass(..., slots=True)`
* `np.ndarray | None`
* `set[str]`

这些都不是 Python 3.6 兼容写法。

如果只是 12G laptop 本地运行，Python 版本较新就没问题；如果你希望把这版逻辑移植回 LSF/Python 3.6，需要另起 `_lsf` 或 py36 兼容版本。

---

## 5. 我建议下一轮重点看哪些日志

请优先看这些新增/关键字段：

```text
rss_collect_markers_mb
rss_exact_cluster_mb
rss_candidate_generation_mb
rss_coverage_eval_mb

candidate_count
candidate_group_count
candidate_object_avoided_count

strict_key_bytes_avoided_estimate_mb
signature_embedding_bytes_avoided_estimate_mb
lazy_signature_embedding_group_count
signature_embedding_live_peak_count

bucketed_coverage_bundle_count
coverage_fill_bin_count
max_fill_bin_group_count
max_bucket_window_group_count
shortlist_max_subgroup_size
```

尤其是这三个比值很有用：

```text
candidate_object_avoided_count / candidate_count
candidate_group_count / candidate_count
signature_embedding_live_peak_count / candidate_group_count
```

如果：

```text
candidate_group_count / candidate_count < 0.7
```

说明 CandidateGroup 合并效果很好。

如果：

```text
signature_embedding_live_peak_count << candidate_group_count
```

说明 lazy signature embedding 真正降低了峰值。

如果：

```text
max_bucket_window_group_count < 200k
```

coverage 在 12G 上基本有希望。

---

## 6. 进一步优化建议

### P0：加 hard memory guard

现在建议不要再“跑到系统 OOM”。加硬阈值：

```text
rss_collect_markers_mb > 9000: abort / 建议切分或 low-memory mode
rss_candidate_generation_mb > 8500: abort / 建议 disk-backed coverage
max_bucket_window_group_count > 500000: abort / 强制 disk-backed coverage
shortlist_max_subgroup_size > 150000: warning / 降低 shortlist 或改 exact strategy
```

这个对 laptop 很重要。

---

### P1：marker/exact streaming

如果 `rss_collect_markers_mb` 仍然高，下一步最该做这个。

思路：

```text
seed -> raster -> exact key
若 exact key 已存在：
    只累计 member_count / seed_weight / representative id
    不长期保留完整 MarkerRecord
否则：
    保留 representative MarkerRecord
```

也就是说，exact clustering 不要等所有 marker records 都建完再做，而是在 collect 阶段就做 online exact grouping。

这会把 marker 常驻量从：

```text
426k marker records
```

压成接近：

```text
126k representative records + member counts
```

这对你之前的日志会非常有帮助。

---

### P1：candidate representative bitmap packbits

当前 group representative 仍然持有 `clip_bitmap`。下一步可以把 candidate group 的 representative bitmap 改成：

```text
packed_clip_bitmap + shape
```

在 `_bundle_geometry_cache()` 或 final verification 时再 unpack。
这会增加 CPU，但对于 90 万 group 的场景，内存收益很明显。

---

### P1：低内存模式参数化

加一个：

```bash
--memory-mode low
```

可自动启用：

```text
COVERAGE_SHORTLIST_MAX_TARGETS = 32
DIAGONAL_SHIFT_MAX_COUNT = 1
只对 array_spacing / residual 开 diagonal
COVERAGE_FILL_BIN_WIDTH = 0.02
禁用 review materialization
启用更频繁 gc
```

这些会有一点 coverage / runtime tradeoff，所以不要默认开，但对 12G laptop 很实用。

---

### P2：disk-backed coverage

如果当前版本仍然在 coverage 阶段 OOM，下一步不要继续做 in-memory 微优化，直接做：

```text
candidate groups -> bucket npz/json
for each bucket window:
    load current source/target window
    compute coverage
    write CSR
merge CSR
greedy set cover
```

这等于把 LSF 版的 CSR/out-of-core 思路移植到 laptop 顺序执行。它是最终兜底方案。

---

## 最终结论

这一轮优化效果我会评为：**很有价值，且是正确方向。**

| 优化点                                 | 落地情况 | 评价                         |
| ----------------------------------- | ---: | -------------------------- |
| CandidateGroup 先行                   |  已落地 | 大幅降低 candidate object 常驻   |
| `candidate_object_avoided_count` 统计 |  已落地 | 能量化收益                      |
| lazy signature embedding            |  已落地 | 明显降低 coverage shortlist 峰值 |
| signature embedding live peak 统计    |  已落地 | 便于判断真实峰值                   |
| bucketed coverage                   |   保留 | 仍是 coverage 防 OOM 关键       |
| strict digest key                   |   保留 | 减少 strict key 内存           |
| marker 阶段 streaming                 |  未落地 | 若 collect 阶段仍高，这是下一步 P0    |
| disk-backed coverage                |  未落地 | 若 coverage 仍 OOM，这是最终兜底    |

一句话总结：

**这版已经把最大的一批 in-memory 问题处理掉了，值得直接重新跑 `tolyu_test2.oas`。如果仍然失败，下一步不要再优先优化 candidate/coverage 小细节，而要转向 marker exact streaming 或 disk-backed coverage。**
