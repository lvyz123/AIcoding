# Optimized V1 Layout Clustering

`layout_clustering_optimized_v1.py` 是当前单进程的 optimized v1 主线实现。

它的目标有两层：

1. 在算法语义上尽量向 `v2_lsf` 看齐，使用 `geometry-driven seed + bounded diagonal shift`。
2. 继续保留 v1 的单脚本、单进程、易调试和易 review 的运行方式。

当前版本已经不再使用旧的 `uniform grid` 全域扫窗主线，也不引入 `v2_lsf` 的 LSF 分布式调度、manifest/shard、CSR/NPZ coverage 落盘等运行架构。

## 算法流程

整体流程如下：

1. 读取单个 OAS 文件，或读取目录下的 `*.oas` 文件。
2. 可选执行 `--apply-layer-ops` / `--register-op` 指定的层间布尔操作。
3. 在版图图形上做 `geometry-driven` 采样，生成四类 seed：
   - `array_representative`
   - `array_spacing`
   - `long_shape_path`
   - `residual_local_grid`
4. 把 seed 栅格化成 marker clip，并立即进入 online exact grouping。
5. 同 exact key 的 marker 聚合成 exact cluster，只保留一个 representative 持续参与后续 candidate generation。
6. 对每个 exact cluster 生成：
   - `base`
   - `left/right/up/down`
   - 少量 `diag_ne/diag_nw/diag_se/diag_sw`
7. candidate generation 结束后，把逻辑 candidate 压成全局 strict bitmap `CoverageCandidateGroup`。
8. coverage 阶段先做 cheap shortlist，再做 lazy full prefilter，最后做 ACC/ECC 几何比较。
9. 通过 greedy set cover 选择 representative candidate。
10. 对 selected candidate 做统一 shift-witness final verification；所有 witness 都失败的成员回退成 singleton。
11. 输出 CSV 主结果；每行对应一个 `CoverageCandidateGroup`，仅在指定 `--review-dir` 时物化 sample / representative 位图文件。

## 运行方式

```bash
python layout_clustering_optimized_v1.py ./design.oas --output clustering_results.csv
```

常用示例：

```bash
python layout_clustering_optimized_v1.py ./design.oas --output clustering_results.csv
python layout_clustering_optimized_v1.py ./input_dir --output clustering_results.csv
python layout_clustering_optimized_v1.py ./design.oas --clip-size 1.35 --geometry-match-mode ecc --output clustering_results.csv
python layout_clustering_optimized_v1.py ./design.oas --review-dir review_out --output clustering_results.csv
python layout_clustering_optimized_v1.py ./design.oas --apply-layer-ops --register-op 2413/0 2410/0 subtract 2411/0 --output clustering_results.csv
```

## 关键参数

- `input_path`：输入 OAS 文件或目录。
- `--output`, `-o`：输出 CSV 路径，默认 `clustering_results.csv`；必须使用 `.csv` 后缀，传入 `.json`、`.txt` 或无后缀路径会立即报错。
- `--clip-size`：clip 边长，单位 `um`，默认 `1.35`。
- `--geometry-match-mode`：最终几何 gate，`acc` 或 `ecc`，默认 `ecc`。
- `--area-match-ratio`：`acc` 模式的面积匹配阈值，默认 `0.96`。
- `--edge-tolerance-um`：`ecc` 模式的边界容差，默认 `0.02`。
- `--pixel-size-nm`：栅格像素尺寸，默认 `10nm`。
- `--review-dir`：导出 review 目录；指定后会物化 `samples/` 和 `representatives/`。
- `--export-cluster-review-dir`：`--review-dir` 的兼容别名。
- `--apply-layer-ops`：启用层操作预处理。
- `--register-op SOURCE_LAYER TARGET_LAYER OPERATION RESULT_LAYER`：注册层操作规则。

当前 v1 不再提供这些旧入口：

- `--seed-strategy`
- `--marker-layer`
- `--hotspot-layer`
- `--format`
- `--graph-invariant-limit`, `--graph-topology-threshold`, `--graph-signature-threshold`
- `--strict-invariant-limit`, `--strict-topology-threshold`, `--strict-signature-threshold`
- `--coverage-shortlist-max-targets`
- 以及 `pair_contact`、`pair_gap`、`drc_component` 等旧 seed 路线相关字段

## 输出字段

常用顶层字段包括：

- `pipeline_mode`, `seed_mode`, `seed_strategy`, `grid_step_ratio`, `grid_step_um`
- `marker_count`, `grid_seed_count`, `bucketed_seed_count`, `seed_bucket_merged_count`
- `array_seed_count`, `array_spacing_seed_count`, `long_shape_seed_count`, `residual_seed_count`
- `array_group_count`, `array_spacing_group_count`, `long_shape_count`, `residual_element_count`
- `array_spacing_weight_total`, `seed_weight_total`, `seed_type_counts`, `seed_audit`
- `exact_cluster_count`, `candidate_count`, `candidate_group_count`
- `candidate_direction_counts`, `diagonal_candidate_count`
- `selected_candidate_count`, `selected_diagonal_candidate_count`, `total_clusters`
- `result_csv_row_count`, `result_csv_release_count`, `result_csv_columns`
- `prefilter_stats`, `coverage_detail_seconds`, `coverage_debug_stats`
- `final_verification_stats`, `final_verification_detail_seconds`
- `file_list`, `config`, `result_summary`

单个 cluster 主要包含：

- `cluster_id`, `size`, `marker_ids`, `exact_cluster_ids`
- `selected_candidate_id`, `selected_shift_direction`, `selected_shift_distance_um`
- `representative_metadata`, `cover_representative_metadata`, `export_representative_metadata`
- `sample_metadata`

其中 `sample_metadata` 保留这些兼容字段：

- `marker_bbox`
- `clip_bbox`
- `seed_weight`
- `seed_type`
- `grid_ix`
- `grid_iy`
- `grid_cell_bbox`

说明：在 `geometry-driven` 版本里，`grid_cell_bbox` 只是兼容别名，语义等同于 `seed_bbox`，不再表示真实全域 grid cell。

## 内部资源表示

当前版本仍保留 packed bitmap、按需缓存和生命周期释放等内部表示策略；这些策略只改变对象表示方式，不改变这些核心语义：

- `GRID_STEP_RATIO=0.5`
- seed 规则
- shortlist recall
- prefilter 阈值
- ACC/ECC 几何匹配阈值
- final verification 逻辑

### Marker / Exact 阶段

- `MarkerRecord`、`ExactCluster`、`CandidateClip`、`LayoutIndex` 使用 `slots=True`。
- `pre_raster_cache` 和 exact bitmap cache 只缓存轻量 raster payload，不缓存完整 `MarkerRecord`。
- `expanded_bitmap` 在 marker collection 阶段尽早 packed 化。
- 采样阶段直接进入 online exact grouping，不再先收集完整 marker 列表再做一次 exact 聚合。
- 默认无 `--review-dir` 时，非 representative 成员会尽早降级成轻量常驻形态：
  - 先缓存 export representative 重排所需特征
  - 再释放 `expanded_bitmap`
  - 若后续不需要 sample 物化，再提前释放 `clip_bitmap`

### Candidate 阶段

- candidate bitmap 通过 strict digest 做 interning，避免重复 ndarray 常驻。
- 同一 exact cluster 内先做 strict duplicate 去重，避免创建完全重复的 shift candidate。
- 所有逻辑 candidate 会先压成全局 `CoverageCandidateGroup`：
  - 保留一个用于 greedy tie-break 的 `best_candidate`
  - 聚合 `origin_ids`
  - 聚合 `direction_counts`
  - 聚合 `logical_candidate_count`
- 非必要的逻辑 candidate 不再长期常驻。

### Coverage 阶段

- mega bundle 使用 4月29日已验证可完成的 `shape + fill-bin` window 顺序处理，而不是在单个超大 bundle 上长期常驻 dense 结构。
- area-density 仍只保留在 `_batch_prefilter()` 的 cheap reject 中，默认不再参与 window 分桶，避免把 mega bundle 切成过多重叠窗口。
- exact-hash coverage 仍然在完整 shape bundle 内传播，不被 fill-bin window 限制。
- shortlist 不再为整个 window 预建全量 `(group_count, 120)` 的 `signature_embeddings`。
- cheap descriptor 和 signature embedding 在 window 内按需计算；full descriptor 使用 window-local cache，并在每个 source prefilter 后释放。
- geometry cache 只保留 packed-first 形态，不长期保留二维 bool 中间结果。

### Result 流式输出

- 大样本默认 `materialized_outputs=false` 时，`_build_results()` 会把每个 `CoverageCandidateGroup` 按最终 cluster 映射写入临时 CSV spool。
- `_save_results()` 只把 CSV spool 复制到用户指定的 `.csv` 主输出路径，不再写旧主结果格式或 summary sidecar。
- 主结果列固定为 `groupID,cluster_id,center_x_um,center_y_um,clip_size`。
- `result_csv_row_count` 等于 `candidate_group_count`。

### Candidate Group Packed-at-Rest

这是当前版本新增的重点：

- `CoverageCandidateGroup` 不再长期持有二维 bool representative bitmap。
- group 常驻形态改为：
  - `packed_clip_bitmap`
  - `clip_bitmap_shape`
  - `area_px`
  - `clip_hash`
  - `origin_ids`
  - `logical_candidate_count`
  - `direction_counts`
  - `coverage`
- coverage window 内需要 bitmap 时，再按需 unpack 到局部缓存。
- window 结束后立即释放局部 unpack bitmap cache。
- final verification 阶段选中的 candidate 与临时 witness candidate 都按需恢复 bitmap，不要求长期常驻二维 bool clip。

## 关键诊断字段

### `coverage_debug_stats`

与 mega bundle 和窗口内临时对象相关的字段包括：

- `bucketed_coverage_bundle_count`
- `coverage_fill_bin_count`
- `max_fill_bin_group_count`
- `max_bucket_window_group_count`
- `bucketed_source_group_count`
- `bucketed_target_group_count`
- `geometry_cache_live_peak_count`
- `geometry_cache_release_count`
- `geometry_cache_live_after_bundle_count`
- `lazy_signature_embedding_group_count`
- `signature_embedding_live_peak_count`
- `window_bitmap_live_peak_count`
- `shortlist_max_subgroup_size`
- `pair_tracker_mode`
- `pair_tracker_disabled_bundle_count`
- `pair_tracker_row_count`

### `coverage_detail_seconds`

coverage 细分耗时里常见字段包括：

- `light_bundle_build`
- `bucket_index_build`
- `bucket_window_index`
- `bucket_window_release`
- `shortlist_index`
- `shortlist_payload_build`
- `shortlist_payload_release`
- `prefilter`
- `full_descriptor_cache`
- `geometry_cache`
- `geometry_cache_release`
- `geometry_match`

## Review 导出

默认情况下：

- `materialized_outputs = false`
- `file_list = []`
- 各 cluster 的 `sample_files` 为空
- `representative_file = null`

指定 `--review-dir` 后，脚本会物化：

- `samples/`
- `representatives/`
- `representative_files.txt`

并按 cluster 子目录组织输出，例如：

- `cluster_0001_size_0012`

## 依赖

```bash
pip install -r requirements.txt
```

当前 v1 主要依赖：

- `gdstk`
- `numpy`
- `scipy`
- `rtree`
- `hnswlib`

以及主线中使用到的图像处理相关依赖。

## 使用建议

- 大样本优先在系统终端中运行，不建议直接在 PyCharm 内置运行环境里拉完整数据。
- 如果只是做精度回归，建议先跑 50um / 100um crop。
- 如果需要 sample / representative 位图导出，再开启 `--review-dir`；默认关闭可减少额外文件输出。
