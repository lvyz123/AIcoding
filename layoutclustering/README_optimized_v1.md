# Optimized V1 Layout Clustering

`layout_clustering_optimized_v1.py` 是当前单进程 optimized v1 主线。它以 `geometry-driven seed` 替代旧的全域 uniform grid 扫窗：前半段按阵列代表点、阵列 spacing、长条路径和 residual 局部区域生成 seed；后半段继续使用 exact hash、coverage pruning、ACC/ECC 几何匹配、greedy set cover 和 final verification 生成最终 cluster。

这个版本不再暴露旧的 `pair/drc` seed strategy，也不引入 v2_lsf 的 LSF、manifest、shard、CSR/NPZ 分布式存储。它的目标是让 v1 在算法语义上更接近 v2_lsf，同时保留单脚本、单进程、易 review 的运行方式。

## 算法流程

1. 读取单个 OASIS 文件，或读取目录下的 `*.oas` 文件。
2. 可选执行 `--apply-layer-ops` / `--register-op` 指定的层间布尔操作，只保留 `result_layer` 进入聚类。
3. 在 pattern geometry 上做 `geometry-driven` 采样；`grid_step_ratio` 固定为 `0.5`，仅作为局部 marker bbox 尺寸和 anchor 量化基准，不再表示全域 grid 扫描。
4. 对规则二维阵列生成 `array_representative` seed，并补充 `array_spacing` 的 `x/y/corner` 三类间距 seed。
5. 对长条图形生成 `long_shape_path` 一维路径 seed；对剩余未分类图形生成 `residual_local_grid` seed。
6. 对 seed 做 anchor 级去重；spacing seed 使用独立 dedupe slot，并把重复次数累积到 `bucket_weight`。
7. 对窗口 bitmap 做 exact hash 聚合，得到 exact clusters。
8. 为每个 exact cluster 生成 `base + left/right/up/down` systematic shift candidates，并补少量 bounded diagonal shift。
9. coverage 阶段先构建轻量 bundle，再通过 cheap shortlist、lazy full prefilter 和 ACC/ECC 几何 gate 生成 candidate 覆盖关系。
10. 使用 greedy set cover 选择 representative candidates。
11. 对 selected candidates 做 final verification；验证失败的成员退回 singleton。
12. 组装 JSON/TXT 结果；只有指定 `--review-dir` 时才物化 sample / representative clip 文件。

## 运行方式

```bash
python layout_clustering_optimized_v1.py ./design.oas --output results.json
```

常用示例：

```bash
python layout_clustering_optimized_v1.py ./design.oas --output results.json
python layout_clustering_optimized_v1.py ./input_dir --output results.json
python layout_clustering_optimized_v1.py ./design.oas --clip-size 1.35 --geometry-match-mode ecc --output results.json
python layout_clustering_optimized_v1.py ./design.oas --review-dir review_out --output results.json
python layout_clustering_optimized_v1.py ./design.oas --format txt --output results.txt
python layout_clustering_optimized_v1.py ./design.oas --apply-layer-ops --register-op 2413/0 2410/0 subtract 2411/0 --output results.json
```

## 关键参数

- `input_path`: 输入 OAS 文件或目录；目录输入默认扫描当前目录下的 `*.oas`，不递归。
- `--output`, `-o`: 输出文件路径，默认 `clustering_optimized_v1_results.json`。
- `--format`, `-f`: 输出格式，`json` 或 `txt`，默认 `json`。
- `--clip-size`: clip 窗口边长，单位 um，默认 `1.35`。
- `--geometry-match-mode`: 最终几何 gate，`acc` 或 `ecc`，默认 `ecc`。
- `--area-match-ratio`: `acc` 模式的面积匹配阈值，默认 `0.96`。
- `--edge-tolerance-um`: `ecc` 模式的边界容差，默认 `0.02um`。
- `--pixel-size-nm`: 栅格化像素尺寸，默认 `10nm`。
- `--review-dir`: 导出 review 目录；指定后会物化 `samples/` 和 `representatives/`。
- `--export-cluster-review-dir`: `--review-dir` 的兼容别名。
- `--apply-layer-ops`: 启用层操作预处理。
- `--register-op SOURCE_LAYER TARGET_LAYER OPERATION RESULT_LAYER`: 注册层操作规则，例如 `--register-op 1/0 2/0 subtract 10/0`。

当前 v1 不再提供这些旧入口：`--seed-strategy`、`--marker-layer`、`--hotspot-layer`，以及 `pair_contact`、`pair_gap`、`drc_component` 等旧 seed 路线相关字段。

## 输出字段

JSON 输出包含聚类结果、样本元数据、seed 审计、coverage 诊断、final verification 统计和运行配置。常用顶层字段包括：

- `pipeline_mode`, `seed_mode`, `seed_strategy`, `grid_step_ratio`, `grid_step_um`
- `marker_count`, `grid_seed_count`, `bucketed_seed_count`, `seed_bucket_merged_count`
- `array_seed_count`, `array_spacing_seed_count`, `long_shape_seed_count`, `residual_seed_count`
- `array_group_count`, `array_spacing_group_count`, `long_shape_count`, `residual_element_count`
- `array_spacing_weight_total`, `seed_weight_total`, `seed_type_counts`, `seed_audit`
- `exact_cluster_count`, `candidate_count`, `candidate_direction_counts`, `diagonal_candidate_count`
- `selected_candidate_count`, `selected_diagonal_candidate_count`, `total_clusters`
- `prefilter_stats`, `coverage_detail_seconds`, `coverage_debug_stats`
- `final_verification_stats`, `final_verification_detail_seconds`
- `memory_debug`, `clusters`, `file_metadata`, `file_list`, `config`, `result_summary`

单个 cluster 主要包含：

- `cluster_id`, `size`, `marker_ids`, `exact_cluster_ids`
- `selected_candidate_id`, `selected_shift_direction`, `selected_shift_distance_um`
- `representative_metadata`, `cover_representative_metadata`, `export_representative_metadata`
- `sample_metadata`: 每个 sample 的 `marker_bbox`、`clip_bbox`、`seed_weight`、`seed_type`、`grid_ix`、`grid_iy`、`grid_cell_bbox` 等

`grid_cell_bbox` 在 geometry-driven 版本中只是兼容字段，语义等同于 `seed_bbox`，不再表示真实全域 grid cell。

## 12G 内存优化

当前版本面向 12G 笔记本做了精度保持型内存优化。原则是只改变表示和计算组织方式，不改变 `GRID_STEP_RATIO=0.5`、seed 规则、shortlist 上限、prefilter 阈值、ECC 容差或 final verification 语义。

已落地的生命周期优化包括：

- `MarkerRecord`、`ExactCluster`、`CandidateClip`、`LayoutIndex` 使用 `slots=True`。
- marker collection 的 `pre_raster_cache` / exact bitmap cache 只保存轻量 raster payload，不缓存完整 `MarkerRecord`。
- `expanded_bitmap` 在 marker collection 阶段 packed 化；candidate generation 只在处理单个 exact cluster 时临时 unpack。
- 非 representative marker 在 exact cluster 后释放 expanded / clip payload；导出代表重排所需分数提前缓存。
- candidate bitmap 使用 interning，共享完全相同的 ndarray。
- coverage 内部优先使用有序 `int32` 数组，不长期保留大规模 Python `set`。
- shortlist 使用 compact numpy matrix 保存 cheap invariants / signature embedding，并按 subgroup 懒构建 payload。
- geometry cache 只保留 packed 形态，target chunk 和 source cache 用完即释放。
- set cover 后释放未选中 candidate bitmap/cache，并丢弃 `all_candidates` 大列表引用。
- 大样本默认走流式 JSON 写出，避免 `file_metadata` 和 `clusters` 在尾部阶段再次形成大峰值。

针对 mega bundle 的新增优化包括：

- strict bitmap key 不再把 packed bytes 直接作为 dict key，而是使用 `shape + blake2b-128 digest` 作为主 key；同 digest 命中时仍逐像素比较 representative bitmap，真实 collision 不会误合并。
- candidate bitmap interning 和 coverage bundle grouping 都复用上述 digest key，减少 Python dict key 常驻内存。
- 当单个 shape bundle 的 group 数超过 `COVERAGE_BUCKETED_GROUP_THRESHOLD` 时，coverage 自动切换到 `shape + fill-bin` 顺序处理。
- fill-bin 邻域半径由 `CHEAP_FILL_ABS_LIMIT` 和 `COVERAGE_FILL_BIN_WIDTH` 推导，范围比 cheap fill gate 更宽，避免因为分桶漏掉原本可能通过 prefilter 的 pair。
- exact hash 直通仍在完整 shape bundle 内全局传播，不受 fill-bin window 限制，也不会触发 geometry cache。
- 同一个 exact cluster 内的 shift proposal 会先做 strict bitmap 去重，只保留 base / 轴向 / diagonal 优先级更高的候选，避免创建完全重复的 `CandidateClip` 对象。

`memory_debug` 诊断字段包括：

- `rss_collect_markers_mb`, `rss_exact_cluster_mb`, `rss_candidate_generation_mb`
- `rss_coverage_eval_mb`, `rss_set_cover_mb`, `rss_result_build_mb`, `rss_peak_estimate_mb`
- `released_marker_expanded_count`, `released_marker_clip_count`, `released_candidate_clip_count`
- `released_cache_owner_count`, `pre_raster_payload_cache_count`, `exact_bitmap_payload_cache_count`
- `packed_marker_expanded_count`, `unpacked_marker_expanded_count`, `packed_marker_clip_count`
- `candidate_bitmap_pool_unique_count`, `candidate_bitmap_pool_hit_count`, `released_candidate_list_ref_count`
- `strict_digest_key_count`, `strict_digest_collision_count`, `strict_key_bytes_avoided_estimate_mb`
- `early_duplicate_shift_candidate_count`

`coverage_debug_stats` / `coverage_detail_seconds` 中与 mega bundle 相关的字段包括：

- `bucketed_coverage_bundle_count`
- `coverage_fill_bin_count`
- `max_fill_bin_group_count`
- `max_bucket_window_group_count`
- `bucketed_source_group_count`
- `bucketed_target_group_count`
- `geometry_cache_live_peak_count`
- `geometry_cache_release_count`
- `geometry_cache_live_after_bundle_count`
- `pair_tracker_mode`
- `pair_tracker_disabled_bundle_count`
- `pair_tracker_row_count`
- `bucket_index_build`
- `bucket_window_index`
- `bucket_window_release`
- `geometry_cache_release`

说明：

- 如果运行环境无法读取进程 RSS，`rss_*` 字段可能为 `null`。
- 即使 RSS 不可用，释放计数仍会输出，可用于判断阶段释放是否发生。
- 大样本更推荐在系统终端运行，而不是在 PyCharm 内置运行环境中直接拉全量样本。

## Review 导出

默认情况下，v1 只返回结果 JSON，不生成 `samples/` 或 `representatives/` 位图文件，此时：

- `materialized_outputs = false`
- `file_list = []`
- 各 cluster 的 `sample_files` 为空，`representative_file` 为 `null`

指定 `--review-dir` 后，脚本会先在临时目录中物化 sample / representative clip，再复制到 review 目录：

- 每个 cluster 一个子目录，例如 `cluster_0001_size_0012`
- 选中的 representative 命名为 `REP__selected__...`
- 成员 sample 命名为 `sample__0000__...`
- 根目录额外生成 `representative_files.txt`

## 依赖

```bash
pip install -r requirements.txt
```

当前 v1 主要依赖 `gdstk`、`rtree`、`numpy`、`scipy`、`hnswlib`，以及共享主线中使用的位图/图像处理依赖。
