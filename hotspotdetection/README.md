# CD-SEM Hotspot Care-Area Recipe Prototype

`recipe_site_selector.py` 是当前 `hotspotdetection` 新任务的主入口，用于构建一版可跑通全流程的 CD-SEM hotspot recipe site selector。

当前版本已经从早期的 `marker -> MP candidate -> recipe site` prototype，升级为 Hu 2020 风格的 seeded care-area 主线：

```text
OAS/OASIS + hotspot marker layer + behavior manifest
-> no-train representative marker selection
-> 从 representative marker 邻域提取 seed care-area families
-> 在 OAS 内搜索同类 care-area instances
-> 对每个 care-area instance 生成 top-K MP candidates
-> candidate-level evidence/context audit + Casati-style objective subset selection
-> AF/AP candidate construction
-> recipe_sites.csv/json + recipe_review
```

本目录中的代码是新预研任务的独立方案。它复用了早期 pattern grouping 任务中的部分 layout rasterize、handcrafted feature 和 coverage selection 能力，但当前语义已经切换为 `high-risk pattern family -> care area instance -> MP/AF/AP recipe site`，不要把它理解成旧的重复 pattern clustering 任务。

## 当前边界

第一版明确要求：

- 输入必须包含 OAS/OASIS layout。
- 输入必须包含 hotspot marker layer。
- 输入必须包含 behavior manifest，且每个有效 marker 只支持一个主图像字段 `aerial_npz`。
- `epe_npz`、`pv_npz`、`nils_npz`、`resist_npz`、`layout_npz` 已下线；manifest 中出现这些字段会直接报错。
- care-area expansion 是 seeded full-OAS search：只从 seed family 派生出的 care-area type/signature 搜索同类实例，不做任意窗口 blind scan。
- 不引入 Hu 2020 的 SEM Printed Pattern Database，不接入监督 ML risk model，不做 candidate-level behavior image recrop。
- NanoPoint 只借鉴 design-aware grouping、care-area expansion、group-aware sampling 和 audit feedback；不引入 optical inspection threshold、noise floor 或 run-hotter sensitivity 语义。
- 不做 layout-only fallback；没有 verified semantic MP candidate 的 representative marker 会被拒绝为 `no_care_area_family`。
- AP 唯一性当前仍是 fingerprint nearest-neighbor proxy，不是完整 search FOV template matching。

## 主要文件

- `recipe_site_selector.py`：当前全流程 CLI 主入口，负责串联 backend、care-area expansion、MP pool selection、AF/AP 构造和 recipe 输出。
- `care_area_generator.py`：Hu-style DDD-lite / care-area expansion 层，负责 seed family 提取、全局 look-alike instance 搜索和 homogeneous group summary。
- `metrology_context.py`：NanoPoint-inspired 但 CD-SEM 语境化的量测优先级、recipe slot 浪费风险和 context audit 汇总。
- `ring_context.py`：Zhang 2016 风格的同心环上下文审查特征，只读取已有 bitmap，不重新 rasterize，不参与当前评分。
- `pattern_memory.py`：Zheng 2025 风格的轻量 pattern outcome export / accumulation，把本次 run 的 compact evidence 写到磁盘，并维护跨 run 的审查型 memory store。
- `subset_objective_selection.py`：Casati-style objective subset selection 层，负责从全局 MP pool 中按多目标边际收益选择 recipe subset。
- `mp_candidate_generator.py`：care-area instance 内的 MP discovery，负责 fragment-aware geometry anchors、MP scoring、NMS 和 MP verification。
- `hotspot_recipe_notrain_backend.py`：无训练 representative marker selection 后端，负责 handcrafted FV、coverage selection 和 behavior verification。
- `layout_utils.py`：OAS 读取、marker rasterize、任意中心点窗口 rasterize 和 review OAS 物化。
- `feature_extractor_handcraft.py`：handcrafted feature vector 生成。
- `preprocess_behavior_inputs.py`：把预裁剪 CD-SEM 图像整理成 backend 可读的 `behavior.jsonl` 和 `aerial_npz/`。
- `test_recipe_site_selector.py`：recipe prototype 的主回归测试。
- `test_handcraft_features.py`：handcrafted FV 和 no-train backend 相关测试。
- `test_preprocess_behavior_inputs.py`：behavior manifest 预处理测试。

保留的 `feature_extractor_autoencoder.py` 目前只作为后续 AE/AI detector 方向参考，不接入当前主线。

## 算法流程

### 1. Representative Marker Selection

主脚本先调用 `hotspot_recipe_notrain_backend.py`：

- 读取 hotspot markers 和 behavior manifest。
- 对 marker window 生成 handcrafted feature vector。
- 执行 exact hash 去重。
- 执行 ANN top-K graph、weighted facility location 和 k-center 补洞。
- 只使用主图像做 behavior final verification。
- 输出 representative marker、cluster provenance 和 backend summary。

这一步的目标是从已有 marker pool 中选择高风险 pattern family 的代表点，不负责在 OAS 内扩展同类 care-area instances。

### 2. Seed Care-Area Family Extraction

`care_area_generator.py` 会对每个 representative marker 调用 `discover_mp_candidates(...)`，只用 verified semantic candidates 生成 seed family。

当前 care-area type 固定为：

- `spacing`
- `line_end`
- `corner_jog`
- `density_transition`

family signature 由以下信息组成：

- `care_area_type`
- fragment corner / line-end / facing-pair metrics
- MP bitmap fingerprint
- quantized gap、line-end exposure、corner context、density transition signal

如果某个 representative marker 周边没有 verified semantic candidate，则不生成虚假 family，该 marker 会进入 rejected provenance，reason 为 `no_care_area_family`。

### 3. Look-Alike Care-Area Expansion

对每个 OAS，`care_area_generator.py` 会构建一次全局 anchor table。大版图触发 cap 时，当前使用 deterministic tile-balanced sampling，而不是简单取 OAS 前若干图元；同一个 tile 内会优先保留靠近 seed care-area 的图元，用于降低 cap 对 seed 周边弱点的采样损失，但不会用 seed 距离硬截断远处 look-alike instance：

- fragment corner anchors
- fragment line-end anchors
- fragment facing-pair / narrow-space anchors
- bbox pair anchors
- density transition anchors

为了控制 Hu-style expansion 后的计算量，当前只对 seed instance 保留完整 top-K MP discovery；非 seed expanded instance 默认使用已通过 care-area match 的 instance center/window 生成轻量 rank-0 MP candidate。这个候选会继承 seed family 的 verified semantic context，但仍需通过轻量 density 检查，避免把几乎空白或几乎全满的窗口直接标为 verified；review 中会标记 `care_area_lightweight_instance=true`。lightweight MP 的 metrology context 直接复用 care-area instance context，避免对同一窗口重复计算，并保持 expanded rarity / localization 语义一致。全局 MP pool 仍保持轻量；只有最终被 budget selection 选中的 expanded lightweight MP，会在 site 构造前自动执行一次 post-selection full MP discovery refine，用 refined MP 替换 rank-0 MP。如果 refine 没有得到 verified MP，该 site 会转为 `recipe_status=rejected`，`reject_reason=post_selection_refine_failed`，不会继续进入 AF/AP 的“假成功”路径。这样避免对所有 expanded instances 重跑 fragment proposal，同时提升最终 recipe site 的 MP core 定位精度。

每个 anchor 会 rasterize 一个 MP-size window，并与 seed family 匹配：

```text
care_area_match_score =
  0.50 * bitmap_similarity
+ 0.35 * fragment_signature_similarity
+ 0.15 * anchor_type_match
```

当前接受阈值为 `0.78`。spacing、line-end、corner/jog 还需要通过 bitmap/signature hard gate；density-transition 使用相对宽松的 density-aware gate。若 signature 信息足够且明显不匹配，会在 rasterize 前直接拒绝，减少无效窗口切图；若 bbox-only anchor 的 fragment signature 过于稀疏，会改用更保守的 bitmap + anchor type gate，避免因缺少 polygon fragment metrics 直接误拒高相似 instance。

在真正 rasterize 前，care-area expansion 会先对同 type anchor 做 cheap pre-score，分数只使用 anchor type match、fragment signature similarity 和 source strength。实例化集合以高分 anchor 为主，同时保留少量 tile-balanced 探索样本，降低 cheap score 排序误伤远处 look-alike 的风险。这个步骤的作用是控制需要切图实例化的 anchor 数量，并把 top/tile/fallback 来源数量、match count、match rate、final instance count、reject reason 和 early-stop 信息写入 `anchor_table_audit`；它不是 seed-distance hard prune，因此远处 look-alike instance 仍然可以通过后续 bitmap/signature match 入选。每个 family 最多保留 `--max-care-area-instances-per-family` 个 instances，默认 `80`。相近且高相似的 instance 会做 NMS，避免同一个 weak geometry 被重复展开。

### 4. Homogeneous Care-Area Groups

每个 family 会输出一个 homogeneous care-area group：

- `care_area_family_id`
- `care_area_type`
- seed marker provenance
- accepted instance count
- match score distribution
- homogeneity score
- expansion confidence
- reject reason distribution

同一 backend cluster 内的重复 seed family 会在 expansion 前合并。合并时保留 `merged_seed_family_ids`、`merged_cluster_ids` 和 `merged_behavior_risk_values`，并使用合并对象中的最大 behavior risk 作为下游继承风险，避免高风险 seed 被低风险重复 family 覆盖。

`homogeneity_score` 使用 accepted instances 的 match score 均值和低分位组合；当前不做复杂自动 split，避免把第一版方案做成难以审查的 rule system。若 family 只有 seed instance，`homogeneity_score` 仍保留为 provenance 分数，但 `care_area_expansion_confidence=0`，表示没有远处同类扩展证据。

seed instance 永远保留为 group member，用作 provenance anchor。

### 5. Care-Area MP Discovery

对每个 accepted care-area instance，`mp_candidate_generator.py` 会在 instance center 周围搜索 top-K MP candidates：

- marker / instance center baseline
- local grid probes
- fragment corner anchors
- fragment line-end anchors
- fragment facing-pair anchors
- critical spacing anchors
- density transition anchors

MP 分数为：

```text
mp_hotspot_score =
  0.30 * critical_geometry_score
+ 0.20 * inherited_behavior_risk
+ 0.20 * known_marker_similarity
+ 0.15 * layout_complexity
+ 0.10 * pattern_rarity
+ 0.05 * voting_confidence
```

其中 `inherited_behavior_risk` 使用的是 instance-level effective risk：

```text
effective_behavior_risk =
  seed_behavior_risk
* care_area_match_score
* care_area_homogeneity_score
```

当前不对 expanded instance 做独立 behavior image recrop，因此非 seed instance 的 risk 会随 look-alike match 和 group homogeneity 做保守衰减，避免把 seed marker 的风险无条件外推到所有实例。seed instance 本身已有 behavior manifest 支撑，`risk_attenuation_factor=1.0`。

### 6. NanoPoint-Inspired Metrology Control

当前只借鉴 NanoPoint 的结构性思想：用 design-aware care-area group 组织候选，再在有限 recipe slot 下做代表性抽样和审查反馈。这里不使用 optical detection sensitivity、noise floor、threshold profile 等光学检测目标函数。

每个 care-area family / instance 会得到 CD-SEM 语境下的量测 context，用于 review；进入全局 MP pool 后，每个 MP candidate 会再基于自己的 bitmap、geometry components、local rarity 和 proposal voting 重新计算一份 candidate-level context，用于排序和输出：

```text
metrology_priority_score =
  0.30 * hotspot_geometry_risk
+ 0.20 * inherited_behavior_risk
+ 0.20 * family_representativeness
+ 0.15 * pattern_rarity
+ 0.15 * mp_localization_confidence
```

`metrology_priority_class` 固定分为 `high / mid / low`，`metrology_context_group_id` 固定为 `{care_area_type}__{metrology_priority_class}`。

同时计算 recipe slot 浪费风险：

```text
site_reliability_risk =
  0.25 * low_family_homogeneity
+ 0.20 * weak_mp_verification
+ 0.20 * high_repetition_or_ap_ambiguity_proxy
+ 0.15 * low_focus_structure_proxy
+ 0.10 * sparse_or_uniform_bitmap_risk
+ 0.10 * signature_sparse_penalty
```

当前 `recipe_waste_penalty = site_reliability_risk`。`metrology_priority_score` 主要用于 priority class、context group 和 audit；全局 priority 只直接使用其独立信号，例如 MP 定位置信度和 recipe waste confidence，避免与 hotspot geometry、behavior risk、rarity 重复计分。这些字段不会放宽 MP verification、AF safety、AP uniqueness 或 AP global duplicate 这些 hard gate。

### 7. Ring-Context Audit

当前版本新增 Zhang 2016 风格的 concentric ring-context audit，但只作为 review evidence，不进入 MP priority、selection gain、AF/AP gate 或 metrology scoring。

`ring_context.py` 对已有 `clip_bitmap` 计算固定半径 `[0.10, 0.20, 0.35, 0.50, 0.80, 1.20] um` 上的上下文特征：

- `ring_density_profile`
- `ring_edge_crossing_profile`
- `ring_asymmetry_profile`
- `ring_pattern_code`
- `ring_selected_radii_um`

其中 selected radii 由无监督 proxy score 加最小半径间距 DP 得到，用于后续人工判断哪些半径值得进入真实 outcome-driven 特征选择。本轮不会使用 hotspot/non-hotspot label，也不会把 ring feature 写入 CSV，避免在缺少可靠反馈时过早改变排序。

### 8. AP/AF Matchability Audit

当前版本新增 Wang 2025 D2DB 对准思想的 layout-side 轻量迁移，但只做审查，不接真实 SEM、不做 SIFT/FLANN、不做 GAN 或 optical flow。

`matchability_audit.py` 对 AP/AF 已有 bitmap 计算以下字段，并写入 candidate details：

- `keypoint_count`
- `keypoint_density_score`
- `keypoint_spread`
- `orientation_entropy`
- `descriptor_margin`
- `periodicity_penalty`
- `layout_matchability_score`

AP 的 `descriptor_margin` 来自现有 template peak margin，periodicity 同时参考 nearest similarity 和 peak count；AF 没有模板峰，`descriptor_margin` 固定为 0.5，并结合 focus quality 生成 matchability audit。上述字段只用于解释 AP/AF 是否可能在真实 CD-SEM/D2DB 执行中稳定匹配，不会放宽或收紧现有 AF/AP hard gate。

### 9. Pattern Memory Accumulation

当前版本会把每次 run 的 `recipe_review/pattern_memory_export/` 追加到默认持久化目录 `hotspotdetection/pattern_memory_store/`。这个目录是跨 run 的磁盘知识库，不是运行时全量内存数据库，已加入 `.gitignore`。

store 文件包括：

- `records.jsonl`：compact provenance、metrology context、ring-context 和 recipe outcome。
- `vectors.npz`：与 records 对齐的 bitmap fingerprint + ring-context float32 向量。
- `manifest.json`：record count、vector shape、增量写入和去重统计。
- `memory_audit.json`：按 `care_area_type`、`metrology_context_group_id`、`mp_candidate_type` 统计 selected、AF/AP fail、AP duplicate 和 smoothed audit prior。
- `ring_outcome_audit.json`：统计每个 ring radius 与 recipe outcome proxy 的关系，并用 DP 给出 outcome-aware selected radii。

主流程还会在 append 当前 run 之前，对当前 MP pool 执行只读 nearest-neighbor prior audit：用当前 compact vector 查询历史 store，输出 `memory_neighbor_count`、`memory_nearest_similarity`、`memory_recipe_success_prior`、`memory_af_success_prior`、`memory_ap_success_prior`、`memory_ap_duplicate_prior` 和 `memory_waste_prior`。store 为空或无近邻时全部使用中性 prior。上述 prior 和 radius evidence 当前只用于 review，不会回写 `mp_priority_score`、`mp_selection_gain`、AF/AP gate，也不会做 retrieval-based hotspot 判定。

### 10. Global MP Pool Selection

所有 care-area instances 的 top-K MP candidates 会进入全局 pool。候选仍会先计算 `mp_priority_score`，但它现在只是候选级价值信号和输出字段，不再直接主导全局选择：

```text
mp_priority_score =
  0.30 * mp_hotspot_score
+ 0.20 * effective_behavior_risk
+ 0.15 * pattern_novelty
+ 0.15 * cluster_coverage
+ 0.10 * mp_localization_confidence
+ 0.10 * (1 - recipe_waste_penalty)
```

如果所有 behavior risk 都为 0，则 behavior risk 权重会按比例分配给其余项。

如果某个候选的 `recipe_waste_penalty > 0.40`，当前会按 waste 连续降低其 `mp_priority_score`，但 `metrology_priority_score` 越高，降权越温和；最低可降到 0.4 倍，不直接拒绝。这样可以让高浪费风险、低量测价值的候选自然排到后面，同时保留 review 证据和极端小 pool 下的兜底可选性。

全局选择已经升级为 Casati-style objective subset selection。selector 会先从当前 pool 自动推导目标 bins：`care_area_type`、`care_area_family_id`、`metrology_context_group_id`、`pattern_taxonomy_class`、`risk_bin` 和 `feasibility_bin`。每个 bin 的 target count 由该 bin 内候选的 `objective_candidate_value` 加权占比决定，不写死业务比例。`htc_like` 不生成正向 taxonomy target，只通过浪费惩罚影响边际收益。

候选级 objective value 为：

```text
objective_risk_score =
  0.35 * mp_hotspot_score
+ 0.20 * effective_behavior_risk
+ 0.20 * defect_evidence_proxy_score
+ 0.15 * metrology_priority_score
+ 0.10 * pattern_novelty

objective_feasibility_score =
  0.70 * expected_recipe_feasibility_proxy
+ 0.30 * (1 - recipe_waste_penalty)

objective_candidate_value =
  objective_risk_score * (0.50 + 0.50 * objective_feasibility_score)
```

每一轮选择最大多目标边际收益：

```text
mp_selection_gain =
  0.24 * risk_coverage_gain
+ 0.16 * family_coverage_gain
+ 0.14 * context_coverage_gain
+ 0.12 * taxonomy_balance_gain
+ 0.14 * recipe_feasibility_gain
+ 0.10 * spatial_diversity_gain
+ 0.10 * priority_anchor_gain
- 0.08 * htc_waste_penalty
```

每个 bin 使用 `1 / sqrt(1 + selected_count_in_bin)` 的递减收益；若已达到 target count，该 bin gain 继续乘以 `0.30`，但不做硬拒绝。空间近邻且 bitmap/fingerprint 高相似的候选仍会被标记为 `mp_pool_duplicate`。未入选但非 duplicate 的候选标记为 `mp_pool_over_budget`。

`mp_selection_gain` 字段名保持不变以兼容 CSV/JSON，但语义已经变成 `subset_objective_marginal_gain`。`recipe_review/subset_objective_audit.json` 会记录 target distribution、selected distribution、coverage gaps、selected marginal gain trace、high-value missed candidates 和 high-risk non-executable candidates。

只有最终仍为 `recipe_status=selected` 的 site 才会被视为有效覆盖。若某个 MP 在 post-selection refine、AF/AP 或 AP global duplicate 阶段失败，它不会再让同 cluster 的其它 marker 被标记成 `covered_by_representative`；这些 marker 会按未覆盖预算语义进入 rejected provenance。

在 rarity 打分前会先执行一次轻量 MP pool pre-dedup，重复候选标记为 `mp_pool_preduplicate`，不再参与 global rarity 计算。重复判断不再使用单纯 `max(shifted_iou, fingerprint_similarity)`，而是要求 shifted bitmap 已经足够接近，或 shifted 接近且 fingerprint 同时接近，避免把“粗特征相同但空间形态不同”的候选误杀。selection 阶段的 duplicate suppression 使用更高阈值，只抑制几乎相同的候选，避免把相似但仍可能贡献 coverage 的 care-area instance 提前删掉。

### 11. AF/AP Construction

对每个 selected MP 搜索 AF candidate：

```text
af_score =
  0.55 * layout_similarity_to_mp
+ 0.30 * focus_quality
+ 0.15 * distance_score
- 0.05 * hotspot_core_risk
```

AF 要求与 MP core 保持最小距离，并在可选 `--sem-image-shift-limit-um` 范围内。`hotspot_core_risk` 同时作为 score penalty 和 hard gate；当候选过于接近 MP hotspot core 时会标记 `too_hotspot_like`，不再作为安全 AF。

对每个 selected MP 搜索 AP candidate：

```text
ap_score =
  0.50 * uniqueness_score
+ 0.20 * entropy_score
+ 0.20 * corner_density_score
+ 0.10 * distance_score
```

AP candidate 需要通过 density、entropy、edge/corner richness gate。最终 selected rows 还会做全局 AP duplicate check。

### 12. Review Evidence / Taxonomy Audit

这一层只做审查，不改变 `mp_priority_score`、`mp_selection_gain`、AF/AP hard gate 或
CSV 主契约。输出位置为 `recipe_review/review_evidence_audit.json`，并在
`mp_candidate_pool.json` 与每个 selected site 的 `site_summary.json` 中保留 candidate 级字段。

- `graph_context_audit`：从已切出的 MP bitmap 中提取 graph-lite 连通块、近邻关系、环境复杂度、
  pool 内 graph rarity 和同 family graph support。
- `evidence_contradiction_audit`：拆分 geometry、behavior、care-area、voting、graph、ring、
  memory evidence，并标记高 priority 低 evidence、强 evidence 但 AF/AP/refine 失败等矛盾。
- `pattern_taxonomy_audit`：给出 `tnsb_like`、`htc_like`、`known_like`、`ambiguous` 审查标签，
  帮助人工区分新颖 weak pattern 与高重复低价值候选。
- `expected_feasibility_audit`：估计 AF/AP/recipe 可执行性 proxy；当前只用于 review，不做
  backfill，不参与排序。

## 运行方式

最小示例：

```bash
python recipe_site_selector.py input.oas \
  --marker-layer 999/0 \
  --behavior-manifest behavior_inputs \
  --output-dir recipe_out \
  --max-sites 100
```

Windows PowerShell 示例：

```powershell
Set-Location -LiteralPath D:\AIcoding\hotspotdetection
python .\recipe_site_selector.py D:\path\to\input.oas `
  --marker-layer 999/0 `
  --behavior-manifest D:\path\to\behavior_inputs `
  --output-dir D:\path\to\recipe_out `
  --max-sites 100
```

`clip_for_lyu` 真实 smoke 建议命令：

```powershell
Set-Location -LiteralPath D:\AIcoding
python -B hotspotdetection\recipe_site_selector.py hotspotdetection\clip_for_lyu.oas `
  --marker-layer 12530/2 `
  --behavior-manifest hotspotdetection\semsim_v11_smoke_clip_for_lyu\behavior.jsonl `
  --output-dir hotspotdetection\_temp_runs\clip_for_lyu_recipe_selector_optimized `
  --max-sites 20 `
  --skip-pattern-memory-store-append
```

这个 smoke run 只追加本次 `recipe_review/pattern_memory_export/`，不会更新持久
`hotspotdetection/pattern_memory_store/`，适合调试和性能回归。

如果需要先把预裁剪 CD-SEM 图像整理成 manifest：

```powershell
python .\preprocess_behavior_inputs.py D:\path\to\input.oas `
  --marker-layer 999/0 `
  --aerial-dir D:\path\to\aerial_images `
  --output-dir D:\path\to\behavior_inputs
```

随后把 `behavior_inputs` 目录传给 `--behavior-manifest`。

## 关键参数

- `input_path`：输入 OAS/OASIS 文件或目录。
- `--marker-layer`：hotspot marker 层，格式为 `layer/datatype`，例如 `999/0`。
- `--behavior-manifest`：`behavior.jsonl` 路径，或 `preprocess_behavior_inputs.py` 的输出目录。
- `--output-dir`：输出目录，写入 CSV、JSON、backend summary 和 review 文件。
- `--clip-size`：MP/AF/AP clip 边长，单位 um，默认 `1.35`。
- `--mp-template-size-um`：MP core template 边长；不提供时使用 `--clip-size`。
- `--af-template-size-um`：AF template 边长；不提供时使用 `--clip-size`。
- `--ap-template-size-um`：AP template 边长；不提供时使用 `--clip-size`。
- `--max-sites`：最多进入 AF/AP 构造的 selected MP 数量，默认 `100`。
- `--mp-coverage-target`：no-train backend 的 representative coverage target，默认 `0.985`。
- `--mp-search-radius-um`：提取 seed family 和 care-area instance 内 MP candidate 的搜索半径，默认 `0.8`。
- `--mp-candidates-per-marker`：每个 care-area instance 进入全局池的 top-K MP candidate 数量，默认 `5`。
- `--max-care-area-instances-per-family`：每个 seed family 最多保留的同类 care-area instances，默认 `80`。
- `--min-feature-um`：可选工艺最小特征尺寸，用于约束 critical spacing gap。
- `--af-search-radius-um`：AF 搜索半径，默认 `3.0`。
- `--sem-image-shift-limit-um`：可选 SEM image-shift 可达半径；设置后 AF candidate 超出该距离会被硬过滤。
- `--ap-search-radius-um`：AP 搜索半径，默认 `5.0`。
- `--candidate-step-um`：MP/AF/AP candidate 滑窗步长，默认 `0.2`。
- `--min-site-distance-um`：AF/AP 与 MP core 的最小距离，默认 `0.5`。
- `--recursive-input`：当 `input_path` 是目录时递归搜索 `.oas` 文件。
- `--apply-layer-ops`：启用 boolean layer operations。
- `--register-op SOURCE_LAYER TARGET_LAYER OPERATION RESULT_LAYER`：注册层操作规则，例如 `--register-op 1/0 2/0 subtract 10/0`。使用该参数时会自动启用 layer operations。
- `--skip-pattern-memory-store-append`：跳过持久 `pattern_memory_store` append，只保留本次 run 的 export artifact。

## Behavior Manifest

`behavior.jsonl` 每行描述一个 marker 样本。当前最小字段如下：

```json
{
  "sample_id": "unit__marker_000000",
  "source_path": "unit.oas",
  "marker_id": "unit__marker_000000",
  "clip_bbox": [-0.675, -0.675, 0.675, 0.675],
  "aerial_npz": "aerial_npz/unit__marker_000000.npz",
  "risk_score": 1.0
}
```

说明：

- `aerial_npz` 必填，NPZ 内默认读取 `image` key；当前该字段承载预裁剪 CD-SEM 图像。
- `risk_score` 可为 0；如果所有 marker risk 都为 0，评分会自动进行权重重分配。
- 不再支持 `epe_npz`、`pv_npz`、`nils_npz`、`resist_npz`、`layout_npz`。
- manifest 可以直接传入 `behavior.jsonl` 文件，也可以传入包含 `behavior.jsonl` 的目录。

## 输出文件

`--output-dir` 下会生成：

- `recipe_sites.csv`：供人工 review / 下游 recipe 构造使用的扁平表。
- `recipe_sites.json`：结构化主输出，包含 config、summary、backend summary、care-area groups、compact MP pool 和 site details。
- `_notrain_backend.json`：no-train backend 的完整中间结果。
- `recipe_review/care_area_groups.json`：seed family、expanded instances、homogeneity 和 rejected seed provenance。
- `recipe_review/mp_candidate_pool.json`：全局 MP candidate pool 审查信息，包含每个 MP candidate 的 ring-context audit、objective components、target bins 和 selection status。
- `recipe_review/source_marker_candidate_index.json`：按 source marker 汇总候选总数、状态分布和 top compact refs，用于替代每个 site summary 中重复嵌入的大列表。
- `recipe_review/metrology_context_audit.json`：按量测优先级、care-area type 和 metrology context group 汇总 recipe slot 使用情况、AF/AP 失败和主要 reject reason。
- `recipe_review/subset_objective_audit.json`：按 Casati-style objective subset selection 汇总 target distribution、coverage gap、marginal gain trace 和 high-risk non-executable candidates。
- `recipe_review/pattern_memory_export/records.jsonl`：本次 run 的 compact pattern outcome records。
- `recipe_review/pattern_memory_export/vectors.npz`：与 records 对齐的 bitmap fingerprint + ring-context float32 向量。
- `recipe_review/site_XXXX/`：每个 selected site 的 review 子目录。

另外会生成 `recipe_review/subset_objective_audit.json`，用于审查 Casati-style objective subset selection 的 target distribution、coverage gap、selected marginal gain trace 和高价值但未执行/不可执行的候选。
此外，默认每次 run 会更新 `hotspotdetection/pattern_memory_store/` 中的 `records.jsonl`、`vectors.npz`、`manifest.json`、`memory_audit.json` 和 `ring_outcome_audit.json`。该目录用于跨 run 积累，不放在 `--output-dir` 下，避免一次性 review 目录被当作长期知识库；调试或 smoke run 可以使用 `--skip-pattern-memory-store-append` 跳过持久 store 写入。

`recipe_sites.json` 的 summary 还会记录 selected expanded MP refine 的聚合计数：`selected_expanded_mp_refine_attempted_count`、`selected_expanded_mp_refined_count`、`selected_expanded_mp_refine_failed_count`，metrology context 覆盖统计：`metrology_context_group_count`、`selected_metrology_context_group_count`、`selected_by_metrology_priority_class`、`selected_by_metrology_context_group`，pattern memory export 的 `pattern_memory_record_count`、`pattern_memory_vector_shape`、`pattern_memory_estimated_disk_bytes`，以及 pattern memory store 的 `pattern_memory_store_record_count`、`pattern_memory_store_added_record_count`、`pattern_memory_store_duplicate_skipped_count` 和 `pattern_memory_store_append_skipped`。单个 site 的 `mp_discovery_components_json` / `site_summary.json` 中会记录 `post_selection_refine`、`pre_refine_mp_x_um`、`pre_refine_mp_y_um` 和 `refine_shift_um`；refine 失败时还会记录 `post_selection_refine_failed` 和失败前的 pool 状态。refine 成功后，`mp_priority_score` 仍表示 selection 时的 pre-refine priority，review 中会用 `post_refine_priority_stale` 标记这一点，避免把 post-refine context 与 pre-refine 排序解释混淆。

### CSV 关键字段

当前 `recipe_sites.csv` 固定包含以下几类字段：

- site 状态：`site_id`、`recipe_status`、`reject_reason`
- seed provenance：`source_marker_id`、`hotspot_cluster_id`
- care-area provenance：`care_area_family_id`、`care_area_instance_id`、`care_area_type`、`care_area_match_score`、`care_area_homogeneity_score`、`care_area_instance_count`、`care_area_seed_marker_id`、`care_area_instance_bbox`
- metrology context：`metrology_priority_score`、`metrology_priority_class`、`site_reliability_risk`、`recipe_waste_penalty`、`metrology_context_group_id`、`selection_profile_id`
- MP：`mp_candidate_id`、`mp_candidate_rank`、`mp_selection_gain`、`mp_x_um`、`mp_y_um`、`mp_candidate_type`、`mp_hotspot_score`、`mp_priority_score`
- AF：`af_x_um`、`af_y_um`、`af_score`、`af_distance_um`、`af_similarity`、`af_reject_reason`、`af_acceptance_checks_json`
- AP：`ap_x_um`、`ap_y_um`、`ap_score`、`ap_uniqueness_score`、`ap_peak_count`、`ap_reject_reason`、`ap_acceptance_checks_json`、`ap_global_duplicate`
- review OAS：`mp_oas`、`af_oas`、`ap_oas`

注意：`source_marker_id` 现在表示 seed marker provenance，不代表 MP 必定位于该 marker 邻域内。

### 关键状态字段

- `recipe_status=selected`：MP/AF/AP 均通过当前规则，且没有被全局 AP duplicate 拒绝。
- `recipe_status=rejected`：该 row 保留用于 provenance 或失败 review。
- `reject_reason=no_care_area_family`：representative marker 周边没有 verified semantic candidate，无法生成 seed care-area family。
- `reject_reason=no_valid_mp`：care-area instance 内没有通过 verification 的 MP candidate。
- `reject_reason=no_safe_af`：没有找到合格 AF candidate。
- `reject_reason=no_unique_ap`：没有找到合格 AP candidate。
- `reject_reason=ap_global_duplicate`：AP 与另一个 selected site 高相似，当前 site 被全局去重拒绝。
- `reject_reason=post_selection_refine_failed`：expanded lightweight MP 被选中后，full MP discovery refine 未得到 verified MP。
- `reject_reason=covered_by_representative`：marker 已被 selected representative 覆盖。
- `reject_reason=over_budget`：超过 `--max-sites` budget。

## Review 目录

典型 review 目录如下：

```text
recipe_review/
  care_area_groups.json
  mp_candidate_pool.json
  source_marker_candidate_index.json
  metrology_context_audit.json
  subset_objective_audit.json
  pattern_memory_export/
    records.jsonl
    vectors.npz
  site_0000/
    mp.oas
    af.oas
    ap.oas
    site_summary.json
```

说明：

- `care_area_groups.json` 用于检查 seed family 是否合理、远处 look-alike instances 是否进入 group、instance reject reason 分布是否异常，以及 anchor table cap / tile coverage / source 分布是否异常。
- `mp_candidate_pool.json` 用于检查每个 care-area instance 的 MP candidate、subset objective marginal gain、recipe waste soft demotion、objective target bins、duplicate suppression、over-budget reject reason 和 ring-context audit。
- `source_marker_candidate_index.json` 用于按 source marker 快速查看候选总量、状态分布和 top compact refs，避免每个 `site_summary.json` 重复写入同源大列表。
- `pattern_memory_export/` 是本次 run 的磁盘导出，不是全量内存数据库；`records.jsonl` 保存 provenance/outcome/vector index，`vectors.npz` 保存 compact fingerprint + ring-context 向量。当前只读取历史 memory 生成中性审查 prior，不会用 retrieval prior 改变评分。
- `mp.oas` 总会在 selected MP site 目录中写出。
- `af.oas` 只在 AF candidate accepted 时写出。
- `ap.oas` 只在 AP candidate accepted 时写出。
- `site_summary.json` 在全局 AP duplicate check 之后写出，因此其状态与最终 CSV/JSON 保持一致。
- AF/AP 的详细失败原因会写入 `site_summary.json` 的 candidate `reject_reason` 和 `acceptance_checks`，例如 `low_similarity`、`low_focus_quality`、`too_hotspot_like`、`low_uniqueness`、`too_many_peaks`、`density_out_of_range`、`low_entropy`、`low_edge_corner`。CSV 的主 `reject_reason` 仍保留 `no_safe_af` / `no_unique_ap` 这类稳定上层状态，同时通过 `af_reject_reason` / `ap_reject_reason` 支持批量统计。
- 未入选的 MP candidates 不单独物化 OAS，完整审查信息在 `mp_candidate_pool.json` 中。

## 依赖

当前主要依赖：

- `gdstk`
- `numpy`
- `scikit-image`
- `hnswlib`
- `Pillow`
- `tifffile`
- `ncempy`，仅在读取 `dm3/dm4` 行为图像时需要

如缺少依赖，可按当前环境实际需求安装，例如：

```bash
pip install gdstk numpy scikit-image hnswlib pillow tifffile
```

## 测试

建议在 `hotspotdetection` 目录下运行：

```powershell
Set-Location -LiteralPath D:\AIcoding\hotspotdetection
python -B -m unittest test_recipe_site_selector -v
python -B -m unittest test_recipe_site_selector test_handcraft_features test_preprocess_behavior_inputs -v
```

使用 `python -B` 是为了避免 Windows 环境中已有 `__pycache__` 权限问题影响测试。

当前重点测试覆盖：

- seed family extraction
- look-alike care-area expansion
- homogeneity reject
- care-area weighted bitmap matching
- anchor table cap / tile coverage audit
- anchor cheap pre-score / instantiate audit
- effective selected coverage after refine / AF / AP reject
- effective behavior risk attenuation
- seed instance risk preservation
- singleton family expansion confidence
- density-transition signature metrics
- care-area MP discovery
- budget diversity
- no care-area family reject
- Top-K MP pool selection
- global budget cap
- duplicate suppression
- valid-only priority normalization
- recipe waste soft demotion
- objective subset selection diversity
- post-selection refine failure reject
- rejected marker rows context propagation
- AF non-overlap / AF hotspot-core risk marking / AF image-shift limit
- AP unique / AP periodic reject / AP global duplicate
- behavior manifest preprocess
- handcrafted FV / no-train backend regression

## 使用建议

- 第一轮 review 先用小版图或 crop 跑通输出，重点检查 `care_area_groups.json`、`mp_candidate_pool.json` 和每个 `site_summary.json`。
- 如果 `no_care_area_family` 很多，优先检查 marker 周边是否真的有 spacing、line-end、corner/jog 或 density-transition 类 semantic geometry。
- 如果 expanded instances 看起来过宽，优先检查 `care_area_match_score`、bitmap/signature 分数、`instance_reject_reasons` 和 `care_area_homogeneity_score`。
- 如果 selected site 大量 `no_safe_af`，优先检查 `--af-search-radius-um`、`--sem-image-shift-limit-um` 和版图中 MP 周边是否有相似可对焦结构；`hotspot_core_risk` 主要作为辅助 review 信号。
- 如果 selected site 大量 `no_unique_ap`，优先检查 AP 搜索半径、候选步长和局部图形是否周期重复。
- 当前版本不建议直接改成任意窗口 full-chip blind scan；care-area expansion 已经是 seeded full-OAS search。

## 后续演进方向

当前 README 只描述已落地能力。下一阶段较有价值的演进方向包括：

- candidate-level CD-SEM recrop：让 expanded care-area instance 自身拥有 behavior 证据，而不只继承 seed marker risk。
- AP template matching proxy：在 search FOV 内滑动模板，计算真实 main peak / second peak / peak count。
- SEM / measurement feedback loop：积累 Printed Pattern Database 或真实 CD-SEM review 结果后，再引入 Hu 2020 中更完整的风险模型。
- 更真实的 AF 安全性检查：继续增强 narrow-space / line-end proxy，降低当前 bitmap 级 `hotspot_core_risk` 的近似误差。
