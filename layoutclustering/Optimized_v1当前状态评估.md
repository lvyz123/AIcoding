我先给结论：**当前 `optimized_v1` 已经能在真实产品版图 `tolyu_test2.oas (550um x 550um)` 上完整跑通，而且最终结果有“保守正确”的特征，但离“高效、好用的产品级聚类结果”还有明显距离。**

这里我主要依据两份材料：
- [running_record.md](C:/Users/81932/Documents/AIcoding/layoutclustering/running_record.md)
- [clustering_results.txt](C:/Users/81932/Documents/AIcoding/layoutclustering/clustering_results.txt)

同时有个注意点：`clustering_results.txt` 被截断在 **200000 字符**，尾部停在 `cluster_sizes` 大数组中间，所以它不是完整可解析 JSON。下面我会把“强结论”和“受截断影响的判断”分开说。

**总体评价**

优点先说：
- 这版算法已经能在真实大版图上**完整跑通**，这是很重要的里程碑。
- 从日志看，mega bundle 的内存生命周期控制是有效的：
  - `coverage` 阶段 RSS 只有 `4135.645 MB`
  - `geometry_cache_live_peak_count=66`
  - `geometry_cache_live_after_bundle_count=0`
- exact 聚合也确实有价值：
  - `426838 samples -> 126174 exact clusters`
  - 说明前端窗口并不是完全噪声，确实抓到了大量重复图样。

但问题也很明显：
- 最终结果的**聚类压缩效果不强**。  
  `426838 -> 108705 final clusters`，平均 cluster size 约 `3.93`。  
  如果目标是把大量重复 pattern 收敛成少量可 review 的族，这个结果还偏散。
- final verification 的**回退比例非常高**：  
  `verified_pass=38074`，`verified_reject=88100`。  
  也就是 `126174` 个 exact clusters 里，大约 **69.8%** 最终没能稳定留在覆盖 candidate 里，被拆回 singleton。
- 换个更直观的角度：  
  `88100 / 108705 ≈ 81%` 的最终 cluster，其实是 final verification 阶段新拆出来的 singleton。  
  这说明当前流程是“**前面合得很积极，最后又拆回很多**”。

**我对当前算法效果的判断**

这版更像是一个：
- **召回优先**
- **最终正确性靠 final verification 兜底**
- **但 coverage / set cover 阶段过于乐观**

的方案。

也就是说，**精度底线大概率是安全的**，因为最后拆 singleton 很激进；  
但代价是：
- 中间算了很多最后并不成立的关系
- 最终输出里保留了太多“没真正聚起来”的 cluster
- 对真实产品版图的“模式提炼能力”还不够强

**日志里最值得警惕的几个信号**

- seed 前端仍然很重：
  - `raw geometry seed = 2,536,893`
  - `dedup seed = 426,838`
  - 其中 `residual_local_grid = 324,804`
  - `long_shape_path = 97,586`
  
  这说明真实产品里，seed 仍然主要被 residual / long-shape 驱动，array seed 占比很小。  
  几何驱动前端虽然比 uniform grid 好很多，但还没有真正把“规则重复结构”充分吃干榨尽。

- candidate 空间仍然巨大：
  - `candidate_count = 1,402,890`
  - `candidate_group_count = 914,532`
  
  即使做了 group-first，搜索空间还是非常大。

- coverage 的主要成本并不在最终 ECC 判定，而在前面的“准备动作”：
  - `prefilter = 9067s`
  - `full_descriptor_cache = 8916s`
  - `full_prefilter = 9024s`
  - `geometry_cache = 7020s`
  - `geometry_match = 410s`
  
  这个信号很关键：**真正慢的不是最后一脚几何比较，而是反复构建 descriptor / bitmap cache / morphology cache。**

- coverage 的误匹配候选很多：
  - `geometry_pass = 1,652,754`
  - `geometry_reject = 33,385,675`
  
  而最终真的通过 final verification 的 exact cluster 只有 `38,074`。  
  说明当前 coverage 阶段放进来的 pair 太多，后面大部分都被否掉了。

- 方向分布也很说明问题：
  - selected candidates 里非 `base` 的有不少
  - 但最终 `final cluster direction` 几乎全是 `base`
  
  最终方向分布：
  - `base = 108276`
  - 非 base 总共只有 `429`
  
  这说明 shift 候选在这个产品版图上**成本很高、最终净收益很低**。

**高价值、且不影响覆盖率/分类准确率的优化点**

下面这些我认为是下一轮最值得做的，且属于“表示优化 / 顺序优化 / 缓存优化”，不会改算法判定语义：

- **P0：把 candidate group 的轻量描述符做跨 window 复用**
  
  现在 `unpacked_candidate_group_bitmap_count = 10,189,074`，说明同一批 group 被反复 unpack。  
  下一步最值钱的是缓存这些更小的派生物，而不是反复还原 bitmap：
  - cheap descriptor
  - full graph descriptor
  - signature embedding
  - 行/列投影或 pooled grid
  
  这些缓存比 bitmap 小得多，也不会改变匹配语义。

- **P0：mega bundle 再做一层“不会漏 recall 的细分桶”**
  
  虽然已经有 `fill-bin`，但这次：
  - `fill bins = 26`
  - `max bucket window = 578171`
  
  还是太大了。  
  可以继续在不改变 cheap gate 语义的前提下，加一层更细的 subgroup/window 组织，比如：
  - shape + fill-bin + cheap topology bucket
  - shape + fill-bin + connected-component coarse signature
  
  前提是：**只做当前 prefilter 必然一致的分桶**，不引入新阈值裁剪。

- **P0：把“当前 full prefilter 已经会拒绝的结构差异”更早前移到 pair 生成前**
  
  现在 `full_prefilter_reject = 921,887`，但 full descriptor 是在巨量 pair 上现场构建出来的。  
  最合适的优化方向不是调阈值，而是把现有 descriptor 的某些严格不变量，提前变成 bucket key / subgroup key，减少 pair 形成。

- **P1：继续压 front-end 的 residual / long-shape seed 常数项**
  
  这次真正主导 seed 数量的是：
  - `residual_local_grid`
  - `long_shape_path`
  
  可以继续做的，是**等价 seed 的更早合并**，例如：
  - 更强的 pre-raster canonicalization
  - 更早的 exact-window 级 dedupe
  - 对 long path 上重复局部窗口做等价折叠
  
  这类优化只要坚持“只合并最终会 exact 相同的窗口”，就不会伤 recall。

- **P1：结果构建阶段继续瘦身**
  
  `rss_result_build_mb = 11171.109 MB`，又回到了接近峰值。  
  这说明尾部结果组装仍然会把大量对象重新拉回内存。  
  这不影响算法正确性，但影响产品可用性。下一步应该继续检查：
  - `clusters` 输出是否还在全量常驻
  - `sample_metadata` 是否重复拷贝过多
  - streaming JSON 是否在这个大样本上真正生效

**可以考虑，但我会放到 A/B 实验里的点**

这些不是不能做，而是它们可能改变召回或聚类语义：

- 限制 diagonal candidate
- 按 seed type 裁剪 shift
- 调大 `GRID_STEP_RATIO`
- 缩 residual seed
- 调 shortlist 上限

从这次产品版图结果看，它们**很可能有收益**，尤其是 diagonal；  
但如果你的要求是“完全不影响覆盖率和分类准确率”，这些我不会直接放进默认路径。

**最后一句实话**

如果站在“真实产品版图是否已经达到可交付效果”的角度，我的判断是：

- **能跑通**：是，而且比前几轮强很多
- **最终正确性风险高不高**：不算高，final verification 把错误合并拆回去了
- **聚类效果够不够好**：还不够，输出仍然太散，singleton 太多
- **下一轮最该攻哪里**：不是再抠 ECC，而是**coverage 前后的 pair 组织、descriptor 复用、以及 residual/long-shape 前端折叠**

如果你愿意，我下一步可以直接把上面这份评估收敛成一版 **“仅含零精度损失项”的 patch plan**。