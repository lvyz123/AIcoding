**结论**
我认同你的目标调整，而且这个调整很关键：如果最终目标是帮助 OPC 团队找到容易出问题的 hotspot，那么 cluster 的主语应该从“几何同类”改成“光刻行为同类”。但我会加一个限定：**几何不应再作为 cluster 的最终相似性定义，而应作为候选生成、解释、去重、review 定位的辅助坐标系**。

换句话说，新目标下的正确方向不是“用 AE 替换 optimized”，而是把系统升级成：

```text
layout/marker windows
-> exact geometry hash 去重
-> lithography behavior representation
-> coverage-first representative selection
-> behavior-domain final verification
-> geometry + aerial/resist review export
```

当前 [layout_clustering_optimized.py](C:/Users/81932/Documents/AIcoding/layoutclustering/layout_clustering_optimized.py) 的 hash/topology/signature/ACC/ECC 很适合“几何 verified clustering”，但它不是“光刻行为 coverage”问题的最终 SOTA。

**AutoEncoder 的价值重新评估**
在你的新目标下，AutoEncoder 的价值明显上升。之前我不建议它进入核心判定，是因为旧目标强调“几何一致”。现在如果目标是“光刻行为 coverage”，AE，尤其是基于 aerial image / resist image / process-window image 的 AE，就应该进入主流程。

但我仍不建议用 **vanilla reconstruction AE** 直接聚类。更推荐：

- **domain-aware AE**：输入 aerial image、resist contour、PV band、NILS/ILS map、slope map，多通道训练。
- **metric-learning AE**：AE latent 之后加 contrastive/triplet/supervised metric loss，让距离真正对应“光刻行为相似”。
- **multi-task encoder**：同时预测 hotspot probability、EPE/CD error、PV sensitivity、bridge/pinch/open 类型。
- **AE 做 coverage space，不做唯一 final gate**：cluster member 最后还要通过 behavior verification，例如 aerial/resist image 差异、EPE map residual、PV band overlap、critical edge signature。

Feng 2023 和 2024 对这个方向的支持很强：2023 说明 aerial image FV 同时包含几何和光学信息，并用 AE 降维做 coverage；2024 直接比较 FFT FV、basic AE FV、domain-aware improved AE FV，结论是 AE，尤其加入 lithography domain knowledge 后，对 anomaly/outlier 更有效。参考：[Feng 2023](https://pubmed.ncbi.nlm.nih.gov/36859995/)、[Feng 2024](https://www.mdpi.com/2304-6732/11/10/990)，以及本地 PDF [Feng_2023.pdf](C:/Users/81932/Documents/AIcoding/layoutclustering/Feng_2023.pdf)、[Feng_2024.pdf](C:/Users/81932/Documents/AIcoding/layoutclustering/Feng_2024.pdf)。

**目标函数也要变**
现在的 greedy set cover 是“用少数 representative 覆盖 exact clusters”。新目标应该改成：

```text
优先最大化 lithography behavior coverage
其次覆盖 rare / high-risk / high-uncertainty patterns
再次减少 representative 数
最后才考虑 cluster size 是否漂亮
```

这意味着大 cluster 不一定是坏事，坏的是“行为覆盖不足”和“critical outlier 被普通 representative 吞掉”。推荐的 selection objective：

- rarity weight：低密度区域权重大。
- hotspot risk weight：模型预测风险、PV sensitivity、EPE/CD error 大的权重大。
- uncertainty weight：encoder/分类器不确定的权重大。
- diversity term：用 k-center、facility-location submodular、DPP 或 farthest-first 选 representative。

**更值得借鉴的方向**
1. **Simulation-based incremental selection**
   Feng 2023 里这个方向比 FV 更贴近真实 model error，实验里 error range 可大幅下降。缺点是慢、依赖 baseline model。它非常适合做第二阶段：AE/embedding 先选候选，simulation-based score 再精排。

2. **Pattern coverage check / unsupervised ML coverage**
   Siemens 2023 的 process model coverage check 思路很贴近你的目标：学习 calibration patterns 的 feature vector space，再找新 design 中覆盖不足、值得加入 calibration/inspection 的 pattern。[Siemens 2023](https://resources.sw.siemens.com/pt-PT/technical-paper-unsupervised-ml-classification-driven-process-model-coverage-check/)

3. **Advanced pattern selection for OPC/EUV**
   Bae 2024 强调 pattern diversity、coverage check、lithographic contrast/illumination effects，这比纯几何 clustering 更接近“OPC 团队可用”的目标。[Bae 2024 SPIE](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12957/129571M/Advanced-pattern-selection-and-coverage-check-for-computational-lithography/10.1117/12.3009845.pdf)

4. **Deep layout metric learning**
   这比普通 AE 更值得重视。它的核心是直接学习“layout clips 之间的距离”，可以让 embedding distance 对 hotspot/non-hotspot 或行为类别更有意义。尤其适合作为你要的 behavior clustering backbone。[Geng et al. 2020/2022](https://dblp.org/rec/conf/iccad/GengYZMY0Y20)、[PDF](https://www.cse.cuhk.edu.hk/~byu/papers/C106-ICCAD2020-Metric-HSD.pdf)

5. **Vector database + pattern database**
   这是我觉得最贴近你下一阶段工程形态的方向：unsupervised metric learning + vector DB + pattern clustering，用于 hotspot retrieval、ILT reuse、参数探索。它天然支持 coverage-first 和大版图检索。[Zheng et al. 2025](https://experts.illinois.edu/en/publications/streamlining-computational-lithography-with-efficient-pattern-dat)

6. **Batch active learning**
   如果高精度 litho simulation/SEM label 成本高，active learning 很适合决定“下一批最值得仿真/量测的 patterns”。这和你的 coverage 最大化目标高度一致。[Yang et al. 2021](https://openreview.net/forum?id=HK89iLJVIR)

7. **GNN / polygon graph encoder**
   Raster AE 很吃 pixel size 和 clip size；GNN 能更直接利用 polygon 拓扑、间距、相对位置。它适合作为几何分支，与 aerial/resist AE 分支融合。[DATE 2022 GNN](https://past.date-conference.com/proceedings-archive/2022/html/0651.html)

8. **Masked layout modeling / self-supervised pretraining**
   如果你有大量无标签 layout，这是比单纯 AE 更现代的预训练方式。它学 layout structure，再 fine-tune 到 hotspot detection/OPC 任务。[Masked Layout Modeling](https://researchportal.hkust.edu.hk/en/publications/masked-layout-modeling-advances-hotspot-detection/)、[Domain-crossing masked layout modeling](https://experts.illinois.edu/en/publications/bridging-hotspot-detection-and-mask-optimization-via-domain-cross/)

9. **Simulator-powered hotspot detector / surrogate simulation**
   对“行为同类”来说，最可靠的特征不是 layout，而是 simulation response。LithoHoD 这类 simulator-powered 框架把 litho simulator latent feature 融进检测器，方向很对。[LithoHoD](https://eurekamag.com/research/099/350/099350566.php)

10. **ORC / PV / NILS / intensity-based checks**
   这类传统 litho metric 不应丢。它们非常适合作为 behavior final verification 和 review 解释层：为什么这个 pattern 是 risk，为什么两个 members 行为同类。[ORC/NILS 扩展检查](https://eurekamag.com/research/105/113/105113239.php)

**我建议的新路线**
我会把下一代版本定义为 `litho_behavior_clustering`，而不是继续叫 geometry clustering：

```text
1. marker / auto-marker / full-chip scan 生成 candidate windows
2. exact geometry hash 去重，只用于减少重复计算
3. 为每个 unique window 生成 behavior tensor:
   layout raster + aerial + resist/PV + slope/NILS + optional EPE/CD maps
4. 训练/加载 behavior encoder:
   domain-aware AE + metric learning + hotspot/severity heads
5. 在 embedding space 做 coverage-first representative selection:
   weighted k-center / submodular facility location / DPP / ANN retrieval
6. cluster assignment:
   embedding 近邻只是候选
   final verification 用 behavior image/EPE/PV/NILS residual
7. 输出:
   cluster representative layout
   member layout
   aerial/resist/PV diff
   hotspot risk score
   coverage score
   uncovered/outlier list
```

最重要的一点：**如果两个 layout 肉眼几何差异很大，但 aerial/resist/PV behavior 非常接近，它们可以同 cluster；但 review 不能只看 layout，必须同时导出 behavior map。** 否则 OPC reviewer 会天然觉得“分错了”。

**优先级建议**
第一优先级：实现 behavior-domain final verification，比是否用 AE 更关键。
第二优先级：训练 domain-aware AE / metric encoder，用于 coverage space。
第三优先级：引入 weighted coverage selection，而不是最小 cluster 数。
第四优先级：vector DB/ANN，加速大版图检索。
第五优先级：active learning，把仿真/SEM 预算用在最能提升 coverage 的样本上。

所以，新的判断是：**有足够 layout + aerial/resist/image/model-error 数据时，引入 AE/metric encoder 是有必要的，但它应该重塑 coverage 和 candidate retrieval，而不是单独替代 verification。**





**第 5 点**
它们不是一个必须串行全跑的流程，而是几类可以组合的“代表样本选择 / 检索”工具。更准确地说：

```text
ANN retrieval: 加速候选近邻搜索
weighted k-center / facility location / DPP: 从候选或全集里选 representatives
```

也就是说，ANN 更像基础设施；weighted k-center、submodular facility location、DPP 才是 selection objective。大版图上通常会组合成：

```text
behavior embedding
-> ANN 找每个点的近邻 / 建稀疏相似图
-> 用一种 representative selection 方法选 reps
-> final verification 分配 members
```

**ANN Retrieval**
作用：加速。

如果你有几十万到几百万个 windows，不可能全量计算 pairwise distance。ANN，例如 FAISS/HNSW/ScaNN，可以快速找到每个 pattern 在 behavior embedding space 里的 top-K 近邻。

它本身不决定选谁当 representative，只是帮你避免 O(N²)。

适用场景：几乎一定要用，尤其大版图。

**Weighted K-Center**
目标：最大化 coverage radius，优先覆盖“最远、最没被代表”的点。

直觉是：每次选一个离当前已选 reps 最远的 pattern。加权版本会让 high-risk、rare、high-uncertainty pattern 更容易被选中。

优点：
- 很适合“coverage 最大化”这个目标。
- 逻辑简单、可解释。
- 能避免所有 representatives 都挤在高密度 memory core 里。

缺点：
- 容易选 outlier，哪怕它只是噪声。
- 对 embedding distance 的质量很敏感。
- 不直接奖励“一个 representative 能代表很多样本”。

适合作为你的第一版 selection baseline。

**Submodular Facility Location**
目标：选一组 reps，使每个 pattern 都能被某个 selected rep 很好代表，同时考虑权重。

常见形式大概是：

```text
maximize Σ_i weight_i * max_{j in selected} similarity(i, j)
```

意思是：每个样本都找最像它的 selected representative，然后把相似度加起来。高风险样本权重大，它被代表好就更重要。

优点：
- 比 k-center 更平衡，既看 coverage，也看总体代表性。
- 不太容易被孤立噪声拖走。
- 很适合“选有限数量 reps 覆盖全集”。

缺点：
- 实现比 k-center 稍复杂。
- 需要 similarity calibration。
- 如果权重设置不好，仍可能偏向大簇。

我认为这是中期最适合你目标的主方法。

**DPP, Determinantal Point Process**
目标：选一组彼此差异大的 reps，强调 diversity。

直觉是：不要选一堆相互很像的点。DPP 会偏好“分散、多样”的集合。

优点：
- diversity 很强。
- 可减少 representatives 挤在同一类 pattern 里。
- 适合做候选补充或 tie-break。

缺点：
- 对大规模数据实现复杂。
- 不天然关心 coverage radius。
- 不天然关心 hotspot risk，除非加入 quality term。
- 比 k-center/facility location 更不直观。

我不建议第一版就把 DPP 当主算法。更适合后面作为 diversity refinement。

**推荐组合**
如果要务实，我建议这样排：

第一版：

```text
behavior embedding
-> ANN top-K graph
-> weighted k-center 选 reps
-> behavior final verification
```

第二版：

```text
behavior embedding
-> ANN top-K graph
-> weighted facility location 选 reps
-> weighted k-center 补 uncovered / farthest / high-risk holes
-> behavior final verification
```

第三版再考虑：

```text
DPP 作为同分候选的 diversity tie-break 或 subset refinement
```

所以，不是四个都必须串起来。我的推荐是：

```text
ANN 是加速层
weighted k-center 是最简单 coverage-first baseline
facility location 是更稳的主 selection objective
DPP 是可选 diversity refinement
```

**第 6 点**
behavior image、EPE、PV、NILS residual 不是同一种东西。它们对应不同层次的光刻行为。

可以只选一种，但取决于你能拿到什么数据、OPC 团队关心什么风险。最稳的是分层使用：

```text
aerial/resist image residual: 通用行为相似性
EPE residual: OPC 几何误差相关
PV residual: process window 鲁棒性
NILS residual: 成像敏感度/边缘质量
```

**Behavior Image Residual**
这里的 behavior image 可以是 aerial image、resist image、contour probability map、litho simulator 输出图等。

比较方式：

```text
MSE / SSIM / edge-weighted MSE / center-weighted MSE / contour-band residual
```

它回答的问题是：

```text
两个 pattern 的整体光刻响应图像是否相似？
```

优点：
- 最通用。
- 适合作为 AE/embedding 的直接输入和 final verification。
- 不需要提取具体 gauge 或边缘点。

缺点：
- 如果只用普通 MSE，可能过度关注大面积背景。
- 对关键 edge/local defect 的敏感性不足。
- 需要对齐、归一化和中心/edge weighting。

适合作为第一版 behavior final verification 的主指标。

**EPE Residual**
EPE 是 edge placement error。它关心目标边缘和模拟/印刷边缘之间的偏差。

它回答的问题是：

```text
两个 pattern 在 OPC 最关心的边缘位置误差上是否相似？
```

优点：
- 和 OPC 团队的语言最一致。
- 对 bridge、pinch、line-end pullback、necking 等问题更直接。
- 比整图 MSE 更接近“是否会出问题”。

缺点：
- 需要 contour extraction、target edge/gauge 定义。
- 对复杂 polygon/gauge 放置比较麻烦。
- 如果没有稳定的 simulator/contour，工程成本高。

如果你已经有 aerial/resist contour 或 OPC verification output，EPE residual 非常值得做。

**PV Residual**
PV 是 process variation，常见是不同 dose/focus/process corner 下的响应差异，例如 PV band、process window margin。

它回答的问题是：

```text
两个 pattern 在工艺波动下是否同样脆弱？
```

优点：
- 非常贴近“容易在光刻工艺过程中出问题”的目标。
- 能区分 nominal 下相似、但 process window 下风险不同的 pattern。
- 适合找 latent hotspot。

缺点：
- 计算成本最高。
- 需要多 corner simulation。
- 数据量和存储都会上去。

如果你的最终目标是 hotspot discovery，PV 是非常强的指标，但不一定适合第一版全量跑。可以先对 high-risk candidates 跑。

**NILS Residual**
NILS 是 normalized image log-slope，反映边缘处 aerial image slope，通常和成像质量、CD sensitivity、工艺鲁棒性相关。

它回答的问题是：

```text
两个 pattern 的边缘成像斜率/可印刷性是否相似？
```

优点：
- 比完整 PV simulation 便宜。
- 对边缘稳定性和 CD sensitivity 有物理意义。
- 很适合做 hotspot risk score 的辅助通道。

缺点：
- 不是最终 failure 本身，只是 proxy。
- 需要准确定位边缘/contour。
- 不同工艺/层的阈值需要校准。

NILS 可以作为 PV 的低成本替代或前置筛选。

**应该选哪一种？**
如果只做第一版，我建议：

```text
behavior image residual + edge/slope weighting
```

原因是它最通用，和 AE 输入一致，工程闭环最快。

如果你能拿到 simulator contour 或 OPC verification gauge，那么升级为：

```text
behavior image residual + EPE residual
```

如果目标明确是 litho hotspot discovery，而你有多 corner simulation 资源，那么最终应该是：

```text
behavior image residual + EPE residual + PV residual
```

NILS 建议作为辅助 score：

```text
NILS residual / low NILS area / min NILS
```

它不一定单独决定 cluster membership，但很适合给 representative 排风险优先级。

**推荐的 final verification 设计**
我会避免一开始做复杂 ML classifier，而是先做可解释的多指标门：

```text
pass if:
  behavior_image_distance <= T_img
  and EPE_residual <= T_epe        # 如果可用
  and PV_band_residual <= T_pv     # 如果可用
  and NILS_delta <= T_nils         # 如果可用
```

但为了避免过硬阈值误杀，也可以做 weighted score：

```text
score =
  0.40 * image_residual
+ 0.30 * EPE_residual
+ 0.20 * PV_residual
+ 0.10 * NILS_residual

pass if score <= threshold
```

我更推荐分阶段：

第一阶段：

```text
AE/behavior embedding: 召回候选
behavior image residual: final verification
```

第二阶段：

```text
加入 EPE residual
```

第三阶段：

```text
对 high-risk / uncovered / representative candidates 加 PV residual
```

第四阶段：

```text
NILS/PV/EPE 共同形成 risk score，用于排序和 coverage weighting
```

**一句话版**
第 5 点里，ANN 是加速检索层，weighted k-center/facility location/DPP 是不同的 representative selection 策略，优先从 weighted k-center 或 facility location 开始。

第 6 点里，behavior image/EPE/PV/NILS 是不同层次的光刻行为指标，不是完全互斥；第一版用 behavior image residual，后续加入 EPE 和 PV，NILS 作为低成本风险/敏感度辅助指标最合适。









# Lithography-Behavior Coverage Clustering 实施计划

## Summary
将 `layout_clustering_optimized.py` 从“几何 verified clustering”升级为默认 marker-driven 的“光刻行为 coverage clustering”。新目标顺序固定为：最大化 lithography behavior coverage，其次减少 representative/cluster 数，同时保证 cluster member 与 representative 通过 behavior final verification。

不再支持 auto-marker 路线；后续版本默认输入一定有 marker layer。新增独立 AutoEncoder 脚本用于训练和导出 FV，主脚本载入 FV 后执行 ANN top-K graph、weighted facility location、weighted k-center 补洞、behavior final verification 和 review/export。

## Key Changes
- `layout_clustering_optimized.py` 作为新的 behavior 主入口：
  - 保留 marker layer、layer ops、review、json/txt 输出能力。
  - exact geometry hash 只用于去重、权重累计和减少重复计算，不再作为最终聚类语义。
  - 主聚类依据改为 AE/FV embedding + behavior verification。
  - 移除 auto-marker 相关入口和后续规划依赖。

- 新增独立脚本 `layout_clustering_autoencoder.py`：
  - 使用 PyTorch，提供 `train` 和 `encode` 两个子命令。
  - 输入采用 Manifest+NPZ 数据契约。
  - `aerial` 必填；`layout/resist/epe/pv/nils` 可选作为额外训练通道。
  - 默认 CNN AutoEncoder，latent dim 默认 `128`。
  - loss 默认使用 aerial SSIM loss + MSE；可选通道存在时加入对应重构 loss。
  - `encode` 输出 `fv_manifest.jsonl` 和 `features.npz`，供主脚本载入。

- 数据契约固定为 Manifest+NPZ：
  - Manifest 每行至少包含 `sample_id`、`source_path`、`marker_id`、`clip_bbox`、`aerial_npz`。
  - 可选字段：`layout_npz`、`resist_npz`、`epe_npz`、`pv_npz`、`nils_npz`、`risk_score`。
  - 每个 NPZ 使用 float32 数组，主键固定为 `image`；所有图像必须同 shape。
  - FV 输出 NPZ 包含 `sample_ids` 和 `features`，`sample_ids` 必须与主脚本 sample id 可一一匹配。

- Representative selection：
  - 先用 `hnswlib` 建 ANN top-K graph，默认 `top_k=64`。
  - 用 weighted facility location 做第一轮 representatives：
    - 相似度由 FV 距离转换：`sim = exp(-dist^2 / tau^2)`。
    - 权重为 `exact_duplicate_count * (1 + risk_score)`；无 `risk_score` 时为 duplicate count。
    - 默认停止条件：coverage score 达到 `0.985` 或 marginal gain 小于 `1e-4`。
  - 再用 weighted k-center 补洞：
    - 补 nearest selected rep 距离过大、verification 失败、或 high-risk 但未被充分覆盖的 samples。
    - high-risk 默认使用 risk score top 10%。
    - 补洞样本直接成为 representative candidate。

- Behavior final verification：
  - `aerial` 一定参与，比较方式固定为 SSIM distance：`1 - SSIM`。
  - `EPE/PV/NILS` 如果在 manifest 中声明为全局可用，则也参与 verification；缺失样本直接报错。
  - 多指标默认 weighted score：
    - aerial: `0.60`
    - EPE: `0.15`
    - PV: `0.15`
    - NILS: `0.10`
    - 仅对可用指标重新归一化权重。
  - 默认 pass 条件：weighted distance `<= 0.08`。
  - verification 失败的 exact cluster 不做跨 cluster reassign，直接补成自身 singleton/base representative，保证解释简单。

- Review/export：
  - 常规输出新增 behavior 字段：FV 维度、ANN top-K、coverage score、facility selected count、k-center added count、behavior verification stats。
  - 可选输出 aerial/resist/PV diff；默认关闭，指定 review dir 时可通过 `--export-diff-channels aerial,resist,pv` 打开。
  - review 中每个 cluster 输出 representative、members、behavior score、SSIM distance、可选 diff map。

## CLI Defaults
- 主脚本新增/调整参数：
  - `--behavior-manifest PATH`，必填。
  - `--feature-npz PATH`，必填。
  - `--ann-top-k 64`
  - `--coverage-target 0.985`
  - `--facility-min-gain 1e-4`
  - `--behavior-verification-threshold 0.08`
  - `--high-risk-quantile 0.90`
  - `--export-diff-channels aerial,resist,pv`，可选。
- AutoEncoder 脚本：
  - `train --manifest train.jsonl --model-out ae.pt --latent-dim 128 --epochs 100 --batch-size 128`
  - `encode --manifest all.jsonl --model ae.pt --features-out features.npz --fv-manifest-out fv_manifest.jsonl`
- 保留：
  - `--marker-layer`
  - `--clip-size`
  - `--output`
  - `--format`
  - `--review-dir`
  - `--apply-layer-ops`
  - `--register-op`
- 不保留：
  - auto-marker 参数
  - HDBSCAN/ILP/FFT/closed-loop 参数
  - 旧几何 ACC/ECC 作为主聚类 final gate 的语义

## Test Plan
- AutoEncoder 脚本：
  - Manifest+NPZ 正常加载。
  - aerial-only 训练/encode 可跑通并输出正确 shape 的 FV。
  - 多通道输入 shape mismatch 会报错。
  - `features.npz` 的 `sample_ids` 与 manifest 顺序一致。
- 主脚本单元测试：
  - marker sample 与 behavior manifest/FV 正确 join。
  - exact hash duplicate 会合并权重但不跳过 behavior selection。
  - ANN top-K graph 返回每个 sample 的近邻集合。
  - weighted facility location 优先选择高权重和高覆盖 representative。
  - weighted k-center 会补 farthest/uncovered/high-risk holes。
  - aerial-only SSIM verification 正确 pass/reject。
  - EPE/PV/NILS 存在时 weighted score 正确归一化。
  - verification 失败会创建 singleton/base representative。
  - review diff channel 只在指定时输出。
- 回归/冒烟：
  - `python -m py_compile layout_clustering_optimized.py layout_clustering_autoencoder.py`
  - 小型 synthetic NPZ 数据跑 train/encode/main 全链路。
  - 带 marker 的 sample OAS + fake aerial NPZ 跑完整聚类。
  - 输出 JSON 不含 auto-marker/HDBSCAN/ILP/FFT/closed-loop 字段。

## Assumptions
- 后续版本默认有 marker layer，不再维护 auto-marker 分支。
- `aerial` 是 behavior verification 的最低必需数据。
- EPE/PV/NILS 是数据集级可选项：一旦声明启用，所有样本都必须提供。
- PyTorch 和 hnswlib 作为新增依赖；SSIM 使用 `scikit-image`。
- 第一版不训练 supervised hotspot classifier；AE/FV 用于 behavior embedding，final correctness 由 SSIM/weighted residual verification 保证。








已完成这版“光刻行为 coverage clustering”实现。核心变化是把 [layout_clustering_optimized.py](C:/Users/81932/Documents/AIcoding/layoutclustering/layout_clustering_optimized.py) 从旧的几何 ACC/ECC verified clustering 改成了 marker-driven behavior 主线：

- 载入 `--behavior-manifest` + `--feature-npz`
- exact hash 只做去重和 duplicate weight 累计
- 用 FV 构建 ANN top-K graph，优先走 `hnswlib`，本机未安装时有 exact top-K fallback
- weighted facility location 选第一批 reps
- weighted k-center 补 farthest / high-risk holes
- final verification 使用 aerial SSIM，EPE/PV/NILS 全局可用时自动加入 weighted score
- review 可选导出 `aerial/resist/pv` diff NPZ
- 输出新增 `behavior_stats`、`behavior_verification_stats`、FV 维度、ANN backend、coverage score、facility/k-center 统计等字段

也新增了独立 AE 脚本 [layout_clustering_autoencoder.py](C:/Users/81932/Documents/AIcoding/layoutclustering/layout_clustering_autoencoder.py)，提供：

```bash
python layout_clustering_autoencoder.py train --manifest train.jsonl --model-out ae.pt
python layout_clustering_autoencoder.py encode --manifest all.jsonl --model ae.pt --features-out features.npz --fv-manifest-out fv_manifest.jsonl
```

它采用 Manifest+NPZ 数据契约，`aerial_npz` 必填，`layout/resist/epe/pv/nils` 可选；PyTorch 是懒加载，所以脚本可编译，但本机未安装 torch 时训练/编码会明确报错。依赖也已补到 [requirements.txt](C:/Users/81932/Documents/AIcoding/layoutclustering/requirements.txt)。

测试更新：

- [test_optimized_clustering.py](C:/Users/81932/Documents/AIcoding/layoutclustering/test_optimized_clustering.py)：覆盖 manifest/FV join、ANN graph、facility selection、k-center、SSIM verification、多通道权重归一化、完整 marker OAS 冒烟、review diff 输出。
- [test_autoencoder.py](C:/Users/81932/Documents/AIcoding/layoutclustering/test_autoencoder.py)：覆盖 Manifest+NPZ、shape mismatch、CLI parser；PyTorch smoke 在本机因未安装 torch 被 skip。
- [test_optimized_auto_marker.py](C:/Users/81932/Documents/AIcoding/layoutclustering/test_optimized_auto_marker.py)：标记 auto-marker 路线 retired 并跳过。

已验证：

```text
python -m py_compile layoutclustering\layout_clustering_optimized.py layoutclustering\layout_clustering_autoencoder.py layoutclustering\test_optimized_clustering.py layoutclustering\test_autoencoder.py layoutclustering\test_optimized_auto_marker.py
python -m unittest test_optimized_clustering -v  # 9 tests OK
python -m unittest test_autoencoder -v           # 3 OK, 1 skipped: PyTorch not installed
python -m unittest test_optimized_auto_marker -v # 1 skipped: retired
python layoutclustering\layout_clustering_optimized.py --help
python layoutclustering\layout_clustering_autoencoder.py --help
git diff --check
```
