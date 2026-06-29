# CD-SEM Hotspot Recipe 选点任务文献调研与现有方案评估 260520

## 0. 任务隔离声明

当前 `hotspotdetection` 是一个新的预研任务。它可以复用旧探索脚本中的 feature vector、ANN 检索、代表性样本选择、行为验证等工程部件，但目标不能沿用旧的“全版图重复 pattern 聚类”定义。本文所有判断以 `CD-SEM_pattern_selection_task.png` 为准：

1. 以 hotspot marker 为起点抓取周围 layout/behavior pattern。
2. 自动搜索、分类并抓取一批有代表性的、可能反映版图潜在 hotspot 的量测位置。
3. 在 hotspot 附近找到相似 pattern 作为 autofocus 点。
4. 在 hotspot 附近找到 unique pattern 作为 addressing 点，且无对准 marker。
5. 最终面向 CD-SEM recipe，而不仅是聚类结果或 representative 列表。

## 1. 当前脚本是否 on the right track

### 1.1 已经对齐的部分

现有脚本对“hotspot marker 周围 pattern 的表征、去重、代表性选择和聚类验证”是有价值的，属于新任务的前半段可复用基础：

- `feature_extractor_autoencoder.py` 是 AE/FV 参考主线：可为后续 task-aware embedding 提供结构参考，但当前 prototype 暂不把 AE 作为必须路径。
- `hotspot_recipe_notrain_backend.py` 是无训练 MP selection backend：自动生成 handcrafted FV，后半段做 ANN/coverage selection/behavior verification，适合作为新任务初期 deterministic baseline。
- `feature_extractor_autoencoder.py` 和 `feature_extractor_handcraft.py` 已经覆盖了 aerial image、layout bitmap、optional behavior map、layout geometry、WL graph signature 等多模态信息，这些都可作为 hotspot/AF/AP 候选点打分的 feature pool。
- `preprocess_behavior_inputs.py` 已经解决了 marker 与 aerial image 的输入对齐问题，对后续 recipe site 数据准备有用。

因此，如果把新任务拆成“先从 hotspot marker 生成候选点和候选 pattern embedding，再做覆盖式代表点选择”，当前脚本是部分 on track 的。

### 1.2 明显偏差和缺口

当前脚本最大的偏差是：它把终点放在 behavior coverage clustering，而 PNG 的终点是 CD-SEM recipe site selection。两者相邻但不等价。

主要 gap 如下：

1. 缺少 measurement point 自动发现层  
   DesignGauge/Miyamoto 这类 recipe 文献通常假设评价点/量测点已经由用户或上游流程给定；当前新任务则需要从 hotspot marker 周边自动搜索潜在 hotspot pattern，并筛选出值得进入 CD-SEM recipe 的 measurement point。现有脚本只把已有 marker 当作聚类样本，没有做 full-layout/marker-neighborhood 的候选点生成、hotspot risk scoring、false alarm 控制和 budget-aware 采样。

2. 缺少 recipe site schema  
   现有输出是 cluster、representative、member、review file。新任务需要输出每个量测 recipe site 的 measurement point、hotspot cluster、AF 点、AP 点、坐标、FOV、距离、相似度、唯一性、失败原因和 review 图。

3. 缺少 autofocus 选点逻辑  
   PNG 要求“在 hotspot 点位附近找相似 pattern 的点位做 autofocus”。当前脚本只判断 representative 与 member 是否相似，没有在每个 hotspot 周边搜索 AF candidate，也没有排除 hotspot core、控制 AF 与 measurement point 的距离、评估 focus robustness。

4. 缺少 addressing unique 选点逻辑  
   PNG 要求“在 hotspot 点位附近找 unique 点位做 addressing，无对准 marker”。当前脚本没有 pattern ambiguity/uniqueness 指标，没有局部自相关、多峰匹配、nearest-neighbor margin、周期阵列排除等 AP 必需逻辑。

5. AutoEncoder 目标偏向重构，不偏向 hotspot 物理风险和定位  
   当前 AE 使用重构损失、局部 SSIM、edge-aware loss，适合生成相似性 FV，但没有用 hotspot/non-hotspot label、error marker、EPE peak、process window fail 等信号学习“为什么这个点危险”。如果后续只是继续调 AE latent，容易得到“外观相似”而不是“hotspot 风险相似”。

6. coverage 定义偏向 pattern diversity，不等于 recipe 有效覆盖  
   Feng 系列 pattern selection 的 coverage 思路适合 OPC calibration pattern set；新任务的 coverage 应该至少同时覆盖：hotspot 类型、版图上下文、AF 可用性、AP 唯一性、量测可执行性、CD-SEM 成像/寻址成功率。

7. 缺少闭环验证指标  
   当前 verification 是 behavior image 层面的 member-representative 相似性。CD-SEM recipe 还需要验证：AF 是否成像稳定、AP 是否唯一锁定、measurement point 是否靠近 hotspot/root cause、recipe 执行是否成功、量测结果是否能区分工艺风险。

简要结论：当前脚本可作为新任务的候选生成和代表性选择底座，但还没有进入 CD-SEM recipe 自动化的核心问题。下一步不应继续单纯优化 cluster purity/recall，而应新增 AF/AP candidate selection 与 recipe 输出契约。

## 2. 文献分层综述

### 2.1 CD-SEM recipe 自动生成：最直接相关

**Miyamoto & Matsuoka, 2015, Automatic Generation of Imaging Sequence for CD-SEM Using Design Data**  
链接：https://www.jstage.jst.go.jp/article/ieejeiss/135/4/135_444/_article/-char/en  
价值：这是最贴近当前 PNG 的文献。论文明确把 CD-SEM imaging sequence 拆成 evaluation point、addressing point、auto-focus point，并从 design data 计算 pattern complexity、uniqueness 等指标来自动选择 AP/AF template。公开摘要中给出 901 个 evaluation points 的 SEM imaging success rate 为 100%，生成时间为 18 分钟，作为当前任务的工程目标形态非常值得参考。  
对当前 gap 的启发：AP/AF 不是聚类附属物，而是 recipe 执行成功的必要点位。当前脚本应增加 `AP_score`、`AF_score`、`uniqueness_margin`、`ambiguity_peak_count`、`distance_to_MP`、`candidate_reject_reason` 等字段。

**Hitachi DesignGauge / RecipeDirector 系列, 2005-2011**  
代表链接：  
- Evolution and Future of Critical Dimension Measurement System for Semiconductor Processes, Hitachi Review 2011：https://www.hitachihyoron.com/rev/pdf/2011/r2011_05_104.pdf  
- Automated CD-SEM Recipe Generation Utilizing Design Pattern Layout, Ward & Page, Hitachi slides：https://nccavs-usergroups.avs.org/wp-content/uploads/PAG2005/PEUG_04_2005_Ward.pdf  
价值：DesignGauge 把 GDS/design DB、matching、recipe creation、SEM image 和 measurement results 串成工业流程；Hitachi Review 2011 明确提到用 CAD data 离线快速生成 measurement recipe，并显著缩短 recipe preparation 时间。需要注意的是，公开资料中 DesignGauge 的典型输入包含 design data 和 measurement-point information，也就是用户或上游流程已经给定待量测点位；它主要解决离线 recipe 生成、design template 生成、SEM-to-design matching 和多点 recipe 批量化，不等价于从全 layout 自动发现潜在 hotspot 量测点。  
对当前 gap 的启发：新任务最终交付不应止于 `clusters.json`，而应形成“离线 recipe generation package”：坐标表、设计模板、AF/AP 模板、review 图、失败项列表、可回灌结果。但 measurement point 的自动搜索和代表性选择仍是当前新任务必须额外完成的上游模块，不能由 DesignGauge 文献直接覆盖。

**Tabery & Page, 2005, Use of Design Pattern Layout for Automatic Metrology Recipe Generation**  
检索来源：https://www.globalsino.com/ICsAndMaterials/page2360.html  
价值：早期 SPIE 工业实践，核心是 design pattern layout 驱动 automatic metrology recipe。虽公开全文不易获取，但被 Hitachi/DesignGauge 相关材料反复引用，是当前任务中“无对准 marker、依靠设计数据寻址”的直接先例。  
对当前 gap 的启发：AP 必须从 design layout 本身选出可稳定识别的局部模板，而不是沿用 hotspot cluster representative。

**Morokuma et al., 2005, A New Matching Engine Between Design Layout and SEM Image of Semiconductor Device**  
检索来源：https://nccavs-usergroups.avs.org/wp-content/uploads/PAG2005/PEUG_04_2005_Ward.pdf  
价值：关注 design layout 与 SEM image matching，是 AP 选点后能否真正对上 SEM 图像的底层问题。  
对当前 gap 的启发：AP candidate 不仅要在 layout 中 unique，还要考虑设计图到 SEM 图像的形变、边缘提取、contrast 和多层 pattern matching 鲁棒性。

### 2.2 Hotspot measurement point 自动发现：补 DesignGauge 上游缺口

这一层回答的问题是：在 DesignGauge/Miyamoto 假设 measurement point 已知之前，如何从 layout 或 hotspot marker 周边自动生成、过滤、排序、去重并选出真正值得 CD-SEM 量测的 hotspot measurement points。它与 AP/AF recipe 自动生成不是同一问题，但两者必须串起来：`MP 自动发现 -> MP 覆盖选择 -> AP/AF 生成 -> recipe 执行`。

**Ding et al., 2009, Machine Learning based Lithographic Hotspot Detection with Critical-Feature Extraction and Classification**  
链接：https://www.cerc.utexas.edu/utda/publications/ICICDT09_ML.pdf  
价值：提出从 layout binary image 中提取 bounded rectangle、T-shape、L-shape 等 critical features，再用监督学习分类 hotspot。论文还提出 proximity detection algorithm，在大 layout 中生成 sampled-pattern-for-testing seeds，并用邻域投票降低误报。  
对当前 gap 的启发：当前任务可以先从 hotspot marker 周围生成密集候选窗口，然后用几何 critical feature + behavior FV 形成 `MP_risk_score`；对邻近候选做 voting/NMS/聚合，避免一个真实 weak point 周围输出一堆重复量测点。

**Gao, Yu & Pan, 2014, Lithography Hotspot Detection and Mitigation in Nanometer VLSI**  
链接：https://arxiv.org/abs/1402.3150  
价值：综述指出 hotspot detection 是 physical verification 和 early physical design 阶段的重要步骤，并将主流方法分为 lithography simulation、pattern matching、machine learning。simulation 准但慢，pattern matching 快但难覆盖 unknown hotspot，ML 适合 full-chip 快速筛选。  
对当前 gap 的启发：MP 自动发现不应只依赖一种信号。建议采用多级 funnel：cheap geometric/pseudo-DRC proposal 负责召回，FV/ML risk score 负责排序，少量 litho/behavior simulation 或 aerial/EPE/PV/NILS 负责确认高风险候选。

**Yang et al., 2021, Bridging the Gap between Layout Pattern Sampling and Hotspot Detection via Batch Active Learning**  
链接：https://www.cse.cuhk.edu.hk/~byu/papers/J54-TCAD2021-AL-HSD.pdf  
价值：把 layout pattern sampling 和 hotspot detection 放在同一个问题里处理，目标是在没有预先固定训练/测试集的真实场景中，用更少 lithography simulation overhead 采样 representative clips，并提升 detector generality。论文的问题定义明确强调从 layout design 中采样能泛化 hotspot pattern space 的 clips。  
对当前 gap 的启发：这正好对应“候选 MP 太多、SEM budget 有限”的情况。当前任务可以把 selected MP 看成 active sampling batch：综合 uncertainty、diversity、risk、coverage，选择一批最值得 SEM recipe 验证的点，而不是只选 cluster centroid。

**Gai et al., 2021, Hotspot detection in large-scale layout with proposal sampling and feature parameters optimization**  
检索来源：https://eurekamag.com/research/099/823/099823395.php  
价值：指出大版图 hotspot detection 的瓶颈在于从待检测 layout 中抽取 clip patterns；其方案通过已知 hotspot 的 clustering analysis 生成 filter rules，高效抽取 layout clips 供 classifier 检测，并优化图形特征参数以减少上下文冗余。  
对当前 gap 的启发：当前任务不是盲扫所有窗口，而应从已有 hotspot marker 出发学习 proposal rules：例如 marker 周边的 edge pair、line-end、jog、bridge/pinch proxy、局部密度突变、EPE/PV proxy 等，先做 proposal sampling，再进入 FV/risk 排序。

**Zhu et al., 2021, Hotspot Detection via Multi-task Learning and Transformer Encoder**  
链接：https://www.cse.cuhk.edu.hk/~byu/papers/C125-ICCAD2021-H-DETR.pdf  
价值：针对传统 clip-wise detector 只能判断小窗口中心是否 hotspot、全版图扫描耗时的问题，提出 single-stage hotspot detector，在大尺度 layout 中直接输出 hotspot bounding boxes，并辅助学习 center/corner 表示；Transformer Encoder 用于捕捉长程上下文关系。  
对当前 gap 的启发：如果未来有足够 label 或 simulation marker，MP 自动发现应从“窗口分类器”升级为“目标检测/定位器”：直接输出 hotspot box、center、confidence 和 defect type。CD-SEM measurement point 可取 box center、defect-location peak，或按量测目标偏移到最合适的 CD/edge 位置。

**Shao et al., 2025, LithoHoD: A Litho Simulator-Powered Framework for IC Layout Hotspot Detection**  
链接：https://arxiv.org/abs/2409.10021  
价值：用 object detection backbone 结合 lithography simulator 的 latent features，通过 cross-attention 同时利用“已知 problematic patterns”和“simulation 估计的可能形变变化”，从而检测潜在 hotspot regions，增强对真实场景的泛化。  
对当前 gap 的启发：当前脚本已有 aerial/EPE/PV/NILS 等 behavior channel 时，可以把它们作为 simulator-guided risk signal，而不是只作为 cluster verification。MP 排序应把 layout pattern similarity 与 litho behavior sensitivity 融合。

**Hu et al., 2020, Pattern-Centric Computational System for Logic and Memory Manufacturing and Process Technology Development**  
链接：https://doaj.org/article/522dc7e2b827493fa0209e082eed9783  
价值：工业 pattern-centric 系统把 GDS/OASIS、die-to-database、ML、care area generation、SEM sampling、defect discovery、full-chip decomposition 和 pattern risk scoring 串成闭环，并明确提到用 ML-based SEM sampling 优化 DOI capture rate 和新 defect type discovery。  
对当前 gap 的启发：当前任务可以借鉴“care area + SEM sampling + risk scoring”的工业形态：先生成候选 care areas/MPs，再按 risk/rarity/coverage 选点，CD-SEM 结果回灌，形成动态改进的 hotspot recipe engine。

**Siemens Calibre SONR, industrial full-chip ML platform**  
链接：https://eda.sw.siemens.com/en-US/ic/calibre-manufacturing/fab-solutions/calibre-sonr/  
价值：公开资料显示 SONR 将 layout/process information 转为 features，支持 full-chip hotspot prediction and analysis、pattern reduction、coverage check、care area generation、layout clustering/comparison，并强调 billion-level full-chip data 的可扩展性。  
对当前 gap 的启发：这是工业产品层面对当前任务的旁证：MP 自动发现必须是 full-chip/large-layout feature platform，而不是单个 recipe 点位工具。当前实现可以先做轻量版：marker-neighborhood feature DB + HNSW + risk scoring + set-cover/batch sampling。

**小结：measurement point 自动发现的技术分层**

- `proposal generation`：从 hotspot marker 周围或 full layout 中生成候选 MP/care area，来源可以是 pseudo-DRC、critical feature、known-hotspot cluster rules、object detector anchors、EPE/PV/aerial risk map。
- `risk scoring`：给每个候选点计算 hotspot likelihood、process sensitivity、defect-location confidence、known/unknown novelty、false-alarm penalty。
- `budgeted selection`：在 SEM recipe 数量有限时，用 active learning / facility location / set cover / k-center 选择高风险且多样的 MP batch。
- `localization refinement`：把候选窗口中心修正到 defect-location peak、bridge/pinch center、line-end/corner root cause 或最适合 CD-SEM 量测的边/线位置。
- `recipe feasibility gating`：只有能找到合格 AF/AP 的 MP 才进入最终 recipe；没有合格 AF/AP 的候选保留为 rejected MP，并写明原因。

### 2.3 Pattern coverage 与 FV：可复用但要重新定义目标

**Feng et al., 2023, Layout Pattern Analysis and Coverage Evaluation in Computational Lithography**  
链接：https://doi.org/10.1364/OE.485206  
本地文件：`../references/Feng_2023.pdf`  
价值：提出在未取得 metrology data 前评估 pattern coverage 的 FV-based 和 simulation-based metrics，并基于 simulation error 做 incremental selection；本地 PDF 摘要指出该方法最多减少 53% model verification error range。  
对当前 gap 的启发：适合做“hotspot marker set 是否覆盖足够多 pattern/risk 类型”的 coverage 度量，但不能直接替代 AF/AP 可执行性指标。

**Feng et al., 2024, Feature Vector Effectiveness Evaluation for Pattern Selection in Computational Lithography**  
链接：https://www.mdpi.com/2304-6732/11/10/990  
本地文件：`../references/Feng_2024.pdf`  
价值：系统比较 AutoEncoder FV 与 FFT FV，并使用 KL divergence、distance ranking 等 KPI 评估 FV 对 pattern selection 的有效性；结论支持 domain-knowledge augmented AE FV 优于 FFT FV。  
对当前 gap 的启发：当前 AE 路线方向合理，但应从“单一重构 embedding”升级为“多任务/双流/物理信号增强 embedding”，尤其加入 hotspot root-cause、AF focus quality、AP uniqueness 的监督或弱监督目标。

**Zheng et al., 2025, Streamlining Computational Lithography With Efficient Pattern Database**  
链接：https://www.cse.cuhk.edu.hk/~byu/papers/J148-TCAD2025-PattBase.pdf  
价值：提出 pattern database 框架：unsupervised metric learning、vector database、pattern clustering，并用 pattern retrieval 支持 hotspot detection、ILT solution reuse 和代表性 pattern selection。  
对当前 gap 的启发：当前脚本的 ANN/HNSW 思路可以升级成“recipe pattern DB”：不仅检索相似 hotspot，还检索 AF 相似候选和 AP 唯一候选；并用 nearest-neighbor margin 作为 AP 唯一性的一部分。

**Chen et al., 2017, Minimizing Cluster Number with Clip Shifting in Hotspot Pattern Classification**  
链接：https://doi.org/10.1145/3061639.3062283  
本地文件：`../references/Chen_2017.pdf`  
价值：面向 hotspot pattern classification，通过 clip shifting 和 set cover 减少 representative hotspot 数量。  
对当前 gap 的启发：clip shifting 对量测点/AF/AP 都有启发，因为 CD-SEM recipe 中 candidate center 不是固定不可动的。可以把 AP/AF 的候选窗口视为带约束的 local search，而不是只取 marker center。

### 2.4 Hotspot detection、定位和可解释性：补“为什么危险”

**Ding, Torres & Pan, 2011, High Performance Lithography Hotspot Detection With Successively Refined Pattern Identifications and Machine Learning**  
链接：https://doi.org/10.1109/TCAD.2011.2164537  
价值：经典 ML hotspot detection 框架，强调逐次细化 pattern identification 和 false alarm 控制。  
对当前 gap 的启发：新任务如果要选“容易出现 hotspot 的位置”，仅按 pattern diversity 不够，需要 hotspot risk scoring 与 false alarm control。

**Zhang, Yu & Young, 2016, Enabling Online Learning in Lithography Hotspot Detection with Information-Theoretic Feature Optimization**  
链接：https://research.cuhk.edu.hk/en/publications/enabling-online-learning-in-lithography-hotspot-detection-with-in-2/  
价值：用信息论特征优化和在线学习处理新出现的 hotspot pattern，强调经过验证的新样本可以回灌模型。  
对当前 gap 的启发：CD-SEM recipe 的实际量测结果应回灌，更新 hotspot risk、AF success、AP ambiguity，而不是一次性静态聚类。

**Sun et al., 2022, Efficient Hotspot Detection via Graph Neural Network**  
链接：https://research.cuhk.edu.hk/en/publications/efficient-hotspot-detection-via-graph-neural-network-2/  
价值：将 layout 表示为 graph，用 GNN 提取局部几何关系 embedding；公开摘要报告在 ICCAD2012 上 over 10x speedup 且 false alarms 更少。  
对当前 gap 的启发：当前 handcrafted WL graph 是好的起点，但如果后续要定位 root-cause 或处理复杂拓扑，GNN 比纯 bitmap AE 更适合表达 polygon 邻接、层间关系和几何约束。

**Jiang et al., 2025, LithoExp: Explainable Two-stage CNN-based Lithographic Hotspot Detection with Layout Defect Localization**  
链接：https://zhiyaoxie.com/files/TODAES25_LithoExp.pdf  
价值：两阶段 CNN 显式学习 defect location map，再结合 layout 和 ROI 做 hotspot detection；公开 PDF 摘要报告 hotspot accuracy 98.1%、false alarm rate 4.0%，并强调可解释定位。  
对当前 gap 的启发：recipe measurement point 应尽量贴近 hotspot root-cause 区域。若有 simulation error marker/EPE map，可训练或弱监督一个 defect-location channel，而不是只聚类整张 clip。

**Zhang et al., 2025, Enhanced Lithographic Hotspot Detection via Multi-Task Deep Learning With Synthetic Pattern Generation**  
链接：https://doi.org/10.1109/OJCS.2024.3510555  
价值：用 synthetic pattern generation、multi-task CNN 和 adaptive loss 处理 truly-never-seen-before hotspots 与 hard-to-classify patterns；DOAJ 摘要报告 ICCAD-2019 上 98.5% accuracy、1.2% false alarm。  
对当前 gap 的启发：真实 hotspot 数据稀缺时，应补 synthetic/DOE pattern，尤其覆盖 edge cases，而不是只依赖已有 marker。

### 2.5 Design-to-SEM / Die-to-database：补“能否对准”

**SPPE-GAN, 2025, Die-to-Database alignment and SEM distortion correction**  
链接：https://www.sciencedirect.com/science/article/pii/S0957417425002805  
价值：面向 SEM image 与 GDS/layout 对齐，使用无监督 SEM pattern extraction、SIFT/FLANN matching 和 optical flow correction，解决 SEM 图像畸变与 design matching。  
对当前 gap 的启发：AP 评分最终不能只看 layout uniqueness，还应预留 SEM-to-design alignment 的验证接口，例如 design template 与实际 SEM 的匹配峰值、畸变残差、alignment confidence。

**Hu et al., 2020, Pattern-Centric Computational System for Logic and Memory Manufacturing and Process Technology Development**  
链接：https://doaj.org/article/522dc7e2b827493fa0209e082eed9783  
价值：工业 pattern-centric 系统，把 GDS/OASIS、die-to-database、ML、care area generation、SEM sampling、defect discovery、risk scoring、OPC verification 串成闭环。  
对当前 gap 的启发：当前新任务可以定位成小型 pattern-centric recipe engine：先从 hotspot marker 出发做 recipe sites，再通过 CD-SEM 结果回灌风险和采样策略。

## 3. 建议采用的新任务方案

### 3.1 总体流程

建议把新任务拆成 5 个模块，避免继续被“聚类输出”牵引：

```text
OAS/GDS + hotspot marker + aerial/SEM/simulation maps
-> marker-neighborhood proposal generation
-> MP candidate risk scoring + localization refinement
-> budget-aware hotspot MP coverage selection
-> 每个 selected MP 周边搜索 AF candidate
-> 每个 selected MP 周边搜索 AP candidate
-> recipe site package + review + measurement feedback loop
```

### 3.2 模块 A：MP 自动发现与 hotspot representative selection

这个模块是 DesignGauge/Miyamoto 没有覆盖、但当前任务必须先补上的上游。建议先做 deterministic baseline，再考虑训练型 detector。

输入：

- hotspot marker 坐标或 hotspot marker layer。
- marker 周围 layout clip。
- 可选 aerial/EPE/PV/NILS/resist/simulation map。
- 可选已知 hotspot type 或 risk label。

步骤：

1. `proposal generation`  
   在 marker 周边生成候选 MP window，而不是只使用 marker center。候选来源包括：marker center、clip 内 line-end/corner/jog/bridge/pinch proxy、edge pair、局部密度突变、critical width/space、EPE/PV/aerial 高响应点、已知 hotspot cluster rule 命中的窗口。

2. `candidate feature extraction`  
   复用当前 AE FV / handcrafted FV / layout geometry / WL graph / behavior stats，为每个候选 MP 生成 feature vector。候选窗口尺寸应和 CD-SEM measurement FOV、hotspot interaction radius 分开定义，避免一个窗口同时承担检测和量测两个尺度。

3. `risk scoring`  
   给每个候选计算 `MP_risk_score`。baseline 可用加权规则：critical geometry score、behavior intensity score、rarity/novelty score、known hotspot similarity、neighbor voting confidence。若后续有 label，可升级为 ML/CNN/object detector。

4. `localization refinement`  
   将候选中心从粗 grid/marker center 修正到最可量测的位置，例如 defect-location peak、EPE/PV peak、bridge/pinch center、line-end/corner root cause 或需要 CD-SEM 量测的 edge/space 中心。

5. `duplicate suppression`  
   对空间邻近且 feature 相似的候选做 NMS 或 cluster suppression，避免同一个 hotspot 周围输出多个重复 MP。

6. `budget-aware selection`  
   在有限 recipe 点数下，按 risk、rarity、coverage、uncertainty、AF/AP 可行性做 batch selection。可以沿用当前 weighted facility location / k-center，但目标要从“cluster coverage”改成“hotspot risk coverage + recipe feasibility coverage”。

输出字段建议：

```text
mp_candidate_id
source_marker_id
mp_x_um
mp_y_um
mp_window_bbox
mp_risk_score
mp_risk_components
mp_localization_reason
mp_hotspot_type
mp_novelty_score
mp_coverage_cluster_id
mp_selected
mp_reject_reason
```

### 3.3 模块 B：已有聚类/FV 底座如何复用

保留当前脚本的可复用部分：

- exact hash 去重。
- AE FV / handcrafted FV。
- ANN top-K 检索。
- weighted facility location / k-center 覆盖选择。
- behavior verification。

需要改造的地方：

- coverage 权重从 `risk_score` 扩展为 `hotspot_risk_score`、`pattern_rarity_score`、`process_sensitivity_score`、`AF_available`、`AP_available` 的组合。
- 输出单位从 cluster representative 改成 recipe measurement point candidate。
- cluster 仍可存在，但只是 recipe site grouping 的中间信息。

### 3.4 模块 C：AF candidate selection

目标：在 hotspot 附近找到与 MP 图形/成像行为足够相似、但不直接落在 hotspot root-cause 上的对焦点。

建议指标：

- `AF_similarity`: 与 MP 的 FV cosine similarity 或 learned metric similarity。
- `AF_distance`: 到 MP 的距离约束，过近可能干扰 measurement，过远降低局部相关性。
- `AF_focus_quality`: 图像/版图局部边缘密度、梯度能量、contrast、pattern complexity。
- `AF_hotspot_exclusion`: 排除高 hotspot risk 或 defect-location map 覆盖区域。
- `AF_stability`: 小 shift 下 FV/SSIM/NCC 稳定，不对 1-2 pixel shift 过敏。

可从当前代码复用：

- AE/handcrafted FV。
- aerial SSIM / residual verification。
- layout geometry 中的 edge/corner/line-end proxy。

### 3.5 模块 D：AP candidate selection

目标：在 hotspot 附近找到可唯一寻址、匹配峰值清晰、非周期重复的 addressing 点。

建议指标：

- `AP_uniqueness_margin`: 最近匹配与第二近匹配的距离差或相似度差。
- `AP_self_match_peak_ratio`: 大 FOV 内自相关或模板匹配的主峰/次峰比。
- `AP_peak_count`: 超过阈值的匹配峰数量，数量多则 ambiguity 高。
- `AP_entropy`: local image/layout entropy，避免过于空旷或过于周期。
- `AP_corner_density`: Harris/FAST-like corner proxy，或 layout corner/jog proxy。
- `AP_periodicity_penalty`: 对 memory array / dense repeating line-space 加惩罚。
- `AP_to_MP_distance`: 满足 CD-SEM beam/stage move 和 FOV 约束。

可从当前代码复用：

- WL graph signature 和 layout geometry 特征。
- ANN/vector DB 检索。
- review 导出框架。

需要新增：

- 局部滑窗候选生成。
- 多峰匹配/自相关检测。
- AP reject reason。
- design-template review 图。

### 3.6 模块 E：recipe 输出契约

建议先定义 CSV/JSON schema，再改算法。最小字段：

```text
site_id
source_layout
hotspot_marker_id
hotspot_cluster_id
measurement_x_um
measurement_y_um
measurement_score
hotspot_risk_score
af_x_um
af_y_um
af_similarity
af_focus_quality
af_distance_um
ap_x_um
ap_y_um
ap_uniqueness_margin
ap_peak_count
ap_entropy
ap_distance_um
recipe_status
reject_reason
review_dir
```

### 3.7 模块 F：闭环学习

从 CD-SEM 执行结果回灌：

- MP 是否真的呈现 hotspot/弱点。
- AF 是否成功对焦。
- AP 是否抓错或多峰。
- 量测结果 CD/EPE/LER 是否异常。
- SEM-to-design alignment confidence。

这样可以把当前一次性聚类脚本升级为 active learning / online learning 的 recipe engine。

## 4. 推荐优先级

### P0：先把任务目标从 clustering 输出切到 recipe site 输出

新增 `recipe_site_selector.py` 作为独立入口，不把 AP/AF 硬塞进旧的 layout clustering 语义。无训练能力由 `hotspot_recipe_notrain_backend.py` 作为内部 backend 提供。

验证标准：

- 输入 marker + aerial/layout 后，至少输出 candidate MP list，并区分 selected/rejected。
- 对 selected MP，输出 MP/AF/AP 三元组；对没有合格 AF/AP 的点，输出明确 reject reason。

### P1：实现 MP 自动发现 deterministic baseline

先不要直接上深度 detector。建议先做可解释规则 + FV 的 baseline：

- 在 hotspot marker 周边滑窗或按 edge/corner/line-end/jog/bridge/pinch proxy 生成 MP candidates。
- 对每个候选计算 layout geometry、handcrafted FV、aerial/EPE/PV/NILS stats、known-hotspot similarity、novelty/rarity。
- 组合出 `MP_risk_score`，并用邻域投票/NMS 抑制重复点。
- 用 weighted facility location / k-center 选出 risk 高且 pattern 多样的 MP batch。

验证标准：

- 对同一 root-cause 周边的重复候选，只保留少量代表点。
- selected MP 能覆盖不同 hotspot morphology，而不是全部落在密集重复阵列。
- 每个 selected/rejected MP 都有 `mp_risk_components` 和 `mp_localization_reason`。

### P2：实现 AP uniqueness baseline

先做无需训练的 deterministic baseline：

- 在 hotspot 周边滑窗生成 AP candidates。
- 用 layout bitmap / aerial image 做自相关或 template matching。
- 计算 peak ratio、peak count、entropy、corner density。
- 选唯一性最高且距离合适的点。

验证标准：

- 对周期性 memory array，AP 应被拒绝或低分。
- 对局部非周期交叉/拐角/复杂结构，AP 应优先选中。

### P3：实现 AF similarity baseline

先用当前 FV 和 SSIM：

- 在 hotspot 周边滑窗生成 AF candidates。
- FV cosine similarity 接近 MP，但避开 hotspot root-cause。
- 加入 edge/gradient focus quality。

验证标准：

- AF 点与 MP 相似但不重叠。
- shift-tolerant verification 通过。

### P4：把 AE 改成任务感知 embedding

不建议一开始就重训复杂大模型。等 MP/AP/AF baseline 有 review 输出后，再考虑：

- layout branch + aerial/SEM branch 的 two-stream encoder。
- defect-location / EPE / PV / NILS 辅助监督。
- contrastive loss：同类 hotspot MP 拉近，AP ambiguous candidate 推远。
- multi-task head：MP hotspot risk、defect localization、AF focus quality、AP uniqueness。

验证标准：

- 相比 handcrafted baseline，MP coverage、AF 成功率、AP uniqueness 同时提升。

### P5：闭环与主动学习

把 CD-SEM 实际结果作为 labels 回灌，参考 online learning 和 pattern-centric manufacturing system 的思路，逐步修正风险分数和选点策略。

## 5. 对当前脚本的处理建议

1. 保留 `hotspot_recipe_notrain_backend.py` 作为 baseline feature/retrieval backend。  
   它无需训练、可解释，适合先服务 AP/AF baseline。

2. 保留 `feature_extractor_autoencoder.py`，但暂时不要继续围绕 reconstruction loss 深挖。  
   等 recipe schema 和 deterministic baseline 成型后，再升级为 task-aware AE。

3. 新增独立入口而不是改名旧入口。  
   建议新入口叫 `recipe_site_selector.py`，输出 `recipe_sites.csv`、`recipe_sites.json` 和 `recipe_review/`。

4. 把旧聚类术语降级为内部模块术语。  
   对用户和后续评审，主输出应使用 MP/AF/AP、recipe、site、addressing、autofocus，而不是 cluster/member/representative。

5. 文档和测试也应按新任务重写。  
   单元测试不应只验证 cluster 数量，而应验证：周期性 AP 被拒绝、unique AP 被选中、AF 与 MP 相似但不重叠、缺失 AF/AP 时 reject reason 清晰。

## 6. 参考文献与链接

1. Miyamoto, A.; Matsuoka, R. Automatic Generation of Imaging Sequence for CD-SEM Using Design Data. IEEJ Transactions on Electronics, Information and Systems, 2015. https://doi.org/10.1541/ieejeiss.135.444
2. Ikegami, T. et al. Evolution and Future of Critical Dimension Measurement System for Semiconductor Processes. Hitachi Review, 2011. https://www.hitachihyoron.com/rev/pdf/2011/r2011_05_104.pdf
3. Ward, B.; Page, L. Automated CD-SEM Recipe Generation Utilizing Design Pattern Layout. Hitachi High Technologies America. https://nccavs-usergroups.avs.org/wp-content/uploads/PAG2005/PEUG_04_2005_Ward.pdf
4. Tabery, C.; Page, L. Use of Design Pattern Layout for Automatic Metrology Recipe Generation. Proc. SPIE 5752, 2005. https://www.globalsino.com/ICsAndMaterials/page2360.html
5. Feng, Y. et al. Layout Pattern Analysis and Coverage Evaluation in Computational Lithography. Optics Express, 2023. https://doi.org/10.1364/OE.485206
6. Feng, Y. et al. Feature Vector Effectiveness Evaluation for Pattern Selection in Computational Lithography. Photonics, 2024. https://doi.org/10.3390/photonics11100990
7. Zheng, S. et al. Streamlining Computational Lithography With Efficient Pattern Database. IEEE TCAD, 2025. https://doi.org/10.1109/TCAD.2025.3562158
8. Chen, K.-J. et al. Minimizing Cluster Number with Clip Shifting in Hotspot Pattern Classification. DAC, 2017. https://doi.org/10.1145/3061639.3062283
9. Ding, D.; Wu, X.; Ghosh, J.; Pan, D. Z. Machine Learning based Lithographic Hotspot Detection with Critical-Feature Extraction and Classification. ICICDT, 2009. https://www.cerc.utexas.edu/utda/publications/ICICDT09_ML.pdf
10. Gao, J.-R.; Yu, B.; Ding, D.; Pan, D. Z. Lithography Hotspot Detection and Mitigation in Nanometer VLSI. ASP-DAC tutorial paper / arXiv, 2014. https://arxiv.org/abs/1402.3150
11. Yang, H.; Li, S.; Tabery, C.; Lin, B.; Yu, B. Bridging the Gap between Layout Pattern Sampling and Hotspot Detection via Batch Active Learning. IEEE TCAD, 2021. https://doi.org/10.1109/TCAD.2020.3015903
12. Gai, T. et al. Hotspot detection in large-scale layout with proposal sampling and feature parameters optimization. Journal of Micro/Nanopatterning, Materials and Metrology, 2021. https://eurekamag.com/research/099/823/099823395.php
13. Zhu, B. et al. Hotspot Detection via Multi-task Learning and Transformer Encoder. ICCAD, 2021. https://www.cse.cuhk.edu.hk/~byu/papers/C125-ICCAD2021-H-DETR.pdf
14. Shao, H.-C. et al. LithoHoD: A Litho Simulator-Powered Framework for IC Layout Hotspot Detection. IEEE TCAD, 2025. https://doi.org/10.48550/arXiv.2409.10021
15. Ding, D.; Torres, J. A.; Pan, D. Z. High Performance Lithography Hotspot Detection With Successively Refined Pattern Identifications and Machine Learning. IEEE TCAD, 2011. https://doi.org/10.1109/TCAD.2011.2164537
16. Zhang, H.; Yu, B.; Young, E. F. Y. Enabling Online Learning in Lithography Hotspot Detection with Information-Theoretic Feature Optimization. ICCAD, 2016. https://doi.org/10.1145/2966986.2967032
17. Sun, S. et al. Efficient Hotspot Detection via Graph Neural Network. DATE, 2022. https://doi.org/10.23919/DATE54114.2022.9774579
18. Jiang, C. et al. LithoExp: Explainable Two-stage CNN-based Lithographic Hotspot Detection with Layout Defect Localization. ACM TODAES, 2025. https://doi.org/10.1145/3721129
19. Zhang, X. et al. Enhanced Lithographic Hotspot Detection via Multi-Task Deep Learning With Synthetic Pattern Generation. IEEE Open Journal of the Computer Society, 2025. https://doi.org/10.1109/OJCS.2024.3510555
20. Wang, Y. et al. SPPE-GAN: A novel model for Die-to-Database alignment and SEM distortion correction framework. Expert Systems with Applications, 2025. https://doi.org/10.1016/j.eswa.2025.126658
21. Hu, C. et al. Pattern-Centric Computational System for Logic and Memory Manufacturing and Process Technology Development. Journal of Microelectronic Manufacturing, 2020. https://doi.org/10.33079/jomm.20030410
22. Siemens EDA. Calibre SONR feature-vector driven full-chip ML platform. https://eda.sw.siemens.com/en-US/ic/calibre-manufacturing/fab-solutions/calibre-sonr/
