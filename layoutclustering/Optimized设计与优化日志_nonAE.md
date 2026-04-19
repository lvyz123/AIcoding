**结论**
我认为有，而且最值得做的备用路线不是把旧 HDBSCAN 特征原样搬回来，而是做一个**非深度训练的 behavior-aware handcrafted FV extractor**：

```text
aerial / PV / EPE / NILS image
-> FFT/DCT 低频谱特征
-> gradient/HOG/Radon/edge/NILS/PV 统计
-> 少量 layout geometry/topology 特征
-> robust normalize
-> 输出 features.npz
-> 继续走现有 ANN + facility location + k-center + SSIM final verification
```

也就是说：AE 失败时，主流程不必退回“几何聚类”；仍然可以保持“光刻行为 coverage clustering”，只是 FV 从 learned AE latent 换成 deterministic handcrafted FV。

**旧 HDBSCAN 特征的位置**
旧版 [layout_clustering_hdbscan.py](C:/Users/81932/Documents/AIcoding/layoutclustering/layout_clustering_hdbscan.py):2198 的 `LayoutFeatureExtractor` 大致包含：

- 全局几何：polygon/path/label 数、总面积、总周长、bbox、density。
- 空间分布：4x4 grid density。
- 形状复杂度：周长面积比、面积分位数、compactness。
- 层间统计：layer area ratio、alignment、distance。
- pattern 统计：repetition、symmetry、regularity、local density variation。
- Radon 投影特征。
- 另有窗口 invariant / signature：例如 [layout_clustering_hdbscan.py](C:/Users/81932/Documents/AIcoding/layoutclustering/layout_clustering_hdbscan.py):623 和 [layout_clustering_hdbscan.py](C:/Users/81932/Documents/AIcoding/layoutclustering/layout_clustering_hdbscan.py):736。

它的优点是快、稳定、无需训练；缺点也明确：主要描述的是**几何相似**，不是**光刻行为相似**。作为 fallback 可以保留一小块，但不应作为核心替代 AE。

**更推荐的备用特征路线**
1. **Aerial-image FFT/DCT spectral FV，最推荐**

这是最接近 AE 论文路线的非训练替代。Feng 2024 明确比较了 AE-based FV 和 FFT-based FV，结论是 AE 更强，但 FFT FV 是被认真验证过的 baseline；论文还指出 aerial image FV 能保留光学和几何关键信息，适合 pattern selection/coverage。见 [Feng 2024](https://www.mdpi.com/2304-6732/11/10/990)。

为什么适合你当前版本：

- 你已经假设 aerial image 一定有。
- 现有主脚本只需要 `features.npz`，不关心 FV 来自 AE 还是别的。
- FFT/DCT 与光刻成像的频域本质更贴近，明显优于纯 layout density。
- 无需训练，只有固定变换和归一化。

建议特征块：

```text
aerial image:
  DCT low-frequency zigzag coefficients
  FFT radial power spectrum
  FFT angular spectrum
  gradient magnitude/orientation histogram
  center/ring energy ratio
  multi-scale downsample density
```

DCT 也在 Litho-NeuralODE 2.0 中被用作 layout image 压缩特征，说明这类谱域压缩在 hotspot 任务里是可行路线，见 [Litho-NeuralODE 2.0](https://www.sciencedirect.com/science/article/abs/pii/S0167926022000244)。FFT hotspot detector 也有专门工作，强调 FFT 能把大规模 layout 压成更小的多维表示，同时保留判别性 pattern 信息，见 [He et al. FFT-based feature extraction](https://www.ivysci.com/articles/2736981__Lithography_Hotspot_Detection_with_FFTbased_Feature_Extraction_and_Imbalanced_Learning_Rate)。

2. **Simulation-response scalar/field FV，优先级同样很高**

如果 manifest 里有 `pv/epe/nils/resist`，这些比 layout 几何更接近 OPC 团队真正关心的风险。Feng 2023 提到 coverage metric 可以基于 intrinsic numerical feature 或 potential model simulation behavior，并且 simulation-error incremental selection 能显著降低 model verification error range，见 [Feng 2023](https://pubmed.ncbi.nlm.nih.gov/36859995/)。

建议特征块：

```text
EPE:
  abs mean / max / q90 / q99
  signed mean / std
  connected high-error area count

PV:
  PV band area
  PV band width histogram
  overlap/IoU-like scalar
  high PV region count

NILS:
  mean / min / q10 along critical edges
  low-NILS area ratio
  gradient/slope histogram

resist:
  contour distance transform stats
  bridge/pinch/open proxy stats
```

这类特征未必适合单独做 final verification，但非常适合做 ANN retrieval 和 representative coverage selection。

3. **Topological classification + critical feature extraction**

Yu/Lin/Jiang/Chiang 的 “topological classification and critical feature extraction” 是非深度训练路线里最相关的 hotspot 文献之一。它用 topological classification 与 critical feature extraction，目标就是减少 false alarm 并提升 hotspot detection；摘要称其优于 ICCAD 2012 contest winner。见 [TCAD 2015 entry](https://ir.lib.nycu.edu.tw/handle/11536/124532)。

对当前场景的价值：

- 很适合作为 layout 几何侧的补充。
- 比旧 HDBSCAN 的全局统计更强调 critical geometry。
- 但它仍然主要是 layout-domain，不应压过 aerial/PV/NILS behavior features。

建议实现为：

```text
critical spacing bins
critical line-end / corner / jog counts
min spacing / min width / notch-like geometry
edge-pair orientation histogram
polygon adjacency / nearest-neighbor distance histogram
```

4. **Polygon graph / WL graph-kernel signature**

如果希望保留 SCP 里“拓扑/图签名”的味道，但不训练 GNN，可以用 polygon adjacency graph + Weisfeiler-Lehman subtree counts。WL graph kernels 的核心优点是用迭代 relabeling 抽取拓扑/label 信息，复杂度可做到接近边数线性，见 [Shervashidze et al. 2011](https://is.mpg.de/ei/publications/shervashidzesvmb2011)。

适合描述：

```text
polygon node: layer, area bin, aspect bin, orientation bin
edge: distance bin, overlap/touch relation, direction bin
WL iterations: 2-3 rounds
feature: subtree label count vector / hashed count vector
```

它的作用不是替代 aerial features，而是帮助区分“看起来频谱相近但拓扑关系不同”的 layout。

5. **Classical image descriptors：HOG / LBP / moments / Radon**

这些不是 lithography 专用，但作为低成本 feature blocks 有价值：

- HOG 描述边缘方向结构，经典来源是 [Dalal & Triggs 2005](https://www.researchgate.net/publication/319770820_Histograms_of_Oriented_Gradients_for_Human_Detection)。
- LBP 描述局部纹理，计算快，对灰度单调变化鲁棒，见 [Ojala et al. 2002](https://colab.ws/articles/10.1109%2FTPAMI.2002.1017623)。
- Hu/Zernike moments 可作为形状 invariant，但对复杂多 polygon layout 表达力有限。
- Radon 已经在旧 HDBSCAN 中用过，可以保留。

我的看法：这些适合做“补充 block”，不要做主干。主干应该是 aerial/PV/EPE/NILS + FFT/DCT。

**与 AE 的取舍**
- AE 优势：能学习非线性 manifold，Feng 2024 也显示 AE 尤其是 domain-aware AE 比 FFT 更会抓 outlier。
- AE 风险：训练依赖数据质量、loss 设计、训练稳定性；失败时会拖住整个 flow。
- Handcrafted FV 优势：确定性、可解释、无需训练、容易 debug；适合做 production fallback。
- Handcrafted FV 风险：非线性表达力不如 AE，coverage space 可能更粗，selected representative 数可能变多。

但对你当前系统来说，final correctness 不是靠 FV 保证，而是靠 behavior SSIM verification 保证。所以 fallback FV 的职责主要是**候选召回和 coverage ordering**。在这个职责下，`aerial DCT/FFT + PV/EPE/NILS stats + layout topology` 是一个很扎实的备用方案。

**我建议的备用版本**
新增独立脚本，例如：

```bash
python layout_clustering_feature_fallback.py encode \
  --manifest behavior.jsonl \
  --features-out features_fallback.npz \
  --method spectral_behavior
```

输出仍然是：

```text
sample_ids
features
```

这样 [layout_clustering_optimized.py](C:/Users/81932/Documents/AIcoding/layoutclustering/layout_clustering_optimized.py) 不需要大改，只是 `--feature-npz` 换成 fallback 生成的 NPZ。

推荐默认 feature block：

```text
aerial_dct_lowfreq        128-256 dims
aerial_fft_radial/angular 48-96 dims
aerial_gradient_hog       64-128 dims
pv/epe/nils/resist stats  20-80 dims, 按可用 channel 加入
layout_density/radon      32-64 dims
layout_topology/WL hash   64-128 dims
```

总维度控制在 `256-512` 比较合适。后面仍然交给 ANN top-K + weighted facility location + k-center 补洞，final verification 保持 SSIM/weighted behavior score。

**优先级建议**
第一优先级：`aerial DCT/FFT + gradient/HOG + radial/angular spectrum`。这是最像 AE FV 的无训练替代。

第二优先级：加入 `PV/EPE/NILS/resist` 的统计特征。它们更贴近 hotspot risk，尤其适合 high-risk hole 补洞。

第三优先级：移植一小部分旧 HDBSCAN layout 特征和 topology/WL signature。它们用于解释、去重和补充分辨率，不做主语义。

第四优先级：只在需要时加入 LBP/Hu/Zernike 等 classical descriptors。收益可能有，但不如前面三类直接。














已按计划实现 no-training 备用版，核心路径现在是：

`layout_clustering_optimized_notrain.py` 自动生成 handcrafted FV，然后复用现有 `OptimizedMainlineRunner` 完成 ANN top-K、facility location、k-center、behavior final verification 和 export。

新增内容：

- [feature_extractor_handcraft.py](C:/Users/81932/Documents/AIcoding/layoutclustering/feature_extractor_handcraft.py)
  - `encode` 子命令
  - 输出兼容 AE 的 `features.npz`
  - 包含 aerial DCT/FFT、radial/angular spectrum、HOG/gradient
  - 可选 `pv/epe/nils/resist` 统计特征
  - layout geometry/critical proxy
  - polygon bbox graph + WL hashed signature
  - block weight + L2 normalize + metadata JSON

- [layout_clustering_optimized_notrain.py](C:/Users/81932/Documents/AIcoding/layoutclustering/layout_clustering_optimized_notrain.py)
  - 不需要 `--feature-npz`
  - 自动生成 `_temp_runs/.../handcraft_features.npz`
  - 输出 `pipeline_mode: optimized_notrain`
  - 记录 `feature_source: handcraft` 和 handcrafted feature metadata

- [layout_clustering_optimized.py](C:/Users/81932/Documents/AIcoding/layoutclustering/layout_clustering_optimized.py)
  - 只做最小兼容扩展：支持 config 传入 `pipeline_mode`、`feature_source`、`feature_metadata`
  - 默认行为仍是 `optimized_behavior`

- [test_handcraft_features.py](C:/Users/81932/Documents/AIcoding/layoutclustering/test_handcraft_features.py)
  - 覆盖 feature NPZ 输出、optional PV block、WL/topology 差异、no-train 主流程、CLI 不暴露 `--feature-npz`

- [requirements.txt](C:/Users/81932/Documents/AIcoding/layoutclustering/requirements.txt)
  - 补充 `scipy>=1.10.0`

已验证：

```text
python -m py_compile layoutclustering\layout_clustering_optimized.py layoutclustering\layout_clustering_optimized_notrain.py layoutclustering\feature_extractor_handcraft.py layoutclustering\test_handcraft_features.py layoutclustering\test_optimized_clustering.py
python layoutclustering\feature_extractor_handcraft.py encode --help
python layoutclustering\layout_clustering_optimized_notrain.py --help
python -m unittest test_optimized_clustering -v
python -m unittest test_handcraft_features -v
git diff --check -- layoutclustering\layout_clustering_optimized.py layoutclustering\layout_clustering_optimized_notrain.py layoutclustering\feature_extractor_handcraft.py layoutclustering\test_handcraft_features.py layoutclustering\requirements.txt
```

结果：`test_optimized_clustering` 11 个通过，`test_handcraft_features` 5 个通过。

我没有动当前工作区里已有的其它改动，包括 `.idea`、`layout_clustering_autoencoder.py` 删除、`feature_extractor_autoencoder.py`、两份优化日志等。


















# `preprocess_notrain.py` 预处理脚本计划

## Summary
是的：对 no-train 主流程来说，带 marker layer 的 layout 加上 `behavior.jsonl` 和每个 marker 对应的 `aerial_npz` 就足够了。新增 `preprocess_notrain.py`，把常见图像格式和 DM3 aerial image 批量转换为 no-train 需要的 NPZ，并按文件名匹配 marker，生成 `behavior.jsonl`。

## Key Changes
- 新增独立脚本 `layoutclustering/preprocess_notrain.py`：
  - 输入：OAS/OASIS、`--marker-layer`、`--aerial-dir`、`--output-dir`。
  - 输出：
    - `behavior.jsonl`
    - `aerial_npz/<marker_id>.npz`
    - `preprocess_summary.json`
  - marker id 复用当前规则：`<oas_stem>__marker_000000`。
  - aerial 文件名匹配规则固定为：
    - 优先匹配完整 marker id，例如 `sample_layout_002__marker_000123.png`
    - 其次匹配 `marker_000123` / `000123`
    - 每个 marker 必须唯一匹配一张 aerial；缺失或重复直接报错。
- 支持图像格式：
  - 常见格式：`.png .jpg .jpeg .tif .tiff .bmp`，用已安装的 Pillow / tifffile 读取。
  - 数组格式：`.npy .npz`，NPZ 优先读 `image` key。
  - DM3/DM4：第一版作为必须支持项，采用懒加载可选依赖，优先 `ncempy.io.dm`；未安装时给出明确错误，提示安装 `ncempy`。
- 图像转换规则：
  - RGB/RGBA 转灰度 luminance。
  - 多页 TIFF 默认取第一页。
  - 3D DM3 若是单通道则 squeeze；若仍是 3D，报错并提示用户先指定/导出二维 aerial plane。
  - 输出统一为 2D `float32`，NPZ key 固定为 `image`。
  - 默认做 finite check；NaN/Inf 报错。
  - 默认 min-max normalize 到 `[0, 1]`，提供 `--no-normalize` 关闭。
- `behavior.jsonl` 每行字段：
  - `sample_id`
  - `source_path`
  - `marker_id`
  - `clip_bbox`
  - `aerial_npz`
  - 可选 `risk_score`，默认 `0.0`
- CLI 示例：
  ```bash
  python layoutclustering/preprocess_notrain.py \
    layoutgenerator/out_oas/sample_layout_002.oas \
    --marker-layer 999/0 \
    --aerial-dir aerial_images \
    --output-dir notrain_inputs
  ```

## Implementation Details
- 直接复用 `layout_utils.MarkerRasterBuilder` 收集 marker records，只用它生成 marker id、clip bbox 和 marker 顺序，不额外改 layout。
- `clip_bbox` 使用 `MarkerRecord.clip_bbox`，保证与 no-train 主脚本重新采样时一致。
- 文件名匹配时只扫描 aerial 目录下支持格式；默认递归扫描，提供 `--no-recursive` 关闭。
- 输出目录结构固定：
  ```text
  output_dir/
    behavior.jsonl
    preprocess_summary.json
    aerial_npz/
      sample_layout_002__marker_000000.npz
      sample_layout_002__marker_000001.npz
  ```
- `preprocess_summary.json` 记录：
  - input layout
  - marker layer
  - marker count
  - matched aerial count
  - supported suffixes
  - normalization mode
  - image shape min/max/unique shape count
  - missing/duplicate preview if failed
- 可选新增到 `requirements.txt`：
  - `ncempy>=1.11.0`，用于 DM3/DM4。
  - 不把 `hyperspy` 作为默认依赖，避免过重。

## Test Plan
- 新增 `test_preprocess_notrain.py`：
  - 构造小 OAS，生成 2-3 个 marker，按 `marker_000000.png` 文件名匹配成功。
  - 生成的 `behavior.jsonl` 能被 `layout_clustering_optimized_notrain.py` 读取。
  - 输出 NPZ 包含 `image` key，dtype 为 `float32`，shape 为 2D。
  - RGB PNG 正确转灰度。
  - `.npy` / `.npz` 输入正确转成 aerial NPZ。
  - 缺失 marker image 报错，并列出缺失 marker 预览。
  - 重复匹配同一 marker 报错。
  - `--no-normalize` 保留原始 float 值范围。
- 回归命令：
  ```bash
  python -m py_compile layoutclustering/preprocess_notrain.py
  python -m unittest test_preprocess_notrain -v
  python layoutclustering/preprocess_notrain.py --help
  ```
- 手动冒烟：
  ```bash
  python layoutclustering/preprocess_notrain.py sample_layout_002.oas \
    --marker-layer 999/0 \
    --aerial-dir aerial_images \
    --output-dir notrain_inputs

  python layoutclustering/layout_clustering_optimized_notrain.py sample_layout_002.oas \
    --marker-layer 999/0 \
    --behavior-manifest notrain_inputs/behavior.jsonl \
    --output results_notrain.json
  ```

## Assumptions
- aerial 图像文件名能包含完整 marker id、`marker_000123`，或裸六位编号 `000123`。
- 每个 marker 必须有且只能有一张 aerial image。
- aerial image 已经是与 marker clip 对应的二维图像；脚本只做格式转换和 manifest 对齐，不做物理坐标配准、不裁剪大图、不做仿真。
- 第一版只生成必需的 `aerial_npz` 和 `behavior.jsonl`；`pv/epe/nils/resist` 后续可按同样模式扩展。
