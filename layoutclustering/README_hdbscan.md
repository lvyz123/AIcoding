# HDBSCAN Layout Clustering

`layout_clustering_hdbscan.py` 是当前 feature-based HDBSCAN 聚类版本。它面向中心窗口特征提取和全样本 HDBSCAN 聚类，适合做工程化分析、特征聚类与 review，不是 Chen 2017 clip shifting 的默认主线实现。

## 算法主线

当前流程大致为：

1. 读取 OASIS 文件或目录，默认匹配 `*.oas`。
2. 对大版图可启用 `--split-layout`，基于候选中心生成中心矩形与外围上下文窗口。
3. 提取中心窗口内部与外围上下文的多块特征，包括 base、spatial、shape、layer、pattern、radon。
4. 对特征进行可选 `log1p` 压缩与标准化，并支持 inner/outer feature 权重。
5. 使用 HDBSCAN 对全样本特征聚类，默认 metric 为 `euclidean`。
6. 对噪声 singleton 做可选轻量后合并，并用 medoid 选择 representative。
7. 可导出 cluster review 目录，便于人工检查结果。

当前版本默认带有高性能工程优化：空间分桶、轻量几何预去重、每窗口局部几何数量上限，以及基于 clip shifting 思路的候选中心边界对齐微调。这些是 HDBSCAN 版本的工程策略，不等同于 clip shifting 主线的 set-cover representative 语义。

## 运行方式

```bash
python layout_clustering_hdbscan.py ./input_data --output results.json
```

常用示例：

```bash
python layout_clustering_hdbscan.py ./design.oas --output results.json
python layout_clustering_hdbscan.py ./large_design.oas --split-layout --clip-size 1.35 --context-width 0.675 --output results.json
python layout_clustering_hdbscan.py ./large_design.oas --split-layout --min-cluster-size 8 --min-samples 4 --feature-workers 2 --output results.json
python layout_clustering_hdbscan.py ./large_design.oas --split-layout --sample-similarity-threshold 0.97 --candidate-bin-size-um 6.0 --hash-precision-nm 5.0 --output results.json
```

## 关键参数

- `input`: 输入文件或目录路径。
- `--output`: 输出文件路径，默认 `clustering_results.json`。
- `--format`: `json` 或 `txt`，默认 `json`。
- `--export-cluster-review-dir`: 导出 review 目录。
- `--min-cluster-size`: HDBSCAN 最小簇大小，默认 `8`。
- `--min-samples`: HDBSCAN `min_samples`，默认 `4`。
- `--split-layout`: 对大版图启用中心窗口采样。
- `--clip-size`: 中心矩形边长，默认 `1.35um`。
- `--context-width`: 外围上下文宽度，默认 `0.675um`。
- `--sample-similarity-threshold`: 几何去重相似度阈值，默认 `0.96`。
- `--candidate-bin-size-um`: 候选中心点空间分桶尺寸；默认自动取值。
- `--max-elements-per-window`: 每窗口保留的最大局部几何元素数，默认 `256`。
- `--hash-precision-nm`: 增强顶点哈希量化精度，默认 `5.0nm`。
- `--feature-workers`: 特征提取并行进程数，默认 `2`。
- `--pattern`: 目录输入时的文件匹配模式，默认 `*.oas`。
- `--apply-layer-ops` / `--register-op`: 可选层操作预处理。

普通帮助只展示常用参数。旧版高级参数仍可解析，用于复现实验或内部调试，但不作为常规用户入口推荐。

## 输出

输出 JSON 会包含聚类、代表样本、样本元数据、特征配置与 review 信息。HDBSCAN 版本的 representative 是基于特征空间/相似度选择的 medoid 样本，不是 set cover 选中的 shifted candidate。

如指定 `--export-cluster-review-dir`，会按聚类结果导出 review 目录，并标记 representative，方便人工比对局部窗口。

## 依赖

安装依赖：

```bash
pip install -r requirements.txt
```

HDBSCAN 版本主要依赖 `gdstk`、`rtree`、`numpy`、`scipy`、`scikit-learn`、`hdbscan`，并在可用时使用 `scikit-image` 的 radon 特征。
