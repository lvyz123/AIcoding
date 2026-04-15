# HDBSCAN Layout Clustering

`layout_clustering_hdbscan.py` 是当前 feature-based HDBSCAN 聚类版本。它面向中心窗口特征提取和全样本 HDBSCAN 聚类，适合做工程化分析、特征聚类与 review。

## 算法主线

当前流程大致为：

1. 读取 OASIS 文件或目录，默认匹配 `*.oas`。
2. 对大版图启用 `--split-layout` 时，必须指定 `--marker-layer layer/datatype`；marker bbox center 作为初始窗口 seed。
3. marker 图形只作为 seed，不进入窗口 pattern 几何、特征提取或输出窗口。
4. 对 marker seed 做 clip shifting 边界对齐微调，再构建中心矩形与外围上下文窗口。
5. 通过几何去重和 shift-cover 压缩减少 clip 位置冗余。
6. 提取中心窗口内部与外围上下文的多块特征，包括 base、spatial、shape、layer、pattern、radon。
7. 使用 HDBSCAN 对全样本特征聚类，并用 medoid 选择 representative。
8. 可导出 cluster review 目录，便于人工检查结果。

## 运行方式

```bash
python layout_clustering_hdbscan.py ./input_data --output results.json
```

常用示例：

```bash
python layout_clustering_hdbscan.py ./design.oas --output results.json
python layout_clustering_hdbscan.py ./large_design.oas --split-layout --marker-layer 999/0 --clip-size 1.35 --context-width 0.675 --output results.json
python layout_clustering_hdbscan.py ./large_design.oas --split-layout --marker-layer 999/0 --min-cluster-size 8 --min-samples 4 --feature-workers 2 --output results.json
python layout_clustering_hdbscan.py ./large_design.oas --split-layout --marker-layer 999/0 --sample-similarity-threshold 0.97 --hash-precision-nm 5.0 --output results.json
```

## 关键参数

- `input`: 输入文件或目录路径。
- `--output`: 输出文件路径，默认 `clustering_results.json`。
- `--format`: `json` 或 `txt`，默认 `json`。
- `--export-cluster-review-dir`: 导出 review 目录。
- `--split-layout`: 对大版图启用中心窗口采样。
- `--marker-layer`: marker 层，格式 `layer/datatype`；`--split-layout` 时必填。
- `--hotspot-layer`: `--marker-layer` 的兼容别名。
- `--clip-size`: 中心矩形边长，默认 `1.35um`。
- `--context-width`: 外围上下文宽度，默认 `0.675um`。
- `--min-cluster-size`: HDBSCAN 最小簇大小，默认 `8`。
- `--min-samples`: HDBSCAN `min_samples`，默认 `4`。
- `--sample-similarity-threshold`: 几何去重相似度阈值，默认 `0.96`。
- `--max-elements-per-window`: 每个窗口保留的最大局部几何元素数，默认 `256`。
- `--hash-precision-nm`: 增强顶点哈希量化精度，默认 `5.0nm`。
- `--feature-workers`: 特征提取并行进程数，默认 `2`。
- `--pattern`: 目录输入时的文件匹配模式，默认 `*.oas`。
- `--apply-layer-ops` / `--register-op`: 可选层操作预处理。

普通帮助只展示常用参数。仍在当前流程中生效的高级参数可隐藏解析，用于复现实验或内部调试；已断开的旧 seed/bin 参数不再作为入口保留。

## 输出

输出 JSON 包含聚类、代表样本、样本元数据、特征配置与 review 信息。HDBSCAN 版本的 representative 是基于特征空间相似度选择的 medoid 样本。

指定 `--export-cluster-review-dir` 时，脚本会按聚类结果导出 review 目录，并把 JSON 中的 sample 和 representative 路径 remap 到 review 目录里的实际拷贝文件。

## 依赖

```bash
pip install -r requirements.txt
```

主要依赖包括 `gdstk`、`rtree`、`numpy`、`scipy`、`scikit-learn`、`hdbscan`，并在可用时使用 `scikit-image` 的 radon 特征。
