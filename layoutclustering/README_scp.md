# SCP Layout Clustering

`layout_clustering_scp.py` 是当前 Liu 2025 marker-driven closed-loop SCP 聚类版本。它从 marker layer 构造局部 pattern item，通过稀疏相似图、surprisal lazy greedy set cover 和 closed-loop refinement 得到最终聚类。

## 算法主线

当前流程大致为：

1. 读取 OASIS 文件或目录，目录输入默认匹配 `*.oas`。
2. 从 `--marker-layer` 获取 marker，并在 `--design-layer` 上提取 marker-centered pattern window。
3. 使用 pattern bbox、图形不变量、签名网格、投影签名等信息构造 pattern item。
4. 基于多阶段 pruning、dual-backend alignment 与 sparse similarity graph 构建 coarse graph。
5. 使用 surprisal lazy greedy SCP 求解 coarse cover。
6. 在有限轮 `closed-loop refinement` 中验证/细化合并关系，输出最终 cluster。

当前 SCP 版本只保留这条 marker-driven 论文主线，不包含旧框架的候选中心导出、feature clustering 或 review/export 后处理链。默认使用 bbox proxy 裁剪 design 几何来近似生成局部 pattern。

## 运行方式

```bash
python layout_clustering_scp.py ./design.oas --marker-layer 999/0 --design-layer 1/0 --output results.json
```

常用示例：

```bash
python layout_clustering_scp.py ./input_dir --pattern "*.oas" --marker-layer 999/0 --design-layer 1/0 --output results.json
python layout_clustering_scp.py ./design.oas --marker-layer 999/0 --design-layer 1/0 --pattern-radius 1.35 --similarity-threshold 0.96
python layout_clustering_scp.py ./design.oas --marker-layer 999/0 --design-layer 1/0 --max-iterations 3 --workers 4
```

## 关键参数

- `input`: 输入 OASIS 文件或目录。
- `--output`: 输出文件路径，默认 `clustering_results.json`。
- `--format`: `json` 或 `txt`，默认 `json`。
- `--pattern`: 目录输入时的文件匹配模式，默认 `*.oas`。
- `--design-layer`: design layer，格式 `layer/datatype`，默认 `1/0`。
- `--marker-layer`: marker layer，格式 `layer/datatype`，默认 `2/0`。
- `--pattern-radius`: marker-centered pattern 半径，单位 um，默认 `1.35`。
- `--similarity-threshold`: closed-loop 相似度阈值，默认 `0.96`。
- `--max-iterations`: closed-loop 最大迭代轮数，默认 `3`。
- `--workers`: graph/refinement 并行 worker 数，默认 CPU 核心数。

## 输出

JSON 输出包含：

- `summary`: 输入规模、cluster 数、singleton 数、设计层/marker 层、相似度阈值、迭代信息等摘要。
- `clusters`: 每个最终聚类的 `cluster_id`、`size`、representative item、member item 信息。

TXT 输出会写出简要 summary 与每个 cluster 的大小、代表样本和成员概览。

## 依赖

安装依赖：

```bash
pip install -r requirements.txt
```

SCP 版本主要依赖 `gdstk`、`rtree`、`numpy` 与 `scikit-image` 的 polygon rasterization。
