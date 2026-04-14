# Clip Shifting Layout Clustering

`layout_clustering_clip_shifting.py` 是当前 Chen 2017 clip shifting 主线版本。它是 marker-driven、raster-first 的精度优先实现，核心算法在 `mainline.py`，入口薄壳保留在 `layout_clustering_clip_shifting.py`。

## 算法主线

当前流程固定为：

1. 从 `--hotspot-layer` 指定的 marker layer 提取 hotspot marker。
2. 以 marker 为中心生成 base clip 与 expanded window，并栅格化为 bitmap。
3. 使用 `clip_bitmap + expanded_bitmap` 做双重 exact clustering，合并完全一致或对称一致的窗口。
4. 基于 expanded bitmap 生成单方向 systematic shift candidates，只允许 `left/right/up/down` 单轴平移。
5. 按论文思路对所有候选 clip 做两两 ACC/ECC 匹配，建立 candidate 到 exact cluster 的覆盖关系。
6. 将覆盖关系转为 set cover，使用 `greedy`、`ilp` 或 `auto` 选择最终 representative candidate clip。

当前实现假定输入图形主要是 Manhattan layout。默认像素尺寸为 `10nm`，主线不再使用历史 feature clustering、HDBSCAN 或 medoid representative 语义。

## 运行方式

```bash
python layout_clustering_clip_shifting.py ./design.oas --hotspot-layer 999/0 --output results.json
```

常用示例：

```bash
python layout_clustering_clip_shifting.py ./design.oas --hotspot-layer 999/0 --matching-mode ecc --solver auto
python layout_clustering_clip_shifting.py ./design.oas --hotspot-layer 999/0 --matching-mode acc --area-match-ratio 0.96
python layout_clustering_clip_shifting.py ./design.oas --hotspot-layer 999/0 --geometry-mode fast --max-elements-per-window 512
```

## 关键参数

- `input`: 输入 OASIS 文件或目录。
- `--hotspot-layer`: 必填 marker 层，格式为 `layer/datatype`，例如 `999/0`。
- `--clip-size`: clip 边长，单位 um，默认 `1.35`。
- `--matching-mode`: `acc` 或 `ecc`，默认 `ecc`。
- `--solver`: `greedy`、`ilp` 或 `auto`，默认 `auto`。
- `--geometry-mode`: `exact` 或 `fast`；两者都是 raster 主线，`fast` 会启用更激进的窗口元素上限。
- `--pixel-size-nm`: raster 像素尺寸，默认 `10`。
- `--area-match-ratio`: ACC 面积匹配阈值，默认 `0.96`。
- `--edge-tolerance-um`: ECC 边界容差，默认 `0.02`。
- `--clip-shift-directions`: 允许的单方向平移集合，默认 `left,right,up,down`。
- `--clip-shift-boundary-tol-um`: systematic shift 边界容差，默认 `0.02`。
- `--apply-layer-ops` / `--register-op`: 可选层操作预处理。

## 输出

JSON 结果中重点字段包括：

- `pipeline_mode`: 固定为 `mainline`。
- `matching_mode`、`solver_used`、`geometry_mode`、`pixel_size_nm`。
- `result_summary.exact_cluster_count`、`candidate_count`、`selected_candidate_count`、`timing_seconds`。
- `clusters[*].representative_file`: set cover 选中的 candidate clip，不是额外 medoid。
- `clusters[*].selected_candidate_id`、`selected_shift_direction`、`selected_shift_distance_um`。
- `clusters[*].exact_cluster_ids` 与 `clusters[*].marker_ids`。

如指定 `--export-cluster-review-dir`，会按 cluster 导出 review 文件夹，包含各簇 sample 与 representative。

## 依赖与测试

安装依赖：

```bash
pip install -r requirements.txt
```

运行回归测试：

```bash
python test_clustering.py
```

当前测试覆盖 marker fail-fast、exact merge、层级展开、中文路径读写、raster hash、单轴 shift、ACC/ECC 位图匹配以及候选 clip 两两 coverage。
