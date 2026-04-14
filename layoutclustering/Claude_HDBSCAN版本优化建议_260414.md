先系统读完关键模块。代码读完，可以给出全面评估。

---

## 整体评估：目标实现情况

你的目标是两层去重：**shift-cover 消除位置冗余 → HDBSCAN 消除语义冗余**。这个架构已经完整实现，核心流程 `_build_initial_window_records → _deduplicate_window_records → _compress_window_records_with_shift_cover → HDBSCAN` 每一层都在做正确的事，没有明显的逻辑错误。

---

## 已正确落地的技术点

**论文核心思想全部覆盖：**

| 论文技术点 | 代码对应位置 | 落地质量 |
|-----------|------------|---------|
| 精确匹配去重（Exact Clip Matching） | `_deduplicate_window_records` exact_hash 路径 | ✅ 且用 8 向 dihedral hash 处理了镜像对称，比论文更完善 |
| 面积约束匹配（ACC）| `_window_xor_ratio` + `area_match_ratio` 参数 | ✅ 用 gdstk.boolean XOR 精确计算 |
| Clip Shifting 候选生成 | `_generate_pattern_shift_candidates_for_record` | ✅ 按 pattern 边触及 clip 边界的事件点生成候选，忠实还原论文 Fig.11 |
| Set Cover 最小化 cluster 数 | `_greedy_select_cover_candidates` + heapq lazy greedy | ✅ O(C log C)，比论文 greedy 更优 |
| 权重驱动的 greedy（高频 pattern 优先） | `cluster_weights = duplicate_count` | ✅ |
| 边界约束（marker constraint）的工程替代 | anchor_bbox + `_build_shift_anchor_bbox` | ✅ 无 marker layer 时用 anchor bbox 合理近似 |

---

## 发现的 3 个真实问题

**问题 1：XOR 对跨源 pattern 存在坐标系不对齐**

`_window_xor_ratio` 对 `normalized_polygons` 做 XOR。`_normalize_polygons_to_local_bbox` 把每个 record 的 polygons 都平移到以其自身 `outer_bbox` 左下角为原点的局部坐标系。这意味着：

- record_a 的 polygons 在以 `(a_x0, a_y0)` 为原点的坐标系
- record_b 的 polygons 在以 `(b_x0, b_y0)` 为原点的坐标系

当两个窗口的 `outer_bbox` 左下角不同时，XOR 实际上是在对两个未对齐的 pattern 做布尔运算，结果偏大，误判两个 pattern 不相似。

在 compress 阶段，shifted candidate 的 `outer_bbox` 已经被移到新位置（`_build_shift_candidate_record` 里重新算了 `outer_bbox`），而 `target_record` 的 `normalized_polygons` 是旧位置的局部坐标，所以这个偏差在 shift-cover 匹配时会系统性地使 XOR 偏高，导致本应能合并的 pattern 被错误拒绝。

修复方案：两个 record 做 XOR 前，先把两者的 normalized_polygons 都平移到同一参考点（如两者 outer_bbox 中心的均值）：

```python
def _window_xor_ratio(record_a, record_b):
    # 统一到相同参考坐标系再做XOR
    cx_a, cy_a = _bbox_center(record_a["sample"].outer_bbox)
    cx_b, cy_b = _bbox_center(record_b["sample"].outer_bbox)
    ref_cx = (cx_a + cx_b) / 2.0
    ref_cy = (cy_a + cy_b) / 2.0

    def _translate_to_ref(polygons, bbox):
        ox = float(bbox[0])
        oy = float(bbox[1])
        offset = np.array([ox - ref_cx, oy - ref_cy], dtype=np.float64)
        result = []
        for poly in polygons or []:
            if poly is None or not hasattr(poly, "points"):
                continue
            pts = np.asarray(poly.points, dtype=np.float64) + offset
            result.append(gdstk.Polygon(pts, layer=getattr(poly,"layer",0),
                                        datatype=getattr(poly,"datatype",0)))
        return result

    polygons_a = _translate_to_ref(record_a["normalized_polygons"],
                                    record_a["sample"].outer_bbox)
    polygons_b = _translate_to_ref(record_b["normalized_polygons"],
                                    record_b["sample"].outer_bbox)
    # ... 后续 XOR 逻辑不变
```

**问题 2：`duplicate_count` 未传入 HDBSCAN**

shift-cover 压缩后每个 window record 携带了 `duplicate_count`（表示它代表了多少个原始 pattern），但 HDBSCAN 的 `fit_predict` 调用处没有传 `sample_weight`：

```python
labels = clusterer.fit_predict(prepared_features)  # 没有 sample_weight
```

对于一个代表了 500 个重复 pattern 的 window，它在 HDBSCAN 的 core distance 计算里和一个 `duplicate_count=1` 的 pattern 权重相同。这导致高频 pattern 在语义聚类阶段的影响力被低估。

修复很简单：

```python
weights = np.asarray(self.pattern_weights, dtype=np.float64)
weights = weights / weights.sum()  # 归一化
labels = clusterer.fit(prepared_features, sample_weight=weights).labels_
```

注意 `hdbscan` 包的 `fit` 支持 `sample_weight` 参数，但 `fit_predict` 不一定，需要用 `fit` 后取 `.labels_`。

**问题 3：compress 阶段的 shifted candidate 的 `normalized_polygons` 用的是 origin_record 的旧多边形**

在 `_build_shift_candidate_record` 里（L1496），shifted candidate 直接复用了 origin_record 的 `normalized_polygons`：

```python
"normalized_polygons": list(origin_record["normalized_polygons"]),  # ← 旧坐标
```

但 candidate 的 `outer_bbox` 已经是新位置了，这就造成 `window_area` 用的是新 bbox 面积，`normalized_polygons` 用的是旧 bbox 坐标系的 polygon，两者不匹配，XOR 结果不可靠。正确做法是重新对新位置的 `outer_polygons` 调用 `_normalize_polygons_to_local_bbox`：

```python
"normalized_polygons": _normalize_polygons_to_local_bbox(
    outer_polygons, outer_bbox  # 用新位置的 bbox
),
```

---

## 可以简化的部分

**S1：`_window_xor_ratio` 的 fallback 路径**

```python
except Exception:
    try:
        inter_polygons = gdstk.boolean(polygons_a, polygons_b, "and")
        ...
    except Exception:
        inter_area = 0.0
    area_a + area_b - 2 * inter_area
```

这个两层 try/except 的 fallback 逻辑在 bbox-proxy polygon（都是矩形）的场景下几乎不会触发，因为 gdstk.boolean 对矩形几乎不会出错。可以简化成单层 try/except，失败就返回 0.5（保守估计不相似但不完全拒绝）。

**S2：`_generate_pattern_shift_candidates_for_record` 的候选去重逻辑冗余**

函数里同时用 `seen_hashes` 和 `seen_centers` 两个 set 做去重，但在 `_compress_window_records_with_shift_cover` 的外层又有 `stable_bucket_key` + 邻域匹配做精确去重。`seen_hashes` 去重实际上是内层的粗糙版本，在候选生成阶段去重会让有价值的 shifted candidate（hash 不同但几何相似）也被过早过滤。可以只保留 `seen_centers` 去重（防止同一位置重复），去掉 `seen_hashes` 去重：

```python
# 删除 seen_hashes = {base_candidate["pattern_hash"]}
# 删除 if candidate["pattern_hash"] in seen_hashes: continue
# 删除 seen_hashes.add(candidate["pattern_hash"])
```

**S3：`_cosine_similarity_1d` 的全零向量特判**

```python
if denom <= 1e-12:
    return 1.0 if np.allclose(a, b) else 0.0
```

对于 bbox-proxy 的空窗口（全零签名），两个空签名会被判定为相似度 1.0，这会让两个空 pattern 错误地被视为完全相同而合并。改为：

```python
if denom <= 1e-12:
    return 0.0  # 无信息向量，视为不相似
```

**S4：`_build_window_record` 里 `extra_meta` 的 pop 顺序隐患**

`extra_meta.pop("stable_bucket_key", ...)` 和后续 `record.update(extra_meta)` 是安全的，但中间有多个 pop 操作，顺序依赖隐式。建议把所有 pop 操作集中到函数顶部，或者改用显式参数而非 `**extra_meta`。

---

## 总结

核心目标完全实现，两层去重架构设计正确，论文的主要思想都已融合。最值得立即修复的是**问题 1（XOR 坐标系不对齐）**和**问题 3（shifted candidate normalized_polygons 坐标错误）**——这两个问题会直接导致 shift-cover 压缩率低于理论值，让本应被合并的 pattern 残留为独立窗口，增加后续 HDBSCAN 的不必要负担。