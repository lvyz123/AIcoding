# OPC任务：
## 1. Pattern grouping：
**任务描述**：自动搜索合适的中心点切clip，然后把切出的layout clip进行聚类分组

**应用场景**：
1）通过有限点位的量测full chip的表征，这一点对weak point检测很关键。
2）利用AI减少辅助OPC建模，减少模型计算过程中的冗余，在保证模型pattern覆盖率的同时提升计算效率。

**任务目标**：实现full chip重复pattern的有效抓取与分组，抓取尽量多的重复pattern（recall），并准确地聚类（precision）

**算法方案**：
1）自动layout clipping：自动seed+分bin+shift-cover确定潜在hotspot位置并切片
2）聚类算法流程：精确哈希聚类去重 -> 预筛选器（invariant score/topo distance/graph signature similarity）聚类去重 -> 候选clip生成 -> ACC/ECC容差匹配 -> 集合覆盖（聚类数最小化）

**交付节点**：7月完成算法方案的交付

**当前使用版图**：tolyu_test2.oas（550um x 550um）