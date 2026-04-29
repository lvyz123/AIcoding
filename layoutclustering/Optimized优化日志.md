太强了！结合你提供的代码，我对这个 v2_lsf 版本进行了一次深度的 Code Review。结论是：**你极其精准、完美地满足了之前提出的 5 点重构要求，这是一次教科书级别的大规模分布式（LSF）架构重构与向下兼容降级。**

你不仅完成了任务，还做出了几个极具前瞻性的架构设计（比如引入 CSR 稀疏矩阵格式），让这个算法真正具备了在工业级 HPC（高性能计算）集群上跑 Full-chip 数据的能力。

以下是对你 5 点要求的具体评估结果，以及下一阶段的演进建议：

### 一、 5 点核心要求评估结果：完美执行 (100% Pass)

**1. LSF 架构适配 (Pass 🌟)**
* **评估：** 你将原本串行的流程漂亮地解耦成了典型的 **MapReduce 范式**：`prepare` (Map分发) -> `run-shard` / `run-coverage-shard` (Worker并行计算) -> `merge` / `merge-coverage` (Reduce汇总)。并且通过 `manifest.json` 极好地管理了分布式任务的上下文，这在 LSF / Slurm 集群运维中是最稳妥的做法。

**2. 依赖脚本隔离与 `_lsf` 后缀 (Pass 🌟)**
* **评估：** `layer_operations_lsf.py`、`layout_utils_lsf.py` 和 `mainline_lsf.py` 均已正确创建。核心逻辑被完全抽离到这些后缀脚本中，职责非常清晰。

**3. 沿用 v1 算法核心，仅做分布式改造 (Pass 🌟)**
* **评估：** v1 中最精华的 Uniform Grid、Exact Hash、ACC/ECC 像素匹配、轻量级 Descriptor 预筛选（`_cheap_bitmap_descriptor` / `_graph_bitmap_descriptor`）全部原汁原味地保留了。
* **亮点：** 为了适应分布式的 Merge，你极其聪明地在 `mainline_lsf.py` 中使用了 **CSR (Compressed Sparse Row) 稀疏矩阵格式** (`coverage_offsets`, `coverage_values`) 来存储图的覆盖关系。这意味着主节点在做最终 Greedy Set Cover 时，内存占用降低了几个数量级，彻底避免了 LSF Master 节点被 OOM（内存溢出）干掉的风险。

**4. 新旧版本完全独立 (Pass 🌟)**
* **评估：** 检查了所有的 `import` 语句。LSF 脚本内部实现了完美闭环，没有任何一行代码去引用 `_v1` 或无后缀的旧脚本。这使得 v2 可以作为一个干净的 standalone package 部署到集群节点。

**5. 严格兼容 Python 3.6 (Pass 🌟)**
* **评估：** 这一步的降级做得非常细致入微：
    * 去掉了所有的 `@dataclass`，换成了经典的 `class ... (object):`。
    * 引入了 `__slots__` 机制（如 `MarkerRecord`, `CandidateClip`），这在 Python 3.6 环境下对创建百万级小对象是非常有效的内存优化手段。
    * 去掉了 `from __future__ import annotations` 和高级类型提示（Type Hints）。
    * 去掉了所有的 `f-string`，改为了 Python 3.6 及以下绝对安全的 `%` 字符串格式化（例如 `"%dx%d" % (...)`）。
    * 去掉了 `scipy.optimize.milp`（旧版 SciPy 不支持），只保留了最稳健的贪心算法。
    * 对 `hnswlib` 做了优雅的 `try...except ImportError` 降级处理，因为老集群节点的 C++ 编译器常常无法编译 hnswlib，回退到基于 NumPy 的 `_exact_cosine_topk_labels` 非常安全。

---

### 二、 进一步的优化建议（聚焦 LSF 集群与 OPC 业务）

既然系统已经具备了在集群上铺开算力的基础，接下来的挑战将从“单机算法性能”转移到**“分布式系统的长尾效应与 IO 瓶颈”**上。针对 Pattern Grouping 任务，建议在以下几个方向继续打磨：

#### 1. 应对“长尾效应”：基于数据密度的动态 Shard 划分
* **隐患：** 目前在 `prepare_stage` 中，你是通过简单的除法 (`shard_size = total / shard_count`) 来均匀切分 Seed 的。但是，芯片版图的特征密度是极度不均的（SRAM 区域全是密集的相同 Pattern，而走线区/Analog 区域极其稀疏）。
* **建议：** 这种按数量均匀切分会导致严重的 **“长尾现象”（Straggler Problem）**——某些 LSF Job 处理的是高度密集的复杂布线区，耗时 2 小时；而另一些 Job 处理的是大块空白区，耗时 5 分钟。
* **做法：** 在 `prepare` 阶段获取 Bounding Box 后，快速扫一次 `layout_index.spatial_index` 计算一下各个网格的图元密度（DensityWeight）。根据 DensityWeight 来动态划分 Shard 的边界，让每个 LSF Job 分配到的**“计算工作量（图元相交次数）”**大致均等，而不是**“物理面积”**均等。

#### 2. 缓解 IO 风暴 (IO Storm)：多级合并或 HDF5 替代 NPZ
* **隐患：** 在 `merge_coverage_stage` 中，你需要汇总所有的 `shard_*.json` 和 `shard_*.npz`。如果 `coverage_shard_count` 开到了 1000 以上，LSF Master 节点在同一时间疯狂读取共享存储（NAS/NFS）上的上千个文件，会引发严重的网络 IO 拥塞，甚至把文件系统打挂。
* **建议：**
    * **方案 A (结构调整):** 采用**树状合并 (Tree Merge)**。不要让主节点一次性读 1000 个 Shard。每 50 个 Shard 分配一个次级 LSF Job 做一次中间态的 CSR 数组合并，最后主节点只读 20 个大文件。
    * **方案 B (存储格式):** 如果集群支持，强烈建议放弃生成海量的散碎 `.npz`，转而使用并发写友好的 **HDF5 (`h5py`)** 或 **Zarr** 格式。所有 Worker Job 将 CSR 数组并发写入同一个 HDF5 文件的不同 dataset 中，最后主节点使用内存映射 (Memory Mapping) 读取，IO 效率会有质的飞跃。

#### 3. OPC 业务的定制化：Cluster 内部的二次对其 (Sub-pixel Alignment)
* **隐患：** 你的 v2 依然沿用了 `pixel_size_nm` (默认 10nm) 的 Raster 匹配。在前面的评审中我也提到过，这对先进工艺的 OPC Hotspot 检测存在语义丢失的风险。
* **建议：** 现在我们有了 LSF 提供的大量算力，不再像单机那样捉襟见肘。
    * 可以在最终的 `merge_coverage` 选出 Representative 之后，利用 LSF 派发最后一个极轻量的任务（例如 `refine-cluster`）：对每个归类好的 Exact Cluster，读取其原始的多边形（Polygon）数据，进行一次矢量的二次对齐和微小容差验证。
    * 这样做既享受了 Raster 带来的召回率和分布式速度，又用 Vector 保证了提交给 OPC 工程师的数据具备纳米级精度。