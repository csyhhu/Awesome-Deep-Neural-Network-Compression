# ZETA: Leveraging Z-order Curves for Efficient Top-k Attention

- **论文链接**: [arXiv:2501.14577](https://arxiv.org/abs/2501.14577)
- **作者**: Qiuhao Zeng, Jerry Huang, Peng Lu, Gezheng Xu, Boxing Chen, Charles Ling, Boyu Wang
- **机构**: University of Western Ontario, Université de Montréal / Mila, Noah's Ark Lab, Vector Institute
- **发表**: ICLR 2025

---

## 一、核心动机

Transformer 的自注意力机制在序列长度 $N$ 增加时，计算和内存复杂度以 $\mathcal{O}(N^2)$ 增长，难以处理长序列。现有的 top-$k$ attention 方法虽然通过仅选择最相关的 $k$ 个 token 降低了复杂度，但在因果掩码（causal mask）条件下不能高效并行处理整个序列，严重限制了训练效率。

**核心问题**: 因果掩码要求当前查询只能关注过去的 token，现有的 top-$k$ attention 方法（如 Reformer、IceFormer）只能逐 token 串行处理，无法充分发挥 GPU 的并行计算能力。

---

## 二、方法概述：ZETA

ZETA（**Z**-order Curves for **E**fficient **T**op-$k$ **A**ttention）的核心思想是利用 $Z$ 阶曲线（$Z$-order curves / Morton codes）将高维的 key 和 query 映射到一维空间，在保持局部性（locality）的同时实现并行排序和邻近 token 检索。

### 2.1 关键创新

| 创新点 | 描述 |
|--------|------|
| **维度分离** | $d_K = d_Q \ll d_V$，即 key/query 维度远小于 value 维度 |
| **$Z$ 阶曲线映射** | 将低维 key/query 映射到一维整数，保持空间局部性 |
| **并行分块检索** | 排序后分块，并行执行二分查找和窗口最近邻检索 |
| **自适应 Cauchy Softmax** | 用可训练的 Cauchy 核替代指数函数，适配欧氏距离度量 |

### 2.2 Key/Query 维度选择的理论分析

论文基于 **Johnson–Lindenstrauss 引理**证明：通过随机投影将高维数据映射到低维空间时，能近似保持成对距离。

**核心定理（Theorem 1）**：在 Lipschitz 条件下，一层注意力模型的期望风险上界为：

$$\mathbb{E}_S [L_{\mathcal{D}}(h_{\mathrm{attn}})] \leq L_{\mathcal{D}}(h^*) + \frac{4lc \sqrt{d_K} B \cdot m^{-1/(d_K+1)}}{\sqrt{1-\sqrt{\frac{C\ln m}{d_K}}}}$$

该定理揭示了 $d_K$ 选择的核心权衡：
- $d_K$ 越大 → 第二项（泛化误差）越大，受维度诅咒影响
- $d_K$ 越小 → $\epsilon$ 项增大，局部性保持变差
- **$d_K$ 不能简单设为等于 $d_V$**，需要精心选择

实验表明 $d_K = 3$ 即可在保持性能的同时显著降低计算量。

### 2.3 一维空间 Top-k 检索流程

$$
\mQ_z = Z\text{-order}(\mQ), \quad \mK_z = Z\text{-order}(\mK)
$$

具体步骤：
1. **$Z$ 阶映射**: 将 $d_K$ 维 key/query 通过位交错映射为一维整数
2. **排序**: 对一维 key 序列并行排序（基数排序，$\mathcal{O}(N)$）
3. **分块**: 将排序后的 key 分为多个 chunk（大小 $M$）
4. **分块因果掩码**: 第 $m = \lfloor i/M \rfloor$ 个 chunk 的 query 仅在前 $m$ 个 chunk 中搜索
5. **最近邻检索**: 对每个 query，二分查找插入位置，在窗口 $K$ 内收集 $k$ 个最近邻居

### 2.4 Adaptive Cauchy Softmax

由于 $Z$ 阶曲线保持的是欧氏距离而非内积相似性，传统的指数 Softmax 不再适用。ZETA 提出 **Adaptive Cauchy Softmax**：

$$\text{softmax}_c(\vq, \mK) = \frac{\left[\|\vq - \mK_i\|^2 + \gamma^2\right]^{-1}}{\sum_{j\in I_q} \left[\|\vq - \mK_j\|^2 + \gamma^2\right]^{-1}}$$

其中 $\gamma$ 是**可训练参数**（通过 sigmoid 约束到 $[0,1]$），每个注意力层独立学习：
- $\gamma$ 小 → 注意力更尖锐，聚焦最相关 token
- $\gamma$ 大 → 注意力更平滑，捕获长距离依赖

**Cauchy 核的优势**：
- 重尾分布，即使距离较远的 token 也能保持不可忽略的影响力
- 与欧氏距离度量天然对齐
- 防止熵崩溃或熵爆炸

---

## 三、复杂度分析

| 操作 | 复杂度 | 说明 |
|------|--------|------|
| $Z$ 阶映射 | $\mathcal{O}(N d_K)$ | 位交错操作 |
| 基数排序 | $\mathcal{O}(N)$ | 一维空间排序 |
| Top-$k$ 检索 | $\mathcal{O}(N \log N)$ | 每个 query 二分查找 |
| 稀疏注意力 | $\mathcal{O}(N k)$ | 仅 top-$k$ token 参与计算 |
| **总体** | **$\mathcal{O}(N \log N)$** | vs 标准注意力的 $\mathcal{O}(N^2)$ |

---

## 四、实验与结果

### 4.1 MQAR（多查询联想记忆）

在合成任务 MQAR 上，ZETA 与标准 Transformer 性能相当，在高维度（256）时达到 100% 准确率，显著优于 Performer。

### 4.2 Long Range Arena (LRA)

| 模型 | ListOps | Text | Retrieval | Image | Pathfinder | **平均** |
|------|---------|------|-----------|-------|------------|----------|
| Vanilla Transformer | 36.37 | 64.27 | 57.46 | 42.44 | 71.40 | 54.39 |
| H-Transformer-1D | **49.53** | **78.69** | 63.99 | 46.05 | 68.78 | 61.41 |
| Hedgehog | 37.15 | 64.60 | **82.24** | 40.15 | 74.16 | 59.66 |
| **ZETA** | 42.52 | 64.52 | 77.92 | **64.39** | 68.20 | **63.51** |

- ZETA 以 **63.51%** 的平均准确率在所有 attention-based 模型中排名第一
- 尤其在 Image 任务上大幅领先（64.39% vs 第二名的 47.38%）

### 4.3 WikiText-103 语言建模

| 模型 | 参数量 | Test PPL |
|------|--------|----------|
| Vanilla Transformer | 125M | 26.2 |
| Performer | 125M | 26.8 |
| Reformer | 125M | 25.6 |
| CosFormer | 125M | **23.1** |
| **ZETA** | 124M | 26.3 |

ZETA 在语言建模上达到与标准 Transformer 相当的性能（26.3 vs 26.2），参数更少（124M vs 125M）。

### 4.4 效率基准测试

在 NVIDIA A100 上（Triton 实现）：

| 序列长度 | Torch Attn FWD | Flash Attn FWD | **ZETA FWD** |
|----------|----------------|----------------|--------------|
| 4096 | 44.3 ms | 3.4 ms | **5.6 ms** |
| 8192 | OOM | 12.8 ms | **11.0 ms** |
| 16384 | OOM | 50.4 ms | **21.7 ms** |
| 32768 | OOM | 198.2 ms | **43.0 ms** |
| 65536 | OOM | 805.3 ms | **85.8 ms** |

- 序列越长，ZETA 相对于 Flash Attention 的优势越明显
- 前向+反向传播同样显著优于 Flash Attention
- 内存消耗略高于 Flash Attention 但仍远低于标准注意力

### 4.5 消融实验

1. **$d_K$ 维度影响**：$d_K \geq 2$ 时 MQAR 性能接近完美，$d_K=1$ 时轻微下降
2. **Softmax 变体**：Cauchy Softmax 在所有 $d_K$ 值下均优于 Negative Euclidean 和 Inverse Euclidean
3. **$Z$ 阶曲线局部性保持**：$d_K$ 越小且样本数越少时，局部性保持越好
4. **Top-$k$ 的 $k$ 值**：$k$ 在 16-48 范围内对模型性能影响不大，$k=32$ 为最优平衡点

---

## 五、关键技术细节

### 5.1 与历史信息融合

为避免 top-$k$ 稀疏注意力导致大量 token 的梯度无法回传，ZETA 将历史 token 的均值向量附加到 top-$k$ tokens 矩阵中（通过 cumsum 实现），类似于 n-gram 语言模型中的平滑技巧。

### 5.2 Triton 优化实现

- 使用 Triton JIT 编译器编写定制 GPU kernel
- 融合前向/反向传播 kernel，减少 I/O 开销
- 利用 `@triton.autotune` 自动调优 block_size 和 num_warps
- 在 kernel block 内计算历史均值向量，避免全局归约开销

---

## 六、局限性与讨论

1. **信息丢失风险**：作为 top-$k$ 方法，仍可能忽略注意力得分较低但实际重要的 token
2. **依赖低维投影质量**：$Z$ 阶曲线的局部性保持随维度增加而下降
3. **与 SSM 模型的比较**：Mamba 等状态空间模型在某些场景下内存效率更高

---

## 七、总结

ZETA 针对因果掩码下 top-$k$ attention 无法并行化的核心问题，提出了利用 $Z$ 阶曲线将 key/query 映射到一维空间进行并行排序和检索的解决方案。其关键洞察是：**key/query 的维度可以远小于 value 的维度**，因为它们只需要保持相对距离信息（由 Johnson-Lindenstrauss 引理保证），而 value 需要高维嵌入来承载丰富的语义信息。配合 Adaptive Cauchy Softmax 和 Triton 优化的 GPU 实现，ZETA 在保持 $\mathcal{O}(N \log N)$ 复杂度的同时，在 LRA 上取得 attention-based 模型的最佳结果，在 WikiText-103 上媲美标准 Transformer，且长序列推理效率显著优于 Flash Attention。

---

## 八、深入讨论 Q&A

### Q1: 详细介绍 Z 阶曲线映射

Z 阶曲线（Z-order Curve / Morton Code）是一种将多维数据映射到一维空间的空间填充曲线，核心特性是**保持局部性**（locality preservation）——原始空间中距离相近的点，映射后在一维空间中的 Z 值也相近。

**核心运算：比特交错（Bit Interleaving）**

给定 $d$ 维向量 $\mathbf{x} = (x_1, \dots, x_d)$，每个坐标 $x_i$ 的二进制表示为 $b_{i1}b_{i2}\dots b_{in}$（$n$ 为量化位数），Z 阶映射为：

$$Z = \underbrace{b_{11}b_{21}\dots b_{d1}}_{\text{所有维度的第1位}} \ \underbrace{b_{12}b_{22}\dots b_{d2}}_{\text{所有维度的第2位}} \ \dots \ \underbrace{b_{1n}b_{2n}\dots b_{dn}}_{\text{所有维度的第n位}}$$

先取每个坐标的**最高位**交错排列，再取**次高位**，依此类推。

**二维示例**：

| 点 | 坐标 | 二进制 | Z 值（交错后） |
|----|------|--------|----------------|
| A | (0,0) | x=00, y=00 | 0000₂ = 0 |
| B | (1,0) | x=01, y=00 | 0001₂ = 1 |
| C | (0,1) | x=00, y=01 | 0010₂ = 2 |
| D | (1,1) | x=01, y=01 | 0011₂ = 3 |

连接路径 A→B→C→D 形成 **Z 字形**，故得名。

**为什么能保持局部性**：高位比特决定了粗粒度的空间位置，低位比特决定细粒度微调。Z 值中排在越前面的比特来自各维度的越高位，Z 值相近意味着粗粒度空间位置相近。这等价于原始空间中的 quadtree/octree 前序遍历。

**ZETA 中为什么 $d_K$ 必须小**：维度诅咒——$d_K$ 越大，高维空间的稀疏性越强，Z 阶映射后的局部性保持效果急剧下降。论文实验数据：

| $d_K$ | $N=512$ | $N=2048$ |
|-------|---------|----------|
| 1-3 | ~80-90% | ~55-65% |
| 32 | ~40% | ~15% |
| 128 | ~15% | ~5% |

（度量方式：映射前后 top-64 最近邻的重叠比例）

ZETA 选择 $d_K=3$ 并配合可训练投影网络（2 层 MLP）来最大化局部性保持。

**ZETA 中的完整使用流程**：

```
高维 token 嵌入 (d_model = 512)
     │
     ▼  f_k, f_q 可训练投影网络（2层 MLP）
低维 key/query (d_K = 3)           高维 value (d_V = 64)
     │                                    │
     ▼ Z-order 比特交错                  │
一维 Z 值 (整数)                         │
     │                                    │
     ▼ torch.sort 并行排序               │
有序 Z 序列                              │
     │                                    │
     ▼ 分块 + causal masking             │
候选 chunk                               │
     │                                    │
     ▼ 二分查找 + 窗口 kNN               │
top-k 索引 I_q ──────────────────────────┘
     │
     ▼ Adaptive Cauchy Softmax + 稀疏注意力
输出
```

**为什么不用内积度量**：Z 阶曲线保持的是欧氏距离的局部性，与内积（dot-product）的局部性不兼容。在低维空间中，两个向量可能内积很大但欧氏距离也很大。因此 ZETA 放弃了传统的内积 Softmax，改用基于欧氏距离的 Adaptive Cauchy Softmax。

**一句话总结**：Z 阶曲线充当了"空间哈希函数"——将三维 key/query 空间映射到一维整数轴，使得空间上邻近的点在一维轴上仍然邻近，从而可以通过**排序 + 二分查找**高效定位最近邻，实现并行的 top-$k$ 注意力检索。

---

### Q2: 为什么需要做分块（chunking）？

**核心问题：Z 阶排序破坏了原始时间顺序。**

排序前（原始时间顺序）：
```
位置:     0      1      2      3      4      5
token:  "我"  "喜"  "欢"  "机"  "器"  "学"
Z 值:    42     7     93    15     61    28
```

排序后（按 Z 值升序）：
```
Z 值:     7     15     28     42     61     93
token:  "喜"  "机"  "学"  "我"  "器"  "欢"
原始位置:  1     3      5      0      4      2
```

在因果自注意力中，token 只能看到过去（位置 $j \le i$），但现在 Z 值最小的 key 可能对应原始序列中靠后的位置（如"学"对应位置 5），直接检索会泄露未来信息。

**分块解决因果泄露**：

将排序后的 key 按**原始时间位置**分成大小为 $M$ 的 chunk：

```
原始位置:   [0  1  2]    [3  4  5]    [6  7  8]    [9 10 11]
            └─chunk 0──┘ └─chunk 1──┘ └─chunk 2──┘ └─chunk 3──┘
               M=3           M=3           M=3           M=3
```

**规则**：原始位置为 $i$ 的 query，处在 chunk $m = \lfloor i/M \rfloor$，只能在前 $m$ 个 chunk 中搜索 key。

```
query 位置 i=5  → m=⌊5/3⌋=1  → 只能搜 chunk 0 和 chunk 1
query 位置 i=8  → m=⌊8/3⌋=2  → 只能搜 chunk 0, 1, 2
query 位置 i=2  → m=⌊2/3⌋=0  → 只能搜 chunk 0
```

**因果性保证**：chunk $m$ 包含的 token 原始位置范围是 $[mM, (m+1)M-1]$。对于位置 $i$ 的 query（处在 chunk $m$），允许访问的 chunk 中的最大原始位置是 $(m)M - 1 < mM \le i$，即所有允许访问的 key 的原始位置都严格小于 $i$。

**代价**：当前 chunk 内的 token（位置在 $[mM, i-1]$）被排除了，丢失了 chunk 内部的上下文信息。通过增大 $M$ 来缓解，但会增加计算量。

**一句话**：分块是 Z 阶排序的"代价"——排序让空间近邻聚集到了一起，但也打乱了时间顺序；分块在排序后的空间中重新建立时间边界，确保因果约束不被违反。

---

### Q3: 为什么分块可以重新建立时间边界？不分块，根据每个 query 的 position idx 先过滤一遍 key，再进行检索不可以吗？

逐 query 过滤的方案看起来很自然，但问题出在**并行效率和批量处理**上。

**逐 query 过滤的问题**：

对 query $i$，先过滤出 `positions < i` 的 key，再在这些 key 上做 Z 阶检索：

```
Query₅ 过滤后: [k₁, k₅, k₀, k₂]   ← 4 个元素，Z 值顺序: 7, 15, 42, 28 (无序！)
Query₆ 过滤后: [k₀, k₁, k₂, k₃, k₄, k₅]  ← 6 个元素
Query₇ 过滤后: [k₀, k₁, k₂, k₃, k₄, k₅, k₆] ← 7 个元素
```

三个致命问题：
1. **每个 query 的子集长度不同** → 无法组成 batch tensor，无法并行
2. **子集中 Z 值不再有序** → 移除元素打乱了顺序，全局排序作废，必须重新排序
3. **重新排序的代价**：$N$ 个 query 各排序 $O(i)$ 个元素 → 总复杂度 $O(N^2 \log N)$

**分块如何保持复用**：

分块的本质是**让一组 query 共享完全相同的 key 候选池**：

```
Chunk 0 (位置 0-2): Query₀,₁,₂ → 候选池 = chunk 0 的 keys
Chunk 1 (位置 3-5): Query₃,₄,₅ → 候选池 = chunk 0 + chunk 1 的 keys
Chunk 2 (位置 6-8): Query₆,₇,₈ → 候选池 = chunk 0 + chunk 1 + chunk 2 的 keys
```

同一 chunk 内的所有 query 使用**同一个**候选池：
- 候选池只排序一次（按 chunk 预先排好）
- 二分查找在当前 chunk 的 batched tensor 上并行执行
- GPU 利用率最大化

**对比总结**：

| 方案 | 并行性 | 复杂度 | 因果满足 |
|------|--------|--------|----------|
| 逐 query 过滤 + 重排序 | ❌ 各 query 独立，无法 batch | $O(N^2 \log N)$ | ✅ |
| 分块方案 | ✅ 同 chunk 内共享候选池，批量并行 | $O(N \log N)$ | ✅（丢失 chunk 内部信息） |

**一句话**：逐 query 过滤本质上是把全局排序的价值清零——每个 query 必须重排序自己的子集，排序变成 per-query 串行操作，效率和暴力搜索差距不大。分块通过"共享候选池"让一次排序服务多个 query，这才是 Z 阶排序 + GPU 批量并行的真正优势所在。

---

### Q4: 给定候选的 chunk，怎么利用 q, k 的 Z 表达来进行检索呢？

检索分为三步，Z 值只做粗筛，最终靠原始空间欧氏距离精筛：

**Step 1: 二分查找确定插入位置**

在已排序的候选 chunk 的 key Z 值序列中，用 `torch.searchsorted` 找到 query 的 Z 值应该插入的位置，$O(\log N)$：

```
排序后的 key Z 值:  [3, 7, 15, 22, 28, 35, 42, 58, 61, 73, 88, 93]
query Z_q = 40 → searchsorted → 插入位置 = 6
```

**Step 2: 以插入位置为中心取窗口**

围绕插入位置取宽度为 $K$ 的窗口（$K > k$，为精筛留余量）：

```
插入位置 = 6，K = 8（窗口半径 = 4）
left = max(0, 2), right = min(11, 10)
窗口候选: Z 值 [15, 22, 28, 35, 42, 58, 61, 73, 88]（9 个）
```

**Step 3: 在窗口中用原始空间欧氏距离做精确 kNN**

在窗口的 $K$ 个候选中，用原始 $d_K$ 维 key 向量计算欧氏距离，取距离最近的 $k$ 个：

```python
window_keys = K_sorted[left:right]      # Shape: (K, d_K)
dists = torch.norm(q - window_keys, dim=-1)  # Shape: (K,)
topk_indices = dists.topk(k, largest=False).indices
I_q = 对应的 value 索引
```

**完整流程图解**：

```
Query q (d_K=3 维) 
    │
    ├──► Z-order 映射 → Z_q = 40
    │
    ▼
候选 chunk 的 sorted keys:
    Z 值: [3, 7, 15, 22, 28, 35, 42, 58, 61, 73, 88, 93]
    高维: [k₀, k₁, k₂, k₃, k₄, k₅, k₆, k₇, k₈, k₉, k₁₀, k₁₁]
    value:[v₀, v₁, v₂, v₃, v₄, v₅, v₆, v₇, v₈, v₉, v₁₀, v₁₁]
    
    Step 1: searchsorted(Z_q) → 位置 6
    
    Step 2: 窗口 K=8 → 索引 [2..10]
            Z: [15, 22, 28, 35, 42, 58, 61, 73, 88]
            高维: [k₂, k₃, k₄, k₅, k₆, k₇, k₈, k₉, k₁₀]
            
    Step 3: 在窗口中用欧氏距离取 top-k (如 k=3)
            ‖q - k₅‖² = 0.08  ← 最近
            ‖q - k₂‖² = 0.12  ← 最近
            ‖q - k₆‖² = 0.31  ← 最近
            I_q = {5, 2, 6}
    
    Step 4: 用 I_q 取 value → [v₅, v₂, v₆]
    
    Step 5: Adaptive Cauchy Softmax 计算注意力权重 → 加权求和
```

**关键设计点**：

| 设计 | 作用 | 说明 |
|------|------|------|
| Z 值做粗筛 | 快速定位大致区域 | $O(\log N)$ 二分查找 |
| 窗口 $K > k$ | 防止 Z 值局部性不完美导致遗漏 | 用 Z 值牺牲一点精度换取速度 |
| 高维欧氏距离做精筛 | 保证检索精度 | 在 $K$ 个候选中精算，复杂度 $O(K \cdot d_K)$ |
| 两阶段筛选 | 平衡速度与精度 | 粗筛 $O(\log N)$，精筛 $O(K)$ |

**Z 值本身不直接用于距离比较**——它只是一个"空间哈希"，帮你快速找到大致区域。真正决定哪个 key 被选中，靠的是第三步的原始空间欧氏距离。

---

### Q5: 如果不做 Z 变换和快速区域定位，直接用低维 kq 通过内积计算距离并获取 k 个最大值，增加了 Z 变换后，整个可选择的序列（考虑到 causal）和 k 需要满足什么条件，Z 变换才有效率空间？

**问题核心**：ZETA 相比"直接用低维 k/q 暴力算内积取 top-k"能省多少计算，以及什么条件下这种节省是显著的。

**计算量对比**：

- **暴力方案**：$\sum_{i=1}^{N} i \cdot d_K = \frac{N(N+1)}{2} d_K \approx \frac{N^2 d_K}{2}$
- **ZETA 方案**：$2N\log N + N \cdot K \cdot d_K$

**ZETA 有效率增益的条件**（单次 query 视角）：

$$\log N + K \cdot d_K < i \cdot d_K \quad\Rightarrow\quad i > \frac{\log N}{d_K} + K$$

即 query 位置 $i$ 足够靠后时，ZETA 才有收益。

**全局效率条件**：

$$\frac{N^2 d_K}{2} > 2N\log N + N K d_K \quad\Rightarrow\quad N > \frac{4\log N}{d_K} + 2K$$

**数值分析**（$d_K = 3$）：

| $N$ | 暴力总量 | ZETA 总量（$K=16$） | 加速比（$K=16$） | 加速比（$K=64$） |
|-----|---------|---------------------|-------------------|-------------------|
| 128 | 24,576 | 7,040 | 3.5× | 1.3× |
| 512 | 393,216 | 29,184 | 13.5× | 5.5× |
| 2048 | 6,291,456 | 120,832 | 52× | 22× |
| 8192 | 100,663,296 | 499,712 | 201× | 84× |
| 32768 | 1,610,612,736 | 2,064,384 | 780× | 325× |

**$N \ge 512$ 时，ZETA 的优势已经非常显著。**

**$k$ 和 $K$ 需要满足的条件**：

1. **$K$ 不能太小**：$K$ 太小则 Z 值局部性的不完美会导致遗漏真正的 top-$k$ 近邻。$d_K=3$ 时 top-64 重叠率约 58.3%，经验上 $K \approx (2 \sim 4) \times k$。
2. **$K$ 不能太大**：$K$ 太大则窗口精筛成本上升，极端情况 $K \rightarrow N$ 退化为暴力搜索。
3. **效率条件整合**：$N > \frac{4\log N}{d_K} + 2K \approx \frac{4\log N}{d_K} + 4k \sim 8k$，即 $N$ 需要大于 $k$ 的 4~8 倍。实际中 $N \gg k$ 几乎总是成立。

**效率条件总结**：

| 条件 | 含义 | 实际场景 |
|------|------|----------|
| $N \gg k$ | 序列长度远大于 top-k | ✅ 几乎总是成立 |
| $N > 2K$ | 窗口远小于序列 | ✅ 长序列场景成立 |
| $K = (2\sim4)k$ | 窗口足够大保证召回率 | ✅ 工程可调 |
| $d_K$ 小（如 3） | 低维保证 Z 阶局部性 | ✅ 论文已验证 |

**核心洞察**：Z 变换的收益来自"用 $\log N$ 的排序+查找代替 $i$ 的线性扫描"。这个收益随 $N$ 线性增长，只要 $N$ 够大（实际中 512 以上就足够），ZETA 的效率优势就是压倒性的。$k$ 的大小只影响窗口精筛的常数项，不会改变渐进复杂度优势。
