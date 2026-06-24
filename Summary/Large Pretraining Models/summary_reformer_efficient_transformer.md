# Reformer: The Efficient Transformer

- **论文链接**: [arXiv:2001.04451](https://arxiv.org/abs/2001.04451)
- **作者**: Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya
- **机构**: U.C. Berkeley & Google Research
- **发表**: ICLR 2020

---

## 一、核心动机

大型 Transformer 模型在众多任务上取得了最先进的结果，但训练代价极其高昂，尤其在长序列场景下。主要瓶颈来自三方面：

1. **Attention 的 $\mathcal{O}(L^2)$ 复杂度**：序列长度 64K 时，单序列的 $QK^T$ 矩阵就需要 16GB 内存
2. **多层激活值存储**：$N$ 层模型需要存储 $N$ 倍的单层激活值用于反向传播
3. **Feed-forward 层的深度**：$d_{ff}$ 通常远大于 $d_{model}$（如 4K vs 1K），占用大量内存

Reformer 通过两项核心技术解决上述问题：**LSH 注意力**和**可逆残差层**。

---

## 二、LSH（Locality-Sensitive Hashing）注意力

### 2.1 核心思想

标准注意力的 softmax 由最大的内积主导，因此对于每个 query $q_i$，只需关注与其最接近的若干 key。问题转化为：**如何在高维空间中快速找到最近邻？** 答案是**局部敏感哈希**——将相似向量以高概率映射到同一个哈希桶中。

### 2.2 Shared-QK

LSH 哈希要求 $Q$ 和 $K$ 相同（否则桶的划分不一致），因此 Reformer 使用 **shared-QK** 设计：$Q$ 和 $K$ 共享同一个线性投影层，且 $K$ 做 L2 归一化（即 $k_j = q_j / \|q_j\|$）。实验表明 shared-QK 不影响模型性能。

### 2.3 Angular LSH

使用**角度 LSH**（基于随机投影）：

1. 固定随机矩阵 $R \in \mathbb{R}^{d_k \times b/2}$
2. 定义哈希函数：$h(x) = \argmax([xR; -xR])$

其中 $[u; v]$ 表示向量拼接。这等价于在高维球面上通过随机超平面划分空间，落在同一侧的向量获得相同哈希值。

### 2.4 LSH 注意力的完整流程

```
Step 1: 将 K 归一化（shared-QK）,计算 hash buckets
Step 2: 按 bucket 排序 → 相似向量聚集在一起
Step 3: 分为 m 个 chunk
Step 4: 每个 chunk 内的 query attend 到同 chunk 和前一 chunk 的 key
Step 5: 因果掩码通过排列位置索引实现
Step 6: 多轮哈希 → 取并集
```

**chunk 大小**：$m = 2l / n_{buckets}$（$l$ 为序列长度），确保平均桶大小 $\le m$ 使得 chunk 内能覆盖整个桶。

**因果掩码**：将每个 query/key 向量与原始位置索引关联，排序后使用相同的排列重排位置索引，然后通过比较操作计算 mask。

**防止自注意**：在 shared-QK 设计中，token 不能 attend 到自身（除第一个 token 外），因为向量与自身的内积总是最大。

### 2.5 多轮哈希

单轮哈希总存在相近向量落入不同桶的概率。使用 $n_{rounds}$ 个独立哈希函数并行执行 LSH 注意力，取结果的并集：

$$\mathcal{P}_i = \bigcup_{r=1}^{n_{rounds}} \mathcal{P}_i^{(r)}, \quad \mathcal{P}_i^{(r)} = \left\{ j : h^{(r)}(q_i) = h^{(r)}(k_j) \right\}$$

### 2.6 复杂度对比

| 注意力类型 | 内存复杂度 | 时间复杂度 |
|-----------|-----------|-----------|
| Scaled Dot-Product | $\max(bn_hld_k, bn_hl^2)$ | 同左 |
| LSH Attention | $\max(bn_hld_k, bn_hln_r(4l/n_c)^2)$ | $\max(bn_hld_k, bn_hn_rl(4l/n_c)^2)$ |

其中 $l$ 为序列长度，$b$ 为 batch size，$n_h$ 为 heads 数，$n_c$ 为 chunk 数，$n_r$ 为哈希轮数。

---

## 三、可逆 Transformer

### 3.1 RevNet 思想

可逆残差网络（RevNets）的核心：**任意层的激活值可以从下一层的激活值恢复**，无需存储中间值。

标准残差层：$y = x + F(x)$

可逆层对：$(x_1, x_2) \mapsto (y_1, y_2)$
$$\begin{aligned} y_1 &= x_1 + F(x_2) \\ y_2 &= x_2 + G(y_1) \end{aligned}$$

反向恢复：
$$\begin{aligned} x_2 &= y_2 - G(y_1) \\ x_1 &= y_1 - F(x_2) \end{aligned}$$

### 3.2 应用于 Transformer

```
Y_1 = X_1 + Attention(X_2)
Y_2 = X_2 + FeedForward(Y_1)
```

将 attention 和 feed-forward 层分别作为 $F$ 和 $G$ 放入可逆块中，Layer Normalization 移到残差块内部。

**内存效果**：整个网络的激活值内存与层数 $n_l$ **解耦**，只需存储一层而非 $n_l$ 层。

### 3.3 Chunking

Feed-forward 层在各位置上的计算完全独立，因此可分块处理：

$$Y_2 = \left[Y_2^{(1)}; \ldots; Y_2^{(c)}\right] = \left[X_2^{(1)} + \mathrm{FeedForward}(Y_1^{(1)}); \ldots; X_2^{(c)} + \mathrm{FeedForward}(Y_1^{(c)})\right]$$

这去除了 $d_{ff}$ 对内存的乘法因子。

### 3.4 完整内存复杂度

| 模型 | 内存复杂度 |
|------|-----------|
| Transformer | $\max(bld_{ff}, bn_hl^2) \cdot n_l$ |
| Reversible Transformer | $\max(bld_{ff}, bn_hl^2)$ |
| Chunked Reversible Transformer | $\max(bld_{model}, bn_hl^2)$ |
| **Reformer** (LSH + Chunked Rev) | $\max(bld_{model}, bn_hln_rc)$ |

---

## 四、实验与结果

### 4.1 合成任务：序列复制

任务形式：$0w0w$，其中 $w$ 长度 511，总序列长 1024。需要非局部注意力，任何有限 span 的稀疏注意力都无法解决。

| 模型 | 准确率 |
|------|--------|
| Full Attention | 100% |
| LSH-4 hashes（训练+评估） | 99.9% |
| LSH-1 hash（训练）+ LSH-8（评估） | 99.9% |

**关键发现**：训练时用较少哈希、评估时增加哈希数可提升准确率。

### 4.2 Shared-QK 和可逆层的影响

- Shared-QK 不损害性能，甚至训练略有加速
- 可逆层与标准 Transformer 学习曲线几乎完全相同

### 4.3 机器翻译（WMT14 EN-DE）

| 模型 | BLEU |
|------|------|
| Transformer base | 27.3 |
| Reversible Transformer base (100K) | 27.6 |
| Reversible Transformer base (500K) | 28.0 |
| Reversible Transformer big (300K) | 29.1 |

### 4.4 大模型训练

- 成功训练 20 层 Reformer 在 enwik8 和 imagenet64 上
- 标准 Transformer 基线因内存和速度问题无法训练
- 12 层 enwik8 模型：1.19 bits/dim → 调优后 1.05 bits/dim

---

## 五、关键技术细节

### 5.1 Shared-QK 中的自注意禁止

在 shared-QK 设计中，token 对自身的内积总是最大，因此修改 mask 禁止 token attend 到自身，除非没有其他有效目标（如序列首 token）。

### 5.2 多轮 LSH 的合并

使用 $N_{i,j}$ 因子（key $j$ 在多少轮中与 query $i$ 共享桶）来避免 union 时的重复计数，实现为 mask 中的附加项。

### 5.3 参数复用

由于 chunked reversible 大幅增加了计算密度（大批量 × 长序列），可以将非活跃层的参数在 GPU 和 CPU 间交换，传输成本被计算量摊平。

---

## 六、局限性与讨论

1. **LSH 是近似注意力**：仔细选择哈希轮数可在准确率和速度间权衡
2. **翻译任务不适用 LSH**：WMT 句子较短（<128 tokens），小于典型 chunk 大小
3. **依赖于 shared-QK 设计**：限制了 query 和 key 的表示差异
4. **大词汇量的 log-probability 计算也需要 chunking**

---

## 七、总结

Reformer 是首个将 Transformer 的注意力复杂度从 $\mathcal{O}(L^2)$ 降至 $\mathcal{O}(L \log L)$ 且**在实践中可训练**的模型。其核心贡献：

1. **LSH 注意力**：通过 angular LSH 分组 + 排序 + 分块实现高效的稀疏注意力
2. **可逆 Transformer**：利用 RevNet 思想消除多层激活值存储的开销
3. **Chunking**：分块处理 feed-forward 层和输出层，进一步降低内存

三项技术叠加使激活值内存与层数无关，使得在单设备上训练 20 层、64K 序列长度的 Transformer 成为可能。

---

## 八、深入讨论 Q&A

### Q1: 在 2.4 的 Step4 中，chunk 是按 key 来聚类的，同一个 chunk 内的 query/key 不一定距离相互接近，这样是否违背了 query 选择 key 时两者应该接近的原理？

首先澄清一个细微的误解：chunk 不是把 key 聚类后分块，而是按 LSH 哈希桶排序后再机械切分。具体流程：

1. **先对所有 query/key 算 LSH 哈希桶编号**
2. **按桶编号排序**（桶内按原始位置排序）
3. **再在排列后的序列上机械切分 chunk**

因为 LSH 保证相似向量以高概率落入同一桶，排序后同一桶内的所有 item 必然连续排列。所以 chunk 内的 item 大多数来自同一个或相邻的哈希桶，并不是随机混在一起的。

**chunk 大小 $m$ 的关键设计**：$m = 2 \times$ 平均桶大小。

$$m = \frac{2l}{n_{buckets}}, \quad \text{平均桶大小} = \frac{l}{n_{buckets}} = \frac{m}{2}$$

这个设计确保了完整的哈希桶在排序后最多跨两个相邻 chunk，而 Reformer 的注意力窗口恰好是**当前 chunk + 前一个 chunk**，总能完整覆盖：

```
排序后（按 bucket #）：

bucket=0         bucket=0(cont)      bucket=1        bucket=2
[●●●●●]          [●●●]               [●●●●●●●●]      [●●●●]
 └─ chunk 0 ──────┘└── chunk 1 ──────┘└── chunk 2 ──┘

m=6，平均 bucket size=3
bucket 0 有 8 个 item → 跨了 chunk 0 和 chunk 1
bucket 1 有 8 个 item → 跨了 chunk 1 和 chunk 2

Chunk 1 的注意力窗口：chunk 0 + chunk 1 → 覆盖了 bucket 0 的全部 ✅
Chunk 2 的注意力窗口：chunk 1 + chunk 2 → 覆盖了 bucket 1 的全部 ✅
```

**不同 bucket 的 item 在同一 chunk 内会怎样？** 这确实会在 chunk 边界处发生——但不会破坏"query 只关注近邻 key"的原则，原因有二：

1. **LSH 保证不同桶的 item 不相似**：如果两个向量落入不同桶，它们以高概率不相似。即使被放在同一个 chunk 中做注意力计算，它们之间的 dot-product 得分天然很低，softmax 会自然抑制。

```
chunk 内的注意力矩阵：

         bucket 0 尾部    bucket 1 头部
            k₃  k₄  k₅    k₆  k₇  k₈
q₃(bucket0) [高  高  高  │ 低  低  低 ]
q₄(bucket0) [高  高  高  │ 低  低  低 ]
─────────────────────────────────────
q₆(bucket1) [低  低  低  │ 高  高  高 ]
q₇(bucket1) [低  低  低  │ 高  高  高 ]
```

跨桶部分的注意力权重由相似度本身自然抑制。

2. **chunk 仅是批处理的工程技巧**：chunk 本质上是把排序后的序列切分以适配 GPU batch 计算，不是语义分组。注意力仍在 $\widetilde{\mathcal{P}}_i$（当前+前一 chunk）内逐对计算。

**与 ZETA 的对比**：

| | Reformer (LSH) | ZETA (Z-order) |
|---|---|---|
| **排序依据** | LSH 哈希桶（语义聚类） | Z 值（空间哈希） |
| **分块后相似性** | 同 bucket 内高度相似；跨 bucket 天然低相似 | Z 值相近 ≈ 空间相近，但边界处可能有跳跃 |
| **违反近邻原则的风险** | 低（不同 bucket 间内积天然小） | 需要窗口 $K > k$ 和多阶段精筛来补救 |

**一句话总结**：chunk 不是"把不相关的 item 硬凑在一起算注意力"，而是"把排序好的桶序列切成 GPU 友好的批次，并确保窗口足够大覆盖完整桶"。跨桶交互由相似度本身自动过滤，不存在违背近邻原则的问题。

---

### Q2: 一个 chunk 内既包含了 query 也包含了 key？

**是的**，Reformer 的 LSH 注意力中，chunk 内同时包含 query 和 key。这是因为 Reformer 采用了 **Shared-QK** 设计：

$$k_j = \frac{q_j}{\|q_j\|}$$

即 key 就是归一化后的 query。这意味着**序列中的每个 token 既是 query 又是 key**。排序后按 chunk 切分，每个 chunk 自然同时包含两者：

```
排序后的序列（按 bucket #，桶内按位置）：
 位置:  s₀    s₁    s₂   │  s₃    s₄    s₅   │  s₆    s₇    s₈
 bucket: 0     0     0   │  0     1     1   │  1     1     1
        └── chunk 0 ────┘ └── chunk 1 ────┘ └── chunk 2 ────┘
```

每个 token 在 chunk 内**同时扮演 query 和 key**：
- 作为 **query**：它在自己的 chunk（以及前一 chunk）中找相似的 key
- 作为 **key**：它被同一 chunk 内其他 query 检索

**注意力计算方式**：

```
Chunk 0 的注意力: Q_chunk0 × K_chunk0        (自注意力)
Chunk 1 的注意力: Q_chunk1 × [K_chunk0; K_chunk1]  (当前+前一)
Chunk 2 的注意力: Q_chunk2 × [K_chunk1; K_chunk2]
```

**Shared-QK 下的自注意力问题**：论文特别指出了 Shared-QK 的副作用——query 和自己做内积的值天然最大（$\|q\|^2 = 1$，而与别的向量内积通常小于 1），这会淹没有意义的跨 token 注意力。因此 Reformer 在因果掩码中额外禁止了 self-attention（除非 token 没有任何其他有效目标，如序列第一个 token）。

**与 ZETA 的对比**：

| | Reformer (LSH) | ZETA (Z-order) |
|---|---|---|
| Q 和 K 关系 | Shared-QK：$k_j = q_j/\|q_j\|$ | 分离：$q, k$ 各自通过投影网络生成 |
| chunk 内容 | Q 和 K 是同一组 token | 排序的是 key，query 通过二分查找搜索 |
| 相似性度量 | 内积（dot-product） | 欧氏距离 |
| chunk 内交互 | Q_chunk × K_chunk（及前一 chunk） | query 在窗口候选 key 中做 kNN |

**一句话总结**：Reformer 因为 Shared-QK，排序后的序列中每个位置既是 query 又是 key，chunk 自然包含两者。注意力在 chunk 内（及前一 chunk）的 query 和 key 之间计算。
