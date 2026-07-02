# Linformer: Self-Attention with Linear Complexity

- **论文链接**: [arXiv:2006.04768](https://arxiv.org/abs/2006.04768)
- **作者**: Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma
- **机构**: Facebook AI
- **发表**: 2020

---

## 一、核心动机

标准 Transformer 的自注意力机制复杂度为 $\mathcal{O}(n^2)$，在长序列场景下成为效率瓶颈。现有方案（稀疏注意力、LSH 注意力等）存在以下问题：

1. **稀疏注意力**：性能下降明显（2% drop，仅 20% 加速），实际增益有限
2. **Reformer (LSH)**：仅在序列长度 > 2048 时才比标准 Transformer 快，且多轮哈希增加了顺序操作数
3. **知识蒸馏**：无法加速 teacher 模型训练，学生模型性能通常会下降

Linformer 的核心洞察：**自注意力矩阵是低秩的**。基于此提出一种新的自注意力机制，将复杂度从 $\mathcal{O}(n^2)$ 降至 $\mathcal{O}(n)$。

| 模型 | 每层复杂度 | 最小顺序操作数 |
|------|-----------|---------------|
| RNN | $\mathcal{O}(n)$ | $\mathcal{O}(n)$ |
| Transformer | $\mathcal{O}(n^2)$ | $\mathcal{O}(1)$ |
| Sparse Transformer | $\mathcal{O}(n\sqrt{n})$ | $\mathcal{O}(1)$ |
| Reformer | $\mathcal{O}(n\log n)$ | $\mathcal{O}(\log n)$ |
| **Linformer** | **$\mathcal{O}(n)$** | **$\mathcal{O}(1)$** |

---

## 二、核心洞察：自注意力是低秩的

### 2.1 经验证据

对预训练 RoBERTa 模型的上下文映射矩阵 $P$ 做奇异值分解（SVD），在 Wiki103 和 IMDB 数据集上分析：

- $P$ 的奇异值呈现**长尾分布**，大部分信息集中在前几个最大的奇异值中
- **高层比低层更低秩**：在高层，更多信息集中在最大的奇异值上，$P$ 的秩更低
- 在 512 维中，取前 128 个奇异值即可恢复绝大多数信息

### 2.2 理论保证（Theorem 1）

> 对任意 $Q, K, V \in \mathbb{R}^{n\times d}$，存在一个低秩矩阵 $\tilde{P}$，使得：
> $$\Pr(\|\tilde{P}w^T - Pw^T\| < \epsilon\|Pw^T\|) > 1 - o(1)$$
> 且 $\text{rank}(\tilde{P}) = \Theta(\log n)$。

证明基于 **Johnson–Lindenstrauss 引理**，构造 $\tilde{P} = \exp(A) \cdot D_A^{-1} R^T R$，其中 $R \in \mathbb{R}^{k \times n}$ 为随机投影矩阵。

---

## 三、核心方法：线性自注意力

### 3.1 基本思路

标准注意力：
$$\text{head}_i = \underbrace{\text{softmax}\left(\frac{QW_i^Q(KW_i^K)^T}{\sqrt{d}}\right)}_{P: n \times n} \cdot VW_i^V$$

Linformer 在 K 和 V 上分别添加线性投影矩阵 $E_i, F_i \in \mathbb{R}^{n \times k}$，将 $n \times d$ 投影到 $k \times d$（$k \ll n$）：

$$\overline{\text{head}}_i = \underbrace{\text{softmax}\left(\frac{QW_i^Q(E_i K W_i^K)^T}{\sqrt{d}}\right)}_{\bar{P}: n \times k} \cdot \underbrace{F_i V W_i^V}_{k \times d}$$

**关键变化**：注意力矩阵从 $n \times n$ 变为 $n \times k$，时间和空间复杂度降为 $\mathcal{O}(nk)$。

```
标准注意力:
  Q(K^T)  →  n×n  →  softmax  →  n×n  →  ×V  →  n×d
  O(n²d) 内存 O(n²)

Linformer:
  Q(EK)^T  →  n×k  →  softmax  →  n×k  →  ×(FV)  →  n×d
  O(nkd) 内存 O(nk)
```

### 3.2 理论保证（Theorem 2）

> 当 $k = \min\{\Theta(9d\log d / \epsilon^2), 5\Theta(\log n / \epsilon^2)\}$ 时，存在矩阵 $E_i, F_i$ 使得线性自注意力以 $\epsilon$ 误差近似标准自注意力。

**核心结论**：$k$ 可以取 $O(d\log d)$ 而与 $n$ 无关。换言之，当嵌入维度 $d$ 固定时，**投影维度 $k$ 可以不随序列长度 $n$ 增长**，从而实现真正的 $\mathcal{O}(n)$ 复杂度。

证明分为两步：
1. 利用分布性 JL 引理，证明 $k = 5\log(nd)/(\epsilon^2 - \epsilon^3)$ 时成立（此时 $k$ 随 $n$ 增长）
2. 进一步利用 $\text{rank}(A) = d$ 的性质，通过取行子矩阵将 $k$ 降至 $9\log d/(\epsilon^2 - \epsilon^3)$，**与 $n$ 无关**

---

## 四、效率优化技术

### 4.1 参数共享策略

为减少额外参数量，Linformer 提供三级参数共享：

| 共享级别 | 说明 | 12层/12头模型参数量 |
|----------|------|-------------------|
| Headwise | 每层共享 $E, F$（跨头） | 24 个投影矩阵 |
| Key-Value | 每层共享 $E = F$（跨头 + KV） | 12 个投影矩阵 |
| Layerwise | 全局共享一个 $E$（跨层 + 跨头 + KV） | 1 个投影矩阵 |

实验表明 Layerwise 共享（仅 1 个额外投影矩阵）的性能几乎不下降。

### 4.2 非均匀投影维度

不同层/头可选用不同的 $k$。高层注意力矩阵更低秩，可使用更小的 $k$。

### 4.3 通用投影方法

除线性投影外，也可使用均值/最大池化、卷积等替代 $E, F$。

---

## 五、实验结果

### 5.1 预训练困惑度

- 在 BookCorpus + Wikipedia（33亿词）上预训练，64 块 V100 GPU，250k updates
- $n=512, k=128$ 或 $n=1024, k=256$ 时，Linformer 的困惑度曲线几乎与标准 Transformer 重合
- **固定 $k=256$，序列长度从 512 增至 4096**：最终困惑度几乎不变，验证了 $\mathcal{O}(n)$ 复杂度
- Layerwise 共享策略（仅 1 个投影矩阵）的困惑度与非共享模型几乎相同

### 5.2 下游任务（GLUE + IMDB）

| $n$ | 模型 | SST-2 | IMDB | QNLI | QQP | 平均 |
|-----|------|-------|------|------|-----|------|
| 512 | RoBERTa-base | 93.1 | 94.1 | 90.9 | 90.9 | 92.25 |
| 512 | Linformer, k=128 | 92.4 | 94.0 | 90.4 | 90.2 | 91.75 |
| 512 | Linformer, k=256, layer-shared | 93.1 | 94.1 | 91.2 | 90.8 | **92.30** |
| 1024 | Linformer, k=256, layer-shared | 93.2 | 94.2 | 90.8 | 90.5 | 92.18 |

- Linformer 在下游任务上与 RoBERTa 性能相当，$k=256$ 时甚至略有超越
- Layerwise 共享策略反而取得最佳精度
- 长序列预训练（$n=1024$）的模型在短序列任务上与 $n=512$ 预训练模型性能相当，说明**性能主要由 $k$ 决定，而非 $n/k$ 比例**

### 5.3 推理效率（V100 16GB）

| 序列长度 $n$ | $k=128$ 加速 | $k=256$ 加速 | $k=128$ 内存节省 | $k=256$ 内存节省 |
|-------------|-------------|-------------|-----------------|-----------------|
| 512 | 1.5× | 1.3× | 1.7× | 1.5× |
| 1024 | 1.7× | 1.6× | 3.0× | 2.9× |
| 4096 | 3.4× | 3.2× | 14× | 13× |
| 32768 | 13× | 12× | 56× | 48× |
| 65536 | 20× | 18× | 60× | 52× |

序列越长，加速和内存节省越显著。

---

## 六、理论保证总结

| 定理 | 内容 |
|------|------|
| Theorem 1 | 自注意力是低秩的：存在 $\text{rank} = \Theta(\log n)$ 的矩阵以 $\epsilon$ 误差近似 $P$ |
| Theorem 2 | 线性自注意力：当 $k = O(d\log d)$ 时，以 $\epsilon$ 误差近似标准注意力（$k$ 与 $n$ 无关） |

---

## 七、与其他方法的对比

| 方法 | 复杂度 | 核心思路 | 限制 |
|------|--------|----------|------|
| Standard Transformer | $\mathcal{O}(n^2)$ | 精确全注意力 | 长序列不可行 |
| Sparse Transformer | $\mathcal{O}(n\sqrt{n})$ | 稀疏注意力模式 | 性能下降较大 |
| Reformer (LSH) | $\mathcal{O}(n\log n)$ | 局部敏感哈希 | 大常数，需 shared-QK |
| Performer (FAVOR+) | $\mathcal{O}(nrd)$ | 随机特征映射 + 核近似 | 近似误差，GPU 不友好 |
| **Linformer** | **$\mathcal{O}(n)$** | **低秩分解 + 线性投影** | 投影维度 $k$ 需调参 |

---

## 八、局限性

1. **$k$ 的选择需要权衡**：$k$ 越大精度越高但速度越慢，$k$ 越小速度越快但可能损失信息
2. **低秩假设可能不总是成立**：对于需要精确细粒度注意力的任务，低秩近似可能丢失关键信息
3. **投影矩阵 $E, F$ 引入额外参数**：虽然可以通过共享策略减少
4. **推理时的序列长度灵活性受限**：$E, F$ 的维度固定为 $n \times k$，序列长度 $n$ 变化时需处理

---

## 九、总结

Linformer 通过自注意力低秩性质的核心洞察，提出了一种简单而高效的线性复杂度自注意力机制：

1. **低秩理论**：证明自注意力矩阵可由 $\Theta(\log n)$ 秩矩阵近似
2. **线性投影**：引入 $E, F \in \mathbb{R}^{n \times k}$ 将 K、V 从 $n \times d$ 投影到 $k \times d$，注意力矩阵从 $n \times n$ 变为 $n \times k$
3. **$k$ 与 $n$ 无关**：从理论证明 $k = O(d\log d)$ 即可保证近似质量
4. **参数共享**：Layerwise 共享仅需 1 个额外投影矩阵，性能几乎无损
5. **实验验证**：在预训练和下游任务上性能与标准 Transformer 相当，推理显著加速

Linformer 以简洁优雅的方式将 Transformer 复杂度降至线性，是高效 Transformer 领域的重要里程碑。

---

## 十、Q&A

### Q1: 投影矩阵 $E, F$ 维度与序列长度 $n$ 绑定，序列长度变化了怎么办？

这是 Linformer 的一个**已知工程局限**。训练时 $E, F \in \mathbb{R}^{n \times k}$ 是固定尺寸的可学习参数矩阵，推理时若序列长度 $n' \neq n$，形状不匹配。论文和工程上提供了以下解决思路：

**论文方案：函数式投影（General Projections, Section 4）**

论文意识到这一问题，提出了用不依赖 $n$ 的投影方法替代固定参数矩阵：

| 投影方式 | 是否依赖 $n$ | 原理 |
|----------|-------------|------|
| 均值/最大池化 | 否 | stride = ceil(n/k)，自适应任意长度，输出恒为 $k$ 维 |
| 一维卷积 | 否 | stride = ratio = n/k，卷积核可训练，输出恒为 $k$ 维 |
| 原论文线性投影（默认） | **是** | $n \times k$ 固定参数矩阵，需处理长度不匹配 |

卷积/池化方式天然适配变长——无论 $n$ 如何变化，始终以自适应 stride 压缩到 $k$ 维，不存在维度绑定问题。

**工程补救方法**：

1. **截断/补齐**：训练时使用最大预期长度 $n_{\text{max}}$ 作为 $E, F$ 的维度，推理时对短序列补零、长序列截断
2. **线性插值**：对 $E$ 的 $n$ 维度进行插值（如双线性插值）到目标长度 $n'$
3. **可分离投影**：将 $E \in \mathbb{R}^{n \times k}$ 分解为 $E = U \cdot S$，其中 $U \in \mathbb{R}^{n \times d}$ 可由位置编码隐式生成，$S \in \mathbb{R}^{d \times k}$ 是可学习且与 $n$ 无关的参数

**本质矛盾**：论文理论上证明了 $k = O(d\log d)$ 与 $n$ 无关，但默认实现中投影矩阵仍与训练时 $n$ 绑定。使用卷积/池化投影可以彻底解决，但可能带来精度损失——这是 Linformer 优雅理论的工程代价。
