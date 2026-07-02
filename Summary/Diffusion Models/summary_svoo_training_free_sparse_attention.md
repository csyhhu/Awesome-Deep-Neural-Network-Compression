# SVOO: Attention Sparsity is Input-Stable —— Training-Free Sparse Attention for Video Generation via Offline Sparsity Profiling and Online QK Co-Clustering

- **会议**: ICML 2026
- **论文链接**: [arXiv:2603.18636](https://arxiv.org/abs/2603.18636)
- **代码**: [GitHub - SVOO](https://github.com/Mutual-Luo/SVOO)
- **作者机构**: 北京航空航天大学、北京大学、中科院自动化所、中国科学技术大学、清华大学、中关村学院

---

## 1. 研究动机与问题

### 1.1 背景

Diffusion Transformers (DiTs) 在视频生成领域取得了显著成功，但其核心的密集 3D 自注意力机制对时空 token 数量呈平方复杂度，导致推理成本极高。训练无关（training-free）的稀疏注意力方法成为降低推理开销的重要途径——它们无需额外训练，直接利用注意力图的稀疏性来削减计算量。

### 1.2 现有方法的两个关键局限（L1 & L2）

作者深入分析现有训练无关稀疏注意力方法，揭示两个尚未被解决的局限：

**L1: 忽略注意力剪枝中的层间异质性（Layer Heterogeneity in Attention Pruning）**
- 现有方法将多个 Transformer 层视为同质堆叠，在不同层上施加统一的稀疏比例
- 实际上，不同层对注意力剪枝的容忍度差异显著，统一策略要么在某些敏感层过度剪枝，要么在冗余层剪枝不足

**L2: 忽略块划分中的 Query-Key 耦合（Q-K Coupling in Block Partitioning）**
- 现有块级稀疏注意力方法独立地对 Query 和 Key 进行分块
- 实际上，最优的 Key 分块是 Query 依赖的，反之亦然
- 独立的 Q-K 分块会导致 Q-K 信息对应关系错位，降低保真度

---

## 2. 核心洞察：注意力稀疏性是输入稳定的层级内禀属性

### 2.1 实证分析

作者在 4 个代表性模型（Wan2.1-14B-T2V, Wan2.1-14B-I2V, Wan2.2-A14B-T2V, HunyuanVideo-T2V）上测量注意力密度（覆盖 80% 注意力质量所需的最小注意力条目比例），得出两个关键发现：

| 性质 | 描述 |
|------|------|
| **层间异质性** | 注意力密度在不同层之间差异显著，表明不同层对剪枝的容忍度不同 |
| **层内稳定性** | 对某一特定层，其注意力稀疏性在不同输入之间高度一致，表明稀疏性主要由层的架构角色和已学参数决定，而非具体输入 |

### 2.2 理论证明（Theorem 1）

作者对 pre-softmax 注意力 logit 的方差给出了高概率的稳定性界。该界依赖于层特定的投影矩阵 $\mathbf{M} = \mathbf{W}_Q \mathbf{W}_K^\top$，解释了**层间异质性**；同时界与 $1/\sqrt{n}$（$n$ 为 token 数）成正比，在视频场景下 $n$ 极大，使得界很小，解释了**层内稳定性**。

---

## 3. 方法：SVOO 框架

SVOO 采用两阶段范式：

### 3.1 离线阶段：逐层稀疏性画像（Offline Layer-Wise Sparsity Profiling）

1. **构建校准集**：用少量随机输入 $\mathcal{D} = \{x^k\}_{k=1}^m$（论文采用随机高斯噪声）
2. **计算逐层注意力密度**：对每层 $\ell$、每头 $h$、每个校准输入 $x^k$，计算覆盖 $\tau=95\%$ 注意力质量所需的最小注意力条目比例 $d_{\ell,h}^k$
3. **高斯拟合**：将 $d_{\ell,h}^k$ 拟合为高斯分布 $\mathcal{N}(\mu_{\ell,h}, \sigma_{\ell,h}^2)$，取上 $\alpha=0.95$ 分位数 $\hat{d}_{\ell,h} = \mu_{\ell,h} + z_\alpha \sigma_{\ell,h}$ 作为保守估计
4. **生成稀疏调度**：$s_{\ell,h} = 1 - \hat{d}_{\ell,h}$

关键优势：由于层内稳定性，校准集可以是任意合理输入。

### 3.2 在线阶段：双向协同聚类（Online Bidirectional Co-Clustering）

**算法流程**（Algorithm 1）：

1. **初始化**：随机采样 Query/Key 锚点作为初始聚类中心 $\mathbf{C}_q^{(0)}, \mathbf{C}_k^{(0)}$
2. **迭代交替执行**（仅 2 次迭代）：
   - **Step A (Query-aware Key 侧聚类)**：计算每个 Key token 对当前 Query 中心 $\mathbf{C}_q^{(i-1)}$ 的亲和度向量 $\mathbf{P}_k$，将具有相似 Query 偏好模式的 Key 分到同一块，更新 Key 中心和分配
   - **Step B (Key-aware Query 侧聚类)**：对称地，计算每个 Query token 对当前 Key 中心的亲和度，分组更新
3. **输出**：耦合的 Q-K 块划分

**设计原理**：
- Query 块内的 token 应具有相似的注意力偏好
- Key 块内的 token 应对相同的 Query 表现出相似的相关性

**顶部块对选择**：
- 通过粗粒度估计 $\bar{\mathbf{A}} = \mathbf{C}_q \mathbf{C}_k^\top$ 进行块级选择
- 采用阈值依赖策略平衡召回率 $\tau$ 和离线预算 $s_{\ell,h}$

**聚类复用**：
- 聚类结果在去噪步之间高度稳定（互信息一致性高）
- 每 20 步重新计算一次聚类，大幅降低开销

**与 SVG2 的关键区别**：
SVG2 使用独立的 K-means 分别对 Q 和 K 分块，忽略 Q-K 耦合；SVOO 通过双向协同聚类联合分块。

---

## 4. 实验

### 4.1 设置

- **7 个模型**: Wan2.1-T2V-1.3B, Wan2.1-T2V-14B, Wan2.1-I2V-14B, Wan2.2-T2V-A14B, Wan2.2-I2V-A14B, HunyuanVideo-T2V, HunyuanVideo-I2V
- **数据集**: VBench (T2V), VBench++ (I2V)
- **指标**: PSNR, SSIM, LPIPS, VBench (ImageQual, AesQual, SubConsist, BackConsist), 延迟, 加速比
- **Baselines**: SpargeAttn, SVG, SVG2, Radial
- **硬件**: NVIDIA H200 GPU, 720p 分辨率

### 4.2 T2V 结果摘要

| 模型 | 方法 | PSNR↑ | 加速比 |
|------|------|-------|--------|
| Wan2.1-1.3B | SVOO | **29.99** | **1.93×** |
| Wan2.1-1.3B | SVG2 | 29.27 | 1.73× |
| Wan2.1-14B | SVOO | **27.79** | **1.64×** |
| Wan2.1-14B | SVG2 | 27.34 | 1.57× |
| Wan2.2-14B | SVOO | **24.85** | **1.63×** |
| Wan2.2-14B | SVG1 | 21.29 | 1.53× |
| HunyuanVideo | SVOO | 24.88 | **2.17×** |
| HunyuanVideo | SVG2 | **25.22** | 1.96× |

SVOO 在所有设置下均取得最高加速比，同时保持最优或接近最优的生成质量。对 Wan2.1-1.3B 实现了 **1.93× 加速**，PSNR 达 **29.99 dB**。

### 4.3 I2V 结果摘要

| 模型 | 方法 | PSNR↑ | 加速比 |
|------|------|-------|--------|
| Wan2.1-14B | SVOO | **27.55** | **1.74×** |
| Wan2.2-14B | SVOO | **29.68** | **1.61×** |
| HunyuanVideo | SVOO | **25.16** | **2.17×** |

在 I2V 任务上，SVOO 在所有模型上全面领先。

### 4.4 消融实验

| 变体 | 说明 | 效果 |
|------|------|------|
| SVOO (w/o Off) | 移除离线画像，固定 recall τ=90% | 效率降低 |
| SVOO (w/o On) | 移除双向协同聚类，使用独立聚类 | 质量下降 |

**结论**: 离线画像实现安全加速，在线协同聚类产出更对齐的块并带来轻微额外开销。

### 4.5 其他分析

- **质量-效率权衡**: 在广泛稀疏范围内保持稳健性能
- **聚类复用**: 聚类结果在步间高度稳定（互信息相似度高），每 20 步重算一次可大幅降低开销
- **注意力召回率**: 双向协同聚类的召回率始终高于 SVG2 的 K-means
- **实现**: 使用 Triton 编写双向协同聚类，采用 FlashInfer 动态块大小核

---

## 5. 关键贡献

1. **揭示核心洞察**：注意力稀疏性是**输入稳定的层内禀属性**——层间差异显著但层内高度一致（附理论证明）
2. **提出 SVOO**：两阶段训练无关稀疏注意力框架
   - 离线逐层稀疏性画像：为每层定制剪枝比例
   - 在线双向协同聚类：考虑 Q-K 耦合的联合块划分
3. **全面实验验证**：在 7 个主流视频生成模型上取得最优质量-效率权衡，最高 **2.17× 加速**

---

## 6. 局限与展望

- 聚类复用策略中每 20 步重算的间隔为经验设定，可进一步优化
- 论文当前仅在视频 DiT 上验证，方法可推广至其他 DiT 应用场景（如图像生成、多模态模型）
- 离线画像阶段需在具体模型上执行一次校准，但对部署场景而言成本可接受

---

## 7. 深入讨论 Q&A

### Q1: 详细介绍离线阶段的动机、理论与流程

离线阶段的设计动机来源于对 DiT 模型注意力机制的深入分析。作者在 4 个代表性视频生成模型上进行了系统实验，测量每层的**注意力密度**（attention density），定义为覆盖 80% 累计注意力质量所需的最小注意力条目比例：

$$d_{\ell,h} = \frac{1}{n}\sum_{i=1}^{n}\frac{|\mathcal{S}_{\ell,h}(i)|}{n}$$

其中 $\mathcal{S}_{\ell,h}(i)$ 是使得第 $i$ 行注意力权重累计和 ≥ τ 的最小位置集合。

实验结果揭示两个关键性质：

| 性质 | 现象 | 含义 |
|------|------|------|
| **层间异质性** | 不同层的注意力密度差异巨大 | 每层对剪枝的容忍度不同，统一剪枝比例必然次优 |
| **层内稳定性** | 同一层跨不同输入，注意力密度高度一致 | 稀疏性是层的"内禀属性"，可以安全地离线测定 |

**核心洞察**：既然稀疏性是层的固有性质且输入无关，那么完全可以用少量随机输入在离线阶段"画像"，推理时直接复用。

#### 理论支撑：Theorem 4.2（层间稀疏性稳定性定理）

**设定**：给定输入 $\mathbf{X} \in \mathbb{R}^{n \times d}$，定义每行 pre-softmax logit 的平均方差：

$$V(\mathbf{X}) \triangleq \frac{1}{n}\sum_{i=1}^{n}\text{Var}\big(\mathbf{z}_i(\mathbf{X})\big)$$

其中 $\mathbf{z}_i(\mathbf{X}) = \frac{(\mathbf{x}_i\mathbf{W}_Q)(\mathbf{X}\mathbf{W}_K)^\top}{\sqrt{d'}}$，方差越大意味着 softmax 后分布越集中（即越稀疏）。

**定理核心结论**：在"token 表示有界"的合理假设下（$\|\mathbf{x}\|_2 \le R$），对任意两个独立输入 $\mathbf{X}, \hat{\mathbf{X}}$，以至少 $1-\delta$ 的概率有：

$$\big|V(\mathbf{X})-V(\hat{\mathbf{X}})\big| \le \frac{d \|\mathbf{M}\|_2^2}{d'} \cdot C R^4 \left(\sqrt{\frac{\log(d/\delta)}{n}} + \frac{\log(d/\delta)}{n}\right)$$

其中 $\mathbf{M} = \mathbf{W}_Q \mathbf{W}_K^\top$。

**两条关键推论**：

1. **层间异质性**：上界依赖于 $\|\mathbf{M}\|_2^2$，这是层特定的参数，不同层不同 → 不同层的 sparsity 差异大
2. **层内稳定性**：在视频生成场景下，$n$（token 数）极大（数万 tokens），上界与 $1/\sqrt{n}$ 成正比 → 上界极小，同一层跨输入的方差差异几乎为零

#### 画像流程

**Step 1 — 构建校准集**：$m$ 个随机输入（论文使用随机高斯噪声，利用了层内稳定性——输入可以任意）

**Step 2 — 逐层逐头计算注意力密度**：对每一层 $\ell$、每一个注意力头 $h$、每一个校准输入 $x^k$：
1. 前向传播得到 post-softmax 注意力矩阵 $\mathbf{A}_{\ell,h}^k$
2. 对每一行 $i$，降序排列后找到最小前缀覆盖 $\tau$（$\tau = 0.95$）注意力质量
3. 计算密度 $d_{\ell,h}^k$

**物理含义**：$d_{\ell,h}^k = 0.3$ 意味着该层只需 30% 的注意力条目即可捕获 95% 的注意力质量，即理论剪枝率为 70%。

**Step 3 — 高斯拟合与保守估计**：将 $m$ 个校准样本的密度值拟合为单变量高斯分布，取上 $\alpha$-分位数作为保守估计（$\alpha = 0.95$）：
$$\hat{d}_{\ell,h} = \mu_{\ell,h} + z_\alpha \cdot \sigma_{\ell,h}$$

取上分位数是一种保守策略，保证真实所需密度超过 $\hat{d}_{\ell,h}$ 的概率 ≤ 5%。

**Step 4 — 生成稀疏调度**：$s_{\ell,h} = 1 - \hat{d}_{\ell,h}$

#### 最终使用方式

在线推理时，逐层稀疏调度指导块对选择：

$$\rho_{\ell,h} = \begin{cases} \min(\text{Recall}(\bar{\mathbf{A}}, \tau), s_{\ell,h}), & s_{\ell,h} > \theta \\ \max(\text{Recall}(\bar{\mathbf{A}}, \tau), s_{\ell,h}), & s_{\ell,h} \le \theta \end{cases}$$

- $\theta = 0.1$ 为阈值
- 对高稀疏层（$s > \theta$），取召回率和预算的最小值，避免过度保留
- 对低稀疏层（$s \le \theta$，即敏感层），取最大值，确保关键信息不丢失

**核心优势**：一次画像，终身使用。无论用户输入什么 prompt，都不需要重新画像，因为稀疏性是层的固有属性而非输入属性。

---

### Q2: 如果实时计算 variance 发现输入间有较大差异，能否反驳 Theorem 4.2？

不能直接反驳。Theorem 4.2 给出的不是等式，而是一个**高概率上界**：

$$|V(\mathbf{X}) - V(\hat{\mathbf{X}})| \le B$$

要反驳定理形式正确性，需要证明差值超过了定理给出的上界 $B$。但在视频场景下这非常困难：

1. **常数 $C$ 未指定**：$C$ 是证明中多次三角不等式和矩阵 Bernstein 不等式累积出的绝对常数，实际值可能非常大（几十到上百），即使测到较大差异，定理仍可通过"$C$ 取大一点"自洽
2. **前提条件突破**：定理假设 $\mathbf{X}$ 和 $\hat{\mathbf{X}}$ 是独立随机样本。如果你的算法**根据 variance 主动挑选输入**，这些输入就不再是独立随机样本，而是被 adversarially selected 的——**违反了定理的前提条件**
3. **Variance 到 Density 到生成质量不是等距映射**：定理 bound 的是 pre-softmax logit variance，但实际稀疏调度用的是 attention density。variance 差 0.01 可能在生成质量上放大为明显退化，放大因子也可能是输入依赖的

**更可能被挑战的不是定理的形式正确性，而是它的实用有效性**：

| 挑战层面 | 含义 |
|---------|------|
| 上界是否紧致？ | $C$ 可能极大，导致实际上界不小 |
| 假设多大程度成立？ | $\|\mathbf{x}\|_2 \le R$ 的 $R$ 到底多大？ |
| 分布外输入 | 真实推理时可能出现校准集未覆盖的极端 prompt |
| Variance 到 Density 映射 | 两者高度相关但不等价 |

**有价值的实验方案**：取 calibration 得到的 schedule，在 100 个真实 prompt 上分别测试 PSNR 下降量。如果某些 prompt PSNR 暴跌而另一些几乎无损，说明 schedule 并非对所有输入等效。

---

### Q3: 详细介绍在线双向协同聚类（Online Bidirectional Co-Clustering）

#### 为什么需要双向协同聚类？

现有块级稀疏方法（如 SVG2）独立地对 Query 和 Key 进行 K-means 分块，这存在致命问题：**最优的 Key 分块是 Query 依赖的**。

对于给定 Query $\mathbf{q}$，两个 Key $\mathbf{k}_1$ 和 $\mathbf{k}_2$ 应该分到同一块，当且仅当 $\mathbf{q}^\top \mathbf{k}_1 \approx \mathbf{q}^\top \mathbf{k}_2$，即 $\mathbf{q}^\top (\mathbf{k}_1 - \mathbf{k}_2) \approx 0$。但 $(\mathbf{k}_1 - \mathbf{k}_2)$ 对不同 $\mathbf{q}$ 的内积不同，因此不存在一个"普适的"Key 分块方案——最优分块必须随 Query 变化。

#### 核心思想：联合分组而非独立分组

> - **Query 块内的 token** 应对 Key 有相似的注意力偏好
> - **Key 块内的 token** 应对 Query 有相似的相关性模式

这需要**同时**而不是**先后**确定 Q 和 K 的块分配。

#### 算法流程（Algorithm 1）

| 参数 | 值 |
|------|-----|
| $K_q$（目标 Query 块数） | 256 |
| $K_k$（目标 Key 块数） | 1024 |
| $I_{\text{max}}$（最大迭代数） | **2**（非常小） |

**初始化**：随机采样 $K_q$ 和 $K_k$ 个 token 作为初始聚类中心 $\mathbf{C}_q^{(0)}, \mathbf{C}_k^{(0)}$。

**第 $i$ 轮迭代**（交替执行 Step A → Step B）：

**─ Step A: Query-aware Key 侧聚类 ─**

目标：根据"哪些 Query 块会关注我"来对 Key 进行分组。

- **A.1** — 计算每个 Key token 对各 Query 中心的亲和度：$\mathbf{P}_k = \mathcal{K} \cdot (\mathbf{C}_q^{(i-1)})^\top \in \mathbb{R}^{N \times K_q}$
  - $\mathbf{P}_k[j, p]$ = Key $j$ 被 Query 块 $p$ 关注的程度——构成 Key $j$ 的"注意力偏好签名"
- **A.2** — 计算每个 Key 中心的签名：$\bar{\mathbf{P}}_k = \mathbf{C}_k^{(i-1)} \cdot (\mathbf{C}_q^{(i-1)})^\top \in \mathbb{R}^{K_k \times K_q}$
- **A.3** — 归一化并分配：每个 Key token 被分配给**注意力签名最相似的 Key 中心**（用 $\ell_2$ 距离）
- **A.4** — 更新 Key 中心：$\mathbf{C}_k^{(i)} \leftarrow \text{Mean}(\mathcal{K} \text{ via } \mathcal{L}_k)$

**─ Step B: Key-aware Query 侧聚类 ─**

与 Step A 完全对称，用刚更新的 $\mathbf{C}_k^{(i)}$ 对 Query 进行分组。

#### 关键设计：信息的双向流动

即使只迭代 2 轮，信息已经在 Q→K→Q→K 方向流动了 4 次：

```
第 i 轮:
  1. Key 侧更新: 利用 C_q 的知识 → Keys 基于对当前 Query 块的亲和模式重新分组
  2. Query 侧更新: 利用刚更新的 C_k → Queries 基于对新 Key 块的注意力模式重新分组
↓ 下一轮
  1. Key 侧更新: 用更新后的 C_q → Keys 的分组反映新的 Query 视角
  2. ...
```

直观效果：独立 K-means 产出的块对可能注意力质量分散，而双向协同聚类使块内注意力偏好趋于一致，注意力质量集中在少量块对上。

#### 块对选择

聚类完成后通过粗估计选择块对：

$$\bar{\mathbf{A}} = \mathbf{C}_q \mathbf{C}_k^\top \in \mathbb{R}^{K_q \times K_k}$$

根据逐层剪枝预算 $s_{\ell,h}$ 和召回率目标 $\tau$ 选择 top 块对计算 dense attention，其余置零。

#### 聚类复用策略

聚类结果在去噪步骤之间高度稳定（互信息相似度 > 0.9），因此**每 20 步才重算一次**，聚类开销摊薄到 1/20。

#### 与 SVG2 的关键对比

| 维度 | SVG2 | SVOO |
|------|------|------|
| 分块方式 | 独立 K-means（先分 K 再分 Q） | 双向协同聚类（交替联合） |
| 耦合建模 | ❌ 无（Q-K 解耦） | ✅ 有（互相指导分块） |
| 聚类目标 | 块内 token 原始特征相似 | 块内 token 对另一侧的注意力模式相似 |
| 信息流 | Q→Q, K→K（各自独立） | Q↔K↔Q↔K（交替双向） |
| 注意力召回率 | 较低 | 始终更高（Figure 7 验证） |

---

### Q4: Online bidirectional co-clustering 是否产生一个 assign 函数，输入 query 输出应计算的 key block index？

**是的，可以这样理解。** 但它是**两级映射**，而非一个直接的端到端函数。

双向协同聚类最终产出三样东西：

| 产物 | 符号 | 作用 |
|------|------|------|
| Query→Query 块的映射 | $\mathcal{L}_q$ | 告诉你这条 Query 属于哪个 Query 块 |
| Key→Key 块的映射 | $\mathcal{L}_k$ | 告诉你这条 Key 属于哪个 Key 块 |
| 块对选择矩阵 | $\mathcal{S}$ | 告诉你哪些 (Query 块, Key 块) 对值得算 |

**assign 函数是两级组合**：

```
给定一条 query token q_i:

  Step 1: p = L_q[i]                         ← 找到 q_i 所属的 Query 块
  Step 2: selected_key_blocks = { m | (p, m) ∈ S }  ← 找到该 Query 块被选中的 Key 块集合
  Step 3: selected_key_tokens = { j | L_k[j] ∈ selected_key_blocks }  ← 展开到具体 Key token

  → 只对 selected_key_tokens 计算 dense attention
```

**伪代码**：

```python
def assign(query_index: int) -> list[int]:
    p = L_q[query_index]                    # Query → Query 块
    selected_blocks = block_pairs[p]         # Query 块 → 对应的 Key 块列表
    selected_keys = L_k_inv[selected_blocks] # Key 块 → Key token 列表
    return selected_keys
```

**关键点**：$\mathcal{S}$ 不是全连接的，而是**逐 Query 块不同**——同一个 Key 块可能对 Query 块 A 被选中，对 Query 块 B 不被选中。这正是"Query-aware"的体现。

**一句话总结**：assign 函数 = $\mathcal{L}_q \circ \mathcal{S} \circ \mathcal{L}_k^{-1}$，输入一条 query token 索引，输出应计算的 key token 索引列表。

---

### Q5: 本方法需要重新训练吗？

**完全不需要重新训练。** 整个 SVOO 方法是 **training-free** 的。

| 阶段 | 需要训练？ | 具体操作 | 是否修改权重？ |
|------|:---:|---|---|
| 离线画像 | ❌ | 前向传播统计 attention density，拟合高斯分布 | 不修改 |
| 在线双向协同聚类 | ❌ | 交替聚类 + 块对选择，纯算法无梯度 | 不修改 |
| 最终推理 | ❌ | 只在选中块对上计算 dense attention | 不修改 |

在线双向协同聚类不需要训练的原因：
1. **聚类中心初始化**：随机采样 token（非参数学习）
2. **交替分配**：基于内积 + ℓ₂ 距离做最近邻分配（无梯度）
3. **中心更新**：取块内 token 均值（无参数）
4. **块对选择**：基于粗估计 $\bar{\mathbf{A}}$ 的 top-k 选择（纯排序）

整个过程只有前向计算，没有反向传播、没有 loss、没有梯度更新。模型权重 $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$ 始终保持不变。

**一句话**：SVOO = **预训练 DiT 权重不变** + **推理时置换注意力计算模式**，是一个即插即用的推理加速插件。

---

### Q6: 方法在文本/图像领域适用吗？论文只做了视频生成

论文确实只在视频 DiT 上验证，但可从理论前提分析跨领域适用性。

#### SVOO 的两个核心前提

| 前提 | 依赖条件 | 视频 | 文本 LLM | 图像 DiT |
|------|---------|:---:|:---:|:---:|
| **层内稳定性**（Theorem 4.2） | $n$（token 数）足够大 → 上界 $O(1/\sqrt{n})$ 紧致 | ✅ $n$ ~ 数万 | ⚠️ 短序列可能不紧致 | ✅ $n$ ~ 几千到数万 |
| **注意力稀疏性**（层内禀属性） | 模型本身存在可剪枝的冗余注意力 | ✅ 已验证 | ⚠️ head 差异极大 | ⚠️ 未经系统验证 |

#### 各领域分析

**文本（LLM）**
- 有利：长上下文场景 $n$ 也很大（128K），Theorem 4.2 上界仍紧致；已有大量工作证明 LLM 注意力存在稀疏性
- 不利：LLM 注意力模式远比 DiT 复杂（induction head 极稀疏 vs 全局头几乎不稀疏）；注意力模式**输入依赖性更强**（写代码 vs 翻译任务稀疏模式可能不同），挑战"层内稳定性"假设；LLM 是**自回归**的（KV cache 逐步增长），需要适配
- **结论**：理论上有扩展空间，但需验证"层内稳定性"是否成立，且需适配自回归模式

**图像 DiT**
- 有利：图像 DiT 是视频 DiT 的特例（帧数 = 1）→ 架构几乎相同；$n$ 仍较大（1024² 图像可到 262K tokens）
- 不利：$n$ 比视频小一个量级（无时间维度），上界稍宽松；论文未在纯图像模型上实验
- **结论**：**最可能直接适用**，因为视频 DiT 就是图像 DiT 沿时间轴的扩展，注意力机制完全一致

#### 论文不声称通用的原因推测

1. **科学严谨性**：Theorem 4.2 只保证 variance 稳定，variance → density → 生成质量的传递链在不同领域可能断裂
2. **实验成本**：已在 7 个模型上实验，扩展到 LLM 需要大量额外工作
3. **架构差异**：LLM 有 causal mask + RoPE 等结构差异
4. **视频已足够有说服力**：视频 DiT 是当前计算量最大、稀疏注意力需求最迫切的应用

#### 适用性判断

| 领域 | 直接适用可能性 | 关键风险 |
|------|:---:|---|
| **图像 DiT** | ⭐⭐⭐⭐⭐ | 几乎无风险，架构本质一致 |
| **长上下文 LLM** | ⭐⭐⭐ | "层内稳定性"需要重新验证；自回归需要适配 |
| **短上下文 LLM** | ⭐⭐ | $n$ 小导致 Theorem 上界不够紧致 |
| **多模态 LLM** | ⭐⭐ | 注意力模式更复杂 |

**最自然的扩展路径**：先在 DiT 图像生成上验证 → 再考虑适配到自回归 LLM。核心障碍不是方法不可行，而是缺少对应领域的实验证据。
