# Veda: Scalable Video Diffusion via Distilled Sparse Attention

- **论文标题**: Veda: Scalable Video Diffusion via Distilled Sparse Attention
- **作者**: Shihao Han, Hao Yang, Xiaofeng Mei, Xinting Hu, Yi Jiang, Xiaojuan Qi (ByteDance, The University of Hong Kong, USTC)
- **会议**: ICML 2026
- **arXiv**: https://arxiv.org/abs/2605.30325

---

## 1. 动机与问题

视频扩散Transformer（Video DiTs）在生成高分辨率、长视频时，自注意力的 **$O(N^2)$ 计算复杂度**成为主要的扩展瓶颈。例如在 Waver-T2V-12B 生成 720P 10 秒视频时，注意力计算占总推理时间的 **92%**。

现有的稀疏注意力方法在 GPU 上的实际加速需要以 **tile（块）** 粒度而非单个 token 粒度跳过计算。因此，稀疏注意力本质上是一个 **tile 选择问题**：将 token 分组为 tile，每个 query tile 只关注少量 key tile。

然而，现有的两类方法在 **高稀疏度（≥90%）** 下都会出现严重的生成质量下降：
- **静态模式方法**（如 SVG、STA）：使用预定义的时空 mask，缺乏对动态注意力分布的适应性。
- **动态方法**（如 VSA、VMOBA）：通过池化特征估计 tile 重要性，但缺乏显式监督来保持全注意力的结构几何，且均值池化无法捕获 tile 内的信号峰值。

## 2. 核心实证发现

作者进行了系统的实证研究，得出以下关键结论：

1. **稀疏度本身不是问题**：使用 full attention 的 Top-k tiles 构造的 "Oracle Mask" 即使在极高稀疏度下也能保持生成质量。
2. **Mask 质量是关键**：生成质量取决于稀疏 mask 与 full attention 的 **tile 级别结构对齐程度**，而非稀疏度比例本身。作者定义了 **Tile Recall@k** 指标来衡量对齐质量。
3. **注意力头的异构性**：不同注意力头在空间-时间维度上表现出显著不同的模式，且随扩散时间步变化，统一的 tiling 策略导致部分头欠聚合或过聚合。

## 3. Veda 方法

Veda 是一个**蒸馏式稀疏注意力框架**，核心思路是将 tile 选择显式地建模为对 full attention 的蒸馏问题。

### 3.1 蒸馏式 Tile 打分（Distilled Tile Scoring）

- **目标构建**：对 full attention 矩阵按 tile 做 **max-pooling**（而非均值池化，因为注意力分布通常是尖峰的），得到参考 tile 分数 $\mathbf{S}^{\text{tgt}}$。
- **TripPool 统计量**：对每个 tile 提取 **三元组统计量** `{Avg, Max, Min}` 拼接，保留 tile 内的重要特征（包括峰值和分布范围）。
- **Head-Specific MLP 投影**：每个注意力头有独立的 MLP 投影 $\phi_q$ 和 $\phi_k$，解决不同头之间的模式差异。
- **蒸馏损失**：使用 KL 散度在 tile 级别对齐预测分数与 target 分数的分布。
- **Stop-Gradient**：对 backbone 特征使用 stop-gradient 操作，**解耦 mask 学习与特征学习**，避免 backbone 被稀疏 mask 学习目标扰动预训练的生成能力。

### 3.2 Head-Aware Tiling（头部感知分块）

- 在固定的硬件 tile 大小预算下（如 $B=128$），每个注意力头 $(l, h)$ 被分配不同的时空 tile 配置 $\pi_{l,h} = (p_t, p_h, p_w)$，满足 $p_t \cdot p_h \cdot p_w = B$。
- 最优配置通过**离线搜索**得到：在校准集上最小化稀疏注意力输出与 full attention 输出的 Frobenius 范数误差。

### 3.3 硬件高效实现

- **Tile-Skipping 稀疏注意力 Kernel**：基于 ThunderKittens DSL 实现，利用 NVIDIA Hopper GPU 的 TMA（异步内存访问）和 Warp Specialization（计算/数据移动解耦），达到 FlashAttention-3 约 **80% 的 MFU**。
- **高效 Ground-Truth Heatmap 生成**：使用 TileLang 实现**两遍 tile 级池化 kernel**，避免存储完整的注意力矩阵，吞吐量达到 FlashAttention-3 的 ~0.9×，且支持**部分 query 监督**加速训练。

## 4. 训练策略

- **两阶段训练**：
  - Stage 1：冻结 backbone，仅训练 score estimator（1000 步，lr=6e-4）。
  - Stage 2：解冻全部参数，进行稀疏微调（backbone lr=6e-5，estimator lr=6e-4）。
- **无需 sparsity warmup**：Stage 1 的 stop-gradient 提供了稳定初始化，Stage 2 直接在目标稀疏度上训练。
- 1B/1.3B 模型训练 23k 步后用 EMA 推理；14B 模型训练 10k 步。

## 5. 实验结果

### 5.1 推理加速

| 模型 | 分辨率/帧数 | End-to-End 加速 | Self-Attn 加速 | 注意力开销降低 |
|------|------------|----------------|---------------|---------------|
| Waver-T2V-12B | 720P / 241帧 | **5.1×** | **10.5×** | 92% → 50% |
| Wan2.1-T2V-14B | 720P / 81帧 | **2.63×** | **7.08×** | — |
| Waver-T2V-1B | 480P / 81帧 | **2.6×** | **3.5×** | — |
| Wan2.1-T2V-1.3B | 480P / 81帧 | **1.59×** | **3.15×** | — |

加速比随序列长度增加而提升，表明 Veda 在长序列场景下具有良好可扩展性。

### 5.2 生成质量

- **Human Evaluation（Waver-bench 1.0）**：Veda 在 90% 稀疏度下与 Full Attention 在视觉质量、运动质量、提示遵循和整体质量四个维度上**无显著差异**（平局率高达 63%）。
- **与 VSA 对比**：Veda 在 95% 稀疏度下优于 VSA 在 87.5% 稀疏度下的表现；在同等 95% 稀疏度下，Veda 视觉质量胜率达 76% vs VSA 的 24%。
- **VBench 量化评估**：Veda 在各项指标上与 Full Attention 相当，同时大幅降低 FLOPs 和 Wall Time。

### 5.3 消融实验

- **TripPool vs 其他池化**：Triplet（Avg+Max+Min）的 MSE loss（0.912）显著优于 Avg pooling（0.965）和 MaxMin（0.982）。
- **Head-Aware Tiling vs 静态 Tiling**：在所有评估维度上一致提升，运动质量 +7.2%，整体质量 +9.6%。

## 6. 理论基础（附录）

论文在附录中提供了完整的理论框架：
- 将注意力视为**能量模型（EBM）**，Log-Sum-Exp 为势函数。mask 错误会通过 softmax 中的分母重归一化被**指数级放大**，导致"结构化幻觉"。
- 论证了均值池化的局限性：均值池化丢弃了二阶统计量（协方差），无法区分均值相似但方向正交的 token。
- 提出基于三元组统计量的注意力重建：利用 **Bhatia-Davis 不等式**，通过极值约束方差，三重统计量 $\{\mu, M, m\}$ 构成 token 分布的"超矩形包围盒"。
- 给出 tile 内聚类条件：tile 内 token 方差需趋向于 0，这通过 Head-Aware Tiling 的**带宽最小化**和**流形嵌入**实现。
- 论证 KL 散度作为蒸馏目标的合理性：KL 散度局部近似于 Fisher 信息度量的自然梯度优化。

## 7. 总结与展望

Veda 提供了一个实用的 video DiT 稀疏注意力解决方案，核心贡献包括：
1. 将稀疏注意力显式建模为 **tile 级别蒸馏问题**，而非依赖隐式学习。
2. **TripPool + Head-Specific MLP** 减少估计误差。
3. **Head-Aware Tiling** 减少结构不匹配。
4. **高效 tile-skipping kernel** 将理论稀疏度转化为实际加速。

未来方向包括：更紧密的 kernel 融合、>95% 稀疏度下的 MFU 保持、跨时间步的 tile 分数缓存、自适应稀疏度分配等。

---

## Q&A 讨论

### Q1: TripPool 统计量是干嘛的？

TripPool 是 Veda 中 tile 打分估计器（score estimator）的核心设计，用来**压缩表示每个 tile 内 token 的信息**，从而在不计算完整 QK 内积的前提下，准确估算 tile 之间的注意力分数。

**为什么需要它？** estimator 不能直接算所有 token 的 QK 内积（那就退化为 full attention 了），必须用 tile 级别的压缩统计量来近似估计。现有方法（如 VSA）只用 Avg pooling，但这有严重缺陷：
- 注意力分布通常是**尖峰分布**，少数 token 贡献了大部分权重。均值池化相当于低通滤波，会把信号峰值淹没在背景噪声中。
- 均值无法反映 tile 内 token 的分布范围（方差），导致两个均值相似但方向正交的 tile 被错误地赋予高相关性。

**TripPool 设计**：对每个 tile 拼接三种统计量 `{Avg, Max, Min}`：
- **Avg**：锚定 tile 的中心位置（一阶矩）
- **Max**：保留注意力中的信号峰值，避免关键 token 被忽略
- **Min**：提供分布下界，与 Max 一起刻画 tile 内 token 的离散程度

论文从理论上证明：QK 内积可分解为均值乘积项 + 协方差项，协方差又可通过 **Bhatia-Davis 不等式**利用极值约束方差 $\sigma^2 \le (M-\mu)(\mu-m)$。三元组 $\{\mu, M, m\}$ 本质上构造了 token 向量的"超矩形包围盒"，保留了对重构注意力分数**最关键的充分统计量**。

消融实验中 TripPool 的 MSE loss 为 **0.912**，明显优于 Avg（0.965）和 MaxMin（0.982）。

---

### Q2: 如何使用 TripPool？

TripPool 在 Veda 的**训练和推理**两个阶段都有使用，完整流程如下：

**训练阶段（蒸馏）：**

1. **分块（Tiling）**：对于某个注意力头的 $\mathbf{Q}, \mathbf{K} \in \mathbb{R}^{N \times d}$，按 head-aware tiling 配置分为 $N_T$ 个 tile，得到 $\widetilde{\mathbf{Q}}, \widetilde{\mathbf{K}} \in \mathbb{R}^{N_T \times B \times d}$。
2. **提取 TripPool 统计量**：对每个 tile 在 $B$ 个 token 维度上分别计算：
   - `Z_q[i] = Avg(Q_tile[i]) ⊕ Max(Q_tile[i]) ⊕ Min(Q_tile[i])` → 维度 $3d$
   - `Z_k[j] = Avg(K_tile[j]) ⊕ Max(K_tile[j]) ⊕ Min(K_tile[j])` → 维度 $3d$
3. **Head-Specific 投影**：每个注意力头 $h$ 有独立的 MLP $\phi_q^{(h)}, \phi_k^{(h)}$，将 $3d$ 维统计量投影到 $d'$ 维隐空间。
4. **计算 tile 分数**：`S_pred[i,j] = φ_q(Z_q[i]) · φ_k(Z_k[j]) / √d'`
5. **蒸馏监督**：将 full attention 矩阵做 max-pool + 行归一化得到 target 分布 $\mathbf{A}^{\text{tgt}}$，对预测分数做 Softmax 得到 $\mathbf{A}^{\text{pred}}$，用 **KL 散度**最小化两者差异。注意 backbone 特征在进入 TripPool 前做了 **stop-gradient**。

**推理阶段：**

1. 按 head-specific tiling 配置对 $\mathbf{Q}, \mathbf{K}$ 分块。
2. 对每个 query tile 和 key tile 提取 TripPool → MLP 投影 → 计算 tile 分数 $\mathbf{S}^{\text{pred}}$。
3. 对每个 query tile 做 **Top-k 选择**，保留分数最高的 $k$ 个 key tile，生成二进制 mask $\widetilde{\mathbf{M}}$。
4. 只将被选中的 key/value tile 送入 sparse attention kernel 计算，其余跳过，实现加速。

**关键点**：TripPool 把一个 $B \times d$ 的 tile 压缩成 $3d$ 维向量，计算量比原始注意力小几个数量级，所以 mask 准备的额外开销极小（论文中仅 **1.9ms**，远低于 sparse attention 的 36.9ms），几乎不拖累整体加速效果。

**澄清 1：蒸馏的目标是 tile 级分数分布，不是 token 级 attention 矩阵。**
TripPool + MLP 做的是**粗粒度的 tile 分数估计**，用来决定"哪些 tile 值得算"。蒸馏的 target 是：
$$\mathbf{A}^{\text{tgt}} = \text{RowNormalize}(\text{MaxPool}(\mathbf{A}^*_{\text{full}}))$$
即把 full attention 的 token 级矩阵先 max-pool 到 tile 级，再做行归一化。所以 KL 散度逼近的是 **tile 级分布**，不是原始的 $N \times N$ attention 矩阵。

**澄清 2：实际的 attention 输出仍由 sparse kernel 用原始 token 计算。**
TripPool + MLP 只负责产出 **tile mask $\widetilde{\mathbf{M}}$**（决定选哪些 key tile），真正的 attention output $\mathbf{O}=\text{softmax}(\mathbf{Q}\hat{\mathbf{K}}^\top/\sqrt{d})\hat{\mathbf{V}}$ 仍然由 token 级别的 $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ 通过 sparse attention kernel 计算，只是 $\hat{\mathbf{K}}, \hat{\mathbf{V}}$ 只包含被选中 tile 中的 token。完整推理流程为：
```
TripPool → MLP → tile 分数 → Top-k mask → sparse kernel 算真实 attention 输出
 ↑_____________ 只看 tile 级别 ________↑    ↑______ 被选中的 token 级别 ______↑
```
TripPool 本质上是"**调度器**"——决定 GPU 把算力花在哪些 tile 上，真正的 attention 计算质量没有被压缩。

---

### Q3: 如何理解时空 tile 配置，是稀疏度吗？

**不是**。时空 tile 配置 $\pi = (p_t, p_h, p_w)$ 和稀疏度是两个完全不同的概念。

**Tile 配置：决定"怎么分组"。** 在硬件 tile 大小固定的前提下（如 $B = 128$），三维时空的 token 需要"拍扁"成一维再分组。$\pi$ 定义了分组的形状：

| 配置 | $p_t$ (时间) | $p_h$ (高度) | $p_w$ (宽度) | 含义 |
|------|------|------|------|------|
| (4, 4, 8) | 4 | 4 | 8 | 时间中等，空间偏宽 |
| (8, 4, 4) | 8 | 4 | 4 | 时间为主，空间均分 |
| (2, 8, 8) | 2 | 8 | 8 | 空间为主，时间少 |

约束：$p_t \times p_h \times p_w = B = 128$。

**稀疏度：决定"选多少"。** 由 Top-k 控制：每个 query tile 在所有 key tile 中选分数最高的 $k$ 个。如果总共有 $N_T$ 个 key tile，选 $k$ 个，则稀疏度 $= 1 - k/N_T$。95% 稀疏度意味着每个 query tile 只看 5% 的 key tile。

**两者的关系：** Tile 配置影响的是每组 token 内部的相似度（方差）。tile 内 token 越相似，TripPool 统计量越准确。Head-Aware Tiling 为每个头选不同的 $(p_t, p_h, p_w)$，让 tile 内部"抱团紧密"，从而在**相同稀疏度**下获得更准确的 tile 选择。

简单总结：
- $\pi = (p_t, p_h, p_w)$ → 决定 **token 如何分组**
- Top-k → 决定 **每组选多少 key tile**（稀疏度）
- Head-Aware Tiling → 先最大化 tile 质量，再施加稀疏度

---

### Q4: 最优配置具体如何计算？

最优 tile 配置 $\pi^*_{l,h}$ 通过**离线穷举搜索**确定，不是在线学习的。

**搜索空间：** 硬件 tile 大小固定 $B = 128$，候选配置是所有三维因式分解：
$$\Omega = \{(p_t, p_h, p_w) \in \mathbb{N}^3 \mid p_t \cdot p_h \cdot p_w = 128\}$$
共约 30~40 种候选，极小离散集合。

**搜索过程（每层每头独立）：** 对每个校准样本 $x$，逐个 layer $l$、head $h$、候选 $\pi$：

1. **获取 full attention 输出作为 ground truth：**
   $$\mathbf{A}^{fu} = \text{Softmax}\left(\frac{\mathbf{QK}^\top}{\sqrt{d}}\right), \quad \mathbf{O}^{fu} = \mathbf{A}^{fu} \cdot \mathbf{V}$$

2. **按候选配置 tiling 并构造 Oracle mask：** 把 $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ 按 $\pi$ 重新分组 → 计算 tiled full attention → max-pool 得到 tile 级分数 → 每个 query tile 取 Top-k 构造 Oracle mask $\widetilde{\mathbf{M}}$。

3. **计算稀疏 attention 误差：** 用 $\widetilde{\mathbf{M}}$ 跑 sparse attention 得到 $\mathbf{O}^{sp}$，与 $\mathbf{O}^{fu}$ 对比 Frobenius 误差：
   $$\mathcal{E}_{l,h,\pi} = \sum_{x \in \mathcal{D}_{cal}} \|\mathbf{O}^{fu}_{l,h} - \mathbf{O}^{sp}_{l,h}(x; \pi)\|_F^2$$

4. **选择最优：** $\pi^*_{l,h} = \arg\min_{\pi \in \Omega} \mathcal{E}_{l,h,\pi}$

| 特点 | 说明 |
|------|------|
| **离线** | 训练前一次性完成，推理直接用 |
| **每层每头独立** | 不同 (l, h) 可选不同配置 |
| **直接优化输出保真度** | 最小化 attention output 的 Frobenius 误差 |
| **校准集很小** | 只需少量样本 |
| **搜索空间小** | 仅几十种因式分解 |

本质上：**哪个分组方式能让 Oracle mask 的 sparse attention 输出最贴近 full attention，就选哪个。**

---

### Q5: 最优配置得在 Distilled Tile Scoring 训练完成之后再进行？

**不需要，恰恰相反。** Head-Aware Tiling 搜索在 score estimator **训练之前**完成。

搜索过程完全不依赖 TripPool 或 MLP estimator，只用预训练 backbone 的 full attention：
- 直接用 full attention 做 max-pool 构造 Oracle mask（不需要 estimator 预测）
- 用 Oracle mask 跑 sparse attention 对比输出误差值

**完整流程时序：**
```
① Head-Aware Tiling 搜索（离线，仅需预训练 backbone）
     ↓ 确定每层每头的 tile 配置 π*_{l,h}
② Stage 1：在选定的 tiling 下冻结 backbone，训练 TripPool + MLP estimator
     ↓
③ Stage 2：在选定的 tiling 下解冻 backbone，sparse 微调
```

---

### Q6: 论文整体流程总结及常见误解修正

**用户总结（已修正）：** 本文先通过 Head-Aware Tiling 搜索每层每头的最佳分块配置；然后训练一个简单 MLP，输入是每个分块的三维统计特征（TripPool），输出是低维的分块表达，用于 block-wise 分数估计，这个估计结果通过 KL 散度与真实 full attention 聚合出来的 block-wise 分数分布进行学习。最终利用估计出的 block-wise 分数进行 Top-k 选择，决定哪些 tile 参与 sparse attention 计算。

**关键修正：MLP 是门控/路由器，不是替代计算模块。**

MLP 的输出 **不是用于替代 attention 计算**，而是决定 **哪些 tile 参与 attention 计算**：

```
TripPool → MLP → tile 分数 S_pred
                         ↓
                  Top-k 选择 → 二进制 mask M̃
                                      ↓
                  sparse kernel: softmax(Q·K̂^T/√d)·V̂  ← 仍用原始 token 计算！
```

MLP 只产出 **mask（调度信号）**，真正的 attention 权重和输出值仍由 token 级别的 Q、K、V 在 sparse kernel 中完整计算，没有被 MLP 的低维表达替代。

**完整流程：**
```
① Head-Aware Tiling → 确定每层每头的分组形状 (p_t, p_h, p_w)
② TripPool + MLP → 输出 tile 分数，用于 Top-k 选择生成 mask
③ KL 蒸馏 → 让 step ② 的 tile 分数分布逼近 full attention 的 tile 级分布
④ 推理时 → 用 MLP 生成的 mask 调度 sparse kernel，在原 token 上算真实 attention
```

**一句话总结：** MLP 是门控路由器，决定 GPU 算力花在哪些 tile 上；真正的 attention 计算质量由原始 token 的 sparse attention kernel 保证，从未被压缩。
