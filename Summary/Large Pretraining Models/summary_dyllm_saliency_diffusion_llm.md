# DyLLM: Efficient Diffusion LLM Inference via Saliency-based Token Selection and Partial Attention

> **论文标题**: DyLLM: Efficient Diffusion LLM Inference via Saliency-based Token Selection and Partial Attention
> **作者**: Younjoo Lee, Seungkyun Dan, Junghoo Lee, Jaiyoung Park, Jung Ho Ahn（首尔大学）
> **会议**: ICML 2026
> **arXiv ID**: 2603.08026
> **代码**: https://github.com/scale-snu/DyLLM.git

---

## 核心速览：综合理解与点评

### 用户综合理解

本文做 Diffusion-LLM 加速。通过分析相邻 step's token 在 attention 后的余弦距离，找到重要 Token。对于重要 Token，执行原本的计算逻辑；对于不重要 Token，在 FFN 时复用上一层的计算结果。本文提出的非显著 Token 在 Attention 上的近似计算没有理论加速。

### 点评

这段理解**核心洞察完全正确**，精准抓住了论文的骨架。有两处细微修正：

1. **"复用上一层的计算结果" → 应为"复用上一步（前一个去噪步）的缓存结果"**
   - FFN 复用发生在**同一层、跨去噪步（step）**：SparseStep 中非显著 token 直接用 `FFN_OUT_cache`，这个缓存来自上一个去噪步在同一层计算的 FFN 输出。
   - 不是跨层（layer）复用——层级之间是逐层传递的，每层独立维护自己的缓存。

2. **"Attention 上的近似计算没有理论加速" → 部分正确**
   - ✅ 对的一面：伪代码层面 `S = Q@K^T` 和 softmax 仍是 $O(N^2 d)$，没有渐近节省；论文声称的 $O(N\|\mathcal{A}\|d)$ 依赖自定义 CUDA 核（FlashAttention 风格）只处理显著 key tile，伪代码未体现。
   - ⚠️ 需修正：Attention 的**第二阶段 A@V → A_sal@ΔV 确实有列稀疏理论加速**（$O(N^2 d) \to O(N\|\mathcal{A}\|d)$），并非完全无加速。
   - 🎯 但你的核心判断是对的：**Attention 的节省远小于 FFN 跳过**。真正的加速大头是 FFN（占 70%+ 推理时间）从 $O(N d^2)$ 降到 $O(\|\mathcal{A}\| d^2)$，而非 attention 稀疏化。这也是 GQA 模型（Dream）加速比更大的原因——GQA 降低 attention 成本占比，让 FFN 主导地位更突出。

**一句话总结**：DyLLM 的加速本质是"用余弦相似度选 token，让大部分 token 跳过 FFN 并复用缓存"，attention 的列稀疏是次要优化。

---

## 1. 核心问题

掩码扩散语言模型（MDLM，如 LLaDA、Dream、Gemini Diffusion）通过并行 token 解码打破了自回归（AR）模型的顺序限制，提供了更高生成吞吐的潜力。然而，其迭代去噪过程存在一个根本性的效率困境：

- **AR 模型**：每个 token 只在一次前向传播中计算，天然支持增量 KV 缓存，仅需计算新增 token。
- **MDLM**：由于双向注意力机制会在每次精化步骤中更新全局上下文，模型必须在每个去噪步中**重复处理完整序列**。这导致去噪步类似于"重复 prefill"操作，计算开销随迭代次数增长而激增。

论文运行时分析表明，原始扩散 LLM 实现中每个步骤的计算主要由 FFN 操作主导，且无法像 AR 那样享受 KV 缓存带来的增益。

---

## 2. 关键洞察：时序稀疏性

通过分析去噪过程中注意力上下文 $C_{t,l}^{(i)}$ 的演化，论文定义了**时序余弦相似度**：

$$s_{t,l}^{(i)} = \frac{C_{t,l}^{(i)} \cdot C_{t-1,l}^{(i)}}{\|C_{t,l}^{(i)}\| \|C_{t-1,l}^{(i)}\|}$$

在 $T=256$ 个扩散步上的分布揭示两个关键发现：

1. **大多数 token 高度稳定**：在所有层中，分布都集中在 $s_{t,l}^{(i)} \approx 1.0$ 附近，表明相邻迭代间注意力上下文基本不变。
2. **稳定性具有层依赖性**：浅层几乎全部 token 都高度相似，而深层呈现更宽的分布和更长的低相似度尾部——即更深层有更多 token 发生有意义更新。

这一时序稀疏性意味着：只有一小部分 token（论文定义为 **salient tokens，显著 token**）真正对下一步更新有贡献，其余 token 的计算是冗余的。

---

## 3. 方法：DyLLM

DyLLM 是一个**免训练（training-free）**的推理框架，基于两个核心观察：(1) MDLM 的层间时序稀疏性；(2) Transformer 块中的位置级 delta 传播。

推理分为两个阶段：
- **FullStep 阶段**（前 $T_{full}=4$ 步）：标准前向传播，缓存 K、V、注意力上下文 C、FFN 输出。
- **SparseStep 阶段**：仅对显著 token 重新计算，复用缓存。

### 3.1 层自适应显著 Token 选择

在步骤 $t$、层 $l$ 的显著 token 集合定义为：

$$\mathcal{A}_{t,l} = \{ i \mid s_{t,l}^{(i)} < \tau \}$$

其中 $\tau$ 为选择阈值。对于非显著 token，跳过后续 FFN 计算并复用 FFN 输出缓存；仅显著 token 执行完整重计算。

- **浅层**：大部分位置敏感度低，可激进剪枝计算。
- **深层**：敏感度更高，基于阈值的策略自动扩展显著 token 集合以保障生成质量。

### 3.2 显著感知近似注意力

识别显著 token 还可优化注意力机制。设 $\Delta C_{t,l}=C_{t,l}-C_{t-1,l}$，$\Delta V_{t,l}=V_{t,l}-V_{t-1,l}$，注意力更新可分解为：

$$\Delta C_{t,l} = S_{t,l}\Delta V_{t,l} + (\Delta S)V_{t-1,l}$$

这揭示了两个传播通道：(1) token 内容更新（$\Delta V$）；(2) 注意力权重重路由（$\Delta S$）。基于此，采用**双路径更新策略**：

- **显著路径（精确行更新）**：对 $i \in \mathcal{A}_{t,l-1}$ 的 token，完全重计算注意力输出矩阵第 $i$ 行。该计算沿 query 维度稀疏——只重计算显著 query 的行，但每个被选 query 仍 attend 所有 key/value。
- **非显著路径（近似更新）**：对 $i \notin \mathcal{A}_{t,l-1}$，query 近似平稳，$\Delta S \approx 0$，更新简化为 $\Delta C_{t,l}^{(i)} \approx S_{t,l}^{(i,\cdot)} \Delta V_{t,l}$。由于 $\Delta V$ 仅在显著 token 索引处非零，更新只需注意力矩阵中对应显著 token 的列。

该策略将注意力复杂度从 $O(N^2 d)$ 降至 $O(N \cdot |\mathcal{A}_{t,l-1}| d)$，其中 $|\mathcal{A}|$ 通常仅占 $N$ 的一小部分。

### 3.3 仅响应步骤（Response-only Step）

基于 RoPE 的相对距离衰减特性，显著 token 主要集中在响应（response）区域。DyLLM 采用自适应策略：

- **响应-only 步**（占总步数 75%）：仅处理响应 token 序列。
- **全序列步**（每 4 步一次）：输入完整 prompt + response 序列，但即便如此，昂贵计算仍只限于显著 token。

这避免了现有缓存方法中周期性全序列刷新带来的吞吐瓶颈。

---

## 4. 理论基础：误差界

论文提出两个命题，为余弦相似度作为选择指标提供理论依据：

### Proposition 1：线性投影下的尺度不变性

设 $W_o$ 为输出投影矩阵，$\alpha > 0$ 为缩放因子，则：

$$\text{RMSNorm}((\alpha C) W_o) = \text{RMSNorm}(C W_o)$$

**含义**：注意力上下文的**幅值**不影响后续 FFN 的输入，只有投影向量的**方向对齐**才重要。

### Proposition 2：方向对齐下的误差界

设 $\delta = \|\text{RMSNorm}(C_{t,l} W_o) - \text{RMSNorm}(C_{t-1,l} W_o)\|_2$ 为层 $l$ 的近似误差，则在 $W_o$ 良态条件下：

$$\delta \le \kappa(W_o)\sqrt{2(1 - s_{t,l})}$$

其中 $\kappa(W_o) = \sigma_{\max}(W_o)/\sigma_{\min}(W_o)$ 为条件数。

**含义**：
- 当 $s_{t,l}^{(i)} \to 1$（token 近平稳），跳过 FFN 引入的误差几乎为零。
- 低 $s_{t,l}^{(i)}$ token 时间偏移大，对生成质量至关重要，应保留为显著 token。
- 通过阈值化 $s_{t,l}$，DyLLM 隐式控制了跨层误差传播预算。

---

## 5. 算法流程

### 高层编排（Algorithm 1）

```
输入: prompt P (长度 L_P), response 长度 L_R, 总步数 T_total,
      完整步数 T_full, 余弦相似度阈值 τ
初始化: R ← [mask] * L_R; 初始化 K_cache, V_cache, C_cache, FFN_OUT_cache
for t = 0 to T_total-1:
    if t < T_full:
        x = Concat(P, R); FullStep(x, caches)
    else:
        if t % 4 == 0: x = Concat(P, R)   // 全序列步
        else: x = R                       // 响应-only 步
        x, idx_sal = SparseStep(x, caches, idx_sal, τ)
    decoded_tokens, decoded_positions = process_logit(x)
    R[decoded_positions] = decoded_tokens
```

### FullStep（Algorithm 2）

标准 Transformer 前向传播，额外缓存 K、V、注意力上下文 C、FFN 输出，供 SparseStep 复用。

### SparseStep（Algorithm 3）

核心效率来源：
1. 仅对显著 token 投影新的 K、V，非显著 token 复用缓存。
2. 计算 $\Delta V = V[idx\_sal] - V_{cache}[idx\_sal]$。
3. 对显著 query 精确计算注意力 $C_{sal}$；对非显著 query 用近似注意力估计残差 $\Delta C$。
4. 合并得到新的 C，与缓存 C 比较余弦相似度，选出下一层显著 token。
5. 仅对新显著 token 应用 FFN，非显著 token 复用 FFN 输出缓存。

### Approximate Attention（Algorithm 4）

1. 计算注意力分数 $S = QK^\top$，Softmax 得 A。
2. 提取 A 中对应显著 token 的列：$A_{sal} = A[:, idx\_sal]$。
3. 残差更新：$\Delta C = A_{sal} \Delta V$。

---

## 6. 实验结果

### 6.1 实验设置

- **硬件**：单张 NVIDIA H100 PCIe 80GB（B200 扩展实验见附录）
- **模型**：LLaDA 8B Instruct、Dream 7B Instruct
- **基准**：GSM8K（数学推理）、MBPP（代码生成）、MATH（数学）、MMLU-Pro（通用知识）
- **基线**：原始实现、Fast-dLLM（PrefixCache / DualCache）、dLLM-Cache
- **吞吐量**：每秒生成 token 数
- 实现采用 PyTorch + 自定义 CUDA 核（FlashAttention 风格融合设计）

### 6.2 主要结果（$n_u=1$，H100）

| 模型 | 基准 | 原始准确率 | DyLLM 最优准确率 | 原始吞吐 | DyLLM 吞吐 | 加速比 |
|------|------|-----------|-----------------|---------|-----------|--------|
| LLaDA 8B | GSM8K | 77.79 | 79.08 | 11.47 | 87.21 | **×7.60** |
| LLaDA 8B | MATH | 33.22 | 38.68 | 15.81 | 106.06 | ×6.71 |
| Dream 7B | GSM8K | 75.59 | 79.30 | 12.57 | 120.62 | **×9.60** |
| Dream 7B | MATH | 37.60 | 45.12 | 17.64 | 142.34 | ×8.07 |

关键观察：
- **准确率保持甚至提升**：在多数基准上 DyLLM 准确率不降反升，归因于 softmax 归一化的去噪效应——抑制低相关 token 的噪声贡献。
- **Dream 加速比更大**：GQA 降低了注意力成本，FFN 占 Dream 推理时间 70%+，更适合显著 token 选择性 FFN 执行。
- **比 dLLM-Cache 快 2.16–3.67×**：避免了固定计算 token 数和繁琐超参调优。

### 6.3 阈值 $\tau$ 的影响

- 降低 $\tau$ 减少计算 token 数、提升吞吐，但过低会导致误差累积而降低准确率。
- 阈值主要**模型相关**而非任务相关：LLaDA 选 $\tau=0.99$，Dream 选 $\tau=0.995$，一次校准即可跨基准泛化。

### 6.4 并行解码度 $n_u$ 的可扩展性

- **Fast-dLLM 的局限**：依赖周期性全序列刷新步骤，随 $n_u$ 增大，刷新频率上升，有效计算量急剧增长，严重限制吞吐。
  - 示例（1024 prompt + 256 response，$n_u=1$）：PrefixCache 平均每步 179.5 token，DualCache 平均 71 token，而刷新步需处理 1280 token。
- **DyLLM 的优势**：无任何需要重计算全序列的步骤，每个步骤都保持稀疏，随 $n_u$ 增长可扩展性显著优于 Fast-dLLM。

### 6.5 与置信感知并行解码的兼容性

DyLLM 与置信感知并行解码方案天然兼容，能维持或提升平均每步解掩码 token 数（avg $n_u$），同时保持更高准确率。在 Dream 上，Fast-dLLM 明显降准确率，而 DyLLM 反而获得更多收益。

---

## 7. 与现有工作的对比

| 方法 | 策略 | 局限 |
|------|------|------|
| dKV-Cache | 周期性刷新 KV | 固定调度，忽略层间动态 |
| Fast-dLLM (Prefix/Dual) | 块级前缀/掩码缓存 | 固定块规则可能遗漏关键 token；刷新步成为吞吐瓶颈 |
| dLLM-Cache | 基于 value 变化的 token 级缓存 | 需逐模型/数据集调超参（$K_P, K_R$）；固定计算 token 数 |
| Elastic-Cache | 基于注意力权重的自适应参与 | 从某层起计算所有 token，未充分利用层间异质性 |
| **DyLLM** | 每层每步动态选择显著 token + 近似注意力 | 依赖模型时序稀疏度和阈值选择 |

**DyLLM 的差异化**：在**每个层和每个去噪步**自适应选择 token 子集，而非固定调度或全局阈值，充分利用层间和 token 间的表示稳定性异质性。

---

## 8. 内存开销与局限

- **额外内存**：相对仅存 KV 缓存，DyLLM 还需缓存注意力上下文 C 和 FFN 输出，内存增长因子为 $(2d/g + 2d)/(2d/g)$（$g$ 为 GQA 共享头数，MHA 时 $g=1$）。
- **实际影响有限**：扩散 LLM 每步处理的 token 数仍比 AR 多数十倍，GPU 常在较小 batch 即达峰值吞吐，额外内存不会带来同等程度的吞吐下降。
- **局限**：性能依赖模型暴露的时序稀疏度和阈值 $\tau$ 选择；阈值需逐模型校准（但可跨任务泛化）。

---

## 9. 总结

DyLLM 抓住了扩散 LLM 推理中的核心冗余：去噪步间大多数 token 表示稳定，仅少量显著 token 发生有意义变化。通过层自适应的显著 token 选择跳过冗余 FFN、显著感知的近似注意力降低二次复杂度、以及仅响应步骤减少处理范围，DyLLM 在免训练条件下实现了：

- LLaDA 上最高 **7.6×**、Dream 上最高 **9.6×** 吞吐提升；
- 在推理、代码、通用基准上**保持或提升**准确率；
- 随并行解码度 $n_u$ 增长的**鲁棒可扩展性**，优于依赖全序列刷新的现有方法。

理论上，两个命题（尺度不变性 + 条件数误差界）为余弦相似度作为显著性指标提供了严格支撑，证明了近平稳 token 跳过计算的误差可控性。

---

## 10. 技术讨论 Q&A

### Q1: $s_{t,l}^{(i)}$ 是注意力输入输出的同一个 Token 的余弦相似度吗？

**不是输入输出之间的比较，而是同一 token 在相邻两个时间步的注意力输出之间的比较。**

$s_{t,l}^{(i)}$ 衡量的是**同一个 token $i$** 的**注意力上下文输出**（attention context，即 $C = \text{softmax}(QK^T)V$ 的第 $i$ 行）在**相邻两个去噪步** $t$ 和 $t-1$ 之间的余弦相似度：

$$s_{t,l}^{(i)} = \frac{C_{t,l}^{(i)} \cdot C_{t-1,l}^{(i)}}{\|C_{t,l}^{(i)}\| \|C_{t-1,l}^{(i)}\|}$$

- $C_{t,l}^{(i)}$：token $i$ 在层 $l$、步 $t$ 的注意力**输出**（论文原文："attention context vector ... computed via softmax-normalized attention scores and value vectors"）。
- 比较对象是**步 $t$ vs 步 $t-1$ 的同一个 token 输出**，不是注意力的输入 vs 输出，也不是不同 token 之间的相似度。
- 用注意力输出而非 hidden state 的原因：Proposition 2 严格证明了 RMSNorm 后 FFN 输入的近似误差与 $s_{t,l}$ 直接相关（$\delta \le \kappa(W_o)\sqrt{2(1-s_{t,l})}$），为跳过低相似度 token 的 FFN 提供理论依据。

### Q2: 显著路径是执行正常的 self-attention 更新吗？还是有特别改进？

**对显著 query 本身是标准精确 attention，但整体不是完整的 self-attention——它是"行稀疏"的。**

从 Algorithm 3 看：`C_sal = attention(Q[idx_sal], K, V)`，只有显著 query 参与，但每个显著 query 都 attend **完整的 K 和 V 矩阵**：

- 显著 query 的注意力计算（QKV 投影、softmax、加权求和）是标准的、**无近似**的精确计算，注意力机制本身没有架构级修改。
- **但只计算显著 query 的行**，非显著 query 的行被跳过（由近似残差代替）。
- 改进点在于 **query 维度的稀疏化**：只对需要更新的显著 query 做完整注意力，避免对稳定 token 做无用的行计算。

### Q3: 非显著路径是用 Token 间的余弦相似度推导成轻量化近似 Attention，且只对非显著 token 使用？

**需要澄清：余弦相似度不是用来推导近似 attention 公式的，它只负责筛选显著 token。** 近似 attention 的推导是独立的另一条路径：

1. **Delta 分解**（Eq. context_decomp）：
   $$\Delta C_{t,l} = S_{t,l}\Delta V_{t,l} + (\Delta S)V_{t-1,l}$$
   注意力上下文变化来自两个通道：value 内容变化（$\Delta V$）和注意力权重重路由（$\Delta S$）。

2. **平稳假设**：对非显著 token，query 近似不变 → $\Delta S^{(i,\cdot)} \approx 0$，简化为：
   $$\Delta C_{t,l}^{(i)} \approx S_{t,l}^{(i,\cdot)} \Delta V_{t,l}$$

3. **利用 $\Delta V$ 的行稀疏性**：$\Delta V$ 只在显著 token 索引处非零（只有显著 token 重算了 V），故只需取注意力矩阵中对应显著 token 的列。

Algorithm 4 实现：`S = QK^T → Softmax → A_sal = A[:, idx_sal] → ΔC = A_sal @ ΔV`（列稀疏）。

关于"只对非显著 token 使用"：
- 近似 attention 实际对**所有 query** 都计算了 $\Delta C$（Algorithm 3 行 243）。
- 但合并时（行 245）：显著 token 用精确值 $C_{sal}$ **覆盖**近似值；非显著 token 用 $C_{cache} + \Delta C$。
- 故近似结果**最终只对非显著 token 生效**。

**总结关系**：余弦相似度 $s_{t,l}$ 负责"选人"（识别显著 token）；Delta 分解 + 平稳假设负责"造公式"（近似 attention 的数学形式）。二者正交，非推导关系。

### Q4: SparseStep 和 Approximate Attention 有什么区别？什么时候用哪个？

它们是**算法层次上的包含关系**，不是二选一：

```
DyLLM (Algorithm 1)
  ├─ FullStep (Algorithm 2)        ← 预热阶段 t < T_full 用
  └─ SparseStep (Algorithm 3)     ← 加速阶段 t ≥ T_full 用（顶层流程）
        └─ Approximate Attention (Algorithm 4)  ← SparseStep 内部的子例程
```

**SparseStep** 是完整的稀疏前向传播步骤，是加速阶段每一步的顶层编排，包含 7 个子步骤：
1. 仅对显著 token 投影 K、V（非显著复用缓存）
2. 计算 $\Delta V$
3. 显著 query 的精确注意力（行稀疏）
4. 调用 **Approximate Attention** 计算残差 $\Delta C$
5. 合并得到新 C，用余弦相似度选出下一层显著 token
6. 仅对显著 token 应用 FFN
7. 更新所有缓存

**Approximate Attention** 是 SparseStep 内部的一个子操作，专门计算非显著 token 的注意力上下文残差更新。它不是独立选择的——只要进入 SparseStep，每一层都会调用它一次。

**何时用哪个**：
- **FullStep**：预热阶段（前 $T_{full}=4$ 步），完整计算所有 token，初始化并稳定缓存。
- **SparseStep**：预热后的所有步骤，是 DyLLM 加速的主体框架。
- **Approximate Attention**：SparseStep 内部自动调用的组件，无需单独决策。

### Q5: 详细介绍近似 Attention 的推导，如整个公式

近似 attention 的核心目标：在不做完整 $O(N^2 d)$ 注意力计算的前提下，估计注意力上下文的残差更新 $\Delta C$。推导分 7 步：

#### 第 0 步：标准注意力定义

层 $l$、去噪步 $t$ 的注意力上下文矩阵：

$$C_{t,l} = S_{t,l} V_{t,l}$$

其中 $S_{t,l} = \text{softmax}(Q_{t,l} K_{t,l}^\top / \sqrt{d})$ 为注意力分数矩阵（$N \times N$），$V_{t,l}$ 为 value 矩阵（$N \times d$）。$C_{t,l}$ 的第 $i$ 行就是 token $i$ 的注意力上下文输出。

#### 第 1 步：定义时间变化量（Delta）

$$
\Delta C_{t,l} = C_{t,l} - C_{t-1,l}, \quad
\Delta S_{t,l} = S_{t,l} - S_{t-1,l}, \quad
\Delta V_{t,l} = V_{t,l} - V_{t-1,l}
$$

且 $S_{t,l} = S_{t-1,l} + \Delta S_{t,l}$，$V_{t,l} = V_{t-1,l} + \Delta V_{t,l}$。

#### 第 2 步：展开 $\Delta C$（核心分解）

$$
\begin{aligned}
\Delta C_{t,l} &= S_{t,l} V_{t,l} - S_{t-1,l} V_{t-1,l} \\
&= (S_{t-1,l} + \Delta S_{t,l})(V_{t-1,l} + \Delta V_{t,l}) - S_{t-1,l} V_{t-1,l} \\
&= \underbrace{S_{t-1,l} V_{t-1,l}}_{\text{抵消}} + S_{t-1,l} \Delta V_{t,l} + \Delta S_{t,l} V_{t-1,l} + \underbrace{\Delta S_{t,l} \Delta V_{t,l}}_{\text{二阶小项}} - \underbrace{S_{t-1,l} V_{t-1,l}}_{\text{抵消}}
\end{aligned}
$$

交叉项抵消后：$\Delta C_{t,l} = S_{t-1,l} \Delta V_{t,l} + \Delta S_{t,l} V_{t-1,l} + \Delta S_{t,l} \Delta V_{t,l}$。

#### 第 3 步：忽略二阶小项 + 改写为 $S_{t,l}$ 形式

$\Delta S \cdot \Delta V$ 是二阶小项，可忽略。再利用 $S_{t-1,l} = S_{t,l} - \Delta S_{t,l}$ 代入第一项，再次忽略二阶小项，得到论文最终形式（Eq. context_decomp）：

$$\boxed{\Delta C_{t,l} = S_{t,l} \Delta V_{t,l} + (\Delta S_{t,l}) V_{t-1,l}}$$

#### 第 4 步：物理解释——两个传播通道

| 通道 | 公式项 | 物理含义 |
|------|--------|---------|
| 内容更新 | $S_{t,l} \Delta V_{t,l}$ | value 矩阵变化（token 内容改变），即使注意力权重不变，上下文也会变。$\Delta V$ 来自 token hidden state 更新。 |
| 权重重路由 | $(\Delta S_{t,l}) V_{t-1,l}$ | token 的 query 变化，导致注意力权重重新分配（"谁关注谁"变了），即使 value 不变，上下文也会变。 |

#### 第 5 步：非显著 Token 的平稳假设

对非显著 token $i \notin \mathcal{A}_{t,l-1}$，假设 query 近似不变 → 第 $i$ 行注意力权重几乎不变：

$$\Delta S_{t,l}^{(i,\cdot)} \approx \mathbf{0}$$

代入分解公式，非显著 token 的上下文更新简化为：

$$\Delta C_{t,l}^{(i)} \approx S_{t,l}^{(i,\cdot)} \cdot \Delta V_{t,l}$$

只剩**内容更新通道**，权重重路由通道被假设消失。物理含义：非显著 token 的上下文变化完全由其他 token 的 value 变化驱动，其自身注意力权重保持不变。

#### 第 6 步：利用 $\Delta V$ 的行稀疏性 → 列稀疏计算

$\Delta V_{t,l} = V_{t,l} - V_{t-1,l}$ 是**行稀疏**的：只有显著 token 重算了 KV 投影，非显著 token 复用缓存，故 $\Delta V$ 在非显著位置为零。设显著索引集为 $\mathcal{A}$：

$$\Delta V_{t,l}^{(j)} = \begin{cases} V_{t,l}^{(j)} - V_{t-1,l}^{(j)} & j \in \mathcal{A} \\ \mathbf{0} & j \notin \mathcal{A} \end{cases}$$

于是矩阵乘法只需取 $S$ 中对应显著 token 的列：

$$\Delta C_{t,l}^{(i)} \approx \sum_{j \in \mathcal{A}} S_{t,l}^{(i,j)} \cdot \Delta V_{t,l}^{(j)} = S_{t,l}^{(i, \mathcal{A})} \cdot \Delta V_{t,l}^{(\mathcal{A})}$$

完整 $S \cdot \Delta V$ 需 $O(N^2 d)$，利用行稀疏后只需 $S$ 的 $|\mathcal{A}|$ 列，复杂度降至 $O(N \cdot |\mathcal{A}| \cdot d)$。这就是"列稀疏"的来源。

#### 第 7 步：Algorithm 4 的实现

```python
# 输入: Q (所有query), K (完整key), ΔV (仅显著token的value残差), idx_sal
S = Q @ K.T                    # [L, L_P+L_R]  注意力分数
A = softmax(S)                 # [L, L_P+L_R]  归一化
A_sal = A[:, idx_sal]          # [L, |idx_sal|]  列稀疏：只取显著token列
ΔC = A_sal @ ΔV                # [L, d]  轻量化残差更新
return ΔC
```

对应推导链：`S` → $S_{t,l}$；`A_sal = A[:, idx_sal]` → 利用 $\Delta V$ 行稀疏性（第 6 步）；`ΔC = A_sal @ ΔV` → 实现 $\Delta C^{(i)} \approx S^{(i,\mathcal{A})} \cdot \Delta V^{(\mathcal{A})}$（第 5-6 步）。

#### 推导链总览

```
C = S·V                                    标准注意力
  ↓ 定义 ΔC, ΔS, ΔV
ΔC = (S_{t-1}+ΔS)(V_{t-1}+ΔV) - S_{t-1}V_{t-1}   展开
  ↓ 交叉项相消 + 忽略二阶小项
ΔC ≈ S_{t-1}ΔV + ΔS·V_{t-1}
  ↓ 改写 S_{t-1} = S_t - ΔS，再忽略二阶小项
ΔC = S_t·ΔV + ΔS·V_{t-1}                   Eq. context_decomp（两通道）
  ↓ 非显著token假设 ΔS^{(i,·)} ≈ 0
ΔC^{(i)} ≈ S_t^{(i,·)}·ΔV                  只剩内容更新通道
  ↓ ΔV 行稀疏（只有显著token重算V）
ΔC^{(i)} ≈ S_t^{(i, A)}·ΔV^{(A)}           列稀疏计算
  ↓ Algorithm 4 实现
A_sal = A[:, idx_sal]; ΔC = A_sal @ ΔV     O(N·|A|·d)
```

### Q6: 非显著 Token 最终的计算公式是怎样的？

非显著 token 在 SparseStep 中几乎不做新计算，最终结果由"缓存值 + 近似残差"组成，分两个层面：

#### 层面 1：注意力上下文（Attention Context）

对非显著 token $i \notin \mathcal{A}_{t,l-1}$，最终注意力上下文是**缓存的旧上下文 + 近似残差更新**：

$$\boxed{C_{t,l}^{(i)} = C_{t-1,l}^{(i)} + \Delta C_{t,l}^{(i)} \approx C_{\text{cache}}^{(i)} + S_{t,l}^{(i, \mathcal{A})} \cdot \Delta V_{t,l}^{(\mathcal{A})}}$$

- $C_{\text{cache}}^{(i)} = C_{t-1,l}^{(i)}$：上一步缓存的注意力上下文（旧值直接复用）。
- $S_{t,l}^{(i, \mathcal{A})}$：当前步注意力矩阵 $S_{t,l} = \text{softmax}(QK^\top)$ 的第 $i$ 行中，对应显著 token 索引集 $\mathcal{A}$ 的列（列稀疏）。
- $\Delta V_{t,l}^{(\mathcal{A})} = V_{t,l}^{(\mathcal{A})} - V_{t-1,l}^{(\mathcal{A})}$：仅显著 token 的 value 残差（只有显著 token 重算了 V）。

对应 Algorithm 3 行 245：`C[idx_sal^c] = C_cache[idx_sal^c] + ΔC[idx_sal^c]`，其中 `ΔC = A_sal @ ΔV`（Algorithm 4）。

**物理含义**：非显著 token 自身注意力权重不变（$\Delta S \approx 0$），上下文更新完全来自显著 token 的 value 变化——其他显著 token 的内容更新通过它原有的注意力权重"流"到了它这里。

#### 层面 2：FFN 输出

非显著 token 的 FFN **完全跳过**，直接复用上一步的 FFN 输出缓存：

$$\boxed{x_{\text{ffn}}^{(i)} = \text{FFN\_OUT}_{\text{cache}}^{(i)}}$$

对应 Algorithm 3 行 248：`x[idx_sal^c] = FFN_OUT_cache[idx_sal^c]`，无任何新计算。

#### 端到端路径对比

| 计算阶段 | 显著 token $i \in \mathcal{A}$ | 非显著 token $i \notin \mathcal{A}$ |
|---------|-------------------------------|-------------------------------------|
| KV 投影 | 重算 $K^{(i)}, V^{(i)}$ | 复用缓存 $K_{\text{cache}}^{(i)}, V_{\text{cache}}^{(i)}$ |
| 注意力 | 精确计算 $C^{(i)} = \text{attn}(Q^{(i)}, K, V)$（行稀疏） | $C^{(i)} = C_{\text{cache}}^{(i)} + S^{(i,\mathcal{A})} \cdot \Delta V^{(\mathcal{A})}$（近似列稀疏） |
| FFN | 重算 $\text{FFN}(x^{(i)})$ | 直接复用 $\text{FFN\_OUT}_{\text{cache}}^{(i)}$ |
| 缓存更新 | 更新所有缓存项 | 更新所有缓存项（用上述近似值） |

#### 关键点

非显著 token 的最终计算只有一处新计算：列稀疏矩阵乘法 $S^{(i,\mathcal{A})} \cdot \Delta V^{(\mathcal{A})}$（复杂度 $O(|\mathcal{A}| \cdot d)$ per token），其余全部复用缓存。相比完整注意力 $O(N \cdot d)$ 和完整 FFN $O(d^2)$，计算量大幅降低，这正是 DyLLM 加速的核心来源。

### Q7: 近似残差中用到的 S（Attention Map）也需要重新计算？在哪里节省了计算量？

**是的，S = Q@K^T 确实需要重新计算。** 这是一个非常尖锐的问题，切中了伪代码与实际复杂度的差距。计算量节省分三层分析：

#### 第一层：S 确实要重算（伪代码层面）

Algorithm 4 伪代码显示 `S = Q @ K.T` 是完整计算，softmax 也是完整计算。因为 K 矩阵是混合的（显著 token 重算 + 非显著 token 复用缓存），K 本身变了，Q 是当前步新 query，所以 Q@K^T 必须重新算。**伪代码层面，注意力分数计算没有渐近节省。**

论文声称的 $O(N \cdot |\mathcal{A}| \cdot d)$ 总复杂度可能依赖**自定义 CUDA 核**（FlashAttention 风格融合设计）：核函数只处理显著 key/value 位置的 tile，跳过非显著 key 的 K 列计算，从而在实现层面避免完整 QK^T。但这在伪代码中没有体现，且会引入额外近似（softmax 只在 $|\mathcal{A}|$ 个显著 key 上归一化，而非完整 $N$ 个）。

#### 第二层：Attention 内部的真实节省

虽然 S 的计算在伪代码层省不掉，但 attention 的**第二阶段**（value 加权求和）确实省了：

| Attention 子步骤 | 完整注意力复杂度 | DyLLM 近似复杂度 | 是否节省 |
|-----------------|----------------|-----------------|---------|
| QK^T 分数计算 | $O(N^2 d)$ | $O(N^2 d)$（伪代码）/ $O(N\|\mathcal{A}\|d)$（CUDA 核） | 伪代码层未省，实现层可省 |
| Softmax 归一化 | $O(N^2)$ | $O(N^2)$ / $O(N\|\mathcal{A}\|)$ | 伪代码层未省 |
| **加权求和 A@V** | $O(N^2 d)$ | $O(N \cdot \|\mathcal{A}\| \cdot d)$ | **明确节省** |
| 精确注意力（显著 query 行） | — | $O(\|\mathcal{A}\| \cdot N \cdot d)$（行稀疏） | 比完整少 $\frac{N - \|\mathcal{A}\|}{N}$ |

attention 内部最明确的节省是 **A@V → A_sal@ΔV**：完整注意力要算 $N \times N$ 的 A 乘 $N \times d$ 的 V（$O(N^2 d)$），DyLLM 只取 A 的 $|\mathcal{A}|$ 列乘 $\Delta V$（$O(N|\mathcal{A}|d)$）。当 $|\mathcal{A}| \ll N$ 时这部分节省显著。

#### 第三层：真正的大头节省不在 Attention，而在 FFN

论文运行时分析显示 **FFN 占推理时间 70%+**（尤其 Dream）。DyLLM 最大的节省来自 FFN 跳过，而非 attention：

| 计算模块 | 完整计算量 | DyLLM 计算量 | 节省幅度 |
|---------|-----------|-------------|---------|
| **FFN**（主导开销） | $O(N \cdot d^2)$ | $O(\|\mathcal{A}\| \cdot d^2)$（非显著跳过） | **巨大**（$N \to \|\mathcal{A}\|$） |
| KV 投影 | $O(N \cdot d^2)$ | $O(\|\mathcal{A}\| \cdot d^2)$ | 大 |
| Q 投影 | $O(N \cdot d^2)$ | $O(N \cdot d^2)$（无法跳过） | 无 |
| Attention 分数 | $O(N^2 d)$ | $O(N^2 d)$ / $O(N\|\mathcal{A}\|d)$（依赖实现） | 小到中 |
| Attention 加权求和 | $O(N^2 d)$ | $O(N\|\mathcal{A}\|d)$ | 中 |
| Response-only 步 | — | N 从 $L_P+L_R$ 降到 $L_R$ | 额外降 N |

**核心结论**：DyLLM 的加速主要不是来自 attention 稀疏化，而是来自 **FFN 跳过**（占 70%+ 开销）和 **KV 投影减少**。Attention 的列稀疏是锦上添花，但不是决定性的。这也解释了为什么 GQA 模型（如 Dream）加速比更大——GQA 降低了 attention 相对成本，让 FFN 主导地位更突出，DyLLM 的 FFN 跳过收益更明显。

#### 直觉总结

把 DyLLM 想象成团队分工：
- **FFN（占 70%+ 工作量）**：只让显著 token 小组干活，非显著 token 全部跳过 → 最大节省
- **KV 投影（次要）**：同样只让显著 token 干
- **Attention（占比小）**：虽然地图（S）要重新画，但搬运工（A@V）只跑显著 token 这条路线
