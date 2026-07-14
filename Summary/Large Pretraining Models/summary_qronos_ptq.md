# Qronos: Correcting the Past by Shaping the Future... in Post-Training Quantization

## 论文信息
- **作者**: Shihao Zhang, Haoyu Zhang (UC San Diego), Ian Colbert (AMD), Rayan Saab (UC San Diego)
- **会议**: ICLR 2026
- **链接**: https://arxiv.org/abs/2505.11695

## 核心贡献

### 1. Qronos 算法
提出了一种新的无反向传播的后训练量化（PTQ）算法 Qronos，其核心特点：

- **双重纠错**：显式地纠正来自权重量化和激活量化两方面的误差
- **跨层纠错**：能够纠正来自之前已量化层的残差量化误差
- **误差扩散**：每步交替执行误差纠正（选择最优量化权重）和误差扩散（将取整误差扩散到未来尚未量化的权重中）

算法每步交替优化：
- Step 1（误差纠正）：固定未量化权重，选择当前权重的最优量化值 $q_t$，使得 $\|XW - \sum q_j \widetilde{X}_j\|$ 最小
- Step 2（误差扩散）：固定已量化权重，调整剩余未量化权重 $w^{(t)}_{\geq t+1}$，最优补偿取整误差

**Pipeline 总览**：

```
逐层处理（layer-wise）:
  对于当前层的权重矩阵 W ∈ ℝ^{N×N'}:
    1. 收集 X（原始输入）和 X̃（前层量化后的实际输入）
    2. 预计算 G = X̃ᵀX, H = X̃ᵀX̃, 对 H⁻¹ 做 Cholesky 分解 H⁻¹ = LLᵀ

    对每一列（输出通道）并行:
      t=1:  q₁ = Q(复杂公式，用 X 和 X̃ 重标定位置 1 的最优值)
            w^{(1)}_{≥2} = 全局重算所有剩余辅助权重          ← "根据前层影响更新本层参数"

      t=2..N:  q_t = Q(w^{(t-1)}_t)                          ← "按固定顺序取下一行做 RTN 量化"
               w^{(t)}_{≥t+1} += Δ^{(t)}（Cholesky 更新）     ← "更新未量化的权重"
```

关键设计：仅 $t=1$ 一次复杂计算（$X$ 和 $\widetilde{X}$ 同时介入）完成跨层纠错，后续全部简化为 RTN + Cholesky 更新。量化顺序为固定自然顺序（$1 \to N$），不做贪心选择（区别于 OPTQ）。

### 2. 高效等价实现（关键定理）
证明了 Qronos 存在等价的简化实现（Theorem 1）：
- **从第 2 个量化迭代起（$t \geq 2$）**，$q_t$ 可以直接通过 RTN（最近邻取整）计算，即 $q_t = \mathcal{Q}(w^{(t-1)}_t)$。注意：这里的"第 2 步"指的是同一列权重向量内部交替优化的第 2 次迭代（$t=2$），**不是训练的第二步或 pipeline 的第二个阶段**，只有 $t=1$ 的第一次迭代需要同时用到 $X$ 和 $\widetilde{X}$ 进行特殊计算
- 剩余权重的更新可以利用 Cholesky 分解高效求解
- **内存优化**：首次迭代通过平方矩阵形式（$G = \widetilde{X}^T X$, $H = \widetilde{X}^T \widetilde{X}$）将峰值内存从 $\mathcal{O}(mN)$ 降至 $\mathcal{O}(N^2)$，在 Llama3 8B 上实现了 **18×** 的内存缩减
- **速度提升**：单层微基准测试达到 **13.8×** 加速

### 3. OPTQ 的新理论解释
论文给出了 OPTQ 的一个新颖几何解释（Corollary 1）：
- OPTQ 看似局部的贪婪更新规则，实际上在每个步骤都**最优地纠正了所有之前迭代的累积量化误差**
- 几何上，OPTQ 每步执行最优网格选择后，进行正交投影到由未来数据列张成的低维超平面上
- 但 OPTQ 在处理 $X \neq \widetilde{X}$（即输入漂移）时存在系统性偏差，因为它只最小化 $\|\widetilde{X}(W - Q)\|$ 而非真正的目标 $\|XW - \widetilde{X}Q\|$

### 4. 与 Qronos 的对比
- **GPFQ** 通过路径跟随处理输入不匹配，但当 $X \neq \widetilde{X}$ 时路径尾部不对齐
- **OPTQ** 仅在 $X = \widetilde{X}$ 时探索权重更新，无法处理激活不匹配
- **Qronos** 通过引入辅助权重 $w^{(t)}$ 来自然处理 $X$ 和 $\widetilde{X}$ 之间的偏差

## 实验设置

### 模型与数据集
- **模型**: Llama3 系列（1B, 3B, 8B foundation 和 instruction fine-tuned）和 Qwen3 系列（0.6B - 32B）
- **评估**: WikiText2 困惑度 + 5 个零样本推理任务（ARC, HellaSwag, PIQA, Winogrande）
- **校准**: 128 个随机序列，每序列 2048 tokens，来自 WikiText2

### 对比基线
RTN, OPTQ, GPFQ, GPTAQ

### 量化变换（Stage 1）
- **仅权重量化**: SmoothQuant, MagR, Hadamard-based Incoherence Processing (HIP)
- **权重-激活量化**: QuaRot, SmoothRot, SpinQuant

## 主要实验结果

### 2-bit / 1.58-bit 权重量化（Llama3）
| 模型 | 2-bit WikiText2 ↓ | 1.58-bit WikiText2 ↓ | 
|------|-------------------|---------------------|
| Llama3.2-1B | OPTQ: 24.6 → **Qronos: 17.8** | OPTQ: 200 → **Qronos: 39.3** |
| Llama3.2-3B | OPTQ: 13.2 → **Qronos: 11.4** | OPTQ: 52.0 → **Qronos: 22.8** |
| Llama3.1-8B | OPTQ: 10.4 → **Qronos: 9.3** | OPTQ: 43.3 → **Qronos: 18.0** |

### 2-bit 权重量化（Qwen3 Instruct, 0.6B-32B）
Qronos 在所有模型尺寸上一致优于其他取整方法。例如 Qwen3-32B: WikiText2 从 OPTQ 的 12.8 → **Qronos 12.0**。

### 3-bit / 4-bit 权重量化（Llama3，配合多种变换）
在不同的量化变换（None, SmoothQuant, MagR, HIP）下，Qronos 均持续提供最优结果。特别是 HIP + Qronos 组合达到最佳整体性能。

### W4A4 权重-激活量化（Llama3）
配合 QuaRot、SmoothRot 和 SpinQuant，以及可选的 KV cache 量化（W4A4KV4）：
- SpinQuant + Qronos 达到最佳整体结果
- Qronos 在更具挑战性的场景（W4A4 vs W4, W4A4KV4 vs W4A4）下提供更大改进
- W3A3 实验进一步验证了这一模式

### 运行时分析
- 校准时间开销随模型增大从 19.7%（0.6B）降至 8.7%（32B）
- 算法运行时间相比 Base 版本提升 13.8×（K=1024）

## 关键洞察

1. **输入漂移问题**：量化早期层会改变后续层的输入分布（$\widetilde{X} \neq X$），现有方法大多忽略这一问题
2. **三角不等式解释**：$\|XW - \widetilde{X}Q\| \leq \|(X - \widetilde{X})W\| + \|\widetilde{X}(W - Q)\|$，OPTQ 只修正第二项，Qronos 修正两项
3. **网格调参的局限**：调优缩放因子和零点只能影响 $Q$ 进而影响第二项，无法触及第一项（激活不匹配）

## 局限性
- 实验限于缩放 min-max 量化网格，未探索非均匀网格或向量量化等更复杂的量化方案
- 需要两次前向传播（分别收集 $X$ 和 $\widetilde{X}$），带来额外校准开销

## Q&A 讨论

### Q1: 误差扩散时，未量化的权重是其它层的参数吗？

**答：不是。** 误差扩散始终在同一层、同一列权重向量内部进行。

Qronos 逐层、逐列地量化权重矩阵 $W \in \mathbb{R}^{N \times N'}$。对于某一列 $w \in \mathbb{R}^N$，算法按行索引 $t = 1, \dots, N$ 逐步量化：每一步交替执行误差纠正（选择当前 $q_t$）和误差扩散（将取整误差补偿到同一列中尚未量化的其余位置 $w^{(t)}_{\geq t+1}$）。这些 $w^{(t)}_{\geq t+1}$ 是论文引入的**辅助变量（auxiliary weights）**，算法结束后被丢弃，只有量化值 $q$ 被保留。

"跨层纠错"的含义不是把误差扩散到其它层参数，而是：由于量化前面层会改变当前层的输入（$X \to \widetilde{X}$），Qronos 在目标函数中直接优化 $\min\|Xw - \widetilde{X}q\|^2$，并引入辅助权重来补偿输入漂移带来的影响，这是 OPTQ（只优化 $\min\|Xw - Xq\|^2$）无法做到的。

### Q2: 本文具体如何拆分量化权重和非量化权重，根据什么节奏选择？

**答：Qronos 按固定自然顺序（$t=1, 2, \dots, N$）逐行量化，不做贪心选择。**

**拆分方式**：对权重矩阵 $W \in \mathbb{R}^{N \times N'}$，外层按列（输出通道）独立并行处理，内层按行索引 $t=1 \to N$ 串行迭代。第 $t$ 步时，状态为 $w^{(t-1)} = (q_{\leq t-1}, w^{(t-1)}_{\geq t})$——前 $t-1$ 个已量化固定，剩余 $N-t+1$ 个是连续的辅助权重。

**节奏与关键转折**：
- **$t=1$（第一步）**：付出较大成本，需同时用到 $X$ 和 $\widetilde{X}$ 通过闭式解计算 $q_1 = \mathcal{Q}\left(\frac{G_{1,\geq1}w - H_{1,\geq2}w_{\geq2}}{H_{11}}\right)$，并初始化所有辅助权重
- **$t \geq 2$（后续步骤）**：简化为 RTN + Cholesky 更新，$q_t = \mathcal{Q}(w^{(t-1)}_t)$，权重更新为 $w^{(t)}_{\geq t+1} = w^{(t-1)}_{\geq t+1} - (w^{(t-1)}_t - q_t) \cdot L_{\geq t+1, t}/L_{tt}$

与 OPTQ 通过 Hessian 对角线贪心选择量化顺序不同，Qronos **不做顺序选择**，直接按自然索引依次量化。其性能优势来自优化目标本身（$\min\|Xw - \widetilde{X}q\|^2$）对输入漂移的显式纠正，而非更聪明的顺序策略。

### Q3: 第一次量化不是 RTN 吗？$t \geq 2$ 之后不需要误差扩散了吗？Cholesky 具体怎么分解？

**答：三个子问题分别回答如下：**

**(1) $t=1$ 的计算**：所有 $t$ 的闭式解外层都是 $\mathcal{Q}(\cdot)$（RTN 算子），关键在于内部不同。$t=1$ 时内部是 $\frac{\langle Xw - \sum_{j=2}^N w_j \widetilde{X}_j, \widetilde{X}_1 \rangle}{\|\widetilde{X}_1\|^2}$，需同时涉及 $X$ 和 $\widetilde{X}$ 的复杂内积；$t \geq 2$ 时内部简化为 $w^{(t-1)}_t$（辅助权重本身），无需任何额外计算，这才是"纯 RTN"的含义。

**(2) 误差扩散每步都在执行**：两个定理分别降低两个步骤的成本——Theorem 1 将 $q_t$ 简化 + 将误差扩散从依赖 $(X, \widetilde{X})$ 简化为仅依赖 $\widetilde{X}$；Lemma 1 (Cholesky) 进一步将误差扩散的最小二乘求解从 $O(N^3)$ 加速到 $O(N)$。但误差扩散的物理含义不变：始终将第 $t$ 步的量化误差 $(w^{(t-1)}_t - q_t)$ 最优补偿到剩余权重中。

**(3) Cholesky 分解细节**：预计算 $\widetilde{X}^\top\widetilde{X}$ 的逆的 Cholesky 分解 $H^{-1} = LL^\top$（$L$ 下三角，每层做一次）。每步误差扩散更新为：
$$\Delta^{(t)} = -(w^{(t-1)}_t - q_t) \cdot \frac{L_{\geq t+1,\, t}}{L_{tt}}, \quad w^{(t)}_{\geq t+1} = w^{(t-1)}_{\geq t+1} + \Delta^{(t)}$$
直觉：$L$ 的第 $t$ 列编码了 Hessian 逆矩阵的结构——$\frac{L_{\geq t+1, t}}{L_{tt}}$ 决定了标量量化误差应按什么比例分配到各剩余权重上，整个过程只需一个标量-向量乘法 + 向量加法，$O(N)$ 复杂度。

### Q4: 每步不应该是"先量化，再计算误差"吗？为什么 $t=1$ 需要复杂公式？它取代了 RTN？

**答：$t=1$ 的公式不是在"量化 $w_1$"，而是在重新确定"输入已变化后第 1 个位置的最优值"。**

Qronos 的优化目标是 $\min\|Xw - \widetilde{X}q\|^2$——输入从 $X$ 变成了 $\widetilde{X}$，但输出要匹配原始 $Xw$。$w$ 是针对原始 $X$ 训练的，当输入变为 $\widetilde{X}$ 时，$w_1$ 不再是该位置的最优值。因此 $t=1$ 公式做的是：在量化网格中找一个值 $p$，使得用 $\widetilde{X}_1 p + \sum_{j=2}^N w_j \widetilde{X}_j$ 逼近原始输出 $Xw$。本质是一次「坐标系对齐」——在输入空间改变后重新确定最优起点。

**验证**：当 $X=\widetilde{X}$（无输入漂移）时，$q_1 = \mathcal{Q}\!\left(\frac{\langle w_1 X_1, X_1\rangle}{\|X_1\|^2}\right) = \mathcal{Q}(w_1)$，完美退化为 RTN。$\sum_{j=2}^N w_j \widetilde{X}_j$ 项在 $X=\widetilde{X}$ 时被消掉，这正是它存在的意义——补偿输入漂移。

**$t \geq 2$ 为何能 RTN**：$t=1$ 后 $w^{(1)}_{\geq 2}$ 已被重算为辅助权重，内化了输入漂移的补偿，所以 $w^{(1)}_2$ 本身就是当前条件下该位置的最优值，直接 RTN 即可。

### Q4-补充: $t=1$ 是否等价于「先通过改变的输入调整 $W$，再选第一列做 RTN」？

**答：是的，这个理解非常精准。** $t=1$ 可以分解为两个子步骤，恰好对应这个描述：

**(a) 先计算"输入改变后位置 1 的最优连续值"，再做 RTN：**

$$q_1 = \mathcal{Q}\!\left(\underbrace{\widetilde{X}_1^{\dagger}\left(Xw - \sum_{j=2}^N w_j \widetilde{X}_j\right)}_{\text{在输入 }\widetilde{X}\text{ 下，位置 1 的最优连续值}}\right)$$

其中 $\widetilde{X}_1^{\dagger} = \widetilde{X}_1^\top / \|\widetilde{X}_1\|^2$。括号内是使 $\widetilde{X}_1 \cdot v + \sum_{j=2}^N w_j \widetilde{X}_j \approx Xw$ 的最小二乘最优连续值 $v$，然后用 $\mathcal{Q}(v)$ snap 到离散网格。这相当于先「用 $\widetilde{X}$ 重新标定位置 1 应该取什么值」，再做 RTN。

**(b) 全局调整所有剩余权重（误差扩散）：**

$$w^{(1)}_{\geq 2} = \widetilde{X}_{\geq 2}^{\dagger} \left( Xw - \widetilde{X}_1 q_1 \right)$$

这步一次性将所有剩余辅助权重重算，使得 $q_1$ 确定后的残差被彻底抹平。

**关键直觉**：$t=1$ 做了一次「全局对齐」——将整列辅助权重全部调整到 $\widetilde{X}$ 空间下，此后只需按自然顺序逐个 RTN + Cholesky 微调即可。这正是 Qronos 仅需 $t=1$ 一次复杂计算就能实现跨层纠错的根本原因。

### Q6: 本文的对比方法包括什么？

本文将量化流程分为两个阶段（见论文 Figure 1），Qronos 定位为 **Stage 2（取整方法）**，对比方法也按此划分：

**Stage 2 对比基线（取整方法，Qronos 直接竞争）**：

| 方法 | 特点 |
|------|------|
| **RTN** | 最近邻取整，无数据驱动 |
| **OPTQ (GPTQ)** | 贪心逐列量化 + Hessian 驱动的误差补偿；仅优化 $\|\widetilde{X}(W-Q)\|$，无法处理输入漂移 |
| **GPFQ** | 贪心路径跟随，逐列量化 + 误差扩散；当 $X \neq \widetilde{X}$ 时路径尾部不对齐 |
| **GPTAQ** | OPTQ 的扩展，支持非对称校准 |

**Stage 1 对比方法（量化变换，Qronos 兼容但不竞争）**：

| 类别 | 方法 | 核心思想 |
|------|------|---------|
| 缩放类 | **SmoothQuant** | 将量化难度在 weight 和 activation 之间平滑迁移 |
| | **MagR** | 通过近端梯度下降直接最小化权重的 $\ell_\infty$ 范数 |
| 旋转类 | **HIP** (基于 QuaRot/QuIP#) | 利用 Hadamard 旋转使权重 incoherent，消除 outlier |
| | **QuaRot** | Hadamard 旋转消除激活值 outlier，端到端 4-bit 量化 |
| | **SmoothRot** | 通道级缩放 + 旋转的组合 |
| | **SpinQuant** | 在 Stiefel 流形上学习最优旋转矩阵 |

**实验设计理念**：论文固定 Stage 1，单独评估 Stage 2 的影响。例如：
- 仅权重量化：HIP + MagR 作为 Stage 1，比较不同 rounding 方法
- 权重-激活量化：QuaRot / SmoothRot / SpinQuant 分别作为 Stage 1，比较不同 rounding 方法

### Q7: Qronos 与 QuaRot 有什么区别？

**核心定位不同：两者不竞争，是互补关系。**

| 维度 | Qronos | QuaRot |
|------|--------|--------|
| **Pipeline 位置** | Stage 2：取整方法（Rounding） | Stage 1：量化变换（Transformation） |
| **解决的问题** | 如何将浮点值映射到离散量化网格？ | 如何修改模型使 weight/activation 更易量化？ |
| **核心技术** | 交替误差纠正 + 扩散的贪心取整算法 | Hadamard 旋转消除激活值 outlier |
| **优化目标** | $\min\|XW - \widetilde{X}Q\|^2$（显式纠正前层量化引起的输入漂移） | 计算不变性：$\mathbf{XQ} \cdot \mathbf{Q}^\top\mathbf{W} = \mathbf{XW}$ |
| **推理开销** | **零**（不修改计算图） | 需在线 Hadamard 变换（Walsh-Hadamard, $O(d\log d)$，约 7% 开销） |

**关键洞察（论文原文）**：

> "The latest innovations in PTQ, including QuaRot, SpinQuant, among many others, are skewed towards proposing and improving transformations that address the quantization challenges exacerbated in LLMs. These studies often only consider RTN and OPTQ. Meanwhile, our work explicitly focuses on improving the rounding method while remaining compatible with these transformations."

即：QuaRot 等方法是改进 Stage 1，但 Stage 2 仅用了 RTN/OPTQ；Qronos 专门改进 Stage 2，可以与任何 Stage 1 方法组合。

**实验验证的互补性**：

论文在 W4A4KV4 设置下（Llama3-8B）：

| Stage 1 | Stage 2 | WikiText2 ↓ | 提升 |
|---------|---------|-------------|------|
| QuaRot | RTN | 15.9 | - |
| QuaRot | OPTQ | 10.3 | -5.6 |
| QuaRot | **Qronos** | **9.3** | **-6.6** |
| SpinQuant | RTN | 13.4 | - |
| SpinQuant | OPTQ | 8.9 | -4.5 |
| SpinQuant | **Qronos** | **8.7** | **-4.7** |

即 Qronos + QuaRot 的组合显著优于 QuaRot + OPTQ，证明了**更好的取整方法可以进一步释放变换方法的潜力**。

**三角不等式解释 Qronos 为何优于 OPTQ**：

$$\underbrace{\|XW - \widetilde{X}Q\|}_{\text{Qronos 目标}} \leq \underbrace{\|(X - \widetilde{X})W\|}_{\text{激活漂移项}} + \underbrace{\|\widetilde{X}(W - Q)\|}_{\text{权重误差项（OPTQ 目标）}}$$

- OPTQ 只最小化第二项（权重误差），忽略了第一项（激活漂移）
- Qronos 直接最小化完整目标，通过 $t=1$ 步骤用 $X$ 和 $\widetilde{X}$ 重新标定最优值来纠正输入漂移
- 调优量化网格参数（scale/zero point）只能影响 $Q$ 从而影响第二项，无法触及第一项

### Q5: 论文有验证过从第一步起就直接 RTN + Cholesky 吗？OPTQ vs Qronos 的实验是干净的消融吗？

**答：论文没有做这个消融实验，且 OPTQ vs Qronos 不是对 $t=1$ 步骤的干净消融。**

Qronos 与 OPTQ 之间至少存在**三个独立差异**，不能简单归因于 $t=1$ 特殊步骤：

| 维度 | Qronos | OPTQ |
|------|--------|------|
| **优化目标** | $\min\|Xw - \widetilde{X}q\|^2$ | $\min\|Xw - Xq\|^2$ |
| **量化顺序** | 固定自然顺序 $1,2,\dots,N$ | Hessian 对角线贪心选择 |
| **$t=1$ 特殊步** | 有，需同时用 $X$ 和 $\widetilde{X}$ | 无 |

此外需要纠正之前的表述：**OPTQ 确实也用了类似 Cholesky 的机制**。论文第 393 行明确指出 "The Cholesky reformulation used in Theorem GPTQ1 also resembles the key mechanism in OPTQ"，因此"Cholesky 微调"并非 Qronos 独有。

论文中所有 OPTQ vs Qronos 实验评估的是上述**三个差异叠加的总效果**，无法单独归因于 $t=1$ 步骤的价值。要干净地消融 $t=1$ 步骤，需要在 Qronos 框架内将 $q_1$ 替换为 $\mathcal{Q}(w_1)$（纯 RTN），其余条件（固定顺序 + $\widetilde{X}$ 目标 + Cholesky 更新）完全不变。
