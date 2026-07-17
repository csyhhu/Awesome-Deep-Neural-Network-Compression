---
title: KronQ: LLM Quantization via Kronecker-Factored Hessian
arxiv: https://arxiv.org/abs/2607.07964
authors: Donghyun Lee, Yuhang Li, Ruokai Yin, Priyadarshini Panda
affiliation: University of Southern California, Yale University
---

# KronQ: 基于 Kronecker 分解 Hessian 的 LLM 量化

## 概述

KronQ 是一种后训练量化（PTQ）框架，通过引入梯度协方差 $\mathbf{H}_G$ 到量化流程中，挑战了现有二阶 PTQ 方法（如 GPTQ）仅依赖输入激活统计的假设。在 Kronecker 分解 Hessian 近似下，量化损失同时依赖激活和梯度协方差，KronQ 在两个互补层面利用这一点：

1. **双向非相干处理（BiIP）**：将现有的输入侧随机旋转扩展到输出维度，使用梯度协方差降低输入和输出维度的权重幅度方差
2. **基于联合 Hessian 迹的层间混合精度分配**：从梯度和激活 Hessian 迹导出新的敏感度指标

## 核心贡献

### 1. Kronecker 分解量化误差

现有 PTQ 方法（如 GPTQ）的量化目标仅使用输入激活协方差 $\mathbf{H}_X$ 作为代理 Hessian，丢弃了输出侧信息。在 Kronecker-factored approximation (K-FAC) 下，完整权重 Hessian 分解为：

$$\mathbf{H} \approx \mathbf{H}_X \otimes \mathbf{H}_G$$

其中 $\mathbf{H}_G = \mathbb{E}[\mathbf{gg}^\top]$ 是梯度协方差。KronQ 将 $\mathbf{H}_G$ 纳入量化目标，形成 Kronecker 分解量化目标：

$$\min_{\widehat{\mathbf{W}}} \mathrm{tr}\left[\mathbf{H}_G\,\Delta\mathbf{W}\,\mathbf{H}_X\,\Delta\mathbf{W}^\top\right]$$

### 2. 双向非相干处理（BiIP）

现有方法（如 QuIP）仅对输入侧进行非相干处理。KronQ 发现 $\mathbf{H}_G$ 同样具有高度相干性，因此提出双向非相干处理：

- **对角缩放**：同时对输入和输出维度进行对角缩放
- **正交变换**：使用随机 Hadamard 变换使 $\mathbf{H}_G$ 和 $\mathbf{H}_X$ 都变得非相干

实验表明，仅处理 $\mathbf{H}_X$ 时输出通道的变异系数（CV_out）几乎不变，而双向处理能同时降低 CV_in 和 CV_out。

### 3. 层间混合精度分配

基于联合 Hessian 迹 $\mathrm{tr}(\mathbf{H}_G) \cdot \mathrm{tr}(\mathbf{H}_X)$ 作为敏感度评分，解决了 Q、K、V 投影共享相同输入统计导致的退化问题。

## 实验结果

### 权重-only 量化

在 LLaMA-2 和 LLaMA-3 系列模型（7B 到 70B）上进行 W4/W3/W2 量化，KronQ 在几乎所有设置下都取得了最低的 WikiText-2 perplexity，在 2-bit 时增益最为显著：

| 模型 | 方法 | W2 PPL | W3 PPL | W4 PPL |
|------|------|--------|--------|--------|
| LLaMA-2-7B | GPTQ | 31.11 | 6.74 | 5.82 |
| | GPTAQ | NaN | 6.21 | 5.69 |
| | **KronQ** | **8.15** | **5.84** | **5.56** |
| LLaMA-2-13B | GPTQ | 35.89 | 9.41 | 5.37 |
| | GPTAQ | 20.43 | 6.52 | 5.08 |
| | **KronQ** | **6.99** | **5.18** | **4.95** |
| LLaMA-2-70B | GPTQ | 9.54 | 5.47 | 3.50 |
| | GPTAQ | 6.41 | 3.93 | 3.46 |
| | **KronQ** | **5.14** | **3.66** | **3.40** |
| LLaMA-3-70B | GPTQ | 2.6e3 | 2.6e3 | 27.49 |
| | GPTAQ | NaN | 1.6e4 | 399.46 |
| | **KronQ** | **7.93** | **4.41** | **3.25** |

### 组量化（Group Quantization）

在 W2 组量化（g=128）下，KronQ 保持领先优势。例如在 LLaMA-2-7B 上，GPTQ 退化到 274.0 PPL，而 KronQ 达到 7.61。

### 权重-激活量化（W2A4）

使用 QuaRot 框架进行激活量化，KronQ 在 LLaMA-2-7B 上将 PPL 从 36.74（GPTQ）降低到 9.38。

### 泛化到新模型和更难基准

在 Gemma-3-12B、DeepSeek-R1-Distill-Llama-8B 和 Phi-4-mini-instruct 上，KronQ 在所有 8 个设置中均获胜。在 GPQA-Diamond、MMLU、AIME-2024 和 LiveCodeBench 等更难基准上，KronQ 在 W4 下均优于 GPTQ 和 GPTAQ。

### 混合精度

KronQ 的联合评分 $\mathrm{tr}(\mathbf{H}_G) \cdot \mathrm{tr}(\mathbf{H}_X)$ 产生比仅激活评分更好的 PPL-比特权衡，在平均约 2.6 比特时达到比 W3 基线更低的 perplexity。

### 推理效率

- **内存**：W4 时 VRAM 减少 3.5-3.9 倍，W2 时减少 4.0-7.5 倍
- **延迟**：解码速度提升 1.25-2.51 倍
- **部署**：bf16 下需要两张 80GB A100 的 70B 模型，在 W4 下可在单张 A100 上运行

## 消融实验

1. **基础量化器**：将 GPTAQ 替换为 GPTQ 会导致 PPL 显著上升，确认漂移校正和 BiIP 是互补的
2. **对角缩放**：移除缩放保留 Hadamard 旋转会导致 PPL 下降，说明对角缩放是必要的
3. **非相干方向**：仅输入侧非相干（$\mathbf{H}_X$ only）已有改善，但仅输出侧（$\mathbf{H}_G$ only）在 LLaMA-2-7B/13B 上严重退化，双向处理效果最佳

## 与本仓库主题的关联

本文属于大模型量化领域的前沿研究，与本仓库中的量化主题高度相关。其核心创新在于：

1. **二阶方法的扩展**：将 K-FAC 引入 PTQ，超越了 GPTQ 系列仅使用 $\mathbf{H}_X$ 的局限
2. **超低位量化**：在 2-bit 场景下表现出色，这是当前量化研究的热点方向
3. **混合精度分配**：利用梯度信息进行更精细的比特分配，这也是量化领域的重要研究方向

## 代码

GitHub: https://github.com/Intelligent-Computing-Lab-Panda/KronQ

## 讨论记录

### Q1: 介绍一下算法流程

KronQ 的算法流程分为四个主要阶段：

**阶段一：统计收集**

通过一次前向-后向传播遍历校准集，收集二阶统计信息：
- $\mathbf{H}_X = \mathbf{X}\mathbf{X}^\top$：输入激活协方差（GPTQ/GPTAQ 已使用）
- $\mathbf{H}_G = \mathbb{E}[\mathbf{gg}^\top]$：梯度协方差（KronQ 新增）

**阶段二：双向非相干处理（BiIP）**

1. **对角缩放**：同时对输入和输出维度进行缩放
   $$\mathbf{W} \leftarrow \mathbf{S}_G \cdot \mathbf{W} \cdot \mathbf{S}_X$$
   其中 $\mathbf{S}_X$ 和 $\mathbf{S}_G$ 分别基于 $\mathbf{H}_X$ 和 $\mathbf{H}_G$ 计算

2. **正交变换**：使用随机 Hadamard 变换使 $\mathbf{H}_G$ 和 $\mathbf{H}_X$ 都变得非相干

**阶段三：量化**

使用 GPTAQ 的列-wise OBS 求解器，关键发现是 $\mathbf{H}_G$ 在更新中代数上抵消，因此量化步骤与 GPTAQ 完全相同，保持计算效率。混合精度分配基于联合 Hessian 迹评分。

**阶段四：推理**

反转 BiIP 变换后进行推理，对角缩放反转成本可忽略，Hadamard 变换引入 $\Theta(d \log d)$ 开销。

完整流程：校准集 → 统计收集 → BiIP 预处理 → GPTAQ 量化 → 反转变换 → 量化模型

### Q2: 对比本文和 GPTQ、GPTAQ 的目标函数，$\mathbf{H}_X$ 和 $\mathbf{H}_G$ 起到了什么作用

**三者目标函数对比（原文精确表达）**：

| 方法 | 目标函数（原文 Eq.） |
|------|---------------------|
| GPTQ | $\min_{\widehat{\mathbf{W}}} \|\mathbf{W}\mathbf{X} - \widehat{\mathbf{W}}\mathbf{X}\|_F^2 = \mathrm{tr}\!\left[(\mathbf{W} - \widehat{\mathbf{W}})\, \mathbf{H}_X\, (\mathbf{W} - \widehat{\mathbf{W}})^\top\right]$（Eq. 1） |
| GPTAQ | $\min_{\widehat{\mathbf{W}}} \|\mathbf{W}\widetilde{\mathbf{X}} - \widehat{\mathbf{W}}\mathbf{X}\|_F^2 = \mathrm{tr}\!\left[\Delta\mathbf{W}\,\mathbf{H}_X\,\Delta\mathbf{W}^\top - \mathbf{W}\,\Delta\mathbf{X}\mathbf{X}^\top\,\Delta\mathbf{W}^\top\right]$（Eq. 2） |
| KronQ（基础） | $\min_{\widehat{\mathbf{W}}} \mathrm{tr}\!\left[\mathbf{H}_G\,\Delta\mathbf{W}\,\mathbf{H}_X\,\Delta\mathbf{W}^\top\right]$（Eq. 3） |
| KronQ（完整） | $\min_{\widehat{\mathbf{W}}} \mathrm{tr}\!\left[\mathbf{H}_G\!\left(\Delta\mathbf{W}\,\mathbf{H}_X\,\Delta\mathbf{W}^\top - \mathbf{W}\,\Delta\mathbf{X}\mathbf{X}^\top\,\Delta\mathbf{W}^\top\right)\right]$（Eq. 4） |

其中 $\Delta\mathbf{W} = \mathbf{W} - \widehat{\mathbf{W}}$，$\Delta\mathbf{X} = \widetilde{\mathbf{X}} - \mathbf{X}$（$\widetilde{\mathbf{X}}$ 为全精度模型激活，$\mathbf{X}$ 为量化模型激活）。

**$\mathbf{H}_X$ 的作用（已有）**：
- $\mathbf{H}_X = \mathbf{X}\mathbf{X}^\top$，输入激活协方差
- 在 OBS 更新中决定量化误差分配方式
- 用于输入侧非相干处理

**$\mathbf{H}_G$ 的作用（KronQ 新增）**：

1. **双向非相干处理（BiIP）**：
   - $\mathbf{H}_G = \mathbb{E}[\mathbf{g}\mathbf{g}^\top]$，梯度协方差
   - 输出侧对角缩放：$\mathbf{S}_G = \mathrm{diag}\!\left(\frac{[\mathbf{H}_G]_{ii}}{\|\mathbf{W}_{i,:}\|^2}\right)^{1/4}$
   - 输出侧正交变换：$\mathbf{H}_G \leftarrow \mathbf{U} \cdot \mathbf{H}_G \cdot \mathbf{U}^\top$
   - 解决 $\mathbf{H}_G$ 高度相干（$\mu$-incoherence 可达 0.99）的问题

2. **混合精度分配**：
   - 敏感度评分：$s_\ell = \mathrm{tr}(\mathbf{H}_G^{(\ell)}) \cdot \mathrm{tr}(\mathbf{H}_X^{(\ell)})$（原文 Eq. 5）
   - 解决 Q、K、V 投影共享相同 $\mathbf{H}_X$ 导致的退化问题

**$\mathbf{H}_G$ 的特殊性质**：
- 在列-wise OBS 更新中代数上抵消，不增加量化步骤的计算复杂度（Proposition 1）
- 价值体现在预处理阶段和混合精度分配
- KronQ 继承了 GPTAQ 的计算效率

**直观理解**：$\mathbf{H}_X$ 告诉我们"输入哪些部分重要"，$\mathbf{H}_G$ 告诉我们"输出哪些部分重要"，两者结合才能做出更精确的量化决策。

### Q3: 我的理解是在 GPTAQ 的基础上，用更精确的方法估计 $\mathbf{H}_X$，以及加上了一个新的调整矩阵 $\mathbf{H}_G$？另外为什么需要加 trace？

**关于 $\mathbf{H}_X$ 的澄清**：

您的理解不完全准确。KronQ **并没有改变 $\mathbf{H}_X$ 的估计方式**：
- GPTQ/GPTAQ/KronQ 三者都使用相同的 $\mathbf{H}_X = \mathbf{X}\mathbf{X}^\top$（输入激活协方差）
- $\mathbf{H}_X$ 的计算方式完全相同，都是通过一次前向传播收集

**关键区别在于 $\mathbf{H}_G$**：
- GPTQ/GPTAQ 隐含假设 $\mathbf{H}_G = \mathbf{I}$（所有输出通道同等重要）
- KronQ 显式估计 $\mathbf{H}_G = \mathbb{E}[\mathbf{g}\mathbf{g}^\top]$，通过一次额外的反向传播收集

$\mathbf{H}_G$ 不是"调整矩阵"，而是梯度协方差，它提供了输出侧的二阶信息。

**关于 $\mathrm{tr}()$（迹）的说明**：

原文确实使用了 $\mathrm{tr}$，这是基于以下原因：

1. **目标函数中的 $\mathrm{tr}$**：
   - GPTQ：$\|\mathbf{W}\mathbf{X} - \widehat{\mathbf{W}}\mathbf{X}\|_F^2 = \mathrm{tr}\!\left[(\mathbf{W} - \widehat{\mathbf{W}})\, \mathbf{H}_X\, (\mathbf{W} - \widehat{\mathbf{W}})^\top\right]$
   - KronQ：$\mathrm{tr}\!\left[\mathbf{H}_G\,\Delta\mathbf{W}\,\mathbf{H}_X\,\Delta\mathbf{W}^\top\right]$
   - 这是 Frobenius 范数的迹表达式，$\|A\|_F^2 = \mathrm{tr}(A^\top A)$

2. **敏感度评分中的 $\mathrm{tr}$（原文 Eq. 5）**：
   $$\mathbb{E}[\mathcal{L}_\ell] \propto \mathrm{tr}(\mathbf{H}_G^{(\ell)}) \cdot \mathrm{tr}(\mathbf{H}_X^{(\ell)})$$
   - 在二阶近似下，期望量化损失与 $\mathrm{tr}(\mathbf{H})$ 成正比（$\mathrm{tr}(\mathbf{H})$ 衡量损失函数的总曲率）
   - 根据 Kronecker 分解性质：$\mathrm{tr}(\mathbf{H}_X \otimes \mathbf{H}_G) = \mathrm{tr}(\mathbf{H}_X) \cdot \mathrm{tr}(\mathbf{H}_G)$
   - $\mathrm{tr}()$ 将矩阵压缩为标量评分，用于子层排序和混合精度分配

3. **计算效率**：$\mathrm{tr}(\mathbf{H}_X) = \sum_i [\mathbf{H}_X]_{ii} = \|\mathbf{X}\|_F^2$，可以在计算 $\mathbf{H}_X$ 时顺便得到

**核心思想**：$\mathbf{H}_G$ 引入了输出侧信息，而 $\mathrm{tr}()$ 将矩阵压缩为标量，使得我们可以用一个数字来衡量每个子层的敏感度，从而进行混合精度分配。

### Q4: 经过阶段三之后已经完成了量化，阶段四有什么额外的工作呢？本文应该只对 Parameters 做量化，Activation 还是保持原来的精度？

**阶段四的额外工作：反转变换**

阶段四不是继续量化，而是**反转阶段二（BiIP）中应用的变换**。在阶段二中，我们对权重做了：
1. **对角缩放**：$\mathbf{W} \leftarrow \mathbf{S}_G \cdot \mathbf{W} \cdot \mathbf{S}_X$
2. **正交变换**：$\mathbf{W} \leftarrow \mathbf{U} \cdot \mathbf{W} \cdot \mathbf{V}^\top$

这些变换是为了让量化更精确，但推理时必须反转这些变换才能得到正确的输出。根据原文（第304行）：
- $\mathbf{S}_X$ 和 $\mathbf{S}_G$ 通过逐元素操作反转，成本可忽略
- $\mathbf{U}$ 和 $\mathbf{V}$ 的逆变换引入 $\Theta(d_{\mathrm{in}} \log d_{\mathrm{in}} + d_{\mathrm{out}} \log d_{\mathrm{out}})$ 的每层开销

**关于激活量化**

您的理解不完全准确。本文考虑了三种量化方案（原文第394行）：

| 方案 | 权重 | 激活 | 说明 |
|------|------|------|------|
| (i) Weight-only quantization | W2/W3/W4 | **A16** | 只量化权重，激活保持16位 |
| (ii) Group quantization | W2/W3/W4, g=128 | **A16** | 组量化，激活保持16位 |
| (iii) Weight-and-activation quantization | W2 | **A4** | 权重和激活都量化 |

论文的主要实验集中在方案 (i)（WxA16），这也是量化领域的主流设置。方案 (iii) 使用 QuaRot 框架对激活进行量化，KronQ 在 W2A4 下仍然优于 GPTQ 和 GPTAQ。