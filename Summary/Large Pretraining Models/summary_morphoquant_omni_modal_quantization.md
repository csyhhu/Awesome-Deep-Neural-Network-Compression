# MorphoQuant: Modality-Aware Quantization for Omni-modal Large Language Models

**论文链接**: [arXiv:2606.04349](https://arxiv.org/abs/2606.04349)  
**收录**: ECCV 2026  
**作者**: Yue Wu, Changyuan Wang, Zixuan Wang, Shilin Ma, Yansong Tang  
**机构**: Tsinghua University

---

## 1. 研究动机

全模态大语言模型（Omni-modal Large Language Models, OLLMs），如 Qwen2.5-Omni，能够同时处理音频、视频、图像和文本等多种模态输入，展现了卓越的跨模态理解和交互能力。然而，其巨大的模型规模和内存消耗严重阻碍了在资源受限设备上的实际部署。

虽然训练后量化（Post-Training Quantization, PTQ）是提升推理效率的关键技术，但将量化精度推至极低的 **W4A4**（4-bit 权重 + 4-bit 激活）时，全模态模型面临 **灾难性的性能退化**。

### 核心问题：跨模态分布鸿沟（Cross-Modal Distribution Chasm）

传统的 VLMs 仅处理双模态（图像+文本），而全模态模型汇聚了异构的感知数据，不同模态具有本质上截然不同的激活分布特征，文章通过实验揭示了三种关键现象：

1. **显著的离群值长尾**：特定通道中出现极端的离群值，主导了整个动态范围，使得最优量化阈值选择变得极其困难。
2. **跨模态形态差异**：不同模态的激活幅度和离群值形态根本不同，全局统一缩放因子要么严重剪裁某些模态的关键高幅度特征，要么将其他模态的细微特征淹没在舍入噪声中。
3. **语义边界退化**：注意力图可视化显示，最严重的量化退化恰好发生在模态间的语义边界上，破坏了跨模态推理。

---

## 2. 方法

针对上述挑战，作者提出了 **MorphoQuant**，一个面向全模态大语言模型的模态感知 PTQ 框架。

### 2.1 Distribution-Aware Bias Compensation (DABC)

**核心思想**：将激活值中被剪裁的离群点信息通过通道级偏置（bias）进行补偿，在不引入混合精度计算开销的前提下保护关键离群点。

**具体设计**：

1. **离差分数（Dispersion Score）**：对每个通道 c，定义离差度量：

   $$\mathcal{D}_c = \frac{|\max(\mathbf{X}_c) - \min(\mathbf{X}_c)|}{\|\overline{\mathbf{X}_c}\|_1 + \epsilon}$$

   分子度量数据分布的范围（离群值会导致该值显著增大），分母为通道均值的 L1 范数，代表密集内点（inlier）的密度。

2. **通道筛选**：按离差分数降序排列，通过阈值 τ 标记显著通道（论文中选择前 5% 的通道），生成二值掩码 $\mathbf{m}$。

3. **偏置补偿前向传播**：

   $$\mathbf{Y} = (\mathbf{W}_Q \mathbf{X}_Q) + \mathbf{b} + \underbrace{\mathbf{W}_Q \left( (\mathbf{X} - \mathbf{X}_Q) \odot \mathbf{m} \right)}_{\Delta \mathbf{b}_{comp}}$$

   离群点的截断残差被投影通过量化后的权重，然后以高精度折叠到偏置项中。

**硬件友好性**：与 ATOM 等混合精度方案不同，DABC 严格保持**纯 4-bit 稠密矩阵乘法**作为计算主体，高精度离群点校正仅作为轻量级的稀疏矩阵乘法和逐元素偏置加法。这种架构解耦使其天然兼容高度优化的 4-bit CUDA 内核。

### 2.2 Morphology-Directed Quantization Function Optimization (MDQFO)

在 DABC 剥离离群点后，剩余的激活分布呈现出**零对称主体 + 单边长尾**的形态特征。MDQFO 利用这一形态特性，进一步优化量化网格。

**关键设计**：

1. **对称量化区间**：限制为 $[-\alpha_c, \alpha_c]$，其中 $\alpha_c$ 为可学习的剪裁边界。

2. **复合损失函数**：
   - **$\ell_p$-norm 损失**（取代传统 MSE，对残差噪声不那么敏感）：
     $$\mathcal{L}_{p} = \frac{1}{N} \sum_{i=1}^{N} \sqrt{|\mathbf{X}_{FP}^{(i)} - \hat{\mathbf{X}}_Q^{(i)}(\alpha_c)|}$$
   - **余弦相似度损失**（保护跨模态语义对齐的高维特征方向）：
     $$\mathcal{L}_{cos} = 1 - \frac{1}{M} \sum_{j=1}^{M} \frac{\mathbf{X}_{FP}^{(j)} \cdot \hat{\mathbf{X}}_Q^{(j)}(\alpha_c)}{\|\mathbf{X}_{FP}^{(j)}\|_2 \|\hat{\mathbf{X}}_Q^{(j)}(\alpha_c)\|_2}$$
   - **总损失**：$\min_{\alpha_c} \mathcal{L}_{total} = \mathcal{L}_{p} + \lambda_{cos} \mathcal{L}_{cos}$

3. **协同搜索**：可学习的剪裁边界 $\alpha_c$ 与偏置补偿掩码联合优化，通过迭代自动收敛到平衡点——偏置补偿专门处理极少数极端尖峰，而收紧的剪裁阈值精细捕获绝大多数零对称内点的语义信息。

---

## 3. 实验

### 实验设置

- **模型**: Qwen2.5-Omni (3B)
- **评估基准**：覆盖全模态谱系
  - **ScienceQA** & **MMMU**：复杂视觉-语言推理
  - **Video-MME**：长上下文视频理解
  - **AIR-Bench**：密集音频信号理解
- **校准数据**：仅使用 128 个多模态样本
- **对比基线**：QLoRA (W4A16)、Naive W4A4、Q-VLM、PoMQ-ViT
- **DABC 阈值**：标记前 5% 的通道

### 主要结果

| 方法 | W/A Bits | ScienceQA | MMMU | Video-MME | AIR-Bench |
|------|----------|-----------|------|-----------|-----------|
| FP16 (Oracle) | 16/16 | 79.39 | 44.78 | 57.33 | 66.25 |
| QLoRA | 4/16 | 75.88 | 43.56 | **54.78** | 64.12 |
| Naive W4A4 | 4/4 | 73.17 | 40.00 | 47.52 | 63.40 |
| Q-VLM | 4/4 | 73.85 | 40.33 | 48.41 | 63.71 |
| PoMQ-ViT | 4/4 | 75.36 | 42.11 | 49.22 | 64.17 |
| **MorphoQuant (Ours)** | **4/4** | **76.63** | **45.11** | **54.33** | **65.09** |

**关键发现**：

1. **全面超越 W4A4 方法**：在所有模态基准上显著优于现有方法。
2. **超越 W4A16 基线**：在 ScienceQA 上达到 76.63%，**超过** QLoRA W4A16 (75.88%)；在 MMMU 上达到 45.11%，甚至**超过** FP16 Oracle (44.78%)，作者将此归因于量化过程对视觉特征噪声的过滤和语义对齐的保护作用。
3. **视频理解鲁棒性**：在 Video-MME 上达到 54.33%，显著缩小了与 W4A16 基线的差距。
4. **音频模态**：在 AIR-Bench 上达到 65.09%，优于 QLoRA W4A16 (64.12%)，证明了 DABC 机制在处理音频高频离群点方面的有效性。

### Video-MME 细分类别结果

| 方法 | Short | Medium | Long | Overall |
|------|-------|--------|------|---------|
| FP16 (Oracle) | 70.79 | 55.83 | 44.64 | 60.52 |
| QLoRA (W4A16) | 67.89 | 52.92 | 49.46 | 58.21 |
| MorphoQuant (W4A4) | **68.56** | 50.90 | 45.23 | **57.61** |

在短视频类别中，MorphoQuant W4A4 甚至超过了 QLoRA W4A16。

### 消融实验

| 变体 | DABC | Collab. Search | Comp. Loss | ScienceQA | MMMU |
|------|------|----------------|------------|-----------|------|
| Baseline (Q-VLM) | - | - | - | 73.85 | 40.33 |
| + DABC | ✓ | ✓ | - | 75.81 (+1.96) | 44.22 (+3.89) |
| Full (Ours) | ✓ | ✓ | ✓ | **76.63** (+2.78) | **45.11** (+4.78) |
| *Ref: QLoRA (W4A16)* | - | - | - | *75.88* | *43.56* |

- **DABC + 协同搜索**：相比 baseline 提升 1.96% (ScienceQA)，证明隔离长尾离群点和收紧内点网格是恢复模型能力的主要驱动力。
- **复合损失函数**：进一步带来约 0.8% 的提升，验证了余弦相似度损失在保护跨模态语义方向对齐方面的重要性。

### 超参数分析

- **$\lambda_{cos}$**：在 4-bit 下模型对余弦损失权重敏感，最优值为 $\lambda_{cos}=0.75$；在 8-bit 下精度变化有限。
- **扩展率 $\gamma$**：控制补偿的激活范围。$\gamma=1.5$ 为最佳平衡点，能在覆盖长尾离群点的同时保持较高精度。

### PCA 可视化

通过 PCA 对激活特征空间进行可视化：
- FP16 Oracle 的全模态激活呈现**密集中心簇 + 长尾离群点**的独特拓扑结构。
- Q-VLM (W4A4) 导致**表征坍塌**，关键的长尾结构完全消失，整个特征空间被压缩为均质簇。
- **MorphoQuant (W4A4)** 完美重建了原始 FP16 的特征拓扑，密集内点和长尾离群点均得到忠实保留。

---

## 4. 贡献总结

1. **首次系统研究**了全模态大语言模型的 W4A4 量化瓶颈，填补了当前多模态压缩研究的关键空白。
2. 提出了 **DABC 机制**，通过将离群点截断残差吸收到通道级偏置中，以硬件友好的方式解决了灾难性离群点剪裁误差。
3. 提出了 **MDQFO 策略**，通过形态导向的复合损失函数协同优化离散化映射和偏置补偿，最大化紧凑内点分布的语义保真度。
4. 在 Qwen2.5-Omni 上的广泛评估表明，W4A4 MorphoQuant **超越**了更高精度的 W4A16 基线，展现了模态感知量化优化的强大潜力。

---

## 5. 局限性与未来工作

- 极端的 4-bit 量化不可避免地导致信息损失，在推理密集型任务上与 FP16 仍存在差距。
- 全模态架构中模态特定离群点的根本来源和最优管理方式仍是一个开放问题。
- 未来工作将继续探索更精细、更硬件友好的离群点处理方法，进一步缩小与全精度全模态智能的差距。

---

## 6. 深入讨论（Q&A）

### Q1: DABC 和 MDQFO 都是基于 Activation 来做的吗？Parameters 的优化怎么处理？

**是的，DABC 和 MDQFO 都只作用于 Activation 侧**。权重的量化通过现有方法处理。

| 组件 | 作用对象 | 具体策略 |
|------|----------|----------|
| **DABC** | Activation | 离群值截断残差 → 偏置补偿 |
| **MDQFO** | Activation | 形态导向的量化边界 + 复合损失优化 |
| **Weight 量化** | Weight | QLoRA (NF4 + Double Quantization) |

论文中明确说明：*"To ensure a strictly fair comparison and isolate the performance gains yielded by our activation compression, all W4A4 evaluated methods employ the identical 4-bit weight compression technique, adopted from QLoRA, for the MLLM backbone."*

所有对比方法（Naive W4A4、Q-VLM、PoMQ-ViT、MorphoQuant）统一使用 QLoRA 的 NF4 (NormalFloat4) 格式存储 + 双重量化（Double Quantization）方案处理权重，以隔离出 Activation 压缩的增益。MorphoQuant 的核心创新完全在于 Activation 量化，DABC 和 MDQFO 可以与任意 Weight 量化方案组合使用，具有良好的拓展性。

---

### Q2: MDQFO 中 $\alpha_c$ 的梯度如何获取？$\mathbf{X}_Q(\alpha_c)$ 的具体公式是什么？

#### $\mathbf{X}_Q(\alpha_c)$ 完整公式

对于 k-bit 对称量化（INT4 有符号，k=4，范围 $[-8, 7]$），$\alpha_c$ 与缩放因子 $s_x$ 的关系为：

$$s_x = \frac{\alpha_c}{2^{k-1} - 1} = \frac{\alpha_c}{7}$$

量化过程：

$$\mathbf{X}_Q(\alpha_c) = \text{clip}\left(\text{round}\left(\frac{\mathbf{X}}{s_x}\right),\; -(2^{k-1}-1),\; 2^{k-1}-1 \right)$$

反量化后与 DABC 补偿合并：

$$\hat{\mathbf{X}}_Q(\alpha_c) = s_x \cdot \mathbf{X}_Q(\alpha_c) + (\mathbf{X} - s_x \cdot \mathbf{X}_Q(\alpha_c)) \odot \mathbf{m}$$

#### $\frac{\partial \mathcal{L}}{\partial \alpha_c}$ 的完整梯度定义

前向依赖链：

$$\alpha_c \;\to\; s_x = \frac{\alpha_c}{7} \;\to\; \mathbf{X}_Q = \text{clip}\left(\text{round}\left(\frac{\mathbf{X}}{s_x}\right), -7, 7\right) \;\to\; \hat{\mathbf{X}}_Q$$

**Step 1 - 损失对 $\hat{\mathbf{X}}_Q$ 的梯度**：由 $\mathcal{L}_{total} = \mathcal{L}_p + \lambda_{cos}\mathcal{L}_{cos}$ 标准反向传播可得。

**Step 2 - $\hat{\mathbf{X}}_Q$ 对 $s_x$ 的梯度**（按 mask 分情况）：

$$
\frac{\partial \hat{x}_i}{\partial s_x} =
\begin{cases}
x_{q,i} - \dfrac{x_i}{s_x}, & m_i = 0 \quad (\text{inlier, 应用 STE}) \\[10pt]
0, & m_i = 1 \quad (\text{outlier, 全精度直通})
\end{cases}
$$

- **Outlier 通道** ($m_i=1$)：$\hat{x}_i = x_i$（全精度直通），与 $s_x$ 和 $\alpha_c$ 无关，梯度为 0。
- **Inlier 通道** ($m_i=0$)：$\frac{\partial \hat{x}_i}{\partial s_x} = x_{q,i} - \frac{x_i}{s_x}$，这是 LSQ 中的经典梯度形式——quantized value 与 scaled full-precision value 的残差驱动 $s_x$ 更新。通过 STE 绕过 round 和 clip 操作（范围内元素梯度为 1，范围外为 0）。

**Step 3 - $s_x$ 对 $\alpha_c$ 的梯度**：

$$s_x = \frac{\alpha_c}{7} \quad\Rightarrow\quad \frac{\partial s_x}{\partial \alpha_c} = \frac{1}{7}$$

**最终汇总**：

$$\boxed{\frac{\partial \mathcal{L}}{\partial \alpha_c} = \sum_{i\;:\;m_i=0,\;|x_i| \leq 7s_x} \frac{\partial \mathcal{L}}{\partial \hat{x}_i} \cdot \left( x_{q,i} - \frac{x_i}{s_x} \right) \cdot \frac{1}{7}}$$

#### 直观解释

| 元素类型 | 是否贡献梯度 | 原因 |
|---------|------------|------|
| **离群值通道** ($m=1$) | ❌ 不贡献 | $\hat{x}=x$（全精度直通），与 $\alpha_c$ 无关 |
| **Inlier 通道中被 clip 的** | ❌ 不贡献 | $\partial \text{clip} / \partial s_x = 0$ (STE) |
| **Inlier 通道中范围内的** | ✅ 贡献 | 按 LSQ 梯度公式计算 |

这体现了 MorphoQuant 的设计哲学：**$\alpha_c$ 仅由密集 inlier 的量化误差驱动优化**，离群点通过 DABC 从梯度流中解耦出去，互不干扰——这正是 "morphology-directed" 的核心含义。
