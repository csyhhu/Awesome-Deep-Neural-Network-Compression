# PermLLM: 面向 N:M 稀疏 LLM 的可学习通道置换

> **论文**: PermLLM: Learnable Channel Permutation for N:M Sparse Large Language Models  
> **arXiv**: https://arxiv.org/abs/2510.10136  
> **作者**: Lancheng Zou, Shuo Yin, Zehua Pei, Tsung-Yi Ho, Farzan Farnia, Bei Yu  
> **机构**: 香港中文大学 (CUHK)  
> **会议**: NeurIPS 2025  
> **代码**: https://github.com/lanchengzou/PermLLM

---

## 🧠 综合理解（一句话把握全文）

本文在 N:M 稀疏架构下，提出**学习一个置换矩阵**来生成对 N:M 稀疏友好的参数布局。该置换矩阵与原权重矩阵的 contraction dimension（$C_{in}$）对应，**只能统一调整 contraction dimension 上元素的分布**（即对所有输出通道施加相同的列重排）。

具体流程：

1. 设置一个**可学习的软置换** $\mathbf{W}_P$；
2. 经 **Sinkhorn** 归一化使其行和、列和均为 1（近似双随机矩阵 $\widehat{\mathbf{P}}$）；
3. 经 **Hungarian** 算法挑选"每行每列恰一个、且总和最大化"的元素，硬化为**硬置换矩阵** $\mathbf{P}$；
4. 对 $\widehat{\mathbf{W}} = \mathbf{W}\mathbf{P}$ 做 **N:M 剪枝**得到 $\widehat{\mathbf{W}}'$；
5. 用**稠密输出 vs 稀疏输出的余弦距离**计算 loss，反向（经 STE）更新软置换 $\mathbf{W}_P$。

### 💡 点评

这段理解**抓住了全文主线**，pipeline 描述准确。补充几点精确化与延伸：

1. **维度澄清**：可学习参数 $\mathbf{W}_P \in \mathbb{R}^{C_{in} \times C_{in}}$ 与**置换矩阵 $\mathbf{P}$ 同形**，并非与权重矩阵 $\mathbf{W} \in \mathbb{R}^{C_{out} \times C_{in}}$ 同形。$\mathbf{W}_P$ 的尺寸由 $C_{in}$ 决定（分块后是 $N_B$ 个 $B \times B$）；权重 $\mathbf{W}$ 仅在前向作为"被置换对象"出现，且全程**冻结**。所以"和原参数矩阵维度一样"应理解为"和 contraction dimension 的置换算子同形"。

2. **"统一调整"的深层含义**：正因为所有 $C_{out}$ 行共享一个 $\mathbf{P}$，才存在**行间张力**——某行受益可能以另一行受损为代价。PermLLM 用端到端输出损失作为仲裁者，让梯度自动平衡各行需求，这正是它超越 RIA"最大化代理指标之和"这类手工启发式的核心。换句话说，**学习的不只是"哪个通道去哪个位置"，更是"在多行冲突下如何折中"**。

3. **Sinkhorn 的角色不止"行列和为 1"**：更重要的是它把 $\mathbf{W}_P$ 投影到 **Birkhoff 多面体**（置换矩阵的凸包）内，从而保证 $\widehat{\mathbf{P}}$ 是合法置换的连续松弛——这是"可微 + 可硬化"的几何基础。温度 $\tau$ 退火则控制软硬程度，从探索走向收敛。

4. **Hungarian "全局最大化"的精确含义**：它最大化的是 $\text{Tr}(\mathbf{P}^\top \widehat{\mathbf{P}})$，即在**内积度量下**找 Birkhoff 多面体最近的顶点。这是对软偏好最忠实的硬化，但要注意——它优化的是"贴合软置换"，不是直接优化剪枝误差；真正的剪枝目标优化发生在反向传播的 loss 上，Hungarian 仅是前向求值的桥梁。

5. **方法定位**：PermLLM 的哲学是 **"不精确求解组合问题，而近似优化真实目标"**——用 Sinkhorn 松弛 + SGD 绕开 NP-hard 的精确置换搜索，用真实输出损失取代手工代理指标。这是 ML 解决组合优化的典型范式，也呼应了压缩领域反复出现的主题：**代理指标与真实误差之间存在不可忽视的错位**。

---

## 📌 核心摘要

本文提出 **PermLLM**，一个针对 N:M 半结构化稀疏 LLM 的后训练剪枝框架，**首次** 将通道置换 (channel permutation) 从手工启发式指标驱动升级为**端到端可学习**的过程。核心贡献是 **可学习通道置换 (Learnable Channel Permutation, LCP)**：

- 利用 **Sinkhorn 归一化** 将离散置换矩阵松弛为可微的软置换矩阵（双随机矩阵）；
- 引入 **分块通道置换 (block-wise LCP)** 策略，把可学习参数从 $C_{in}^2$ 降至 $C_{in} \cdot B$，并大幅降低 Hungarian 算法的求解复杂度；
- 直接最小化稠密模型与稀疏模型输出的**余弦相似度损失**，而非依赖手工质量代理指标；
- 与 Wanda / RIA 等一次性剪枝方法无缝兼容，并开发了定制 CUDA kernel 实现 84× 加速。

在 LLaMA、LLaMA-2、LLaMA-3.1、Qwen-2.5、OPT 等模型上的 2:4 稀疏实验表明，PermLLM 显著超越现有通道置换方法，在 LLaMA-3.1 8B、Qwen-2.5 7B 等新模型上甚至超过需要更新权重的 SparseGPT。

---

## 🎯 研究动机与问题

### 背景：N:M 稀疏与通道置换

- **N:M 稀疏**（如 2:4）受 NVIDIA Ampere Sparse Tensor Core 原生支持，理论可获得 2× 计算吞吐提升，是半结构化剪枝的实用范式。
- **通道置换** 通过重排权重矩阵的输入通道顺序，使重要权重落入更易保留的位置，从而提升 N:M 稀疏后的精度。
- 主流方法 RIA 采用两阶段启发式：先按重要性将通道分配到不同 block，再通过线性分配问题 (linear sum assignment) 最大化保留权重重要性分数之和。

### 现有方法的核心缺陷

**手工质量指标与真实剪枝误差之间存在错位**。如图 1 (motivation) 所示：

- 直接 2:4 剪枝的输出 loss = 12.375；
- 采用最大化保留权重重要性分数 $S$ 的通道置换后，loss 反而升至 **13.662**；
- 而另一个分数 $S$ 较低的置换方案，loss 却降到 **8.716**。

这说明**最大化手工指标 ≠ 最小化剪枝误差**。此外，启发式方法无法捕捉复杂的层间交互，错失了补偿剪枝误差的机会。

### 学习置换矩阵的两大挑战

| 挑战 | 描述 |
|------|------|
| **离散性与组合约束** | 置换矩阵 $\mathbf{P}$ 为二值矩阵，每行每列恰有一个 1，非可微，难以梯度优化 |
| **解空间爆炸** | $C_{in}$ 个通道有 $C_{in}!$ 种置换；N:M 约束下虽降为 $\frac{C_{in}!}{(M!)^G \cdot G!}$，但 $C_{in}=16, M=4$ 时仍有 260 万候选，LLM 中 $C_{in}$ 常超过一千 |

---

## 💡 方法：PermLLM 框架

### 1. 软置换矩阵松弛

将硬置换矩阵 $\mathbf{P}$ 松弛为**双随机矩阵 (doubly stochastic matrix)** $\widehat{\mathbf{P}}$（每行每列和为 1，非负），通过 **Sinkhorn 归一化** 迭代实现：

$$
S^0(\mathbf{X}) = \exp(\mathbf{X}), \quad S^i(\mathbf{X}) = \mathcal{T}_c(\mathcal{T}_r(S^{i-1}(\mathbf{X})))
$$

其中 $\mathcal{T}_r, \mathcal{T}_c$ 分别为行、列归一化。软置换矩阵：

$$
\widehat{\mathbf{P}} = S^L(\mathbf{W}_P / \tau)
$$

- $\mathbf{W}_P$：与 $\widehat{\mathbf{P}}$ 同形状的可学习矩阵；
- $\tau$：温度系数，训练中从 1 线性衰减到 0.1，控制软度（趋近 0 时收敛到硬置换）；
- $L$：Sinkhorn 迭代次数，默认取 5。

**前向硬化 + 反向直通**：前向用 Hungarian 算法将 $\widehat{\mathbf{P}}$ 硬化为最近邻的严格置换矩阵 $\mathbf{P}$：

$$
\mathbf{P} = \arg\max_{\mathbf{P} \in \mathcal{P}} \text{Tr}(\mathbf{P}^\top \widehat{\mathbf{P}})
$$

反向采用 **Straight-Through Estimator (STE)**，令 $\partial \mathbf{P} / \partial \widehat{\mathbf{P}} = 1$，保持梯度流通。

### 2. 分块可学习通道置换 (Block-wise LCP)

**动机**：全矩阵 LCP 的可学习参数量为 $C_{in}^2$（与权重矩阵同形），训练负担过重。

**方案**：将 $C_{in}$ 个通道划分为 $N_B = C_{in}/B$ 个大小为 $B$ 的 block，置换仅在 block 内进行：

- **参数量**：从 $C_{in}^2$ 降至 $N_B \times B^2 = C_{in} \cdot B$，压缩比为 $B / C_{in}$；
- **硬化复杂度**：Hungarian 算法从 $O(C_{in}^3)$ 降至 $O(N_B \cdot B^3) = O(C_{in} \cdot B^2)$；
- **结构**：$\widehat{\mathbf{W}}_B = \mathbf{W}\mathbf{P}_B$，其中 $\mathbf{P}_B = \text{diag}(\mathbf{P}_1, \dots, \mathbf{N_B})$ 为块对角矩阵。

**默认 $B=64$**：兼顾性能与效率，$B=128$ 会使运行时间翻倍且收敛更慢。

### 3. 与一次性剪枝整合 + 直接损失优化

**剪枝掩码生成**：基于 Wanda/RIA 的重要性指标 $\mathbf{S}_{ij} = |\mathbf{W}_{ij}| \cdot \|\mathbf{X}_j\|_2$，置换后重要性矩阵变为 $\widehat{\mathbf{S}} = \mathbf{S}\mathbf{P}_B$，掩码通过 argmax 选取每组 $M$ 中最大的 $M-N$ 个。

**反向梯度**：argmax 不可微，前向用硬掩码 $\mathbf{M}$，反向用 softmax 软掩码 $\widehat{\mathbf{M}} = \text{Softmax}(\widehat{\mathbf{S}})$ 近似梯度。

**训练目标**：直接最小化稠密模型输出 $\mathbf{y}$ 与稀疏模型输出 $\widetilde{\mathbf{y}}$ 的余弦相似度损失：

$$
\mathcal{L}_{cosine}(\mathbf{y}, \widetilde{\mathbf{y}}) = 1 - \frac{\mathbf{y} \cdot \widetilde{\mathbf{y}}}{\|\mathbf{y}\| \cdot \|\widetilde{\mathbf{y}}\|}
$$

- 训练中**仅 $\mathbf{W}_P^i$ 可学习，权重 $\mathbf{W}$ 固定**；
- 掩码 $\mathbf{M}$ 随 $\mathbf{P}_B$ 动态更新；
- 训练后：$\widehat{\mathbf{W}}' = \mathbf{M}^* \odot (\mathbf{W}\mathbf{P}_B^*)$。

### 4. 输入激活对齐与 CUDA 加速

- **激活置换**：当前层输入通道被置换后，需对前一层的输出通道（即前一层权重行）做相同置换：$\widehat{\mathbf{W}}''_{l-1} = \mathbf{P}_{l,B}^* \widehat{\mathbf{W}}'_{l-1}$。这是行操作，保持前层的 N:M 稀疏性。
- **共享输入的层**（如 Q/K/V、Up/Gate）须采用相同置换。
- **定制 CUDA kernel**：通道置换从 3.288ms 降至 0.039ms，**84× 加速**，整体线性层加速约 1.67×。

---

## 🔬 实验设置

- **模型**：LLaMA 7B/13B、LLaMA-2 7B/13B、LLaMA-3.1 8B、Qwen-2.5 7B、OPT 6.7B；
- **校准数据**：C4 数据集随机采样 128 样本，每样本 1024 tokens；
- **评估**：Wikitext2 (PPL) + 5 个 zero-shot 任务 (HellaSwag, ARC-E, ARC-C, OpenBookQA, RTE)；
- **剪枝范围**：所有线性层（约占 99% 参数），跳过 embedding 和 classification head；
- **超参**：AdamW 优化器，lr ∈ {1e-3 (Wanda), 5e-3 (RIA)}，Sinkhorn 迭代 5 次，温度 1→0.1 线性衰减，block size 64，50 次迭代；
- **硬件**：7B 模型 4×A100 约 2.5h，13B 模型 8×A100 约 5.5h。

---

## 📊 主要实验结果

### 1. 语言建模 (Wikitext2 PPL，越低越好)

| Method | OPT 6.7B | LLaMA 7B | LLaMA 13B | LLaMA-2 7B | LLaMA-2 13B | LLaMA-3.1 8B | Qwen-2.5 7B |
|--------|----------|----------|-----------|------------|-------------|--------------|-------------|
| Dense | 10.86 | 5.68 | 5.09 | 5.47 | 4.89 | 6.24 | 7.74 |
| SparseGPT | 14.33 | 11.19 | 9.17 | 11.12 | 9.03 | 16.62 | 14.34 |
| Wanda | 16.29 | 11.59 | 9.60 | 12.16 | 9.05 | 23.42 | 24.44 |
| Wanda+CP | 15.28 | 11.07 | 8.69 | 11.00 | 8.51 | 21.09 | 18.76 |
| **PermLLM_Wanda** | 14.27 | **9.41** | 8.06 | **9.39** | 8.20 | **14.03** | **13.58** |
| RIA | 15.93 | 11.14 | 8.96 | 11.30 | 8.51 | 22.62 | 22.67 |
| RIA+CP | 15.13 | 10.99 | 8.15 | 10.26 | 8.08 | 19.80 | 17.58 |
| **PermLLM_RIA** | **14.23** | 9.95 | **7.81** | 9.60 | **7.97** | 15.79 | 15.93 |

**关键发现**：
- PermLLM 在绝大多数模型上取得最优 PPL；
- 在 LLaMA-3.1 8B 和 Qwen-2.5 7B 上，Wanda/RIA+CP 严重退化（PPL 20+），而 PermLLM 将 PPL 压到 13-15，**甚至超过需要更新权重的 SparseGPT**；
- 在 LLaMA-2 7B 上，PermLLM_Wanda (9.39) 相比 Wanda (12.16) 实现 **29% PPL 下降**。

### 2. Zero-shot 平均准确率 (越高越好)

PermLLM 在 OPT 6.7B、LLaMA 7B、LLaMA-2 7B、LLaMA-3.1 8B、Qwen-2.5 7B 上均取得**最高平均准确率**，验证了学习式置换在不同任务上的泛化性。

### 3. 推理加速 (LLaMA-2 7B, 2048 tokens)

| 组件 | Dense | 2:4 + CP | 加速比 |
|------|-------|----------|--------|
| Q/K/V/O_proj | 1.513ms | 0.927ms | 1.632× |
| Up/Gate_proj | 2.607ms | 1.526ms | 1.708× |
| Down_proj | 2.614ms | 1.535ms | 1.703× |
| CP 本身 | - | 0.039ms | (CUDA kernel 84× 加速) |

定制 CUDA kernel 使通道置换开销极小，整体线性层加速约 **1.67×**。

### 4. 4:8 稀疏 (LLaMA-2 7B)

PermLLM_Wanda 平均准确率 47.97%，PPL 7.96，均优于 SparseGPT 和 Wanda+CP，证明方法**不局限于 2:4**。

---

## 🔍 消融实验

### 1. Sinkhorn 迭代次数

迭代 0 次（不收敛到双随机矩阵）vs 5 次：
- LLaMA-3.1 8B：平均准确率 49.18% → **52.17%**，PPL 14.43 → **13.58**；
- Qwen-2.5 7B：平均准确率 42.96% → **43.33%**，PPL 14.12 → **14.03**。

证明双随机矩阵结构对置换学习的有效性。

### 2. 校准数据集鲁棒性 (LLaMA-2 7B)

Pile / Wikitext2 / C4 三种数据集各 128 样本，平均准确率分别为 44.74 / 44.61 / 46.59，PPL 分别为 8.96 / 8.31 / 9.39。学习到的置换在不同数据集上表现稳定。

### 3. Block size (LLaMA-2 7B)

| Block size | 平均 Acc | Wikitext2 PPL | 时间 |
|------------|----------|---------------|------|
| 32 | 43.58 | 9.50 | 2h |
| **64** | 46.59 | 9.39 | 2.5h |
| 128 | 47.09 | 9.07 | 6h |

更大 block 提供更大优化空间但收敛更慢，64 是性能与效率的最佳平衡点。

### 4. 部分 PermLLM (仅最后 6 层用 LCP)

LLaMA-2 7B 上：仅对最后 6 层 decoder 应用 LCP，其余用传统 CP。单 GPU 0.4h（与传统 CP 相当），平均 Acc 43.78% > RIA+CP 43.42%，PPL 10.10 < 10.26。是计算资源受限场景下的实用折中方案。

---

## 🎨 掩码可视化

对 LLaMA-2 7B 第 30 层 down_proj 的 128×128 掩码（置换回原序）对比显示：
- Wanda / RIA + CP 旨在**最大化保留权重重要性指标之和**；
- PermLLM 旨在**最小化稠密-稀疏输出差异**；
- 二者保留的权重分布明显不同，PermLLM 的保留模式更贴合真实输出保真目标。

---

## ⚠️ 局限性

1. **领域扩展**：方法专为半结构化剪枝设计，但通道置换在量化（如 DuQuant、RPTQ）中也被证明有效，向量化等任务扩展是开放方向。
2. **训练开销**：虽然分块策略显著降低了开销，但 PermLLM 仍比传统通道置换方法需要更多计算资源，提升训练效率是未来工作方向。

---

## 📝 个人理解与启示

### 核心创新点
PermLLM 的最大贡献在于**将"置换选择"从启发式指标优化升级为端到端梯度优化**。这呼应了压缩领域一个反复出现的主题：**手工代理指标与真实损失之间存在不可忽视的错位**——无论多精心设计的指标（Wanda 的 $|W| \cdot \|X\|$、RIA 的相对重要性），都难以完全捕捉剪枝对最终输出的影响。

### 技术亮点
1. **Sinkhorn + STE 组合**：用双随机矩阵松弛离散置换，前向 Hungarian 硬化、反向 STE 传梯度，是处理组合优化不可微问题的经典范式；
2. **分块策略**：同时降低参数量（$C_{in}^2 → C_{in}B$）和 Hungarian 复杂度（$O(C_{in}^3) → O(C_{in}B^2)$），是实用化的关键；
3. **与现有方法解耦**：作为插件兼容 Wanda/RIA，不修改权重、不改掩码生成逻辑，工程友好。

### 与相关工作的对比
- **vs SparseGPT**：SparseGPT 通过更新未剪枝权重补偿误差（重量级），PermLLM 仅学置换不改权重（轻量级），却在新模型上反超 SparseGPT；
- **vs MaskLLM**：MaskLLM 学习掩码本身，需大量数据；PermLLM 学习置换、掩码仍由一次性指标决定，数据需求低（128 样本即可）；
- **vs ELSA**：ELSA 是全局非结构化剪枝（ADMM），PermLLM 是半结构化 N:M 剪枝，两者面向不同稀疏范式。

### 实践意义
对于需要硬件加速的 LLM 部署（N:M 稀疏 + NVIDIA GPU），PermLLM 提供了一个**不更新权重、不改训练流程、即插即用**的精度提升手段，尤其在 LLaMA-3.1 / Qwen-2.5 等新模型上效果显著。

---

## 深度问答 (Q&A)

### Q1: N:M 稀疏具体如何进行？是把 $C_{in} \times C_{out}$ 矩阵展平成一维再切块挑选吗？

**A: 不是展平成一维。N:M 稀疏保留二维结构，沿 $C_{in}$ 维度（列方向）逐行分组。**

具体流程（以 2:4 为例，权重 $\mathbf{W} \in \mathbb{R}^{C_{out} \times C_{in}}$）：

```
对每一行 i (每个输出通道, 共 C_out 行):
    将该行的 C_in 个元素按顺序切分为 C_in/M 个 group, 每个 group M=4 个连续元素
    对每个 group:
        计算重要性分数 S = |W| * ||X||  (Wanda) 或 RIA 指标
        保留分数最大的 M-N = 2 个 (mask 置 1)
        剪掉其余 N = 2 个 (mask 置 0)
```

论文公式 (8) 的约束 $\|\mathbf{M}_{i, kM:(k+1)M}\|_0 = M-N$ 正是这个含义：每个长度为 $M$ 的 group 里非零元素个数为 $M-N$。

**关键点**：
- 分组**逐行独立**，每行的 group 划分相同（按 $C_{in}$ 顺序每 $M$ 个一组），但每行保留哪些元素独立决定；
- "连续 $M$ 个"指 $C_{in}$ 维度上位置相邻，所以**重排 $C_{in}$ 通道顺序 = 改变哪些权重落入同一 group**，这正是通道置换起作用的根本原因；
- NVIDIA Ampere 的 Sparse Tensor Core 硬件要求就是这个 2:4 pattern（每 4 个连续元素中恰好 2 个非零），所以分组必须按 $C_{in}$ 连续切分，不能任意打乱。

---

### Q2: Permutation 只能从 contraction dimension ($C_{in}$) 进行吗？$C_{in}^2$ 参数是否意味着产生两个置换矩阵？如何把变换传递到上一层保证输出不变？

**A: 只对 $C_{in}$（列）做置换，只产生一个置换矩阵 $\mathbf{P}$。$C_{in}^2$ 是"生成置换的可学习参数矩阵 $\mathbf{W}_P$ 的尺寸"，不是要产生两个置换矩阵。**

#### 如何保证输出不变？（变换传递）

线性层 $\mathbf{y}_l = \mathbf{W}_l \mathbf{x}_l$。若对 $\mathbf{W}_l$ 的列做置换 $\mathbf{P}$：$\widehat{\mathbf{W}}_l = \mathbf{W}_l \mathbf{P}$，那么必须同时把输入 $\mathbf{x}_l$ 也置换：$\widehat{\mathbf{x}}_l = \mathbf{P}^\top \mathbf{x}_l$，才能保持 $\mathbf{y}_l = \widehat{\mathbf{W}}_l \widehat{\mathbf{x}}_l = \mathbf{W}_l \mathbf{P} \mathbf{P}^\top \mathbf{x}_l = \mathbf{W}_l \mathbf{x}_l$。

而 $\mathbf{x}_l = \mathbf{y}_{l-1} = \mathbf{W}_{l-1} \mathbf{x}_{l-1}$，要让 $\widehat{\mathbf{x}}_l = \mathbf{P}^\top \mathbf{x}_l$，等价于把上一层的权重改为 $\mathbf{P}^\top \mathbf{W}_{l-1}$，即**对 $\mathbf{W}_{l-1}$ 的行（$C_{out}$）做相同置换**。对应论文公式 (13)：

$$\widehat{\mathbf{W}}''_{l-1} = \mathbf{P}_{l,B}^* \widehat{\mathbf{W}}'_{l-1}$$

（$\mathbf{P}$ 左乘 = 行置换；由于 $\mathbf{P}$ 是置换矩阵，$\mathbf{P}^\top = \mathbf{P}^{-1}$，符号约定一致即可。）

**为什么行置换保持上一层的 N:M 稀疏性？** N:M 分组是每行内部独立划分的，整行调换顺序只是把"第 3 行和第 5 行"对调，每行内部的 2:4 pattern 完全不变，所以传递操作"无损"。

**特例**：Q/K/V 共享同一输入 $\mathbf{x}$，所以它们的列置换必须相同；同理 Up/Gate。这就是论文说"共享输入的层必须采用相同置换"的原因。

---

### Q3: Sinkhorn 归一化有什么特点？每次迭代能保证行列之和为 1 吗？

**A: 不能。单次迭代后只有"后做的那一步归一化"被保证，另一个不保证。收敛到双随机矩阵是渐近的。**

#### Sinkhorn 流程

```
输入: 任意方阵 X
Step 0:  S^0 = exp(X)                     # 变非负
Step i:  S^i = T_c(T_r(S^{i-1}))          # 先行归一化, 再列归一化
         T_r: 每行除以该行和 → 每行和=1
         T_c: 每列除以该列和 → 每列和=1
```

#### 单次迭代后的状态

假设做了一次 $T_r$ 再 $T_c$：
- 做完 $T_r$：所有行和 = 1，但列和不一定；
- 再做 $T_c$：所有列和 = 1，但**行和又被破坏了**（不再是 1）。

所以一次迭代后**只有列和为 1，行和不为 1**。继续迭代会反复在"行和=1"与"列和=1"之间逼近，行和列同时为 1 的双随机矩阵是**不动点**，当且仅当迭代收敛时达到。

#### Sinkhorn 定理

对任意正方阵，迭代必然收敛到**唯一**的双随机矩阵（Sinkhorn 定理）。收敛速度是线性的（几何衰减）。论文用 $L=5$ 次，实测足够接近双随机——消融实验中 iter=0 vs iter=5 差距明显（LLaMA-3.1 8B 平均 Acc 49.18% → 52.17%）。

#### 为什么适合这里？

1. **可微**：行/列归一化都是 element-wise 除法，梯度可流畅传播；
2. **保结构**：收敛到双随机矩阵，恰是置换矩阵的凸组合（Birkhoff-von Neumann 定理），是置换矩阵集合（Birkhoff 多面体）的连续松弛；
3. **温度 $\tau$ 控制硬度**：$\tau \to 0$ 时双随机矩阵趋近于硬置换矩阵（越靠近 Birkhoff 多面体的顶点），训练中 $\tau$ 从 1 线性衰减到 0.1 实现"由软到硬"的退火。

---

### Q4: Hungarian 算法是怎么做的？为什么可以硬化 surrogate permutation？对双随机矩阵有其他硬化方法吗？

**A: Hungarian 求解线性分配问题，给出 Birkhoff 多面体上内积意义下的最优顶点投影，是天然的硬化方式。**

#### Hungarian 在做什么？

它求解**线性分配问题 (Linear Assignment Problem)**：给定 $n \times n$ 收益矩阵 $\widehat{\mathbf{P}}$，选 $n$ 个元素，每行每列恰选一个，使总收益最大。即：

$$\max_{\mathbf{P} \in \mathcal{P}} \sum_{i,j} P_{ij} \widehat{P}_{ij} = \max_{\mathbf{P} \in \mathcal{P}} \text{Tr}(\mathbf{P}^\top \widehat{\mathbf{P}})$$

这正是论文公式 (6)。算法复杂度 $O(n^3)$，是精确算法（非近似）。

#### 为什么这是"硬化"？

**Birkhoff-von Neumann 定理**：双随机矩阵集合（Birkhoff 多面体）的顶点恰好是所有置换矩阵。任何双随机矩阵 $\widehat{\mathbf{P}}$ 都可写成置换矩阵的凸组合：

$$\widehat{\mathbf{P}} = \sum_k \theta_k \mathbf{P}_k, \quad \theta_k \geq 0, \sum \theta_k = 1$$

"硬化" = 把 $\widehat{\mathbf{P}}$ 投影到最近的顶点。在内积度量下，最近的顶点就是使 $\text{Tr}(\mathbf{P}^\top \widehat{\mathbf{P}})$ 最大的那个置换矩阵——这正是 Hungarian 求解的目标。所以 **Hungarian 给出了 Birkhoff 多面体上内积意义下的最优顶点投影**。

#### 其他硬化方法对比

| 方法 | 描述 | 缺点 |
|------|------|------|
| **逐行 argmax（贪心）** | 每行选最大元素位置 | 无法保证列唯一性，可能产生冲突（两行选同列），不是合法置换 |
| **Gumbel-Softmax + 匹配** | 加 Gumbel 噪声采样后做匹配 | 引入随机性，训练用更多，硬化仍需匹配 |
| **Auction algorithm** | 另一种分配算法，与 Hungarian 等价 | 本质相同，只是实现不同 |
| **迭代 argmax + 冲突消解** | 启发式处理冲突 | 不保证最优，可能陷入局部解 |
| **Gumbel-Sinkhorn（训练时）** | 训练阶段直接用带噪声的 Sinkhorn 采样 | 是训练技巧，最终硬化仍要 Hungarian |

所以 Hungarian 是**确定性、精确、与 Birkhoff 几何最一致**的硬化方法。PermLLM 用它做前向硬化、STE 做反向传梯度（$\partial \mathbf{P} / \partial \widehat{\mathbf{P}} = 1$），是这类组合优化不可微问题的标准范式（Mena et al. 2018, Gumbel-Sinkhorn 系列工作都是这套）。

---

### Q5: "仅对最后 6 层 decoder 应用 LCP，其余用传统 CP"——传统 CP 是怎么样的？为什么仅对部分层进行 LCP？

**A: 传统 CP 是 RIA 的两阶段启发式（通道分配 + 线性分配细化），无梯度无学习。只对部分层用 LCP 是为了在训练开销与性能间折中。**

#### 传统 CP（RIA 的通道置换）

完全无梯度、无学习的两阶段启发式：

1. **通道分配阶段 (Channel Allocation)**：基于手工重要性指标（RIA 指标），按贪心策略把"重要的"通道分配到不同的 block，确保每个 block 内既有重要也有不重要的通道（避免一个 block 全是重要通道导致剪枝时无处下手）。
2. **细化阶段 (Refinement)**：把通道置换建模为**线性分配问题**，用 Hungarian 算法最大化"保留权重重要性分数之和"。

**核心特征**：
- 优化目标是**手工代理指标**（保留权重重要性之和），不是真实输出误差；
- 无可学习参数，单次求解，速度极快；
- 与论文图 1 现象一致：最大化代理指标 ≠ 最小化剪枝误差，可能反而变差。

#### 为什么只对部分层用 LCP？

三个原因：

1. **训练开销**：LCP 需要梯度反向传播 + Sinkhorn 迭代 + Hungarian 求解，比传统 CP 贵得多。全模型 LCP 在 7B 上需 4×A100、2.5h；只做 6 层则单卡 0.4h 就够，与传统 CP 相当。
2. **层间敏感度差异**：论文引用"不同层对输出影响不同"。通常**靠后的层**对最终 logits / 语言建模损失影响更直接（更接近输出），在这些层用更精细的 LCP 收益更高；前面的层用传统 CP 足够。
3. **效率-性能折中**：partial PermLLM（6 层 LCP）平均 Acc 43.78% > RIA+CP 43.42%，PPL 10.10 < 10.26，虽不及 full PermLLM（44.30% / 9.60），但已在"零额外时间成本"下提升明显，适合算力受限场景。

#### 为什么选"最后 6 层"？

论文未给出严格理论依据，应是经验选择。一般直觉：靠近输出的 decoder 层对 next-token prediction 损失的梯度信号更强、对最终输出更敏感，LCP 直接优化余弦相似度损失时这些层的优化收益更显著。具体层数（6）是精度-开销的权衡点。

---

### Q6: 列置换的行间影响——第一行的置换会不会破坏第二行已优化好的 N:M 分布？

**A: 不会，因为所有行共享同一个置换矩阵 $\mathbf{P}$，不存在"逐行串行置换"。行间冲突通过端到端输出损失自动平衡。**

#### 关键澄清

你的疑问基于"每行有独立置换"的误解。实际上 $\widehat{\mathbf{W}} = \mathbf{W}\mathbf{P}$ 中，$\mathbf{P} \in \mathbb{R}^{C_{in} \times C_{in}}$ 是**一个**列置换矩阵，右乘后对 $\mathbf{W}$ 的**每一行施加完全相同的列重排**。

N:M 分组按列位置定义（每行都是"列 0-3 一组、列 4-7 一组…"），只有统一的列置换才能重新定义所有行的分组结构。若每行用不同置换，硬件无法用统一 2:4 pattern 加速。

#### 行间冲突如何解决

确实存在张力：$\mathbf{P}$ 让某些行受益，可能让另一些行变差。PermLLM 的处理是**不逐行优化，而是端到端优化全局输出损失**：

$$\mathcal{L} = 1 - \cos(\mathbf{y}, \widetilde{\mathbf{y}}), \quad \widetilde{\mathbf{y}} = (\mathbf{M} \odot \mathbf{W}\mathbf{P})\mathbf{x}$$

输出 $\mathbf{y}$ 是所有行的加权和，梯度反向传播时**自动平衡各行需求**——找到的 $\mathbf{P}$ 是所有行的"全局折中"。

#### 与传统 CP 的对比

| 方法 | 如何处理行间冲突 |
|------|----------------|
| **RIA+CP** | 同一 $\mathbf{P}$，优化"所有行保留权重重要性之和"（手工代理指标），无法真实反映各行对输出的贡献 |
| **PermLLM** | 用真实输出余弦损失，梯度自然加权各行（重要行梯度大），找到**对最终输出最优的折中** |

所以"是否处理行间影响"的答案是：**显式地通过端到端输出损失处理**，这正是 PermLLM 相对启发式方法的核心优势之一——它不再需要手工设计"对所有行公平"的代理指标。

---

### Q7: 双随机矩阵是什么？Sinkhorn 多次迭代后得到什么结果？

**A: 双随机矩阵 = 元素非负 + 行和为 1 + 列和为 1。Sinkhorn 多次迭代后渐近收敛到（唯一的）双随机矩阵，有限次迭代得到近似双随机。**

#### 双随机矩阵定义

方阵 $\mathbf{A} \in \mathbb{R}^{n \times n}$ 满足：
- 所有元素 $A_{ij} \geq 0$；
- 每行之和 = 1；
- 每列之和 = 1。

例：$\begin{pmatrix} 0.5 & 0.5 \\ 0.5 & 0.5 \end{pmatrix}$、$\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$（单位阵也是双随机的特例，即置换矩阵）。

#### Birkhoff-von Neumann 定理（关键）

**所有 $n \times n$ 双随机矩阵构成的集合（Birkhoff 多面体）是一个凸多面体，其顶点恰好是所有置换矩阵。** 任何双随机矩阵可写成置换矩阵的凸组合：

$$\widehat{\mathbf{P}} = \sum_k \theta_k \mathbf{P}_k, \quad \theta_k \geq 0, \sum_k \theta_k = 1$$

这就是为什么双随机矩阵是"置换矩阵的连续松弛"——它在置换矩阵集合的凸包内，梯度可流动；硬化 = 投影到最近顶点。

#### Sinkhorn 多次迭代后的结果

| 迭代次数 | 结果 |
|---------|------|
| 0 次（仅 exp） | 非负矩阵，行列和都不为 1 |
| 1 次（行归一化 + 列归一化） | 列和 = 1，行和 ≠ 1（被列归一化破坏） |
| 有限 $L$ 次 | **近似**双随机：行和、列和都接近 1，误差随 $L$ 几何衰减 |
| $L \to \infty$ | **精确**双随机矩阵（唯一，由 Sinkhorn 定理保证） |

论文用 $L=5$，实测足够接近双随机。消融实验：iter=0 vs iter=5 在 LLaMA-3.1 8B 上平均 Acc 从 49.18% 提升到 52.17%，证明"接近双随机"对学习效果至关重要。

---

### Q8: Hungarian 中的收益指什么？具体怎么做？

**A: 收益就是软置换矩阵 $\widehat{\mathbf{P}}$ 的元素本身；目标是选 $n$ 个元素（每行每列恰选一个）使和最大；Hungarian 是 $O(n^3)$ 精确算法。**

#### 收益是什么

收益矩阵就是软置换矩阵 $\widehat{\mathbf{P}}$ 本身。目标：

$$\max_{\mathbf{P} \in \mathcal{P}} \sum_{i,j} P_{ij} \widehat{P}_{ij}$$

即从 $\widehat{\mathbf{P}}$ 中选 $n$ 个元素（每行每列恰选一个），使**选中的元素之和最大**。$\widehat{P}_{ij}$ 就是"把行 $i$ 分配给列 $j$"的收益。

#### 直觉解释

$\widehat{\mathbf{P}}$ 的每一行是"该位置想去哪个列的概率分布"（行和为 1）。Hungarian 找一个合法的一对一匹配，最大化"捕获的总质量"——即最贴合软偏好的硬置换。

#### 贪心 argmax 会出错（反例）

```
row1 [ 0.5  0.4 ]      贪心: row1→col1 (0.5), row2→col2 (0.1)  sum=0.6
row2 [ 0.6  0.1 ]      最优: row1→col2 (0.4), row2→col1 (0.6)  sum=1.0
```
贪心无法回溯，可能次优。所以需要 Hungarian 精确求解。

#### Hungarian 算法步骤

本质是**原始-对偶算法**：
1. 把 MAX 问题转成 MIN（用 $c_{ij} = \max(\widehat{\mathbf{P}}) - \widehat{P}_{ij}$ 作为代价）；
2. 维护对偶变量 $u_i$（行势）和 $v_j$（列势），满足 $u_i + v_j \leq c_{ij}$；
3. 迭代地：找增广路径 → 增加匹配数 → 更新对偶变量保持可行；
4. $O(n^3)$ 时间精确求解，返回最优匹配。


---

### Q9: 本文的 Loss 是什么？梯度如何一路传导回 $\mathbf{W}_P$？

**A: Loss 是稠密输出与稀疏输出的余弦相似度。前向 $\mathbf{W}_P \to \text{Sinkhorn} \to \widehat{\mathbf{P}} \to \text{Hungarian} \to \mathbf{P} \to$ 剪枝 $\to \widetilde{\mathbf{y}}$，反向在 Hungarian 和 argmax 两处用 STE 直通梯度。**

#### 前向流程（Forward）

```
W_P (可学习, B×B)
   │
   ├──Sinkhorn (5次迭代, 温度τ)──→ P̂ (软置换, 双随机)
   │                                   │
   │                              Hungarian (前向硬化)
   │                                   │
   │                                   ▼
   │                                   P (硬置换矩阵)
   │                                   │
   ├──W (固定权重) ──→ Ŵ = W P (列置换)
   │                   │
   │              计算重要性 Ŝ = S P
   │                   │
   │              argmax (前向硬掩码) ──→ M (硬掩码)
   │                   │
   │              Ŵ' = M ⊙ Ŵ (剪枝后权重)
   │                   │
   │              ỹ = Ŵ' x (稀疏模型输出)
   │                   │
   └──→  y = W x (稠密模型输出, 直接取, 无需重新计算)
                       │
                  L = 1 - cos(y, ỹ)   ← 这就是 Loss
```

#### 反向流程（Backward）

损失 $\mathcal{L}$ 对 $\mathbf{W}_P$ 求梯度，路径上有两个不可微操作需要 STE：

| 不可微操作 | 前向 | 反向（STE） |
|-----------|------|-----------|
| Hungarian 硬化 $\widehat{\mathbf{P}} \to \mathbf{P}$ | argmax 选最优置换 | $\partial \mathbf{P} / \partial \widehat{\mathbf{P}} = \mathbf{I}$（直通） |
| 掩码 argmax $\widehat{\mathbf{S}} \to \mathbf{M}$ | 每组取 top-$M-N$ | 用 softmax 软掩码 $\widehat{\mathbf{M}} = \text{Softmax}(\widehat{\mathbf{S}})$ 近似梯度 |

Sinkhorn 本身可微（element-wise 除法），梯度正常传播。

#### 关键点

1. **Loss 是余弦相似度损失**：$\mathcal{L} = 1 - \frac{\mathbf{y} \cdot \widetilde{\mathbf{y}}}{\|\mathbf{y}\| \|\widetilde{\mathbf{y}}\|}$，直接衡量稠密-稀疏输出差异；
2. **唯一可学习参数是 $\mathbf{W}_P$**，权重 $\mathbf{W}$ 全程冻结；
3. **掩码 $\mathbf{M}$ 随 $\mathbf{P}$ 动态变化**（不是预先固定），梯度通过掩码选择间接影响 $\mathbf{W}_P$；
4. **温度退火**：$\tau$ 从 1 线性衰减到 0.1，训练初期软（探索空间大），后期硬（收敛到具体置换）。

---

### Q10: RIA 的 Channel Allocation 是 NP-hard 吗？还是按重要性排序再组合？

**A: RIA 的 Stage 2（细化）是多项式可解的 LSAP，不是 NP-hard；但 Stage 1（通道分配）本质是组合划分问题，精确求解是 NP-hard，RIA 用贪心近似。"按重要性排序再组合"抓住了 Stage 1 的精神但不完全。**

#### 分阶段看复杂度

| 阶段 | 问题类型 | 复杂度 | 是否 NP-hard |
|------|---------|--------|-------------|
| **Stage 1: 通道分配** | 把通道分到 block 的划分问题 | 启发式贪心，近似求解 | 精确求解类似 bin-packing 变体，**是 NP-hard**；RIA 用贪心近似 |
| **Stage 2: 细化** | 线性分配问题 (LSAP) | Hungarian $O(n^3)$ 精确求解 | **不是** NP-hard，多项式可解 |

#### "按重要性排序再组合"的直觉

你的直觉抓住了 Stage 1 的精神，但不完全：

- **简单排序不够**：若只是按重要性排序后 round-robin 分到各 block，能保证 block 内重要性分布均衡，但**没有考虑权重的联合分布**（哪些通道组合在一起对剪枝影响最大）；
- **RIA 实际做法**：先用 RIA 指标量化每个通道的重要性，然后**贪心地把重要通道分散到不同 block**（避免一个 block 全是重要通道导致剪枝时无法满足 2:4 约束），再用 Hungarian 在 block 内做细化。

#### 为什么完整问题是难的

完整最优置换需同时决定：
1. 哪些通道分到同一 block（partition，组合）；
2. block 内通道顺序（permutation，组合）；
3. 跨所有 $C_{out}$ 行联合优化（共享 $\mathbf{P}$）。

解空间 $\frac{C_{in}!}{(M!)^G \cdot G!}$ 在 $C_{in}=4096, M=4$ 时是天文数字。即便用代理指标，**联合优化分组 + 排列 + 多行共享** 也不是单一 LSAP 能精确求解的。RIA 通过"先贪心分组、再 Hungarian 细化"的分解绕开难题，但分解本身不保证全局最优。

#### PermLLM 的视角

PermLLM **绕开了"是否 NP-hard"的问题**：
1. 用 Sinkhorn 松弛成连续优化；
2. 用 SGD/Adam 在软置换空间搜索；
3. 用真实损失（而非代理指标）作为目标。

代价是不保证全局最优（可能陷入局部解），但**优化方向直接对齐真实目标**，实验证明比启发式代理指标更有效。这其实是 ML 解决组合优化的常见范式：**与其精确求解代理问题，不如近似优化真实目标**。
