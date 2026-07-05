# LIVEditor-14B: Lightning Unified Video Editing via In-Context Sparse Attention

**论文信息：** ICML 2026 | [arXiv:2605.04569](https://arxiv.org/abs/2605.04569)

**作者：** Shitong Shao\*, Zikai Zhou\*, Haopeng Li, Yingwei Song, Wenliang Zhong, Lichen Bai, Zeke Xie† (HKUST(GZ), University of Arizona)

---

## 1. 研究动机

统一视频编辑模型正从基于 Cross-Attention 的框架（如 VACE）向基于 In-Context Learning (ICL) 的全注意力（Full-Attention）范式演进。ICL 通过直接拼接 Context Token 和 Source Token 实现信息融合，但会导致序列长度翻倍，使得注意力计算的二次复杂度成为严重的推理瓶颈。

**核心挑战：** 现有的稀疏注意力机制（如 VSA、Sparge Attn、STA、Radial Attn）均为通用视频生成场景设计，未充分利用 ICL 场景中 Context Token 和 Source Token 之间的结构性差异。

---

## 2. 核心贡献

### 2.1 关键发现：Context Token 的低显著性

通过注意力分布分析，作者发现：
- **$Q^\text{src}(K^\text{src})^\top$** 的注意力分数显著高于 **$Q^\text{src}(K^\text{ctx})^\top$**
- 这一趋势在深层网络中更加明显
- 绝大多数 Context Token 可以被安全剪枝而不损害表示保真度

### 2.2 理论分析：Query Sharpness 与近似误差的关系

**定理 1（论文核心定理）：** 令 $S_1(i) = \sigma(z_i)$ 和 $S_2(i) = \sigma(\tilde{z}_i)$ 分别为 Token $i$ 的真实和近似注意力分布。定义 **块级锐度（Sharpness）度量** $M_u \triangleq \mathbb{E}_{i \in \text{Block}_u}[\|S_2(i)\|_2^2]$。假设注意力能量 $f(i) = \|S_2(i)\|_2^2$ 在块内满足 $L$-Lipschitz 连续，则期望近似误差满足：

$$\mathcal{E}_u \le (M_u + L\delta) \cdot \mathbb{E}[\|\Delta z_i\|_2^2] + \mathcal{C}_H \mathbb{E}[\|\Delta z_i\|_2^4]$$

其中 $\delta$ 为块直径，$\mathcal{C}_H$ 为与 softmax Hessian 最大值相关的常数。

**核心洞察：** Query 的 Sharpness 与 0 阶 Taylor 展开的近似误差成正比，可作为动态分组的有效指标——高 Sharpness 的 Query 需要精确计算，低 Sharpness 的 Query 可以安全近似。

### 2.3 ISA：In-Context Sparse Attention

ISA 包含三个核心组件：

**（1）Pre-Selection（预选择）**
- 利用 Pooling Attention 计算压缩的注意力分数矩阵 $S_\text{coarse}$
- 通过 $\text{TopK}(\text{Mean}(S^\text{ctx}_\text{coarse}))$ 保留最重要的 Context Token 块
- 超参数 $\alpha_s$（Select Ratio）控制保留比例
- 复杂度从 $\mathcal{O}(NSD)$ 降至 $\mathcal{O}(N(L_\text{src} + \alpha_s L_\text{ctx})D)$

**（2）Block-Wise 0-th Order Taylor Sparse Attention（块级 0 阶泰勒稀疏注意力）**
- 对选中的稀疏块：使用标准 FlashAttention 精确计算 $S_{ij} = Q_iK_j^\top$
- 对未被选中的块：使用 0 阶泰勒近似 $S_{ij}^c = Q_i(K_j^c)^\top$，其中 $K_j^c, V_j^c$ 为块均值
- 近似块的计算复杂度从 $\mathcal{O}(L_QL_KD)$ 降至 $\mathcal{O}(L_QD)$
- 总复杂度从 $\mathcal{O}(N^2D)$ 降至 $\mathcal{O}(N^2D/b)$（$b$ 为块大小）

**（3）Grouped Computation（分组计算）**
- 基于 Sharpness $M_i$ 将 Query 分为两组：
  - **高误差 Query** → 标准 Full Attention（保证精细特征）
  - **低误差 Query** → Block-Wise 0-th Taylor Sparse Attention（编码粗粒度信息如背景）
- 超参数 $\alpha_f$（Flat Ratio）控制路由到稀疏注意力的比例
- Taylor 稀疏注意力部分可达 **93.75% 的稀疏度**，且性能几乎无损

### 2.4 LIVEditor-14B

基于 ISA 构建的 14B 参数闪电视频编辑模型，包含三个关键设计：
1. **ISA 作为核心注意力机制**：处理长 ICL 序列，计算开销可忽略
2. **两阶段训练策略**：
   - Stage 1：170 万混合质量数据预训练（lr=1e-5, batch=16）
   - Stage 2：8.9 万高质量数据精调（lr=1e-6, EMA decay=0.995）
3. **解耦 RoPE 策略**：对 Source 和 Context Token 独立应用 RoPE，位置索引分别归零，避免位置偏差

### 2.5 数据管线

自动化四阶段合成管线构建了 170 万+高质量视频编辑对：
1. **指令准备**：VLM（Gemini-2.5-Pro/GPT-4o）采样编辑任务并生成目标图像指令
2. **目标帧生成**：Gemini 2.5 Image Preview 合成锚帧 + VLM 质量过滤
3. **目标视频生成**：内部 14B TI2V 扩散模型合成 + YOLO/GroundingDINO/SAM/DOVER 多级质量评估
4. **后处理与潜在编码**：VAE 编码 + 文本编码

数据分布七大类：Style Transfer (33.87%), Object Swap, Object Addition, Object Stylization, Object Removal, Human Edit (12.80%), Other (1.09%)

---

## 3. 实验评估

### 3.1 效率

- ISA 相比 SDPA 和 FlashAttention v2/3 **减少约 60% 的注意力模块延迟**
- 端到端场景下实现 **1.47× 加速**（32 步推理含 CFG）
- 加速比随序列长度增长而进一步增大

### 3.2 性能（EditVerseBench）

| 方法 | Quality | Text Align | Temporal | Editing Q | Frame Pick | Video Pick |
|------|---------|------------|----------|-----------|------------|------------|
| **LIVEditor-14B (ISA)** | **7.89** | **20.09** | **27.19** | **24.55** | **99.32** | 99.22 |
| LIVEditor-14B (full-attn) | 7.83 | 20.02 | 27.12 | 24.47 | 99.21 | **99.24** |
| EditVerse | 7.65 | 20.07 | 27.14 | 24.32 | 98.56 | 98.44 |

- ISA 版本在几乎所有指标上超越 Full-Attention 版本
- 一致优于 SOTA 方法（Ditto, InsV2V, Lucy Edit 等）

### 3.3 稀疏注意力对比

ISA 在训练无关（training-free）场景下直接应用于预训练模型，不仅超越 VSA、SWA、Sparge Attn、Radial Attn，甚至优于原始 Full-Attention 模型。

### 3.4 消融实验

- **$\alpha_f$（Flat Ratio）**：最敏感的参数，默认设为 0.5
- **$\alpha_{ns}$（No Sparsity Ratio）**：可设低值（0.0625）
- **$\alpha_s$（Select Ratio）**：默认 0.125
- 训练后 ISA 与 Full Attention 的输出误差在几乎所有 Block 中显著降低
- 两阶段训练中 Stage 2 精调在所有指标上带来提升

### 3.5 其他 Benchmark

- **IVE-Bench**：ISA 版本在 Video Quality 四个子指标上达到最高分
- **VIE-Bench**：在 Object Addition/Swapping/Removal/Stylization 等任务中匹配或超越 Full-Attention 基线
- **FiVE-Bench**：在刚性/非刚性物体替换、颜色修改、材质修改、物体添加等任务中持续优于 Full-Attention

---

## 4. 实现细节

- **前向传播**：Triton 和 TileLang 双版本实现，序列长度 >64K 时 Triton 更优
- **反向传播**：Triton 实现，使 ISA 成为完全可微可训练的稀疏注意力
- **基础模型**：在 Wan 2.2 的高噪声分支上后训练
- **优化策略**：DeepSpeed ZeRO-3 Offload，32 张 80GB GPU（4 节点），序列并行度 2

---

## 5. 关键洞察

1. **ISA 超越 Full Attention 的原因**：TopK 预选择过滤了不相关的噪声 Token，Block-Wise Taylor 注意力引入了一种不同的注意力范式而非仅仅是数学近似
2. **ICL 场景的特殊性**：Context Token 作为条件先验而非生成目标，高语义密度是剪枝可行的基础
3. **Sharpness 作为高效代理指标**：避免了计算完整的 $Q(K-K^c)^\top$，直接从 Pooling Score Matrix 提取

---

## 6. 总结

LIVEditor-14B 是首个针对 ICL 视频编辑设计的近乎无损稀疏注意力框架。ISA 通过预选择剪枝冗余 Context Token、Query 动态分组路由、块级 0 阶泰勒近似三个机制，实现了约 60% 的注意力延迟降低。在四个 Benchmark 上超越 SOTA 全注意力方法，同时保持（甚至增强）视觉保真度。ISA 的设计原则为未来长序列视频生成任务提供了可扩展的解决方案。

---

**代码仓库：** [github.com/xie-lab-ml/Lightning-Unified-Video-Editor-via-In-Context-Sparse-Attention](https://github.com/xie-lab-ml/Lightning-Unified-Video-Editor-via-In-Context-Sparse-Attention)

---

## 7. 讨论记录

### Q1：Pre-Selection 中的 Pooling Attention 具体怎么做？

Pooling Attention 的核心思想是**在粗粒度上先算一遍注意力，用低成本的分数图来指导细粒度的选择**。具体分 4 步：

**第 1 步：块划分（Block Partition）**

由于视频模型按时空 tile 顺序展平为序列，将其按块大小 $b$ 切分：
- $Q$ → $\{Q_1, Q_2, ..., Q_{N_Q}\}$，每块 $b \times D$
- $K, V$ → $\{K_1,...,K_{N_K}\}, \{V_1,...,V_{N_K}\}$，每块 $b \times D$
- $N_Q = \lceil N/b \rceil$，$N_K = \lceil S/b \rceil$

**第 2 步：块级均值压缩（Pooling）**

对每个块沿序列维求均值，得到压缩后的表示：
$$Q^c_i = \text{mean}(Q_i) \in \mathbb{R}^{1 \times D},\quad K^c_j = \text{mean}(K_j) \in \mathbb{R}^{1 \times D}$$
最终得到 $Q^c \in \mathbb{R}^{B \times H \times N_Q \times D}$，$K^c, V^c \in \mathbb{R}^{B \times H \times N_K \times D}$。

这一步将块内 $b$ 个 token 压缩成 1 个"代表向量"，保留了局部结构信息但大幅降低了计算量。

**第 3 步：粗粒度注意力计算**

用压缩后的表示做标准 Scaled Dot-Product Attention：
$$S_\text{coarse} = \text{softmax}\left(\frac{Q^c (K^c)^\top}{\sqrt{D}}\right) \in \mathbb{R}^{B \times H \times N_Q \times N_K}$$
复杂度从 $\mathcal{O}(NSD)$ 降为 $\mathcal{O}(N_Q N_K D)$，约为原来的 $1/b^2$。

**第 4 步：Context Token 重要性排序 + TopK 选择**

由于 Source 和 Context Token 在内存中是连续存放的（$L_{src}$ 在前，$L_{ctx}$ 在后），先取出 $S_\text{coarse}$ 中仅与 Context Key 相关的子矩阵：
$$S^\text{ctx}_\text{coarse} = S_\text{coarse}[:, :, :\lceil L_{src}/b\rceil, \lceil L_{src}/b\rceil:]$$
然后沿 Query 维度求和，得到每个 Context 块的"全局重要性分数"：
$$I_\text{topk} = \text{TopK}\left(\text{Mean}(S^\text{ctx}_\text{coarse}, \text{axis=}Q\_dim),\; k=\alpha_s \cdot \lceil L_{ctx}/b\rceil\right)$$
最后通过 Gather + Concat 重构稀疏的 K、V：
$$K_\text{new} = \text{Concat}([K^\text{src}, \text{Gather}(K^\text{ctx}, I_\text{topk})])$$

**数值示例**：假设 $L_{src}=L_{ctx}=4096$，块大小 $b=64$，则 Source 和 Context 各有 64 个块。$\alpha_s=0.125$ 意味着只保留 $64 \times 0.125 = 8$ 个 Context 块（512 个 token），剪枝掉其余 56 个块（3584 个 token），即 **87.5% 的 Context Token 被认为是冗余的**，而实验表明这几乎不影响性能。

### Q2：ISA 对未选中块的处理——Block-Wise 0 阶泰勒稀疏注意力

当选中的稀疏块（$M_{ij}=1$）用标准 FlashAttention 精确计算后，未选中的块（$M_{ij}=0$）通过 0 阶泰勒近似来加速。

**核心思想**：用整个块的均值向量 $K^c_j, V^c_j$（$1\times D$）替代块内所有 $L_K$ 个 token 的逐个交互。

**数学推导**：

① 0 阶展开——将 $V_j$ 在块均值 $V^c_j$ 处展开，忽略所有高阶项：
$$V_j \approx V^c_j$$

② K 的近似——将 softmax 内的 $K_j$ 替换为块均值：
$$\text{softmax}\left(\frac{Q_i K_j^\top}{\sqrt{D}}\right) \approx \text{Expand}\left(\text{softmax}\left(\frac{Q_i (K^c_j)^\top}{\sqrt{D}}\right)\right)$$
此处需要 Expand 操作是因为 $Q_i (K^c_j)^\top$ 只产生 $L_Q\times 1$ 的标量分数，但需要对应 $L_K$ 个 token，因此沿列方向复制 $L_K$ 次。

③ 最终的近似计算公式：
$$\begin{aligned}
S_{ij}^c &= Q_i (K^c_j)^\top \cdot \frac{1}{\sqrt{D}} \quad \in \mathbb{R}^{L_Q \times 1} \\[2pt]
P_{ij}^c &= \exp(S_{ij}^c) \quad \in \mathbb{R}^{L_Q \times 1} \\[2pt]
\ell_i &\;+\!\!=\; P_{ij}^c \cdot L_K \qquad \text{(归一化分母，因为每个近似 token 贡献相同权重)} \\[2pt]
O_i &\;+\!\!=\; P_{ij}^c \cdot V^c_j \cdot L_K \qquad \text{(块均值被 $L_K$ 个 token 共享加权)}
\end{aligned}$$

**为什么要乘以 $L_K$？**

精确计算时 $P_{ij} \in \mathbb{R}^{L_Q \times L_K}$ 有 $L_K$ 个不同的注意力权重；近似计算时 $P_{ij}^c \in \mathbb{R}^{L_Q \times 1}$ 只有 1 个。因为假设块内所有 token 被均质化为同一个均值，这 1 个权重需要"代表" $L_K$ 个 token，所以归一化分母和输出聚合都要乘以 $L_K$。

**复杂度对比**：

| 操作 | 精确计算（选中块） | 0 阶泰勒近似（未选中块） |
|------|:---:|:---:|
| 得分计算 | $Q_i K_j^\top$：$\mathcal{O}(L_Q L_K D)$ | $Q_i (K^c_j)^\top$：$\mathcal{O}(L_Q D)$ |
| Value 聚合 | $P_{ij} V_j$：$\mathcal{O}(L_Q L_K D)$ | $P_{ij}^c V^c_j$：$\mathcal{O}(L_Q D)$ |
| 内存加载 | $K_j, V_j \in \mathbb{R}^{L_K \times D}$ | $K^c_j, V^c_j \in \mathbb{R}^{1 \times D}$ |

**加速比 = $L_K$ 倍**，默认 $L_K=64$ 时未选中块的计算速度理论上是精确计算的 64 倍。

**为什么不用 1 阶/2 阶泰勒？**

论文尝试过更高阶展开但放弃：1 阶展开需要计算 Jacobian-vector 乘积，引入不规则内存访问，无法高效融合进 kernel；高阶项的额外计算开销会抵消稀疏带来的加速收益。

**与 FlashAttention 的融合实现**：

整个逻辑融合在单个 Triton kernel 中，选中和未选中块共享同一套 online softmax 状态 $(m_i, \ell_i, O_i)$，避免中间结果的 HBM 读写。伪代码结构：
```
对每个 Query 块 Q_i：
  初始化 m_i = -∞, ℓ_i = 0, O_i = 0
  
  // Part A: 精确稀疏注意力（M_{ij}=1）
  for j in TopK 选中块：
      加载 K_j, V_j → 标准 FlashAttention 计算 → 更新 m_i, ℓ_i, O_i
  
  // Part B: 0 阶泰勒近似（M_{ij}=0）
  for j not in TopK：
      加载 K^c_j, V^c_j（仅 1×D 的均值向量）
      S_avg = Q_i (K^c_j)^T · scale        ← O(L_Q D)
      P_avg = exp(S_avg - m_new)           ← L_Q×1 标量
      ℓ_i += rowsum(P_avg) · L_K           ← 乘以块大小
      O_i += (P_avg · V^c_j) · L_K         ← 均值加权 × L_K
  
  O_i = O_i / ℓ_i
```

**Part B 逐维度走查（以 $L_Q=L_K=64, D=128$ 为例）：**

| 步骤 | 公式 | 输入维度 | 输出维度 | 说明 |
|:---:|------|:---:|:---:|------|
| 0（前置） | 从 Part A 继承 | — | $m_i, \ell_i \in \mathbb{R}^{64\times 1}, O_i \in \mathbb{R}^{64\times 128}$ | online softmax 运行状态 |
| 1 | 加载 $K^c_j, V^c_j$ | — | $\mathbb{R}^{1\times 128}$ | 块均值，仅 128 个元素（精确计算需加载 $64\times 128$） |
| 2 | $S_{avg} = Q_i (K^c_j)^\top / \sqrt{D}$ | $Q_i \in \mathbb{R}^{64\times 128}$, $K^c_j \in \mathbb{R}^{1\times 128}$ | $S_{avg} \in \mathbb{R}^{\color{red}{64\times 1}}$ | 每个 Query 只产出一个标量分数（精确：$64\times 64$），FLOPs 减少 64× |
| 3 | $m_{new} = \max(m_i, S_{avg})$ | $m_i, S_{avg} \in \mathbb{R}^{64\times 1}$ | $m_{new} \in \mathbb{R}^{64\times 1}$ | Element-wise max |
| 4 | $P_{avg} = \exp(S_{avg} - m_{new})$ | $S_{avg}, m_{new} \in \mathbb{R}^{64\times 1}$ | $P_{avg} \in \mathbb{R}^{64\times 1}$ | 每 Query 对**整个块**的块级注意力权重 |
| 5 | $\ell_i^{new} = \ell_i \cdot e^{m_i-m_{new}} + \mathbf{P_{avg} \cdot L_K}$ | $P_{avg} \in \mathbb{R}^{64\times 1}$ | $\ell_i^{new} \in \mathbb{R}^{64\times 1}$ | ⚠️ 乘以 $L_K$：因为 1 个块级权重要"代表" $L_K$ 个被均质化的 token |
| 6 | $O_i^{new} = \text{diag}(e^{m_i-m_{new}}) O_i + \mathbf{(P_{avg} \cdot V^c_j) \cdot L_K}$ | $P_{avg} \in \mathbb{R}^{64\times 1}$, $V^c_j \in \mathbb{R}^{1\times 128}$ | $O_i^{new} \in \mathbb{R}^{64\times 128}$ | 外积 $P_{avg} \cdot V^c_j$ 的 $(k,d)$ 元素 = $P_{avg}[k] \times V^c_j[0,d]$ |
| 7 | $O_i = O_i / \ell_i$ | $O_i \in \mathbb{R}^{64\times 128}$, $\ell_i \in \mathbb{R}^{64\times 1}$ | 最终输出 $\mathbb{R}^{64\times 128}$ | broadcast 除法，完成 softmax 归一化 |

**乘以 $L_K$ 的深层原因**：

精确计算中，块 $j$ 对 $\ell_i$ 的贡献是 $\sum_{t=1}^{L_K} \exp(S_{ij}[:,t])$，每个 token 有独立权重。0 阶近似假设块内 token 均质化——所有 $L_K$ 个 token 共享同一个 $P_{avg}[k]$，因此等价于累加 $L_K$ 次相同的值：
$$\underbrace{P_{avg}[k] + \cdots + P_{avg}[k]}_{L_K \text{ 次}} = P_{avg}[k] \cdot L_K$$
输出同理：64 个 token 的 Value 全部被近似为同一个 $V^c_j$，每个都按 $P_{avg}[k]$ 加权，总贡献 = $P_{avg}[k] \cdot V^c_j \cdot L_K$。

**数据流图**：
```
Q_i [64,128]          K^c_j [1,128]          V^c_j [1,128]
     |                      |                      |
     +------- matmul -------+                      |
     |                                              |
  S_avg [64,1]                                     |
     |  exp(S_avg - m_new)                          |
  P_avg [64,1] ---------+                          |
     |                   |                          |
     | ×L_K              +-------- outer ----------+
     v                   |                          |
  ℓ_i [64,1]     P_avg·V^c_j [64,128]              |
     |                   | ×L_K                     |
     +---- O_i / ℓ_i ----+--------------------------+
                 |
           Final O_i [64,128]
```

**Part B 变量速查表**：

| 变量 | 维度 | 含义 |
|------|:---:|------|
| $Q_i$ | $\mathbb{R}^{L_Q \times D}$ | 当前 Query 块 |
| $K^c_j$ | $\mathbb{R}^{1 \times D}$ | 第 $j$ 个 Key 块沿序列维的均值（块内 pooling） |
| $V^c_j$ | $\mathbb{R}^{1 \times D}$ | 第 $j$ 个 Value 块沿序列维的均值（块内 pooling） |
| $S_{avg}$ | $\mathbb{R}^{L_Q \times 1}$ | 粗粒度注意力分数（Query 与 Key 均值的内积） |
| $m_i$ | $\mathbb{R}^{L_Q \times 1}$ | 当前 Query 块已见过的最大 logit（online softmax 状态） |
| $m_{new}$ | $\mathbb{R}^{L_Q \times 1}$ | 更新后的 running max |
| $P_{avg}$ | $\mathbb{R}^{L_Q \times 1}$ | 块级注意力权重（已做 exp + 数值稳定） |
| $\ell_i$ | $\mathbb{R}^{L_Q \times 1}$ | softmax 分母累加（online softmax 状态） |
| $O_i$ | $\mathbb{R}^{L_Q \times D}$ | 加权输出累加（online softmax 状态） |
| $L_K$ | 标量 | 块大小（如 64），用于"1 个均值代表 $L_K$ 个 token"的缩放因子 |

### Q3：可不可以理解为用块内的 Pooling K/V 来近似计算块内的 Full-Attention？

**完全可以这么理解，而且这是最直观的认知模型。** 但有一个关键细节需要补充——"乘以 $L_K$"的校正。

#### 直观理解

Full-Attention 对块 $j$ 做的事情：
```
Q_i [64×128] × K_j^T [64×128]^T → S [64×64]  → softmax → P [64×64] → × V_j [64×128] → O [64×128]
     ↑ 64 个不同的 Key token                   ↑ 64 个不同的权重              ↑ 64 个不同的 Value token
```

0 阶泰勒近似对块 $j$ 做的事情：
```
Q_i [64×128] × (K^c_j)^T [1×128]^T → S_avg [64×1] → softmax → P_avg [64×1] → × V^c_j [1×128] → O [64×128]
     ↑ 只有 1 个 Key 均值                          ↑ 只有 1 个权重                  ↑ 只有 1 个 Value 均值
```

**本质就是**：把 "Query × 64 个不同 Key → 64 个不同权重 → 聚合 64 个不同 Value" 替换为 "Query × 1 个 Key 均值 → 1 个权重 → 聚合 1 个 Value 均值 × 64"。

#### 关键校正：为什么不能直接用，必须乘 $L_K$？

直接写 `O += P_avg · V^c_j` 会**低估该块的贡献**。因为 1 个均值 $V^c_j$ 代替了 $L_K$ 个独立 token 的 Value，输出幅值会缩小 $L_K$ 倍。

**类比**：全班 64 个学生的平均分是 80 分，如果你用 1 个"平均学生"代表全班，总贡献 = 80 × 1 = 80。但全班真实总贡献 = 80 × 64 = 5120。正确的近似是 `总贡献 = 均值 × 64`。

因此正确的写法是：

$$\text{Attention}_{\text{approx}}(Q_i, K_j, V_j) \;\approx\; \underbrace{\text{softmax}\left(\frac{Q_i (K^c_j)^\top}{\sqrt{D}}\right) \cdot V^c_j}_{\text{用 Pooling K/V 算一遍 Attention}} \;\times\; L_K$$

**总结**：0 阶泰勒稀疏注意力的核心就是 **"用块内 Pooling K/V 近似块内 Full-Attention，再乘以 $L_K$ 补偿 token 数量的缩放"**。这个校正因子是整个近似不可或缺的一部分。

### Q4：Pre-Selection 的 TopK 和 Grouped Computation 的 Sharpness 是否都是选择 Metric？两者有什么区别？

**是的，两者都是"选择判断用哪种 Attention"的 Metric，但它们作用在不同维度、回答不同问题。**

#### 两个 Metric 的对比

| 维度 | Pre-Selection（TopK） | Grouped Computation（Sharpness） |
|:---|------|------|
| **选择对象** | Context 的 K/V **块**（哪些保留、哪些剪枝） | Query **块**（哪些精确算、哪些近似算） |
| **选择轴** | Key/Value 维度 | Query 维度 |
| **原始数据** | 粗粒度注意力分数 $S_{coarse}$ | 粗粒度注意力分布 $S_3 = \text{softmax}(Q^c (K^c)^\top)$ |
| **Metric 计算** | $\text{Mean}(S_{coarse}^{ctx}, \text{axis=}Q\_dim)$ → 取 TopK | $M_u = \mathbb{E}[\|S_2(i)\|_2^2]$ → 按阈值分组 |
| **语义** | 某个 Context 块被**所有 Query 平均关注多少**（全局重要性） | 某个 Query 块的注意力分布**有多尖锐**（对近似的敏感度） |
| **回答的问题** | "哪些 Context 块对整体最有价值？" | "哪些 Query 块经不起近似，必须精确计算？" |
| **控制超参** | $\alpha_s$（Select Ratio）= 0.125 | $\alpha_f$（Flat Ratio）= 0.5 |

#### 为什么需要两个不同的 Metric？

因为稀疏发生在两个不同的阶段，需要回答两个不同的问题：

**Pre-Selection 阶段**（先做）：目标是从 Context Token 中**删除冗余**。问题本质是 "哪些 Context 块不重要？"，需要用**量级**（magnitude）来判断——被关注得少的 Context 块自然可以删。因此用 Pooling Attention 的分数求和后 TopK，是自然的**全局重要性排序**。

**Grouped Computation 阶段**（后做）：目标是在处理剩余块时**动态分配算力**。问题本质是 "哪些 Query 对近似误差敏感？"，需要用**分布形态**（distribution shape）来判断——

具体来说，Sharpness $M_u$ 的定义是：

$$M_u = \mathbb{E}_{i \in \text{Block}_u} [\|S_2(i)\|_2^2]$$

其中 $S_2(i) = \text{softmax}(Q_i (K^c)^\top)$ 是第 $i$ 个 Query token 的近似注意力分布。直观理解：

- **高 Sharpness（尖锐分布）**：注意力集中在少数几个 Key 块上，softmax 后形成尖峰 → 近似误差大 → 必须用 Full Attention
- **低 Sharpness（平坦分布）**：注意力均匀分布在多个 Key 块上，softmax 后接近均匀 → 近似误差小 → 可以用 Sparse Attention

**论文定理的核心洞察**：上界 $\mathcal{E}_u \le (M_u + L\delta) \cdot \mathbb{E}[\|\Delta z_i\|_2^2] + \cdots$ 表明，近似误差与 Sharpness $M_u$ 成正比——锐度越高的 Query，0 阶泰勒展开的误差越大，因此这些 Query 需要路由到精确计算分支。

#### 一个具体的类比

想象一个图书馆：

- **Pre-Selection = 决定去哪些书架**：从 100 个书架中，根据"被读者查阅的总次数"，选出最热门的 12 个书架（$\alpha_s = 0.125$），其余 88 个书架直接忽略。**衡量标准：总查阅量（量级）**。

- **Grouped Computation = 决定每个读者怎么看书**：对于读者 A（查询很精准，只看固定的 2-3 本书 → 高 Sharpness），需要给他精确的书籍信息（Full Attention）；对于读者 B（随便翻阅，每本书都扫一眼 → 低 Sharpness），给他每层书架的内容概要就够了（Sparse Attention）。**衡量标准：查阅模式是集中还是分散（分布形态）**。

#### 为什么论文选了 Sharpness 而不是其他候选指标？

论文明确提到他们还考虑了 $||Q(K - K^c)^\top||_\infty^2$（块内方差）作为候选，但：
1. 计算代价太高，无法高效集成
2. 实验表明它与泰勒误差的**相关性弱**，不是有效的代理指标

而 Sharpness $M_u$ 可以直接从 Pooling Attention 的分数矩阵中**零额外成本**导出，且与泰勒误差呈**强正相关**，因此成为最优选择。

### Q5：为什么 Sharpness 与 Query 相关，而不是与 K/V 相关？

**核心原因：同样的 K/V 近似误差，对不同 Query 的影响是不同的——Sharpness 衡量的是 Query 对误差的"放大系数"。**

#### 从误差上界公式看分工

回忆定理 1 的核心不等式：

$$\mathcal{E}_u \le \underbrace{(M_u + L\delta)}_{\text{Query 侧的放大因子}} \cdot \underbrace{\mathbb{E}[\|\Delta z_i\|_2^2]}_{\text{K/V 侧的原始误差}} + \cdots$$

这个公式揭示了两个独立因素：

| 因素 | 来源 | 含义 |
|------|:---:|------|
| $\mathbb{E}[\|\Delta z_i\|_2^2]$ | **K/V 侧** | Key 被块均值替代造成的基础误差（$\Delta z_i = Q_i(K - K^c)^\top$） |
| $M_u$（Sharpness） | **Query 侧** | Query 对这个基础误差的**放大倍数** |

K/V 侧误差是"原材料"——你可以把它理解为 **信号失真程度**，对于所有 Query 来说，给定的 K/V 块质量是固定的。但不同 Query 对这个失真的**容忍度**不同，这正是 Sharpness 要衡量的。

#### 直观理解：Sharpness 是什么？

给定同一个 Key 块集合的均值 $K^c$，两个不同的 Query $Q_A$ 和 $Q_B$ 分别计算：

$$S_A = \text{softmax}(Q_A (K^c)^\top),\qquad S_B = \text{softmax}(Q_B (K^c)^\top)$$

- **$Q_A$ 是"尖锐"的**：$S_A = [0.02, 0.03, \mathbf{0.85}, 0.04, 0.01, \dots]$，注意力高度集中在第 3 个 Key 块 → **$M_u$ 高**
- **$Q_B$ 是"平坦"的**：$S_B = [0.10, 0.12, 0.11, 0.09, 0.10, \dots]$，注意力均匀分布 → **$M_u$ 低**

当 K/V 被均值近似后，第 3 个 Key 块的真实 token 和均值之间的偏差，对 $Q_A$ 的影响远大于对 $Q_B$ 的影响——因为 $Q_A$ 的 85% 注意力都押在这个块上，而 $Q_B$ 只有 11%。

**类比**：K/V 的块均值近似相当于"用摘要代替原文"。Sharpness 衡量的是"你有多依赖某个具体段落"：
- 如果你只关心第 3 段（尖锐），摘要丢失的细节会让你误判严重
- 如果你每段都差不多关注（平坦），摘要已经足够

#### 为什么 K/V 不能作为分组依据？

假设我们用 K/V 侧的某个指标（如块内方差 $\|K - K^c\|^2$）来分组：

- **块 A**：块内方差大（token 之间差异大，均值代表性差）
- **块 B**：块内方差小（token 接近均值，均值代表性好）

如果用这个来路由——对块 A 精确计算、块 B 近似计算——问题在于：**一个 Query 可能对块 A 只分配了 1% 的注意力，但对块 B 分配了 90% 的注意力**。那么精确计算块 A 是浪费，近似计算块 B 反而引入大误差。

换句话说，**K/V 侧的误差需要乘以 Query 侧的注意力权重后才是真正的输出误差**。块内方差大但没被关注 → 不重要；块内方差小但被高度关注 → 近似仍可能累积误差。

#### 一图总结

```
                     K/V 侧误差                          Query 侧放大
                  (块内 token 偏离均值)                  (注意力是否集中在该块)
                         │                                      │
                         ▼                                      ▼
              ┌──────────────────┐                  ┌──────────────────┐
  块 A:      │ K 方差大 → 误差源大│    ×    Q_flat  │ 注意力 1% → ×0.01│  =  实际影响小
              └──────────────────┘                  └──────────────────┘
              
              ┌──────────────────┐                  ┌──────────────────┐
  块 B:      │ K 方差小 → 误差源小│    ×    Q_sharp │ 注意力 90% → ×0.9│  =  实际影响大
              └──────────────────┘                  └──────────────────┘
```

**这就是为什么需要按 Query 分组而非按 K/V 分组**：最终输出误差 = K/V 误差 × Query 注意力权重，Query 的 Sharpness 决定了这个乘法中的放大因子。两个因素中，Query 侧的差异（从 1% 到 90%）远大于 K/V 侧的差异，因此 Sharpness 是更有效的分组依据。

### Q6：Sharpness 是用聚合后的 K/V（0 阶泰勒近似）计算的吗？

**是的。Sharpness 的定义本身就建立在近似注意力分布 $S_2$ 上，而非精确分布 $S_1$。**

论文定理 1 的符号定义：

- $S_1(i) = \text{softmax}(Q_i K^\top)$ —— 用逐 token 的真实 $K$（精确）
- $S_2(i) = \text{softmax}(Q_i (K^c)^\top)$ —— 用块均值 $K^c$（近似）
- $M_u = \mathbb{E}_{i \in \text{Block}_u} [\|S_2(i)\|_2^2]$ —— Sharpness 定义在近似分布上

实践中更进一步，直接复用 Pre-Selection 的粗粒度矩阵：

$$M_u = \text{Var}\big(\text{softmax}(Q^c_u (K^c)^\top)\big)$$

即用 **Query 均值 × Key 均值** 的方差来近似 Sharpness，连逐 token 的 $Q_i$ 都省了。

**为什么可以用近似分布来预测近似误差？** 表面看像循环论证（用近似分布预测近似误差），但实际上 Sharpness 衡量的是**分布形态**（尖锐 vs 平坦），而非误差本身。定理 1 证明了两者正相关，但它们是不同的量——就像温度计读数可以预测发烧，但读数本身不是发烧。

更重要的是实践理由：如果用精确 $S_1$ 算 Sharpness，必须先算一遍 Full Attention，那还需要加速什么？而用 $S_2$ 可以直接复用 Pooling Attention 中已有的 $K^c$，**零额外开销**。

**ISA 的设计闭环**：

```
Pooling Attention 一次计算
         │
         ├──→ S_coarse 的 TopK ──→ Pre-Selection（选 Context 块）
         │
         └──→ S_3 的方差（Sharpness）──→ Grouped Computation（分 Query 组）
```

### Q7：ISA 与视频编辑任务的耦合程度？能否用于其他 Attention 任务？

**ISA 的三个组件与 ICL 视频编辑的耦合程度不同，需要分开看。**

#### 论文对 ICL 视频编辑 Attention 特征的分析

论文对 ICL 场景下的注意力模式做了系统分析（Fig. QK_analysis + Fig. analysis_context_token），发现了两个 ICL 特有的结构特征：

**特征 1：注意力矩阵的四象限结构**

ICL 中 Source Token 和 Context Token 在内存中连续存放，注意力矩阵天然分为四个区域：

| | $K^{src}$ | $K^{ctx}$ |
|---|---|---|
| $Q^{src}$ | 极高 | 很低 |
| $Q^{ctx}$ | 中等 | 中等 |

**特征 2：Context Token 的显著性随深度下降**

$$Q^{src}(K^{src})^\top \;\gg\; Q^{src}(K^{ctx})^\top$$

而且这个差距在网络深层**越来越大**。换句话说，Source Token 之间的交互主导了注意力，Context Token 的作用在深层几乎可以忽略。

#### 三个组件的耦合分析

| 组件 | 耦合对象 | 耦合程度 | 能否迁移到其他任务 |
|------|------|:---:|:---:|
| **Pre-Selection** | ICL 特有的 Source/Context 结构 | 🔴 强耦合 | ❌ 需要任务有类似的"两段式"输入结构（如 RAG、多模态拼接） |
| **0-th Taylor Sparse Attention** | 注意力矩阵本身 | 🟢 零耦合 | ✅ 任何 Attention 计算都可以用，纯粹是矩阵近似技术 |
| **Grouped Computation（Sharpness）** | 注意力分布形态 | 🟢 零耦合 | ✅ 任何有 $\text{softmax}(QK^\top)$ 的地方都可以用 |

#### 具体分析

**Pre-Selection 为什么是 ICL 特化的？**

论文明确说（line 233）：

> "Most existing sparse attention mechanisms are designed based on general video generation and fail to account for the distinction between context tokens and source tokens"

Pre-Selection 的整个设计前提是"Context Token 可以被安全剪枝"，这个假设建立在对 ICL 注意力矩阵的分析之上。

#### Q8：Source/Context 两段式结构不是也出现在一般 AR 框架中吗？（LLM、video continuation）

**是的，两段式结构在语法上普遍存在，但注意力语义完全不同。** 这是理解 ISA 适用范围的关键。

| 场景 | "Context" 的角色 | "Source" 的角色 | $Q^{src}(K^{ctx})^\top$ 的重要性 |
|------|------|------|:---:|
| **ICL 视频编辑** | 示例帧（展示编辑风格） | 待编辑帧 | 🟢 **很低** — 深层几乎可忽略 |
| **LLM（auto-regressive）** | 已生成的 prefix/prompt | 当前预测 token | 🔴 **极高** — 这是 Attention 的核心运算 |
| **Video Continuation** | 已给定的前置帧 | 待生成的后续帧 | 🔴 **极高** — 生成帧必须依赖前置帧做条件生成 |

**关键区别：Context 的语义角色不同。**

- **ICL 中**：Context 是"风格参考卡片"——模型看一眼就懂了编辑意图，后续主要靠 Source 内部的空间-时间一致性来完成生成。论文的 Fig. QK_analysis 证实了 $Q^{src}(K^{ctx})^\top$ 在深层几乎归零。

- **LLM 中**：Context 是"语义历史"——下一个 token 就是靠前面所有 token 做条件概率预测的。$Q_{current}(K_{past})^\top$ **就是 Attention 的全部意义**，无法安全剪枝。

- **Video Continuation 中**：Context 是"条件信号"——要生成的帧必须强烈地 attend 到前置帧才能保持时间连贯性。$$Q^{future}(K^{past})^\top$$ 不仅不能剪，反而是最关键的依赖。

**核心结论**：Pre-Selection 的可行性不来源于"输入语法上的两段式结构"，而是来源于 **ICL 特有的注意力语义**——Context 是"风格参考"而非"语义条件"，其显著性随深度递减。在 LLM 和 video continuation 中，虽然输入语法也有两段，但 Context 是强语义条件，剪掉会直接破坏任务质量。因此 Pre-Selection 是 ICL 特化的，不能直接迁移到通用 AR 场景。

**Sharpness + 0-th Taylor 为什么是通用的？**

这两个组件不依赖任何 ICL 特定假设：

- **0-th Taylor Sparse Attention**：是一种通用的矩阵近似——对任意 $Q, K, V$，用块均值近似块内 token，复杂度从 $\mathcal{O}(L_Q L_K D)$ 降至 $\mathcal{O}(L_Q D)$
- **Sharpness Grouping**：定理 1 的推导不涉及 Source/Context 区分，只假设 attention distribution 的 Lipschitz 连续性——这对任何经过训练的 Transformer 都成立

因此，如果将 ISA 用于非 ICL 的长序列 Attention（如长视频生成、长文档理解），只需要：
1. **去掉** Pre-Selection（或改为通用的 TopK 选块策略）
2. **保留** Sharpness-based Grouped Computation + 0-th Taylor Sparse Attention

这两组件组合仍然能提供显著的加速收益，且理论上界成立。

同一份粗粒度注意力矩阵同时喂给两个决策模块——一次廉价的前置计算，驱动两个维度的稀疏决策。

---

### 全局总结：ISA 的两阶段 Attention 计算优化

ISA 的完整管线可以总结为在两个维度上的两级路由决策：

```
Pooling Attention（一次廉价前置计算）
         │
         ├──→ 阶段 1：Pre-Selection（K/V 维度，块级路由）
         │        选出 TopK 个 Context 块 → Full Attention
         │        其余 Context 块 → 0-th Taylor 近似（pooling K/V）
         │
         └──→ 阶段 2：Grouped Computation（Query 维度，token/块级路由）
                  高 Sharpness 的 Query 块 → 全程 Full Attention（跳过稀疏管线）
                  低 Sharpness 的 Query 块 → 走阶段 1 的混合管线
                          ├── TopK 块 → Full Attention
                          └── 其余块 → 0-th Taylor 近似
```

**注意**：阶段 2 的 Sharpness 路由是**前置决策**——高 Sharpness Query 块从一开始就直接跑 Full Attention，并非"在 0-th Taylor 计算中检测到误差大再切换"。这一设计保证了实现的简洁性和 GPU 流水线的规整性。

**两个阶段的操作对象与粒度**：

| | 阶段 1（Pre-Selection） | 阶段 2（Grouped Computation） |
|---|---|---|
| 操作轴 | Key/Value 维度 | Query 维度 |
| 粒度 | Context K/V **块** | Query **块** |
| 选取标准 | 全局重要性（量级） | 分布形态（Sharpness） |
| 决策方式 | TopK 硬选择 | 阈值分组 |
| 结果 | 削减 K/V 序列长度 | 为不同 Query 分配不同计算精度 |
