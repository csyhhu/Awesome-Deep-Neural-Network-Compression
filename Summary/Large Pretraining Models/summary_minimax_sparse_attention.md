# MiniMax Sparse Attention (MSA) 论文总结

> **论文**: MiniMax Sparse Attention
> **arXiv**: https://arxiv.org/abs/2606.13392
> **机构**: MiniMax, Peking University, NVIDIA, Zhejiang University 等
> **代码**: https://github.com/MiniMax-AI/MSA

---

## 1. 摘要

超长上下文能力已成为前沿大语言模型（LLM）的必备能力：agentic 工作流、仓库级代码推理和持久记忆都要求模型能够同时关注数十万到数百万个 token。然而，softmax attention 的二次方计算成本使得这在部署规模上难以为继。

本文提出了 **MiniMax Sparse Attention (MSA)**，一种基于 Grouped Query Attention (GQA) 的块级稀疏注意力机制。MSA 的核心设计理念是简单性和可扩展性，使其能够在广泛的 GPU 架构上高效部署。

### 主要特点
- **Index Branch（索引分支）**：轻量级分支，通过 max-pooling 评分为每个 GQA 组独立选择 Top-k 个 KV 块
- **Main Branch（主分支）**：仅在选定的块上执行精确的块稀疏注意力计算
- **硬件协同设计**：无 exp 的 Top-k 选择、KV-outer 稀疏注意力，提高张量核心利用率

### 主要结果
- 在 109B 参数的原生多模态训练模型上，MSA 与 GQA 性能相当
- 在 1M 上下文长度下，每个 token 的注意力计算量减少 **28.4×**
- 配合协同设计的内核，在 H800 上实现 **14.2× prefill** 和 **7.6× decoding** 的实测加速

---

## 2. 引言

### 研究背景
大语言模型正从短文本单次交互快速转向跨越数百个推理和行动步骤的长 horizon agentic 工作流。然而，这些任务所需的超长上下文对训练和推理都带来了严重的计算和内存瓶颈，其中二次方成本的 softmax attention 是主要瓶颈。

### 现有方法的两类方向
1. **混合架构**：用线性注意力或滑动窗口注意力替代部分 softmax attention 层
2. **稀疏化 softmax attention**：直接稀疏化 softmax attention 以突破计算瓶颈

### MSA 的设计理念
遵循奥卡姆剃刀原则：经过大量消融实验后，只保留最本质的组件。MSA 采用块级 token 选择配合较小的 top-k，在放宽先前设计施加的 head 维度约束的同时，能够在更广泛的 GPU 架构上高效执行。

---

## 3. 方法

### 3.1 架构

MSA 是一个基于 GQA 的双分支稀疏注意力机制。

#### Index Branch（索引分支）
- 为每个 GQA 组引入一个索引 query head，以及一个跨组共享的索引 key head
- 对可见的 key token 进行评分，然后聚合到块级别（通过 max-pooling）
- 选择 top-k 个块索引
- **强制包含本地块**（包含当前 query 位置的块）

关键公式：
- 索引评分：$S^{\text{idx},(r)}_{i,j} = \frac{(Q^{\text{idx}})^{(r)}_i (K^{\text{idx}})_j^{\top}}{\sqrt{d_{\text{idx}}}}$
- 块级评分：$M^{\text{idx},(r)}_{i,b} = \max_{j \in B_b, j \le i} S^{\text{idx},(r)}_{i,j}$
- Top-k 选择：$\mathcal{I}_i^{(r)} = \text{TopK}(M^{\text{idx},(r)}_{i,\cdot}, k)$

#### Main Branch（主分支）
- 仅在 Index Branch 选择的块中的 token 上计算 softmax attention
- 每个 query 的注意力成本从 $O(N)$ 降低到 $O(kB_k)$

### 3.2 训练

由于 top-k 选择不可微分，Index Branch 通过以下机制训练：

#### KL 对齐损失
- 在选定的 token 上，将 Index Branch 的分布与 Main Branch 的分布进行 KL 散度对齐
- Teacher 分布 $P$ 对所有 query head 的 Main Branch 分布在概率级别取平均
- KL 损失仅更新索引投影矩阵 $W^{\text{idx}}_q$ 和 $W^{\text{idx}}_k$

#### 梯度截断（Gradient Detach）
- Index Branch 的输入 $X$ 被截断梯度，使得 KL 损失不会影响 backbone
- Teacher 分布 $P$ 也被截断梯度

#### Indexer 热启动（Indexer Warmup）
- 在前几次迭代中，两个分支都运行完整注意力，用 KL 损失训练新添加的索引投影
- 热启动后，切换到稀疏注意力

#### 本地块强制（Local Block）
- 每个 query 位置的本地块总是被选中，防止退化选择

### 3.3 计算复杂度

| 方法 | FLOPs |
|------|-------|
| GQA | $2 H_q d_h N^2$ |
| MSA | $H_{kv} d_{\text{idx}} N^2 + 4 H_q d_h N k B_k$ |

当 $kB_k \ll N$ 且 $H_{kv}d_{\text{idx}} \ll H_qd_h$ 时，MSA 的 FLOPs 优势随 $N$ 增长。

---

## 4. 内核设计

### 4.1 Index & TopK

#### 无 exp 选择（Exp-free selection）
- 由于 softmax 保持顺序，可以直接对索引评分 $s$ 进行排名
- 前向传播绕过 softmax 的 max/exp/sum 步骤，直接将原始评分传递给选择

#### 每线程寄存器 top-k
- 采用 $B_k = 128$，$k = 16$ 的配置
- 每个 warp 的 32 个 lane 流式处理输入行，在共享内存中维护 k 元素最小堆
- 专用内核在所有测试设置中都是最快的

### 4.2 稀疏注意力（KV-outer 迭代）

#### 选择 KV-outer 迭代顺序
- Q-outer 迭代的算术强度约为 $G$
- KV-outer 迭代的算术强度约为 $\frac{2}{3} B_k$
- 由于 $\frac{2}{3} B_k \gg G$，选择 KV-outer 迭代以最大化算术强度

#### 预调度块分块（Pre-scheduled tile chunking）
- GPU 调度器内核将每个 KV 块沿着其 query 维度分割成块
- 热块（被几乎所有 query 选择的块）被分散到许多 CTA 上
- 预分配每个 (query, chunk) 对在输出缓冲区中的槽位，无需原子操作

#### 两阶段前向传播（Two-phase forward）
- 由于每个 query 的 k 个部分由 k 个不同的 CTA 产生，禁止内联 softmax 归一化
- 分为两个内核：注意力内核和合并内核
- 使用可编程依赖启动（Programmatic Dependent Launch）隐藏内核间启动延迟

#### Query 拼接（Query concatenation）
- KV-outer 迭代将所有收集的、共享相同 KV 操作数的位置打包在一起
- 将 $\lceil 128/G \rceil$ 个 query 位置与它们的 G 个关联 query head 打包到 $128 \times 128$ 的 score MMA 中

### 4.3 稀疏 KL 损失

#### LSE 融合（LSE fusion）
- 在主传播过程中直接将 LSE 值发送到全局内存
- 跳过 KL 损失前向传播
- 后向内核直接将这些标量加载到 softmax 中

#### 动态负载均衡（Dynamic load balancing）
- 内核作为持久网格运行，CTA 通过全局原子计数器声明工作
- 每个块沿着其收集的 query 维度被划分成子块

---

## 5. 实验

### 5.1 实验设置

#### 模型结构
- 109B 总参数，6B 激活参数（MoE 模型）
- 41 层（前 3 层稠密，其余 38 层 MoE）
- 64 个 query head，4 个 KV head，head 维度 128
- 块大小 $B_k = 128$，每个 query 和 GQA 组保留 $k = 16$ 个 KV 块

#### 训练预算
- 总预算 3T tokens
- MSA-PT（从头开始稀疏预训练）
- MSA-CPT（从 Full-Attention checkpoint 继续预训练）

### 5.2 训练动态
- MSA-PT 的 LM loss 曲线与 Full Attention 几乎无法区分
- 梯度范数曲线也在整个训练过程中保持相同范围
- MSA-CPT 的 indexer 热启动阶段快速降低 KL 损失

### 5.3 主要结果

#### 与 Full Attention 基线比较
- 两个稀疏模型在总体上与 Full-Attention 基线保持竞争力
- MSA-PT 在许多数学、图像、视频和长上下文检索基准上获得最强结果
- MSA-CPT 更为保守，在大多数文本、代码和 PPL 评估上保持接近

#### 代表性评估结果（部分）
| 组别 | 基准 | Full | MSA-PT | MSA-CPT |
|------|------|------|--------|---------|
| 通用 | MMLU | 67.0 | **67.2** | 66.8 |
| 数学 | GSM8K | 76.2 | **77.7** | 73.7 |
| 代码 | HumanEval | 61.0 | **64.0** | 57.9 |
| 检索 | RULER-8K | 79.8 | **84.2** | 77.2 |
| 图像 | AI2D | 68.3 | **70.6** | 67.3 |
| 视频 | EgoSchema | 29.6 | **37.6** | 25.8 |

### 5.4 效率

- 在 1M token 下，FLOPs 减少达到 **28.4×**
- 实测加速：
  - Prefill: **14.2×**
  - Decoding: **7.6×**

---

## 6. 结论

本文提出了 MSA，一种与 GQA 协同设计的稀疏注意力机制。该架构为标准的 GQA 层附加了一个轻量级的 Index Branch：每个 GQA 组通过块级 dot-product indexer 独立选择一小组 KV 块，Main Branch 仅在选定的块上执行 softmax attention。

在 109B MoE 规模上，MSA 在大多预训练和 agentic 基准上保持了 GQA Full-Attention 基线的能力，同时在 1M 上下文下将每个 token 的注意力计算减少了 28.4×。

### 未来展望
1. 缩小残余的长上下文检索差距（更长的稀疏训练、更大的推理选择预算、更丰富的索引评分函数）
2. 将相同的选择器设计扩展到预训练之外的场景（强化学习后训练、agentic 部署）

---

## 7. 与现有方法的比较

| 方法 | 稀疏模式 | 训练方式 | 特点 |
|------|---------|---------|------|
| H2O/SnapKV | 推理时稀疏化 | 预训练 Full Attention | 继承训练成本 |
| NSA | 三类并行分支 | 原生训练 | 针对 MQA/MHA |
| MoBA | 大 KV 块 | 原生训练 | 块平均 key 评分 |
| DSA | Token 级选择 | 原生训练 | 基于 ReLU 的索引器 |
| **MSA** | **块级选择** | **原生训练** | **每 GQA 组独立 Top-k** |

---

## 参考文献

- Yuan, J. et al. (2025). Native Sparse Attention for Sliding Window Attention.
- DeepSeek-AI (2025). DeepSeek-V3 Technical Report.
- Lai, X. et al. (2025). FlexPrefill: A Context-Aware Sparse Attention Mechanism for Efficient Long-Sequence Inference.
- Lu, Y. et al. (2025). MoBA: Mixture of Block Attention.
