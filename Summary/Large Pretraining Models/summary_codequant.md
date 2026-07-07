# CodeQuant: Unified Clustering and Quantization for Enhanced Outlier Smoothing in Low-Precision Mixture-of-Experts

- **论文链接**: [arXiv:2604.10496](https://arxiv.org/abs/2604.10496)
- **作者**: Xiangyang Yin, Xingyu Liu, Tianhua Xia, Bo Bao, Vithursan Thangarasa, Valavan Manohararajah, Eric Sather, Sai Qian Zhang (NYU & Cerebras Systems)
- **会议**: ICLR 2026

---

## 1. 研究动机

Mixture-of-Experts (MoE) 架构已成为大规模语言模型的核心范式之一，但其巨大的总参数量带来了高昂的内存和通信开销。低精度量化（如 4-bit）是常用的压缩策略，但 MoE 模型中普遍存在的 **激活值离群点（activation outliers）** 会严重扩大量化动态范围，导致量化误差急剧增大、精度大幅下降。现有的旋转平滑方法（如 QuaRot、SmoothQuant）虽能部分缓解，但残差误差依然存在。另一方面，**基于聚类的非均匀量化**（如 SqueezeLLM）能通过将权重映射到紧凑的质心集合来吸收离群点，同时 LUT（查找表）实现也具有硬件友好性。

---

## 2. 核心方法

CodeQuant 提出了一个统一聚类与量化的框架，包含四个阶段：

### 2.1 Activation-Oriented Outlier Smoothing (AOS) — 激活导向的离群点平滑

- 在 MoE 的 Self-Attention 和 FFN 模块中引入**可学习的旋转矩阵** $R$（通过 Cayley 变换保证正交性并维持可微性）。
- 优化目标为最小化旋转后激活值的量化误差：$\mathop{\arg\min}_{R} ||X_R - Q(X_R)||^2$
- 由于 MoE 中 Router + 多 Expert 的结构对旋转变换具有不变性，旋转矩阵可在所有 Expert 间共享，不引入额外在线计算。
- 通过将离群点从激活空间"迁移"到权重空间，AOS 为后续的权重聚类创造了更友好的条件。

### 2.2 Adaptive Weight Clustering with Centroid Finetuning (ACCF) — 自适应权重聚类与质心微调

- 对旋转后的权重 $W_R = R^\top W$ 进行**逐行聚类**，每行有 $K$ 个质心。
- 核心优化目标：最小化聚类后矩阵乘法的输出差异 $|| X_R W_R - \tilde{X}_R W_c ||^2$。
- 针对 MoE FFN 的路由机制，额外引入 **KL 散度正则项**以保持聚类前后 token-expert 分配的一致性。
- 采用**交替优化**策略：
  1. **固定 assignment，优化质心**：通过梯度下降微调。
  2. **固定质心，优化 assignment**：推导了一种基于梯度的解析解来决定每个权重的归属，而非简单的最近邻分配。
- 通过引入输入激活的 Hessian 信息的对角近似（$D_1, D_2$），将分配决策转化为加权最近邻搜索。

### 2.3 Permutation-Invariant Outlier Grouping (POG) — 排列不变离群点分组

- 在 Block-wise 聚类场景下，某些分块内的权重大方差会导致聚类误差较大。
- POG 将权重按列划分为更小的子组，按方差排序后进行**跨组重排列**，使得高方差子组与低方差子组均匀混合。
- 排列通过置换矩阵 $P$ 形式实现，可折叠进 Self-Attention 和 MoE FFN 的权重矩阵中，不引入额外计算。
- 仅在 Block-wise 场景下有效（Embedding-wise 下 K-means 对顺序不敏感）。

### 2.4 LUT Kernel 设计

- 设计高效的 LUT-based GEMM 内核，每个权重组的质心与量化激活值的所有可能组合预先计算为查找表。
- 通过两级 Mux 选择输出，利用共享内存减少访存冲突。
- 使用 Accel-Sim 模拟器评估实际 GPU 性能，针对现代 GPU 的 bank conflict 问题优化了共享内存结构。
- 大幅减少传统量化中的反量化与乘法操作。

---

## 3. 实验设置

- **模型**: Phi-mini-MoE-Instruct, Qwen3-30B-A3B, DeepSeek-V2-Lite, Mixtral 8x7B
- **基线方法**: RTN, SmoothQuant, QuaRot, SqueezeLLM, DuQuant, SpinQuant
- **评估基准**: WikiText2/C4（PPL）、ARC/HellaSwag/MMLU/PIQA/WinoGrande（零样本 QA）、GSM8K/MATH500（数学推理）
- **配置**: A4W4 和 A8W4，Block-wise（组大小 $g=1024$）和 Embedding-wise
- **预处理**: AOS 阶段 15-50 分钟，ACCF 阶段 30-240 分钟（取决于模型大小，全离线）

---

## 4. 主要结果

### 4.1 语言建模与零样本 QA (A4W4 Embedding-wise)

| 模型 | 指标 | CodeQuant vs QuaRot |
|------|------|---------------------|
| Qwen3-30B-A3B | Wiki2 PPL | 10.31 vs 16.04（↓5.73） |
| Qwen3-30B-A3B | Avg Acc | 69.4% vs 58.1%（↑11.3%） |
| DeepSeek-V2-Lite | Wiki2 PPL | 7.08 vs 7.75（↓0.67） |
| Mixtral 8×7B | Wiki2 PPL | 4.65 vs 16.79（↓12.14） |
| Mixtral 8×7B | Avg Acc | 72.5% vs 49.7%（↑22.8%） |

CodeQuant 在所有模型和配置下均显著优于 RTN、SmoothQuant、SqueezeLLM 和 QuaRot。

### 4.2 数学推理能力

| 模型 | 任务 | CodeQuant | QuaRot | BF16 |
|------|------|-----------|--------|------|
| Qwen3-30B-A3B | GSM8K (8-shot) | **86.7%** | 50.8% | 92.4% |
| Qwen3-30B-A3B | MATH500 (4-shot) | **24.1%** | 12.8% | 32.2% |

CodeQuant 在推理密集型任务上优势尤为突出。

### 4.3 延迟评估

- GPU (Accel-Sim 模拟): CodeQuant 相比 BF16 实现平均 **2.63× 加速**，优于 QuaRot 和 SqueezeLLM。
- CPU (T-MAC 内核实测): CodeQuant 可达 **4.15× 加速**（相较于 BF16 Llama.cpp）。

### 4.4 与在线旋转方法的比较

$\text{CodeQuant}_{had}$（启用在线 Hadamard 变换）在 DeepSeek-V2-Lite 和 Qwen3-30B-A3B 上均优于 SpinQuant 和 DuQuant。

---

## 5. 消融实验

| 消融项 | 结论 |
|--------|------|
| **AOS 学习旋转 vs 随机旋转** | AOS 带来 1.4% 准确率提升，PPL 降低 0.23（Wiki2） |
| **KL 散度正则** | 引入 KL 散度可提升准确率并稳定 Router 行为（token-expert 分配变化率降低） |
| **POG 排列** | Block-wise 下 POG 带来一致的精度提升 |
| **极限低比特 (A4W2)** | CodeQuant 相比 SqueezeLLM 优势随比特数降低而增大（从 1.5%→7.2%） |

---

## 6. 关键创新点总结

1. **首次将聚类的离群点吸收能力与旋转平滑统一**：AOS 将激活离群点迁移到权重空间，ACCF 利用聚类质心吸收权重的极端值。
2. **端到端的可学习框架**：旋转矩阵（Cayley）、权重聚类（梯度驱动 assignment）、质心微调全部可微分优化。
3. **MoE 专用设计**：KL 散度保持 token-expert 路由一致性、共享旋转矩阵避免额外计算。
4. **硬件协同设计**：专用的 LUT GEMM 内核设计，充分利用现代 GPU 共享内存，消除反量化开销。
5. **全离线预处理**：所有优化在部署前完成，推理时权重矩阵固定不变，无运行时开销。

---

## 7. 潜在局限

- 预处理时间较长（Model 越大越长，Mixtral 8×7B 需约 290 分钟），但对于离线场景可接受。
- GPU 的 LUT 内核部分依赖模拟评估（Accel-Sim），实际硬件部署仍需进一步验证。
- 在少数 A8W4 Block-wise 配置下，POG 带来的提升有限（如 DeepSeek-V2-Lite 准确率仅提升 0.3%）。
