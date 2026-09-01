# 统一 Token 压缩器：DiT 中 Token Compression 与 Step-wise Sparsity Prediction 的统一框架

> **目标**：用一个统一的可学习框架，同时完成 DiT 中的 **token compression（空间维度：压缩哪些 / 多少 token）** 与 **step-wise sparsity prediction（时间维度：每个去噪步该压缩多少）**。
>
> **核心组件**：
> 1. **Token Compressor**：一个可学习的 $N \times N$ 矩阵 $W$，将 $[N, d] \to [N, d]$ 做 token 混合，再从中选择 $M$ 个 token（$M < N$）。
> 2. **Sparsity Predictor**：一个轻量 DNN，输入压缩前的原始 token $[N, d]$，输出标量 $s \in [0,1]$，由 $s$ 决定 $M = s \cdot N$。

---

## 目录

- [1. 动机](#1-动机)
- [2. Related Works（来自 Summary/Diffusion Models）](#2-related-works来自-summarydiffusion-models)
  - [2.1 ToMeSD — Token Merging](#21-tomesd--token-merging)
  - [2.2 SparseDiT — Token Sparsification](#22-sparsedit--token-sparsification)
  - [2.3 DiffSparse — Learned Token Sparsity](#23-diffsparse--learned-token-sparsity)
  - [2.4 TokenCache — Token Caching](#24-tokencache--token-caching)
  - [2.5 DyDiT — Dynamic Diffusion Transformer](#25-dydit--dynamic-diffusion-transformer)
  - [2.6 BudCache — Step-Level Caching](#26-budcache--step-level-caching)
  - [2.7 ERTACache — Error Rectification](#27-ertacache--error-rectification)
  - [2.8 RT-Lynx — Activation Sparsity](#28-rt-lynx--activation-sparsity)
  - [2.9 现有方法的维度分析小结](#29-现有方法的维度分析小结)
- [3. 核心思想](#3-核心思想)
- [4. 框架设计](#4-框架设计)
  - [4.1 Token Compressor（可学习 N×N 矩阵）](#41-token-compressor可学习-nn-矩阵)
  - [4.2 Sparsity Predictor（DNN → Scalar）](#42-sparsity-predictordnn--scalar)
  - [4.3 统一前向流程](#43-统一前向流程)
- [5. 数学形式化](#5-数学形式化)
- [6. 与现有方法的统一关系](#6-与现有方法的统一关系)
- [7. 设计选择与讨论](#7-设计选择与讨论)
- [8. 训练策略](#8-训练策略)
- [9. 开放问题与未来方向](#9-开放问题与未来方向)

---

## 1. 动机

DiT（Diffusion Transformer）的推理瓶颈在于：每个去噪步都要对 $N$ 个 token 做完整的 self-attention（$O(N^2)$）和 MLP（$O(N \cdot d)$）计算，且需重复数十到数百步。现有加速方法沿着两条近乎正交的路径展开：

| 路径 | 解决的问题 | 代表方法 | 共同缺陷 |
|------|-----------|---------|---------|
| **空间压缩**（token compression） | 减少单步内的 token 数量 $N \to M$ | ToMeSD, SparseDiT, DyDiT | 压缩率多为**手工设定**或**固定调度**，缺乏对输入内容的自适应 |
| **时间复用**（step-wise caching） | 跨步复用计算结果，减少 NFE | TokenCache, BudCache, ERTACache, DiffSparse | 不改变单步内的 token 数量，与空间压缩正交但往往不联合优化 |

**关键观察**：

1. **空间压缩率应当随去噪步变化**（SparseDiT 已验证）：早期步处理全局低频结构，少量 token 足够；后期步处理高频细节，需要更多 token。
2. **空间压缩率还应随输入内容变化**：简单图像（纯色背景、单一物体）可大幅压缩，复杂图像（密集纹理、多物体）需保守压缩。DiffSparse 的代价表与 token 长度解耦但**不随实例变化**；TokenCache 的 $\alpha$ 虽内容感知但只做缓存复用、不做真正的 token 降维。
3. **压缩方式（如何 mix/select）与压缩率（保留多少）应当联合学习**：现有方法二者分离——ToMe 用固定 attention merge + 固定比例；SparseDiT 用固定 pooling + 线性调度；DiffSparse 学习比例但沿用已有 selector。

**本框架的命题**：将「token compression」与「step-wise sparsity prediction」统一为**两个可学习模块的协同**——compressor 学习「怎么压缩」，sparsity predictor 学习「压多少」，二者以原始 token 为桥梁联合训练，实现**时间步 × 内容**双自适应。

---

## 2. Related Works（来自 Summary/Diffusion Models）

### 2.1 ToMeSD — Token Merging

> **来源**：[summary_tome_token_merging_diffusion.md](file:///d:/WorkSpace/Awesome-Deep-Neural-Network-Compression/Summary/Diffusion%20Models/summary_tome_token_merging_diffusion.md)
> **论文**：Bolya et al., CVPR 2023

**核心机制**：无训练的推理时操作。将 token 分为 src/dst 集合，按 attention 相似度把 src 中 $r$ 个 token 合并到 dst（取平均），计算后 unmerge（复制回原位）。

**与本框架的关系**：
- ToMe 的 merge 矩阵 $W_\downarrow \in \mathbb{R}^{M \times N}$ 由 **attention 权重动态生成**（内容驱动），是本框架中 $W$ 的一种「内容驱动 + 不可学习」退化形式。
- ToMe 的压缩率 $r$ 是**固定常数**（无 sparsity predictor），是本框架中 predictor 退化为常数函数 $f_\psi = c$ 的特例。
- ToMeSD 的总结文档中 Part III 已给出 TokenCompress 统一公式 $T' = W \cdot T$ 与 Scheduling 统一公式 $r(t) = \sigma(f_\psi(t, T, x_t))$，本框架正是在此基础上提出**具体的可学习实例化**。

**关键数据**：60% token reduction 下 2× 加速、5.6× 内存节省，但 FID 上升；仅对 self-attention 应用、仅对高分辨率层应用效果最佳。

### 2.2 SparseDiT — Token Sparsification

> **来源**：[summary_sparse_dit_token_sparsification.md](file:///d:/WorkSpace/Awesome-Deep-Neural-Network-Compression/Summary/Diffusion%20Models/summary_sparse_dit_token_sparsification.md)
> **论文**：Chang et al., NeurIPS 2025

**核心机制**：三段式空间架构 + 时间步动态剪枝。
- **底层**：Poolingformer（全局平均池化替代 self-attention，$W = \frac{1}{N}\mathbf{1}^T$，秩-1 均匀矩阵，token 数不变）
- **中层**：SDTM（Sparse-Dense Token Module）——自适应空间池化下采样 $N \to M$ → 稀疏 Transformer 处理 → 上采样 + cross-attention 恢复
- **顶层**：标准 Dense Transformer
- **时间步**：剪枝率 $r$ 随去噪推进线性递减（早期 $r$ 大 token 少，后期 $r$ 小 token 多）

**与本框架的关系**：
- SparseDiT 的 Poolingformer 是本框架 $W$ 的「位置无关 + 固定均匀」退化；SDTM 是 $W$ 的「位置驱动 + 块对角池化」退化。
- SparseDiT 的时间步调度是**线性函数** $r(t) = r_{\min} + (r_{\max} - r_{\min}) \cdot t/T$，是本框架 sparsity predictor 退化为「仅时间步感知、无内容感知」的线性形式的特例。
- SparseDiT 的每层 token 数**固定预设**，缺乏实例自适应——这正是本框架要解决的。

**关键数据**：DiT-XL 512×512 上 FLOPs -55%、速度 +175%、FID 仅 +0.09。

### 2.3 DiffSparse — Learned Token Sparsity

> **来源**：[summary_diffsparse_learned_token_sparsity.md](file:///d:/WorkSpace/Awesome-Deep-Neural-Network-Compression/Summary/Diffusion%20Models/summary_diffsparse_learned_token_sparsity.md)
> **论文**：Zhu et al., ICLR 2026 投稿

**核心机制**：**与本框架最接近的现有工作**。由三部分组成：
1. **Token 选择器**：沿用已有重要性分数（attention/similarity/norm），决定「哪些 token 重算」
2. **可学习稀疏代价预测器 $C$**：一张 $(T \times L) \times |S|$ 的参数表（非神经网络，仅几千参数），编码每个 (步, 层) 在各候选稀疏率下的代理代价
3. **动态规划求解器**：在全局压缩约束 $R$ 下求最优分配

**关键特点**：
- 代价表 $C$ 与 token 长度 $N$ **解耦**，可跨分辨率迁移
- 用 STE 打通离散 mask 的梯度，LPIPS 蒸馏端到端优化 $C$
- 模型权重**全程冻结**，只训练轻量代价表
- 推理时 DP 不运行，用预计算 mask，零推理开销

**与本框架的关系与关键区别**：

| 维度 | DiffSparse | 本框架 |
|------|-----------|--------|
| 学习对象 | 代价表 $C$（决定保留多少 token） | 压缩矩阵 $W$（决定怎么压缩）**+** sparsity predictor（决定压多少） |
| 压缩方式 | 不学习——沿用已有 selector（top-K by importance） | **学习** $N \times N$ 矩阵做 token 混合后再选择 |
| 内容自适应 | ❌ 代价表对同一 (t, layer) 对所有实例相同 | ✅ Predictor 输入原始 token，按实例输出 $s$ |
| 时间步感知 | ✅ 代价表维度含 $T$ | ✅ Predictor 可接收 $t$（或通过 token 统计量隐式感知） |
| 决策粒度 | 离散候选集 $S$（如 $\{0, 0.25, ..., 1.0\}$） | 连续标量 $s \in [0,1]$ |
| 求解方式 | 训练时 DP 全局优化 | 端到端梯度下降 |

**启发**：DiffSparse 证明了「学习稀疏分配」优于手工设定和搜索方法，且「学多少」与「学哪些」正交。本框架进一步将「学哪些/怎么压缩」也纳入学习，并加入实例级内容自适应。

### 2.4 TokenCache — Token Caching

> **来源**：[summary_tokencache_token_caching_dit.md](file:///d:/WorkSpace/Awesome-Deep-Neural-Network-Compression/Summary/Diffusion%20Models/summary_tokencache_token_caching_dit.md)
> **论文**：Lou et al., IEEE TSSP 2025

**核心机制**：三级缓存调度——Token 级（重要性分数 $\alpha$）/ Block 级（自适应分配比例）/ Timestep 级（I-step 全计算 vs P-step 缓存复用）。Cache Predictor 是轻量 DiT block，输出 $L \times N$ 个重要性分数。

**与本框架的关系**：
- TokenCache 的 $\alpha \in [0,1]^N$ 是**逐 token**的混合系数（复用 vs 重算），本框架的 $s \in [0,1]$ 是**全局标量**（保留比例）。前者是时间维度复用（$N \to N$），后者是空间维度降维（$N \to M$）。
- TokenCache 的 Cache Predictor **是内容感知的**（输入 $x_t$），本框架的 sparsity predictor 与之类似但输出粒度不同。
- 二者正交可叠加：本框架决定「每步保留多少 token 做计算」，TokenCache 决定「哪些 token 复用上一步结果」。

### 2.5 DyDiT — Dynamic Diffusion Transformer

> **来源**：[summary_dydt_dynamic_diffusion_transformer.md](file:///d:/WorkSpace/Awesome-Deep-Neural-Network-Compression/Summary/Diffusion%20Models/summary_dydt_dynamic_diffusion_transformer.md)
> **论文**：Zhao et al., ICLR 2025

**核心机制**：两个动态维度——
- **TDW（Timestep-wise Dynamic Width）**：根据时间步 embedding 动态激活不同的 attention head 和 MLP channel group
- **SDT（Spatial-wise Dynamic Token）**：根据 token 难度跳过 MLP 计算（对角 0/1 掩码，token 数不变）

**与本框架的关系**：
- DyDiT 的 SDT 是「token 级跳过」（$N \to N$，对角矩阵），本框架是「token 级降维」（$N \to M$，非对角混合矩阵）。
- DyDiT 的 Router 输入是时间步 embedding $E_t$（TDW）或 token 特征（SDT），本框架的 predictor 输入是原始 token——与 SDT 的内容感知思路一致，但输出从「逐 token 0/1」变为「全局比例标量」。
- DyDiT 需要**微调 + Router 训练**，本框架同样需要训练（compressor + predictor）。

### 2.6 BudCache — Step-Level Caching

> **来源**：[summary_budcache_step_level_caching.md](file:///d:/WorkSpace/Awesome-Deep-Neural-Network-Compression/Summary/Diffusion%20Models/summary_budcache_step_level_caching.md)
> **论文**：Lei et al., ICML 2026

**核心机制**：**预算约束的步级缓存**。翻转传统范式——先确定计算预算 $B$（NFE），再通过模拟退火 + 爬山法**离线搜索**最优二进制掩码 $\mathbf{m} \in \{0,1\}^K$（哪些步计算、哪些步复用缓存）。

**关键洞见**：
- 步级冗余模式是 **ODE/模型的内在属性**，跨 prompt 高度一致 → 固定 mask 即可泛化
- 离线搜索 → 在线推理零开销，NFE 严格确定
- 进一步用梯度下降优化时间离散化 $\boldsymbol{\sigma}$（缓存感知的时间表对齐）

**与本框架的关系**：
- BudCache 处理的是「时间维度：哪些步跳过」，本框架处理的是「空间维度：每步保留多少 token」——正交可叠加。
- BudCache 的 mask 是**离线搜索的固定策略**（非内容自适应），本框架的 predictor 是**在线内容自适应**。二者代表两种哲学：BudCache 追求延迟确定性，本框架追求质量-效率的实例级最优。
- BudCache 的「预算先定、策略后搜」思路可借鉴：本框架也可设定 FLOPs 预算，让 predictor 在预算约束下学习最优 $s$。

### 2.7 ERTACache — Error Rectification

> **来源**：[summary_ertacache.md](file:///d:/WorkSpace/Awesome-Deep-Neural-Network-Compression/Summary/Diffusion%20Models/summary_ertacache.md)
> **论文**：Peng et al., ICLR 2026

**核心机制**：将缓存误差分解为**特征偏移误差**（$\varepsilon_i$）和**步骤放大误差**（$\Delta t_i \cdot \varepsilon_i$），分别用闭式解误差修正（$v_i^{corr} = v_i + \sigma(K_i v_i + B_i)$）和轨迹感知时间步调整（$\Delta t_i = \Delta t_c \cdot \phi_i$）校正。

**与本框架的关系**：
- ERTACache 的误差修正公式 $v_i^{corr} = v_i + \sigma(K_i v_i + B_i)$ 与本框架的 compressor $T' = W \cdot T$ 形式上类似（都是线性变换 + 非线性），但目的不同：ERTACache 修正缓存偏差，本框架做 token 压缩。
- ERTACache 的离线标定思路（跨 prompt 一致性假设）提示：本框架的 predictor 是否也需要考虑跨实例的稳定性。

### 2.8 RT-Lynx — Activation Sparsity

> **来源**：[summary_rt_lynx_activation_sparsity.md](file:///d:/WorkSpace/Awesome-Deep-Neural-Network-Compression/Summary/Diffusion%20Models/summary_rt_lynx_activation_sparsity.md)
> **论文**：Cong et al., arXiv 2026

**核心机制**：利用 DiT 激活值的天然稀疏性（叠加机制导致每 token 仅激活 5%~10% 神经元），对激活施加 2:4 N:M 半结构化稀疏化 + 范数补偿 + LoRA 适应。

**与本框架的关系**：
- RT-Lynx 压缩的是**通道维度**（$d \to d'$），本框架压缩的是 **token 维度**（$N \to M$）——正交可叠加。
- RT-Lynx 的「激活天然稀疏」观察暗示：token 维度同样存在天然冗余（图像的平滑区域），本框架的 compressor 可学习利用这种冗余。

### 2.9 现有方法的维度分析小结

| 方法 | 压缩维度 | 压缩方式（怎么压） | 压缩率（压多少） | 内容自适应 | 时间步自适应 | 可学习 |
|------|---------|------------------|----------------|-----------|------------|--------|
| ToMeSD | Token 数量 | Attention merge（固定） | 固定常数 | ✗（merge 是，比例不是） | ✗ | ✗ |
| SparseDiT | Token 密度 | Pooling/SDTM（固定） | 线性函数 of $t$ | ✗ | ✅（线性） | ✗（结构） |
| DiffSparse | Token 缓存 | 沿用已有 selector | 可学习代价表 + DP | ✗ | ✅ | ✅（仅代价表） |
| TokenCache | Token 结果复用 | 重要性分数 $\alpha$ | 可学习 $\alpha$ + 调度 | ✅ | ✅ | ✅（predictor） |
| DyDiT | Head/Channel/Token | Router 路由 | Router 决策 | ✅（SDT） | ✅（TDW） | ✅ |
| BudCache | Step 计算复用 | — | 离线搜索 mask | ✗ | ✅（固定 mask） | ✗ |
| ERTACache | Step 误差修正 | 闭式解 $K, B$ | 阈值 $\lambda$ | ✗ | ✅ | ✗（闭式） |
| RT-Lynx | 通道激活 | 2:4 Top-K | 固定 50% | ✗ | ✗ | ✅（仅 LoRA） |
| **本框架** | **Token 数量** | **可学习 $N \times N$ 矩阵** | **可学习 predictor → 标量** | **✅** | **✅** | **✅（双模块）** |

**本框架的定位**：首个同时实现「可学习压缩方式 + 可学习压缩率 + 内容自适应 + 时间步自适应」的 token compression 框架。DiffSparse 最接近但压缩方式不可学习且无内容自适应；TokenCache 内容自适应但不做真正降维。

---

## 3. 核心思想

### 3.1 一句话概括

> **用一个可学习的 $N \times N$ 矩阵做 token 混合（学习「怎么压缩」），用一个轻量 DNN 从原始 token 预测稀疏比例标量（学习「压多少」），二者以原始 token 为输入桥梁联合训练，实现 DiT 中 token compression 与 step-wise sparsity prediction 的统一。**

### 3.2 设计理念

将 token compression 分解为两个**正交但协同**的可学习子问题：

```
┌─────────────────────────────────────────────────────────┐
│                    原始 Token T ∈ [N, d]                 │
│                         │                               │
│           ┌─────────────┴─────────────┐                 │
│           ▼                           ▼                 │
│   ┌───────────────┐         ┌──────────────────┐       │
│   │ Sparsity      │         │ Token Compressor │       │
│   │ Predictor     │         │  T' = W · T       │       │
│   │ (DNN)         │         │  [N,d] → [N,d]   │       │
│   │               │         │                   │       │
│   │ [N,d] → s     │         │  + Select M       │       │
│   │  s ∈ [0,1]    │         │  tokens           │       │
│   └───────┬───────┘         └────────┬──────────┘       │
│           │                          │                  │
│           │    M = s · N             │                  │
│           └──────────┬───────────────┘                  │
│                      ▼                                  │
│            ┌──────────────────┐                         │
│            │  Compressed T'   │                         │
│            │  ∈ [M, d]        │                         │
│            │  M < N           │                         │
│            └──────────────────┘                         │
│                      │                                  │
│                      ▼                                  │
│            后续 DiT Block 在 M 个 token 上计算          │
└─────────────────────────────────────────────────────────┘
```

**两个子问题**：
1. **「怎么压缩」**（Compressor）：通过 $N \times N$ 矩阵 $W$ 学习 token 间的混合关系，将原始 token 变换到便于选择/聚合的表示空间。这取代了 ToMe 的 attention merge、SparseDiT 的 pooling 等固定策略。
2. **「压多少」**（Predictor）：通过 DNN 从原始 token 预测稀疏比例 $s$，决定保留 $M = s \cdot N$ 个 token。这取代了 ToMe 的固定常数、SparseDiT 的线性函数、DiffSparse 的代价表。

**为什么用原始 token 作为 predictor 输入**：原始 token 最直接地反映当前输入的复杂度/冗余度。压缩后的 token 已丢失信息，不利于评估「还能压多少」。这与 TokenCache 的 Cache Predictor（输入 $x_t$）和 DyDiT 的 SDT Router（输入 token 特征）思路一致。

**为什么是标量而非逐 token 向量**：
- 标量 $s$ 决定全局保留比例，与 SparseDiT/DiffSparse 的「每层一个稀疏率」对齐，工程实现简单（Top-M 选择）。
- 逐 token 向量（如 TokenCache 的 $\alpha \in [0,1]^N$）更适合缓存复用场景；本框架做真正降维，标量 + Top-M 已足够。
- 若需更细粒度，predictor 可扩展为输出逐 token 重要性分数（见 §7 讨论）。

### 3.3 与「step-wise」的统一

「Step-wise sparsity prediction」体现在：每个去噪步 $t$ 的输入 token $T_t$ 不同，predictor 自然输出不同的 $s(t)$。这使得：

- **早期步**（$t$ 大，接近噪声）：token 冗余高、结构简单 → predictor 输出小 $s$ → 保留少量 token → 加速全局结构生成
- **后期步**（$t$ 小，接近图像）：token 差异大、细节丰富 → predictor 输出大 $s$ → 保留更多 token → 保证细节质量

这一行为与 SparseDiT 的经验观察（「早期全局 → 后期细节」）一致，但从「手工线性调度」升级为「数据驱动的实例自适应」。

**关键区别**：SparseDiT 的调度是 $r(t) = f(t)$（仅依赖时间步，对所有实例相同）；本框架是 $s(t) = f_\psi(T_t)$（依赖当前 token，对每个实例不同）。后者严格更强——若所有实例在步 $t$ 的 token 统计量一致，则退化为前者。

---

## 4. 框架设计

### 4.1 Token Compressor（可学习 N×N 矩阵）

#### 4.1.1 基本形式

设输入 token $T \in \mathbb{R}^{N \times d}$，compressor 的核心操作：

$$T' = W \cdot T, \quad W \in \mathbb{R}^{N \times N}$$

其中 $T' \in \mathbb{R}^{N \times d}$ 是混合后的 token（数量仍为 $N$）。随后从中选择 $M$ 个 token。

**$W$ 的语义**：$W_{ij}$ 表示「输出 token $i$ 从输入 token $j$ 聚合多少信息」。$W$ 的每一行是一个聚合核，对全部 token 做加权求和。

#### 4.1.2 选择机制

混合后 $T' \in \mathbb{R}^{N \times d}$ 中选 $M$ 个 token 的方式有几种设计选择：

| 选择方式 | 公式 | 特点 |
|---------|------|------|
| **Top-M by score** | $\text{score}_i = g(T'_i)$，取 top-M | 需要额外打分函数 $g$ |
| **$W$ 行范数** | $\text{score}_i = \|W_i\|_2$，取 top-M | 无额外参数，行范数反映该 token 聚合的信息量 |
| **可学习选择向量** | $\text{score} = \text{softmax}(W \cdot \mathbf{1})$ | $W$ 自带选择语义 |
| **Soft 选择（可微）** | $T''_i = \sigma(\text{score}_i) \cdot T'_i$，再 threshold | 端到端可微 |

**推荐方案**：让 $W$ 同时承担「混合」与「选择」双重职责。具体地，对 $W$ 的每一行计算信息量分数，选 top-M 行对应的 token：

$$\text{score}_i = \|W_i\|_2 \cdot \|T'_i\|_2, \quad \mathcal{I}_M = \text{TopM}(\text{score})$$
$$T_{\text{out}} = T'[\mathcal{I}_M] \in \mathbb{R}^{M \times d}$$

#### 4.1.3 $N \times N$ 矩阵的参数化策略

直接学习完整的 $N \times N$ 矩阵存在两个问题：(1) 参数量 $O(N^2)$，当 $N = 4096$ 时达 16M/层；(2) 无法适配变化的 $N$（不同分辨率）。因此需要参数化：

**策略 A：低秩分解（推荐用于固定 N）**

$$W = U \cdot V^T, \quad U, V \in \mathbb{R}^{N \times r}, \quad r \ll N$$

参数量降为 $O(Nr)$。$r$ 为秩，控制混合的范围（$r$ 小 → 局部混合，$r$ 大 → 全局混合）。

**策略 B：内容生成（推荐用于变化 N / 内容自适应）**

$$W = g_\phi(T) \in \mathbb{R}^{N \times N}$$

其中 $g_\phi$ 是生成函数。常见实例化：
- $g_\phi(T) = \text{softmax}(T \cdot T^T / \sqrt{d})$（attention，退化为 ToMe）
- $g_\phi(T) = \text{softmax}(\text{MLP}([T, \text{pos}]))$（学习驱动，退化为 SDTM）
- $g_\phi(T) = \text{Linear}_\phi(\text{meanpool}(T)) \cdot \mathbf{1}^T$（全局，退化为 Poolingformer）

但内容生成方式计算 $T \cdot T^T$ 仍是 $O(N^2)$，需配合线性注意力或稀疏化。

**策略 C：块对角 / 带状（兼顾效率与局部性）**

$$W = \text{blkdiag}(W_1, W_2, \ldots, W_{N/k}), \quad W_i \in \mathbb{R}^{k \times k}$$

每个块仅混合局部 $k$ 个 token，参数量 $O(Nk)$，计算量 $O(Nk)$。这与 SDTM 的空间 cell 池化、DiTFastAttn 的 window attention 思路一致。

**策略 D：固定基 + 可学习系数（推荐用于跨分辨率泛化）**

$$W = \sum_{j=1}^{J} \alpha_j \cdot B_j, \quad B_j \text{ 为固定基矩阵}, \alpha_j \text{ 可学习}$$

基矩阵 $B_j$ 可选：恒等、全局均值、空间池化核、DCT 基等。参数量 $O(J)$，与 $N$ 无关，天然支持变 $N$。

#### 4.1.4 恢复机制（可选）

若压缩后需恢复到 $N$ 个 token（类似 ToMe 的 unmerge / SparseDiT 的 SDTM 恢复），引入升维矩阵 $W_\uparrow \in \mathbb{R}^{N \times M}$：

$$\hat{T} = W_\uparrow \cdot T_{\text{out}} \in \mathbb{R}^{N \times d}$$

- $W_\uparrow = W_\downarrow^T$（伪逆恢复）
- $W_\uparrow$ 独立可学习（如 SDTM 的上采样 + 线性融合）
- $W_\uparrow$ 可省略（若后续层在 $M$ 个 token 上计算，最后再上采样，类似 SparseDiT 中层处理）

### 4.2 Sparsity Predictor（DNN → Scalar）

#### 4.2.1 基本形式

$$s = \sigma(f_\psi(T)), \quad T \in \mathbb{R}^{N \times d}, \quad s \in [0, 1]$$

其中 $f_\psi$ 是轻量 DNN，$\sigma$ 是 sigmoid。$M = \text{round}(s \cdot N)$。

#### 4.2.2 网络结构设计

Predictor 需从 $[N, d]$ 的 token 集合输出一个标量，本质是 **set-to-scalar** 的映射。设计选择：

**方案 1：均值池化 + MLP（最简）**

$$f_\psi(T) = \text{MLP}_\psi(\text{meanpool}(T)) \in \mathbb{R}$$

参数量极小（MLP 几层全连接），计算量 $O(Nd + d^2)$。缺点：丢失 token 间方差信息。

**方案 2：统计量拼接 + MLP（推荐起步方案）**

$$f_\psi(T) = \text{MLP}_\psi([\text{mean}(T), \text{std}(T), \text{entropy}(\text{attn}(T)), \ldots])$$

输入 token 的多种统计量（均值、方差、注意力熵、频域能量比等），MLP 输出标量。参考 ToMeSD 统一框架中 §21.2 的反馈信号列表。优点：可解释、计算量低、无需训练大网络。

**方案 3：轻量 Transformer + 池化（最强表达力）**

$$f_\psi(T) = \text{MLP}(\text{meanpool}(\text{LightTransformer}_\psi(T)))$$

类似 TokenCache 的 Cache Predictor（复用第一层权重初始化）或 DyDiT 的 Router。参数量约 3.57%，表达力最强但开销最大。

**方案 4：与时间步联合（显式 step-wise）**

$$s = \sigma(f_\psi([T, t/T, \text{time\_embed}(t)]))$$

显式输入时间步 $t$，让 predictor 同时感知「当前是第几步」和「当前 token 长什么样」。这最直接地实现「step-wise sparsity prediction」。

**推荐**：从方案 2（统计量 + MLP）起步，验证有效后升级到方案 4（联合时间步）。

#### 4.2.3 离散化与可微性

$M = \text{round}(s \cdot N)$ 是不可微的。训练时需处理：

- **Straight-Through Estimator (STE)**：前向用 round，反向梯度直通（DiffSparse、DyDiT 均用此法）
- **Gumbel-Sigmoid**：$M = \text{GumbelSigmoid}(s \cdot N)$，可微采样（DyDiT 用此法）
- **Soft 选择**：不用 hard round，而是对每个 token 加 soft mask $\sigma(\alpha_i)$，$\alpha$ 由 $s$ 参数化

#### 4.2.4 预算约束

为控制总 FLOPs，可加入预算约束（借鉴 DyDiT 的 FLOPs-aware 训练、BudCache 的预算约束思想）：

$$\mathcal{L}_{\text{FLOPs}} = \left(\frac{1}{T}\sum_{t=1}^{T} s(t) - \lambda\right)^2$$

其中 $\lambda$ 为目标平均保留率。总损失：

$$\mathcal{L} = \mathcal{L}_{\text{diffusion}} + \lambda_1 \mathcal{L}_{\text{FLOPs}} + \lambda_2 \mathcal{L}_{\text{recon}}$$

### 4.3 统一前向流程

给定去噪步 $t$ 的输入 latent $x_t$，经 patchify 得到 token $T_t \in \mathbb{R}^{N \times d}$：

```
Step t:
  1. T_t = Patchify(x_t)                          # [N, d]
  
  2. s(t) = σ(f_ψ(T_t))                           # Sparsity Predictor → scalar ∈ [0,1]
  
  3. M_t = round(s(t) · N)                        # 保留 token 数
  
  4. T'_t = W · T_t                                # Token Compressor 混合, [N, d] → [N, d]
  
  5. score_i = ‖W_i‖₂ · ‖T'_t[i]‖₂               # 计算 token 重要性
     I_M = TopM(score)                            # 选择 M_t 个 token 的索引
  
  6. T_comp = T'_t[I_M]                           # [M_t, d] 压缩后 token
  
  7. 在 T_comp 上执行 DiT Block 计算               # self-attention: O(M_t²) 而非 O(N²)
  
  8. (可选) T̂ = W_↑ · T_comp                       # 恢复到 [N, d]
  
  9. x_{t-1} = Update(x_t, T̂)                     # ODE/flow 更新
```

**关键性质**：
- 步骤 2-3 实现 **step-wise sparsity prediction**（每个去噪步自适应决定压缩率）
- 步骤 4-6 实现 **token compression**（可学习混合 + 选择）
- 二者通过 $M_t$ 耦合：predictor 决定 $M_t$，compressor 据此选 $M_t$ 个 token
- 整个流程端到端可微（配合 STE/Gumbel），可联合训练 $W$ 和 $\psi$

---

## 5. 数学形式化

### 5.1 统一公式

将 compressor 和 predictor 联合表达：

$$\boxed{T_{\text{out}}^{(t)} = \text{Select}_M\Big(\underbrace{W_\phi \cdot T_t}_{\text{Compressor: 混合}}, \quad M = \underbrace{\text{round}(\sigma(f_\psi(T_t)) \cdot N)}_{\text{Predictor: 压多少}}\Big)}$$

其中：
- $W_\phi \in \mathbb{R}^{N \times N}$：可学习压缩矩阵（参数 $\phi$）
- $f_\psi: \mathbb{R}^{N \times d} \to \mathbb{R}$：可学习 sparsity predictor（参数 $\psi$）
- $\text{Select}_M(\cdot, M)$：从 $N$ 个 token 中选 $M$ 个（Top-M by score）

### 5.2 退化关系

通过固定/特化各组件，本框架退化为现有方法：

| 退化配置 | $W_\phi$ | $f_\psi$ | 退化为 |
|---------|----------|---------|--------|
| $W$ = attention 权重, $f_\psi = c$（常数） | 内容驱动 | 常数 | **ToMeSD** |
| $W$ = $\frac{1}{N}\mathbf{1}^T$（均匀）, $f_\psi = c$ | 固定均匀 | 常数 | **SparseDiT-Poolingformer** |
| $W$ = 块对角池化, $f_\psi$ = 线性函数 of $t$ | 位置驱动 | 线性 | **SparseDiT-SDTM** |
| $W$ = 对角 0/1 掩码, $f_\psi$ = Router | 跳过 | Router | **DyDiT-SDT** |
| $W$ = 恒等, $f_\psi$ = 可学习代价表 | 不混合 | 代价表 | **DiffSparse**（压缩方式不可学部分） |
| $W$ = 恒等, $f_\psi(T) = \alpha$（逐 token） | 不混合 | 逐 token | **TokenCache**（缓存复用） |

### 5.3 与 ToMeSD 统一框架的关系

ToMeSD 总结文档（Part III）提出了两个统一公式：
- TokenCompress：$T' = W \cdot T, \quad W = g_\phi(T, \text{pos})$
- Scheduling：$r(t) = \sigma(f_\psi(t, T, x_t))$

本框架是这两个公式的**具体可学习实例化**：
- TokenCompress 的 $g_\phi$ 在本框架中实例化为**可学习 $N \times N$ 矩阵**（而非 attention/pooling 等固定形式）
- Scheduling 的 $f_\psi$ 在本框架中实例化为**输入原始 token 的 DNN**（而非常数/线性/阶跃函数）
- ToMeSD 文档中 §22.4 的「统一框架最终形式」$T'_t = g_\phi(T_t, \text{pos}) \cdot \sigma(f_\psi(t, T, L_{\text{recon}}))$ 与本框架高度一致，本框架进一步明确：
  - $g_\phi$ = $N \times N$ 矩阵（含选择）
  - $f_\psi$ 的输入为**原始 token**（而非压缩后 token），且输出为**标量**（而非逐 token）
  - 可选加入 $L_{\text{recon}}$ 反馈形成闭环（见 §7.4）

---

## 6. 与现有方法的统一关系

### 6.1 统一视角下的方法谱系

```
                    本框架（可学习 W + 可学习 predictor）
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        W 可学习          predictor       二者协同
        (本框架)          可学习           (本框架)
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         固定 W         固定 predictor    W/predictor
         (ToMe/         (SparseDiT        部分可学习
          SparseDiT)     线性调度)        (DiffSparse: 仅代价表;
                                          TokenCache: 仅 α;
                                          DyDiT: 仅 Router)
```

### 6.2 本框架相对各方法的增量

| 对比方法 | 本框架的增量 |
|---------|------------|
| vs ToMeSD | 压缩方式从「固定 attention merge」→「可学习矩阵」；压缩率从「固定常数」→「内容自适应标量」 |
| vs SparseDiT | 压缩方式从「固定 pooling/SDTM」→「可学习矩阵」；压缩率从「线性函数 of $t$」→「内容自适应（隐含时间步）」 |
| vs DiffSparse | 压缩方式从「不可学习（沿用 selector）」→「可学习矩阵」；压缩率从「代价表（非实例自适应）」→「DNN（实例自适应）」 |
| vs TokenCache | 从「时间维度缓存复用（$N \to N$）」→「空间维度真正降维（$N \to M$）」 |
| vs DyDiT | 从「token 级跳过（对角掩码）」→「token 级混合降维（非对角矩阵）」 |
| vs BudCache | 从「时间维度步级缓存」→「空间维度 token 压缩」（正交可叠加） |

### 6.3 正交可叠加的方法

本框架与以下方法正交，可叠加使用：

| 叠加方法 | 叠加方式 | 预期收益 |
|---------|---------|---------|
| BudCache / ERTACache | 本框架决定每步 token 数，BudCache 决定哪些步跳过 | 空间 + 时间双重压缩 |
| RT-Lynx | 本框架压缩 token 数，RT-Lynx 压缩通道激活 | Token + 通道双重压缩 |
| 量化（W8A8） | 本框架减少 token，量化减少每 token 计算精度开销 | Token + 精度双重压缩 |
| 蒸馏（少步模型） | 本框架在蒸馏模型上仍可应用（DiffSparse 已验证） | 步数 + token 双重压缩 |

---

## 7. 设计选择与讨论

### 7.1 $W$ 应该是内容相关还是内容无关？

| 方案 | $W$ 依赖 | 优点 | 缺点 |
|------|---------|------|------|
| **内容无关**（固定参数） | 仅 $\phi$ | 推理快（预计算 $W$）；跨实例一致 | 无法适配不同输入 |
| **内容相关**（生成式） | $T$ | 实例自适应；理论表达力强 | 推理需生成 $W$，$O(N^2)$ 开销 |
| **混合**（低秩 + 内容） | $\phi + T$ | 平衡效率与自适应 | 设计复杂 |

**建议**：起步用内容无关的低秩分解（策略 A），验证框架有效性后再探索内容相关（策略 B）。

### 7.2 Predictor 输入为什么是压缩前的 token？

三个理由：
1. **信息完整性**：压缩前 token 保留全部信息，能准确评估冗余度；压缩后 token 已丢失信息，评估「还能压多少」不可靠。
2. **解耦**：predictor 不依赖 compressor 的具体实现，compressor 可替换/升级而 predictor 无需重训（参考 ToMeSD 文档 §22.3 的「条件 Scheduler」策略）。
3. **闭环基础**：predictor 评估原始输入，compressor 执行压缩，二者形成 Actor-Critic 式协作（predictor = Critic 评估难度，compressor = Actor 执行压缩）。

### 7.3 标量 vs 逐 token 向量

本框架选择标量 $s \in [0,1]$ 而非逐 token 向量 $\mathbf{s} \in [0,1]^N$：

| 维度 | 标量 $s$ | 逐 token 向量 $\mathbf{s}$ |
|------|---------|--------------------------|
| 语义 | 全局保留比例 | 逐 token 重要性 |
| 选择 | Top-M（需配合打分） | 直接 threshold |
| 与现有方法对齐 | SparseDiT/DiffSparse（每层一个率） | TokenCache/DyDiT（逐 token） |
| 压缩类型 | 真正降维 $N \to M$ | 跳过/缓存 $N \to N$ |

**扩展**：若需更细粒度，predictor 可输出 $s$ 决定 $M$，再由 compressor 的 $W$ 行范数决定「哪 $M$ 个」。这样 predictor 管「量」，compressor 管「质」，职责清晰。

### 7.4 闭环反馈（可选增强）

借鉴 ToMeSD 文档 §21.2 的建议，可将 compressor 的重构损失反馈给 predictor：

$$s(t) = \sigma\Big(f_\psi\big(T_t, \underbrace{L_{\text{recon}}(t)}_{\text{压缩器反馈}}\big)\Big)$$

其中 $L_{\text{recon}}(t) = \|T_t - W_\uparrow \cdot \text{Select}(W_\downarrow \cdot T_t, M)\|^2$。

**反馈逻辑**：$L_{\text{recon}}$ 高 → 压缩损失大 → predictor 降低 $s$（保留更多 token）；$L_{\text{recon}}$ 低 → 压缩安全 → predictor 提高 $s$（压缩更多）。

这形成**自适应闭环**：简单图像自动多压、复杂图像自动少压。风险是闭环可能导致振荡，需用 EMA 平滑或设 $s \in [s_{\min}, s_{\max}]$ 上下界。

### 7.5 $N \times N$ 矩阵的合理性讨论

**Q：为什么是 $N \times N$ 而非 $M \times N$？**

用户明确设计为 $[N, d] \to [N, d]$（混合）再选 $M$，而非直接 $[N, d] \to [M, d]$（降维）。理由：

1. **解耦混合与选择**：$N \times N$ 矩阵只管「混合」（学习 token 间关系），选择 $M$ 由 predictor 决定。若用 $M \times N$，则矩阵形状依赖 $M$，而 $M$ 每步变化，无法用固定参数。
2. **支持可变 $M$**：同一 $W$ 可配合不同 $M$（不同步、不同实例），只需改变选择数量。
3. **统一视角**：ToMeSD 文档 §19 的矩阵分析显示，Poolingformer 也是 $N \times N$（秩-1 均匀），DyDiT-SDT 也是 $N \times N$（对角掩码）。$N \times N$ 是更一般的形式，$M \times N$ 是其子矩阵。

**代价**：$N \times N$ 矩阵的计算量 $O(N^2 d)$ 与 self-attention 同阶，可能抵消压缩收益。**必须配合低秩/块对角/稀疏参数化**（§4.1.3）才能实际加速。

### 7.6 多层应用策略

是否每层都应用 compressor + predictor？

| 策略 | 描述 | 参考 |
|------|------|------|
| 仅底层 | 只在 token 最多的层压缩 | ToMeSD（min_tokens=4096） |
| 三段式 | 底层压缩、中层交替、顶层全计算 | SparseDiT |
| 每层独立 | 每层独立 predictor | TokenCache、DiffSparse |
| 共享 predictor | 所有层共享一个 predictor，输出 per-layer $s$ | 轻量方案 |

**建议**：参考 SparseDiT 的三段式 + DiffSparse 的每层独立分配。每层可有自己的 $W^{(l)}$ 和 $s^{(l)}$，或共享 predictor 输出 $L$ 个 $s$。

---

## 8. 训练策略

### 8.1 训练目标

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{diffusion}}}_{\text{生成质量}} + \lambda_1 \underbrace{\mathcal{L}_{\text{FLOPs}}}_{\text{计算预算}} + \lambda_2 \underbrace{\mathcal{L}_{\text{recon}}}_{\text{压缩质量}}$$

- $\mathcal{L}_{\text{diffusion}}$：扩散损失（或 LPIPS 蒸馏损失，参考 DiffSparse）
- $\mathcal{L}_{\text{FLOPs}} = (\frac{1}{T}\sum_t s(t) - \lambda)^2$：FLOPs 约束（参考 DyDiT）
- $\mathcal{L}_{\text{recon}} = \|T - W_\uparrow W_\downarrow T\|^2$：压缩-恢复一致性（可选）

### 8.2 训练方式

参考 ToMeSD 文档 §22 的三种策略：

**策略 1：联合训练（完全耦合）**
- 同时优化 $\phi$（compressor）和 $\psi$（predictor）
- 优点：理论最优
- 缺点：训练不稳定，梯度耦合

**策略 2：交替训练（半解耦，推荐）**
```
Phase 1: 固定 W_φ，训练 predictor ψ（学「压多少」）
Phase 2: 固定 ψ，训练 compressor φ（学「怎么压」）
重复直至收敛
```
- 优点：每个子问题更稳定（类似 EM 算法）
- 缺点：可能不收敛到全局最优

**策略 3：条件 Predictor（解耦，最灵活）**
- Predictor 不直接学习压缩策略，而是学习「根据 compressor 状态调整策略」
- Compressor 可替换，predictor 通用
- 类比 Actor-Critic：Compressor = Actor，Predictor = Critic

**推荐**：策略 2（交替训练）起步，验证后尝试策略 1。

### 8.3 训练数据

参考 DiffSparse：仅用 10,000 个 captions/class indices（不用图像），4-10 GPU·h。本框架的 predictor 和 compressor 参数量同样很小，预期训练成本相近。

### 8.4 模型权重处理

两种选择：
- **冻结 DiT 权重**（参考 DiffSparse）：只训练 $W$ 和 predictor，student = 原模型 + 压缩调度
- **微调 DiT 权重**（参考 SparseDiT、DyDiT）：联合微调，质量更高但成本更大

**建议**：先冻结权重验证框架，再视效果决定是否微调。

### 8.5 离散化梯度

$M = \text{round}(s \cdot N)$ 和 Top-M 选择都不可微，需用：
- **STE**（Straight-Through Estimator）：前向离散，反向恒等（DiffSparse 用此法）
- **Gumbel-Sigmoid**：可微采样（DyDiT 用此法）
- **Soft Top-K**：用可微排序近似 Top-M

---

## 9. 开放问题与未来方向

### 9.1 跨分辨率泛化

$N \times N$ 矩阵绑定 $N$，如何跨分辨率迁移？参考 DiffSparse 的「代价表与 $N$ 解耦」：
- 用策略 D（固定基 + 可学习系数），$W = \sum_j \alpha_j B_j$，与 $N$ 无关
- 或用内容生成式 $W = g(T)$，天然适配任意 $N$

### 9.2 与 KV Cache 的交互

推理时若使用 KV cache，被压缩掉的 token 的 K/V 如何处理？
- SparseDiT 用 cross-attention 恢复稠密表示
- 本框架需设计 $W_\uparrow$ 恢复机制，或让被选 token 的 K/V 代表未选 token

### 9.3 Predictor 的实例自适应 vs 延迟确定性

BudCache 选择固定 mask 追求延迟确定；本框架的 predictor 是实例自适应的，延迟会波动。权衡：
- 设 $s \in [s_{\min}, s_{\max}]$ 限制波动范围
- 或预计算若干档位 $s$，推理时按 predictor 输出量化到最近档位

### 9.4 从标量到结构化稀疏

当前 predictor 输出全局标量 $s$。可扩展为：
- **逐层标量** $s^{(l)}$：每层不同压缩率（参考 DiffSparse 的 per-layer 分配）
- **逐 token 分数** $\mathbf{s} \in [0,1]^N$：直接做 soft 选择（参考 TokenCache）
- **结构化 mask**：按空间区域分配不同 $s$（参考 SparseDiT 的空间三段式）

### 9.5 与 step-level caching 的联合优化

本框架（空间压缩）与 BudCache/ERTACache（时间缓存）正交。可设计**时空联合优化**：
- 每步的 $s(t)$ 不仅决定 token 压缩率，还影响是否触发缓存复用
- 用动态规划联合优化「哪些步缓存」+「每步保留多少 token」

### 9.6 理论分析

- **压缩-质量权衡的理论边界**：给定 $M$，最优 $W$ 的形式是什么？（信息论角度）
- **predictor 的最优输入**：原始 token 的哪些统计量最能预测最优 $s$？
- **与 attention 稀疏化的等价关系**：$W \cdot T$ 后选 Top-M，是否等价于某种 attention mask？

### 9.7 实验验证路线

1. **基线复现**：在 DiT-XL/2 ImageNet 256×256 上复现 ToMeSD/SparseDiT 基线
2. **消融实验**：
   - 仅 compressor（固定 $s$）vs 仅 predictor（固定 $W$）vs 联合
   - $W$ 的参数化策略 A/B/C/D 对比
   - predictor 的方案 1/2/3/4 对比
3. **对比实验**：与 ToMeSD、SparseDiT、DiffSparse、TokenCache 在同等 FLOPs 下比 FID
4. **跨任务验证**：图像（DiT）、文生图（PixArt）、视频（Latte/Wan2.1）
5. **叠加实验**：与 BudCache、RT-Lynx、量化叠加验证正交性

---

## 附录：关键符号表

| 符号 | 含义 |
|------|------|
| $T \in \mathbb{R}^{N \times d}$ | 原始 token（$N$ 个，$d$ 维） |
| $W \in \mathbb{R}^{N \times N}$ | 可学习压缩矩阵（参数 $\phi$） |
| $T' = W \cdot T$ | 混合后 token，$\in \mathbb{R}^{N \times d}$ |
| $f_\psi: \mathbb{R}^{N \times d} \to \mathbb{R}$ | Sparsity predictor（参数 $\psi$） |
| $s = \sigma(f_\psi(T)) \in [0,1]$ | 稀疏比例标量 |
| $M = \text{round}(s \cdot N)$ | 保留 token 数 |
| $T_{\text{out}} \in \mathbb{R}^{M \times d}$ | 压缩后输出 token |
| $W_\uparrow \in \mathbb{R}^{N \times M}$ | 恢复矩阵（可选） |
| $t$ | 去噪时间步 |
| $T$ | 总去噪步数（注意与 token $T$ 区分，上下文明确） |

---

## 参考文档索引

| 方法 | 总结文档 |
|------|---------|
| ToMeSD | [summary_tome_token_merging_diffusion.md](file:///d:/WorkSpace/Awesome-Deep-Neural-Network-Compression/Summary/Diffusion%20Models/summary_tome_token_merging_diffusion.md)（含 Part III 统一框架） |
| SparseDiT | [summary_sparse_dit_token_sparsification.md](file:///d:/WorkSpace/Awesome-Deep-Neural-Network-Compression/Summary/Diffusion%20Models/summary_sparse_dit_token_sparsification.md) |
| DiffSparse | [summary_diffsparse_learned_token_sparsity.md](file:///d:/WorkSpace/Awesome-Deep-Neural-Network-Compression/Summary/Diffusion%20Models/summary_diffsparse_learned_token_sparsity.md) |
| TokenCache | [summary_tokencache_token_caching_dit.md](file:///d:/WorkSpace/Awesome-Deep-Neural-Network-Compression/Summary/Diffusion%20Models/summary_tokencache_token_caching_dit.md) |
| DyDiT | [summary_dydt_dynamic_diffusion_transformer.md](file:///d:/WorkSpace/Awesome-Deep-Neural-Network-Compression/Summary/Diffusion%20Models/summary_dydt_dynamic_diffusion_transformer.md) |
| BudCache | [summary_budcache_step_level_caching.md](file:///d:/WorkSpace/Awesome-Deep-Neural-Network-Compression/Summary/Diffusion%20Models/summary_budcache_step_level_caching.md) |
| ERTACache | [summary_ertacache.md](file:///d:/WorkSpace/Awesome-Deep-Neural-Network-Compression/Summary/Diffusion%20Models/summary_ertacache.md) |
| RT-Lynx | [summary_rt_lynx_activation_sparsity.md](file:///d:/WorkSpace/Awesome-Deep-Neural-Network-Compression/Summary/Diffusion%20Models/summary_rt_lynx_activation_sparsity.md) |
