# VSA: Video Sparse Attention — 可训练的视频扩散稀疏注意力

> **Paper**: [arXiv:2505.13389](https://arxiv.org/abs/2505.13389)  
> **Authors**: Peiyuan Zhang, Yongqi Chen, Haofeng Huang, Will Lin, Zhengzhong Liu, Ion Stoica, Eric P. Xing, Hao Zhang (UCSD / MBZUAI / UC Berkeley)  
> **Venue**: NeurIPS 2025  
> **Code**: https://github.com/hao-ai-lab/FastVideo  
> **Date**: 2026-08-03

---

## 🧠 读者综合理解

> **读者理解**：本文先对 VAE encode 结果做分块，每个分块包含若干个 token，取他们的平均数作为块的表达。然后算 attention，只取 top-k 的块出来，里面的 token 才进行真实计算。

**点评：基本正确，抓住了 VSA 的核心流程，但有两点需要补充和修正。**

**正确部分：**
- ✅ "VAE encode 结果做分块"——准确。VSA 在 $(T,H,W)$ 网格上划分 $(4,4,4)$ 立方体
- ✅ "每个分块包含若干个 token，取平均数作为块的表达"——准确。每个立方体 64 个 token，mean pooling 得到 cube-level 表示
- ✅ "算 attention，只取 top-k 的块出来"——准确。Coarse stage 在立方体级计算注意力，选 Top-K 个关键立方体
- ✅ "里面的 token 才进行真实计算"——准确。Fine stage 只在选中的立方体内部做 token 级稀疏注意力

**需要补充/修正的两点：**

1. **Fine stage 不是"真实计算"，而是"块内稀疏计算"**：VSA 的 Fine stage 仍然是稀疏注意力，不是全注意力。它在选中的立方体内部计算 token 级注意力，但立方体之间的计算被跳过了。

2. **最终还有门控融合步骤**：VSA 的输出不是只有 Fine stage 的结果，而是：
   $$O = G_c \odot \text{broadcast}(O_c) + G_f \odot O_f$$
   其中 $O_c$ 是 Coarse stage 的输出（立方体级，提供全局信息），$O_f$ 是 Fine stage 的输出（token 级，提供局部精细信息）。门控 $G_c, G_f$ 由模型学习，平衡两个阶段的贡献。

---

## 核心问题与动机

视频扩散 Transformer（DiT）的 3D 全注意力是主要瓶颈——即使 5 秒 720p 视频也有 100K+ tokens，全注意力的 $\mathcal{O}(N^2)$ 复杂度导致训练和推理都极为缓慢。

**现有方法的局限：**
- 多数后处理方法（如 STA、Sparge）在**训练后**才引入稀疏模式，存在 train-test mismatch
- 固定模式方法（如滑动窗口）无法适应不同层/头/时间步的注意力分布变化
- 缺少与 GPU block-sparse 布局对齐的实现，理论加速难以转化为实际 wall-clock 加速

**核心洞察：** 视频 DiT 的注意力矩阵天然稀疏——大部分权重集中在少量"critical tokens"上。关键是如何在不计算全注意力矩阵的前提下高效识别这些 critical tokens。

---

## VSA 方法架构

### 两阶段层次化设计

```
输入: 视频潜空间 Z ∈ ℝ^{T×H×W×d}
  │
  ▼ 划分立方体 (C_t, C_h, C_w) = (4, 4, 4)，tile size B=64
  │
  ┌─────────────────────────────────────────────────┐
  │  Coarse Stage (粗粒度阶段)                        │
  │  1. 每个立方体做 mean pooling → 立方体级表示          │
  │  2. 计算立方体间全注意力 → 轻量级全局信息建模        │
  │  3. 每行 Top-K 选择 → 预测 critical 立方体位置       │
  │  4. 输出 O_c（粗粒度注意力结果）                     │
  └────────────┬────────────────────────────────────┘
               │ Top-K 选择的 block indices
               ▼
  ┌─────────────────────────────────────────────────┐
  │  Fine Stage (细粒度阶段)                          │
  │  1. 仅在选中的 K 个立方体内部做 token 级注意力       │
  │  2. block-sparse kernel 实现 → 硬件高效             │
  │  3. 输出 O_f（细粒度注意力结果）                     │
  └────────────┬────────────────────────────────────┘
               │
               ▼
  ┌─────────────────────────────────────────────────┐
  │  门控融合: O = O_c ⊙ G_c + O_f ⊙ G_f              │
  │  G_c, G_f 从输入 hidden states 线性投影得到          │
  └─────────────────────────────────────────────────┘
```

### 关键设计选择

| 参数 | 选择 | 原因 |
|------|------|------|
| **Tile size B** | 64 (对应 $(4,4,4)$ 立方体) | 平衡表达力和硬件效率：更小 tile 精度更高但吞吐低；64×64 tile 达 85% FA3 MFU |
| **Top-K** | 32 | 在 16K-25K 序列长度下 consistently 最优 |
| **池化方式** | Mean Pooling | 优于 Max Pooling 和 Conv Pooling；Conv 导致训练不稳定 |
| **稀疏率** | 87.5% (K=32 / 256 cubes) | 每 256 个立方体选 32 个，对应 8× FLOPS 减少 |
| **Local 模块** | 不需要 | 消融证明显式局部建模（如 $(3,3,3)$ 窗口）无额外收益 |

### 硬件高效实现

1. **Block-sparse kernel**：基于 ThunderKittens 实现 fine stage，利用 Hopper GPU 的 TMA 和 Warp Specialization，达 **85% FA3 MFU**
2. **Coarse stage 融合**：将 softmax、Top-K 选择、mask-to-index 转换融合为单一 kernel，开销仅占总注意力的 <1%
3. **Tile 映射**：每个 $(4,4,4)$ 立方体映射为 GPU SM 上的一个 tile，天然对齐 block-sparse 计算布局

### 与 Distillation 的集成（Sparse-Distill）

VSA 是**首个与蒸馏兼容的稀疏注意力方法**：
- 学生模型使用 VSA + 少步采样
- 教师模型保持全注意力
- 保持原始 DMD2 蒸馏损失和超参数不变
- Wan-1.3B 实现 **50.9× 加速** 且质量无下降

---

## 实验结果

### 预训练：从 60M 到 1.4B 参数

| 指标 | Full Attention | VSA | 改进 |
|------|---------------|-----|------|
| Training FLOPS | baseline | **2.53× 减少** | 相同 loss 下 |
| Attention FLOPS | baseline | **8× 减少** | 87.5% 稀疏度 |
| Loss (120M, 4.5×10²⁰ FLOPS) | 0.13877 | **0.13162** | 更低 |
| Loss (120M, 4×10²¹ FLOPS) | 0.12703 | **0.12687** | 更低 |

- VSA 在所有规模上都优于 full attention，形成更好的 Pareto frontier
- 最优 Top-K 依赖序列长度和训练预算：长序列需要更大 K，更多训练预算也支持更大 K

### 后训练：适配预训练模型

**Wan2.1-1.3B (480P, 81帧)：**

| 指标 | Full Attention | VSA |
|------|---------------|-----|
| DiT 推理时间 | 31s | **18s** (1.7×) |
| Attention 加速 | — | **6×** |
| VBench Total | 82.56% | 82.77% |
| VBench Quality | 83.71% | 83.60% |
| VBench Semantic | 77.98% | 79.47% |

- VBench 分数与 full attention 相当，部分指标略优
- 注意力仅占总运行时间的 20%（原来占 92%）

**Wan2.1-14B (720P)：**
- 人类评估：VSA 与 full attention 质量相当
- 推理时间从 1274s 降至 **576s**

### Sparse-Distill 结果

| 指标 | Full Attention + DMD2 | VSA + DMD2 |
|------|----------------------|-----------|
| 加速比 | baseline | **50.9×** |
| 视频生成时间 | — | ~5s (5s 视频) |
| 质量 | — | 无下降 |

### Kernel Benchmark

| 方法 | 相对 FA3 加速 | MFU |
|------|-------------|-----|
| FlexAttention (相同 mask) | 2× | — |
| **VSA (fine stage)** | **7×** | **85%** |
| **VSA (端到端)** | **6×** | — |

### Critical Token 预测精度

- 随机选择 32/386 个立方体仅覆盖 **8%** 注意力分数
- VSA 预测覆盖 **60-90%** 注意力分数
- 精度随扩散时间步单调递增；跨层级呈 zig-zag 模式

---

## 与相关工作的对比

| 方法 | 类型 | 训练 | 硬件效率 | 视频适配 |
|------|------|------|---------|---------|
| **STA** | 固定模式 | 后处理 | 高 | 有 |
| **Sparge** | 固定模式 | 后处理 | 中 | 有 |
| **SViDeo** | 低秩预测器 | 多阶段 | 中 | 有 |
| **MoBA** | 可训练 | 端到端 | 中 | 无（LLM 设计） |
| **NSA** | 可训练 | 端到端 | 中 | 无（LLM 设计） |
| **VSA** | **可训练 + 层次化** | **端到端** | **高 (85% MFU)** | **有** |

**VSA 的独特性：**
1. **首个在视频 DiT 上验证可训练稀疏注意力**的大规模研究
2. **端到端可训练**，无需后处理 profile 或多阶段训练
3. **硬件对齐**：cube-to-tile 映射确保 block-sparse 布局
4. **与蒸馏兼容**：第一个展示稀疏注意力 + 蒸馏联合工作的方法

---

## USV 中 VSA 的使用方式

在 USV 框架中，VSA 被用作**基础的稀疏注意力计算内核**：

1. **Token Merge 先于 VSA**：先将冗余 token 合并减少数量（N → N-r）
2. **VSA 在合并后的 token 上运行**：通过二值掩码 M^(l) 进一步稀疏化注意力
3. **熵感知策略统一调度**：同时控制 VSA 的注意力稀疏率 ρ^attn 和 token merge 的合并率 ρ^token
4. **与 DMD2 蒸馏结合**：学生模型使用 VSA + token merging + 少步采样

USV 在 VSA 基础上的创新：
- 引入 token merging 减少 token 数量（VSA 本身不减少 token 数量）
- 引入熵感知动态策略，替换 VSA 原有的固定稀疏模式
- 将 VSA 与 token merging 协同设计，获得超加性收益

---

## 局限与展望

### VSA 本身的局限

1. **固定 cube 尺寸 $(4,4,4)$**：要求视频潜空间各维度是 4 的倍数，限制了可生成的分辨率
2. **最优 Top-K 选择**：依赖序列长度和训练预算，缺乏理论预测
3. **长序列扩展**：16K 序列长度已验证，更长序列（>100K）待探索
4. **自适应 Top-K**：不同层/头/时间步使用不同 K 值是未来方向
5. **与 KV Cache 结合**：VSA 可与 TeaCache 等缓存方法结合进一步加速

### 在 USV 框架中的局限

VSA 在 USV 中作为唯一的稀疏注意力实现，存在以下局限：

1. **未验证与其他稀疏注意力方法的组合**：USV 的 token merge 仅与 VSA 组合测试，未探索与 VMoBA、SVOO、SALAD 等方法的组合
2. **协同效应来源不明**：无法确定"token merge + 稀疏注意力"的协同效应是否为 VSA 特有，还是任何可训练稀疏注意力都能获得
3. **固定稀疏率分配**：VSA 原有的固定稀疏率选择被 USV 的熵感知策略替代，但两者的交互（如 coarse stage 的 Top-K 是否应随熵调整）未被研究

> 关于 USV 中 token merge + VSA 协同的详细分析，参见 [USV 论文摘要的扩展分析部分](file:///d:/WorkSpace/Awesome-Deep-Neural-Network-Compression/Summary/Diffusion%20Models/summary_usv_unified_sparsification_video.md)

---

## 💬 深度问答 (Q&A)

### Q1: 潜空间是 4 维的 (T,H,W,C)，为什么用 3 维划分立方体？每个立方体包含多少 token？Vision patch 如何形成 token？

**数据流全解析：**

```
原始视频像素: T_raw × H_raw × W_raw × 3
  │
  ▼  VAE 编码器
  │  空间 8× 压缩, 时间 4× 压缩
  │
  ▼  VAE 潜空间: T × H × W × C     (例如: 21 × 60 × 106 × 16)
  │  ↑ 4 维: (T, H, W, C)
  │  T=时间帧数, H=空间高, W=空间宽, C=通道数(固定)
  │
  ▼  展平为 token 序列
  │  L = T × H × W 个 tokens      (例如: 131K)
  │  每个 token 对应 (T,H,W) 网格上的一个位置 (t,h,w)
  │  每个 token 有 C 维特征
  │
  ▼  Attention 计算
     Q, K, V ∈ ℝ^{L × d}
     每个 token 被线性投影为 d 维的 query/key/value
```

**为什么立方体划分在 (T,H,W) 3 维上？**
- 立方体划分的目的是在**时空邻域**内做稀疏选择
- 每个立方体对应视频中一个局部时空 tubelet（一小段时间内的一块空间区域）
- C 是通道/特征维度，不参与立方体划分——它是每个 token 内部的特征，用于 attention 计算
- 类比：一张 RGB 图像的像素是 (H,W,3)，但划分 patch 时只在 (H,W) 上划分，通道维度保留

**立方体大小：**
- 默认 $(C_t, C_h, C_w) = (4, 4, 4)$
- 每个立方体包含 $B = 4 \times 4 \times 4 = 64$ 个 token
- Token 索引映射：$n = (\lfloor t/4 \rfloor \times N_h N_w + \lfloor h/4 \rfloor \times N_w + \lfloor w/4 \rfloor) \times 64 + (t \mod 4) \times 16 + (h \mod 4) \times 4 + (w \mod 4)$
- 这个映射保证每个 cube 内的 64 个 token 在 1D 序列中连续，对应 GPU 上的一个 tile

**回答：**
1. 4 维 → 3 维的原因：C 是通道维，不参与立方体划分；立方体划分在时空 (T,H,W) 上进行
2. 每个立方体 64 个 token（默认参数）
3. Vision patch → token：VAE 编码器将像素压缩为 $(T,H,W,C)$ 潜空间，每个 $(t,h,w)$ 位置就是一个 token，有 C 维特征

---

### Q2: Coarse 阶段是不是找到注意力最集中的 Top-K 立方体？

**是的，这正是 Coarse 阶段的核心功能。** 完整流程如下：

```
输入: L 个 token（属于 N_cubes 个立方体，每个立方体 B=64 个 token）

Step 1: Mean Pooling (立方体级)
  每个 (4,4,4) 立方体的 64 个 token → 取均值 → 1 个 cube-level 表示
  结果: Q_c, K_c, V_c ∈ ℝ^{N_cubes × d}   (例如: 2048 × d)

Step 2: 立方体间全注意力
  A_c = Softmax(Q_c × K_c^T / √d) ∈ ℝ^{N_cubes × N_cubes}
  即 2048 × 2048 的注意力矩阵（计算量远小于 token 级）

Step 3: 每行 Top-K 选择 (K=32)
  对 A_c 的每一行，选 32 个最大 attention weight 对应的立方体
  生成稀疏 mask: 标记哪些立方体是 "critical" 的

Step 4: 广播到 token 级
  每个选中的立方体 → B×B = 64×64 个 Q-K 对在 Fine Stage 中计算
  每个未选中的立方体 → 完全跳过，不做任何 token 级计算

结果: 约 87.5% 的立方体被跳过 (32/256 = 1/8 被选中)
```

**关键点：**
- Coarse 阶段计算的是 **"哪个立方体对其他立方体重要"**（通过 attention weight 衡量）
- 不是计算 "哪个 token 对其他 token 重要"——那是 Fine Stage 的工作
- 立方体级的决策是一种**粗粒度过滤**：先确定哪些区域重要，再在这些区域内做精细计算
- 这避免了在全 token 级别计算 attention 矩阵的 O(L²) 开销

---

### Q3: O_c 和 O_f 的维度？O_c 是块矩阵吗？

**O_c 和 O_f 都是值矩阵（加权和的结果），不是"token 数量的平方"。**

| 变量 | 维度 | 含义 |
|------|------|------|
| Q_c, K_c, V_c | (N_cubes) × d | Cube 级的 Q/K/V |
| A_c | (N_cubes) × (N_cubes) | Cube 间的注意力权重矩阵 |
| **O_c** | **(N_cubes) × d** | Cube 级的注意力输出 = A_c × V_c |
| Q, K, V | L × d | Token 级的 Q/K/V |
| M | L × L | Token 级的注意力 mask（由 A_c broadcast 而来） |
| **O_f** | **L × d** | Token 级的稀疏注意力输出（仅选中的立方体内部） |
| G_c, G_f | L × d | 门控向量（每个 token 有独立值） |
| **O** | **L × d** | 最终输出 = G_c ⊙ O_c + G_f ⊙ O_f |

**O_c 的处理：**
- O_c 维度是 (N_cubes) × d，不是 L × d
- 在最终融合时，O_c 需要 broadcast 到 token 级：每个 cube 的输出复制 B=64 次
- broadcast 后的 O_c 变成 L × d，才能与 O_f (L × d) 逐元素相加

**Mask M 的块稀疏结构：**
- M ∈ ℝ^{L × L} 是 token 级的 mask
- 当 A_c 中 (i,j) 被选中时，M 中对应的区域是一个 B×B 的稠密块（64×64）
- 所以 M 确实是块稀疏的——这保证了 Fine Stage 的计算能被 GPU 高效处理

**门控机制的作用：**
```
O = G_c ⊙ broadcast(O_c) + G_f ⊙ O_f
```
- G_c 初始化为 0（训练初期不使用 coarse stage）
- G_f 初始化为 1（训练初期完全依赖 fine stage）
- 训练过程中，模型学习调整两个门控值，平衡 coarse（全局信息）和 fine（局部精细）的贡献

---

### Q4: VSA 在 USV merge token 基础上再做分块？

**是的，在 USV 框架中，VSA 在 token merge 之后运行。** 完整数据流：

```
输入: Z ∈ ℝ^{N×d}  (N 个 token，来自 VAE 压缩的潜空间)
  │
  │  ┌─── Token Merge ───────────────────────────┐
  │  │  1. (T,H,W) 网格划分 3D 二部图块             │
  │  │  2. 计算描述子 → 贪心合并最相似的 token        │
  │  │  3. 输出: Z̃ ∈ ℝ^{(N-r)×d}                  │
  │  │     (合并后的 token 序列，数量减少)            │
  │  └────────────────────────────────────────────┘
  │
  ▼
  Z̃ ∈ ℝ^{(N-r)×d}  (merge 后的 token)
  │
  │  ┌─── VSA 稀疏注意力 ─────────────────────────┐
  │  │  1. 将 Z̃ 按 (4,4,4) 划分为立方体             │
  │  │     (注意：划分在 merge 后的 token 上)         │
  │  │  2. Coarse Stage:                            │
  │  │     mean pooling → 立方体注意力 → Top-K 选择  │
  │  │  3. Fine Stage:                              │
  │  │     仅在选中的立方体内部做 token 级注意力      │
  │  │  4. 门控融合: O = G_c ⊙ O_c + G_f ⊙ O_f      │
  │  │  5. 输出: Z̃' ∈ ℝ^{(N-r)×d}                 │
  │  └────────────────────────────────────────────┘
  │
  ▼
  Z̃' ∈ ℝ^{(N-r)×d}
  │
  │  ┌─── Unmerge ─────────────────────────────────┐
  │  │  1. 查映射表，恢复被合并的 token               │
  │  │  2. 输出: Ẑ ∈ ℝ^{N×d}                       │
  │  └────────────────────────────────────────────┘
  │
  ▼
  Ẑ ∈ ℝ^{N×d}  (稠密 token，继续下一层)
```

**注意事项：**
- 如果 merge 后的 token 数 $(N-r)$ 不是 64 的倍数，VSA 论文未说明如何处理——可能需要 padding
- VSA 的立方体划分逻辑在 USV 中保持不变，只是输入 token 数量减少了
- USV 的熵感知策略同时控制 VSA 的 K（Top-K 立方体数量）和 token merge 的合并率

**对比：独立 VSA vs USV 中的 VSA**

| 维度 | 独立 VSA | USV 中的 VSA |
|------|---------|-------------|
| 输入 token 数 | L（全部 token） | N-r（merge 后减少） |
| 立方体数量 | L/64 | (N-r)/64 |
| Top-K 选择 | 固定 K=32 | 由熵感知策略动态调整 |
| 稀疏率 | 固定 87.5% | 动态，随时间步和层变化 |
