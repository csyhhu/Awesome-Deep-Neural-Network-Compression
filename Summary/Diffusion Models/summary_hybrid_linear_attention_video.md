# Sana: 基于混合线性注意力与注意力残差的高效视频生成

> **论文**: Hybrid Linear Attention with Attention Residuals for Efficient Video Generation  
> **arXiv**: https://arxiv.org/abs/2607.21553  
> **机构**: NVIDIA  
> **作者**: Junsong Chen, Jincheng Yu, Yitong Li, Shuchen Xue, Haozhe Liu, Jingyu Xin, Yuyang Zhao, Tian Ye, Zhangjie Wu, Zian Wang, Daquan Zhou, Ping Luo, Song Han, Enze Xie

---

## 一、核心问题与动机

当前最先进的视频生成模型（如 Wan 2.1/2.2、HunyuanVideo、Seedance 2.0 等）普遍采用全 3D softmax 注意力机制，其计算复杂度为 O(N²)。对于高分辨率（如 1080p）长视频，压缩后的 latent token 数量可达数万级别，quadratic 开销成为瓶颈。

线性注意力将 token 交互压缩为固定大小状态矩阵 S ∈ R^(d×d)，复杂度降为 O(N)，但表达能力受限——固定秩的状态矩阵无法表示所有 token-token 交互，会削弱精确的时空对应关系和细节质量。

**核心问题**：能否在保持 O(N) 长序列扩展能力的同时，通过混合注意力设计恢复 softmax 注意力的表达能力？

---

## 二、核心方法

### 2.1 混合线性-Softmax 注意力 (Hybrid Linear-Softmax Attention)

采用 **3:1 比例**（75% 线性 + 25% softmax）：

- **线性层**：使用门控双线性线性注意力（bidirectional gated linear attention），构成主要的 O(N) token 混合
- **Softmax 锚点**：每 4 层插入一个 softmax 层作为"锚点"，周期性地恢复秩不受限的 token 交互
- 线性和 softmax 头使用不同的 head 维度，在效率与单头容量间权衡
- 两种路径保留独立的 RoPE 张量和门控参数化

**设计选择**：通过从头训练的小规模代理实验（28层、3072宽度），在 0%–100% softmax 比例间扫描，确定 **25% softmax 为质量-效率的 Pareto 拐点**。

### 2.2 块注意力残差 (Block Attention Residuals, AttnRes)

灵感来自 LLM 中的 Attention Residuals 机制（Kimi K3 等），适配到视频扩散场景：

- **块组织**：每 S=8 个连续 Transformer 层划分为一个块
- **深度路由**：路由器从以下来源集合中动态聚合特征：
  - 初始 token 嵌入 b₀
  - 已完成块的特征摘要 b₁, b₂, ...
  - 当前块的部分累积和 pₗ
- **共享路由查询**：所有深度共享同一套路由查询（每个分支一个），相比逐层设计节省 4× 内存
- **路由公式**：

  hₗ(x) = Σ α^(τ)_{i→l}(x) · vᵢ(x)
  
  α^(τ)_{i→l}(x) = softmax_i((w^(τ) + φ_τ(t))^T · RMSNorm(vᵢ(x)))

- **无时间步条件**：路由器不使用显式的 diffusion timestep 输入（验证发现其影响可忽略），时间信息通过 AdaLN 和条件特征隐式传递

### 2.3 从零训练 (From-Scratch Training)

与后处理线性化不同，本研究直接训练混合骨干网络：

- **数据管道**：6 阶段数据处理（镜头分割、黑白边/字幕清理、低质量过滤、多轴评分）
- **质量/运动漏斗**：预训练使用宽松阈值的广泛数据，持续训练提高质量和运动门槛，SFT 使用最可信子集
- **分辨率/时长课程**：480p → 720p 渐进训练
- **训练目标**：
  - Flow Matching + logit-normal 时间步采样
  - 内容感知流偏移（TQD）：高运动片段 → 高噪声，高质量片段 → 低噪声
  - 自适应 token-count-aware 流偏移（3→6，随 token 数线性插值）
  - 时间步分层验证（10 桶 macro MSE）
  - Self-Flow 蒸馏（预训练和持续训练阶段）
  - 权重 EMA
- **后训练**：
  - Diffusion-DPO：基于偏好的视觉对齐
  - 在线强化学习（ReFL）：HPSv3++ + DeQA-Score + UniPercept 三模型联合奖励

---

## 三、实验结果

### 3.1 主要性能

| 指标 | Sana 5B | Sana 14B | Bernini-R 14B | Wan 2.2-A14B | Cosmos-3 Nano |
|------|---------|----------|----------------|---------------|---------------|
| VBench Total | **84.30** | ~84.5 | 84.64 | ~83.5 | ~83.0 |
| VBench Quality | **85.61** | - | 85.13 | - | - |
| 延迟 (单 H100, 480p×832×81) | **13.2s** | 29.1s | 421s | ~650s | ~45s |

- 在 720p/60s 形状下，DiT 前向传播比全 softmax 基线快 **3.2×**
- 延迟优势随视频时长增大：从 5s 到 60s，速度比从 1.40× 扩大到 2.73×

### 3.2 消融分析

**Softmax 比例消融**（28 层代理模型）：
- 0%（纯线性）：val loss 0.955
- 14.3%（4/28 锚点）：val loss 0.914
- 25%（3:1）：val loss 0.905
- 50%：val loss 0.897（但延迟增加 1.29×）
- 100%（全 softmax）：val loss 0.945
- **结论**：25% softmax 为实际 Pareto 拐点

**AttnRes 消融**：
- 启用 AttnRes 使深层状态有效秩提升 **~12%**
- 共享路由查询在 loss 相当的情况下节省 4× 内存
- 块大小 S∈{4,8,16} 性能相近，S=8 为工程默认值
- 路由在深度上重用已完成块的 50%+ 特征（最近完成块占比最高）

### 3.3 效率分析

**分辨率/时长扩展**：
- 从 480p 到 1080p，Full/Hybrid 速度比从 1.16× 增至 2.01×
- 在 720p/60s（1441 帧）下达到 3.17× 加速

**模型规模扩展**：
- 从 1.2B 到 28.9B 参数，混合方案在所有尺度均更快
- 绝对时间节省从 128ms 增至 948ms

**硬件扩展**：
- H100 → GB200：14B 骨干加速 1.98×
- GB200 上 Full/Hybrid 速度比从 1.58× 扩大到 3.07×

---

## 四、部署与应用

### 4.1 Sol-Engine 全栈优化

在 NVIDIA B200 上实现 3.58× 端到端加速：
1. Kernel 融合 + 执行优化：62.65s → 30.74s
2. 残差复用（50 步中计算 33 步）：30.74s → 20.89s
3. Softmax 锚点稀疏注意力：20.89s → 17.52s

最终 5B 管线在 720p/5s 下达到 **13.06s**，比 Wan 2.2-A14B 快 **120×**。

### 4.2 低精度推理 (QAT)

- 采用 MXFP4 权重 + MXFP8 激活量化
- VBench Total 与 BF16 基线持平
- 模型存储从 8.94GB 降至 2.87GB（-67.9%）
- 峰值内存从 10.74GB 降至 4.63GB（-56.9%）

### 4.3 Physical AI 应用

在约 5000 小时真实机器人和自我中心视频上微调 100k 步：
- 生成逼真且物理合理的机器人操作视频
- 优于同规模的 Cosmos3-Edge (4B)
- 与 Cosmos3-Nano (16B) 和 Lingbot-Video (30B) 竞争力相当

---

## 五、核心贡献总结

1. **混合注意力架构**：提出 75% 线性 + 25% softmax 的混合视频扩散 Transformer，在质量上匹配全 softmax 模型，同时保留线性注意力的 O(N) 扩展能力

2. **Block Attention Residuals**：适配到视频扩散的跨深度路由机制，有效秩提升 ~12%，实现锚点特征的跨深度重用

3. **从头训练方法**：完整的多阶段训练管线（数据策展、分辨率/时长课程、结构化标注、自蒸馏、DPO/ReFL 后训练），使混合架构直接学习而不依赖预训练模型的后处理线性化

4. **高效部署**：与 Sol-Engine 优化栈无缝组合，实现单 GPU 上的 720p 高分辨率视频生成，延迟仅为大模型的 1/100

5. **实用开源**：5B 模型可在消费级 GPU 上运行，为研究者和日常用户提供高效的视频生成基础

---

## 六、与 SANA-WM/SANA-Streaming 的关系

本文的 SANA 是 SANA 系列的延续：
- **SANA（本文）**：混合线性-Softmax 视频扩散 Transformer，面向通用视频生成
- **SANA-WM**：基于高效骨干的世界模型
- **SANA-Streaming**：面向流式视频编辑的扩展

相比前代纯线性 SANA，本文通过引入 25% softmax 锚点和 AttnRes 机制，在保持 O(N) 扩展的同时显著提升了生成质量。

---

## 七、未来方向

1. **长视频扩展**：将课程从当前 8s 延长到分钟级训练
2. **因果生成**：将双向算子改为因果 Gated DeltaNet，应用于机器人/自动驾驶/世界模型的流式推理
3. **锚点特定稀疏内核**
4. **低精度格式**（NVFP4 等）
5. **少步蒸馏**：补充训练无关的部署缓存
6. **内容自适应锚点放置和 AttnRes 路由**

---

## 八、深度问答

### Q1: Bidirectional Gated Linear Attention 的具体公式与维度

#### 公式

对于单个注意力头，给定归一化的查询 qⱼ 和键 kₙ（经 RoPE 旋转后分别为 qⱼʳ 和 kₙʳ），线性注意力分为两步：

**Step 1 — 状态写入（所有 token 依次处理）：**

$$S = \sum_{n=1}^{N} v_n \cdot (\beta_n \odot k_n^r)^\top$$

**Step 2 — 状态读取（对每个 query token 计算输出）：**

$$o_j = W_o \Bigg[ \operatorname{RMSNorm}\Bigg( \frac{S \cdot q_j^r}{\sum_{n=1}^{N} \phi(k_n)^\top \phi(q_j) + \epsilon} \Bigg) \odot \sigma(g_j) \Bigg]$$

其中 φ(·) = ReLU(·)。

#### 核心元素维度（以 5B 模型为例）

| 符号 | 含义 | 维度 | 说明 |
|------|------|------|------|
| N | 序列长度（token 数） | 标量 | 如 480p×832×81 约 22K latent token |
| dₕ | 单头维度 | 标量 = 128 | 线性注意力头的 dₕ，softmax 头的 dₕ=256 |
| qⱼʳ, kₙʳ | RoPE 旋转后的 query/key | (dₕ,) | 每个 head 独立 |
| βₙ | 写入门控（write gate） | (dₕ,) 或标量 | 控制每个 token 写入状态的强度 |
| vₙ | value 向量 | (dₕ,) | 每个 token 的 value |
| **S** | **压缩状态矩阵** | **(dₕ, dₕ) = (128, 128)** | 固定大小，与 N 无关——这是 O(N) 的关键 |
| S·qⱼʳ | 状态读取 | (dₕ,) | 用 query 从固定状态读取信息 |
| φ(kₙ)ᵀφ(qⱼ) | 线性注意力归一化分母 | 标量 | 类似 softmax 中的 Σ exp，防止输出过大 |
| gⱼ | 输出门控（output gate） | (dₕ,) | σ 为 sigmoid，逐元素门控 |
| Wₒ | 输出投影 | (dₕ, dₕ) | 线性变换 |
| oⱼ | 单头输出 | (dₕ,) | 最终拼接所有 head |

#### 核心直觉

- 状态矩阵 S 是 **dₕ×dₕ = 128×128 ≈ 16K** 参数，与序列长度 N 完全无关
- 所有 token 按顺序写入 S（每个 token 贡献 vₙ·(βₙkₙʳ)ᵀ），然后每个 query 直接从 S 读取
- 这使得写入为 O(N·dₕ²)，读取为 O(N·dₕ²)，总复杂度 O(N)，常数因子为 dₕ²
- **瓶颈**：S 只有 128 秩，无法表示所有 token 间的精确交互——这就是为什么需要周期性 softmax 锚点来补充

#### 与 LLM 版本的区别

- 本文使用 **双向（bidirectional）** 操作，不是因果 delta-rule 递归
- 去掉 delta-rule 更新后，成为 Gated DeltaNet 的自然初始化
- 线性层使用 **双门控**（write gate β + output gate g），而 softmax 锚点只用 sigmoid 输出门控

---

### Q2: "路由器从来源集合中动态聚合特征"是什么意思？路由的输入是这些特征，而不是原本 attention 输出？

#### 答：路由器是 attention/FFN **之前** 的一个额外步骤，不替换 attention

整体数据流如下（对每一层 l）：

```
输入 hₗ₋₁
  │
  ├── AttnRes 路由器（新增）
  │     ├── 从来源集合 Vₗ 计算加权聚合
  │     └── 输出 hₗ = Σ αᵢ·vᵢ  （替代 hₗ₋₁ 作为后续子层输入）
  │
  ├── AdaLN 调制
  │
  ├── Self-Attention（线性 or softmax）  ← 仍然正常执行
  │
  ├── Cross-Attention（文本条件）        ← 仍然正常执行
  │
  ├── SwiGLU FFN                         ← 仍然正常执行
  │
  └── 输出 hₗ
```

#### 来源集合 Vₗ 的组成

Vₗ 包含当前层 l 可以"看到"的所有深度特征：

| 来源 | 符号 | 含义 | 形状 |
|------|------|------|------|
| 初始嵌入 | b₀ | 整个模型的 token 嵌入输入 | (N, d) |
| 已完成块摘要 | b₁, b₂, ... | 已完成块（每 S=8 层一个块）的累积特征 | (N, d)，每个块一个 |
| 当前块部分和 | pₗ | 当前正在执行的块内，到目前为止的累积特征 | (N, d) |

在 5B 模型中（32 层，S=8），最多有 4 个块 + 1 个嵌入 + 1 个部分和 = 最多 6 个来源。

#### 路由计算（以 attention 分支为例）

$$h_l(x) = \sum_{v_i \in V_l} \alpha^{(\text{attn})}_{i \to l}(x) \cdot v_i(x)$$

$$\alpha^{(\text{attn})}_{i \to l}(x) = \operatorname{softmax}_i\!\Big( w^{(\text{attn})^\top} \operatorname{RMSNorm}(v_i(x)) \Big)$$

- w^(attn) 是一个可学习的查询向量（**所有深度共享**，见 Q3）
- softmax 在来源维度 i 上计算，对每个 token 位置 x 独立
- 结果 hₗ(x) 替代原始输入 hₗ₋₁(x) 送入后续的 attention 和 FFN

#### 关键理解

1. **Attention 仍然正常工作**：路由器不会移除或替换 self-attention / cross-attention / FFN。它只是在这些子层之前插入了一个跨深度的特征混合步骤
2. **路由器的作用**：把之前块学到的、包含 softmax 锚点更新的特征"注入"当前层，让线性层也能利用高秩信息
3. **实验证据**：去掉路由器后，深层有效秩下降 12%；路由权重在深度上重用 50%+ 的已完成块特征

---

### Q3: "所有深度共享同一套路由查询"是指 8 个块中的 MoE 都用同一套 expert 吗？

#### 答：不是 MoE。"共享路由查询"指的是路由公式中的查询向量 w^(τ) 在所有深度复用

#### 路由公式回顾

$$\alpha^{(\tau)}_{i \to l}(x) = \operatorname{softmax}_i\!\Big( \big(w^{(\tau)} + \phi_\tau(t)\big)^{\!\top} \operatorname{RMSNorm}(v_i(x)) \Big)$$

其中：
- **w^(τ)**：路由查询向量（可学习参数）
- τ ∈ {attn, ffn}：区分 attention 分支和 FFN 分支

#### "共享" vs. "不共享"的区别

| 方案 | 参数 | 含义 |
|------|------|------|
| **不共享（原始 AttnRes / Kimi K3）** | 每层有独立的 w^(τ)ₗ | 32 层 × 2 分支 = 64 个路由查询向量 |
| **共享（本文）** | 所有层共用同一 w^(attn) 和 w^(ffn) | 仅 2 个路由查询向量 |

#### 为什么共享不影响效果？

共享查询后，路由权重 α 仍然会**随深度变化**，原因有两个：

1. **来源集合 Vₗ 随深度变化**：在第 l 层时，能看到的已完成块摘要数量取决于前面完成了多少个块（⌊l/8⌋ 个已完成块 + 当前部分和）。不同的来源 vᵢ(x) 会产生不同的 αᵢ
2. **来源特征本身随深度变化**：即使是同一个来源（如 b₁），其特征值也会随训练演化，而 softmax 是对来源特征做的 RMSNorm + 投影

所以"共享查询"只是减少了参数数量，路由的输出仍然是深度自适应的。

#### 内存节省

路由查询参数占比极小（远 < 0.001%），但共享后节省了 **4× 的路由参数内存**（从 N_layers×2 减到 2），同时 loss 基本持平（0.496 vs 0.495）。