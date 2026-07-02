# LongCat-Video Technical Report 论文总结

- **论文标题**: LongCat-Video Technical Report
- **作者**: Meituan LongCat Team（美团）
- **arXiv**: https://arxiv.org/abs/2510.22200
- **发布时间**: 2025年10月
- **代码/模型**: https://github.com/meituan-longcat/LongCat-Video

---

## 1. 核心贡献

LongCat-Video 是美团团队提出的 **13.6B 参数视频生成基础模型**，为迈向世界模型（World Model）的第一步。主要贡献包括：

1. **统一多任务架构**：基于 DiT 框架，单一模型同时支持 Text-to-Video、Image-to-Video、Video-Continuation 三个任务。
2. **长视频生成能力**：通过在 Video-Continuation 任务上预训练，可生成长达数分钟的高质量视频，无色彩漂移和质量退化。
3. **高效推理**：采用由粗到精（Coarse-to-Fine）生成策略和 Block Sparse Attention，将推理效率提升 10× 以上，可在数分钟内生成 720p/30fps 视频。
4. **多奖励 GRPO 训练**：在 Flow Matching 框架下设计多奖励 RLHF，性能媲美最新闭源和顶尖开源模型。

---

## 2. 模型架构

### 2.1 网络结构

采用标准 **DiT（Diffusion Transformer）** 架构，具体参数：

| 参数 | 数值 |
|------|------|
| 层数 | 48 |
| 隐藏维度 | 4096 |
| FFN 隐藏维度 | 16384 (SwiGLU) |
| 注意力头数 | 32 |
| AdaLN 嵌入维度 | 512 |
| 总参数量 | 13.6B |

关键技术细节：
- **AdaLN-Zero** 调制机制，每层配备专用调制 MLP
- **RMSNorm** 作为 QKNorm，应用于自注意力和交叉注意力
- **3D RoPE** 用于视觉 token 位置编码
- **WAN2.1 VAE** 进行视频像素压缩（压缩比 4×8×8），DiT 内 patchify 进一步压缩（1×2×2），总压缩比 4×16×16
- **umT5** 作为多语言文本编码器，支持中英文

### 2.2 统一多任务架构

将所有任务统一为"视频续写"范式：

- **Text-to-Video**：条件帧数为 0（仅噪声）
- **Image-to-Video**：条件帧数为 1（单张图片）
- **Video-Continuation**：条件帧数为多帧

输入格式为 $X = [X_{cond}, X_{noisy}]$，条件帧的时间步 $t_{cond}$ 固定为 0，噪声帧的时间步 $t_{noisy}$ 在 [0,1] 内采样。损失计算时忽略条件帧部分。

### 2.3 Block Attention with KVCache

设计了专门的注意力机制：
- $X_{cond} = Attention(Q_{cond}, K_{cond}, V_{cond})$ —— 条件 token 仅依赖自身
- $X_{noisy} = Attention(Q_{noisy}, [K_{cond}, K_{noisy}], [V_{cond}, V_{noisy}])$ —— 噪声 token 可同时关注条件和噪声区域

条件 token 的 KV 特征可跨所有采样步复用（KVCache），提升长视频生成效率。

---

## 3. 多奖励 GRPO 训练

### 3.1 核心洞察：GRPO 作为随机噪声搜索

论文揭示了 Flow Matching 模型中的关键关系：
$$\frac{dR}{dv_\theta} \approx -\frac{3}{2} \hat{A}_t^i \cdot \epsilon$$

即 GRPO 通过相对优势 $\hat{A}_t^i$ 和 SDE 噪声 $\epsilon$ 来近似奖励对速度场的梯度。

### 3.2 四大技术创新

#### （1）固定 SDE 随机时间步（Fix SDE Timestep）
- 每个 prompt 的所有样本共享相同初始噪声，仅在随机选取的**单个关键时间步** $t'$ 上使用 SDE 采样（注入噪声），其余时间步使用确定性 ODE 采样
- 解决了时序信用分配（Temporal Credit Assignment）模糊的问题

#### （2）截断噪声调度（Truncated Noise Schedule）
- SDE 扩散系数 $\sigma_t\sqrt{\Delta t}$ 接近 $t=1$ 时会过大，导致训练不稳定
- 引入阈值截断：$\sigma_t\sqrt{\Delta t} \to \min(\sigma_t\sqrt{\Delta t}, \tau)$，其中 $\tau=0.45$

#### （3）策略损失和 KL 损失重加权
- 发现梯度幅度被因子 $\kappa(t,\Delta t)=\sqrt{\frac{\Delta t(1-t)}{t}}$ 缩放，导致 $t\to1$ 时梯度消失
- 引入重加权系数消除时间步依赖：
  - $\lambda_{policy} = \sqrt{\frac{t}{\Delta t(1-t)}}$
  - $\lambda_{KL} = \frac{t}{\Delta t(1-t)}$

#### （4）最大组标准差（Max Group Std）
- 用所有组中的**最大标准差**替代各组自身的标准差进行优势归一化
- 防止低方差组的不可靠优势估计对训练的负面影响

### 3.3 多奖励设计

使用三个专业奖励模型：

| 奖励类型 | 模型基础 | 特点 |
|---------|---------|------|
| 视觉质量（VQ） | HPSv3 | HPSv3-general（全帧均值）+ HPSv3-percentile（top 30% 帧） |
| 运动质量（MQ） | VideoAlign | 使用灰度视频训练，聚焦运动特征而非颜色 |
| 文本-视频对齐（TA） | VideoAlign | 保留原始颜色输入，评估语义对应关系 |

多奖励训练通过各奖励间的相互约束自然防止奖励黑客（Reward Hacking），例如运动奖励可抑制 HPSv3 导致的静态倾向。

---

## 4. 数据管理

### 4.1 数据预处理
- 基于源视频 ID 和 MD5 哈希去重
- 使用 PySceneDetect + 自研 TransNetV2 进行转场分割
- FFMPEG 裁剪黑边

### 4.2 数据标注
- **基础视频描述**：微调 LLaVA-Video 模型，增强时序理解和动作描述能力
- **摄影和视觉风格**：训练专用相机运动分类器（平移/倾斜/缩放/晃摄），使用 Qwen2.5VL 标注镜头类型和视觉风格
- **描述增强**：中英翻译、摘要生成、随机组合摄影/风格元素

### 4.3 数据分布分析
- 对描述文本嵌入进行聚类分析
- 使用 LLM 为每个聚类打标签
- 针对性补充或再平衡数据

---

## 5. 训练流程

### 5.1 渐进式预训练

| 阶段 | 任务 | 分辨率 | 学习率 | 迭代数 |
|------|------|--------|--------|--------|
| 1 | T2I | 256p | 1e-4 | 285k |
| 2 | T2I + T2V | 256p×93帧 | 1e-4 | 140k |
| 3 | T2I+T2V+I2V+VC | 256p×93帧 | 5e-5 | 164k |
| 4 | T2I+T2V+I2V+VC | 480p×93帧 | 5e-5 | 36k |
| 5 | T2I+T2V+I2V+VC | 480p+720p×93帧 | 2e-5 | 53k |

采用 Bucket 策略进行混合分辨率训练，优化器为 AdamW。

### 5.2 有监督微调（SFT）
- 使用高质量筛选数据集
- 按美学分数、视频质量、运动质量等过滤
- 包含相机运动和视觉风格专用数据集
- 学习率 1e-5，7.5k 迭代

### 5.3 RLHF 训练
- 仅使用 Text-to-Video 任务进行 GRPO
- Group size=4, prompts per step=64
- 16 采样步，SDE 步范围 [0,6]
- 学习率 1e-4，500 迭代
- 使用 LoRA（rank=128, alpha=64）

### 5.4 加速训练
- **CFG 蒸馏** + **一致性模型（CM）蒸馏**：使推理步数从 50+ 降至 16
- **Refinement Expert LoRA**：负责 Coarse-to-Fine 第二阶段超分辨率

---

## 6. 高效推理

### 6.1 由粗到精（Coarse-to-Fine）生成

两阶段流程：
1. **粗阶段**：生成 480p/15fps 视频（16 步采样）
2. **精阶段**：三线性插值上采样至 720p/30fps，Refinement Expert 进行 5 步去噪

Refinement Expert 使用 Flow Matching 训练，添加适度噪声 $t_{thresh}=0.5$，仅需 5 步采样。

### 6.2 Block Sparse Attention（BSA）

- 将视频序列划分为 $t\times h\times w$ 3D 块
- 通过块级平均池化计算块间相似度分数
- 每个 query 块仅关注 top-r 个最相关的 key 块
- 计算量降至原注意力 **< 10%**，质量近乎无损

关键实现细节：
- 基于 Triton + Flash Attention 实现，支持前向/反向传播和 Context Parallelism
- 3D 块大小 $t=h=w=4$
- 蒸馏阶段稀疏度：$r=\frac{1}{8}N_k$（约 87.5% 稀疏）
- Refinement 阶段稀疏度：$r=\frac{1}{16}N_k$（约 93.75% 稀疏）
- 支持 Ring Block Sparse Attention 的 Context Parallelism

### 6.3 推理效率对比

| 配置 | LCM | C2F | BSA | 采样步数 | 延迟 | 加速比 |
|------|-----|-----|-----|---------|------|--------|
| 720p×93帧 原始 | ✗ | ✗ | ✗ | 50 | 1429.5s | 1.0× |
| 720p×93帧 +蒸馏 | ✓ | ✗ | ✗ | 16 | 244.6s | 5.8× |
| 480p→720p×93帧 | ✓ | ✓ | ✗ | 16+5 | 135.3s | 10.6× |
| 480p→720p×93帧 **全部** | ✓ | ✓ | ✓ | 16+5 | **116.5s** | **12.3×** |
| 480p→720p×189帧 **全部** | ✓ | ✓ | ✓ | 16+5 | **142.0s** | **10.1×** |

（单卡 H800 GPU 测试）

---

## 7. 评测结果

### 7.1 内部基准（Text-to-Video）

评测维度：文本对齐、视觉质量、运动质量、综合质量。

**MOS 评估**：LongCat-Video 综合质量优于 PixVerse-V5 和 Wan2.2-T2V-A14B，视觉质量与 Wan2.2 持平，仅次于 Veo3。

**GSB 评估**：与 PixVerse-V5 综合质量持平（242 vs 246），视觉质量有优势；显著优于 Wan2.2-T2V-A14B。

### 7.2 内部基准（Image-to-Video）

与 Seedance 1.0、Hailuo-2、Wan2.2-I2V-A14B 对比：
- **视觉质量最高**（3.27），领先所有对比模型
- 图像对齐（4.04）和运动质量（3.59）略逊于竞品，是未来改进方向

### 7.3 公开基准 VBench 2.0

总分 **62.11%**，仅次于 Veo3（66.72%）和 Vidu Q1（62.70%）：
- **Commonsense 维度领先所有方法**（70.94%），体现运动合理性和物理规律方面的优势
- Controllability（44.79%）表现突出

---

## 8. 训练基础设施

- **DeepSpeed-Zero2** + Context Parallelism + Ring Attention + Activation Checkpointing
- Bucket 策略支持混合分辨率批量训练
- VAE 操作跨 rank 缓存机制消除计算气泡
- MFU 达 33%–38%

---

## 9. 总结与展望

LongCat-Video 作为 13.6B 视频生成基础模型，在统一多任务、长视频生成和高效推理方面均展现强竞争力。开源了代码、模型权重和 Block Sparse Attention 模块。

未来方向：
- 更好的物理知识建模
- 多模态记忆集成
- 融入 LLM/MLLM 的知识

---

## 10. 架构深入：Q&A

### 10.1 Token 与 VAE 基础

#### Q1: 条件 Token 和噪声 Token 如何理解？

| 维度 | 条件 Token ($X_{cond}$) | 噪声 Token ($X_{noisy}$) |
|------|------------------------|-------------------------|
| **来源** | 用户提供的参考帧 → VAE 编码 | 纯高斯噪声 $\epsilon \sim \mathcal{N}(0,I)$ |
| **时间步** | $t=0$ 固定不变 | $t$ 从 $1 \to 0$（16 步 ODE 逐步减小） |
| **值变化** | 16 步中始终不变 | 每步更新（逐渐接近真实视频） |
| **语义** | "已知的上下文/参考" | "待预测的未来帧" |
| **KVCache** | 可缓存复用 | 每步重新计算 |
| **参与 Loss** | ❌ Mask 掉 | ✅ 计算 MSE Loss |

不同任务的唯一区别 = 条件 Token 数量：

```
T2V:  [████████████████████████████████]  条件=0,   噪声=93帧
I2V:  [█|███████████████████████████████]  条件=1,   噪声=93帧
VC:   [████████████████████|███████████]  条件=73,  噪声=93帧
```

#### Q2: VAE 处理后帧数会变化吗？

**会。** WAN2.1 VAE 压缩比为 $4 \times 8 \times 8$（时间 × 高 × 宽），DiT 内 patchify 再压缩 $1 \times 2 \times 2$，总压缩比 $4 \times 16 \times 16$。

```
输入: 93 帧 RGB (480×854)
  ↓ VAE (4×8×8)
Latent: ~23 temporal × 60×107 spatial
  ↓ DiT Patchify (1×2×2)
DiT Tokens: ~23 temporal × 30×53 spatial ≈ 38K tokens
```

#### Q3: 在 DiT 中，每一帧是一个 Token 吗？

**不是。** 每 latent 时间步有 $H' \times W'$ 个空间 token，总 token 数 = $N_t \times H' \times W'$：

| 分辨率 | $H'$ | $W'$ | 每时间步 token | 总 token（~23 步） |
|--------|------|------|---------------|------------------|
| 480p | 30 | ~53 | ~1590 | ~38K |
| 720p | 45 | ~80 | ~3600 | ~83K |

每个 token 为 $d=4096$ 维向量，送入 48 层 DiT。

#### Q4: 3D VAE 是否会混淆不同帧的信息？

3D 卷积会在**时间维度聚合相邻帧**（每 4 帧融合为 1 个 latent 时间步），但这是**有损压缩**而非"混淆"，VAE 经过重建训练可以解码复原。

**关键边界**：条件 Token 和噪声 Token **不会在 VAE 中交叉**：

```
条件帧 (RGB)  →  VAE Encode  →  条件 latent token  ─┐
                                                     ├→ 在 DiT latent space 中拼接
噪声帧         →  直接从 N(0,I) 采样  →  噪声 latent token  ─┘
                    (在 latent space 中直接初始化)
```

噪声 Token 不经过 VAE Encoder，两者在 DiT 输入层才第一次相遇。

#### Q5: DiT 如何区分条件 Token 和噪声 Token？

四层机制共同作用：

1. **时间步嵌入（最主要）**：$t_{cond}=0$，$t_{noisy} \in (0,1]$，通过 AdaLN 调制 token 特征。$t=0$ → "已干净，作为参考"；$t>0$ → "当前噪声水平，需要去噪"。

2. **Block Attention Mask**：条件 Token 不关注噪声 Token，形成结构性隔离。

3. **3D RoPE**：条件 Token 的 temporal 坐标在前段，噪声 Token 在后段，位置编码天然区分。

4. **隐式分布差异**：条件 Token 是结构化 latent（编码自真实图像），噪声 Token 从高方差逐步收敛，模型通过训练习得差异。

---

### 10.2 注意力机制与 KV Cache

#### Q6: DiT 中使用 Causal Mask 还是 Full Attention？

**都不是。** LongCat-Video 使用 **Block Attention（块级注意力）**：

```
                  Query
            ┌─────────┬──────────┐
            │  X_cond │ X_noisy  │
     ┌──────┼─────────┼──────────┤
  K  │X_cond│    ✓    │    ✓     │  ← 条件 K/V 可被所有人关注
  e  ├──────┼─────────┼──────────┤
  y  │X_noisy   ✗    │    ✓     │  ← 噪声 K/V 不影响条件 Token
     └──────┴─────────┴──────────┘
```

公式表达：
$$X_{cond} = \text{Attention}(Q_{cond}, K_{cond}, V_{cond})$$
$$X_{noisy} = \text{Attention}(Q_{noisy}, [K_{cond}, K_{noisy}], [V_{cond}, V_{noisy}])$$

| 对比维度 | 标准 Causal Mask (LLM) | Full Attention | LongCat Block Attention |
|---------|----------------------|----------------|------------------------|
| 条件 ↔ 条件 | N/A | ✓ 双向 | ✓ **双向** |
| 噪声 ↔ 噪声 | N/A | ✓ 双向 | ✓ **双向** |
| 噪声 → 条件 | N/A | ✓ | ✓ 允许 |
| 条件 → 噪声 | N/A | ✓ | ✗ **阻断** |

核心设计意图：条件 Token 不受噪声 Token 影响，保证训练-推理一致性。

#### Q7: Full Attention / Block Attention 如何做 KV Cache？

DiT 的 KV Cache **完全不同于 LLM**——不是 token-by-token 串行复用，而是**跨去噪步复用**：

```
去噪步 1 (t=1.0):
  全部 token 同时前向
  → 计算并缓存 K_cond, V_cond  ← 条件 Token 不变

去噪步 2~16:
  条件 Token 不变 → 直接复用 K_cond, V_cond
  噪声 Token 逐步去噪 → 每步重新计算 K_noisy, V_noisy
```

**为什么可行**：条件 Token 的 $t_{cond}=0$ 在整个 16 步中保持干净的 latent 值不变，其 K、V 只需计算一次。

**跨 Chunk 的 KV Cache 累积**（长视频场景）：

```
Chunk 1: 生成 93 帧 → 全部成为"条件" → 缓存 93 帧的 K, V
Chunk 2: 条件 = 73 帧(来自 Chunk 1) → 复用缓存 + 生成 20 帧 → 追加缓存
Chunk 3: 条件 = 73 帧(来自 Chunk 1+2) → 复用 + 追加
...
→ KV Cache 线性增长 → QVG 量化在这里发挥作用
```

QVG 量化的正是 48 层 × 32 head 的全部条件 K、V，从 BF16 → INT2/INT4，实现最高 **7.05× 压缩**（34 GB → ~4.8 GB）。

---

### 10.3 长视频生成与跨 Chunk 机制

#### Q8: Video-Continuation 是否使用 Autoregressive？

**Chunk 内部完全非自回归**：所有 93 帧噪声 Token **并行去噪生成**（16 步 ODE），不存在帧与帧之间的因果依赖。

长视频通过 **段级链接（segment-level chaining）** 实现：

```
Chunk 1: [条件帧×0 (T2V)] → 并行去噪 → 93帧 (~6秒)
                              ↓
Chunk 2: [后73帧作为条件] → 并行去噪 → 93帧 (续写)
                              ↓
Chunk 3: [后73帧作为条件] → 并行去噪 → 93帧 (续写)
                              ↓
                            ...
                              ↓
                      累计数分钟长视频
```

段间串行（前一 Chunk 的输出作为后一 Chunk 的条件），段内并行（Flow Matching 去噪）。

#### Q9: 跨 Chunk 生成是领域标准做法吗？

**不是唯一标准**，当前领域存在多种策略：

| 策略 | 代表模型 | KV Cache | 优点 | 缺点 |
|------|---------|----------|------|------|
| **原生长生成** | Sora, Veo3（闭源） | 不详 | 一次生成，质量最高 | 训练成本极高 |
| **Chunk-wise 全上下文** | **LongCat-Video** | 线性增长（~34GB/5s） | 无色彩漂移，时序一致 | KV Cache 膨胀 |
| **滑动窗口截断** | Wan Self-Forcing (21帧), HY-WorldPlay (20帧) | 恒定（小） | 内存友好 | 丢失远距离上下文 |
| **噪声调度扩展** | FreeNoise, Gen-L-Video | N/A | 绕过 AR 问题 | 运动重复、多样性差 |
| **分层生成** | VideoPoet, Lumiere | N/A | 先关键帧后插值 | 需额外模型 |

QVG 论文 Motivation 一节直接指出这一 tradeoff：
> *"Bounding the context effectively shrinks the model's working memory, which can exacerbate long-horizon drift and limit revisitability and temporal consistency."*

QVG 的核心贡献就是通过 KV Cache 量化，使全上下文策略的内存开销逼近滑动窗口策略。

#### Q10: 三类任务都使用跨 Chunk 生成吗？

| 任务 | 标准用法 | 是否跨 Chunk？ | 说明 |
|------|---------|:---:|------|
| **T2V** | 单次 93 帧（~6 秒） | ❌ 不需要 | 论文评测均在此长度下进行 |
| **I2V** | 单次 1 + 93 帧（~6 秒） | ❌ 不需要 | 同上 |
| **VC** | 单次或 **多 Chunk 串行** | ✅ 唯一使用 | 论文 "Long-Video Generation Examples" 展示的正是此模式 |

但 T2V / I2V **可以通过跨 Chunk 扩展为长视频**：

```
Chunk 1:  T2V (N_cond=0) 或 I2V (N_cond=1) → 生成 93 帧
Chunk 2+: VC  (N_cond=73) → 持续续写，可每段换 prompt
```

论文将此称为 "interactive video generation with changing instructions for each clip"。

---

### 10.4 Video-Continuation 训练与推理全流程

#### 一、核心设计理念

LongCat-Video 将 Text-to-Video、Image-to-Video、Video-Continuation 三种任务统一为同一个框架——**"视频续写"**。三者的唯一区别在于条件帧的数量：

| 任务 | $N_{cond}$ | $N_{noisy}$ |
|------|:---:|:---:|
| Text-to-Video | 0 | 93 |
| Image-to-Video | 1 | 93 |
| Video-Continuation | 可变（如 21/73） | 93 |

---

#### 二、训练过程

**2.1 训练数据构造**

对于 Video-Continuation，从一段完整视频中随机采样子片段：

```
原始视频: [帧1, 帧2, 帧3, ..., 帧N]

随机采样起点 k，选取:
  X_cond = [帧k, 帧k+1, ..., 帧k+N_cond-1]      ← 条件帧（干净 latent）
  X_0    = [帧k+N_cond, ..., 帧k+N_cond+N_noisy-1] ← 目标帧（GT）

其中 X_0 是可变的 N_noisy 帧，如 93 帧
条件帧数量 N_cond 在训练时也是随机变化的，使模型学会适应不同长度的上下文。
```

**2.2 统一输入表示与 Flow Matching**

Step 1 — VAE 编码：所有 RGB 帧通过 WAN2.1 VAE 编码为 latent：

$$X_{cond} = \text{VAE_Encode}(\text{条件帧}),\quad X_0 = \text{VAE_Encode}(\text{目标帧})$$

Step 2 — 加噪：仅对目标帧加噪（条件帧保持干净）：

$$X_{noisy} = (1-t) \cdot X_0 + t \cdot \epsilon,\quad \epsilon \sim \mathcal{N}(0,I)$$

Step 3 — 拼接输入：

$$X_{input} = [X_{cond}; X_{noisy}] \in \mathbb{R}^{B \times (N_{cond}+N_{noisy}) \times H \times W \times C}$$

$$t_{input} = [\underbrace{0,0,...,0}_{N_{cond}}; \underbrace{t,t,...,t}_{N_{noisy}}]$$

关键设计：$t_{cond}=0$ 固定不变，向模型显式传递"这些是干净信息，不需要去噪"的信号。

Step 4 — DiT 前向传播：输入 $X_{input}$ 和 $t_{input}$ 送入 48 层 DiT，模型预测速度场 $v_{pred}$。

Step 5 — 损失计算（仅噪声帧）：

$$v_{gt} = X_0 - \epsilon$$

$$\mathcal{L} = \frac{1}{N_{noisy}} \sum_{i \in \text{noisy}} \|v_{pred}^{(i)} - v_{gt}^{(i)}\|^2$$

条件帧部分的预测被 mask 掉，不参与梯度反传。

**2.3 Block Attention 机制（训练与推理一致）**

```
Self-Attention:
  X_cond = Attention(Q_cond, K_cond, V_cond)           ← 只看自己
  X_noisy = Attention(Q_noisy, [K_cond, K_noisy],        ← 看条件+噪声
                                [V_cond, V_noisy])

Cross-Attention:
  仅 X_noisy 做 cross-attn with text embeddings
  X_cond 不参与 cross-attn
```

关键特性：
- 条件 token 不受噪声 token 影响（单向信息流），保证训练-推理一致性
- 条件部分的 K、V 在推理时可直接缓存复用

**2.4 抗颜色漂移：Per-Frame 独立噪声扰动**

论文特别提到（第 4 节第 30 行）：

> *"For Video-Continuation task, we also perturb conditional frames with per-frame independent noise levels to enhance robustness to color drift."*

这借鉴了 Diffusion Forcing 的思想：训练时对条件帧也加入微小噪声（每帧独立采样），让模型学会从"不完美的条件"中续写，避免推理时因累积误差导致的颜色漂移。

**2.5 渐进式多阶段预训练**

VC 任务在预训练的第三阶段才加入，与 T2I、T2V、I2V 联合多任务训练：

| 阶段 | 任务 | 分辨率 | LR | Iter |
|------|------|--------|-----|------|
| 1 | T2I | 256p | 1e-4 | 285k |
| 2 | T2I + T2V | 256p×93f | 1e-4 | 140k |
| 3 | T2I+T2V+I2V+VC | 256p×93f | 5e-5 | 164k |
| 4 | T2I+T2V+I2V+VC | 480p×93f | 5e-5 | 36k |
| 5 | T2I+T2V+I2V+VC | 480p+720p×93f | 2e-5 | 53k |

多任务采样比例约为 **T2I:T2V:I2V:VC = 2:5:3:2**。

**2.6 SFT 微调**

预训练后，使用高质量筛选数据对**全部四种任务（含 VC）**进行 SFT：

| 任务 | 分辨率 | LR | Iter |
|------|--------|-----|------|
| T2I + T2V + I2V + VC | 480p + 720p×93f | 1e-5 | 7.5k |

**2.7 GRPO RLHF（仅 T2V，但对 VC 泛化有效）**

GRPO 阶段仅使用 T2V 任务训练，但论文指出：

> *"We find that the improvements of instruction-following, visual quality and motion quality generalize well to Image-to-Video and Video-Continuation tasks."*

**2.8 蒸馏训练（CFG + CM，含 VC）**

CFG 蒸馏和 Consistency Model 蒸馏阶段四任务全开：

| 阶段 | 任务 | 分辨率 | LR | Iter |
|------|------|--------|-----|------|
| CFG蒸馏 | T2I+T2V+I2V+VC | 480p+720p×93f | 5e-5 | 2k |
| CM蒸馏 | T2I+T2V+I2V+VC | 480p+720p×93f | 5e-5 | 3k |

蒸馏后推理步数从 50 → 16 步，CFG 被隐式蒸馏进模型，无需双倍前向。

**2.9 Refinement Expert 训练（含条件帧 C2F）**

对于带条件帧的 C2F 精化训练，流程如下：

训练数据构造：

```
高分辨率原始视频 → 取条件帧 X_hr^cond + 目标帧
         ↓
下采样条件帧: X_lr^cond = Downsample(X_hr^cond)
低分辨率目标: X_lr = Decode(BaseModel([Encode(X_lr^cond), ϵ]))  ← 第一阶段模拟
         ↓
拼接上采样: X_up = [X_hr^cond, Upsample(X_lr)]
         ↓
VAE编码+加噪: x_thresh = (1-0.5)·Encode(X_up) + 0.5·ϵ
         ↓
GT速度: v_t' = (X_0_hr - x_thresh) / 0.5  ← 归一化到与 base model 一致的数值范围
```

训练参数：$t_{thresh}=0.5$，先用 Full Attention 收敛 → 再切 BSA（93.75% 稀疏）。

---

#### 三、推理过程

**3.1 单次 Video-Continuation（非自回归）**

这是 VC 的基本单元——一次生成 93 帧续写：

```
输入条件: X_cond (如 21 帧，VAE编码后的 latent)
         +
        X_noisy = ɛ ∼ N(0,I)  (93 帧纯噪声)

文本条件: c (umT5 编码的文本)

时间步: t_cond = [0,...,0], t_noisy = [1,...,1]

→ 48层 DiT，16步 ODE 迭代去噪
→ VAE Decode
→ 输出 93 帧 RGB 视频续写
```

KVCache 优化：第一步 forward 时计算并缓存 $K_{cond}, V_{cond}$（条件帧 21 帧的 K、V），后续 15 步直接复用。

**3.2 Coarse-to-Fine 两阶段生成（带条件帧）**

第一阶段：粗生成（480p, 15fps）

```
Step 1: 条件帧下采样
  X_lr^cond = Downsample(X_hr^cond)   ← 高分辨率→480p

Step 2: VAE编码 + 拼接噪声
  X_lr_input = [Encode(X_lr^cond), ɛ]

Step 3: Base Model 16步去噪
  X_lr = BaseModel(X_lr_input, c)

Step 4: VAE解码
  RGB_lr = Decode(X_lr)   ← 480p/15fps
```

第二阶段：精化（720p, 30fps）

```
Step 1: 三线性上采样
  X_up = [X_hr^cond, TrilinearUpsample(X_lr)]
  → 条件帧保持原始高分辨率，生成部分上采样到 720p/30fps

Step 2: VAE编码
  X_up_latent = Encode(X_up)

Step 3: 加噪 (t_thresh=0.5)
  X_start = 0.5·X_up_latent + 0.5·ɛ   ← 仅50%噪声，保留布局

Step 4: Refinement Expert (LoRA) 5步去噪 + BSA
  X_sr = Refinement(X_start, c)

Step 5: VAE解码
  最终输出 720p/30fps RGB 视频
```

条件帧始终保留原始高分辨率，避免了低分辨率阶段的细节损失。

**3.3 长视频生成：Chunk-wise 段级自回归**

对于超长时间的视频，LongCat-Video 采用逐段链接：

```
Chunk 1: [0帧条件] + 93帧噪声 → T2V生成93帧 (约6秒@15fps)
                                    ↓
Chunk 2: [后73帧作为条件] + 93帧噪声 → VC续写93帧
                                    ↓
Chunk 3: [后73帧作为条件] + 93帧噪声 → VC续写93帧
                                    ↓
                                  ...
                                    ↓
                            累计数分钟
```

每段仍需经历 C2F 两阶段（480p粗 + 720p精），但 KV Cache 跨段线性累积，QVG 方法的 KV Cache 量化正是在这一步发挥作用。

**3.4 推理数据流全览**

```
┌─────────────────────────────────────────────────────────────┐
│                    第一阶段 (Coarse)                          │
│                                                              │
│  X_hr^cond (高分辨率条件帧)                                   │
│       ↓ Downsample                                           │
│  X_lr^cond (480p条件帧)                                      │
│       ↓ VAE Encode                                           │
│  [Encode(X_lr^cond) ‖ ɛ (93帧噪声)]                          │
│       ↓ Base Model × 16步 ODE (含KVCache)                    │
│  X_lr (480p/15fps latent, 93帧)                              │
│       ↓ VAE Decode                                           │
│  RGB_lr (480p/15fps, 93帧)                                   │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                    第二阶段 (Refine)                          │
│                                                              │
│  X_up = [X_hr^cond ‖ TrilinearUpsample(RGB_lr)]             │
│       ↓ VAE Encode                                           │
│  X_up_latent (720p/30fps latent)                             │
│       ↓ Add noise (t_thresh=0.5)                             │
│  X_start = (1-0.5)·X_up_latent + 0.5·ɛ                      │
│       ↓ Refinement Expert LoRA × 5步 + BSA (93.75%稀疏)     │
│  X_sr (720p/30fps latent, 93帧)                              │
│       ↓ VAE Decode                                           │
│  最终输出: 720p/30fps RGB 视频 (~6秒)                         │
└─────────────────────────────────────────────────────────────┘
```

---

#### 四、训练 vs 推理对比总结

| 维度 | 训练 | 推理 |
|------|------|------|
| 输入构造 | $X_{cond}$ (干净) + $X_{noisy}$ (加噪GT)，$N_{cond}$ 随机可变 | $X_{cond}$ (编码后的给定帧) + $\epsilon$ (纯噪声) |
| 去噪步数 | 1步（单次前向，预测速度场） | 16步 ODE（蒸馏后） |
| 条件帧处理 | 可加入 per-frame 独立噪声抗漂移 | 保持干净，VAE编码后直接拼接 |
| 注意力 | Block Causal（训练-推理一致） | Block Causal + KVCache |
| C2F | 需单独训练 Refinement Expert LoRA | 两阶段：480p×16步 + 720p×5步 |
| 长视频 | 单次 93 帧训练 | Chunk-wise 段级链接 |
| 损失 | MSE，仅噪声帧 | 无（纯前向） |
| CFG | 蒸馏进模型（CFG-Zero） | 无需额外负提示前向 |
