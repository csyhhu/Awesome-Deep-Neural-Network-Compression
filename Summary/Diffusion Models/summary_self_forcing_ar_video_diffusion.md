# Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion

**论文信息**：NeurIPS 2025 | Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, Eli Shechtman (Adobe Research & UT Austin)

**项目页面**：https://self-forcing.github.io/

---

## 1. 核心问题：Exposure Bias

自回归（AR）视频扩散模型将视频生成分解为 $p(x^{1:N}) = \prod_{i=1}^{N} p(x^i|x^{<i})$，每帧去噪时以历史帧为条件。现有训练范式存在 **train-test distribution gap**：

| 训练范式 | 训练时 context 来源 | 推理时 context 来源 | 问题 |
|---------|-------------------|-------------------|------|
| **Teacher Forcing (TF)** | 干净的真实帧 $x^{<i} \sim p_{\text{data}}$ | 模型自己生成的帧 $\hat{x}^{<i} \sim p_\theta$ | 分布不匹配 → 误差累积 |
| **Diffusion Forcing (DF)** | 加噪的真实帧 $\tilde{x}^{<i}$ | 干净的自生成帧 | 分布不匹配 + 牺牲时序一致性 |

这种分布不匹配导致 **exposure bias**：模型在训练时只见过"完美"历史，推理时却必须依赖自己不完美的输出，错误随时间累积，表现为饱和度上升、画面质量退化。

#### Q&A: Teacher Forcing 是 AR 训练的标准吗？

**是的**，Teacher Forcing 是序列自回归建模中最经典、最广泛使用的标准训练范式。

- **LLM 领域**：从 GPT 到 LLaMA 到 DeepSeek，标准训练方式就是 Teacher Forcing——每个 token 的预测条件都是前面的 ground-truth token（next-token prediction）
- **传统 AR 视频生成**：VideoGPT、CogVideo 等基于 VQ tokenizer 的模型也使用 TF
- **AR 视频扩散模型**：Pyramidal Flow、NOVA、ACDiT、CA2 等工作均采用 TF 范式训练
- 论文消融实验中 **TF 的 VBench 始终优于 DF**（chunk-wise: 83.58 vs 82.95; frame-wise: 80.34 vs 77.24），作者明确指出"TF 普遍优于 DF 作为 AR 扩散预训练策略"

TF 唯一的"原罪"就是 exposure bias——训练时永远给模型完美 context，推理时却让模型依赖自己不完美的输出。

#### Q&A: Diffusion Forcing 是 Video-AR 训练的标准吗？

**不是**。DF 是一个有影响力的替代方案，但远未成为"标准"。

- **DF 的动机**：让每帧噪声水平独立采样，使训练 context 的噪声分布覆盖推理场景，试图通过扩大训练分布来缓解 exposure bias
- **实践中并非主流**：论文消融中 TF **始终显著优于** DF，尤其在 frame-wise 场景下差距更大（80.34 vs 77.24，差 3.1 分）。Pyramid Flow、SkyReels-V2 等近期工作的骨干网络仍用 TF 范式
- **DF 的固有问题**：牺牲时序一致性、使 KV cache 设计复杂化、增加推理延迟，且并未从根本上解决 exposure bias（训练时 context 仍是真实帧的加噪版本，不是自生成帧）
- **CausVid 的教训**：CausVid 用 DF + DMD 组合，但 DF 产生的输出与推理时的模型分布不匹配，DMD 实际上在匹配错误的分布
- **第三路线 Rolling Diffusion**：FIFO-Diffusion 等用渐进噪声调度，不严格遵循 AR chain-rule 分解，是独立于 TF/DF 的另一条技术路线

综上：**TF 是基础标准，DF 是一次改良尝试**，但 Self Forcing 的消融表明 DF 不如 TF 作为预训练方案，且 DF 自身的 train-test gap 仍存在。

#### Q&A: TF 和 DF 的区别只是 context 来源（干净 vs 加噪）吗？

**不只是**。"context 是否加噪"是表面区别，更深层差异在于**噪声时间步的采样方式**：

| | Teacher Forcing | Diffusion Forcing |
|---|---|---|
| Context 来源 | 干净 ground-truth $x^{<i}$ | 加噪 ground-truth $x^{j<i}_{t^j}$ |
| 时间步采样 | **所有帧共享同一个 $t$** | **每帧独立采样 $t^i$** |
| 训练分布 | 所有帧处于同一噪声水平 | 不同帧可处于不同噪声水平 |

**独立采样时间步才是 DF 的核心设计意图**。推理时 AR 场景天然是"干净 context（$t=0$）+ 噪声当前帧（$t \approx T$）"的组合。DF 的独立采样确保训练中有一定概率覆盖到这种组合——前面帧恰好采样到低噪声、当前帧采样到高噪声。TF 共享 $t$，训练时永远见不到这种组合。

**但 DF 仍未解决根本问题**：即使被加噪了，DF 的 context 帧**内容仍是 ground-truth 质量**——只是被噪声模糊。推理时 context 是**模型自己生成的内容**，可能包含结构性错误、伪影、失真。**"带噪声的完美帧" ≠ "干净的自生成帧"**。

这也是为什么消融中 frame-wise DF（77.24）大幅落后于 TF（80.34）——frame-wise 需要更多次 AR unrolling，模型对自生成 context 中的缺陷毫无准备，DF 的"加噪分幕"完全无法应对。

---

## 2. 核心方法

### 2.1 Self Forcing 训练范式

**核心思想**：训练时做自回归 self-rollout，每帧的去噪条件不再是 ground-truth 帧，而是**模型自己之前生成并缓存到 KV cache 的帧**。

$$x^{1:N}_\theta \sim p_\theta(x^{1:N}) = \prod_{i=1}^{N} p_\theta(x^i|x^{<i}_\theta)$$

三个关键设计保证效率：

1. **少步扩散模型（4-step）**：用均匀时间步 $[1000, 750, 500, 250]$，大幅减少展开长度
2. **随机梯度截断**：每次训练随机采样 denoise step $s \sim \text{Uniform}(1, ..., T)$，仅对第 $s$ 步的输出计算梯度，之前的去噪步骤 detach 梯度。同时 detach KV cache 中历史帧的梯度
3. **训练中使用 KV cache**：TF/DF 用特殊 attention mask 并行训练，Self Forcing 直接用 KV cache 做序列推演（无需特殊 mask），可用 FlashAttention-3 加速

**关键洞察**：Self Forcing 的每轮迭代时间与 TF/DF **可比甚至更优**——因为 TF/DF 的因果 mask 有额外开销，而 Self Forcing 用 full attention + KV cache，可利用高度优化的 kernel。

### 2.2 Holistic Distribution Matching Loss

因为 self-rollout 产生的是完整的生成视频 $\hat{x}^{1:N} \sim p_\theta$，可以对其施加**视频级别**的分布匹配损失：

$$\min_\theta \; D(p_{\text{data}}(x^{1:N}) \| p_\theta(x^{1:N}))$$

论文验证了三种散度：

| 损失函数 | 散度度量 | 是否需真实视频数据 | 14B 网络需求 |
|---------|---------|-----------------|------------|
| **DMD** (Distribution Matching Distillation) | Reverse KL | ❌ 数据自由 | ✅ 作为 real score network |
| **SiD** (Score Identity Distillation) | Fisher Divergence | ❌ 数据自由 | ❌ 1.3B 即可 |
| **GAN** (R3GAN, relativistic + R1/R2 reg) | JS Divergence | ✅ 需 70k 生成视频 | ❌ |

这与 CausVid 的根本区别：CausVid 用 **DF 产生的输出**去做 DMD，匹配的是错误的分布；Self Forcing 匹配的是真正的推理时分布。

### 2.3 Rolling KV Cache 用于长视频生成

三种视频外推方案的复杂度对比：

| 方案 | 复杂度 | 是否需要 recompute KV |
|-----|--------|---------------------|
| 双向 attention 滑动窗口 (TF/DF) | $O(TL^2)$ | ❌ 不支持 KV cache |
| 因果 attention + 窗口重叠 recompute | $O(L^2 + TL)$ | ✅ 窗口 shift 时重算 |
| **Rolling KV Cache (本文)** | $O(TL)$ | ❌ 固定大小 cache，FIFO 淘汰 |

**问题**：朴素 rolling KV cache 在第一个 latent frame（图像 latent，无时间压缩）被淘汰后出现 flickering 伪影。

**解决**：训练时限制 attention window，让模型学习在看不到第一个 chunk 的情况下也能去噪最后一帧，模拟长视频生成场景。

---

## 3. 算法流程

### Algorithm 1: Self Forcing Training

```
对每个训练迭代：
  1. 初始化 KV cache = []
  2. 随机采样 denoise stop step s ∈ [1, T]
  3. For i = 1 to N:  // 逐帧/chunk
       x^i_{t_T} ~ N(0, I)
       For j = T down to s:
         If j == s:
           开启梯度 → G_θ 去噪得到 x̂^i_0 → 关闭梯度
           将 x̂^i_0 的 KV 嵌入写入 cache
         Else:
           关闭梯度 → G_θ 去噪 → 加噪(forward process)进入下一去噪步
  4. 对生成的完整视频 x̂^{1:N} 计算分布匹配损失 → 更新 θ
```

关键：第 $s$ 步是所有帧都去噪到位的那一步，只有这步计算梯度；$s$ 的随机采样保证所有中间去噪步都能收到监督信号。

### Algorithm 2: Inference with Rolling KV Cache

```
1. KV cache 固定大小为 L 帧
2. For i = 1 to M:  // 生成 M 帧
     从噪声开始 T-step 去噪
     最后一步输出 clean frame → 写入 KV cache
     若 cache 满 → pop 最旧的
```

---

## 4. 实验与结果

### 4.1 主实验 (Table 1)

基于 Wan2.1-T2V-1.3B，$832 \times 480$，5s 视频 @16fps：

| 模型 | VBench Total | Quality | Semantic | FPS | Latency (s) |
|------|-------------|---------|----------|-----|-------------|
| Wan2.1-1.3B (双向扩散) | 84.26 | **85.30** | 80.09 | 0.78 | 103 |
| SkyReels-V2 (chunk AR) | 82.67 | 84.70 | 74.53 | 0.49 | 112 |
| CausVid (chunk AR) | 81.20 | 84.05 | 69.80 | **17.0** | 0.69 |
| **Self Forcing (chunk-wise)** | **84.31** | 85.07 | **81.28** | **17.0** | 0.69 |
| **Self Forcing (frame-wise)** | 84.26 | 85.25 | 80.30 | 8.9 | **0.45** |

- chunk-wise: 3 latent frames/chunk, VBench 超越原版 Wan2.1，同时速度提升约 **130×**
- frame-wise: 最低延迟 0.45s，适合交互式应用
- 用户偏好测试中一致优于所有 baseline（包括原版 Wan2.1）

### 4.2 消融实验 (Table 2)

| 配置 | Chunk-wise VBench | Frame-wise VBench |
|------|-------------------|-------------------|
| DF (50×2-step, MSE) | 82.95 | 77.24 |
| TF (50×2-step, MSE) | 83.58 | 80.34 |
| DF + DMD (≈CausVid) | 82.76 | 80.56 |
| TF + DMD | 82.32 | 78.12 |
| **SF + DMD** | **84.31** | **84.26** |
| **SF + SiD** | 84.07 | 83.54 |
| **SF + GAN** | 83.88 | 83.27 |

关键发现：
- Self Forcing 在三种 loss 下均稳定且均优于所有 baseline
- 其他方法从 chunk-wise 切换至 frame-wise 时显著退化（DF: -5.71, TF: -3.24），而 Self Forcing 几乎无退化（-0.05）
- TF 普遍优于 DF 作为 AR 扩散预训练策略

### 4.3 训练效率

- 每轮迭代时间：Self Forcing (s=1) < TF ≈ DF
- 相同 wall-clock time 下，Self Forcing 质量显著优于 TF/DF
- DMD 版本 64×H100 约 1.5 小时收敛，SiD/GAN 约 2-3 小时

### 4.4 Rolling KV Cache

- Recomputing KV: 4.6 FPS (10s 视频)
- Rolling KV (朴素): 15.8 FPS，但有伪影
- Rolling KV (本文训练方案): **16.1 FPS**，无伪影

---

## 5. 讨论与启示

1. **"并行预训练 + 串行后训练"范式**：Self Forcing 首次在视频领域实践了这一思路（LLM 中已有 RL post-training），兼顾预训练的可扩展性和推理时的分布对齐。

2. **AR、Diffusion、GAN 的互补性**：论文展示了三者如何有效融合——AR 提供 chain-rule 分解，Diffusion 提供条件生成，GAN 提供分布匹配目标。

3. **局限**：
   - 超出训练 context length 的长视频仍有质量退化
   - 梯度截断可能限制长程依赖学习
   - 未来方向：SSM 等循环架构、更好的外推技术

---

## 6. 关键实现细节

- **基础模型**：Wan2.1-T2V-1.3B (Flow Matching)，初始化自 16k ODE solution pairs fine-tuning
- **去噪步数**：4-step uniform schedule $[1000, 750, 500, 250]$
- **移步因子**：$k=5$ (time step shifting)
- **Prompt 数据**：VidProS 子集 (~250k 过滤后 prompts)，用 Qwen2.5-7B 扩展
- **Real score network**：DMD 用 14B 模型，SiD 用 1.3B 模型
- **训练硬件**：64×H100 (80GB)，per-GPU batch size=1
- **评估**：VBench + MovieGenBench (1003 prompts) 用户偏好测试
