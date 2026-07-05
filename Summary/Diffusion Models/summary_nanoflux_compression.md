# NanoFLUX: 面向移动设备的大规模文生图模型蒸馏压缩

> **论文标题**: NanoFLUX: Distillation-Driven Compression of Large Text-to-Image Generation Models for Mobile Devices  
> **作者**: Ruchika Chavhan, Malcolm Chadwick, Alberto Gil Couto Pimentel Ramos, Luca Morreale, Mehdi Noroozi, Abhinav Mehrotra  
> **机构**: Samsung AI Center, Cambridge  
> **发表**: ICML 2026  
> **arXiv**: 2602.06879

---

## 1. 核心动机

当前最先进的文生图（T2I）扩散模型（如 FLUX.1 系列，17B 参数；Qwen-Image，20B 参数）虽然生成质量极高，但巨大的计算开销使其只能在昂贵 GPU 或云端运行，严重限制了普通用户的使用。本文目标是将 **17B 的 FLUX.1-Schnell** 压缩为可在移动设备上运行的 **2.4B 模型（NanoFLUX）**，同时尽可能保持生成质量。

---

## 2. 方法总览

NanoFLUX 采用**渐进式蒸馏压缩流水线**，分别压缩 FLUX.1-Schnell 的两个最重组件：

| 组件 | 教师模型 | 学生模型 | 压缩比 |
|------|---------|----------|--------|
| Diffusion Transformer (DiT) | 12B | 2B | 6× |
| Text Encoder | T5-XXL (5B) | T5-Large (330M) | 15× |
| **总计** | **17B** | **2.4B** | **~7×** |

---

## 3. DiT 压缩四阶段（12B → 2B）

### 阶段 C1：注意力头剪枝（12B → 5B）

- **核心观察**：通过 SVD 对每个注意力头输出 `softmax(QK^T)V` 进行低秩分析，发现保留 `r=16` 个奇异分量（共 H=24 个头）即可重建高质量图像，表明 attention head 高度冗余。
- **操作**：注意力头数从 24 减至 16，head 维度 d_H=128 不变，模型维度从 3072 降至 2048。学生用教师的前 16 个 head 初始化。
- **损失函数**：知识蒸馏损失 + 注意力头特征损失（Head-level feature loss）：
  $$\mathcal{L}_{\mathrm{features}} = \sum_{l=1}^L \Bigg\| \frac{1}{H_T}\sum_{h=1}^{H_T} o_T^{l, h} - \frac{1}{H_S}\sum_{h=1}^{H_S} o^{l, h}_S\Bigg\|$$

  其中 `o` 是每个 head 的 `softmax(Q_h K_h^T)V_h`，即 **output projection（W_O）之前**的 per-head attention 输出（shape: tokens × d_H）。论文对每个 token 取平均，得到该 head 的整体特征表示。由于 head 数从 24→16 后维度变了，常规 block-level feature distillation 不适用，因此设计了这一 head-level 损失。

- **训练细节**：γ=1，275 epochs，lr=1e-4，40 H100 GPU。
- **推理步数**：压缩后需从 4 步增加到 8 步以保证质量（低容量模型需要更多推理步数）。

### 阶段 C2：特征维度缩减（5B → 3B）

- 在 5B 模型上再次进行 SVD 分析（按 head 内特征），保留 `r=96` 个奇异分量。
- **操作**：head 维度从 128 降至 96，模型维度从 2048 降至 1536。
- 先用 5B 模型蒸馏 50 epochs，再用 12B FLUX.1-Schnell 蒸馏 100 epochs（cosine annealing）。

### 阶段 C3：深度剪枝 / Block Merging（3B → 2.5B）

- **核心观察**：计算各 transformer block 输入-输出余弦相似度，发现 SS（Single-Stream）块 7~23 的相似度均 >0.85（图像）和 >0.9（文本），表明大量连续 SS 块功能冗余。
- **操作**：将 15 个高相似度 SS 块**参数平均合并**为 1 个 block（非直接删除），从 38 个 block 减至 24 个。
- **消融实验**：Block Merging 优于 Prune 和 IterPrune，且只对 SS 块操作有效（对 DS 块操作会显著降低质量）。

### 阶段 C4：静态层归一化 Static-LN（2.5B → 1.8B）

- **核心观察**：AdaLN 层约占 3.2B 参数，跨 1000 样本分析发现预测系数 variance-to-norm 比值极低，说明不同样本间系数变化很小。
- **操作**：用少量校准集预计算所有 1000 个 timestep 的平均归一化系数，替换动态 AdaLN。
- 只需 2 个样本即可达到可比较性能，移除 0.7B 参数且几乎**无需额外训练**。Fine-tune 后质量完全恢复。

---

## 4. 渐进式 Token 下采样（PTD）

### 动机
自注意力是 FLUX 推理的主要延迟瓶颈（O(n²)）。已有工作（token pruning/merging, linear attention）在大规模 MMDiT 上效果有限。原始 FLUX.1-Schnell 是纯 MMDiT 架构，所有层都在相同 token 分辨率下运行，**没有任何 UNet 式的多尺度上下采样结构**。

### 方法
- **ResNet 下采样器 D**（本文新增）：包含 ResNet block + stride-2 的 3×3 卷积，将 token 数减少 4 倍。
- **ResNet 上采样器 U**（本文新增）：包含 ResNet block + 双线性插值 + 卷积 → 再一个 ResNet block。
- 上下采样器均含 timestep-dependent 的 scale/shift conditioning，通道数等于模型特征维度 d=1536。
- **插入位置**：下采样器放在第一个 SS block 后，上采样器放在最后一个 SS block 前。43 个 block 中有 23 个操作在低分辨率空间。DS block 保持高分辨率处理以保留精细信息。
- **混合 RoPE**：高/低分辨率 block 分别使用高/低频位置编码。
- **时间步阈值机制**：设定 `t_thresh=0.5`，前半段采样用低分辨率（捕捉全局结构），后半段用全分辨率（恢复细节）。消融实验表明 t_thresh=0.5 是延迟-质量最优平衡点。
- **渐进式训练**：从一开始直接插入 D 和 U 会导致训练不稳定。因此采用渐进策略：先将 D 放在最后一个 block 前、U 放在其后，只训练 D、U 和相邻 block；之后逐步将 D 向前移动，每次只训练新集成的 block + D。U 和 projection 层在初始阶段后冻结。每个阶段用知识蒸馏损失训练 80 epochs (lr=1e-4)，全部 block 集成后再做 20 epochs 端到端微调。每次迭代需 48 H100、~10 小时。
- **参数增量**：D 和 U 为模型增加约 **200M** 参数，但减少的 attention 计算量使总延迟显著下降。

---

## 5. 文本编码器蒸馏（5B → 330M）

### 背景：FLUX 的双文本编码器

FLUX.1-Schnell 使用两个文本编码器：
- **T5-XXL（5B）**：生成 token-level prompt embeddings，与图像 token 拼接后进入 MMDiT 的 joint attention
- **CLIP-Large（120M）**：生成 pooled representation，与 timestep embedding 结合后送入 AdaLN 层

由于 CLIP-Large 已经很轻量（120M），压缩重点放在 T5-XXL → T5-Large（330M，压缩 15×，仅原大小的 6.6%）。

### 已有方法的局限

已有文本编码器蒸馏工作主要分两类：

| 方法 | 策略 | 问题 |
|------|------|------|
| `scaling-down-t5` | 每个去噪步都对最终输出做蒸馏 | 高噪声区域梯度不稳定，训练震荡 |
| Neodragon | 在 transformer block 前用浅层 embedding 做蒸馏 | 未解决后续层中的误差累积 |

### 本文的核心洞察

得益于 MMDiT 的 **joint attention** 机制，image token 会直接 attend prompt token，因此 **中间的 prompt hidden states 天然编码了视觉线索**，且比 image features 的高频噪声更少。基于此，本文提出利用 transformer **中间层**的 prompt hidden states 作为蒸馏信号。

### 两阶段训练策略（详见 Algorithm 1）

**阶段一：Student Warm-up**

- 在 T5-Large 输出端接一个**两层 MLP**，将其输出维度匹配到 T5-XXL 的维度
- 最小化教师和学生 prompt embeddings 之间的 MSE：
  $$\mathcal{L}_{\text{init}} = \|p_T - p_S\|_2^2$$

**阶段二：Block-wise 蒸馏 + 去噪 Rollout**

这是本文的核心创新，流程如下：

1. **冻结 DiT 权重**，只训练 T5-Large + MLP
2. 采样随机噪声 $x_T \sim \mathcal{N}(0,I)$
3. 采样**随机截止时间步 $\hat{t} \sim \mathcal{U}(\{1,\dots,T\})$**
4. 从 $t=T$ 到 $t=1$ 执行去噪 rollout：
   - 使用学生编码器做一次 forward，收集所有块的 prompt hidden states $\{h_t^i(p_S)\}$
   - 使用教师编码器做一次 forward，收集各块 prompt hidden states $\{h_t^i(p_T)\}$
   - **关键机制**：对 $t < \hat{t}$（高噪声阶段），对学生的 prompt states 执行 `stopgrad`，梯度不反向传播，避免高噪声信号干扰训练
   - 对 $t \ge \hat{t}$（低噪声阶段），累积 block-wise MSE 损失：
     $$\mathcal{L}_{\text{block}} = \sum_{i=1}^L \alpha_i \cdot \|h_t^i(p_S) - h_t^i(p_T)\|_2^2$$
   - 用学生预测的速度场 $\hat{v}_t^S$ 进行 Euler 更新：$x_{t-1} = x_t + \Delta\tau_t \cdot \hat{v}_t^S$
5. 用 $\mathcal{L}_{\text{block}}$ 更新学生编码器参数

### 为什么只需监督前 3 层？

论文设置 $\alpha_1=\alpha_2=\alpha_3=0.1$，其余 $\alpha_i=0$，即**仅监督前 3 个 transformer 层的 prompt hidden states**。原因：

- 在第 3 个 Double-Stream block 中发现 **"super-weight" 现象**（引用自 yu2025）：该 block 会使 prompt hidden states 的范数急剧增大
- 从 block similarity 分析（Figure 6）可验证：DS 块中输入-输出余弦相似度从第 2 层的 ~0.3 跳变到第 3 层的 ~1.0
- **直觉**：在第 3 层匹配 prompt 范数对于限制后续层的误差积累至关重要，一旦前几层对齐了，后续层的误差就不会放大

这种设计大幅减少了训练开销——相比 `scaling-down-t5` 需要在每一步都施加损失，本文只需监督前 3 层。

### 关键参数

- 所有实验用 **Step C3 的 2.5B DiT** 作为冻结的 diffusion transformer 进行蒸馏
- 蒸馏后的 T5-Large **可直接迁移**到 Step C4 的 1.8B 模型，性能无明显下降
- 文本编码器推理仅需 **15ms**（Samsung S25U）

### 消融对比

| 方法 | 2.5B DiT HPSv3 | 1.8B DiT HPSv3 |
|------|---------------|---------------|
| `scaling-down-t5` (每步输出蒸馏) | 10.37 | 10.21 |
| 仅阶段一 (warmup) | 10.42 | 10.40 |
| **完整两阶段 (Ours)** | **10.45** | **10.45** |

> 与使用 T5-XXL 的 2.5B 模型（HPSv3=10.68）相比，蒸馏后仅损失 ~0.23 HPSv3，而文本编码器缩小了 15×。

---

## 6. 实验结果

### 主要指标对比

| 模型 | DiT | 文本编码器 | 步数 | One-IG↑ | DPG↑ | GenEval↑ | HPSv3↑ |
|------|-----|-----------|------|---------|------|----------|--------|
| FLUX.1-Schnell | 12B | T5-XXL(5B) | 4 | 43.1 | 84.3 | 66.0 | 11.45 |
| SANA-1.5 | 1.6B | Gemma(2.6B) | 20 | 42.6 | 84.3 | 65.8 | 10.59 |
| SANA Sprint | 1.6B | Gemma(2.6B) | 2 | 43.2 | 63.9 | 73.0 | 10.26 |
| **NanoFLUX** | **2B** | **T5-Large(330M)** | **10** | **42.1** | **75.5** | **49.7** | **10.41** |

### 各压缩阶段 DiT 性能变化

| 阶段 | 参数量 | 步数 | One-IG↑ | DPG↑ | GenEval↑ | HPSv3↑ |
|------|--------|------|---------|------|----------|--------|
| FLUX.1-Schnell | 12B | 4 | 43.1 | 84.3 | 66.0 | 11.45 |
| C1 (Head Prune) | 5B | 8 | 46.8 | 83.8 | 62.4 | 11.04 |
| C2 (d_H Reduce) | 3B | 10 | 43.2 | 82.3 | 54.1 | 10.74 |
| C3 (Block Merge) | 2.5B | 10 | 43.2 | 82.6 | 53.5 | 10.68 |
| C4 (Static-LN) | 1.8B | 10 | 43.2 | 82.4 | 53.1 | 10.60 |

### 端侧延迟（Samsung S25U / Snapdragon 8 Elite）

| 模型 | 步数 | 去噪延迟 | 
|------|------|----------|
| 12B (FLUX) | 4 | 14.00s |
| 5B | 8 | 4.56s |
| 3B | 10 | 3.70s |
| 2.5B | 10 | 2.80s |
| 1.8B | 10 | 2.75s |
| 2.5B + PTD | 10 | 2.45s |
| **1.8B + PTD (NanoFLUX)** | **10** | **2.40s** |

> 加上 VAE Decoder（160ms）和 T5-Large（15ms），NanoFLUX 端到端生成一张 512×512 图像约 **2.5 秒**。

---

## 7. 训练数据与评估

- **训练数据**：Ye-PoP 数据集（~480K 图像），用 FLUX.1-Schnell 教师模型重新生成高质量图像，并用 Qwen-Image 重标注（short/medium/tags/long 四种描述）。
- **评估基准**：One-IG、DPG、GenEval、HPDv3。

---

## 8. 主要贡献总结

1. **首个大规模文生图模型系统性压缩框架**：将 17B FLUX.1-Schnell 压缩至 2.4B（~7×），建立渐进式蒸馏流水线。
2. **Block Merging**：通过参数平均合并冗余 block，优于直接剪枝。
3. **Static-LN**：发现 AdaLN 系数跨样本方差极低，用预计算均值替换可移除 0.7B 参数而几乎不影响质量。
4. **Progressive Token Downsampling (PTD)**：结合 ResNet 下/上采样与时间步阈值机制，在保持质量的同时显著降低延迟。
5. **Block-wise 文本编码器蒸馏**：利用早期 MMDiT 层的视觉信号进行蒸馏，仅需监督前 3 层即可高效压缩文本编码器。
6. **移动端部署验证**：在 Samsung S25U 上实现 2.5 秒生成 512×512 图像，证明高质量端侧文生图的可行性。

---

## 9. 关键启示

- **低容量模型需要更多推理步数**：12B 只需 4 步，1.8B 需要 10 步来弥补容量损失。
- **SS 块 vs DS 块的冗余模式不同**：SS 块高度冗余适合合并，DS 块保留全局结构不宜大幅压缩。
- **Block Merging > Block Pruning**：参数平均保留了被合并 block 的知识，优于直接删除。
- **AdaLN 条件信号的贡献有限**：时间步嵌入 + CLIP 池化特征对归一化系数的跨样本变化极小，可静态化。
- **文本编码器蒸馏的 "super-weight" 现象**：第 3 层 prompt 范数跳变是关键，匹配此层可有效限制误差传播。

---

## 10. Q&A（阅读讨论）

**Q1: C1 中 head-level feature loss 的 `o` 是 attention 的结果吗？**

是的。`o` 指每个 head 的 `softmax(Q_h K_h^T) V_h`，即**在 output projection（W_O）之前**的 per-head attention 输出（shape: tokens × d_H）。论文的 SVD 分析也是在这一层进行的，用 "prior to the output projection" 明确描述。选择这一层做分析和蒸馏的原因是：每个 head 特征独立（d_H=128），head 间冗余性直接体现在特征线性相关性上，比经过 W_O 混合后分析更准确。

**Q2: FLUX 中在哪里使用 ResNet 上下采样？还是本文新增的？**

ResNet 上下采样是本文新增的模块，**原始 FLUX 中没有**。FLUX.1-Schnell 是纯 MMDiT 架构，所有层在同一 token 分辨率下运行，没有 UNet 式的多尺度设计。已有工作如 U-DiTs 在 DiT 中引入了下采样，但仅限于小规模 class-conditional 生成。本文是**首次在 MMDiT + 大规模文生图**场景下引入 ResNet 上下采样（Progressive Token Downsampling），将 token 数减少 4 倍并将 23/43 个 block 切换到低分辨率空间。

**Q3: CLIP-Large 的输入是什么？**

CLIP-Large 的输入和 T5-XXL 一样，都是**同一个文本 prompt**（用户输入的文本描述），但两者的输出和使用方式完全不同：T5-XXL 输出 token-level embeddings（序列，与 image token 拼接后进入 joint attention）；CLIP-Large 输出 **pooled representation**（单个向量 `p_CLIP ∈ ℝ^d`，与 timestep embedding 相加后送入 AdaLN 层生成 scale/shift 系数）。论文原文（preliminaries.tex）描述为 "CLIP provides a pooled representation from its final hidden state"。即同一个 prompt 被两个编码器分别处理：T5 提供细粒度 token 级语义（用于 attention），CLIP 提供全局语义摘要（用于归一化调制）。

**Q4: 双文本编码器（T5 + CLIP）范式是 FLUX 首创的吗？其他 image generation 方法也这么设计吗？**

**不是 FLUX 首创**。这一设计继承自 **Stable Diffusion 3**（Esser et al. 2024, "Scaling Rectified Flow Transformers"）。SD3 首次提出 MMDiT 架构，同时使用 T5-XXL + CLIP-L（以及可选的 CLIP-G）作为双文本编码器，FLUX 由同一团队（Black Forest Labs，即原 SD 团队）开发，沿用了这一定位。

其他工作的文本编码器设计：

| 模型 | 文本编码器 | 设计思路 |
|------|-----------|----------|
| SD1.x / SD2.x | CLIP-L (123M) | 单编码器，cross-attention 注入 |
| **SDXL** | **CLIP-L + OpenCLIP-G** | 双 CLIP 编码器，但仍为 cross-attention 模式 |
| **SD3** | **T5-XXL + CLIP-L + CLIP-G** | 首创 T5+CLIP 双编码器 + MMDiT joint attention |
| **FLUX.1** | **T5-XXL + CLIP-L** | 继承 SD3 范式，去掉 CLIP-G |
| DALL-E 3 | T5-XXL | 单编码器 |
| PixArt-α | T5 | 单编码器 + cross-attention |

关键区别在于 **SD3/FLUX 的双编码器不仅仅是有两个编码器**，而是两个编码器的输出作用于模型的不同路径：T5 做 token-level joint attention，CLIP 做 pooled AdaLN conditioning。SDXL 虽然也有双 CLIP，但两者输出都走 cross-attention，没有这种"细粒度+全局摘要"的分工设计。

**Q5: Block-wise 蒸馏中，高噪声阶段 prompt states 被 stopgrad，学生如何训练？**

关键在于 **stopgrad 只切断了 prompt hidden states 的梯度通道，velocity（速度场预测 `v̂_t^S`）的梯度通道仍然保留**。学生通过两条路径学习：

```
高噪声步 (t < t̂):
  p_S → DiT → v̂_t^S → x_{t-1} → ... → L_block    ✅ 梯度畅通（Euler 链）
  p_S → DiT → h_t^i(p_S) [stopgrad]               ❌ 梯度阻断

低噪声步 (t ≥ t̂):
  p_S → DiT → v̂_t^S → x_{t-1} → ... → L_block    ✅ 梯度畅通
  p_S → DiT → h_t^i(p_S) → L_block                ✅ 直接监督
```

低噪声步有**双重梯度**（velocity 链 + hidden states 直接监督），高噪声步只有 velocity 链的隐式优化。损失从低噪声步的 `L_block` 出发，沿 Euler 更新链反向传播，经过每一帧的 `v̂_t^S` → DiT → `p_S`，最终传导到 T5-Large+MLP。

**Q6: 但 velocity 来自 DiT，DiT 不是已被冻结了吗？**

**冻结 DiT ≠ 梯度不能流经 DiT**。冻结只意味着 DiT 的参数不更新，但其计算图仍然可微。DiT 在此充当一个固定的非线性函数，梯度可以穿过它继续回传：

```
L_block → h_t^i(p_S) → DiT_frozen(x_t, p_S) → p_S → MLP → T5-Large (trainable)
```

这是特征蒸馏（feature distillation）的标准范式——冻结的教师网络作为"梯度管道"，将监督信号从中间层特征路由回学生。PyTorch 中 `model.eval()` + `torch.no_grad()` 才阻断梯度，仅冻结参数（`requires_grad=False`）不阻断反向传播。

**Q7: 高噪声下，T5-Large+MLP 的梯度来自 DiT 而不来自自身的 hidden states？**

是的。精确对比：

| 时间步 | 梯度来源 |
|--------|----------|
| t < t̂ | **仅 velocity 链**：`L_block → ... → v̂_t^S → DiT → p_S → T5-Large` |
| t ≥ t̂ | **双重**：velocity 链 + `h_t^i(p_S)` 直接对齐 `h_t^i(p_T)` |

高噪声步的梯度完全依赖 velocity 预测质量——学生编码器仍然被"考核"：它生成的 `p_S` 是否能让 DiT 输出合理的 velocity，从而让后续低噪声步的 prompt hidden states 对齐教师目标。

**Q8: `h_t^i(p_S)` 的监督信号来自 teacher？**

是的。`h_t^i(p_T)` 是教师（T5-XXL）生成的 prompt hidden states，作为蒸馏目标：

```
教师：prompt → T5-XXL → p_T → DiT(x_t, p_T, t) → h_t^i(p_T)  ← 目标（无梯度）
学生：prompt → T5-Large+MLP → p_S → DiT(x_t, p_S, t) → h_t^i(p_S) ← 需对齐

L_block = Σ α_i · ||h_t^i(p_S) - h_t^i(p_T)||²    （仅 t ≥ t̂）
```

在高噪声阶段（大量纯噪声、joint attention 尚未建立有意义的视觉关联），教师自己的 `h_t^i(p_T)` 也不可靠，用它做监督会引入噪声梯度。因此论文选择高噪声步完全丢弃这一直接对齐信号，只保留 velocity 链的隐式优化。
