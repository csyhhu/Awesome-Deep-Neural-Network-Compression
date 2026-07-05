# Light Forcing: Accelerating Autoregressive Video Diffusion via Sparse Attention

**论文信息**
- **标题**: Light Forcing: Accelerating Autoregressive Video Diffusion via Sparse Attention
- **作者**: Chengtao Lv, Yumeng Shi, Yushi Huang, Ruihao Gong, Shen Ren, Wenya Wang
- **机构**: Nanyang Technological University, HKUST, Beihang University, Sensetime Research, AUMOVIO
- **会议**: ICML 2026
- **链接**: https://arxiv.org/abs/2602.04789
- **代码**: https://github.com/chengtao-lv/LightForcing

---

## 1. 研究动机

自回归（AR）视频生成模型通过逐块（chunk-by-chunk）生成范式，在视觉保真度和交互性方面取得了显著进展。然而，时空 3D full attention 的**二次计算复杂度**仍然是高效部署的主要瓶颈。以 Self Forcing 1.3B 为例，当生成 480p 视频至第 14 个 chunk 时，注意力计算消耗的时间约占端到端延迟的 **~75%**。

现有的稀疏注意力方法主要针对**双向（bidirectional）视频扩散模型**设计。作者发现，将这些方法直接应用于 AR 模型会导致严重的性能退化，原因有二：

1. **忽略 chunk 的异质性贡献**：不同 chunk 对全局误差积累的贡献不同。当前 chunk 本质上是基于过去干净 chunk 预测下一噪声水平，因此后期 chunk 会继承早期 chunk 的质量。
2. **对过去关键上下文利用不足**：不同层、注意力头和去噪时间步对历史信息的关注点不同，滑动窗口等方法不可避免地丢弃关键信息，损害长程一致性和运动丰富度。

---

## 2. 核心方法

Light Forcing 提出两个互补的核心组件：

### 2.1 Chunk-Aware Growth (CAG) 机制

**核心思想**：早期 chunk 对后续生成质量起"视觉锚点"作用，应分配较低的稀疏度；后期 chunk 可容救更高的稀疏度。

- 从理论角度，chunk $i$ 的生成质量与去噪步数 $T$（Term 1: $\propto 1/\sqrt{T}$）和分数估计误差 $\varepsilon_{\text{score}}$（Term 2）相关
- 基于 Total Variation 上界公式建立 chunk 的难度估计
- 对第一个 chunk 保持 dense attention，对后续 chunk $i>1$ 分配稀疏度：
  $$s_i = s_{base} - \alpha_i \beta$$

其中 $\alpha_i$ 随 chunk 索引缩放（$\propto 1/\sqrt{T}$），$\beta$ 通过 FLOPs 约束求解，确保整体计算量与目标稀疏率一致。

### 2.2 Hierarchical Sparse Attention (HSA)

**核心思想**：采用从粗到细（coarse-to-fine）的 Pipeline，在帧级别和块级别两个粒度上选择稀疏注意力掩码。

三阶段流程：
1. **Token Compression**：对 query 做块级池化，对 key 做块级+帧级池化
2. **Mask Selection**（两级）：
   - **帧级**：对每个 query 块，在压缩后的帧级表示上做 Top-K 选择
   - **块级**：在选中的帧内，进一步在块级表示上做 Top-K 选择（含始终保留的 sink 帧、最近邻帧和当前 chunk 帧）
3. **Blockwise Sparse Attention**：基于两级掩码执行高效块稀疏注意力计算

HSA 将注意力复杂度与历史帧总数解耦，保持固定的计算预算。

#### HSA 概念辨析

**Q1: "块（block）"是 chunk-by-chunk 中的 chunk 吗？**

**不是**，这是两个完全不同层级的概念：

- **Chunk**：自回归视频生成中的**生成单元**，一个 chunk 包含**多个帧**（论文中为 3 个 latent 帧）。模型一次生成一个 chunk，然后自回归地生成下一个。这是**时间轴上的划分**。

- **Block**：注意力计算中为了 GPU 硬件效率而做的**序列切分**。将 query/key/value 的 token 序列按固定大小（论文设 block size = 64）分成多个 block，然后以 block 为粒度进行稀疏注意力（见论文 Section "Blockwise Sparse Attention"）。这是**空间/序列维度的划分**。

**层级关系**：`Chunk → Frame → Token → Block`（多个 token 组成一个 block）

---

**Q2: HSA 第一阶段的产出是什么？**

HSA 的 Mask Selection 分两级，第一阶段是**帧级粗选**：

- **输入**：对当前 chunk $i$ 的每个 query block $\tilde{\vq}^{(i)}_r$，与所有候选过去帧的**帧级压缩 key** $\hat{\vk}^{\mathcal{M}}$ 做内积，得到帧级相关性分数 $p^{(i)}_r$

- **产出**：为每个 query block $r$ 选出一个**帧集合** $\mathcal{T}_r$：
  $$\mathcal{T}_r = \text{TopK}_{\text{idx}}(p^{(i)}_r) \cup \mathcal{F}^{(i)}$$

  其中 $\mathcal{F}^{(i)}$ 是始终保留的帧集：
  - 少数 **sink 帧**（最早的几个历史帧）
  - **最近邻帧**（滑动窗口内的帧）
  - **当前 chunk $i$ 自身的帧**（保证 chunk 内时间依赖）

- 第二阶段（块级精选）在 $\mathcal{T}_r$ 选中的帧内，进一步做**块级 Top-K 选择**，得到最终要参与注意力计算的 block 集合 $\mathcal{J}_r$。

---

**Q3: 完整流程是怎样的？是否有 token 级别的选择？**

完整流程可总结为：

```
对于当前 chunk 的每个 query block r：
  ├─ Step 1 Token Compression:
  │    ├─ query 做块级池化 → q̃^(i)_r（每个 query block 一个向量）
  │    ├─ key 做块级池化    → k̃^(τ)_j（每帧的每个 key block 一个向量）
  │    └─ key 再做帧级池化  → k̂^M（每个过去帧 → 一个向量）
  │
  ├─ Step 2 Mask Selection:
  │    ├─ 帧级: 用 q̃^(i)_r × k̂^M → Top-K 帧 → T_r
  │    └─ 块级: 在 T_r 内用 q̃^(i)_r × k̃^(τ)_j → Top-K key blocks → J_r
  │
  └─ Step 3 Blockwise Sparse Attention:
       用原始高精度 qr × (选中的 key blocks) → 注意力输出
```

关键点：
- 选中的是 **key block**（key 侧被关注的分块），而非 query token
- query 侧始终是**当前 chunk 的全部 token**（按 block 组织）
- 本质是回答"**哪些历史信息对这个 query block 重要**"——先粗选帧，再精确定位到帧内的具体 token block
- 选择过程的额外开销极小（约 2% 的端到端运行时间）

---

## 3. 实验与结果

### 3.1 实验设置
- **模型**: Self Forcing 1.3B, LongLive 1.3B, Infinite-Forcing
- **基准**: VBench（5s 视频, 16 维度）和 VBench-Long（15s 视频）
- **对比方法**: STA, Radial Attention, SVG2, VMoBA, SLA
- **微调**: 对 VMoBA, SLA, Light Forcing 在预训练权重基础上额外训练 2000 步
- **推理**: 基于 SpargeAttention kernel，RTX 5090 GPU

### 3.2 主要结果

**Self Forcing 1.3B 上的 VBench 结果**：

| 方法 | 延迟(s) | 加速比 | 总得分 |
|------|---------|--------|--------|
| Dense (FlashAttention2) | 9.61 | 1.00× | 84.1 |
| STA | 8.27 | 1.16× | 83.6 |
| Radial | 7.39 | 1.30× | 73.7 |
| SVG2 | 21.38 | 0.45× | 82.8 |
| VMoBA | 7.42 | 1.29× | 83.6 |
| SLA | 7.71 | 1.25× | 83.2 |
| **Light Forcing** | **7.39** | **1.30×** | **84.5** |

- Light Forcing 端到端加速 1.3×，注意力部分加速 **3.79×**
- 总得分 84.5 **超越** Dense Attention 的 84.1

**LongLive 1.3B 上**：Light Forcing 达到 1.19× 加速，总得分 83.9 vs Dense 83.2。

**15s 长视频生成**（Infinite-Forcing）：
- Light Forcing 总得分 84.1 vs Dense 83.6
- 质量得分从 84.6 提升至 85.4

### 3.3 高效部署

结合优化后，进一步加速：

| GPU | 5s 视频 | 15s 视频 |
|-----|---------|----------|
| RTX 5090 | 3.07× (27.4 FPS) | 3.17× |
| A100 | 2.35× | 2.12× |
| H100 | 2.01× (33.9 FPS) | 1.98× |

优化技术栈：FP8 线性层 + 高效 Kernel（RoPE/RMSNorm/fuse_scale_shift）+ LightVAE。

### 3.4 消融实验

| 变体 | 总得分 |
|------|--------|
| 直接 1D Sparse Attention (90%) | 严重崩溃 |
| +Fine-tuning | 82.8 |
| +Fine-tuning + CAG | 83.x (质量和美学提升，动态性下降) |
| +Fine-tuning + CAG + HSA (Light Forcing) | **84.5** |

### 3.5 HSA 超参数敏感性

对 $topk \in \{6, 9, 12\}$ 均表现稳定，表明每个 query 块只需关注少数过去帧即可缓解不一致问题。

---

## 4. 局限性

1. 仅在 1.3B 模型上验证，扩展至更大模型（如 14B）是重要方向
2. HSA 虽能缓解白色带状伪影，极少数样本中仍有残留
3. 部分 kernel fusion 在特定 GPU（如 A100）上加速效果有限
4. 稀疏性与极少量去噪步骤（1-3 步）、低比特量化等方法的组合尚未探索

---

## 5. 总结

Light Forcing 是**首个**专门为自回归视频生成模型设计的稀疏注意力解决方案。通过 Chunk-Aware Growth（宏观分配稀疏度预算）和 Hierarchical Sparse Attention（微观检索关键上下文），在保持甚至超越 Dense Attention 生成质量的同时，实现显著的加速效果。结合高效部署优化后，首次在消费级 GPU（RTX 5090）上以 27.4 FPS 实现实时视频生成。
