# Q-Sched: Pushing the Boundaries of Few-Step Diffusion Models with Quantization-Aware Scheduling

## 论文信息

- **标题**: Q-Sched: Pushing the Boundaries of Few-Step Diffusion Models with Quantization-Aware Scheduling
- **作者**: Natalia Frumkin, Diana Marculescu (The University of Texas at Austin)
- **会议**: NeurIPS 2025
- **arXiv**: 2509.01624v1
- **代码**: https://github.com/enyac-group/q-sched

---

## 核心贡献

Q-Sched 提出了一种全新的后训练量化（PTQ）范式：**不修改模型权重，而是修改扩散模型的 scheduler**。通过引入两个可学习的预条件系数和参考无关的图像质量损失函数 JAQ，Q-Sched 在仅需少量标定提示（5个）的情况下，实现量化后的少步扩散模型达到甚至超越全精度模型的性能。

### 主要贡献

1. **量化感知调度器 (Q-Sched)**：引入两个标量预条件系数 \( c_x \) 和 \( c_\epsilon \)，分别应用于采样轨迹中的 \( x_t \) 和噪声预测 \( \epsilon_\theta^Q \)，通过修改 TCD scheduler 的采样过程来补偿量化误差。

2. **JAQ (Joint Alignment Quality) 损失函数**：将文本-图像一致性指标（如 CLIPScore）与图像质量指标（如 CLIP-IQA）相结合，提出参考无关的优化目标：
   \[
   \text{JAQ}(x) = \text{TC}(x) + k \cdot \text{IQ}(x)
   \]

3. **大规模用户偏好研究**：收集了超过 80,000 条人类标注，证明 Q-Sched 在感知质量上优于 MixDQ 和 SVDQuant。

---

## 方法详解

### 动机

少步扩散模型（如 LCM、PCM）将推理步数从 40-1000 步蒸馏至 2-8 步，显著提升了推理速度。但这些模型仍然依赖大型 U-Net 或 DiT 骨干网络，推理成本仍然很高。现有的后训练量化方法通常需要全精度模型进行标定，这在资源受限场景下不可行。

更重要的是，蒸馏后的少步扩散模型对量化极其敏感，因为量化破坏了原本脆弱的 ODE/SDE 采样轨迹。传统的量化方法试图将量化模型的分布矫正回全精度模型，但 Q-Sched 另辟蹊径——直接学习一个适应量化噪声的新采样轨迹。

### Q-Sched 调度器

原始的 TCD Strategic Stochastic Sampling (SSS) 采样公式为：

\[
\mathbf{x_s} = \frac{\alpha_s}{\alpha_{s'}} \Big(\alpha_{s'}\frac{\mathbf{x_t} - \sigma_t \mathcal{E}^Q_\theta(x_t, t)}{\alpha_t} + \sigma_{s'}\mathcal{E}^Q_\theta(x_t, t) \Big) + \eta \mathbf{z}
\]

Q-Sched 引入两个可学习系数后：

\[
\mathbf{x_s} = \frac{\alpha_s}{\alpha_{s'}} \Big(\alpha_{s'}\frac{c_x\mathbf{x_t} - \sigma_t c_{\epsilon}\mathcal{E}^Q_\theta(x_t, t)}{\alpha_t} + \sigma_{s'}c_{\epsilon}\mathcal{E}^Q_\theta(x_t, t) \Big) + \eta \mathbf{z}
\]

通过网格搜索优化 \( c_x \) 和 \( c_\epsilon \)，以 JAQ 损失为目标找到最优采样轨迹。

### JAQ 损失函数

- **TC(x)**: 文本-图像一致性指标（CLIPScore 或 AQ-MAP）
- **IQ(x)**: 纯图像质量指标（CLIP-IQA 或 HPSV2）
- **k = 2**: 平衡两个目标的超参数

JAQ 的优势在于完全不需要参考图像或全精度模型输出，只需少量标定提示即可。

### 与 PTQD 的区别

| 维度 | PTQD | Q-Sched |
|------|------|---------|
| 核心理念 | 校正量化模型分布以匹配全精度模型 | 学习全新的量化感知采样轨迹 |
| 标定需求 | 需要 1024 张全精度模型输出的图像 | 仅需 5-20 个标定提示 |
| 假设前提 | 量化误差可线性建模 + 高斯噪声假设 | 无分布假设 |
| 优化目标 | 均值/标准差校正 | JAQ 图像质量指标 |

---

## 实验与结果

### 实验设置

- **量化方案**: W4A8（权重4bit、激活8bit，模型缩小4倍）和 W8A8
- **评估模型**: LCM、PCM（Stable Diffusion v1-5/XL 骨干）、SDXL-Turbo、FLUX.1[schnell]
- **评估指标**: FID、CLIPScore、FID-SD、ELO（用户偏好评分）
- **硬件**: 单张 Nvidia A6000，Q-Sched 优化仅需约 20 分钟

### Consistency Models 上的主要结果

| NFEs | Precision | Schedule | PCM FID | LCM FID |
|------|-----------|----------|---------|---------|
| 2 | FP16 | Original | 24.17 | 38.74 |
| 2 | W4A8 | Q-Sched | **22.24** | **32.50** |
| 4 | FP16 | Original | 23.29 | 31.94 |
| 4 | W4A8 | Q-Sched | **17.39** | **26.98** |
| 8 | FP16 | Original | 20.15 | 27.34 |
| 8 | W4A8 | Q-Sched | 16.83 | **25.82** |

**关键发现**：
- Q-Sched 的 W4A8 量化模型在 2步/4步/8步 设置下分别比 FP16 模型提升 **16.1%/15.5%/5.6%** 的 FID
- 量化 + 少步蒸馏是互补的压缩技术，而非相互冲突

### 大规模模型（SDXL-Turbo 和 FLUX.1）上的结果

- SDXL-Turbo W4A8: Q-Sched FID = 21.41，显著优于 MixDQ (25.36) 和 Naive (25.75)
- SDXL 2步 PCM: Q-Sched W4A8 的 FID 仅比 FP16 退化 1.2%
- PTQD 在少步大模型上完全失效（FID = 161.96），因为其高斯噪声假设不再成立

### 用户偏好研究

- 超过 80,000 条人类标注
- Q-Sched 在 FLUX.1[schnell] 上优于 SVDQuant，在 SDXL-Turbo 上优于 MixDQ
- 在模型大小 vs. ELO 评分的 Pareto 前沿上，Q-Sched 达到最优

---

## 消融实验

### 预条件系数选择
联合优化 \( c_\epsilon \) 和 \( c_x \) 始终优于只优化单个系数，在两个指标上都达到最佳。

### k 值选择
- 过小的 k 导致颜色失真
- 过大的 k（如 k=5）使输出偏离真实数据分布
- k=2 在所有实验中表现稳定

### 损失函数对比
- CLIPScore 单独优化：图像过饱和，缺乏深度
- Brisque 单独优化：图像过于平滑，细节丢失
- CLIP-IQA-Q：图像质量高，但无法处理幻觉问题
- JAQ（CLIPScore + CLIP-IQA-Q）：在图像质量和文本一致性之间取得最佳平衡

### 随机性分析
在不同随机性水平 \( \eta \in \{0, 0.1, 0.3, 0.5, 0.7, 0.9\} \) 下，Q-Sched 在所有设置下均优于 PTQD。

---

## 理论保证

论文给出了严格的理论分析：由于最终图像的期望量化误差 \( E[||\Delta x_0||] \) 是关于采样系数 \( k_t, m_t \) 的线性函数，且存在全局最小值（零误差），因此理论上一定存在一组量化感知系数 \( \tilde{m}_t^*, \tilde{k}_t^* \) 严格改进期望量化误差。

---

## 量化引入的伪影类型

1. **颜色失真 (Color Distortion)**
2. **图像退化 (Degradation)**
3. **幻觉结构 (Hallucinations)**

Q-Sched 通过修改采样轨迹能有效缓解这些伪影，甚至在某些情况下生成比全精度模型更清晰的图像。

---

## 模型大小分析

量化只应用于 U-Net/DiT 骨干网络（占总模型大小的主要部分）：

| 模型 | FP16 总量 | DiT FP16 | DiT W4A8 | DiT W8A8 |
|------|-----------|----------|----------|----------|
| SDXL-Turbo | 1.37 GB | 1.03 GB | 0.26 GB | 0.51 GB |
| FLUX.1 | 6.73 GB | 4.76 GB | 1.19 GB | 2.38 GB |

---

## 局限与不足

1. W4A4 过于激进，图像质量不可接受
2. 多人图像、文字生成场景仍存在挑战
3. 部分 prompt 下会出现偏离原始语义的情况
4. JAQ 损失中的 k 值需要手工调整

---

## 总结与启示

Q-Sched 提出了一种全新视角的模型压缩方法——通过修改采样调度器而非模型权重来实现量化补偿。该方法的优势在于：

- **极低标定成本**：仅需 5 个提示，无需全精度模型
- **高度模块化**：独立于模型骨干架构（U-Net/DiT），适用于任何 TCD 类 scheduler
- **互补性**：量化与少步蒸馏是互补的压缩策略，两者叠加可进一步提升效率
- **实用性强**：20 分钟即可完成优化

这项工作为量化感知采样打开了新的研究方向，也为在资源受限设备上部署高质量生成模型提供了可行方案。

---

## 深入讨论 Q&A

### Q1：Q-Sched 调度器详解

**Q: 详细介绍一下 Q-Sched 调度器的设计原理。**

#### 背景：TCD 原始采样公式

Q-Sched 建立在 TCD 的 Strategic Stochastic Sampling (SSS) 之上：

\[
\mathbf{x_s} = \frac{\alpha_s}{\alpha_{s'}} \Big(\alpha_{s'}\frac{\mathbf{x_t} - \sigma_t \mathcal{E}^Q_\theta(x_t, t)}{\alpha_t} + \sigma_{s'}\mathcal{E}^Q_\theta(x_t, t) \Big) + \eta \mathbf{z}
\]

关键观察：公式依赖两个来自上一时间步的量 —— \( \mathbf{x_t} \) 和 \( \mathcal{E}^Q_\theta(x_t, t) \)，这为 Q-Sched 提供了两个独立的修正自由度。

#### Q-Sched 的核心修改

引入两个标量预条件系数：

\[
\mathbf{x_s} = \frac{\alpha_s}{\alpha_{s'}} \Big(\alpha_{s'}\frac{\textcolor{red}{c_x}\mathbf{x_t} - \sigma_t \textcolor{red}{c_{\epsilon}}\mathcal{E}^Q_\theta(x_t, t)}{\alpha_t} + \sigma_{s'}\textcolor{red}{c_{\epsilon}}\mathcal{E}^Q_\theta(x_t, t) \Big) + \eta \mathbf{z}
\]

| 系数 | 作用对象 | 含义 |
|------|----------|------|
| \( c_x \) | 当前样本 \( x_t \) | 控制"保留多少原始图像信息"，调节重建项的权重 |
| \( c_\epsilon \) | 噪声预测 \( \mathcal{E}^Q_\theta \) | 控制"去噪力度"，调节模型预测的贡献程度 |

当 \( c_x = c_\epsilon = 1 \) 时，退化为原始 TCD 调度器。

#### 为什么两个系数就够了 —— 理论分析

定义量化误差传播的递归形式：

\[
\Delta x_t = k_t \cdot \Delta x_{t+1} + m_t \cdot \Delta \mathcal{E}_\theta(t+1)
\]

其中 \( k_t = \frac{\alpha_t}{\alpha_{t+1}}, \; m_t = \frac{\alpha_t}{\alpha_{t'}}(\sigma_{t'} - \frac{\sigma_{t+1}}{\alpha_{t+1}}) \)。

最终图像的期望误差：

\[
E[||\Delta x_0||] = \sum_{s=1}^{S} \Big( \prod_{v=0}^{s-2} k_v \Big) \cdot m_{s-1} \cdot E[||\Delta \mathcal{E}_\theta(s)||]
\]

由于 \( k_t, m_t \in \mathbb{R}^+ \)，且 \( E[||\Delta x_0||] \) 是关于它们的**线性函数**，存在全局最小值（零误差），因此**理论上一定存在一组量化感知系数 \( \tilde{m}_t^*, \tilde{k}_t^* \) 严格改进期望量化误差**。

#### 搜索 / 优化过程

```
输入: 量化模型 E_Q, 标定提示集 P, JAQ 损失
输出: 最优系数 (c_x*, c_ε*)

1. 定义搜索空间: c_x ∈ [a, b], c_ε ∈ [c, d]
2. 对每个 (c_x, c_ε) 组合:
   a. 用修改后的 scheduler 完成完整 N 步采样，生成最终图像 x_0
   b. 计算 JAQ(x_0)
3. 选择使 JAQ 分数最大化的 (c_x*, c_ε*)
```

关键特点：
- **网格搜索**：两个一维参数的组合空间很小，穷举可接受
- **全局固定系数**：所有时间步共享同一组 \( (c_x, c_\epsilon) \)，所有 prompt 通用
- **推理时零开销**：系数是预先搜好的，推理时只是替换 scheduler 的乘数

#### 与不同模型骨干的兼容性

```
┌─────────────────────────────────┐
│          Prompt Text            │
└─────────────┬───────────────────┘
              ▼
┌─────────────────────────────────┐
│       Text Encoder(s)           │  ◄── 不量化
└─────────────┬───────────────────┘
              ▼
┌─────────────────────────────────┐
│   ┌─────────────────────────┐   │
│   │  Q-Sched Scheduler      │   │  ◄── Q-Sched 在这里！
│   │  c_x · x_t , c_ε · ε_Q  │   │
│   └─────────────────────────┘   │
│              │                  │
│              ▼                  │
│   ┌─────────────────────────┐   │
│   │  U-Net / DiT (Quantized)│   │  ◄── 量化骨干
│   └─────────────────────────┘   │
└─────────────────────────────────┘
```

已验证支持：U-Net 骨干（LCM/PCM/SDXL-Turbo）和 DiT 骨干（FLUX.1），涵盖一致性蒸馏和 Flow Matching 策略。

---

### Q2：JAQ 损失函数的详细定义

**Q: JAQ 具体怎么定义的？采样中如何使用？**

#### 核心公式

\[
\texttt{JAQ}(x) = \texttt{TC}(x) + k \cdot \texttt{IQ}(x)
\]

- \( \texttt{TC}(x) \)：Text-Image Compatibility，文本-图像一致性
- \( \texttt{IQ}(x) \)：Image Quality，纯图像质量
- \( k \)：平衡超参数，**k = 2**
- **目标**：最大化 JAQ 分数（分数越高越好）

#### 两种具体实现

| 使用场景 | TC(x) | IQ(x) | 公式 |
|----------|-------|-------|------|
| LCM/PCM (SD v1-5) | CLIPScore | CLIP-IQA-Q | JAQ = CLIPScore + 2·CLIP-IQA-Q |
| SDXL-Turbo / FLUX.1 | AQ-MAP | HPSV2 | JAQ = AQ-MAP + 2·HPSV2 |

#### 各子指标详解

**CLIPScore**：用 CLIP 图像/文本编码器计算生成图像与 prompt 的余弦相似度。局限：对量化伪影不敏感。

**CLIP-IQA-Q**：CLIP-IQA 的 "Quality" prompt 变体。比较输入图像与预定义质量锚文本（"high quality photo" / "low quality photo"）的嵌入相对相似度。局限：完全不理解图像语义，一张高质量但跑题的图像也能得高分。

**AQ-MAP**：空间对齐评分（来自 QRef），将图像分块计算每个区域与 prompt 的 CLIP 对齐度。比 CLIPScore 更细粒度。

**HPSV2**：在真实人类偏好数据上微调的图像质量模型，直接对齐人类审美。

#### 优化流程 —— 不是每步调整！

Q-Sched 的做法是**事先全局搜索，采样时固定系数**：

```
离线优化阶段（约 20 分钟）:
  For each (c_x, c_ε) in grid:
    For each prompt in P (5个):
      1. 用 (c_x, c_ε) 运行完整 N 步采样 → 最终图像 x_0
      2. 计算 JAQ(x_0)
    → 取所有 prompt 的平均 JAQ
  选择平均 JAQ 最高的 (c_x*, c_ε*)

推理阶段:
  用搜好的固定 (c_x*, c_ε*) 进行所有图像生成
  不再重新计算 JAQ，不动态调整
```

**关键点**：系数是全局共享的（所有时间步 + 所有 prompt 通用），推理时零额外开销。

#### 为什么要"联合"损失？

| 仅优化目标 | 现象 | 原因 |
|-----------|------|------|
| 仅 CLIPScore | 图像过饱和、色彩失真 | 只看语义对齐，不管视觉质量 |
| 仅 CLIP-IQA-Q | 可能出现幻觉、偏离 prompt | 只看图像质量，不理解 prompt |
| 仅 Brisque | 过于平滑、细节丢失 | 传统指标不适合生成任务 |
| **JAQ (两者联合)** | **细节丰富 + 语义准确** | 互补约束，互相制衡 |

k 值消融：k → 0 只关注 CLIPScore → 颜色失真；k = 5 偏向图像质量 → 偏离真实数据分布。k = 2 是经验最佳值。

---

### Q3：Q-Sched 能否用于非量化（FP16）模型？

**Q: 使用 JAQ 搜索最佳预条件系数与模型无关，那不用量化模型也可以吗？**

**短答案**：流程上可以，但几乎没有增益。

#### 理论分析

回顾最终图像的期望误差公式：

\[
E[||\Delta x_0||] = \sum_{s=1}^{S} \Big( \prod_{v=0}^{s-2} k_v \Big) \cdot m_{s-1} \cdot \textcolor{red}{E[||\Delta \mathcal{E}_\theta(s)||]}
\]

关键在红色项 \( E[||\Delta \mathcal{E}_\theta(s)||] \)：

| 场景 | \( \Delta \mathcal{E}_\theta \) | 误差来源 | 调度器调整空间 |
|------|-------------------------------|----------|---------------|
| **FP 模型** | \( \mathcal{E}_\theta - \mathcal{E}_\theta = 0 \) | 无 | 调整 \( k_t, m_t \) 但乘的是零 → 无意义 |
| **量化模型** | \( \mathcal{E}_\theta - \mathcal{E}^Q_\theta \neq 0 \) | 量化引入的噪声预测偏差 | 调整 \( k_t, m_t \) 乘的是非零值 → 有增益空间 |

对于 FP 模型，\( E[||\Delta \mathcal{E}_\theta||] = 0 \)，所以 \( E[||\Delta x_0||] = 0 \)。原始 scheduler 参数 \( c_x = c_\epsilon = 1 \) 已经是全局最优，调了也是白调。

#### 少步蒸馏 FP 模型的边界情况

蒸馏过程本身也可能引入偏差，论文的实验数据侧面印证了这一点：

| 模型 | 精度 | 调度器 | FID |
|------|------|--------|-----|
| 4-step PCM | FP16 | Original | 23.29 |
| 4-step PCM | W4A8 | **Q-Sched** | **17.39** |
| 8-step PCM | FP16 | Original | 20.15 |
| 8-step PCM | W4A8 | **Q-Sched** | **16.83** |

Q-Sched 的 W4A8 量化模型**反超**了 FP16 原始模型，说明蒸馏模型的 trajectory 并非"绝对最优"。理论上 FP16 蒸馏模型也可能有微小增益空间，但：

1. 原始 scheduler 是随模型训练/蒸馏耦合设计的
2. 对 FP 蒸馏模型的改进空间远小于量化模型（\( \Delta \mathcal{E}_\theta \) 是主要误差源）
3. 论文的核心主张是"量化 + 少步蒸馏互补"

#### 总结

| 问题 | 答案 |
|------|------|
| JAQ 搜索能否用于 FP 模型？ | 流程上可以，无技术障碍 |
| FP 模型用 Q-Sched 有效果吗？ | 几乎为零 — 原始 scheduler 已经是该模型的全局最优 |
| 少步蒸馏 FP 模型呢？ | 理论上可能有微小增益，但量化才是 Q-Sched 真正发挥价值的场景 |

一句话：**JAQ 是"模型无关"的评分工具，但 Q-Sched 的增益来源是"量化引入的 \( \Delta \mathcal{E}_\theta \neq 0 \)"**，FP 模型没有这个误差源。

---

### Q4：Q-Sched 是否绑定 TCD？换成 DDPM / DDIM / Flow Matching 是否失效？

**Q: Q-Sched 是基于 TCD 这种采样方式进行的吗？如果换成 DDPM 或者 Flow Matching 是否就失效了？**

**短答案**：Q-Sched **不绑定 TCD**，适用于任何具有 \( \{x_t, \text{model\_prediction}\} \) 二元结构的采样器。论文已验证在 TCD、Flow Matching、ADD 三种采样器上均有效。

#### 各类采样器的原始公式

##### DDPM (Denoising Diffusion Probabilistic Models)

\[
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} x_t - \frac{1-\alpha_t}{\sqrt{\alpha_t}\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) + \sigma_t z
\]

其中 \( \sigma_t = \sqrt{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t} \)。

##### DDIM (Denoising Diffusion Implicit Models)

\[
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}} + \sqrt{1-\bar{\alpha}_{t-1}}\epsilon_\theta(x_t, t)
\]

DDPM 和 DDIM 统称为 "predicting \( \epsilon \)" 模式：模型直接预测噪声。

##### Flow Matching / Rectified Flow (FM)

\[
x_{t-1} = x_t - \Delta t \cdot v_\theta(x_t, t)
\]

称为 "predicting \( v \)"（velocity）模式：模型预测速度场。Euler 离散化后每一步沿速度方向前进 \( \Delta t \)。

##### TCD / SSS (Trajectory Consistency Distillation)

\[
x_s = \frac{\alpha_s}{\alpha_{s'}} \Big(\alpha_{s'}\frac{x_t - \sigma_t \epsilon_\theta(x_t, t)}{\alpha_t} + \sigma_{s'} \epsilon_\theta(x_t, t) \Big) + \eta z
\]

TCD 同样预测 \( \epsilon \)，但引入了子步 \( s' = (1-\gamma)t \)，使得一步可以跨越更大的时间跨度。

#### 各类采样器适配 Q-Sched 后的公式

**DDPM 适配：**

\[
x_{t-1} = \frac{\textcolor{red}{c_x}}{\sqrt{\alpha_t}} x_t - \frac{1-\alpha_t}{\sqrt{\alpha_t}\sqrt{1-\bar{\alpha}_t}} \textcolor{red}{c_\epsilon}\epsilon^Q_\theta(x_t, t) + \sigma_t z
\]

**DDIM 适配**（与 TCD 几乎一致，因为 DDIM 是 TCD + \( \gamma = 1 \) + 无噪声的特例）：

\[
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \frac{\textcolor{red}{c_x}x_t - \sqrt{1-\bar{\alpha}_t}\textcolor{red}{c_\epsilon}\epsilon^Q_\theta}{\sqrt{\bar{\alpha}_t}} + \sqrt{1-\bar{\alpha}_{t-1}}\textcolor{red}{c_\epsilon}\epsilon^Q_\theta
\]

**Flow Matching / Rectified Flow 适配：**

\[
x_{t-1} = \textcolor{red}{c_x} \cdot x_t - \textcolor{red}{c_\epsilon} \cdot \Delta t \cdot v^Q_\theta(x_t, t)
\]

这里 \( v_\theta \) 是速度场预测，等价于 score-based 框架中 \( \epsilon_\theta \) 的角色。论文中 SDXL-Turbo 和 FLUX.1[schnell] 用的就是这种适配。

**TCD 适配**（原始论文形式）：

\[
x_s = \frac{\alpha_s}{\alpha_{s'}} \Big(\alpha_{s'}\frac{c_x x_t - \sigma_t c_\epsilon \epsilon^Q_\theta}{\alpha_t} + \sigma_{s'} c_\epsilon \epsilon^Q_\theta \Big) + \eta z
\]

#### 统一视角

```
        采样器分类
        │
   ┌────┴──────────┐
   │               │
Score-based            Velocity-based
(预测噪声 ε)            (预测速度 v)
   │                      │
   ├─ DDPM               ├─ Rectified Flow
   ├─ DDIM               ├─ Flow Matching
   ├─ TCD/SSS            └─ ADD (SDXL-Turbo)
   └─ LCM/PCM
   
   共同的二元结构:
   { x_t , model_prediction }
         │
         ▼
   Q-Sched 注入:
   c_x · x_t + c_ε · model_prediction
```

无论采样器预测的是噪声 \( \epsilon \) 还是速度 \( v \)，Q-Sched 的操作完全一致：在状态项和模型预测项之间重新分配权重。

#### 论文中的实验覆盖

| 模型 | 底层采样器 | 类型 | Q-Sched 结果 |
|------|-----------|------|-------------|
| LCM (SD v1-5) | LCM scheduler | score-based | ✅ 4-step FID 15.5% 提升 |
| PCM (SD v1-5/XL) | TCD/SSS | score-based | ✅ 2/4/8-step 全面优于 FP16 |
| SDXL-Turbo | ADD + Flow Matching | velocity-based | ✅ W4A8 FID 超 MixDQ 4 个点 |
| FLUX.1[schnell] | Rectified Flow | velocity-based | ✅ 用户偏好超 SVDQuant |

#### 什么情况下会失效？

| 场景 | 是否适配 | 原因 |
|------|:--:|------|
| DDPM / DDIM | ✅ | \( x_t \) 和 \( \epsilon_\theta \) 天然分离 |
| TCD / SSS | ✅ | 同上，且系数空间更大 |
| Flow Matching (Euler) | ✅ | 速度场替代噪声预测，结构一致 |
| Heun / RK45 等高阶 ODE solver | ⚠️ | 多步子步骤使系数引入更复杂，需额外设计 |
| 纯随机采样（无模型预测） | ❌ | 没有 \( \epsilon_\theta \) 可供缩放 |
| 一步生成（如 GAN-based） | ❌ | 不存在多步采样轨迹 |

#### 为什么 TCD 仍是"最佳适配"？

1. **公式天然解耦**：DDPM 中两个项的系数 \( \frac{1}{\sqrt{\alpha_t}} \) 和 \( \frac{1-\alpha_t}{\sqrt{\alpha_t}\sqrt{1-\bar{\alpha}_t}} \) 共享 \( \alpha_t \)，而 TCD 中 \( \alpha, \sigma \) 独立设计，两系数调整范围更大。
2. **少步对量化最敏感**：蒸馏将 50-100 步压缩至 2-8 步后，每步的量化偏差被放大，Q-Sched 的修正空间最大。标准 DDPM 多步可"平均掉"偏差。
3. **论文已验证的最优场景**：最大增益出现在 2-step PCM（16.1%）和 4-step LCM（15.5%），步数越少增益越大。

#### 一句话总结

> Q-Sched 的泛化边界不是"是否使用 TCD"，而是**采样器是否有 \( \{x_t, \text{model\_prediction}\} \) 的二元结构**。DDPM、DDIM、TCD、Flow Matching、Rectified Flow 都有此结构，均可适配。TCD 是最佳场景因为公式天然解耦且少步蒸馏对量化最敏感。
